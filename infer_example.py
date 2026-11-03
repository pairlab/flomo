#!/usr/bin/env python3
"""Minimal FloMo robot-action inference example.

Loads a `FloMo` checkpoint and runs one inference pass:
image(s) + task text + current end-effector pose -> a 14D YPR robot action
chunk.

Pass one `--image` per camera view the checkpoint conditions on (its `img_<view>`
inputs), repeated in canonical view order: primary, left, right. A single-view
checkpoint takes one `--image`. Steps:

    1. Load the checkpoint (rebuild from saved cfg, apply LoRA, load weights).
    2. LANCZOS-squash the conditioning image(s) to the checkpoint's square size.
    3. Encode the task text (UMT5) and conditioning image(s) (Wan VAE).
    4. Build a B=1 cotrain batch and run flow-matching denoising (`model.infer`).
    5. Denormalize positions, convert relative rotations to absolute rotations,
       and convert the model's 20D rot6d actions to the robot's 14D YPR format.

What it deliberately does NOT do:
    * Any transport / control loop (this just returns one chunk).
    * Camera acquisition or hardware control.

Requires a CUDA GPU and a trained checkpoint. Example:

    python infer_example.py \
        --checkpoint /path/to/epoch.pt \
        --wan-ckpt-dir ./wan_checkpoints \
        --image primary.png --image left.png --image right.png \
        --task "pick up the croissant and place it in the oven" \
        --ee-pose="x,y,z,yaw,pitch,roll,grip,x,y,z,yaw,pitch,roll,grip"
"""

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

# Frozen prefixes: non-LoRA checkpoints strip these (reloaded from the Wan .pth
# in __init__), so they're "missing" on load_state_dict — expected, not an error.
FROZEN_PREFIXES = ("text_encoder.", "vae.")


def load_model(ckpt, wan_ckpt_dir, device):
    """Rebuild the model from the loaded checkpoint dict and load weights. Returns (model, cfg)."""
    from flomo.models.FloMo import FloMo
    from flomo.utils.train_utils import rename_legacy_state_keys

    cfg = OmegaConf.create(ckpt["config"])

    # The cfg references the training machine's filesystem; point Wan weights at
    # the local download dir instead.
    cfg.forward_dynamics.wan.vae_weights = f"{wan_ckpt_dir}/Wan2.2_VAE.pth"
    cfg.forward_dynamics.wan.text_encoder_weights = f"{wan_ckpt_dir}/models_t5_umt5-xxl-enc-bf16.pth"

    # Encode the cond image live from raw pixels (no preprocessed latents here),
    # and supply a precomputed text embedding (see encode_text).
    for mod_cfg in cfg.forward_dynamics.modalities.values():
        if mod_cfg.kind == "image":
            mod_cfg.use_preprocessed = False
    cfg.forward_dynamics.cotrain.use_preprocessed_text_embs = True
    # Legacy: checkpoints trained before `tracks_normalizer` became a required
    # dataset arg omit it; omission meant no Ramen normalization.
    for ds in cfg.datasets:
        if "tracks_normalizer" not in ds.args:
            ds.args.tracks_normalizer = None

    model = FloMo(cfg).to(device)
    # setup_model() applies LoRA wrapping (renames backbone params), so it MUST
    # run before load_state_dict or every key mismatches.
    model.setup_model()
    is_lora = bool(ckpt["lora_enabled"])
    state = rename_legacy_state_keys(ckpt["model"])
    res = model.load_state_dict(state, strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"unexpected keys: {res.unexpected_keys[:10]}")
    if not is_lora:
        bad = [k for k in res.missing_keys if not k.startswith(FROZEN_PREFIXES)]
        if bad:
            raise RuntimeError(f"missing non-frozen keys: {bad[:10]}")
    model.eval()
    return model, cfg


@torch.no_grad()
def encode_text(cfg, text, device):
    """Encode the task string to a (text_len, text_dim) UMT5 embedding."""
    from flomo.models.FloMo import WanTextEncoder
    from transformers import AutoTokenizer

    text_len = int(cfg.forward_dynamics.modalities.text.text_len)
    tok = AutoTokenizer.from_pretrained("google/umt5-xxl")
    enc = WanTextEncoder()
    enc.load_state_dict(torch.load(cfg.forward_dynamics.wan.text_encoder_weights, map_location="cpu"))
    enc = enc.to(device).eval()
    batch = tok(
        [text], return_tensors="pt", padding="max_length",
        truncation=True, max_length=text_len,
    ).to(device)
    ctx = enc(batch["input_ids"], batch["attention_mask"])
    ctx = ctx * batch["attention_mask"].unsqueeze(-1).to(ctx.dtype)  # zero the pad
    return ctx[0].detach().to(device=device, dtype=torch.float32)  # (L, D)


def load_image(path, size):
    """Load RGB pixels as `(1, size, size, 3)` using the training-time LANCZOS squash."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)[None]  # (1, size, size, 3)


def relative_rotation_actions(cfg):
    """Read the checkpoint's action rotation convention and require consistency."""
    action_ds = [ds for ds in cfg.datasets if "action" in list(ds.outputs)]
    if not action_ds:
        raise RuntimeError("checkpoint config has no action-producing dataset")
    missing = [str(ds.name) for ds in action_ds if "relative_rotation_actions" not in ds.args]
    if missing:
        raise RuntimeError(f"action datasets are missing relative_rotation_actions: {missing}")
    values = {str(ds.name): bool(ds.args.relative_rotation_actions) for ds in action_ds}
    if len(set(values.values())) != 1:
        raise RuntimeError(f"relative_rotation_actions disagrees across action datasets: {values}")
    return next(iter(values.values()))


def parse_ee_pose(value, action_space):
    """Parse the robot's current 14D YPR or 20D rot6d pose into the model action space."""
    from flomo.utils.action_preprocess import convert_ypr_to_rot6d

    ee = np.array([float(x) for x in value.split(",")], dtype=np.float32)
    if ee.shape == (14,):
        if action_space == "eepose_6drot":
            return convert_ypr_to_rot6d(ee)
        if action_space == "eepose_ypr":
            return ee
        raise ValueError(f"14D robot poses are incompatible with action_space={action_space}")
    if ee.shape == (20,):
        if action_space != "eepose_6drot":
            raise ValueError(f"20D robot poses are incompatible with action_space={action_space}")
        return ee
    raise ValueError(f"--ee-pose must contain 14 or 20 comma-separated values, got {ee.shape[0]}")


def postprocess_robot_actions(actions, norm, cfg, ee_pose):
    """Convert model actions to the robot's 14D YPR format and return `(b, t, 14)`."""
    from flomo.utils.action_preprocess import convert_rot6d_to_ypr, relative_rot6d_to_absolute

    feat_dim = int(cfg.forward_dynamics.modalities.action.feat_dim)
    t_steps = int(cfg.forward_dynamics.modalities.action.t_steps)
    if actions.ndim != 3 or actions.shape[1:] != (t_steps, feat_dim):
        raise ValueError(f"actions must be (b, {t_steps}, {feat_dim}), got {tuple(actions.shape)}")
    if norm.horizon != t_steps:
        raise ValueError(f"normalizer horizon={norm.horizon} but model t_steps={t_steps}")
    if norm.action_space == "eepose_6drot" and feat_dim != 20:
        raise ValueError(f"eepose_6drot requires feat_dim=20, got {feat_dim}")
    if norm.action_space == "eepose_ypr" and feat_dim != 14:
        raise ValueError(f"eepose_ypr requires feat_dim=14, got {feat_dim}")

    ee_ref = parse_ee_pose(ee_pose, norm.action_space)
    actions = norm.denormalize(actions, ee_ref)
    if relative_rotation_actions(cfg):
        if norm.action_space != "eepose_6drot":
            raise RuntimeError(f"relative rotations require eepose_6drot, got {norm.action_space}")
        actions = relative_rot6d_to_absolute(actions, ee_ref)
    if norm.action_space == "eepose_6drot":
        actions = convert_rot6d_to_ypr(actions)
    return actions


def build_normalizer(ckpt, normalizer_type):
    """Load the ActionNormalizer baked into the checkpoint."""
    from flomo.utils.action_normalizer import ActionNormalizer

    stats = ckpt["normalizer_stats"]
    if stats is None:
        raise RuntimeError("checkpoint has no embedded normalizer_stats; pass one explicitly")
    if "action_space" in stats:  # legacy single-normalizer format
        return ActionNormalizer.from_stats_dict(stats)
    if normalizer_type not in stats:  # typed dict: pick the dataset type
        raise RuntimeError(f"normalizer_stats has no '{normalizer_type}'; available: {list(stats)}")
    return ActionNormalizer.from_stats_dict(stats[normalizer_type])


def build_batch(model, images, text_emb, device):
    """Hand-roll a B=1 cotrain batch (mirrors CotrainCollate for one sample)."""
    from flomo.models.modalities import _WanVAELatentModality

    batch = {}
    img_mods = [n for n in model.active_inputs if n == "img" or n.startswith("img_")]
    if len(img_mods) != len(images):
        raise ValueError(f"model has {len(img_mods)} image inputs {img_mods}; got {len(images)} --image")
    for name, px in zip(img_mods, images):
        batch[name] = [torch.from_numpy(px).to(device)]  # (n_pix, S, S, 3) uint8
        batch[f"{name}_mask"] = torch.tensor([True], device=device)
    if "text" in model.active_inputs:
        batch["text_emb"] = text_emb.unsqueeze(0)  # (1, L, D)
        batch["text_mask"] = torch.tensor([True], device=device)
    for name in model.active_outputs:  # outputs: True mask + ignored placeholder
        mod = model.modalities[name]
        if isinstance(mod, _WanVAELatentModality):
            shape = (mod.LATENT_CHANNELS, mod.t_frames, mod.h_lat, mod.w_lat)
        else:  # action
            shape = (int(mod.t_steps), int(mod.feat_dim))
        batch[name] = [torch.zeros(shape, dtype=torch.float32, device=device)]
        batch[f"{name}_mask"] = torch.tensor([True], device=device)
    return batch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="cotrain checkpoint (.pt) saved by train.py")
    p.add_argument("--wan-ckpt-dir", default="./wan_checkpoints", help="dir with Wan2.2_VAE.pth + umT5 weights")
    p.add_argument("--image", action="append", required=True, help="cond image path (repeat for multi-view)")
    p.add_argument("--task", required=True, help="task description (encoded by UMT5)")
    p.add_argument("--inference-steps", type=int, default=10, help="flow-matching denoising steps")
    p.add_argument("--normalizer-type", default="realyam", help="dataset-type key for the checkpoint's normalizer")
    p.add_argument(
        "--ee-pose",
        required=True,
        help="comma-separated current EE pose: 14D robot YPR format (preferred) or 20D model rot6d format",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("FloMo requires a CUDA GPU (the Wan backbone initializes CUDA at import).")
    device = torch.device("cuda")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, cfg = load_model(ckpt, args.wan_ckpt_dir, device)

    size = int(cfg.forward_dynamics.wan.vae_spatial)
    images = [load_image(path, size) for path in args.image]
    text_emb = encode_text(cfg, args.task, device)
    batch = build_batch(model, images, text_emb, device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model.infer(batch, args.inference_steps)

    if "action" not in pred:
        raise RuntimeError(f"model has no 'action' output; got {list(pred)}")
    actions = pred["action"].to(torch.float32)  # (1, T_a, feat_dim)

    norm = build_normalizer(ckpt, args.normalizer_type)
    actions = postprocess_robot_actions(actions, norm, cfg, args.ee_pose)

    chunk = actions[0].cpu().numpy()  # (T_a, 14)
    print(f"action chunk: shape={chunk.shape}, space=eepose_ypr")
    print(chunk)


if __name__ == "__main__":
    main()
