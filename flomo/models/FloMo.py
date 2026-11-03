"""Any-to-any forward dynamics on bidirectional Wan DiT.

The trunk attends over concatenated modality tokens. Clean inputs use timestep
0; noised outputs use per-modality flow-matching timesteps.

WanModel.forward patchifies `(B, 48, F, H, W)` inputs rather than accepting
`(B, T_m, D)` tokens. This wrapper instead calls its text/time embeddings,
blocks, head, and RoPE directly, relying on the Wan 2.2 component layout.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import repeat

from flomo.loaders.cotrain_loader import (
    base_modality_name,
    derive_active_modalities,
    expand_modality_order,
)
from flomo.models.modalities import (
    ROLE_INPUT,
    ROLE_OUTPUT,
    ActionModality,
    ImageModality,
    MotionTokensWanPretrainedModality,
    TextModality,
    VideoModality,
    _WanVAELatentModality,
)

# Wan2.2 and dreamzero aren't pip-installable; add them to sys.path at import.
# Clone both next to this repo (see README).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _add_sibling_repo(name: str, clone: str) -> None:
    path = os.path.join(_REPO_ROOT, name)
    if not os.path.isdir(path):
        raise ImportError(f"{name} not found at {path}. Run: {clone}")
    if path not in sys.path:
        sys.path.insert(0, path)


_add_sibling_repo("dreamzero", "git clone https://github.com/dreamzero0/dreamzero.git dreamzero")
_add_sibling_repo("Wan2.2", "git clone https://github.com/Wan-Video/Wan2.2.git")

from wan.modules import attention as _wan_attn  # noqa: E402
from wan.modules import model as _wan_model  # noqa: E402

_wan_model.flash_attention = _wan_attn.attention                                        # SDPA fallback
from wan.modules.model import WanModel, sinusoidal_embedding_1d  # noqa: E402

from groot.vla.model.dreamzero.modules import attention as _dz_attn  # noqa: E402
from groot.vla.model.dreamzero.modules.flow_match_scheduler import (  # noqa: E402
    FlowMatchScheduler,
)
from groot.vla.model.dreamzero.modules.wan_video_dit import WanModelStateDictConverter  # noqa: E402
from groot.vla.model.dreamzero.modules.wan_video_text_encoder import (  # noqa: E402
    WanTextEncoder,
)
from groot.vla.model.dreamzero.modules.wan_video_vae import WanVideoVAE38  # noqa: E402

from flomo.utils.constants import resolve_wan_size  # noqa: E402


def _is_hopper_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name(0).lower()
    return "h100" in name or "h200" in name or "hopper" in name


# FA3 hopper kernels are H100/H200 only. Gate the module flags by GPU type so
# Ampere/Ada don't crash at kernel launch.
if not _is_hopper_gpu():
    _wan_attn.FLASH_ATTN_3_AVAILABLE = False
    _dz_attn.FLASH_ATTN_3_AVAILABLE = False


def _resolve_motion_tokens_supervision(cfg) -> Tuple[str, bool]:
    """Return the shared motion-token `(supervision, tracks_ramen_normalized)`.

    All output datasets must agree to share one VAE latent space.
    Returns `("cumulative_disp", False)` if none emits motion tokens.
    """
    sup_values: set = set()
    ramen_values: set = set()
    for ds in cfg.datasets:
        outputs = list(ds.outputs)
        if "motion_tokens" not in outputs:
            continue
        args = ds.args
        sup = args.supervision
        if sup == "cum_disp":  # legacy alias for pre-rename checkpoints
            sup = "cumulative_disp"
        ramen = (args.tracks_normalizer == "ramen")
        sup_values.add(sup)
        ramen_values.add(bool(ramen))
    if len(sup_values) > 1:
        raise ValueError(
            f"datasets producing motion_tokens disagree on supervision: {sorted(sup_values)}; "
            f"all motion_tokens-producing datasets must share a single supervision mode."
        )
    if len(ramen_values) > 1:
        raise ValueError(
            f"datasets producing motion_tokens disagree on tracks_normalizer (ramen vs none): "
            f"{sorted(ramen_values)}"
        )
    if not sup_values:
        return "cumulative_disp", False
    return next(iter(sup_values)), next(iter(ramen_values))


def _build_modality(name: str, role: str, cfg, dim: int) -> nn.Module:
    """Build a modality from its base-name config and optional camera-view suffix."""
    base = base_modality_name(name)
    view = name[len(base) + 1:] if name != base else None
    m_cfg = cfg.forward_dynamics.modalities[base]
    kind = m_cfg.kind
    if kind == "image":
        return ImageModality(
            name, dim=dim,
            h_lat=m_cfg.h_lat, w_lat=m_cfg.w_lat, view=view,
        )
    if kind == "video":
        return VideoModality(
            name, role=role, dim=dim, t_frames=m_cfg.t_frames,
            h_lat=m_cfg.h_lat, w_lat=m_cfg.w_lat, view=view,
        )
    if kind == "action":
        return ActionModality(name, dim=dim, t_steps=m_cfg.t_steps, action_dim=m_cfg.feat_dim)
    if kind == "text":
        return TextModality(name, dim=dim, text_len=m_cfg.text_len, text_dim=m_cfg.text_dim)
    if kind == "motion_tokens_wan_pretrained":
        sup, ramen = _resolve_motion_tokens_supervision(cfg)
        return MotionTokensWanPretrainedModality(
            name, role=role, dim=dim,
            t_frames=m_cfg.t_frames, h_lat=m_cfg.h_lat, w_lat=m_cfg.w_lat,
            supervision=sup, tracks_ramen_normalized=ramen,
        )
    raise ValueError(f"Unknown modality kind {kind!r} for {name!r}")


def cotrain_loss(
    per_sample_losses: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Weighted sum of per-modality means over real rows.

    Losses have shape `[B, ...]`; masks are `[B]` with True for real rows.
    Returns `(total, breakdown)` with detached per-modality means for logging.
    Empty modalities contribute zero while remaining in the autograd graph.
    """
    total: torch.Tensor = None
    breakdown: Dict[str, torch.Tensor] = {}
    for m, raw in per_sample_losses.items():
        per_row = raw if raw.dim() == 1 else raw.flatten(1).mean(dim=1)                 # [B]
        mask = masks[m].to(per_row.dtype)                                               # [B]   1.0 where real
        n_real = mask.sum()
        if n_real.item() == 0:
            m_loss = (mask * per_row).sum() * 0.0
        else:
            m_loss = (mask * per_row).sum() / n_real                                    # mean over real rows
        weighted = weights[m] * m_loss
        total = weighted if total is None else total + weighted
        breakdown[m] = m_loss.detach()
    return total, breakdown


def per_dataset_breakdown(
    per_sample_losses: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    dataset_names: List[str],
) -> Dict[Tuple[str, str], torch.Tensor]:
    """Per-(dataset, modality) mean over real rows for logging.

    Returns `{(dataset, modality): scalar Tensor (detached)}`. A pair with zero
    real rows on this batch is omitted — wandb sees a gap rather than a 0/NaN.
    `None` entries in `dataset_names` are skipped.
    """
    out: Dict[Tuple[str, str], torch.Tensor] = {}
    unique = sorted(d for d in set(dataset_names) if d is not None)
    for m, raw in per_sample_losses.items():
        per_row = raw if raw.dim() == 1 else raw.flatten(1).mean(dim=1)                 # [B]
        mod_mask = masks[m].to(per_row.dtype)                                           # [B]
        for ds in unique:
            ds_mask = torch.tensor(
                [d == ds for d in dataset_names], dtype=per_row.dtype, device=per_row.device,
            )
            combined = mod_mask * ds_mask
            n_real = combined.sum()
            if n_real.item() == 0:
                continue
            out[(ds, m)] = ((combined * per_row).sum() / n_real).detach()
    return out


class FloMo(nn.Module):
    """Any-to-any forward dynamics on the bidirectional Wan DiT trunk."""

    def __init__(self, cfg):
        super().__init__()
        wan_cfg = cfg.forward_dynamics.wan
        cotrain_cfg = cfg.forward_dynamics.cotrain

        size = resolve_wan_size(wan_cfg)
        self.dim = size["dim"]
        self.num_heads = size["num_heads"]
        self.head_dim = self.dim // self.num_heads
        self.patch_size = (1, 2, 2)

        self.finetune_mode = wan_cfg.finetune_mode
        self.lora_cfg = wan_cfg.lora

        if wan_cfg.backbone != "bidirectional":
            raise ValueError(
                f"cotrain only supports bidirectional Wan DiT; got {wan_cfg.backbone!r}"
            )
        print("[Wan] Initializing WanModel (bidirectional, model_type='ti2v') for cotrain ...")
        self.backbone = WanModel(
            model_type="ti2v",
            patch_size=self.patch_size,
            text_len=512,
            in_dim=48, out_dim=48,
            dim=self.dim, ffn_dim=size["ffn_dim"], freq_dim=256,
            text_dim=4096,
            num_heads=self.num_heads, num_layers=size["num_layers"],
            window_size=(-1, -1),
            qk_norm=True, cross_attn_norm=True, eps=1e-6,
        )

        # Expand camera streams before checking the configured token order.
        active_inputs, active_outputs, _universal, _partial = derive_active_modalities(
            cfg.datasets, cfg.forward_dynamics.modalities,
        )
        active_set = set(active_inputs) | set(active_outputs)
        order = expand_modality_order(cotrain_cfg.modality_order, active_inputs, active_outputs)
        if set(order) != active_set or len(order) != len(set(order)):
            raise ValueError(
                f"expanded modality_order={order} must exactly cover active "
                f"modalities {sorted(active_set)} with no duplicates"
            )

        self.modalities = nn.ModuleDict()
        active_input_set = set(active_inputs)
        for name in order:
            role = ROLE_INPUT if name in active_input_set else ROLE_OUTPUT
            self.modalities[name] = _build_modality(name, role, cfg, self.dim)

        self.active_inputs: List[str] = [n for n in order if n in active_input_set]
        self.active_outputs: List[str] = [n for n in order if n in set(active_outputs)]

        scheduler = FlowMatchScheduler(
            num_train_timesteps=wan_cfg.num_train_timesteps,
            num_inference_steps=wan_cfg.num_inference_steps,
            shift=wan_cfg.scheduler_shift,
        )
        scheduler.set_timesteps(wan_cfg.num_train_timesteps, training=True)
        for mod in self.modalities.values():
            mod.scheduler = scheduler
        # Separate instance for inference so set_timesteps(num_inference_steps,
        # training=False) doesn't clobber the training schedule the modalities reference.
        self.inference_scheduler = FlowMatchScheduler(
            num_train_timesteps=wan_cfg.num_train_timesteps,
            num_inference_steps=wan_cfg.num_inference_steps,
            shift=wan_cfg.scheduler_shift,
        )

        self.loss_weights: Dict[str, float] = {
            n: float(w) for n, w in cfg.forward_dynamics.loss_weights.items()
        }

        # Image/video preprocessing is per modality; text uses the cotrain flag.
        self.use_preprocessed_text_embs = bool(cotrain_cfg.use_preprocessed_text_embs)
        per_mod_use_preproc: Dict[str, bool] = {
            name: bool(cfg.forward_dynamics.modalities[base_modality_name(name)].use_preprocessed)
            for name, mod in self.modalities.items()
            if isinstance(mod, _WanVAELatentModality)
        }
        # Preprocessed modalities still need the VAE for visualization.
        need_vae = len(per_mod_use_preproc) > 0

        if not self.use_preprocessed_text_embs:
            from transformers import AutoTokenizer
            print(f"[Wan] Loading UMT5-XXL text encoder from {wan_cfg.text_encoder_weights} ...")
            self.text_encoder = WanTextEncoder()
            self.text_encoder.load_state_dict(
                torch.load(wan_cfg.text_encoder_weights, map_location="cpu")
            )
            self.tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
        else:
            self.text_encoder = None
            self.tokenizer = None

        if need_vae:
            self.vae = WanVideoVAE38(vae_pretrained_path=wan_cfg.vae_weights)
            self.vae.model.load_state_dict(
                torch.load(wan_cfg.vae_weights, map_location="cpu", weights_only=True)
            )
        else:
            self.vae = None

        # Share encoder and projection weights across modality instances.
        self.text_mod: Optional[TextModality] = None
        for name, mod in self.modalities.items():
            if isinstance(mod, _WanVAELatentModality):
                mod.use_preprocessed = per_mod_use_preproc[name]
                mod.vae_spatial = int(wan_cfg.vae_spatial)
                mod.vae = self.vae
                mod.patch_embedding = self.backbone.patch_embedding
                mod.unpatch_proj = self.backbone.head.head
            elif isinstance(mod, TextModality):
                if self.text_mod is not None:
                    raise RuntimeError("multiple TextModality instances in modality_order; only one supported")
                self.text_mod = mod
                mod.use_preprocessed = self.use_preprocessed_text_embs
                mod.text_encoder = self.text_encoder
                mod.tokenizer = self.tokenizer

        # Without text inputs, encode_context supplies Wan's zero context.
        if self.text_mod is None:
            text_cfg = cfg.forward_dynamics.modalities.text
            self.text_mod = TextModality(
                "text", dim=self.dim, text_len=text_cfg.text_len, text_dim=text_cfg.text_dim,
            )
            self.text_mod.use_preprocessed = self.use_preprocessed_text_embs
            self.text_mod.text_encoder = self.text_encoder
            self.text_mod.tokenizer = self.tokenizer
            # TextModality never touches the trunk, so its mask_token is unused —
            # mark untrained so DDP doesn't see it as an unused parameter.
            self.text_mod.mask_token.requires_grad_(False)

        if wan_cfg.init_weights == "pretrained":
            self._load_pretrained_weights(wan_cfg.model_id)

        self._log_modality_summary(cfg)

    def _log_modality_summary(self, cfg) -> None:
        """Log token order, conditioning, RoPE geometry, and losses at construction.

        Non-main ranks are silenced by `suppress_non_main_prints`.
        """
        order = self._trunk_ordered_names()
        input_set = set(self.active_inputs)

        rows, offset = [], 0
        for i, name in enumerate(order):
            mod = self.modalities[name]
            kind = str(cfg.forward_dynamics.modalities[base_modality_name(name)].kind)
            is_input = name in input_set
            t_m = mod.num_tokens()
            rows.append({
                "i": i, "name": name, "kind": kind,
                "role": "INPUT" if is_input else "OUTPUT",
                "state": "clean, t=0" if is_input else "FM-noised",
                "tokens": t_m, "lo": offset, "hi": offset + t_m,
            })
            offset += t_m

        name_w = max([8] + [len(r["name"]) for r in rows])
        kind_w = max([8] + [len(r["kind"]) for r in rows])
        hdr = (f"   {'#':>2}  {'modality':<{name_w}}  {'kind':<{kind_w}}  "
               f"{'role':<6}  {'trunk state':<11}  {'tokens':>6}  token range")
        rule = "  " + "-" * (len(hdr) - 2)

        lines = [
            "", "=" * len(hdr),
            f"  FloMo — any-to-any training layout "
            f"({self.num_params / 1e6:.1f}M params)",
            "=" * len(hdr),
            "  TRUNK — bidirectional self-attention over the concatenated token stream:",
            "", hdr, rule,
        ]
        for r in rows:
            lines.append(
                f"   {r['i']:>2}  {r['name']:<{name_w}}  {r['kind']:<{kind_w}}  "
                f"{r['role']:<6}  {r['state']:<11}  {r['tokens']:>6}  "
                f"[{r['lo']:>5}:{r['hi']:>5})"
            )
        lines += [rule, f"  trunk total: {offset} tokens", ""]

        # Text rides cross-attention, never the trunk.
        text_active = self.text_mod is not None and self.text_mod.name in self.modalities
        lines.append("  CROSS-ATTENTION CONTEXT")
        if text_active:
            lines.append(f"    text — ACTIVE: {self.text_mod.text_len} encoded tokens / sample")
        else:
            lines.append("    text — inactive: zero context fed (Wan unconditional path)")

        # RoPE grid covers the VAE-latent prefix (F = sum of t_frames).
        f_total, grid_h, grid_w = 0, None, None
        for name in order:
            mod = self.modalities[name]
            if isinstance(mod, _WanVAELatentModality):
                f_total += mod.t_frames
                grid_h, grid_w = mod.h_tok, mod.w_tok
        lines += [
            "",
            f"  RoPE — unified Wan grid over the VAE-latent prefix: "
            f"F={f_total}  H={grid_h}  W={grid_w}",
            "",
            "  LOSS — flow-matching MSE, OUTPUT modalities only:",
        ]
        for name in self.active_outputs:
            w = self.loss_weights[base_modality_name(name)]
            lines.append(f"    {name}  (weight {w})")
        if not self.active_outputs:
            lines.append("    (none)")

        lines += [
            "",
            "  Legend: INPUT modalities are encoded clean at timestep 0; OUTPUT",
            "  modalities are flow-matching-noised at a per-row timestep, denoised by",
            "  the trunk, and supervised against the FM target.",
            "=" * len(hdr), "",
        ]
        print("\n".join(lines))

    def _load_pretrained_weights(self, model_id: str) -> None:
        from diffusers import WanTransformer3DModel
        print(f"[Wan] Loading pretrained Wan backbone weights from {model_id} ...")
        diffusers_model = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
        )
        diffusers_sd = diffusers_model.state_dict()
        del diffusers_model
        converter = WanModelStateDictConverter()
        native_sd, _ = converter.from_diffusers(diffusers_sd)
        missing, unexpected = self.backbone.load_state_dict(native_sd, strict=False)
        print(
            f"[Wan] cotrain backbone pretrained loaded. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def setup_model(self) -> None:
        if self.vae is not None:
            for p in self.vae.parameters():
                p.requires_grad = False
            self.vae.eval()
        if self.text_encoder is not None:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_encoder.eval()

        if self.finetune_mode == "lora":
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=self.lora_cfg.rank,
                lora_alpha=self.lora_cfg.alpha,
                target_modules=list(self.lora_cfg.target_modules),
                lora_dropout=self.lora_cfg.dropout,
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)
            self.backbone.print_trainable_parameters()

        print(
            f"[Wan] cotrain setup_model: "
            f"{self.num_trainable_params/1e6:.1f}M trainable / {self.num_params/1e6:.1f}M total"
        )

    def forward(self, batch: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Return `(predictions, total_loss)` for a cotrain-collated batch.

        Each modality has a length-B list of optional tensors and a `[B]` bool
        mask. Universal modalities have no missing entries and all-True masks.
        """
        device = self._infer_device(batch)
        B = self._infer_batch_size(batch)
        self._stash_batch_ds_names(batch)

        masks: Dict[str, torch.Tensor] = {
            name: batch[f"{name}_mask"].to(device=device, dtype=torch.bool)
            for name in self.modalities
        }

        # Wan RoPE rotates only the VAE-token prefix; see _trunk_ordered_names.
        ordered_names = self._trunk_ordered_names()
        trunk_tokens: Dict[str, torch.Tensor] = {}
        targets: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}                       # (target, real_idx)
        output_t_per_row: Dict[str, torch.Tensor] = {}                                   # [B], FM-t per row
        for name in ordered_names:
            mod = self.modalities[name]
            tokens, target, t_row, real_idx = self._encode_trunk_input(
                mod, batch[name], masks[name], B, device,
            )
            trunk_tokens[name] = tokens
            if mod.role == ROLE_OUTPUT:
                targets[name] = (target, real_idx)
                output_t_per_row[name] = t_row

        x = torch.cat([trunk_tokens[n] for n in ordered_names], dim=1)                   # [B, N_total, D]

        t_tok = self._build_per_token_timesteps(ordered_names, output_t_per_row, B, x.shape[1], device)

        grid_sizes = self._compute_grid_sizes(B, device)

        # Text is absent from `masks` when no dataset declares it (fallback zero-context text_mod).
        text_mask = masks[self.text_mod.name] if self.text_mod.name in masks else None
        text_context = self.text_mod.encode_context(batch, text_mask, B)                # [B, text_len, text_dim]

        x_out = self._trunk_call(x, t_tok, text_context, grid_sizes)                     # [B, N_total, D]

        predictions: Dict[str, torch.Tensor] = {}
        per_sample_losses: Dict[str, torch.Tensor] = {}
        for name, span in self._output_slices(ordered_names):
            mod = self.modalities[name]
            pred_latent = mod.untokenize(x_out[:, span])                                # [B, *latent_shape]
            target_real, real_idx = targets[name]
            per_sample = self._compute_per_sample_loss(
                mod, pred_latent, target_real, real_idx, B,
            )
            # Absent rows use t=0 and are excluded by cotrain_loss.
            t_w = mod.scheduler.training_weight(output_t_per_row[name].to(device)).to(per_sample)
            per_sample_losses[name] = per_sample * t_w
            predictions[name] = pred_latent

        # All camera views share their base modality's loss weight.
        out_masks = {n: masks[n] for n in self.active_outputs}
        out_weights = {n: self.loss_weights[base_modality_name(n)] for n in self.active_outputs}
        total, breakdown = cotrain_loss(per_sample_losses, out_masks, out_weights)

        # Validation also breaks down dataset/subname; training keeps a flat namespace.
        # Absent when the batch did not come from a cotrain dataset (e.g. infer_example).
        dataset_names = batch["dataset_name"] if "dataset_name" in batch else None
        dataset_subnames = batch["dataset_subname"] if "dataset_subname" in batch else None
        ds_breakdown = (
            per_dataset_breakdown(per_sample_losses, out_masks, dataset_names)
            if dataset_names is not None else {}
        )
        if (not self.training) and dataset_names is not None and dataset_subnames is not None:
            composite = [
                f"{n}/{s}" if s is not None else None
                for n, s in zip(dataset_names, dataset_subnames)
            ]
            ds_breakdown = {
                **ds_breakdown,
                **per_dataset_breakdown(per_sample_losses, out_masks, composite),
            }
        self.last_train_info = {
            "loss_breakdown": breakdown,
            "per_dataset_breakdown": ds_breakdown,
        }
        return predictions, total

    @torch.no_grad()
    def infer(self, batch: Dict[str, Any], num_inference_steps: int) -> Dict[str, torch.Tensor]:
        """Return clean output latents from a flow-matching ODE solve.

        Encode inputs once at t=0. Initialize outputs with N(0, I), then update
        their latents with inference_scheduler while keeping inputs fixed.
        """
        device = self._infer_device(batch)
        B = self._infer_batch_size(batch)
        self._stash_batch_ds_names(batch)

        masks: Dict[str, torch.Tensor] = {
            name: batch[f"{name}_mask"].to(device=device, dtype=torch.bool)
            for name in self.modalities
        }
        ordered_names = self._trunk_ordered_names()

        input_tokens: Dict[str, torch.Tensor] = {}
        for name in ordered_names:
            mod = self.modalities[name]
            if mod.role == ROLE_INPUT:
                input_tokens[name] = mod.mask_and_encode(batch[name], masks[name])

        # Match training autocast precision throughout the scheduler updates.
        noisy_latents: Dict[str, torch.Tensor] = {
            name: self.modalities[name].init_noise(B, device, dtype=torch.bfloat16)
            for name in self.active_outputs
        }

        self.inference_scheduler.set_timesteps(num_inference_steps, training=False)
        timesteps = self.inference_scheduler.timesteps.to(device)

        text_mask = masks[self.text_mod.name] if self.text_mod.name in masks else None
        text_context = self.text_mod.encode_context(batch, text_mask, B)
        grid_sizes = self._compute_grid_sizes(B, device)

        for t_val in timesteps:
            t_row = torch.full((B,), float(t_val.item()), device=device, dtype=torch.float32)

            trunk_tokens: Dict[str, torch.Tensor] = dict(input_tokens)
            output_t_per_row: Dict[str, torch.Tensor] = {}
            for name in self.active_outputs:
                mod = self.modalities[name]
                trunk_tokens[name] = mod.tokenize(noisy_latents[name])
                output_t_per_row[name] = t_row

            x = torch.cat([trunk_tokens[n] for n in ordered_names], dim=1)
            t_tok = self._build_per_token_timesteps(
                ordered_names, output_t_per_row, B, x.shape[1], device,
            )
            x_out = self._trunk_call(x, t_tok, text_context, grid_sizes)

            for name, span in self._output_slices(ordered_names):
                pred_velocity = self.modalities[name].untokenize(x_out[:, span])
                noisy_latents[name] = self.inference_scheduler.step(
                    pred_velocity, t_val, noisy_latents[name],
                )

        return noisy_latents

    @staticmethod
    def _infer_device(batch: Dict[str, Any]) -> torch.device:
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v.device
            if isinstance(v, list):
                for t in v:
                    if isinstance(t, torch.Tensor):
                        return t.device

    @staticmethod
    def _infer_batch_size(batch: Dict[str, Any]) -> int:
        for k, v in batch.items():
            if k.endswith("_mask"):
                return int(v.shape[0])
            if isinstance(v, torch.Tensor):
                return int(v.shape[0])
            if isinstance(v, list):
                return len(v)

    def _stash_batch_ds_names(self, batch: Dict[str, Any]) -> None:
        """Push per-batch dataset names onto modalities that consume them in encode."""
        ds_names = batch["dataset_name"] if "dataset_name" in batch else None
        for mod in self.modalities.values():
            if hasattr(mod, "_ds_to_idx"):
                mod._batch_ds_names = ds_names

    def _encode_trunk_input(
        self, mod, field, mask: torch.Tensor, B: int, device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return `(tokens, target_real, t_per_row, real_idx)` for one modality.

        Tokens have shape `[B, T_m, D]`. Inputs are clean; their other returns
        are None. Outputs are noised on real rows and return targets
        `[N_real, *latent_shape]`, timesteps `[B]`, and indices `[N_real]`.
        """
        if mod.role == ROLE_INPUT:
            return mod.mask_and_encode(field, mask), None, None, None

        T_m = mod.num_tokens()
        real_modality_batch, real_idx = mod.gather_real(field, mask)
        if real_modality_batch is None:
            tokens = mod.scatter_tokens(
                torch.empty(0, T_m, self.dim, device=device, dtype=mod.mask_token.dtype),
                real_idx, B,
            )
            return (
                tokens,
                torch.empty(0, device=device),
                torch.zeros(B, device=device, dtype=torch.float32),
                real_idx,
            )

        clean_latent = mod.encode_rows(real_modality_batch, real_idx)                   # [N_real, *latent_shape]
        t_real = mod.sample_t(clean_latent.shape[0], device)                             # [N_real]
        noisy_latent, target_real = mod.add_noise(clean_latent, t_real)
        tokens_real = mod.tokenize(noisy_latent)                                         # [N_real, T_m, D]
        if int(real_idx.numel()) == B:
            tokens = tokens_real                                                          # all-real: skip scatter
        else:
            tokens = mod.scatter_tokens(tokens_real, real_idx, B)

        t_per_row = torch.zeros(B, device=device, dtype=t_real.dtype)
        t_per_row[real_idx] = t_real
        return tokens, target_real, t_per_row, real_idx

    def _build_per_token_timesteps(
        self, ordered_names: List[str], output_t_per_row: Dict[str, torch.Tensor],
        B: int, N_total: int, device: torch.device,
    ) -> torch.Tensor:
        """[B, N_total] timestep tensor: 0 on clean-input tokens, FM-t on noised-output tokens.
        FM-t broadcasts across the T_m tokens of each output modality."""
        t_tok = torch.zeros(B, N_total, device=device, dtype=torch.float32)
        for name, span in self._output_slices(ordered_names):
            t_row = output_t_per_row[name].to(torch.float32)                            # [B]
            t_tok[:, span] = repeat(t_row, "b -> b n", n=span.stop - span.start)
        return t_tok

    def _output_slices(self, ordered_names: List[str]):
        """Yield output token spans in the same order used to assemble the trunk."""
        offset = 0
        for name in ordered_names:
            end = offset + self.modalities[name].num_tokens()
            if name in self.active_outputs:
                yield name, slice(offset, end)
            offset = end

    def _trunk_ordered_names(self) -> List[str]:
        """Order VAE tokens before non-VAE tokens; omit zero-token modalities.

        Within each partition, inputs precede outputs in modality_order.
        Wan RoPE rotates only the first F*H*W tokens, so the non-VAE tail
        relies on learned position embeddings.
        """
        vae_in, vae_out, non_in, non_out = [], [], [], []
        for name in self.modalities:                                                     # nn.ModuleDict preserves modality_order
            mod = self.modalities[name]
            if mod.num_tokens() == 0:
                continue
            is_vae = isinstance(mod, _WanVAELatentModality)
            is_input = name in self.active_inputs
            (vae_in if is_vae and is_input else
             vae_out if is_vae else
             non_in if is_input else
             non_out).append(name)
        return vae_in + vae_out + non_in + non_out

    def _compute_grid_sizes(self, B: int, device: torch.device) -> torch.Tensor:
        """Return `[B, 3]` Wan RoPE grids for the VAE-token prefix.

        F sums modality t_frames; H/W must share h_tok/w_tok. This places
        conditioning and predicted frames on Wan's pretrained temporal grid.
        """
        F_total = 0
        H = W = None
        for name in self.modalities:
            mod = self.modalities[name]
            if not isinstance(mod, _WanVAELatentModality) or mod.num_tokens() == 0:
                continue
            F_total += mod.t_frames
            if H is None:
                H, W = mod.h_tok, mod.w_tok
            elif (mod.h_tok, mod.w_tok) != (H, W):
                raise ValueError(
                    f"VAE-latent modalities must share (h_tok, w_tok); got "
                    f"{name}=({mod.h_tok}, {mod.w_tok}) vs prior ({H}, {W})"
                )
        if F_total == 0:
            raise ValueError("cotrain trunk needs at least one VAE-latent modality")
        return torch.tensor([[F_total, H, W]] * B, dtype=torch.long, device=device)

    def _trunk_call(
        self, x: torch.Tensor, t_tok: torch.Tensor, text_context: torch.Tensor, grid_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Run Wan blocks on pre-tokenized `[B, N_total, D]` inputs.

        t_tok is `[B, N_total]`; text_context is `[B, text_len, text_dim]`;
        grid_sizes is `[B, 3]` for RoPE over the VAE-token prefix.
        Wan blocks broadcast per-token modulation `[B, N_total, 6, D]`.

        Inline the head's AdaLN for per-token modulation. Output projections
        remain in each modality's untokenize method.
        """
        B, N, _ = x.shape
        device = x.device
        # Wan blocks require float32 timestep modulation, even under outer autocast.
        with torch.amp.autocast("cuda", dtype=torch.float32):
            t_flat = t_tok.reshape(B * N).to(torch.float32)
            t_emb = self.backbone.time_embedding(
                sinusoidal_embedding_1d(self.backbone.freq_dim, t_flat).float()
            )                                                                            # [B*N, D]
            t_emb = t_emb.view(B, N, self.dim)
            t_mod = self.backbone.time_projection(t_emb).unflatten(2, (6, self.dim))     # [B, N, 6, D]
            assert t_emb.dtype == torch.float32 and t_mod.dtype == torch.float32

        context = self.backbone.text_embedding(text_context)                             # [B, text_len, D]

        seq_lens = torch.full((B,), N, dtype=torch.long, device=device)
        context_lens = torch.full((B,), context.shape[1], dtype=torch.long, device=device)
        if self.backbone.freqs.device != device:
            self.backbone.freqs = self.backbone.freqs.to(device)
        freqs = self.backbone.freqs                                                      # [1024, head_dim_complex]

        for block in self.backbone.blocks:
            x = block(x, t_mod, seq_lens, grid_sizes, freqs, context, context_lens)

        # AdaLN finalize from WanModel.head: per-token scale/shift built from
        # `head.modulation [1, 2, D]` and our per-token `t_emb [B, N, D]`.
        head = self.backbone.head
        with torch.amp.autocast("cuda", dtype=torch.float32):
            e = head.modulation.unsqueeze(0) + t_emb.float().unsqueeze(2)                # [B, N, 2, D]
            x = head.norm(x) * (1 + e[:, :, 1]) + e[:, :, 0]                             # [B, N, D]
        return x

    @staticmethod
    def _compute_per_sample_loss(
        mod, pred_latent: torch.Tensor, target_real: torch.Tensor,
        real_idx: torch.Tensor, B: int,
    ) -> torch.Tensor:
        """[B] per-sample loss. Absent rows contribute 0 (cotrain_loss masks them)."""
        device = pred_latent.device
        if real_idx.numel() == 0:
            return torch.zeros(B, device=device, dtype=pred_latent.dtype)
        pred_real = pred_latent[real_idx]
        per_sample_real = mod.compute_loss(pred_real, target_real)                       # [N_real]
        per_sample = torch.zeros(B, device=device, dtype=per_sample_real.dtype)
        per_sample[real_idx] = per_sample_real
        return per_sample
