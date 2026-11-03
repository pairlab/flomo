"""Weighted any-to-any cotraining with FloMo."""

import datetime
import os
import time
from collections import defaultdict
from typing import Dict, List
import hydra
import numpy as np
import torch
import wandb
from einops import rearrange, repeat
from omegaconf import OmegaConf
from tqdm import tqdm

from torch.utils.data import ConcatDataset, DataLoader, RandomSampler

from flomo.loaders.cotrain_loader import (
    CotrainCollate,
    build_cotrain_dataloader,
    build_cotrain_dataset,
    cotrain_batch_to_device,
    derive_active_modalities,
)
from flomo.models.modalities import (
    ActionModality,
    MotionTokensWanPretrainedModality,
)
from flomo.models.FloMo import FloMo
from flomo.utils.log_utils import BrokenSampleLog
from flomo.utils.grad_norm_logging import log_module_gradients
from flomo.utils.config_validation import validate_train_config
from flomo.utils.distributed import (
    create_accelerator,
    is_main_process,
    log_main,
    prepare_training_components,
    register_grad_comm_hook,
    save_training_checkpoint,
    suppress_non_main_prints,
    wandb_log,
)
from flomo.utils.motion_stats import _install_motion_token_stats
from flomo.utils.train_utils import (
    compute_mid_epoch_resume,
    get_checkpoint_dir,
    get_lr_scheduler,
    get_optim,
    get_root_dir,
    latest_checkpoint_from_dir,
    load_checkpoint,
    safe_torch_compile,
    segfault_resilient_iter,
    setup_wandb_tmp_dirs,
    unwrap_model_for_checkpoint,
)


def _compute_ar_metrics(pred_pos, gt_pos):
    """MSE/L1 + moving/nonmoving variants on (B, V, T, N, D) positions.
    Moving/nonmoving keys are skipped when their mask is empty."""
    from flomo.utils.data_utils import compute_moving_indices
    diff = pred_pos.float() - gt_pos.float()
    out = {"mse": diff.pow(2).mean().item(), "l1": diff.abs().mean().item()}
    moving = compute_moving_indices(gt_pos).bool()                                         # (B, V, N)
    moving_mask = repeat(moving, "b v n -> b v t n d", t=gt_pos.shape[2], d=gt_pos.shape[-1])
    for tag, mask in (("moving", moving_mask), ("nonmoving", ~moving_mask)):
        if mask.any():
            d = diff[mask]
            out[f"mse_{tag}"] = d.pow(2).mean().item()
            out[f"l1_{tag}"] = d.abs().mean().item()
    return out


@torch.no_grad()
def _ar_metrics_from_pred_latents(inner, batch, pred_latents):
    """Return per-modality AR MSE/L1; skip modalities without position decoders.

    Cumulative-displacement tracks are zero-anchored. Reconstruct GT and
    predicted positions from `batch["init_pts"]` in physical units.
    """
    out = {}
    for name, pred_latent in pred_latents.items():
        mod = inner.modalities[name]
        if isinstance(mod, ActionModality):
            gt_raw, real_idx = mod.gather_real(batch[name], batch[f"{name}_mask"])       # (N_real, T_a, action_dim)
            if gt_raw is None:
                continue
            diff = pred_latent[real_idx].float() - gt_raw.float()
            out[name] = {"mse": diff.pow(2).mean().item(), "l1": diff.abs().mean().item()}
            continue
        gt_raw, real_idx = mod.gather_real(batch[name], batch[f"{name}_mask"])           # (N_real, V, T, N, D)
        if gt_raw is None:
            continue
        ds_idx = None
        if isinstance(mod, MotionTokensWanPretrainedModality):
            # Ramen encoding skips ds_idx, but decoding needs per-dataset bounds.
            ds_idx = torch.tensor(
                [mod._ds_to_idx[mod._batch_ds_names[i]] for i in real_idx.tolist()],
                device=pred_latent.device, dtype=torch.long,
            )
        if getattr(mod, "supervision", None) == "cumulative_disp":
            if "init_pts" not in batch:
                continue
            init_real = torch.stack(
                [batch["init_pts"][i] for i in real_idx.tolist()], dim=0,
            ).to(gt_raw)                                                                 # (N_real, V, 1, N, D)
            T_vel = gt_raw.shape[2] - 1
            # Undo loader Ramen normalization so GT and predictions share physical units.
            if getattr(mod, "tracks_ramen_normalized", False) and ds_idx is not None:
                side = int(round(gt_raw.shape[3] ** 0.5))
                gt_grid = rearrange(gt_raw[:, 0, 1:], "b t (h w) c -> b c t h w", h=side, w=side)
                gt_phys = torch.stack([
                    mod._un_ramen(gt_grid[b], int(ds_idx[b]), t_lo=1, t_hi=1 + T_vel)
                    for b in range(gt_grid.shape[0])
                ], dim=0)
                gt_cd = rearrange(gt_phys, "b c t h w -> b 1 t (h w) c").to(init_real)
            else:
                gt_cd = gt_raw[:, :, 1:]
            gt_pos = torch.cat([init_real, init_real + gt_cd], dim=2)
            init_pts = init_real
        else:
            init_pts = gt_raw[:, :, 0:1]
            gt_pos = gt_raw
        pred_real = pred_latent[real_idx]
        T_vel = gt_pos.shape[2] - 1
        if isinstance(mod, MotionTokensWanPretrainedModality):
            pred_pos = mod.decode_positions(pred_real, ds_idx, init_pts, T_vel)
        else:
            continue
        out[name] = _compute_ar_metrics(pred_pos, gt_pos[:, :, : pred_pos.shape[2]])
    return out



def _share_normalizer_within_type(per_ds, ds_cfgs, accelerator):
    """Share one action normalizer over each dataset type's combined train data.

    Types with at most one normalizer need no recomputation.
    """
    by_type: Dict[str, List[int]] = {}
    for i, (d, c) in enumerate(zip(per_ds, ds_cfgs)):
        if getattr(d, "normalizer", None) is None:
            continue
        by_type.setdefault(str(c.type), []).append(i)

    for type_name, idxs in by_type.items():
        if len(idxs) < 2:
            continue
        # Aggregate unique (task, demo_id) across all instances' index_maps.
        seen: set = set()
        seqs: List[np.ndarray] = []
        action_space = per_ds[idxs[0]].action_space
        horizon = per_ds[idxs[0]].true_horizon
        for i in idxs:
            d = per_ds[i]
            for entry in d.index_map:
                key = (entry["task"], entry["demo_id"])
                if key in seen:
                    continue
                seen.add(key)
                with d._open(entry["demo_path"]) as f:
                    seqs.append(f[f"actions/{action_space}"][:].astype(np.float32))
        from flomo.utils.action_normalizer import compute_normalizer_stats
        shared = compute_normalizer_stats(
            action_sequences=seqs, reference_sequences=seqs,
            action_space=action_space, horizon=horizon,
        )
        shared.metadata = {
            "tasks": sorted({t for t, _ in seen}),
            "num_demos": len(seqs),
            "shared_across": [str(ds_cfgs[i].name) for i in idxs],
        }
        for i in idxs:
            per_ds[i].set_normalizer(shared)
        log_main(
            f"[normalizer-share] type={type_name}: shared normalizer across "
            f"{len(idxs)} train datasets ({[str(ds_cfgs[i].name) for i in idxs]}); "
            f"{len(seqs)} demos, {len(seen)} unique (task, demo).",
            accelerator=accelerator,
        )


def _normalizers_by_type(per_ds, ds_cfgs):
    """Pick each type's normalizer after `_share_normalizer_within_type`."""
    by_type = {}
    for d, c in zip(per_ds, ds_cfgs):
        if c.type in by_type:
            continue
        nrm = getattr(d, "normalizer", None)
        if nrm is not None:
            by_type[c.type] = nrm
    return by_type


def _build_val_loader(cfg, universal, partial, type_normalizers):
    """Sequential val loader. Each val dataset takes its type's normalizer so
    train and val share one action scale per dataset type."""
    if not cfg.val_datasets:
        return None
    val_ds = [
        build_cotrain_dataset(d, cfg.forward_dynamics, None)
        for d in cfg.val_datasets
    ]
    for val_d, val_cfg in zip(val_ds, cfg.val_datasets):
        if not getattr(val_d, "normalize_actions", False):
            continue
        if val_cfg.type not in type_normalizers:
            raise RuntimeError(
                f"Val dataset '{val_cfg.name}' (type={val_cfg.type}) has "
                f"normalize_actions=True but no train dataset of type "
                f"{val_cfg.type} provides a normalizer."
            )
        val_d.set_normalizer(type_normalizers[val_cfg.type])
    combined = ConcatDataset(val_ds)
    collate = CotrainCollate(universal=universal, partial=partial)
    n_steps = cfg.val_steps_per_epoch
    sampler = RandomSampler(
        combined,
        replacement=False,
        num_samples=int(n_steps) * int(cfg.batch_size) if n_steps is not None else len(combined),
    )
    return DataLoader(
        combined,
        batch_size=int(cfg.gpu_max_bs),
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
    )


def _reduce_batched(mean_groups, sum_groups, accelerator):
    """ONE all-reduce. `mean_groups`: {grp: ({k: sum}, {k: count})} → world means.
    `sum_groups`: {grp: {k: sum}} → world sums. Mean entries with zero world count
    are omitted. All ranks must agree on every key set."""
    mean_meta = [(g, sorted(s.keys())) for g, (s, _) in mean_groups.items()]
    sum_meta  = [(g, sorted(s.keys())) for g, s in sum_groups.items()]
    flat_s = [float(mean_groups[g][0][k]) for g, ks in mean_meta for k in ks]
    flat_c = [float(mean_groups[g][1][k]) for g, ks in mean_meta for k in ks]
    flat_p = [float(sum_groups[g][k])      for g, ks in sum_meta  for k in ks]
    if not flat_s and not flat_p:
        return {g: {} for g, _ in mean_meta}, {g: {} for g, _ in sum_meta}
    payload = torch.tensor(flat_s + flat_c + flat_p, device=accelerator.device, dtype=torch.float64)
    payload = accelerator.reduce(payload, "sum")
    n = len(flat_s)
    means, i = {}, 0
    for g, ks in mean_meta:
        d = {}
        for k in ks:
            cnt = payload[i + n].item()
            if cnt > 0:
                d[k] = (payload[i] / payload[i + n]).item()
            i += 1
        means[g] = d
    sums, j = {}, 2 * n
    for g, ks in sum_meta:
        sums[g] = {k: payload[j + off].item() for off, k in enumerate(ks)}
        j += len(ks)
    return means, sums


class _MicroBatchMeans:
    """Per-rank accumulator over micro-batches; `flush` does ONE all-reduce returning
    world means (loss/bd/pd) and world sums (per-dataset sample counts).
    `pd_keys` is the canonical (dataset, modality) cross-product; `ds_keys` is the
    canonical dataset list — both fixed so all ranks agree on the key set."""

    def __init__(self, pd_keys, ds_keys):
        self._pd_keys = pd_keys
        self._ds_keys = ds_keys
        self.reset()

    def reset(self):
        self.loss_sum, self.loss_n = 0.0, 0
        self.bd_sum, self.bd_n = defaultdict(float), defaultdict(int)
        self.pd_sum = {k: 0.0 for k in self._pd_keys}
        self.pd_n = {k: 0 for k in self._pd_keys}
        self.ds_count = {k: 0 for k in self._ds_keys}

    def add(self, loss, info, dataset_names):
        self.loss_sum += float(loss.item())
        self.loss_n += 1
        for k, v in info["loss_breakdown"].items():
            self.bd_sum[k] += float(v.item())
            self.bd_n[k] += 1
        for k, v in info["per_dataset_breakdown"].items():
            if k in self.pd_sum:
                self.pd_sum[k] += float(v.item())
                self.pd_n[k] += 1
        for name in dataset_names:
            self.ds_count[name] += 1

    def flush(self, accelerator):
        means, sums = _reduce_batched(
            mean_groups={
                "loss": ({"_": self.loss_sum}, {"_": self.loss_n}),
                "bd":   (dict(self.bd_sum), dict(self.bd_n)),
                "pd":   (self.pd_sum, self.pd_n),
            },
            sum_groups={"ds": dict(self.ds_count)},
            accelerator=accelerator,
        )
        loss = means["loss"]["_"]
        bd, pd, ds = means["bd"], means["pd"], sums["ds"]
        self.reset()
        return loss, bd, pd, ds


def _reduce_ar(per_mod, inner, accelerator):
    """World-mean AR metrics. Canonical key set = decoder-bearing modalities × metric names."""
    ar_keys = []
    for name, mod in inner.modalities.items():
        if isinstance(mod, MotionTokensWanPretrainedModality):
            for k in ("mse", "l1", "mse_moving", "l1_moving", "mse_nonmoving", "l1_nonmoving"):
                ar_keys.append((name, k))
        elif isinstance(mod, ActionModality):
            for k in ("mse", "l1"):
                ar_keys.append((name, k))
    local_sum = {k: 0.0 for k in ar_keys}
    local_n = {k: 0 for k in ar_keys}
    for name, m in per_mod.items():
        for k, v in m.items():
            key = (name, k)
            if key in local_sum:
                local_sum[key] += float(v)
                local_n[key] += 1
    return _reduce_batched({"_": (local_sum, local_n)}, {}, accelerator)[0]["_"]


def _val_sub_dataset_keys(val_loader, active_outputs):
    """Return `(dataset/subname, modality)` val keys, omitting None subnames."""
    raw = val_loader.dataset
    datasets = raw.datasets if hasattr(raw, "datasets") else [raw]
    out = []
    for d in datasets:
        if not (hasattr(d, "cotrain_name") and hasattr(d, "index_map")):
            continue
        if not hasattr(d, "_cotrain_subnames_cache"):
            d._cotrain_subnames_cache = sorted(
                {d._cotrain_subname(idx) for idx in d.index_map} - {None}
            )
        for s in d._cotrain_subnames_cache:
            composite = f"{d.cotrain_name}/{s}"
            for m in active_outputs:
                out.append((composite, m))
    return out


@torch.no_grad()
def _validate(model, val_loader, accelerator, cfg, num_inf_steps, epoch, train_steps, val_steps, broken_log):
    """Log teacher-forced metrics and periodic AR metrics as world means.

    Reduce at optimizer-step boundaries and keep val_steps monotonic across epochs.
    """
    model.eval()
    inner = unwrap_model_for_checkpoint(model)
    pd_keys = [(str(d.name), m) for d in cfg.val_datasets for m in inner.active_outputs]
    pd_keys += _val_sub_dataset_keys(val_loader, inner.active_outputs)
    ds_keys = [str(d.name) for d in cfg.val_datasets]
    tf_sum, tf_count = defaultdict(float), defaultdict(int)
    pd_sum, pd_count = defaultdict(float), defaultdict(int)
    ar_sum = defaultdict(lambda: defaultdict(float))
    ar_count = defaultdict(lambda: defaultdict(int))
    loss_sum, loss_count = 0.0, 0
    ar_interval = int(cfg.val_ar_interval)
    grad_accum = accelerator.gradient_accumulation_steps
    n_batches = len(val_loader)
    mtr = _MicroBatchMeans(pd_keys, ds_keys)
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="val",
                                           disable=not is_main_process(accelerator))):
        broken_log.ingest(batch.pop("_broken_samples", []), phase="val")
        batch = cotrain_batch_to_device(batch, accelerator.device, non_blocking=True)
        with accelerator.autocast():
            _pred, loss = model(batch)
        mtr.add(loss, inner.last_train_info, batch["dataset_name"])

        is_opt_step = ((batch_idx + 1) % grad_accum == 0) or (batch_idx + 1 == n_batches)
        if not is_opt_step:
            continue

        val_steps += 1
        avg_loss, avg_breakdown, avg_pd_breakdown, _ = mtr.flush(accelerator)
        log_dict = {
            "val_steps": val_steps,
            "val/loss": avg_loss,
            "val/epoch": epoch,
            **{f"val/loss/{k}": v for k, v in avg_breakdown.items()},
            **{f"val/loss/{ds}/{mod}": v for (ds, mod), v in avg_pd_breakdown.items()},
        }
        loss_sum += avg_loss
        loss_count += 1
        for k, v in avg_breakdown.items():
            tf_sum[k] += v
            tf_count[k] += 1
        for (ds, mod), v in avg_pd_breakdown.items():
            pd_key = f"{ds}/{mod}"
            pd_sum[pd_key] += v
            pd_count[pd_key] += 1

        if ar_interval > 0 and (val_steps % ar_interval == 0):
            with accelerator.autocast():
                pred_latents = inner.infer(batch, num_inf_steps)
            per_mod = _ar_metrics_from_pred_latents(inner, batch, pred_latents)
            avg_ar = _reduce_ar(per_mod, inner, accelerator)
            for (name, k), v in avg_ar.items():
                log_dict[f"val/ar/{name}/{k}"] = v
                ar_sum[name][k] += v
                ar_count[name][k] += 1
            _log_pred_viz(model, batch, val_steps, accelerator, num_inf_steps,
                          phase="val", pred_latents=pred_latents)
            _log_cond_img_viz(model, batch, val_steps, accelerator, phase="val")
        wandb_log(log_dict, accelerator)

    avg_dict = {"val_steps": val_steps, "train_steps": train_steps, "val/epoch": epoch}
    if loss_count > 0:
        v = loss_sum / loss_count
        avg_dict["val/loss_avg"] = v
        avg_dict["val/loss_avg_by_train"] = v
    for k in tf_sum:
        v = tf_sum[k] / tf_count[k]
        avg_dict[f"val/loss_avg/{k}"] = v
        avg_dict[f"val/loss_avg_by_train/{k}"] = v
    for k in pd_sum:
        v = pd_sum[k] / pd_count[k]
        avg_dict[f"val/loss_avg/{k}"] = v
        avg_dict[f"val/loss_avg_by_train/{k}"] = v
    for name, ks in ar_sum.items():
        for k, s in ks.items():
            avg_dict[f"val/ar_avg/{name}/{k}"] = s / ar_count[name][k]
    if len(avg_dict) > 1:
        wandb_log(avg_dict, accelerator)
    if is_main_process(accelerator) and broken_log.unique:
        wandb_log({"skipped/table": broken_log.wandb_table()}, accelerator)
    model.train()
    return val_steps


def _log_pred_viz(model, batch, step, accelerator, num_inference_steps, phase, pred_latents=None):
    """Log source/GT/prediction/error panels and projected tracks per modality."""
    if not is_main_process(accelerator):
        return
    inner = unwrap_model_for_checkpoint(model)
    if pred_latents is None:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred_latents = inner.infer(batch, num_inference_steps)
    log_dict = {f"{phase}_steps": step}
    for name, pred_latent in pred_latents.items():
        mod = inner.modalities[name]
        flow_out = mod.viz_pred(batch, pred_latent)
        if flow_out is not None:
            video, caption = flow_out
            log_dict[f"viz/{phase}_flow/{name}"] = wandb.Video(video, fps=4.0, format="mp4", caption=caption)
        # Also log the modality's optional loss-target-space panel.
        if hasattr(mod, "viz_pred_normalized"):
            norm_out = mod.viz_pred_normalized(batch, pred_latent)
            if norm_out is not None:
                video, caption = norm_out
                log_dict[f"viz/{phase}_flow_normalized/{name}"] = wandb.Video(
                    video, fps=4.0, format="mp4", caption=caption,
                )
        if hasattr(mod, "viz_tracks"):
            tr_out = mod.viz_tracks(batch, pred_latent)
            if tr_out is not None:
                video, caption = tr_out
                log_dict[f"viz/{phase}_tracks/{name}"] = wandb.Video(video, fps=4.0, format="mp4", caption=caption)
    if len(log_dict) > 1:
        wandb_log(log_dict, accelerator)


@torch.no_grad()
def _log_cond_img_viz(model, batch, step, accelerator, phase):
    """VAE-decode each `img_<view>` input stream for the first real row and log
    under `viz/{phase}_cond_img/{view}`. Sanity check for the wrist cameras,
    which the pred-viz panel doesn't cover (img is input-only)."""
    if not is_main_process(accelerator):
        return
    inner = unwrap_model_for_checkpoint(model)
    log_dict = {f"{phase}_steps": step}
    for name, mod in inner.modalities.items():
        if not name.startswith("img_"):
            continue
        mask = batch[f"{name}_mask"]
        real_idx = mask.nonzero(as_tuple=True)[0]
        if real_idx.numel() == 0:
            continue
        i = int(real_idx[0])
        raw = batch[name][i].unsqueeze(0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            latent = mod.encode(raw)
            rgb = mod.vae.single_decode(latent.to(torch.bfloat16))                # (1, 3, T_pix, S, S) in [-1, 1]
        frame = rgb[0, :, 0].float()                                              # (3, S, S) cond frame
        u8 = ((frame + 1.0) * 0.5 * 255.0).clamp(0, 255).to(torch.uint8)
        img = rearrange(u8, "c h w -> h w c").cpu().numpy()
        view = name[len("img_"):]
        log_dict[f"viz/{phase}_cond_img/{view}"] = wandb.Image(img, caption=view)
    if len(log_dict) > 1:
        wandb_log(log_dict, accelerator)


@hydra.main(config_path="cfg", config_name="train", version_base="1.2")
def main(cfg):
    """Any-to-any cotrain entry."""
    validate_train_config(cfg)
    run_name = str(cfg.run_name) or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.root_dir is None:
        cfg.root_dir = get_root_dir()
    setup_wandb_tmp_dirs()

    accelerator = create_accelerator(cfg)
    suppress_non_main_prints(accelerator)
    log_main("================== FINAL CONFIG (cotrain) ==================", accelerator=accelerator)
    log_main(OmegaConf.to_yaml(cfg), accelerator=accelerator)
    device = accelerator.device

    # Per-dataset instances + sample weights.
    per_ds = [
        build_cotrain_dataset(d, cfg.forward_dynamics, cfg.augmentations)
        for d in cfg.datasets
    ]
    per_ds_weights = [float(d.weight) for d in cfg.datasets]

    # When a type has multiple train instances with their own normalizers
    # (e.g. realyam_train + realyam_play_train), compute one shared normalizer
    # over the union and apply to all. Otherwise each instance's actions get
    # normalized to its own scale and the action head sees mixed targets.
    _share_normalizer_within_type(per_ds, cfg.datasets, accelerator)

    # One normalizer per dataset type, used by val datasets of the same type
    # and baked into the checkpoint keyed by dataset type. Inference picks the
    # entry by type (default "realyam", overridable via --normalizer-type).
    type_normalizers = _normalizers_by_type(per_ds, cfg.datasets)
    normalizer_stats_dict = (
        {t: nrm.to_dict() for t, nrm in type_normalizers.items()}
        if type_normalizers else None
    )
    _, _, universal, partial = derive_active_modalities(
        cfg.datasets, cfg.forward_dynamics.modalities,
    )
    log_main(
        f"[cotrain] {len(per_ds)} datasets; universal={universal}; partial={partial}",
        accelerator=accelerator,
    )

    steps_per_epoch = cfg.train_steps_per_epoch or (sum(len(d) for d in per_ds) // cfg.batch_size)
    # DataLoader yields micro-batches of size gpu_max_bs; accelerator.accumulate
    # collects (batch_size / gpu_max_bs) of them per optimizer step. Sampler draws
    # train_steps_per_epoch * batch_size samples per epoch — i.e. one weighted-replacement
    # draw per forward pass.
    train_loader = build_cotrain_dataloader(
        per_dataset_instances=per_ds,
        per_dataset_weights=per_ds_weights,
        universal_modalities=universal,
        partial_modalities=partial,
        batch_size=cfg.gpu_max_bs,
        num_samples=steps_per_epoch * cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False,
    )
    val_loader = _build_val_loader(cfg, universal, partial, type_normalizers)

    model = FloMo(cfg).to(device)
    model.setup_model()
    _install_motion_token_stats(model, cfg, per_ds, accelerator)

    optim_cfg = cfg.optim
    optimizer = get_optim(
        model, lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay,
        betas=optim_cfg.adam_betas,
    )
    scheduler = get_lr_scheduler(
        optimizer, lr_schedule=optim_cfg.lr_schedule,
        warmup_steps=optim_cfg.warmup_steps,
        decay_steps=optim_cfg.decay_steps,
        eta_min=optim_cfg.eta_min,
    )
    scaler = getattr(accelerator, "scaler", None)
    lora_enabled = (cfg.forward_dynamics.wan.finetune_mode == "lora")

    checkpoint_dir = get_checkpoint_dir(stage="forward_dynamics", run_name=run_name, resume=cfg.resume)
    log_main(f"Checkpoint dir: {checkpoint_dir}", accelerator=accelerator)

    if cfg.resume and cfg.checkpoint is None:
        cfg.checkpoint = latest_checkpoint_from_dir(checkpoint_dir)
        if cfg.checkpoint is not None:
            log_main(f"[resume] auto-resume from {cfg.checkpoint}", accelerator=accelerator)

    if cfg.compile:
        log_main("[compile] wrapping model with torch.compile (first step will be slow)", accelerator=accelerator)
        model = safe_torch_compile(model)

    train_steps = 0
    val_steps = 0
    wandb_run_id = None
    samples_seen_ckpt = None
    first_epoch_remaining = None
    if cfg.checkpoint is not None:
        init_only = not cfg.resume
        model, optimizer, scheduler, scaler, ckpt_info = load_checkpoint(
            cfg.checkpoint, model, optimizer, scheduler, scaler,
            skip_optimizer=init_only or cfg.resume_skip_optimizer,
        )
        if init_only:
            start_epoch = 1
            log_main(
                f"[init] loaded model weights from {cfg.checkpoint}; starting a fresh run",
                accelerator=accelerator,
            )
        else:
            train_steps = ckpt_info["train_steps"]
            val_steps = ckpt_info["val_steps"]
            # Resume mid-epoch; pass the global (unprepared-loader) grad_accum.
            start_epoch, first_epoch_remaining = compute_mid_epoch_resume(
                train_steps, train_loader, cfg.batch_size // cfg.gpu_max_bs, ckpt_info,
            )
            if first_epoch_remaining == 0:
                start_epoch, first_epoch_remaining = start_epoch + 1, None
            wandb_run_id = ckpt_info["wandb_run_id"]
            samples_seen_ckpt = ckpt_info["samples_seen"]
            # When scheduler is None, the saved (possibly mid-decay) lr sticks; push cfg.optim.lr back.
            if scheduler is None:
                for pg in optimizer.param_groups:
                    pg["lr"] = float(cfg.optim.lr)
                log_main(f"[resume] scheduler disabled; lr -> {cfg.optim.lr}", accelerator=accelerator)
    else:
        start_epoch = 1

    if cfg.wandb_run_id is not None:
        wandb_run_id = cfg.wandb_run_id

    model, optimizer, train_loader, scheduler, val_loader = prepare_training_components(
        accelerator, model, optimizer, train_loader, scheduler=scheduler, val_loader=val_loader,
    )
    register_grad_comm_hook(model, accelerator, cfg.grad_comm)

    if is_main_process(accelerator):
        wandb.init(
            entity=cfg.wandb_entity, project=cfg.wandb_project, group=cfg.wandb_group,
            name=run_name,
            id=wandb_run_id,
            resume="allow" if wandb_run_id is not None else None,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            mode='disabled' if not cfg.use_wandb else 'online',
            allow_val_change=True,
        )
        wandb.define_metric("train_steps")
        wandb.define_metric("val_steps")
        wandb.define_metric("train/*", step_metric="train_steps")
        wandb.define_metric("val/loss_avg_by_train*", step_metric="train_steps")
        wandb.define_metric("val/*", step_metric="val_steps")
        wandb.define_metric("viz/train_*", step_metric="train_steps")
        wandb.define_metric("viz/val_*", step_metric="val_steps")
        wandb.define_metric("skipped/*", step_metric="train_steps")

    broken_log = BrokenSampleLog(checkpoint_dir, accelerator.process_index)

    num_inf_steps = int(cfg.forward_dynamics.wan.num_inference_steps)
    interval = int(cfg.log_train_video_interval)
    # Track global per-dataset draws for data_pass metrics and checkpoint resume.
    ds_keys = [str(d.name) for d in cfg.datasets]
    ds_sizes = [float(len(p)) for p in per_ds]
    samples_seen = list(samples_seen_ckpt) if samples_seen_ckpt is not None else [0.0] * len(ds_keys)
    # New dataset slots start at zero; discard removed slots on resume.
    samples_seen = (samples_seen + [0.0] * len(ds_keys))[:len(ds_keys)]
    inner = unwrap_model_for_checkpoint(model)
    pd_keys = [(str(d.name), m) for d in cfg.datasets for m in inner.active_outputs]

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        model.train()
        t_step = time.perf_counter()
        mtr = _MicroBatchMeans(pd_keys, ds_keys)
        epoch_budget = first_epoch_remaining if epoch == start_epoch else None
        epoch_opt_steps = 0
        for batch in tqdm(segfault_resilient_iter(train_loader, accelerator),
                          desc=f"cotrain ep{epoch}", total=len(train_loader),
                          disable=not is_main_process(accelerator)):
            try:
                broken_log.ingest(batch.pop("_broken_samples", []), phase="train")
                batch = cotrain_batch_to_device(batch, device, non_blocking=True)
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        _pred, loss = model(batch)
                    accelerator.backward(loss)
                    grad_norm = None
                    grad_norms = {}
                    if accelerator.sync_gradients:
                        # Measure synchronized gradients before clipping. DDP averages
                        # across ranks; bf16 needs no GradScaler correction.
                        if cfg.logging.log_grad_norms and is_main_process(accelerator):
                            grad_norms = log_module_gradients(
                                model, train_steps, modules_to_monitor=list(cfg.logging.modules_to_monitor)
                            )
                        if optim_cfg.clip_grad > 0:
                            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=optim_cfg.clip_grad)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if accelerator.sync_gradients and scheduler is not None:
                        scheduler.step()
                mtr.add(loss, inner.last_train_info, batch["dataset_name"])
                if accelerator.sync_gradients:
                    train_steps += 1
                    epoch_opt_steps += 1
                    ckpt_every = cfg.train_steps_per_epoch // cfg.ckpt_saves_per_epoch if cfg.ckpt_saves_per_epoch > 0 else 0
                    if ckpt_every > 0 and train_steps % ckpt_every == 0:
                        save_training_checkpoint(
                            os.path.join(checkpoint_dir, "latest.pt"),
                            epoch, cfg, model, optimizer, scaler,
                            train_loss=loss.item(), val_loss=None, scheduler=scheduler,
                            train_steps=train_steps, val_steps=val_steps, accelerator=accelerator,
                            steps_per_epoch=steps_per_epoch,
                            lora_enabled=lora_enabled,
                            normalizer_stats=normalizer_stats_dict,
                            samples_seen=list(samples_seen),
                        )
                    avg_loss, avg_breakdown, avg_pd_breakdown, ds_counts = mtr.flush(accelerator)
                    for i, k in enumerate(ds_keys):
                        samples_seen[i] += ds_counts[k]
                    if is_main_process(accelerator):
                        now = time.perf_counter()
                        log_dict = {
                            "train_steps": train_steps,
                            "train/loss": avg_loss,
                            "train/it_per_sec": 1.0 / max(now - t_step, 1e-9),
                            "train/epoch": epoch,
                            **{f"train/data_pass/{k}": samples_seen[i] / ds_sizes[i]
                               for i, k in enumerate(ds_keys)},
                            **{f"train/loss/{k}": v for k, v in avg_breakdown.items()},
                            **{f"train/loss/{ds}/{mod}": v for (ds, mod), v in avg_pd_breakdown.items()},
                            **broken_log.counter_log(),
                        }
                        if scheduler is not None:
                            log_dict["train/lr"] = scheduler.get_last_lr()[0]
                        if grad_norm is not None:
                            log_dict["train/grad_norm"] = float(grad_norm)
                        log_dict.update({f"grad_norms/{k}": v for k, v in grad_norms.items()})
                        t_step = now
                        wandb_log(log_dict, accelerator)
                    if cfg.generate_video and interval > 0 and (train_steps % interval == 0):
                        with accelerator.autocast():
                            pred_latents = inner.infer(batch, num_inf_steps)
                        per_mod = _ar_metrics_from_pred_latents(inner, batch, pred_latents)
                        avg_ar = _reduce_ar(per_mod, inner, accelerator)
                        if is_main_process(accelerator):
                            _log_pred_viz(model, batch, train_steps, accelerator, num_inf_steps,
                                          phase="train", pred_latents=pred_latents)
                            _log_cond_img_viz(model, batch, train_steps, accelerator, phase="train")
                            ar_log = {"train_steps": train_steps,
                                      **{f"train/ar/{name}/{k}": v for (name, k), v in avg_ar.items()}}
                            if len(ar_log) > 1:
                                wandb_log(ar_log, accelerator)
                    if epoch_budget is not None and epoch_opt_steps >= epoch_budget:
                        break
            except RuntimeError as e:
                if "DataLoader worker" not in str(e):
                    raise
                log_main(f"[recover] worker died mid-step: {e}; zeroing grads and continuing",
                         accelerator=accelerator)
                optimizer.zero_grad(set_to_none=True)
                continue

        if val_loader is not None and cfg.eval_interval > 0 and (epoch % cfg.eval_interval == 0):
            val_steps = _validate(model, val_loader, accelerator, cfg, num_inf_steps, epoch, train_steps, val_steps, broken_log)

        save_training_checkpoint(
            os.path.join(checkpoint_dir, "latest.pt"),
            epoch, cfg, model, optimizer, scaler,
            train_loss=loss.item(), val_loss=None, scheduler=scheduler,
            train_steps=train_steps, val_steps=val_steps, accelerator=accelerator,
            steps_per_epoch=steps_per_epoch,
            lora_enabled=lora_enabled,
            normalizer_stats=normalizer_stats_dict,
            samples_seen=list(samples_seen),
        )
        if epoch == 1 or epoch % cfg.save_interval == 0:
            save_training_checkpoint(
                os.path.join(checkpoint_dir, f"{epoch}.pt"),
                epoch, cfg, model, optimizer, scaler,
                train_loss=loss.item(), val_loss=None, scheduler=scheduler,
                train_steps=train_steps, val_steps=val_steps, accelerator=accelerator,
                steps_per_epoch=steps_per_epoch,
                lora_enabled=lora_enabled,
                normalizer_stats=normalizer_stats_dict,
                samples_seen=list(samples_seen),
            )


if __name__ == "__main__":
    main()
