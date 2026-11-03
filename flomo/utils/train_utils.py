"""Training and checkpoint helpers for the FloMo entrypoint."""

import functools
import multiprocessing.process
import os
import atexit
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from omegaconf import OmegaConf

import wandb
import math


def unwrap_model_for_checkpoint(model):
    # Accelerate's unwrap_model() crashes on DDP(OptimizedModule) with KeyError('_orig_mod').
    # Unwrap DDP (.module) before torch.compile (._orig_mod) manually.
    while hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def setup_wandb_tmp_dirs() -> str:
    """Return a local WandB directory keyed by SLURM_JOBID (or "local").

    Keeps WandB cache/media staging under /tmp instead of the home directory,
    and redirects TMPDIR into this directory so media files outlive external
    /tmp cleanup until the run exits.
    """
    job_id = os.environ.get("SLURM_JOBID", "local")
    wandb_base = f"/tmp/wandb_{job_id}"
    tmp_dir = os.path.join(wandb_base, "tmp")
    for sub in ("cache", "data", "dir", "tmp"):
        os.makedirs(os.path.join(wandb_base, sub), exist_ok=True)
    os.environ["WANDB_CACHE_DIR"] = os.path.join(wandb_base, "cache")
    os.environ["WANDB_DATA_DIR"] = os.path.join(wandb_base, "data")
    os.environ["WANDB_DIR"] = os.path.join(wandb_base, "dir")
    os.environ["TMPDIR"] = tmp_dir
    # tempfile caches the resolved tempdir, so update the cache directly.
    tempfile.tempdir = tmp_dir
    multiprocessing.process.current_process()._config.pop("tempdir", None)
    atexit.register(shutil.rmtree, wandb_base, ignore_errors=True)
    return wandb_base


def get_optim(
    model,
    lr,
    weight_decay,
    betas,
):
    """AdamW over trainable parameters."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        trainable, lr=lr, weight_decay=weight_decay, betas=betas, fused=True
    )


def get_lr_scheduler(
    optimizer,
    lr_schedule,
    warmup_steps: Optional[int] = None,
    decay_steps: Optional[int] = None,
    eta_min: float = 0.0,
):
    """Schedule LR by optimizer updates; None disables scheduling.

    "linear" warms up then holds LR. "cosine_with_warmup" adds cosine decay
    to eta_min. Required warmup/decay step counts must be positive.
    """
    if lr_schedule is None:
        return None
    elif lr_schedule == "linear":
        if warmup_steps is None or int(warmup_steps) <= 0:
            raise ValueError("linear lr_schedule requires positive 'warmup_steps' to be set in the config")
        warmup_iters = int(warmup_steps)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
        print('LR Scheduler: Linear with warmup for {} steps'.format(warmup_iters))
        return warmup_scheduler
    elif lr_schedule == "cosine_with_warmup":
        if warmup_steps is None or decay_steps is None or int(warmup_steps) <= 0 or int(decay_steps) <= 0:
            raise ValueError("cosine_with_warmup lr_schedule requires positive 'warmup_steps' and 'decay_steps' to be set in the config")
        warmup_iters = int(warmup_steps)
        decay_iters = int(decay_steps)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, decay_iters), eta_min=float(eta_min))
        print('LR Scheduler: Cosine with warmup for {} warmup steps, {} decay steps, eta_min={}'.format(warmup_iters, decay_iters, eta_min))
        return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iters])

    raise ValueError(f"Unknown lr_schedule: {lr_schedule!r}. Supported: 'linear', 'cosine_with_warmup'.")


def segfault_resilient_iter(loader, accelerator):
    """Rebuild failed DataLoader workers on every rank together.

    Accelerate broadcasts RNG state in iter(loader), so independent rebuilds
    desynchronize ranks. All-reduce (died, stop) each batch to coordinate.

    Callers must also catch RuntimeError matching "DataLoader worker" around
    the loop body: SIGCHLD can raise asynchronously outside next().
    """
    from flomo.utils.distributed import log_main
    it = iter(loader)
    use_dist = accelerator.num_processes > 1
    while True:
        died = 0
        stop = 0
        batch = None
        try:
            batch = next(it)
        except StopIteration:
            stop = 1
        except RuntimeError as e:
            if "worker" not in str(e):
                raise
            log_main(
                f"[recover] worker died on rank {accelerator.process_index} in next(): {e}",
                accelerator=accelerator,
            )
            died = 1
        if use_dist:
            status = torch.tensor([died, stop], device=accelerator.device, dtype=torch.int64)
            torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MAX)
            any_died = status[0].item() > 0
            any_stop = status[1].item() > 0
        else:
            any_died = bool(died)
            any_stop = bool(stop)
        if any_died:
            log_main("[recover] rebuilding loader on all ranks", accelerator=accelerator)
            loader._iterator = None
            it = iter(loader)
            continue
        if any_stop:
            return
        yield batch


def get_checkpoint_dir(stage, run_name, resume=False):
    """Select checkpoints/{stage}/{run_name}[_id].

    Resume the highest existing id, or allocate the next id for a new run.
    """
    assert stage in ["motion_tokenizer", "forward_dynamics", "inverse_dynamics"]
    run_name = str(run_name)
    stage_dir = os.path.join("checkpoints", stage)
    os.makedirs(stage_dir, exist_ok=True)

    matching_runs = [f for f in os.listdir(stage_dir) if f.startswith(run_name)]
    matching_run_ids = []

    for f in matching_runs:
        parts = f.split("_")
        if f == run_name:  # Handle case where base run_name exists
            matching_run_ids.append(0)
        elif len(parts) > 1 and parts[-1].isdigit():
            matching_run_ids.append(int(parts[-1]))

    matching_run_ids.sort()

    if matching_run_ids:
        if resume:
            run_id = matching_run_ids[-1]
        else:
            run_id = matching_run_ids[-1] + 1
    else:
        run_id = 0

    checkpoint_dir = os.path.join(stage_dir, f"{run_name}_{run_id}" if run_id > 0 else run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    return checkpoint_dir


def latest_checkpoint_from_dir(checkpoint_dir):
    """
    Returns the newest checkpoint path in a given directory.
    """
    checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt")
    ]
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}, nothing to resume")
        return None
    return max(checkpoints, key=os.path.getmtime)


def _prune_checkpoints(checkpoint_dir, predicate, keep: int = 3):
    """Keep only the newest `keep` checkpoints matching `predicate(filename)`. keep<0 disables pruning."""
    if keep < 0:
        return
    try:
        files = [
            os.path.join(checkpoint_dir, f)
            for f in os.listdir(checkpoint_dir)
            if predicate(f)
        ]
    except FileNotFoundError:
        return

    if len(files) <= keep:
        return

    files.sort(key=os.path.getmtime, reverse=True)
    for old in files[keep:]:
        try:
            os.remove(old)
            print(f"Pruned old checkpoint {old}")
        except Exception as e:
            print(f"Failed to prune checkpoint {old}: {e}")


def prune_step_checkpoints(checkpoint_dir, keep: int = 3):
    """Keep only the newest `step_*.pt` checkpoints in a directory."""
    _prune_checkpoints(
        checkpoint_dir,
        lambda f: f.startswith("step_") and f.endswith(".pt"),
        keep,
    )


def prune_epoch_checkpoints(checkpoint_dir, keep: int = 3):
    """Keep only the newest `{N}.pt` epoch checkpoints (excludes `latest.pt`)."""
    _prune_checkpoints(
        checkpoint_dir,
        lambda f: f.endswith(".pt") and f[:-3].isdigit(),
        keep,
    )


def _load_checkpoint_file(path):
    try:
        return torch.load(path, weights_only=False)
    except Exception as e:
        print(f"Failed to load checkpoint {path}: {e}")
        return None


def _atomic_torch_save(payload, path: str):
    """
    Save to a temp file in the target directory and publish via atomic rename.

    This avoids exposing partially written checkpoint files during preemption.
    """
    path = str(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
    )
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Frozen pretrained sub-modules: reloaded from canonical .pth paths in
# FloMo.__init__, so they don't belong in our checkpoint deltas.
_FROZEN_ENCODER_PREFIXES = ("text_encoder.", "vae.")


def save_checkpoint(
    checkpoint_path,
    epoch,
    cfg,
    model,
    optimizer,
    scaler=None,
    train_loss=None,
    val_loss=None,
    scheduler=None,
    train_steps=None,
    val_steps=None,
    steps_per_epoch=None,
    lora_enabled=False,
    **extra_checkpoint_fields,
):
    """
    Saves the model and optimizer state.
    For LoRA training, only saves trainable parameters to reduce checkpoint size.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    if lora_enabled:
        model_state = {k: v for k, v in model.state_dict().items()
                      if any(k.startswith(n) for n, p in model.named_parameters() if p.requires_grad)}
        print(f"LoRA checkpoint: saving {len(model_state)} trainable parameter tensors (original: {len(model.state_dict())})")
    else:
        model_state = model.state_dict()

    # FloMo reloads frozen encoders from pretrained weights; omit them here.
    # Full checkpoints permit missing keys only under these encoder prefixes.
    n_before = len(model_state)
    model_state = {k: v for k, v in model_state.items() if not k.startswith(_FROZEN_ENCODER_PREFIXES)}
    if n_before != len(model_state):
        print(f"Checkpoint: stripped {n_before - len(model_state)} frozen encoder tensors "
              f"(prefixes {_FROZEN_ENCODER_PREFIXES})")

    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "config": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "wandb_run_id": wandb.run.id,
        "lora_enabled": lora_enabled,  # Flag to indicate this is a LoRA-only checkpoint
        "train_steps": train_steps,
        "val_steps": val_steps,
        "steps_per_epoch": steps_per_epoch,
    }
    checkpoint.update(extra_checkpoint_fields)
    _atomic_torch_save(checkpoint, checkpoint_path)
    print("Saved checkpoint at ", checkpoint_path)

    basename = os.path.basename(checkpoint_path)
    if basename.startswith("step_"):
        prune_step_checkpoints(os.path.dirname(checkpoint_path), keep=cfg.keep_ckpts)
    elif basename.endswith(".pt") and basename[:-3].isdigit():
        prune_epoch_checkpoints(os.path.dirname(checkpoint_path), keep=cfg.keep_ckpts)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None,
                    skip_optimizer=False):
    """Load a checkpoint with automatic fallback to older files in the same dir.

    `skip_optimizer=True` loads model weights only (weights-only init, or a
    resume whose trainable-param set changed). Returns the raw checkpoint dict
    last; callers index progress fields only when resuming.
    """
    checkpoint = _load_checkpoint_file(checkpoint_path)

    if checkpoint is None:
        base_dir = os.path.dirname(checkpoint_path)
        files = sorted(
            [f for f in os.listdir(base_dir) if f.endswith(".pt")],
            key=lambda f: os.path.getmtime(os.path.join(base_dir, f)),
            reverse=True,
        )
        for fname in files:
            candidate = os.path.join(base_dir, fname)
            if candidate == checkpoint_path:
                continue
            checkpoint = _load_checkpoint_file(candidate)
            if checkpoint is not None:
                print(f"Successfully fell back to previous checkpoint {candidate}")
                checkpoint_path = candidate
                break

    if checkpoint is None:
        raise RuntimeError(f"Unable to load any checkpoint from {checkpoint_path}")

    if any(k.startswith("_orig_mod.") for k in checkpoint["model"].keys()):
        checkpoint["model"] = unwrap_compiled_state_dict(checkpoint["model"])
    checkpoint["model"] = rename_legacy_state_keys(checkpoint["model"])

    load_target = unwrap_model_for_checkpoint(model)

    is_lora_checkpoint = checkpoint["lora_enabled"]

    if is_lora_checkpoint:
        print("Loading LoRA checkpoint (trainable parameters only)")
        load_result = load_target.load_state_dict(checkpoint["model"], strict=False)
        if load_result.missing_keys:
            print(f"  Expected missing keys (frozen params): {len(load_result.missing_keys)} keys")
        if load_result.unexpected_keys:
            print(f"  WARNING: Unexpected keys: {load_result.unexpected_keys}")
    else:
        # Frozen encoders reload at model initialization. Reject other missing
        # keys and all unexpected keys as architecture mismatches.
        load_result = load_target.load_state_dict(checkpoint["model"], strict=False)
        unexpected_missing = [k for k in load_result.missing_keys if not k.startswith(_FROZEN_ENCODER_PREFIXES)]
        if unexpected_missing:
            raise RuntimeError(
                f"load_checkpoint: {len(unexpected_missing)} unexpected missing keys "
                f"(not under {_FROZEN_ENCODER_PREFIXES}); first 10: {unexpected_missing[:10]}"
            )
        n_skipped = len(load_result.missing_keys) - len(unexpected_missing)
        if n_skipped:
            print(f"  Skipped {n_skipped} frozen encoder keys (reloaded from pretrained .pth at init)")
        if load_result.unexpected_keys:
            raise RuntimeError(
                f"load_checkpoint: {len(load_result.unexpected_keys)} unexpected keys in checkpoint; "
                f"first 10: {load_result.unexpected_keys[:10]}"
            )

    if skip_optimizer:
        print("[load_checkpoint] skip_optimizer=True: not restoring optimizer/scheduler/scaler state")
    else:
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint["scaler"] is not None:
            scaler.load_state_dict(checkpoint["scaler"])

    print(f"Loaded checkpoint from {checkpoint_path}")

    # Progress fields are read by the resume path only; weights-only init must
    # not require them.
    return model, optimizer, scheduler, scaler, checkpoint


def rename_legacy_state_keys(state):
    """Legacy alias: pre-rename checkpoints use cum_disp buffer key names."""
    return {k.replace("cum_disp", "cumulative_disp"): v for k, v in state.items()}


def compute_mid_epoch_resume(train_steps, train_loader, grad_accum, checkpoint_info):
    """Map saved epoch progress to the current loader size, allowing world-size changes.

    Return (start_epoch, remaining_optimizer_steps); None means a full next epoch.
    """
    epoch = checkpoint_info["epoch"]
    new_steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    saved_steps_per_epoch = checkpoint_info["steps_per_epoch"]
    completed_fraction = (train_steps % saved_steps_per_epoch) / saved_steps_per_epoch
    completed_new_steps = round(completed_fraction * new_steps_per_epoch)
    if completed_new_steps == 0:
        return epoch + 1, None
    return epoch, new_steps_per_epoch - completed_new_steps



def unwrap_compiled_state_dict(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        # remove leading "_orig_mod." if present
        if k.startswith("_orig_mod."):
            new_sd[k[len("_orig_mod.") :]] = v
        else:
            new_sd[k] = v
    return new_sd


@functools.lru_cache(maxsize=1)
def patch_torch_compile_nvcc_probe() -> bool:
    """
    Work around a PyTorch 2.9.x Inductor bug where generating debug repros
    probes `nvcc --version` and lets PermissionError abort torch.compile.
    """
    try:
        from torch._dynamo import debug_utils
        from torch._dynamo.repro import after_aot
    except Exception:
        return False

    def wrap_probe(fn):
        if getattr(fn, "_flomo_nvcc_probe_safe", False):
            return fn

        @functools.wraps(fn)
        def wrapped():
            try:
                return fn()
            except PermissionError as exc:
                return f"# nvcc probe skipped due to permission error: {exc}\n"

        wrapped._flomo_nvcc_probe_safe = True
        return wrapped

    debug_utils._cuda_system_info_comment = wrap_probe(debug_utils._cuda_system_info_comment)
    after_aot._cuda_system_info_comment = wrap_probe(after_aot._cuda_system_info_comment)
    return True


def safe_torch_compile(module: torch.nn.Module, *args, **kwargs) -> torch.nn.Module:
    patch_torch_compile_nvcc_probe()
    return torch.compile(module, *args, **kwargs)


def get_root_dir():
    file_path = Path(__file__).resolve()
    root_dir = file_path.parents[2]
    print(f"Using root_dir: {root_dir}")

    return root_dir
