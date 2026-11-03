import builtins
import math
import os
from datetime import timedelta

import wandb

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
    bf16_compress_hook,
    bf16_compress_wrapper,
)
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import (
    PowerSGDState,
    powerSGD_hook,
)

from flomo.utils.train_utils import (
    save_checkpoint,
    unwrap_model_for_checkpoint,
)


_ORIGINAL_PRINT = builtins.print
_PRINT_SUPPRESSED = False

def suppress_non_main_prints(accelerator):
    """Suppress duplicate `print()` output on non-main ranks while keeping stderr intact."""
    global _PRINT_SUPPRESSED
    if is_main_process(accelerator) or _PRINT_SUPPRESSED:
        return False

    def _quiet_print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if force:
            return _ORIGINAL_PRINT(*args, **kwargs)

    _PRINT_SUPPRESSED = True
    builtins.print = _quiet_print
    return True




def create_accelerator(cfg):
    # torchrun sets WORLD_SIZE; single-GPU direct-python runs have no WORLD_SIZE (defaults to 1).
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    gradient_accumulation_steps = max(1, math.ceil(cfg.batch_size / (cfg.gpu_max_bs * world_size)))
    mixed_precision = "bf16" if cfg.optim.amp else "no"
    # find_unused_parameters=True needed when batches conditionally skip output modalities
    # (e.g. cotrain: action heads get no gradient on egoverse-only batches).
    # gradient_as_bucket_view=True saves ~1 grad copy; safe with set_to_none zero_grad.
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=cfg.ddp_find_unused_parameters,
        gradient_as_bucket_view=True,
    )
    # PyTorch default c10d store get / PG collective timeout is 10 min. Cold
    # cotrain runs do per-dataset flow_pct stat compute on rank 0 inside a
    # wait_for_everyone barrier, which can exceed that window on a fresh FS
    # snapshot. 30 min also covers single-rank transient stalls on multi-node
    # TCP setups (e.g. blosc decompression retries on a slow network FS).
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))
    return Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs, pg_kwargs],
    )


def register_grad_comm_hook(model, accelerator, mode):
    """Register a DDP gradient-communication hook on the prepared model.

    mode: none | bf16 | powersgd | bf16_powersgd. No-op for single-process runs
    (no DDP wrapper). `model` must be the post-`accelerator.prepare` DDP model.
    """
    if mode == "none" or accelerator.num_processes == 1:
        return
    if mode == "bf16":
        model.register_comm_hook(state=None, hook=bf16_compress_hook)
        return
    # PowerSGD warms up with a few vanilla all-reduces before low-rank kicks in.
    state = PowerSGDState(
        process_group=None,
        matrix_approximation_rank=1,
        start_powerSGD_iter=10,
    )
    hook = powerSGD_hook if mode == "powersgd" else bf16_compress_wrapper(powerSGD_hook)
    model.register_comm_hook(state, hook)


def is_main_process(accelerator):
    return accelerator.is_main_process


def log_main(*args, accelerator, **kwargs):
    if is_main_process(accelerator):
        print(*args, **kwargs)


def wandb_log(log_dict, accelerator):
    if is_main_process(accelerator) and wandb.run is not None:
        wandb.log(log_dict)


def prepare_training_components(
    accelerator,
    model,
    optimizer,
    train_loader,
    scheduler=None,
    val_loader=None,
):
    optional = [x for x in [scheduler, val_loader] if x is not None]
    prepared = accelerator.prepare(model, optimizer, train_loader, *optional)
    model, optimizer, train_loader = prepared[:3]
    idx = 3
    if scheduler is not None:
        scheduler = prepared[idx]
        idx += 1
    if val_loader is not None:
        val_loader = prepared[idx]
    return model, optimizer, train_loader, scheduler, val_loader

def save_training_checkpoint(
    checkpoint_path,
    epoch,
    cfg,
    traj_model,
    optimizer,
    scaler,
    train_loss,
    val_loss,
    scheduler,
    train_steps,
    val_steps,
    accelerator,
    lora_enabled,
    steps_per_epoch=None,
    **extra_checkpoint_fields,
):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            checkpoint_path,
            epoch,
            cfg,
            unwrap_model_for_checkpoint(traj_model),
            optimizer,
            scaler,
            train_loss=train_loss,
            val_loss=val_loss,
            scheduler=scheduler,
            train_steps=train_steps,
            val_steps=val_steps,
            steps_per_epoch=steps_per_epoch,
            lora_enabled=lora_enabled,
            **extra_checkpoint_fields,
        )
    accelerator.wait_for_everyone()
