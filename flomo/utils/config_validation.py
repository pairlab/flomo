import math
import os
from typing import Any

from flomo.loaders.cotrain_loader import (
    base_modality_name,
    derive_active_modalities,
    expand_modality_order,
)

from flomo.utils.constants import (
    MULTI_VIEW_TRACKERS,
    POSSIBLE_OUTPUTS,
    SPATIAL_TRACKER_OUTPUTS,
    TRACKS_FORMAT,
    WAN_MODEL_SIZES,
    VGGT_OUTPUTS,
    resolve_wan_size,
)


PREPROCESS_MODES = {
    "egoverse": ("tracks", "video", "text", "text_emb", "meta", "vae_latent_h16"),
    "molmoact2": (
        "tracks",
        "video",
        "text",
        "text_emb",
        "vae_latent",
        "vae_latent_h16",
        "actions",
    ),
    "realyam": ("tracks", "text_emb", "vae_latent_h16"),
    "robotwin": ("tracks", "text_emb", "vae_latent_h16"),
}

TRACKER_OUTPUTS = {
    "spatialtracker": tuple(SPATIAL_TRACKER_OUTPUTS),
    "vggt": tuple(VGGT_OUTPUTS),
}

SPLITS = ("train", "val", "all")
GRAD_COMM_MODES = ("none", "bf16", "powersgd", "bf16_powersgd")
FINETUNE_MODES = ("full", "lora")
INIT_WEIGHTS = ("pretrained", "scratch")
LR_SCHEDULES = ("linear", "cosine_with_warmup")
DATASET_TYPES = ("egoverse", "molmoact2", "realyam", "robotwin")
WAN_VAE_STRIDE = 16
WAN_VAE_T_STRIDE = 4


def validate_train_config(cfg: Any) -> None:
    _validate_positive_int("batch_size", cfg.batch_size)
    _validate_positive_int("gpu_max_bs", cfg.gpu_max_bs)
    _validate_train_batching(cfg)
    _validate_positive_int("num_workers", cfg.num_workers)
    _validate_positive_int("num_epochs", cfg.num_epochs)
    if cfg.train_steps_per_epoch is not None:
        _validate_positive_int("train_steps_per_epoch", cfg.train_steps_per_epoch)
    _validate_nonnegative_int("eval_interval", cfg.eval_interval)
    _validate_positive_int("save_interval", cfg.save_interval)
    _validate_nonnegative_int("ckpt_saves_per_epoch", cfg.ckpt_saves_per_epoch)
    if cfg.train_steps_per_epoch is None and cfg.ckpt_saves_per_epoch != 0:
        raise ValueError(
            "ckpt_saves_per_epoch > 0 requires train_steps_per_epoch to be set."
        )
    _validate_nonnegative_int("log_train_video_interval", cfg.log_train_video_interval)
    _validate_nonnegative_int("val_ar_interval", cfg.val_ar_interval)
    if cfg.val_steps_per_epoch is not None:
        _validate_positive_int("val_steps_per_epoch", cfg.val_steps_per_epoch)

    if cfg.grad_comm not in GRAD_COMM_MODES:
        raise ValueError(f"grad_comm={cfg.grad_comm} is unsupported; choices={GRAD_COMM_MODES}.")

    if cfg.forward_dynamics.model != "wan":
        raise ValueError(f"forward_dynamics.model={cfg.forward_dynamics.model} is unsupported; use 'wan'.")
    _validate_wan_train_config(cfg.forward_dynamics.wan)
    _validate_optim_config(cfg.optim)
    _validate_train_dataset_contracts(cfg)


def validate_preprocess_config(dataset: str, cfg: Any) -> None:
    if dataset not in PREPROCESS_MODES:
        raise ValueError(f"Unknown preprocessing dataset '{dataset}'.")

    if cfg.mode not in PREPROCESS_MODES[dataset]:
        raise ValueError(
            f"{dataset} preprocessing supports modes={PREPROCESS_MODES[dataset]}; "
            f"got '{cfg.mode}'."
        )

    _validate_common(cfg)
    _validate_tracker_type(cfg.tracker_type)

    if cfg.mode == "tracks":
        validate_track_config(cfg)
        _validate_track_dataset_config(dataset, cfg)
    if cfg.mode == "actions":
        _validate_action_config(cfg)
    if cfg.mode in {"vae_latent", "vae_latent_h16"}:
        _validate_vae_config(dataset, cfg)
    if dataset == "egoverse":
        _validate_egoverse_config(cfg)


def _validate_wan_train_config(wan_cfg: Any) -> None:
    if wan_cfg.backbone != "bidirectional":
        raise ValueError(
            f"forward_dynamics.wan.backbone={wan_cfg.backbone} is unsupported; "
            "cotrain requires 'bidirectional'."
        )
    if wan_cfg.finetune_mode not in FINETUNE_MODES:
        raise ValueError(
            f"forward_dynamics.wan.finetune_mode={wan_cfg.finetune_mode} is "
            f"unsupported; choices={FINETUNE_MODES}."
        )
    if wan_cfg.init_weights not in INIT_WEIGHTS:
        raise ValueError(
            f"forward_dynamics.wan.init_weights={wan_cfg.init_weights} is "
            f"unsupported; choices={INIT_WEIGHTS}."
        )
    if wan_cfg.init_weights == "pretrained":
        size = resolve_wan_size(wan_cfg)
        if size != WAN_MODEL_SIZES["5B"]:
            raise ValueError(
                "forward_dynamics.wan.init_weights=pretrained requires the 5B "
                f"Wan shape, got {size}."
            )
    if wan_cfg.finetune_mode == "lora":
        _validate_lora_config(wan_cfg.lora)


def _validate_lora_config(lora_cfg: Any) -> None:
    _validate_positive_int("forward_dynamics.wan.lora.rank", lora_cfg.rank)
    _validate_positive_int("forward_dynamics.wan.lora.alpha", lora_cfg.alpha)
    if type(lora_cfg.dropout) not in (float, int) or not 0 <= lora_cfg.dropout < 1:
        raise ValueError(
            "forward_dynamics.wan.lora.dropout must be numeric in [0, 1), "
            f"got {lora_cfg.dropout}."
        )
    if not list(lora_cfg.target_modules):
        raise ValueError("forward_dynamics.wan.lora.target_modules must be non-empty.")


def _validate_optim_config(optim_cfg: Any) -> None:
    if optim_cfg.lr_schedule is not None and optim_cfg.lr_schedule not in LR_SCHEDULES:
        raise ValueError(
            f"optim.lr_schedule={optim_cfg.lr_schedule} is unsupported; "
            f"choices={LR_SCHEDULES} or null."
        )
    if optim_cfg.lr_schedule == "linear":
        _validate_positive_int("optim.warmup_steps", optim_cfg.warmup_steps)
    if optim_cfg.lr_schedule == "cosine_with_warmup":
        _validate_positive_int("optim.warmup_steps", optim_cfg.warmup_steps)
        _validate_positive_int("optim.decay_steps", optim_cfg.decay_steps)


def _validate_train_batching(cfg: Any) -> None:
    world = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    divisor = int(cfg.gpu_max_bs) * world
    if int(cfg.batch_size) % divisor != 0:
        effective = math.ceil(int(cfg.batch_size) / divisor) * divisor
        raise ValueError(
            f"batch_size={cfg.batch_size} must be divisible by gpu_max_bs * "
            f"WORLD_SIZE ({cfg.gpu_max_bs} * {world}); otherwise gradient "
            f"accumulation rounds the effective batch to {effective}."
        )


def _validate_train_dataset_contracts(cfg: Any) -> None:
    train_datasets = list(cfg.datasets)
    val_datasets = [] if cfg.val_datasets is None else list(cfg.val_datasets)
    all_datasets = train_datasets + val_datasets
    for ds in all_datasets:
        _validate_dataset_entry(ds, cfg)

    active_inputs, active_outputs, _universal, partial = derive_active_modalities(
        train_datasets, cfg.forward_dynamics.modalities,
    )
    order = expand_modality_order(
        cfg.forward_dynamics.cotrain.modality_order,
        active_inputs,
        active_outputs,
    )
    active = set(active_inputs) | set(active_outputs)
    if set(order) != active or len(order) != len(set(order)):
        raise ValueError(
            f"forward_dynamics.cotrain.modality_order expands to {order}, "
            f"which must exactly cover active modalities {sorted(active)} "
            "with no duplicates."
        )

    _validate_modality_geometry(cfg, all_datasets, active_inputs, active_outputs)
    _validate_augmentation_contract(cfg)
    _validate_ddp_contract(cfg, partial)
    _validate_val_pairing(cfg, active_outputs)


def _validate_dataset_entry(ds: Any, cfg: Any) -> None:
    fd = cfg.forward_dynamics
    if ds.type not in DATASET_TYPES:
        raise ValueError(f"dataset '{ds.name}': unknown type {ds.type!r}; choices={DATASET_TYPES}.")

    inputs = set(ds.inputs)
    outputs = set(ds.outputs)
    both = inputs & outputs
    if both:
        raise ValueError(f"dataset '{ds.name}': {sorted(both)} listed as both input and output.")
    for m in inputs | outputs:
        if m not in fd.modalities:
            raise ValueError(f"dataset '{ds.name}': modality {m!r} has no config entry.")
    for m in ("img", "text"):
        if m in outputs:
            raise ValueError(f"dataset '{ds.name}': {m!r} is conditioning-only and cannot be an output.")

    keys = set(_require_arg(ds, "keys_to_load"))
    if "motion_tokens" in outputs and "motion_tokens" not in keys:
        raise ValueError(f"dataset '{ds.name}': outputs include motion_tokens but keys_to_load lacks 'motion_tokens'.")
    if "action" in outputs and "actions" not in keys:
        raise ValueError(f"dataset '{ds.name}': outputs include action but keys_to_load lacks 'actions'.")
    if ("img" in inputs or "video" in inputs or "video" in outputs) and "vae_latent" not in keys:
        raise ValueError(
            f"dataset '{ds.name}': camera modalities are dispatched by keys_to_load='vae_latent'."
        )
    if "text" in inputs:
        if fd.cotrain.use_preprocessed_text_embs and "text_emb" not in keys:
            raise ValueError(
                f"dataset '{ds.name}': use_preprocessed_text_embs=true but keys_to_load lacks 'text_emb'."
            )
        if not fd.cotrain.use_preprocessed_text_embs and "text" not in keys:
            raise ValueError(
                f"dataset '{ds.name}': use_preprocessed_text_embs=false but keys_to_load lacks 'text'."
            )

    img_shape = list(_require_arg(ds, "img_shape"))
    vae_spatial = int(fd.wan.vae_spatial)
    if img_shape != [vae_spatial, vae_spatial]:
        raise ValueError(
            f"dataset '{ds.name}': img_shape={img_shape} must equal "
            f"[forward_dynamics.wan.vae_spatial, forward_dynamics.wan.vae_spatial] "
            f"= [{vae_spatial}, {vae_spatial}]."
        )

    interp = str(_require_arg(ds, "interp_method"))
    if interp == "none":
        true_h = int(_require_arg(ds, "true_horizon"))
        pred_h = int(_require_arg(ds, "track_pred_horizon"))
        if true_h != pred_h:
            raise ValueError(
                f"dataset '{ds.name}': interp_method='none' requires "
                f"track_pred_horizon={pred_h} to equal true_horizon={true_h}."
            )

    if "motion_tokens" in outputs:
        n = int(_require_arg(ds, "num_tracks"))
        side = math.isqrt(n)
        if side * side != n:
            raise ValueError(
                f"dataset '{ds.name}': num_tracks={n} must be a perfect square "
                "for the motion-token flow grid."
            )


def _validate_modality_geometry(
    cfg: Any,
    datasets: list[Any],
    active_inputs: list[str],
    active_outputs: list[str],
) -> None:
    fd = cfg.forward_dynamics
    active_bases = {base_modality_name(n) for n in set(active_inputs) | set(active_outputs)}
    horizons = {int(_require_arg(ds, "true_horizon")) for ds in datasets}
    if len(horizons) != 1:
        raise ValueError(f"datasets disagree on true_horizon: {sorted(horizons)}.")
    true_h = horizons.pop()

    vae_spatial = int(fd.wan.vae_spatial)
    lat = vae_spatial // WAN_VAE_STRIDE
    for name in ("img", "video", "motion_tokens"):
        if name not in active_bases:
            continue
        mod = fd.modalities[name]
        if int(mod.h_lat) != lat or int(mod.w_lat) != lat:
            raise ValueError(
                f"forward_dynamics.modalities.{name}.h_lat/w_lat="
                f"({mod.h_lat}, {mod.w_lat}) must equal "
                f"wan.vae_spatial // {WAN_VAE_STRIDE} = {lat}."
            )

    if {"video", "motion_tokens"} & active_bases and true_h % WAN_VAE_T_STRIDE != 0:
        raise ValueError(
            f"true_horizon={true_h} must be a multiple of {WAN_VAE_T_STRIDE} "
            "for Wan VAE video/motion-token targets."
        )

    if "video" in active_bases:
        want = true_h // WAN_VAE_T_STRIDE
        if any(base_modality_name(n) == "video" for n in active_inputs):
            want += 1
        if int(fd.modalities.video.t_frames) != want:
            raise ValueError(
                f"forward_dynamics.modalities.video.t_frames={fd.modalities.video.t_frames} "
                f"must be {want} for true_horizon={true_h}."
            )
    if "motion_tokens" in active_bases:
        want = 1 + true_h // WAN_VAE_T_STRIDE
        if int(fd.modalities.motion_tokens.t_frames) != want:
            raise ValueError(
                f"forward_dynamics.modalities.motion_tokens.t_frames="
                f"{fd.modalities.motion_tokens.t_frames} must be {want} for "
                f"true_horizon={true_h}."
            )
    if "action" in active_bases and int(fd.modalities.action.t_steps) != true_h:
        raise ValueError(
            f"forward_dynamics.modalities.action.t_steps={fd.modalities.action.t_steps} "
            f"must equal true_horizon={true_h}."
        )

    for name in active_outputs:
        base = base_modality_name(name)
        if base not in fd.loss_weights:
            raise ValueError(f"active output modality {name!r} has no loss weight.")


def _validate_augmentation_contract(cfg: Any) -> None:
    aug = cfg.augmentations
    aug_on = bool(aug.color_jitter or aug.gaussian_blur or aug.random_resized_crop)
    if not aug_on:
        return
    mods = cfg.forward_dynamics.modalities
    if bool(mods.img.use_preprocessed) or bool(mods.video.use_preprocessed):
        raise ValueError(
            "augmentations operate on raw pixels, so img/video use_preprocessed "
            "must be false when augmentations are enabled."
        )


def _validate_ddp_contract(cfg: Any, partial: list[str]) -> None:
    world = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if world > 1 and partial and not bool(cfg.ddp_find_unused_parameters):
        raise ValueError(
            f"multi-GPU cotrain with partial modalities {partial} requires "
            "ddp_find_unused_parameters=true."
        )


def _validate_val_pairing(cfg: Any, active_outputs: list[str]) -> None:
    if cfg.val_datasets is None:
        return
    if not any(base_modality_name(n) == "motion_tokens" for n in active_outputs):
        return
    if len(cfg.val_datasets) > len(cfg.datasets):
        raise ValueError(
            "motion-token stats pair val datasets to train datasets positionally; "
            f"got len(val_datasets)={len(cfg.val_datasets)} > len(datasets)={len(cfg.datasets)}."
        )


def _require_arg(ds: Any, key: str) -> Any:
    if key not in ds.args:
        raise ValueError(f"dataset '{ds.name}': args.{key} is required.")
    return ds.args[key]


def validate_track_config(cfg: Any) -> None:
    tracker = cfg.tracker_type
    _validate_tracker_type(tracker)

    requested = list(cfg.track_outputs)
    if not requested:
        raise ValueError("mode=tracks requires at least one track_outputs entry.")

    invalid = sorted(set(requested) - set(POSSIBLE_OUTPUTS))
    if invalid:
        raise ValueError(f"Invalid track_outputs={invalid}; choices={POSSIBLE_OUTPUTS}.")

    unsupported = sorted(set(requested) - set(TRACKER_OUTPUTS[tracker]))
    if unsupported:
        raise ValueError(
            f"tracker_type={tracker} cannot produce track_outputs={unsupported}; "
            f"allowed={list(TRACKER_OUTPUTS[tracker])}."
        )

    if bool(cfg.multi_view) and tracker not in MULTI_VIEW_TRACKERS:
        raise ValueError(f"tracker_type={tracker} does not support multi_view=true.")

    if "tracks_format" in cfg and cfg.tracks_format != TRACKS_FORMAT[tracker]:
        raise ValueError(
            f"tracks_format={cfg.tracks_format} is incompatible with "
            f"tracker_type={tracker}; expected {TRACKS_FORMAT[tracker]}."
        )

    _validate_positive_int("horizon", cfg.horizon)
    _validate_positive_int("n_tracks", cfg.n_tracks)
    _validate_positive_int("chunk_size", cfg.chunk_size)
    _validate_subsampling(cfg)

    if cfg.init_queries != "uniform":
        raise ValueError(f"init_queries={cfg.init_queries} is unsupported; use 'uniform'.")
    if cfg.reinit is not True:
        raise ValueError("Remaining trackers always reinitialize windows; set reinit=true.")
    if cfg.chunk_size <= cfg.horizon:
        raise ValueError(
            f"chunk_size={cfg.chunk_size} must be greater than horizon={cfg.horizon}."
        )
    if cfg.subsampling != 1 and tracker != "spatialtracker":
        raise ValueError(
            f"subsampling={cfg.subsampling} is only supported by spatialtracker, "
            f"not tracker_type={tracker}."
        )


def validate_track_chunk_overlap(dataset: str, cfg: Any) -> None:
    num_views = _track_num_views(dataset, cfg)
    min_chunk = cfg.horizon * num_views
    if cfg.chunk_size <= min_chunk:
        raise ValueError(
            f"chunk_size={cfg.chunk_size} must be greater than "
            f"horizon * num_views = {cfg.horizon} * {num_views}."
        )


def _validate_common(cfg: Any) -> None:
    _validate_positive_int("img_size", cfg.img_size)
    _validate_workers(cfg)
    if "split" in cfg and cfg.split not in SPLITS:
        raise ValueError(f"split={cfg.split} is unsupported; choices={SPLITS}.")
    if "val_ratio" in cfg:
        if type(cfg.val_ratio) not in (float, int) or not 0 <= cfg.val_ratio < 1:
            raise ValueError(f"val_ratio must be numeric in [0, 1), got {cfg.val_ratio}.")
    _validate_optional_positive_int(cfg, "max_episodes")
    _validate_optional_positive_int(cfg, "max_clips")
    _validate_optional_positive_int(cfg, "max_frames")
    _validate_optional_positive_int(cfg, "max_demos")


def _validate_workers(cfg: Any) -> None:
    _validate_positive_int("num_workers", cfg.num_workers)
    if type(cfg.worker_id) is not int:
        raise ValueError(f"worker_id must be an int, got {cfg.worker_id}.")
    if not 0 <= cfg.worker_id < cfg.num_workers:
        raise ValueError(
            f"worker_id={cfg.worker_id} must be in [0, {cfg.num_workers})."
        )
    if "sub_workers" in cfg:
        _validate_positive_int("sub_workers", cfg.sub_workers)
        if type(cfg.sub_id) is not int:
            raise ValueError(f"sub_id must be an int, got {cfg.sub_id}.")
        if not 0 <= cfg.sub_id < cfg.sub_workers:
            raise ValueError(
                f"sub_id={cfg.sub_id} must be in [0, {cfg.sub_workers})."
            )


def _validate_track_dataset_config(dataset: str, cfg: Any) -> None:
    if dataset == "molmoact2" and bool(cfg.multi_view):
        raise ValueError("MolmoAct2 tracks mode is top-view only; set multi_view=false.")
    validate_track_chunk_overlap(dataset, cfg)


def _validate_action_config(cfg: Any) -> None:
    _validate_subsampling(cfg)
    if cfg.subsampling != 1:
        raise ValueError(
            f"actions mode requires subsampling=1, got {cfg.subsampling}; "
            f"otherwise action frame_indices will not match track frame_indices."
        )


def _validate_vae_config(dataset: str, cfg: Any) -> None:
    _validate_positive_int("horizon", cfg.horizon)
    image_hw = list(cfg.wan.image_hw)
    if len(image_hw) != 2:
        raise ValueError(f"wan.image_hw must have two entries, got {image_hw}.")
    for axis, size in zip(("height", "width"), image_hw):
        _validate_positive_int(f"wan.image_hw.{axis}", size)
    if image_hw[0] != image_hw[1]:
        raise ValueError(f"Wan preprocessing expects square image_hw, got {image_hw}.")
    if dataset in {"egoverse", "molmoact2"} and image_hw != [cfg.img_size, cfg.img_size]:
        raise ValueError(
            f"wan.image_hw={image_hw} must match img_size={cfg.img_size} because "
            "latent modes read the preprocessed video store."
        )
    if cfg.mode == "vae_latent_h16" and cfg.horizon % 4 != 0:
        raise ValueError(
            f"vae_latent_h16 requires horizon % 4 == 0, got horizon={cfg.horizon}."
        )


def _validate_egoverse_config(cfg: Any) -> None:
    if cfg.output_format != "zarr":
        raise ValueError(f"output_format={cfg.output_format} is unsupported; EgoVerseDataset reads zarr only.")


def _validate_tracker_type(tracker: str) -> None:
    if tracker not in TRACKER_OUTPUTS:
        raise ValueError(
            f"Unsupported tracker_type={tracker}; choices={tuple(TRACKER_OUTPUTS)}."
        )


def _validate_subsampling(cfg: Any) -> None:
    if type(cfg.subsampling) is not int or not 1 <= cfg.subsampling <= 60:
        raise ValueError(
            f"subsampling must be an int in [1, 60], got {cfg.subsampling}."
        )


def _validate_positive_int(name: str, val: Any) -> None:
    if type(val) is not int or val <= 0:
        raise ValueError(f"{name} must be a positive int, got {val}.")


def _validate_nonnegative_int(name: str, val: Any) -> None:
    if type(val) is not int or val < 0:
        raise ValueError(f"{name} must be a nonnegative int, got {val}.")


def _validate_optional_positive_int(cfg: Any, name: str) -> None:
    if name in cfg and cfg[name] is not None:
        _validate_positive_int(name, cfg[name])


def _track_num_views(dataset: str, cfg: Any) -> int:
    if not bool(cfg.multi_view):
        return 1
    if dataset == "egoverse":
        return 1
    if dataset == "molmoact2":
        return 1
    return len(list(cfg.camera_names))
