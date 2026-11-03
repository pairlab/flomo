"""Weighted cotrain loading, shared modality/view expansion, and collation."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler


# Camera modalities expand into per-view streams such as img_primary.
CAMERA_KINDS = ("image", "video")

# Canonical camera-view vocabulary, in trunk-concat order. Each dataset class
# maps its internal view keys onto these via `CANONICAL_VIEWS`.
CANONICAL_VIEW_ORDER = ("primary", "left", "right")


def base_modality_name(name: str) -> str:
    """Strip a canonical view suffix, preserving underscores in base names."""
    for view in CANONICAL_VIEW_ORDER:
        suffix = f"_{view}"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _dataset_class(dataset_type: str):
    """Map a dataset `type` onto its `BaseDataset` subclass."""
    if dataset_type == "egoverse":
        from flomo.loaders.egoverse_dataset import EgoVerseDataset
        return EgoVerseDataset
    if dataset_type == "realyam":
        from flomo.loaders.realyam_dataset import RealYamDataset
        return RealYamDataset
    if dataset_type == "molmoact2":
        from flomo.loaders.molmoact2_dataset import MolmoAct2Dataset
        return MolmoAct2Dataset
    if dataset_type == "robotwin":
        from flomo.loaders.robotwin_dataset import RoboTwinDataset
        return RoboTwinDataset
    raise NotImplementedError(f"cotrain dataset type {dataset_type!r} not wired")


def build_cotrain_dataset(ds_cfg, model_cfg, aug_cfg) -> Dataset:
    """Build one dataset with model-owned encoding settings.

    `ds_cfg.args` supplies storage and dataset settings. `model_cfg` is the
    forward_dynamics block; its image/video preprocessing settings take
    precedence over the corresponding dataset arguments.
    Resolve these arguments locally without mutating the authored config.
    Pass `aug_cfg=None` for validation.
    """
    args = OmegaConf.to_container(ds_cfg.args, resolve=True)
    args["inputs"] = list(ds_cfg.inputs)
    args["outputs"] = list(ds_cfg.outputs)
    args["img_use_preprocessed"] = bool(model_cfg.modalities.img.use_preprocessed)
    args["video_use_preprocessed"] = bool(model_cfg.modalities.video.use_preprocessed)
    if aug_cfg is not None:
        aug_dict = (
            OmegaConf.to_container(aug_cfg, resolve=True)
            if not isinstance(aug_cfg, dict) else aug_cfg
        )
        # `apply_to_types` is required; an explicit null applies augmentations
        # to every dataset type.
        apply_types = aug_dict["apply_to_types"]
        if apply_types is None or ds_cfg.type in apply_types:
            # Pixel augmentations cannot operate on stored Wan-VAE latents.
            aug_on = (
                aug_dict["color_jitter"]
                or aug_dict["gaussian_blur"]
                or aug_dict["random_resized_crop"]
            )
            if aug_on:
                img_pp = args["img_use_preprocessed"]
                vid_pp = args["video_use_preprocessed"]
                if img_pp or vid_pp:
                    raise ValueError(
                        f"dataset '{ds_cfg.name}': augmentations enabled "
                        f"(color_jitter={bool(aug_dict['color_jitter'])}, "
                        f"gaussian_blur={bool(aug_dict['gaussian_blur'])}, "
                        f"random_resized_crop={bool(aug_dict['random_resized_crop'])}) "
                        f"require live VAE encoding, but img_use_preprocessed="
                        f"{img_pp} / video_use_preprocessed={vid_pp}. Override "
                        f"`forward_dynamics.modalities.img.use_preprocessed=false` "
                        f"and `forward_dynamics.modalities.video.use_preprocessed=false`."
                    )
            args["aug_cfg"] = aug_dict
    ds = _dataset_class(ds_cfg.type)(**args)
    ds.cotrain_name = str(ds_cfg.name)
    return ds


def dataset_canonical_views(ds_cfg, kind: str, role: str) -> List[str]:
    """Return validated camera views in CANONICAL_VIEW_ORDER.

    Inputs use args.cond_views; outputs use args.pred_views.
    Images are conditioning-only.
    """
    if kind not in CAMERA_KINDS:
        raise ValueError(f"{ds_cfg.name}: {kind!r} is not a camera modality kind")
    if kind == "image" and role != "input":
        raise ValueError(
            f"{ds_cfg.name}: 'image' modality is conditioning-only and cannot "
            f"be an output"
        )
    arg = "cond_views" if role == "input" else "pred_views"
    cls = _dataset_class(ds_cfg.type)
    requested = set(ds_cfg.args[arg])
    unknown = requested - set(cls.CANONICAL_VIEWS)
    if unknown:
        raise ValueError(
            f"{ds_cfg.name}: {arg} has views {sorted(unknown)} not in dataset "
            f"type {ds_cfg.type!r} CANONICAL_VIEWS {sorted(cls.CANONICAL_VIEWS)}"
        )
    return [v for v in CANONICAL_VIEW_ORDER if v in requested]


def _expand_dataset_modalities(
    base_names: Sequence[str], ds_cfg, role: str, modalities_cfg,
) -> List[str]:
    """Expand camera modalities to name_view streams; leave other names unchanged."""
    out: List[str] = []
    for m in base_names:
        kind = str(modalities_cfg[m].kind)
        if kind in CAMERA_KINDS:
            views = dataset_canonical_views(ds_cfg, kind, role)
            if not views:
                view_arg = "cond_views" if role == "input" else "pred_views"
                canonical = sorted(_dataset_class(ds_cfg.type).CANONICAL_VIEWS)
                raise ValueError(
                    f"dataset {ds_cfg.name!r}: {role} modality {m!r} is camera-kind "
                    f"and needs per-view streams, but args.{view_arg} is empty. "
                    f"Set args.{view_arg} to a non-empty subset of {canonical}."
                )
            out.extend(f"{m}_{v}" for v in views)
        else:
            out.append(m)
    return out


def expand_modality_order(
    modality_order: Sequence[str],
    active_inputs: Sequence[str],
    active_outputs: Sequence[str],
) -> List[str]:
    """Expand base names to active name_view streams in CANONICAL_VIEW_ORDER.

    active_inputs/outputs must already be expanded by derive_active_modalities.
    """
    active = set(active_inputs) | set(active_outputs)
    out: List[str] = []
    for base in modality_order:
        if base in active:  # single-view keys (actions, motion tokens, etc.)
            out.append(base)
        else:  # camera keys (img, video) -> per-view streams
            out.extend(
                f"{base}_{v}" for v in CANONICAL_VIEW_ORDER if f"{base}_{v}" in active
            )
    return out


def derive_active_modalities(
    datasets_cfg: Sequence[Any],
    modalities_cfg: Any,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Return sorted `(active_inputs, active_outputs, universal, partial)` lists.

    Expand camera views before classifying streams. A stream is universal only
    if every dataset supplies it in the same role; other streams are partial.
    Reject empty datasets, duplicate names, conflicting roles, and nonpositive
    total weight.
    """
    if len(datasets_cfg) == 0:
        raise ValueError("cfg.datasets is empty")
    seen_names: List[str] = []
    for ds in datasets_cfg:
        if ds.name in seen_names:
            raise ValueError(f"duplicate dataset name {ds.name!r} in cfg.datasets")
        seen_names.append(ds.name)

    per_ds_inputs = [
        set(_expand_dataset_modalities(ds.inputs, ds, "input", modalities_cfg))
        for ds in datasets_cfg
    ]
    per_ds_outputs = [
        set(_expand_dataset_modalities(ds.outputs, ds, "output", modalities_cfg))
        for ds in datasets_cfg
    ]
    active_inputs_set = set().union(*per_ds_inputs)
    active_outputs_set = set().union(*per_ds_outputs)

    role_conflicts = active_inputs_set & active_outputs_set
    if role_conflicts:
        raise ValueError(
            f"modality(ies) appear as BOTH input and output across datasets "
            f"(role conflict): {sorted(role_conflicts)}"
        )

    weights_sum = sum(float(ds.weight) for ds in datasets_cfg)
    if not (weights_sum > 0.0):
        raise ValueError(
            f"dataset weights must sum to a positive number; got {weights_sum}"
        )

    universal_in = {m for m in active_inputs_set if all(m in s for s in per_ds_inputs)}
    universal_out = {m for m in active_outputs_set if all(m in s for s in per_ds_outputs)}
    universal = sorted(universal_in | universal_out)
    partial = sorted((active_inputs_set | active_outputs_set) - set(universal))
    return sorted(active_inputs_set), sorted(active_outputs_set), universal, partial


def _to_tensor(v):
    """Numpy arrays from BaseDataset.__getitem__ → torch.Tensor; strings/None pass through."""
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    return v


class CotrainCollate:
    """Picklable collator for spawned DataLoader workers.

    Collates each modality as a length-B list and a `[B]` bool presence mask.
    Universal modalities must exist in every sample. Partial modalities use
    None entries and False masks where absent. Both follow the same batch schema.
    """

    def __init__(self, universal: Sequence[str], partial: Sequence[str]):
        overlap = set(universal) & set(partial)
        if overlap:
            raise ValueError(f"cotrain_collate: modality cannot be both universal and partial: {sorted(overlap)}")
        self.universal = list(universal)
        self.partial = list(partial)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(samples) == 0:
            raise ValueError("cotrain_collate: empty sample list")
        B = len(samples)
        out: Dict[str, Any] = {}

        for m in self.universal:
            items: List[Any] = []
            for i, s in enumerate(samples):
                if m not in s or s[m] is None:
                    raise ValueError(
                        f"cotrain_collate: universal modality {m!r} is None/missing on "
                        f"sample {i}. Check dataset YAML inputs/outputs vs `universal` set."
                    )
                items.append(_to_tensor(s[m]))
            out[m] = items
            out[f"{m}_mask"] = torch.ones(B, dtype=torch.bool)

        for m in self.partial:
            field: List[Any] = []
            mask = torch.zeros(B, dtype=torch.bool)
            for i, s in enumerate(samples):
                v = s[m] if m in s else None
                if v is None:
                    field.append(None)
                else:
                    field.append(_to_tensor(v))
                    mask[i] = True
            out[m] = field
            out[f"{m}_mask"] = mask

        for key in ("dataset_name", "dataset_subname"):
            if any(key in s for s in samples):
                out[key] = [s[key] if key in s else None for s in samples]

        # Broken-sample records flow up to the train loop for logging.
        broken = [r for s in samples if "_broken_samples" in s for r in s["_broken_samples"]]
        if broken:
            out["_broken_samples"] = broken

        # TextModality selects embeddings or raw strings via use_preprocessed.
        if any("text_emb" in s for s in samples):
            out["text_emb"] = torch.stack([_to_tensor(s["text_emb"]) for s in samples], dim=0)

        # Keep visualization fields as lists: mixed datasets may have different lengths.
        # init_pts anchors cumulative displacement when reconstructing metric tracks.
        for key in ("viz_video", "intrs", "init_pts"):
            if any(key in s for s in samples):
                out[key] = [_to_tensor(s[key]) if key in s else None for s in samples]

        return out


def cotrain_batch_to_device(batch: Dict[str, Any], device, non_blocking: bool = False) -> Dict[str, Any]:
    """Move tensor fields/list entries to device; preserve None and non-tensor values."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=non_blocking)
        elif isinstance(v, list):
            out[k] = [
                (t.to(device, non_blocking=non_blocking) if isinstance(t, torch.Tensor) else t)
                for t in v
            ]
        else:
            out[k] = v
    return out


def build_cotrain_sample_weights(
    per_dataset_instances: Sequence[Dataset],
    per_dataset_weights: Sequence[float],
) -> torch.Tensor:
    """Assign each sample weight w_d/N_d, giving dataset d total mass w_d.

    WeightedRandomSampler normalizes these masses into expected batch shares.
    """
    return torch.cat([
        torch.full((len(ds),), float(w) / len(ds), dtype=torch.double)
        for ds, w in zip(per_dataset_instances, per_dataset_weights)
    ])


def build_cotrain_dataloader(
    per_dataset_instances: Sequence[Dataset],
    per_dataset_weights: Sequence[float],
    universal_modalities: Sequence[str],
    partial_modalities: Sequence[str],
    batch_size: int,
    num_samples: int,
    num_workers: int,
    drop_last: bool,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Build a weighted DataLoader with cotrain collation.

    Dataset instances and weights must have the same order. num_samples sets
    the epoch's total draws; generator controls sampler reproducibility.
    prefetch_factor is ignored when num_workers=0.
    """
    combined = ConcatDataset(list(per_dataset_instances))
    sample_weights = build_cotrain_sample_weights(per_dataset_instances, per_dataset_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=int(num_samples),
        replacement=True,
        generator=generator,
    )
    collate = CotrainCollate(universal=universal_modalities, partial=partial_modalities)
    return DataLoader(
        combined,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers),
        prefetch_factor=int(prefetch_factor) if int(num_workers) > 0 else None,
        drop_last=bool(drop_last),
        collate_fn=collate,
    )
