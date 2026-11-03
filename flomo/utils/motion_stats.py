"""Per-dataset motion-token velocity statistics.

Samples raw tracks from each cotrain dataset, derives per-channel velocity
percentile bounds ("flow_pct"), caches them on disk keyed on the tracker
config, and installs them on the live motion-token modality at train startup.
"""

import hashlib
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from flomo.utils.train_utils import unwrap_model_for_checkpoint


def _track_stats_cache_dir():
    """Return the track-stat cache under FLOMO_CACHE_DIR or ~/.cache/flomo."""
    root = (
        os.environ["FLOMO_CACHE_DIR"]
        if "FLOMO_CACHE_DIR" in os.environ
        else os.path.expanduser("~/.cache/flomo")
    )
    return os.path.join(root, "track_stats")


def _per_ds_flow_pct_cache_path(ds_args, ds_name, ds_subsets, cache_dir):
    """Per-dataset cache path keyed on tracker config + dataset name + subset list."""
    key = "|".join([
        f"tracker={ds_args.tracker}",
        f"tracker_mode={ds_args.tracker_mode}",
        f"track_method={ds_args.track_method}",
        f"supervision={ds_args.supervision}",
        f"ds={ds_name}",
        f"subs={'+'.join(sorted(str(s) for s in ds_subsets))}",
    ])
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"flow_pct_{ds_name}_{digest}.json"), key


def _compute_flow_pct_for_dataset(ds, ts_cfg, accelerator):
    """Sample raw tracks from one dataset; return per-channel velocity (and
    cumulative_disp, when supervision='cumulative_disp') percentile bounds.

    For `supervision='cumulative_disp'` the loader returns motion_tokens already
    zero-anchored (cumulative_disp = tracks - tracks[0]); the velocity diff
    `cumulative_disp[t+1] - cumulative_disp[t]` still equals true velocity, so the legacy
    flow_pct keys remain meaningful. cumulative_disp_pct keys are added for the
    cumulative_disp viz path."""
    from flomo.utils.distributed import is_main_process, log_main
    n_target = int(ts_cfg.max_trajectories)
    p_lo = float(ts_cfg.velocity_percentile_low)
    p_hi = float(ts_cfg.velocity_percentile_high)
    n = min(n_target, len(ds))
    g = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(ds), generator=g)[:n].tolist()
    is_cumulative_disp = getattr(ds, "supervision", "") == "cumulative_disp"
    vels, cds = [], []
    log_main(f"[cotrain] flow_pct stats for {ds.cotrain_name}: sampling {n} trajectories",
             accelerator=accelerator)
    for li in tqdm(indices, desc=f"flow_pct [{ds.cotrain_name}]",
                   disable=not is_main_process(accelerator)):
        idx_dict = ds.index_map[li]
        data = ds.load_data(idx_dict, explicit_keys=["motion_tokens"])
        tracks = data["motion_tokens"]                                                  # (V=1, H, N, 3)
        if isinstance(tracks, np.ndarray):
            tracks = torch.from_numpy(tracks)
        vel = tracks[:, 1:] - tracks[:, :-1]                                            # (V, H-1, N, 3)
        vels.append(rearrange(vel, "v t n d -> (v t n) d"))
        if is_cumulative_disp:
            # cumulative_disp = tracks - tracks[0]; for the loader's cumulative_disp path
            # tracks[0] is already 0, so motion_tokens IS cumulative_disp. Skip t=0
            # so the zero-anchor doesn't skew the lo tail toward zero.
            cds.append(rearrange(tracks[:, 1:], "v t n d -> (v t n) d"))
    vel_all = torch.cat(vels, dim=0).numpy()
    pct_low = np.percentile(vel_all, p_lo, axis=0).tolist()
    pct_high = np.percentile(vel_all, p_hi, axis=0).tolist()
    out = {
        "pct_low": pct_low,
        "pct_high": pct_high,
        "percentile_low": p_lo,
        "percentile_high": p_hi,
        "count": int(vel_all.shape[0]),
    }
    if is_cumulative_disp:
        cd_all = torch.cat(cds, dim=0).numpy()
        out["cumulative_disp_pct_low"] = np.percentile(cd_all, p_lo, axis=0).tolist()
        out["cumulative_disp_pct_high"] = np.percentile(cd_all, p_hi, axis=0).tolist()
        out["cumulative_disp_count"] = int(cd_all.shape[0])
    return out


def _install_motion_token_stats(model, cfg, per_ds, accelerator):
    """Install per-dataset flow_pct velocity bounds on live (non-preprocessed)
    MotionTokensWanPretrainedModality instances. Cached on disk; computes fresh
    on miss. Val dataset names alias the matching train dataset's bounds
    (positional pairing of cfg.datasets / cfg.val_datasets). For
    supervision='cumulative_disp' also installs cumulative_disp percentile
    bounds and, when loaders carry a TracksNormalizer, display-only ramen
    stats."""
    from flomo.utils.distributed import is_main_process, log_main
    from flomo.models.modalities import MotionTokensWanPretrainedModality
    inner = unwrap_model_for_checkpoint(model)
    for name, mod in inner.modalities.items():
        if isinstance(mod, MotionTokensWanPretrainedModality) and not mod.use_preprocessed:
            ds_args = cfg.datasets[0].args
            cache_dir = _track_stats_cache_dir()
            os.makedirs(cache_dir, exist_ok=True)
            is_cumulative_disp = mod.supervision == "cumulative_disp"
            bounds = {}
            cumulative_disp_bounds: Dict[str, Tuple] = {}
            for d, ds in zip(cfg.datasets, per_ds):
                cache_path, cache_key = _per_ds_flow_pct_cache_path(
                    ds_args, str(d.name), list(d.args.dataset_names), cache_dir,
                )
                cached = None
                if os.path.exists(cache_path):
                    with open(cache_path, "r") as f:
                        cached = json.load(f)
                # Recompute when the cache predates the cumulative_disp_pct extension
                # (missing key) for a cumulative_disp supervision.
                cache_stale = cached is not None and is_cumulative_disp and "cumulative_disp_pct_low" not in cached
                if cached is not None and not cache_stale:
                    stats = cached
                    log_main(f"[cotrain] flow_pct cache hit ({d.name}): {cache_path}",
                             accelerator=accelerator)
                else:
                    if is_main_process(accelerator):
                        log_main(
                            f"[cotrain] flow_pct cache {'stale' if cache_stale else 'miss'} ({d.name}); key={cache_key}",
                            accelerator=accelerator,
                        )
                        stats = _compute_flow_pct_for_dataset(ds, cfg.track_stats, accelerator)
                        with open(cache_path, "w") as f:
                            json.dump(stats, f)
                        log_main(f"[cotrain] wrote flow_pct cache ({d.name}): {cache_path}",
                                 accelerator=accelerator)
                    accelerator.wait_for_everyone()
                    if not is_main_process(accelerator):
                        with open(cache_path, "r") as f:
                            stats = json.load(f)
                bounds[str(d.name)] = (stats["pct_low"], stats["pct_high"])
                if is_cumulative_disp:
                    cumulative_disp_bounds[str(d.name)] = (
                        stats["cumulative_disp_pct_low"], stats["cumulative_disp_pct_high"],
                    )
            # Alias val dataset names to matching train dataset's bounds (positional pairing).
            for d_t, d_v in zip(cfg.datasets, cfg.val_datasets or []):
                bounds[str(d_v.name)] = bounds[str(d_t.name)]
                if is_cumulative_disp:
                    cumulative_disp_bounds[str(d_v.name)] = cumulative_disp_bounds[str(d_t.name)]
            mod.set_flow_pct_bounds_per_dataset(bounds)
            log_main(
                f"[cotrain] {name}: per-DS flow_pct installed for {sorted(bounds.keys())}",
                accelerator=accelerator,
            )
            if is_cumulative_disp:
                mod.set_cumulative_disp_pct_bounds_per_dataset(cumulative_disp_bounds)
                log_main(
                    f"[cotrain] {name}: per-DS cumulative_disp_pct installed for {sorted(cumulative_disp_bounds.keys())}",
                    accelerator=accelerator,
                )
                # Ramen stats (display-only) are read directly from each loader's
                # TracksNormalizer when present; idle for cumulative_disp + no_norm.
                ramen_stats: Dict[str, Tuple] = {}
                for d, ds in zip(cfg.datasets, per_ds):
                    tn = getattr(ds, "tracks_normalizer", None)
                    if tn is not None:
                        ramen_stats[str(d.name)] = (tn.p02.tolist(), tn.p98.tolist())
                if ramen_stats:
                    if len(ramen_stats) != len(cfg.datasets):
                        raise RuntimeError(
                            f"[cotrain] {name}: tracks_normalizer set on some but not all datasets — "
                            f"{sorted(ramen_stats.keys())} vs {sorted(d.name for d in cfg.datasets)}"
                        )
                    for d_t, d_v in zip(cfg.datasets, cfg.val_datasets or []):
                        ramen_stats[str(d_v.name)] = ramen_stats[str(d_t.name)]
                    mod.set_ramen_stats_per_dataset(ramen_stats)
                    log_main(
                        f"[cotrain] {name}: per-DS ramen stats installed for {sorted(ramen_stats.keys())}",
                        accelerator=accelerator,
                    )
