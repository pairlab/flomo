"""Ramen-style per-timestep, per-channel normalization for cumulative_disp tracks.

Structurally mirrors `ActionNormalizer` in `flomo/utils/action_normalizer.py`:
- Stats: `(H, 3)` per-timestep, per-channel p02 / p98 of cumulative_disp values
  (cumulative_disp[t] = tracks[t] - tracks[0]) pooled across (sample, point).
- Forward:  y = clip((4/3)*(x - p02)/(p98 - p02) - (2/3), -1, 1)
- Inverse:  x = ((3/4)*y + 1/2) * (p98 - p02) + p02

The output range is [-1, 1] to match Wan VAE's pretrained decode range.

Used by the dataloaders when `supervision='cumulative_disp'` and
`tracks_normalizer='ramen'`. The on-disk stats file lives at
`<dataset_path>/normalizer_stats/tracks_cumulative_disp_h<H>.json`; the loader
computes it on the fly with `compute_and_save_tracks_stats` if missing.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch


class TracksNormalizer:
    """Per-timestep, per-channel Ramen normalizer for cumulative_disp tracks.

    `p02` and `p98` are `(horizon, 3)` arrays of percentiles computed once over
    a corpus of cumulative_disp values. Forward maps each `(t, dim)` cell of an input
    `(..., horizon, ..., 3)` tensor into [-1, 1] using its own stats; the
    inverse recovers metric units (modulo clip tails).
    """

    def __init__(
        self,
        p02: np.ndarray,
        p98: np.ndarray,
        horizon: int,
    ):
        self.p02 = np.asarray(p02, dtype=np.float32)
        self.p98 = np.asarray(p98, dtype=np.float32)
        self.horizon = int(horizon)

        if self.p02.shape != (self.horizon, 3):
            raise ValueError(
                f"p02 shape mismatch: {self.p02.shape} vs ({self.horizon}, 3)"
            )
        if self.p98.shape != (self.horizon, 3):
            raise ValueError(
                f"p98 shape mismatch: {self.p98.shape} vs ({self.horizon}, 3)"
            )

        self.range = self.p98 - self.p02
        self.range = np.where(self.range < 1e-8, 1e-8, self.range)
        self.metadata: Dict = {}

    # ------------- (de)serialization ----------------------------------------

    @classmethod
    def from_stats_dict(cls, data: Dict) -> "TracksNormalizer":
        norm = cls(
            p02=np.array(data["stats"]["p02"]),
            p98=np.array(data["stats"]["p98"]),
            horizon=data["horizon"],
        )
        norm.metadata = {
            k: v for k, v in data.items() if k not in {"stats", "horizon"}
        }
        return norm

    @classmethod
    def from_stats_file(cls, stats_path: Union[str, Path]) -> "TracksNormalizer":
        stats_path = Path(stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"Tracks stats file not found: {stats_path}")
        with open(stats_path, "r") as f:
            data = json.load(f)
        return cls.from_stats_dict(data)

    def to_dict(self) -> Dict:
        return {
            "horizon": self.horizon,
            "stats": {
                "p02": self.p02.tolist(),
                "p98": self.p98.tolist(),
            },
        }

    def save(
        self,
        path: Union[str, Path],
        num_samples: int = 0,
        metadata: Optional[Dict] = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        data["num_samples"] = num_samples
        data["computed_at"] = datetime.now().isoformat()
        if metadata:
            data.update(metadata)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved tracks normalizer stats to {path}")

    # ------------- forward / inverse ----------------------------------------

    def _apply_along_horizon(
        self,
        x: np.ndarray,
        lo: np.ndarray,
        rng: np.ndarray,
        out_lo: float,
        out_hi: float,
    ) -> np.ndarray:
        """Broadcast `(H, 3)` stats over an input shaped `(..., H, N, 3)`.
        Returns the normalized array, clipped to `[out_lo, out_hi]`.
        """
        if x.ndim < 3 or x.shape[-1] != 3:
            raise ValueError(f"expected (..., H, N, 3); got {x.shape}")
        h_used = x.shape[-3]
        if h_used > self.horizon:
            raise ValueError(
                f"horizon axis length {h_used} exceeds normalizer horizon {self.horizon}"
            )
        # (H, 3) -> (H, 1, 3) so right-to-left broadcasting aligns against
        # (..., H, N, 3); leading axes are added by numpy as needed.
        lo_b = lo[:h_used].reshape(h_used, 1, 3)
        rng_b = rng[:h_used].reshape(h_used, 1, 3)
        y = (4.0 / 3.0) * (x - lo_b) / rng_b - (2.0 / 3.0)
        return np.clip(y, out_lo, out_hi)

    def normalize(
        self,
        tracks: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Forward Ramen normalization.

        `tracks` is cumulative_disp of shape `(..., H, N, 3)` (typically `(V=1, H, N, 3)`
        from the dataloader, or `(B, V, H, N, 3)` from a stacked batch). Values
        outside `[p02, p98]` are soft-extended toward `[-1, 1]` and then clipped.
        """
        is_torch = isinstance(tracks, torch.Tensor)
        device = tracks.device if is_torch else None
        x = tracks.detach().cpu().numpy() if is_torch else np.asarray(tracks)
        x = x.astype(np.float32, copy=False)
        y = self._apply_along_horizon(x, self.p02, self.range, -1.0, 1.0)
        return torch.from_numpy(y).to(device) if is_torch else y

    def denormalize(
        self,
        tracks_norm: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Inverse of `normalize`. Clipping at ±1 is destructive in the
        forward direction; this inverse just undoes the affine."""
        is_torch = isinstance(tracks_norm, torch.Tensor)
        device = tracks_norm.device if is_torch else None
        y = tracks_norm.detach().cpu().numpy() if is_torch else np.asarray(tracks_norm)
        y = y.astype(np.float32, copy=False)
        if y.ndim < 3 or y.shape[-1] != 3:
            raise ValueError(f"expected (..., H, N, 3); got {y.shape}")
        h_used = y.shape[-3]
        if h_used > self.horizon:
            raise ValueError(
                f"horizon axis length {h_used} exceeds normalizer horizon {self.horizon}"
            )
        lo_b = self.p02[:h_used].reshape(h_used, 1, 3)
        rng_b = self.range[:h_used].reshape(h_used, 1, 3)
        x = ((3.0 / 4.0) * y + 0.5) * rng_b + lo_b
        return torch.from_numpy(x).to(device) if is_torch else x


def sample_cumulative_disp_from_dataset(
    ds,
    n_samples: int,
    horizon: int,
    seed: int = 0,
) -> np.ndarray:
    """Stream up to `n_samples` cumulative_disp windows from a loader's `index_map`.

    Duck-typed: `ds` must expose `index_map`, `_track_idx(idx_dict)`,
    `_tracks_group(idx_dict)` context manager, and `tracks_key`. Returns
    `(K, horizon, N, 3)` with `K <= n_samples`.
    """
    rng = np.random.default_rng(seed)
    n_total = len(ds.index_map)
    if n_total == 0:
        raise RuntimeError("dataset has empty index_map")
    take = min(n_samples, n_total)
    indices = sorted(rng.choice(n_total, size=take, replace=False).tolist())

    samples = []
    for idx in indices:
        idx_dict = ds.index_map[idx]
        track_idx = ds._track_idx(idx_dict)
        with ds._tracks_group(idx_dict) as vg:
            window = np.asarray(
                vg[ds.tracks_key][track_idx:track_idx + 1]
            ).astype(np.float32)                                          # (1, H_pp, N, 3)
        if window.shape[0] != 1 or window.shape[1] < horizon:
            continue
        traj = window[0, :horizon]                                        # (H, N, 3)
        samples.append(traj - traj[:1])                                   # cumulative_disp
    if not samples:
        raise RuntimeError("no usable cumulative_disp samples produced")
    return np.stack(samples, axis=0)                                      # (K, H, N, 3)


def compute_tracks_normalizer_stats(
    cumulative_disp_samples: np.ndarray,
    horizon: int,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> TracksNormalizer:
    """Compute per-timestep, per-channel `p_low`/`p_high` percentiles over a
    pool of cumulative_disp samples.

    `cumulative_disp_samples` is `(K, H, N, 3)` — K observed windows, each with H
    timesteps of `(N, 3)` cumulative_disp values (frame-0 subtracted). Returns a fitted
    `TracksNormalizer`.
    """
    arr = np.asarray(cumulative_disp_samples, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected (K, H, N, 3); got {arr.shape}")
    if arr.shape[1] != horizon:
        raise ValueError(f"horizon mismatch: array has {arr.shape[1]}, requested {horizon}")
    # Pool (sample, point) per (t, channel): shape (H, K*N, 3) -> percentiles over axis=1.
    pooled = arr.transpose(1, 0, 2, 3).reshape(horizon, -1, 3)            # (H, K*N, 3)
    p02 = np.percentile(pooled, p_low, axis=1).astype(np.float32)         # (H, 3)
    p98 = np.percentile(pooled, p_high, axis=1).astype(np.float32)        # (H, 3)
    return TracksNormalizer(p02=p02, p98=p98, horizon=horizon)


def compute_and_save_tracks_stats(
    ds,
    horizon: int,
    out_path,
    n_samples: int = 5000,
    p_low: float = 2.0,
    p_high: float = 98.0,
    seed: int = 0,
    metadata: Optional[Dict] = None,
) -> TracksNormalizer:
    """Convenience: sample from `ds`, fit, save, return. Used by the loader's
    on-the-fly auto-compute path when the cache file is missing."""
    cd = sample_cumulative_disp_from_dataset(ds, n_samples=n_samples, horizon=horizon, seed=seed)
    norm = compute_tracks_normalizer_stats(cd, horizon=horizon, p_low=p_low, p_high=p_high)
    md = {"p_low": p_low, "p_high": p_high}
    if metadata:
        md.update(metadata)
    norm.save(out_path, num_samples=int(cd.shape[0]), metadata=md)
    return norm
