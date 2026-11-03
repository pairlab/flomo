"""
Ramen-style action normalization.

Implements per-timestep, per-feature percentile-based normalization:
- Normalizes position dimensions (first 3 per arm) using 2nd/98th percentiles
- Maps [p02, p98] -> [-1, 1], clips outliers to [-1.5, 1.5]
- Skips rotation (6D) and gripper dimensions

Reference: Ramen paper Section on Data Normalization
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch


# Action dimension structure for bimanual robot
#
# eepose_6drot (20D):
#   Left arm:  [pos(3), rot6d(6), gripper(1)] = dims 0-9
#   Right arm: [pos(3), rot6d(6), gripper(1)] = dims 10-19
#   Normalize: position only (0-2, 10-12)
#   Skip: 6D rotation (3-8, 13-18) - already in [-1, 1] range
#   Skip: gripper (9, 19) - binary values (0 or 1)
#
# eepose_ypr (14D):
#   Left arm:  [pos(3), ypr(3), gripper(1)] = dims 0-6
#   Right arm: [pos(3), ypr(3), gripper(1)] = dims 7-13
#   Normalize: position only (0-2, 7-9)
#   Skip: yaw/pitch/roll (3-5, 10-12) - angular values
#   Skip: gripper (6, 13) - binary values (0 or 1)
#
ACTION_SPACE_CONFIGS = {
    "eepose_6drot": {
        "total_dim": 20,
        "normalize_dims": [0, 1, 2, 10, 11, 12],  # Position only (gripper is binary, skip)
        "per_arm_dim": 10,
    },
    "eepose_ypr": {
        "total_dim": 14,
        "normalize_dims": [0, 1, 2, 7, 8, 9],  # Position only (gripper is binary, skip)
        "per_arm_dim": 7,
    },
    "joints": {
        "total_dim": 14,
        "normalize_dims": list(range(14)),  # All joints
        "per_arm_dim": 7,
    },
}


class ActionNormalizer:
    """Ramen-style percentile-based action normalizer with relative actions.

    Actions are relative to the observation-frame pose (Ramen-style):
    - During normalize: subtract the reference action before normalizing
    - During denormalize: add reference action (current EE pose) after denormalizing

    Only normalizes specified dimensions (typically XYZ positions) using per-timestep
    2nd/98th percentile statistics. Other dimensions pass through unchanged.

    Formula: y = clip(2 * (x - p02) / (p98 - p02) - 1, -1.5, 1.5)
    """

    def __init__(
        self,
        p02: np.ndarray,
        p98: np.ndarray,
        normalize_dims: List[int],
        action_space: str,
        horizon: int,
    ):
        """Initialize normalizer with precomputed statistics.

        Args:
            p02: 2nd percentile stats, shape (horizon, num_normalize_dims)
            p98: 98th percentile stats, shape (horizon, num_normalize_dims)
            normalize_dims: List of dimension indices to normalize
            action_space: Action space name (e.g., 'eepose_6drot')
            horizon: Action prediction horizon (number of timesteps)
        """
        self.p02 = np.array(p02, dtype=np.float32)
        self.p98 = np.array(p98, dtype=np.float32)
        self.normalize_dims = normalize_dims
        self.action_space = action_space
        self.horizon = horizon

        # Validate shapes
        assert self.p02.shape == (horizon, len(normalize_dims)), \
            f"p02 shape mismatch: {self.p02.shape} vs ({horizon}, {len(normalize_dims)})"
        assert self.p98.shape == (horizon, len(normalize_dims)), \
            f"p98 shape mismatch: {self.p98.shape} vs ({horizon}, {len(normalize_dims)})"

        # Precompute range for efficiency
        self.range = self.p98 - self.p02
        # Avoid division by zero - use small epsilon
        self.range = np.where(self.range < 1e-8, 1e-8, self.range)

    @classmethod
    def from_stats_dict(cls, data: Dict) -> "ActionNormalizer":
        """Construct from an in-memory stats dict (as produced by `to_dict`).
        Extra fields (tasks, val_fraction, num_demos, computed_at) are exposed
        via `.metadata`.
        """
        norm = cls(
            p02=np.array(data["stats"]["p02"]),
            p98=np.array(data["stats"]["p98"]),
            normalize_dims=data["normalize_dims"],
            action_space=data["action_space"],
            horizon=data["horizon"],
        )
        norm.metadata = {
            k: v for k, v in data.items()
            if k not in {"stats", "normalize_dims", "action_space", "horizon"}
        }
        return norm

    @classmethod
    def from_stats_file(cls, stats_path: Union[str, Path]) -> "ActionNormalizer":
        """Load normalizer from a stats JSON file. Extra metadata fields
        (tasks, val_fraction, num_demos, computed_at) are exposed via .metadata.
        """
        stats_path = Path(stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
        with open(stats_path, "r") as f:
            data = json.load(f)
        return cls.from_stats_dict(data)

    def normalize(
        self,
        actions: Union[np.ndarray, torch.Tensor],
        ref_action: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Normalize actions using Ramen formula with relative actions.

        1. Convert to relative at normalize_dims only: act[dim] -= ref_action[dim]
        2. Normalize: y = clip(2 * (x - p02) / (p98 - p02) - 1, -1.5, 1.5)

        Args:
            actions: Action tensor, shape (T, D) or (B, T, D)
            ref_action: Absolute pose at the observation frame, shape (D,). Datasets
                whose action[0] is the current pose pass action[0].

        Returns:
            Normalized relative actions, same shape as input
        """
        is_torch = isinstance(actions, torch.Tensor)
        device = actions.device if is_torch else None

        if is_torch:
            act = actions.cpu().numpy().copy()
        else:
            act = actions.copy()

        # Ensure 3D: (B, T, D)
        added_batch = False
        if act.ndim == 2:
            act = act[np.newaxis, :, :]  # (T, D) -> (1, T, D)
            added_batch = True

        B, T, D = act.shape
        if T > self.horizon:
            raise ValueError(f"T={T} exceeds horizon={self.horizon}")

        if isinstance(ref_action, torch.Tensor):
            ref = ref_action.cpu().numpy()
        else:
            ref = np.asarray(ref_action, dtype=np.float32)
        if ref.shape != (D,):
            raise ValueError(f"ref_action must be ({D},), got shape {ref.shape}")

        # Convert to relative and normalize at normalize_dims only
        for t in range(T):
            for i, dim in enumerate(self.normalize_dims):
                x = act[:, t, dim] - ref[dim]
                # Normalize
                p02_t = self.p02[t, i]
                rng_t = self.range[t, i]
                act[:, t, dim] = np.clip(2.0 * (x - p02_t) / rng_t - 1.0, -1.5, 1.5)

        if added_batch:
            act = act[0]  # (1, T, D) -> (T, D)

        if is_torch:
            return torch.from_numpy(act).to(device)
        return act

    def denormalize(
        self,
        actions: Union[np.ndarray, torch.Tensor],
        ref_action: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Denormalize actions and convert from relative to absolute.

        1. Denormalize at normalize_dims: x = (y + 1) * (p98 - p02) / 2 + p02
        2. Convert to absolute at normalize_dims: act[dim] += ref_action[dim]

        Args:
            actions: Normalized relative actions, shape (T, D) or (B, T, D)
            ref_action: Reference action (current EE pose), shape (D,)
                       Only values at normalize_dims are used.

        Returns:
            Denormalized absolute actions, same shape as input
        """
        is_torch = isinstance(actions, torch.Tensor)
        device = actions.device if is_torch else None

        if is_torch:
            act = actions.cpu().numpy().copy()
        else:
            act = actions.copy()

        # Ensure 3D: (B, T, D)
        added_batch = False
        if act.ndim == 2:
            act = act[np.newaxis, :, :]  # (T, D) -> (1, T, D)
            added_batch = True

        B, T, D = act.shape
        if T > self.horizon:
            raise ValueError(f"T={T} exceeds horizon={self.horizon}")

        # Prepare reference
        if isinstance(ref_action, torch.Tensor):
            ref = ref_action.cpu().numpy()
        else:
            ref = np.asarray(ref_action, dtype=np.float32)
        if ref.ndim != 1:
            raise ValueError(f"ref_action must be 1D, got shape {ref.shape}")

        # Denormalize and convert to absolute at normalize_dims only
        for t in range(T):
            for i, dim in enumerate(self.normalize_dims):
                y = act[:, t, dim]
                p02_t = self.p02[t, i]
                rng_t = self.range[t, i]
                # Denormalize then add reference
                act[:, t, dim] = (y + 1.0) * rng_t / 2.0 + p02_t + ref[dim]

        if added_batch:
            act = act[0]  # (1, T, D) -> (T, D)

        if is_torch:
            return torch.from_numpy(act).to(device)
        return act

    def to_dict(self) -> Dict:
        """Convert normalizer to dictionary for serialization."""
        return {
            "action_space": self.action_space,
            "horizon": self.horizon,
            "normalize_dims": self.normalize_dims,
            "stats": {
                "p02": self.p02.tolist(),
                "p98": self.p98.tolist(),
            },
        }

    def save(self, path: Union[str, Path], num_demos: int = 0, metadata: Optional[Dict] = None):
        """Save normalizer stats to JSON file.

        Args:
            path: Output file path
            num_demos: Number of demos used to compute stats
            metadata: Extra fields merged into the saved JSON (e.g. tasks, val_fraction)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        data["num_demos"] = num_demos
        data["computed_at"] = datetime.now().isoformat()
        if metadata:
            data.update(metadata)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved normalizer stats to {path}")


def compute_normalizer_stats(
    action_sequences: List[np.ndarray],
    reference_sequences: List[np.ndarray],
    action_space: str,
    horizon: int,
) -> ActionNormalizer:
    """Compute normalizer statistics for relative actions.

    Samples ALL possible chunk starting positions from each demo sequence.
    Stats are computed on relative actions: action[start + t] - reference[start].

    Args:
        action_sequences: List of action arrays, each shape (T, D)
        reference_sequences: Per-demo observation-frame poses aligned with
            action_sequences, each shape (T, D). Datasets whose action[t] is the
            pose at frame t pass action_sequences.
        action_space: Action space name (e.g., 'eepose_6drot')
        horizon: Action prediction horizon

    Returns:
        ActionNormalizer with computed percentile statistics
    """
    config = ACTION_SPACE_CONFIGS[action_space]
    norm_dims = config["normalize_dims"]
    total_dim = config["total_dim"]
    n_dims = len(norm_dims)

    # Collect values: values[timestep][dim_idx] = list of values
    values = [[[] for _ in range(n_dims)] for _ in range(horizon)]

    n_chunks = 0
    for seq, refs in zip(action_sequences, reference_sequences, strict=True):
        if seq.shape[1] != total_dim:
            raise ValueError(f"Action dim {seq.shape[1]} != expected {total_dim}")
        if refs.shape != seq.shape:
            raise ValueError(f"reference shape {refs.shape} != action shape {seq.shape}")

        seq_len = seq.shape[0]

        # Sample all possible chunk starting positions
        for start in range(max(1, seq_len - horizon + 1)):
            chunk = seq[start:start + horizon]
            chunk_len = chunk.shape[0]

            # Collect relative values at norm_dims only
            for t in range(chunk_len):
                for i, dim in enumerate(norm_dims):
                    values[t][i].append(chunk[t, dim] - refs[start, dim])

            n_chunks += 1

    print(f"  Sampled {n_chunks} chunks from {len(action_sequences)} demos")

    # Compute percentiles
    p02 = np.zeros((horizon, n_dims), dtype=np.float32)
    p98 = np.zeros((horizon, n_dims), dtype=np.float32)

    for t in range(horizon):
        for i in range(n_dims):
            if len(values[t][i]) == 0:
                raise ValueError(f"No data for timestep {t}, dim {norm_dims[i]}")
            arr = np.array(values[t][i])
            p02[t, i] = np.percentile(arr, 2)
            p98[t, i] = np.percentile(arr, 98)

    return ActionNormalizer(
        p02=p02,
        p98=p98,
        normalize_dims=norm_dims,
        action_space=action_space,
        horizon=horizon,
    )
