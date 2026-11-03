"""
Shared action conversion utilities for training and robot inference.

This module is intentionally small and fail-hard:
- no silent fallbacks for missing timestamps or bad shapes
- strict validation of action conventions
"""

import numpy as np
import torch
from einops import rearrange


ROT6D_DIMS = ((3, 9), (13, 19))


def _require_2d(x: np.ndarray, name: str) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {x.shape}")
    if x.shape[0] <= 0 or x.shape[1] <= 0:
        raise ValueError(f"{name} must have positive shape, got {x.shape}")


def rot6d_to_matrix_np(rot6: np.ndarray) -> np.ndarray:
    """
    Convert rot6d to rotation matrices.

    rot6: [N, 6] where each row is [R00, R01, R10, R11, R20, R21].
    out : [N, 3, 3]
    """
    _require_2d(rot6, "rot6")
    if rot6.shape[1] != 6:
        raise ValueError(f"rot6 must have D=6, got shape {rot6.shape}")

    rot_cols = rot6.reshape((rot6.shape[0], 3, 2))
    a1 = rot_cols[:, :, 0]
    a2 = rot_cols[:, :, 1]

    eps = 1e-8
    b1_n = np.linalg.norm(a1, axis=1, keepdims=True)
    b1 = a1 / np.maximum(b1_n, eps)

    proj = np.sum(a2 * b1, axis=1, keepdims=True) * b1
    b2_t = a2 - proj
    b2_n = np.linalg.norm(b2_t, axis=1, keepdims=True)
    b2 = b2_t / np.maximum(b2_n, eps)

    b3 = np.cross(b1, b2, axis=1)
    return np.stack([b1, b2, b3], axis=2).astype(np.float32)


def matrix_to_rot6d_np(r: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrices to row-major 6D columns format.

    r   : [N, 3, 3]
    out : [N, 6] as [R00, R01, R10, R11, R20, R21]
    """
    if r.ndim != 3 or r.shape[1:] != (3, 3):
        raise ValueError(f"r must be [N,3,3], got shape {r.shape}")
    return np.stack([r[:, 0, 0], r[:, 0, 1], r[:, 1, 0], r[:, 1, 1], r[:, 2, 0], r[:, 2, 1]], axis=1).astype(
        np.float32
    )


def to_relative_rot6d_eepose20(a_abs: np.ndarray, ref_action: np.ndarray) -> np.ndarray:
    """
    Convert absolute eepose_6drot actions to relative-rotation eepose_6drot.

    a_abs: [T, 20]
      left : xyz(0:3), rot6d(3:9), grip(9)
      right: xyz(10:13), rot6d(13:19), grip(19)
    ref_action: [20] absolute pose at the observation frame.

    output has the same shape. Positions/grippers are unchanged.
    Rotations become relative to the reference for each arm:
      R_rel[t] = R_ref^T @ R[t]
    """
    _require_2d(a_abs, "a_abs")
    if a_abs.shape[1] != 20:
        raise ValueError(f"eepose_6drot actions must be [T,20], got {a_abs.shape}")
    ref = np.asarray(ref_action, dtype=np.float32)
    if ref.shape != (20,):
        raise ValueError(f"ref_action must be (20,), got {ref.shape}")

    out = np.asarray(a_abs, dtype=np.float32).copy()
    for s, e in ROT6D_DIMS:
        r_abs = rot6d_to_matrix_np(out[:, s:e])
        r_ref_t = rot6d_to_matrix_np(rearrange(ref[s:e], "d -> 1 d"))[0].T
        r_rel = np.einsum("ij,tjk->tik", r_ref_t, r_abs)
        out[:, s:e] = matrix_to_rot6d_np(r_rel)
    return out


def rot6d_to_matrix_torch(rot6: torch.Tensor) -> torch.Tensor:
    """Convert `(b, t, 6)` row-major first-two-column rotations to `(b, t, 3, 3)`."""
    if rot6.ndim != 3 or rot6.shape[2] != 6:
        raise ValueError(f"rot6 must be (b, t, 6), got {tuple(rot6.shape)}")
    rc = rearrange(rot6, "b t (r c) -> b t r c", r=3, c=2)
    c1 = rc[:, :, :, 0]
    c2 = rc[:, :, :, 1]
    c1 = c1 / c1.norm(dim=2, keepdim=True).clamp_min(1e-8)
    proj = (c2 * c1).sum(dim=2, keepdim=True) * c1
    c2 = c2 - proj
    c2 = c2 / c2.norm(dim=2, keepdim=True).clamp_min(1e-8)
    c3 = torch.cross(c1, c2, dim=2)
    return torch.stack([c1, c2, c3], dim=3)


def matrix_to_rot6d_torch(r: torch.Tensor) -> torch.Tensor:
    """Convert `(b, t, 3, 3)` rotations to `(b, t, 6)` row-major first-two columns."""
    if r.ndim != 4 or r.shape[2:] != (3, 3):
        raise ValueError(f"r must be (b, t, 3, 3), got {tuple(r.shape)}")
    return torch.stack(
        [r[:, :, 0, 0], r[:, :, 0, 1], r[:, :, 1, 0], r[:, :, 1, 1], r[:, :, 2, 0], r[:, :, 2, 1]],
        dim=2,
    )


def relative_rot6d_to_absolute(actions: torch.Tensor, ref_action: np.ndarray) -> torch.Tensor:
    """Convert relative rotations in `(b, t, 20)` actions using a `(20,)` absolute reference."""
    if actions.ndim != 3 or actions.shape[2] != 20:
        raise ValueError(f"actions must be (b, t, 20), got {tuple(actions.shape)}")
    ref = np.asarray(ref_action, dtype=np.float32)
    if ref.shape != (20,):
        raise ValueError(f"ref_action must be (20,), got {ref.shape}")
    ref_t = torch.from_numpy(ref).to(device=actions.device, dtype=actions.dtype)

    out = actions.clone()
    for s, e in ROT6D_DIMS:
        r0 = rot6d_to_matrix_torch(rearrange(ref_t[s:e], "d -> 1 1 d"))[0, 0]
        r_rel = rot6d_to_matrix_torch(out[:, :, s:e])
        r_abs = torch.einsum("ij,btjk->btik", r0, r_rel)
        out[:, :, s:e] = matrix_to_rot6d_torch(r_abs)
    return out


def rot_matrix_to_ypr(r: torch.Tensor) -> torch.Tensor:
    """Convert `(b, t, 3, 3)` rotations to `(b, t, 3)` ZYX yaw-pitch-roll."""
    if r.ndim != 4 or r.shape[2:] != (3, 3):
        raise ValueError(f"r must be (b, t, 3, 3), got {tuple(r.shape)}")
    sy = (-r[:, :, 2, 0]).clamp(-1 + 1e-6, 1 - 1e-6)
    pitch = torch.asin(sy)
    yaw = torch.atan2(r[:, :, 1, 0], r[:, :, 0, 0])
    roll = torch.atan2(r[:, :, 2, 1], r[:, :, 2, 2])
    return torch.stack([yaw, pitch, roll], dim=2)


def convert_rot6d_to_ypr(actions: torch.Tensor) -> torch.Tensor:
    """Convert bimanual `(b, t, 20)` rot6d actions to `(b, t, 14)` robot YPR actions."""
    if actions.ndim != 3 or actions.shape[2] != 20:
        raise ValueError(f"actions must be (b, t, 20), got {tuple(actions.shape)}")
    l_ypr = rot_matrix_to_ypr(rot6d_to_matrix_torch(actions[:, :, 3:9]))
    r_ypr = rot_matrix_to_ypr(rot6d_to_matrix_torch(actions[:, :, 13:19]))
    return torch.cat(
        [actions[:, :, 0:3], l_ypr, actions[:, :, 9:10], actions[:, :, 10:13], r_ypr, actions[:, :, 19:20]],
        dim=2,
    )


def convert_ypr_to_rot6d(ee_pose: np.ndarray) -> np.ndarray:
    """Convert one bimanual `(14,)` robot YPR pose to the model's `(20,)` rot6d space."""
    from scipy.spatial.transform import Rotation

    ee = np.asarray(ee_pose, dtype=np.float32)
    if ee.shape != (14,):
        raise ValueError(f"ee_pose must be (14,), got {ee.shape}")
    l_mat = Rotation.from_euler("ZYX", ee[3:6]).as_matrix()
    r_mat = Rotation.from_euler("ZYX", ee[10:13]).as_matrix()
    l_rot6 = rearrange(l_mat[:, 0:2], "r c -> (r c)")
    r_rot6 = rearrange(r_mat[:, 0:2], "r c -> (r c)")
    return np.concatenate([ee[0:3], l_rot6, ee[6:7], ee[7:10], r_rot6, ee[13:14]]).astype(np.float32)


def robotwin_ee_to_eepose20(
    left_pose: np.ndarray, left_grip: np.ndarray, right_pose: np.ndarray, right_grip: np.ndarray,
) -> np.ndarray:
    """Convert RoboTwin `(t, 7)` xyz+wxyz EE poses and `(t, 1)` grippers to `(t, 20)` eepose_6drot."""
    from scipy.spatial.transform import Rotation

    arms = []
    for pose, grip in ((left_pose, left_grip), (right_pose, right_grip)):
        rot = Rotation.from_quat(pose[:, [4, 5, 6, 3]]).as_matrix()  # wxyz -> scipy xyzw
        arms += [pose[:, 0:3], matrix_to_rot6d_np(rot), grip]
    return np.concatenate(arms, axis=1).astype(np.float32)
