import random

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from scipy.interpolate import CubicSpline
from torchvision import transforms
from typing import Tuple


def normalize_traj(traj_input, img_shape):
    """
    transforms the trajectory so that the pixel coordinates are between -1 and 1
    traj: (bs, horizon, num_queries, 2) with (row, col) in (height, width)

    Returns:
    traj: (bs, horizon, num_queries, 2) with (row, col) in (-1, 1)
    """
    initial_shape = traj_input.shape

    # making sure traj is a float numpy array or tensor
    if isinstance(traj_input, np.ndarray):
        traj = traj_input.copy().astype(np.float32)
    elif isinstance(traj_input, torch.Tensor):
        traj = traj_input.clone().float()
    else:
        raise ValueError(f"traj should be a numpy array or tensor, but is of type {type(traj)}")

    traj[..., 0] = (traj[..., 0] - float(img_shape[0] / 2)) / (float(img_shape[0]) / 2)
    traj[..., 1] = (traj[..., 1] - float(img_shape[1] / 2)) / (float(img_shape[1]) / 2)

    # clamping/clipping the values to be between -1 and 1
    if isinstance(traj, np.ndarray):
        traj = np.clip(traj, -1, 1)
    elif isinstance(traj, torch.Tensor):
        traj = torch.clamp(traj, -1, 1)

    return traj.reshape(initial_shape)


def unnormalize_traj(traj_input, img_shape):
    """
    transforms the trajectory so that the pixel coordinates are between height and width of the image
    traj: (bs, num_views, horizon, num_queries, 2) with (row, col) in (-1, 1)

    Returns:
    traj: (bs, num_views, horizon, num_queries, 2) with (row, col) in (height, width)
    """
    initial_shape = traj_input.shape

    # making sure traj is a float numpy array or tensor
    if isinstance(traj_input, np.ndarray):
        traj = traj_input.copy().astype(np.float32)
    elif isinstance(traj_input, torch.Tensor):
        traj = traj_input.clone().float()
    else:
        raise ValueError(f"traj should be a numpy array or tensor, but is of type {type(traj_input)}")

    traj[..., 0] = (traj[..., 0] * float(img_shape[0]) / 2) + float(img_shape[0]) / 2
    traj[..., 1] = (traj[..., 1] * float(img_shape[1]) / 2) + float(img_shape[1]) / 2

    return traj.reshape(initial_shape)



def optical_flow_traj(raw: np.ndarray, extrs: np.ndarray) -> np.ndarray:
    """Synthetic cumsum trajectory whose per-step diffs are the camera-rotated
    optical-flow velocities — the training-side optical_flow track tensor shared
    by the EgoVerse / RealYam / MolmoAct2 loaders.

    raw:   (H, 2, N, 3) -- raw[t, 0]/raw[t, 1] = fresh-reinit start/end of window t.
    extrs: (H-1, 4, 4)  -- camera extrinsics for the first H-1 windows.

    Each window's 1-step flow is rotated into window 0's camera frame, then the
    flows are cumulatively summed onto raw[0, 0]. Returns (V=1, H, N, 3) so
    `traj[0, k] = raw[0, 0] + sum(flow[:k])` and `traj[0, 0] = raw[0, 0]`.
    """
    F = raw.shape[0] - 1
    vel = raw[:F, 1] - raw[:F, 0]                                            # (F, N, 3)
    R_to_init = np.einsum(
        "ij,kjl->kil", np.linalg.inv(extrs[0]), extrs
    )[:, :3, :3]                                                             # (F, 3, 3)
    vel = np.einsum("kij,knj->kni", R_to_init, vel)                          # (F, N, 3)
    init = raw[0, 0]                                                         # (N, 3)
    traj = np.concatenate([init[None], init[None] + np.cumsum(vel, axis=0)], axis=0)
    return traj[None]                                                        # (1, H, N, 3)


def interpolate_traj(traj, new_seq_len):
    """
    Interpolates the time dim to new_seq_len, maintaining the first
    and last points of the trajectory.

    Args:
        traj (torch.Tensor or np.ndarray): The input trajectory of shape (..., horizon, num_queries, dim).
        new_seq_len (int): The desired number of points in the output trajectory.

    Returns:
        Same type as input: The trajectory of shape (num_views, new_seq_len, num_queries, 2) after interpolation.
    """
    input_is_numpy = isinstance(traj, np.ndarray)
    if input_is_numpy:
        traj = torch.from_numpy(traj)

    if traj.shape[-3] == new_seq_len:
        return traj.numpy().astype(np.float32) if input_is_numpy else traj

    if traj.ndim == 4:
        v, t, n, d = traj.shape
        traj = rearrange(traj, 'v t n d -> v (n d) t')
        traj = F.interpolate(traj, size=new_seq_len, mode='linear', align_corners=True)
        traj = rearrange(traj, 'v (n d) t -> v t n d', n=n)
    else:
        b, v, t, n, d = traj.shape
        traj = rearrange(traj, 'b v t n d -> b (v n d) t')
        traj = F.interpolate(traj, size=new_seq_len, mode='linear', align_corners=True)
        traj = rearrange(traj, 'b (v n d) t -> b v t n d', v=v, n=n)

    return traj.numpy().astype(np.float32) if input_is_numpy else traj

def interpolate_traj_spline(traj, new_seq_len):
    """
    Interpolates the time dimension to new_seq_len using cubic spline interpolation,
    maintaining the first and last points of the trajectory.

    Args:
        traj (torch.Tensor or np.ndarray): The input trajectory of shape (..., horizon, num_queries, dim).
        new_seq_len (int): The desired number of points in the output trajectory.

    Returns:
        Same type as input: The trajectory of shape (..., new_seq_len, num_queries, dim) after interpolation.
    """
    if traj.shape[-3] == new_seq_len:
        return traj
    original_seq_len = traj.shape[-3]
    time_orig = np.linspace(0, 1, original_seq_len)
    time_new = np.linspace(0, 1, new_seq_len)

    input_is_tensor = isinstance(traj, torch.Tensor)
    if input_is_tensor:
        traj = traj.numpy()
    spline = CubicSpline(time_orig, traj, axis=-3)
    interpolated = spline(time_new)

    return torch.tensor(interpolated, dtype=torch.float32) if input_is_tensor else interpolated.astype(np.float32)



def compute_moving_indices(traj, threshold=0.05):
    """
    Compute moving indices for the given trajectory tensor.

    Parameters:
    traj (torch.Tensor): Input tensor of shape (b, v, t, n, d) (normalized to [-1, 1]).

    Returns:
    torch.Tensor: Tensor containing moving indices (1 if moving, 0 if not moving) with shape (b, v, n).
    """
    diff = traj[:, :, 1:] - traj[:, :, :-1]
    total_distance = torch.sum(torch.norm(diff.float(), dim=-1), dim=2)
    if threshold <= 0:
        moving_indices = torch.ones_like(total_distance)
    else:
        moving_indices = (total_distance > threshold).float()

    return moving_indices









class RandomGaussianBlur(object):
    def __init__(self, kernel_sizes=[3, 5, 7], sigma_min=0.1, sigma_max=2.0, p=0.5):
        self.kernel_sizes = kernel_sizes
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            kernel_size = random.choice(self.kernel_sizes)
            sigma = random.uniform(
                self.sigma_min,
                self.sigma_max
            )
            return transforms.functional.gaussian_blur(img, kernel_size, sigma)
        return img



def eulerian_track_motion(flows: np.ndarray) -> float:
    """Mean over probes of the sum of per-window flow magnitudes. flows: (F, N, 3)."""
    return float(np.linalg.norm(flows, axis=-1).sum(axis=0).mean())




def compute_cumulative_camera_motion(extrs: np.ndarray) -> Tuple[float, float]:
    """
    Compute cumulative camera rotation and translation over a horizon window.

    Args:
        extrs: (horizon, 4, 4) camera extrinsics (camera-to-world)

    Returns:
        (cumulative_rotation_deg, cumulative_translation_meters)
    """
    R_curr = extrs[:-1, :3, :3]  # (H-1, 3, 3)
    R_next = extrs[1:, :3, :3]  # (H-1, 3, 3)

    # Relative rotations: R_rel[i] = R_next[i] @ R_curr[i].T
    R_rel = R_next @ R_curr.transpose(0, 2, 1)  # (H-1, 3, 3)

    # cos(θ) from trace
    traces = np.trace(R_rel, axis1=1, axis2=2)  # (H-1,)
    cos_angles = np.clip((traces - 1) / 2, -1.0, 1.0)

    # sin(θ) from skew-symmetric part
    skew = R_rel - R_rel.transpose(0, 2, 1)  # (H-1, 3, 3)
    skew_norms = np.linalg.norm(skew.reshape(-1, 9), axis=1)  # Frobenius per matrix
    sin_angles = skew_norms / (2 * np.sqrt(2))

    angles = np.arctan2(sin_angles, cos_angles)  # (H-1,)
    cumulative_rotation = float(np.sum(angles))

    # Translation
    t_diffs = extrs[1:, :3, 3] - extrs[:-1, :3, 3]  # (H-1, 3)
    cumulative_translation = float(np.sum(np.linalg.norm(t_diffs, axis=1)))

    return float(np.degrees(cumulative_rotation)), cumulative_translation


def frame_to_chunk_position(t, num_chunks, chunk_size, horizon):
    """Map from original frame number to chunk index and position within chunk.

    Returns:
        chunk_idx: Which chunk contains frame t (N dimension)
        start_pos: Position of frame t within that chunk (C dimension)
        end_pos: End position for horizon window (start_pos + horizon)

    Example:
        chunk_size=96, stride=80, horizon=16
        Chunk 0: frames [0, 96)
        Chunk 1: frames [80, 176)
        Chunk 2: frames [160, 256)

        t=0  -> chunk=0, pos=0:16
        t=79 -> chunk=0, pos=79:95
        t=80 -> chunk=1, pos=0:16  (frame 80 is in both chunks, we use later one)
        t=96 -> chunk=1, pos=16:32
    """
    stride = chunk_size - horizon
    chunk_idx = min(t // stride, num_chunks - 1)

    start_pos = t - (chunk_idx * stride)
    end_pos = start_pos + horizon

    assert end_pos <= chunk_size, (
        f"Window [{start_pos, end_pos}) exceeds chunk size {chunk_size}. "
    )
    return chunk_idx, start_pos, end_pos
