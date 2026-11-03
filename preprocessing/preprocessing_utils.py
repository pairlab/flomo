import hashlib
import os
import sys
from typing import Any, Dict

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from einops import rearrange, repeat
from flomo.utils.cfg_utils import get_device
from flomo.utils.config_validation import validate_track_config

spatracker_path = os.path.join(os.getcwd(), "SpaTrackerV2")
if os.path.isdir(spatracker_path) and spatracker_path not in sys.path:
    sys.path.insert(0, spatracker_path)
vggt_path = os.path.join(os.getcwd(), "vggt")
if os.path.isdir(vggt_path) and vggt_path not in sys.path:
    sys.path.insert(0, vggt_path)

def _get_tracks_common_dir(cfg, base_dir):
    """Get the common directory for storing tracks.
    e.g. <base_dir>/uniform_400_reinit_16
    """
    reinit_str = f"_reinit_{cfg.horizon}" if (cfg.reinit) else ""
    common_params_folder = f"{cfg.init_queries}_{cfg.n_tracks}{reinit_str}"
    common_dir = os.path.join(base_dir, common_params_folder)
    return common_dir

def _get_output_subfolder(cfg, output_type: str) -> str:
    """Get the output-specific subfolder name within the common params
    folder, e.g. spatialtracker_2d.
    """
    if cfg.mode == "tracks":
        # e.g., "spatialtracker_2d"
        return f"{cfg.tracker_type}_{output_type}"
    else:
        # For non-track modes, no subfolder needed
        raise ValueError("Non-track modes do not have output specific sub-folders.")

def save_tracks_config(cfg: Any, base_dir: str) -> None:
    """
    Save config.yaml for each of the outputs produced by the TrackProcessor
    Example: if track_outputs = ['2d', '3d', 'pose', 'depth'], then 4 copies of config.yaml will be saved at each output location
    """
    validate_track_config(cfg)

    common_dir = _get_tracks_common_dir(cfg, base_dir)
    os.makedirs(common_dir, exist_ok=True)
    for output_type in cfg.track_outputs:
        output_subfolder = f"{cfg.tracker_type}_{output_type}"
        save_dir = os.path.join(common_dir, output_subfolder)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(save_dir, "config.yaml"))

def load_tracker(cfg: Any, models: Dict[str, Any]) -> Any:
    """
    Load tracker model(s) based on the configuration into 'models' dict.
    """
    if cfg.tracker_type == "spatialtracker":
        from SpaTrackerV2.models.SpaTrackV2.models.predictor import Predictor
        from SpaTrackerV2.models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track

        device = get_device()
        vggt4track = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
        predictor = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
        if vggt4track is None:
            raise RuntimeError("Failed to load VGGT4Track model")
        if predictor is None:
            raise RuntimeError("Failed to load SpatialTracker Predictor model")

        models["vggt4track"] = vggt4track.eval().to(device)
        predictor.eval()
        predictor.to(device)
        models["spatialtracker"] = predictor
    elif cfg.tracker_type == "vggt":
        from vggt.models.vggt import VGGT
        device = get_device()
        vggt = VGGT.from_pretrained("facebook/VGGT-1B")
        if vggt is None:
            raise RuntimeError("Failed to load VGGT model")
        models["vggt"] = vggt.eval().to(device)
    else:
        raise ValueError(f"{cfg.tracker_type} is not a valid tracker type")
    return models


def get_track_processor_func(cfg: Any, models: Dict[str, Any]) -> (Any, Dict[str, Any]):
    """
    Get the appropriate track extraction function and required models based on the tracker type.
    """
    # Lazy imports to avoid circular dependencies
    from preprocessing.SpatialTracker import spatial_tracks_from_video
    from preprocessing.VGGT import vggt_tracks_from_video

    if cfg.tracker_type == "spatialtracker":
        track_fn = spatial_tracks_from_video
        models = {
            "vggt_model": models["vggt4track"],
            "track_model": models["spatialtracker"]
        }
    elif cfg.tracker_type == "vggt":
        track_fn = vggt_tracks_from_video
        models = {
            "vggt_model": models["vggt"]
        }
    else:
        raise ValueError(f"{cfg.tracker_type} is not "
                         f"a valid tracker type")

    return track_fn, models


def inital_save_h5(path, skip_exist, view_names=["agentview", "eye_in_hand"]):
    try:
        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                file_processed = True
                for view in view_names:
                    if f"root/{view}" not in f:
                        file_processed = False
                        break
                if file_processed and skip_exist:
                    print(f"File {path} already exists, skipping.")
                    return None
    except Exception as e:
        print(f"Error in opening file {path}, {e}")

    f = h5py.File(path, 'w')
    return f


def concatenate_views(videos_dict: Dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """
    Concatenate multiple camera views temporally for multi-view processing.

    Args:
        videos_dict: Dict mapping view names to videos, each (T, C, H, W)

    Returns:
        concatenated_video: (T*num_views, C, H, W) array with interleaved views
        view_order: List of view names in the order they appear in concatenation

    Concatenation pattern: [v0_t0, v1_t0, v0_t1, v1_t1, ..., v0_tn, v1_tn]
    """
    view_order = list(videos_dict.keys())
    videos = [videos_dict[view] for view in view_order]

    # Check all videos have same shape
    T, C, H, W = videos[0].shape
    for video in videos[1:]:
        assert video.shape == (T, C, H, W), f"All videos must have same shape, got {video.shape} vs {(T, C, H, W)}"

    # Stack and interleave: (num_views, T, C, H, W) -> (T, num_views, C, H, W) -> (T*num_views, C, H, W)
    stacked = np.stack(videos, axis=0)  # (num_views, T, C, H, W)
    stacked = np.transpose(stacked, (1, 0, 2, 3, 4))  # (T, num_views, C, H, W)
    concatenated = stacked.reshape(T * len(view_order), C, H, W)

    return concatenated, view_order


def prepare_video_for_vggt(video, dim_order, multi_view, device):
    """Convert input video(s) to tensor and concatenate if multi-view.

    Returns:
        video_tensor: (T_total, C, H, W) tensor
        T: Original timesteps per view
        num_views: Number of views (1 if single-view)
        view_order: List of view names (None if single-view)
    """
    if multi_view:
        assert isinstance(video, dict), "multi_view=True requires video to be Dict[str, np.ndarray]"

        # Convert each view from thwc to tchw if needed
        if dim_order == 'thwc':
            video = {view: rearrange(v, "t h w c -> t c h w") for view, v in video.items()}

        # Concatenate views temporally
        video_concat, view_order = concatenate_views(video)
        T = video[view_order[0]].shape[0]  # Original T per view
        video_tensor = torch.from_numpy(video_concat).to(device).float()
        num_views = len(view_order)
    else:
        if dim_order == 'thwc':
            video = rearrange(video, "t h w c -> t c h w")
        video_tensor = torch.from_numpy(video).to(device).float()
        T = video_tensor.shape[0]
        num_views = 1
        view_order = None

    assert video_tensor.shape[1] == 3, "Video must have 3 channels (RGB)"
    return video_tensor, T, num_views, view_order


def pad_video_for_tracking(video_preprocessed, horizon, num_views, T):
    """Pad video so last frame can also be tracked.

    Args:
        video_preprocessed: (T_total, C, H, W) preprocessed video
        horizon: Number of frames to track forward
        num_views: Number of views (affects padding pattern)
        T: Original timesteps per view

    Returns:
        video_padded: (T_total + (horizon-1)*num_views, C, H, W)
    """
    if num_views > 1:
        # For multi-view, pad with alternating pattern of last frame from each view
        last_frames = []
        for i in range(num_views):
            last_frame_idx = T * num_views - num_views + i  # Last occurrence of view i
            last_frames.append(video_preprocessed[last_frame_idx])

        # Create padding by repeating [v0_last, v1_last, ...] (horizon-1) times
        padding_frames = []
        for _ in range(horizon - 1):
            padding_frames.extend(last_frames)
        padding = torch.stack(padding_frames, dim=0)
    else:
        # Single-view: just repeat last frame
        padding = repeat(video_preprocessed[-1], "c h w -> t c h w", t=horizon - 1)

    return torch.cat([video_preprocessed, padding], dim=0)


def preprocess_image(img_tensor, mode="crop", target_size=518, keep_ratio=False):
    """
    Taken from SpaTrackerV2.models.SpaTrackV2.models.vggt4track.utils.load_fn.
    Preprocess image tensor(s) to target size with crop or pad mode.
    Args:
        img_tensor (torch.Tensor): Image tensor of shape (C, H, W) or (T, C, H, W), values in [0, 1]
        mode (str): 'crop' or 'pad'
        target_size (int): Target size for width/height
    Returns:
        torch.Tensor: Preprocessed image tensor(s), same batch dim as input normalized between 0 and 1
    """
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")
    if img_tensor.dim() == 3:
        tensors = [img_tensor]
        squeeze = True
    elif img_tensor.dim() == 4:
        tensors = list(img_tensor)
        squeeze = False
    else:
        raise ValueError("Input tensor must be (C, H, W) or (T, C, H, W)")
    processed = []
    for img in tensors:
        C, H, W = img.shape
        if mode == "pad":
            if W >= H:
                new_W = target_size
                new_H = round(H * (new_W / W) / 14) * 14
            else:
                new_H = target_size
                new_W = round(W * (new_H / H) / 14) * 14
            out = torch.nn.functional.interpolate(img.unsqueeze(0), size=(new_H, new_W), mode="bicubic", align_corners=False).squeeze(0)
            h_padding = target_size - new_H
            w_padding = target_size - new_W
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            if h_padding > 0 or w_padding > 0:
                out = torch.nn.functional.pad(
                    out, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
        else:  # crop
            new_W = target_size
            new_H = round(H * (new_W / W) / 14) * 14
            out = torch.nn.functional.interpolate(img.unsqueeze(0), size=(new_H, new_W), mode="bicubic", align_corners=False).squeeze(0)
            if keep_ratio==False:
                if new_H > target_size:
                    start_y = (new_H - target_size) // 2
                    out = out[:, start_y : start_y + target_size, :]
        processed.append(out)
    result = torch.stack(processed)
    if squeeze:
        return result[0]
    return result


def pad_tensor(tensor: torch.tensor, chunk_size: int, identity: bool = False) -> torch.tensor:
    """Pad tensor to chunk size."""
    pad_len = chunk_size - tensor.shape[0]

    if identity and tensor.ndim > 3:
        raise ValueError("Identity padding only supported for 3D tensors.")

    if tensor.dim() == 3:  # Special case where identity padding might be used for 3D matrices
        d1, d2 = tensor.shape[1:]
        if identity and d1 == d2:
            pad_item = torch.eye(d1, device=tensor.device, dtype=tensor.dtype)
        else:
            pad_item = torch.zeros((d1, d2), device=tensor.device, dtype=tensor.dtype)
        padding = repeat(pad_item, "d1 d2 -> t d1 d2", t=pad_len)

    # For tensors with more dimensions, zero-padding is used
    else:
        padding = torch.zeros(
            (pad_len, *tensor.shape[1:]),
            device=tensor.device,
            dtype=tensor.dtype
        )

    return torch.cat([tensor, padding], dim=0)

def scale_intrinsics(intrinsics, original_size: tuple, target_size: tuple):
    """
    Scale camera intrinsics to match a new image resolution.

    Args:
        intrinsics: (..., 3, 3) camera intrinsic matrices
        original_size: (H, W) original image size
        target_size: (H, W) target image size

    Returns:
        Scaled intrinsics with same shape as input
    """
    original_h, original_w = original_size
    target_h, target_w = target_size

    scale_y = target_h / original_h
    scale_x = target_w / original_w

    # Clone to avoid modifying original
    if isinstance(intrinsics, torch.Tensor):
        scaled_intrinsics = intrinsics.clone()
    elif isinstance(intrinsics, np.ndarray):
        scaled_intrinsics = np.copy(intrinsics)
    else:
        raise TypeError("intrinsics must be a torch.Tensor or np.ndarray")

    # Scale focal lengths and principal point
    scaled_intrinsics[..., 0, 0] *= scale_x  # fx
    scaled_intrinsics[..., 1, 1] *= scale_y  # fy
    scaled_intrinsics[..., 0, 2] *= scale_x  # cx
    scaled_intrinsics[..., 1, 2] *= scale_y  # cy

    return scaled_intrinsics

def split_views(concat_array: np.ndarray, view_order: list[str]) -> Dict[str, np.ndarray]:
    """
    Split concatenated multi-view array back into separate views.

    Args:
        concat_array: (N, C, H, W) where C contains interleaved views (VGGT outputs)
        view_order: List of view names in the order they were concatenated

    Returns:
        Dict mapping view names to their split arrays, each (N, C//num_views, ...)
    """
    num_views = len(view_order)
    results = {}

    # Extract each view using slicing with stride
    for i, view_name in enumerate(view_order):
        results[view_name] = concat_array[:, i::num_views]

    return results



def split_indices(total: int, num_workers: int, worker_id: int) -> np.ndarray:
    """Deterministic (num_workers, worker_id) shard of range(total)."""
    if num_workers <= 1:
        return np.arange(total)
    if not 0 <= worker_id < num_workers:
        raise ValueError(f"worker_id {worker_id} must be in [0, {num_workers}).")
    return np.array_split(np.arange(total), num_workers)[worker_id]


def _stable_bucket(sample_id: str, seed: int) -> float:
    key = f"{seed}:{sample_id}".encode("utf-8")
    return int(hashlib.sha1(key).hexdigest()[:8], 16) / float(0xFFFFFFFF)


def in_split(sample_id: str, split: str, val_ratio: float, seed: int) -> bool:
    """Return True iff `sample_id` belongs to `split` ('train' / 'val' / 'all').

    A stable sha1(seed:sample_id) bucket below `val_ratio` lands the sample in
    'val', the rest in 'train'.
    """
    if split == "all":
        return True
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    is_val = _stable_bucket(sample_id, seed) < val_ratio
    if split == "train":
        return not is_val
    if split in {"val", "validation"}:
        return is_val
    raise ValueError(f"Unsupported split '{split}'. Expected train, val, validation, or all.")
