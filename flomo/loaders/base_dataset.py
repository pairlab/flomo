import json
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import torch
import torch.nn.functional as Fnn
import torchvision.transforms as T
import torchvision.transforms.functional as TVF
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

from flomo.utils.data_utils import RandomGaussianBlur, optical_flow_traj
from flomo.utils.tracks_normalizer import TracksNormalizer, compute_and_save_tracks_stats


def _lanczos_resize_video(video: np.ndarray, hw) -> np.ndarray:
    """LANCZOS-resize `(T, H, W, 3)` RGB to (h, w), preserving dtype.

    Preprocessing already crops frames to squares; do not crop again.
    """
    h, w = int(hw[0]), int(hw[1])
    if video.shape[1] == h and video.shape[2] == w:
        return video
    frames = video.astype(np.uint8)
    out = np.stack(
        [np.asarray(Image.fromarray(frames[t]).resize((w, h), Image.LANCZOS))
         for t in range(frames.shape[0])],
        axis=0,
    )
    return out.astype(video.dtype)


class BaseDataset(Dataset, ABC):
    """Shared indexing, selective loading, and pixel augmentation, plus the
    backend-independent cotrain loading over EgoVerse Zarr and RealYam/MolmoAct2
    HDF5 storage.

    Subclasses provide index/cache construction, storage primitives, and
    process_data, which defines final sample keys, shapes, and normalization.
    Each index-map entry must contain the paths, frame window, and other
    metadata needed to load one sample without another filesystem scan.
    Subclasses initialize tracks_key, true_horizon, supervision, preprocessing
    settings, img_shape, and CANONICAL_VIEWS, then call
    _setup_views to configure input and output cameras.
    """

    def __init__(
        self,
        root_dir: str,
        dataset_names: List[str],
        track_method: str = 'uniform_4096_reinit_16',
        cond_cameraviews: List[str] = ('agentview',),
        keys_to_load: List[str] = ('images', 'tracks', 'actions'),
        img_shape: Tuple[int, int] = (128, 128),
        true_horizon: int = 16,
        track_pred_horizon: int = 8,
        interp_method: str = 'linear',
        num_tracks: int = 400,
        use_cached_index_map: bool = False,
        aug_cfg: Dict = None,
        subsampling: int = 1,
        tracker: str = '',
        sample_stride: int = 1,
        supervision: str = "point_tracks",
    ):
        super().__init__()
        if supervision not in ("point_tracks", "optical_flow", "cumulative_disp"):
            raise ValueError(
                f"supervision must be 'point_tracks', 'optical_flow', or "
                f"'cumulative_disp', got {supervision!r}"
            )
        self.root_dir = root_dir
        self.dataset_names = list(dataset_names)
        self.track_method = track_method
        self.cond_cameraviews = list(cond_cameraviews)
        self.keys_to_load = list(keys_to_load)
        self.img_shape = img_shape
        self.true_horizon = true_horizon
        self.track_pred_horizon = track_pred_horizon
        self.interp_method = interp_method
        self.num_tracks = num_tracks
        self.use_cached_index_map = use_cached_index_map
        self.aug_cfg = aug_cfg
        self.subsampling = subsampling
        self.tracker = tracker
        self.sample_stride = sample_stride
        self.supervision = supervision

        if subsampling > 1:
            raise ValueError(
                f"subsampling > 1 is not completely supported (got {subsampling}). "
                "Depth and pose loading assume subsampling=1."
            )

        # Current loaders assume reinit-based track files.
        self.reinit = 'reinit' in track_method

        self.index_map = self.get_index_map()

        if self.aug_cfg and self.aug_cfg['color_jitter']:
            cj = self.aug_cfg['color_jitter_strength']
            hue = self.aug_cfg["hue"]
            self.color_jitter = T.ColorJitter(brightness=cj, contrast=cj, saturation=cj, hue=hue)
        if self.aug_cfg and self.aug_cfg['gaussian_blur']:
            self.gaussian_blur = RandomGaussianBlur(
                kernel_sizes=self.aug_cfg['gaussian_blur_kernel_size'],
                sigma_min=self.aug_cfg['gaussian_blur_sigma_min'],
                sigma_max=self.aug_cfg['gaussian_blur_sigma_max'],
                p=self.aug_cfg['gaussian_blur_p'],
            )
        if self.aug_cfg and self.aug_cfg['random_resized_crop']:
            self.rrc_scale = tuple(self.aug_cfg['random_resized_crop_scale'])
            self.rrc_ratio = tuple(self.aug_cfg['random_resized_crop_ratio'])

    @classmethod
    def _check_views(cls, views) -> list:
        """Validate canonical view names against `CANONICAL_VIEWS`."""
        unknown = set(views) - set(cls.CANONICAL_VIEWS)
        if unknown:
            raise ValueError(
                f"{cls.__name__}: camera views {sorted(unknown)} not in "
                f"CANONICAL_VIEWS {sorted(cls.CANONICAL_VIEWS)}"
            )
        return list(views)

    def _setup_views(self, cond_views, pred_views, inputs, outputs) -> None:
        """Configure cond_views for inputs and pred_views for outputs.

        load_vae_latent emits img_<view> and video_<view> according to these roles.
        Images are conditioning-only.
        """
        if "img" in outputs:
            raise ValueError(
                f"{type(self).__name__}: 'img' modality is conditioning-only "
                f"and cannot be an output"
            )
        self.cond_views = self._check_views(list(cond_views))
        self.pred_views = self._check_views(list(pred_views))
        self.img_is_input = "img" in inputs
        self.video_is_input = "video" in inputs
        self.video_is_output = "video" in outputs

    def _setup_tracks_normalizer(
        self,
        normalizer_type: Optional[str],
        stats_path: Optional[str],
        default_stats_path: Path,
        n_samples: int = 5000,
    ) -> Optional[TracksNormalizer]:
        """Load Ramen stats from stats_path or default_stats_path; compute if missing.

        normalizer_type=None disables normalization. Call after building index_map.
        """
        if normalizer_type is None:
            return None
        if normalizer_type != "ramen":
            raise ValueError(f"unknown tracks_normalizer={normalizer_type!r}; only 'ramen' is supported")
        if stats_path is not None:
            path = Path(os.path.expanduser(stats_path))
        else:
            path = Path(default_stats_path)
        if path.exists():
            norm = TracksNormalizer.from_stats_file(path)
        else:
            print(
                f"[tracks_normalizer] {path} missing; computing on the fly from "
                f"this dataset's index_map (this may take a few minutes)…",
                flush=True,
            )
            norm = compute_and_save_tracks_stats(
                self,
                horizon=self.true_horizon,
                out_path=path,
                n_samples=n_samples,
                metadata={"dataset_names": list(self.dataset_names)},
            )
        if norm.horizon != self.true_horizon:
            raise ValueError(
                f"tracks_normalizer horizon ({norm.horizon}) != dataset.true_horizon "
                f"({self.true_horizon}); regenerate stats with the matching horizon"
            )
        return norm

    @abstractmethod
    def get_cache_file(self) -> str:
        """Return the absolute path of the JSON index-map cache."""
        raise NotImplementedError

    @abstractmethod
    def create_index_map(self) -> List[Dict]:
        """Build sample metadata, including paths and frame windows.

        Keep filesystem traversal here rather than in per-sample loaders.
        """
        raise NotImplementedError

    def get_index_map(self) -> List[Dict]:
        cache_file = self.get_cache_file()
        if self.use_cached_index_map:
            try:
                with open(cache_file, 'r') as f:
                    index_map = json.load(f)
                # Intern string paths to reduce memory (paths are duplicated per frame)
                index_map = self._intern_index_map_strings(index_map)
                print(f"Loaded index map from cache: {cache_file}")

                return self._apply_sample_stride(index_map)
            except FileNotFoundError:
                pass

        # Only rank 0 scans the filesystem. File polling avoids NCCL startup
        # timeouts during long scans. Use torchrun's environment because the
        # process group may not exist yet at dataset construction.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        distributed = world_size > 1

        if distributed and rank != 0:
            import time
            print(f"[rank {rank}] Waiting for rank 0 to build index map at {cache_file}...", flush=True)
            while not os.path.exists(cache_file):
                time.sleep(5)
            with open(cache_file, 'r') as f:
                index_map = json.load(f)
            index_map = self._intern_index_map_strings(index_map)
            print(f"[rank {rank}] Loaded index map from cache: {cache_file}", flush=True)
            return self._apply_sample_stride(index_map)

        print("Creating index map...")
        index_map = self.create_index_map()
        print("Index map length: ", len(index_map))
        assert len(index_map) > 0, "Index map is empty"

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        # Atomic publication prevents peers from reading a partial cache.
        import tempfile
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(cache_file), suffix=".tmp")
        with os.fdopen(fd, 'w') as f:
            json.dump(index_map, f)
        os.chmod(tmp, 0o775)
        os.replace(tmp, cache_file)
        print(f"Saved index map to cache: {cache_file}")

        partial_file = cache_file + ".partial"
        if os.path.exists(partial_file):
            os.remove(partial_file)
            print(f"Removed partial checkpoint: {partial_file}")

        return self._apply_sample_stride(index_map)

    def _apply_sample_stride(self, index_map: List[Dict]) -> List[Dict]:
        """Subsample the in-memory index; the on-disk cache retains every entry."""
        if self.sample_stride <= 1:
            return index_map
        filtered = index_map[::self.sample_stride]
        print(
            f"[sample_stride={self.sample_stride}] Using {len(filtered)}/{len(index_map)} samples."
        )
        return filtered
    
    @staticmethod
    def _intern_index_map_strings(index_map: List[Dict]) -> List[Dict]:
        """Intern string values in index_map to reduce memory from path duplication."""
        path_keys = {'track_path', 'video_path', 'text_path', 'text_emb_path',
                     'vae_latent_path', 'vae_latent_h16_path',
                     'depth_path', 'pose_path',
                     'task_file', 'demo_key', 'dataset', 'split'}
        for entry in index_map:
            for key in path_keys:
                if key in entry and isinstance(entry[key], str):
                    entry[key] = sys.intern(entry[key])
        return index_map

    def load_data(self, idx_dict: Dict, explicit_keys: Optional[List[str]] = None) -> Dict:
        """Load one sample's requested modalities.

        explicit_keys overrides keys_to_load without changing the training config.
        motion_tokens aliases raw tracks `(V=1, true_horizon, N, 3)`; viz_video is
        `(1+4*t_frames, H, W, 3)` RGB and intrs is `(1, 3, 3)` — both requested
        independently for visualization.
        """
        want = set(explicit_keys) if explicit_keys is not None else set(self.keys_to_load)
        data: Dict = {}

        if "images" in want:
            data.update(self.load_images(idx_dict))
        if "actions" in want:
            data.update(self.load_actions(idx_dict))
        if "tracks" in want or "vis" in want:
            data.update(
                self.load_tracks(
                    idx_dict,
                    include_tracks="tracks" in want,
                    include_vis="vis" in want,
                )
            )
        if "text" in want or "text_emb" in want:
            data.update(
                self.load_text(
                    idx_dict,
                    include_text="text" in want,
                    include_text_emb="text_emb" in want,
                )
            )
        if "depth" in want:
            data.update(self.load_depth(idx_dict))
        if "pose" in want:
            data.update(self.load_pose(idx_dict))
        if "vae_latent" in want:
            data.update(self.load_vae_latent(idx_dict))
        if "motion_tokens" in want:
            tr = self.load_tracks(idx_dict, include_tracks=True, include_vis=False)
            data["motion_tokens"] = np.nan_to_num(
                tr["tracks"], nan=0.0, posinf=0.0, neginf=0.0,
            ).astype(np.float32)                                    # (V=1, true_horizon, N, 3)
            if "init_pts" in tr:
                # cumulative_disp side channel: frame-0 XYZ for track viz reconstruction.
                data["init_pts"] = tr["init_pts"].astype(np.float32)
        if "viz_video" in want and "viz_video" not in data:
            data["viz_video"] = self._read_video_window(
                self._rgb_video_path(idx_dict), self.view, int(idx_dict["start_t"])
            )
        if "intrs" in want and "intrs" not in data:
            data["intrs"] = self.load_pose(idx_dict)["intrs"]
        return data

    @abstractmethod
    def load_images(self, idx_dict: Dict) -> Dict:
        """Return float32 images `(V, H, W, C)` in [0, 255] or [0, 1]."""
        raise NotImplementedError

    @abstractmethod
    def load_actions(self, idx_dict: Dict) -> Dict:
        """Return float32 actions `(T, D)` for [start_t, end_t)."""
        raise NotImplementedError

    def load_tracks(self, idx_dict: Dict, include_tracks: bool = True,
                    include_vis: bool = False) -> Dict:
        """Load tracks `(1, true_horizon, N, 3)` and optional visibility `(1, horizon, N)`.

        Optical flow integrates fresh one-step windows into window-0's camera
        frame and also returns raw_windows. Point tracks use one native world-frame
        trajectory. Cumulative displacement subtracts its frame-0 anchor;
        process_data may then apply Ramen normalization.
        """
        track_idx = self._track_idx(idx_dict)
        H = self.true_horizon
        out: Dict[str, np.ndarray] = {}
        with self._tracks_group(idx_dict) as vg:
            if include_tracks:
                if self.supervision == "optical_flow":
                    # Keep each reinitialized window's first step.
                    raw = np.asarray(
                        vg[self.tracks_key][track_idx:track_idx + H, :2]
                    ).astype(np.float32)                                       # (H, 2, N, 3)
                    if raw.shape[0] != H:
                        raise IndexError(
                            f"optical_flow: requested {H} reinit windows from "
                            f"track_idx={track_idx}, got {raw.shape[0]} "
                            f"(rollout_len={idx_dict['rollout_len']})"
                        )
                    extrs = self._read_flow_extrs(idx_dict, H - 1)             # (H-1, 4, 4)
                    out["tracks"] = optical_flow_traj(raw, extrs)              # (1, H, N, 3)
                    out["raw_windows"] = raw[None]                             # (1, H, 2, N, 3)
                else:
                    tracks = np.asarray(
                        vg[self.tracks_key][track_idx:track_idx + 1]
                    ).astype(np.float32)                                       # (1, H_pp, N, 3)
                    if tracks.shape[0] != 1:
                        raise IndexError(
                            f"{self.supervision}: window track_idx={track_idx} out of range"
                        )
                    if self.supervision == "cumulative_disp":
                        # Stash frame-0 XYZ before zero-anchoring; viz_tracks
                        # needs it to reconstruct metric tracks for projection.
                        out["init_pts"] = tracks[:, :1].copy()                 # (1, 1, N, 3)
                        tracks = tracks - tracks[:, :1]                        # (1, H_pp, N, 3)
                    out["tracks"] = tracks
                out["valid_velocity_mask"] = np.ones(
                    (1, out["tracks"].shape[2]), dtype=np.float32
                )
            if include_vis and "vis" in vg:
                out["vis"] = np.asarray(
                    vg["vis"][track_idx:track_idx + 1]
                ).astype(np.float32)
        return out

    def load_vae_latent(self, idx_dict: Dict) -> Dict:
        """Return requested img_<view>/video_<view> latents or live RGB frames.

        Preprocessed shapes: img `(48, 1, h, w)` (window start_t, latent 0);
        video `(48, T, h, w)` with T=5 for inputs (including latent 0) or T=4
        for output targets. Live shapes are `(H, W, 3)` for images and
        `(1 + 4*t_frames, H, W, 3)` for video.

        Input views come from cond_views; output views come from pred_views.
        """
        start_t = int(idx_dict["start_t"])
        out: Dict[str, np.ndarray] = {}

        img_views = set(self.cond_views) if self.img_is_input else set()

        video_input_views = set(self.cond_views) if self.video_is_input else set()
        video_output_views = set(self.pred_views) if self.video_is_output else set()
        video_views = video_input_views | video_output_views

        for canon in sorted(img_views | video_views):
            view_key = self.CANONICAL_VIEWS[canon]
            want_img = canon in img_views
            want_video = canon in video_views

            if (want_img and self.img_use_preprocessed) or (
                want_video and self.video_use_preprocessed
            ):
                with self._vae_latent_array(idx_dict, view_key) as ds:
                    if start_t >= ds.shape[0]:
                        raise IndexError(
                            f"start_t={start_t} past vae_latent_h16 length {ds.shape[0]}"
                        )
                    if want_img and self.img_use_preprocessed:
                        cond = np.asarray(ds[start_t][:, 0])                  # (48, h, w)
                        out[f"img_{canon}"] = cond[:, None].astype(np.float32) # (48, 1, h, w)
                    if want_video and self.video_use_preprocessed:
                        # Output video excludes the separately conditioned latent 0.
                        lo = 0 if self.video_is_input else 1
                        targets = np.asarray(ds[start_t][:, lo:5])            # (48, 5|4, h, w)
                        out[f"video_{canon}"] = targets.astype(np.float32)

            live_img = want_img and not self.img_use_preprocessed
            if live_img or (want_video and not self.video_use_preprocessed):
                window = self._read_video_window(
                    self._rgb_video_path(idx_dict), view_key, start_t
                )
                # Match the live VAE's pixel resolution without another crop.
                window = _lanczos_resize_video(window, self.img_shape)
                if live_img:
                    out[f"img_{canon}"] = window[0].astype(np.float32)         # (H, W, 3)
                if want_video and not self.video_use_preprocessed:
                    out[f"video_{canon}"] = window.astype(np.float32)
        return out

    @abstractmethod
    def load_text(
        self,
        idx_dict: Dict,
        include_text: bool = True,
        include_text_emb: bool = False,
    ) -> Dict:
        """Return optional text strings and/or float32 text_emb arrays."""
        raise NotImplementedError

    def _tracks_group(self, idx_dict: Dict):
        """Context manager yielding the per-view group that holds
        `self.tracks_key` (and optionally a `vis` array)."""
        raise NotImplementedError

    def _vae_latent_array(self, idx_dict: Dict, view_key: str):
        """Context manager yielding the `(N, 48, 5, h, w)` `vae_latent_h16`
        array for `view_key`."""
        raise NotImplementedError

    def _track_idx(self, idx_dict: Dict) -> int:
        """Index into the tracks array for this window's `start_t`."""
        raise NotImplementedError

    def _read_flow_extrs(self, idx_dict: Dict, n: int) -> np.ndarray:
        """`(n, 4, 4)` camera extrinsics for the first `n` windows from `start_t`."""
        raise NotImplementedError

    def _read_video_window(self, video_path: str, view_key: str, start_t: int) -> np.ndarray:
        """Read the raw RGB video window (start_t : start_t+horizon+1) for live VAE encoding."""
        raise NotImplementedError

    def _rgb_video_path(self, idx_dict: Dict) -> str:
        """Path passed to `_read_video_window` for live (un-preprocessed) RGB."""
        raise NotImplementedError

    @abstractmethod
    def process_data(self, data: Dict) -> Dict:
        """Finalize sample shapes, ranges, normalization, and output keys."""
        raise NotImplementedError

    def _apply_color_blur_chw01(self, batch: torch.Tensor) -> torch.Tensor:
        """Share one color/blur transform across `(K, 3, H, W)` float frames in [0, 1]."""
        if self.aug_cfg['color_jitter']:
            batch = self.color_jitter(batch)
        if self.aug_cfg['gaussian_blur']:
            batch = self.gaussian_blur(batch)
        return torch.clamp(batch, 0.0, 1.0)

    def _sample_rrc_params(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """One (top, left, height, width) crop window at (h, w)."""
        ref = torch.empty(1, 1, h, w)
        return T.RandomResizedCrop.get_params(ref, scale=list(self.rrc_scale), ratio=list(self.rrc_ratio))

    @staticmethod
    def _apply_rrc_chw(batch: torch.Tensor, params: Tuple[int, int, int, int]) -> torch.Tensor:
        """Crop+resize a (K, C, H, W) float batch back to (K, C, H, W)."""
        H, W = batch.shape[-2], batch.shape[-1]
        i, j, h, w = params
        return TVF.resized_crop(batch, i, j, h, w, size=[H, W], antialias=True)

    @staticmethod
    def _apply_rrc_intrs(intrs, params: Tuple[int, int, int, int], src_size: int):
        """Update `(V, 3, 3)` intrinsics for crop+resize, preserving numpy/torch type.

        fx' = fx*src_size/w; cx' = (cx-j)*src_size/w, and likewise for fy/cy.
        This keeps projected tracks aligned with augmented pixels.
        """
        is_torch = torch.is_tensor(intrs)
        arr = intrs.detach().cpu().numpy().copy() if is_torch else np.asarray(intrs).copy()
        i, j, h, w = params
        sy, sx = float(src_size) / float(h), float(src_size) / float(w)
        arr[..., 0, 0] = arr[..., 0, 0] * sx
        arr[..., 1, 1] = arr[..., 1, 1] * sy
        arr[..., 0, 2] = (arr[..., 0, 2] - j) * sx
        arr[..., 1, 2] = (arr[..., 1, 2] - i) * sy
        return torch.from_numpy(arr) if is_torch else arr

    def _apply_rrc_motion_tokens(
        self, mt, params: Tuple[int, int, int, int], src_size: int,
    ) -> torch.Tensor:
        """Crop `(1, T, N, 3)` motion tracks with the primary camera's window.

        Preserve the model's `(w h)` flattening order. Upsample to src_size
        before the pixel-space crop, then resize back to the square track grid.
        """
        if isinstance(mt, np.ndarray):
            mt = torch.from_numpy(mt)
        V, T_, N, D = mt.shape
        if V != 1:
            raise ValueError(f"motion_tokens crop requires V=1, got V={V}")
        if D != 3:
            raise ValueError(f"motion_tokens last dim must be 3, got {D}")
        side = int(math.isqrt(N))
        if side * side != N:
            raise ValueError(f"motion_tokens N must be a perfect square, got N={N}")
        x = rearrange(mt.float(), 'v t (w h) d -> (v t) d h w', h=side, w=side)
        x = Fnn.interpolate(x, size=(src_size, src_size), mode='bilinear', align_corners=False)
        i, j, h, w = params
        x = TVF.resized_crop(x, i, j, h, w, size=[side, side], antialias=True)
        return rearrange(x, '(v t) d h w -> v t (w h) d', v=V, t=T_)

    def augment_data(self, data: Dict) -> Dict:
        """Share augmentation parameters across each camera's frames.

        Supports live img_<view>/video_<view> RGB arrays and legacy images
        `(V, F, H, W, 3)`; skips preprocessed latents. Each camera gets an
        independent transform, shared by its conditioning and target frames.

        Motion tokens `(1, T, N, 3)` receive the primary camera's spatial crop,
        but no color jitter or blur.
        """
        if not self.aug_cfg:
            return data
        do_color = bool(self.aug_cfg['color_jitter'])
        do_blur = bool(self.aug_cfg['gaussian_blur'])
        do_crop = bool(self.aug_cfg['random_resized_crop'])
        if not (do_color or do_blur or do_crop):
            return data

        if 'images' in data:
            imgs = data['images']
            if isinstance(imgs, np.ndarray):
                imgs = torch.from_numpy(imgs)
            V = imgs.shape[0]
            per_view = []
            for v in range(V):
                x = rearrange(imgs[v], 'f h w c -> f c h w')
                if do_color or do_blur:
                    x = self._apply_color_blur_chw01(x)
                if do_crop:
                    params = self._sample_rrc_params(x.shape[-2], x.shape[-1])
                    x = self._apply_rrc_chw(x, params)
                per_view.append(rearrange(x, 'f c h w -> f h w c'))
            data['images'] = torch.stack(per_view, dim=0)

        by_view: Dict[str, List[Tuple[str, object]]] = {}
        for k, v in data.items():
            if k == 'images' or not isinstance(v, (np.ndarray, torch.Tensor)):
                continue
            if k.startswith('img_'):
                view = k[len('img_'):]
            elif k.startswith('video_'):
                view = k[len('video_'):]
            else:
                continue
            if len(v.shape) == 0 or int(v.shape[-1]) != 3:
                continue  # latent or non-RGB; skip
            by_view.setdefault(view, []).append((k, v))

        primary_rrc_params: Optional[Tuple[int, int, int, int]] = None
        primary_src_size: Optional[int] = None
        for view, items in by_view.items():
            # Batch the view's frames to sample one transform; restore shapes/ranges below.
            tensors: List[torch.Tensor] = []
            meta: List[Tuple[str, Tuple, int, int, bool]] = []
            for k, v in items:
                t = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                orig_shape = tuple(t.shape)
                was_high = bool(t.float().max() > 1.0)
                tf = t.float() / 255.0 if was_high else t.float()
                if tf.ndim == 3:
                    tt = rearrange(tf, 'h w c -> 1 c h w')
                elif tf.ndim == 4:
                    tt = rearrange(tf, 't h w c -> t c h w')
                else:
                    raise ValueError(
                        f"augment_data: unexpected shape {orig_shape} for key {k!r}"
                    )
                start = sum(x.shape[0] for x in tensors)
                tensors.append(tt)
                meta.append((k, orig_shape, start, start + tt.shape[0], was_high))

            batch = torch.cat(tensors, dim=0).contiguous()  # (K, 3, H, W) in [0, 1]
            if do_color or do_blur:
                batch = self._apply_color_blur_chw01(batch)
            if do_crop:
                params = self._sample_rrc_params(batch.shape[-2], batch.shape[-1])
                batch = self._apply_rrc_chw(batch, params)
                if view == 'primary':
                    primary_rrc_params = params
                    primary_src_size = int(batch.shape[-2])

            for k, orig_shape, s, e, was_high in meta:
                chunk = batch[s:e]
                if len(orig_shape) == 3:
                    out = rearrange(chunk, '1 c h w -> h w c')
                else:
                    out = rearrange(chunk, 't c h w -> t h w c')
                if was_high:
                    out = out * 255.0
                data[k] = out

        # --- Motion tokens: crop with primary view's crop ---------------------
        if do_crop and 'motion_tokens' in data and by_view:
            if primary_rrc_params is None:
                raise ValueError(
                    "augment_data: random_resized_crop enabled with motion_tokens "
                    "present, but no primary-view RGB key (img_primary / video_primary) "
                    "found to derive a matching crop from."
                )
            data['motion_tokens'] = self._apply_rrc_motion_tokens(
                data['motion_tokens'], primary_rrc_params, primary_src_size,
            )
            # cumulative_disp viz reconstructs absolute XYZ as `init_pts + cumulative_disp`;
            # the cropped cumulative_disp grid is sampled at the cropped image's pixel
            # positions, so init_pts must be resampled the same way or the sum
            # is geometrically meaningless (token n's old anchor + token n's new
            # displacement -> projects to junk pixels).
            if 'init_pts' in data:
                data['init_pts'] = self._apply_rrc_motion_tokens(
                    data['init_pts'], primary_rrc_params, primary_src_size,
                )

        # --- Intrs: crop with primary view's crop so downstream projections ---
        # (e.g. the model's `viz_tracks`) line up with the cropped pixels.
        if do_crop and 'intrs' in data and primary_rrc_params is not None:
            data['intrs'] = self._apply_rrc_intrs(
                data['intrs'], primary_rrc_params, primary_src_size,
            )

        return data

    def _start_valid(self, start_t: int, rollout_len: int) -> bool:
        """Is `start_t` a valid window start under the current supervision mode?
        optical_flow needs `true_horizon` consecutive reinit windows, so the
        last `true_horizon-1` starts per episode are invalid; point_tracks
        accepts any in-range start (short tails are truncated/padded).
        """
        if self.supervision == "optical_flow":
            return (start_t // self.subsampling) + self.true_horizon <= rollout_len // self.subsampling
        return 0 <= start_t < rollout_len

    def __len__(self):
        return len(self.index_map)

    # Treated as "broken sample" — caught, recorded, resampled.
    _BROKEN_EXC = (OSError, RuntimeError, KeyError, IndexError, ValueError, EOFError)

    def __getitem__(self, idx, **kwargs):
        # Records ride up via `_broken_samples` for wandb/JSONL logging.
        n = len(self.index_map)
        if not hasattr(self, "_resample_rng"):
            self._resample_rng = np.random.default_rng()
        broken: List[Dict] = []
        cur = idx
        ds_name = getattr(self, "cotrain_name", type(self).__name__)
        for _ in range(8):
            idx_dict = self.index_map[cur]
            try:
                data = self.load_data(idx_dict)
                break
            except self._BROKEN_EXC as e:
                broken.append({
                    "dataset": ds_name,
                    "idx": int(cur),
                    "clip": idx_dict.get("demo_id") or idx_dict.get("video_path", "?"),
                    "start_t": int(idx_dict["start_t"]),
                    "err_type": type(e).__name__,
                    "err_msg": str(e)[:200],
                })
                print(f"[broken] {ds_name} idx={cur}: {type(e).__name__}: {e}", flush=True)
                cur = int(self._resample_rng.integers(0, n))
        else:
            raise RuntimeError(f"8 consecutive broken samples starting at idx={idx}")
        data = self.process_data(data)

        if self.aug_cfg is not None:
            data = self.augment_data(data)

        data["__meta__"] = {"idx_dict": idx_dict.copy()}
        if hasattr(self, "cotrain_name"):
            data["dataset_name"] = self.cotrain_name
            sub = self._cotrain_subname(idx_dict)
            if sub is not None:
                data["dataset_subname"] = sub
        if broken:
            data["_broken_samples"] = broken
        return data

    def _cotrain_subname(self, idx_dict) -> Optional[str]:
        """Optional sub-folder label for the per-(dataset, sub) val breakdown."""
        return None
