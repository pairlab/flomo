import glob
import json
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import zarr
from einops import rearrange
from torchvision.transforms import Resize
from omegaconf import OmegaConf
from tqdm import tqdm

from pathlib import Path

from flomo.loaders.base_dataset import BaseDataset
from flomo.utils.data_utils import (
    compute_cumulative_camera_motion,
    eulerian_track_motion,
    interpolate_traj,
    interpolate_traj_spline,
)
from flomo.utils.tracks_normalizer import TracksNormalizer


class EgoVerseDataset(BaseDataset):
    """Zarr-backed EgoVerse loader.

    Preprocessing layout (split = train | val | test):
        <root>/<split>/<track_method>/<tracker>_<mode>/<task>/<prefix>/<sample>.zarr
            root/<view>/{<tracks_key>(T, H_pp, N, 3), ...}
        <root>/<split>/{video,text,meta,vae_latent_h16,segment_text_emb}/<task>/<prefix>/<sample>.zarr
        <root>/<split>/<track_method>/<tracker>_pose/.../camera_extrinsics(num_chunks, chunk_size, 4, 4)
        <root>/<split>/<track_method>/<tracker>_depth/.../depth_map(num_chunks, chunk_size, H, W)

    Tracks are stored as fresh-grid + 1-step-flow per window: shape `(T, H_pp, N, 3)`
    where `H_pp` (= preprocessing_horizon, e.g. 16) is unrelated to `true_horizon`
    (= the loader's window length, e.g. 16). The loader only consumes `tracks[:, :2]`
    (start + 1-step displacement). Pose/depth are stored in overlapping chunks of
    size `chunk_size` with step `chunk_size - preprocessing_horizon`; per-window
    reads must fit in one chunk, enforced by the `true_horizon <= chunk_size`
    precondition in `__init__`.
    """

    # EgoVerse exposes one canonical camera.
    CANONICAL_VIEWS = {"primary": "default_view"}

    def __init__(
        self,
        root_dir: str,
        dataset_names: List[str],
        track_method: str,
        cond_views: List[str] = ("primary",),
        pred_views: List[str] = (),
        inputs: List[str] = ("img",),
        outputs: List[str] = (),
        keys_to_load: List[str] = ("images", "tracks", "text"),
        img_shape: Tuple[int, int] = (256, 256),
        true_horizon: int = 16,
        track_pred_horizon: int = 16,
        interp_method: str = "none",
        num_tracks: Optional[int] = None,
        subsampling: Optional[int] = None,
        use_cached_index_map: bool = False,
        aug_cfg: Optional[Dict] = None,
        dataset_path: Optional[str] = None,
        tracker: str = "spatialtracker",
        tracker_mode: str = "3d",
        sample_stride: int = 1,
        supervision: str = "point_tracks",
        min_track_motion: Optional[float] = None,
        max_track_motion: Optional[float] = None,
        max_cam_rotation: Optional[float] = None,
        max_cam_translation: Optional[float] = None,
        img_use_preprocessed: bool = True,
        video_use_preprocessed: bool = True,
        tracks_normalizer: Optional[str] = None,
        tracks_normalizer_stats_path: Optional[str] = None,
    ) -> None:
        # Set fields needed for _load_preprocessing_cfg() before we call it.
        self.root_dir = root_dir
        self.dataset_names = list(dataset_names)
        self.track_method = track_method
        self.dataset_path = dataset_path
        self.tracker = tracker
        self.tracker_mode = tracker_mode
        self._setup_views(cond_views, pred_views, inputs, outputs)
        self.view = self.CANONICAL_VIEWS["primary"]
        self.true_horizon = int(true_horizon)

        if self.tracker_mode == "3d":
            self.tracks_key = "3d_tracks"
        elif self.tracker_mode == "2d":
            raise NotImplementedError(
                "EgoVerse zarrs only ship '3d_tracks'; tracker_mode='2d' is not supported."
            )
        else:
            raise ValueError(f"Unknown tracker mode '{self.tracker_mode}' for EgoVerse.")

        # Read stored geometry from preprocessing config, not caller defaults.
        pp_cfg = self._load_preprocessing_cfg()
        self.chunk_size = int(pp_cfg["chunk_size"])
        self.preprocessing_horizon = int(pp_cfg["horizon"])
        self.stride = self.chunk_size - self.preprocessing_horizon
        if subsampling is not None and int(subsampling) != int(pp_cfg["subsampling"]):
            raise ValueError(f"subsampling mismatch: caller={subsampling}, config={int(pp_cfg['subsampling'])}")
        if num_tracks is not None and int(num_tracks) != int(pp_cfg["n_tracks"]):
            raise ValueError(f"num_tracks mismatch: caller={num_tracks}, config={int(pp_cfg['n_tracks'])}")
        self.subsampling = int(pp_cfg["subsampling"])
        num_tracks = int(pp_cfg["n_tracks"])
        if self.stride <= 0:
            raise ValueError(f"Invalid preprocessing config: chunk_size={self.chunk_size} <= horizon={self.preprocessing_horizon}")

        self.img_use_preprocessed = bool(img_use_preprocessed)
        self.video_use_preprocessed = bool(video_use_preprocessed)
        self.min_track_motion = None if min_track_motion is None else float(min_track_motion)
        self.max_track_motion = None if max_track_motion is None else float(max_track_motion)
        self.max_cam_rotation = None if max_cam_rotation is None else float(max_cam_rotation)
        self.max_cam_translation = None if max_cam_translation is None else float(max_cam_translation)
        if supervision not in {"point_tracks", "optical_flow", "cumulative_disp"}:
            raise ValueError(
                f"supervision must be 'point_tracks', 'optical_flow', or "
                f"'cumulative_disp', got '{supervision}'"
            )
        if supervision == "optical_flow" and self.true_horizon < 3:
            raise ValueError(f"true_horizon must be >= 3 in optical_flow mode, got {self.true_horizon}")
        if tracks_normalizer is not None and tracks_normalizer != "ramen":
            raise ValueError(
                f"tracks_normalizer must be null or 'ramen'; got {tracks_normalizer!r}"
            )
        if tracks_normalizer == "ramen" and supervision != "cumulative_disp":
            raise ValueError(
                "tracks_normalizer='ramen' requires supervision='cumulative_disp' "
                f"(got supervision='{supervision}')"
            )
        if self.true_horizon > self.chunk_size:
            raise ValueError(
                f"true_horizon={self.true_horizon} exceeds preprocessing chunk_size={self.chunk_size}; "
                "no window can fit in a single chunk."
            )

        super().__init__(
            root_dir=root_dir,
            dataset_names=dataset_names,
            track_method=track_method,
            cond_cameraviews=self.cond_views,
            keys_to_load=list(keys_to_load),
            img_shape=img_shape,
            true_horizon=true_horizon,
            track_pred_horizon=track_pred_horizon,
            interp_method=interp_method,
            num_tracks=num_tracks,
            use_cached_index_map=use_cached_index_map,
            aug_cfg=aug_cfg,
            subsampling=self.subsampling,
            tracker=tracker,
            sample_stride=sample_stride,
            supervision=supervision,
        )

        self.resize = Resize(self.img_shape, antialias=False)

        if "images" in self.keys_to_load:
            video_source = self.index_map[0]["video_path"]
            vg = self._open_view_group(video_source)
            if "video" not in vg:
                raise KeyError(f"Missing 'video' array in {video_source}")
            video = vg["video"]
            if video.ndim != 4:
                raise ValueError(f"Expected video with 4 dims, got {video.ndim}")
            self.data_img_size = (int(video.shape[1]), int(video.shape[2]))
        else:
            self.data_img_size = tuple(self.img_shape)

        default_tracks_stats = self._default_tracks_stats_path()
        self.tracks_normalizer: Optional[TracksNormalizer] = self._setup_tracks_normalizer(
            normalizer_type=tracks_normalizer,
            stats_path=tracks_normalizer_stats_path,
            default_stats_path=default_tracks_stats,
        )

    def _default_tracks_stats_path(self) -> Path:
        """`<egoverse_dataset_root>/normalizer_stats/tracks_cumulative_disp_h<H>.json`.
        Anchored on the first index entry's `track_path`'s grandparent so
        train/val splits of the same egoverse dataset share a cache."""
        track_path = self.index_map[0]["track_path"]
        # track_path = <root>/<dataset_name>/<track_method>/<tracker>_<mode>/<demo>.zarr
        # We want <root>/<dataset_name>/normalizer_stats/.
        ds_root = Path(track_path).parents[2]
        return ds_root / "normalizer_stats" / f"tracks_cumulative_disp_h{self.true_horizon}.json"

    def _dataset_root(self) -> str:
        base = self.dataset_path
        if base is None:
            return os.path.join(self.root_dir, "preprocessed_data")
        base = os.path.expanduser(base)
        if not os.path.isabs(base):
            base = os.path.join(self.root_dir, base)
        return base

    def _load_preprocessing_cfg(self) -> Dict:
        """Read the preprocessing config.yaml for this track_method (authoritative source for chunk_size / horizon / n_tracks / subsampling)."""
        dataset_dir, split = self._parse_dataset_name(self.dataset_names[0])
        config_path = os.path.join(
            self._dataset_root(), dataset_dir, split,
            self.track_method, f"{self.tracker}_{self.tracker_mode}", "config.yaml",
        )
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing preprocessing config at {config_path}")
        return OmegaConf.load(config_path)

    def _chunk_pos(self, t: int, num_chunks: int) -> Tuple[int, int]:
        """Map global frame t to (chunk_idx, pos_in_chunk) under the preprocessing
        stride. chunk_idx is clamped to the last chunk so end-of-clip frames don't
        index past the array — matches frame_to_chunk_position, the shared helper
        used by the molmoact2 / realyam loaders."""
        chunk_idx = min(t // self.stride, num_chunks - 1)
        return chunk_idx, t - chunk_idx * self.stride

    def _read_chunked_window(self, arr, t: int, n: int) -> np.ndarray:
        """Read n frames starting at global t from a (num_chunks, chunk_size, ...) zarr; must fit in one chunk."""
        chunk_idx, pos = self._chunk_pos(t, arr.shape[0])
        if pos + n > self.chunk_size:
            raise ValueError(
                f"Window [t={t}, n={n}] spans chunks (chunk_idx={chunk_idx}, pos={pos}, "
                f"cs={self.chunk_size}); preprocessing must satisfy "
                f"true_horizon <= chunk_size - (chunk_size - preprocessing_horizon)."
            )
        return np.asarray(arr[chunk_idx, pos:pos + n])

    @staticmethod
    def _parse_dataset_name(name: str) -> Tuple[str, str]:
        parts = [p for p in name.split("_") if p]
        split = parts[-1].lower()
        valid = {"train", "val", "validation", "test"}
        if split not in valid:
            raise ValueError(
                f"EgoVerse dataset '{name}' must end with one of {sorted(valid)}."
            )
        dataset_dir = "_".join(parts[:-1])
        return dataset_dir, split

    _INDEX_MAP_CACHE_DIR = os.path.expanduser("~/.cache/flomo/index_maps/egoverse")

    def get_cache_file(self) -> str:
        dataset_str = "_".join(self.dataset_names)
        cache_dir = self._INDEX_MAP_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        suffix = f"{self.tracker}_{self.tracker_mode}"
        if any(v is not None for v in (
            self.min_track_motion, self.max_track_motion,
            self.max_cam_rotation, self.max_cam_translation,
        )):
            suffix += (
                f"_minmot{self.min_track_motion}_maxmot{self.max_track_motion}"
                f"_camrot{self.max_cam_rotation}_camtrans{self.max_cam_translation}"
            )
        return os.path.join(cache_dir, f"{dataset_str}_{self.track_method}_{suffix}_base.json")

    def _windows_per_sample(self) -> int:
        """Consecutive tracks rows required per sample. Bound by the video modality's
        T_pix = true_horizon + 1 pixel-frame window (1 cond + true_horizon prediction
        frames). OF (reads true_horizon rows) and PT (reads true_horizon - 1 rows)
        both fit comfortably under this bound.
        """
        return self.true_horizon + 1

    def create_index_map(self) -> List[Dict]:
        """Build the cache JSON with all per-entry filters inline:
        tail-trim, motion. Resumable via .partial checkpoints."""
        do_motion_filter = any(v is not None for v in (
            self.min_track_motion, self.max_track_motion,
            self.max_cam_rotation, self.max_cam_translation,
        ))
        if do_motion_filter:
            print(
                f"[Motion Filter] min_track_motion={self.min_track_motion}, "
                f"max_track_motion={self.max_track_motion}, "
                f"max_cam_rotation={self.max_cam_rotation}, "
                f"max_cam_translation={self.max_cam_translation}, F={self.true_horizon - 1}"
            )
        wps = self._windows_per_sample()
        filter_stats = {
            'total': 0,
            'short_clip': 0, 'tail_trim': 0,
            'low_track_motion': 0, 'high_track_motion': 0,
            'high_cam_rotation': 0, 'high_cam_translation': 0,
        }
        CHECKPOINT_INTERVAL = 500

        partial_file = self.get_cache_file() + ".partial"
        already_processed: set = set()
        index: List[Dict] = []
        if os.path.exists(partial_file):
            with open(partial_file, "r") as f:
                partial_state = json.load(f)
            already_processed = set(partial_state["processed_paths"])
            index = partial_state["index"]
            filter_stats = partial_state["filter_stats"]
            print(
                f"[Index Map] Resuming from partial checkpoint: "
                f"{len(already_processed)} files already processed, "
                f"{len(index)} entries collected so far."
            )

        base = self._dataset_root()
        needs_images = "images" in self.keys_to_load
        needs_text_emb = "text_emb" in self.keys_to_load
        needs_text = "text" in self.keys_to_load or needs_text_emb
        files_processed_this_run = 0
        skipped_corrupt = 0

        for dataset in self.dataset_names:
            dataset_dir, split = self._parse_dataset_name(dataset)
            split_root = os.path.join(base, dataset_dir, split)
            split_tracks_root = os.path.join(
                split_root, self.track_method, f"{self.tracker}_{self.tracker_mode}"
            )
            if not os.path.isdir(split_tracks_root):
                raise FileNotFoundError(
                    f"EgoVerse split tracks directory missing: {split_tracks_root}. "
                    "Run preprocessing/preprocess_egoverse.py to generate the processed data."
                )

            zarr_paths = sorted(
                glob.glob(os.path.join(split_tracks_root, "*", "*", "*.zarr"))
            )
            if not zarr_paths:
                raise FileNotFoundError(f"No .zarr files found under {split_tracks_root}")

            for track_path in tqdm(zarr_paths, desc="Building index map"):
                if track_path in already_processed:
                    continue

                rel = os.path.relpath(track_path, split_tracks_root)
                parts = rel.split(os.sep)
                if len(parts) < 3:
                    print(f"[EgoVerseDataset] Skipping malformed path {track_path}")
                    continue
                task, prefix = parts[0], parts[1]
                sample_name = parts[-1]

                video_path = os.path.join(split_root, "video", task, prefix, sample_name)
                text_path = os.path.join(split_root, "text", task, prefix, sample_name)
                meta_path = os.path.join(split_root, "meta", task, prefix, sample_name)

                try:
                    vg = self._open_view_group(track_path)
                    if self.tracks_key not in vg:
                        raise KeyError(f"{track_path} missing {self.tracks_key} array")
                    tracks = vg[self.tracks_key]
                    T = int(tracks.shape[0])
                    if T == 0:
                        raise ValueError(f"Empty tracks tensor in {track_path}")
                    # Read tracks/pose once per Zarr store when filtering.
                    tracks_data = np.asarray(tracks[:, :2]).astype(np.float32) if do_motion_filter else None
                    extrs_data = None
                    if do_motion_filter and (
                        self.max_cam_rotation is not None or self.max_cam_translation is not None
                    ):
                        pose_path = self._derive_sidecar_path(track_path, "pose")
                        pose_vg = self._open_view_group(pose_path)
                        extrs_data = np.asarray(pose_vg["camera_extrinsics"][:]).astype(np.float32)
                except Exception as e:
                    print(f"[EgoVerseDataset] Skipping corrupt/incomplete zarr {track_path}: {e}")
                    skipped_corrupt += 1
                    already_processed.add(track_path)
                    files_processed_this_run += 1
                    continue

                if needs_images and not os.path.isdir(video_path):
                    raise FileNotFoundError(f"Missing EgoVerse video zarr: {video_path}")
                if needs_text and not os.path.isdir(text_path):
                    raise FileNotFoundError(f"Missing EgoVerse text zarr: {text_path}")
                if needs_text_emb and not os.path.isdir(
                    self._derive_segment_text_emb_path(text_path)
                ):
                    raise FileNotFoundError(
                        f"Missing EgoVerse segment_text_emb zarr: "
                        f"{self._derive_segment_text_emb_path(text_path)}"
                    )

                # Apply the same short-clip filter to optical flow and point tracks.
                if T < self.true_horizon:
                    filter_stats['short_clip'] += 1
                    already_processed.add(track_path)
                    files_processed_this_run += 1
                    continue

                for track_idx in range(T):
                    start_t = track_idx * self.subsampling
                    end_t = start_t + self.true_horizon * self.subsampling
                    filter_stats['total'] += 1
                    # Tail-trim: slice [track_idx, track_idx + wps) must fit in T_tracks.
                    if track_idx + wps > T:
                        filter_stats['tail_trim'] += 1
                        continue
                    if do_motion_filter:
                        if not self._passes_motion_filter(tracks_data, extrs_data, start_t, filter_stats):
                            continue
                    index.append(
                        {
                            "track_path": track_path,
                            "video_path": video_path,
                            "text_path": text_path,
                            "meta_path": meta_path,
                            "dataset": dataset,
                            "split": split,
                            "start_t": int(start_t),
                            "end_t": int(end_t),
                            "rollout_len": T,
                        }
                    )

                already_processed.add(track_path)
                files_processed_this_run += 1
                if files_processed_this_run % CHECKPOINT_INTERVAL == 0:
                    try:
                        with open(partial_file, "w") as f:
                            json.dump(
                                {"processed_paths": list(already_processed), "index": index, "filter_stats": filter_stats},
                                f,
                            )
                    except Exception as e:
                        print(f"[Index Map] Warning: could not write partial checkpoint: {e}")

        if skipped_corrupt:
            print(f"[EgoVerseDataset] Skipped {skipped_corrupt} corrupt/incomplete zarr(s); indexed {len(index)} entries.")
        if filter_stats['total'] > 0:
            kept = len(index)
            total = filter_stats['total']
            print(f"[Index Filters] Kept {kept}/{total} samples ({100*kept/total:.1f}%)")
            print(f"  - Short clip (T < true_horizon):    {filter_stats['short_clip']}")
            print(f"  - Tail-trim (track_idx + wps > T):  {filter_stats['tail_trim']}")
            print(f"  - Low track motion:                 {filter_stats['low_track_motion']}")
            print(f"  - High track motion:                {filter_stats['high_track_motion']}")
            print(f"  - High cam rotation:                {filter_stats['high_cam_rotation']}")
            print(f"  - High cam translation:             {filter_stats['high_cam_translation']}")
        return self._intern_index_map_strings(index)

    def _passes_motion_filter(
        self,
        tracks_data: np.ndarray,
        extrs_data: Optional[np.ndarray],
        start_t: int,
        filter_stats: Dict[str, int],
    ) -> bool:
        """Score one F-window against track-motion (Eulerian sum-of-flows, mean over N) and
        cam-motion (cumulative rot/trans) thresholds. F=true_horizon-1. Chunk-safe + tail-trim
        are gated upstream so window/chunk overflow is impossible here."""
        F = self.true_horizon - 1

        if self.min_track_motion is not None or self.max_track_motion is not None:
            raw = tracks_data[start_t:start_t + F]                            # (F, 2, N, 3)
            flows = raw[:, 1] - raw[:, 0]                                     # (F, N, 3)
            track_motion = eulerian_track_motion(flows)
            if self.min_track_motion is not None and track_motion < self.min_track_motion:
                filter_stats['low_track_motion'] += 1
                return False
            if self.max_track_motion is not None and track_motion > self.max_track_motion:
                filter_stats['high_track_motion'] += 1
                return False

        if extrs_data is not None:
            chunk_idx, pos = self._chunk_pos(start_t, extrs_data.shape[0])
            extrs = extrs_data[chunk_idx, pos:pos + F]                        # (F, 4, 4)
            cam_rot, cam_trans = compute_cumulative_camera_motion(extrs)
            if self.max_cam_rotation is not None and cam_rot > self.max_cam_rotation:
                filter_stats['high_cam_rotation'] += 1
                return False
            if self.max_cam_translation is not None and cam_trans > self.max_cam_translation:
                filter_stats['high_cam_translation'] += 1
                return False

        return True

    @staticmethod
    def _open_group(path: str):
        return zarr.open_group(path, mode="r")

    def _open_view_group(self, path: str):
        """Open <path>/root/<view>. Text zarrs use a different layout; use _open_group instead."""
        g = self._open_group(path)
        if "root" not in g or self.view not in g["root"]:
            raise KeyError(f"{path} missing root/{self.view}")
        return g["root"][self.view]

    def load_images(self, idx_dict: Dict) -> Dict:
        start_t = int(idx_dict["start_t"])
        vg = self._open_view_group(idx_dict["video_path"])
        frames = np.asarray(vg["video"][start_t]).astype(np.float32)[None, None]  # (V=1, F=1, H, W, C)
        return {"images": frames}

    def load_actions(self, idx_dict: Dict) -> Dict:
        raise NotImplementedError("EgoVerse does not provide actions.")

    def _track_idx(self, idx_dict: Dict) -> int:
        # EgoVerse's index map `start_t` is in frame space; tracks are subsampled.
        return int(idx_dict["start_t"]) // self.subsampling

    @contextmanager
    def _tracks_group(self, idx_dict: Dict):
        yield self._open_view_group(idx_dict["track_path"])

    def _read_flow_extrs(self, idx_dict: Dict, n: int) -> np.ndarray:
        pose_vg = self._open_view_group(
            self._derive_sidecar_path(idx_dict["track_path"], "pose")
        )
        return self._read_chunked_window(
            pose_vg["camera_extrinsics"], int(idx_dict["start_t"]), n
        ).astype(np.float32)

    @contextmanager
    def _vae_latent_array(self, idx_dict: Dict, view_key: str):
        # EgoVerse is single-camera; the view group already navigates to the
        # primary view, so `view_key` is unused.
        yield self._open_view_group(
            self._derive_vae_latent_h16_path(idx_dict["video_path"])
        )["vae_latent_h16"]

    def _rgb_video_path(self, idx_dict: Dict) -> str:
        return idx_dict["video_path"]

    def _read_video_window(self, video_path: str, view_key: str, start_t: int) -> np.ndarray:
        """Raw RGB window (1 + 4*t_frames, H, W, 3) for live VAE encoding / flow viz.

        The index map guarantees a full 1+true_horizon window ahead; a short
        slice means the index map is wrong, so fail hard rather than pad.
        """
        T = self.true_horizon + 1
        vg = self._open_group(video_path)["root"][view_key]
        frames = np.asarray(vg["video"][start_t: start_t + T]).astype(np.float32)
        if frames.shape[0] != T:
            raise IndexError(
                f"video window [start_t={start_t}, T={T}] runs past the clip "
                f"({frames.shape[0]} frames) in {video_path}; the index map "
                f"must guarantee a full {T}-frame window."
            )
        return frames

    def _derive_sidecar_path(self, track_path: str, suffix: str) -> str:
        """Rewrite .../<tracker>_<tracker_mode>/... → .../<tracker>_<suffix>/... so depth/pose paths share the index_map's track_path."""
        stamp = f"/{self.tracker}_{self.tracker_mode}/"
        target = f"/{self.tracker}_{suffix}/"
        if stamp not in track_path:
            raise ValueError(f"track_path does not contain '{stamp}': {track_path}")
        return track_path.replace(stamp, target, 1)

    def _derive_vae_latent_h16_path(self, video_path: str) -> str:
        """`<split_root>/video/<...>/<clip>.zarr` → `<split_root>/vae_latent_h16/<...>/<clip>.zarr`."""
        stamp = "/video/"
        target = "/vae_latent_h16/"
        if stamp not in video_path:
            raise ValueError(f"video_path does not contain '{stamp}': {video_path}")
        return video_path.replace(stamp, target, 1)

    def _derive_segment_text_emb_path(self, text_path: str) -> str:
        """`<split_root>/text/<...>/<clip>.zarr` → `<split_root>/segment_text_emb/<...>/<clip>.zarr`."""
        stamp = "/text/"
        target = "/segment_text_emb/"
        if stamp not in text_path:
            raise ValueError(f"text_path does not contain '{stamp}': {text_path}")
        return text_path.replace(stamp, target, 1)

    def load_depth(self, idx_dict: Dict) -> Dict:
        depth_source = self._derive_sidecar_path(idx_dict["track_path"], "depth")
        start_t = int(idx_dict["start_t"])
        vg = self._open_view_group(depth_source)
        depth_arr = vg["depth_map"]
        chunk_idx, pos = self._chunk_pos(start_t, depth_arr.shape[0])
        return {"depth": np.asarray(depth_arr[chunk_idx, pos])[None].astype(np.float32)}

    def load_pose(self, idx_dict: Dict) -> Dict:
        pose_source = self._derive_sidecar_path(idx_dict["track_path"], "pose")
        start_t = int(idx_dict["start_t"])
        vg = self._open_view_group(pose_source)
        extrs_arr = vg["camera_extrinsics"]
        intrs_arr = vg["camera_intrinsics"]
        chunk_idx, pos = self._chunk_pos(start_t, extrs_arr.shape[0])
        return {
            "extrs": np.asarray(extrs_arr[chunk_idx, pos])[None].astype(np.float32),
            "intrs": np.asarray(intrs_arr[chunk_idx, pos])[None].astype(np.float32),
        }

    def load_text(
        self,
        idx_dict: Dict,
        include_text: bool = True,
        include_text_emb: bool = False,
    ) -> Dict:
        if not include_text and not include_text_emb:
            return {}
        out: Dict = {}
        text_source = idx_dict["text_path"]
        g = self._open_group(text_source)
        # Segment whose [start, end) contains the window's first frame
        # (start_t / fps). `text` and `text_emb` share this segment index.
        fps    = int(g["fps"][()])
        t_sec  = idx_dict["start_t"] / fps
        starts = g["segments_start_sec"][:]
        seg_idx = max(0, int(np.searchsorted(starts, t_sec, side="right")) - 1)
        if include_text:
            if "segments_label" not in g:
                raise KeyError(f"Missing 'segments_label' array in {text_source}")
            label = g["segments_label"][:][seg_idx]
            out["text"] = label.decode("utf-8") if isinstance(label, bytes) else str(label)
        if include_text_emb:
            # Per-window per-segment UMT5 embedding from segment_text_emb zarr.
            # Picks the segment whose [start, end) contains start_t/fps — same
            # selection rule as the live `include_text` path above so the
            # embedding matches the text label that would be produced live.
            emb_path = self._derive_segment_text_emb_path(text_source)
            eg = self._open_group(emb_path)
            if "segments_text_emb" not in eg:
                raise KeyError(f"Missing 'segments_text_emb' array in {emb_path}")
            out["text_emb"] = np.asarray(eg["segments_text_emb"][seg_idx]).astype(np.float32)  # (text_len, text_dim)
        return out

    def process_data(self, data: Dict) -> Dict:
        if "images" in data:
            imgs = data["images"] / 255.0
            imgs = imgs.astype(np.float32)
            if self.data_img_size != self.img_shape:
                v, f = imgs.shape[:2]
                img_t = torch.from_numpy(rearrange(imgs, "v f h w c -> (v f) c h w")).float()
                resized = self.resize(img_t).numpy().astype(np.float32)
                imgs = rearrange(resized, "(v f) c h w -> v f h w c", v=v, f=f)
            data["images"] = imgs

        if "tracks" in data:
            tr = data["tracks"]
            tr = np.nan_to_num(tr, nan=0.0, posinf=0.0, neginf=0.0)
            if self.track_pred_horizon != self.true_horizon:
                if self.interp_method == "linear":
                    tr = interpolate_traj(tr, self.track_pred_horizon)
                elif self.interp_method == "spline":
                    tr = interpolate_traj_spline(tr, self.track_pred_horizon)
                elif self.interp_method != "none":
                    raise NotImplementedError(f"Unsupported interp method {self.interp_method}")
            if self.tracks_normalizer is not None:
                tr = self.tracks_normalizer.normalize(tr)
            data["traj"] = tr.astype(np.float32)
            data.pop("tracks", None)

        if "raw_windows" in data:
            data["raw_windows"] = np.nan_to_num(
                data["raw_windows"], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)

        # `motion_tokens` (cotrain alias for raw tracks) bypasses the tracks->traj
        # path in load_data, so apply the Ramen normalizer here too.
        if "motion_tokens" in data and self.tracks_normalizer is not None:
            data["motion_tokens"] = self.tracks_normalizer.normalize(
                data["motion_tokens"]
            ).astype(np.float32)

        return data
