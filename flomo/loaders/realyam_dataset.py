import glob
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import Resize

from flomo.loaders.base_dataset import BaseDataset
from flomo.utils.action_normalizer import ActionNormalizer, compute_normalizer_stats
from flomo.utils.action_preprocess import to_relative_rot6d_eepose20
from flomo.utils.data_utils import (
    frame_to_chunk_position,
    interpolate_traj,
    interpolate_traj_spline,
    normalize_traj,
)
from flomo.utils.tracks_normalizer import (
    TracksNormalizer,
)


# Absolute EE pose: two arms × [position(3), continuous rot6d(6), gripper(1)].
ACTION_SPACE = "eepose_6drot"

# Single-view (front camera). Per-modality HDF5 lookups go through this key.
DEFAULT_VIEW = "front_img_1"
CAMERA_KEY_MAP = {DEFAULT_VIEW: "observations/images/front_img_1"}

RAW_SUBDIR = "raw"


class RealYamDataset(BaseDataset):
    """Bimanual RealYam loader. The dataset name encodes task and
    split (e.g. ``real_yam_croissant_oven_randomized_train``); on-disk splits
    live at ``<dataset_path>/splits/<task>.json``.

    Single tree under ``dataset_path``:
        raw/<task>/<demo_id>.hdf5                            (symlink to raw demos)
        <track_method>/spatialtracker_<2d|3d|depth|pose>/...
        text_emb/<task>/<demo_id>.hdf5
        vae_latent_h16/<task>/<demo_id>.hdf5                 (17-frame sliding-window Wan VAE latents)
        splits/<task>.json
        normalizer_stats/h<horizon>.json                     (used iff normalize_actions)
    """

    # Canonical views map to front and wrist-camera HDF5 keys.
    CANONICAL_VIEWS = {
        "primary": "front_img_1",
        "left": "left_wrist_img",
        "right": "right_wrist_img",
    }

    def __init__(
        self,
        root_dir: str,
        dataset_names: List[str],
        track_method: str = "uniform_4096_reinit_16",
        cond_views: List[str] = ("primary",),
        pred_views: List[str] = (),
        inputs: List[str] = ("img",),
        outputs: List[str] = (),
        keys_to_load: List[str] = ("images", "tracks", "vis", "text"),
        img_shape: Tuple[int, int] = (128, 128),
        true_horizon: int = 16,
        track_pred_horizon: int = 16,
        interp_method: str = "none",
        num_tracks: int = 4096,
        use_cached_index_map: bool = False,
        aug_cfg: Optional[Dict] = None,
        dataset_path: Optional[str] = None,
        tracker: str = "spatialtracker",
        tracker_mode: str = "3d",
        subsampling: int = 1,
        sample_stride: int = 1,
        normalize_actions: bool = True,
        relative_rotation_actions: bool = True,
        normalizer_stats_path: Optional[str] = None,
        splits_dir: Optional[str] = None,
        supervision: str = "point_tracks",
        img_use_preprocessed: bool = True,
        video_use_preprocessed: bool = True,
        tracks_normalizer: Optional[str] = None,
        tracks_normalizer_stats_path: Optional[str] = None,
        skip_first_frame: bool = False,
        filter_no_motion: bool = False,
    ) -> None:
        # Camera-view state (see load_vae_latent). CANONICAL_VIEWS maps
        # canonical names onto internal HDF5 keys at read time. No view is
        # required: a run may condition on no image at all. `self.view` is the
        # primary internal view used for the single-view modalities (tracks /
        # actions / pose), independent of these lists.
        self._setup_views(cond_views, pred_views, inputs, outputs)
        if supervision not in {"point_tracks", "optical_flow", "cumulative_disp"}:
            raise ValueError(
                f"supervision must be 'point_tracks', 'optical_flow', or "
                f"'cumulative_disp', got '{supervision}'"
            )
        if supervision == "optical_flow" and int(true_horizon) < 3:
            raise ValueError(f"true_horizon must be >= 3 in optical_flow mode, got {true_horizon}")
        if tracks_normalizer is not None and tracks_normalizer != "ramen":
            raise ValueError(
                f"tracks_normalizer must be null or 'ramen'; got {tracks_normalizer!r}"
            )
        if tracks_normalizer == "ramen" and supervision != "cumulative_disp":
            raise ValueError(
                "tracks_normalizer='ramen' requires supervision='cumulative_disp' "
                f"(got supervision='{supervision}')"
            )

        self.dataset_path = dataset_path
        # Optional override for splits_dir (default: <dataset_path>/splits/).
        self.splits_dir = os.path.expanduser(splits_dir) if splits_dir is not None else None
        if tracker_mode not in {"2d", "3d"}:
            raise ValueError(f"tracker_mode must be '2d' or '3d', got {tracker_mode!r}")
        self.tracker = tracker
        self.tracker_mode = tracker_mode
        self.tracks_key = "3d_tracks" if tracker_mode == "3d" else "tracks"
        self.action_space = ACTION_SPACE
        self.relative_rotation_actions = bool(relative_rotation_actions)
        if self.relative_rotation_actions and self.action_space != "eepose_6drot":
            raise ValueError(
                f"relative_rotation_actions=True requires action_space='eepose_6drot', got '{self.action_space}'"
            )
        self.normalize_actions = bool(normalize_actions)
        self.skip_first_frame = bool(skip_first_frame)
        self.filter_no_motion = bool(filter_no_motion)
        self.view = self.CANONICAL_VIEWS["primary"]
        self.img_use_preprocessed = bool(img_use_preprocessed)
        self.video_use_preprocessed = bool(video_use_preprocessed)
        # Resolve the stored horizon before indexing to trim latent tails without
        # opening every latent file; depth/pose readers use the same horizon.
        self.root_dir = root_dir
        self.track_method = track_method
        self.preprocessing_horizon = self._resolve_preprocessing_horizon()

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
            subsampling=subsampling,
            tracker=tracker,
            sample_stride=sample_stride,
            supervision=supervision,
        )

        self.resize = Resize(self.img_shape, antialias=False)

        if "images" in self.keys_to_load and len(self.index_map) > 0:
            with self._open(self.index_map[0]["demo_path"]) as f:
                arr = f[CAMERA_KEY_MAP[self.view]]
                self.data_img_size = (int(arr.shape[1]), int(arr.shape[2]))
        else:
            self.data_img_size = self.img_shape

        # Optical flow needs true_horizon-1 consecutive pose frames in one chunk;
        # frame_to_chunk_position guarantees preprocessing_horizon frames.
        if self.supervision == "optical_flow" and self.true_horizon - 1 > self.preprocessing_horizon:
            raise ValueError(
                f"optical_flow needs {self.true_horizon - 1} consecutive pose frames in a "
                f"single chunk, but preprocessing_horizon={self.preprocessing_horizon}. "
                "Rerun preprocessing with a larger horizon."
            )

        # Optional action normalizer. Explicit path wins; otherwise compute stats
        # on the fly from this dataset's demos at construction time.
        self.normalizer: Optional[ActionNormalizer] = None
        if self.normalize_actions and "actions" in self.keys_to_load:
            if normalizer_stats_path is not None:
                stats_path = os.path.expanduser(normalizer_stats_path)
                self.normalizer = ActionNormalizer.from_stats_file(stats_path)
                if self.normalizer.action_space != self.action_space:
                    raise ValueError(
                        f"Normalizer action_space ({self.normalizer.action_space}) != dataset ({self.action_space})"
                    )
                if self.normalizer.horizon != self.true_horizon:
                    raise ValueError(
                        f"Normalizer horizon ({self.normalizer.horizon}) != dataset ({self.true_horizon})"
                    )
                # Subset check: every task this dataset loads must have been included
                # when stats were computed. A single stats file covering all train+val
                # tasks is the common case (each train/val RealYam instance is a subset).
                stats_tasks = set(self.normalizer.metadata.get("tasks", []))
                ds_tasks = {e["task"] for e in self.index_map}
                missing = ds_tasks - stats_tasks
                if stats_tasks and missing:
                    raise ValueError(
                        f"Dataset tasks {sorted(missing)} are not covered by normalizer stats at "
                        f"{stats_path} (stats covers {sorted(stats_tasks)}). "
                        f"Omit normalizer_stats_path to recompute stats from the index."
                    )
            else:
                self.normalizer = self._compute_normalizer_from_index()

        # Optional tracks normalizer (cumulative_disp + Ramen). Cache file co-located
        # with action stats under <dataset_path>/normalizer_stats/.
        default_tracks_stats = (
            Path(os.path.expanduser(self.dataset_path))
            / "normalizer_stats"
            / f"tracks_cumulative_disp_h{self.true_horizon}.json"
        )
        self.tracks_normalizer: Optional[TracksNormalizer] = self._setup_tracks_normalizer(
            normalizer_type=tracks_normalizer,
            stats_path=tracks_normalizer_stats_path,
            default_stats_path=default_tracks_stats,
        )

    def set_normalizer(self, normalizer: ActionNormalizer) -> None:
        """Replace this dataset's normalizer (e.g. to share train's stats with val)."""
        if normalizer.action_space != self.action_space:
            raise ValueError(
                f"set_normalizer: action_space ({normalizer.action_space}) != dataset ({self.action_space})"
            )
        if normalizer.horizon != self.true_horizon:
            raise ValueError(
                f"set_normalizer: horizon ({normalizer.horizon}) != dataset ({self.true_horizon})"
            )
        self.normalizer = normalizer

    _INDEX_MAP_CACHE_DIR = os.path.expanduser("~/.cache/flomo/index_maps/realyam")

    @staticmethod
    def _parse_dataset_name(name: str) -> Tuple[str, str]:
        """``real_yam_<task>_<split>`` -> (task, split)."""
        if not name.startswith("real_yam_"):
            raise ValueError(f"RealYam dataset name must start with 'real_yam_', got '{name}'")
        parts = name.split("_")
        split = parts[-1].lower()
        if split == "validation":
            split = "val"
        if split not in {"train", "val"}:
            raise ValueError(f"RealYam dataset '{name}' must end with '_train' or '_val'.")
        # 'real', 'yam', <task parts...>, <split>
        task = "_".join(parts[2:-1])
        if not task:
            raise ValueError(f"RealYam dataset '{name}' missing task between 'real_yam_' and '_{split}'.")
        return task, split

    def _cotrain_subname(self, idx_dict) -> str:
        return idx_dict["task"]

    def _dataset_root(self) -> str:
        if self.dataset_path is None:
            return os.path.join(self.root_dir, "preprocessed_data", "real_yam")
        base = os.path.expanduser(self.dataset_path)
        return base if os.path.isabs(base) else os.path.join(self.root_dir, base)

    def get_cache_file(self) -> str:
        os.makedirs(self._INDEX_MAP_CACHE_DIR, exist_ok=True)
        ds = "_".join(self.dataset_names)
        suffix = f"{self.track_method}_{self.tracker}_{self.tracker_mode}"
        if self.skip_first_frame:
            suffix += "_sff"
        if self.filter_no_motion:
            suffix += "_fnm"
        suffix += ".json"
        # dataset_path lives in the cache key: same (task_list, dataset_path)
        # pair shares; switching real_yam <-> real_yam_squash invalidates.
        import hashlib
        path_tag = hashlib.sha1((self.dataset_path or "").encode()).hexdigest()[:8]
        # Filename must fit in 255 bytes; hash long dataset name lists.
        if len(ds) > 200:
            ds = f"realyam_multi_{hashlib.sha1(ds.encode()).hexdigest()[:12]}"
        return os.path.join(self._INDEX_MAP_CACHE_DIR, f"{ds}_p{path_tag}_{suffix}")

    def create_index_map(self) -> List[Dict]:
        base = self._dataset_root()
        if not os.path.isdir(base):
            raise FileNotFoundError(f"RealYam dataset_path does not exist: {base}")

        # Motion filter only runs when actions are loaded (needs EE positions).
        apply_motion_filter = self.filter_no_motion and "actions" in self.keys_to_load
        _MOTION_THRESH = 1e-6
        n_skipped_first_frame = 0
        n_skipped_no_motion = 0

        index: List[Dict] = []
        for dataset in self.dataset_names:
            task, split = self._parse_dataset_name(dataset)

            if self.splits_dir is not None:
                # Use explicit splits file (intra-task train/val partition).
                splits_path = os.path.join(self.splits_dir, f"{task}.json")
                if not os.path.isfile(splits_path):
                    raise FileNotFoundError(
                        f"splits_dir set but missing splits file for task '{task}': {splits_path}"
                    )
                with open(splits_path) as f:
                    splits = json.load(f)
                if split not in splits:
                    raise KeyError(f"Splits file {splits_path} has no '{split}' key.")
                demo_ids = splits[split]
                if not demo_ids:
                    raise ValueError(f"Empty '{split}' split for task '{task}' in {splits_path}")
            else:
                # No splits provided: take all demos that have full preprocessing
                task_tracks_dir = os.path.join(
                    base, self.track_method, f"{self.tracker}_{self.tracker_mode}", task,
                )
                if not os.path.isdir(task_tracks_dir):
                    raise FileNotFoundError(f"Tracks task dir not found: {task_tracks_dir}")
                demo_files = sorted(glob.glob(os.path.join(task_tracks_dir, "demo_*.hdf5")))
                if not demo_files:
                    raise FileNotFoundError(f"No demo_*.hdf5 in {task_tracks_dir}")
                demo_ids = [os.path.basename(f)[: -len(".hdf5")] for f in demo_files]

            for demo_id in demo_ids:
                paths = self._modal_paths(base, task, demo_id)
                # Corrupted/missing demos: skip at index-build time so a single
                # bad HDF5 doesn't blow up dataset init. Logs the reason so
                # corruption is visible. Pairs with __getitem__'s runtime catch
                # for chunk-level errors that pass the open-and-shape probe.
                # Broad catch covers the 14 known-bad realyam demos whose
                # failure modes vary (truncated headers, missing root keys,
                # partial chunks → OSError / KeyError / ValueError / EOFError).
                try:
                    self._validate_paths(paths)
                    with self._open(paths["track_path"]) as f:
                        T = int(f["root"][self.view][self.tracks_key].shape[0])
                    if T == 0:
                        raise ValueError(f"Empty tracks tensor in {paths['track_path']}")
                    has_motion = None
                    if apply_motion_filter:
                        with self._open(paths["demo_path"]) as f:
                            acts = f[f"actions/{self.action_space}"][:].astype(np.float32)
                        pos_l = acts[:, 0:3]                                    # (rollout_len, 3)
                        pos_r = acts[:, 10:13]                                  # (rollout_len, 3)
                        delta_l = np.linalg.norm(np.diff(pos_l, axis=0), axis=1)  # (rollout_len-1,)
                        delta_r = np.linalg.norm(np.diff(pos_r, axis=0), axis=1)  # (rollout_len-1,)
                        has_motion = (delta_l > _MOTION_THRESH) | (delta_r > _MOTION_THRESH)
                except (OSError, RuntimeError, KeyError, IndexError, ValueError, EOFError) as e:
                    print(f"[RealYam] index-build skipping {task}/{demo_id}: {type(e).__name__}: {e}")
                    continue

                for i in range(T):
                    start_t = i * self.subsampling
                    end_t = start_t + self.true_horizon * self.subsampling
                    # Under optical_flow, drop the trailing true_horizon-1 starts per clip
                    # because the loader reads true_horizon consecutive reinit rows.
                    if not self._start_valid(start_t, T):
                        continue
                    if start_t >= T - self.preprocessing_horizon:
                        continue
                    # Drop the t=0 chunk: covers the homing artifact (large jump from
                    # robot home pose into demo start, frame 0 -> 1).
                    if self.skip_first_frame and start_t == 0:
                        n_skipped_first_frame += 1
                        continue
                    if has_motion is not None:
                        # has_motion has length rollout_len-1; chunk_end is clamped
                        # to rollout_len so the slice stays in-bounds.
                        chunk_end = min(end_t, len(has_motion) + 1)
                        motion_slice = has_motion[start_t:chunk_end - 1]
                        if len(motion_slice) > 0 and not np.any(motion_slice):
                            n_skipped_no_motion += 1
                            continue

                    index.append({
                        "task": task,
                        "demo_id": demo_id,
                        "split": split,
                        "dataset": dataset,
                        "start_t": int(start_t),
                        "end_t": int(end_t),
                        "rollout_len": T,
                        **{k: os.path.realpath(v) for k, v in paths.items()},
                    })

        if n_skipped_first_frame > 0:
            print(f"[RealYam] Filtered {n_skipped_first_frame} chunks with start_t=0 (skip_first_frame)")
        if n_skipped_no_motion > 0:
            print(f"[RealYam] Filtered {n_skipped_no_motion} chunks with no EE motion (both arms stationary)")

        return self._intern_index_map_strings(index)

    def _modal_paths(self, base: str, task: str, demo_id: str) -> Dict[str, str]:
        track_root = os.path.join(base, self.track_method)
        return {
            "demo_path":       os.path.join(base, RAW_SUBDIR, task, f"{demo_id}.hdf5"),
            "track_path":      os.path.join(track_root, f"{self.tracker}_{self.tracker_mode}", task, f"{demo_id}.hdf5"),
            "tracks2d_path":   os.path.join(track_root, f"{self.tracker}_2d", task, f"{demo_id}.hdf5"),
            "depth_path":      os.path.join(track_root, f"{self.tracker}_depth", task, f"{demo_id}.hdf5"),
            "pose_path":       os.path.join(track_root, f"{self.tracker}_pose", task, f"{demo_id}.hdf5"),
            "text_emb_path":       os.path.join(base, "text_emb", task, f"{demo_id}.hdf5"),
            "vae_latent_h16_path": os.path.join(base, "vae_latent_h16", task, f"{demo_id}.hdf5"),
        }

    def _validate_paths(self, paths: Dict[str, str]) -> None:
        if not os.path.isfile(paths["track_path"]):
            raise FileNotFoundError(f"Missing track_path: {paths['track_path']}")
        checks = [
            ("images",         "demo_path"),
            ("actions",        "demo_path"),
            ("text",           "demo_path"),
            ("depth",          "depth_path"),
            ("pose",           "pose_path"),
            ("text_emb",       "text_emb_path"),
        ]
        for key, path_key in checks:
            if key in self.keys_to_load and not os.path.isfile(paths[path_key]):
                raise FileNotFoundError(f"Missing {path_key} for keys_to_load={key}: {paths[path_key]}")
        # vae_latent_h16 is only required when at least one camera modality reads
        # preprocessed latents; with both flags False the loader live-encodes raw RGB.
        if (
            "vae_latent" in self.keys_to_load
            and (self.img_use_preprocessed or self.video_use_preprocessed)
            and not os.path.isfile(paths["vae_latent_h16_path"])
        ):
            raise FileNotFoundError(
                f"Missing vae_latent_h16_path for keys_to_load=vae_latent: {paths['vae_latent_h16_path']}"
            )
        # optical_flow reads pose internally regardless of keys_to_load.
        if self.supervision == "optical_flow" and not os.path.isfile(paths["pose_path"]):
            raise FileNotFoundError(
                f"optical_flow needs pose_path: {paths['pose_path']}"
            )

    def _compute_normalizer_from_index(self) -> ActionNormalizer:
        """Compute action normalizer stats over unique (task, demo_id) demos."""
        seen: set[Tuple[str, str]] = set()
        seqs: List[np.ndarray] = []
        for entry in self.index_map:
            key = (entry["task"], entry["demo_id"])
            if key in seen:
                continue
            seen.add(key)
            with self._open(entry["demo_path"]) as f:
                seqs.append(f[f"actions/{self.action_space}"][:].astype(np.float32))
        if not seqs:
            raise RuntimeError("Cannot compute normalizer stats: index map is empty.")
        normalizer = compute_normalizer_stats(
            action_sequences=seqs,
            reference_sequences=seqs,  # action[t] is the pose at frame t
            action_space=self.action_space,
            horizon=self.true_horizon,
        )
        normalizer.metadata = {"tasks": sorted({t for t, _ in seen}), "num_demos": len(seqs)}
        return normalizer

    @staticmethod
    def _open(path: str):
        return h5py.File(path, "r")

    def _resolve_preprocessing_horizon(self) -> int:
        """Read on-disk preprocessing horizon from <tracker_dir>/config.yaml.

        The chunk stride used to write depth/pose is `chunk_size - horizon` (with
        multi_view=False). The loader must use this same horizon when decoding,
        not `self.true_horizon`. Cross-checks against the `_reinit_<N>` token in
        track_method.
        """
        # config.yaml lives at <base>/<track_method>/<tracker_subdir>/config.yaml.
        tracker_dir = os.path.join(
            self._dataset_root(), self.track_method, f"{self.tracker}_{self.tracker_mode}",
        )
        cfg_path = os.path.join(tracker_dir, "config.yaml")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f"Missing preprocessing config.yaml at {cfg_path}; cannot resolve "
                "on-disk chunk stride for depth/pose."
            )
        cfg = OmegaConf.load(cfg_path)
        if "horizon" not in cfg:
            raise KeyError(f"{cfg_path} has no 'horizon' field.")
        h = int(cfg["horizon"])
        if bool(cfg["multi_view"]):
            raise NotImplementedError(
                f"RealYam loader assumes multi_view=False, got multi_view=True in {cfg_path}."
            )
        if "reinit_" in self.track_method:
            token = int(self.track_method.split("reinit_")[-1].split("_")[0])
            if token != h:
                raise ValueError(
                    f"track_method={self.track_method!r} implies reinit horizon {token}, "
                    f"but {cfg_path} says horizon={h}."
                )
        return h

    def load_images(self, idx_dict: Dict) -> Dict:
        start_t = idx_dict["start_t"]
        with self._open(idx_dict["demo_path"]) as f:
            frame = f[CAMERA_KEY_MAP[self.view]][start_t].astype(np.float32)  # (H, W, C)
        return {"images": frame[None, None]}                                  # (V=1, F=1, H, W, C)

    def load_actions(self, idx_dict: Dict) -> Dict:
        s, e = idx_dict["start_t"], idx_dict["end_t"]
        with self._open(idx_dict["demo_path"]) as f:
            acts = f[f"actions/{self.action_space}"][s:e:self.subsampling].astype(np.float32)
        # actions[0] is the pose at the observation frame.
        return {"actions": acts, "action_reference": acts[0].copy()}

    def _read_extrs_window(self, pose_path: str, start_t: int, n: int) -> np.ndarray:
        """Read n consecutive camera extrinsics at global frame start_t (must fit in one chunk)."""
        with self._open(pose_path) as f:
            extrs_ds = f["root"][self.view]["camera_extrinsics"]
            num_chunks, chunk_size = extrs_ds.shape[:2]
            # chunk math uses the on-disk preprocessing stride; we then slice n frames manually.
            chunk_idx, start_pos, _ = frame_to_chunk_position(
                start_t, num_chunks, chunk_size, self.preprocessing_horizon
            )
            if start_pos + n > chunk_size:
                raise ValueError(
                    f"Pose window [start_t={start_t}, n={n}] spans chunks "
                    f"(chunk_idx={chunk_idx}, start_pos={start_pos}, chunk_size={chunk_size})."
                )
            return extrs_ds[chunk_idx, start_pos:start_pos + n].astype(np.float32)

    def _track_idx(self, idx_dict: Dict) -> int:
        # RealYam's index map `start_t` is in frame space; tracks are subsampled.
        return int(idx_dict["start_t"]) // self.subsampling

    @contextmanager
    def _tracks_group(self, idx_dict: Dict):
        with self._open(idx_dict["track_path"]) as f:
            yield f["root"][self.view]

    def _read_flow_extrs(self, idx_dict: Dict, n: int) -> np.ndarray:
        return self._read_extrs_window(idx_dict["pose_path"], int(idx_dict["start_t"]), n)

    @contextmanager
    def _vae_latent_array(self, idx_dict: Dict, view_key: str):
        with self._open(idx_dict["vae_latent_h16_path"]) as f:
            yield f["root"][view_key]["vae_latent_h16"]

    def _rgb_video_path(self, idx_dict: Dict) -> str:
        return idx_dict["demo_path"]

    def load_text(
        self,
        idx_dict: Dict,
        include_text: bool = True,
        include_text_emb: bool = False,
    ) -> Dict:
        out: Dict = {}
        if include_text:
            with self._open(idx_dict["demo_path"]) as f:
                t = f["text"][()]
                out["text"] = t.decode("utf-8") if isinstance(t, bytes) else str(t)
        if include_text_emb:
            with self._open(idx_dict["text_emb_path"]) as f:
                out["text_emb"] = f["text_emb"][()].astype(np.float32)
        return out

    def load_depth(self, idx_dict: Dict) -> Dict:
        with self._open(idx_dict["depth_path"]) as f:
            ds = f["root"][self.view]["depth_map"]
            num_chunks, chunk_size = ds.shape[:2]
            chunk_idx, start_pos, _ = frame_to_chunk_position(
                idx_dict["start_t"], num_chunks, chunk_size, self.preprocessing_horizon
            )
            depth = ds[chunk_idx, start_pos][None].astype(np.float32)
        return {"depth": depth}

    def load_pose(self, idx_dict: Dict) -> Dict:
        with self._open(idx_dict["pose_path"]) as f:
            vg = f["root"][self.view]
            extrs_ds = vg["camera_extrinsics"]
            intrs_ds = vg["camera_intrinsics"]
            num_chunks, chunk_size = extrs_ds.shape[:2]
            chunk_idx, start_pos, _ = frame_to_chunk_position(
                idx_dict["start_t"], num_chunks, chunk_size, self.preprocessing_horizon
            )
            return {
                "extrs": extrs_ds[chunk_idx, start_pos][None].astype(np.float32),
                "intrs": intrs_ds[chunk_idx, start_pos][None].astype(np.float32),
            }

    def _read_video_window(self, demo_path: str, view_key: str, start_t: int) -> np.ndarray:
        """RGB window (1+4*t_frames, img_size, img_size, 3) for VAE/viz. Mirrors
        preprocess_realyam._resize_video (LANCZOS squash to img_size x img_size)
        so viz_video lands on the same pixel grid as the preprocessed VAE latents."""
        T = self.true_horizon + 1
        with self._open(demo_path) as f:
            frames = np.asarray(
                f[f"observations/images/{view_key}"][start_t: start_t + T]
            ).astype(np.uint8)
        if frames.shape[0] != T:
            raise IndexError(
                f"video window [start_t={start_t}, T={T}] runs past the clip "
                f"({frames.shape[0]} frames) in {demo_path}; the index map "
                f"must guarantee a full {T}-frame window."
            )
        return self._lanczos_squash(frames).astype(np.float32)

    def _lanczos_squash(self, frames: np.ndarray) -> np.ndarray:
        """LANCZOS-resize to a square of width img_shape[0], matching preprocessing."""
        img_size = int(self.img_shape[0])
        if frames.shape[1] == img_size and frames.shape[2] == img_size:
            return frames
        T = frames.shape[0]
        out = np.empty((T, img_size, img_size, 3), dtype=np.uint8)
        for t in range(T):
            pil = Image.fromarray(frames[t]).resize((img_size, img_size), Image.LANCZOS)
            out[t] = np.asarray(pil, dtype=np.uint8)
        return out

    def process_data(self, data: Dict) -> Dict:
        if "images" in data:
            imgs = (data["images"] / 255.0).astype(np.float32)
            if self.data_img_size != self.img_shape:
                v, fr = imgs.shape[:2]
                t = torch.from_numpy(rearrange(imgs, "v f h w c -> (v f) c h w")).float()
                imgs = rearrange(self.resize(t).numpy(), "(v f) c h w -> v f h w c", v=v, f=fr).astype(np.float32)
            data["images"] = imgs

        if "actions" in data:
            acts = data["actions"]
            ref = data.pop("action_reference")
            # Pad with last action (delta=0 under relative normalization).
            if acts.shape[0] < self.true_horizon:
                pad = np.repeat(acts[-1:], self.true_horizon - acts.shape[0], axis=0)
                acts = np.concatenate([acts, pad], axis=0)
            if self.relative_rotation_actions:
                acts = to_relative_rot6d_eepose20(acts, ref)
            if self.normalize_actions:
                acts = self.normalizer.normalize(acts, ref)
            data["actions"] = acts.astype(np.float32)

        if "tracks" in data:
            tr = np.nan_to_num(data["tracks"], nan=0.0, posinf=0.0, neginf=0.0)
            if self.tracker_mode == "2d":
                if tr.shape[1] > self.true_horizon:
                    tr = tr[:, : self.true_horizon]
                tr = tr[..., [1, 0]]
                tr = normalize_traj(tr, self.data_img_size)
            if self.track_pred_horizon != self.true_horizon:
                if self.interp_method == "linear":
                    tr = interpolate_traj(tr, self.track_pred_horizon)
                elif self.interp_method == "spline":
                    tr = interpolate_traj_spline(tr, self.track_pred_horizon)
                elif self.interp_method != "none":
                    raise NotImplementedError(f"Unsupported interp method {self.interp_method}")
            if self.tracks_normalizer is not None:
                # load_tracks already removed the cumulative-displacement anchor.
                tr = self.tracks_normalizer.normalize(tr)
            data["traj"] = tr.astype(np.float32)
            data.pop("tracks", None)

        if "vis" in data:
            vis = data["vis"]
            if vis.shape[1] > self.true_horizon:
                vis = vis[:, : self.true_horizon]
            if vis.ndim == 3:
                vis = vis[..., None]
            vis = np.nan_to_num(vis, nan=0.0, posinf=0.0, neginf=0.0)
            if self.track_pred_horizon != self.true_horizon:
                if self.interp_method == "linear":
                    vis = interpolate_traj(vis, self.track_pred_horizon)
                elif self.interp_method == "spline":
                    vis = interpolate_traj_spline(vis, self.track_pred_horizon)
                elif self.interp_method != "none":
                    raise NotImplementedError(f"Unsupported interp method {self.interp_method}")
            data["vis"] = vis.astype(np.float32)

        if "raw_windows" in data:
            data["raw_windows"] = np.nan_to_num(
                data["raw_windows"], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)

        # motion_tokens bypasses tracks->traj, so normalize that alias separately.
        if "motion_tokens" in data and self.tracks_normalizer is not None:
            data["motion_tokens"] = self.tracks_normalizer.normalize(
                data["motion_tokens"]
            ).astype(np.float32)

        if "actions" in data:
            data["action"] = data.pop("actions")

        return data
