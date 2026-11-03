import glob
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from einops import rearrange
from torchvision.transforms import Resize
from omegaconf import OmegaConf

from flomo.loaders.base_dataset import BaseDataset
from flomo.utils.action_normalizer import ActionNormalizer, compute_normalizer_stats
from flomo.utils.action_preprocess import to_relative_rot6d_eepose20
from flomo.utils.data_utils import (
    interpolate_traj,
    interpolate_traj_spline,
    frame_to_chunk_position,
)
from flomo.utils.tracks_normalizer import TracksNormalizer
from pathlib import Path

from tqdm import tqdm


# Tracks are stored only for the top view.
TRACK_VIEW = "top"

# Absolute EE pose: two arms × [position(3), continuous rot6d(6), gripper(1)].
ACTION_SPACE = "eepose_6drot"

# Sum absolute joint deltas over 14 DOFs and true_horizon-1 steps (radians).
# The threshold separates the static (~0.018) and moving clusters near 0.035.
STATIC_EXCURSION_THRESH = 0.035


class MolmoAct2Dataset(BaseDataset):
    """
    MolmoAct2-BimanualYAM dataset loader built on top of BaseDataset.

    Produced by preprocessing/preprocess_molmoact2.py. The on-disk train/val
    split is materialized into directories (egoverse-style stable hash applied
    at preprocess time), so each dataset entry names exactly one split.

    Directory layout (per dataset entry):

        base = <dataset_root>/<dataset_dir>/<split>

        Tracks (top view only):
            base/<track_method>/<tracker>_3d/<task>/<sample_id>.hdf5
            base/<track_method>/<tracker>_depth/<task>/<sample_id>.hdf5
            base/<track_method>/<tracker>_pose/<task>/<sample_id>.hdf5

        Other modalities:
            base/video/<task>/<sample_id>.hdf5            # root/<view>/video
            base/text/<task>/<sample_id>.hdf5             # text
            base/text_emb/<task>/<sample_id>.hdf5         # text_emb
            base/vae_latent_h16/<task>/<sample_id>.hdf5   # root/<view>/vae_latent_h16
            base/actions/<task>/<sample_id>.hdf5          # actions/<space>, observations/joint_positions

    `<sample_id>` is `{subset}__ep{episode_index:04d}`; `<task>` is the
    filesystem-sanitized coarse task name.
    """

    # The top camera is primary; left/right are scene cameras.
    CANONICAL_VIEWS = {"primary": "top", "left": "left", "right": "right"}

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
        normalize_actions: bool = True,
        relative_rotation_actions: bool = True,
        normalizer_stats_path: Optional[str] = None,
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
        # Tracks/actions use the primary view even without image conditioning.
        self.view = self.CANONICAL_VIEWS["primary"]
        self.true_horizon = int(true_horizon)

        if self.tracker_mode == "3d":
            self.tracks_key = "3d_tracks"
        elif self.tracker_mode == "2d":
            self.tracks_key = "tracks"
        else:
            raise ValueError(f"Unknown tracker mode '{self.tracker_mode}' for MolmoAct2.")

        # Read stored geometry from preprocessing config, not caller defaults.
        pp_cfg = self._load_preprocessing_cfg()
        if bool(pp_cfg["multi_view"]):
            raise NotImplementedError("MolmoAct2 loader assumes multi_view=False.")
        self.preprocessing_horizon = int(pp_cfg["horizon"])
        # Optical flow needs true_horizon-1 consecutive pose frames in one chunk;
        # frame_to_chunk_position guarantees preprocessing_horizon frames.
        if supervision == "optical_flow" and self.true_horizon - 1 > self.preprocessing_horizon:
            raise ValueError(
                f"optical_flow needs {self.true_horizon - 1} consecutive pose frames in one "
                f"chunk, but preprocessing_horizon={self.preprocessing_horizon}. "
                "Rerun preprocessing with a larger horizon."
            )
        if subsampling is not None and int(subsampling) != int(pp_cfg["subsampling"]):
            raise ValueError(f"subsampling mismatch: caller={subsampling}, config={int(pp_cfg['subsampling'])}")
        if num_tracks is not None and int(num_tracks) != int(pp_cfg["n_tracks"]):
            raise ValueError(f"num_tracks mismatch: caller={num_tracks}, config={int(pp_cfg['n_tracks'])}")
        self.subsampling = int(pp_cfg["subsampling"])
        num_tracks = int(pp_cfg["n_tracks"])

        self.action_space = ACTION_SPACE
        self.normalize_actions = bool(normalize_actions)
        self.relative_rotation_actions = bool(relative_rotation_actions)
        if tracks_normalizer is not None and tracks_normalizer != "ramen":
            raise ValueError(
                f"tracks_normalizer must be null or 'ramen'; got {tracks_normalizer!r}"
            )
        if tracks_normalizer == "ramen" and supervision != "cumulative_disp":
            raise ValueError(
                "tracks_normalizer='ramen' requires supervision='cumulative_disp' "
                f"(got supervision='{supervision}')"
            )
        # Per-modality preprocessed/live toggle for the VAE-latent modalities.
        # Live (False) emits raw RGB; the modality VAE-encodes.
        self.img_use_preprocessed = bool(img_use_preprocessed)
        self.video_use_preprocessed = bool(video_use_preprocessed)

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

        # Discover the stored video resolution from the first sample.
        if "images" in self.keys_to_load:
            video_source = self.index_map[0]["video_path"]
            with self._open(video_source) as f:
                video = f["root"][self.view]["video"]
                if video.ndim != 4:
                    raise ValueError(f"Expected video with 4 dims, got {video.ndim}")
                self.data_img_size = (int(video.shape[1]), int(video.shape[2]))
        else:
            self.data_img_size = tuple(self.img_shape)

        # Action normalizer. Explicit path wins; otherwise compute on the fly
        # from this dataset's demos. Mirrors RealYamDataset.
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
                stats_tasks = set(self.normalizer.metadata.get("tasks", []))
                ds_tasks = {e["task"] for e in self.index_map}
                missing = ds_tasks - stats_tasks
                if stats_tasks and missing:
                    raise ValueError(
                        f"Dataset tasks {sorted(missing)} are not covered by normalizer stats at "
                        f"{stats_path} (stats covers {sorted(stats_tasks)})."
                    )
            else:
                self.normalizer = self._compute_normalizer_from_index()

        # Optional tracks normalizer (cumulative_disp + Ramen). Cache file lives next
        # to action stats under the per-dataset normalizer_stats/ dir (same
        # parent the action normalizer uses).
        dataset_dir, _ = self._parse_dataset_name(self.dataset_names[0])
        default_tracks_stats = (
            Path(self._dataset_root())
            / dataset_dir
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

    def _compute_normalizer_from_index(self) -> ActionNormalizer:
        """Compute action normalizer stats over unique actions files in the index."""
        seen: set = set()
        seqs: List[np.ndarray] = []
        for entry in self.index_map:
            path = entry["actions_path"]
            if path in seen:
                continue
            seen.add(path)
            with self._open(path) as f:
                seqs.append(f[f"actions/{self.action_space}"][:].astype(np.float32))
        if not seqs:
            raise RuntimeError("Cannot compute normalizer stats: index map is empty.")
        normalizer = compute_normalizer_stats(
            action_sequences=seqs,
            reference_sequences=seqs,  # action[t] is the pose at frame t
            action_space=self.action_space,
            horizon=self.true_horizon,
        )
        tasks = sorted({e["task"] for e in self.index_map})
        normalizer.metadata = {"tasks": tasks, "num_demos": len(seqs)}
        return normalizer

    def _dataset_root(self) -> str:
        base = self.dataset_path
        if base is None:
            return os.path.join(self.root_dir, "preprocessed_data")
        base = os.path.expanduser(base)
        if not os.path.isabs(base):
            base = os.path.join(self.root_dir, base)
        return base

    @staticmethod
    def _parse_dataset_name(name: str) -> Tuple[str, str]:
        parts = [p for p in name.split("_") if p]
        split = parts[-1].lower()
        valid = {"train", "val", "validation", "test"}
        if split not in valid:
            raise ValueError(
                f"MolmoAct2 dataset '{name}' must end with one of {sorted(valid)}."
            )
        dataset_dir = "_".join(parts[:-1])
        return dataset_dir, split

    @staticmethod
    def _modal_path(split_root: str, folder: str, task: str, sample_id: str) -> str:
        return os.path.join(split_root, folder, task, f"{sample_id}.hdf5")

    @staticmethod
    def _derive_modal_path(anchor_path: str, anchor_folder: str, target_folder: str) -> str:
        """Derive a sibling modality path from an anchor path via folder-name string replace."""
        needle = f"{os.sep}{anchor_folder}{os.sep}"
        if needle not in anchor_path:
            raise ValueError(
                f"Cannot derive modal path: anchor folder '{anchor_folder}' not found in {anchor_path}"
            )
        return anchor_path.replace(needle, f"{os.sep}{target_folder}{os.sep}", 1)

    def _load_preprocessing_cfg(self):
        """Read the preprocessing config.yaml for this track_method (authoritative
        source for horizon / n_tracks / subsampling)."""
        dataset_dir, split = self._parse_dataset_name(self.dataset_names[0])
        config_path = os.path.join(
            self._dataset_root(), dataset_dir, split,
            self.track_method, f"{self.tracker}_{self.tracker_mode}", "config.yaml",
        )
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing preprocessing config at {config_path}")
        return OmegaConf.load(config_path)

    _INDEX_MAP_CACHE_DIR = os.path.expanduser("~/.cache/flomo/index_maps/molmoact2")

    def get_cache_file(self) -> str:
        dataset_str = "_".join(self.dataset_names)
        os.makedirs(self._INDEX_MAP_CACHE_DIR, exist_ok=True)
        suffix = f"{self.tracker}_{self.tracker_mode}"
        # The static-clip filter is baked into the index; encode its threshold
        # so changing STATIC_EXCURSION_THRESH invalidates stale caches.
        suffix += f"_st{STATIC_EXCURSION_THRESH}"
        return os.path.join(
            self._INDEX_MAP_CACHE_DIR, f"{dataset_str}_{self.track_method}_{suffix}_base.json"
        )

    def create_index_map(self) -> List[Dict]:
        """Scan track HDF5 files and emit one index entry per window start.

        Saves incremental progress to a .partial file every CHECKPOINT_INTERVAL
        files so a preempted scan resumes where it left off.
        """
        CHECKPOINT_INTERVAL = 500

        partial_file = self.get_cache_file() + ".partial"
        already_processed: set = set()
        index: List[Dict] = []
        if os.path.exists(partial_file):
            with open(partial_file, "r") as f:
                partial_state = json.load(f)
            already_processed = set(partial_state["processed_paths"])
            index = partial_state["index"]
            print(f"[Index Map] Resuming: {len(already_processed)} files done, "
                  f"{len(index)} entries collected.")

        base = self._dataset_root()
        needs_images = "images" in self.keys_to_load
        needs_text = "text" in self.keys_to_load
        needs_text_emb = "text_emb" in self.keys_to_load
        needs_depth = "depth" in self.keys_to_load
        needs_pose = "pose" in self.keys_to_load
        files_processed_this_run = 0
        n_starts, n_static = 0, 0  # static-filter stats for the run

        for dataset in self.dataset_names:
            dataset_dir, split = self._parse_dataset_name(dataset)
            split_root = os.path.join(base, dataset_dir, split)

            tracks_root = os.path.join(
                split_root, self.track_method, f"{self.tracker}_{self.tracker_mode}"
            )
            if not os.path.isdir(tracks_root):
                raise FileNotFoundError(
                    f"MolmoAct2 tracks directory missing: {tracks_root}. "
                    "Run preprocessing/preprocess_molmoact2.py to generate the data."
                )

            h5_paths = sorted(glob.glob(os.path.join(tracks_root, "*", "*.hdf5")))
            if not h5_paths:
                raise FileNotFoundError(f"No HDF5 files found under {tracks_root}")

            for track_path in tqdm(h5_paths, desc="Building index map"):
                if track_path in already_processed:
                    continue

                rel_path = os.path.relpath(track_path, tracks_root)
                parts = rel_path.split(os.sep)
                if len(parts) != 2:
                    raise ValueError(f"Malformed MolmoAct2 track path: {track_path}")
                task = parts[0]
                sample_id = os.path.splitext(parts[-1])[0]

                video_path = self._modal_path(split_root, "video", task, sample_id)
                text_path = self._modal_path(split_root, "text", task, sample_id)
                text_emb_path = self._modal_path(split_root, "text_emb", task, sample_id)
                actions_path = self._modal_path(split_root, "actions", task, sample_id)
                depth_path = self._modal_path(
                    os.path.join(split_root, self.track_method),
                    f"{self.tracker}_depth", task, sample_id,
                )
                pose_path = self._modal_path(
                    os.path.join(split_root, self.track_method),
                    f"{self.tracker}_pose", task, sample_id,
                )

                with self._open(track_path) as f:
                    vg = f["root"][TRACK_VIEW]
                    if self.tracks_key not in vg:
                        raise KeyError(f"{track_path} missing {self.tracks_key} dataset")
                    T = int(vg[self.tracks_key].shape[0])
                    if T == 0:
                        raise ValueError(f"Empty tracks tensor in {track_path}")
                    if "vis" in self.keys_to_load and "vis" not in vg:
                        raise KeyError(f"{track_path} missing vis dataset")

                # Validate presence only for modalities that will be loaded.
                # The actions file is always needed for the static-clip filter.
                if needs_images and not os.path.isfile(video_path):
                    raise FileNotFoundError(f"Missing MolmoAct2 video file: {video_path}")
                if needs_text and not os.path.isfile(text_path):
                    raise FileNotFoundError(f"Missing MolmoAct2 text file: {text_path}")
                if needs_text_emb and not os.path.isfile(text_emb_path):
                    raise FileNotFoundError(f"Missing MolmoAct2 text_emb file: {text_emb_path}")
                if not os.path.isfile(actions_path):
                    raise FileNotFoundError(f"Missing MolmoAct2 actions file: {actions_path}")
                if needs_depth and not os.path.isfile(depth_path):
                    raise FileNotFoundError(f"Missing MolmoAct2 depth file: {depth_path}")
                if needs_pose and not os.path.isfile(pose_path):
                    raise FileNotFoundError(f"Missing MolmoAct2 pose file: {pose_path}")

                # Joint state drives the static-clip filter: a window with no
                # real joint/gripper motion is the teleop-disabled setup period.
                with self._open(actions_path) as f:
                    obs_state = f["observations/joint_positions"][:]  # (T, 14)
                if obs_state.shape[0] != T:
                    raise ValueError(
                        f"joint_positions length {obs_state.shape[0]} != tracks length "
                        f"{T} for {sample_id}"
                    )

                for start_t in range(T):
                    if not self._start_valid(start_t, T):  # drop any clips which extend past the end of the video
                        continue
                    
                    if start_t >= T - self.preprocessing_horizon:  # vae_latent_h16 has only T-preprocessing_horizon windows
                        continue

                    n_starts += 1
                    # Drop the window if total joint movement is below the
                    # threshold (sum of |delta| over all 14 dofs and all steps).
                    win = obs_state[start_t: start_t + self.true_horizon]
                    if np.abs(np.diff(win, axis=0)).sum() < STATIC_EXCURSION_THRESH:
                        n_static += 1
                        continue
                    index.append({
                        "track_path": os.path.realpath(track_path),
                        "video_path": os.path.realpath(video_path),
                        "text_path": os.path.realpath(text_path),
                        "actions_path": os.path.realpath(actions_path),
                        "depth_path": os.path.realpath(depth_path),
                        "pose_path": os.path.realpath(pose_path),
                        "dataset": dataset,
                        "split": split,
                        "task": task,
                        "start_t": int(start_t),
                        "end_t": int(start_t + self.true_horizon),
                        "rollout_len": T,
                    })

                already_processed.add(track_path)
                files_processed_this_run += 1
                if files_processed_this_run % CHECKPOINT_INTERVAL == 0:
                    fd, tmp_path = tempfile.mkstemp(
                        dir=os.path.dirname(partial_file), suffix=".tmp"
                    )
                    with os.fdopen(fd, "w") as f:
                        json.dump({"processed_paths": list(already_processed),
                                   "index": index}, f)
                    os.replace(tmp_path, partial_file)

        if n_starts > 0:
            print(f"[Static Filter] Dropped {n_static}/{n_starts} static windows "
                  f"({100 * n_static / n_starts:.1f}%); thresh={STATIC_EXCURSION_THRESH}")
        return self._intern_index_map_strings(index)

    @staticmethod
    def _open(path: str):
        return h5py.File(path, "r")

    def load_images(self, idx_dict: Dict) -> Dict:
        start_t = idx_dict["start_t"]
        with self._open(idx_dict["video_path"]) as f:
            frame = f["root"][self.view]["video"][start_t].astype(np.float32)  # (H, W, C)
        return {"images": frame[None, None]}                                    # (V=1, F=1, H, W, C)

    def load_actions(self, idx_dict: Dict) -> Dict:
        """14-D bimanual joint actions for [start_t, end_t). Shape (T_window, 14)."""
        s, e = idx_dict["start_t"], idx_dict["end_t"]
        with self._open(idx_dict["actions_path"]) as f:
            acts = f[f"actions/{ACTION_SPACE}"][s:e].astype(np.float32)
        return {"actions": acts}

    def _track_idx(self, idx_dict: Dict) -> int:
        # MolmoAct2's index map `start_t` indexes the tracks array directly.
        return int(idx_dict["start_t"])

    @contextmanager
    def _tracks_group(self, idx_dict: Dict):
        with self._open(idx_dict["track_path"]) as f:
            yield f["root"][TRACK_VIEW]

    def _read_flow_extrs(self, idx_dict: Dict, n: int) -> np.ndarray:
        """`n` consecutive camera extrinsics at `start_t` (must fit one chunk)."""
        with self._open(idx_dict["pose_path"]) as f:
            extrs_ds = f["root"][TRACK_VIEW]["camera_extrinsics"]
            num_chunks, chunk_size, _, _ = extrs_ds.shape
            chunk_idx, start_pos, _ = frame_to_chunk_position(
                int(idx_dict["start_t"]), num_chunks, chunk_size,
                self.preprocessing_horizon,
            )
            if start_pos + n > chunk_size:
                raise ValueError(
                    f"optical_flow pose window [start_t={idx_dict['start_t']}, "
                    f"n={n}] spans chunks (chunk_idx={chunk_idx}, "
                    f"start_pos={start_pos}, chunk_size={chunk_size})."
                )
            return extrs_ds[chunk_idx, start_pos:start_pos + n].astype(np.float32)

    @contextmanager
    def _vae_latent_array(self, idx_dict: Dict, view_key: str):
        source = self._derive_modal_path(idx_dict["text_path"], "text", "vae_latent_h16")
        with self._open(source) as f:
            yield f["root"][view_key]["vae_latent_h16"]

    def _rgb_video_path(self, idx_dict: Dict) -> str:
        return idx_dict["video_path"]

    def load_text(
        self,
        idx_dict: Dict,
        include_text: bool = True,
        include_text_emb: bool = False,
    ) -> Dict:
        if not include_text and not include_text_emb:
            return {}
        out: Dict = {}
        if include_text:
            with self._open(idx_dict["text_path"]) as f:
                if "text" not in f:
                    raise KeyError(f"Missing text in {idx_dict['text_path']}")
                text = f["text"][()]
                if isinstance(text, bytes):
                    text = text.decode("utf-8")
                out["text"] = str(text)
        if include_text_emb:
            emb_source = self._derive_modal_path(idx_dict["text_path"], "text", "text_emb")
            with self._open(emb_source) as f:
                if "text_emb" not in f:
                    raise KeyError(f"Missing text_emb in {emb_source}")
                out["text_emb"] = f["text_emb"][()].astype(np.float32)
        return out

    def load_depth(self, idx_dict: Dict) -> Dict:
        with self._open(idx_dict["depth_path"]) as f:
            ds = f["root"][TRACK_VIEW]["depth_map"]
            num_chunks, chunk_size, _, _ = ds.shape
            chunk_idx, start_pos, _ = frame_to_chunk_position(
                idx_dict["start_t"], num_chunks, chunk_size, self.preprocessing_horizon
            )
            depth = ds[chunk_idx, start_pos][None].astype(np.float32)
        return {"depth": depth}

    def load_pose(self, idx_dict: Dict) -> Dict:
        with self._open(idx_dict["pose_path"]) as f:
            vg = f["root"][TRACK_VIEW]
            extrs_ds = vg["camera_extrinsics"]
            intrs_ds = vg["camera_intrinsics"]
            num_chunks, chunk_size, _, _ = extrs_ds.shape
            chunk_idx, start_pos, _ = frame_to_chunk_position(
                idx_dict["start_t"], num_chunks, chunk_size, self.preprocessing_horizon
            )
            return {
                "extrs": extrs_ds[chunk_idx, start_pos][None].astype(np.float32),
                "intrs": intrs_ds[chunk_idx, start_pos][None].astype(np.float32),
            }

    def _read_video_window(self, video_path: str, view_key: str, start_t: int) -> np.ndarray:
        """RGB window (1+4*t_frames, img_size, img_size, 3) for VAE/viz. Source
        is already crop+resized by preprocess_molmoact2._VideoProcessor."""
        T = self.true_horizon + 1
        with self._open(video_path) as f:
            frames = np.asarray(
                f["root"][view_key]["video"][start_t: start_t + T]
            ).astype(np.float32)
        if frames.shape[0] != T:
            raise IndexError(
                f"video window [start_t={start_t}, T={T}] runs past the clip "
                f"({frames.shape[0]} frames) in {video_path}; the index map "
                f"must guarantee a full {T}-frame window."
            )
        return frames

    def process_data(self, data: Dict) -> Dict:
        if "images" in data:
            imgs = (data["images"] / 255.0).astype(np.float32)
            if self.data_img_size != self.img_shape:
                v, fr = imgs.shape[:2]
                t = torch.from_numpy(rearrange(imgs, "v f h w c -> (v f) c h w")).float()
                imgs = rearrange(
                    self.resize(t).numpy(), "(v f) c h w -> v f h w c", v=v, f=fr
                ).astype(np.float32)
            data["images"] = imgs

        if "actions" in data:
            acts = data["actions"]
            # Pad short tails by repeating the last action.
            if acts.shape[0] < self.true_horizon:
                pad = np.repeat(acts[-1:], self.true_horizon - acts.shape[0], axis=0)
                acts = np.concatenate([acts, pad], axis=0)
            # Match RealYam's relative-rotation and normalization conventions.
            ref = acts[0].copy()
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

        # motion_tokens bypasses tracks->traj, so normalize that alias separately.
        if "motion_tokens" in data and self.tracks_normalizer is not None:
            data["motion_tokens"] = self.tracks_normalizer.normalize(
                data["motion_tokens"]
            ).astype(np.float32)

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

        if "actions" in data:
            data["action"] = data.pop("actions")

        return data
