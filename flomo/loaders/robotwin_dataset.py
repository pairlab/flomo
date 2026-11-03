import io
import json
import os
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image

from flomo.loaders.realyam_dataset import RealYamDataset
from flomo.utils.action_normalizer import ActionNormalizer, compute_normalizer_stats
from flomo.utils.action_preprocess import robotwin_ee_to_eepose20


CAMERA_KEY_MAP = {
    "cam_head": "vision/cam_head/colors",
    "cam_left_wrist": "vision/cam_left_wrist/colors",
    "cam_right_wrist": "vision/cam_right_wrist/colors",
}

# JPEG comment marking standard-RGB frames (demo_clean archives refreshed 2026-09-14).
RGB_MARKER = b"XPL-RGB1"


def decode_rgb(buf: bytes) -> np.ndarray:
    """Decode one JPEG frame. Unmarked frames (older archives) store BGR, so reverse their channels."""
    img = Image.open(io.BytesIO(buf))
    rgb = np.asarray(img.convert("RGB"))
    if "comment" in img.info and img.info["comment"] == RGB_MARKER:
        return rgb
    return rgb[:, :, ::-1]


def read_instruction(f) -> str:
    """Episode task string: `instruction` in older archives, else the first entry of the
    `instructions` JSON list (refreshed archives; equal to the old `instruction`)."""
    if "instruction" in f:
        return f["instruction"][()].decode("utf-8")
    return json.loads(f["instructions"][()])[0]


def read_rgb_frames(f, view: str, frame_indices) -> np.ndarray:
    """Decode `view` frames from an open RoboTwin episode -> `(t, h, w, 3)` uint8."""
    arr = f[CAMERA_KEY_MAP[view]]
    return np.stack([decode_rgb(arr[i]) for i in frame_indices], axis=0)


def read_eepose20(f, group: str, rows: slice) -> np.ndarray:
    """Read `state` or `action` EE poses and grippers as `(t, 20)` eepose_6drot."""
    return robotwin_ee_to_eepose20(
        f[f"{group}/left_ee_poses"][rows],
        f[f"{group}/left_ee_joint_states"][rows],
        f[f"{group}/right_ee_poses"][rows],
        f[f"{group}/right_ee_joint_states"][rows],
    )


class RoboTwinDataset(RealYamDataset):
    """RoboTwin 2.0 episodes plus RealYam-layout sidecars; names are `<task>_<train|val>`.

    Actions are next-state targets (`action[t] == state[t + 1]`), so each window
    is relative to the observed `state[start_t]`.
    """

    CANONICAL_VIEWS = {
        "primary": "cam_head",
        "left": "cam_left_wrist",
        "right": "cam_right_wrist",
    }
    _INDEX_MAP_CACHE_DIR = os.path.expanduser("~/.cache/flomo/index_maps/robotwin")
    _UNSUPPORTED_KEYS = {"images", "depth"}

    def __init__(
        self,
        raw_dir: str,
        embodiment: str,
        splits_dir: str,
        keys_to_load: List[str],
        filter_no_motion: bool,
        **kwargs,
    ) -> None:
        bad = self._UNSUPPORTED_KEYS & set(keys_to_load)
        if bad:
            raise ValueError(f"RoboTwinDataset does not support keys_to_load={sorted(bad)}.")
        if filter_no_motion:
            raise ValueError("RoboTwinDataset does not support filter_no_motion=true.")
        self.raw_dir = os.path.expanduser(raw_dir)
        self.embodiment = embodiment
        super().__init__(
            splits_dir=splits_dir,
            keys_to_load=keys_to_load,
            filter_no_motion=filter_no_motion,
            **kwargs,
        )

    @staticmethod
    def _parse_dataset_name(name: str) -> Tuple[str, str]:
        """``<task>_<split>`` -> (task, split)."""
        task, _, split = name.rpartition("_")
        if not task or split not in {"train", "val"}:
            raise ValueError(f"RoboTwin dataset name must be '<task>_train' or '<task>_val', got {name!r}.")
        return task, split

    def _modal_paths(self, base: str, task: str, demo_id: str) -> Dict[str, str]:
        paths = super()._modal_paths(base, task, demo_id)
        paths["demo_path"] = os.path.join(self.raw_dir, task, self.embodiment, "data", f"{demo_id}.hdf5")
        return paths

    def _compute_normalizer_from_index(self) -> ActionNormalizer:
        """Stats over unique demos, relative to the observed state at each start."""
        seen: Set[Tuple[str, str]] = set()
        seqs: List[np.ndarray] = []
        refs: List[np.ndarray] = []
        # Drop the last row so chunk starts match index starts [0, T - true_horizon).
        rows = slice(0, -1)
        for entry in self.index_map:
            key = (entry["task"], entry["demo_id"])
            if key in seen:
                continue
            seen.add(key)
            with self._open(entry["demo_path"]) as f:
                seqs.append(read_eepose20(f, "action", rows))
                refs.append(read_eepose20(f, "state", rows))
        if not seqs:
            raise RuntimeError("Cannot compute normalizer stats: index map is empty.")
        normalizer = compute_normalizer_stats(
            action_sequences=seqs,
            reference_sequences=refs,
            action_space=self.action_space,
            horizon=self.true_horizon,
        )
        normalizer.metadata = {"tasks": sorted({t for t, _ in seen}), "num_demos": len(seqs)}
        return normalizer

    def load_actions(self, idx_dict: Dict) -> Dict:
        s, e = idx_dict["start_t"], idx_dict["end_t"]
        with self._open(idx_dict["demo_path"]) as f:
            acts = read_eepose20(f, "action", slice(s, e, self.subsampling))
            ref = read_eepose20(f, "state", slice(s, s + 1))[0]
        return {"actions": acts, "action_reference": ref}

    def load_text(
        self,
        idx_dict: Dict,
        include_text: bool = True,
        include_text_emb: bool = False,
    ) -> Dict:
        out = super().load_text(idx_dict, include_text=False, include_text_emb=include_text_emb)
        if include_text:
            with self._open(idx_dict["demo_path"]) as f:
                out["text"] = read_instruction(f)
        return out

    def _read_video_window(self, demo_path: str, view_key: str, start_t: int) -> np.ndarray:
        """RGB window `(true_horizon + 1, img_size, img_size, 3)`, LANCZOS-squashed like preprocessing."""
        n = self.true_horizon + 1
        with self._open(demo_path) as f:
            T = f[CAMERA_KEY_MAP[view_key]].shape[0]
            if start_t + n > T:
                raise IndexError(f"video window [{start_t}, {start_t + n}) runs past {T} frames in {demo_path}.")
            frames = read_rgb_frames(f, view_key, range(start_t, start_t + n))
        return self._lanczos_squash(frames).astype(np.float32)
