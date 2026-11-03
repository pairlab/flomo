"""RoboTwin 2.0 preprocessing: head-camera tracks, text embeddings, and Wan latents
for all three cameras, written in the RealYam sidecar layout."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import h5py
import hydra
from omegaconf import DictConfig, OmegaConf

from flomo.loaders.robotwin_dataset import CAMERA_KEY_MAP, read_instruction, read_rgb_frames
from flomo.utils.config_validation import PREPROCESS_MODES, validate_preprocess_config
from preprocessing.preprocess_base import Sample, run_dataset
from preprocessing.preprocess_realyam import PreprocessRealYam, _resize_video
from preprocessing.preprocessing_utils import save_tracks_config, split_indices


VALID_MODES = PREPROCESS_MODES["robotwin"]


def _task_demos(source_root: str, task: str, embodiment: str) -> List[Path]:
    task_dir = Path(source_root) / task / embodiment / "data"
    demos = sorted(task_dir.glob("episode_*.hdf5"))
    if not demos:
        raise FileNotFoundError(f"No episode_*.hdf5 in {task_dir}")
    return demos


def write_splits(cfg: DictConfig, base_dir: str) -> None:
    """Write `splits/<task>.json` with every episode in both train and val."""
    split_dir = os.path.join(base_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)
    for task in cfg.task_names:
        ids = [p.stem for p in _task_demos(os.path.abspath(cfg.source_dir), task, cfg.embodiment)]
        path = os.path.join(split_dir, f"{task}.json")
        if os.path.exists(path):
            with open(path) as f:
                splits = json.load(f)
            if splits["train"] != ids or splits["val"] != ids:
                raise ValueError(f"{path} does not list every episode in {cfg.source_dir}/{task}.")
            continue
        # Atomic replace: concurrent workers write identical content.
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"train": ids, "val": ids}, f, indent=2)
        os.replace(tmp, path)


class PreprocessRoboTwin(PreprocessRealYam):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        if cfg.mode not in VALID_MODES:
            raise ValueError(f"RoboTwin preprocessing supports modes={VALID_MODES}; got '{cfg.mode}'.")
        if list(cfg.camera_names) != ["cam_head"]:
            raise ValueError(f"RoboTwin tracks use camera_names=[cam_head], got {list(cfg.camera_names)}.")
        if not cfg.task_names:
            raise ValueError("RoboTwin preprocessing requires task_names.")

        self.source_root = os.path.abspath(cfg.source_dir)
        self.dest_root = os.path.abspath(cfg.dest_dir)
        self.output_root = os.path.join(self.dest_root, cfg.output_name)
        self.camera_names = list(cfg.camera_names)

        items = [
            {"task_name": task, "demo_file": str(p)}
            for task in cfg.task_names
            for p in _task_demos(self.source_root, task, cfg.embodiment)
        ]
        if cfg.max_demos is not None:
            items = items[: int(cfg.max_demos)]
        idxs = split_indices(len(items), int(cfg.num_workers), int(cfg.worker_id))
        self.items = [items[i] for i in idxs]

        print(f"[Init] source_dir={self.source_root}")
        print(f"[Init] output_root={self.output_root}")
        print(f"[Init] tasks={list(cfg.task_names)} total demos={len(items)}")
        print(f"[Init] worker {cfg.worker_id}/{cfg.num_workers} demos={len(self.items)}")

    def to_sample(self, item: Any, cfg: DictConfig) -> Sample:
        demo_file = item["demo_file"]
        meta = {"task_name": item["task_name"], "demo_file": demo_file}
        sample_id = Path(demo_file).stem

        with h5py.File(demo_file, "r") as f:
            if cfg.mode == "text_emb":
                text = read_instruction(f)
                return Sample(id=sample_id, videos={}, text=text, meta=meta)

            if cfg.mode == "tracks":
                views, img_size = self.camera_names, int(cfg.img_size)
            elif cfg.mode == "vae_latent_h16":
                views, img_size = list(CAMERA_KEY_MAP), int(cfg.wan.image_hw[0])
            else:
                raise ValueError(f"Unknown mode: {cfg.mode}")
            T = f[CAMERA_KEY_MAP["cam_head"]].shape[0]
            videos: Dict[str, Any] = {
                view: _resize_video(read_rgb_frames(f, view, range(T)), img_size) for view in views
            }
        return Sample(id=sample_id, videos=videos, meta=meta)


@hydra.main(config_path="../cfg/preprocessing", config_name="preprocess_robotwin", version_base="1.2")
def main(cfg: DictConfig) -> None:
    validate_preprocess_config("robotwin", cfg)

    base_dir = os.path.join(os.path.abspath(cfg.dest_dir), cfg.output_name)
    write_splits(cfg, base_dir)

    if cfg.mode == "tracks":
        save_tracks_config(cfg, base_dir)
    else:
        save_dir = os.path.join(base_dir, cfg.mode)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(save_dir, "config.yaml"))

    run_dataset(PreprocessRoboTwin(cfg), cfg)


if __name__ == "__main__":
    main()
