"""
RealYam preprocessing for SpatialTracker track export.

Reads already-exported per-demo HDF5 files under:

    <source_dir>/<task_name>/demo_<id>.hdf5

Tracks are written under:

    <dest_dir>/<output_name>/<common_params>/<tracker_output>/<task_name>/demo_<id>.hdf5
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from flomo.utils.config_validation import PREPROCESS_MODES, validate_preprocess_config
from preprocessing.preprocess_base import (
    PreprocessDataset,
    Sample,
    TrackProcessor,
    WanTextEmbProcessor,
    WanVaeLatentH16Processor,
    run_dataset,
)
from preprocessing.preprocessing_utils import (
    _get_output_subfolder,
    get_track_processor_func,
    load_tracker,
    save_tracks_config,
    split_indices,
)

# DreamZero modules (cloned at ./dreamzero)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dreamzero"))
from groot.vla.model.dreamzero.modules.wan_video_text_encoder import WanTextEncoder
from groot.vla.model.dreamzero.modules.wan_video_vae import WanVideoVAE38


CAMERA_KEY_MAP = {
    "front_img_1": "observations/images/front_img_1",
    "left_wrist_img": "observations/images/left_wrist_img",
    "right_wrist_img": "observations/images/right_wrist_img",
}




def _resize_video(video: np.ndarray, img_size: int) -> np.ndarray:
    # LANCZOS squash to (img_size, img_size), matching the training data.
    if video.ndim != 4 or video.shape[3] != 3:
        raise ValueError(f"Expected (T,H,W,3) RGB video, got {video.shape}")
    if video.shape[1] == img_size and video.shape[2] == img_size:
        return video

    T = video.shape[0]
    out = np.empty((T, img_size, img_size, 3), dtype=np.uint8)
    for t in range(T):
        pil = Image.fromarray(video[t]).resize((img_size, img_size), Image.LANCZOS)
        out[t] = np.asarray(pil, dtype=np.uint8)
    return out


VALID_MODES = PREPROCESS_MODES["realyam"]


class PreprocessRealYam(PreprocessDataset):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        if cfg.mode not in VALID_MODES:
            raise ValueError(f"RealYam preprocessing supports modes={VALID_MODES}; got '{cfg.mode}'.")
        if cfg.img_size <= 0:
            raise ValueError(f"img_size must be > 0, got {cfg.img_size}")
        if cfg.max_demos is not None and int(cfg.max_demos) <= 0:
            raise ValueError(f"max_demos must be > 0, got {cfg.max_demos}")

        self.source_root = os.path.abspath(cfg.source_dir)
        self.dest_root = os.path.abspath(cfg.dest_dir)
        self.output_root = os.path.join(self.dest_root, cfg.output_name)

        if not os.path.isdir(self.source_root):
            raise FileNotFoundError(f"Source directory not found: {self.source_root}")

        self.camera_names = list(cfg.camera_names)
        for view in self.camera_names:
            if view not in CAMERA_KEY_MAP:
                raise ValueError(
                    f"Invalid camera view '{view}'. Must be one of {list(CAMERA_KEY_MAP.keys())}"
                )

        task_dirs: Dict[str, Path] = {}
        for path in sorted(Path(self.source_root).iterdir()):
            if not path.is_dir():
                continue
            demo_paths = sorted(path.glob("demo_*.hdf5"))
            if demo_paths:
                task_dirs[path.name] = path

        if not task_dirs:
            raise FileNotFoundError(
                f"No task directories with demo_*.hdf5 files found under {self.source_root}"
            )

        requested_tasks = list(cfg.task_names)
        if requested_tasks:
            missing = sorted(set(requested_tasks) - set(task_dirs))
            if missing:
                raise FileNotFoundError(f"Requested task_names not found: {missing}")
            sel_tasks = requested_tasks
        else:
            sel_tasks = sorted(task_dirs)

        items: List[Dict[str, str]] = []
        for task_name in sel_tasks:
            task_dir = task_dirs[task_name]
            for demo_path in sorted(task_dir.glob("demo_*.hdf5")):
                items.append(
                    {
                        "task_name": task_name,
                        "demo_file": str(demo_path),
                    }
                )

        if cfg.max_demos is not None:
            items = items[: int(cfg.max_demos)]

        if not items:
            raise ValueError("No RealYam demos selected for preprocessing.")

        idxs = split_indices(len(items), int(cfg.num_workers), int(cfg.worker_id))
        self.items = [items[i] for i in idxs]

        print(f"[Init] source_dir={self.source_root}")
        print(f"[Init] output_root={self.output_root}")
        print(f"[Init] tasks={sel_tasks}")
        print(f"[Init] total demos={len(items)}")
        print(f"[Init] worker {cfg.worker_id}/{cfg.num_workers} demos={len(self.items)}")

    @staticmethod
    def _prep_model(model, device):
        model = model.to(device=device, dtype=torch.bfloat16).eval()
        model.requires_grad_(False)
        return model

    def build_models(self, cfg: DictConfig) -> Dict[str, Any]:
        models: Dict[str, Any] = {}
        if cfg.mode == "tracks":
            load_tracker(cfg, models)
            return models

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models["device"] = device

        if cfg.mode == "text_emb":
            text_encoder = WanTextEncoder()
            text_encoder.load_state_dict(torch.load(cfg.text_encoder_weights, map_location="cpu"))
            models["text_encoder"] = self._prep_model(text_encoder, device)
            models["tokenizer"] = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
        elif cfg.mode == "vae_latent_h16":
            vae = WanVideoVAE38(vae_pretrained_path=cfg.vae_weights)
            vae.model.load_state_dict(torch.load(cfg.vae_weights, map_location="cpu", weights_only=True))
            models["vae"] = self._prep_model(vae, device)
        return models

    def build_processors(self, cfg: DictConfig, models: Dict[str, Any]) -> Dict[str, Any]:
        if cfg.mode == "tracks":
            track_fn, models = get_track_processor_func(cfg, models)
            return {
                "tracks": TrackProcessor(
                    track_fn=track_fn,
                    models=models,
                    init_queries=cfg.init_queries,
                    reinit=cfg.reinit,
                    horizon=cfg.horizon,
                    n_tracks=cfg.n_tracks,
                    batch_size=cfg.batch_size,
                    track_outputs=cfg.track_outputs,
                    multi_view=cfg.multi_view,
                    output_size=cfg.output_size,
                    chunk_size=cfg.chunk_size,
                    subsampling=cfg.subsampling,
                )
            }
        if cfg.mode == "text_emb":
            return {"text_emb": WanTextEmbProcessor(
                text_encoder=models["text_encoder"],
                tokenizer=models["tokenizer"],
                device=models["device"],
            )}
        if cfg.mode == "vae_latent_h16":
            return {"vae_latent_h16": WanVaeLatentH16Processor(
                vae=models["vae"],
                device=models["device"],
                image_hw=cfg.wan.image_hw,
                horizon=cfg.horizon,
            )}
        raise ValueError(f"Unknown mode: {cfg.mode}")

    def iter_items(self, cfg: DictConfig) -> Iterable[Any]:
        rows = tqdm(self.items, desc="RealYam demos", leave=False) if cfg.progress else self.items
        yield from rows

    def to_sample(self, item: Any, cfg: DictConfig) -> Sample:
        demo_file = item["demo_file"]
        meta = {"task_name": item["task_name"], "demo_file": demo_file}
        sample_id = Path(demo_file).stem

        with h5py.File(demo_file, "r") as reader:
            if cfg.mode == "tracks":
                videos: Dict[str, np.ndarray] = {}
                for view in self.camera_names:
                    img_key = CAMERA_KEY_MAP[view]
                    if img_key not in reader:
                        raise KeyError(f"Missing {img_key} in {demo_file}")
                    video = np.asarray(reader[img_key], dtype=np.uint8)
                    videos[view] = _resize_video(video, int(cfg.img_size))
                return Sample(id=sample_id, videos=videos, meta=meta)

            if cfg.mode == "text_emb":
                if "text" not in reader:
                    raise KeyError(f"Missing top-level 'text' in {demo_file}")
                t = reader["text"][()]
                text = t.decode("utf-8") if isinstance(t, bytes) else str(t)
                return Sample(id=sample_id, videos={}, text=text, meta=meta)

            if cfg.mode == "vae_latent_h16":
                view = "front_img_1"
                img_key = CAMERA_KEY_MAP[view]
                if img_key not in reader:
                    raise KeyError(f"Missing {img_key} in {demo_file}")
                video = np.asarray(reader[img_key], dtype=np.uint8)
                resized = _resize_video(video, int(cfg.wan.image_hw[0]))
                return Sample(id=sample_id, videos={view: resized}, meta=meta)

        raise ValueError(f"Unknown mode: {cfg.mode}")

    def _get_common_params_folder(self) -> str:
        cfg = self.cfg
        if cfg.mode == "tracks":
            reinit_str = f"_reinit_{cfg.horizon}" if cfg.reinit else ""
            return f"{cfg.init_queries}_{cfg.n_tracks}{reinit_str}"
        if cfg.mode in ("text_emb", "vae_latent_h16"):
            return cfg.mode
        raise ValueError(f"Unknown mode: {cfg.mode}")

    def output_paths(self, sample: Sample, cfg: DictConfig) -> Dict[str, str]:
        task_name = sample.meta["task_name"]
        common_params_folder = self._get_common_params_folder()

        if cfg.mode == "tracks":
            base_dir = os.path.join(self.output_root, common_params_folder)
            paths: Dict[str, str] = {}
            for output_type in cfg.track_outputs:
                output_subfolder = _get_output_subfolder(cfg, output_type)
                output_dir = os.path.join(base_dir, output_subfolder, task_name)
                paths[output_type] = os.path.join(output_dir, f"{sample.id}.hdf5")
            return paths

        # text_emb / vae_latent_h16: <output_root>/<mode>/<task>/<demo_id>.hdf5
        output_dir = os.path.join(self.output_root, common_params_folder, task_name)
        return {cfg.mode: os.path.join(output_dir, f"{sample.id}.hdf5")}

    def on_end(self, cfg: DictConfig, models: Dict[str, Any]) -> None:
        pass


@hydra.main(config_path="../cfg/preprocessing", config_name="preprocess_realyam", version_base="1.2")
def main(cfg: DictConfig) -> None:
    validate_preprocess_config("realyam", cfg)

    base_dir = os.path.join(os.path.abspath(cfg.dest_dir), cfg.output_name)

    if cfg.mode == "tracks":
        save_tracks_config(cfg, base_dir)
    else:
        save_dir = os.path.join(base_dir, cfg.mode)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(save_dir, "config.yaml"))

    dataset = PreprocessRealYam(cfg)
    run_dataset(dataset, cfg)


if __name__ == "__main__":
    main()
