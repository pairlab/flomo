"""MolmoAct2-BimanualYAM preprocessing + selection-index CLI.

Preprocessing modes: 'tracks', 'video', 'text', 'text_emb', 'vae_latent',
'vae_latent_h16', 'actions'. Driven by a selection index built with the
`index` subcommand: `python -m preprocessing.preprocess_molmoact2 index select ...`.

Output layout (egoverse-style, with split as a dir near the top so dataset-side
walks naturally pick up train/val):

    <dest_dir>/<output_name>/<split>/<common_params>/<task_safe>/<sample_id>.hdf5

For tracks the layout has an extra <subfolder> for each track output type:

    <dest_dir>/<output_name>/<split>/<common_params>/<subfolder>/<task_safe>/<sample_id>.hdf5

Tracks always use the `top` view only (project decision); video / vae_latent /
vae_latent_h16 default to all three views (configurable via cfg.camera_names).

Modality dependency (egoverse pattern): only `mode=video` decodes the raw mp4s.
`tracks`, `vae_latent` and `vae_latent_h16` read the already-preprocessed
`video` HDF5 store instead of decoding again — so run `mode=video`
first. The store is located at <dest_dir>/<source_output_name>/<split>/video/...
(`source_output_name` defaults to `output_name`). The video run must cover every
view these modes need (cfg.camera_names) at the same cfg.img_size.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

import av
import h5py
import hydra
import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from flomo.utils.config_validation import PREPROCESS_MODES, validate_preprocess_config
from preprocessing.preprocess_base import (
    PreprocessDataset,
    Processor,
    Sample,
    TrackProcessor,
    WanTextEmbProcessor,
    WanVaeLatentH16Processor,
    WanVaeLatentProcessor,
    run_dataset,
)
from preprocessing.preprocessing_utils import (
    _get_output_subfolder,
    get_track_processor_func,
    in_split,
    load_tracker,
    save_tracks_config,
    split_indices,
)

# Heavy imports (transformers / DreamZero / Wan encoders) are loaded lazily
# inside build_models() so that lightweight modes (text / actions) start fast.
# Make sure dreamzero is importable when those branches do fire.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dreamzero"))


VALID_MODES = PREPROCESS_MODES["molmoact2"]
TRACKS_VIEW = "top"  # tracks mode is hardcoded to top view per project decision

VIEWS = ("top", "left", "right")
VIEW_KEY = {v: f"observation.images.{v}" for v in VIEWS}

FULL_INDEX_NAME = "episodes_full.jsonl"


# ---------------------------------------------------------------------------
# Episode refs + task string normalization
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_TRAIL = re.compile(r"[.\s]+$")


@dataclass(frozen=True)
class ViewRef:
    chunk: int
    file: int
    from_t: float
    to_t: float


@dataclass(frozen=True)
class EpisodeRef:
    """Pointer to one episode in one subset, with everything needed to fetch it."""
    subset: str
    subset_dir: str
    episode_index: int
    fps: int
    length: int
    duration_s: float
    task_coarse: str          # raw string from tasks.parquet
    task_coarse_norm: str     # lowercased / whitespace-collapsed / trailing-punct-stripped
    task_annotated: str       # per-episode string from tasks_annotated.parquet
    data_chunk: int
    data_file: int
    data_from_index: int
    data_to_index: int
    views: Dict[str, ViewRef] = field(default_factory=dict)

    @property
    def sample_id(self) -> str:
        return f"{self.subset}__ep{self.episode_index:04d}"


def normalize_task(s: str) -> str:
    """Normalize a coarse task string for grouping.

    Lowercase, collapse internal whitespace, strip trailing periods/whitespace.
    Keeps strings like 'sort waste into correct bins and load items into the
    dishwasher rack' equal regardless of trailing punctuation/casing.
    """
    if s is None:
        raise ValueError("task string is None")
    out = _WS.sub(" ", str(s).strip().lower())
    out = _TRAIL.sub("", out)
    if not out:
        raise ValueError("normalized task string is empty")
    return out


def task_safe_dirname(s: str) -> str:
    """Convert a normalized task string to a filesystem-safe directory name."""
    return re.sub(r"[^a-z0-9_]+", "_", s).strip("_") or "unknown"


# ---------------------------------------------------------------------------
# Subset scanning
# ---------------------------------------------------------------------------
# Source dataset layout (per subset, e.g. allenai--01122025-box-01):
#     meta/info.json                       fps + feature schemas
#     meta/tasks.parquet                   coarse task string (1 row per subset)
#     meta/tasks_annotated.parquet         per-episode natural-language task
#     meta/episodes/chunk-*/file-*.parquet per-episode metadata (data/video pointers)
#     data/chunk-*/file-*.parquet          frame rows
#     videos/<view>/chunk-*/file-*.mp4     concatenated av1 mp4 (multi-episode)
#
# Only `top` is used for tracking; left/right are loaded for raw video / vae outputs.

def list_subsets(source_root: str) -> List[str]:
    if not os.path.isdir(source_root):
        raise FileNotFoundError(f"MolmoAct2 source root not found: {source_root}")
    out = sorted(
        n for n in os.listdir(source_root)
        if os.path.isdir(os.path.join(source_root, n)) and not n.startswith("_")
    )
    if not out:
        raise FileNotFoundError(f"No subsets found under {source_root}")
    return out


def _read_info(subset_dir: str) -> Dict:
    with open(os.path.join(subset_dir, "meta/info.json"), "r") as f:
        return json.load(f)


def _read_coarse_task(subset_dir: str) -> str:
    """tasks.parquet has one row per task_index; the task string is the row index."""
    import pandas as pd
    df = pd.read_parquet(os.path.join(subset_dir, "meta/tasks.parquet"))
    vals = list(df.index.values)
    if len(vals) == 0:
        raise ValueError(f"empty tasks.parquet in {subset_dir}")
    if len(vals) > 1:
        # Some subsets may have multiple coarse tasks (we haven't seen any in
        # MolmoAct2-YAM, but be explicit if it ever appears).
        raise ValueError(
            f"{subset_dir}: tasks.parquet has {len(vals)} rows; expected 1. "
            f"Per-task_index handling not implemented."
        )
    return str(vals[0])


def _read_annotated(subset_dir: str) -> Dict[int, str]:
    import pandas as pd
    df = pd.read_parquet(os.path.join(subset_dir, "meta/tasks_annotated.parquet"))
    out: Dict[int, str] = {}
    for ep_idx, row in df.iterrows():
        out[int(ep_idx)] = str(row["task"])
    return out


def _episode_meta_paths(subset_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(subset_dir, "meta/episodes/chunk-*/file-*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no meta/episodes/*.parquet in {subset_dir}")
    return paths


_EP_COLS_BASE = [
    "episode_index", "length",
    "data/chunk_index", "data/file_index",
    "dataset_from_index", "dataset_to_index",
]


def _ep_view_cols(view: str) -> List[str]:
    k = VIEW_KEY[view]
    return [
        f"videos/{k}/chunk_index",
        f"videos/{k}/file_index",
        f"videos/{k}/from_timestamp",
        f"videos/{k}/to_timestamp",
    ]


def iter_subset_episodes(subset_dir: str, subset_name: str) -> List[EpisodeRef]:
    info = _read_info(subset_dir)
    fps = int(info["fps"])
    coarse_raw = _read_coarse_task(subset_dir)
    coarse_norm = normalize_task(coarse_raw)
    annotated = _read_annotated(subset_dir)

    cols = list(_EP_COLS_BASE)
    for v in VIEWS:
        cols.extend(_ep_view_cols(v))

    out: List[EpisodeRef] = []
    for path in _episode_meta_paths(subset_dir):
        d = pq.read_table(path, columns=cols).to_pydict()
        n = len(d["episode_index"])
        for i in range(n):
            ep_idx = int(d["episode_index"][i])
            length = int(d["length"][i])
            views: Dict[str, ViewRef] = {}
            for v in VIEWS:
                k = VIEW_KEY[v]
                views[v] = ViewRef(
                    chunk=int(d[f"videos/{k}/chunk_index"][i]),
                    file=int(d[f"videos/{k}/file_index"][i]),
                    from_t=float(d[f"videos/{k}/from_timestamp"][i]),
                    to_t=float(d[f"videos/{k}/to_timestamp"][i]),
                )
            if ep_idx not in annotated:
                raise KeyError(
                    f"{subset_name} ep {ep_idx}: missing tasks_annotated row"
                )
            duration = length / float(fps)
            out.append(EpisodeRef(
                subset=subset_name,
                subset_dir=subset_dir,
                episode_index=ep_idx,
                fps=fps,
                length=length,
                duration_s=duration,
                task_coarse=coarse_raw,
                task_coarse_norm=coarse_norm,
                task_annotated=annotated[ep_idx],
                data_chunk=int(d["data/chunk_index"][i]),
                data_file=int(d["data/file_index"][i]),
                data_from_index=int(d["dataset_from_index"][i]),
                data_to_index=int(d["dataset_to_index"][i]),
                views=views,
            ))
    return out


# ---------------------------------------------------------------------------
# JSONL serialization (used by the index builder + preprocessor)
# ---------------------------------------------------------------------------

def episode_to_jsonl(ep: EpisodeRef) -> str:
    payload = {
        "subset": ep.subset,
        "subset_dir": ep.subset_dir,
        "episode_index": ep.episode_index,
        "fps": ep.fps,
        "length": ep.length,
        "duration_s": ep.duration_s,
        "task_coarse": ep.task_coarse,
        "task_coarse_norm": ep.task_coarse_norm,
        "task_annotated": ep.task_annotated,
        "data_chunk": ep.data_chunk,
        "data_file": ep.data_file,
        "data_from_index": ep.data_from_index,
        "data_to_index": ep.data_to_index,
        "views": {
            v: {"chunk": r.chunk, "file": r.file, "from_t": r.from_t, "to_t": r.to_t}
            for v, r in ep.views.items()
        },
    }
    return json.dumps(payload)


def episode_from_dict(d: Dict) -> EpisodeRef:
    return EpisodeRef(
        subset=str(d["subset"]),
        subset_dir=str(d["subset_dir"]),
        episode_index=int(d["episode_index"]),
        fps=int(d["fps"]),
        length=int(d["length"]),
        duration_s=float(d["duration_s"]),
        task_coarse=str(d["task_coarse"]),
        task_coarse_norm=str(d["task_coarse_norm"]),
        task_annotated=str(d["task_annotated"]),
        data_chunk=int(d["data_chunk"]),
        data_file=int(d["data_file"]),
        data_from_index=int(d["data_from_index"]),
        data_to_index=int(d["data_to_index"]),
        views={
            v: ViewRef(chunk=int(r["chunk"]), file=int(r["file"]),
                       from_t=float(r["from_t"]), to_t=float(r["to_t"]))
            for v, r in d["views"].items()
        },
    )


def read_jsonl(path: str) -> List[EpisodeRef]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"index file not found: {path}")
    out: List[EpisodeRef] = []
    with open(path, "r") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(episode_from_dict(json.loads(line)))
            except Exception as e:
                raise ValueError(f"{path}:{ln} parse error: {e}")
    return out


# ---------------------------------------------------------------------------
# Video decode (av1 via pyav)
# ---------------------------------------------------------------------------

def video_path_for(ep: EpisodeRef, view: str) -> str:
    if view not in ep.views:
        raise KeyError(f"view {view!r} not present in episode {ep.sample_id}")
    r = ep.views[view]
    return os.path.join(
        ep.subset_dir, "videos", VIEW_KEY[view],
        f"chunk-{r.chunk:03d}", f"file-{r.file:03d}.mp4",
    )


def crop_resize_4x3(img: np.ndarray, img_size: int) -> np.ndarray:
    """Center-crop a frame to 4:3, then resize (squash) to (img_size, img_size).

    The molmoact2 views are 16:9 (640x360). A shortest-side square crop would
    discard ~44% of the width, clipping the edges of the bimanual workspace.
    Cropping to 4:3 first keeps far more of the scene; the residual 4:3 -> 1:1
    squash distorts only mildly. LANCZOS resampling keeps molmoact2 frames on
    the same pixel grid Wan was trained on (PIL Lanczos).
    """
    if img_size <= 0:
        raise ValueError(f"img_size must be positive, got {img_size}")
    h, w = img.shape[:2]
    # Crop whichever dimension makes the frame depart from 4:3 (w/h == 4/3).
    if w * 3 > h * 4:        # wider than 4:3 -> crop width
        cw = (h * 4) // 3
        x1 = (w - cw) // 2
        img = img[:, x1:x1 + cw]
    elif w * 3 < h * 4:      # taller than 4:3 -> crop height
        ch = (w * 3) // 4
        y1 = (h - ch) // 2
        img = img[y1:y1 + ch, :]
    pil = Image.fromarray(img).resize((img_size, img_size), Image.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def decode_and_resize_episode_view(ep: EpisodeRef, view: str, img_size: int) -> np.ndarray:
    """Stream-decode the [from_t, to_t) span of one view and crop+resize each
    frame on the fly to (img_size, img_size) uint8 via crop_resize_4x3 (center
    -crop to 4:3, then squash). Never materializes the full-res decoded clip in
    memory — peak per call is ~length * img_size**2 * 3 bytes instead of
    ~length * 360 * 640 * 3 bytes (~13x reduction at 256/640).
    """
    r = ep.views[view]
    path = video_path_for(ep, view)

    container = av.open(path)
    try:
        stream = container.streams.video[0]
        # Seek backwards to the nearest keyframe before from_t, then drop frames
        # whose pts is below from_t (sub-keyframe accuracy).
        seek_pts = max(int(r.from_t / float(stream.time_base)) - 1, 0)
        container.seek(seek_pts, stream=stream, any_frame=False)

        frames: List[np.ndarray] = []
        # to_t is exclusive (= from_t of the next episode in the same mp4),
        # so the upper bound check must skip frames at pts == to_t.
        # Tolerance is half a frame interval to absorb float rounding only.
        tol = 0.5 / float(ep.fps)
        for frame in container.decode(stream):
            pts_s = float(frame.pts * stream.time_base)
            if pts_s < r.from_t - tol:
                continue
            if pts_s >= r.to_t - tol:
                break
            frames.append(crop_resize_4x3(frame.to_ndarray(format="rgb24"), img_size))
    finally:
        container.close()

    if len(frames) != ep.length:
        raise ValueError(
            f"{ep.sample_id} view={view}: decoded {len(frames)} frames, "
            f"expected {ep.length} (from_t={r.from_t}, to_t={r.to_t}, path={path})"
        )
    return np.stack(frames, axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Action / state reader (data/chunk-XXX/file-YYY.parquet)
# ---------------------------------------------------------------------------

def data_path_for(ep: EpisodeRef) -> str:
    return os.path.join(
        ep.subset_dir, "data",
        f"chunk-{ep.data_chunk:03d}",
        f"file-{ep.data_file:03d}.parquet",
    )


def read_episode_actions(ep: EpisodeRef) -> Dict[str, np.ndarray]:
    """Load per-frame `action` (T, 14) and `observation.state` (T, 14) for one
    episode. Filters the rows of the data parquet by `episode_index` (the
    `dataset_from_index`/`dataset_to_index` fields are GLOBAL `index` column
    values across all data parquets in the subset, not local row offsets).
    Validates that exactly `length` rows are read."""
    import pyarrow.compute as pc
    path = data_path_for(ep)
    table = pq.read_table(path, columns=["action", "observation.state",
                                         "frame_index", "episode_index"])
    sub = table.filter(pc.equal(table.column("episode_index"), ep.episode_index))
    if sub.num_rows != ep.length:
        raise ValueError(
            f"{ep.sample_id}: filtered {sub.num_rows} rows for episode_index "
            f"{ep.episode_index} in {path}, expected length={ep.length}"
        )
    # Each row's action / observation.state column is itself a list-of-floats.
    actions = np.asarray(sub.column("action").to_pylist(), dtype=np.float32)
    states = np.asarray(sub.column("observation.state").to_pylist(), dtype=np.float32)
    # Frames must be contiguous 0..T-1 — episodes always start at frame 0.
    fi = np.asarray(sub.column("frame_index").to_pylist(), dtype=np.int64)
    if not np.array_equal(fi, np.arange(ep.length)):
        raise ValueError(
            f"{ep.sample_id}: frame_index column is non-contiguous "
            f"(first 5: {fi[:5].tolist()}, last 5: {fi[-5:].tolist()})"
        )
    return {"actions": actions, "observation_state": states}


# ---------------------------------------------------------------------------
# Selection-index CLI: build / select / stats
# ---------------------------------------------------------------------------

def cmd_build(source_dir: str, index_dir: str, force: bool) -> None:
    out_path = os.path.join(index_dir, FULL_INDEX_NAME)
    if os.path.exists(out_path) and not force:
        raise FileExistsError(
            f"Full index already exists at {out_path}. Pass --force to rebuild."
        )

    subsets = list_subsets(source_dir)
    print(f"[build] scanning {len(subsets)} subsets under {source_dir}")
    os.makedirs(index_dir, exist_ok=True)

    tmp_path = out_path + ".tmp"
    n_eps = 0
    n_subsets_ok = 0
    failures: List[Tuple[str, str]] = []
    with open(tmp_path, "w") as f:
        for subset in tqdm(subsets, desc="subsets"):
            subset_dir = os.path.join(source_dir, subset)
            try:
                eps = iter_subset_episodes(subset_dir, subset)
            except Exception as e:
                failures.append((subset, repr(e)))
                continue
            for ep in eps:
                f.write(episode_to_jsonl(ep) + "\n")
                n_eps += 1
            n_subsets_ok += 1

    os.replace(tmp_path, out_path)
    total_h = _index_total_hours(out_path)
    print(f"[build] wrote {n_eps} episodes from {n_subsets_ok}/{len(subsets)} "
          f"subsets ({total_h:.1f}h) -> {out_path}")
    if failures:
        print(f"[build] {len(failures)} subsets failed:")
        for s, e in failures[:20]:
            print(f"  {s}: {e}")
        if len(failures) > 20:
            print(f"  ... +{len(failures) - 20} more")
        # Treat any failure as fatal — failing hard matches AGENTS.md.
        raise RuntimeError(f"{len(failures)} subsets failed during index build")


def _index_total_hours(path: str) -> float:
    total_s = 0.0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                total_s += float(json.loads(line)["duration_s"])
    return total_s / 3600.0


def _group_by_task(eps: List[EpisodeRef]) -> "OrderedDict[str, List[EpisodeRef]]":
    """Group episodes by normalized coarse task. Tasks ordered alphabetically
    so round-robin output is deterministic across machines."""
    buckets: Dict[str, List[EpisodeRef]] = defaultdict(list)
    for ep in eps:
        buckets[ep.task_coarse_norm].append(ep)
    return OrderedDict((k, buckets[k]) for k in sorted(buckets))


def _round_robin_select(
    eps: List[EpisodeRef], target_hours: float, seed: int,
) -> Tuple[List[EpisodeRef], Dict[str, int], float]:
    """Round-robin episodes by normalized coarse task, shuffled within each
    task, until cumulative duration >= target_hours.

    Returns: (selected episodes in pick order, per-task picked counts, hours).
    """
    if target_hours <= 0:
        raise ValueError(f"target_hours must be > 0, got {target_hours}")

    rng = random.Random(seed)
    buckets = _group_by_task(eps)
    if not buckets:
        raise ValueError("No episodes available for selection.")

    # Shuffle each bucket once, then iterate round-robin via per-task indices.
    for task in buckets:
        rng.shuffle(buckets[task])

    target_s = target_hours * 3600.0
    cum_s = 0.0
    picked: List[EpisodeRef] = []
    counts: Dict[str, int] = defaultdict(int)
    pos: Dict[str, int] = {t: 0 for t in buckets}
    task_order = list(buckets.keys())
    while cum_s < target_s:
        progressed = False
        for task in task_order:
            if pos[task] >= len(buckets[task]):
                continue
            ep = buckets[task][pos[task]]
            pos[task] += 1
            picked.append(ep)
            counts[task] += 1
            cum_s += ep.duration_s
            progressed = True
            if cum_s >= target_s:
                break
        if not progressed:
            # We've drained every bucket and still didn't hit target.
            break

    return picked, dict(counts), cum_s / 3600.0


def _normalize_and_check(items: List[str], available: set, label: str) -> List[str]:
    norm = [normalize_task(t) for t in items]
    missing = [t for t in norm if t not in available]
    if missing:
        raise ValueError(
            f"{label} tasks not in index: {missing}. "
            f"Available (sample): {sorted(available)[:10]} ..."
        )
    return norm


def cmd_select(
    index_dir: str,
    full_index_name: str,
    name: str,
    hours: float,
    seed: int,
    tasks_arg: List[str],
    exclude_tasks_arg: List[str],
    force: bool,
) -> None:
    if tasks_arg and exclude_tasks_arg:
        raise ValueError(
            "Pass either --tasks or --exclude-tasks, not both."
        )

    full_path = os.path.join(index_dir, full_index_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Full index not found at {full_path}. Run `... build` first."
        )

    out_dir = os.path.join(index_dir, name)
    out_index = os.path.join(out_dir, "index.jsonl")
    out_manifest = os.path.join(out_dir, "manifest.json")
    if os.path.exists(out_index) and not force:
        raise FileExistsError(
            f"Selection {out_index} already exists. Pass --force to overwrite."
        )

    eps = read_jsonl(full_path)
    print(f"[select] loaded {len(eps)} episodes from {full_path} "
          f"({sum(e.duration_s for e in eps)/3600:.1f}h total)")

    available = {ep.task_coarse_norm for ep in eps}
    included_norm: List[str] = []
    excluded_norm: List[str] = []

    if tasks_arg:
        included_norm = _normalize_and_check(tasks_arg, available, "Requested")
        keep = set(included_norm)
    elif exclude_tasks_arg:
        excluded_norm = _normalize_and_check(exclude_tasks_arg, available, "Excluded")
        keep = available - set(excluded_norm)
        if not keep:
            raise ValueError("After --exclude-tasks, no tasks remain.")
    else:
        keep = available

    candidates = [ep for ep in eps if ep.task_coarse_norm in keep]
    print(f"[select] candidate pool: {len(candidates)} eps "
          f"({sum(e.duration_s for e in candidates)/3600:.1f}h) across "
          f"{len(keep)} task(s) "
          f"(included={len(included_norm) or 'all'}, excluded={len(excluded_norm)})")

    picked, counts, achieved = _round_robin_select(candidates, hours, seed)
    if achieved < hours:
        # Fail hard — the caller should know the requested subset is too small.
        raise ValueError(
            f"Only {achieved:.2f}h available (target {hours}h). "
            f"Picked {len(picked)} eps from {len(counts)} task(s)."
        )

    os.makedirs(out_dir, exist_ok=True)
    tmp = out_index + ".tmp"
    with open(tmp, "w") as f:
        for ep in picked:
            f.write(episode_to_jsonl(ep) + "\n")
    os.replace(tmp, out_index)

    manifest = {
        "name": name,
        "target_hours": hours,
        "achieved_hours": achieved,
        "n_clips": len(picked),
        "seed": seed,
        "source_full_index": os.path.relpath(full_path, out_dir),
        "included_tasks_norm": included_norm,
        "excluded_tasks_norm": excluded_norm,
        "per_task_clip_counts": dict(sorted(counts.items())),
    }
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"[select] wrote {len(picked)} clips ({achieved:.2f}h) -> {out_index}")
    print(f"[select] manifest -> {out_manifest}")
    print(f"[select] task coverage: {len(counts)} / {len(keep)} tasks picked from")


def cmd_stats(index_dir: str, name: str | None) -> None:
    if name is None:
        path = os.path.join(index_dir, FULL_INDEX_NAME)
        label = "FULL INDEX"
    else:
        path = os.path.join(index_dir, name, "index.jsonl")
        label = f"SELECTION '{name}'"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"index not found: {path}")

    rows: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    n_lines = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                rows[d["task_coarse_norm"]].append(
                    (int(d["length"]), float(d["duration_s"]))
                )
                n_lines += 1

    table = []
    for task, eps in rows.items():
        lens = np.array([L for L, _ in eps])
        durs = np.array([d for _, d in eps])
        table.append({
            "task": task,
            "hours": float(durs.sum() / 3600.0), "eps": len(eps),
            "med_len": int(np.median(lens)), "mean_s": float(durs.mean()),
            "min_s": float(durs.min()), "max_s": float(durs.max()),
        })
    table.sort(key=lambda r: -r["hours"])

    header = (
        f'{"task (normalized)":<60} {"hours":>7} {"#eps":>6} '
        f'{"med_len":>7} {"mean_s":>7} {"min_s":>6} {"max_s":>6}'
    )
    print(f"\n{label}: {path}")
    print(header)
    print("-" * len(header))
    for r in table:
        print(f'{r["task"]:<60} {r["hours"]:>6.2f}h {r["eps"]:>6d} '
              f'{r["med_len"]:>7d} {r["mean_s"]:>6.1f}s {r["min_s"]:>5.1f}s {r["max_s"]:>5.1f}s')
    print("-" * len(header))
    total_h = sum(r["hours"] for r in table)
    print(f"TOTAL: {len(table)} tasks, {n_lines} eps, {total_h:.1f}h")


def _split_tasks(s: str) -> List[str]:
    if s is None:
        return []
    return [t for t in (x.strip() for x in s.split(";")) if t]


def index_main(argv: List[str] | None = None) -> None:
    """Build the MolmoAct2-BimanualYAM episode index and select 10-hour subsets.

    Three subcommands:

      build   Walk every subset under <source_dir> and emit one JSONL row per
              episode at <index_dir>/episodes_full.jsonl. Idempotent unless --force.

      select  Read the full index and write a selection at
              <index_dir>/<name>/{index.jsonl, manifest.json}. Always round-robins
              across normalized coarse tasks (shuffled within each) until cumulative
              duration >= --hours. Optional task filter (mutually exclusive):

                --tasks "task_a;task_b;..."
                    Restrict to ONLY these normalized coarse tasks.

                --exclude-tasks "task_a;task_b;..."
                    Drop these tasks from the full set; round-robin the rest.

              Omit both to round-robin across the entire dataset.

              Train/val split is NOT applied here — it's a preprocessing-time
              decision (see `cfg.split` / `cfg.val_ratio` in this file, mirroring
              the egoverse pattern: a stable sha1(seed:sample_id) hash assigns
              each clip to train or val, so rebuilding selections doesn't
              reshuffle splits.

      stats   Print a per-coarse-task breakdown: hours, episode count, median
              frame length, mean/min/max episode duration. Defaults to the full
              index; pass --name <sel> to inspect a saved selection.

    Examples:
      python -m preprocessing.preprocess_molmoact2 index --index-dir /idx build --source-dir /raw
      python -m preprocessing.preprocess_molmoact2 index --index-dir /idx select --name diverse_10h --hours 10
      python -m preprocessing.preprocess_molmoact2 index --index-dir /idx select --name boxes_only --hours 10 \\
          --tasks "Pack container"
      python -m preprocessing.preprocess_molmoact2 index --index-dir /idx select --name no_spellout_10h --hours 10 \\
          --exclude-tasks "spell out ai2;spell out neurips;spell out robot"
      python -m preprocessing.preprocess_molmoact2 index --index-dir /idx stats
      python -m preprocessing.preprocess_molmoact2 index --index-dir /idx stats --name diverse_10h
    """
    ap = argparse.ArgumentParser(description=index_main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", required=True, help="where to write/read the index")

    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="scan source_dir and write the full episode index")
    p_build.add_argument("--force", action="store_true",
                         help="overwrite existing full index file")
    p_build.add_argument("--source-dir", required=True, help="raw dataset root")

    p_sel = sub.add_parser(
        "select",
        help="round-robin N hours of clips from the full index, optionally filtered",
    )
    p_sel.add_argument("--name", required=True,
                       help="selection name (directory under index_dir)")
    p_sel.add_argument("--hours", type=float, default=10.0,
                       help="target cumulative duration in hours (default: 10)")
    p_sel.add_argument("--seed", type=int, default=0,
                       help="seed for within-task shuffling (default: 0)")
    p_sel.add_argument("--tasks", default="",
                       help="';'-separated coarse task strings to INCLUDE (matched after "
                            "normalization). Mutually exclusive with --exclude-tasks.")
    p_sel.add_argument("--exclude-tasks", default="",
                       help="';'-separated coarse task strings to EXCLUDE. "
                            "Mutually exclusive with --tasks.")
    p_sel.add_argument("--force", action="store_true",
                       help="overwrite existing selection")

    p_stats = sub.add_parser("stats", help="print per-coarse-task breakdown")
    p_stats.add_argument("--name", default=None,
                         help="selection name to inspect (default: full index)")

    args = ap.parse_args(argv)

    if args.cmd == "build":
        cmd_build(args.source_dir, args.index_dir, args.force)
        return
    if args.cmd == "select":
        cmd_select(
            index_dir=args.index_dir,
            full_index_name=FULL_INDEX_NAME,
            name=args.name,
            hours=args.hours,
            seed=args.seed,
            tasks_arg=_split_tasks(args.tasks),
            exclude_tasks_arg=_split_tasks(args.exclude_tasks),
            force=args.force,
        )
        return
    if args.cmd == "stats":
        cmd_stats(index_dir=args.index_dir, name=args.name)
        return
    raise SystemExit(f"unknown subcommand {args.cmd!r}")


# ---------------------------------------------------------------------------
# Completion markers
# ---------------------------------------------------------------------------
# A clip's HDF5 is written in place, and a job preempted mid-write can leave a
# partial file that still passes a structural (group-exists) skip check. To make
# skip-exist trustworthy we write a sidecar `<path>.done` marker — but only
# AFTER the HDF5 has been flushed to disk (see `_DoneMarkingProcessor`). The
# marker therefore exists iff the clip was written completely; a preempted clip
# has no marker and is reprocessed.

def _done_marker(path: str) -> str:
    return f"{path}.done"


def _is_done(path: str) -> bool:
    """True iff the HDF5 and its `.done` marker both exist — i.e. a previous run
    finished writing this clip."""
    return os.path.exists(path) and os.path.exists(_done_marker(path))


# ---------------------------------------------------------------------------
# processors
# ---------------------------------------------------------------------------

class _VideoProcessor(Processor):
    """Write THWC uint8 frames per view at root/<view>/video. Frames have
    already been LANCZOS-resized + center-cropped in to_sample()."""

    def process(self, out_h5_dict: Dict[str, h5py.File], sample: Sample) -> None:
        out_h5 = self._get_single_file(out_h5_dict)
        root = out_h5.require_group("root")
        for view, video in sample.videos.items():
            vg = root.require_group(view)
            if "video" in vg:
                del vg["video"]
            vg.create_dataset("video", data=np.asarray(video, dtype=np.uint8),
                              dtype="uint8")


class _TextProcessor(Processor):
    """Write the per-episode annotated language string at top-level `text`."""

    def process(self, out_h5_dict: Dict[str, h5py.File], sample: Sample) -> None:
        if sample.text is None:
            raise ValueError(f"text mode requires sample.text; got None for {sample.id}")
        out_h5 = self._get_single_file(out_h5_dict)
        dt = h5py.string_dtype(encoding="utf-8")
        if "text" in out_h5:
            del out_h5["text"]
        out_h5.create_dataset("text", data=np.array(sample.text, dtype=dt), dtype=dt)


class _ActionProcessor(Processor):
    """Read per-frame action + observation.state from the data parquet and write
    top-level `actions` (T, 14) fp32, `observation_state` (T, 14) fp32, and
    `frame_indices` (T,) int32 = arange(T)."""

    def process(self, out_h5_dict: Dict[str, h5py.File], sample: Sample) -> None:
        out_h5 = self._get_single_file(out_h5_dict)
        if sample.actions is None:
            raise ValueError(f"actions mode requires sample.actions; got None for {sample.id}")
        actions = np.asarray(sample.actions, dtype=np.float32)
        states = np.asarray(sample.meta["observation_state"], dtype=np.float32)
        T = actions.shape[0]
        for key, arr in (("actions", actions), ("observation_state", states)):
            if key in out_h5:
                del out_h5[key]
            out_h5.create_dataset(key, data=arr, dtype="float32")
        if "frame_indices" in out_h5:
            del out_h5["frame_indices"]
        out_h5.create_dataset("frame_indices", data=np.arange(T, dtype=np.int32), dtype="int32")


class _DoneMarkingProcessor(Processor):
    """Wraps any Processor: after the inner processor writes, flush each output
    file to disk and drop a sidecar `<path>.done` marker.

    `flush()` makes the on-disk HDF5 self-consistent BEFORE the marker is
    created, so the marker can never race ahead of the data. A job preempted
    mid-write leaves the HDF5 with no marker, so `_is_done` reports it as
    incomplete and it gets reprocessed. Used for every mode (incl. the shared
    TrackProcessor) so skip-exist no longer trusts a structural check alone."""

    def __init__(self, inner: Processor):
        self.inner = inner

    def process(self, out_h5_dict: Dict[str, h5py.File], sample: Sample) -> None:
        self.inner.process(out_h5_dict, sample)
        for h5_file in out_h5_dict.values():
            if h5_file is None:        # file was already complete -> skipped
                continue
            h5_file.flush()
            with open(_done_marker(h5_file.filename), "w"):
                pass


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class PreprocessMolmoAct2(PreprocessDataset):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        if cfg.mode not in VALID_MODES:
            raise ValueError(f"MolmoAct2 supports modes={VALID_MODES}; got '{cfg.mode}'.")
        if cfg.img_size <= 0:
            raise ValueError(f"img_size must be > 0, got {cfg.img_size}")
        if cfg.max_clips is not None and int(cfg.max_clips) <= 0:
            raise ValueError(f"max_clips must be > 0 or null, got {cfg.max_clips}")
        if cfg.max_frames is not None and int(cfg.max_frames) <= 0:
            raise ValueError(f"max_frames must be > 0 or null, got {cfg.max_frames}")

        self.dest_root = os.path.abspath(cfg.dest_dir)
        self.output_root = os.path.join(self.dest_root, cfg.output_name)

        index_path = os.path.abspath(cfg.index_path)
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"Selection index not found: {index_path}. "
                f"Build it with `python -m preprocessing.preprocess_molmoact2 index select ...`"
            )
        eps = read_jsonl(index_path)
        # Egoverse-style hash split: filter the selection by `cfg.split` using
        # a stable sha1(seed:sample_id) bucket BEFORE sharding, so each
        # (num_workers, worker_id) slice is deterministic across train/val runs.
        self.split = str(cfg.split)
        eps = [ep for ep in eps
               if in_split(ep.sample_id, self.split, float(cfg.val_ratio), int(cfg.split_seed))]

        if cfg.mode == "tracks":
            self.camera_names = [TRACKS_VIEW]
        else:
            self.camera_names = list(cfg.camera_names)
        for v in self.camera_names:
            if v not in VIEWS:
                raise ValueError(
                    f"Invalid camera view '{v}'. Must be one of {list(VIEWS)}"
                )

        ids = split_indices(len(eps), int(cfg.num_workers), int(cfg.worker_id))
        self.episodes: List[EpisodeRef] = [eps[i] for i in ids]
        if cfg.max_clips is not None:
            self.episodes = self.episodes[: int(cfg.max_clips)]

        if not self.episodes:
            raise ValueError(
                f"No clips for split={self.split} worker {cfg.worker_id}/{cfg.num_workers} in {index_path}"
            )

        print(f"[Init] index={index_path}")
        print(f"[Init] mode={cfg.mode} split={self.split} views={self.camera_names}")
        print(f"[Init] output_root={self.output_root}")
        print(f"[Init] split={self.split} clips={len(eps)}; "
              f"worker {cfg.worker_id}/{cfg.num_workers} -> {len(self.episodes)} clips"
              f"{' (capped by max_clips)' if cfg.max_clips is not None else ''}")

    @staticmethod
    def _prep_model(model, device):
        return model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    def build_models(self, cfg: DictConfig) -> Dict[str, Any]:
        models: Dict[str, Any] = {}
        if cfg.mode == "tracks":
            load_tracker(cfg, models)
            return models
        if cfg.mode in {"video", "text", "actions"}:
            return models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models["device"] = device
        if cfg.mode == "text_emb":
            from transformers import AutoTokenizer
            from groot.vla.model.dreamzero.modules.wan_video_text_encoder import WanTextEncoder
            text_encoder = WanTextEncoder()
            text_encoder.load_state_dict(torch.load(cfg.text_encoder_weights, map_location="cpu"))
            models["text_encoder"] = self._prep_model(text_encoder, device)
            models["tokenizer"] = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
        elif cfg.mode in {"vae_latent", "vae_latent_h16"}:
            from groot.vla.model.dreamzero.modules.wan_video_vae import WanVideoVAE38
            vae = WanVideoVAE38(vae_pretrained_path=cfg.vae_weights)
            vae.model.load_state_dict(torch.load(cfg.vae_weights, map_location="cpu", weights_only=True))
            models["vae"] = self._prep_model(vae, device)
        return models

    def build_processors(self, cfg: DictConfig, models: Dict[str, Any]) -> Dict[str, Processor]:
        # Wrap every processor so each fully-written clip gets a `.done` marker.
        raw = self._build_raw_processors(cfg, models)
        return {k: _DoneMarkingProcessor(p) for k, p in raw.items()}

    def _build_raw_processors(self, cfg: DictConfig, models: Dict[str, Any]) -> Dict[str, Processor]:
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
        if cfg.mode == "video":
            return {"video": _VideoProcessor()}
        if cfg.mode == "text":
            return {"text": _TextProcessor()}
        if cfg.mode == "text_emb":
            return {"text_emb": WanTextEmbProcessor(
                text_encoder=models["text_encoder"],
                tokenizer=models["tokenizer"],
                device=models["device"],
            )}
        if cfg.mode == "vae_latent":
            return {"vae_latent": WanVaeLatentProcessor(
                vae=models["vae"],
                device=models["device"],
                image_hw=cfg.wan.image_hw,
            )}
        if cfg.mode == "vae_latent_h16":
            return {"vae_latent_h16": WanVaeLatentH16Processor(
                vae=models["vae"],
                device=models["device"],
                image_hw=cfg.wan.image_hw,
                horizon=cfg.horizon,
            )}
        if cfg.mode == "actions":
            return {"actions": _ActionProcessor()}
        raise ValueError(f"Unknown mode: {cfg.mode}")

    def iter_items(self, cfg: DictConfig) -> Iterable[EpisodeRef]:
        rows = tqdm(self.episodes, desc="MolmoAct2 clips", leave=False) if cfg.progress else self.episodes
        yield from rows

    def to_sample(self, ep: EpisodeRef, cfg: DictConfig) -> Sample:
        # Skip-exist via `.done` markers (see `_is_done`). Done BEFORE the
        # expensive decode below — run_dataset only skip-checks after
        # to_sample(), so an already-done clip would otherwise pay a full
        # mp4 decode just to be discarded.
        if getattr(cfg, "skip_exist", True):
            paths = self._paths_for(ep.sample_id, ep.task_coarse_norm, cfg)
            if all(_is_done(p) for p in paths.values()):
                return None
            # A path without its `.done` marker is a stale partial write (a
            # prior run was preempted mid-write). Delete it so the shared
            # inital_save_h5 rewrites it from scratch instead of skipping it on
            # its weaker structural (group-exists) check.
            for p in paths.values():
                if not _is_done(p) and os.path.exists(p):
                    os.remove(p)

        meta = {
            "subset": ep.subset,
            "episode_index": np.int32(ep.episode_index),
            "fps": np.int32(ep.fps),
            "length": np.int32(ep.length),
            "task_coarse": ep.task_coarse,
            "task_coarse_norm": ep.task_coarse_norm,
            "task_annotated": ep.task_annotated,
            "task_safe": task_safe_dirname(ep.task_coarse_norm),
            "split": self.split,
        }

        if cfg.mode == "actions":
            data = read_episode_actions(ep)
            meta["observation_state"] = data["observation_state"]
            return Sample(id=ep.sample_id, videos={}, actions=data["actions"], meta=meta)

        if cfg.mode in {"text", "text_emb"}:
            return Sample(id=ep.sample_id, videos={}, text=ep.task_annotated, meta=meta)

        # Only `video` decodes raw mp4s; the video-consuming modes (tracks /
        # vae_latent / vae_latent_h16) read the already-preprocessed
        # `video` store so the decode+resize work happens exactly once.
        if cfg.mode == "video":
            target_size = int(cfg.img_size)
            videos: Dict[str, np.ndarray] = {}
            for view in self.camera_names:
                videos[view] = decode_and_resize_episode_view(ep, view, target_size)
        else:
            videos = self._load_preprocessed_video(ep, cfg)

        # Testing knob: truncate every view to the first `max_frames` frames so
        # the expensive tracking/encoding pass runs on a short clip.
        if cfg.max_frames is not None:
            n = int(cfg.max_frames)
            videos = {v: vid[:n] for v, vid in videos.items()}

        return Sample(id=ep.sample_id, videos=videos, text=ep.task_annotated, meta=meta)

    def _common_params_folder(self, cfg: DictConfig) -> str:
        if cfg.mode == "tracks":
            reinit_str = f"_reinit_{cfg.horizon}" if cfg.reinit else ""
            return f"{cfg.init_queries}_{cfg.n_tracks}{reinit_str}"
        return cfg.mode

    def _video_input_path(self, ep: EpisodeRef, cfg: DictConfig) -> str:
        """Path to the preprocessed `video` HDF5 for one clip — the input that
        tracks / vae_latent / vae_latent_h16 read instead of decoding
        the raw mp4. Produced by a prior `mode=video` run; `source_output_name`
        defaults to `output_name` but can point at a video store elsewhere."""
        src_root = os.path.join(
            self.dest_root, str(cfg.source_output_name), self.split, "video",
        )
        task_safe = task_safe_dirname(ep.task_coarse_norm)
        return os.path.join(src_root, task_safe, f"{ep.sample_id}.hdf5")

    def _load_preprocessed_video(
        self, ep: EpisodeRef, cfg: DictConfig,
    ) -> Dict[str, np.ndarray]:
        """Read root/<view>/video for each needed view from the preprocessed
        `video` store. Fails hard if the store or a view is missing."""
        src = self._video_input_path(ep, cfg)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Missing preprocessed video store for {ep.sample_id}: {src}. "
                f"Run `mode=video` first (it must cover camera_names="
                f"{self.camera_names} at img_size={cfg.img_size})."
            )
        videos: Dict[str, np.ndarray] = {}
        with h5py.File(src, "r") as f:
            for view in self.camera_names:
                key = f"root/{view}/video"
                if key not in f:
                    raise KeyError(
                        f"{src} has no dataset '{key}'. Re-run `mode=video` with "
                        f"'{view}' in camera_names."
                    )
                videos[view] = f[key][:]
        return videos

    def _paths_for(self, sample_id: str, task_coarse_norm: str,
                   cfg: DictConfig) -> Dict[str, str]:
        """Output path(s) for one clip, keyed by output type. Pure function of
        its args + cfg — used both for writing and for the early skip check."""
        common = self._common_params_folder(cfg)
        # <output_root>/<split>/<common> is the per-mode root; tracks fan out
        # one more level by output_type (3d/depth/pose/2d).
        split_root = os.path.join(self.output_root, self.split, common)
        task_safe = task_safe_dirname(task_coarse_norm)

        if cfg.mode == "tracks":
            paths: Dict[str, str] = {}
            for output_type in cfg.track_outputs:
                sub = _get_output_subfolder(cfg, output_type)
                d = os.path.join(split_root, sub, task_safe)
                paths[output_type] = os.path.join(d, f"{sample_id}.hdf5")
            return paths

        d = os.path.join(split_root, task_safe)
        return {cfg.mode: os.path.join(d, f"{sample_id}.hdf5")}

    def output_paths(self, sample: Sample, cfg: DictConfig) -> Dict[str, str]:
        return self._paths_for(
            sample.id, sample.meta["task_coarse_norm"], cfg,
        )

    def on_end(self, cfg: DictConfig, models: Dict[str, Any]) -> None:
        pass


@hydra.main(config_path="../cfg/preprocessing", config_name="preprocess_molmoact2", version_base="1.2")
def main(cfg: DictConfig) -> None:
    validate_preprocess_config("molmoact2", cfg)

    # Include split so config.yaml lives next to the actual data, matching
    # egoverse: <output_root>/<split>/<common>/config.yaml
    base_dir = os.path.join(os.path.abspath(cfg.dest_dir), cfg.output_name, cfg.split)

    if cfg.mode == "tracks":
        save_tracks_config(cfg, base_dir)
    else:
        save_dir = os.path.join(base_dir, cfg.mode)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(save_dir, "config.yaml"))

    dataset = PreprocessMolmoAct2(cfg)
    run_dataset(dataset, cfg)


if __name__ == "__main__":
    # `index build|select|stats ...` runs the selection-index CLI; any other
    # invocation is the ordinary Hydra preprocessing entry point.
    if len(sys.argv) > 1 and sys.argv[1] == "index":
        index_main(sys.argv[2:])
    else:
        main()
