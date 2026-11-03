"""
EgoVerse preprocessing pipeline following the PreprocessDataset API.

Outputs are zarr stores (the only format EgoVerseDataset reads), split by
modality under:
    <dest_dir>/<output_name>/<split>/
        video/<task>/<prefix>/<sample_id>.zarr
        text/<task>/<prefix>/<sample_id>.zarr
        segment_text_emb/<task>/<prefix>/<sample_id>.zarr   (mode=text_emb)
        meta/<task>/<prefix>/<sample_id>.zarr
        vae_latent_h16/<task>/<prefix>/<sample_id>.zarr
        <track_method>/<tracker>_<output>/<task>/<prefix>/<sample_id>.zarr

`tracks` and `vae_latent_h16` read the `video` stores; `text_emb` reads the
`text` stores.

Episode discovery walks the raw EgoVerse zarr episode directories (with a
locked, content-hashed JSON cache so repeated runs don't rescan), and video /
camera reads decode the raw Zarr v3 shards directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import h5py
import hydra
import numpy as np
import torch
import zarr
from numcodecs import Blosc
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from flomo.utils.config_validation import PREPROCESS_MODES, validate_preprocess_config
from flomo.utils.wan_encoding import encode_wan_text
from preprocessing.preprocess_base import (
    PreprocessDataset,
    Processor,
    Sample,
    TrackProcessor,
    WanVaeLatentH16Processor,
    run_dataset,
)
from preprocessing.preprocessing_utils import (
    _get_output_subfolder,
    _get_tracks_common_dir,
    get_track_processor_func,
    in_split,
    load_tracker,
    save_tracks_config,
    split_indices,
)


VIEW = "default_view"
VALID_MODES = PREPROCESS_MODES["egoverse"]
# Store dir per non-track mode. text_emb's dir must match
# EgoVerseDataset._derive_segment_text_emb_path (/text/ -> /segment_text_emb/).
MODE_DIRS = {
    "video": "video",
    "text": "text",
    "text_emb": "segment_text_emb",
    "meta": "meta",
    "vae_latent_h16": "vae_latent_h16",
}

MECKA_INTRINSICS_360_640 = np.array(
    [
        [250.823578428, 0.0, 320.608314256],
        [0.0, 251.000532666, 184.415298569],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

_MISSING_U64 = np.uint64((1 << 64) - 1)


# ---------------------------------------------------------------------------
# Episode scan + cache
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EgoVerseEpisode:
    source_name: str
    source_root: str
    task_name: str
    episode_id: str
    sample_id: str
    episode_dir: str
    frame_count: int
    fps: int
    frame_height: int
    frame_width: int
    text: str
    meta: Dict[str, Any]
    image_backend: str


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _jsonable(val: Any) -> Any:
    if isinstance(val, dict):
        return {str(k): _jsonable(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_jsonable(v) for v in val]
    if isinstance(val, tuple):
        return [_jsonable(v) for v in val]
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _resolve_source(src: Dict[str, Any]) -> Dict[str, str]:
    if "name" not in src:
        raise KeyError(f"Source entry missing 'name': {src}")
    if "root" not in src:
        raise KeyError(f"Source entry missing 'root': {src}")
    name = str(src["name"])
    root = os.path.abspath(os.path.expanduser(str(src["root"])))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"EgoVerse source root does not exist: {root}")
    return {"name": name, "root": root}


def _is_episode_dir(ep_dir: Path) -> bool:
    return (ep_dir / "images.front_1").is_dir()


def _scan_source(src: Dict[str, str]) -> List[tuple[str, Path]]:
    root = Path(src["root"])
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not dirs:
        raise ValueError(f"EgoVerse source root has no child directories: {root}")

    items: List[tuple[str, Path]] = []
    for node in dirs:
        if _is_episode_dir(node):
            items.append((src["name"], node))
            continue
        for ep_dir in sorted([p for p in node.iterdir() if p.is_dir() and _is_episode_dir(p)]):
            items.append((node.name, ep_dir))
    if not items:
        raise ValueError(f"No EgoVerse episodes found under source root: {root}")
    return items


def _pick_text(task_name: str, root_attr: Dict[str, Any]) -> str:
    task_desc = str(root_attr.get("task_description", "")).strip()
    task_meta = str(root_attr.get("task_name", "")).strip()
    if task_desc:
        return task_desc
    if task_meta and task_meta.lower() != "debug":
        return task_meta
    return task_name


def _image_meta_from_root(root_attr: Dict[str, Any]) -> tuple[int, int, int]:
    shp = root_attr["features"]["images.front_1"]["shape"]
    if not isinstance(shp, list) or len(shp) != 3:
        raise ValueError(f"Unexpected image feature shape: {shp}")
    h = int(shp[0])
    w = int(shp[1])
    c = int(shp[2])
    if c != 3:
        raise ValueError(f"Expected 3 RGB channels, got {c}")
    return h, w, c


def _local_frame_count(ep_dir: Path) -> int:
    chunk_dir = ep_dir / "images.front_1" / "c"
    if not chunk_dir.is_dir():
        raise FileNotFoundError(f"Missing image chunk dir: {chunk_dir}")
    count = len([p for p in chunk_dir.iterdir() if p.is_file()])
    if count == 0:
        raise ValueError(f"No image chunks found in {chunk_dir}")
    return count


def _build_episode(
    source_name: str,
    source_root: str,
    task_name: str,
    ep_dir: Path,
    default_fps: int,
) -> EgoVerseEpisode:
    root_meta_path = ep_dir / "zarr.json"
    root_attr: Dict[str, Any] = {}
    if root_meta_path.is_file():
        root_meta = _load_json(root_meta_path)
        root_attr = dict(root_meta.get("attributes", {}))

    task = str(root_attr.get("task_name") or task_name).strip() or task_name
    ep_id = str(root_attr.get("episode_id") or ep_dir.name)
    sample_id = f"{source_name}_{ep_id}"

    img_meta_path = ep_dir / "images.front_1" / "zarr.json"
    if root_meta_path.is_file() and img_meta_path.is_file():
        n_frames = int(root_attr["total_frames"])
        h, w, c = _image_meta_from_root(root_attr)
        img_backend = "zarr"
    else:
        n_frames = _local_frame_count(ep_dir)
        h = -1
        w = -1
        img_backend = "local_chunks"

    anns_path = ep_dir / "annotations" / "zarr.json"
    ann_count = 0
    if anns_path.is_file():
        ann_meta = _load_json(anns_path)
        shp = ann_meta["shape"]
        if shp:
            ann_count = int(shp[0])

    txt = _pick_text(task, root_attr)
    fps = int(root_attr.get("fps", default_fps))
    meta = {
        "source_name": source_name,
        "source_root": source_root,
        "task_name": task,
        "episode_id": ep_id,
        "sample_id": sample_id,
        "episode_dir": str(ep_dir),
        "frame_count": np.int32(n_frames),
        "fps": np.int32(fps),
        "annotation_count": np.int32(ann_count),
        "image_backend": img_backend,
        "embodiment": str(root_attr.get("embodiment", "")),
        "task_description": str(root_attr.get("task_description", "")),
    }
    return EgoVerseEpisode(
        source_name=source_name,
        source_root=source_root,
        task_name=task,
        episode_id=ep_id,
        sample_id=sample_id,
        episode_dir=str(ep_dir),
        frame_count=n_frames,
        fps=fps,
        frame_height=h,
        frame_width=w,
        text=txt,
        meta=meta,
        image_backend=img_backend,
    )


def _episode_to_dict(ep: EgoVerseEpisode) -> Dict[str, Any]:
    return {
        "source_name": ep.source_name,
        "source_root": ep.source_root,
        "task_name": ep.task_name,
        "episode_id": ep.episode_id,
        "sample_id": ep.sample_id,
        "episode_dir": ep.episode_dir,
        "frame_count": ep.frame_count,
        "fps": ep.fps,
        "frame_height": ep.frame_height,
        "frame_width": ep.frame_width,
        "text": ep.text,
        "meta": _jsonable(ep.meta),
        "image_backend": ep.image_backend,
    }


def _episode_from_dict(row: Dict[str, Any]) -> EgoVerseEpisode:
    return EgoVerseEpisode(
        source_name=str(row["source_name"]),
        source_root=str(row["source_root"]),
        task_name=str(row["task_name"]),
        episode_id=str(row["episode_id"]),
        sample_id=str(row["sample_id"]),
        episode_dir=str(row["episode_dir"]),
        frame_count=int(row["frame_count"]),
        fps=int(row["fps"]),
        frame_height=int(row["frame_height"]),
        frame_width=int(row["frame_width"]),
        text=str(row["text"]),
        meta=dict(row["meta"]),
        image_backend=str(row["image_backend"]),
    )


def _episode_cache_path(sources: Iterable[Dict[str, str]], default_fps: int) -> Path:
    key_rows = [{"name": src["name"], "root": src["root"]} for src in sources]
    payload = json.dumps({"sources": key_rows, "default_fps": default_fps}, sort_keys=True)
    dig = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    env_root = os.environ.get("EGOVERSE_CACHE_ROOT", "").strip()
    if env_root:
        cache_root = Path(env_root).expanduser().resolve()
    else:
        cache_root = Path(key_rows[0]["root"]).resolve().parent / "_egoverse_cache"
    return cache_root / "egoverse_episode_index" / f"{dig}.json"


def _write_episode_cache(path: Path, rows: List[EgoVerseEpisode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump([_episode_to_dict(row) for row in rows], f)
    os.replace(tmp, path)


def _read_episode_cache(path: Path) -> List[EgoVerseEpisode]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"EgoVerse cache must contain a list, got {type(data)}")
    return [_episode_from_dict(row) for row in data]


def _build_episode_index(sources: Iterable[Dict[str, str]], default_fps: int) -> List[EgoVerseEpisode]:
    rows: List[EgoVerseEpisode] = []
    for src in sources:
        for task_name, ep_dir in _scan_source(src):
            rows.append(
                _build_episode(
                    source_name=src["name"],
                    source_root=src["root"],
                    task_name=task_name,
                    ep_dir=ep_dir,
                    default_fps=default_fps,
                )
            )
    rows = sorted(rows, key=lambda x: (x.task_name, x.sample_id))
    if not rows:
        raise ValueError("No EgoVerse episodes found while building cache.")
    return rows


def _load_episode_index(sources: Iterable[Dict[str, str]], default_fps: int) -> List[EgoVerseEpisode]:
    src_rows = [dict(src) for src in sources]
    path = _episode_cache_path(src_rows, default_fps)
    if path.is_file():
        return _read_episode_cache(path)

    lock = path.with_suffix(".lock")
    deadline = time.time() + 3600.0
    path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if path.is_file():
                return _read_episode_cache(path)
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for EgoVerse cache lock: {lock}")
            time.sleep(1.0)

    try:
        rows = _build_episode_index(src_rows, default_fps)
        _write_episode_cache(path, rows)
        return rows
    finally:
        if lock.exists():
            lock.unlink()


def list_egoverse_episodes(
    sources: Iterable[Dict[str, Any]],
    split: str,
    val_ratio: float,
    split_seed: int,
    default_fps: int,
    max_episodes: Optional[int],
) -> List[EgoVerseEpisode]:
    src_rows = [_resolve_source(dict(src_raw)) for src_raw in sources]
    all_rows = _load_episode_index(src_rows, default_fps)
    rows: List[EgoVerseEpisode] = []
    for ep in all_rows:
        if in_split(ep.sample_id, split, val_ratio, split_seed):
            rows.append(ep)
            if max_episodes is not None and len(rows) >= max_episodes:
                return rows
    if not rows:
        raise ValueError("No EgoVerse episodes matched the requested split and source roots.")
    return rows


# ---------------------------------------------------------------------------
# Raw Zarr v3 decoding (sharded numeric + JPEG arrays)
# ---------------------------------------------------------------------------

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _load_array_meta(path: Path) -> Dict[str, Any]:
    meta = _load_json(path / "zarr.json")
    if meta["node_type"] != "array":
        raise ValueError(f"Expected zarr array metadata at {path}, got {meta['node_type']}")
    if int(meta["zarr_format"]) != 3:
        raise ValueError(f"Expected zarr_format=3 at {path}, got {meta['zarr_format']}")
    return meta


def _outer_grid_shape(shape: List[int], outer_chunk: List[int]) -> List[int]:
    return [_ceil_div(int(s), int(c)) for s, c in zip(shape, outer_chunk)]


def _chunk_path(path: Path, outer_idx: tuple[int, ...], sep: str) -> Path:
    out = path / "c"
    if sep != "/":
        raise ValueError(f"Unsupported zarr separator '{sep}'")
    for idx in outer_idx:
        out = out / str(idx)
    return out


def _decode_index(buf: bytes, n_inner: int, has_crc32c: bool) -> np.ndarray:
    n_bytes = n_inner * 16 + (4 if has_crc32c else 0)
    if len(buf) < n_bytes:
        raise ValueError(f"Shard buffer too small for index: len={len(buf)} need={n_bytes}")
    raw = buf[-n_bytes:]
    if has_crc32c:
        raw = raw[:-4]
    idx = np.frombuffer(raw, dtype="<u8")
    if idx.size != n_inner * 2:
        raise ValueError(f"Unexpected sharding index size: got {idx.size}, expected {n_inner * 2}")
    return idx.reshape(n_inner, 2)


def _decode_local_chunk(buf: bytes) -> bytes:
    if len(buf) < 8:
        raise ValueError("Local JPEG chunk too small.")
    n_val = struct.unpack_from("<I", buf, 0)[0]
    n_bytes = struct.unpack_from("<I", buf, 4)[0]
    if n_val != 1:
        raise ValueError(f"Expected single JPEG entry, got {n_val}")
    st = 8
    ed = st + n_bytes
    if ed != len(buf):
        raise ValueError(f"Corrupt JPEG chunk: payload ends at {ed}, file len={len(buf)}")
    return bytes(buf[st:ed])


def _zstd_decode(buf: bytes) -> bytes:
    from numcodecs.zstd import Zstd
    return Zstd().decode(buf)


def _load_numeric_zarr_array(path: Path) -> np.ndarray:
    meta = _load_array_meta(path)
    shape = [int(x) for x in meta["shape"]]
    dtype = np.dtype(meta["data_type"]).newbyteorder("<")
    outer_chunk = [int(x) for x in meta["chunk_grid"]["configuration"]["chunk_shape"]]
    shard = meta["codecs"][0]
    if shard["name"] != "sharding_indexed":
        raise ValueError(f"Unsupported codec stack for numeric array at {path}: {meta['codecs']}")
    inner_chunk = [int(x) for x in shard["configuration"]["chunk_shape"]]
    data_codecs = [c["name"] for c in shard["configuration"]["codecs"]]
    index_codecs = [c["name"] for c in shard["configuration"]["index_codecs"]]
    if data_codecs != ["bytes", "zstd"]:
        raise ValueError(f"Unsupported numeric data codecs at {path}: {data_codecs}")
    if index_codecs != ["bytes", "crc32c"]:
        raise ValueError(f"Unsupported numeric index codecs at {path}: {index_codecs}")
    sep = meta["chunk_key_encoding"]["configuration"]["separator"]
    fill_value = meta["fill_value"]
    arr = np.full(shape, fill_value, dtype=dtype)

    inner_grid = [int(o // i) for o, i in zip(outer_chunk, inner_chunk)]
    n_inner = int(np.prod(inner_grid, dtype=np.int64))
    outer_grid = _outer_grid_shape(shape, outer_chunk)
    for outer_idx in np.ndindex(*outer_grid):
        shard_path = _chunk_path(path, outer_idx, sep)
        if not shard_path.is_file():
            continue
        buf = shard_path.read_bytes()
        idx = _decode_index(buf, n_inner, has_crc32c=True)
        for inner_flat, inner_idx in enumerate(np.ndindex(*inner_grid)):
            off = idx[inner_flat, 0]
            n_bytes = idx[inner_flat, 1]
            if off == _MISSING_U64 or n_bytes == _MISSING_U64:
                continue
            st = int(off)
            ed = st + int(n_bytes)
            raw = _zstd_decode(buf[st:ed])
            chunk = np.frombuffer(raw, dtype=dtype).copy().reshape(inner_chunk)
            sl = []
            ch_sl = []
            for d in range(len(shape)):
                base = int(outer_idx[d]) * outer_chunk[d] + int(inner_idx[d]) * inner_chunk[d]
                end = min(base + inner_chunk[d], shape[d])
                sl.append(slice(base, end))
                ch_sl.append(slice(0, end - base))
            arr[tuple(sl)] = chunk[tuple(ch_sl)]
    return arr


def _load_jpeg_zarr_array(path: Path) -> List[bytes]:
    meta = _load_array_meta(path)
    shape = [int(x) for x in meta["shape"]]
    if len(shape) != 1:
        raise ValueError(f"Expected 1D JPEG array at {path}, got shape={shape}")
    outer_chunk = [int(x) for x in meta["chunk_grid"]["configuration"]["chunk_shape"]]
    shard = meta["codecs"][0]
    if shard["name"] != "sharding_indexed":
        raise ValueError(f"Unsupported codec stack for JPEG array at {path}: {meta['codecs']}")
    inner_chunk = [int(x) for x in shard["configuration"]["chunk_shape"]]
    data_codecs = [c["name"] for c in shard["configuration"]["codecs"]]
    index_codecs = [c["name"] for c in shard["configuration"]["index_codecs"]]
    if data_codecs != ["vlen-bytes", "zstd"]:
        raise ValueError(f"Unsupported JPEG data codecs at {path}: {data_codecs}")
    if index_codecs != ["bytes", "crc32c"]:
        raise ValueError(f"Unsupported JPEG index codecs at {path}: {index_codecs}")
    if inner_chunk != [1]:
        raise ValueError(f"Expected JPEG inner chunk [1], got {inner_chunk}")
    sep = meta["chunk_key_encoding"]["configuration"]["separator"]

    outer_len = outer_chunk[0]
    n_frames = shape[0]
    n_outer = _ceil_div(n_frames, outer_len)
    out: List[bytes] = [b""] * n_frames
    for outer_idx_0 in range(n_outer):
        shard_path = _chunk_path(path, (outer_idx_0,), sep)
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing JPEG shard file: {shard_path}")
        buf = shard_path.read_bytes()
        idx = _decode_index(buf, outer_len, has_crc32c=True)
        for inner_flat in range(outer_len):
            t = outer_idx_0 * outer_len + inner_flat
            if t >= n_frames:
                break
            off = idx[inner_flat, 0]
            n_bytes = idx[inner_flat, 1]
            if off == _MISSING_U64 or n_bytes == _MISSING_U64:
                raise ValueError(f"Missing JPEG payload for frame {t} in {path}")
            st = int(off)
            ed = st + int(n_bytes)
            raw = _zstd_decode(buf[st:ed])
            out[t] = _decode_local_chunk(raw)
    return out


def _decode_jpeg(jpeg: bytes) -> np.ndarray:
    with Image.open(BytesIO(jpeg)) as img:
        rgb = img.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def _resize_frame(img: np.ndarray, img_size: int) -> np.ndarray:
    # LANCZOS aspect-preserving resize (shortest side = img_size) + center crop.
    # A bilinear squash distorts aspect ratio and makes VAE latents OOD vs Wan pretraining.
    if img_size <= 0:
        raise ValueError(f"img_size must be positive, got {img_size}")
    if img.shape[0] == img_size and img.shape[1] == img_size:
        return img
    h, w = img.shape[:2]
    s = img_size / min(h, w)
    nw, nh = round(w * s), round(h * s)
    pil = Image.fromarray(img).resize((nw, nh), Image.LANCZOS)
    x1 = (nw - img_size) // 2
    y1 = (nh - img_size) // 2
    return np.asarray(pil.crop((x1, y1, x1 + img_size, y1 + img_size)), dtype=np.uint8)


def load_episode_video(ep: EgoVerseEpisode, img_size: int) -> np.ndarray:
    ep_dir = Path(ep.episode_dir)
    if ep.image_backend == "zarr":
        jpeg_frames = _load_jpeg_zarr_array(ep_dir / "images.front_1")
    elif ep.image_backend == "local_chunks":
        chunk_dir = ep_dir / "images.front_1" / "c"
        chunk_files = sorted([p for p in chunk_dir.iterdir() if p.is_file()])
        jpeg_frames = [_decode_local_chunk(p.read_bytes()) for p in chunk_files]
    else:
        raise ValueError(f"Unsupported EgoVerse image backend '{ep.image_backend}'")

    frames = [_resize_frame(_decode_jpeg(jpeg), img_size) for jpeg in jpeg_frames]
    return np.stack(frames, axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Camera decoding (head pose -> extrinsics, metadata -> scaled intrinsics)
# ---------------------------------------------------------------------------

def _quat_to_rot_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with 4 entries, got shape {q.shape}")
    nrm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(nrm == 0.0):
        raise ValueError("Encountered zero-norm quaternion in EgoVerse head pose.")
    q = q / nrm
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    rot = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    rot[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[..., 0, 1] = 2.0 * (x * y - z * w)
    rot[..., 0, 2] = 2.0 * (x * z + y * w)
    rot[..., 1, 0] = 2.0 * (x * y + z * w)
    rot[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[..., 1, 2] = 2.0 * (y * z - x * w)
    rot[..., 2, 0] = 2.0 * (x * z - y * w)
    rot[..., 2, 1] = 2.0 * (y * z + x * w)
    rot[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


def _camera_matrix_from_metadata(ep: EgoVerseEpisode) -> np.ndarray:
    root_attr = _load_json(Path(ep.episode_dir) / "zarr.json").get("attributes", {})
    # Intrinsics may live under either metadata key; fall back to the fixed
    # MECKA matrix for MECKA embodiments.
    intr = root_attr.get("camera_intrinsics")
    if not (isinstance(intr, dict) and {"fx", "fy", "cx", "cy"} <= set(intr.keys())):
        intr = root_attr.get("intrinsics")
    if isinstance(intr, dict) and {"fx", "fy", "cx", "cy"} <= set(intr.keys()):
        return np.array(
            [
                [float(intr["fx"]), 0.0, float(intr["cx"])],
                [0.0, float(intr["fy"]), float(intr["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    emb = str(root_attr.get("embodiment", ""))
    if emb.startswith("MECKA"):
        return MECKA_INTRINSICS_360_640.copy()
    raise ValueError(f"No usable intrinsics for embodiment '{emb}' in {ep.episode_dir}")


def load_episode_camera(ep: EgoVerseEpisode, img_size: int) -> Dict[str, np.ndarray]:
    ep_dir = Path(ep.episode_dir)
    head = _load_numeric_zarr_array(ep_dir / "obs_head_pose")
    if head.ndim != 2 or head.shape[1] != 7:
        raise ValueError(f"Expected obs_head_pose shape (T, 7), got {head.shape}")

    pos = head[:, :3].astype(np.float64)
    quat = head[:, 3:].astype(np.float64)
    rot = _quat_to_rot_wxyz(quat)
    t = head.shape[0]

    extr = np.tile(np.eye(4, dtype=np.float64), (t, 1, 1))
    extr[:, :3, :3] = rot
    extr[:, :3, 3] = pos

    if ep.frame_height <= 0 or ep.frame_width <= 0:
        raise ValueError(f"Episode {ep.sample_id} is missing raw frame dimensions.")
    # LANCZOS aspect-preserving uniform scale + center crop — mirrors
    # _resize_frame so intrinsics describe the same pixel grid the tracker
    # consumes. fx,fy and cx,cy scale by the uniform factor; cx,cy shift
    # by the center-crop offset on the long axis.
    intr = _camera_matrix_from_metadata(ep)
    s = float(img_size) / float(min(ep.frame_height, ep.frame_width))
    rw = round(ep.frame_width * s)
    rh = round(ep.frame_height * s)
    crop_x = (rw - img_size) // 2
    crop_y = (rh - img_size) // 2
    intr_scaled = intr.copy()
    intr_scaled[0, 0] *= s
    intr_scaled[1, 1] *= s
    intr_scaled[0, 2] = intr_scaled[0, 2] * s - crop_x
    intr_scaled[1, 2] = intr_scaled[1, 2] * s - crop_y
    intrs = np.repeat(intr_scaled[None], t, axis=0)

    return {
        "camera_extrinsics": extr.astype(np.float32),
        "camera_intrinsics": intrs.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Output-path / store helpers
# ---------------------------------------------------------------------------

def _load_skip_ids(path: str) -> set[str]:
    """Read a JSONL manifest of finished clips. Each line must be a JSON
    object with a `sample_id` field; other fields are ignored. Raises if the
    file is missing or any line lacks `sample_id`."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"skip_ids_file does not exist: {path}")
    out: set[str] = set()
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "sample_id" not in rec:
                raise ValueError(f"skip_ids_file {path}:{ln} missing sample_id field")
            out.add(str(rec["sample_id"]))
    return out


def _sanitize_part(val: str) -> str:
    txt = str(val).strip()
    if not txt:
        raise ValueError("Encountered empty path component while building EgoVerse output path.")
    return txt.replace("/", "_").replace(" ", "_")


def _read_store_array(path: str, key: str) -> np.ndarray:
    return np.asarray(zarr.open_group(path, mode="r")[key])


# ---------------------------------------------------------------------------
# processors
# ---------------------------------------------------------------------------

class _VideoProcessor(Processor):
    def process(self, out_h5_dict: Dict[str, h5py.File], sample: Sample) -> None:
        out_h5 = self._get_single_file(out_h5_dict)
        root = out_h5.require_group("root")
        for view, video in sample.videos.items():
            vg = root.require_group(view)
            if "video" in vg:
                del vg["video"]
            arr = np.asarray(video, dtype=np.uint8)
            # One chunk holds up to 246 frames at full HxW, Blosc-lz4
            # compressed, keeping the inode count per clip low.
            T, H, W, C = arr.shape
            vg.create_dataset(
                "video", data=arr, dtype="uint8", chunks=(min(246, T), H, W, C),
                compressor=Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE),
            )


class _TextProcessor(Processor):
    """Clip text plus the segment timeline EgoVerseDataset.load_text reads:
    fps, segments_start_sec (K,), segments_label (K,). Raw EgoVerse carries no
    subtask timeline, so the clip is one segment (K=1) starting at 0.0 s."""

    def process(self, out_h5_dict: Dict[str, h5py.File], sample: Sample) -> None:
        out = self._get_single_file(out_h5_dict)
        out.create_dataset("text", data=np.asarray(sample.text))
        out.create_dataset("fps", data=np.int32(sample.meta["fps"]), dtype="int32")
        out.create_dataset("segments_start_sec", data=np.zeros(1, dtype=np.float32), dtype="float32")
        out.create_dataset("segments_label", data=np.asarray([sample.text]))


class _SegmentTextEmbProcessor(Processor):
    """UMT5-XXL embedding of every text-store segment label:
    segments_text_emb (K, S, D) fp32, row k embeds segments_label[k]."""

    def __init__(self, text_encoder, tokenizer, device):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device

    def process(self, out_h5_dict: Dict[str, h5py.File], sample: Sample) -> None:
        out = self._get_single_file(out_h5_dict)
        with torch.no_grad():
            embs = [
                encode_wan_text(self.text_encoder, self.tokenizer, label, self.device, torch.float32)
                for label in sample.meta["segments_label"]
            ]  # K x (1, S, D)
        emb = torch.cat(embs, dim=0).cpu().numpy()  # (K, S, D)
        out.create_dataset(
            "segments_text_emb", data=emb, dtype="float32", chunks=(1, *emb.shape[1:]),
            compressor=Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE),
        )


class _MetaProcessor(Processor):
    def process(self, out_h5_dict: Dict[str, h5py.File], sample: Sample) -> None:
        out_h5 = self._get_single_file(out_h5_dict)
        if sample.meta is None:
            raise ValueError("EgoVerse samples must carry metadata.")
        grp = out_h5.require_group("meta")
        for k, v in sample.meta.items():
            if k in grp:
                del grp[k]
            if isinstance(v, str):
                dt = h5py.string_dtype(encoding="utf-8")
                grp.create_dataset(k, data=np.array(v, dtype=dt), dtype=dt)
                continue
            if isinstance(v, (int, np.integer)):
                grp.create_dataset(k, data=np.int32(v), dtype="int32")
                continue
            if isinstance(v, float):
                grp.create_dataset(k, data=np.float32(v), dtype="float32")
                continue
            grp.create_dataset(k, data=np.asarray(v))


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class PreprocessEgoVerse(PreprocessDataset):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dest_root = os.path.abspath(os.path.expanduser(cfg.dest_dir))
        self.rows = list_egoverse_episodes(
            sources=list(cfg.sources),
            split=str(cfg.split),
            val_ratio=float(cfg.val_ratio),
            split_seed=int(cfg.split_seed),
            default_fps=int(cfg.default_fps),
            max_episodes=cfg.max_episodes,
        )
        ids = split_indices(len(self.rows), int(cfg.num_workers), int(cfg.worker_id))
        self.rows = [self.rows[i] for i in ids]
        sub_workers = int(cfg.sub_workers)
        if sub_workers > 1:
            sub_id = int(cfg.sub_id)
            self.rows = self.rows[sub_id::sub_workers]
        if cfg.skip_ids_file is not None:
            skip_ids = _load_skip_ids(str(cfg.skip_ids_file))
            n_before = len(self.rows)
            self.rows = [r for r in self.rows if r.sample_id not in skip_ids]
            print(
                f"[skip_ids_file] {cfg.skip_ids_file}: {len(skip_ids)} ids in manifest; "
                f"worker {cfg.worker_id}/{cfg.num_workers} dropped {n_before - len(self.rows)} "
                f"already-finished clips; {len(self.rows)} remain."
            )
        if not self.rows:
            raise ValueError(
                f"No EgoVerse episodes remain for worker {cfg.worker_id}/{cfg.num_workers}."
            )

    def build_models(self, cfg) -> Dict[str, Any]:
        models: Dict[str, Any] = {}
        if cfg.mode == "tracks":
            load_tracker(cfg, models)
        elif cfg.mode in {"vae_latent_h16", "text_emb"}:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dreamzero"))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            models["device"] = device
            if cfg.mode == "vae_latent_h16":
                from groot.vla.model.dreamzero.modules.wan_video_vae import WanVideoVAE38
                vae = WanVideoVAE38(vae_pretrained_path=cfg.vae_weights)
                vae.model.load_state_dict(torch.load(cfg.vae_weights, map_location="cpu", weights_only=True))
                models["vae"] = vae.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
            else:
                from transformers import AutoTokenizer
                from groot.vla.model.dreamzero.modules.wan_video_text_encoder import WanTextEncoder
                enc = WanTextEncoder()
                enc.load_state_dict(torch.load(cfg.text_encoder_weights, map_location="cpu"))
                models["text_encoder"] = enc.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
                models["tokenizer"] = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
        return models

    def build_processors(self, cfg, models: Dict[str, Any]) -> Dict[str, Processor]:
        if cfg.mode == "tracks":
            track_fn, track_models = get_track_processor_func(cfg, models)
            return {
                "tracks": TrackProcessor(
                    track_fn=track_fn,
                    models=track_models,
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
            return {"text_emb": _SegmentTextEmbProcessor(
                text_encoder=models["text_encoder"],
                tokenizer=models["tokenizer"],
                device=models["device"],
            )}
        if cfg.mode == "meta":
            return {"meta": _MetaProcessor()}
        if cfg.mode == "vae_latent_h16":
            return {"vae_latent_h16": WanVaeLatentH16Processor(
                vae=models["vae"],
                device=models["device"],
                image_hw=cfg.wan.image_hw,
                horizon=cfg.horizon,
            )}
        raise ValueError(f"Unsupported EgoVerse mode '{cfg.mode}'.")

    def iter_items(self, cfg) -> Iterable[Any]:
        rows = tqdm(self.rows, desc="EgoVerse episodes", leave=False) if cfg.progress else self.rows
        yield from rows

    def to_sample(self, item: EgoVerseEpisode, cfg) -> Sample:
        vid = None
        intrs = None
        extrs = None
        if cfg.mode == "video":
            vid = load_episode_video(item, int(cfg.img_size))
        elif cfg.mode in {"tracks", "vae_latent_h16"}:
            vid = _read_store_array(self._input_store_path(item, "video"), f"root/{VIEW}/video")
            if (
                cfg.mode == "tracks"
                and cfg.tracker_type == "spatialtracker"
                and cfg.camera_source == "egoverse"
            ):
                cam = load_episode_camera(item, int(cfg.img_size))
                if cam["camera_intrinsics"].shape[0] != vid.shape[0]:
                    raise ValueError(
                        f"Camera/video length mismatch for {item.sample_id}: "
                        f"intrs={cam['camera_intrinsics'].shape[0]} video={vid.shape[0]}"
                    )
                intrs = {VIEW: cam["camera_intrinsics"]}
                extrs = {VIEW: cam["camera_extrinsics"]}

        meta = dict(item.meta)
        if cfg.mode == "text_emb":
            meta["segments_label"] = [
                str(s) for s in _read_store_array(self._input_store_path(item, "text"), "segments_label")
            ]
        if vid is not None:
            meta["frame_count"] = np.int32(int(vid.shape[0]))
            meta["frame_height"] = np.int32(int(vid.shape[1]))
            meta["frame_width"] = np.int32(int(vid.shape[2]))
        meta["task_dir"] = _sanitize_part(item.task_name)
        meta["path_prefix"] = item.sample_id[:2]
        return Sample(
            id=item.sample_id,
            videos={VIEW: vid},
            intrs=intrs,
            extrs=extrs,
            text=item.text,
            meta=meta,
        )

    def _input_store_path(self, item: EgoVerseEpisode, mode: str) -> str:
        """Store an earlier `mode` stage wrote under source_output_name."""
        path = os.path.join(
            self.dest_root, str(self.cfg.source_output_name), self.cfg.split, MODE_DIRS[mode],
            _sanitize_part(item.task_name), item.sample_id[:2], f"{item.sample_id}.zarr",
        )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing EgoVerse {mode} store for {item.sample_id}: {path}. "
                f"Run preprocessing/preprocess_egoverse.py mode={mode} first."
            )
        return path

    def output_paths(self, sample: Sample, cfg) -> Dict[str, str]:
        if sample.meta is None:
            raise ValueError("EgoVerse sample metadata is required to build output paths.")
        task = _sanitize_part(str(sample.meta["task_dir"]))
        pref = _sanitize_part(str(sample.meta["path_prefix"]))
        base = os.path.join(self.dest_root, cfg.output_name, cfg.split)
        if cfg.mode == "tracks":
            common = _get_tracks_common_dir(cfg, base)
            out: Dict[str, str] = {}
            for out_type in cfg.track_outputs:
                sub = _get_output_subfolder(cfg, out_type)
                out[out_type] = os.path.join(common, sub, task, pref, f"{sample.id}.zarr")
            return out
        mode_dir = MODE_DIRS[cfg.mode]
        return {mode_dir: os.path.join(base, mode_dir, task, pref, f"{sample.id}.zarr")}


@hydra.main(config_path="../cfg/preprocessing", config_name="preprocess_egoverse", version_base="1.2")
def main(cfg: DictConfig) -> None:
    validate_preprocess_config("egoverse", cfg)
    base_dir = os.path.join(cfg.dest_dir, cfg.output_name, cfg.split)
    if cfg.mode == "tracks":
        save_tracks_config(cfg, base_dir)
    else:
        save_dir = os.path.join(base_dir, MODE_DIRS[cfg.mode])
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(save_dir, "config.yaml"))

    ds = PreprocessEgoVerse(cfg)
    run_dataset(ds, cfg)


if __name__ == "__main__":
    main()
