"""Dataset-independent preprocessing with HDF5 and Zarr output.

Implement PreprocessDataset's model/processor setup, item iteration, sample
conversion, and output paths; run_dataset applies the processors and writes stores.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import h5py
import numpy as np
from einops import rearrange

from preprocessing.preprocessing_utils import inital_save_h5

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# (T, H, N, xyz) chunks sized for 16-window training reads; ~12 MiB in float32.
TRACKS_3D_CHUNKS = (16, 16, 4096, 3)


def _prepare_zarr_data(data: Any, dtype: Any) -> np.ndarray:
    arr = np.asarray(data)
    if arr.dtype.kind == "O" and arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return np.asarray(item.decode("utf-8"), dtype=str)
        if isinstance(item, str):
            return np.asarray(item, dtype=str)
    if arr.dtype.kind in {"S", "U"}:
        return arr.astype(str)
    if dtype is None:
        return arr
    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        return arr
    if np_dtype.kind == "O":
        return arr
    return np.asarray(arr, dtype=np_dtype)


class _ZarrNode:
    """Expose the h5py-style group/dataset operations used by TrackProcessor."""

    def __init__(self, node: Any):
        self.node = node

    def create_group(self, name: str) -> "_ZarrNode":
        return _ZarrNode(self.node.create_group(name))

    def require_group(self, name: str) -> "_ZarrNode":
        if hasattr(self.node, "require_group"):
            return _ZarrNode(self.node.require_group(name))
        if name in self.node:
            return _ZarrNode(self.node[name])
        return self.create_group(name)

    def create_dataset(self, name: str, data: Any, dtype: Any = None,
                       chunks: Optional[tuple] = None, compressor: Any = None) -> Any:
        arr = _prepare_zarr_data(data, dtype)
        kw: Dict[str, Any] = {}
        if chunks is not None:
            # Cap chunks to actual shape — zarr errors if chunk dim > shape dim
            kw["chunks"] = tuple(min(c, s) for c, s in zip(chunks, arr.shape))
        if compressor is not None:
            kw["compressor"] = compressor
        return self.node.create_dataset(name, data=arr, **kw)

    def __contains__(self, key: str) -> bool:
        return key in self.node

    def __getitem__(self, key: str) -> Any:
        obj = self.node[key]
        if hasattr(obj, "create_group") or hasattr(obj, "group_keys"):
            return _ZarrNode(obj)
        return obj

    def __delitem__(self, key: str) -> None:
        del self.node[key]

    def close(self) -> None:
        return None


def _required_store_paths(output_type: str, view_names: Iterable[str]) -> List[str]:
    views = list(view_names)
    if output_type == "video":
        return [f"root/{view}/video" for view in views]
    if output_type == "text":
        return ["text"]
    if output_type == "text_emb":
        return ["text_emb"]
    if output_type == "segment_text_emb":
        return ["segments_text_emb"]
    if output_type == "meta":
        return ["meta"]
    if output_type == "actions":
        return ["actions", "observation_state", "frame_indices"]
    if output_type == "vae_latent":
        return [f"root/{view}/vae_latent" for view in views]
    if output_type == "vae_latent_h16":
        return [f"root/{view}/vae_latent_h16" for view in views]
    if output_type == "2d":
        return [p for view in views for p in [f"root/{view}/tracks", f"root/{view}/vis"]]
    if output_type == "3d":
        return [p for view in views for p in [f"root/{view}/3d_tracks", f"root/{view}/vis"]]
    if output_type == "pose":
        return [p for view in views
                for p in [f"root/{view}/camera_extrinsics", f"root/{view}/camera_intrinsics"]]
    if output_type == "depth":
        return [p for view in views
                for p in [f"root/{view}/depth_map", f"root/{view}/depth_confidence"]]
    return []


def _store_has_path(node: Any, path: str) -> bool:
    cur = node
    for part in path.split("/"):
        if part not in cur:
            return False
        cur = cur[part]
    return True


def _initial_save_zarr(path: str, skip_exist: bool, required_paths: List[str]) -> Optional[_ZarrNode]:
    import zarr  # local import; only required for output_format='zarr'

    try:
        if os.path.exists(path):
            root = zarr.open_group(path, mode="r")
            # `.zmetadata` is the post-consolidate completion marker — absent means previous run died mid-write.
            zmeta_exists = os.path.exists(os.path.join(path, ".zmetadata"))
            ok = zmeta_exists and all(_store_has_path(root, req) for req in required_paths)
            if ok and skip_exist:
                print(f"Store {path} already exists, skipping.")
                return None
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    except Exception as exc:
        print(f"Error while opening zarr store {path}: {exc}")
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    return _ZarrNode(zarr.open_group(path, mode="w"))

@dataclass
class Sample:
    """A single training example to write.

    Required
    - id: Unique identifier used in output file naming
    - videos: dict mapping view -> THWC uint8/float32 arrays

    Optional
    - text: str instruction for this sample
    - actions: np.ndarray action sequence for this sample
    - meta: arbitrary metadata
    """

    id: str
    videos: Dict[str, np.ndarray]
    tracks: Optional[Dict[str, np.ndarray]] = None
    vis: Optional[Dict[str, np.ndarray]] = None
    depth: Optional[Dict[str, np.ndarray]] = None
    extrs: Optional[Dict[str, np.ndarray]] = None
    intrs: Optional[Dict[str, np.ndarray]] = None
    text: Optional[str] = None
    actions: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None


class Processor(ABC):
    """Writes a modality for one Sample into an opened HDF5 file."""

    @abstractmethod
    def process(self, out_h5_dict: Dict[str, h5py.File], sample: Sample) -> None:
        ...

    def _get_single_file(self, out_h5_dict: Dict[str, h5py.File]) -> h5py.File:
        """Helper for simple processors that write to a single file."""
        assert len(out_h5_dict) == 1, f"Expected single file, got {len(out_h5_dict)}"
        return list(out_h5_dict.values())[0]


class TrackProcessor(Processor):
    """Computes dense tracks per view and writes root/<view>/{tracks,vis}."""

    def __init__(
            self,
            track_fn,
            models,
            init_queries: str = "uniform",
            reinit: bool = True,
            horizon: int = 16,
            n_tracks: int = 400,
            batch_size: int = 8,
            track_outputs: list = None,
            multi_view: bool = False,
            output_size: int = None,
            chunk_size: int = None,
            subsampling: int = 1
    ):
        self.track_fn = track_fn
        self.models = models
        self.init_queries = init_queries
        self.reinit = reinit
        self.horizon = horizon
        self.n_tracks = n_tracks
        self.batch_size = batch_size
        self.track_outputs = track_outputs if track_outputs is not None else ['3d']
        self.multi_view = multi_view
        self.output_size = output_size
        self.chunk_size = chunk_size
        self.subsampling = subsampling

    def process(self, out_h5_dict: Dict[str, h5py.File], sample: Sample) -> None:
        if self.multi_view:
            # Collect all videos into dict and convert to tchw
            videos_dict = {}
            T = None
            for view, video_thwc in sample.videos.items():
                video_tchw = rearrange(video_thwc, "t h w c -> t c h w")
                if T is None: T = video_tchw.shape[0]
                videos_dict[view] = video_tchw

            # Call track_fn once with multi_view enabled
            results_by_view = self.track_fn(
                video=videos_dict,
                models=self.models,
                init_queries=self.init_queries,
                reinit=self.reinit,
                horizon=self.horizon,
                n_tracks=self.n_tracks,
                batch_size=self.batch_size,
                dim_order="tchw",  # Already converted above
                track_outputs=self.track_outputs,
                multi_view=True,
                chunk_size=self.chunk_size,
                output_size=self.output_size,
                subsampling=self.subsampling
            )

            # Write results for each view
            for view, results in results_by_view.items():
                # Write each output type to its corresponding file
                for output_type, h5_file in out_h5_dict.items():
                    if h5_file is None:
                        continue
                    # Create root and view groups if needed
                    root = h5_file.create_group("root") if "root" not in h5_file else h5_file["root"]
                    vg = root.create_group(view) if view not in root else root[view]

                    # Store the indicies of the frames so we can assert that the same frames are
                    # provided at dataloading time during training
                    vg.create_dataset("frame_indices", data=np.arange(0, T, self.subsampling), dtype="int32")

                    # Determine which keys from results belong to this output type
                    if output_type == '2d':
                        keys_to_write = ['tracks', 'vis']
                    elif output_type == '3d':
                        keys_to_write = ['3d_tracks', 'vis']
                    elif output_type == 'pose':
                        keys_to_write = ['camera_extrinsics', 'camera_intrinsics']
                    elif output_type == 'depth':
                        keys_to_write = ['depth_map', 'depth_confidence']
                    else:
                        continue

                    # Write the relevant data
                    for key in keys_to_write:
                        if key in results:
                            if key in vg:
                                vg.__delitem__(key)
                            kw: Dict[str, Any] = {}
                            if key == "3d_tracks":
                                kw["chunks"] = TRACKS_3D_CHUNKS
                            vg.create_dataset(key, data=results[key], dtype="float32", **kw)

        else:
            # Run tracking once for all views (existing per-view loop)
            for view, video_thwc in sample.videos.items():
                video_tchw = rearrange(video_thwc, "t h w c -> t c h w")
                T = video_tchw.shape[0]
                results = self.track_fn(
                    video=video_tchw,
                    models=self.models,
                    init_queries=self.init_queries,
                    reinit=self.reinit,
                    horizon=self.horizon,
                    n_tracks=self.n_tracks,
                    batch_size=self.batch_size,
                    dim_order="tchw",
                    track_outputs=self.track_outputs,
                    chunk_size=self.chunk_size,
                    output_size=self.output_size,
                    subsampling=self.subsampling
                )

                # Write each output type to its corresponding file
                for output_type, h5_file in out_h5_dict.items():
                    if h5_file is None:
                        continue
                    # Create root and view groups if needed
                    root = h5_file.create_group("root") if "root" not in h5_file else h5_file["root"]
                    vg = root.create_group(view) if view not in root else root[view]

                    # Store the indicies of the frames so we can assert that the same frames are
                    # provided at dataloading time during training
                    vg.create_dataset("frame_indices", data=np.arange(0, T, self.subsampling), dtype="int32")

                    # Determine which keys from results belong to this output type
                    if output_type == '2d':
                        keys_to_write = ['tracks', 'vis']
                    elif output_type == '3d':
                        keys_to_write = ['3d_tracks', 'vis']
                    elif output_type == 'pose':
                        keys_to_write = ['camera_extrinsics', 'camera_intrinsics']
                    elif output_type == 'depth':
                        keys_to_write = ['depth_map', 'depth_confidence']
                    else:
                        continue

                    # Write the relevant data
                    for key in keys_to_write:
                        if key in results:
                            if key in vg:
                                vg.__delitem__(key)
                            kw: Dict[str, Any] = {}
                            if key == "3d_tracks":
                                kw["chunks"] = TRACKS_3D_CHUNKS
                            vg.create_dataset(key, data=results[key], dtype="float32", **kw)


class WanTextEmbProcessor(Processor):
    """UMT5-XXL embedding of sample.text as top-level (S, 4096) fp32."""

    def __init__(self, text_encoder, tokenizer, device):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device

    def process(self, out_h5_dict: Dict[str, h5py.File], sample: Sample) -> None:
        import torch
        from flomo.utils.wan_encoding import encode_wan_text
        if sample.text is None:
            raise ValueError(f"text_emb mode requires sample.text; got None for {sample.id}")
        out_h5 = self._get_single_file(out_h5_dict)
        with torch.no_grad():
            emb = encode_wan_text(
                self.text_encoder, self.tokenizer, sample.text, self.device, torch.float32,
            )  # (1, S, 4096)
        emb = emb.cpu().numpy()
        if "text_emb" in out_h5:
            del out_h5["text_emb"]
        out_h5.create_dataset("text_emb", data=emb[0], dtype="float32")


class WanVaeLatentProcessor(Processor):
    """Per-frame Wan2.2 VAE latent at root/<view>/vae_latent (T, z, h, w).
    Encoder runs at `image_hw` (set to match the LANCZOS pre-resize size)."""

    def __init__(self, vae, device, image_hw):
        self.vae = vae
        self.device = device
        self.image_hw = tuple(image_hw)

    def process(self, out_h5_dict: Dict[str, h5py.File], sample: Sample) -> None:
        import torch
        from flomo.utils.wan_encoding import encode_wan_vae
        out_h5 = self._get_single_file(out_h5_dict)
        root = out_h5.require_group("root")
        for view, video in sample.videos.items():
            vg = root.require_group(view)
            latents = []
            for t in range(video.shape[0]):
                frame = torch.from_numpy(video[t:t + 1])
                with torch.no_grad():
                    latent = encode_wan_vae(
                        self.vae, frame, self.image_hw, self.device, torch.bfloat16,
                    )
                latents.append(latent[0].cpu().float().numpy())
            latents = np.stack(latents, axis=0)  # (T, z, h_lat, w_lat)
            if "vae_latent" in vg:
                del vg["vae_latent"]
            vg.create_dataset("vae_latent", data=latents, dtype="float32")


class WanVaeLatentH16Processor(Processor):
    """Sliding-window Wan2.2 VAE latent at root/<view>/vae_latent_h16.

    video: (t, h, w, c) uint8
    windows: (b, horizon + 1, h, w, c)
    output: (t - horizon, z, t_lat, h_lat, w_lat)
    """

    WINDOW_BATCH = 16

    def __init__(self, vae, device, image_hw, horizon):
        self.vae = vae
        self.device = device
        self.image_hw = tuple(image_hw)
        self.win = int(horizon) + 1

    def process(self, out_h5_dict: Dict[str, h5py.File], sample: Sample) -> None:
        import torch
        from flomo.utils.wan_encoding import encode_wan_vae_video
        out_h5 = self._get_single_file(out_h5_dict)
        root = out_h5.require_group("root")
        for view, video in sample.videos.items():
            T = video.shape[0]
            if T < self.win:
                raise ValueError(
                    f"{sample.id}/{view}: video has {T} frames but "
                    f"vae_latent_h16 needs at least {self.win}."
                )
            chunks = []
            for st in range(0, T - self.win + 1, self.WINDOW_BATCH):
                ed = min(st + self.WINDOW_BATCH, T - self.win + 1)
                windows = np.stack([video[t:t + self.win] for t in range(st, ed)], axis=0)
                with torch.no_grad():
                    latent = encode_wan_vae_video(
                        self.vae, torch.from_numpy(windows), self.image_hw,
                        self.device, torch.bfloat16,
                    )  # (b, z, t_lat, h_lat, w_lat)
                chunks.append(latent.cpu().float().numpy())
            latents = np.concatenate(chunks, axis=0)  # (t - horizon, z, t_lat, h_lat, w_lat)
            vg = root.require_group(view)
            if "vae_latent_h16" in vg:
                del vg["vae_latent_h16"]
            kw: Dict[str, Any] = {}
            if isinstance(vg, _ZarrNode):
                from numcodecs import Blosc
                N = latents.shape[0]
                kw["chunks"] = (min(64, N), *latents.shape[1:])
                kw["compressor"] = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
            vg.create_dataset("vae_latent_h16", data=latents, dtype="float32", **kw)


class PreprocessDataset(ABC):
    """Implement these five methods to define your dataset preprocessing."""

    @abstractmethod
    def build_models(self, cfg) -> Dict[str, Any]:
        """Create and return any models needed by processors (e.g., trackers)."""
        ...

    @abstractmethod
    def build_processors(self, cfg, models: Dict[str, Any]) -> Dict[str, Processor]:
        """Return a dict of processors, e.g., {"tracks": TrackProcessor(...)}."""
        ...

    @abstractmethod
    def iter_items(self, cfg) -> Iterable[Any]:
        """Yield raw items to be converted into Samples (e.g., file paths/keys)."""
        ...

    @abstractmethod
    def to_sample(self, item: Any, cfg) -> Sample:
        """Map a raw item to a Sample (id, videos, optional text/actions/meta)."""
        ...

    @abstractmethod
    def output_paths(self, sample: Sample, cfg) -> Dict[str, str]:
        """Return dict mapping output type to full output .hdf5 file path."""
        ...

    # Optional hooks
    def on_begin(self, cfg, models: Dict[str, Any]) -> None:
        pass

    def on_end(self, cfg, models: Dict[str, Any]) -> None:
        pass


def _open_outfile(save_path: str, output_type: str, view_names: Iterable[str],
                  skip_exist: bool, output_format: str) -> Optional[Any]:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if output_format == "zarr":
        return _initial_save_zarr(save_path, skip_exist, _required_store_paths(output_type, view_names))
    if output_format == "hdf5":
        return inital_save_h5(save_path, skip_exist, view_names=list(view_names))
    raise ValueError(f"Unsupported output_format '{output_format}' (expected 'hdf5' or 'zarr')")

def _open_outfiles(
        save_paths: Dict[str, str],
        view_names: Iterable[str],
        skip_exist: bool,
        output_format: str,
) -> Dict[str, Optional[Any]]:
    out_files = {}
    for output_type, path in save_paths.items():
        out_files[output_type] = _open_outfile(path, output_type, view_names, skip_exist, output_format)
    return out_files

def run_dataset(defn: PreprocessDataset, cfg) -> None:
    """Run preprocessing: build models, iterate items, write outputs."""
    models = defn.build_models(cfg)
    processors = defn.build_processors(cfg, models)
    defn.on_begin(cfg, models)

    output_format = str(getattr(cfg, "output_format", "hdf5"))

    try:
        for item in defn.iter_items(cfg):
            sample = defn.to_sample(item, cfg)
            if sample is None:
                continue

            save_paths = defn.output_paths(sample, cfg)

            # Open all output files
            out_h5_dict = _open_outfiles(save_paths, sample.videos.keys(),
                                         getattr(cfg, "skip_exist", True), output_format)

            # Skip if all files already exist
            if all(h5 is None for h5 in out_h5_dict.values()):
                continue

            try:
                for proc in processors.values():
                    # Pass dict of file handles to processor
                    # Simple processors get single-item dict, track processors get multi-item dict
                    proc.process(out_h5_dict, sample)
            finally:
                # Zarr: consolidate_metadata = dataloader speedup + completion marker for skip-exist.
                for output_type, h5_file in out_h5_dict.items():
                    if h5_file is None:
                        continue
                    h5_file.close()
                    if output_format == "zarr":
                        import zarr
                        zarr.consolidate_metadata(save_paths[output_type])
    finally:
        defn.on_end(cfg, models)
