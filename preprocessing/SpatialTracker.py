import time

import numpy as np
import torch

from tqdm import tqdm
from flomo.utils.cfg_utils import get_device
from flomo.utils.data_utils import frame_to_chunk_position
from flomo.utils.query import grid_queries
from flomo.utils.constants import VGGT_INPUT_SIZE
from preprocessing.preprocessing_utils import (preprocess_image,
                                               prepare_video_for_vggt,
                                               pad_video_for_tracking,
                                               scale_intrinsics,
                                               pad_tensor,
                                               split_views
                                               )

def spatial_tracks_from_video(
        video,
        models,
        init_queries='uniform',
        horizon=16,
        n_tracks=400,
        reinit=True,
        batch_size=8,
        dim_order='tchw',
        chunk_size=64,
        track_outputs=['3d', '2d', 'depth', 'pose'],
        multi_view=False,
        output_size=None,
        subsampling=1
):
    """
    Args:
        video: Video frames as numpy array (T, C, H, W) or dict of arrays for multi_view
        models: Dictionary containing 'vggt_model' (depth estimation) and 'track_model' (SpatialTracker)
        init_queries: Query initialization method ('uniform' for grid sampling)
        horizon: Number of frames to track forward from each timestep
        n_tracks: Number of points to track (default 400 for 20x20 grid)
        reinit: Kept for API compatibility but ignored (always reinitialized at each frame)
        batch_size: Batch size for processing (unsupported by SpatialTracker)
        dim_order: Input video dimension order, either 'tchw' or 'thwc'
        chunk_size: Number of frames to process per VGGT inference call (reduces GPU memory usage)
        track_outputs: List of outputs to extract. Valid values:
            - '2d': 2D tracks and visibility
            - '3d': 3D tracks and visibility
            - 'pose': camera extrinsics and intrinsics
            - 'depth': depth map and confidence per pixel
        multi_view: If True, process multiple concatenated views for VGGT cross-view attention
        output_size: Desired output size for depth and point maps, if None, uses original
        subsampling: FPS subsampling rate, 1 does nothing, 2 skips every other frame, etc.

    Returns:
        Dictionary with requested outputs:
            If multi_view=False: Dict[str, np.ndarray] (existing behavior)
            If multi_view=True: Dict[str, Dict[str, np.ndarray]] nested by view
    """
    start_total = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] spatial_tracks_from_video started")

    # Subsample the video before processing if necessary
    if subsampling != 1:
        if isinstance(video, dict):
            video = {view: video[::subsampling] for view, video in video.items()}
        else:
            video = video[::subsampling]

    device = get_device()
    vggt_model = models["vggt_model"]
    track_model = models["track_model"]

    with torch.no_grad():
        # 1. Concatenate the views by interleaving frames from each view
        # if multi_view=True
        video_tensor, T, num_views, view_order = prepare_video_for_vggt(
            video, dim_order, multi_view, device
        )

        # 2. Preprocess to 518x518
        video_preprocessed = preprocess_image(
            video_tensor, mode="crop", target_size=VGGT_INPUT_SIZE, keep_ratio=False
        )
        T_total, C, H, W = video_preprocessed.shape
        if multi_view:
            T = T_total // num_views  # Original timesteps per view

        # 3. Pad video so the last frame can also be tracked with reinit
        video_padded = pad_video_for_tracking(
            video_preprocessed, horizon, num_views, T
        )

        # 4. Pass the video through VGGT to get depth and camera poses
        depth_full, intrs_full, extrs_full, unc_metric_full = _run_vggt_inference(
            video_padded, vggt_model, chunk_size,
            multi_view, num_views, T, horizon
        )

        # 5. Split the video batch into individual views if multi_view=True
        tracking_inputs = prepare_tracking_inputs(
            video_padded, depth_full, intrs_full, extrs_full, unc_metric_full,
            multi_view, view_order, device
        )

        # 6. Generate queries
        queries_st = _generate_queries(init_queries, n_tracks, H, W, device)

        # 7. Run tracking
        if multi_view:
            results = _track_multi_view_reinit(
                tracking_inputs, track_model, queries_st,
                T, horizon, n_tracks, track_outputs, output_size
            )
        else:
            results = _track_single_view_reinit(
                tracking_inputs, track_model, queries_st,
                T, horizon, n_tracks, track_outputs, output_size
            )

        total_time = time.time() - start_total
        print(f"[{time.strftime('%H:%M:%S')}] spatial_tracks_from_video complete: {total_time:.1f}s")

        return results


def _run_vggt_inference(video_padded, vggt_model, chunk_size,
                        multi_view, num_views, T, horizon):
    """Run VGGT depth estimation in chunks to avoid OOM.
    Args:
        video_padded: (T + (horizon-1)*num_views, C, H, W)

    Returns:
        depth_full: (N, C, H, W)
        intrs_full: (N, C, 3, 3)
        extrs_full: (N, C, 4, 4)
        unc_metric_full: (N, C, H, W)

        where N = number of chunks and C = chunk_size
    """
    T_padded = len(video_padded)

    print(f"[{time.strftime('%H:%M:%S')}] Starting chunked VGGT inference")

    if multi_view:
        print(f"  Multi-view mode: {num_views} views, {T} timesteps per view")

    print(f"  Video length: {T_padded} frames")

    # Calculate the number of chunks
    stride = chunk_size - (horizon * num_views) # how far forward each chunk jumps
    assert stride > 0, ("chunk_size must be larger than horizon * num_views "
                        "to allow each view to have an overlap or horizon")
    num_chunks = max(1, int(np.ceil((T_padded - (horizon * num_views)) / stride)))
    print(f"  Processing in {num_chunks} chunk(s) of {chunk_size} frames"
          f" with stride {stride} (overlap: {chunk_size - stride})")

    vggt_start = time.time()
    all_depths, all_intrs, all_extrs, all_uncs = [], [], [], []

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * stride
            chunk_end = min(chunk_start + chunk_size, T_padded)
            actual_chunk_size = chunk_end - chunk_start

            print(f"  Processing chunk {chunk_idx + 1}/{num_chunks}: frames"
                  f" {chunk_start}-{chunk_end} ({actual_chunk_size} frames)")

            video_chunk = video_padded[chunk_start:chunk_end]
            # VGGT expects a batch dim and values in [0, 1]
            vggt_output = vggt_model(video_chunk.unsqueeze(0) / 255.0)

            # Extract outputs, handling batch dimension
            depth = vggt_output["points_map"][..., 2] # Grab Z coordinate [actual_chunk_size, H, W]
            if depth.dim() == 5: depth = depth.squeeze(0)

            unc = vggt_output["unc_metric"]
            if unc.dim() == 4: unc = unc.squeeze(0) # [actual_chunk_size, H, W]

            intrs = vggt_output["intrs"]
            if intrs.dim() == 4: intrs = intrs.squeeze(0) # [actual_chunk_size, 3, 3]

            extrs = vggt_output["poses_pred"]
            if extrs.dim() == 4: extrs = extrs.squeeze(0) # [actual_chunk_size, 4, 4]

            # Check if we need to pad tensors in order to stack them
            if chunk_idx == num_chunks - 1 and actual_chunk_size < chunk_size:
                depth = pad_tensor(depth, chunk_size)
                unc = pad_tensor(unc, chunk_size)
                extrs = pad_tensor(extrs, chunk_size, identity=True)  # Pad by identity so that we can take the inverse later
                intrs = pad_tensor(intrs, chunk_size)

            all_depths.append(depth)
            all_uncs.append(unc)
            all_intrs.append(intrs)
            all_extrs.append(extrs)

            del vggt_output
            torch.cuda.empty_cache()

    # Concatenate chunks
    depth_full = torch.stack(all_depths, dim=0)
    intrs_full = torch.stack(all_intrs, dim=0)
    extrs_full = torch.stack(all_extrs, dim=0)
    unc_metric_full = torch.stack(all_uncs, dim=0)

    vggt_time = time.time() - vggt_start
    print(f"[{time.strftime('%H:%M:%S')}] VGGT inference complete: {vggt_time:.1f}s")

    return depth_full, intrs_full, extrs_full, unc_metric_full


def prepare_tracking_inputs(video_padded, depth_full, intrs_full, extrs_full,
                            unc_metric_full, multi_view, view_order, device):
    """Split VGGT outputs by view if multi-view, otherwise return single dict.

    Returns:
        If multi_view: Dict[view_name, Dict[key, tensor]]
        If single_view: Dict[key, tensor]
    """
    if multi_view:
        # Convert to numpy for splitting
        video_np = video_padded.cpu().numpy()
        depth_np = depth_full.cpu().numpy()
        intrs_np = intrs_full.cpu().numpy()
        extrs_np = extrs_full.cpu().numpy()
        unc_np = unc_metric_full.cpu().numpy()

        # Split each output by view
        unc_by_view = split_views(unc_np, view_order)
        extrs_by_view = split_views(extrs_np, view_order)
        intrs_by_view = split_views(intrs_np, view_order)
        depth_by_view = split_views(depth_np, view_order)
        # Handle video separately because it has a different shape: (T, C, H, W)
        video_by_view = {view_name: video_np[i::len(view_order)] for i, view_name in enumerate(view_order)}

        # Convert back to tensors and organize by view
        tracking_inputs = {}
        for view in view_order:
            tracking_inputs[view] = {
                'video': torch.from_numpy(video_by_view[view]).to(device),
                'depth': torch.from_numpy(depth_by_view[view]).to(device),
                'intrs': torch.from_numpy(intrs_by_view[view]).to(device),
                'extrs': torch.from_numpy(extrs_by_view[view]).to(device),
                'unc': torch.from_numpy(unc_by_view[view]).to(device),
            }
        return tracking_inputs
    else:
        return {
            'video': video_padded.to(device),
            'depth': depth_full.to(device),
            'intrs': intrs_full.to(device),
            'extrs': extrs_full.to(device),
            'unc': unc_metric_full.to(device),
        }


def _generate_queries(init_queries, n_tracks, H, W, device):
    """Generate query points for tracking."""
    if init_queries == 'uniform':
        queries = grid_queries(views=1, n_tracks=n_tracks, device=device)
    else:
        raise NotImplementedError(f"Query initialization method {init_queries} not implemented.")
    return queries.spatialtracker(H, W)


def _track_multi_view_reinit(tracking_inputs, track_model, queries_st, T, horizon,
                             n_tracks, track_outputs, output_size):
    """Track multiple views and return nested results dict."""
    results_by_view = {}

    for view, inputs in tracking_inputs.items():
        print(f"[{time.strftime('%H:%M:%S')}] Tracking view: {view}")

        view_results = _track_single_view_reinit(
            inputs, track_model, queries_st,
            T, horizon, n_tracks, track_outputs, output_size,
        )
        results_by_view[view] = view_results

    return results_by_view


def _track_single_view_reinit(inputs, track_model, queries_st, T, horizon,
                              n_tracks, track_outputs, output_size):
    tracking_start = time.time()
    device = inputs['video'].device
    N, C, _, _ = inputs['depth'].shape

    # Initialize output arrays
    tracks_2d = torch.zeros(T, horizon, n_tracks, 2).to(device) if '2d' in track_outputs else None
    tracks_3d = torch.zeros(T, horizon, n_tracks, 3).to(device) if '3d' in track_outputs else None
    vis = torch.zeros(T, horizon, n_tracks).to(device)

    # Track each frame
    for t in tqdm(range(T), desc="Tracking with SpatialTracker"):
        # Find the chunk number and position in that chunk for t
        chunk_idx, start_pos, end_pos = frame_to_chunk_position(t, N, C, horizon)

        # Extract window
        video_window = inputs['video'][t:t + horizon]
        depth_window = inputs['depth'][chunk_idx, start_pos:end_pos]
        intrs_window = inputs['intrs'][chunk_idx, start_pos:end_pos]
        extrs_window = inputs['extrs'][chunk_idx, start_pos:end_pos]
        unc_window = inputs['unc'][chunk_idx, start_pos:end_pos]

        # Normalize extrinsics to first camera pose
        # (world -> camera_0) @ (camera_h -> world)
        extrs_window = torch.linalg.inv(extrs_window[0]) @ extrs_window

        # Call SpatialTracker
        try:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                (_, _, _, _, track3d_pred, track2d_pred, vis_pred, _, _) = track_model.forward(
                    video_window,
                    depth=depth_window,
                    intrs=intrs_window,
                    extrs=extrs_window,
                    unc_metric=unc_window,
                    queries=queries_st,
                    iters_track=4,
                    stage=1,
                    # full_point=False filters query tracks by depth confidence and
                    # randomly duplicates survivors, breaking track ordering.
                    full_point=True,
                    support_frame=horizon - 1,
                    fixed_cam=True,
                    query_no_BA=True,
                    replace_ratio=0.2
                )

            # Store results
            if '3d' in track_outputs:
                # Camera poses are in camera-to-world format, so we transform the points to be in world frame (first frame in window)
                R = extrs_window[:, :3, :3]  # (horizon, 3, 3)
                translation = extrs_window[:, :3, 3]   # (horizon, 3)
                track3d_pred = (R[:, None, ...] @ track3d_pred[:, :, :3, None]).squeeze(-1) + translation[:, None, :]  # (horizon, N, 3)
                tracks_3d[t] = track3d_pred
            if '2d' in track_outputs:
                tracks_2d[t] = track2d_pred[..., :2]
            vis[t] = vis_pred.squeeze(-1)

        except RuntimeError as e:
            if "random_ expects 'from' to be less than 'to'" in str(e):
                tqdm.write(f"\n⚠️  Warning: Frame {t}/{T} had no valid points, using zero tracks")
                if '3d' in track_outputs:
                    tracks_3d[t] = 0
                if '2d' in track_outputs:
                    tracks_2d[t] = 0
                vis[t] = 0
            else:
                raise

    total_tracking = time.time() - tracking_start
    print(f"[{time.strftime('%H:%M:%S')}] Tracking complete: {total_tracking:.1f}s")

    # Build results dict
    return _build_results_dict(tracks_2d, tracks_3d, vis, inputs, track_outputs, output_size)


def _build_results_dict(tracks_2d, tracks_3d, vis, inputs, track_outputs, output_size):
    """Build final results dictionary with requested outputs."""
    results = {}

    # Original VGGT output size
    original_size = VGGT_INPUT_SIZE

    print(f"Resizing relevant matrices to output_size={output_size}")

    # Resize 2D tracks since they're in pixel coordinates
    if '2d' in track_outputs and tracks_2d is not None:
        if output_size and output_size != original_size:
            scale_factor = output_size / original_size
            tracks_2d_scaled = tracks_2d * scale_factor
            results['tracks'] = tracks_2d_scaled.cpu().numpy()
        else:
            results['tracks'] = tracks_2d.cpu().numpy()

    # 3D tracks don't need resizing since they're in world coordinates (meters)
    if '3d' in track_outputs and tracks_3d is not None:
        results['3d_tracks'] = tracks_3d.cpu().numpy()

    # Visibility doesn't need resizing
    if ('2d' in track_outputs or '3d' in track_outputs) and vis is not None:
        results['vis'] = vis.cpu().numpy()

    # Handle pose outputs
    if 'pose' in track_outputs:
        # Extrinsics don't change with image resolution
        results['camera_extrinsics'] = inputs['extrs'].cpu().numpy()

        # Scale intrinsics if output size differs
        if output_size and output_size != original_size:
            intrs_np = inputs['intrs'].cpu().numpy()
            intrs_scaled = scale_intrinsics(
                intrs_np,
                (original_size, original_size),
                (output_size, output_size)
            )
            results['camera_intrinsics'] = intrs_scaled
        else:
            results['camera_intrinsics'] = inputs['intrs'].cpu().numpy()

    # Handle depth outputs
    if 'depth' in track_outputs:
        depth = inputs['depth']
        assert depth.ndim == 4, (f"Expected depth to have 4 dimensions (N, C, H, W), got {depth.ndim} "
                                 f"dimensions with shape {depth.shape}")

        if output_size and output_size != original_size:
            # Resize depth maps
            N, C, H, W = depth.shape
            depth = depth.view(N * C, 1, H, W)
            depth_resized = torch.nn.functional.interpolate(
                depth,
                size=(output_size, output_size),
                mode='bilinear',
                align_corners=False
            )

            unc = inputs['unc']
            assert unc.ndim == 4, (f"Expected unc to have 4 dimensions (N, C, H, W), got {unc.ndim} "
                                   f"dimensions with shape {unc.shape}")
            unc = unc.view(N * C, 1, H, W)
            unc_resized = torch.nn.functional.interpolate(
                unc,
                size=(output_size, output_size),
                mode='bilinear',
                align_corners=False
            )

            # Reshape back to (N, C, H', W')
            results['depth_map'] = depth_resized.view(N, C, output_size, output_size).cpu().numpy()
            results['depth_confidence'] = unc_resized.view(N, C, output_size, output_size).cpu().numpy()
        else:
            results['depth_map'] = inputs['depth'].cpu().numpy()
            results['depth_confidence'] = inputs['unc'].cpu().numpy()

    return results
