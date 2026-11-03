import time

import numpy as np
import torch

from flomo.utils.cfg_utils import get_device
from preprocessing.SpatialTracker import prepare_tracking_inputs
from preprocessing.preprocessing_utils import (preprocess_image,
                                               prepare_video_for_vggt,
                                               pad_video_for_tracking,
                                               scale_intrinsics,
                                               pad_tensor
                                               )
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

def vggt_tracks_from_video(
        video,
        models,
        init_queries='uniform',
        reinit=True,
        horizon=16,
        n_tracks=400,
        batch_size=8,
        dim_order='tchw',
        track_outputs=['3d'],
        multi_view=False,
        output_size=None,
        chunk_size=None,
        subsampling=None
):
    """
    Args:
        video: Video frames as numpy array (T, C, H, W) or dict of arrays for multi_view
        models: Dictionary containing 'vggt_model' (depth estimation) and 'track_model' (SpatialTracker)
        init_queries: Query initialization method ('uniform' for grid sampling)
        reinit: If True, reinitialize tracking at each frame; if False, track across full video once
        horizon: Number of frames to track forward from each timestep
        n_tracks: Number of points to track (default 400 for 20x20 grid)
        batch_size: Batch size for processing (unsupported by SpatialTracker)
        dim_order: Input video dimension order, either 'tchw' or 'thwc'
        vggt_chunk_size: Number of frames to process per VGGT inference call (reduces GPU memory usage)
        track_outputs: List of outputs to extract. Valid values:
            - '2d': 2D tracks and visibility
            - '3d': 3D tracks and visibility
            - 'pose': camera extrinsics and intrinsics
            - 'depth': depth map and confidence per pixel
        multi_view: If True, process multiple concatenated views for VGGT cross-view attention
        output_size: Desired output size for the depth and point maps. If None, use original size.
        chunk_size: Number of frames per VGGT inference chunk.
    Returns:
        Dictionary with requested outputs:
            If multi_view=False: Dict[str, np.ndarray] (existing behavior)
            If multi_view=True: Dict[str, Dict[str, np.ndarray]] nested by view
    """
    start_total = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] spatial_tracks_from_video started")

    device = get_device()
    vggt_model = models["vggt_model"]

    with torch.no_grad():
        # 1. Prepare video for VGGT (handles multi-view concatenation)
        video_tensor, T, num_views, view_order = prepare_video_for_vggt(
            video, dim_order, multi_view, device
        )

        # 2. Preprocess to 518x518
        # We can reuse the same preprocessing function from SpatialTracker
        video_preprocessed = preprocess_image(
            video_tensor / 255.0, mode="crop", target_size=518, keep_ratio=False
        )
        T_total, C, H, W = video_preprocessed.shape
        if multi_view:
            T = T_total // num_views  # Original timesteps per view

        # 3. Pad video for tracking windows
        video_padded = pad_video_for_tracking(
            video_preprocessed, horizon, num_views, T
        )

        # 4. Run VGGT inference
        depth_full, intrs_full, extrs_full, unc_metric_full = _run_vggt_inference(
            video_padded, vggt_model, chunk_size,
            multi_view, num_views, T, output_size, horizon
        )

        # 5. Prepare tracking inputs (handles splitting for multi-view)
        tracking_inputs = prepare_tracking_inputs(
            video_padded, depth_full, intrs_full, extrs_full, unc_metric_full,
            multi_view, view_order, device
        )

        total_time = time.time() - start_total
        print(f"[{time.strftime('%H:%M:%S')}] spatial_tracks_from_video complete: {total_time:.1f}s")

        results = {}

        if multi_view:
            for view in tracking_inputs:
                results[view] = _build_results_dict(None, None, None, tracking_inputs[view], track_outputs)
        else:
            results = _build_results_dict(None, None, None, tracking_inputs, track_outputs)

        return results


def _run_vggt_inference(video_padded, vggt_model, chunk_size,
                        multi_view, num_views, T, output_size, horizon):
    """Run VGGT depth estimation in chunks to avoid OOM.

    Returns:
        depth_full: (N, C, H, W)
        intrs_full: (N, C, 3, 3)
        extrs_full: (N, C, 4, 4)
        unc_metric_full: (N, C, H, W)

        where N = number of chunks and C = chunk_size
    """
    T_padded = len(video_padded)

    if chunk_size is None:
        raise ValueError("chunk_size must be provided for VGGT inference")

    print(f"[{time.strftime('%H:%M:%S')}] Starting chunked VGGT inference")
    if multi_view:
        print(f"  Multi-view mode: {num_views} views, {T} timesteps per view")

    print(f"  Video length: {T_padded} frames")

    # Calculate number of chunks with stride
    stride = chunk_size - num_views * horizon
    assert stride > 0, "chunk_size must be larger than num_views * horizon to allow each view to have an overlap of horizon"
    num_chunks = max(1, int(np.ceil((T_padded - horizon * num_views) / stride)))
    print(f"  Processing in {num_chunks} chunk(s) of {chunk_size} frames with stride {stride} (overlap: {chunk_size - stride})")

    vggt_start = time.time()
    
    all_depths, all_uncs, all_extrs, all_intrs = [], [], [], []

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * stride
            chunk_end = min(chunk_start + chunk_size, T_padded)
            actual_chunk_size = chunk_end - chunk_start

            print(
                f"  Processing chunk {chunk_idx + 1}/{num_chunks}: frames {chunk_start}-{chunk_end} ({actual_chunk_size} frames)")

            video_chunk = video_padded[chunk_start:chunk_end].unsqueeze(0)

            # Get aggregated tokens
            aggregated_tokens_list, ps_idx = vggt_model.aggregator(video_chunk)

            # Extract outputs, handling batch dimension
            depth, unc = vggt_model.depth_head(aggregated_tokens_list, video_chunk, ps_idx)
            if depth.dim() == 5: depth = depth.squeeze(0)  # [T, H, W, 1]
            if unc.dim() == 4: unc = unc.squeeze(0)  # [T, H, W]

            pose_enc = vggt_model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, video_chunk.shape[-2:])
            if extrinsic.dim() == 4:
                extrinsic = extrinsic.squeeze(0)
                # turn into 4x4 homogenous matrix (originally 3x4)
                bottom = torch.tensor([0, 0, 0, 1], device=get_device()).view(1, 1, 4).repeat(extrinsic.shape[0], 1, 1)
                extrinsic = torch.cat([extrinsic, bottom], dim=1)
                        
            if intrinsic.dim() == 4: intrinsic = intrinsic.squeeze(0)

            # Resize depth and unc to output_size, and scale intrinsics accordingly
            if output_size is not None:
                original_h, original_w = depth.shape[-3:-1]  # Get current size before resizing
                
                depth = depth.squeeze(-1).unsqueeze(1)  # [T, 1, H, W]
                unc = unc.unsqueeze(1)  # [T, 1, H, W]
                
                depth = torch.nn.functional.interpolate(depth, size=output_size, mode="bilinear", align_corners=False)
                unc = torch.nn.functional.interpolate(unc, size=output_size, mode="bilinear", align_corners=False)
                
                # Reshape back: [T, 1, H', W'] -> [T, H', W']
                depth = depth.squeeze(1)
                unc = unc.squeeze(1)
                
                # Scale intrinsics to match new resolution
                intrinsic = scale_intrinsics(intrinsic, (original_h, original_w), (output_size, output_size))
            else:
                # Remove trailing dimension from depth if no resizing
                depth = depth.squeeze(-1)

            if chunk_idx == num_chunks - 1:  # might need to pad tensors to stack
                depth = pad_tensor(depth, chunk_size)
                unc = pad_tensor(unc, chunk_size)
                extrinsic = pad_tensor(extrinsic, chunk_size, identity=True)  # Pad by identity so that we can take the inverse later
                intrinsic = pad_tensor(intrinsic, chunk_size)

            # Append all outputs
            all_depths.append(depth)
            all_uncs.append(unc)
            all_extrs.append(extrinsic)
            all_intrs.append(intrinsic)

            del aggregated_tokens_list, ps_idx
            torch.cuda.empty_cache()

    # Average accumulated predictions
    depth_full = torch.stack(all_depths, dim=0)
    unc_metric_full = torch.stack(all_uncs, dim=0)
    intrs_full = torch.stack(all_intrs, dim=0)
    extrs_full = torch.stack(all_extrs, dim=0)

    vggt_time = time.time() - vggt_start
    print(f"[{time.strftime('%H:%M:%S')}] VGGT inference complete: {vggt_time:.1f}s")

    return depth_full, intrs_full, extrs_full, unc_metric_full


def _build_results_dict(tracks_2d, tracks_3d, vis, inputs, track_outputs):
    """Build final results dictionary with requested outputs."""
    results = {}

    if '2d' in track_outputs and tracks_2d is not None:
        results['tracks'] = tracks_2d.cpu().numpy()

    if '3d' in track_outputs and tracks_3d is not None:
        results['3d_tracks'] = tracks_3d.cpu().numpy()

    if ('2d' in track_outputs or '3d' in track_outputs) and vis is not None:
        results['vis'] = vis.cpu().numpy()

    if 'pose' in track_outputs:
        inputs['extrs'] = torch.linalg.inv(inputs['extrs'])  # convert from world-to-camera to camera-to-world
        results['camera_extrinsics'] = inputs['extrs'].cpu().numpy()
        results['camera_intrinsics'] = inputs['intrs'].cpu().numpy()

    if 'depth' in track_outputs:
        results['depth_map'] = inputs['depth'].cpu().numpy()
        results['depth_confidence'] = inputs['unc'].cpu().numpy()

    return results