import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from flomo.utils.data_utils import interpolate_traj, unnormalize_traj, compute_moving_indices


def _grid_subsample_indices(num_tracks, target_grid):
    """Indices for a uniform target_grid x target_grid subsample of an N=side^2 row-major grid."""
    side = int(round(num_tracks ** 0.5))
    if side * side != num_tracks:
        raise ValueError(f"num_tracks={num_tracks} is not a perfect square; cannot 2D-grid-subsample.")
    if target_grid is None or int(target_grid) >= side:
        return None
    target_grid = int(target_grid)
    rows = np.linspace(0, side - 1, target_grid).round().astype(int)
    cols = np.linspace(0, side - 1, target_grid).round().astype(int)
    return (rows[:, None] * side + cols[None, :]).reshape(-1)


def vis_pred(images, trajs, cmap='autumn_r', interp_seq_len=64, opacity=0.5, sample_ratio=1.0, show_all=False):
    """
    images (b, v, h, w, c) tensor, (0-1)
    trajs (b, v, t, n, d) tensor, (-1, 1), (row, col)
    show_all: if True, ignore the moving-only filter and render every track at full opacity.

    Returns:
    vis_imgs (b, h, w, c) tensor, (0-255)
    """
    assert trajs.dim() == 5
    assert images.dim() == 5

    # Move to CPU for visualization (result is converted to numpy anyway)
    images = images.cpu()
    trajs = trajs.cpu()

    image_shape = images.shape[2:4]  # (h, w)

    # Interpolate for visualization
    vis_traj = interpolate_traj(trajs.clone(), interp_seq_len)

    # Unnormalize for visualization
    vis_images = images.clone() * 255
    vis_traj = unnormalize_traj(vis_traj, image_shape).round().long()
    vis_traj = vis_traj.clamp(0, image_shape[0] - 1)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    horizon = vis_traj.shape[2]
    vis_img_list = []
    batch_size = images.shape[0]

    if show_all:
        moving_indices = torch.ones(trajs.shape[0], trajs.shape[1], trajs.shape[3], device=trajs.device, dtype=trajs.dtype)
    else:
        moving_indices = compute_moving_indices(trajs)  # (b, v, n)

    for v in range(trajs.shape[1]):
        vis_img = vis_images[:, v].clone()  # Shape: [batch_size, h, w, c]
        for t in range(horizon):
            # Set all points at timestep to color
            color = cmap(t / horizon)[:3]
            color = [int(c * 255) for c in color]

            r1 = vis_traj[:, v, t, :, 0]  # Shape: [batch_size, num_points]
            c1 = vis_traj[:, v, t, :, 1]  # Shape: [batch_size, num_points]
            r2 = torch.clamp(r1 + 1, 0, image_shape[0] - 1)
            c2 = torch.clamp(c1 + 1, 0, image_shape[1] - 1)

            num_points = r1.shape[1]
            # Sample indices according to sample_ratio
            if sample_ratio < 1.0:
                target_num_points = max(1, int(num_points * sample_ratio))
                sampled_indices = torch.linspace(0, num_points - 1, target_num_points, dtype=torch.long, device=trajs.device)
            else:
                sampled_indices = torch.arange(0, num_points, device=trajs.device)

            # Index the trajectory coordinates with the sampled indices.
            r1 = r1[:, sampled_indices]
            c1 = c1[:, sampled_indices]
            r2 = r2[:, sampled_indices]
            c2 = c2[:, sampled_indices]
            num_points = r1.shape[1]  # Update number of points after sampling

            batch_indices = torch.arange(batch_size, device=trajs.device).unsqueeze(1).expand(batch_size, num_points)

            color_tensor = torch.tensor(color, device=trajs.device, dtype=vis_images.dtype).view(1, 1, 3).expand(batch_size, num_points, 3)

            # Sample moving indices accordingly.
            opacity_tensor = moving_indices[:, v][:, sampled_indices].unsqueeze(2).expand(batch_size, num_points, 3) * opacity
            inv_opacity_tensor = 1 - opacity_tensor

            vis_img[batch_indices, r1, c1, :] = inv_opacity_tensor * vis_img[batch_indices, r1, c1, :] + opacity_tensor * color_tensor
            vis_img[batch_indices, r2, c1, :] = inv_opacity_tensor * vis_img[batch_indices, r2, c1, :] + opacity_tensor * color_tensor
            vis_img[batch_indices, r1, c2, :] = inv_opacity_tensor * vis_img[batch_indices, r1, c2, :] + opacity_tensor * color_tensor
            vis_img[batch_indices, r2, c2, :] = inv_opacity_tensor * vis_img[batch_indices, r2, c2, :] + opacity_tensor * color_tensor

        vis_img_list.append(vis_img)

    cat_vis_imgs = torch.cat(vis_img_list, dim=-2)
    return cat_vis_imgs.to(dtype=torch.uint8)


# ============ 3D Video Generation Functions ============

