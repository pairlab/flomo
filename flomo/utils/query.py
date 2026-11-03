"""Query coordinates and grid sampling in normalized (row, col) space."""

import torch
from einops import repeat


class Query:
    """Convert `(B, N, 2)` row/col queries in [-1, 1] to tracker formats."""
    def __init__(self, tensor, allow_out_of_frame=False):
        assert tensor.ndim == 3
        assert tensor.shape[-1] == 2
        if not allow_out_of_frame:
            assert torch.all(tensor >= -1) and torch.all(tensor <= 1)
        self.tensor = tensor
        self.device = tensor.device

    def standard(self):
        return self.tensor

    def spatialtracker(self, H, W):
        """
        Input (self.tensor): (1, n_tracks, 2) with (row, col) in [-1, 1]
        Output: (n_tracks, 3) numpy array with [t, x, y] in pixel coords
        """

        tensor = self.tensor.squeeze(0) # (n_tracks, 2)

        # Convert normalized (row, col) to pixel (x, y)
        row = (tensor[:, 0] + 1) * H / 2  # [-1, 1] -> [0, H]
        col = (tensor[:, 1] + 1) * W / 2  # [-1, 1] -> [0, W]
        t = torch.zeros_like(row)

        result = torch.stack([t, col, row], dim=-1) # (n_tracks, 3)
        return result.cpu().numpy()

    def sample_indices(self, n_samples, manual_queries=None):
        """Return `(B, n_samples)` random or nearest-manual-query indices."""
        n_queries = self.tensor.shape[1]
        assert n_samples <= n_queries, f"Number of samples ({n_samples}) must be less than or equal to number of queries  available ({n_queries})."
        if manual_queries is None:
            indices = torch.randperm(n_queries)[:n_samples].to(self.device)
            indices = indices.repeat(self.tensor.shape[0], 1)
        else:
            assert n_samples == manual_queries.tensor.shape[1], f"Number of samples ({n_samples}) must be equal to number of manual queries provided ({manual_queries.tensor.shape[1]})."
            indices = torch.cdist(self.tensor, manual_queries.standard()).argmin(dim=1)

        return indices

    def sample(self, n_samples, manual_queries=None):
        """Sample random points, or the nearest points to manual_queries."""
        indices = self.sample_indices(n_samples, manual_queries)
        sampled_queries = torch.gather(self.tensor, 1, indices.unsqueeze(-1).expand(-1, -1, 2))
        sampled_queries = Query(sampled_queries, allow_out_of_frame=True)

        return sampled_queries


def grid_queries(views, n_tracks, device, sample=False, grid_size=None):
    """Return an edge-padded `(views, grid_size**2, 2)` row/col query grid.

    With sample=True, select n_tracks points independently for each view.
    """
    grid_size = int((n_tracks)**0.5) if grid_size is None else grid_size
    padding = 1 / grid_size
    min_val = -1 + padding
    max_val = 1 - padding
    x = torch.linspace(min_val, max_val, grid_size)
    y = torch.linspace(min_val, max_val, grid_size)
    xx, yy = torch.meshgrid(x, y)
    points = torch.stack([yy, xx], dim=-1).reshape(-1, 2).to(device) # (n_tracks, 2)
    points = repeat(points, "n d -> v n d", v=views) # (views, n_tracks, 2)

    if sample:
        # (views, grid_size**2, 2) -> (views, n_tracks, 2)
        new_points = torch.empty((views, n_tracks, 2), device=device)
        for i in range(views):
            new_points[i] = points[i][torch.randperm(points[i].shape[0])[:n_tracks]]
        points = new_points

    return Query(points)
