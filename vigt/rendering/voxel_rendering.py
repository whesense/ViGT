from __future__ import annotations

from typing import Callable

import torch

@torch.no_grad()
def render_voxels(
    occupancy_fn: Callable[..., torch.Tensor],
    roi: dict,
    grid_shape: tuple[int, int, int] = (200, 200, 16),
    voxel_size: float | tuple[float, float, float] = 0.4,
    points_per_voxel: int = 8,
    threshold: float = 0.5,
    device: torch.device | str | None = "cuda",
    max_voxel_chunk: int = 32768,
) -> torch.Tensor:

    if isinstance(voxel_size, (int, float)):
        voxel_size_xyz = (float(voxel_size), float(voxel_size), float(voxel_size))

    device = torch.device(device)
    roi_min = torch.as_tensor(roi["min"], dtype=torch.float32, device=device)
    voxel_size_t = torch.as_tensor(voxel_size_xyz, dtype=torch.float32, device=device)

    nx, ny, nz = grid_shape
    grid_x = torch.arange(nx, device=device)
    grid_y = torch.arange(ny, device=device)
    grid_z = torch.arange(nz, device=device)
    voxels_ijk = torch.stack(
        torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij"),
        dim=-1,
    ).view(-1, 3)

    voxel_base = roi_min + voxels_ijk * voxel_size_t
    voxel_logits = torch.empty(voxel_base.shape[0], dtype=torch.float32, device=device)

    for start in range(0, voxel_base.shape[0], max_voxel_chunk):
        end = min(voxel_base.shape[0], start + max_voxel_chunk)
        chunk_base = voxel_base[start:end]

        offsets = torch.rand(
            (chunk_base.shape[0], points_per_voxel, 3),
            device=device,
            dtype=torch.float32,
        )
        queries = chunk_base[:, None, :] + offsets * voxel_size_t[None, None, :]
        scores = occupancy_fn(queries.reshape(-1, 3))
        scores = scores.view(-1, points_per_voxel)
        voxel_logits[start:end] = scores.max(dim=1).values

    occupied = (voxel_logits >= threshold).to(torch.uint8)
    return occupied.view(nx, ny, nz)
