from typing import Callable, Optional

import nerfacc
import torch

from .occupancy_rendering import ImplicitOccupancyRenderer

def _rayalpha_fn(
    occupancy_fn: Callable[..., torch.Tensor],
    origins: torch.Tensor,
    directions: torch.Tensor,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    @torch.no_grad()
    def rgba_fn(t_starts, t_ends, ray_indices):
        current_origins = origins[ray_indices]
        current_directions = directions[ray_indices]
        queries = current_origins + current_directions * (t_starts + t_ends)[:, None] / 2.0
        alphas = occupancy_fn(queries).view_as(t_starts)
        colors = torch.ones((len(ray_indices), 3), device=queries.device, dtype=queries.dtype)
        return colors, alphas

    return rgba_fn


@torch.no_grad()
def render_points(
    occupancy_fn,
    ray_origins: torch.Tensor,
    ray_dirs: torch.Tensor,
    roi: dict,
    max_rays_chunk: Optional[int] = None,
) -> torch.Tensor:
    roi_min = roi["min"]
    roi_max = roi["max"]
    renderer = ImplicitOccupancyRenderer(
        render_step_size=0.05,
        render_chunk=8192,
        roi_vmin=(roi_min[0], roi_min[1], roi_min[2]),
        roi_vmax=(roi_max[0], roi_max[1], roi_max[2]),
        zmin=roi_min[2],
        zmax=roi_max[2],
    ).to(ray_origins.device)

    assert ray_origins.device.type == "cuda", f"ray_origins on {ray_origins.device}, expected cuda"
    assert ray_dirs.device.type == "cuda", f"ray_dirs on {ray_dirs.device}, expected cuda"

    n_rays = ray_dirs.shape[0]
    chunk = max_rays_chunk or renderer.render_chunk
    points = []

    for start in range(0, n_rays, chunk):
        end = min(n_rays, start + chunk)
        origins_chunk = ray_origins[start:end]
        dirs_chunk = ray_dirs[start:end]

        assert origins_chunk.device.type == "cuda", (
            f"origins_chunk on {origins_chunk.device}, expected cuda"
        )
        assert dirs_chunk.device.type == "cuda", f"dirs_chunk on {dirs_chunk.device}, expected cuda"

        ray_indices, t_starts, t_ends = renderer.estimator.sampling(
            origins_chunk,
            dirs_chunk,
            render_step_size=renderer.render_step_size,
        )

        _, _, depth, _ = nerfacc.rendering(
            t_starts,
            t_ends,
            ray_indices,
            render_bkgd=renderer.background,
            rgb_alpha_fn=_rayalpha_fn(occupancy_fn, origins_chunk, dirs_chunk),
            n_rays=len(dirs_chunk),
        )
        points.append(origins_chunk + dirs_chunk * depth)

    return torch.cat(points, dim=0)
