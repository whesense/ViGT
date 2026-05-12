from typing import Callable

import nerfacc
import torch
from torch.nn import functional as F

from utils.camera import Camera
from utils.colormap import Colormap


class ImplicitOccupancyRenderer:
    def __init__(
        self,
        viewport: tuple[int, int] | None = None,
        zmin: float = -1.0,
        zmax: float = 2.5,
        cmap: str = "turbo",
        background: tuple[int, int, int] = (0, 0, 0),
        render_step_size: float = 0.05,
        render_chunk: int = 8192,
        roi_vmin: tuple[float, float, float] = (-100.0, -100.0, -4.5),
        roi_vmax: tuple[float, float, float] = (100.0, 100.0, 4.5),
    ) -> None:
        super().__init__()
        self.viewport = viewport
        self.render_step_size = render_step_size
        self.render_chunk = render_chunk

        if zmin < roi_vmin[2] or zmax > roi_vmax[2]:
            raise RuntimeError("ImplicitOccupancyRenderer requires zmin, zmax to be in roi.")
        self.zmin, self.zmax = zmin, zmax

        aabb = roi_vmin[:2] + (zmin,) + roi_vmax[:2] + (zmax,)
        self.estimator = nerfacc.OccGridEstimator(
            roi_aabb=torch.as_tensor(aabb).contiguous(),
        )
        self.estimator.binaries[:] = True

        self.cmap = Colormap(cmap)
        self.register_buffer("background", torch.tensor(background) / 255.0)

    def raygen(self, camera: Camera) -> tuple[torch.Tensor, torch.Tensor]:
        if list(self.parameters()):
            device = next(self.parameters()).device
        else:
            device = camera.world_se3_camera.translation.device

        xs = torch.linspace(0, camera.viewport[1] - 1, steps=camera.viewport[1], device=device)
        ys = torch.linspace(0, camera.viewport[0] - 1, steps=camera.viewport[0], device=device)

        x, y = torch.meshgrid(xs, ys, indexing="xy")
        depth = torch.ones_like(x)

        uvd = torch.stack([x, y, depth], dim=-1)
        directions = camera.unproject(uvd.view(-1, 3), from_pixels=True).to(device)

        origins = camera.world_se3_camera.translation.to(device).expand_as(directions)
        directions = F.normalize(directions - origins, dim=-1).view(-1, 3)
        return origins, directions

    def raycolor(
        self,
        occupancy_fn: Callable[..., torch.Tensor],
        origins: torch.Tensor,
        directions: torch.Tensor,
    ) -> Callable[..., torch.Tensor]:
        @torch.no_grad()
        def rgba_fn(t_starts, t_ends, ray_indices):
            current_origins = origins[ray_indices]
            current_directions = directions[ray_indices]

            queries = current_origins + current_directions * (t_starts + t_ends)[:, None] / 2.0

            z = queries[..., 2]
            z = (z - self.zmin) / (self.zmax - self.zmin)
            colors = self.cmap(z)

            alphas = occupancy_fn(queries).view_as(t_starts)
            return colors, alphas

        return rgba_fn

    @torch.no_grad()
    def forward(
        self,
        occupancy_fn: Callable[..., torch.Tensor],
        camera: Camera,
    ) -> torch.Tensor:
        origins, directions = self.raygen(camera)
        num_rays = directions.shape[0]

        frame = []
        for i in range(0, num_rays, self.render_chunk):
            current_origins = origins[i : i + self.render_chunk]
            current_directions = directions[i : i + self.render_chunk]

            ray_indices, t_starts, t_ends = self.estimator.sampling(
                current_origins,
                current_directions,
                render_step_size=self.render_step_size,
            )
            colors, _, _, _ = nerfacc.rendering(
                t_starts,
                t_ends,
                ray_indices,
                render_bkgd=self.background,
                rgb_alpha_fn=self.raycolor(
                    occupancy_fn,
                    current_origins,
                    current_directions,
                ),
                n_rays=len(current_directions),
            )
            frame.append(colors)

        frame = torch.cat(frame, dim=0)
        return frame.view(*camera.viewport + (3,))
