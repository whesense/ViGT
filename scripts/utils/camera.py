from __future__ import annotations

import torch
from torch.nn import functional as F


def intrinsics_from_focal(fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    return torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

def intrinsics_from_fov(
    fovy: float,
    fovx: float | None = None,
    cx: float = 0.5,
    cy: float = 0.5,
    aspect: float = 1.0,
) -> torch.Tensor:
    fovy_rad = torch.deg2rad(torch.tensor(fovy, dtype=torch.float32))
    fy = 0.5 / torch.tan(fovy_rad / 2.0)
    if fovx is not None:
        fovx_rad = torch.deg2rad(torch.tensor(fovx, dtype=torch.float32))
        fx = 0.5 / torch.tan(fovx_rad / 2.0)
    else:
        fx = fy / aspect
    return intrinsics_from_focal(float(fx), float(fy), cx, cy)


class CameraPose:
    def __init__(self, rotation: torch.Tensor, translation: torch.Tensor):
        self.rotation = rotation.float()
        self.translation = translation.float()

    @staticmethod
    def identity() -> "CameraPose":
        return CameraPose(torch.eye(3), torch.zeros(3))

    def to(self, device: torch.device | str) -> "CameraPose":
        self.rotation = self.rotation.to(device)
        self.translation = self.translation.to(device)
        return self

    def transform_points(self, points: torch.Tensor) -> torch.Tensor:
        return points @ self.rotation.T + self.translation


def lookat(
    eye: torch.Tensor | list[float],
    at: torch.Tensor | list[float],
    up: torch.Tensor | list[float] = (0.0, 0.0, 1.0),
) -> CameraPose:
    eye_t = torch.as_tensor(eye, dtype=torch.float32)
    at_t = torch.as_tensor(at, dtype=torch.float32)
    up_t = torch.as_tensor(up, dtype=torch.float32)

    z = F.normalize(at_t - eye_t, dim=-1, eps=1e-5)
    if torch.abs(torch.dot(z, up_t) - 1.0) < 1e-3:
        raise RuntimeError("Direction and up vectors cannot be collinear.")

    x = F.normalize(torch.cross(up_t, z, dim=-1), dim=-1, eps=1e-5)
    y = F.normalize(torch.cross(z, x, dim=-1), dim=-1, eps=1e-5)
    rotation = torch.stack([-x, -y, z], dim=0).T
    return CameraPose(rotation=rotation, translation=eye_t)


class PinholeCamera:
    def __init__(
        self,
        viewport: tuple[int, int],
        intrinsics: torch.Tensor,
        world_se3_camera: CameraPose | None = None,
    ):
        self.viewport = viewport
        self.world_se3_camera = world_se3_camera or CameraPose.identity()
        self.intrinsics = intrinsics.float()

    @staticmethod
    def lookat(
        eye: torch.Tensor | list[float],
        at: torch.Tensor | list[float],
        up: torch.Tensor | list[float] = (0.0, 0.0, 1.0),
        fov: float = 90.0,
        viewport: tuple[int, int] = (1024, 1024),
    ) -> "PinholeCamera":
        aspect = viewport[1] / viewport[0]
        return PinholeCamera(
            viewport=viewport,
            intrinsics=intrinsics_from_fov(fovy=fov, aspect=aspect),
            world_se3_camera=lookat(eye=eye, at=at, up=up),
        )

    def unproject(self, uvd: torch.Tensor, from_pixels: bool = True) -> torch.Tensor:
        depth = uvd[..., 2:3]
        uv = uvd[..., :2]

        if from_pixels:
            height, width = self.viewport
            scale = torch.tensor([width, height], device=uv.device, dtype=uv.dtype)
            uv = uv / scale

        uv = uv * depth
        points = torch.cat([uv, depth], dim=-1)

        k_inv = torch.linalg.inv(self.intrinsics.to(points.device))
        points_cam = (k_inv @ points.T).T
        return self.world_se3_camera.to(points.device).transform_points(points_cam)


Camera = PinholeCamera
