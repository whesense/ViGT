import warnings
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .layers.decoder_layers import OccupancyPredictor, ResidualFullyConnected, make_linear

try:
    import tinycudann as tcnn
    USE_TINYCUDA = True
except (OSError, ImportError):
    USE_TINYCUDA = False


def grid_sample(
    features: torch.Tensor,
    locations: torch.Tensor,
    sampling_mode: str = "bilinear",
) -> torch.Tensor:
    """Sample BEV features at normalized XY locations."""
    batch_size, num_points, _ = locations.shape
    locations_xy = torch.stack([locations[..., 1], locations[..., 0]], dim=-1)
    sampled = F.grid_sample(
        features,
        locations_xy.view(batch_size, 1, num_points, 2) * 2.0 - 1.0,
        mode=sampling_mode,
        padding_mode="border",
        align_corners=False,
    )
    return sampled[:, :, 0, :].permute(0, 2, 1)


class ImplicitDecoder(nn.Module):

    def __init__(
        self,
        f_channels: int = 128,
        query_encoding: dict | None = None,
        width: int = 32,
        num_blocks: int = 5,
        sampling_mode: str = "bilinear",
        n_input_dims: int = 3,
        roi_min: Sequence[float] = (0.0, 0.0, 0.0),
        roi_max: Sequence[float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.sampling_mode = sampling_mode
        self.n_input_dims = n_input_dims

        if query_encoding is None:
            query_encoding = {"otype": "Identity"}

        if USE_TINYCUDA:
            self.query_encoding = tcnn.Encoding(
                n_input_dims=n_input_dims,
                encoding_config=query_encoding,
                dtype=torch.float32,
            )
        else:
            if query_encoding.get("otype", "Identity") != "Identity":
                raise ValueError(
                    "tinycudann is not installed and non-identity encoding was requested."
                )
            self.query_encoding = nn.Identity()
            self.query_encoding.n_output_dims = n_input_dims
            warnings.warn(
                "tinycudann is not installed; using Identity query encoding.",
                stacklevel=2,
            )

        roi_min_tensor = torch.tensor(roi_min, dtype=torch.float32)
        roi_max_tensor = torch.tensor(roi_max, dtype=torch.float32)
        if roi_min_tensor.numel() != 3 or roi_max_tensor.numel() != 3:
            raise ValueError("roi_min and roi_max must contain 3 values each.")
        self.register_buffer("roi_min", roi_min_tensor, persistent=False)
        self.register_buffer("roi_max", roi_max_tensor, persistent=False)

        self.encoded_query_dim = self.query_encoding.n_output_dims
        self.zq_projection = make_linear(f_channels, width, relu=False, bias=False)
        self.q_projection = make_linear(self.encoded_query_dim, width, relu=False, bias=False)

        attn_offset_linear = make_linear(width, 2, relu=False, bias=False)
        nn.init.normal_(attn_offset_linear.weight, 0, 0.01)
        self.dq_predictor = nn.Sequential(
            ResidualFullyConnected(in_channels=width),
            attn_offset_linear,
        )

        self.occ_predictor = OccupancyPredictor(
            query_channels=self.encoded_query_dim,
            feature_channels=2 * f_channels,
            num_blocks=num_blocks,
            width=width,
        )

    def _normalize_roi(self, queries: torch.Tensor) -> torch.Tensor:
        queries_norm = queries.clone()
        denom = torch.clamp(self.roi_max - self.roi_min, min=1e-6)
        queries_norm[..., :3] = ((queries_norm[..., :3] - self.roi_min) / denom).clamp(0.0, 1.0)
        return queries_norm

    def forward(self, features: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """features: [B, C, H, W], queries: [B, N, >=3]."""
        if queries.shape[-1] < self.n_input_dims:
            raise ValueError(
                f"queries last dim must be >= {self.n_input_dims}, got {queries.shape[-1]}"
            )

        queries = self._normalize_roi(queries)
        queries_for_encoding = queries[..., : self.n_input_dims]
        encoded_queries = self.query_encoding(
            queries_for_encoding.reshape(-1, self.n_input_dims)
        ).view(
            queries.shape[0],
            queries.shape[1],
            self.encoded_query_dim,
        )

        q_points = queries[..., :2]
        zq_features = grid_sample(features, q_points, sampling_mode=self.sampling_mode)
        dq = self.dq_predictor(
            self.q_projection(encoded_queries) + self.zq_projection(zq_features)
        )
        r_points = q_points + dq

        zr_features = grid_sample(features, r_points, sampling_mode=self.sampling_mode)
        zf_features = torch.cat([zq_features, zr_features], dim=-1)
        return self.occ_predictor(encoded_queries, zf_features)
