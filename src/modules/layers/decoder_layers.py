import torch
from torch import nn

FloatTensor = torch.Tensor


def make_linear(
    in_features: int,
    out_features: int,
    relu: bool = False,
    bias: bool = True,
) -> nn.Module:
    layer = nn.Linear(in_features, out_features, bias=bias)
    if relu:
        return nn.Sequential(layer, nn.ReLU(inplace=True))
    return layer


class ResidualFullyConnected(nn.Module):
    """Residual fully connected block used by occupancy decoders."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        width: int | None = None,
        zero_init: bool = True,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        width = width or min(in_channels, out_channels)

        self.fc1 = make_linear(in_channels, width, relu=False)
        self.fc2 = make_linear(width, out_channels, relu=False)
        self.relu = nn.ReLU(inplace=True)
        self.skip = None
        if in_channels != out_channels:
            self.skip = make_linear(in_channels, out_channels, relu=False, bias=False)

        self.reset_parameters(zero_init=zero_init)

    def reset_parameters(self, zero_init: bool = True):
        if zero_init:
            nn.init.zeros_(self.fc2.weight)

    def forward(self, x: FloatTensor) -> FloatTensor:
        dx = self.fc1(self.relu(x))
        dx = self.fc2(self.relu(dx))
        if self.skip is not None:
            x = self.skip(x)
        return x + dx


class OccupancyPredictor(nn.Module):
    """Implicit occupancy decoder from query points and latent features."""

    def __init__(
        self,
        query_channels: int = 4,
        feature_channels: int = 128,
        num_blocks: int = 3,
        width: int = 16,
    ):
        super().__init__()
        self.query_projection = make_linear(query_channels, width, relu=False)
        self.feature_projections = nn.ModuleList(
            [make_linear(feature_channels, width, relu=False) for _ in range(num_blocks)]
        )
        self.residuals = nn.ModuleList(
            [ResidualFullyConnected(in_channels=width) for _ in range(num_blocks)]
        )
        self.occupancy = make_linear(width, 1, relu=False)
        self.relu = nn.ReLU()

    def forward(self, queries: FloatTensor, features: FloatTensor) -> FloatTensor:
        qproj = self.query_projection(queries)
        for fproj, residual in zip(self.feature_projections, self.residuals):
            qproj = qproj + fproj(features)
            qproj = residual(qproj)
        return self.occupancy(self.relu(qproj))
