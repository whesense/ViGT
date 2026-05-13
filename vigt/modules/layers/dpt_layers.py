from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_DPT_OUT_CHANNELS = (256, 512, 1024, 1024)
DEFAULT_DPT_LAYER_IDX = (4, 11, 17, 23)
DPT_LEVELS = 4


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100.0) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega = 1.0 / (omega_0 ** (omega / (embed_dim / 2.0)))
    out = torch.einsum("m,d->md", pos.reshape(-1), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100.0) -> torch.Tensor:
    h, w, grid_dim = pos_grid.shape
    if grid_dim != 2:
        raise ValueError(f"Expected last dim=2 for position grid, got {grid_dim}")
    flat = pos_grid.reshape(-1, grid_dim)
    emb_x = make_sincos_pos_embed(embed_dim // 2, flat[:, 0], omega_0=omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, flat[:, 1], omega_0=omega_0)
    return torch.cat([emb_x, emb_y], dim=-1).view(h, w, embed_dim)


def create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    diag = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag
    span_y = 1.0 / diag
    x_coords = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, steps=height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    return torch.stack((uu, vv), dim=-1)


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    if size is None:
        if scale_factor is None:
            raise ValueError("Either size or scale_factor must be provided.")
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    int_max = 1610612736
    elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if elements <= int_max:
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)

    chunks = torch.chunk(x, chunks=(elements // int_max) + 1, dim=0)
    resized = [F.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks]
    return torch.cat(resized, dim=0).contiguous()


def activate_head(out: torch.Tensor, activation: str = "norm_exp", conf_activation: str = "expp1") -> Tuple[torch.Tensor, torch.Tensor]:
    fmap = out.permute(0, 2, 3, 1)
    xyz = fmap[:, :, :, :-1]
    conf = fmap[:, :, :, -1]

    if activation == "norm_exp":
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pts3d = (xyz / d) * torch.expm1(d)
    elif activation == "norm":
        pts3d = xyz / xyz.norm(dim=-1, keepdim=True)
    elif activation == "exp":
        pts3d = torch.exp(xyz)
    elif activation == "relu":
        pts3d = F.relu(xyz)
    elif activation == "inv_log":
        pts3d = torch.sign(xyz) * torch.expm1(torch.abs(xyz))
    elif activation == "xy_inv_log":
        xy, z = xyz.split([2, 1], dim=-1)
        z = torch.sign(z) * torch.expm1(torch.abs(z))
        pts3d = torch.cat([xy * z, z], dim=-1)
    elif activation == "sigmoid":
        pts3d = torch.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")

    if conf_activation == "expp1":
        conf_out = 1 + conf.exp()
    elif conf_activation == "expp0":
        conf_out = conf.exp()
    elif conf_activation == "sigmoid":
        conf_out = torch.sigmoid(conf)
    else:
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    return pts3d, conf_out


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation: nn.Module, groups: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=groups)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.activation(x))
        out = self.conv2(self.activation(out))
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features: int,
        activation: nn.Module,
        align_corners: bool = True,
        size: Tuple[int, int] | None = None,
        has_residual: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.size = size
        self.has_residual = has_residual
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True, groups=groups)
        self.res1 = ResidualConvUnit(features, activation, groups=groups) if has_residual else None
        self.res2 = ResidualConvUnit(features, activation, groups=groups)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs: torch.Tensor, size: Tuple[int, int] | None = None) -> torch.Tensor:
        output = xs[0]
        if self.has_residual and self.res1 is not None:
            output = self.skip_add.add(output, self.res1(xs[1]))
        output = self.res2(output)
        target_size = size if size is not None else self.size
        kwargs = {"size": target_size} if target_size is not None else {"scale_factor": 2}
        output = custom_interpolate(output, **kwargs, mode="bilinear", align_corners=self.align_corners)
        return self.out_conv(output)


def _make_scratch(in_shape: Sequence[int], out_shape: int, groups: int = 1) -> nn.Module:
    if len(in_shape) < DPT_LEVELS:
        raise ValueError(f"Expected {DPT_LEVELS} input feature scales for DPT scratch network.")
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch


class DPTHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = DEFAULT_DPT_OUT_CHANNELS,
        intermediate_layer_idx: Sequence[int] = DEFAULT_DPT_LAYER_IDX,
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {patch_size}")
        if down_ratio <= 0:
            raise ValueError(f"down_ratio must be > 0, got {down_ratio}")
        if len(out_channels) != DPT_LEVELS:
            raise ValueError(f"out_channels must have {DPT_LEVELS} values, got {len(out_channels)}")
        if len(intermediate_layer_idx) != DPT_LEVELS:
            raise ValueError(
                f"intermediate_layer_idx must have {DPT_LEVELS} values, got {len(intermediate_layer_idx)}"
            )

        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = list(intermediate_layer_idx)
        self.out_channels = list(out_channels)

        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0) for oc in self.out_channels])
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(self.out_channels[0], self.out_channels[0], kernel_size=4, stride=4, padding=0),
                nn.ConvTranspose2d(self.out_channels[1], self.out_channels[1], kernel_size=2, stride=2, padding=0),
                nn.Identity(),
                nn.Conv2d(self.out_channels[3], self.out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )

        self.scratch = _make_scratch(self.out_channels, features)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = FeatureFusionBlock(features, nn.ReLU(inplace=True))
        self.scratch.refinenet2 = FeatureFusionBlock(features, nn.ReLU(inplace=True))
        self.scratch.refinenet3 = FeatureFusionBlock(features, nn.ReLU(inplace=True))
        self.scratch.refinenet4 = FeatureFusionBlock(features, nn.ReLU(inplace=True), has_residual=False)

        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, output_dim, kernel_size=1, stride=1, padding=0),
            )

    def forward(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int | None = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if images.ndim != 5:
            raise ValueError(f"Expected images of shape [B, S, 3, H, W], got ndim={images.ndim}")
        _, seq_len, _, _, _ = images.shape
        if frames_chunk_size is None or frames_chunk_size >= seq_len:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)
        if frames_chunk_size <= 0:
            raise ValueError("frames_chunk_size must be > 0.")

        preds_all = []
        conf_all = []
        for start in range(0, seq_len, frames_chunk_size):
            end = min(start + frames_chunk_size, seq_len)
            chunk = self._forward_impl(aggregated_tokens_list, images, patch_start_idx, start, end)
            if self.feature_only:
                preds_all.append(chunk)
            else:
                pred_chunk, conf_chunk = chunk
                preds_all.append(pred_chunk)
                conf_all.append(conf_chunk)

        if self.feature_only:
            return torch.cat(preds_all, dim=1)
        return torch.cat(preds_all, dim=1), torch.cat(conf_all, dim=1)

    def _extract_level_tokens(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        layer_idx: int,
        patch_start_idx: int,
        frames_start_idx: int | None = None,
        frames_end_idx: int | None = None,
    ) -> torch.Tensor:
        if layer_idx >= len(aggregated_tokens_list):
            raise IndexError(
                f"Requested layer_idx={layer_idx}, but got only {len(aggregated_tokens_list)} feature tensors."
            )
        x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
        if frames_start_idx is not None and frames_end_idx is not None:
            x = x[:, frames_start_idx:frames_end_idx]
        return x

    def _forward_impl(
        self,
        aggregated_tokens_list: Sequence[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int | None = None,
        frames_end_idx: int | None = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        batch_size, seq_len, _, height, width = images.shape
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size

        outs = []
        for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = self._extract_level_tokens(
                aggregated_tokens_list=aggregated_tokens_list,
                layer_idx=layer_idx,
                patch_start_idx=patch_start_idx,
                frames_start_idx=frames_start_idx,
                frames_end_idx=frames_end_idx,
            )

            x = x.view(batch_size * seq_len, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, width, height)
            x = self.resize_layers[dpt_idx](x)
            outs.append(x)

        out = self.scratch_forward(outs)
        out = custom_interpolate(
            out,
            size=(int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, width, height)

        if self.feature_only:
            return out.view(batch_size, seq_len, *out.shape[1:])

        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)
        return preds.view(batch_size, seq_len, *preds.shape[1:]), conf.view(batch_size, seq_len, *conf.shape[1:])

    def _apply_pos_embed(self, x: torch.Tensor, width: int, height: int, ratio: float = 0.1) -> torch.Tensor:
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=width / height, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1]) * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        out = self.scratch.refinenet1(out, layer_1_rn)
        return self.scratch.output_conv1(out)
