import torch
import torch.nn as nn


class Sequential(nn.Sequential):
    def forward(self, *inputs, **kwargs):
        for module in self:
            if isinstance(inputs, tuple):
                inputs = module(*inputs, **kwargs)
            else:
                inputs = module(inputs, **kwargs)
        return inputs


def mlp(num_channels: int) -> Sequential:
    return Sequential(
        nn.LayerNorm(num_channels),
        nn.Linear(num_channels, num_channels),
        nn.GELU(),
        nn.Linear(num_channels, num_channels),
    )


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 0.0):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((dim,)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class LayerScaleModule(nn.Module):
    def __init__(self, module: nn.Module, dim: int, init_value: float = 0.0):
        super().__init__()
        self.module = module
        self.layer_scale = LayerScale(dim, init_value) if init_value > 0 else nn.Identity()

    def forward(self, *args, **kwargs):
        return self.layer_scale(self.module(*args, **kwargs))


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs) + args[0]


class MultiHeadAttention(nn.Module):
    def __init__(self, num_q_channels: int, num_kv_channels: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, pad_mask=None, attn_mask=None) -> torch.Tensor:
        return self.attention(
            x_q,
            x_kv,
            x_kv,
            average_attn_weights=False,
            key_padding_mask=pad_mask,
            attn_mask=attn_mask,
        )[0]


class CrossAttention(nn.Module):
    def __init__(self, num_q_channels: int, num_kv_channels: int, num_heads: int):
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attention = MultiHeadAttention(
            num_q_channels=num_q_channels,
            num_kv_channels=num_kv_channels,
            num_heads=num_heads,
        )

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, pad_mask=None, attn_mask=None) -> torch.Tensor:
        return self.attention(self.q_norm(x_q), self.kv_norm(x_kv), pad_mask=pad_mask, attn_mask=attn_mask)


class SelfAttention(nn.Module):
    def __init__(self, num_channels: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_q_channels=num_channels,
            num_kv_channels=num_channels,
            num_heads=num_heads,
        )

    def forward(self, x: torch.Tensor, pad_mask=None, attn_mask=None) -> torch.Tensor:
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)


def cross_attention_layer(
    num_q_channels: int,
    num_kv_channels: int,
    num_heads: int,
    scale_init: float = 0.0,
    residual_ca: bool = True,
):
    if residual_ca:
        ca = Residual(
            LayerScaleModule(
                CrossAttention(num_q_channels, num_kv_channels, num_heads),
                num_q_channels,
                scale_init,
            )
        )
    else:
        ca = CrossAttention(num_q_channels, num_kv_channels, num_heads)

    return Sequential(
        ca,
        Residual(LayerScaleModule(mlp(num_q_channels), num_q_channels, scale_init)),
    )


def self_attention_layer(
    num_channels: int,
    num_heads: int,
    scale_init: float = 0.0,
):
    return Sequential(
        Residual(
            LayerScaleModule(SelfAttention(num_channels, num_heads), num_channels, scale_init),
        ),
        Residual(LayerScaleModule(mlp(num_channels), num_channels, scale_init)),
    )


def self_attention_block(
    num_layers: int,
    num_channels: int,
    num_heads: int,
    scale_init: float = 0.0,
):
    return Sequential(
        *[
            self_attention_layer(
                num_channels=num_channels,
                num_heads=num_heads,
                scale_init=scale_init,
            )
            for _ in range(num_layers)
        ]
    )
