from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.dpt_layers import DEFAULT_DPT_OUT_CHANNELS, DPT_LEVELS, DPTHead


class DPTWrapper(nn.Module):
    def __init__(
        self,
        input_channels: int = 1024,
        output_channels: int = 256,
        out_channels: Sequence[int] = DEFAULT_DPT_OUT_CHANNELS,
        pos_embed: bool = True,
        original_shape: Tuple[int, int] = (32, 32),
        output_shape: Tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()

        self.original_shape = original_shape
        self.output_shape = output_shape
        self.intermediate_shape = (output_shape[0] // 4, output_shape[1] // 4)
        self.fake_patch_size = 4

        self.dpt = DPTHead(
            dim_in=input_channels,
            patch_size=self.fake_patch_size,
            features=output_channels,
            out_channels=out_channels,
            intermediate_layer_idx=tuple(range(DPT_LEVELS)),
            pos_embed=pos_embed,
            feature_only=True,
            down_ratio=1,
        )

    def _resize_level_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, feat_dim = tokens.shape
        expected_tokens = self.original_shape[0] * self.original_shape[1]
        if num_tokens != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} tokens for original_shape={self.original_shape}, got {num_tokens}."
            )

        tokens_2d = tokens.permute(0, 2, 1).reshape(batch_size, feat_dim, self.original_shape[0], self.original_shape[1])
        tokens_2d = F.interpolate(tokens_2d, size=self.intermediate_shape, mode="bilinear", align_corners=True)
        return tokens_2d.permute(0, 2, 3, 1).reshape(batch_size, -1, feat_dim)

    def forward(self, bev_tokens: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(bev_tokens) != DPT_LEVELS:
            raise ValueError(f"Expected {DPT_LEVELS} feature levels for DPT, got {len(bev_tokens)}.")

        batch_size = bev_tokens[0].shape[0]
        feat_dim = bev_tokens[0].shape[-1]
        for idx, level_tokens in enumerate(bev_tokens):
            if level_tokens.shape[0] != batch_size:
                raise ValueError(
                    f"All levels must have same batch size. Level 0: {batch_size}, level {idx}: {level_tokens.shape[0]}"
                )
            if level_tokens.shape[-1] != feat_dim:
                raise ValueError(
                    f"All levels must have same embedding size. Level 0: {feat_dim}, level {idx}: {level_tokens.shape[-1]}"
                )

        resized_tokens = [self._resize_level_tokens(level_tokens).unsqueeze(1) for level_tokens in bev_tokens]
        images = torch.zeros(
            batch_size,
            1,
            3,
            self.output_shape[0],
            self.output_shape[1],
            device=bev_tokens[0].device,
            dtype=bev_tokens[0].dtype,
        )
        out = self.dpt(resized_tokens, images, 0)
        return out.squeeze(1)
