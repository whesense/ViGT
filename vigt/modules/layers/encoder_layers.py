from typing import Optional, Sequence

import torch
from torch import nn

from .attention_layers import cross_attention_layer, self_attention_block


class BevProjector(nn.Module):
    """Project image patch tokens to BEV latent queries."""

    def __init__(
        self,
        input_channels: int = 1024,
        feature_size: int = 1024,
        bev_shape: tuple[int, int] = (32, 32),
        use_sa: bool = False,
    ) -> None:
        super().__init__()
        bev_h, bev_w = bev_shape
        self.num_bev_queries = bev_h * bev_w
        self.bev_queries = nn.Parameter(torch.randn(self.num_bev_queries, feature_size))

        self.attn_1 = cross_attention_layer(
            num_q_channels=feature_size,
            num_kv_channels=input_channels,
            num_heads=16,
            residual_ca=True,
        )
        self.attn_2 = cross_attention_layer(
            num_q_channels=feature_size,
            num_kv_channels=input_channels,
            num_heads=8,
            residual_ca=True,
        )
        self.sa_block = (
            self_attention_block(
                num_layers=1,
                num_channels=feature_size,
                num_heads=8,
            )
            if use_sa
            else None
        )

    def forward(
        self,
        img_patches: torch.Tensor,
        pad_mask: Optional[torch.Tensor],
        additional_queries: Optional[torch.Tensor] = None,
    ):
        batch_size = img_patches.shape[0]
        bev_queries = self.bev_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = torch.cat([bev_queries, additional_queries], dim=-2) if additional_queries is not None else bev_queries

        features = self.attn_1(queries, img_patches, pad_mask)
        if self.sa_block is not None:
            features = self.sa_block(features)
        features = self.attn_2(features, img_patches, pad_mask)

        if additional_queries is None:
            return features

        bev_features = features[:, : self.num_bev_queries]
        additional_features = features[:, self.num_bev_queries :]
        return bev_features, additional_features


class MultiLayerBevProjector(nn.Module):
    """Apply BEV projection across multiple backbone layers."""

    def __init__(
        self,
        num_layers: int = 4,
        input_channels: int = 1024,
        feature_size: int = 1024,
        bev_shape: tuple[int, int] = (32, 32),
        use_sa: bool = False,
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleList(
            [
                BevProjector(
                    input_channels=input_channels,
                    feature_size=feature_size,
                    bev_shape=bev_shape,
                    use_sa=use_sa,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        multilayer_feats: Sequence[torch.Tensor],
        cam_embs: Optional[torch.Tensor],
        pad_mask: Optional[torch.Tensor],
        camera_tokens: Optional[torch.Tensor] = None,
    ):
        if len(multilayer_feats) != len(self.projectors):
            raise ValueError(
                f"Expected {len(self.projectors)} feature levels, got {len(multilayer_feats)}."
            )

        bev_features = []
        for layer_feats, projector in zip(multilayer_feats, self.projectors):
            feat = torch.cat([layer_feats, cam_embs], dim=-1) if cam_embs is not None else layer_feats
            if camera_tokens is not None:
                bev_feat, camera_tokens = projector(feat, pad_mask, camera_tokens)
            else:
                bev_feat = projector(feat, pad_mask)
            bev_features.append(bev_feat)

        if camera_tokens is None:
            return bev_features
        return bev_features, camera_tokens
