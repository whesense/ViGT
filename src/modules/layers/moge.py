import logging
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from moge.model.v1 import MoGeModel as MoGeModel_v1

logger = logging.getLogger(__name__)


class MoGeBackbone(nn.Module):
    """MoGe ViT-L wrapper with intermediate token extraction for multi-camera input."""

    def __init__(
        self,
        model_name: str = "Ruicheng/moge-vitl",
        intermediate_layers: Union[Sequence[int], int] = (20, 21, 22, 23),
        add_cls_tokens: bool = True,
    ) -> None:
        super().__init__()
        logger.info("Loading MoGe model from %s", model_name)
        model = MoGeModel_v1.from_pretrained(model_name)
        self.backbone = model.backbone

        self.num_tokens_range = model.num_tokens_range
        self.intermediate_layers = self._normalize_intermediate_layers(intermediate_layers)
        self.num_layers = len(self.intermediate_layers) if isinstance(self.intermediate_layers, list) else self.intermediate_layers
        self.patch_size = self.backbone.patch_size
        self.add_cls_tokens = add_cls_tokens

        self.backbone.mask_token.requires_grad = False
        self.init_cls_token_from_pretrained()

    @staticmethod
    def _normalize_intermediate_layers(intermediate_layers: Union[Sequence[int], int]) -> Union[list[int], int]:
        if isinstance(intermediate_layers, int):
            if intermediate_layers <= 0:
                raise ValueError("intermediate_layers must be > 0 when passed as int.")
            return intermediate_layers

        layers = list(intermediate_layers)
        if not layers:
            raise ValueError("intermediate_layers sequence cannot be empty.")
        return layers

    def init_cls_token_from_pretrained(self) -> None:
        original_cls_token = self.backbone.cls_token
        front_cls = original_cls_token.clone()
        other_cls = original_cls_token.clone()

        differentiation_scale = 0.001
        front_cls = front_cls + torch.randn_like(front_cls) * differentiation_scale
        other_cls = other_cls + torch.randn_like(other_cls) * differentiation_scale

        new_cls_tokens = torch.cat([front_cls, other_cls], dim=1)
        self.cls_token = nn.Parameter(new_cls_tokens)
        self.backbone.cls_token.requires_grad = False

    def _resize_images(self, x_list: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        resized = []
        for img in x_list:
            original_height, original_width = img.shape[-2:]
            img_14 = F.interpolate(
                img,
                (original_height // 14 * 14, original_width // 14 * 14),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized.append(img_14)
        return resized

    @staticmethod
    def _get_front_camera_mask(
        x_list: Sequence[torch.Tensor], front_camera_mask: Optional[Sequence[bool]]
    ) -> list[bool]:
        if front_camera_mask is None:
            return [False] * len(x_list)
        if len(front_camera_mask) != len(x_list):
            raise ValueError("front_camera_mask length must match x_list length.")
        return list(front_camera_mask)

    def _get_intermediate_layers_list(
        self,
        x_list: Sequence[torch.Tensor],
        n: Union[Sequence[int], int],
        return_class_token: bool = False,
        norm: bool = True,
        front_camera_mask: Optional[Sequence[bool]] = None,
    ):
        front_camera_mask = self._get_front_camera_mask(x_list, front_camera_mask)
        x = [self.prepare_tokens_with_masks(img, is_front_camera=is_front) for img, is_front in zip(x_list, front_camera_mask)]

        total_block_len = len(self.backbone.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n

        outputs = []
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in blocks_to_take:
                layer_results = []
                for tokens in x:
                    tokens_norm = self.backbone.norm(tokens) if norm else tokens
                    cls_token = tokens_norm[:, 0:1]
                    patch_tokens = tokens_norm[:, 1 + self.backbone.num_register_tokens :]

                    if return_class_token:
                        layer_results.append((patch_tokens, cls_token))
                    else:
                        layer_results.append(patch_tokens)

                outputs.append(layer_results)

        if return_class_token:
            return tuple(
                ([result[0] for result in layer_results], [result[1] for result in layer_results]) for layer_results in outputs
            )
        return tuple(outputs)

    def prepare_tokens_with_masks(self, x: torch.Tensor, is_front_camera: bool = False) -> torch.Tensor:
        _, _, img_h, img_w = x.shape
        x = self.backbone.patch_embed(x)

        cls_token = self.cls_token[:, 1:, :] if is_front_camera else self.cls_token[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.backbone.interpolate_pos_encoding(x, img_h, img_w)

        if self.backbone.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.backbone.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x

    def forward(
        self,
        x_list: Sequence[torch.Tensor],
        front_camera_mask: Optional[Sequence[bool]] = None,
    ):
        """Extract intermediate token features for each input camera image."""
        x_list = self._resize_images(x_list)
        if self.add_cls_tokens:
            layer_outs = self._get_intermediate_layers_list(
                x_list=x_list,
                n=self.intermediate_layers,
                return_class_token=True,
                norm=True,
                front_camera_mask=front_camera_mask,
            )
            features = []
            for feature_maps, cls_tokens in layer_outs:
                combined_feats = [torch.cat([feat_map, cls_tok], dim=-2) for feat_map, cls_tok in zip(feature_maps, cls_tokens)]
                features.append(combined_feats)
            return features

        return self._get_intermediate_layers_list(
            x_list=x_list,
            n=self.intermediate_layers,
            return_class_token=False,
            norm=True,
            front_camera_mask=front_camera_mask,
        )
