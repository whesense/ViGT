from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

from .modules.dpt import DPTWrapper
from .modules.encoder import ViGTEncoder
from .modules.implicit_decoder import ImplicitDecoder
from .modules.layers.encoder_layers import MultiLayerBevProjector
from .modules.layers.moge import MoGeBackbone
from .vigt import ViGT


@dataclass
class ViGTHFConfig:
    # Backbone
    backbone_model_name: str = "Ruicheng/moge-vitl"
    intermediate_layers: list[int] = field(default_factory=lambda: [20, 21, 22, 23])
    add_cls_tokens: bool = True

    # Cam2BEV
    cam2bev_num_layers: int = 4
    cam2bev_input_channels: int = 1024
    cam2bev_feature_size: int = 1024
    cam2bev_bev_shape: tuple[int, int] = (32, 32)
    cam2bev_use_sa: bool = False

    # DPT latent decoder
    dpt_input_channels: int = 1024
    dpt_output_channels: int = 256
    dpt_out_channels: tuple[int, int, int, int] = (256, 512, 1024, 1024)
    dpt_original_shape: tuple[int, int] = (32, 32)
    dpt_output_shape: tuple[int, int] = (256, 256)
    dpt_pos_embed: bool = True

    # Implicit decoder
    implicit_f_channels: int = 256
    implicit_width: int = 32
    implicit_num_blocks: int = 5
    implicit_sampling_mode: str = "bilinear"
    implicit_n_input_dims: int = 3
    roi_min: tuple[float, float, float] = (-40.0, -40.0, -1.0)
    roi_max: tuple[float, float, float] = (40.0, 40.0, 5.4)


class ViGTForInference(nn.Module):
    """HF-style wrapper"""

    config_name = "config.json"
    weights_name = "model.safetensors"

    def __init__(self, config: ViGTHFConfig):
        super().__init__()
        self.config = config
        self.model = self._build_model(config)

    @staticmethod
    def _build_model(config: ViGTHFConfig) -> ViGT:
        encoder = ViGTEncoder(
            cam_encoder=MoGeBackbone(
                model_name=config.backbone_model_name,
                intermediate_layers=config.intermediate_layers,
                add_cls_tokens=config.add_cls_tokens,
            ),
            latent_encoder=MultiLayerBevProjector(
                num_layers=config.cam2bev_num_layers,
                input_channels=config.cam2bev_input_channels,
                feature_size=config.cam2bev_feature_size,
                bev_shape=config.cam2bev_bev_shape,
                use_sa=config.cam2bev_use_sa,
            ),
            latent_decoder=DPTWrapper(
                input_channels=config.dpt_input_channels,
                output_channels=config.dpt_output_channels,
                out_channels=config.dpt_out_channels,
                pos_embed=config.dpt_pos_embed,
                original_shape=config.dpt_original_shape,
                output_shape=config.dpt_output_shape,
            ),
        )

        decoder = ImplicitDecoder(
            f_channels=config.implicit_f_channels,
            width=config.implicit_width,
            num_blocks=config.implicit_num_blocks,
            sampling_mode=config.implicit_sampling_mode,
            n_input_dims=config.implicit_n_input_dims,
            roi_min=config.roi_min,
            roi_max=config.roi_max,
        )

        roi = {"min": config.roi_min, "max": config.roi_max}
        return ViGT(encoder=encoder, decoder=decoder, roi=roi)

    def forward(self, samples, queries):
        return self.model(samples, queries)

    def save_pretrained(self, save_directory: str | Path):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / self.config_name, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)
        safetensors_save_file(self.state_dict(), str(save_dir / self.weights_name))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        device: Optional[str | torch.device] = None,
    ) -> "ViGTForInference":
        root = Path(pretrained_path)
        config_path = root / cls.config_name
        weights_path = root / cls.weights_name

        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights file: {weights_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        model = cls(config=ViGTHFConfig(**cfg_dict))

        state_dict = safetensors_load_file(str(weights_path), device=str(map_location))
        model.load_state_dict(state_dict, strict=strict)

        if device is not None:
            model = model.to(device)
        model.eval()
        return model


def init_vigt_model(
    weights_path: str | Path,
    device: Optional[str | torch.device] = None,
    strict: bool = True,
) -> ViGTForInference:
    model = ViGTForInference.from_pretrained(
        pretrained_path=weights_path,
        map_location="cpu",
        strict=strict,
        device=device,
    )
    model.eval()
    return model
