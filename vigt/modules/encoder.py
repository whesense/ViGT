from typing import Any

import torch
import torch.nn as nn


class ViGTEncoder(nn.Module):
    """Encode multi-camera images into BEV latents."""

    def __init__(
        self,
        cam_encoder: nn.Module,
        latent_encoder: nn.Module,
        latent_decoder: nn.Module,
    ):
        super().__init__()
        self.cam_encoder = cam_encoder
        self.latent_encoder = latent_encoder
        self.latent_decoder = latent_decoder

    def _get_multilayer_cam_feats(self, samples: list[Any]):
        camera_names_sorted = []
        front_camera_mask = []
        x_list = []
        for sample in samples:
            images = sample["images"] if isinstance(sample, dict) else sample.images
            front_camera_idx = (
                sample["front_camera_idx"] if isinstance(sample, dict) else sample.front_camera_idx
            )
            if front_camera_idx < 0 or front_camera_idx >= len(images):
                raise ValueError("`front_camera_idx` is out of range for provided `images`.")

            camera_names_sorted.append(list(range(len(images))))
            for cam_idx, image in enumerate(images):
                front_camera_mask.append(cam_idx == front_camera_idx)
                x_list.append(image[None] if image.ndim == 3 else image)

        feats = self.cam_encoder(x_list, front_camera_mask)
        multilayer_feats = []
        for layer_feats in feats:
            sample_feats = []
            feat_idx = 0
            for cam_names in camera_names_sorted:
                num_cams = len(cam_names)
                cam_feats = [layer_feats[feat_idx + i].squeeze(0) for i in range(num_cams)]
                sample_feats.append(torch.cat(cam_feats, dim=0))
                feat_idx += num_cams
            multilayer_feats.append(sample_feats)

        last_layer_feats = feats[-1]
        camera_tokens_last = []
        feat_idx = 0
        for cam_names in camera_names_sorted:
            num_cams = len(cam_names)
            cam_tokens = []
            for i in range(num_cams):
                cam_feat = last_layer_feats[feat_idx + i].squeeze()
                cam_tokens.append(cam_feat[-1])
            camera_tokens_last.append(torch.stack(cam_tokens, dim=0))
            feat_idx += num_cams

        return multilayer_feats, torch.stack(camera_tokens_last, dim=0)

    @staticmethod
    def _pad_multilayer_cam_feats(multilayer_feats_nested: list[list[torch.Tensor]]):
        multilayer_feats = [
            torch.nn.utils.rnn.pad_sequence(layer_feats, batch_first=True, padding_value=0)
            for layer_feats in multilayer_feats_nested
        ]
        seq_lens = [len(feat) for feat in multilayer_feats_nested[0]]
        max_seq_len = max(seq_lens)
        device = multilayer_feats_nested[0][0].device
        pad_mask = torch.arange(max_seq_len, device=device) > torch.tensor(seq_lens, device=device)[:, None]
        return multilayer_feats, pad_mask

    def encode(self, samples: list[Any]):
        multilayer_feats, camera_tokens = self._get_multilayer_cam_feats(samples)
        multilayer_feats, pad_mask = self._pad_multilayer_cam_feats(multilayer_feats)
        latent_array, _camera_tokens = self.latent_encoder(
            multilayer_feats,
            cam_embs=None,
            pad_mask=pad_mask,
            camera_tokens=camera_tokens,
        )
        return latent_array

    def forward(self, samples: list[Any]):
        latent_array = self.encode(samples)
        return self.latent_decoder(latent_array)
