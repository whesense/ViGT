from typing import Any, Callable

import torch
import torch.nn as nn


class ViGT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        roi: Any = None,
    ):
        super().__init__()
        self.roi = roi
        self.encoder = encoder
        self.decoder = decoder

        if self.encoder is None or self.decoder is None:
            raise ValueError("ViGT requires both `encoder` and `decoder`.")

        if self.roi is not None:
            if hasattr(self.encoder, "roi"):
                self.encoder.roi = self.roi
            if hasattr(self.decoder, "roi"):
                self.decoder.roi = self.roi

    def encode(self, samples: list[Any] | Any) -> torch.Tensor:
        if not isinstance(samples, list):
            samples = [samples]
        return self.encoder(samples)

    def decode(self, context: torch.Tensor, queries: torch.Tensor):
        return self.decoder(context, queries)

    def forward(self, samples: list[Any] | Any, queries: torch.Tensor):
        encodings = self.encode(samples)
        return self.decode(encodings, queries)

    def build_occupancy_fn(
        self,
        samples: list[Any] | Any,
        chunk: int = 2**17,
    ) -> Callable[..., torch.Tensor]:
        context = self.encode(samples)

        def occupancy_fn(queries: torch.Tensor):
            results = []
            for i in range(0, len(queries), chunk):
                chunk_queries = queries[i : i + chunk]
                logits = self.decode(context, chunk_queries.view(1, -1, queries.shape[-1]))
                densities = logits.sigmoid().view(-1)

                # Remove points inside ego box (ego coordinates).
                x = chunk_queries[:, 0]
                y = chunk_queries[:, 1]
                in_box = (x > -1.0) & (x < 3.5) & (torch.abs(y) < 1.0)
                densities[in_box] = 0.0

                results.append(densities)
            return torch.cat(results, dim=0)

        return occupancy_fn
