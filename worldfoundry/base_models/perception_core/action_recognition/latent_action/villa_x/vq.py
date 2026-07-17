"""Vector quantizer required by Villa-X latent-action extraction."""

from __future__ import annotations

from torch import Tensor, nn


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float,
        **_: object,
    ) -> None:
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_e, e_dim)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        flattened = values.reshape(-1, self.e_dim)
        distances = (
            flattened.square().sum(dim=1, keepdim=True)
            + self.embedding.weight.square().sum(dim=1)
            - 2 * flattened @ self.embedding.weight.t()
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).reshape(values.shape)
        return quantized, indices
