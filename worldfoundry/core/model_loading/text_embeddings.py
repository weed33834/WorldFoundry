"""Text-embedding configuration types."""

from enum import Enum


class EmbeddingConcatStrategy(str, Enum):
    FULL_CONCAT = "full_concat"
    MEAN_POOLING = "mean_pooling"
    POOL_EVERY_N_LAYERS_AND_CONCAT = "pool_every_n_layers_and_concat"

    def __str__(self) -> str:
        return self.value


__all__ = ["EmbeddingConcatStrategy"]
