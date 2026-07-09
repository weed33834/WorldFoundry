"""Visual-Semantic (VS) Similarity metric — WorldFoundry paper reimplementation."""

from __future__ import annotations

import numpy as np

from worldfoundry.evaluation.tasks.metrics.registry import metric_module_from_globals

from .wrapper import (
    compute_vs_similarity,
    compute_vs_similarity_from_scores,
    compute_vs_similarity_matrix,
)

METRIC_ID = "vs_similarity"
ALIASES = ("vs-similarity", "vs_sim", "visual_semantic_similarity", "hdgan_vs")
HIGHER_IS_BETTER = True
FAMILY = "scorer"
TAGS = ("scorer", "text_to_image", "hdgan", "paper_reimplementation")

METRIC_MODULE = metric_module_from_globals(
    metric_id=METRIC_ID,
    aliases=ALIASES,
    description=(
        "Visual-Semantic Similarity from HDGAN (arXiv:1802.09178): paired cosine similarity "
        "between image and sentence embeddings. WorldFoundry reimplementation from paper."
    ),
    family=FAMILY,
    higher_is_better=HIGHER_IS_BETTER,
    tags=TAGS,
)


def compute(image_features: np.ndarray, text_features: np.ndarray, **kwargs) -> float:
    return compute_vs_similarity(image_features, text_features, **kwargs)


__all__ = [
    "ALIASES",
    "FAMILY",
    "HIGHER_IS_BETTER",
    "METRIC_ID",
    "METRIC_MODULE",
    "compute",
    "compute_vs_similarity",
    "compute_vs_similarity_from_scores",
    "compute_vs_similarity_matrix",
]
