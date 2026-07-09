"""Packed attention sequence metadata used by long-context inference runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass(frozen=True)
class PackedCoreAttnParams:
    q_range: torch.Tensor
    k_range: torch.Tensor
    np_q_range: np.ndarray
    np_k_range: np.ndarray
    max_seqlen_q: int
    max_seqlen_k: int


@dataclass(frozen=True)
class PackedCrossAttnParams:
    q_ranges: torch.Tensor | None = None
    kv_ranges: torch.Tensor | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_kv: torch.Tensor | None = None
    max_seqlen_q: int | None = None
    max_seqlen_kv: int | None = None


@dataclass(frozen=True)
class ModelMetaArgs:
    H: int
    W: int
    cp_pad_size: int
    cp_split_sizes: List[int]
    slice_point: int
    denoising_range_num: int
    range_num: int
    extract_prefix_video_feature: bool
    fwd_extra_1st_chunk: bool
    distill_nearly_clean_chunk: bool
    clip_token_nums: int
    enable_cuda_graph: bool
    core_attn_params: PackedCoreAttnParams
    cross_attn_params: PackedCrossAttnParams


__all__ = ["ModelMetaArgs", "PackedCoreAttnParams", "PackedCrossAttnParams"]
