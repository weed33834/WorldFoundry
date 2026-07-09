"""Canonical names and validation helpers for benchmark execution run modes.

This module exposes the defined modes supported by the benchmark execution engine,
distinguishing between lightweight contract/mock evaluations and full official
benchmark executions.
"""

from __future__ import annotations

# Modes where execution runs a fast contract-based synthetic generation (e.g. for testing and validation)
BENCHMARK_RUN_CONTRACT_MODES = frozenset({"contract"})

# Modes focusing strictly on normalising and parsing pre-existing raw results into WorldFoundry scorecards
BENCHMARK_RUN_NORMALIZER_MODES = frozenset({"normalizer"})

# Modes executing official benchmark procedures, ranging from validation checks to full evaluation runs
BENCHMARK_RUN_OFFICIAL_MODES = BENCHMARK_RUN_NORMALIZER_MODES | frozenset({"official-validation", "official-run"})

# Modes exposed through public user-facing CLI surfaces.
BENCHMARK_RUN_PUBLIC_MODES = BENCHMARK_RUN_OFFICIAL_MODES

# Union of all valid execution modes recognized by the WorldFoundry evaluation runner.
# ``contract`` remains accepted for narrow internal test fixtures, but public
# commands should route users to runnable official modes.
BENCHMARK_RUN_SUPPORTED_MODES = BENCHMARK_RUN_CONTRACT_MODES | BENCHMARK_RUN_OFFICIAL_MODES


def normalize_benchmark_run_mode(value: str) -> str:
    """Validates and returns a normalized canonical benchmark run mode.

    Args:
        value: The raw string representation of the benchmark execution mode.

    Returns:
        The validated, canonical mode string recognized by the execution pipeline.

    Raises:
        ValueError: If the provided mode is not in the supported set of modes.
    """

    if value not in BENCHMARK_RUN_SUPPORTED_MODES:
        supported = ", ".join(sorted(BENCHMARK_RUN_SUPPORTED_MODES))
        raise ValueError(f"benchmark run mode must be one of: {supported}")
    return value


__all__ = [
    "BENCHMARK_RUN_CONTRACT_MODES",
    "BENCHMARK_RUN_NORMALIZER_MODES",
    "BENCHMARK_RUN_OFFICIAL_MODES",
    "BENCHMARK_RUN_PUBLIC_MODES",
    "BENCHMARK_RUN_SUPPORTED_MODES",
    "normalize_benchmark_run_mode",
]
