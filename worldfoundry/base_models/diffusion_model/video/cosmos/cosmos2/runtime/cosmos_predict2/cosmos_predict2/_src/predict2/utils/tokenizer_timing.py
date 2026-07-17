"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos2 -> runtime -> cosmos_predict2 -> cosmos_predict2 -> _src -> predict2 -> utils -> tokenizer_timing.py functionality."""

from dataclasses import dataclass


@dataclass
class TokenizerTimes:
    """Tokenizer times implementation."""

    model_invocation: float = 0.0
    total: float = 0.0


BenchmarkTimes = TokenizerTimes
