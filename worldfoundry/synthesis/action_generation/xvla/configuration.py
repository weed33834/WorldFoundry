"""Hugging Face configuration for the in-tree X-VLA runtime."""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig

from .configuration_florence2 import Florence2Config


class XVLAConfig(PretrainedConfig):
    """Checkpoint-compatible X-VLA architecture configuration."""

    model_type = "xvla"

    def __init__(
        self,
        florence_config: dict[str, Any] | Florence2Config | None = None,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 30,
        len_soft_prompts: int = 32,
        dim_time: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
        soft_prompt_length: int = 32,
        max_action_dim: int = 20,
        real_action_dim: int = 20,
        num_actions: int = 30,
        action_mode: str = "ee6d",
        use_proprio: bool = True,
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        if isinstance(florence_config, dict):
            self.florence_config = Florence2Config(**florence_config)
        elif isinstance(florence_config, Florence2Config):
            self.florence_config = florence_config
        else:
            self.florence_config = Florence2Config()
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.num_domains = int(num_domains)
        self.len_soft_prompts = int(len_soft_prompts)
        self.dim_time = int(dim_time)
        self.max_len_seq = int(max_len_seq)
        self.use_hetero_proj = bool(use_hetero_proj)
        self.soft_prompt_length = int(soft_prompt_length)
        self.max_action_dim = int(max_action_dim)
        self.real_action_dim = int(real_action_dim)
        self.num_actions = int(num_actions)
        self.action_mode = str(action_mode)
        self.use_proprio = bool(use_proprio)
        # The released Florence encoder stores a single shared embedding and
        # omits its encoder alias.  Keep this explicit on the outer composite
        # config so Transformers 5 expands XVLA's fully-qualified tie mapping.
        super().__init__(tie_word_embeddings=bool(tie_word_embeddings), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output["florence_config"] = self.florence_config.to_dict()
        return output


__all__ = ["XVLAConfig"]
