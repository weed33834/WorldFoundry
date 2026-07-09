"""Module for base_models -> diffusion_model -> diffsynth -> models -> flux_text_encoder.py functionality."""

import torch
from transformers import T5EncoderModel, T5Config
from .sd_text_encoder import SDTextEncoder



class FluxTextEncoder2(T5EncoderModel):
    """Flux text encoder implementation."""
    def __init__(self, config):
        """Init.

        Args:
            config: The config.
        """
        super().__init__(config)
        self.eval()

    def forward(self, input_ids):
        """Forward.

        Args:
            input_ids: The input ids.
        """
        outputs = super().forward(input_ids=input_ids)
        prompt_emb = outputs.last_hidden_state
        return prompt_emb

    @staticmethod
    def state_dict_converter():
        """State dict converter."""
        return FluxTextEncoder2StateDictConverter()



class FluxTextEncoder2StateDictConverter():
    """Flux text encoder state dict converter implementation."""
    def __init__(self):
        """Init."""
        pass

    def from_diffusers(self, state_dict):
        """From diffusers.

        Args:
            state_dict: The state dict.
        """
        state_dict_ = state_dict
        return state_dict_

    def from_civitai(self, state_dict):
        """From civitai.

        Args:
            state_dict: The state dict.
        """
        return self.from_diffusers(state_dict)
