"""Module for base_models -> diffusion_model -> diffsynth -> prompters -> cog_prompter.py functionality."""

from .base_prompter import BasePrompter
from ..models.flux_text_encoder import FluxTextEncoder2
from transformers import T5TokenizerFast
import os


class CogPrompter(BasePrompter):
    """Cog prompter implementation."""
    def __init__(
        self,
        tokenizer_path=None
    ):
        """Init.

        Args:
            tokenizer_path: The tokenizer path.
        """
        if tokenizer_path is None:
            base_path = os.path.dirname(os.path.dirname(__file__))
            tokenizer_path = os.path.join(base_path, "tokenizer_configs/cog/tokenizer")
        super().__init__()
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)
        self.text_encoder: FluxTextEncoder2 = None


    def fetch_models(self, text_encoder: FluxTextEncoder2 = None):
        """Fetch models.

        Args:
            text_encoder: The text encoder.
        """
        self.text_encoder = text_encoder


    def encode_prompt_using_t5(self, prompt, text_encoder, tokenizer, max_length, device):
        """Encode prompt using t5.

        Args:
            prompt: The prompt.
            text_encoder: The text encoder.
            tokenizer: The tokenizer.
            max_length: The max length.
            device: The device.
        """
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        ).input_ids.to(device)
        prompt_emb = text_encoder(input_ids)
        prompt_emb = prompt_emb.reshape((1, prompt_emb.shape[0]*prompt_emb.shape[1], -1))

        return prompt_emb
    

    def encode_prompt(
        self,
        prompt,
        positive=True,
        device="cuda"
    ):
        """Encode prompt.

        Args:
            prompt: The prompt.
            positive: The positive.
            device: The device.
        """
        prompt = self.process_prompt(prompt, positive=positive)
        prompt_emb = self.encode_prompt_using_t5(prompt, self.text_encoder, self.tokenizer, 226, device)
        return prompt_emb
