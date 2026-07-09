"""Module for base_models -> diffusion_model -> diffsynth -> models -> sdxl_vae_decoder.py functionality."""

from .sd_vae_decoder import SDVAEDecoder, SDVAEDecoderStateDictConverter


class SDXLVAEDecoder(SDVAEDecoder):
    """Sdxlvae decoder implementation."""
    def __init__(self, upcast_to_float32=True):
        """Init.

        Args:
            upcast_to_float32: The upcast to float32.
        """
        super().__init__()
        self.scaling_factor = 0.13025

    @staticmethod
    def state_dict_converter():
        """State dict converter."""
        return SDXLVAEDecoderStateDictConverter()
    

class SDXLVAEDecoderStateDictConverter(SDVAEDecoderStateDictConverter):
    """Sdxlvae decoder state dict converter implementation."""
    def __init__(self):
        """Init."""
        super().__init__()

    def from_diffusers(self, state_dict):
        """From diffusers.

        Args:
            state_dict: The state dict.
        """
        state_dict = super().from_diffusers(state_dict)
        return state_dict, {"upcast_to_float32": True}
    
    def from_civitai(self, state_dict):
        """From civitai.

        Args:
            state_dict: The state dict.
        """
        state_dict = super().from_civitai(state_dict)
        return state_dict, {"upcast_to_float32": True}
