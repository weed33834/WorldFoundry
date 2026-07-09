"""Module for base_models -> diffusion_model -> diffsynth -> utils -> state_dict_converters -> flux_text_encoder_t5.py functionality."""

def FluxTextEncoderT5StateDictConverter(state_dict):
    """Fluxtextencodert5statedictconverter.

    Args:
        state_dict: The state dict.
    """
    state_dict_ = {i: state_dict[i] for i in state_dict}
    state_dict_["encoder.embed_tokens.weight"] = state_dict["shared.weight"]
    return state_dict_
