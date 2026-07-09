"""Module for base_models -> diffusion_model -> diffsynth -> utils -> state_dict_converters -> qwen_image_text_encoder.py functionality."""

def QwenImageTextEncoderStateDictConverter(state_dict):
    """Qwenimagetextencoderstatedictconverter.

    Args:
        state_dict: The state dict.
    """
    state_dict_ = {}
    for k in state_dict:
        v = state_dict[k]
        if k.startswith("visual."):
            k = "model." + k
        elif k.startswith("model."):
            k = k.replace("model.", "model.language_model.")
        state_dict_[k] = v
    return state_dict_
