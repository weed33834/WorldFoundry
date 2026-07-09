"""Module for base_models -> diffusion_model -> diffsynth -> utils -> state_dict_converters -> flux_infiniteyou.py functionality."""

def FluxInfiniteYouImageProjectorStateDictConverter(state_dict):
    """Fluxinfiniteyouimageprojectorstatedictconverter.

    Args:
        state_dict: The state dict.
    """
    return state_dict['image_proj']