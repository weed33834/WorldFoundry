"""Module for base_models -> diffusion_model -> diffsynth -> utils -> state_dict_converters -> nexus_gen.py functionality."""

def NexusGenAutoregressiveModelStateDictConverter(state_dict):
    """Nexusgenautoregressivemodelstatedictconverter.

    Args:
        state_dict: The state dict.
    """
    new_state_dict = {}
    for key in state_dict:
        value = state_dict[key]
        new_state_dict["model." + key] = value
    return new_state_dict