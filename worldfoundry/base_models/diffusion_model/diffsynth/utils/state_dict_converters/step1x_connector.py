"""Module for base_models -> diffusion_model -> diffsynth -> utils -> state_dict_converters -> step1x_connector.py functionality."""

def Qwen2ConnectorStateDictConverter(state_dict):
    """Qwen2connectorstatedictconverter.

    Args:
        state_dict: The state dict.
    """
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("connector."):
            name_ = name[len("connector."):]
            state_dict_[name_] = state_dict[name]
    return state_dict_