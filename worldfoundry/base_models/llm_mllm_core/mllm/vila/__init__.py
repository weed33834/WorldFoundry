"""Inference-only in-tree VILA/NVILA runtime."""

__all__ = [
    "Image",
    "VILAConfig",
    "VILAForCausalLM",
    "VILAGenerator",
    "Video",
    "generate_vila_response",
    "load_vila_model",
]


def __getattr__(name: str):
    if name == "VILAConfig":
        from .configuration_vila import VILAConfig

        return VILAConfig
    if name in {"Image", "Video"}:
        from .media import Image, Video

        return {"Image": Image, "Video": Video}[name]
    if name == "VILAForCausalLM":
        from .modeling_vila import VILAForCausalLM

        return VILAForCausalLM
    if name in {"VILAGenerator", "generate_vila_response", "load_vila_model"}:
        from .inference import VILAGenerator, generate_vila_response, load_vila_model

        return {
            "VILAGenerator": VILAGenerator,
            "generate_vila_response": generate_vila_response,
            "load_vila_model": load_vila_model,
        }[name]
    raise AttributeError(name)
