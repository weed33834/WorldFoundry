"""Module for base_models -> diffusion_model -> video -> skyreels_v3 -> skyreels_v3 -> pipelines -> __init__.py functionality."""

__all__ = [
    "ShotSwitchingExtensionPipeline",
    "SingleShotExtensionPipeline",
    "ReferenceToVideoPipeline",
    "TalkingAvatarPipeline",
]


def __getattr__(name):
    """Getattr.

    Args:
        name: The name.
    """
    if name == "ReferenceToVideoPipeline":
        from .reference_to_video_pipeline import ReferenceToVideoPipeline

        return ReferenceToVideoPipeline
    if name == "ShotSwitchingExtensionPipeline":
        from .shot_switching_extension_pipeline import ShotSwitchingExtensionPipeline

        return ShotSwitchingExtensionPipeline
    if name == "SingleShotExtensionPipeline":
        from .single_shot_extension_pipeline import SingleShotExtensionPipeline

        return SingleShotExtensionPipeline
    if name == "TalkingAvatarPipeline":
        from .talking_avatar_pipeline import TalkingAvatarPipeline

        return TalkingAvatarPipeline
    raise AttributeError(name)
