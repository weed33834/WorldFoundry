"""GEN3C Cosmos Predict1 base-model integration."""

__all__ = [
    "DEFAULT_GEN3C_ALIAS",
    "DEFAULT_GEN3C_MOGE1_REPO",
    "DEFAULT_GEN3C_NEGATIVE_PROMPT",
    "Gen3CRuntime",
    "build_subprocess_env",
    "ensure_packaged_gen3c_runtime",
    "load_pil_image",
    "load_video_frames",
    "official_gen3c_repo_root",
    "packaged_gen3c_cosmos_root",
    "prepare_gen3c_checkpoint_root",
    "project_root",
    "resolve_gen3c_runtime_root",
    "resolve_gen3c_checkpoint_arg",
    "resolve_gen3c_moge_pretrained",
]

_RUNTIME_ENV_EXPORTS = {
    "DEFAULT_GEN3C_ALIAS",
    "DEFAULT_GEN3C_MOGE1_REPO",
    "DEFAULT_GEN3C_NEGATIVE_PROMPT",
    "build_subprocess_env",
    "ensure_packaged_gen3c_runtime",
    "load_pil_image",
    "load_video_frames",
    "official_gen3c_repo_root",
    "packaged_gen3c_cosmos_root",
    "prepare_gen3c_checkpoint_root",
    "project_root",
    "resolve_gen3c_runtime_root",
    "resolve_gen3c_checkpoint_arg",
    "resolve_gen3c_moge_pretrained",
}


def __getattr__(name: str):
    """Getattr.

    Args:
        name: The name.
    """
    if name == "Gen3CRuntime":
        from .worldfoundry_runtime import Gen3CRuntime

        return Gen3CRuntime
    if name in _RUNTIME_ENV_EXPORTS:
        from . import runtime_env

        return getattr(runtime_env, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
