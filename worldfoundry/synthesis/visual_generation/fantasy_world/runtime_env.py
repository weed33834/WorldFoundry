"""
Utilities for managing paths, caching, and dependency resolution for FantasyWorld models.

This module provides functions to locate project roots, cache directories, and
model checkpoints, including specific resolution logic for various FantasyWorld
models like WAN2.1, WAN2.2, and MoGe-2. It handles both local paths and
Hugging Face Hub downloads, ensuring that required files are present for model
operation.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

from huggingface_hub import snapshot_download

from worldfoundry.base_models.diffusion_model.video.wan import wan_variant_root


DEFAULT_FANTASY_WORLD_WAN21_REPO = "acvlab/FantasyWorld-Wan2.1-I2V-14B-480P"
DEFAULT_FANTASY_WORLD_WAN22_REPO = "acvlab/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera"
DEFAULT_FANTASY_WORLD_WAN21_BASE_REPO = "Wan-AI/Wan2.1-I2V-14B-480P"
DEFAULT_FANTASY_WORLD_WAN22_BASE_REPO = "alibaba-pai/Wan2.2-Fun-A14B-Control-Camera"
DEFAULT_FANTASY_WORLD_WAN22_LORA_REPO = "alibaba-pai/Wan2.2-Fun-Reward-LoRAs"
DEFAULT_FANTASY_WORLD_WAN22_MODELSCOPE_ID = "PAI/Wan2.2-Fun-A14B-Control-Camera"
DEFAULT_FANTASY_WORLD_WAN22_LORA_MODELSCOPE_ID = "PAI/Wan2.2-Fun-Reward-LoRAs"
DEFAULT_FANTASY_WORLD_MOGE2_REPO = "Ruicheng/moge-2-vitl-normal"

DEFAULT_FANTASY_WORLD_WAN21_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in "
    "the background, walking backwards"
)

WAN21_REQUIRED_FILES = [
    *(f"diffusion_pytorch_model-0000{i}-of-00007.safetensors" for i in range(1, 8)),
    "Wan2.1_VAE.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
]

WAN22_BASE_REQUIRED_PATTERNS = [
    "high_noise_model/*",
    "low_noise_model/*",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
]

WAN22_LORA_HIGH_NAME = "Wan2.2-Fun-A14B-InP-high-noise-HPS2.1.safetensors"
WAN22_LORA_LOW_NAME = "Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors"


def project_root() -> Path:
    """
    Determines the root directory of the current project.

    It assumes the project root is the directory containing a 'pyproject.toml' file.
    If 'pyproject.toml' is not found, it falls back to a default parent directory.

    Returns:
        Path: The absolute path to the project root directory.
    """
    current = Path(__file__).resolve()
    # Iterate through parent directories to find 'pyproject.toml'
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # Fallback if 'pyproject.toml' is not found in any parent
    return current.parents[6]


def cache_root() -> Path:
    """
    Returns the root directory for caching Hugging Face downloads.

    This path is typically within the project's 'cache' directory.

    Returns:
        Path: The absolute path to the cache root directory.
    """
    return project_root() / "cache" / "hfd"


def checkpoint_root() -> Path:
    """
    Returns the root directory for model checkpoints.

    Prioritizes the path specified by the 'WORLDFOUNDRY_CKPT_DIR' environment variable.
    If not set, it defaults to a 'ckpt' directory adjacent to the project root.

    Returns:
        Path: The absolute path to the checkpoint root directory.
    """
    configured = os.environ.get("WORLDFOUNDRY_CKPT_DIR")
    if configured:
        return Path(configured).expanduser()
    return project_root().parent / "ckpt"


def runtime_root() -> Path:
    """
    Returns the root directory for FantasyWorld runtime components.

    This path points to the 'fantasy_world_runtime' directory relative to the
    current file.

    Returns:
        Path: The absolute path to the FantasyWorld runtime directory.
    """
    return Path(__file__).resolve().parent / "fantasy_world_runtime"


def fantasy_wan_root() -> Path:
    """
    Returns the root directory for the FantasyWorld WAN variant model.

    This resolves to a specific path within the `worldfoundry` package structure.

    Returns:
        Path: The absolute path to the FantasyWorld WAN variant root.
    """
    return wan_variant_root("fantasy-world")


def fantasy_vggt_root() -> Path:
    """
    Returns the root directory for the FantasyWorld VGGT model.

    This resolves to a specific path within the `worldfoundry` package structure.

    Returns:
        Path: The absolute path to the FantasyWorld VGGT root.
    """
    return (
        Path(__file__).resolve().parents[3]
        / "base_models"
        / "three_dimensions"
        / "point_clouds"
        / "vggt"
        / "vggt"
        / "variants"
        / "fantasy_world"
    )


def ensure_fantasy_world_runtime() -> Path:
    """
    Ensures that the necessary FantasyWorld runtime directories are added to `sys.path`.

    This allows Python to find modules and packages required by FantasyWorld
    models, prioritizing them by inserting them at the beginning of `sys.path`.
    It also removes any duplicate entries before adding.

    Returns:
        Path: The resolved absolute path to the main FantasyWorld runtime directory.
    """
    root = runtime_root().resolve()
    for import_root in (
        root,
        fantasy_wan_root().resolve(),
    ):
        root_str = str(import_root)
        # Remove existing paths to avoid duplicates and ensure new insertion is at the front
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
    return root


def _iter_candidate_paths(source_value: Optional[str | os.PathLike]) -> Iterable[Path]:
    """
    Generates an iterable of potential absolute paths based on a source value.

    This function attempts to resolve a given source value (which could be a local path,
    a repository ID, or a filename) into actual file system paths. It checks
    direct existence, then constructs candidate names from the source, and finally
    looks for these candidates within predefined cache and checkpoint roots.

    Args:
        source_value (Optional[str | os.PathLike]): The input string or path to resolve.
            Can be a repository ID, a local path, or a filename.

    Yields:
        Path: Resolved absolute `Path` objects that exist on the filesystem.
    """
    if not source_value:
        return

    raw = Path(source_value).expanduser()
    # If the raw path already exists, yield it directly
    if raw.exists():
        yield raw.resolve()
        return

    source_str = str(source_value)
    candidate_names = []
    if source_str:
        # Generate candidate names from the source string, e.g., by replacing slashes
        candidate_names.append(source_str.replace("/", "--"))
        candidate_names.append(source_str.split("/")[-1])

    # Deduplicate candidate names to avoid redundant checks
    deduped = []
    for name in candidate_names:
        if name and name not in deduped:
            deduped.append(name)

    # Check for candidates within standard cache and checkpoint directories
    for name in deduped:
        for base in (cache_root(), checkpoint_root(), checkpoint_root() / "hfd"):
            candidate = base / name
            if candidate.exists():
                yield candidate.resolve()


def _download_snapshot(repo_id: str, allow_patterns: Sequence[str]) -> Path:
    """
    Downloads a snapshot of a Hugging Face Hub repository.

    Args:
        repo_id (str): The ID of the repository to download.
        allow_patterns (Sequence[str]): A list of glob patterns to filter which files
            are downloaded from the repository.

    Returns:
        Path: The absolute path to the downloaded repository snapshot directory.
    """
    return Path(snapshot_download(repo_id=repo_id, allow_patterns=list(allow_patterns))).resolve()


def _resolve_file(
    source_value: Optional[str | os.PathLike],
    expected_filenames: Sequence[str],
    *,
    fallback_repo_id: Optional[str] = None,
) -> Path:
    """
    Resolves the absolute path to a specific file.

    It first checks local candidate paths derived from `source_value`. If the file
    is not found locally, it attempts to download it from the Hugging Face Hub
    using `fallback_repo_id` (or `source_value` if it's a repo ID).

    Args:
        source_value (Optional[str | os.PathLike]): A local path, a Hugging Face repo ID,
            or a custom identifier for the file.
        expected_filenames (Sequence[str]): A list of possible filenames to look for.
            The first one found will be returned.
        fallback_repo_id (Optional[str]): The Hugging Face repository ID to use
            if `source_value` does not directly specify an existing local path or
            a valid repo ID for download.

    Returns:
        Path: The absolute path to the resolved file.

    Raises:
        FileNotFoundError: If the file cannot be located locally or downloaded
            from the specified repository.
    """
    # Check local candidate paths first
    for candidate in _iter_candidate_paths(source_value):
        # If the candidate itself is the file, return it
        if candidate.is_file():
            return candidate.resolve()
        # Otherwise, check if any of the expected filenames exist within the candidate directory
        for filename in expected_filenames:
            target = candidate / filename
            if target.is_file():
                return target.resolve()

    # If source_value points to an existing path but the required file isn't there
    if source_value and Path(str(source_value)).expanduser().exists():
        raise FileNotFoundError(
            f"Unable to locate any of {list(expected_filenames)} under {source_value}."
        )

    # Determine the repository ID for download
    repo_id = str(source_value) if source_value else fallback_repo_id
    if not repo_id:
        raise FileNotFoundError(f"Unable to locate expected file candidates: {list(expected_filenames)}")

    # Attempt to download from Hugging Face Hub
    snapshot_root = _download_snapshot(repo_id, expected_filenames)
    for filename in expected_filenames:
        target = snapshot_root / filename
        if target.is_file():
            return target.resolve()

    raise FileNotFoundError(f"Unable to locate any of {list(expected_filenames)} in snapshot for {repo_id}.")


def _resolve_dir_with_files(
    source_value: Optional[str | os.PathLike],
    required_files: Sequence[str],
    *,
    fallback_repo_id: Optional[str] = None,
    allow_patterns: Optional[Sequence[str]] = None,
) -> Path:
    """
    Resolves the absolute path to a directory that contains a specific set of files.

    It first checks local candidate paths derived from `source_value`. If a directory
    containing all `required_files` is not found locally, it attempts to download
    them from the Hugging Face Hub using `fallback_repo_id` (or `source_value`
    if it's a repo ID).

    Args:
        source_value (Optional[str | os.PathLike]): A local path, a Hugging Face repo ID,
            or a custom identifier for the directory.
        required_files (Sequence[str]): A list of filenames that *must* all be present
            in the resolved directory.
        fallback_repo_id (Optional[str]): The Hugging Face repository ID to use
            if `source_value` does not directly specify an existing local path or
            a valid repo ID for download.
        allow_patterns (Optional[Sequence[str]]): Specific glob patterns to pass to
            `snapshot_download` if downloading. If None, `required_files` will be used.

    Returns:
        Path: The absolute path to the resolved directory.

    Raises:
        FileNotFoundError: If the directory containing all required files cannot
            be located locally or downloaded from the specified repository.
    """
    # Check local candidate paths first
    for candidate in _iter_candidate_paths(source_value):
        # If the candidate itself is a directory, or its parent is, check for required files
        candidate_dir = candidate if candidate.is_dir() else candidate.parent
        # Verify that all required files exist within this candidate directory
        if all((candidate_dir / name).exists() for name in required_files):
            return candidate_dir.resolve()

    # If source_value points to an existing path but the required files aren't there
    if source_value and Path(str(source_value)).expanduser().exists():
        raise FileNotFoundError(
            f"Unable to locate required files {list(required_files)} under {source_value}."
        )

    # Determine the repository ID for download
    repo_id = str(source_value) if source_value else fallback_repo_id
    if not repo_id:
        raise FileNotFoundError(f"Unable to locate required files: {list(required_files)}")

    # Attempt to download from Hugging Face Hub
    snapshot_root = _download_snapshot(repo_id, allow_patterns or required_files)
    # After download, verify that all required files exist in the snapshot root
    if all((snapshot_root / name).exists() for name in required_files):
        return snapshot_root.resolve()

    raise FileNotFoundError(f"Unable to locate required files {list(required_files)} in snapshot for {repo_id}.")


def resolve_fantasy_world_wan21_checkpoint(source_value: Optional[str | os.PathLike]) -> Path:
    """
    Resolves the path to the FantasyWorld WAN2.1 model checkpoint file.

    Args:
        source_value (Optional[str | os.PathLike]): A local path, Hugging Face repo ID,
            or custom identifier for the WAN2.1 checkpoint.

    Returns:
        Path: The absolute path to the WAN2.1 checkpoint file.

    Raises:
        FileNotFoundError: If the checkpoint cannot be found.
    """
    return _resolve_file(
        source_value,
        ("model.pth",),
        fallback_repo_id=DEFAULT_FANTASY_WORLD_WAN21_REPO,
    )


def resolve_fantasy_world_wan22_checkpoint_dir(source_value: Optional[str | os.PathLike]) -> Path:
    """
    Resolves the path to the directory containing FantasyWorld WAN2.2 model checkpoints.

    This typically includes 'high_noise_model.pth' and 'low_noise_model.pth'.

    Args:
        source_value (Optional[str | os.PathLike]): A local path, Hugging Face repo ID,
            or custom identifier for the WAN2.2 checkpoint directory.

    Returns:
        Path: The absolute path to the WAN2.2 checkpoint directory.

    Raises:
        FileNotFoundError: If the directory with required checkpoints cannot be found.
    """
    return _resolve_dir_with_files(
        source_value,
        ("high_noise_model.pth", "low_noise_model.pth"),
        fallback_repo_id=DEFAULT_FANTASY_WORLD_WAN22_REPO,
        allow_patterns=("high_noise_model.pth", "low_noise_model.pth"),
    )


def resolve_fantasy_world_wan21_base_dir(source_value: Optional[str | os.PathLike]) -> Path:
    """
    Resolves the path to the directory containing the FantasyWorld WAN2.1 base model files.

    Args:
        source_value (Optional[str | os.PathLike]): A local path, Hugging Face repo ID,
            or custom identifier for the WAN2.1 base model directory.

    Returns:
        Path: The absolute path to the WAN2.1 base model directory.

    Raises:
        FileNotFoundError: If the directory with required base model files cannot be found.
    """
    return _resolve_dir_with_files(
        source_value,
        WAN21_REQUIRED_FILES,
        fallback_repo_id=DEFAULT_FANTASY_WORLD_WAN21_BASE_REPO,
    )


def _resolve_wan22_snapshot_or_cached_dir(
    source_value: Optional[str | os.PathLike],
    *,
    fallback_repo_id: str,
    extra_cache_aliases: Sequence[str] = (),
    allow_patterns: Sequence[str],
    must_contain: Sequence[str],
) -> Path:
    """
    Helper function to resolve a WAN2.2 model directory (base or LoRA).

    It attempts to find the directory locally by checking various aliases and
    source values. If not found, it downloads the snapshot from Hugging Face Hub.

    Args:
        source_value (Optional[str | os.PathLike]): The primary source for the directory,
            can be a local path or a Hugging Face repo ID.
        fallback_repo_id (str): The default Hugging Face repo ID to use if
            `source_value` is not provided or doesn't resolve locally.
        extra_cache_aliases (Sequence[str]): Additional names/identifiers to
            check in local cache/checkpoint directories.
        allow_patterns (Sequence[str]): Glob patterns to specify which files to
            download from the Hugging Face repository.
        must_contain (Sequence[str]): A list of filenames that *must* all be present
            in the resolved directory for it to be considered valid.

    Returns:
        Path: The absolute path to the resolved directory.

    Raises:
        FileNotFoundError: If the directory with all required files cannot be
            located locally or downloaded from the specified repository.
    """
    cache_probe_values = []
    if source_value:
        cache_probe_values.append(str(source_value))
    cache_probe_values.extend(extra_cache_aliases)

    # Iterate through probe values to find a locally cached or existing directory
    for probe in cache_probe_values:
        for candidate in _iter_candidate_paths(probe):
            candidate_dir = candidate if candidate.is_dir() else candidate.parent
            # Check if all required files are present in the candidate directory
            if all((candidate_dir / name).exists() for name in must_contain):
                return candidate_dir.resolve()

    # If source_value points to an existing path but the required files aren't in it
    if source_value and Path(str(source_value)).expanduser().exists():
        candidate_dir = Path(source_value).expanduser().resolve()
        if candidate_dir.is_file():
            candidate_dir = candidate_dir.parent
        if all((candidate_dir / name).exists() for name in must_contain):
            return candidate_dir
        raise FileNotFoundError(f"Unable to locate required files {list(must_contain)} under {candidate_dir}.")

    # If not found locally, attempt to download from Hugging Face Hub
    snapshot_root = _download_snapshot(
        str(source_value) if source_value else fallback_repo_id,
        allow_patterns=allow_patterns,
    )
    # After download, verify that all required files are in the snapshot root
    if all((snapshot_root / name).exists() for name in must_contain):
        return snapshot_root.resolve()
    raise FileNotFoundError(
        f"Unable to locate required files {list(must_contain)} in snapshot for "
        f"{str(source_value) if source_value else fallback_repo_id}."
    )


def resolve_fantasy_world_wan22_base_dir(source_value: Optional[str | os.PathLike]) -> Path:
    """
    Resolves the path to the directory containing the FantasyWorld WAN2.2 base model files.

    Args:
        source_value (Optional[str | os.PathLike]): A local path, Hugging Face repo ID,
            or custom identifier for the WAN2.2 base model directory.

    Returns:
        Path: The absolute path to the WAN2.2 base model directory.

    Raises:
        FileNotFoundError: If the directory with required base model files cannot be found.
    """
    return _resolve_wan22_snapshot_or_cached_dir(
        source_value,
        fallback_repo_id=DEFAULT_FANTASY_WORLD_WAN22_BASE_REPO,
        extra_cache_aliases=(
            DEFAULT_FANTASY_WORLD_WAN22_BASE_REPO,
            DEFAULT_FANTASY_WORLD_WAN22_MODELSCOPE_ID,
            "Wan-AI/Wan2.2-I2V-A14B",
            "Wan2.2-I2V-A14B",
            "Wan-AI/Wan2.2-T2V-A14B",
            "Wan2.2-T2V-A14B",
        ),
        allow_patterns=WAN22_BASE_REQUIRED_PATTERNS,
        must_contain=("high_noise_model", "low_noise_model"), # These are sub-directories
    )


def resolve_fantasy_world_wan22_lora_dir(source_value: Optional[str | os.PathLike]) -> Path:
    """
    Resolves the path to the directory containing FantasyWorld WAN2.2 LoRA models.

    Args:
        source_value (Optional[str | os.PathLike]): A local path, Hugging Face repo ID,
            or custom identifier for the WAN2.2 LoRA directory.

    Returns:
        Path: The absolute path to the WAN2.2 LoRA directory.

    Raises:
        FileNotFoundError: If the directory with required LoRA files cannot be found.
    """
    return _resolve_wan22_snapshot_or_cached_dir(
        source_value,
        fallback_repo_id=DEFAULT_FANTASY_WORLD_WAN22_LORA_REPO,
        extra_cache_aliases=(
            DEFAULT_FANTASY_WORLD_WAN22_LORA_REPO,
            DEFAULT_FANTASY_WORLD_WAN22_LORA_MODELSCOPE_ID,
        ),
        allow_patterns=(WAN22_LORA_HIGH_NAME, WAN22_LORA_LOW_NAME),
        must_contain=(WAN22_LORA_HIGH_NAME, WAN22_LORA_LOW_NAME),
    )


def _ensure_utils3d_alias() -> None:
    """
    Ensures that the `utils3d` module used by MoGe-2 points to the vendored
    WorldFoundry implementation.

    This prevents conflicts with other `utils3d` modules that might be
    present in the Python environment, ensuring MoGe-2 uses the correct version
    for panorama helpers. It clears any existing `utils3d` related modules from
    `sys.modules` and then sets the alias.
    """
    from worldfoundry.base_models.three_dimensions.general_3d.eastern_journalist import (
        utils3d as vendored_utils3d,
    )

    # FantasyWorld's MoGe-2 utilities expect the WorldFoundry-pinned utils3d
    # implementation when code reaches the panorama helpers.
    # Identify and remove any potentially conflicting `utils3d` modules from sys.modules
    stale_modules = [
        name for name in sys.modules if name == "utils3d" or name.startswith("utils3d.")
    ]
    for name in stale_modules:
        del sys.modules[name]

    # Alias the vendored utils3d to 'utils3d' in sys.modules
    sys.modules["utils3d"] = vendored_utils3d


def resolve_moge_pretrained(source_value: Optional[str | os.PathLike]) -> str:
    """
    Resolves the path to the MoGe-2 pretrained model checkpoint.

    Args:
        source_value (Optional[str | os.PathLike]): A local path to the MoGe-2
            checkpoint file or its containing directory, or a Hugging Face repo ID.

    Returns:
        str: The absolute path to the MoGe-2 checkpoint file. If `source_value`
            is a repo ID and no local file is found, the repo ID itself is returned,
            indicating it needs to be downloaded by a downstream system.

    Raises:
        FileNotFoundError: If `source_value` is a local path to a directory,
            but a 'model.pt' file is not found within it.
    """
    if source_value is None:
        # Check default repo and its potential local cached paths
        for probe in (DEFAULT_FANTASY_WORLD_MOGE2_REPO,):
            for candidate in _iter_candidate_paths(probe):
                if candidate.is_file():
                    return str(candidate.resolve())
                model_file = candidate / "model.pt"
                if model_file.is_file():
                    return str(model_file.resolve())
        # If no local default found, return the default repo ID
        return DEFAULT_FANTASY_WORLD_MOGE2_REPO

    candidate = Path(source_value).expanduser()
    if candidate.exists():
        # If the candidate itself is a file, return its absolute path
        if candidate.is_file():
            return str(candidate.resolve())
        # If it's a directory, check for 'model.pt' inside
        model_file = candidate / "model.pt"
        if model_file.is_file():
            return str(model_file.resolve())
        raise FileNotFoundError(f"Expected a MoGe checkpoint file at {model_file}")

    # Check for local candidates derived from the source_value
    for local_candidate in _iter_candidate_paths(str(source_value)):
        if local_candidate.is_file():
            return str(local_candidate.resolve())
        model_file = local_candidate / "model.pt"
        if model_file.is_file():
            return str(model_file.resolve())

    # If no local file or path could be resolved, assume source_value is a repo ID
    return str(source_value)


def ensure_moge2_runtime(moge_path: Optional[str | os.PathLike] = None) -> None:
    """
    Configures the runtime environment for MoGe-2.

    This involves ensuring the correct `utils3d` module alias is set.
    Note: Direct external MoGe source checkout paths are no longer supported.

    Args:
        moge_path (Optional[str | os.PathLike]): DEPRECATED. This argument is no longer
            supported and will raise a RuntimeError if provided.

    Raises:
        RuntimeError: If `moge_path` is provided, indicating deprecated usage.
    """
    _ensure_utils3d_alias()

    if moge_path:
        raise RuntimeError(
            "FantasyWorld no longer accepts external MoGe source checkout paths. "
            "Use worldfoundry.base_models.three_dimensions.depth.moge and pass `moge_pretrained` for weights."
        )


def prepare_wan22_runtime_root(
    base_dir: str | os.PathLike,
    lora_dir: str | os.PathLike,
) -> tuple[str, Optional[tempfile.TemporaryDirectory[str]]]:
    """
    Prepares a runtime root directory for WAN2.2 models, potentially using symlinks.

    WAN2.2 expects a specific directory structure (e.g., `root/PAI/Wan2.2-Fun-A14B-Control-Camera`).
    This function checks if an existing structure is already present in a parent
    directory. If not, it creates a temporary directory and symlinks the provided
    `base_dir` and `lora_dir` into the expected structure.

    Args:
        base_dir (str | os.PathLike): The path to the WAN2.2 base model directory.
        lora_dir (str | os.PathLike): The path to the WAN2.2 LoRA model directory.

    Returns:
        tuple[str, Optional[tempfile.TemporaryDirectory[str]]]:
            A tuple containing:
            - str: The absolute path to the prepared runtime root directory.
            - Optional[tempfile.TemporaryDirectory[str]]: The `TemporaryDirectory`
                object if a temporary directory was created, otherwise `None`.
                This object should be kept to ensure proper cleanup of the temp directory.
    """
    base_path = Path(base_dir).expanduser().resolve()
    lora_path = Path(lora_dir).expanduser().resolve()

    # Check if a suitable 'PAI' structure already exists in parent directories
    # This avoids creating unnecessary temporary directories if models are already laid out correctly
    direct_root = base_path.parent.parent if base_path.parent.name == "PAI" else None
    if direct_root is not None:
        model_candidate = direct_root / "PAI" / "Wan2.2-Fun-A14B-Control-Camera"
        lora_candidate = direct_root / "PAI" / "Wan2.2-Fun-Reward-LoRAs"
        if model_candidate.exists() and lora_candidate.exists():
            return str(direct_root.resolve()), None

    # If no existing structure found, create a temporary directory
    temp_root = tempfile.TemporaryDirectory(prefix="fantasyworld_wan22_")
    root = Path(temp_root.name)
    pai_root = root / "PAI"
    pai_root.mkdir(parents=True, exist_ok=True)

    # Create symlinks within the temporary directory to match the expected 'PAI' structure
    model_link = pai_root / "Wan2.2-Fun-A14B-Control-Camera"
    lora_link = pai_root / "Wan2.2-Fun-Reward-LoRAs"

    if not model_link.exists():
        os.symlink(base_path, model_link, target_is_directory=True)
    if not lora_link.exists():
        os.symlink(lora_path, lora_link, target_is_directory=True)

    return str(root.resolve()), temp_root
