"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos1 -> cosmos_predict1_gen3c -> runtime_env.py functionality."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from worldfoundry.core import cuda_visible_devices_from_device
from worldfoundry.core.io.paths import project_root

DEFAULT_GEN3C_ALIAS = "gen3c"
DEFAULT_GEN3C_MOGE1_REPO = "Ruicheng/moge-vitl"
DEFAULT_GEN3C_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special "
    "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and "
    "flickering. Overall, the video is of poor quality."
)

_GEN3C_CHECKPOINT_COMPONENT_REPOS = {
    "Gen3C-Cosmos-7B": "nvidia/GEN3C-Cosmos-7B",
    "Cosmos-Tokenize1-CV8x8x8-720p": "nvidia/Cosmos-Tokenize1-CV8x8x8-720p",
    os.path.join("google-t5", "t5-11b"): "google-t5/t5-11b",
}
_GEN3C_ALIASES = {"", DEFAULT_GEN3C_ALIAS, "GEN3C", "gen-3c", "Gen3C"}


def packaged_gen3c_cosmos_root() -> Path:
    """Packaged gen3c cosmos root.

    Returns:
        The return value.
    """
    return Path(__file__).resolve().parent


def _env_truthy(key: str) -> bool:
    """Helper function to env truthy.

    Args:
        key: The key.

    Returns:
        The return value.
    """
    return os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}


def _hf_local_files_only() -> bool:
    """Return whether WorldFoundry should restrict HF resolution to local cache."""
    return _env_truthy("WORLDFOUNDRY_HF_LOCAL_FILES_ONLY")


def _snapshot_download_repo(repo_id: str) -> Path:
    """Resolve a Hugging Face model repo through the standard hub cache."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency checked by env setup.
        raise ImportError("GEN3C requires huggingface_hub to resolve model snapshots.") from exc

    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_files_only=_hf_local_files_only(),
    )
    return Path(snapshot).expanduser()


def official_gen3c_repo_root() -> Optional[Path]:
    """Official gen3c repo root.

    Returns:
        The return value.
    """
    candidates = []
    env_value = os.environ.get("WORLDFOUNDRY_GEN3C_OFFICIAL_REPO")
    if env_value:
        candidates.append(Path(env_value).expanduser())

    deduped: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in deduped:
            deduped.append(resolved)

    for candidate in deduped:
        if (candidate / "cosmos_predict1" / "diffusion" / "inference" / "gen3c_pipeline.py").is_file():
            return candidate
    return None


def resolve_gen3c_runtime_root() -> str:
    """Resolve gen3c runtime root.

    Returns:
        The return value.
    """
    official_root = official_gen3c_repo_root()
    if official_root is not None:
        return str(official_root)
    return ensure_packaged_gen3c_runtime()


def is_gen3c_alias(value: Optional[str]) -> bool:
    """Is gen3c alias.

    Args:
        value: The value.

    Returns:
        The return value.
    """
    return value is None or str(value).strip() in _GEN3C_ALIASES


def ensure_packaged_gen3c_runtime() -> str:
    """Ensure packaged gen3c runtime.

    Returns:
        The return value.
    """
    sentinel = packaged_gen3c_cosmos_root() / "cosmos_predict1" / "diffusion" / "inference" / "gen3c_pipeline.py"
    if not sentinel.is_file():
        raise FileNotFoundError(f"Packaged GEN3C runtime is incomplete: {sentinel}")
    return str(packaged_gen3c_cosmos_root())


def resolve_gen3c_checkpoint_arg(pretrained_model_path: Optional[str]) -> Optional[str]:
    """Resolve gen3c checkpoint arg.

    Args:
        pretrained_model_path: The pretrained model path.

    Returns:
        The return value.
    """
    if is_gen3c_alias(pretrained_model_path):
        return None
    return str(Path(str(pretrained_model_path)).expanduser())


def _local_moge_model_path_candidates(source_value: str) -> list[Path]:
    """Helper function to local moge model path candidates.

    Args:
        source_value: The source value.

    Returns:
        The return value.
    """
    names = []
    if source_value:
        names.extend(
            [
                source_value.replace("/", "--"),
                source_value.split("/")[-1],
            ]
        )

    deduped = []
    for name in names:
        if name and name not in deduped:
            deduped.append(name)

    roots: list[Path] = []
    for env_key in ("WORLDFOUNDRY_HFD_ROOT", "WORLDFOUNDRY_CKPT_DIR"):
        value = os.environ.get(env_key)
        if not value:
            continue
        root = Path(value).expanduser()
        roots.append(root)
        if env_key == "WORLDFOUNDRY_CKPT_DIR":
            roots.extend([root / "hfd", root / "huggingface" / "hub"])
    adjacent_ckpt = project_root().parent / "ckpt"
    roots.extend(
        [
            adjacent_ckpt,
            adjacent_ckpt / "hfd",
            adjacent_ckpt / "huggingface" / "hub",
            project_root() / "cache" / "hfd",
        ]
    )
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    for env_key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(env_key)
        if value:
            roots.append(Path(value).expanduser())

    candidates: list[Path] = []
    for root in roots:
        for name in deduped:
            candidates.append(root / name / "model.pt")
            candidates.extend(_hf_snapshot_files(root, source_value, "model.pt"))
    result: list[Path] = []
    for candidate in candidates:
        if candidate not in result:
            result.append(candidate)
    return result


def resolve_gen3c_moge_pretrained(pretrained_model_name_or_path: Optional[str]) -> str:
    """Resolve gen3c moge pretrained.

    Args:
        pretrained_model_name_or_path: The pretrained model name or path.

    Returns:
        The return value.
    """
    if pretrained_model_name_or_path is None:
        for local_candidate in _local_moge_model_path_candidates(DEFAULT_GEN3C_MOGE1_REPO):
            if local_candidate.is_file():
                return str(local_candidate.resolve())
        snapshot = _snapshot_download_repo(DEFAULT_GEN3C_MOGE1_REPO)
        model_file = snapshot / "model.pt"
        if not model_file.is_file():
            raise FileNotFoundError(f"Expected MoGe checkpoint file in Hugging Face snapshot: {model_file}")
        return str(model_file)

    candidate = Path(str(pretrained_model_name_or_path)).expanduser()
    if candidate.exists():
        if candidate.is_file():
            return str(candidate.resolve())
        model_file = candidate / "model.pt"
        if model_file.is_file():
            return str(model_file.resolve())
        raise FileNotFoundError(f"Expected a MoGe checkpoint file at {model_file}")

    source_value = str(pretrained_model_name_or_path)
    for local_candidate in _local_moge_model_path_candidates(source_value):
        if local_candidate.is_file():
            return str(local_candidate.resolve())
    if "/" in source_value and not source_value.startswith(("./", "../", "/")):
        snapshot = _snapshot_download_repo(source_value)
        model_file = snapshot / "model.pt"
        if not model_file.is_file():
            raise FileNotFoundError(f"Expected MoGe checkpoint file in Hugging Face snapshot: {model_file}")
        return str(model_file)
    return source_value


def _checkpoint_layout_complete(root: Path) -> bool:
    """Helper function to checkpoint layout complete.

    Args:
        root: The root.

    Returns:
        The return value.
    """
    required_files = [
        root / "Gen3C-Cosmos-7B" / "model.pt",
        root / "Cosmos-Tokenize1-CV8x8x8-720p" / "mean_std.pt",
        root / "google-t5" / "t5-11b" / "config.json",
    ]
    return all(path.is_file() for path in required_files)


def _hf_snapshot_files(root: Path, repo_id: str, filename: str) -> list[Path]:
    """Return files from a Hugging Face hub cache directory without downloading."""
    sanitized = repo_id.replace("/", "--")
    model_dir = root / f"models--{sanitized}"
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(snapshots.glob(f"*/{filename}"))


def _hf_snapshot_dirs(root: Path, repo_id: str) -> list[Path]:
    """Return snapshot directories from a Hugging Face hub cache directory."""
    sanitized = repo_id.replace("/", "--")
    snapshots = root / f"models--{sanitized}" / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(path for path in snapshots.iterdir() if path.is_dir())


def _component_required_file(layout_name: str) -> Path:
    if layout_name == "Gen3C-Cosmos-7B":
        return Path("model.pt")
    if layout_name == "Cosmos-Tokenize1-CV8x8x8-720p":
        return Path("mean_std.pt")
    if layout_name == os.path.join("google-t5", "t5-11b"):
        return Path("config.json")
    return Path("config.json")


def _component_dir_candidates(repo_id: str, root: Path) -> list[Path]:
    """Helper function to component dir candidates.

    Args:
        repo_id: The repo id.
        root: The root.

    Returns:
        The return value.
    """
    raw_name = repo_id.split("/")[-1]
    sanitized_name = repo_id.replace("/", "--")
    google_nested = root / repo_id
    candidates = [
        google_nested,
        root / sanitized_name,
        root / raw_name,
    ]
    candidates.extend(_hf_snapshot_dirs(root, repo_id))
    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _find_component_dir(repo_id: str, root: Path) -> Optional[Path]:
    """Helper function to find component dir.

    Args:
        repo_id: The repo id.
        root: The root.

    Returns:
        The return value.
    """
    for candidate in _component_dir_candidates(repo_id, root):
        if candidate.exists():
            return candidate.resolve()
    return None


def _find_component_dir_with_file(layout_name: str, repo_id: str, root: Path) -> Optional[Path]:
    required = _component_required_file(layout_name)
    for candidate in _component_dir_candidates(repo_id, root):
        if (candidate / required).is_file():
            return candidate.resolve()
    return None


def _direct_layout_component_sources(root: Path) -> Optional[dict[str, Path]]:
    """Find GEN3C components in a direct local checkpoint root."""
    component_sources: dict[str, Path] = {}
    alias_names = {
        "Gen3C-Cosmos-7B": (
            "Gen3C-Cosmos-7B",
            "GEN3C-Cosmos-7B",
            "nvidia--GEN3C-Cosmos-7B",
        ),
        "Cosmos-Tokenize1-CV8x8x8-720p": (
            "Cosmos-Tokenize1-CV8x8x8-720p",
            "nvidia--Cosmos-Tokenize1-CV8x8x8-720p",
        ),
        os.path.join("google-t5", "t5-11b"): (
            os.path.join("google-t5", "t5-11b"),
            "google-t5--t5-11b",
            "t5-11b",
        ),
    }
    for layout_name, repo_id in _GEN3C_CHECKPOINT_COMPONENT_REPOS.items():
        required = _component_required_file(layout_name)
        found = None
        for name in alias_names.get(layout_name, (layout_name, repo_id.replace("/", "--"), repo_id.split("/")[-1])):
            candidate = root / name
            if (candidate / required).is_file():
                found = candidate.resolve()
                break
        if found is None:
            found = _find_component_dir_with_file(layout_name, repo_id, root)
        if found is None:
            return None
        component_sources[layout_name] = found
    return component_sources


def _reset_dir(path: Path) -> None:
    """Helper function to reset dir.

    Args:
        path: The path.

    Returns:
        The return value.
    """
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Helper function to link or copy.

    Args:
        src: The src.
        dst: The dst.

    Returns:
        The return value.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        _reset_dir(dst)
    dst.symlink_to(src, target_is_directory=src.is_dir())


def _iter_hfd_roots(path_value: Optional[Path]) -> list[Path]:
    """Helper function to iter hfd roots.

    Args:
        path_value: The path value.

    Returns:
        The return value.
    """
    roots = []
    if path_value is not None:
        for current in (path_value, *path_value.parents):
            if current.name == "hfd":
                roots.append(current.resolve())
                break
    for env_key in ("WORLDFOUNDRY_HFD_ROOT", "WORLDFOUNDRY_CKPT_DIR"):
        value = os.environ.get(env_key)
        if value:
            roots.append(Path(value).expanduser().resolve())
            if env_key == "WORLDFOUNDRY_CKPT_DIR":
                roots.append((Path(value).expanduser() / "hfd").resolve())
    adjacent_ckpt = project_root().parent / "ckpt"
    if adjacent_ckpt.is_dir():
        roots.append(adjacent_ckpt.resolve())
        roots.append((adjacent_ckpt / "hfd").resolve())
        roots.append((adjacent_ckpt / "huggingface" / "hub").resolve())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append((Path(hf_home).expanduser() / "hub").resolve())
    for env_key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(env_key)
        if value:
            roots.append(Path(value).expanduser().resolve())
    default_hfd = (project_root() / "cache" / "hfd").resolve()
    roots.append(default_hfd)
    deduped: list[Path] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def _iter_direct_checkpoint_roots() -> list[Path]:
    """Helper function to iter direct checkpoint roots.

    Returns:
        The return value.
    """
    roots = []
    env_value = os.environ.get("WORLDFOUNDRY_GEN3C_CHECKPOINT_DIR")
    if env_value:
        roots.append(Path(env_value).expanduser().resolve())
    adjacent_ckpt = project_root().parent / "ckpt"
    if adjacent_ckpt.is_dir():
        roots.append(adjacent_ckpt.resolve())
    deduped: list[Path] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def _stage_hfd_checkpoints(hfd_root: Path) -> Optional[Path]:
    """Helper function to stage hfd checkpoints.

    Args:
        hfd_root: The hfd root.

    Returns:
        The return value.
    """
    component_sources = {}
    for layout_name, repo_id in _GEN3C_CHECKPOINT_COMPONENT_REPOS.items():
        component_dir = _find_component_dir_with_file(layout_name, repo_id, hfd_root)
        if component_dir is None:
            return None
        component_sources[layout_name] = component_dir

    return _stage_component_sources(component_sources)


def _stage_component_sources(component_sources: dict[str, Path]) -> Path:
    """Stage component directories into GEN3C's official checkpoint layout."""
    stage_root = project_root() / "cache" / "runtime" / "gen3c_checkpoints"
    if stage_root.exists():
        _reset_dir(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    for layout_name, source_dir in component_sources.items():
        _link_or_copy(source_dir, stage_root / layout_name)
    return stage_root.resolve()


def _stage_hf_snapshot_checkpoints() -> Optional[Path]:
    """Build GEN3C's official checkpoint_dir layout from HF snapshots."""
    component_sources: dict[str, Path] = {}
    try:
        for layout_name, repo_id in _GEN3C_CHECKPOINT_COMPONENT_REPOS.items():
            component_sources[layout_name] = _snapshot_download_repo(repo_id)
    except Exception:
        if _hf_local_files_only():
            return None
        raise

    return _stage_component_sources(component_sources)


def prepare_gen3c_checkpoint_root(checkpoint_dir: Optional[str]) -> str:
    """Prepare gen3c checkpoint root.

    Args:
        checkpoint_dir: The checkpoint dir.

    Returns:
        The return value.
    """
    candidates = []
    if checkpoint_dir is not None:
        candidates.append(Path(checkpoint_dir).expanduser())

    for candidate in candidates:
        if candidate.exists() and _checkpoint_layout_complete(candidate):
            return str(candidate.resolve())
        if candidate.exists():
            sources = _direct_layout_component_sources(candidate)
            if sources is not None:
                staged = _stage_component_sources(sources)
                if _checkpoint_layout_complete(staged):
                    return str(staged)

    for candidate in _iter_direct_checkpoint_roots():
        if candidate.exists() and _checkpoint_layout_complete(candidate):
            return str(candidate.resolve())
        if candidate.exists():
            sources = _direct_layout_component_sources(candidate)
            if sources is not None:
                staged = _stage_component_sources(sources)
                if _checkpoint_layout_complete(staged):
                    return str(staged)

    for candidate in candidates:
        for hfd_root in _iter_hfd_roots(candidate if candidate.exists() else candidate.parent):
            staged = _stage_hfd_checkpoints(hfd_root)
            if staged is not None:
                return str(staged)

    for hfd_root in _iter_hfd_roots(None):
        staged = _stage_hfd_checkpoints(hfd_root)
        if staged is not None:
            return str(staged)

    staged = _stage_hf_snapshot_checkpoints()
    if staged is not None and _checkpoint_layout_complete(staged):
        return str(staged)

    raise FileNotFoundError(
        "Unable to locate a complete GEN3C checkpoint root. Expected either a directory containing "
        "'Gen3C-Cosmos-7B/', 'Cosmos-Tokenize1-CV8x8x8-720p/', and 'google-t5/t5-11b/', or cached "
        "HFD repos for those components under ${WORLDFOUNDRY_HFD_ROOT}."
    )


def build_subprocess_env(
    device: Optional[str] = None,
) -> dict[str, str]:
    """Build subprocess env.

    Args:
        device: The device.

    Returns:
        The return value.
    """
    env = os.environ.copy()
    runtime_root = Path(resolve_gen3c_runtime_root())
    pythonpath_entries = []
    staged_prepend = env.get("WORLDFOUNDRY_STUDIO_CHILD_PYTHONPATH_PREPEND", "").strip()
    if staged_prepend:
        pythonpath_entries.append(staged_prepend)
    pythonpath_entries.append(str(runtime_root))
    packaged_root = packaged_gen3c_cosmos_root()
    if packaged_root != runtime_root:
        pythonpath_entries.append(str(packaged_root))
    pythonpath_entries.append(str(project_root() / "src"))
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TERM", "dumb")
    env.setdefault("TERMINFO_DIRS", "/usr/share/terminfo:/lib/terminfo")
    env.pop("TERMINFO", None)
    if env.get("NCCL_DEBUG", "").upper() in {"INFO", "TRACE"}:
        env["NCCL_DEBUG"] = "WARN"
    for key in ("NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "NCCL_IB_QPS_PER_CONNECTION"):
        env.pop(key, None)
    visible_device = cuda_visible_devices_from_device(device)
    if visible_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_device
    return env
