"""Dispatch Studio inference jobs into isolated Conda runtime environments.

WorldFoundry Studio typically runs in a base Python environment, while individual
models require dedicated Conda envs (different CUDA/PyTorch stacks). This module:

1. Resolves which Conda env a model should use (``workspace_runtime_spec``).
2. Decides whether to spawn a child process (``dispatch_spec_for_inference``).
3. Remaps GPU indices and torchrun settings for the child (``_run_kwargs_for_child``).
4. Launches ``worldfoundry.studio.runtime_job`` inside that env and streams logs
   back to the Studio UI (``run_manager_payload_in_conda``).

The child process is marked with ``WORLDFOUNDRY_STUDIO_CONDA_CHILD=1`` so nested
calls do not re-dispatch indefinitely.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import select
import signal
import subprocess
import sys
import time
import codecs
import tempfile
import atexit
import threading
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

from worldfoundry.core.io.paths import project_root, resolve_worldfoundry_path, worldfoundry_path_tokens
from worldfoundry.runtime import resolve_ckpt_dir, resolve_hfd_root, resolve_hf_cache_dir
from worldfoundry.runtime.conda import RuntimeCondaEnvSpec, apply_unified_env_override, load_runtime_conda_env_spec

from .execution import RunRecord, TORCHRUN_DISTRIBUTED_ENV
from .launch_config import resolve_lingbot_fast_num_procs, wmfactory_interactive_model_spec


# Environment variable set in child processes to prevent recursive dispatch.
STUDIO_CONDA_CHILD_ENV = "WORLDFOUNDRY_STUDIO_CONDA_CHILD"
# Comma-separated model IDs (or "all"/"*") that always run in a subprocess.
FORCE_SUBPROCESS_MODELS_ENV = "WORLDFOUNDRY_STUDIO_FORCE_SUBPROCESS_MODELS"
RESIDENT_WORKERS_ENV = "WORLDFOUNDRY_STUDIO_RESIDENT_WORKERS"
RESIDENT_WORKER_MODELS_ENV = "WORLDFOUNDRY_STUDIO_RESIDENT_WORKER_MODELS"
RESIDENT_WORKER_REQUEST_TIMEOUT_ENV = "WORLDFOUNDRY_STUDIO_RESIDENT_WORKER_REQUEST_TIMEOUT"
DISPATCH_API_KEY_ENV = "WORLDFOUNDRY_STUDIO_DISPATCH_API_KEY"
CHILD_PYTHONPATH_PREPEND_ENV = "WORLDFOUNDRY_STUDIO_CHILD_PYTHONPATH_PREPEND"
# Placeholder key in serialized payload; actual secret is passed via env.
SECRET_ENV_REF_KEY = "__worldfoundry_secret_env__"
DEFAULT_FORCE_SUBPROCESS_MODELS = frozenset(
    {
        "cameractrl",
        "hunyuan-game-craft",
        "hunyuan-gamecraft",
        "hunyuan-world-voyager",
        "lingbot-world",
        "hunyuan-worldplay",
        "wan-2p2",
        "matrix-game-2",
        "matrix-game-3",
    }
)
LINGBOT_WORLD_MODEL_ID = "lingbot-world"
TORCHRUN_LINGBOT_ENV = "WORLDFOUNDRY_STUDIO_TORCHRUN_LINGBOT_FAST"
TORCHRUN_MODEL_NPROC_KEYS: Mapping[str, tuple[str, ...]] = {
    "hunyuan-game-craft": ("torchrun_nproc_per_node", "torchrun_nproc", "nproc_per_node"),
    "hunyuan-gamecraft": ("torchrun_nproc_per_node", "torchrun_nproc", "nproc_per_node"),
    "hunyuan-world-voyager": (
        "torchrun_nproc_per_node",
        "torchrun_nproc",
        "nproc_per_node",
        "ulysses_degree",
    ),
    "hunyuan-worldplay": ("torchrun_nproc_per_node", "torchrun_nproc", "nproc_per_node"),
}
DISPATCH_ONLY_LOAD_KWARGS = frozenset(
    {
        "cuda_visible_devices",
        "visible_devices",
        "cuda_devices",
        "gpu_ids",
        "torchrun_nproc_per_node",
        "torchrun_nproc",
        "nproc_per_node",
    }
)
DISPATCH_ONLY_CALL_KWARGS = frozenset(
    {
        # In a dispatched conda child, nested official entrypoints must run with
        # the child env's interpreter instead of the parent Studio default.
        "python_executable",
    }
)

LogCallback = Callable[[str, str], None]
CancelCallback = Callable[[], bool]


@dataclass
class _ResidentWorker:
    key: tuple[str, ...]
    model_id: str
    process: subprocess.Popen[Any]
    lock: threading.RLock
    decoder: codecs.IncrementalDecoder
    command: list[str]
    created_at: float
    last_used_at: float


class _ResidentWorkerUnavailable(RuntimeError):
    """Raised when resident-worker startup fails before a request can run."""


@dataclass(frozen=True)
class _ResidentRunContext:
    child_run_kwargs: dict[str, Any]
    payload_run_kwargs: dict[str, Any]
    env: dict[str, str]
    key: tuple[str, ...]


_RESIDENT_WORKERS: dict[tuple[str, ...], _ResidentWorker] = {}
_RESIDENT_WORKERS_LOCK = threading.RLock()


def _env_flag(name: str) -> bool:
    """Return True when an environment variable is set to a truthy string."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def in_conda_child_process() -> bool:
    """Return True when running inside a Studio-dispatched Conda child process."""
    return _env_flag(STUDIO_CONDA_CHILD_ENV)


def _force_subprocess_for_model(model_id: str) -> bool:
    """Return True when the model must run in a subprocess even if Python paths match."""
    raw = os.getenv(FORCE_SUBPROCESS_MODELS_ENV)
    if raw is None:
        candidates = DEFAULT_FORCE_SUBPROCESS_MODELS
    else:
        candidates = frozenset(item.strip() for item in raw.split(",") if item.strip())
    return "*" in candidates or "all" in candidates or model_id in candidates


def _resident_workers_enabled_for_model(model_id: str) -> bool:
    """Return True when eligible conda jobs may reuse a long-lived worker."""
    value = os.getenv(RESIDENT_WORKERS_ENV, "auto").strip().lower()
    if value in {"0", "false", "no", "off", "never"}:
        return False
    raw_models = os.getenv(RESIDENT_WORKER_MODELS_ENV, "").strip()
    if not raw_models:
        return True
    candidates = {item.strip() for item in raw_models.split(",") if item.strip()}
    if not candidates:
        return True
    return "*" in candidates or "all" in candidates or model_id in candidates


def _resident_worker_request_timeout() -> float:
    try:
        return max(float(os.getenv(RESIDENT_WORKER_REQUEST_TIMEOUT_ENV, "0") or "0"), 0.0)
    except ValueError:
        return 0.0


def _same_executable(left: str | Path, right: str | Path) -> bool:
    """Compare two Python interpreter paths after resolving symlinks."""
    try:
        return Path(left).expanduser().absolute() == Path(right).expanduser().absolute()
    except OSError:
        return str(left) == str(right)


def _dedupe_paths(values: list[Path]) -> tuple[Path, ...]:
    rows: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path not in rows:
            rows.append(path)
    return tuple(rows)


def _dedupe_text(values: tuple[str, ...]) -> tuple[str, ...]:
    rows: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in rows:
            rows.append(text)
    return tuple(rows)


def _candidate_env_roots(configured: Path) -> tuple[Path, ...]:
    """Collect likely Conda env root directories to search for a model runtime."""
    tokens = worldfoundry_path_tokens()
    repo = _repo_root()
    conda_root = Path(tokens["WORLDFOUNDRY_CONDA_ROOT"]).expanduser()
    conda_envs_root = Path(tokens["WORLDFOUNDRY_CONDA_ENVS_ROOT"]).expanduser()
    candidates = [
        configured,
        conda_envs_root,
        conda_root / "envs",
        conda_root,
        repo.parent / "conda" / "envs",
        repo.parent / "conda",
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix).expanduser()
        candidates.extend([prefix.parent, prefix.parent.parent / "envs"])
    if configured.name == "conda":
        candidates.append(configured / "envs")
    elif configured.name == "envs":
        candidates.append(configured.parent)
    return _dedupe_paths(candidates)


def _candidate_env_names(model_id: str, env_name: str) -> tuple[str, ...]:
    """Generate common Conda env name variants for a model (hyphen/underscore aliases)."""
    return _dedupe_text((env_name, model_id, model_id.replace("-", "_"), model_id.replace("_", "-")))


def _discover_existing_env_spec(spec: RuntimeCondaEnvSpec) -> RuntimeCondaEnvSpec | None:
    """Locate an on-disk Conda env when the configured path does not exist yet."""
    if spec.exists:
        return spec
    for root in _candidate_env_roots(spec.env_root):
        for name in _candidate_env_names(spec.model_id, spec.env_name):
            candidate = root / name
            if (candidate / "bin" / "python").is_file():
                return replace(
                    spec,
                    env_name=name,
                    env_root=root,
                    notes=spec.notes + (f"studio_env_discovery:{candidate}",),
                )
    return None


@lru_cache(maxsize=None)
def workspace_runtime_spec(model_id: str) -> RuntimeCondaEnvSpec | None:
    """Resolve the Conda runtime spec for a model, with discovery and unified-env fallback."""
    spec = load_runtime_conda_env_spec(model_id)
    if spec is not None:
        discovered = _discover_existing_env_spec(spec)
        if discovered is not None:
            return discovered

    fallback = RuntimeCondaEnvSpec(model_id=model_id, env_name=model_id)
    unified_fallback = apply_unified_env_override(fallback)
    if unified_fallback != fallback:
        discovered = _discover_existing_env_spec(unified_fallback)
        if discovered is not None:
            return discovered
    return _discover_existing_env_spec(fallback)


def dispatch_spec_for_inference(model_id: str, *, backend: str = "auto") -> RuntimeCondaEnvSpec | None:
    """Return a Conda spec when inference should run in a separate env, else None.

    Dispatch is skipped when already in a child process, when using ``api_init``,
    when no runtime env exists, or when the current interpreter already matches
    the target env (unless the model is force-subprocess listed).
    """
    if in_conda_child_process():
        return None
    if str(backend or "auto").strip() == "api_init":
        return None
    spec = workspace_runtime_spec(model_id)
    if spec is None:
        return None
    if _same_executable(sys.executable, spec.python_executable) and not _force_subprocess_for_model(model_id):
        return None
    return spec


def _visible_device_items(value: str | None) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _cuda_device_index(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text.startswith("cuda:"):
        return None
    index = text.split(":", maxsplit=1)[1].strip()
    return index if index.isdigit() else None


def _model_cuda_device_indices(run_kwargs: Mapping[str, Any]) -> list[str]:
    indices: list[str] = []
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    call_kwargs = _json_object_from_text(run_kwargs.get("call_kwargs_text"))
    for source in (load_kwargs, call_kwargs):
        for key, value in source.items():
            if key != "device" and not str(key).endswith("_device"):
                continue
            index = _cuda_device_index(value)
            if index is not None and index not in indices:
                indices.append(index)
    return indices


def _expand_visible_devices_for_model_devices(
    visible_devices: str,
    run_kwargs: Mapping[str, Any],
) -> str:
    visible = _visible_device_items(visible_devices)
    if not visible:
        return visible_devices
    model_indices = _model_cuda_device_indices(run_kwargs)
    if not model_indices or not any(index in visible for index in model_indices):
        return visible_devices
    expanded = list(visible)
    for index in model_indices:
        if index not in expanded:
            expanded.append(index)
    return ",".join(expanded)


def _rewrite_cuda_devices_for_child(
    values: Mapping[str, Any],
    visible_devices: str,
) -> dict[str, Any]:
    """Remap ``cuda:N`` device strings to child-local indices under CUDA_VISIBLE_DEVICES."""
    visible = _visible_device_items(visible_devices)
    if not visible:
        return dict(values)
    rewritten = dict(values)
    for key, value in values.items():
        if key != "device" and not str(key).endswith("_device"):
            continue
        index = _cuda_device_index(value)
        if index is not None and index in visible:
            rewritten[key] = f"cuda:{visible.index(index)}"
    return rewritten


def _run_kwargs_for_child(
    run_kwargs: Mapping[str, Any],
    *,
    cuda_visible_devices: str = "",
) -> dict[str, Any]:
    """Prepare run kwargs for a dispatched child: remap devices and strip dispatch-only keys."""
    child_kwargs = dict(run_kwargs)
    device = str(child_kwargs.get("device") or "")
    if device.startswith("cuda:"):
        # _runtime_env pins CUDA_VISIBLE_DEVICES to the selected physical GPU.
        # Inside that child process the selected GPU is exposed as logical cuda:0.
        child_kwargs["device"] = "cuda"
    load_kwargs = _json_object_from_text(child_kwargs.get("load_kwargs_text"))
    if load_kwargs:
        original_load_kwargs = dict(load_kwargs)
        load_kwargs = _rewrite_cuda_devices_for_child(load_kwargs, cuda_visible_devices)
        cleaned = {key: value for key, value in load_kwargs.items() if key not in DISPATCH_ONLY_LOAD_KWARGS}
        if cleaned != original_load_kwargs:
            child_kwargs["load_kwargs_text"] = json.dumps(cleaned)
    call_kwargs = _json_object_from_text(child_kwargs.get("call_kwargs_text"))
    if call_kwargs:
        original_call_kwargs = dict(call_kwargs)
        call_kwargs = _rewrite_cuda_devices_for_child(call_kwargs, cuda_visible_devices)
        cleaned = {key: value for key, value in call_kwargs.items() if key not in DISPATCH_ONLY_CALL_KWARGS}
        if cleaned != original_call_kwargs:
            child_kwargs["call_kwargs_text"] = json.dumps(cleaned)
    return child_kwargs


def _json_object_from_text(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _truthy_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _official_video_internal_torchrun_nproc(
    model_id: str,
    *,
    call_kwargs: Mapping[str, Any],
    load_kwargs: Mapping[str, Any],
) -> int:
    """Return nproc for official video configs whose command launches torchrun."""
    config_path = _package_root() / "data" / "models" / "runtime" / "configs" / "video_official" / f"{model_id}.yaml"
    if not config_path.is_file():
        return 0
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    if "{torchrun}" not in text:
        return 0
    requested = (
        call_kwargs.get("nproc_per_node")
        or call_kwargs.get("torchrun_nproc_per_node")
        or call_kwargs.get("torchrun_nproc")
        or load_kwargs.get("nproc_per_node")
        or load_kwargs.get("torchrun_nproc_per_node")
        or load_kwargs.get("torchrun_nproc")
    )
    if requested in {None, ""}:
        match = re.search(r"(?m)^\s+nproc_per_node:\s*([0-9]+)\s*$", text)
        requested = match.group(1) if match else 0
    try:
        return int(requested or 0)
    except Exception:
        return 0


def _lingbot_torchrun_nproc(model_id: str, run_kwargs: Mapping[str, Any]) -> int:
    """Resolve torchrun process count for LingBot-World (FSDP / ulysses fast path)."""
    if model_id != LINGBOT_WORLD_MODEL_ID:
        return 0
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    requested = load_kwargs.get("torchrun_nproc_per_node") or load_kwargs.get("ulysses_size")
    try:
        nproc = int(requested or 0)
    except Exception:
        nproc = 0
    if nproc > 1:
        return nproc
    if _truthy_value(load_kwargs.get("dit_fsdp")) or _truthy_value(load_kwargs.get("t5_fsdp")):
        visible_count = _cuda_visible_device_count(os.getenv("CUDA_VISIBLE_DEVICES"))
        return resolve_lingbot_fast_num_procs(visible_count=visible_count or None)
    return 0


def _explicit_torchrun_nproc(model_id: str, run_kwargs: Mapping[str, Any]) -> int:
    """Return explicit multi-GPU torchrun count from load/call kwargs, or 0."""
    requested = _requested_torchrun_nproc(model_id, run_kwargs)
    return requested if requested is not None and requested > 1 else 0


def _requested_torchrun_nproc(model_id: str, run_kwargs: Mapping[str, Any]) -> int | None:
    keys = TORCHRUN_MODEL_NPROC_KEYS.get(model_id)
    if not keys:
        return None
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    call_kwargs = _json_object_from_text(run_kwargs.get("call_kwargs_text"))
    for source in (load_kwargs, call_kwargs):
        for key in keys:
            if key not in source or source.get(key) in {None, ""}:
                continue
            try:
                return int(source.get(key) or 0)
            except Exception:
                return None
    return None


def _default_torchrun_nproc(model_id: str, run_kwargs: Mapping[str, Any]) -> int:
    """Apply model-specific default torchrun sizing when the user did not specify nproc."""
    if _requested_torchrun_nproc(model_id, run_kwargs) is not None:
        return 0
    if model_id in {"hunyuan-game-craft", "hunyuan-gamecraft"}:
        return int(os.getenv("WORLDFOUNDRY_STUDIO_GAMECRAFT_TORCHRUN_NPROC", "8") or "8")
    if model_id != "hunyuan-worldplay":
        return 0
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    wm_spec = wmfactory_interactive_model_spec(model_id, load_kwargs=load_kwargs)
    return int(wm_spec.preferred_visible_devices or 0) if wm_spec is not None else 0


def _internal_torchrun_nproc(model_id: str, run_kwargs: Mapping[str, Any]) -> int:
    """Return torchrun nproc for models that launch distributed jobs inside their own code."""
    call_kwargs = _json_object_from_text(run_kwargs.get("call_kwargs_text"))
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    if model_id == "gen3c":
        requested = call_kwargs.get("num_gpus") or load_kwargs.get("num_gpus")
    elif model_id == "wan2.1-vace":
        requested = (
            call_kwargs.get("nproc_per_node")
            or call_kwargs.get("torchrun_nproc_per_node")
            or load_kwargs.get("nproc_per_node")
            or load_kwargs.get("torchrun_nproc_per_node")
            or 8
        )
    elif model_id == "skyreels-v3":
        if not _truthy_value(call_kwargs.get("use_usp")):
            return 0
        requested = (
            call_kwargs.get("nproc_per_node")
            or call_kwargs.get("torchrun_nproc_per_node")
            or call_kwargs.get("torchrun_nproc")
            or load_kwargs.get("nproc_per_node")
            or load_kwargs.get("torchrun_nproc_per_node")
            or load_kwargs.get("torchrun_nproc")
        )
    elif model_id == "kairos-sensenova":
        requested = (
            call_kwargs.get("nproc_per_node")
            or call_kwargs.get("torchrun_nproc_per_node")
            or call_kwargs.get("torchrun_nproc")
            or load_kwargs.get("nproc_per_node")
            or load_kwargs.get("torchrun_nproc_per_node")
            or load_kwargs.get("torchrun_nproc")
        )
    else:
        return _official_video_internal_torchrun_nproc(
            model_id,
            call_kwargs=call_kwargs,
            load_kwargs=load_kwargs,
        )
    try:
        return int(requested or 0)
    except Exception:
        return 0


def _cuda_visible_device_count(value: str | None) -> int:
    if not value:
        return 0
    return len([item for item in value.split(",") if item.strip()])


def _parse_nvidia_smi_rows() -> list[dict[str, float]]:
    """Query GPU memory/utilization stats via ``nvidia-smi`` for auto device selection."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(command, capture_output=True, text=True, check=True)
    rows: list[dict[str, float]] = []
    for raw_line in proc.stdout.strip().splitlines():
        parts = [part.strip().replace("%", "") for part in raw_line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "memory_used": float(parts[1]),
                "memory_total": float(parts[2]),
                "utilization": float(parts[3]),
            }
        )
    return rows


def _select_wmfactory_visible_devices(num_devices: int | None = None) -> str:
    """Pick idle GPUs for WMFactory-style models using nvidia-smi heuristics."""
    requested = None if num_devices is None else max(int(num_devices or 1), 1)
    if os.getenv("WM_AUTO_CUDA_VISIBLE_DEVICES", "1") != "1":
        return ""

    try:
        max_mem_fraction = float(os.getenv("WM_AUTO_GPU_MAX_MEMORY_FRACTION", "0.5"))
        max_util_fraction = float(os.getenv("WM_AUTO_GPU_MAX_UTILIZATION_FRACTION", "0.5"))
    except ValueError:
        max_mem_fraction = 0.5
        max_util_fraction = 0.5

    try:
        rows = _parse_nvidia_smi_rows()
    except Exception:
        rows = []
    if rows:
        def _rank_key(row: Mapping[str, float]) -> tuple[float, float, float]:
            return (
                0.0 if row["memory_total"] <= 0 else row["memory_used"] / row["memory_total"],
                row["utilization"],
                row["index"],
            )

        ranked = sorted(
            rows,
            key=_rank_key,
        )

        eligible = []
        for row in rows:
            mem_fraction = 0.0 if row["memory_total"] <= 0 else row["memory_used"] / row["memory_total"]
            util_fraction = row["utilization"] / 100.0
            if mem_fraction < max_mem_fraction and util_fraction < max_util_fraction:
                eligible.append(row)

        if requested is None:
            picked = eligible or ranked[:1]
        else:
            ranked_eligible = sorted(eligible, key=_rank_key)
            picked = ranked_eligible[:requested] if len(ranked_eligible) >= requested else ranked[:requested]
        if picked:
            return ",".join(str(int(row["index"])) for row in picked)

    visible = _normalize_cuda_visible_devices(os.getenv("CUDA_VISIBLE_DEVICES", ""))
    if visible:
        if requested is None:
            return visible
        parts = [part.strip() for part in visible.split(",") if part.strip()]
        return ",".join(parts[:requested])
    return ""


def _wmfactory_visible_devices_for_run(model_id: str, run_kwargs: Mapping[str, Any]) -> str:
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    spec = wmfactory_interactive_model_spec(model_id, load_kwargs=load_kwargs)
    if spec is None:
        return ""
    override = os.getenv(f"WM_{spec.env_prefix}_CUDA_VISIBLE_DEVICES")
    if override is not None:
        return _normalize_cuda_visible_devices(override)
    return _select_wmfactory_visible_devices(spec.preferred_visible_devices)


def _apply_wmfactory_env_hints(model_id: str, run_kwargs: Mapping[str, Any], env: dict[str, str]) -> None:
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    spec = wmfactory_interactive_model_spec(model_id, load_kwargs=load_kwargs)
    if spec is None:
        return
    visible = [part.strip() for part in str(env.get("CUDA_VISIBLE_DEVICES", "")).split(",") if part.strip()]
    if spec.use_dual_device_hint:
        env.setdefault(f"WM_{spec.env_prefix}_GEN_DEVICE", "cuda:0")
        env.setdefault(f"WM_{spec.env_prefix}_DECODE_DEVICE", "cuda:1" if len(visible) >= 2 else "cuda:0")


def _normalize_cuda_visible_devices(value: Any) -> str:
    if value in {None, ""}:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return _normalize_cuda_visible_devices(parsed)
    return text


def _cuda_visible_devices_from_kwargs(run_kwargs: Mapping[str, Any]) -> str:
    load_kwargs = _json_object_from_text(run_kwargs.get("load_kwargs_text"))
    call_kwargs = _json_object_from_text(run_kwargs.get("call_kwargs_text"))
    for source in (load_kwargs, call_kwargs):
        for key in ("cuda_visible_devices", "visible_devices", "cuda_devices", "gpu_ids"):
            value = _normalize_cuda_visible_devices(source.get(key))
            if value:
                return value
    return ""


def _cuda_visible_devices_from_device(run_kwargs: Mapping[str, Any]) -> str:
    device = str(run_kwargs.get("device") or "").strip()
    if not device.startswith("cuda:"):
        return ""
    gpu_idx = device.split(":", maxsplit=1)[1].strip()
    if not gpu_idx.isdigit():
        return ""
    parent_visible = _visible_device_items(os.getenv("CUDA_VISIBLE_DEVICES", ""))
    if parent_visible:
        logical_index = int(gpu_idx)
        if 0 <= logical_index < len(parent_visible):
            return parent_visible[logical_index]
    return gpu_idx


def _explicit_cuda_visible_devices_for_run(run_kwargs: Mapping[str, Any]) -> str:
    return _cuda_visible_devices_from_kwargs(run_kwargs) or _cuda_visible_devices_from_device(run_kwargs)


def _torchrun_executable(spec: RuntimeCondaEnvSpec) -> str:
    candidate = Path(str(spec.python_executable)).with_name("torchrun")
    return str(candidate if candidate.is_file() else "torchrun")


def _terminate_process_tree(process: subprocess.Popen[Any], *, force: bool = False) -> None:
    """Terminate a subprocess and its children when it owns a process group."""
    if process.poll() is not None:
        return
    if sys.platform != "win32":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL if force else signal.SIGTERM)
            return
        except ProcessLookupError:
            return
        except Exception:
            pass
    if force:
        process.kill()
    else:
        process.terminate()


def _shutdown_resident_worker(worker: _ResidentWorker, *, force: bool = False) -> None:
    process = worker.process
    if process.poll() is None:
        try:
            if process.stdin is not None and not force:
                process.stdin.write(b'{"command":"shutdown"}\n')
                process.stdin.flush()
        except Exception:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process, force=force)
    try:
        if process.stdin is not None:
            process.stdin.close()
    except Exception:
        pass
    try:
        if process.stdout is not None:
            process.stdout.close()
    except Exception:
        pass


def _shutdown_all_resident_workers() -> None:
    with _RESIDENT_WORKERS_LOCK:
        workers = list(_RESIDENT_WORKERS.values())
        _RESIDENT_WORKERS.clear()
    for worker in workers:
        _shutdown_resident_worker(worker, force=False)


atexit.register(_shutdown_all_resident_workers)


def _drain_process_output(
    *,
    process: subprocess.Popen[Any],
    decoder: codecs.IncrementalDecoder,
    log_callback: LogCallback | None,
) -> bool:
    """Drain currently available child output without waiting for a newline."""
    if process.stdout is None:
        return True
    fd = process.stdout.fileno()
    reached_eof = False
    while True:
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            break
        except InterruptedError:
            continue
        except OSError:
            reached_eof = True
            break
        if not chunk:
            reached_eof = True
            break
        text = decoder.decode(chunk)
        if text:
            _append_log(log_callback, "stdout", text)
    return reached_eof


def run_record_from_manifest(payload: Mapping[str, Any]) -> RunRecord:
    output_dir = str(payload.get("output_dir") or "")
    return RunRecord(
        run_id=str(payload.get("run_id") or Path(output_dir).name),
        model_id=str(payload.get("model_id") or ""),
        display_name=str(payload.get("display_name") or payload.get("model_id") or ""),
        mode=str(payload.get("mode") or ""),
        status=str(payload.get("status") or ""),
        output_dir=output_dir,
        manifest_path=str(payload.get("manifest_path") or (Path(output_dir) / "manifest.json")),
        preview_video=payload.get("preview_video") or None,
        preview_image=payload.get("preview_image") or None,
        preview_splat=payload.get("preview_splat") or None,
        preview_model=payload.get("preview_model") or None,
        gallery=[str(item) for item in payload.get("gallery") or ()],
        rrd_path=payload.get("rrd_path") or None,
        artifacts=[str(item) for item in payload.get("artifacts") or ()],
        metadata=dict(payload.get("metadata") or {}),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _payload_run_kwargs_with_secret_refs(run_kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    payload_kwargs = dict(run_kwargs)
    secret_env: dict[str, str] = {}
    api_key = payload_kwargs.get("api_key")
    if isinstance(api_key, str) and api_key:
        payload_kwargs["api_key"] = {SECRET_ENV_REF_KEY: DISPATCH_API_KEY_ENV}
        secret_env[DISPATCH_API_KEY_ENV] = api_key
    return payload_kwargs, secret_env


def _repo_root() -> Path:
    return project_root(Path(__file__).resolve())


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _runtime_pythonpath(spec: RuntimeCondaEnvSpec, env: Mapping[str, str]) -> str:
    repo = _repo_root()
    paths: list[str] = []
    staged_prepend = os.getenv(CHILD_PYTHONPATH_PREPEND_ENV, "").strip()
    if staged_prepend:
        paths.append(staged_prepend)
    src_root = repo / "src"
    if (src_root / "worldfoundry").is_dir():
        paths.append(str(src_root))
    if (repo / "worldfoundry").is_dir():
        paths.append(str(repo))
    for item in spec.pythonpath_dirs:
        resolved = resolve_worldfoundry_path(str(item), env)
        if not resolved.is_absolute():
            resolved = repo / resolved
        paths.append(str(resolved))
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    deduped: list[str] = []
    for item in paths:
        if item and item not in deduped:
            deduped.append(item)
    return os.pathsep.join(deduped)


def _venv_base_prefix(env_prefix: Path) -> Path | None:
    pyvenv_cfg = env_prefix / "pyvenv.cfg"
    if not pyvenv_cfg.is_file():
        return None
    try:
        lines = pyvenv_cfg.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        key, _, value = line.partition("=")
        if key.strip().lower() == "home" and value.strip():
            home = Path(value.strip())
            return home.parent if home.name == "bin" else home
    return None


def _runtime_library_dirs(spec: RuntimeCondaEnvSpec, env: Mapping[str, str]) -> str:
    prefixes: list[Path] = [spec.env_prefix]
    base_prefix = _venv_base_prefix(spec.env_prefix)
    if base_prefix is not None:
        prefixes.append(base_prefix)

    dirs: list[str] = []
    for prefix in prefixes:
        candidates = [prefix / "lib"]
        candidates.extend(prefix.glob("lib/python*/site-packages/torch/lib"))
        candidates.extend(prefix.glob("lib/python*/site-packages/nvidia/*/lib"))
        for candidate in candidates:
            if candidate.is_dir():
                item = str(candidate)
                if item not in dirs:
                    dirs.append(item)

    shim_dirs = _runtime_library_shim_dirs(spec, dirs)
    dirs = shim_dirs + dirs

    existing = env.get("LD_LIBRARY_PATH", "")
    for item in existing.split(os.pathsep):
        if item and item not in dirs:
            dirs.append(item)
    return os.pathsep.join(dirs)


def _runtime_library_shim_dirs(spec: RuntimeCondaEnvSpec, library_dirs: list[str]) -> list[str]:
    """Create transient soname shims required by some NVIDIA Python wheels."""
    digest = hashlib.sha1(str(spec.env_prefix).encode("utf-8")).hexdigest()[:12]
    root = Path(
        os.getenv(
            "WORLDFOUNDRY_RUNTIME_LIB_SHIM_DIR",
            str(Path(tempfile.gettempdir()) / "worldfoundry-runtime-lib-shims"),
        )
    )
    shim_dir = root / digest
    created = False
    for library_name in ("libcudnn.so",):
        if any((Path(item) / library_name).exists() for item in library_dirs):
            continue
        target: Path | None = None
        for item in library_dirs:
            candidates = sorted(Path(item).glob(f"{library_name}.*"))
            if candidates:
                target = candidates[0]
                break
        if target is None:
            continue
        try:
            shim_dir.mkdir(parents=True, exist_ok=True)
            link_path = shim_dir / library_name
            if link_path.is_symlink() or link_path.exists():
                try:
                    if link_path.resolve() == target.resolve():
                        created = True
                        continue
                except OSError:
                    pass
                if link_path.is_symlink():
                    link_path.unlink()
                else:
                    continue
            link_path.symlink_to(target)
            created = True
        except OSError:
            continue
    return [str(shim_dir)] if created else []


def _runtime_cache_dir(spec: RuntimeCondaEnvSpec, name: str) -> str:
    root = Path(
        os.getenv(
            "WORLDFOUNDRY_RUNTIME_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "worldfoundry-runtime-cache"),
        )
    )
    path = root / spec.model_id / name
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(path)


def _runtime_env(spec: RuntimeCondaEnvSpec, device: str | None = None) -> dict[str, str]:
    """Build the child-process environment: Conda paths, PYTHONPATH, and optional GPU pin."""
    env = os.environ.copy()
    # Studio inference jobs are expected to resolve missing public/gated model
    # assets when a token/network is available. Do not let a parent shell's
    # offline debugging flags silently turn official demo jobs into failures.
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    env.pop("HF_DATASETS_OFFLINE", None)
    env[STUDIO_CONDA_CHILD_ENV] = "1"
    env["WORLDFOUNDRY_REPO_ROOT"] = str(_repo_root())
    ckpt_dir = resolve_ckpt_dir(env)
    hfd_root = resolve_hfd_root(env)
    env.setdefault("WORLDFOUNDRY_CKPT_DIR", str(ckpt_dir))
    env.setdefault("WORLDFOUNDRY_HFD_ROOT", str(hfd_root))
    local_hf_home = ckpt_dir / "huggingface"
    if local_hf_home.is_dir():
        env.setdefault("HF_HOME", str(local_hf_home))
        env.setdefault("HF_HUB_CACHE", str(local_hf_home / "hub"))
        env.setdefault("HUGGINGFACE_HUB_CACHE", str(local_hf_home / "hub"))
    else:
        env.setdefault("HF_HUB_CACHE", str(resolve_hf_cache_dir(env)))
    env["CONDA_PREFIX"] = str(spec.env_prefix)
    env["WORLDFOUNDRY_CONDA_ENVS_ROOT"] = str(spec.env_prefix.parent)
    env["WORLDFOUNDRY_CONDA_ENV_ROOT"] = str(spec.env_prefix.parent)
    if spec.env_prefix.parent.name == "envs":
        env["WORLDFOUNDRY_CONDA_ROOT"] = str(spec.env_prefix.parent.parent)
    env["PATH"] = os.pathsep.join([str(spec.env_prefix / "bin"), env.get("PATH", "")])
    env["LD_LIBRARY_PATH"] = _runtime_library_dirs(spec, env)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TERM", "dumb")
    env.setdefault("TERMINFO_DIRS", "/usr/share/terminfo:/lib/terminfo")
    env.pop("TERMINFO", None)
    if spec.model_id == "gen3c":
        cuda_home_override = os.getenv("WORLDFOUNDRY_CUDA_HOME_OVERRIDE", "").strip()
        if cuda_home_override.lower() in {"0", "none", "unset"}:
            env.pop("CUDA_HOME", None)
        else:
            env["CUDA_HOME"] = cuda_home_override or str(spec.env_prefix)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("WORLDFOUNDRY_GEN3C_IMPORT_STAGGER_SECONDS", "1")
        env.setdefault("TORCH_COMPILE_DISABLE", "1")
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", _runtime_cache_dir(spec, "torchinductor"))
        env.setdefault("TRITON_CACHE_DIR", _runtime_cache_dir(spec, "triton"))
        env.setdefault("CUDA_CACHE_PATH", _runtime_cache_dir(spec, "cuda"))
        env.setdefault("XDG_CACHE_HOME", _runtime_cache_dir(spec, "xdg"))
    env["PYTHONPATH"] = _runtime_pythonpath(spec, env)
    # Map device string like "cuda:3" to CUDA_VISIBLE_DEVICES=3
    # so each job can run on a specific GPU instead of all defaulting to GPU 0.
    if device and device.startswith("cuda"):
        gpu_idx = device.split(":")[1] if ":" in device else None
        if gpu_idx is not None and gpu_idx.isdigit():
            env["CUDA_VISIBLE_DEVICES"] = gpu_idx
    return env


def _prepare_resident_run_context(
    *,
    model_id: str,
    spec: RuntimeCondaEnvSpec,
    workspace_root: str,
    run_kwargs: Mapping[str, Any],
) -> _ResidentRunContext | None:
    if not _resident_workers_enabled_for_model(model_id):
        return None

    requested_torchrun_nproc = (
        _lingbot_torchrun_nproc(model_id, run_kwargs)
        or _explicit_torchrun_nproc(model_id, run_kwargs)
        or _default_torchrun_nproc(model_id, run_kwargs)
    )
    requested_internal_torchrun_nproc = _internal_torchrun_nproc(model_id, run_kwargs)
    if requested_torchrun_nproc > 1 or requested_internal_torchrun_nproc > 1:
        return None

    kwargs_cuda_visible_devices = _cuda_visible_devices_from_kwargs(run_kwargs)
    device_cuda_visible_devices = _cuda_visible_devices_from_device(run_kwargs)
    parent_cuda_visible_devices = _normalize_cuda_visible_devices(os.getenv("CUDA_VISIBLE_DEVICES", ""))
    explicit_cuda_visible_devices = _expand_visible_devices_for_model_devices(
        kwargs_cuda_visible_devices
        or device_cuda_visible_devices
        or parent_cuda_visible_devices,
        run_kwargs,
    )
    child_run_kwargs = _run_kwargs_for_child(
        run_kwargs,
        cuda_visible_devices=explicit_cuda_visible_devices,
    )
    payload_run_kwargs, secret_env = _payload_run_kwargs_with_secret_refs(child_run_kwargs)
    if secret_env:
        return None

    wmfactory_visible_devices = "" if explicit_cuda_visible_devices else _wmfactory_visible_devices_for_run(model_id, run_kwargs)
    env = _runtime_env(spec, device=None if wmfactory_visible_devices else str(run_kwargs.get("device", "")))
    if explicit_cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = explicit_cuda_visible_devices
    elif wmfactory_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = wmfactory_visible_devices
    _apply_wmfactory_env_hints(model_id, run_kwargs, env)

    key = (
        model_id,
        str(spec.python_executable),
        str(spec.env_prefix),
        str(Path(workspace_root).expanduser()),
        env.get("CUDA_VISIBLE_DEVICES", ""),
        env.get("WORLDFOUNDRY_CKPT_DIR", ""),
        env.get("WORLDFOUNDRY_HFD_ROOT", ""),
        env.get("PYTHONPATH", ""),
    )
    return _ResidentRunContext(
        child_run_kwargs=child_run_kwargs,
        payload_run_kwargs=payload_run_kwargs,
        env=env,
        key=key,
    )


def _start_resident_worker(
    *,
    model_id: str,
    spec: RuntimeCondaEnvSpec,
    workspace_root: str,
    context: _ResidentRunContext,
    log_callback: LogCallback | None,
) -> _ResidentWorker:
    command = [
        str(spec.python_executable),
        "-m",
        "worldfoundry.studio.runtime_job",
        "run-manager-worker",
        "--workspace-root",
        workspace_root,
    ]
    process = subprocess.Popen(
        command,
        cwd=str(_repo_root()),
        env=context.env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=sys.platform != "win32",
    )
    assert process.stdout is not None
    try:
        os.set_blocking(process.stdout.fileno(), False)
    except (AttributeError, OSError):
        pass
    worker = _ResidentWorker(
        key=context.key,
        model_id=model_id,
        process=process,
        lock=threading.RLock(),
        decoder=codecs.getincrementaldecoder("utf-8")("replace"),
        command=command,
        created_at=time.monotonic(),
        last_used_at=time.monotonic(),
    )
    _append_log(
        log_callback,
        "system",
        f"started resident conda worker for {model_id} in {spec.resolved_env_name}: {spec.python_executable}\n",
    )
    return worker


def _resident_worker_for(
    *,
    model_id: str,
    spec: RuntimeCondaEnvSpec,
    workspace_root: str,
    context: _ResidentRunContext,
    log_callback: LogCallback | None,
) -> _ResidentWorker:
    with _RESIDENT_WORKERS_LOCK:
        existing = _RESIDENT_WORKERS.get(context.key)
        if existing is not None and existing.process.poll() is None:
            return existing
        if existing is not None:
            _RESIDENT_WORKERS.pop(context.key, None)
        worker = _start_resident_worker(
            model_id=model_id,
            spec=spec,
            workspace_root=workspace_root,
            context=context,
            log_callback=log_callback,
        )
        _RESIDENT_WORKERS[context.key] = worker
        return worker


def _drop_resident_worker(worker: _ResidentWorker, *, force: bool = False) -> None:
    with _RESIDENT_WORKERS_LOCK:
        if _RESIDENT_WORKERS.get(worker.key) is worker:
            _RESIDENT_WORKERS.pop(worker.key, None)
    _shutdown_resident_worker(worker, force=force)


def _send_resident_worker_request(
    worker: _ResidentWorker,
    *,
    request_payload: Mapping[str, Any],
) -> None:
    if worker.process.poll() is not None:
        raise _ResidentWorkerUnavailable(f"resident worker exited before request: {worker.process.returncode}")
    if worker.process.stdin is None:
        raise _ResidentWorkerUnavailable("resident worker stdin is closed")
    data = json.dumps(request_payload, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
    try:
        worker.process.stdin.write(data)
        worker.process.stdin.flush()
    except BrokenPipeError as exc:
        raise _ResidentWorkerUnavailable("resident worker pipe closed before request") from exc


def _wait_for_resident_worker_result(
    worker: _ResidentWorker,
    *,
    result_path: Path,
    error_path: Path,
    log_callback: LogCallback | None,
    cancel_requested: CancelCallback | None,
) -> RunRecord:
    timeout = _resident_worker_request_timeout()
    deadline = time.monotonic() + timeout if timeout > 0 else None
    stdout_eof = False
    process = worker.process
    if process.stdout is None:
        raise _ResidentWorkerUnavailable("resident worker stdout is closed")
    stdout_fd = process.stdout.fileno()

    while True:
        if cancel_requested is not None and cancel_requested():
            _append_log(log_callback, "system", "termination requested for resident inference worker\n")
            _drop_resident_worker(worker, force=True)
            raise RuntimeError("resident inference worker cancelled")

        if result_path.is_file():
            worker.last_used_at = time.monotonic()
            _drain_process_output(process=process, decoder=worker.decoder, log_callback=log_callback)
            return run_record_from_manifest(json.loads(result_path.read_text(encoding="utf-8")))

        if error_path.is_file():
            worker.last_used_at = time.monotonic()
            _drain_process_output(process=process, decoder=worker.decoder, log_callback=log_callback)
            payload = json.loads(error_path.read_text(encoding="utf-8"))
            detail = payload.get("traceback") or payload.get("error") or f"see {error_path}"
            raise RuntimeError(str(detail))

        if process.poll() is not None:
            while not stdout_eof:
                stdout_eof = _drain_process_output(process=process, decoder=worker.decoder, log_callback=log_callback)
                if not stdout_eof:
                    time.sleep(0.05)
            with _RESIDENT_WORKERS_LOCK:
                if _RESIDENT_WORKERS.get(worker.key) is worker:
                    _RESIDENT_WORKERS.pop(worker.key, None)
            raise RuntimeError(f"resident worker for {worker.model_id} exited with status {process.returncode}")

        if deadline is not None and time.monotonic() > deadline:
            _drop_resident_worker(worker, force=True)
            raise TimeoutError(f"resident worker request timed out after {timeout:.1f}s")

        if not stdout_eof:
            readable, _, _ = select.select([stdout_fd], [], [], 0.5)
            if readable:
                stdout_eof = _drain_process_output(process=process, decoder=worker.decoder, log_callback=log_callback)
        else:
            time.sleep(0.05)


def _run_manager_payload_in_resident_conda(
    *,
    model_id: str,
    spec: RuntimeCondaEnvSpec,
    workspace_root: str,
    run_kwargs: Mapping[str, Any],
    dispatch_root: str | Path,
    log_callback: LogCallback | None = None,
    cancel_requested: CancelCallback | None = None,
) -> RunRecord | None:
    context = _prepare_resident_run_context(
        model_id=model_id,
        spec=spec,
        workspace_root=workspace_root,
        run_kwargs=run_kwargs,
    )
    if context is None:
        return None

    dispatch_dir = Path(dispatch_root)
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    request_id = f"{int(time.time() * 1000)}-{os.getpid()}-{threading.get_ident()}"
    payload_path = dispatch_dir / "payload.json"
    result_path = dispatch_dir / "result.json"
    error_path = dispatch_dir / "error.json"
    for stale_path in (result_path, error_path):
        try:
            stale_path.unlink()
        except FileNotFoundError:
            pass
    payload_path.write_text(
        json.dumps(
            {
                "workspace_root": workspace_root,
                "run_kwargs": _json_safe(context.payload_run_kwargs),
                "resident_worker": True,
                "request_id": request_id,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    worker = _resident_worker_for(
        model_id=model_id,
        spec=spec,
        workspace_root=workspace_root,
        context=context,
        log_callback=log_callback,
    )
    with worker.lock:
        _append_log(log_callback, "system", f"dispatching {model_id} to resident conda worker\n")
        _send_resident_worker_request(
            worker,
            request_payload={
                "request_id": request_id,
                "run_kwargs": _json_safe(context.payload_run_kwargs),
                "result_path": str(result_path),
                "error_path": str(error_path),
            },
        )
        return _wait_for_resident_worker_result(
            worker,
            result_path=result_path,
            error_path=error_path,
            log_callback=log_callback,
            cancel_requested=cancel_requested,
        )


def _append_log(log_callback: LogCallback | None, stream: str, text: str) -> None:
    if log_callback is not None:
        log_callback(stream, text)
    elif stream == "stderr":
        sys.stderr.write(text)
    else:
        sys.stdout.write(text)


def run_manager_payload_in_conda(
    *,
    model_id: str,
    spec: RuntimeCondaEnvSpec,
    workspace_root: str,
    run_kwargs: Mapping[str, Any],
    dispatch_root: str | Path,
    log_callback: LogCallback | None = None,
    cancel_requested: CancelCallback | None = None,
) -> RunRecord:
    """Run a Studio manager job in an isolated Conda env and return the child RunRecord.

    Writes ``payload.json``, spawns ``runtime_job run-manager-payload`` (optionally
    via ``torchrun``), streams stdout to ``log_callback``, and reads ``result.json``.
    """
    dispatch_dir = Path(dispatch_root)
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    try:
        resident_record = _run_manager_payload_in_resident_conda(
            model_id=model_id,
            spec=spec,
            workspace_root=workspace_root,
            run_kwargs=run_kwargs,
            dispatch_root=dispatch_dir,
            log_callback=log_callback,
            cancel_requested=cancel_requested,
        )
    except _ResidentWorkerUnavailable as exc:
        _append_log(log_callback, "system", f"resident worker unavailable; falling back to one-shot subprocess: {exc}\n")
        resident_record = None
    if resident_record is not None:
        return resident_record

    payload_path = dispatch_dir / "payload.json"
    result_path = dispatch_dir / "result.json"
    for stale_log in ("stdout.log", "stderr.log"):
        stale_log_path = dispatch_dir / stale_log
        if stale_log_path.exists():
            stale_log_path.unlink()
    # Resolve torchrun sizing from explicit kwargs, model defaults, or internal rules.
    requested_torchrun_nproc = (
        _lingbot_torchrun_nproc(model_id, run_kwargs)
        or _explicit_torchrun_nproc(model_id, run_kwargs)
        or _default_torchrun_nproc(model_id, run_kwargs)
    )
    requested_internal_torchrun_nproc = _internal_torchrun_nproc(model_id, run_kwargs)
    requested_multigpu_nproc = max(requested_torchrun_nproc, requested_internal_torchrun_nproc)
    kwargs_cuda_visible_devices = _cuda_visible_devices_from_kwargs(run_kwargs)
    device_cuda_visible_devices = _cuda_visible_devices_from_device(run_kwargs)
    parent_cuda_visible_devices = _normalize_cuda_visible_devices(os.getenv("CUDA_VISIBLE_DEVICES", ""))
    explicit_cuda_visible_devices = _expand_visible_devices_for_model_devices(
        kwargs_cuda_visible_devices
        or device_cuda_visible_devices
        or parent_cuda_visible_devices,
        run_kwargs,
    )
    # Strip parent-only kwargs and remap cuda:N indices for the child process.
    child_run_kwargs = _run_kwargs_for_child(
        run_kwargs,
        cuda_visible_devices=explicit_cuda_visible_devices,
    )
    payload_run_kwargs, secret_env = _payload_run_kwargs_with_secret_refs(child_run_kwargs)
    payload_path.write_text(
        json.dumps(
            {
                "workspace_root": workspace_root,
                "run_kwargs": _json_safe(payload_run_kwargs),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if result_path.exists():
        result_path.unlink()

    torchrun_nproc = _lingbot_torchrun_nproc(model_id, child_run_kwargs)
    if torchrun_nproc <= 1:
        torchrun_nproc = _explicit_torchrun_nproc(model_id, run_kwargs)
    if torchrun_nproc <= 1:
        torchrun_nproc = _default_torchrun_nproc(model_id, run_kwargs)
    internal_torchrun_nproc = _internal_torchrun_nproc(model_id, child_run_kwargs)
    wmfactory_visible_devices = "" if explicit_cuda_visible_devices else _wmfactory_visible_devices_for_run(model_id, run_kwargs)
    runtime_args = [
        "-m",
        "worldfoundry.studio.runtime_job",
        "run-manager-payload",
        "--payload-path",
        str(payload_path),
        "--result-path",
        str(result_path),
    ]
    # Multi-GPU models launch via torchrun; single-GPU uses the env's python directly.
    if torchrun_nproc > 1:
        command = [
            str(spec.python_executable),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={torchrun_nproc}",
            *runtime_args,
        ]
    else:
        command = [
            str(spec.python_executable),
            *runtime_args,
        ]
    _append_log(
        log_callback,
        "system",
        (
            f"dispatching {model_id} to conda env {spec.resolved_env_name} with torchrun "
            f"nproc={torchrun_nproc}: {spec.python_executable}\n"
            if torchrun_nproc > 1
            else f"dispatching {model_id} to conda env {spec.resolved_env_name}: {spec.python_executable}\n"
        ),
    )
    env = _runtime_env(
        spec,
        device=(
            None
            if torchrun_nproc > 1 or internal_torchrun_nproc > 1 or wmfactory_visible_devices
            else str(run_kwargs.get("device", ""))
        ),
    )
    if explicit_cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = explicit_cuda_visible_devices
    elif wmfactory_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = wmfactory_visible_devices
    env.update(secret_env)
    _apply_wmfactory_env_hints(model_id, run_kwargs, env)
    if torchrun_nproc > 1:
        if model_id == LINGBOT_WORLD_MODEL_ID:
            env[TORCHRUN_LINGBOT_ENV] = "1"
        if model_id in {"hunyuan-game-craft", "hunyuan-gamecraft", "hunyuan-worldplay"}:
            env[TORCHRUN_DISTRIBUTED_ENV] = "1"
    if torchrun_nproc > 1 or internal_torchrun_nproc > 1:
        requested_nproc = max(torchrun_nproc, internal_torchrun_nproc)
        visible_count = _cuda_visible_device_count(env.get("CUDA_VISIBLE_DEVICES"))
        if explicit_cuda_visible_devices and visible_count < requested_nproc:
            raise RuntimeError(
                f"{model_id} requested nproc={requested_nproc} but CUDA_VISIBLE_DEVICES="
                f"{explicit_cuda_visible_devices!r} exposes only {visible_count} device(s)"
            )
        # Let torchrun see all GPUs when the pinned set is too small for nproc.
        if visible_count < requested_nproc:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        env.setdefault("OMP_NUM_THREADS", "1")
    # Stream child stdout and honor cooperative cancellation (terminate, then kill).
    process = subprocess.Popen(
        command,
        cwd=str(_repo_root()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=sys.platform != "win32",
    )
    assert process.stdout is not None
    stdout_fd = process.stdout.fileno()
    try:
        os.set_blocking(stdout_fd, False)
    except (AttributeError, OSError):
        pass
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    termination_sent_at: float | None = None
    stdout_eof = False
    while True:
        if cancel_requested is not None and cancel_requested() and process.poll() is None:
            now = time.monotonic()
            if termination_sent_at is None:
                _terminate_process_tree(process, force=False)
                termination_sent_at = now
                _append_log(log_callback, "system", "termination requested for active inference process\n")
            elif now - termination_sent_at > 10:
                _terminate_process_tree(process, force=True)
                termination_sent_at = now
                _append_log(log_callback, "system", "active inference process did not exit after terminate; killed\n")

        if not stdout_eof:
            readable, _, _ = select.select([stdout_fd], [], [], 0.5)
            if readable:
                stdout_eof = _drain_process_output(
                    process=process,
                    decoder=decoder,
                    log_callback=log_callback,
                )
        else:
            time.sleep(0.05)

        if process.poll() is not None:
            while not stdout_eof:
                stdout_eof = _drain_process_output(
                    process=process,
                    decoder=decoder,
                    log_callback=log_callback,
                )
                if not stdout_eof:
                    time.sleep(0.05)
            remainder = decoder.decode(b"", final=True)
            if remainder:
                _append_log(log_callback, "stdout", remainder)
            break

    exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(f"{model_id} conda runtime exited with status {exit_code}: {spec.python_executable}")
    if not result_path.is_file():
        raise RuntimeError(f"{model_id} conda runtime did not write result payload: {result_path}")
    return run_record_from_manifest(json.loads(result_path.read_text(encoding="utf-8")))
