"""Lazy WorldFoundry runtime for the in-tree Cosmos3 Diffusers inference code."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .artifacts import (
    candidate_repo_dirs,
    candidate_repo_dirs_at_revision,
    checkpoint_revision,
    cosmos3_revision_for_repo_id,
    resolve_cosmos3_model_source,
    resolve_cosmos3_variant_id,
    resolve_local_artifact_path,
    strip_cosmos3_loader_metadata,
)


_SUPPORTED_TASKS = (
    "text-to-image",
    "text-to-video",
    "image-to-video",
    "video-to-video",
    "world-generation",
)
_REQUIRED_LAYOUT_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "vae/config.json",
)
_SUPPORTED_PIPELINE_CLASSES = {"Cosmos3OmniDiffusersPipeline", "Cosmos3OmniPipeline"}


@dataclass(frozen=True)
class Cosmos3RuntimePlan:
    """Resolved local execution plan for one Cosmos3 checkpoint variant."""

    model_path: str
    variant_id: str
    revision: str | None
    resolved_model_path: str | None
    backend: str
    supported_tasks: tuple[str, ...]
    blocked: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Cosmos3RuntimeOutput:
    """Multimodal result returned for Cosmos3 sound or action inference."""

    video: Any
    sound: Any = None
    action: Any = None
    audio_sample_rate: int | None = None


def _missing_weight_files(component_dir: Path) -> tuple[str, ...]:
    """Validate either a single safetensors file or every shard named by an index."""

    index_paths = sorted(component_dir.glob("*.safetensors.index.json"))
    if index_paths:
        missing: list[str] = []
        for index_path in index_paths:
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                shard_names = set(index.get("weight_map", {}).values())
            except (OSError, json.JSONDecodeError, AttributeError):
                missing.append(f"{index_path.relative_to(component_dir.parent)} (valid JSON weight_map)")
                continue
            if not shard_names:
                missing.append(f"{index_path.relative_to(component_dir.parent)} (non-empty weight_map)")
                continue
            missing.extend(
                str((component_dir / shard_name).relative_to(component_dir.parent))
                for shard_name in sorted(shard_names)
                if not (component_dir / shard_name).is_file()
            )
        return tuple(missing)
    if any(candidate.is_file() for candidate in component_dir.glob("*.safetensors")):
        return ()
    return (f"{component_dir.name}/*.safetensors",)


def _missing_diffusers_layout_files(
    path: Path | None,
    *,
    load_sound_tokenizer: bool = True,
) -> tuple[str, ...]:
    if path is None or not path.is_dir():
        return _REQUIRED_LAYOUT_FILES

    missing = [relative_path for relative_path in _REQUIRED_LAYOUT_FILES if not (path / relative_path).is_file()]
    model_index_path = path / "model_index.json"
    has_sound_tokenizer = False
    if model_index_path.is_file():
        try:
            model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            missing.append("model_index.json (valid JSON)")
        else:
            pipeline_class = model_index.get("_class_name")
            if pipeline_class not in _SUPPORTED_PIPELINE_CLASSES:
                missing.append("model_index.json (_class_name=Cosmos3OmniPipeline)")
            for component in ("scheduler", "text_tokenizer", "transformer", "vae"):
                if not model_index.get(component):
                    missing.append(f"model_index.json ({component} component)")
            has_sound_tokenizer = bool(model_index.get("sound_tokenizer"))

    if not (path / "text_tokenizer/tokenizer.json").is_file() and not (
        (path / "text_tokenizer/vocab.json").is_file() and (path / "text_tokenizer/merges.txt").is_file()
    ):
        missing.append("text_tokenizer/tokenizer.json (or vocab.json + merges.txt)")
    missing.extend(_missing_weight_files(path / "transformer"))
    missing.extend(_missing_weight_files(path / "vae"))
    if load_sound_tokenizer:
        if not has_sound_tokenizer:
            missing.append("model_index.json (sound_tokenizer component)")
        else:
            if not (path / "sound_tokenizer/config.json").is_file():
                missing.append("sound_tokenizer/config.json")
            missing.extend(_missing_weight_files(path / "sound_tokenizer"))
    return tuple(dict.fromkeys(missing))


def _has_diffusers_layout(path: Path | None, *, load_sound_tokenizer: bool = True) -> bool:
    return not _missing_diffusers_layout_files(path, load_sound_tokenizer=load_sound_tokenizer)


def _select_checkpoint_candidate(
    model_source: str,
    revision: str | None = None,
    *,
    load_sound_tokenizer: bool = True,
) -> Path | None:
    """Select a complete cached snapshot without silently changing path or revision."""

    explicit_path = Path(model_source).expanduser()
    if explicit_path.exists():
        if revision is not None and checkpoint_revision(explicit_path) != revision:
            return None
        return explicit_path.resolve()

    candidates = (
        candidate_repo_dirs_at_revision(model_source, revision)
        if revision is not None
        else candidate_repo_dirs(model_source)
    )
    for candidate in candidates:
        if _has_diffusers_layout(candidate, load_sound_tokenizer=load_sound_tokenizer):
            return candidate
    return candidates[0] if candidates else None


def _torch_dtype(dtype: Any) -> Any:
    if dtype is None or not isinstance(dtype, str):
        return dtype

    import torch

    normalized = dtype.lower().replace("torch.", "")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported Cosmos3 torch dtype: {dtype!r}")
    return aliases[normalized]


def _create_safety_checker() -> Any:
    """Initialize the official guardrail before loading the much larger generator."""

    try:
        from cosmos_guardrail import CosmosSafetyChecker
    except ImportError as exc:
        raise RuntimeError(
            "Cosmos3 safety checking requires `cosmos-guardrail==0.3.1`. Install the "
            "Cosmos3 runtime environment or explicitly set `enable_safety_checker=False`."
        ) from exc

    try:
        return CosmosSafetyChecker()
    except Exception as exc:
        detail = str(exc).lower()
        if any(
            marker in detail
            for marker in ("401", "403", "authorized list", "gated repo", "access to model")
        ):
            raise RuntimeError(
                "Cosmos3 safety checking requires approved Hugging Face access to "
                "`nvidia/Cosmos-1.0-Guardrail`. Request access, provide an authorized HF_TOKEN, "
                "and retry; or explicitly set `enable_safety_checker=False` for an unscreened run."
            ) from exc
        raise RuntimeError(
            "Failed to initialize the official Cosmos3 safety checker. Fix the guardrail runtime "
            "or explicitly set `enable_safety_checker=False` for an unscreened run."
        ) from exc


def _runtime_request(
    model_path: str | Mapping[str, Any] | None,
    options: Mapping[str, Any],
) -> tuple[str, str, str | None, dict[str, Any]]:
    request = dict(model_path) if isinstance(model_path, Mapping) else {}
    request.update(options)
    selector_model_id = request.get("model_id")
    source_request: str | Mapping[str, Any] | None = model_path if isinstance(model_path, str) else request
    source = resolve_cosmos3_model_source(source_request, model_id=selector_model_id)
    variant_id = resolve_cosmos3_variant_id(
        request,
        model_id=selector_model_id,
        model_source=source,
    )
    requested_revision = str(request.get("revision") or "").strip() or None
    revision = requested_revision or cosmos3_revision_for_repo_id(source)
    return source, variant_id, revision, strip_cosmos3_loader_metadata(request)


def _pipeline_cls() -> type[Any]:
    from .diffusers_cosmos3 import Cosmos3OmniPipeline

    return Cosmos3OmniPipeline


def _transformer_cls() -> type[Any]:
    from .diffusers_cosmos3 import Cosmos3OmniTransformer

    return Cosmos3OmniTransformer


def _balanced_decoder_device_map(
    transformer_path: Path,
    device_map: Any,
    *,
    device: str,
) -> Any:
    """Keep Cosmos3 packing/output heads together while balancing decoder layers.

    Accelerate's generic ``balanced`` map also moves the projection and rotary
    modules to the last GPU. Cosmos3 performs indexing and packing outside those
    modules, so that generic layout mixes the first and last GPU during forward.
    This map explicitly pins every non-layer state prefix to the execution GPU
    and assigns only complete decoder layers to other GPUs. It deliberately has
    no root entry because root hooks override descendant execution hooks.
    """

    if device_map is None:
        return device_map

    import torch

    parsed_device = torch.device(device)
    if parsed_device.type != "cuda":
        raise ValueError(
            f"Cosmos3 sharded device maps require a CUDA execution device, got {device!r}."
        )
    main_device = (
        int(parsed_device.index)
        if parsed_device.index is not None
        else int(torch.cuda.current_device())
    )
    try:
        config = json.loads((transformer_path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot build a Cosmos3 device map without a valid transformer config: {exc}") from exc
    num_layers = int(config.get("num_hidden_layers") or 0)
    if num_layers < 1:
        raise ValueError("Cosmos3 transformer config does not declare a positive num_hidden_layers.")
    state_prefixes: set[str] = set()
    for index_path in sorted(transformer_path.glob("*.index.json")):
        try:
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        weight_map = index_payload.get("weight_map")
        if isinstance(weight_map, Mapping):
            state_prefixes.update(str(name).split(".", 1)[0] for name in weight_map)
    if not state_prefixes:
        try:
            from safetensors import safe_open

            for weights_path in sorted(transformer_path.glob("*.safetensors")):
                with safe_open(weights_path, framework="pt", device="cpu") as handle:
                    state_prefixes.update(str(name).split(".", 1)[0] for name in handle.keys())
        except (ImportError, OSError, RuntimeError):
            pass
    if not state_prefixes:
        state_prefixes.update(
            {
                "embed_tokens",
                "norm",
                "norm_moe_gen",
                "lm_head",
                "proj_in",
                "proj_out",
                "time_embedder",
            }
        )
        if bool(config.get("action_gen")):
            state_prefixes.update(
                {"action_modality_embed", "action_proj_in", "action_proj_out"}
            )
        if bool(config.get("sound_gen")):
            state_prefixes.update(
                {"audio_modality_embed", "audio_proj_in", "audio_proj_out"}
            )

    if isinstance(device_map, Mapping):
        custom_map = dict(device_map)

        def cuda_index(value: Any) -> int | None:
            if isinstance(value, int):
                return value
            mapped = torch.device(value)
            if mapped.type != "cuda":
                return None
            return int(mapped.index) if mapped.index is not None else int(torch.cuda.current_device())

        if "" in custom_map:
            if len(custom_map) != 1:
                raise ValueError(
                    "Unsafe Cosmos3 device map: a root entry cannot be combined with decoder-layer overrides."
                )
            if cuda_index(custom_map[""]) != main_device:
                raise ValueError("Cosmos3's single-device root map must target the execution CUDA device.")
            return custom_map

        required_non_layers = sorted(name for name in state_prefixes if name != "layers")
        required_layers = [f"layers.{index}" for index in range(num_layers)]
        missing = [name for name in (*required_non_layers, *required_layers) if name not in custom_map]
        if missing:
            raise ValueError(
                "Incomplete Cosmos3 device map; explicitly map every non-layer state prefix and decoder layer. "
                f"Missing: {missing}."
            )
        misplaced = [
            name for name in required_non_layers if cuda_index(custom_map[name]) != main_device
        ]
        if misplaced:
            raise ValueError(
                "Cosmos3 packing/projection state must remain on the execution CUDA device. "
                f"Misplaced: {misplaced}."
            )
        return custom_map

    if device_map != "balanced":
        raise ValueError(
            "Cosmos3 string device maps support only `balanced`, which preserves packing/projection "
            "device invariants; pass a complete explicit device-map dict for a custom placement."
        )
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        raise RuntimeError(
            "Cosmos3 `device_map='balanced'` requires at least two visible CUDA GPUs. "
            "Expose a high-memory multi-GPU configuration or pass a complete custom device-map dict."
        )
    if main_device >= gpu_count:
        raise ValueError(
            f"Cosmos3 execution device {device!r} is not available; visible CUDA device count is {gpu_count}."
        )
    gpu_ids = [main_device, *(index for index in range(gpu_count) if index != main_device)]
    base, remainder = divmod(num_layers, len(gpu_ids))

    # Do not use a root entry here. Accelerate propagates a root execution hook to
    # descendants before installing layer overrides, which pulls sharded layer inputs
    # back to the main device while their weights remain on another GPU.
    result: dict[str, int] = {
        name: main_device for name in sorted(state_prefixes) if name != "layers"
    }
    # These parameter-free modules must follow the packing device as well.
    result.update({"rotary_emb": main_device, "time_proj": main_device})
    layer_index = 0
    for rank, gpu_id in enumerate(gpu_ids):
        layer_count = base + (1 if rank < remainder else 0)
        for _ in range(layer_count):
            result[f"layers.{layer_index}"] = gpu_id
            layer_index += 1
    return result


class Cosmos3Runtime:
    """Local Cosmos3 runtime backed by the pinned in-tree Diffusers implementation."""

    def __init__(
        self,
        pipeline: Any,
        model_path: str,
        *,
        variant_id: str = "cosmos3-nano",
        revision: str | None = None,
        device: str = "cuda",
        backend: str = "diffusers",
    ) -> None:
        self.pipeline = pipeline
        self.model_path = model_path
        self.variant_id = variant_id
        self.revision = revision
        self.device = device
        self.backend = backend
        scheduler = getattr(pipeline, "scheduler", None)
        self._scheduler_config = dict(getattr(scheduler, "config", {}) or {})

    @classmethod
    def plan(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        backend: str = "diffusers",
        **kwargs: Any,
    ) -> dict[str, Any]:
        raw_model_path, variant_id, revision, plan_kwargs = _runtime_request(model_path, kwargs)
        load_sound_tokenizer = bool(plan_kwargs.get("load_sound_tokenizer", True))
        resolved = _select_checkpoint_candidate(
            raw_model_path,
            revision,
            load_sound_tokenizer=load_sound_tokenizer,
        )
        blockers: list[str] = []
        if backend != "diffusers":
            blockers.append(f"Unsupported Cosmos3 backend: {backend!r}.")
        if resolved is None:
            if revision is not None:
                stale = [
                    f"{candidate} ({checkpoint_revision(candidate) or 'unverified'})"
                    for candidate in candidate_repo_dirs(raw_model_path)
                ]
                blockers.append(
                    f"Missing Cosmos3 checkpoint at pinned revision {revision}. "
                    f"Cached non-matching candidates: {stale or 'none'}."
                )
            else:
                blockers.append(
                    "Missing Cosmos3 checkpoint. Download the official Diffusers repository into "
                    "WORLDFOUNDRY_HFD_ROOT or the Hugging Face Hub cache."
                )
        else:
            missing = _missing_diffusers_layout_files(
                resolved,
                load_sound_tokenizer=load_sound_tokenizer,
            )
            if missing:
                blockers.append(
                    "Cosmos3 checkpoint staging is incomplete; missing: " + ", ".join(missing)
                )

        return Cosmos3RuntimePlan(
            model_path=raw_model_path,
            variant_id=variant_id,
            revision=revision,
            resolved_model_path=None if resolved is None else str(resolved),
            backend=backend,
            supported_tasks=_SUPPORTED_TASKS,
            blocked=bool(blockers),
            blockers=tuple(blockers),
        ).to_dict()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        *,
        device: str = "cuda",
        backend: str = "diffusers",
        torch_dtype: Any = "bfloat16",
        device_map: Any = None,
        enable_safety_checker: bool = True,
        load_sound_tokenizer: bool = True,
        **kwargs: Any,
    ) -> "Cosmos3Runtime":
        if backend != "diffusers":
            raise ValueError(f"Unsupported Cosmos3 backend: {backend!r}")
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE"):
            value = os.environ.get(name)
            if value is not None and value.strip().lower() not in {"1", "true", "yes", "on"}:
                raise ValueError(
                    f"Cosmos3 inference requires offline model loading; {name}={value!r} is not allowed."
                )
            os.environ[name] = "1"

        raw_model_path, variant_id, revision, load_kwargs = _runtime_request(model_path, kwargs)
        requested_local_only = load_kwargs.pop("local_files_only", True)
        if requested_local_only is not True:
            raise ValueError("Cosmos3 inference does not allow local_files_only=False.")
        mapped_load_sound = bool(load_kwargs.pop("load_sound_tokenizer", load_sound_tokenizer))
        resolved_model_path = _select_checkpoint_candidate(
            raw_model_path,
            revision,
            load_sound_tokenizer=mapped_load_sound,
        )
        if resolved_model_path is None:
            if revision is not None:
                raise FileNotFoundError(
                    f"Cosmos3 {variant_id} requires pinned revision {revision}; no matching local snapshot was found. "
                    "Run scripts/inference/prepare_model_infer.sh --model cosmos3 --download and use the reported "
                    "exact model-ref."
                )
            resolved_model_path = resolve_local_artifact_path(raw_model_path)
        missing = _missing_diffusers_layout_files(
            resolved_model_path,
            load_sound_tokenizer=mapped_load_sound,
        )
        if missing:
            raise FileNotFoundError(
                "Cosmos3 requires a complete downloaded official Diffusers repository; missing "
                f"{', '.join(missing)} under {resolved_model_path}."
            )

        mapped_dtype = load_kwargs.pop("torch_dtype", torch_dtype)
        mapped_device_map = load_kwargs.pop("device_map", device_map)
        mapped_safety = bool(load_kwargs.pop("enable_safety_checker", enable_safety_checker))
        safety_checker = load_kwargs.pop("safety_checker", None)
        load_kwargs["torch_dtype"] = _torch_dtype(mapped_dtype)
        if mapped_safety:
            load_kwargs["safety_checker"] = (
                safety_checker if safety_checker is not None else _create_safety_checker()
            )
        if not mapped_load_sound:
            # Diffusers treats an explicitly supplied None as an override for the
            # component declared in model_index.json, avoiding the ~1.9 GB AVAE.
            load_kwargs["sound_tokenizer"] = None

        if mapped_device_map is not None:
            # DiffusionPipeline-level device maps distribute whole components. Cosmos3-Super's
            # transformer is larger than one 80 GB GPU, so loading the pipeline with
            # device_map="balanced" offloads that entire component to CPU. Load the transformer
            # independently so Accelerate can split its decoder blocks across visible GPUs.
            if mapped_device_map == "balanced" and "max_memory" in load_kwargs:
                raise ValueError(
                    "Cosmos3's safe balanced map does not infer placements from max_memory. "
                    "Remove max_memory or provide a complete explicit device-map dict."
                )
            transformer_device_map = _balanced_decoder_device_map(
                resolved_model_path / "transformer",
                mapped_device_map,
                device=device,
            )
            transformer_kwargs = {
                "torch_dtype": load_kwargs["torch_dtype"],
                "device_map": transformer_device_map,
                "local_files_only": True,
            }
            for key in ("max_memory", "offload_folder", "offload_state_dict", "low_cpu_mem_usage"):
                if key in load_kwargs:
                    transformer_kwargs[key] = load_kwargs.pop(key)
            load_kwargs["transformer"] = _transformer_cls().from_pretrained(
                str(resolved_model_path / "transformer"),
                **transformer_kwargs,
            )

        pipeline = _pipeline_cls().from_pretrained(
            str(resolved_model_path),
            enable_safety_checker=mapped_safety,
            local_files_only=True,
            **load_kwargs,
        )
        if mapped_device_map is None and hasattr(pipeline, "to"):
            pipeline = pipeline.to(device)
        elif mapped_device_map is not None:
            # The transformer is already block-sharded. Move only the smaller decode components
            # to the execution GPU; calling pipeline.to() would collapse the transformer shards.
            for component_name in ("vae", "sound_tokenizer"):
                component = getattr(pipeline, component_name, None)
                if component is not None and hasattr(component, "to"):
                    component.to(device)
        return cls(
            pipeline=pipeline,
            model_path=str(resolved_model_path),
            variant_id=variant_id,
            revision=revision,
            device=device,
            backend=backend,
        )

    def api_init(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError("Cosmos3Runtime is a local in-tree runtime and does not use an API backend.")

    def predict(
        self,
        prompt: str | list[str],
        *,
        image: Any = None,
        video: Any = None,
        negative_prompt: str | list[str] | None = None,
        num_frames: int = 189,
        height: int = 720,
        width: int = 1280,
        fps: float = 24.0,
        num_inference_steps: int = 35,
        guidance_scale: float = 6.0,
        enable_sound: bool = False,
        action: Any = None,
        flow_shift: float | None = None,
        use_karras_sigmas: bool | None = None,
        seed: int | None = None,
        output_type: str = "video",
        return_omni_output: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        if enable_sound and getattr(self.pipeline, "sound_tokenizer", None) is None:
            raise ValueError(
                "Cosmos3 sound generation was requested, but the AVAE sound tokenizer is not loaded. "
                "Reload the runtime with `load_sound_tokenizer=True` (the default)."
            )

        generator = kwargs.pop("generator", None)
        if seed is not None and generator is None:
            import torch

            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        scheduler = getattr(self.pipeline, "scheduler", None)
        scheduler_type = type(scheduler)
        if scheduler is None or not hasattr(scheduler_type, "from_config"):
            raise RuntimeError("Cosmos3 pipeline scheduler cannot be configured for inference.")
        official_shifted_mode = action is not None or (num_frames > 1 and (video is not None or image is None))
        resolved_flow_shift = float(flow_shift) if flow_shift is not None else (10.0 if official_shifted_mode else 1.0)
        resolved_karras = (
            bool(use_karras_sigmas)
            if use_karras_sigmas is not None
            else not official_shifted_mode
        )
        self.pipeline.scheduler = scheduler_type.from_config(
            self._scheduler_config,
            flow_shift=resolved_flow_shift,
            use_karras_sigmas=resolved_karras,
        )

        normalized_output_type = {"video": "pt", "tensor": "pt"}.get(output_type, output_type)
        kwargs.pop("return_dict", None)
        call_num_frames = None if action is not None else num_frames
        call_height = None if action is not None else height
        call_width = None if action is not None else width
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            video=video,
            num_frames=call_num_frames,
            height=call_height,
            width=call_width,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            enable_sound=enable_sound,
            action=action,
            generator=generator,
            output_type=normalized_output_type,
            return_dict=True,
            **kwargs,
        )
        video_output = getattr(result, "video", result)
        sound_output = getattr(result, "sound", None)
        action_output = getattr(result, "action", None)
        wants_omni_output = bool(return_omni_output) or enable_sound or action is not None
        if not wants_omni_output:
            return video_output

        sample_rate = None
        sound_tokenizer = getattr(self.pipeline, "sound_tokenizer", None)
        sound_config = getattr(sound_tokenizer, "config", None)
        if sound_output is not None and sound_config is not None:
            configured_sample_rate = getattr(sound_config, "sampling_rate", None)
            if configured_sample_rate is not None:
                sample_rate = int(configured_sample_rate)
        return Cosmos3RuntimeOutput(
            video=video_output,
            sound=sound_output,
            action=action_output,
            audio_sample_rate=sample_rate,
        )


__all__ = ["Cosmos3Runtime", "Cosmos3RuntimeOutput", "Cosmos3RuntimePlan"]
