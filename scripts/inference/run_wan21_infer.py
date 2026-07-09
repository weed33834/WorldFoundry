import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" if (ROOT / "src" / "worldfoundry").is_dir() else ROOT
PACKAGE_ROOT = SOURCE_ROOT / "worldfoundry"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from worldfoundry.pipelines.wan.pipeline_wan_2p1_i2v import Wan2p1I2VPipeline
from worldfoundry.pipelines.wan.pipeline_wan_2p1_t2v import Wan2p1T2VPipeline


DEFAULT_IMAGE = str(PACKAGE_ROOT / "data" / "test_cases" / "test_image_case1" / "ref_image.png")
DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
    "intensely on a spotlighted stage."
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float | None = None) -> float | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _require_existing_path(path_value: str, label: str) -> str:
    path = Path(path_value).expanduser()
    if path.exists():
        return str(path)
    if path.is_absolute() or path_value.startswith("."):
        raise FileNotFoundError(f"{label} not found: {path_value}")
    return path_value


def _add_if_set(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


generation_type = os.getenv("WAN21_GENERATION_TYPE", "i2v").strip().lower()
if generation_type not in {"t2v", "i2v"}:
    raise ValueError("WAN21_GENERATION_TYPE must be 't2v' or 'i2v'.")

model_path = _require_existing_path(
    os.environ.get("WAN21_MODEL_PATH", "Wan-AI/Wan2.1-I2V-14B-480P"),
    "Wan2.1 model path",
)
task = os.getenv("WAN21_TASK", "t2v-1.3B" if generation_type == "t2v" else "i2v-14B")
size = os.getenv("WAN21_SIZE", "832*480")
frame_num = int(os.getenv("WAN21_FRAME_NUM", "81"))
fps = int(os.getenv("WAN21_FPS", "16"))
output_path = os.getenv("WAN21_OUTPUT_PATH", "./wan21_output.mp4")
prompt = os.getenv("WAN21_PROMPT", DEFAULT_PROMPT)

runtime_components: dict[str, Any] = {
    "task": task,
    "size": size,
    "frames": frame_num,
    "fps": fps,
    "offload_model": _env_bool("WAN21_OFFLOAD_MODEL", True),
    "t5_cpu": _env_bool("WAN21_T5_CPU", True),
    "t5_fsdp": _env_bool("WAN21_T5_FSDP", False),
    "dit_fsdp": _env_bool("WAN21_DIT_FSDP", False),
    "ulysses_size": int(os.getenv("WAN21_ULYSSES_SIZE", "1")),
    "ring_size": int(os.getenv("WAN21_RING_SIZE", "1")),
    "sample_solver": os.getenv("WAN21_SAMPLE_SOLVER", "unipc"),
}
_add_if_set(runtime_components, "sample_steps", _env_int("WAN21_SAMPLE_STEPS"))
_add_if_set(runtime_components, "sample_shift", _env_float("WAN21_SAMPLE_SHIFT"))
_add_if_set(runtime_components, "sample_guide_scale", _env_float("WAN21_SAMPLE_GUIDE_SCALE"))
_add_if_set(runtime_components, "base_seed", _env_int("WAN21_SEED"))

pipeline_cls = Wan2p1T2VPipeline if generation_type == "t2v" else Wan2p1I2VPipeline
pipeline = pipeline_cls.from_pretrained(
    model_path=model_path,
    required_components=runtime_components,
    lazy=True,
)

image = None
if generation_type == "i2v":
    image_path = _require_existing_path(os.getenv("WAN21_IMAGE_PATH", DEFAULT_IMAGE), "Wan2.1 input image")
    image = Image.open(image_path).convert("RGB")

Path(output_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
result = pipeline(
    prompt=prompt,
    images=image,
    output_path=output_path,
    fps=fps,
    return_dict=True,
)

print(f"Wan2.1 {task} video saved to: {result.get('generated_video_path') or output_path}")
