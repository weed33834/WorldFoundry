#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${WORLDFOUNDRY_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

MODE="${1:-stage-validation}"
if [[ $# -gt 0 ]]; then
  shift
fi

MODEL_ROOT="${WORLDFOUNDRY_MODEL_DIR:-${REPO_ROOT}/cache/models}"
SOURCE_ROOT="${WORLDFOUNDRY_MODEL_SOURCE_DIR:-${MODEL_ROOT}/sources}"
DATA_ROOT="${WORLDFOUNDRY_DATA_ROOT:-${WORLDFOUNDRY_DATA_DIR:-${REPO_ROOT}/cache/worldfoundry/data}}"
RUNTIME_ENVS_ROOT="${WORLDFOUNDRY_CONDA_ENVS_ROOT:-${REPO_ROOT}/cache/conda/envs}"
CKPT_ROOT="${WORLDFOUNDRY_CKPT_DIR:-${MODEL_ROOT}/checkpoints}"
WORLDSCORE_ROOT="${WORLDFOUNDRY_WORLDSCORE_ROOT:-${SOURCE_ROOT}/WorldScore}"
WORLDSCORE_DATA_PATH="${WORLDFOUNDRY_WORLDSCORE_DATA_PATH:-${DATA_ROOT}/Howieeeee/WorldScore}"
WORLDSCORE_CHECKPOINT_DIR="${WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR:-${REPO_ROOT}/cache/worldfoundry/assets/worldscore_official/metrics/checkpoints}"
WORLDFOUNDRY_PYTHON="${WORLDFOUNDRY_PYTHON:-python}"
WORLDSCORE_PYTHON="${WORLDSCORE_PYTHON:-${RUNTIME_ENVS_ROOT}/worldscore-runtime/bin/python}"
WORLDSCORE_BERT_BASE_UNCASED_PATH="${WORLDSCORE_BERT_BASE_UNCASED_PATH:-${CKPT_ROOT}/WorldScore/bert-base-uncased}"
WORLDSCORE_CLIP_VIT_BASE_PATCH16_PATH="${WORLDSCORE_CLIP_VIT_BASE_PATCH16_PATH:-${CKPT_ROOT}/WorldScore/openai--clip-vit-base-patch16}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/cache/huggingface}"
TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/cache/torch}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/cache/xdg}"
CUDA_HOME="${CUDA_HOME:-${RUNTIME_ENVS_ROOT}/lyra}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WORLDFOUNDRY_WORLDSCORE_ROOT="${WORLDSCORE_ROOT}"
export WORLDFOUNDRY_WORLDSCORE_DATA_PATH="${WORLDSCORE_DATA_PATH}"
export WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR="${WORLDSCORE_CHECKPOINT_DIR}"
export WORLDSCORE_BERT_BASE_UNCASED_PATH
export WORLDSCORE_CLIP_VIT_BASE_PATCH16_PATH
export HF_HOME
export TORCH_HOME
export XDG_CACHE_HOME
export CUDA_HOME
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1},download.pytorch.org"
export no_proxy="${no_proxy:-localhost,127.0.0.1},download.pytorch.org"

if [[ -x "${WORLDSCORE_PYTHON}" && -d "${CUDA_HOME}" ]]; then
  TORCH_LIB="$("${WORLDSCORE_PYTHON}" - <<'PY'
import pathlib
import torch

print(pathlib.Path(torch.__file__).parent / "lib")
PY
)"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${TORCH_LIB}:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_worldscore_eval_local.sh contract [output_dir]
  scripts/run_worldscore_eval_local.sh stage-validation [output_dir]
  scripts/run_worldscore_eval_local.sh normalize <worldscore.json> [generated_root] [output_dir]
  scripts/run_worldscore_eval_local.sh official-run <model_name> <model_path> [output_dir]

Environment overrides:
  WORLDFOUNDRY_REPO_ROOT
  WORLDFOUNDRY_MODEL_DIR
  WORLDFOUNDRY_MODEL_SOURCE_DIR
  WORLDFOUNDRY_WORLDSCORE_ROOT
  WORLDFOUNDRY_WORLDSCORE_DATA_PATH
  WORLDFOUNDRY_WORLDSCORE_ASSET_CHECKPOINT_DIR
  WORLDFOUNDRY_PYTHON
  WORLDSCORE_PYTHON
  WORLDSCORE_BERT_BASE_UNCASED_PATH
  WORLDSCORE_CLIP_VIT_BASE_PATCH16_PATH
  HF_HOME
  TORCH_HOME
  XDG_CACHE_HOME
  CUDA_HOME
USAGE
}

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "${path}" ]]; then
    echo "missing ${label}: ${path}" >&2
    exit 2
  fi
}

require_worldscore_basics() {
  require_path "WorldFoundry repo" "${REPO_ROOT}/pyproject.toml"
  require_path "WorldScore repo" "${WORLDSCORE_ROOT}/worldscore/run_evaluate.py"
  require_path "WorldScore data root" "${WORLDSCORE_DATA_PATH}"
  require_path "WorldScore dynamic manifest" "${WORLDSCORE_DATA_PATH}/WorldScore-Dataset/dynamic/dynamic.json"
  require_path "WorldScore static manifest" "${WORLDSCORE_DATA_PATH}/WorldScore-Dataset/static/static.json"
}

case "${MODE}" in
  -h|--help|help)
    usage
    ;;

  contract)
    require_worldscore_basics
    OUTPUT_DIR="${1:-${REPO_ROOT}/tmp/local_open_eval/worldscore_contract_local}"
    "${WORLDFOUNDRY_PYTHON}" -m worldfoundry.cli zoo benchmark-run \
      --benchmark-id worldscore \
      --mode contract \
      --generated-artifact-dir "${REPO_ROOT}/tmp/worldfoundry_all_benchmarks_contract/runs/fake-world-model__worldscore/generated_artifacts" \
      --output-dir "${OUTPUT_DIR}" \
      --json
    ;;

  stage-validation)
    require_worldscore_basics
    OUTPUT_DIR="${1:-${REPO_ROOT}/tmp/local_open_eval/worldscore_stage_validation_local}"
    FRAME_DIR="${OUTPUT_DIR}/source_frames"
    UPSTREAM_OUTPUT_DIR="${OUTPUT_DIR}/upstream_output"
    rm -rf "${OUTPUT_DIR}"
    "${WORLDFOUNDRY_PYTHON}" - <<PY
from pathlib import Path
from PIL import Image, ImageDraw

root = Path("${FRAME_DIR}")
root.mkdir(parents=True, exist_ok=True)
for i in range(4):
    image = Image.new("RGB", (320, 192), (40 + i * 35, 70, 150 - i * 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30 + i * 22, 60, 120 + i * 22, 145), fill=(220, 180 - i * 20, 60 + i * 30))
    draw.text((10, 10), f"worldscore validation {i:03d}", fill=(255, 255, 255))
    image.save(root / f"{i:03d}.png")
PY
    "${WORLDFOUNDRY_PYTHON}" "${REPO_ROOT}/worldfoundry/evaluation/tasks/execution/runners/worldscore/run_worldscore_official_runner.py" \
      --worldscore-root "${WORLDSCORE_ROOT}" \
      --data-path "${WORLDSCORE_DATA_PATH}" \
      --stage-dynamic-source "${FRAME_DIR}" \
      --stage-target-frames 4 \
      --worldscore-output-dir "${UPSTREAM_OUTPUT_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --stage-overwrite \
      --stage-only \
      --json
    ;;

  normalize)
    require_worldscore_basics
    RESULTS_PATH="${1:-}"
    if [[ -z "${RESULTS_PATH}" ]]; then
      usage >&2
      exit 2
    fi
    GENERATED_ROOT="${2:-${REPO_ROOT}/tmp/local_open_eval/worldscore_stage_validation_local/upstream_output}"
    OUTPUT_DIR="${3:-${REPO_ROOT}/tmp/local_open_eval/worldscore_normalize_local}"
    require_path "WorldScore result JSON" "${RESULTS_PATH}"
    "${WORLDFOUNDRY_PYTHON}" -m worldfoundry.cli zoo benchmark-run \
      --benchmark-id worldscore \
      --mode official-validation \
      --official-results-path "${RESULTS_PATH}" \
      --generated-artifact-dir "${GENERATED_ROOT}" \
      --output-dir "${OUTPUT_DIR}" \
      --env "WORLDFOUNDRY_WORLDSCORE_ROOT=${WORLDSCORE_ROOT}" \
      --env "WORLDFOUNDRY_WORLDSCORE_DATA_PATH=${WORLDSCORE_DATA_PATH}" \
      --json
    ;;

  official-run)
    require_worldscore_basics
    MODEL_NAME="${1:-}"
    MODEL_PATH="${2:-}"
    OUTPUT_DIR="${3:-${REPO_ROOT}/tmp/local_open_eval/worldscore_official_run_local}"
    if [[ -z "${MODEL_NAME}" || -z "${MODEL_PATH}" ]]; then
      usage >&2
      exit 2
    fi
    require_path "WorldScore python" "${WORLDSCORE_PYTHON}"
    require_path "model path" "${MODEL_PATH}"
    "${WORLDFOUNDRY_PYTHON}" "${REPO_ROOT}/worldfoundry/evaluation/tasks/execution/runners/worldscore/run_worldscore_official_runner.py" \
      --worldscore-root "${WORLDSCORE_ROOT}" \
      --data-path "${WORLDSCORE_DATA_PATH}" \
      --model-name "${MODEL_NAME}" \
      --model-path "${MODEL_PATH}" \
      --output-dir "${OUTPUT_DIR}" \
      --python "${WORLDSCORE_PYTHON}" \
      --json
    ;;

  *)
    usage >&2
    exit 2
    ;;
esac
