#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

if [ -d /usr/local/cuda-12.4/compat ]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
  export LD_LIBRARY_PATH="/usr/local/cuda-12.4/compat:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"
fi

# Set environment variable for CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MODEL_ROOT="${MODEL_ROOT:-${SCRIPT_DIR}/models/matrixgame}"
export DIT_PATH="${DIT_PATH:-${MODEL_ROOT}/dit}"
export TEXTENC_PATH="${TEXTENC_PATH:-${MODEL_ROOT}}"
export VAE_PATH="${VAE_PATH:-${MODEL_ROOT}/vae}"
export MOUSE_ICON_PATH="${MOUSE_ICON_PATH:-${MODEL_ROOT}/assets/mouse.png}"
export IMAGE_PATH="${IMAGE_PATH:-${SCRIPT_DIR}/initial_image}"
export OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/test}"
export INFERENCE_STEPS="${INFERENCE_STEPS:-50}"

for required_path in "${DIT_PATH}" "${TEXTENC_PATH}" "${VAE_PATH}" "${MOUSE_ICON_PATH}" "${IMAGE_PATH}"; do
  if [ ! -e "${required_path}" ]; then
    echo "Missing required path: ${required_path}" >&2
    echo "Set MODEL_ROOT/IMAGE_PATH or place the downloaded Matrix-Game weights and input image there." >&2
    exit 2
  fi
done

# Execute inference script with parameters
python inference_bench.py \
    --dit_path "${DIT_PATH}" \
    --textenc_path "${TEXTENC_PATH}" \
    --vae_path "${VAE_PATH}" \
    --mouse_icon_path "${MOUSE_ICON_PATH}" \
    --image_path "${IMAGE_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --inference_steps "${INFERENCE_STEPS}" \
    --bfloat16
