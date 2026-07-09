#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDFOUNDRY_ROOT="${WORLDFOUNDRY_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
HUNYUANVIDEO15_REPO_DIR="${HUNYUANVIDEO15_REPO_DIR:-${WORLDFOUNDRY_ROOT}/worldfoundry/base_models/diffusion_model/video/hunyuan_video/official_hunyuan_video_1_5_runtime}"
HUNYUANVIDEO15_CKPT_DIR="${HUNYUANVIDEO15_CKPT_DIR:-${WORLDFOUNDRY_CKPT_DIR:-${HOME}/.cache/worldfoundry/checkpoints}/HunyuanVideo-1.5}"
HUNYUANVIDEO15_OUTPUT_DIR="${WORLDFOUNDRY_MODEL_PARITY_OUTPUT_DIR:-${WORLDFOUNDRY_ROOT}/tmp/model_zoo/parity/hunyuanvideo-1.5}"
HUNYUANVIDEO15_OUTPUT="${HUNYUANVIDEO15_OUTPUT:-${HUNYUANVIDEO15_OUTPUT_DIR}/hunyuanvideo15_t2v_native_seed42_848x480_9f.mp4}"
HUNYUANVIDEO15_PYTHON="${HUNYUANVIDEO15_PYTHON:-${PYTHON:-python}}"
HUNYUANVIDEO15_TORCHRUN="${HUNYUANVIDEO15_TORCHRUN:-$(dirname "$("${HUNYUANVIDEO15_PYTHON}" -c 'import sys; print(sys.executable)' 2>/dev/null || command -v "${HUNYUANVIDEO15_PYTHON}" || printf '%s' "${HUNYUANVIDEO15_PYTHON}")")/torchrun}"
if [[ ! -x "${HUNYUANVIDEO15_TORCHRUN}" ]]; then
  HUNYUANVIDEO15_TORCHRUN="${HUNYUANVIDEO15_TORCHRUN_FALLBACK:-torchrun}"
fi

if [[ ! -f "${HUNYUANVIDEO15_REPO_DIR}/generate.py" ]]; then
  echo "missing in-tree HunyuanVideo-1.5 runtime: ${HUNYUANVIDEO15_REPO_DIR}/generate.py" >&2
  exit 3
fi

if [[ ! -d "${HUNYUANVIDEO15_CKPT_DIR}" ]]; then
  echo "missing HunyuanVideo-1.5 checkpoint directory: ${HUNYUANVIDEO15_CKPT_DIR}" >&2
  echo "run: bash scripts/inference/prepare_model_infer.sh hunyuanvideo-1.5-t2v --download" >&2
  exit 4
fi

mkdir -p "${HUNYUANVIDEO15_OUTPUT_DIR}"
export PYTHONPATH="${HUNYUANVIDEO15_REPO_DIR}:${WORLDFOUNDRY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${HUNYUANVIDEO15_REPO_DIR}"
"${HUNYUANVIDEO15_TORCHRUN}" \
  --standalone \
  --nproc_per_node="${HUNYUANVIDEO15_NPROC_PER_NODE:-8}" \
  generate.py \
  --prompt "${HUNYUANVIDEO15_PROMPT:-A cat walks on a snowy street, cinematic, high quality.}" \
  --resolution "${HUNYUANVIDEO15_RESOLUTION:-480p}" \
  --aspect_ratio "${HUNYUANVIDEO15_ASPECT_RATIO:-16:9}" \
  --num_inference_steps "${HUNYUANVIDEO15_STEPS:-8}" \
  --video_length "${HUNYUANVIDEO15_FRAMES:-9}" \
  --seed "${HUNYUANVIDEO15_SEED:-42}" \
  --rewrite False \
  --cfg_distilled True \
  --enable_step_distill False \
  --sparse_attn False \
  --use_sageattn False \
  --enable_cache False \
  --sr False \
  --save_pre_sr_video False \
  --overlap_group_offloading False \
  --model_path "${HUNYUANVIDEO15_CKPT_DIR}" \
  --output_path "${HUNYUANVIDEO15_OUTPUT}"
