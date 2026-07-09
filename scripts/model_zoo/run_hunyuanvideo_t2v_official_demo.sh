#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDFOUNDRY_ROOT="${WORLDFOUNDRY_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
HUNYUANVIDEO_REPO_DIR="${HUNYUANVIDEO_REPO_DIR:-${WORLDFOUNDRY_ROOT}/worldfoundry/base_models/diffusion_model/video/hunyuan_video/official_hunyuan_video_runtime}"
HUNYUANVIDEO_CKPT_DIR="${HUNYUANVIDEO_CKPT_DIR:-${WORLDFOUNDRY_CKPT_DIR:-${HOME}/.cache/worldfoundry/checkpoints}/HunyuanVideo}"
HUNYUANVIDEO_OUTPUT_DIR="${WORLDFOUNDRY_MODEL_PARITY_OUTPUT_DIR:-${WORLDFOUNDRY_ROOT}/tmp/model_zoo/parity/hunyuanvideo}"
HUNYUANVIDEO_OUTPUT="${HUNYUANVIDEO_OUTPUT:-${HUNYUANVIDEO_OUTPUT_DIR}/hunyuanvideo_t2v_official_cat_grass_1280x720_129f.mp4}"
HUNYUANVIDEO_PYTHON="${HUNYUANVIDEO_PYTHON:-${PYTHON:-python}}"
HUNYUANVIDEO_TORCHRUN="${HUNYUANVIDEO_TORCHRUN:-$(dirname "$("${HUNYUANVIDEO_PYTHON}" -c 'import sys; print(sys.executable)' 2>/dev/null || command -v "${HUNYUANVIDEO_PYTHON}" || printf '%s' "${HUNYUANVIDEO_PYTHON}")")/torchrun}"
if [[ ! -x "${HUNYUANVIDEO_TORCHRUN}" ]]; then
  HUNYUANVIDEO_TORCHRUN="${HUNYUANVIDEO_TORCHRUN_FALLBACK:-torchrun}"
fi

if [[ ! -f "${HUNYUANVIDEO_REPO_DIR}/sample_video.py" ]]; then
  echo "missing in-tree HunyuanVideo runtime: ${HUNYUANVIDEO_REPO_DIR}/sample_video.py" >&2
  exit 3
fi

if [[ ! -d "${HUNYUANVIDEO_CKPT_DIR}" ]]; then
  echo "missing HunyuanVideo checkpoint directory: ${HUNYUANVIDEO_CKPT_DIR}" >&2
  echo "run: bash scripts/inference/prepare_model_infer.sh hunyuanvideo-t2v --download" >&2
  exit 4
fi

mkdir -p "${HUNYUANVIDEO_OUTPUT_DIR}"
RUN_DIR="${HUNYUANVIDEO_OUTPUT_DIR}/official_raw"
mkdir -p "${RUN_DIR}"
START_TS="$(date +%s)"

export PYTHONPATH="${HUNYUANVIDEO_REPO_DIR}:${WORLDFOUNDRY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MODEL_BASE="${HUNYUANVIDEO_CKPT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${HUNYUANVIDEO_REPO_DIR}"
"${HUNYUANVIDEO_TORCHRUN}" \
  --nproc_per_node="${HUNYUANVIDEO_NPROC_PER_NODE:-8}" \
  sample_video.py \
  --model-base "${HUNYUANVIDEO_CKPT_DIR}" \
  --dit-weight "${HUNYUANVIDEO_CKPT_DIR}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" \
  --model-resolution "${HUNYUANVIDEO_MODEL_RESOLUTION:-720p}" \
  --video-size "${HUNYUANVIDEO_HEIGHT:-720}" "${HUNYUANVIDEO_WIDTH:-1280}" \
  --video-length "${HUNYUANVIDEO_FRAMES:-129}" \
  --infer-steps "${HUNYUANVIDEO_STEPS:-50}" \
  --prompt "${HUNYUANVIDEO_PROMPT:-A cat walks on the grass, realistic style.}" \
  --text-encoder llm \
  --tokenizer llm \
  --prompt-template dit-llm-encode \
  --prompt-template-video dit-llm-encode-video \
  --flow-reverse \
  --flow-shift "${HUNYUANVIDEO_FLOW_SHIFT:-7.0}" \
  --embedded-cfg-scale "${HUNYUANVIDEO_EMBEDDED_CFG_SCALE:-6.0}" \
  --seed "${HUNYUANVIDEO_SEED:-42}" \
  --ulysses-degree "${HUNYUANVIDEO_ULYSSES_DEGREE:-8}" \
  --ring-degree "${HUNYUANVIDEO_RING_DEGREE:-1}" \
  --save-path "${RUN_DIR}"

PRODUCED="$(
  find "${RUN_DIR}" -type f -name '*.mp4' -newermt "@${START_TS}" -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
)"
if [[ -z "${PRODUCED}" || ! -f "${PRODUCED}" ]]; then
  echo "HunyuanVideo official demo completed but no new mp4 was found in ${RUN_DIR}" >&2
  exit 5
fi

cp -f "${PRODUCED}" "${HUNYUANVIDEO_OUTPUT}"
printf '%s\n' "${HUNYUANVIDEO_OUTPUT}"
