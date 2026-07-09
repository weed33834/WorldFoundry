#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDFOUNDRY_ROOT="${WORLDFOUNDRY_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
HUNYUANVIDEO_I2V_REPO_DIR="${HUNYUANVIDEO_I2V_REPO_DIR:-${WORLDFOUNDRY_ROOT}/worldfoundry/base_models/diffusion_model/video/hunyuan_video/official_hunyuan_video_i2v_runtime}"
HUNYUANVIDEO_I2V_CKPT_DIR="${HUNYUANVIDEO_I2V_CKPT_DIR:-${WORLDFOUNDRY_CKPT_DIR:-${HOME}/.cache/worldfoundry/checkpoints}/HunyuanVideo-I2V}"
HUNYUANVIDEO_I2V_IMAGE="${HUNYUANVIDEO_I2V_IMAGE:-${WORLDFOUNDRY_ROOT}/worldfoundry/data/test_cases/hunyuanvideo_i2v/0.jpg}"
HUNYUANVIDEO_I2V_OUTPUT_DIR="${WORLDFOUNDRY_MODEL_PARITY_OUTPUT_DIR:-${WORLDFOUNDRY_ROOT}/tmp/model_zoo/parity/hunyuanvideo-i2v}"
HUNYUANVIDEO_I2V_OUTPUT="${HUNYUANVIDEO_I2V_OUTPUT:-${HUNYUANVIDEO_I2V_OUTPUT_DIR}/hunyuanvideo_i2v_official_firework_864x1024_129f.mp4}"
HUNYUANVIDEO_I2V_PYTHON="${HUNYUANVIDEO_I2V_PYTHON:-${PYTHON:-python}}"
HUNYUANVIDEO_I2V_TORCHRUN="${HUNYUANVIDEO_I2V_TORCHRUN:-$(dirname "$("${HUNYUANVIDEO_I2V_PYTHON}" -c 'import sys; print(sys.executable)' 2>/dev/null || command -v "${HUNYUANVIDEO_I2V_PYTHON}" || printf '%s' "${HUNYUANVIDEO_I2V_PYTHON}")")/torchrun}"
if [[ ! -x "${HUNYUANVIDEO_I2V_TORCHRUN}" ]]; then
  HUNYUANVIDEO_I2V_TORCHRUN="${HUNYUANVIDEO_I2V_TORCHRUN_FALLBACK:-torchrun}"
fi

if [[ ! -f "${HUNYUANVIDEO_I2V_REPO_DIR}/sample_image2video.py" ]]; then
  echo "missing in-tree HunyuanVideo I2V runtime: ${HUNYUANVIDEO_I2V_REPO_DIR}/sample_image2video.py" >&2
  exit 3
fi

if [[ ! -d "${HUNYUANVIDEO_I2V_CKPT_DIR}" ]]; then
  echo "missing HunyuanVideo-I2V checkpoint directory: ${HUNYUANVIDEO_I2V_CKPT_DIR}" >&2
  echo "run: bash scripts/inference/prepare_model_infer.sh hunyuanvideo-i2v --download" >&2
  exit 4
fi

if [[ ! -f "${HUNYUANVIDEO_I2V_IMAGE}" ]]; then
  echo "missing HunyuanVideo-I2V demo image: ${HUNYUANVIDEO_I2V_IMAGE}" >&2
  exit 5
fi

mkdir -p "${HUNYUANVIDEO_I2V_OUTPUT_DIR}"
RUN_DIR="${HUNYUANVIDEO_I2V_OUTPUT_DIR}/official_raw"
mkdir -p "${RUN_DIR}"
START_TS="$(date +%s)"

export PYTHONPATH="${HUNYUANVIDEO_I2V_REPO_DIR}:${WORLDFOUNDRY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MODEL_BASE="${HUNYUANVIDEO_I2V_CKPT_DIR}"
export ALLOW_RESIZE_FOR_SP="${ALLOW_RESIZE_FOR_SP:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${HUNYUANVIDEO_I2V_REPO_DIR}"
"${HUNYUANVIDEO_I2V_TORCHRUN}" \
  --nproc_per_node="${HUNYUANVIDEO_I2V_NPROC_PER_NODE:-8}" \
  sample_image2video.py \
  --model-base "${HUNYUANVIDEO_I2V_CKPT_DIR}" \
  --i2v-dit-weight "${HUNYUANVIDEO_I2V_CKPT_DIR}/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt" \
  --model "${HUNYUANVIDEO_I2V_MODEL:-HYVideo-T/2}" \
  --prompt "${HUNYUANVIDEO_I2V_PROMPT:-An Asian man with short hair in black tactical uniform and white clothes waves a firework stick.}" \
  --i2v-mode \
  --i2v-image-path "${HUNYUANVIDEO_I2V_IMAGE}" \
  --i2v-resolution "${HUNYUANVIDEO_I2V_RESOLUTION:-720p}" \
  --i2v-stability \
  --infer-steps "${HUNYUANVIDEO_I2V_STEPS:-50}" \
  --video-length "${HUNYUANVIDEO_I2V_FRAMES:-129}" \
  --flow-reverse \
  --flow-shift "${HUNYUANVIDEO_I2V_FLOW_SHIFT:-7.0}" \
  --embedded-cfg-scale "${HUNYUANVIDEO_I2V_EMBEDDED_CFG_SCALE:-6.0}" \
  --seed "${HUNYUANVIDEO_I2V_SEED:-0}" \
  --ulysses-degree "${HUNYUANVIDEO_I2V_ULYSSES_DEGREE:-8}" \
  --ring-degree "${HUNYUANVIDEO_I2V_RING_DEGREE:-1}" \
  --save-path "${RUN_DIR}"

PRODUCED="$(
  find "${RUN_DIR}" -type f -name '*.mp4' -newermt "@${START_TS}" -printf '%T@ %p\n' \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
)"
if [[ -z "${PRODUCED}" || ! -f "${PRODUCED}" ]]; then
  echo "HunyuanVideo I2V official demo completed but no new mp4 was found in ${RUN_DIR}" >&2
  exit 6
fi

cp -f "${PRODUCED}" "${HUNYUANVIDEO_I2V_OUTPUT}"
printf '%s\n' "${HUNYUANVIDEO_I2V_OUTPUT}"
