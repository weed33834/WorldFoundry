#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDFOUNDRY_ROOT="${WORLDFOUNDRY_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WAN22_REPO_DIR="${WAN22_REPO_DIR:-${WORLDFOUNDRY_ROOT}/tmp/model_zoo/repos/github.com_Wan-Video_Wan2.2}"
WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-${WORLDFOUNDRY_ROOT}/cache/hfd/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e}"
WAN22_OUTPUT_DIR="${WORLDFOUNDRY_MODEL_PARITY_OUTPUT_DIR:-${WORLDFOUNDRY_ROOT}/tmp/model_zoo/parity/wan2.2}"
WAN22_OUTPUT="${WAN22_OUTPUT:-${WAN22_OUTPUT_DIR}/wan22_ti2v_5b_seed42_1280x704_121f.mp4}"
WAN22_PYTHON="${WAN22_PYTHON:-${WORLDFOUNDRY_ROOT}/.venv/bin/python}"

if [[ ! -f "${WAN22_REPO_DIR}/generate.py" ]]; then
  echo "missing official Wan2.2 repo: ${WAN22_REPO_DIR}" >&2
  exit 3
fi

if [[ ! -d "${WAN22_CKPT_DIR}" ]]; then
  echo "missing Wan2.2 TI2V-5B checkpoint snapshot: ${WAN22_CKPT_DIR}" >&2
  echo "run: python scripts/model_zoo/download_checkpoints.py --model-id wan2.2 --repo-id Wan-AI/Wan2.2-TI2V-5B --execute" >&2
  exit 4
fi

mkdir -p "${WAN22_OUTPUT_DIR}"

cd "${WAN22_REPO_DIR}"
"${WAN22_PYTHON}" generate.py \
  --task ti2v-5B \
  --size '1280*704' \
  --ckpt_dir "${WAN22_CKPT_DIR}" \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --base_seed 42 \
  --frame_num 121 \
  --sample_steps 50 \
  --sample_shift 5 \
  --sample_guide_scale 5 \
  --save_file "${WAN22_OUTPUT}" \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
