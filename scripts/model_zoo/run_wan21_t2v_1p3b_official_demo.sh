#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDFOUNDRY_ROOT="${WORLDFOUNDRY_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WAN21_REPO_DIR="${WAN21_REPO_DIR:-${WORLDFOUNDRY_ROOT}/tmp/model_zoo/repos/github.com_Wan-Video_Wan2.1}"
WAN21_CKPT_DIR="${WAN21_CKPT_DIR:-${WORLDFOUNDRY_ROOT}/cache/hfd/models--Wan-AI--Wan2.1-T2V-1.3B/snapshots/37ec512624d61f7aa208f7ea8140a131f93afc9a}"
WAN21_OUTPUT_DIR="${WORLDFOUNDRY_MODEL_PARITY_OUTPUT_DIR:-${WORLDFOUNDRY_ROOT}/tmp/model_zoo/parity/wan2.1}"
WAN21_OUTPUT="${WAN21_OUTPUT:-${WAN21_OUTPUT_DIR}/wan21_t2v_1p3b_seed42_832x480_81f.mp4}"
if [[ -x "${WORLDFOUNDRY_ROOT}/tmp/envs/wan21-cu118/bin/python" ]]; then
  WAN21_PYTHON_DEFAULT="${WORLDFOUNDRY_ROOT}/tmp/envs/wan21-cu118/bin/python"
else
  WAN21_PYTHON_DEFAULT="${WORLDFOUNDRY_ROOT}/.venv/bin/python"
fi
WAN21_PYTHON="${WAN21_PYTHON:-${WAN21_PYTHON_DEFAULT}}"

if [[ ! -f "${WAN21_REPO_DIR}/generate.py" ]]; then
  echo "missing official Wan2.1 repo: ${WAN21_REPO_DIR}" >&2
  exit 3
fi

if [[ ! -d "${WAN21_CKPT_DIR}" ]]; then
  echo "missing Wan2.1 T2V-1.3B checkpoint snapshot: ${WAN21_CKPT_DIR}" >&2
  echo "run: python scripts/model_zoo/download_checkpoints.py --model-id wan2.1 --repo-id Wan-AI/Wan2.1-T2V-1.3B --execute" >&2
  exit 4
fi

mkdir -p "${WAN21_OUTPUT_DIR}"

cd "${WAN21_REPO_DIR}"
"${WAN21_PYTHON}" generate.py \
  --task t2v-1.3B \
  --size '832*480' \
  --ckpt_dir "${WAN21_CKPT_DIR}" \
  --offload_model True \
  --t5_cpu \
  --sample_shift 8 \
  --sample_guide_scale 6 \
  --base_seed 42 \
  --frame_num 81 \
  --save_file "${WAN21_OUTPUT}" \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
