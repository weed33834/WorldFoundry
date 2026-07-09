#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDFOUNDRY_ROOT="${WORLDFOUNDRY_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MOCHI1_REPO_DIR="${MOCHI1_REPO_DIR:-${WORLDFOUNDRY_ROOT}/tmp/model_zoo/repos/github.com_genmoai_models}"
MOCHI1_CKPT_DIR="${MOCHI1_CKPT_DIR:-${WORLDFOUNDRY_ROOT}/cache/hfd/models--genmo--mochi-1-preview/snapshots/14be5fcea23095ed330cb214647916a451e38b6e}"
MOCHI1_OUTPUT_DIR="${WORLDFOUNDRY_MODEL_PARITY_OUTPUT_DIR:-${WORLDFOUNDRY_ROOT}/tmp/model_zoo/parity/mochi-1}"
MOCHI1_OUTPUT_NAME="${MOCHI1_OUTPUT_NAME:-mochi1_preview_seed12345_848x480_31f.mp4}"
MOCHI1_OUTPUT="${MOCHI1_OUTPUT:-${MOCHI1_OUTPUT_DIR}/${MOCHI1_OUTPUT_NAME}}"
MOCHI1_PROMPT="${MOCHI1_PROMPT:-A bright yellow lemon rolls across a wooden table under soft studio lighting.}"
MOCHI1_WIDTH="${MOCHI1_WIDTH:-848}"
MOCHI1_HEIGHT="${MOCHI1_HEIGHT:-480}"
MOCHI1_NUM_FRAMES="${MOCHI1_NUM_FRAMES:-31}"
MOCHI1_SEED="${MOCHI1_SEED:-12345}"
MOCHI1_CFG_SCALE="${MOCHI1_CFG_SCALE:-6.0}"
MOCHI1_NUM_STEPS="${MOCHI1_NUM_STEPS:-64}"
MOCHI1_THRESHOLD_NOISE="${MOCHI1_THRESHOLD_NOISE:-0.025}"
MOCHI1_CPU_OFFLOAD="${MOCHI1_CPU_OFFLOAD:-1}"
MOCHI1_CUDA_VISIBLE_DEVICES="${MOCHI1_CUDA_VISIBLE_DEVICES:-}"

if [[ -n "${MOCHI1_CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${MOCHI1_CUDA_VISIBLE_DEVICES}"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "${MOCHI1_CPU_OFFLOAD}" != "0" ]]; then
  export CUDA_VISIBLE_DEVICES="0"
fi

if [[ -x "${WORLDFOUNDRY_ROOT}/tmp/envs/mochi1/bin/python" ]]; then
  MOCHI1_PYTHON_DEFAULT="${WORLDFOUNDRY_ROOT}/tmp/envs/mochi1/bin/python"
elif [[ -x "${WORLDFOUNDRY_ROOT}/.venv/bin/python" ]]; then
  MOCHI1_PYTHON_DEFAULT="${WORLDFOUNDRY_ROOT}/.venv/bin/python"
else
  MOCHI1_PYTHON_DEFAULT="python3"
fi
MOCHI1_PYTHON="${MOCHI1_PYTHON:-${MOCHI1_PYTHON_DEFAULT}}"

if [[ ! -f "${MOCHI1_REPO_DIR}/demos/cli.py" ]]; then
  echo "missing official Mochi-1 repo: ${MOCHI1_REPO_DIR}" >&2
  exit 3
fi

for required_file in dit.safetensors decoder.safetensors encoder.safetensors; do
  if [[ ! -f "${MOCHI1_CKPT_DIR}/${required_file}" ]]; then
    echo "missing Mochi-1 preview checkpoint file: ${MOCHI1_CKPT_DIR}/${required_file}" >&2
    echo "run: python scripts/model_zoo/download_checkpoints.py --model-id mochi-1 --repo-id genmo/mochi-1-preview --execute" >&2
    exit 4
  fi
done

mkdir -p "${MOCHI1_OUTPUT_DIR}"
MOCHI1_RAW_OUTPUT_DIR="$(mktemp -d "${MOCHI1_OUTPUT_DIR}/official_raw.XXXXXX")"

cd "${MOCHI1_REPO_DIR}"
MOCHI1_ARGS=(
  "demos/cli.py"
  "--model_dir" "${MOCHI1_CKPT_DIR}"
  "--out_dir" "${MOCHI1_RAW_OUTPUT_DIR}"
  "--prompt" "${MOCHI1_PROMPT}"
  "--width" "${MOCHI1_WIDTH}"
  "--height" "${MOCHI1_HEIGHT}"
  "--num_frames" "${MOCHI1_NUM_FRAMES}"
  "--seed" "${MOCHI1_SEED}"
  "--cfg_scale" "${MOCHI1_CFG_SCALE}"
  "--num_steps" "${MOCHI1_NUM_STEPS}"
  "--threshold-noise" "${MOCHI1_THRESHOLD_NOISE}"
)
if [[ "${MOCHI1_CPU_OFFLOAD}" != "0" ]]; then
  MOCHI1_ARGS+=("--cpu_offload")
fi

PYTHONPATH="${MOCHI1_REPO_DIR}/src:${PYTHONPATH:-}" "${MOCHI1_PYTHON}" "${MOCHI1_ARGS[@]}"

MOCHI1_GENERATED="$(find "${MOCHI1_RAW_OUTPUT_DIR}" -maxdepth 1 -type f -name 'output_*.mp4' -print | sort | tail -n 1)"
if [[ -z "${MOCHI1_GENERATED}" ]]; then
  echo "official Mochi-1 CLI completed without writing output_*.mp4 in ${MOCHI1_RAW_OUTPUT_DIR}" >&2
  exit 5
fi

cp "${MOCHI1_GENERATED}" "${MOCHI1_OUTPUT}"
if [[ -f "${MOCHI1_GENERATED%.mp4}.json" ]]; then
  cp "${MOCHI1_GENERATED%.mp4}.json" "${MOCHI1_OUTPUT%.mp4}.json"
fi

echo "Mochi-1 preview official demo artifact: ${MOCHI1_OUTPUT}"
