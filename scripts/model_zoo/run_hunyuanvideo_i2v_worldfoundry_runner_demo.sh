#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDFOUNDRY_ROOT="${WORLDFOUNDRY_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ENV_ROOT="${WORLDFOUNDRY_CONDA_ENVS_ROOT:-${WORLDFOUNDRY_CONDA_ENV_ROOT:-${HOME}/.cache/worldfoundry/conda_envs}}"

if [[ -n "${HUNYUANVIDEO_I2V_PYTHON:-}" ]]; then
  PYTHON_BIN="${HUNYUANVIDEO_I2V_PYTHON}"
elif [[ -x "${ENV_ROOT}/HunyuanVideo/bin/python" ]]; then
  PYTHON_BIN="${ENV_ROOT}/HunyuanVideo/bin/python"
elif [[ "${CONDA_PREFIX:-}" == */HunyuanVideo && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="${PYTHON:-python}"
fi

export PYTHONPATH="${WORLDFOUNDRY_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_hunyuanvideo_i2v_worldfoundry_runner_demo.py" "$@"
