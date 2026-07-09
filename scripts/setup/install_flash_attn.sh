#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERIFY_SCRIPT="${ROOT}/scripts/setup/verify_flash_attn.py"
source "${ROOT}/scripts/setup/conda_utils.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup/install_flash_attn.sh flash_attn_fa25
  bash scripts/setup/install_flash_attn.sh flash_attn_fa28

Behavior:
  - validates the currently installed flash-attn in the WorldFoundry conda env first
  - reinstalls if the import or GPU kernel check fails
  - builds flash-attn from source inside the active conda env
  - uses WORLDFOUNDRY_CONDA_ENV_PREFIX or WORLDFOUNDRY_CONDA_ENV_NAME to select the env
EOF
}

if (($# == 0)); then
  usage >&2
  exit 1
fi

BUCKET=""
while (($#)); do
  case "$1" in
    flash_attn_fa25|flash_attn_fa28)
      if [[ -n "$BUCKET" ]]; then
        echo "Only one flash-attn bucket may be specified." >&2
        usage >&2
        exit 2
      fi
      BUCKET="$1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$BUCKET" ]]; then
  echo "Missing flash-attn bucket." >&2
  usage >&2
  exit 2
fi

if ! command -v "${CONDA_EXE_PATH}" >/dev/null 2>&1; then
  echo "conda executable not found. Install Miniconda/Anaconda or set CONDA_EXE." >&2
  exit 1
fi

verify_current() {
  worldfoundry_conda_run python "${VERIFY_SCRIPT}"
}

if [[ "${WORLDFOUNDRY_FORCE_FLASH_ATTN_REINSTALL:-0}" != "1" ]] && verify_current; then
  echo "flash-attn is already healthy; skipping reinstall."
  exit 0
fi

install_from_source() {
  echo "Installing flash-attn from source for ${BUCKET}."
  worldfoundry_conda_pip uninstall -y flash-attn >/dev/null 2>&1 || true
  local trusted_host_args=()
  if [[ -n "${WORLDFOUNDRY_PIP_TRUSTED_HOST:-}" ]]; then
    local host
    for host in ${WORLDFOUNDRY_PIP_TRUSTED_HOST}; do
      trusted_host_args+=(--trusted-host "${host}")
    done
  fi
  FLASH_ATTENTION_FORCE_BUILD=TRUE \
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
  MAX_JOBS="${MAX_JOBS:-8}" \
  worldfoundry_conda_pip install \
    "${trusted_host_args[@]}" \
    --no-deps \
    --no-build-isolation \
    --no-binary flash-attn \
    --no-cache-dir \
    --force-reinstall \
    flash-attn
}

install_from_source
verify_current
