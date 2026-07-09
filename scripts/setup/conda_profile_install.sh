#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTORCH_BUNDLE="clip"
TRANSFORMERS_PROFILE="high"
INSTALL_THREE_D_CORE=0
FLASH_ATTN_BUCKET=""
CONDA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/setup/conda_profile_install.sh --group GROUP [--group GROUP ...] [conda_install options]

Install selected dependency profile groups through the unified conda installer.
Accepted groups:
  runtime_clip_bundle, runtime_timm_bundle
  transformers_high, transformers_low
  three_d_core
  flash_attn_fa25, flash_attn_fa28

Additional arguments are forwarded to scripts/setup/conda_install.sh.
EOF
}

while (($#)); do
  case "$1" in
    --group)
      if (($# < 2)); then
        echo "Missing value for --group" >&2
        exit 1
      fi
      group="$2"
      case "$group" in
        runtime_clip_bundle)
          PYTORCH_BUNDLE="clip"
          ;;
        runtime_timm_bundle)
          PYTORCH_BUNDLE="timm"
          ;;
        transformers_high)
          TRANSFORMERS_PROFILE="high"
          ;;
        transformers_low)
          TRANSFORMERS_PROFILE="low"
          ;;
        three_d_core)
          INSTALL_THREE_D_CORE=1
          ;;
        flash_attn_fa25|flash_attn_fa28)
          if [[ -n "$FLASH_ATTN_BUCKET" && "$FLASH_ATTN_BUCKET" != "$group" ]]; then
            echo "Conflicting flash-attn groups: ${FLASH_ATTN_BUCKET} vs ${group}" >&2
            exit 2
          fi
          FLASH_ATTN_BUCKET="$group"
          ;;
        *)
          echo "Unknown conda profile group: ${group}" >&2
          usage >&2
          exit 2
          ;;
      esac
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      CONDA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$INSTALL_THREE_D_CORE" == "1" ]]; then
  CONDA_ARGS+=(--three-d-core)
else
  CONDA_ARGS+=(--skip-three-d-core)
fi

if [[ -n "$FLASH_ATTN_BUCKET" ]]; then
  CONDA_ARGS+=(--flash-attn "$FLASH_ATTN_BUCKET")
else
  CONDA_ARGS+=(--skip-flash-attn)
fi

bash "$ROOT/scripts/setup/conda_install.sh" \
  --preset slim \
  --pytorch-bundle "$PYTORCH_BUNDLE" \
  --transformers "$TRANSFORMERS_PROFILE" \
  "${CONDA_ARGS[@]}"
