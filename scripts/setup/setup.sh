#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BOOTSTRAP="${ROOT}/scripts/setup/bootstrap_worldfoundry.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup/setup.sh [options]

Set up WorldFoundry for local inference, Studio, and benchmark execution.
This is the public setup entrypoint and forwards the same options accepted by
scripts/setup/bootstrap_worldfoundry.sh.

Common examples:
  bash scripts/setup/setup.sh
  bash scripts/setup/setup.sh --cuda cu128 --data-root /path/to/data --model-root /path/to/models
  bash scripts/setup/setup.sh --with-model lingbot-world --prepare-model lingbot-world
  bash scripts/setup/link_hf_checkpoints.sh --ckpt-dir /path/to/checkpoints --default-world

Run the full option reference:
  bash scripts/setup/bootstrap_worldfoundry.sh --help
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  echo
  exec bash "$BOOTSTRAP" --help
fi

exec bash "$BOOTSTRAP" "$@"
