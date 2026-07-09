#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model="${1:-}"
if [[ -z "$model" ]]; then
  cat >&2 <<'EOF'
Usage: bash scripts/inference/test_nav_video_gen.sh <model> [--input image_path] [run_infer options]

Models:
  matrix-game-2, matrix-game-2-universal
  matrix-game-2-gta-drive
  matrix-game-2-templerun
  matrix-game-3
  yume-1p5, yume-1p5-i2v, yume-1p5-t2v, yume-1p5-v2v
  lingbot-world, lingbot-world-base-cam
  lingbot-world-base-act-preview
  lingbot-world-fast
EOF
  exit 2
fi
shift
exec bash "$ROOT/scripts/inference/run_infer.sh" --category navigation-video --model "$model" "$@"
