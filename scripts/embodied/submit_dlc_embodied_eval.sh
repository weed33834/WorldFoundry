#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: submit_dlc_embodied_eval.sh CONFIG [OUTPUT_DIR]

Set WF_DLC_SUBMIT=1 to submit. Without it, the script prints the DLC command.
Override WF_DLC_WORKER_IMAGE to use a PAI-accessible image path.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${WORLDFOUNDRY_REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CONFIG=$1
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${REPO_ROOT}/${CONFIG}"
fi
OUTPUT_DIR=${2:-${REPO_ROOT}/tmp/embodied_dlc/$(basename "${CONFIG}" .yaml)-$(date -u +%Y%m%dT%H%M%SZ)}
DLC_BIN=${DLC_BIN:-/etc/dsw/runtime/export_bin/dlc}

readarray -t CONFIG_VALUES < <(
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 - "${CONFIG}" <<'PY'
import sys
from worldfoundry.evaluation.tasks.embodied.config_loader import load_canonical_embodied_config

config = load_canonical_embodied_config(sys.argv[1])
docker = config.get("docker") or {}
print(docker.get("image") or "")
print(docker.get("source_image") or "")
print(docker.get("python_env") or "")
print(config.get("id") or "embodied_eval")
PY
)

CONFIG_IMAGE=${CONFIG_VALUES[0]}
CONFIG_SOURCE_IMAGE=${CONFIG_VALUES[1]}
CONFIG_CONDA_ENV=${CONFIG_VALUES[2]}
CONFIG_ID=${CONFIG_VALUES[3]}
if [[ "${WF_DLC_USE_SOURCE_IMAGE:-0}" == "1" ]]; then
  DEFAULT_WORKER_IMAGE=${CONFIG_SOURCE_IMAGE:-${CONFIG_IMAGE}}
else
  DEFAULT_WORKER_IMAGE=${CONFIG_IMAGE}
fi
WORKER_IMAGE=${WF_DLC_WORKER_IMAGE:-${DEFAULT_WORKER_IMAGE}}
CONDA_ENV=${WF_EMBODIED_CONDA_ENV-${CONFIG_CONDA_ENV}}
JOB_NAME=${WF_DLC_JOB_NAME:-sft_innovator_science_data}

if [[ -z "${WORKER_IMAGE}" ]]; then
  echo "No worker image found. Set WF_DLC_WORKER_IMAGE or add docker.image to the config." >&2
  exit 2
fi

INNER_COMMAND=$(cat <<EOF
set -euo pipefail
cd "${REPO_ROOT}"
export WORLDFOUNDRY_REPO_ROOT="${REPO_ROOT}"
export WF_EMBODIED_CONDA_ENV="${CONDA_ENV}"
export WF_EMBODIED_SERVER_URL="${WF_EMBODIED_SERVER_URL:-}"
export WF_EMBODIED_SERVE_CONFIG="${WF_EMBODIED_SERVE_CONFIG:-}"
export WF_EMBODIED_SERVE_PORT="${WF_EMBODIED_SERVE_PORT:-8000}"
export WF_EMBODIED_PLAN_ONLY="${WF_EMBODIED_PLAN_ONLY:-0}"
export WF_EMBODIED_NO_SAVE="${WF_EMBODIED_NO_SAVE:-0}"
export WF_EMBODIED_BOOTSTRAP="${WF_EMBODIED_BOOTSTRAP:-0}"
export WF_EMBODIED_BOOTSTRAP_PACKAGES="${WF_EMBODIED_BOOTSTRAP_PACKAGES:-pyyaml msgpack packaging tqdm websockets}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
bash scripts/embodied/run_dlc_embodied_eval.sh "${CONFIG}" "${OUTPUT_DIR}"
EOF
)
COMMAND=$(cat <<EOF
bash <<'WF_DLC_COMMAND'
${INNER_COMMAND}
WF_DLC_COMMAND
EOF
)

CMD=(
  "${DLC_BIN}" submit pytorchjob
  --name="${JOB_NAME}"
  --command="${COMMAND}"
  --resource_id="${WF_DLC_RESOURCE_ID:-quotaev2tl4w6aw0}"
  --workspace_id="${WF_DLC_WORKSPACE_ID:-240810}"
  --vpc_id="${WF_DLC_VPC_ID:-vpc-0jl5rpw5qokp6p2ettip6}"
  --switch_id="${WF_DLC_SWITCH_ID:-vsw-0jlmr9rjzed093yr9c0kz}"
  --security_group_id="${WF_DLC_SECURITY_GROUP_ID:-sg-0jl0pd5qaerdj75wmred}"
  --extended_cidrs="${WF_DLC_EXTENDED_CIDRS:-10.1.255.0/29,10.1.255.8/29,10.1.16.0/20}"
  --priority="${WF_DLC_PRIORITY:-1}"
  --workers="${WF_DLC_WORKERS:-1}"
  --worker_image="${WORKER_IMAGE}"
  --worker_cpu="${WF_DLC_WORKER_CPU:-116}"
  --worker_memory="${WF_DLC_WORKER_MEMORY:-1800Gi}"
  --worker_shared_memory="${WF_DLC_WORKER_SHARED_MEMORY:-1800Gi}"
  --worker_gpu="${WF_DLC_WORKER_GPU:-8}"
  --job_max_running_time_minutes="${WF_DLC_MAX_RUNNING_MINUTES:-1440}"
)

if [[ -n "${WF_DLC_DATA_SOURCE_URIS:-}" ]]; then
  CMD+=(--data_source_uris="${WF_DLC_DATA_SOURCE_URIS}")
fi

if [[ -n "${WF_DLC_IMAGE_REPO_USERNAME:-}" || -n "${WF_DLC_IMAGE_REPO_PASSWORD:-}" ]]; then
  if [[ -z "${WF_DLC_IMAGE_REPO_USERNAME:-}" || -z "${WF_DLC_IMAGE_REPO_PASSWORD:-}" ]]; then
    echo "Set both WF_DLC_IMAGE_REPO_USERNAME and WF_DLC_IMAGE_REPO_PASSWORD for private image registries." >&2
    exit 2
  fi
  CMD+=(--image_repo_username="${WF_DLC_IMAGE_REPO_USERNAME}" --image_repo_password="${WF_DLC_IMAGE_REPO_PASSWORD}")
fi

PRINT_CMD=("${CMD[@]}")
for i in "${!PRINT_CMD[@]}"; do
  if [[ "${PRINT_CMD[$i]}" == --image_repo_password=* ]]; then
    PRINT_CMD[$i]="--image_repo_password=REDACTED"
  fi
done

printf 'DLC worker image: %s\n' "${WORKER_IMAGE}"
printf 'DLC conda env: %s\n' "${CONDA_ENV}"
printf 'DLC output dir: %s\n' "${OUTPUT_DIR}"
printf '+'
printf ' %q' "${PRINT_CMD[@]}"
printf '\n'

if [[ "${WF_DLC_SUBMIT:-0}" == "1" ]]; then
  exec "${CMD[@]}"
fi
