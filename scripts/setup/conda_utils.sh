#!/usr/bin/env bash

# Shared helpers for running commands inside the WorldFoundry conda environment.
# Source this file from setup, validation, and demo scripts instead of invoking a
# project-local virtualenv.

WORLDFOUNDRY_CONDA_ENV_NAME="${WORLDFOUNDRY_CONDA_ENV_NAME:-worldfoundry}"
WORLDFOUNDRY_CONDA_ENV_PREFIX="${WORLDFOUNDRY_CONDA_ENV_PREFIX:-}"
CONDA_EXE_PATH="${CONDA_EXE:-conda}"

worldfoundry_conda_env_args() {
  if [[ -n "${WORLDFOUNDRY_CONDA_ENV_PREFIX}" ]]; then
    printf '%s\n' "-p" "${WORLDFOUNDRY_CONDA_ENV_PREFIX}"
  else
    printf '%s\n' "-n" "${WORLDFOUNDRY_CONDA_ENV_NAME}"
  fi
}

worldfoundry_conda_selector_text() {
  if [[ -n "${WORLDFOUNDRY_CONDA_ENV_PREFIX}" ]]; then
    printf '%s' "${WORLDFOUNDRY_CONDA_ENV_PREFIX}"
  else
    printf '%s' "${WORLDFOUNDRY_CONDA_ENV_NAME}"
  fi
}

worldfoundry_in_target_conda_env() {
  if [[ -n "${WORLDFOUNDRY_CONDA_ENV_PREFIX}" ]]; then
    [[ "${CONDA_PREFIX:-}" == "${WORLDFOUNDRY_CONDA_ENV_PREFIX}" ]]
  else
    [[ "${CONDA_DEFAULT_ENV:-}" == "${WORLDFOUNDRY_CONDA_ENV_NAME}" ]]
  fi
}

worldfoundry_conda_run() {
  if worldfoundry_in_target_conda_env; then
    "$@"
    return $?
  fi
  local env_args=()
  mapfile -t env_args < <(worldfoundry_conda_env_args)
  "${CONDA_EXE_PATH}" run --no-capture-output "${env_args[@]}" "$@"
}

worldfoundry_conda_pip() {
  worldfoundry_conda_run python -m pip "$@"
}
