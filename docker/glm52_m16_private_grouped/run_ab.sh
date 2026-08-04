#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
CT="${CT:-nholmber-inference-testing-amd-1}"
MODE="${1:-perf}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${ROOT}/docker/glm52_m16_private_grouped/logs/${STAMP}}"
CONTAINER_ROOT="/app/repos/aiter-glm52-flydsl/docker/glm52_m16_private_grouped"

case "${MODE}" in
  perf)
    CONFIGS=(
      "configs/vllm_glm52_tp4_m16_baseline.yaml"
      "configs/vllm_glm52_tp4_m16_candidate.yaml"
    )
    ;;
  profile)
    CONFIGS=(
      "configs/vllm_glm52_tp4_m16_baseline_profile.yaml"
      "configs/vllm_glm52_tp4_m16_candidate_profile.yaml"
    )
    ;;
  all)
    CONFIGS=(
      "configs/vllm_glm52_tp4_m16_baseline.yaml"
      "configs/vllm_glm52_tp4_m16_candidate.yaml"
      "configs/vllm_glm52_tp4_m16_baseline_profile.yaml"
      "configs/vllm_glm52_tp4_m16_candidate_profile.yaml"
    )
    ;;
  *)
    echo "usage: $0 [perf|profile|all]" >&2
    exit 2
    ;;
esac

mkdir -p "${LOG_DIR}"

for relative_config in "${CONFIGS[@]}"; do
  name="$(basename "${relative_config}" .yaml)"
  log="${LOG_DIR}/${name}.log"
  container_config="${CONTAINER_ROOT}/${relative_config}"
  echo "[$(date +%H:%M:%S)] ${name}"
  docker exec -i "${CT}" \
    inference-testing -c "${container_config}" \
    2>&1 | tee "${log}"
done

echo "logs: ${LOG_DIR}"
