#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
INFERENCE_TESTING_CONTAINER="${INFERENCE_TESTING_CONTAINER:-nholmber-inference-testing-amd-1}"
EXTRA_ARGS=()
if [[ "${DRYRUN:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--dryrun)
fi

run_config() {
    local config="$1"
    if command -v inference-testing >/dev/null 2>&1; then
        inference-testing "${EXTRA_ARGS[@]}" --platform amd -c "${config}"
        return
    fi

    local relative="${config#"${REPO_ROOT}/"}"
    docker exec "${INFERENCE_TESTING_CONTAINER}" \
        inference-testing "${EXTRA_ARGS[@]}" --platform amd \
        -c "/app/repos/aiter-glm52-flydsl/${relative}"
}

echo "=== GLM-5.2 MXFP4 fused-MoE versus production MXMOE profiling ==="
echo "Workloads: 60k input / 600 output, concurrency 1, 2, 4, 8, 16"

echo
echo "[$(date +%H:%M:%S)] 1/2: production MXMOE baseline"
run_config "${SCRIPT_DIR}/vllm_glm52_mxfp4_mxmoe_profile.yaml"

echo
echo "[$(date +%H:%M:%S)] 2/2: fused M=1-16 kernels"
run_config "${SCRIPT_DIR}/vllm_glm52_mxfp4_fused_moe_profile.yaml"

echo
echo "=== Profiling A/B complete ==="
echo "MXMOE traces:    /tmp/traces_glm52_mxmoe_ab_20260730"
echo "Fused-MoE traces: /tmp/traces_glm52_fused_moe_ab_20260730"
