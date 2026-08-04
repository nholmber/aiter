#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
BASE_IMAGE="${BASE_IMAGE:-amdsiloai/vllm-private:vllm_69715823_aiter_4a1cc77_glm52_production_tp4_pr1_pr2_pr3_pr4_pr50008}"
IMAGE="${IMAGE:-amdsiloai/vllm-private:vllm_69715823_aiter_4bec01e6a_glm52_m16_private_grouped_pr1_pr2_pr3_pr4_pr50008}"

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "AITER_COMMIT=4bec01e6a" \
  -f "${ROOT}/docker/glm52_m16_private_grouped/Dockerfile" \
  -t "${IMAGE}" \
  "${ROOT}"

echo "${IMAGE}"
