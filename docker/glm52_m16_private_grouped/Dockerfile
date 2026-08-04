ARG BASE_IMAGE=amdsiloai/vllm-private:vllm_69715823_aiter_4a1cc77_glm52_production_tp4_pr1_pr2_pr3_pr4_pr50008
FROM ${BASE_IMAGE}

ARG AITER_COMMIT=4bec01e6a

LABEL org.opencontainers.image.title="GLM-5.2 TP4 M16 private-grouped MXFP4 A/B"
LABEL org.opencontainers.image.revision="${AITER_COMMIT}"
LABEL com.amd.glm52.m16_private_grouped="true"

ENV AITER_GLM52_M16_PRIVATE_GROUPED=1

# Install private copies so generic production GEMM1/GEMM2 paths remain
# untouched. Only the exact GLM-5.2 TP4 M=16 dispatcher imports these files.
COPY aiter/ops/flydsl/kernels/mxfp4_gemm1.py \
     /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_gemm1_private_grouped.py
COPY aiter/ops/flydsl/kernels/mxfp4_gemm2.py \
     /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_gemm2_private_grouped.py
COPY aiter/ops/flydsl/kernels/mxfp4_routed_compact_shared_moe.py \
     /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_routed_compact_shared_moe.py
COPY aiter/ops/flydsl/mxfp4_routed_compact_shared_moe_kernels.py \
     /tmp/aiter-main/aiter/ops/flydsl/mxfp4_routed_compact_shared_moe_kernels.py
COPY docker/glm52_m16_private_grouped/patch_fused_moe_m16.py \
     /tmp/patch_fused_moe_m16.py

RUN sed -i \
      's/from \.mxfp4_gemm1 import (/from .mxfp4_gemm1_private_grouped import (/' \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_routed_compact_shared_moe.py && \
    sed -i \
      's/from \.mxfp4_gemm2 import (/from .mxfp4_gemm2_private_grouped import (/' \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_routed_compact_shared_moe.py && \
    python /tmp/patch_fused_moe_m16.py && \
    rm /tmp/patch_fused_moe_m16.py && \
    find /tmp/aiter-main/aiter -type f -name '*.pyc' -delete && \
    python -m py_compile \
      /tmp/aiter-main/aiter/fused_moe.py \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_gemm1_private_grouped.py \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_gemm2_private_grouped.py \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_routed_compact_shared_moe.py \
      /tmp/aiter-main/aiter/ops/flydsl/mxfp4_routed_compact_shared_moe_kernels.py && \
    grep -q AITER_GLM52_M16_PRIVATE_GROUPED \
      /tmp/aiter-main/aiter/fused_moe.py && \
    grep -q mxfp4_gemm1_private_grouped \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_routed_compact_shared_moe.py && \
    echo "GLM-5.2 M16 private-grouped AITER overlay installed"
