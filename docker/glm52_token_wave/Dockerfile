ARG BASE_IMAGE=amdsiloai/vllm-private:vllm_69715823_aiter_4a1cc77_glm52_production_tp4_pr1_pr50008
FROM ${BASE_IMAGE}

ARG AITER_COMMIT=ff3b34254

LABEL org.opencontainers.image.title="GLM-5.2 token-wave MXFP4 validation"
LABEL org.opencontainers.image.revision="${AITER_COMMIT}"
LABEL com.amd.glm52.token_wave="true"

ENV AITER_GLM52_TOKEN_WAVE=1
ENV AITER_GLM52_TOKEN_WAVE_MIN_M=5
ENV AITER_GLM52_TOKEN_WAVE_MAX_M=16

COPY overlay/aiter/ops/flydsl/kernels/mxfp4_gemm1.py \
     /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_gemm1.py
COPY overlay/aiter/ops/flydsl/kernels/mxfp4_gemm2.py \
     /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_gemm2.py
COPY overlay/aiter/ops/flydsl/kernels/mxfp4_routed_compact_shared_moe.py \
     /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_routed_compact_shared_moe.py
COPY overlay/aiter/ops/flydsl/kernels/mxfp4_shared_hybrid_moe.py \
     /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_shared_hybrid_moe.py
COPY overlay/aiter/ops/flydsl/kernels/mxfp4_token_wave_shared_moe.py \
     /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_token_wave_shared_moe.py
COPY overlay/aiter/ops/flydsl/mxfp4_token_wave_shared_moe_kernels.py \
     /tmp/aiter-main/aiter/ops/flydsl/mxfp4_token_wave_shared_moe_kernels.py

COPY patch_fused_moe_dispatch.py \
     /tmp/patch_fused_moe_dispatch.py

RUN python /tmp/patch_fused_moe_dispatch.py && \
    rm /tmp/patch_fused_moe_dispatch.py && \
    SITE=/usr/local/lib/python3.12/dist-packages/aiter && \
    cp /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_gemm1.py \
       ${SITE}/ops/flydsl/kernels/mxfp4_gemm1.py && \
    cp /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_gemm2.py \
       ${SITE}/ops/flydsl/kernels/mxfp4_gemm2.py && \
    cp /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_routed_compact_shared_moe.py \
       ${SITE}/ops/flydsl/kernels/mxfp4_routed_compact_shared_moe.py && \
    cp /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_shared_hybrid_moe.py \
       ${SITE}/ops/flydsl/kernels/mxfp4_shared_hybrid_moe.py && \
    cp /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_token_wave_shared_moe.py \
       ${SITE}/ops/flydsl/kernels/mxfp4_token_wave_shared_moe.py && \
    cp /tmp/aiter-main/aiter/ops/flydsl/mxfp4_token_wave_shared_moe_kernels.py \
       ${SITE}/ops/flydsl/mxfp4_token_wave_shared_moe_kernels.py && \
    cp /tmp/aiter-main/aiter/fused_moe.py ${SITE}/fused_moe.py && \
    python -m py_compile \
      /tmp/aiter-main/aiter/fused_moe.py \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_gemm1.py \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_gemm2.py \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_routed_compact_shared_moe.py \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_shared_hybrid_moe.py \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_token_wave_shared_moe.py \
      /tmp/aiter-main/aiter/ops/flydsl/mxfp4_token_wave_shared_moe_kernels.py && \
    grep -q AITER_GLM52_TOKEN_WAVE /tmp/aiter-main/aiter/fused_moe.py && \
    echo "GLM-5.2 token-wave AITER overlay installed"
