ARG BASE_IMAGE=amdsiloai/vllm-private:vllm_69715823_aiter_4a1cc77_glm52_production_tp4_pr1_pr50008
FROM ${BASE_IMAGE}

ARG AITER_COMMIT=47208fb1a

LABEL org.opencontainers.image.title="GLM-5.2 fused low-M MXFP4 validation"
LABEL org.opencontainers.image.revision="${AITER_COMMIT}"
LABEL com.amd.glm52.fused_moe="true"

ENV AITER_GLM52_FUSED_MOE=1
ENV AITER_GLM52_FUSED_MOE_MIN_M=1
ENV AITER_GLM52_FUSED_MOE_MAX_M=4

COPY overlay/aiter/ops/flydsl/kernels/mxfp4_flat_moe.py \
     /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_flat_moe.py
COPY overlay/aiter/ops/flydsl/kernels/mxfp4_flat_single_stage_moe.py \
     /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_flat_single_stage_moe.py
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
COPY overlay/aiter/ops/flydsl/mxfp4_flat_moe_kernels.py \
     /tmp/aiter-main/aiter/ops/flydsl/mxfp4_flat_moe_kernels.py
COPY overlay/aiter/ops/flydsl/mxfp4_flat_single_stage_moe_kernels.py \
     /tmp/aiter-main/aiter/ops/flydsl/mxfp4_flat_single_stage_moe_kernels.py
COPY overlay/aiter/ops/flydsl/mxfp4_shared_hybrid_moe_kernels.py \
     /tmp/aiter-main/aiter/ops/flydsl/mxfp4_shared_hybrid_moe_kernels.py
COPY overlay/aiter/ops/flydsl/mxfp4_token_wave_shared_moe_kernels.py \
     /tmp/aiter-main/aiter/ops/flydsl/mxfp4_token_wave_shared_moe_kernels.py

COPY patch_fused_moe_dispatch.py \
     /tmp/patch_fused_moe_dispatch.py

RUN python /tmp/patch_fused_moe_dispatch.py && \
    rm /tmp/patch_fused_moe_dispatch.py && \
    SITE=/usr/local/lib/python3.12/dist-packages/aiter && \
    cp /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_flat_moe.py \
       ${SITE}/ops/flydsl/kernels/mxfp4_flat_moe.py && \
    cp /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_flat_single_stage_moe.py \
       ${SITE}/ops/flydsl/kernels/mxfp4_flat_single_stage_moe.py && \
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
    cp /tmp/aiter-main/aiter/ops/flydsl/mxfp4_flat_moe_kernels.py \
       ${SITE}/ops/flydsl/mxfp4_flat_moe_kernels.py && \
    cp /tmp/aiter-main/aiter/ops/flydsl/mxfp4_flat_single_stage_moe_kernels.py \
       ${SITE}/ops/flydsl/mxfp4_flat_single_stage_moe_kernels.py && \
    cp /tmp/aiter-main/aiter/ops/flydsl/mxfp4_shared_hybrid_moe_kernels.py \
       ${SITE}/ops/flydsl/mxfp4_shared_hybrid_moe_kernels.py && \
    cp /tmp/aiter-main/aiter/ops/flydsl/mxfp4_token_wave_shared_moe_kernels.py \
       ${SITE}/ops/flydsl/mxfp4_token_wave_shared_moe_kernels.py && \
    cp /tmp/aiter-main/aiter/fused_moe.py ${SITE}/fused_moe.py && \
    python -m py_compile \
      /tmp/aiter-main/aiter/fused_moe.py \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_flat_moe.py \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_flat_single_stage_moe.py \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_gemm1.py \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_gemm2.py \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_routed_compact_shared_moe.py \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_shared_hybrid_moe.py \
      /tmp/aiter-main/aiter/ops/flydsl/kernels/mxfp4_token_wave_shared_moe.py \
      /tmp/aiter-main/aiter/ops/flydsl/mxfp4_flat_moe_kernels.py \
      /tmp/aiter-main/aiter/ops/flydsl/mxfp4_flat_single_stage_moe_kernels.py \
      /tmp/aiter-main/aiter/ops/flydsl/mxfp4_shared_hybrid_moe_kernels.py \
      /tmp/aiter-main/aiter/ops/flydsl/mxfp4_token_wave_shared_moe_kernels.py && \
    grep -q AITER_GLM52_FUSED_MOE /tmp/aiter-main/aiter/fused_moe.py && \
    echo "GLM-5.2 fused low-M AITER overlay installed"
