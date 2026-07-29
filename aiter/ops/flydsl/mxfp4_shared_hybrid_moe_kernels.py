# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import torch

from aiter.ops.flydsl import moe_kernels as _moe_kernels


@functools.cache
def _get_stage1(
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    routed_use_nt,
    shared_use_nt,
    interleave,
):
    from .kernels.mxfp4_shared_hybrid_moe import (
        compile_mxfp4_shared_hybrid_stage1,
    )

    return compile_mxfp4_shared_hybrid_stage1(
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=TOPK,
        routed_use_nt=routed_use_nt,
        shared_use_nt=shared_use_nt,
        interleave=interleave,
    )


@functools.cache
def _get_stage2(
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    routed_use_nt,
    shared_use_nt,
    shared_weight_is_one,
):
    from .kernels.mxfp4_shared_hybrid_moe import (
        compile_mxfp4_shared_hybrid_stage2,
    )

    return compile_mxfp4_shared_hybrid_stage2(
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=TOPK,
        routed_use_nt=routed_use_nt,
        shared_use_nt=shared_use_nt,
        shared_weight_is_one=shared_weight_is_one,
    )


def flydsl_mxfp4_shared_hybrid_moe(
    *,
    hidden_states,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk_ids,
    topk_weights,
    out=None,
    stage1_routed_use_nt=True,
    stage1_shared_use_nt=False,
    stage2_routed_use_nt=False,
    stage2_shared_use_nt=False,
    shared_weight_is_one=True,
    interleave=False,
    stream=None,
):
    """Two-launch GLM-5.2 path with a deterministic final shared route.

    The first ``TOPK-1`` routes are sparse and route-direct. The final route is
    assumed to target expert ``NE-1`` for every token and is processed as one
    grouped BM16 expert block.
    """
    M, D_HIDDEN = hidden_states.shape
    D_INTER = w1.shape[1] // 2
    NE = w1.shape[0]
    TOPK = topk_ids.shape[1]
    BM = 16
    if M < 1 or M > BM:
        raise ValueError(
            f"shared-hybrid path supports 1 <= M <= {BM}, got {M}"
        )
    if TOPK < 2:
        raise ValueError("shared-hybrid path requires TOPK >= 2")
    if D_HIDDEN % 256 != 0 or D_INTER % 256 != 0:
        raise ValueError(
            "shared-hybrid path requires hidden/inter dimensions divisible by 256"
        )

    routed_topk = TOPK - 1
    max_m_blocks = M * routed_topk + 1
    private_rows = max_m_blocks * BM
    if out is None:
        out = torch.empty(
            (M, D_HIDDEN), dtype=torch.bfloat16, device=hidden_states.device
        )
    inter_q = torch.empty(
        (private_rows, D_INTER // 2),
        dtype=torch.uint8,
        device=hidden_states.device,
    )
    inter_scale = torch.empty(
        (max_m_blocks * D_INTER,),
        dtype=torch.uint8,
        device=hidden_states.device,
    )

    hidden_states = hidden_states.contiguous()
    w1 = w1.contiguous()
    w1_scale = w1_scale.contiguous().view(torch.uint8)
    w2 = w2.contiguous()
    w2_scale = w2_scale.contiguous().view(torch.uint8)
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    run_stream = torch.cuda.current_stream() if stream is None else stream

    stage1 = _get_stage1(
        D_HIDDEN,
        D_INTER,
        NE,
        TOPK,
        stage1_routed_use_nt,
        stage1_shared_use_nt,
        interleave,
    )
    _moe_kernels._run_compiled(
        stage1,
        (
            hidden_states.data_ptr(),
            w1.data_ptr(),
            w1_scale.data_ptr(),
            topk_ids.data_ptr(),
            inter_q.data_ptr(),
            inter_scale.data_ptr(),
            out.data_ptr(),
            M,
            run_stream,
        ),
    )

    stage2 = _get_stage2(
        D_HIDDEN,
        D_INTER,
        NE,
        TOPK,
        stage2_routed_use_nt,
        stage2_shared_use_nt,
        shared_weight_is_one,
    )
    _moe_kernels._run_compiled(
        stage2,
        (
            inter_q.data_ptr(),
            inter_scale.data_ptr(),
            w2.data_ptr(),
            w2_scale.data_ptr(),
            topk_ids.data_ptr(),
            topk_weights.data_ptr(),
            out.data_ptr(),
            M,
            run_stream,
        ),
    )
    return out
