# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import torch

from aiter.ops.flydsl import moe_kernels as _moe_kernels
from aiter.ops.flydsl.kernels.mxfp4_embedded_sort_route_moe import (
    PRIVATE_BLOCK_STRIDE,
)


@functools.cache
def _get_stage1(
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    use_nt,
    interleave,
    dispatch_n_groups,
):
    from .kernels.mxfp4_embedded_sort_route_moe import (
        compile_mxfp4_embedded_sort_stage1,
    )

    return compile_mxfp4_embedded_sort_stage1(
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=TOPK,
        use_nt=use_nt,
        interleave=interleave,
        dispatch_n_groups=dispatch_n_groups,
    )


def flydsl_mxfp4_embedded_sort_moe(
    *,
    hidden_states,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk_ids,
    topk_weights,
    out=None,
    stage1_use_nt=True,
    stage2_use_nt=False,
    interleave=False,
    stage1_dispatch_n_groups=None,
    stage2_dispatch_n_groups=0,
    stream=None,
):
    """Two-stage embedded-sort MXFP4 MoE path for GLM-5.2 M=8..16.

    Stage 1 embeds route compaction and BF16->MXFP4 quantization, then writes a
    route-private FP4 intermediate. Stage 2 consumes that intermediate
    route-directly and atomically accumulates the routed weights.
    """
    M, D_HIDDEN = hidden_states.shape
    D_INTER = w1.shape[1] // 2
    NE = w1.shape[0]
    TOPK = topk_ids.shape[1]
    BM = 16

    if M < 1 or M > BM:
        raise ValueError(f"embedded-sort path supports 1 <= M <= {BM}, got {M}")
    if D_HIDDEN % 256 != 0 or D_INTER % 256 != 0:
        raise ValueError(
            "embedded-sort path requires hidden/inter dimensions divisible by 256"
        )
    if stage1_dispatch_n_groups is None:
        # M=8 prefers two groups; M=16 prefers four. M=12 is effectively tied.
        stage1_dispatch_n_groups = 2 if M <= 12 else 4

    num_routes = M * TOPK
    max_m_blocks = num_routes * PRIVATE_BLOCK_STRIDE
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
        stage1_use_nt,
        interleave,
        stage1_dispatch_n_groups,
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

    from aiter.ops.flydsl.mxfp4_flat_moe_kernels import (
        flydsl_mxfp4_flat_stage2,
    )

    return flydsl_mxfp4_flat_stage2(
        inter_q=inter_q,
        inter_scale=inter_scale,
        w2=w2,
        w2_scale=w2_scale,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        out=out,
        D_INTER=D_INTER,
        use_nt=stage2_use_nt,
        dispatch_n_groups=stage2_dispatch_n_groups,
        stream=run_stream,
    )


# Backward-compatible name used by the development benchmark harness.
flydsl_mxfp4_embedded_sort_private_moe = flydsl_mxfp4_embedded_sort_moe
