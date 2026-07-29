# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import torch

from aiter.ops.flydsl import moe_kernels as _moe_kernels
from aiter.ops.flydsl.kernels.mxfp4_flat_moe import ROUTE_BLOCK_STRIDE


@functools.cache
def _get_flat_stage1(
    D_HIDDEN, D_INTER, NE, TOPK, BM, BN, BK, use_nt, interleave
):
    from .kernels.mxfp4_flat_moe import compile_mxfp4_flat_stage1

    return compile_mxfp4_flat_stage1(
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=TOPK,
        BM=BM,
        BN=BN,
        BK=BK,
        use_nt=use_nt,
        interleave=interleave,
    )


@functools.cache
def _get_flat_stage2(
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    BM,
    BN,
    BK,
    use_nt,
    dispatch_n_groups,
):
    from .kernels.mxfp4_flat_moe import compile_mxfp4_flat_stage2

    return compile_mxfp4_flat_stage2(
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=TOPK,
        BM=BM,
        BN=BN,
        BK=BK,
        use_nt=use_nt,
        dispatch_n_groups=dispatch_n_groups,
    )


def _logical_shape(hidden_states, w1, topk_ids):
    M, D_HIDDEN = hidden_states.shape
    NE = w1.shape[0]
    D_INTER = w1.shape[1] // 2
    TOPK = topk_ids.shape[1]
    return M, D_HIDDEN, D_INTER, NE, TOPK


def flydsl_mxfp4_flat_stage1(
    *,
    hidden_states,
    w1,
    w1_scale,
    topk_ids,
    out,
    use_nt=True,
    interleave=False,
    BM=16,
    BN=256,
    BK=256,
    stream=None,
):
    M, D_HIDDEN, D_INTER, NE, TOPK = _logical_shape(
        hidden_states, w1, topk_ids
    )
    max_m_blocks = M * TOPK * ROUTE_BLOCK_STRIDE
    private_rows = max_m_blocks * BM
    inter_q = torch.empty(
        (private_rows, D_INTER // 2),
        dtype=torch.uint8,
        device=hidden_states.device,
    )
    inter_scale = torch.empty(
        (max_m_blocks * 1024,),
        dtype=torch.uint8,
        device=hidden_states.device,
    )

    hidden_states = hidden_states.contiguous()
    w1 = w1.contiguous()
    w1_scale = w1_scale.contiguous().view(torch.uint8)
    topk_ids = topk_ids.to(torch.int32).contiguous()

    launch = _get_flat_stage1(
        D_HIDDEN,
        D_INTER,
        NE,
        TOPK,
        BM,
        BN,
        BK,
        use_nt,
        interleave,
    )
    _moe_kernels._run_compiled(
        launch,
        (
            hidden_states.data_ptr(),
            w1.data_ptr(),
            w1_scale.data_ptr(),
            topk_ids.data_ptr(),
            inter_q.data_ptr(),
            inter_scale.data_ptr(),
            out.data_ptr(),
            M,
            torch.cuda.current_stream() if stream is None else stream,
        ),
    )
    return inter_q, inter_scale


def flydsl_mxfp4_flat_stage2(
    *,
    inter_q,
    inter_scale,
    w2,
    w2_scale,
    topk_ids,
    topk_weights,
    out,
    D_INTER,
    use_nt=False,
    BM=16,
    BN=256,
    BK=256,
    dispatch_n_groups=0,
    stream=None,
):
    M = topk_ids.shape[0]
    D_HIDDEN = out.shape[1]
    NE = w2.shape[0]
    TOPK = topk_ids.shape[1]
    w2 = w2.contiguous()
    w2_scale = w2_scale.contiguous().view(torch.uint8)
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()

    launch = _get_flat_stage2(
        D_HIDDEN,
        D_INTER,
        NE,
        TOPK,
        BM,
        BN,
        BK,
        use_nt,
        dispatch_n_groups,
    )
    _moe_kernels._run_compiled(
        launch,
        (
            inter_q.data_ptr(),
            inter_scale.data_ptr(),
            w2.data_ptr(),
            w2_scale.data_ptr(),
            topk_ids.data_ptr(),
            topk_weights.data_ptr(),
            out.data_ptr(),
            M,
            torch.cuda.current_stream() if stream is None else stream,
        ),
    )
    return out


def flydsl_mxfp4_flat_moe(
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
    stage1_bn=None,
    interleave=False,
    stage2_dispatch_n_groups=0,
    stream=None,
):
    M, D_HIDDEN, D_INTER, _, _ = _logical_shape(hidden_states, w1, topk_ids)
    if stage1_bn is None:
        stage1_bn = 128 if M <= 4 else 256
    if out is None:
        out = torch.empty(
            (M, D_HIDDEN), dtype=torch.bfloat16, device=hidden_states.device
        )
    inter_q, inter_scale = flydsl_mxfp4_flat_stage1(
        hidden_states=hidden_states,
        w1=w1,
        w1_scale=w1_scale,
        topk_ids=topk_ids,
        out=out,
        use_nt=stage1_use_nt,
        interleave=interleave,
        BN=stage1_bn,
        stream=stream,
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
        stream=stream,
    )
