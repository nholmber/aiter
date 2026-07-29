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
    compact_a=True,
    shared_bn=64,
    routed_bn=64,
    role_batch_size=4,
):
    from .kernels.mxfp4_token_wave_shared_moe import (
        compile_mxfp4_token_wave_shared_stage1,
    )

    return compile_mxfp4_token_wave_shared_stage1(
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=TOPK,
        routed_use_nt=routed_use_nt,
        shared_use_nt=shared_use_nt,
        compact_a=compact_a,
        shared_bn=shared_bn,
        BN=routed_bn,
        role_batch_size=role_batch_size,
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
    token_wave,
    role_batch_size,
):
    if token_wave:
        from .kernels.mxfp4_token_wave_shared_moe import (
            compile_mxfp4_token_wave_shared_stage2,
        )

        if not shared_weight_is_one:
            raise ValueError(
                "token-wave Stage 2 requires shared_weight_is_one=True"
            )
        return compile_mxfp4_token_wave_shared_stage2(
            D_HIDDEN=D_HIDDEN,
            D_INTER=D_INTER,
            NE=NE,
            TOPK=TOPK,
            routed_use_nt=routed_use_nt,
            shared_use_nt=shared_use_nt,
            role_batch_size=role_batch_size,
        )
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


def flydsl_mxfp4_token_wave_shared_moe(
    *,
    hidden_states,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk_ids,
    topk_weights,
    out=None,
    inter_q=None,
    inter_scale=None,
    stage1_routed_use_nt=None,
    stage1_shared_use_nt=False,
    stage1_compact_a=True,
    stage1_shared_bn=64,
    stage1_routed_bn=None,
    stage1_role_batch_size=4,
    stage2_routed_use_nt=False,
    stage2_shared_use_nt=False,
    stage2_token_wave=None,
    stage2_role_batch_size=4,
    shared_weight_is_one=True,
    stream=None,
):
    """Experimental shared-A/four-wave GLM-5.2 MXFP4 MoE."""
    M, D_HIDDEN = hidden_states.shape
    D_INTER = w1.shape[1] // 2
    NE = w1.shape[0]
    TOPK = topk_ids.shape[1]
    BM = 16
    if M < 1 or M > BM:
        raise ValueError(
            f"token-wave path supports 1 <= M <= {BM}, got {M}"
        )
    if TOPK != 9:
        raise ValueError(
            "token-wave path requires eight routed slots plus one shared slot"
        )
    if D_HIDDEN % 256 != 0 or D_INTER % 256 != 0:
        raise ValueError(
            "token-wave path requires hidden/inter dimensions divisible by 256"
        )
    use_large_bucket = M > 8
    if stage1_routed_use_nt is None:
        stage1_routed_use_nt = use_large_bucket
    if stage1_routed_bn is None:
        stage1_routed_bn = 128 if use_large_bucket else 64
    if stage2_token_wave is None:
        stage2_token_wave = use_large_bucket

    routed_topk = TOPK - 1
    max_m_blocks = M * routed_topk + 1
    private_rows = max_m_blocks * BM
    if out is None:
        out = torch.empty(
            (M, D_HIDDEN), dtype=torch.bfloat16, device=hidden_states.device
        )
    if inter_q is None:
        inter_q = torch.empty(
            (private_rows, D_INTER // 2),
            dtype=torch.uint8,
            device=hidden_states.device,
        )
    elif (
        inter_q.dtype != torch.uint8
        or inter_q.device != hidden_states.device
        or inter_q.numel() < private_rows * (D_INTER // 2)
    ):
        raise ValueError(
            "inter_q must be a uint8 device buffer with at least "
            f"{private_rows * (D_INTER // 2)} elements"
        )
    if inter_scale is None:
        inter_scale = torch.empty(
            (max_m_blocks * D_INTER,),
            dtype=torch.uint8,
            device=hidden_states.device,
        )
    elif (
        inter_scale.dtype != torch.uint8
        or inter_scale.device != hidden_states.device
        or inter_scale.numel() < max_m_blocks * D_INTER
    ):
        raise ValueError(
            "inter_scale must be a uint8 device buffer with at least "
            f"{max_m_blocks * D_INTER} elements"
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
        stage1_compact_a,
        stage1_shared_bn,
        stage1_routed_bn,
        stage1_role_batch_size,
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
        stage2_token_wave,
        stage2_role_batch_size,
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
