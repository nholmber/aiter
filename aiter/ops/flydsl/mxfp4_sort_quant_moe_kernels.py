# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.2 BM16 MXFP4 MoE with fused sort and one-time input quantization."""

import torch

from aiter.ops.moe_mxfp4_aux import mxfp4_moe_sort_quant_shared

from .mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1
from .mxfp4_gemm2_kernels import flydsl_mxfp4_gemm2


def flydsl_mxfp4_sort_quant_moe(
    *,
    hidden_states,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk_ids,
    topk_weights,
    out=None,
    stream=None,
):
    """Run the deterministic-shared GLM-5.2 decode path in three launches.

    The final top-k slot must route to expert 256 with weight exactly one.
    The preparation launch sorts routes, zeros the atomic output, and
    quantizes each BF16 token to MXFP4 once. GEMM1 gathers the token scales
    through ``m_indices`` and reuses them from LDS across all four waves.
    """

    M, D_HIDDEN = hidden_states.shape
    D_INTER = w1.shape[1] // 2
    NE = w1.shape[0]
    TOPK = topk_ids.shape[1]
    BM = 16
    if M < 5 or M > BM:
        raise ValueError(f"sort-quant GLM-5.2 path supports 5 <= M <= {BM}, got {M}")
    if (D_HIDDEN, D_INTER, NE, TOPK) != (6144, 512, 257, 9):
        raise ValueError(
            "sort-quant path requires GLM-5.2 TP4 shape "
            "(H=6144, I=512, E=257, topk=9)"
        )

    hidden_states = hidden_states.contiguous()
    w1 = w1.contiguous()
    w1_scale = w1_scale.contiguous().view(torch.uint8)
    w2 = w2.contiguous()
    w2_scale = w2_scale.contiguous().view(torch.uint8)
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    device = hidden_states.device

    active = min(NE, M * TOPK)
    max_sorted = (M * TOPK + active * (BM - 1) + BM - 1) // BM * BM
    sorted_ids = torch.empty(max_sorted, dtype=torch.int32, device=device)
    sorted_expert_ids = torch.empty(
        max_sorted // BM,
        dtype=torch.int32,
        device=device,
    )
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    reverse_sorted = torch.empty(M * TOPK, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(
        max_sorted,
        dtype=torch.float32,
        device=device,
    )
    m_indices = torch.empty(max_sorted, dtype=torch.int32, device=device)
    a_quant = torch.empty(
        (M, D_HIDDEN // 2),
        dtype=torch.uint8,
        device=device,
    )
    a_scale = torch.empty(
        (M, D_HIDDEN // 32),
        dtype=torch.uint8,
        device=device,
    )
    if out is None:
        out = torch.empty(
            (M, D_HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
    inter_quant = torch.empty(
        (max_sorted, D_INTER // 2),
        dtype=torch.uint8,
        device=device,
    )
    inter_scale_bytes = max_sorted * max(
        (1024 // 64) * 4,
        (D_INTER // 32) * 2,
    )
    inter_scale = torch.empty(
        inter_scale_bytes,
        dtype=torch.uint8,
        device=device,
    )

    mxfp4_moe_sort_quant_shared(
        a_input=hidden_states,
        topk_ids=topk_ids,
        topk_weight=topk_weights,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=num_valid_ids,
        reverse_sorted=reverse_sorted,
        sorted_weights=sorted_weights,
        a_quant=a_quant,
        a_scale=a_scale,
        m_indices=m_indices,
        bf16_zero_out=out,
        NE=NE,
        TOPK=TOPK,
        D_HIDDEN=D_HIDDEN,
        MB=BM,
    )

    # Cached W1 wins through the rounded M=8 bucket. The M=16 bucket prefers
    # non-temporal W1; a wider XCD group starts winning once M reaches 10.
    stage1_use_nt = M >= 9
    stage1_xcd = 8 if not stage1_use_nt else (1 if M == 9 else 4)
    flydsl_mxfp4_gemm1(
        a_quant=a_quant,
        a_scale_sorted_shuffled=a_scale,
        w1_u8=w1,
        w1_scale_u8=w1_scale,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=num_valid_ids,
        m_indices=m_indices,
        inter_sorted_quant=inter_quant,
        inter_sorted_shuffled_scale=inter_scale,
        hidden_states=hidden_states,
        n_tokens=M,
        BM=BM,
        use_nt=stage1_use_nt,
        inline_quant=False,
        NE=NE,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        topk=TOPK,
        BN=256,
        BK=256,
        xcd_swizzle=stage1_xcd,
        direct_token_scales=True,
        stream=stream,
    )
    flydsl_mxfp4_gemm2(
        inter_sorted_quant=inter_quant,
        inter_sorted_shuffled_scale=inter_scale,
        w2_u8=w2,
        w2_scale_u8=w2_scale,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=num_valid_ids,
        sorted_token_ids=sorted_ids,
        sorted_weights=sorted_weights,
        flat_out=out,
        M_logical=M,
        max_sorted=max_sorted,
        BM=BM,
        use_nt=False,
        atomic=True,
        mxfp4out=False,
        NE=NE,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        topk=TOPK,
        BN=256,
        BK=256,
        stream=stream,
    )
    return out
