# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Python wrapper for the GLM-5.2 flat single-stage FlyDSL prototype."""

import functools

import torch

from aiter.ops.flydsl import moe_kernels as _moe_kernels


@functools.cache
def _get_kernel(D_HIDDEN, D_INTER, NE, TOPK, use_nt_g1, use_nt_g2):
    from .kernels.mxfp4_flat_single_stage_moe import (
        compile_mxfp4_flat_single_stage,
    )

    return compile_mxfp4_flat_single_stage(
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=TOPK,
        use_nt_g1=use_nt_g1,
        use_nt_g2=use_nt_g2,
    )


_SYNC_WORKSPACES = {}


def _get_sync_workspace(device, tokens):
    key = (device.index, tokens)
    workspace = _SYNC_WORKSPACES.get(key)
    if workspace is None:
        workspace = torch.zeros(
            (tokens, 3),
            dtype=torch.int32,
            device=device,
        )
        torch.cuda.synchronize(device)
        _SYNC_WORKSPACES[key] = workspace
    return workspace


def flydsl_mxfp4_flat_single_stage_moe(
    *,
    hidden_states,
    w1,
    w1_scale,
    w2,
    w2_scale,
    topk_ids,
    topk_weights,
    out=None,
    use_nt_g1=True,
    use_nt_g2=False,
    stream=None,
):
    M, D_HIDDEN = hidden_states.shape
    D_INTER = w1.shape[1] // 2
    NE = w1.shape[0]
    TOPK = topk_ids.shape[1]
    if M < 1 or M > 4:
        raise ValueError(
            f"single-stage flat prototype supports 1 <= M <= 4, got {M}"
        )
    if D_HIDDEN != 6144 or D_INTER != 512 or NE != 257 or TOPK != 9:
        raise ValueError(
            "single-stage flat prototype requires GLM-5.2 TP4 shape "
            "(H=6144, I=512, E=257, topk=9)"
        )

    hidden_states = hidden_states.contiguous()
    w1 = w1.contiguous()
    w1_scale = w1_scale.contiguous().view(torch.uint8)
    w2 = w2.contiguous()
    w2_scale = w2_scale.contiguous().view(torch.uint8)
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    if out is None:
        out = torch.empty(
            (M, D_HIDDEN),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )

    max_m_blocks = M * TOPK
    inter_scale = torch.empty(
        (max_m_blocks * 1024,),
        dtype=torch.uint8,
        device=hidden_states.device,
    )
    sync = _get_sync_workspace(hidden_states.device, M)
    launch = _get_kernel(
        D_HIDDEN,
        D_INTER,
        NE,
        TOPK,
        use_nt_g1,
        use_nt_g2,
    )
    _moe_kernels._run_compiled(
        launch,
        (
            hidden_states.data_ptr(),
            w1.data_ptr(),
            w1_scale.data_ptr(),
            w2.data_ptr(),
            w2_scale.data_ptr(),
            topk_ids.data_ptr(),
            topk_weights.data_ptr(),
            inter_scale.data_ptr(),
            sync.data_ptr(),
            out.data_ptr(),
            M,
            torch.cuda.current_stream() if stream is None else stream,
        ),
    )
    return out
