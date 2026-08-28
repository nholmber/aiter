# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Adapted from flash-linear-attention / vLLM (see _triton_kernels copy).

from __future__ import annotations

import torch
import triton

from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.fused_rearrange_sigmoid_gdr import (
    fused_rearrange_sigmoid_gated_delta_rule_rmsnorm_silu_update_kernel,
    fused_rearrange_sigmoid_gated_delta_rule_update_kernel,
)


def fused_rearrange_sigmoid_gated_delta_rule(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    qkv: torch.Tensor,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
    core_attn_out: torch.Tensor | None = None,
    output_gate: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Triton sigmoid-gated delta rule over packed QKV (decode-oriented).

    Passing ``output_gate``, ``norm_weight``, and ``norm_eps`` opts into the
    Qwen TP8-specialized decode kernel. That path computes
    ``RMSNorm(output) * SiLU(output_gate)`` in the recurrent kernel and writes
    the fused result to ``core_attn_out``. Omitting all three arguments keeps
    the existing tiled GDR kernel unchanged.
    """
    expected_shape = (qkv.shape[0], key_dim * 2 + value_dim)
    assert qkv.shape == expected_shape, (
        f"expect qkv to be in shape {expected_shape}, got {qkv.shape}"
    )
    if scale is None:
        scale = head_k_dim**-0.5
    else:
        assert scale > 0, "scale must be positive"

    B = 1
    T = qkv.shape[0]
    H = key_dim // head_k_dim
    HV = value_dim // head_v_dim
    K = head_k_dim
    V = head_v_dim
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 4
    norm_fusion_args = (output_gate, norm_weight, norm_eps)
    use_rmsnorm_silu_fusion = all(arg is not None for arg in norm_fusion_args)
    if any(arg is not None for arg in norm_fusion_args) and not use_rmsnorm_silu_fusion:
        raise ValueError(
            "output_gate, norm_weight, and norm_eps must be provided together"
        )

    if inplace_final_state and ssm_state_indices is None:
        raise ValueError(
            "ssm_state_indices is required when inplace_final_state=True "
            "(kernel indexes final state slots per token)."
        )

    o = (
        core_attn_out[: NK * B * T * HV * V].view(NK, B, T, HV, V)
        if core_attn_out is not None
        else qkv.new_empty(NK, B, T, HV, V)
    )
    if inplace_final_state:
        if initial_state is None:
            raise ValueError("initial_state is required when inplace_final_state=True")
        final_state = initial_state
    else:
        st_dtype = initial_state.dtype if initial_state is not None else qkv.dtype
        final_state = qkv.new_empty(T, HV, V, K, dtype=st_dtype)

    stride_init_state_token = (
        int(initial_state.stride(0)) if initial_state is not None else 0
    )
    stride_final_state_token = int(final_state.stride(0))

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    stride_qkv_l, stride_qkv_hd = qkv.stride()

    if use_rmsnorm_silu_fusion:
        assert output_gate is not None
        assert norm_weight is not None
        assert norm_eps is not None
        if qkv.dtype != torch.bfloat16:
            raise ValueError("fused GDR RMSNorm/SiLU requires BF16 qkv")
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise ValueError("fused GDR RMSNorm/SiLU requires BF16 a and b")
        if dt_bias.dtype != torch.bfloat16:
            raise ValueError("fused GDR RMSNorm/SiLU requires BF16 dt_bias")
        if A_log.dtype != torch.float32:
            raise ValueError("fused GDR RMSNorm/SiLU requires FP32 A_log")
        if (H, HV, K, V, BK, BV) != (2, 16, 128, 128, 128, 32):
            raise ValueError(
                "fused GDR RMSNorm/SiLU requires Qwen TP8 shape H=2, HV=16, K=V=128"
            )
        if not use_qk_l2norm_in_kernel:
            raise ValueError("fused GDR RMSNorm/SiLU requires in-kernel Q/K L2 norm")
        if is_kda:
            raise ValueError("fused GDR RMSNorm/SiLU does not support KDA")
        if not inplace_final_state or initial_state is None:
            raise ValueError(
                "fused GDR RMSNorm/SiLU requires an in-place initial-state pool"
            )
        if cu_seqlens is None or N != T:
            raise ValueError(
                "fused GDR RMSNorm/SiLU requires one indexed decode token per sequence"
            )
        if num_accepted_tokens is not None:
            raise ValueError("fused GDR RMSNorm/SiLU does not support spec decoding")
        if ssm_state_indices is None or ssm_state_indices.ndim != 1:
            raise ValueError(
                "fused GDR RMSNorm/SiLU requires one state-pool index per sequence"
            )
        if ssm_state_indices.numel() < N:
            raise ValueError("ssm_state_indices is shorter than the decode batch")
        if initial_state.ndim != 4 or initial_state.shape[1:] != (HV, V, K):
            raise ValueError(
                "initial_state must have shape "
                f"(state_pool, {HV}, {V}, {K}), got {tuple(initial_state.shape)}"
            )
        if initial_state.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("initial_state must be BF16 or FP32")
        if output_gate.shape != (T, HV, V):
            raise ValueError(
                "output_gate must have shape "
                f"{(T, HV, V)}, got {tuple(output_gate.shape)}"
            )
        if output_gate.dtype != torch.bfloat16:
            raise ValueError("fused GDR RMSNorm/SiLU requires BF16 output_gate")
        if norm_weight.ndim != 1 or norm_weight.numel() != V:
            raise ValueError(f"norm_weight must have shape ({V},)")
        if norm_weight.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("norm_weight must be BF16 or FP32")
        if norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if o.dtype != torch.bfloat16:
            raise ValueError("fused GDR RMSNorm/SiLU requires BF16 output")
        if not o.is_contiguous():
            raise ValueError("fused GDR RMSNorm/SiLU requires contiguous output")

        stride_gate_l, stride_gate_h, stride_gate_v = output_gate.stride()
        grid = (N * HV,)
        fused_rearrange_sigmoid_gated_delta_rule_rmsnorm_silu_update_kernel[grid](
            A_log=A_log,
            a=a.contiguous(),
            b=b.contiguous(),
            dt_bias=dt_bias,
            beta=beta,
            threshold=threshold,
            qkv=qkv,
            output_gate=output_gate,
            norm_weight=norm_weight,
            o=o,
            h0=initial_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            scale=scale,
            norm_eps=norm_eps,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            stride_qkv_l=stride_qkv_l,
            stride_qkv_hd=stride_qkv_hd,
            stride_gate_l=stride_gate_l,
            stride_gate_h=stride_gate_h,
            stride_gate_v=stride_gate_v,
            stride_init_state_token=stride_init_state_token,
            stride_indices_seq=stride_indices_seq,
            stride_norm_weight=norm_weight.stride(0),
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return o.squeeze(0), final_state

    grid = (NK, NV, N * HV)
    fused_rearrange_sigmoid_gated_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        qkv=qkv,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_qkv_l=stride_qkv_l,
        stride_qkv_hd=stride_qkv_hd,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state
