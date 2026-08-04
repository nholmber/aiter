#!/usr/bin/env python3

from pathlib import Path

path = Path("/tmp/aiter-main/aiter/fused_moe.py")
text = path.read_text()

sentinel = "AITER_GLM52_M16_PRIVATE_GROUPED"
if sentinel in text:
    raise SystemExit(f"{path} is already patched")

marker = "\n    grouped_a8w4_out = None\n"
if text.count(marker) != 1:
    raise RuntimeError(
        f"expected one grouped_a8w4_out marker in {path}, "
        f"found {text.count(marker)}"
    )

insertion = r"""
    if (
        os.environ.get("AITER_GLM52_M16_PRIVATE_GROUPED", "0") == "1"
        and M == 16
        and get_gfx() == "gfx950"
        and E == 257
        and model_dim == 6144
        and inter_dim == 512
        and topk == 9
        and dtype == dtypes.bf16
        and hidden_states.dtype == dtypes.bf16
        and activation == ActivationType.Silu
        and quant_type == QuantType.per_1x32
        and q_dtype_w == dtypes.fp4x2
        and isG1U1
        and isShuffled
        and gate_mode == GateMode.SEPARATED
        and not doweight_stage1
        and expert_mask is None
        and bias1 is None
        and bias2 is None
        and hidden_pad == 0
        and intermediate_pad == 0
        and w1_scale is not None
        and w2_scale is not None
    ):
        from aiter.ops.flydsl.mxfp4_routed_compact_shared_moe_kernels import (
            flydsl_mxfp4_routed_compact_shared_moe,
        )

        return flydsl_mxfp4_routed_compact_shared_moe(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weight,
            stage1_dispatch_n_groups=4,
            stage1_bn=256,
            stage2_dispatch_n_groups=0,
        )
"""

path.write_text(text.replace(marker, insertion + marker))
