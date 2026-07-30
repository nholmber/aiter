from pathlib import Path


path = Path("/tmp/aiter-main/aiter/fused_moe.py")
text = path.read_text()

if "AITER_GLM52_FUSED_MOE" in text:
    raise SystemExit("GLM-5.2 fused-MoE dispatch is already installed")

marker = "    grouped_a8w4_out = None\n"
if text.count(marker) != 1:
    raise RuntimeError(
        f"expected one grouped-a8w4 marker in {path}, found {text.count(marker)}"
    )

dispatch = '''    glm52_fused_moe_enabled = (
        os.environ.get(
            "AITER_GLM52_FUSED_MOE",
            os.environ.get("AITER_GLM52_TOKEN_WAVE", "0"),
        )
        == "1"
    )
    glm52_fused_moe_min_m = int(
        os.environ.get(
            "AITER_GLM52_FUSED_MOE_MIN_M",
            os.environ.get("AITER_GLM52_TOKEN_WAVE_MIN_M", "1"),
        )
    )
    glm52_fused_moe_max_m = int(
        os.environ.get(
            "AITER_GLM52_FUSED_MOE_MAX_M",
            os.environ.get("AITER_GLM52_TOKEN_WAVE_MAX_M", "16"),
        )
    )
    if (
        glm52_fused_moe_enabled
        and glm52_fused_moe_min_m <= M <= glm52_fused_moe_max_m
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
        if M == 1:
            from aiter.ops.flydsl.mxfp4_flat_single_stage_moe_kernels import (
                flydsl_mxfp4_flat_single_stage_moe,
            )

            return flydsl_mxfp4_flat_single_stage_moe(
                hidden_states=hidden_states,
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                topk_ids=topk_ids,
                topk_weights=topk_weight,
            )
        if M == 2:
            from aiter.ops.flydsl.mxfp4_flat_moe_kernels import (
                flydsl_mxfp4_flat_moe,
            )

            return flydsl_mxfp4_flat_moe(
                hidden_states=hidden_states,
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                topk_ids=topk_ids,
                topk_weights=topk_weight,
            )
        if M <= 4:
            from aiter.ops.flydsl.mxfp4_shared_hybrid_moe_kernels import (
                flydsl_mxfp4_shared_hybrid_moe,
            )

            return flydsl_mxfp4_shared_hybrid_moe(
                hidden_states=hidden_states,
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                topk_ids=topk_ids,
                topk_weights=topk_weight,
            )

        from aiter.ops.flydsl.mxfp4_token_wave_shared_moe_kernels import (
            flydsl_mxfp4_token_wave_shared_moe,
        )

        return flydsl_mxfp4_token_wave_shared_moe(
            hidden_states=hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            topk_ids=topk_ids,
            topk_weights=topk_weight,
        )

'''

path.write_text(text.replace(marker, dispatch + marker))
print(f"patched {path}")
