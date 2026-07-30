#!/usr/bin/env python3
"""Benchmark GLM-5.2 TP4 flat single-stage assembly against FlyDSL flat."""

import argparse
import os

os.environ.setdefault("AITER_GLM52_FUSED_MOE", "1")
os.environ.setdefault("AITER_GLM52_FUSED_MOE_MIN_M", "1")
os.environ.setdefault("AITER_GLM52_FUSED_MOE_MAX_M", "4")

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import (
    _moe_prepare_unsorted_input,
    fused_moe,
    fused_moe_1stage,
    fused_topk,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.flydsl.mxfp4_flat_moe_kernels import flydsl_mxfp4_flat_moe
from aiter.ops.flydsl.mxfp4_flat_single_stage_moe_kernels import (
    _SYNC_WORKSPACES,
    flydsl_mxfp4_flat_single_stage_moe,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils


HIDDEN = 6144
INTER = 512
EXPERTS = 257
TOPK = 9
ASM_KERNELS = {
    128: "_ZN5aiter50fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x128E",
    256: "_ZN5aiter50fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256E",
}


def normalized_diff(reference, actual):
    a = reference.double()
    b = actual.double()
    return (
        1 - 2 * (a * b).sum() / (a.square() + b.square()).sum()
    ).item()


def elapsed_us(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--correctness-repeats", type=int, default=5)
    parser.add_argument("--graph-replays", type=int, default=0)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(41)

    max_m = max(args.m)
    hidden_all = torch.randn((max_m, HIDDEN), dtype=torch.bfloat16)
    routed_scores = torch.randn(
        (max_m, EXPERTS - 1), dtype=torch.bfloat16
    )
    w1 = torch.randn(
        (EXPERTS, 2 * INTER, HIDDEN), dtype=torch.bfloat16
    )
    w2 = torch.randn((EXPERTS, HIDDEN, INTER), dtype=torch.bfloat16)

    quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1_q, w1_scale = quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(EXPERTS, 2 * INTER, HIDDEN // 2)
    w2_q = w2_q.view(EXPERTS, HIDDEN, INTER // 2)
    w1_kernel = shuffle_weight(w1_q, layout=(16, 16))
    w2_kernel = shuffle_weight(w2_q, layout=(16, 16))
    w1_kernel.is_shuffled = True
    w2_kernel.is_shuffled = True
    w1_scale_kernel = fp4_utils.e8m0_shuffle(w1_scale)
    w2_scale_kernel = fp4_utils.e8m0_shuffle(w2_scale)

    for m in args.m:
        hidden = hidden_all[:m]
        routed_weights, routed_ids = fused_topk(
            hidden, routed_scores[:m], TOPK - 1, True
        )
        topk_ids = torch.cat(
            (
                routed_ids,
                torch.full((m, 1), EXPERTS - 1, dtype=torch.int32),
            ),
            dim=1,
        ).contiguous()
        topk_weights = torch.cat(
            (
                routed_weights,
                torch.ones((m, 1), dtype=torch.float32),
            ),
            dim=1,
        ).contiguous()

        hidden_q, hidden_scale = quant(
            hidden,
            quant_dtype=dtypes.fp4x2,
        )
        ref_stage1 = torch_moe_stage1(
            hidden_q,
            w1_q,
            w2_q,
            topk_weights,
            topk_ids,
            dtype=torch.float32,
            activation=aiter.ActivationType.Silu,
            quant_type=aiter.QuantType.per_1x32,
            a1_scale=hidden_scale,
            w1_scale=w1_scale,
        )
        ref_stage1_q, ref_stage1_scale = quant(
            ref_stage1,
            quant_dtype=dtypes.fp4x2,
        )
        reference = torch_moe_stage2(
            ref_stage1_q,
            w1_q,
            w2_q,
            topk_weights,
            topk_ids,
            dtype=torch.bfloat16,
            quant_type=aiter.QuantType.per_1x32,
            w2_scale=w2_scale,
            a2_scale=ref_stage1_scale,
        )

        flat_out = flydsl_mxfp4_flat_moe(
            hidden_states=hidden,
            w1=w1_kernel,
            w1_scale=w1_scale_kernel,
            w2=w2_kernel,
            w2_scale=w2_scale_kernel,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        torch.cuda.synchronize()

        print(f"M={m}")
        print(
            "  reference diffs: "
            f"flydsl-2stage={normalized_diff(reference, flat_out):.8e}"
        )
        public_out = fused_moe(
            hidden,
            w1_kernel,
            w2_kernel,
            topk_weights,
            topk_ids,
            activation=aiter.ActivationType.Silu,
            quant_type=aiter.QuantType.per_1x32,
            w1_scale=w1_scale_kernel,
            w2_scale=w2_scale_kernel,
        )
        torch.cuda.synchronize()
        print(
            f"  public dispatch diff={normalized_diff(reference, public_out):.8e}"
        )
        flat_us = elapsed_us(
            lambda: flydsl_mxfp4_flat_moe(
                hidden_states=hidden,
                w1=w1_kernel,
                w1_scale=w1_scale_kernel,
                w2=w2_kernel,
                w2_scale=w2_scale_kernel,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            ),
            args.warmup,
            args.iterations,
        )
        print(f"  flydsl-2stage: {flat_us:.3f} us")

        single_out = flydsl_mxfp4_flat_single_stage_moe(
            hidden_states=hidden,
            w1=w1_kernel,
            w1_scale=w1_scale_kernel,
            w2=w2_kernel,
            w2_scale=w2_scale_kernel,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        torch.cuda.synchronize()
        single_diff = normalized_diff(flat_out, single_out)
        single_us = elapsed_us(
            lambda: flydsl_mxfp4_flat_single_stage_moe(
                hidden_states=hidden,
                w1=w1_kernel,
                w1_scale=w1_scale_kernel,
                w2=w2_kernel,
                w2_scale=w2_scale_kernel,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            ),
            args.warmup,
            args.iterations,
        )
        print(
            f"  flydsl-1stage-chunk128: {single_us:.3f} us "
            f"diff_vs_flat={single_diff:.8e} "
            f"diff_vs_ref={normalized_diff(reference, single_out):.8e}"
        )
        repeat_diffs = []
        for _ in range(args.correctness_repeats):
            repeated = flydsl_mxfp4_flat_single_stage_moe(
                hidden_states=hidden,
                w1=w1_kernel,
                w1_scale=w1_scale_kernel,
                w2=w2_kernel,
                w2_scale=w2_scale_kernel,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )
            torch.cuda.synchronize()
            repeat_diffs.append(normalized_diff(reference, repeated))
        if repeat_diffs:
            print(
                "    repeated correctness: "
                f"min={min(repeat_diffs):.8e} "
                f"max={max(repeat_diffs):.8e}"
            )

        if args.graph_replays:
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                graph_out = flydsl_mxfp4_flat_single_stage_moe(
                    hidden_states=hidden,
                    w1=w1_kernel,
                    w1_scale=w1_scale_kernel,
                    w2=w2_kernel,
                    w2_scale=w2_scale_kernel,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                )
            graph_diffs = []
            for _ in range(args.graph_replays):
                graph.replay()
                torch.cuda.synchronize()
                graph_diffs.append(normalized_diff(reference, graph_out))
            print(
                "    graph replay correctness: "
                f"min={min(graph_diffs):.8e} "
                f"max={max(graph_diffs):.8e}"
            )
        if single_diff > 1e-3:
            single_out_2 = flydsl_mxfp4_flat_single_stage_moe(
                hidden_states=hidden,
                w1=w1_kernel,
                w1_scale=w1_scale_kernel,
                w2=w2_kernel,
                w2_scale=w2_scale_kernel,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
            )
            torch.cuda.synchronize()
            print(
                "    debug: "
                f"ref_norm={flat_out.float().norm().item():.6f} "
                f"out_norm={single_out.float().norm().item():.6f} "
                f"max_abs={(flat_out.float() - single_out.float()).abs().max().item():.6f} "
                f"nan={torch.isnan(single_out).sum().item()} "
                f"repeat_diff={normalized_diff(single_out, single_out_2):.8e} "
                f"repeat_ref_diff={normalized_diff(flat_out, single_out_2):.8e} "
                f"repeat_norm={single_out_2.float().norm().item():.6f} "
                f"sync={[x.cpu().tolist() for x in _SYNC_WORKSPACES.values()]}"
            )

        for subgu, kernel_name in ASM_KERNELS.items():
            (
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                moe_buf,
            ) = _moe_prepare_unsorted_input(
                topk_ids,
                topk_weights,
                HIDDEN,
                dtypes.bf16,
            )

            def run_asm():
                return fused_moe_1stage(
                    hidden,
                    w1_kernel,
                    w2_kernel,
                    TOPK,
                    sorted_ids,
                    sorted_weights,
                    sorted_expert_ids,
                    num_valid_ids,
                    moe_buf,
                    True,
                    block_size_M=16,
                    activation=aiter.ActivationType.Silu,
                    quant_type=aiter.QuantType.per_1x32,
                    xbf16=True,
                    kernelName=kernel_name,
                    q_dtype_a=dtypes.fp4x2,
                    q_dtype_w=dtypes.fp4x2,
                    w1_scale=w1_scale_kernel,
                    w2_scale=w2_scale_kernel,
                    M=m,
                    device=hidden.device,
                    doweight_stage1=False,
                )

            asm_out = run_asm()
            torch.cuda.synchronize()
            diff = normalized_diff(flat_out, asm_out)
            asm_us = elapsed_us(
                run_asm,
                args.warmup,
                args.iterations,
            )
            print(
                f"  asm-1stage-subgu{subgu}: {asm_us:.3f} us "
                f"diff_vs_flat={diff:.8e} "
                f"diff_vs_ref={normalized_diff(reference, asm_out):.8e}"
            )


if __name__ == "__main__":
    main()
