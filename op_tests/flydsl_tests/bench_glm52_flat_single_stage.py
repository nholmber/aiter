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
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.flydsl.mxfp4_flat_moe_kernels import flydsl_mxfp4_flat_moe
from aiter.ops.flydsl.mxfp4_flat_single_stage_moe_kernels import (
    _SYNC_WORKSPACES,
    flydsl_mxfp4_flat_single_stage_moe,
)
from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1
from aiter.ops.flydsl.mxfp4_gemm2_kernels import flydsl_mxfp4_gemm2
from aiter.ops.quant import mxfp4_moe_sort_fwd
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
ASM_SORTED_KERNELS = {
    256: "_ZN5aiter49fmoe_bf16_pertokenMXfp4_g1u1_novs_silu_2tg_32x256E",
    512: "_ZN5aiter49fmoe_bf16_pertokenMXfp4_g1u1_novs_silu_1tg_32x512E",
}


def normalized_diff(reference, actual):
    a = reference.double()
    b = actual.double()
    return (1 - 2 * (a * b).sum() / (a.square() + b.square()).sum()).item()


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
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--bm16-sweep", action="store_true")
    parser.add_argument("--bm16-selected", action="store_true")
    args = parser.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(args.seed)

    max_m = max(args.m)
    hidden_all = torch.randn((max_m, HIDDEN), dtype=torch.bfloat16)
    routed_scores = torch.randn((max_m, EXPERTS - 1), dtype=torch.bfloat16)
    w1 = torch.randn((EXPERTS, 2 * INTER, HIDDEN), dtype=torch.bfloat16)
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
        print(f"  public dispatch diff={normalized_diff(reference, public_out):.8e}")
        public_us = elapsed_us(
            lambda: fused_moe(
                hidden,
                w1_kernel,
                w2_kernel,
                topk_weights,
                topk_ids,
                activation=aiter.ActivationType.Silu,
                quant_type=aiter.QuantType.per_1x32,
                w1_scale=w1_scale_kernel,
                w2_scale=w2_scale_kernel,
            ),
            args.warmup,
            args.iterations,
        )
        print(f"  public-selected: {public_us:.3f} us")
        if m >= 5 and args.graph_replays:
            public_graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(public_graph):
                public_graph_out = fused_moe(
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
            public_graph_us = elapsed_us(
                public_graph.replay,
                args.warmup,
                args.iterations,
            )
            public_graph_diffs = []
            for _ in range(args.graph_replays):
                public_graph.replay()
                torch.cuda.synchronize()
                public_graph_diffs.append(normalized_diff(reference, public_graph_out))
            print(
                f"    graph replay: {public_graph_us:.3f} us "
                f"diff_min={min(public_graph_diffs):.8e} "
                f"diff_max={max(public_graph_diffs):.8e}"
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

        if m <= 4:
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
        else:
            print("  flydsl-1stage-chunk128: skipped (M>4 residency bound)")

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

        for subgu, kernel_name in ASM_SORTED_KERNELS.items():

            def run_sorted_asm():
                (
                    sorted_ids,
                    sorted_weights,
                    sorted_expert_ids,
                    num_valid_ids,
                    moe_buf,
                ) = moe_sorting(
                    topk_ids,
                    topk_weights,
                    EXPERTS,
                    HIDDEN,
                    dtypes.bf16,
                    block_size=32,
                    accumulate=True,
                )
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
                    block_size_M=32,
                    activation=aiter.ActivationType.Silu,
                    quant_type=aiter.QuantType.per_1x32,
                    xbf16=False,
                    kernelName=kernel_name,
                    q_dtype_a=dtypes.fp4x2,
                    q_dtype_w=dtypes.fp4x2,
                    w1_scale=w1_scale_kernel,
                    w2_scale=w2_scale_kernel,
                    M=m,
                    device=hidden.device,
                    doweight_stage1=False,
                )

            sorted_out = run_sorted_asm()
            torch.cuda.synchronize()
            sorted_us = elapsed_us(
                run_sorted_asm,
                args.warmup,
                args.iterations,
            )
            print(
                f"  asm-sorted-1stage-subgu{subgu}: {sorted_us:.3f} us "
                f"diff_vs_ref={normalized_diff(reference, sorted_out):.8e}"
            )

            (
                pre_sorted_ids,
                pre_sorted_weights,
                pre_sorted_expert_ids,
                pre_num_valid_ids,
                pre_moe_buf,
            ) = moe_sorting(
                topk_ids,
                topk_weights,
                EXPERTS,
                HIDDEN,
                dtypes.bf16,
                block_size=32,
                accumulate=True,
            )
            pre_a1, pre_a1_scale = quant(
                hidden,
                quant_dtype=dtypes.fp4x2,
            )
            pre_a1_scale = mxfp4_moe_sort_fwd(
                pre_a1_scale,
                sorted_ids=pre_sorted_ids,
                num_valid_ids=pre_num_valid_ids,
                token_num=m,
                cols=HIDDEN,
            )

            def run_sorted_kernel_only():
                pre_moe_buf.zero_()
                aiter.fmoe_g1u1(
                    pre_moe_buf,
                    pre_a1,
                    w1_kernel,
                    w2_kernel,
                    pre_sorted_ids,
                    pre_sorted_weights,
                    pre_sorted_expert_ids,
                    pre_num_valid_ids,
                    TOPK,
                    pre_a1_scale,
                    w1_scale_kernel.view(EXPERTS, -1),
                    w2_scale_kernel.view(EXPERTS, -1),
                    kernel_name,
                    fc2_smooth_scale=None,
                    activation=aiter.ActivationType.Silu,
                )
                return pre_moe_buf

            kernel_only_out = run_sorted_kernel_only()
            torch.cuda.synchronize()
            kernel_only_us = elapsed_us(
                run_sorted_kernel_only,
                args.warmup,
                args.iterations,
            )
            print(
                f"    prequantized kernel+zero: {kernel_only_us:.3f} us "
                f"diff_vs_ref={normalized_diff(reference, kernel_only_out):.8e}"
            )

        if m >= 5:
            active = min(EXPERTS, m * TOPK)
            max_sorted = (m * TOPK + active * 15 + 15) // 16 * 16
            sorted_ids = torch.empty(
                max_sorted,
                dtype=torch.int32,
                device=hidden.device,
            )
            sorted_expert_ids = torch.empty(
                max_sorted // 16,
                dtype=torch.int32,
                device=hidden.device,
            )
            num_valid_ids = torch.empty(
                2,
                dtype=torch.int32,
                device=hidden.device,
            )
            reverse_sorted = torch.empty(
                m * TOPK,
                dtype=torch.int32,
                device=hidden.device,
            )
            sorted_weights = torch.empty(
                max_sorted,
                dtype=torch.float32,
                device=hidden.device,
            )
            m_indices = torch.empty(
                max_sorted,
                dtype=torch.int32,
                device=hidden.device,
            )
            a_quant = torch.empty(
                (m, HIDDEN // 2),
                dtype=torch.uint8,
                device=hidden.device,
            )
            a_scale = torch.empty(
                (m, HIDDEN // 32),
                dtype=torch.uint8,
                device=hidden.device,
            )
            out = torch.empty(
                (m, HIDDEN),
                dtype=torch.bfloat16,
                device=hidden.device,
            )
            inter_quant = torch.empty(
                (max_sorted, INTER // 2),
                dtype=torch.uint8,
                device=hidden.device,
            )
            inter_scale = torch.empty(
                (max_sorted * 1024,),
                dtype=torch.uint8,
                device=hidden.device,
            )
            main_sorted_ids = torch.empty_like(sorted_ids)
            main_sorted_expert_ids = torch.empty_like(sorted_expert_ids)
            main_num_valid_ids = torch.empty_like(num_valid_ids)
            main_reverse_sorted = torch.empty_like(reverse_sorted)
            main_sorted_weights = torch.empty_like(sorted_weights)
            main_m_indices = torch.empty_like(m_indices)
            main_out = torch.empty_like(out)
            main_inter_quant = torch.empty_like(inter_quant)
            main_inter_scale = torch.empty_like(inter_scale)
            main_dummy = torch.empty(
                1,
                dtype=torch.uint8,
                device=hidden.device,
            )
            empty_bf16 = torch.empty(
                0,
                dtype=torch.bfloat16,
                device=hidden.device,
            )

            def run_bm16_sort_quant():
                aiter.mxfp4_moe_sort_quant_shared(
                    a_input=hidden,
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
                    NE=EXPERTS,
                    TOPK=TOPK,
                    D_HIDDEN=HIDDEN,
                    MB=16,
                )

            def run_bm16_gemm1(use_nt, bn=256, xcd_swizzle=0):
                flydsl_mxfp4_gemm1(
                    a_quant=a_quant,
                    a_scale_sorted_shuffled=a_scale,
                    w1_u8=w1_kernel,
                    w1_scale_u8=w1_scale_kernel,
                    sorted_expert_ids=sorted_expert_ids,
                    cumsum_tensor=num_valid_ids,
                    m_indices=m_indices,
                    inter_sorted_quant=inter_quant,
                    inter_sorted_shuffled_scale=inter_scale,
                    hidden_states=hidden,
                    n_tokens=m,
                    BM=16,
                    use_nt=use_nt,
                    inline_quant=False,
                    NE=EXPERTS,
                    D_HIDDEN=HIDDEN,
                    D_INTER=INTER,
                    topk=TOPK,
                    BN=bn,
                    xcd_swizzle=xcd_swizzle,
                    direct_token_scales=True,
                    pipeline_direct_scales=m <= 8,
                )

            def run_bm16_gemm2():
                flydsl_mxfp4_gemm2(
                    inter_sorted_quant=inter_quant,
                    inter_sorted_shuffled_scale=inter_scale,
                    w2_u8=w2_kernel,
                    w2_scale_u8=w2_scale_kernel,
                    sorted_expert_ids=sorted_expert_ids,
                    cumsum_tensor=num_valid_ids,
                    sorted_token_ids=sorted_ids,
                    sorted_weights=sorted_weights,
                    flat_out=out,
                    M_logical=m,
                    max_sorted=max_sorted,
                    BM=16,
                    use_nt=False,
                    atomic=True,
                    mxfp4out=False,
                    NE=EXPERTS,
                    D_HIDDEN=HIDDEN,
                    D_INTER=INTER,
                    topk=TOPK,
                )

            def run_bm16_quant_once(use_nt, bn=256, xcd_swizzle=0):
                run_bm16_sort_quant()
                run_bm16_gemm1(use_nt, bn, xcd_swizzle)
                run_bm16_gemm2()
                return out

            def run_main_sort():
                aiter.mxfp4_moe_sort(
                    topk_ids=topk_ids,
                    topk_weight=topk_weights,
                    sorted_token_ids=main_sorted_ids,
                    sorted_expert_ids=main_sorted_expert_ids,
                    cumsum_tensor=main_num_valid_ids,
                    reverse_sorted=main_reverse_sorted,
                    sorted_weights=main_sorted_weights,
                    m_indices=main_m_indices,
                    bf16_zero_out=main_out,
                    bf16_zero_workspace=empty_bf16,
                    M_logical=m,
                    NE=EXPERTS,
                    TOPK=TOPK,
                    D_HIDDEN=HIDDEN,
                    D_INTER=1,
                    MB=16,
                    prologue=0,
                )

            def run_main_gemm1():
                flydsl_mxfp4_gemm1(
                    a_quant=main_dummy,
                    a_scale_sorted_shuffled=main_dummy,
                    w1_u8=w1_kernel,
                    w1_scale_u8=w1_scale_kernel,
                    sorted_expert_ids=main_sorted_expert_ids,
                    cumsum_tensor=main_num_valid_ids,
                    m_indices=main_m_indices,
                    inter_sorted_quant=main_inter_quant,
                    inter_sorted_shuffled_scale=main_inter_scale,
                    hidden_states=hidden,
                    n_tokens=m,
                    BM=16,
                    use_nt=True,
                    inline_quant=True,
                    NE=EXPERTS,
                    D_HIDDEN=HIDDEN,
                    D_INTER=INTER,
                    topk=TOPK,
                )

            def run_main_gemm2():
                flydsl_mxfp4_gemm2(
                    inter_sorted_quant=main_inter_quant,
                    inter_sorted_shuffled_scale=main_inter_scale,
                    w2_u8=w2_kernel,
                    w2_scale_u8=w2_scale_kernel,
                    sorted_expert_ids=main_sorted_expert_ids,
                    cumsum_tensor=main_num_valid_ids,
                    sorted_token_ids=main_sorted_ids,
                    sorted_weights=main_sorted_weights,
                    flat_out=main_out,
                    M_logical=m,
                    max_sorted=max_sorted,
                    BM=16,
                    use_nt=False,
                    atomic=True,
                    mxfp4out=False,
                    NE=EXPERTS,
                    D_HIDDEN=HIDDEN,
                    D_INTER=INTER,
                    topk=TOPK,
                )

            def run_main_f16in():
                run_main_sort()
                run_main_gemm1()
                run_main_gemm2()
                return main_out

            main_us = elapsed_us(
                run_main_f16in,
                args.warmup,
                args.iterations,
            )
            main_result = run_main_f16in()
            torch.cuda.synchronize()
            main_diff = normalized_diff(reference, main_result)
            main_sort_us = elapsed_us(
                run_main_sort,
                args.warmup,
                args.iterations,
            )
            run_main_sort()
            main_gemm1_us = elapsed_us(
                run_main_gemm1,
                args.warmup,
                args.iterations,
            )
            run_main_gemm1()
            torch.cuda.synchronize()
            main_gemm2_us = elapsed_us(
                run_main_gemm2,
                args.warmup,
                args.iterations,
            )
            print(
                f"  forced-main-f16in: {main_us:.3f} us " f"diff_vs_ref={main_diff:.8e}"
            )
            print(
                "    stages: "
                f"sort={main_sort_us:.3f} us "
                f"g1={main_gemm1_us:.3f} us "
                f"g2={main_gemm2_us:.3f} us "
                f"sum={main_sort_us + main_gemm1_us + main_gemm2_us:.3f} us"
            )
            if args.graph_replays:
                main_graph = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                with torch.cuda.graph(main_graph):
                    main_graph_out = run_main_f16in()
                main_graph_us = elapsed_us(
                    main_graph.replay,
                    args.warmup,
                    args.iterations,
                )
                main_graph.replay()
                torch.cuda.synchronize()
                print(
                    f"    graph replay: {main_graph_us:.3f} us "
                    f"diff={normalized_diff(reference, main_graph_out):.8e}"
                )

            sort_quant_us = elapsed_us(
                run_bm16_sort_quant,
                args.warmup,
                args.iterations,
            )
            bm16_candidates = [
                (256, True, 0),
                (256, False, 0),
            ]
            if args.bm16_selected:
                bm16_candidates = [
                    (256, False, 8),
                    (256, True, 1),
                    (256, True, 4),
                    (512, True, 0),
                    (512, True, 1),
                ]
            elif args.bm16_sweep:
                bm16_candidates.extend(
                    (bn, use_nt, 0)
                    for bn in (512, 128, 64)
                    for use_nt in (True, False)
                )
                bm16_candidates.extend(
                    (256, use_nt, xcd_swizzle)
                    for xcd_swizzle in (1, 2, 4, 8)
                    for use_nt in (True, False)
                )
                bm16_candidates.extend(
                    (512, True, xcd_swizzle)
                    for xcd_swizzle in (1, 2, 4, 8)
                )

            for bn, use_nt, xcd_swizzle in bm16_candidates:
                quant_once_us = elapsed_us(
                    lambda: run_bm16_quant_once(
                        use_nt,
                        bn,
                        xcd_swizzle,
                    ),
                    args.warmup,
                    args.iterations,
                )
                quant_once_out = run_bm16_quant_once(
                    use_nt,
                    bn,
                    xcd_swizzle,
                )
                torch.cuda.synchronize()
                quant_once_diff = normalized_diff(reference, quant_once_out)

                run_bm16_sort_quant()
                gemm1_us = elapsed_us(
                    lambda: run_bm16_gemm1(
                        use_nt,
                        bn,
                        xcd_swizzle,
                    ),
                    args.warmup,
                    args.iterations,
                )
                run_bm16_gemm1(use_nt, bn, xcd_swizzle)
                torch.cuda.synchronize()
                gemm2_us = elapsed_us(
                    run_bm16_gemm2,
                    args.warmup,
                    args.iterations,
                )
                stage_sum_us = sort_quant_us + gemm1_us + gemm2_us
                print(
                    "  bm16-sortquant-preq-g1-"
                    f"{'nt' if use_nt else 'cached'}"
                    f"-bn{bn}-xcd{xcd_swizzle}: "
                    f"{quant_once_us:.3f} us "
                    f"diff_vs_ref={quant_once_diff:.8e}"
                )
                print(
                    "    stages: "
                    f"sort+quant={sort_quant_us:.3f} us "
                    f"g1={gemm1_us:.3f} us "
                    f"g2={gemm2_us:.3f} us "
                    f"sum={stage_sum_us:.3f} us"
                )
                selected_candidate = (
                    (not use_nt and xcd_swizzle == 8)
                    or (use_nt and xcd_swizzle in (1, 4))
                    or (bn == 512 and use_nt and xcd_swizzle == 0)
                )
                if args.graph_replays and selected_candidate:
                    quant_graph = torch.cuda.CUDAGraph()
                    torch.cuda.synchronize()
                    with torch.cuda.graph(quant_graph):
                        quant_graph_out = run_bm16_quant_once(
                            use_nt,
                            bn,
                            xcd_swizzle,
                        )
                    quant_graph_us = elapsed_us(
                        quant_graph.replay,
                        args.warmup,
                        args.iterations,
                    )
                    quant_graph.replay()
                    torch.cuda.synchronize()
                    print(
                        f"    graph replay: {quant_graph_us:.3f} us "
                        f"diff={normalized_diff(reference, quant_graph_out):.8e}"
                    )


if __name__ == "__main__":
    main()
