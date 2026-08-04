#!/usr/bin/env python3

import argparse
import os

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

M = 16
HIDDEN = 6144
INTER = 512
EXPERTS = 257
TOPK = 9


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
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--replays", type=int, default=5)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    torch.manual_seed(args.seed)

    hidden = torch.randn((M, HIDDEN), dtype=torch.bfloat16)
    routed_scores = torch.randn(
        (M, EXPERTS - 1),
        dtype=torch.bfloat16,
    )
    w1 = torch.randn(
        (EXPERTS, 2 * INTER, HIDDEN),
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        (EXPERTS, HIDDEN, INTER),
        dtype=torch.bfloat16,
    )

    routed_weights, routed_ids = fused_topk(
        hidden,
        routed_scores,
        TOPK - 1,
        True,
    )
    topk_ids = torch.cat(
        (
            routed_ids,
            torch.full((M, 1), EXPERTS - 1, dtype=torch.int32),
        ),
        dim=1,
    ).contiguous()
    topk_weights = torch.cat(
        (
            routed_weights,
            torch.ones((M, 1), dtype=torch.float32),
        ),
        dim=1,
    ).contiguous()

    assert torch.all(topk_ids[:, -1] == EXPERTS - 1)
    assert torch.all(topk_weights[:, -1] == 1)

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

    def run():
        return fused_moe(
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

    eager_out = run()
    torch.cuda.synchronize()
    eager_diff = normalized_diff(reference, eager_out)
    eager_us = elapsed_us(run, args.warmup, args.iterations)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_out = run()

    graph_us = elapsed_us(
        graph.replay,
        args.warmup,
        args.iterations,
    )
    graph_diffs = []
    for _ in range(args.replays):
        graph.replay()
        torch.cuda.synchronize()
        graph_diffs.append(normalized_diff(reference, graph_out))

    unique_routed = int(torch.unique(routed_ids).numel())
    print(
        "private_grouped=" f"{os.environ.get('AITER_GLM52_M16_PRIVATE_GROUPED', '0')}"
    )
    print(f"seed={args.seed} unique_routed={unique_routed}")
    print(f"eager_us={eager_us:.3f} eager_diff={eager_diff:.8e}")
    print(
        f"graph_us={graph_us:.3f} "
        f"graph_diff_min={min(graph_diffs):.8e} "
        f"graph_diff_max={max(graph_diffs):.8e}"
    )


if __name__ == "__main__":
    main()
