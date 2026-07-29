# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Token-centric serial-route MXFP4 GEMM1 experiment for GLM-5.2."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .mxfp4_gemm1 import _bm_constants, _gemm1_body, _global_i32_at
from .mxfp4_gemm_common import (
    _gep1,
    _global_base_ptr1,
    _raw,
    k_tiles_total_for,
    num_n_blocks_for,
)


def compile_mxfp4_token_serial_shared_stage1(
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    BM=16,
    BN=64,
    BK=256,
    routed_use_nt=True,
    shared_use_nt=False,
):
    """Group four route-private BN64 GEMM1 bodies by token and expert phase.

    This is a correctness-first stepping stone toward a wave-native
    token-centric kernel. It deliberately repeats input quantization and GEMM
    work four times inside each workgroup; only the dispatch geometry changes.
    """
    if BM != 16 or BN != 64 or BK != 256:
        raise ValueError(
            "token-serial Stage 1 requires BM=16, BN=64, and BK=256"
        )
    if TOPK != 9:
        raise ValueError(
            "token-serial Stage 1 requires eight routed slots plus one shared slot"
        )

    routed_topk = TOPK - 1
    routes_per_phase = 4
    num_phases = routed_topk // routes_per_phase
    kh_tile = BK // 2
    n_out = 2 * D_INTER
    n_blocks = num_n_blocks_for(n_out, BN)
    _, _, _, lds_bytes = _bm_constants(
        BM, BN, kh_tile, k_tiles_total_for(D_HIDDEN, BK)
    )
    rnt_tag = "nt" if routed_use_nt else "cached"
    snt_tag = "nt" if shared_use_nt else "cached"
    name = (
        f"mxfp4_token_serial_shared_g1_h{D_HIDDEN}_i{D_INTER}"
        f"_ne{NE}_tk{TOPK}_bm{BM}_bn{BN}_r{rnt_tag}_s{snt_tag}_v1"
    )

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def stage1_kernel(
        arg_hidden: fx.Int64,
        arg_w1: fx.Int64,
        arg_w1_scale: fx.Int64,
        arg_topk_ids: fx.Int64,
        arg_inter_q: fx.Int64,
        arg_inter_scale: fx.Int64,
        arg_final_out: fx.Int64,
        i32_ntok: fx.Int32,
    ):
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))

        num_routed = i32_ntok * fx.Int32(routed_topk)
        routed_blocks = (
            i32_ntok * fx.Int32(num_phases * n_blocks)
        )
        max_m_blocks = num_routed + fx.Int32(1)

        # Stage 2 atomically accumulates into final_out. The token-centric
        # Stage-1 grid has enough threads to clear every output dword once.
        zero_dword = bx_i32 * fx.Int32(256) + tx_i32
        total_out_dwords = i32_ntok * fx.Int32(D_HIDDEN // 2)
        if zero_dword < total_out_dwords:
            llvm.StoreOp(
                _raw(fx.Int32(0)),
                _gep1(
                    _global_base_ptr1(arg_final_out),
                    zero_dword * fx.Int32(4),
                ),
            )

        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        if bx_i32 < routed_blocks:
            token_phase = bx_i32 // fx.Int32(n_blocks)
            n_block = bx_i32 - token_phase * fx.Int32(n_blocks)
            token = token_phase // fx.Int32(num_phases)
            phase = token_phase - token * fx.Int32(num_phases)

            for route_in_phase in range_constexpr(routes_per_phase):
                slot = (
                    phase * fx.Int32(routes_per_phase)
                    + fx.Int32(route_in_phase)
                )
                raw_route = token * fx.Int32(TOPK) + slot
                expert = rocdl.readfirstlane(
                    T.i32, _raw(_global_i32_at(arg_topk_ids, raw_route))
                )
                route = token * fx.Int32(routed_topk) + slot
                tile = route * fx.Int32(n_blocks) + n_block
                _gemm1_body(
                    lds_raw_ptr,
                    arg_inter_q,
                    arg_inter_scale,
                    arg_w1,
                    arg_w1_scale,
                    arg_topk_ids,
                    arg_topk_ids,
                    arg_inter_q,
                    arg_inter_scale,
                    arg_hidden,
                    tile,
                    lane,
                    wave,
                    routed_use_nt,
                    i32_ntok,
                    max_m_blocks,
                    BM=BM,
                    BN=BN,
                    BK=BK,
                    inline_quant=True,
                    K=D_HIDDEN,
                    N_OUT=n_out,
                    NE=NE,
                    interleave=False,
                    direct_route=True,
                    direct_expert=expert,
                    direct_token=token,
                )
                if route_in_phase != routes_per_phase - 1:
                    gpu.barrier()
        else:
            n_block = bx_i32 - routed_blocks
            shared_m_block = num_routed
            tile = shared_m_block * fx.Int32(n_blocks) + n_block
            _gemm1_body(
                lds_raw_ptr,
                arg_inter_q,
                arg_inter_scale,
                arg_w1,
                arg_w1_scale,
                arg_topk_ids,
                arg_topk_ids,
                arg_inter_q,
                arg_inter_scale,
                arg_hidden,
                tile,
                lane,
                wave,
                shared_use_nt,
                i32_ntok,
                max_m_blocks,
                BM=BM,
                BN=BN,
                BK=BK,
                inline_quant=True,
                K=D_HIDDEN,
                N_OUT=n_out,
                NE=NE,
                interleave=False,
                direct_route=True,
                direct_expert=fx.Int32(NE - 1),
                direct_sequential_rows=True,
            )

    @flyc.jit
    def launch_stage1(
        arg_hidden: fx.Int64,
        arg_w1: fx.Int64,
        arg_w1_scale: fx.Int64,
        arg_topk_ids: fx.Int64,
        arg_inter_q: fx.Int64,
        arg_inter_scale: fx.Int64,
        arg_final_out: fx.Int64,
        i32_ntok: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = (
            arith.index_cast(T.index, _raw(i32_ntok))
            * fx.Index(num_phases * n_blocks)
            + fx.Index(n_blocks)
        )
        stage1_kernel(
            arg_hidden,
            arg_w1,
            arg_w1_scale,
            arg_topk_ids,
            arg_inter_q,
            arg_inter_scale,
            arg_final_out,
            i32_ntok,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    launch_stage1.compile_hints["llvm_options"] = {
        "enable-post-misched": False
    }
    return launch_stage1
