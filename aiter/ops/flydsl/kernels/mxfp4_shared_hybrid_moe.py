# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic shared-expert hybrid MXFP4 MoE kernels."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .mxfp4_gemm1 import _bm_constants, _gemm1_body, _global_i32_at
from .mxfp4_gemm2 import (
    _gemm2_body,
    _issue_a_load_lds,
    saq_slot_bytes,
    tiling,
)
from .mxfp4_gemm_common import (
    _buffer_rsrc,
    _gep1,
    _global_base_ptr1,
    _raw,
    k_half_for,
    k_tiles_total_for,
    kStages,
    lds_acc_bytes_for,
    num_n_blocks_for,
)


def compile_mxfp4_shared_hybrid_stage1(
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    BM=16,
    BN=256,
    BK=256,
    routed_use_nt=True,
    shared_use_nt=False,
    interleave=False,
):
    """Sparse routed GEMM1 plus one grouped shared-expert GEMM1."""
    if BM != 16 or BN != 256 or BK != 256:
        raise ValueError(
            "shared-hybrid Stage 1 requires BM=16 and BN=BK=256"
        )
    if TOPK < 2:
        raise ValueError("shared-hybrid Stage 1 requires TOPK >= 2")

    routed_topk = TOPK - 1
    kh_tile = BK // 2
    n_out = 2 * D_INTER
    n_blocks = num_n_blocks_for(n_out, BN)
    _, _, _, lds_bytes = _bm_constants(
        BM, BN, kh_tile, k_tiles_total_for(D_HIDDEN, BK)
    )
    gu_tag = "il" if interleave else "sep"
    rnt_tag = "nt" if routed_use_nt else "cached"
    snt_tag = "nt" if shared_use_nt else "cached"
    name = (
        f"mxfp4_shared_hybrid_g1_h{D_HIDDEN}_i{D_INTER}_ne{NE}_tk{TOPK}"
        f"_bm{BM}_r{rnt_tag}_s{snt_tag}_{gu_tag}_v1"
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
        routed_blocks = num_routed * fx.Int32(n_blocks)
        total_blocks = routed_blocks + fx.Int32(n_blocks)
        max_m_blocks = num_routed + fx.Int32(1)
        work_id = (bx_i32 < routed_blocks).select(
            bx_i32 + fx.Int32(n_blocks), bx_i32 - routed_blocks
        )

        # Stage 2 uses atomic accumulation. Spread output zeroing over the full
        # Stage-1 grid; completion of this kernel is the device-wide barrier.
        zero_dword0 = bx_i32 * fx.Int32(256) + tx_i32
        zero_stride = total_blocks * fx.Int32(256)
        total_out_dwords = i32_ntok * fx.Int32(D_HIDDEN // 2)
        for zero_pass in range_constexpr(2):
            zero_dword = zero_dword0 + fx.Int32(zero_pass) * zero_stride
            if zero_dword < total_out_dwords:
                llvm.StoreOp(
                    _raw(fx.Int32(0)),
                    _gep1(
                        _global_base_ptr1(arg_final_out),
                        zero_dword * fx.Int32(4),
                    ),
                )

        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        if work_id < fx.Int32(n_blocks):
            n_block = work_id
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
                interleave=interleave,
                direct_route=True,
                direct_expert=fx.Int32(NE - 1),
                direct_sequential_rows=True,
            )
        else:
            routed_pid = work_id - fx.Int32(n_blocks)
            route = routed_pid // fx.Int32(n_blocks)
            n_block = routed_pid - route * fx.Int32(n_blocks)
            token = route // fx.Int32(routed_topk)
            slot = route - token * fx.Int32(routed_topk)
            raw_route = token * fx.Int32(TOPK) + slot
            expert = rocdl.readfirstlane(
                T.i32, _raw(_global_i32_at(arg_topk_ids, raw_route))
            )
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
                interleave=interleave,
                direct_route=True,
                direct_expert=expert,
                direct_token=token,
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
            * fx.Index(routed_topk * n_blocks)
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


def compile_mxfp4_shared_hybrid_stage2(
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    BM=16,
    BN=256,
    BK=256,
    routed_use_nt=False,
    shared_use_nt=False,
    shared_weight_is_one=False,
):
    """Route-direct GEMM2 plus one grouped shared-expert GEMM2."""
    if BM != 16 or BN != 256 or BK != 256:
        raise ValueError(
            "shared-hybrid Stage 2 requires BM=16 and BN=BK=256"
        )
    if TOPK < 2:
        raise ValueError("shared-hybrid Stage 2 requires TOPK >= 2")

    routed_topk = TOPK - 1
    kh_tile = BK // 2
    n_blocks = num_n_blocks_for(D_HIDDEN, BN)
    k_half = k_half_for(D_INTER)
    k_tiles = k_tiles_total_for(D_INTER, BK)
    a_stages = kStages if k_tiles <= kStages else 3
    slot_bytes = saq_slot_bytes(BM, kh_tile)
    lds_bytes = lds_acc_bytes_for(BM, BN) + a_stages * slot_bytes
    rnt_tag = "nt" if routed_use_nt else "cached"
    snt_tag = "nt" if shared_use_nt else "cached"
    sw_tag = "_sw1" if shared_weight_is_one else ""
    name = (
        f"mxfp4_shared_hybrid_g2_h{D_HIDDEN}_i{D_INTER}_ne{NE}_tk{TOPK}"
        f"_bm{BM}_r{rnt_tag}_s{snt_tag}{sw_tag}_v1"
    )

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def stage2_kernel(
        arg_inter_q: fx.Int64,
        arg_inter_scale: fx.Int64,
        arg_w2: fx.Int64,
        arg_w2_scale: fx.Int64,
        arg_topk_ids: fx.Int64,
        arg_topk_weights: fx.Int64,
        arg_out: fx.Int64,
        i32_ntok: fx.Int32,
    ):
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        num_routed = i32_ntok * fx.Int32(routed_topk)
        routed_blocks = num_routed * fx.Int32(n_blocks)
        work_id = (bx_i32 < routed_blocks).select(
            bx_i32 + fx.Int32(n_blocks), bx_i32 - routed_blocks
        )
        max_m_blocks = num_routed + fx.Int32(1)
        aq_records = fx.Int64(max_m_blocks) * fx.Int64(BM * k_half)
        aq_rsrc = _buffer_rsrc(arg_inter_q, aq_records)
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        saq_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
        n_load_waves, rows_per_wave, k_subblocks = tiling(BM)

        if work_id < fx.Int32(n_blocks):
            n_block = work_id
            m_block = num_routed
            m_row = m_block * fx.Int32(BM)
            if wave < fx.Int32(n_load_waves):
                for k_slot in range(k_tiles):
                    for sub in range(k_subblocks):
                        lds_row = (
                            wave * fx.Int32(rows_per_wave)
                            + fx.Int32(sub * 8)
                        )
                        car = m_row + lds_row + (lane // fx.Int32(8))
                        _issue_a_load_lds(
                            aq_rsrc,
                            saq_base_i32,
                            k_slot,
                            k_slot,
                            car,
                            lane,
                            slot_bytes,
                            lds_row,
                            KH_TILE=kh_tile,
                            k_half=k_half,
                        )
            rocdl.sched_barrier(0)
            tile = m_block * fx.Int32(n_blocks) + n_block
            _gemm2_body(
                lds_raw_ptr,
                arg_inter_scale,
                arg_w2,
                arg_w2_scale,
                arg_topk_ids,
                arg_topk_ids,
                arg_topk_weights,
                i32_ntok,
                max_m_blocks,
                arg_out,
                arg_inter_scale,
                tile,
                lane,
                wave,
                BM,
                shared_use_nt,
                NE,
                D_HIDDEN,
                "atomic",
                aq_rsrc=aq_rsrc,
                D_INTER=D_INTER,
                D_INTER_REAL=D_INTER,
                aStages=a_stages,
                BN=BN,
                BK=BK,
                KH_TILE=kh_tile,
                direct_route=True,
                direct_expert=fx.Int32(NE - 1),
                direct_sequential_rows=True,
                direct_weight_stride=TOPK,
                direct_weight_col=TOPK - 1,
                direct_sequential_weight_one=shared_weight_is_one,
            )
        else:
            routed_pid = work_id - fx.Int32(n_blocks)
            route = routed_pid // fx.Int32(n_blocks)
            n_block = routed_pid - route * fx.Int32(n_blocks)
            token = route // fx.Int32(routed_topk)
            slot = route - token * fx.Int32(routed_topk)
            raw_route = token * fx.Int32(TOPK) + slot
            expert = rocdl.readfirstlane(
                T.i32, _raw(_global_i32_at(arg_topk_ids, raw_route))
            )
            weight = fx.Float32(
                llvm.load(
                    T.f32,
                    _gep1(
                        _global_base_ptr1(arg_topk_weights),
                        raw_route * fx.Int32(4),
                    ),
                    invariant=True,
                )
            )
            m_block = route
            m_row = m_block * fx.Int32(BM)
            if wave < fx.Int32(n_load_waves):
                for k_slot in range(k_tiles):
                    for sub in range(k_subblocks):
                        lds_row = (
                            wave * fx.Int32(rows_per_wave)
                            + fx.Int32(sub * 8)
                        )
                        car = m_row + lds_row + (lane // fx.Int32(8))
                        _issue_a_load_lds(
                            aq_rsrc,
                            saq_base_i32,
                            k_slot,
                            k_slot,
                            car,
                            lane,
                            slot_bytes,
                            lds_row,
                            KH_TILE=kh_tile,
                            k_half=k_half,
                        )
            rocdl.sched_barrier(0)
            tile = m_block * fx.Int32(n_blocks) + n_block
            _gemm2_body(
                lds_raw_ptr,
                arg_inter_scale,
                arg_w2,
                arg_w2_scale,
                arg_topk_ids,
                arg_topk_ids,
                arg_topk_weights,
                i32_ntok,
                max_m_blocks,
                arg_out,
                arg_inter_scale,
                tile,
                lane,
                wave,
                BM,
                routed_use_nt,
                NE,
                D_HIDDEN,
                "atomic",
                aq_rsrc=aq_rsrc,
                D_INTER=D_INTER,
                D_INTER_REAL=D_INTER,
                aStages=a_stages,
                BN=BN,
                BK=BK,
                KH_TILE=kh_tile,
                direct_route=True,
                direct_expert=expert,
                direct_token=token,
                direct_weight=weight,
            )

    @flyc.jit
    def launch_stage2(
        arg_inter_q: fx.Int64,
        arg_inter_scale: fx.Int64,
        arg_w2: fx.Int64,
        arg_w2_scale: fx.Int64,
        arg_topk_ids: fx.Int64,
        arg_topk_weights: fx.Int64,
        arg_out: fx.Int64,
        i32_ntok: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = (
            arith.index_cast(T.index, _raw(i32_ntok))
            * fx.Index(routed_topk * n_blocks)
            + fx.Index(n_blocks)
        )
        stage2_kernel(
            arg_inter_q,
            arg_inter_scale,
            arg_w2,
            arg_w2_scale,
            arg_topk_ids,
            arg_topk_weights,
            arg_out,
            i32_ntok,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    launch_stage2.compile_hints["llvm_options"] = {
        "enable-post-misched": False
    }
    return launch_stage2
