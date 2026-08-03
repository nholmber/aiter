# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Route-private GEMM1 plus grouped GEMM2 for deterministic shared MXFP4 MoE."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .mxfp4_gemm1 import (
    _bm_constants,
    _gemm1_body,
    _global_i32_at,
)
from .mxfp4_gemm2 import (
    _gemm2_body,
    _issue_a_load_lds,
    saq_slot_bytes,
    tiling,
)
from .mxfp4_gemm_common import (
    _buffer_rsrc,
    _gep1,
    _gep3,
    _global_base_ptr1,
    _global_ptr1,
    _lds_ptr3,
    _raw,
    k_half_for,
    k_tiles_total_for,
    kStages,
    lds_acc_bytes_for,
    num_n_blocks_for,
)


def compile_mxfp4_routed_compact_shared_stage1(
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
    dispatch_n_groups=2,
):
    """Embed routed leader detection and scatter GEMM1 rows by private route."""
    if BM != 16 or BN not in (128, 256) or BK != 256:
        raise ValueError(
            "routed-compact/shared Stage 1 requires BM=16, "
            "BN in {128,256}, and BK=256"
        )
    if BN == 128 and interleave:
        raise ValueError("routed-compact/shared BN128 supports separated gate/up only")
    if TOPK < 2:
        raise ValueError("routed-compact/shared Stage 1 requires TOPK >= 2")

    routed_topk = TOPK - 1
    kh_tile = BK // 2
    n_out = 2 * D_INTER
    n_blocks = num_n_blocks_for(n_out, BN)
    if dispatch_n_groups not in (1, 2, 4):
        raise ValueError("dispatch_n_groups must be one of {1,2,4}")
    if n_blocks % dispatch_n_groups != 0:
        raise ValueError(
            f"N blocks ({n_blocks}) must be divisible by dispatch_n_groups "
            f"({dispatch_n_groups})"
        )

    _, _, _, body_lds_bytes = _bm_constants(
        BM, BN, kh_tile, k_tiles_total_for(D_HIDDEN, BK)
    )
    routes_offset = 32
    metadata_bytes = ((routes_offset + BM * 4 + 15) // 16) * 16
    lds_bytes = body_lds_bytes + metadata_bytes
    gu_tag = "il" if interleave else "sep"
    rnt_tag = "nt" if routed_use_nt else "cached"
    snt_tag = "nt" if shared_use_nt else "cached"
    bn_tag = "" if BN == 256 else f"_bn{BN}"
    name = (
        f"mxfp4_routed_compact_shared_g1_h{D_HIDDEN}_i{D_INTER}"
        f"_ne{NE}_tk{TOPK}_bm{BM}_r{rnt_tag}_s{snt_tag}_{gu_tag}"
        f"{bn_tag}_ng{dispatch_n_groups}_private_v2"
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
        arg_topk_weights: fx.Int64,
        arg_inter_q: fx.Int64,
        arg_inter_scale: fx.Int64,
        arg_sorted_token_ids: fx.Int64,
        arg_sorted_weights: fx.Int64,
        arg_expert_ids: fx.Int64,
        arg_counts: fx.Int64,
        arg_final_out: fx.Int64,
        i32_ntok: fx.Int32,
    ):
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        num_routed = i32_ntok * fx.Int32(routed_topk)
        routed_candidate_blocks = num_routed * fx.Int32(dispatch_n_groups)
        total_blocks = routed_candidate_blocks + fx.Int32(n_blocks)
        max_m_blocks = num_routed + fx.Int32(1)

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
        if bx_i32 < routed_candidate_blocks:
            route0 = bx_i32 // fx.Int32(dispatch_n_groups)
            n_group = bx_i32 - route0 * fx.Int32(dispatch_n_groups)
            token0 = route0 // fx.Int32(routed_topk)
            slot0 = route0 - token0 * fx.Int32(routed_topk)
            raw_route0 = token0 * fx.Int32(TOPK) + slot0
            expert = rocdl.readfirstlane(
                T.i32,
                _raw(_global_i32_at(arg_topk_ids, raw_route0)),
            )

            metadata_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr)) + fx.Int32(
                body_lds_bytes
            )
            metadata_ptr = _lds_ptr3(metadata_base_i32, fx.Int32(0))
            lds_flag = metadata_ptr
            lds_count = _lds_ptr3(metadata_base_i32, fx.Int32(4))
            lds_routes = _lds_ptr3(metadata_base_i32, fx.Int32(routes_offset))

            if tx_i32 == fx.Int32(0):
                llvm.StoreOp(
                    _raw(fx.Int32(0)),
                    lds_flag,
                    alignment=4,
                )
                llvm.StoreOp(
                    _raw(fx.Int32(0)),
                    lds_count,
                    alignment=4,
                )
            if tx_i32 < fx.Int32(BM):
                llvm.StoreOp(
                    _raw(num_routed),
                    _gep3(lds_routes, tx_i32 * fx.Int32(4)),
                    alignment=4,
                )
            gpu.barrier()

            if tx_i32 < route0:
                prev_token = tx_i32 // fx.Int32(routed_topk)
                prev_slot = tx_i32 - prev_token * fx.Int32(routed_topk)
                prev_raw = prev_token * fx.Int32(TOPK) + prev_slot
                prev_expert = _global_i32_at(arg_topk_ids, prev_raw)
                if prev_expert == expert:
                    llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.add,
                        lds_flag,
                        _raw(fx.Int32(1)),
                        llvm.AtomicOrdering.monotonic,
                        syncscope="workgroup",
                        alignment=4,
                    )
            gpu.barrier()
            is_leader = fx.Int32(
                rocdl.readfirstlane(T.i32, llvm.load(T.i32, lds_flag))
            ) == fx.Int32(0)

            if is_leader:
                if tx_i32 < num_routed:
                    route_token = tx_i32 // fx.Int32(routed_topk)
                    route_slot = tx_i32 - route_token * fx.Int32(routed_topk)
                    route_raw = route_token * fx.Int32(TOPK) + route_slot
                    route_expert = _global_i32_at(arg_topk_ids, route_raw)
                    if route_expert == expert:
                        slot = llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.add,
                            lds_count,
                            _raw(fx.Int32(1)),
                            llvm.AtomicOrdering.monotonic,
                            syncscope="workgroup",
                            alignment=4,
                        ).result
                        slot_i32 = fx.Int32(slot)
                        if slot_i32 < fx.Int32(BM):
                            llvm.StoreOp(
                                _raw(tx_i32),
                                _gep3(
                                    lds_routes,
                                    slot_i32 * fx.Int32(4),
                                ),
                                alignment=4,
                            )
                gpu.barrier()

            count = fx.Int32(
                llvm.load(
                    T.i32,
                    lds_count,
                    alignment=4,
                )
            )

            if n_group == fx.Int32(0):
                if is_leader:
                    if tx_i32 == fx.Int32(0):
                        llvm.StoreOp(
                            _raw(expert),
                            _global_ptr1(arg_expert_ids, route0 * fx.Int32(4)),
                            alignment=4,
                        )
                        llvm.StoreOp(
                            _raw(count),
                            _global_ptr1(arg_counts, route0 * fx.Int32(4)),
                            alignment=4,
                        )
                    if tx_i32 < fx.Int32(BM):
                        compact_route = fx.Int32(
                            llvm.load(
                                T.i32,
                                _gep3(
                                    lds_routes,
                                    tx_i32 * fx.Int32(4),
                                ),
                            )
                        )
                        row_valid = compact_route < num_routed
                        safe_route = row_valid.select(compact_route, fx.Int32(0))
                        token = safe_route // fx.Int32(routed_topk)
                        slot = safe_route - token * fx.Int32(routed_topk)
                        raw_route = token * fx.Int32(TOPK) + slot
                        weight = fx.Float32(
                            llvm.load(
                                T.f32,
                                _gep1(
                                    _global_base_ptr1(arg_topk_weights),
                                    raw_route * fx.Int32(4),
                                ),
                            )
                        )
                        weight = fx.Float32(
                            arith.select(
                                row_valid,
                                _raw(weight),
                                _raw(fx.Float32(0.0)),
                            )
                        )
                        out_token = row_valid.select(token, i32_ntok)
                        packed_token = out_token | (safe_route << fx.Int32(24))
                        row = route0 * fx.Int32(BM) + tx_i32
                        llvm.StoreOp(
                            _raw(packed_token),
                            _global_ptr1(
                                arg_sorted_token_ids,
                                row * fx.Int32(4),
                            ),
                            alignment=4,
                        )
                        llvm.StoreOp(
                            _raw(weight),
                            _global_ptr1(
                                arg_sorted_weights,
                                row * fx.Int32(4),
                            ),
                            alignment=4,
                        )
                elif tx_i32 == fx.Int32(0):
                    llvm.StoreOp(
                        _raw(fx.Int32(0)),
                        _global_ptr1(arg_counts, route0 * fx.Int32(4)),
                        alignment=4,
                    )

            if is_leader:
                for pass_idx in range_constexpr(n_blocks // dispatch_n_groups):
                    n_block = n_group + fx.Int32(pass_idx * dispatch_n_groups)
                    tile = route0 * fx.Int32(n_blocks) + n_block
                    _gemm1_body(
                        lds_raw_ptr,
                        arg_inter_q,
                        arg_inter_scale,
                        arg_w1,
                        arg_w1_scale,
                        arg_expert_ids,
                        arg_sorted_token_ids,
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
                        direct_mapped_rows=True,
                        mapped_output=True,
                        output_block_stride=1,
                        output_row_limit=num_routed,
                        local_route_map_ptr=lds_routes,
                        local_route_topk=routed_topk,
                    )
                    gpu.barrier()
        else:
            n_block = bx_i32 - routed_candidate_blocks
            shared_m_block = max_m_blocks - fx.Int32(1)
            tile = shared_m_block * fx.Int32(n_blocks) + n_block
            _gemm1_body(
                lds_raw_ptr,
                arg_inter_q,
                arg_inter_scale,
                arg_w1,
                arg_w1_scale,
                arg_expert_ids,
                arg_sorted_token_ids,
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

    @flyc.jit
    def launch_stage1(
        arg_hidden: fx.Int64,
        arg_w1: fx.Int64,
        arg_w1_scale: fx.Int64,
        arg_topk_ids: fx.Int64,
        arg_topk_weights: fx.Int64,
        arg_inter_q: fx.Int64,
        arg_inter_scale: fx.Int64,
        arg_sorted_token_ids: fx.Int64,
        arg_sorted_weights: fx.Int64,
        arg_expert_ids: fx.Int64,
        arg_counts: fx.Int64,
        arg_final_out: fx.Int64,
        i32_ntok: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = arith.index_cast(T.index, _raw(i32_ntok)) * fx.Index(
            routed_topk * dispatch_n_groups
        ) + fx.Index(n_blocks)
        stage1_kernel(
            arg_hidden,
            arg_w1,
            arg_w1_scale,
            arg_topk_ids,
            arg_topk_weights,
            arg_inter_q,
            arg_inter_scale,
            arg_sorted_token_ids,
            arg_sorted_weights,
            arg_expert_ids,
            arg_counts,
            arg_final_out,
            i32_ntok,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    launch_stage1.compile_hints["llvm_options"] = {"enable-post-misched": False}
    return launch_stage1


def compile_mxfp4_routed_compact_shared_stage2(
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
    dispatch_n_groups=0,
):
    """Gather private routed rows and run grouped routed/shared GEMM2."""
    if BM != 16 or BN != 256 or BK != 256:
        raise ValueError("routed-compact/shared Stage 2 requires BM=16 and BN=BK=256")
    routed_topk = TOPK - 1
    kh_tile = BK // 2
    n_blocks = num_n_blocks_for(D_HIDDEN, BN)
    if dispatch_n_groups == 0:
        dispatch_n_groups = n_blocks
    if dispatch_n_groups <= 0 or n_blocks % dispatch_n_groups != 0:
        raise ValueError(
            f"output N blocks ({n_blocks}) must be divisible by "
            f"dispatch_n_groups ({dispatch_n_groups})"
        )
    k_half = k_half_for(D_INTER)
    k_tiles = k_tiles_total_for(D_INTER, BK)
    a_stages = kStages if k_tiles <= kStages else 3
    slot_bytes = saq_slot_bytes(BM, kh_tile)
    lds_bytes = lds_acc_bytes_for(BM, BN) + a_stages * slot_bytes
    rnt_tag = "nt" if routed_use_nt else "cached"
    snt_tag = "nt" if shared_use_nt else "cached"
    name = (
        f"mxfp4_routed_compact_shared_g2_h{D_HIDDEN}_i{D_INTER}"
        f"_ne{NE}_tk{TOPK}_bm{BM}_r{rnt_tag}_s{snt_tag}"
        f"_ng{dispatch_n_groups}_private_v2"
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
        arg_topk_weights: fx.Int64,
        arg_sorted_token_ids: fx.Int64,
        arg_sorted_weights: fx.Int64,
        arg_expert_ids: fx.Int64,
        arg_counts: fx.Int64,
        arg_out: fx.Int64,
        i32_ntok: fx.Int32,
    ):
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        num_routed = i32_ntok * fx.Int32(routed_topk)
        routed_blocks = num_routed * fx.Int32(dispatch_n_groups)
        max_m_blocks = num_routed + fx.Int32(1)
        aq_records = fx.Int64(max_m_blocks) * fx.Int64(BM * k_half)
        aq_rsrc = _buffer_rsrc(arg_inter_q, aq_records)
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        saq_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
        n_load_waves, rows_per_wave, k_subblocks = tiling(BM)

        if bx_i32 < routed_blocks:
            route0 = bx_i32 // fx.Int32(dispatch_n_groups)
            n_group = bx_i32 - route0 * fx.Int32(dispatch_n_groups)
            count = fx.Int32(
                llvm.load(
                    T.i32,
                    _global_ptr1(arg_counts, route0 * fx.Int32(4)),
                )
            )
            if count > fx.Int32(0):
                m_row = route0 * fx.Int32(BM)
                if wave < fx.Int32(n_load_waves):
                    for k_slot in range(k_tiles):
                        for sub in range(k_subblocks):
                            lds_row = wave * fx.Int32(rows_per_wave) + fx.Int32(sub * 8)
                            row_in_block = lds_row + lane // fx.Int32(8)
                            packed_token = fx.Int32(
                                llvm.load(
                                    T.i32,
                                    _global_ptr1(
                                        arg_sorted_token_ids,
                                        (m_row + row_in_block) * fx.Int32(4),
                                    ),
                                    invariant=True,
                                )
                            )
                            private_block = packed_token.shrui(fx.Int32(24)) & fx.Int32(
                                0xFF
                            )
                            car = private_block * fx.Int32(BM)
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
                for pass_idx in range_constexpr(n_blocks // dispatch_n_groups):
                    n_block = n_group + fx.Int32(pass_idx * dispatch_n_groups)
                    tile = route0 * fx.Int32(n_blocks) + n_block
                    _gemm2_body(
                        lds_raw_ptr,
                        arg_inter_scale,
                        arg_w2,
                        arg_w2_scale,
                        arg_expert_ids,
                        arg_sorted_token_ids,
                        arg_sorted_weights,
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
                        private_route_scales=True,
                    )
                    gpu.barrier()
        else:
            n_group = bx_i32 - routed_blocks
            m_block = max_m_blocks - fx.Int32(1)
            m_row = m_block * fx.Int32(BM)
            if wave < fx.Int32(n_load_waves):
                for k_slot in range(k_tiles):
                    for sub in range(k_subblocks):
                        lds_row = wave * fx.Int32(rows_per_wave) + fx.Int32(sub * 8)
                        car = m_row + lds_row + lane // fx.Int32(8)
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
            for pass_idx in range_constexpr(n_blocks // dispatch_n_groups):
                n_block = n_group + fx.Int32(pass_idx * dispatch_n_groups)
                tile = m_block * fx.Int32(n_blocks) + n_block
                _gemm2_body(
                    lds_raw_ptr,
                    arg_inter_scale,
                    arg_w2,
                    arg_w2_scale,
                    arg_expert_ids,
                    arg_sorted_token_ids,
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
                    direct_sequential_weight_one=True,
                )
                gpu.barrier()

    @flyc.jit
    def launch_stage2(
        arg_inter_q: fx.Int64,
        arg_inter_scale: fx.Int64,
        arg_w2: fx.Int64,
        arg_w2_scale: fx.Int64,
        arg_topk_weights: fx.Int64,
        arg_sorted_token_ids: fx.Int64,
        arg_sorted_weights: fx.Int64,
        arg_expert_ids: fx.Int64,
        arg_counts: fx.Int64,
        arg_out: fx.Int64,
        i32_ntok: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = arith.index_cast(T.index, _raw(i32_ntok)) * fx.Index(
            routed_topk * dispatch_n_groups
        ) + fx.Index(dispatch_n_groups)
        stage2_kernel(
            arg_inter_q,
            arg_inter_scale,
            arg_w2,
            arg_w2_scale,
            arg_topk_weights,
            arg_sorted_token_ids,
            arg_sorted_weights,
            arg_expert_ids,
            arg_counts,
            arg_out,
            i32_ntok,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    launch_stage2.compile_hints["llvm_options"] = {"enable-post-misched": False}
    return launch_stage2
