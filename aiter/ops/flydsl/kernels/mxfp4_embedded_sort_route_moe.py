# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Embedded-sort MXFP4 MoE Stage 1 for GLM-5.2 M<=16."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .mxfp4_gemm1 import _bm_constants, _gemm1_body, _global_i32_at
from .mxfp4_gemm_common import (
    _gep1,
    _gep3,
    _global_base_ptr1,
    _lds_ptr3,
    _raw,
    k_tiles_total_for,
    num_n_blocks_for,
)

# BM16 owns one complete shuffled scale chunk, so adjacent routes do not need
# an empty guard block between them.
PRIVATE_BLOCK_STRIDE = 1


def compile_mxfp4_embedded_sort_stage1(
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    BM=16,
    BN=256,
    BK=256,
    use_nt=True,
    interleave=False,
    dispatch_n_groups=4,
):
    """Compile fused route compaction + BF16 quantization + GEMM1.

    One candidate workgroup is launched per ``(route, N-dispatch-group)``.
    Only the first route for an expert continues. It gathers all routes for
    that expert into an LDS list, runs GEMM1, and scatters each compacted row
    to that route's private FP4 intermediate block. Stage 2 can therefore be
    route-direct and requires no sorting metadata.
    """
    if BM != 16 or BN != 256 or BK != 256:
        raise ValueError(
            "embedded-sort Stage 1 requires BM=16 and BN=BK=256"
        )

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
    metadata_bytes = ((8 + BM * 4 + 15) // 16) * 16
    lds_bytes = body_lds_bytes + metadata_bytes
    gu_tag = "il" if interleave else "sep"
    nt_tag = "nt" if use_nt else "cached"
    name = (
        f"mxfp4_emsort_g1_h{D_HIDDEN}_i{D_INTER}_ne{NE}_tk{TOPK}"
        f"_bm{BM}_{nt_tag}_{gu_tag}_ng{dispatch_n_groups}_rs"
        f"{PRIVATE_BLOCK_STRIDE}_v1"
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
        num_routes = i32_ntok * fx.Int32(TOPK)
        route0 = bx_i32 // fx.Int32(dispatch_n_groups)
        n_group = bx_i32 - route0 * fx.Int32(dispatch_n_groups)
        expert = rocdl.readfirstlane(
            T.i32, _raw(_global_i32_at(arg_topk_ids, route0))
        )

        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
        metadata_base_i32 = lds_base_i32 + fx.Int32(body_lds_bytes)
        lds_flag = _lds_ptr3(metadata_base_i32, fx.Int32(0))
        lds_count = _lds_ptr3(metadata_base_i32, fx.Int32(4))

        # Stage 2 atomically accumulates into final_out. Distribute the clear
        # across the full Stage-1 grid; kernel completion is the global barrier.
        zero_dword0 = bx_i32 * fx.Int32(256) + tx_i32
        zero_stride = i32_ntok * fx.Int32(
            TOPK * dispatch_n_groups * 256
        )
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

        if tx_i32 == fx.Int32(0):
            llvm.StoreOp(_raw(fx.Int32(0)), lds_flag, alignment=4)
            llvm.StoreOp(_raw(fx.Int32(0)), lds_count, alignment=4)
        if tx_i32 < fx.Int32(BM):
            llvm.StoreOp(
                _raw(num_routes),
                _gep3(
                    lds_count, fx.Int32(4) + tx_i32 * fx.Int32(4)
                ),
                alignment=4,
            )
        gpu.barrier()

        # Any earlier route for this expert disqualifies the candidate.
        if tx_i32 < route0:
            prev_expert = _global_i32_at(arg_topk_ids, tx_i32)
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
            # Gather matching route indices. Atomic slot order may differ
            # between N groups, but every row is scattered by original route,
            # so no cross-workgroup ordering contract is needed.
            if tx_i32 < num_routes:
                route_expert = _global_i32_at(arg_topk_ids, tx_i32)
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
                                lds_count,
                                fx.Int32(4) + slot_i32 * fx.Int32(4),
                            ),
                            alignment=4,
                        )
            gpu.barrier()

            for pass_idx in range_constexpr(
                n_blocks // dispatch_n_groups
            ):
                n_block = n_group + fx.Int32(
                    pass_idx * dispatch_n_groups
                )
                tile = bx_i32 * fx.Int32(n_blocks) + n_block
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
                    use_nt,
                    i32_ntok,
                    num_routes * fx.Int32(dispatch_n_groups),
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
                    output_block_stride=PRIVATE_BLOCK_STRIDE,
                    output_row_limit=num_routes,
                    local_route_map_ptr=lds_count,
                    local_route_map_byte_offset=4,
                    local_route_topk=TOPK,
                )
                gpu.barrier()

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
            * fx.Index(TOPK * dispatch_n_groups)
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
