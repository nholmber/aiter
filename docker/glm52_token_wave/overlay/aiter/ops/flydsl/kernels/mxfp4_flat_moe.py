# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Route-direct two-stage MXFP4 MoE kernels for small-token gfx950 decode."""

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


# BM16 owns one complete shuffled scale chunk, so adjacent routes can use
# consecutive 16-row private blocks without aliasing.
ROUTE_BLOCK_STRIDE = 1


def compile_mxfp4_flat_stage1(
    *,
    D_HIDDEN: int,
    D_INTER: int,
    NE: int,
    TOPK: int,
    BM: int = 16,
    BN: int = 256,
    BK: int = 256,
    use_nt: bool = True,
    interleave: bool = False,
):
    if BM != 16 or BN not in (64, 128, 256) or BK != 256:
        raise ValueError(
            "flat Stage 1 requires BM=16, BN in {64,128,256}, and BK=256"
        )
    if BN in (64, 128) and interleave:
        raise ValueError(
            "flat Stage-1 BN64/BN128 supports separated gate/up only"
        )

    kh_tile = BK // 2
    n_out = 2 * D_INTER
    n_blocks = num_n_blocks_for(n_out, BN)
    _, _, _, lds_bytes = _bm_constants(
        BM, BN, kh_tile, k_tiles_total_for(D_HIDDEN, BK)
    )
    gu_tag = "il" if interleave else "sep"
    nt_tag = "nt" if use_nt else "cached"
    bn_tag = "" if BN == 256 else f"_bn{BN}"
    name = (
        f"mxfp4_flat_g1_h{D_HIDDEN}_i{D_INTER}_ne{NE}_tk{TOPK}"
        f"_bm{BM}_{nt_tag}_{gu_tag}{bn_tag}_v1"
    )

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def flat_stage1_kernel(
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

        route = bx_i32 // fx.Int32(n_blocks)
        n_block = bx_i32 - route * fx.Int32(n_blocks)
        token = route // fx.Int32(TOPK)
        slot = route - token * fx.Int32(TOPK)
        expert = rocdl.readfirstlane(
            T.i32, _raw(_global_i32_at(arg_topk_ids, route))
        )
        private_m_block = route * fx.Int32(ROUTE_BLOCK_STRIDE)
        max_m_blocks = i32_ntok * fx.Int32(TOPK * ROUTE_BLOCK_STRIDE)

        # Stage 2 atomically accumulates into final_out. One Stage-1 workgroup
        # per token clears that row, and kernel completion is the global barrier.
        if (slot == fx.Int32(0)) & (n_block == fx.Int32(0)):
            out_base = _global_base_ptr1(arg_final_out)
            for i in range(D_HIDDEN // 2 // 256):
                dword = tx_i32 + fx.Int32(i * 256)
                byte_off = (
                    token * fx.Int32(D_HIDDEN * 2) + dword * fx.Int32(4)
                )
                llvm.StoreOp(_raw(fx.Int32(0)), _gep1(out_base, byte_off))

        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tile = private_m_block * fx.Int32(n_blocks) + n_block
        _gemm1_body(
            lds_raw_ptr,
            arg_inter_q,  # unread for inline quant
            arg_inter_scale,  # unread for inline quant
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
    def launch_flat_stage1(
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
        grid_x = arith.index_cast(T.index, _raw(i32_ntok)) * fx.Index(
            TOPK * n_blocks
        )
        flat_stage1_kernel(
            arg_hidden,
            arg_w1,
            arg_w1_scale,
            arg_topk_ids,
            arg_inter_q,
            arg_inter_scale,
            arg_final_out,
            i32_ntok,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    launch_flat_stage1.compile_hints["llvm_options"] = {"enable-post-misched": False}
    return launch_flat_stage1


def compile_mxfp4_flat_stage2(
    *,
    D_HIDDEN: int,
    D_INTER: int,
    NE: int,
    TOPK: int,
    BM: int = 16,
    BN: int = 256,
    BK: int = 256,
    use_nt: bool = False,
    dispatch_n_groups: int = 0,
):
    if BM != 16 or BN != 256 or BK != 256:
        raise ValueError("flat Stage 2 currently requires BM=16 and BN=BK=256")

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
    nt_tag = "nt" if use_nt else "cached"
    name = (
        f"mxfp4_flat_g2_h{D_HIDDEN}_i{D_INTER}_ne{NE}_tk{TOPK}"
        f"_bm{BM}_{nt_tag}_ng{dispatch_n_groups}_v1"
    )

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def flat_stage2_kernel(
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

        route = bx_i32 // fx.Int32(dispatch_n_groups)
        n_group = bx_i32 - route * fx.Int32(dispatch_n_groups)
        token = route // fx.Int32(TOPK)
        expert = rocdl.readfirstlane(
            T.i32, _raw(_global_i32_at(arg_topk_ids, route))
        )
        weight = fx.Float32(
            llvm.load(
                T.f32,
                _gep1(
                    _global_base_ptr1(arg_topk_weights),
                    route * fx.Int32(4),
                ),
                invariant=True,
            )
        )

        private_m_block = route * fx.Int32(ROUTE_BLOCK_STRIDE)
        max_m_blocks = i32_ntok * fx.Int32(TOPK * ROUTE_BLOCK_STRIDE)
        aq_records = fx.Int64(max_m_blocks) * fx.Int64(BM * k_half)
        aq_rsrc = _buffer_rsrc(arg_inter_q, aq_records)

        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        saq_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
        n_load_waves, rows_per_wave, k_subblocks = tiling(BM)
        m_row = private_m_block * fx.Int32(BM)
        if wave < fx.Int32(n_load_waves):
            for slot in range(k_tiles):
                for sub in range(k_subblocks):
                    lds_row = wave * fx.Int32(rows_per_wave) + fx.Int32(sub * 8)
                    car = m_row + lds_row + (lane // fx.Int32(8))
                    _issue_a_load_lds(
                        aq_rsrc,
                        saq_base_i32,
                        slot,
                        slot,
                        car,
                        lane,
                        slot_bytes,
                        lds_row,
                        KH_TILE=kh_tile,
                        k_half=k_half,
                    )
        rocdl.sched_barrier(0)

        for pass_idx in range_constexpr(n_blocks // dispatch_n_groups):
            n_block = (
                n_group + fx.Int32(pass_idx * dispatch_n_groups)
            )
            tile = private_m_block * fx.Int32(n_blocks) + n_block
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
                use_nt,
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
            gpu.barrier()

    @flyc.jit
    def launch_flat_stage2(
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
        grid_x = arith.index_cast(T.index, _raw(i32_ntok)) * fx.Index(
            TOPK * dispatch_n_groups
        )
        flat_stage2_kernel(
            arg_inter_q,
            arg_inter_scale,
            arg_w2,
            arg_w2_scale,
            arg_topk_ids,
            arg_topk_weights,
            arg_out,
            i32_ntok,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    launch_flat_stage2.compile_hints["llvm_options"] = {"enable-post-misched": False}
    return launch_flat_stage2
