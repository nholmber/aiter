# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-launch route-direct MXFP4 MoE prototype for GLM-5.2 low-M decode."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.typing import T

from .mxfp4_gemm1 import _bm_constants, _gemm1_body, _global_i32_at
from .mxfp4_gemm2 import _gemm2_body, saq_slot_bytes
from .mxfp4_gemm_common import (
    _gep1,
    _gep3,
    _global_base_ptr1,
    _lds_ptr3,
    _raw,
    k_tiles_total_for,
    lds_acc_bytes_for,
    num_n_blocks_for,
)


def compile_mxfp4_flat_single_stage(
    *,
    D_HIDDEN: int,
    D_INTER: int,
    NE: int,
    TOPK: int,
    use_nt_g1: bool = True,
    use_nt_g2: bool = False,
):
    """Compile the correctness-first one-launch flat kernel.

    Each workgroup owns one ``(token, route, 128-wide intermediate chunk)``.
    G1 produces that chunk into private LDS, G2 consumes the chunk immediately
    and atomically accumulates its partial result over all hidden-output tiles.
    """
    BM = 16
    BN_G1 = 256
    BN_G2 = 256
    BK = 256
    CHUNK = 128
    N_CHUNKS = D_INTER // CHUNK
    WGS_PER_TOKEN = TOPK * N_CHUNKS
    if D_INTER != 512 or D_HIDDEN % BN_G2 != 0:
        raise ValueError(
            "single-stage prototype currently requires D_INTER=512 and "
            "D_HIDDEN divisible by 256"
        )

    kh_tile = BK // 2
    g1_n_blocks = num_n_blocks_for(2 * D_INTER, BN_G1)
    if g1_n_blocks != N_CHUNKS:
        raise AssertionError(
            f"G1 blocks {g1_n_blocks} must match chunks {N_CHUNKS}"
        )
    _, _, _, g1_lds_bytes = _bm_constants(
        BM,
        BN_G1,
        kh_tile,
        k_tiles_total_for(D_HIDDEN, BK),
    )
    g2_slot_bytes = saq_slot_bytes(BM, kh_tile)
    g2_lds_bytes = g2_slot_bytes + lds_acc_bytes_for(BM, BN_G2)
    flag_offset = g1_lds_bytes + g2_lds_bytes
    lds_bytes = flag_offset + 16
    out_n_blocks = num_n_blocks_for(D_HIDDEN, BN_G2)
    name = (
        f"mxfp4_flat_1stage_h{D_HIDDEN}_i{D_INTER}_ne{NE}_tk{TOPK}"
        f"_bm16_chunk128_v1"
    )

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def flat_single_stage_kernel(
        arg_hidden: fx.Int64,
        arg_w1: fx.Int64,
        arg_w1_scale: fx.Int64,
        arg_w2: fx.Int64,
        arg_w2_scale: fx.Int64,
        arg_topk_ids: fx.Int64,
        arg_topk_weights: fx.Int64,
        arg_inter_scale: fx.Int64,
        arg_sync: fx.Int64,
        arg_out: fx.Int64,
        i32_ntok: fx.Int32,
    ):
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))

        route_chunk = bx_i32
        route = route_chunk // fx.Int32(N_CHUNKS)
        chunk = route_chunk - route * fx.Int32(N_CHUNKS)
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

        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        lds_raw_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
        g2_raw_ptr = fx.add_offset(lds_raw_ptr, g1_lds_bytes)
        g2_base_i32 = lds_raw_i32 + fx.Int32(g1_lds_bytes)
        local_flag_ptr = _gep3(
            _lds_ptr3(lds_raw_i32, fx.Int32(0)),
            fx.Int32(flag_offset),
        )

        def _sync_elem(index):
            base = arith.index_cast(T.index, _raw(fx.Int64(arg_sync)))
            elem = arith.index_cast(
                T.index,
                _raw(token * fx.Int32(3) + fx.Int32(index)),
            )
            addr = fx.Index(base) + fx.Index(elem) * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
            return ptr._value if hasattr(ptr, "_value") else ptr

        def _atomic_add(index, value):
            return ArithValue(
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    _sync_elem(index),
                    _raw(fx.Int32(value)),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                ).result
            )

        is_leader = tx_i32 == fx.Int32(0)
        arrive_if = scf.IfOp(_raw(is_leader))
        with ir.InsertionPoint(arrive_if.then_block):
            arrival = _atomic_add(0, 1)
            is_last = arith.cmpi(
                CmpIPredicate.eq,
                _raw(arrival),
                _raw(fx.Int32(WGS_PER_TOKEN - 1)),
            )
            llvm.StoreOp(
                arith.select(
                    is_last,
                    _raw(fx.Int32(1)),
                    _raw(fx.Int32(0)),
                ),
                local_flag_ptr,
            )
            scf.YieldOp([])
        gpu.barrier()

        last_flag = fx.Int32(llvm.load(T.i32, local_flag_ptr))
        is_last_block = last_flag != fx.Int32(0)
        is_not_last_block = last_flag == fx.Int32(0)
        zero_if = scf.IfOp(_raw(is_last_block))
        with ir.InsertionPoint(zero_if.then_block):
            for i in range(D_HIDDEN // 2 // 256):
                dword = tx_i32 + fx.Int32(i * 256)
                byte_off = (
                    token * fx.Int32(D_HIDDEN * 2)
                    + dword * fx.Int32(4)
                )
                out_base = arith.index_cast(
                    T.index,
                    _raw(fx.Int64(arg_out)),
                )
                out_off = arith.index_cast(T.index, _raw(byte_off))
                out_addr = fx.Index(out_base) + fx.Index(out_off)
                out_ptr = buffer_ops.create_llvm_ptr(
                    out_addr,
                    address_space=1,
                )
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.xchg,
                    (
                        out_ptr._value
                        if hasattr(out_ptr, "_value")
                        else out_ptr
                    ),
                    _raw(fx.Int32(0)),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
            rocdl.s_waitcnt(0)
            gpu.barrier()
            publish_if = scf.IfOp(_raw(is_leader))
            with ir.InsertionPoint(publish_if.then_block):
                _atomic_add(1, 1)
                scf.YieldOp([])
            scf.YieldOp([])

        wait_if = scf.IfOp(_raw(is_leader & is_not_last_block))
        with ir.InsertionPoint(wait_if.then_block):
            initial = arith.constant(0, type=T.i32)
            wait = scf.WhileOp([T.i32], [initial])
            before = ir.Block.create_at_start(wait.before, [T.i32])
            after = ir.Block.create_at_start(wait.after, [T.i32])
            with ir.InsertionPoint(before):
                current = before.arguments[0]
                keep_waiting = arith.cmpi(
                    CmpIPredicate.eq,
                    current,
                    initial,
                )
                scf.ConditionOp(keep_waiting, [current])
            with ir.InsertionPoint(after):
                observed = _atomic_add(1, 0)
                scf.YieldOp([_raw(observed)])
            scf.YieldOp([])
        gpu.barrier()
        total_m_blocks = i32_ntok * fx.Int32(TOPK)
        g1_tile = route * fx.Int32(g1_n_blocks) + chunk
        _gemm1_body(
            lds_raw_ptr,
            arg_inter_scale,
            arg_inter_scale,
            arg_w1,
            arg_w1_scale,
            arg_topk_ids,
            arg_topk_ids,
            arg_inter_scale,
            arg_inter_scale,
            arg_hidden,
            g1_tile,
            lane,
            wave,
            use_nt_g1,
            i32_ntok,
            total_m_blocks,
            BM=BM,
            BN=BN_G1,
            BK=BK,
            inline_quant=True,
            K=D_HIDDEN,
            N_OUT=2 * D_INTER,
            NE=NE,
            interleave=False,
            direct_route=True,
            direct_expert=expert,
            direct_token=token,
            local_aq_base_i32=g2_base_i32,
            local_aq_row_bytes=kh_tile,
        )
        rocdl.s_waitcnt(0)
        gpu.barrier()

        for k_half in range_constexpr(N_CHUNKS):
            chunk_if = scf.IfOp(_raw(chunk == fx.Int32(k_half)))
            with ir.InsertionPoint(chunk_if.then_block):
                for n_block in range_constexpr(out_n_blocks):
                    g2_tile = (
                        route * fx.Int32(out_n_blocks)
                        + fx.Int32(n_block)
                    )
                    _gemm2_body(
                        g2_raw_ptr,
                        arg_inter_scale,
                        arg_w2,
                        arg_w2_scale,
                        arg_topk_ids,
                        arg_topk_ids,
                        arg_topk_weights,
                        i32_ntok,
                        total_m_blocks,
                        arg_out,
                        arg_inter_scale,
                        g2_tile,
                        lane,
                        wave,
                        BM,
                        use_nt_g2,
                        NE,
                        D_HIDDEN,
                        "atomic",
                        D_INTER=D_INTER,
                        D_INTER_REAL=D_INTER,
                        aStages=1,
                        BN=BN_G2,
                        BK=BK,
                        KH_TILE=kh_tile,
                        direct_route=True,
                        direct_expert=expert,
                        direct_token=token,
                        direct_weight=weight,
                        direct_k_half=k_half,
                    )
                    gpu.barrier()
                scf.YieldOp([])

        rocdl.s_waitcnt(0)
        gpu.barrier()
        finish_if = scf.IfOp(_raw(is_leader))
        with ir.InsertionPoint(finish_if.then_block):
            done = _atomic_add(2, 1)
            is_done = arith.cmpi(
                CmpIPredicate.eq,
                _raw(done),
                _raw(fx.Int32(WGS_PER_TOKEN - 1)),
            )
            reset_if = scf.IfOp(is_done)
            with ir.InsertionPoint(reset_if.then_block):
                _atomic_add(0, -WGS_PER_TOKEN)
                _atomic_add(1, -1)
                _atomic_add(2, -WGS_PER_TOKEN)
                rocdl.s_waitcnt(0)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_flat_single_stage(
        arg_hidden: fx.Int64,
        arg_w1: fx.Int64,
        arg_w1_scale: fx.Int64,
        arg_w2: fx.Int64,
        arg_w2_scale: fx.Int64,
        arg_topk_ids: fx.Int64,
        arg_topk_weights: fx.Int64,
        arg_inter_scale: fx.Int64,
        arg_sync: fx.Int64,
        arg_out: fx.Int64,
        i32_ntok: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = arith.index_cast(T.index, _raw(i32_ntok)) * fx.Index(
            WGS_PER_TOKEN
        )
        flat_single_stage_kernel(
            arg_hidden,
            arg_w1,
            arg_w1_scale,
            arg_w2,
            arg_w2_scale,
            arg_topk_ids,
            arg_topk_weights,
            arg_inter_scale,
            arg_sync,
            arg_out,
            i32_ntok,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    launch_flat_single_stage.compile_hints["llvm_options"] = {
        "enable-post-misched": False
    }
    return launch_flat_single_stage
