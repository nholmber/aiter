# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Wave-native token-centric MXFP4 GEMM1 experiment for GLM-5.2."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
)
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .mxfp4_gemm1 import (
    _bm_constants,
    _gemm1_body,
    _global_i32_at,
    _inline_e8m0,
    _pkmax_u16,
    _silu_mul_batch,
)
from .mxfp4_gemm2 import (
    _gemm2_body,
    _issue_a_load_lds,
    saq_slot_bytes,
    tiling,
)
from .mxfp4_gemm_common import (
    _buffer_rsrc,
    _e8m0_from_amax,
    _fabs_f32,
    _gep1,
    _gep3,
    _global_base_ptr1,
    _inline_dpp_quad_amax,
    _lds_ptr3,
    _lds_swizzle_mask,
    _raw,
    _umax_i32,
    bq_bytes_for,
    bscale_bytes_for,
    k_half_for,
    k_tiles_total_for,
    kas_per_chunk_dw_for,
    kbs_per_expert_dw_for,
    kBS_stride_k0_dw,
    kbs_stride_n0_dw_for,
    kStages,
    lds_acc_bytes_for,
    num_n_blocks_for,
)


def compile_mxfp4_token_wave_shared_stage1(
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    BM=16,
    BN=64,
    BK=256,
    routed_use_nt=False,
    shared_use_nt=False,
    compact_a=True,
    shared_bn=64,
    role_batch_size=4,
):
    """Quantize one token once, then assign one routed expert to each wave."""
    if BM != 16 or BN not in (64, 128) or BK != 256:
        raise ValueError(
            "token-wave Stage 1 requires BM=16, BN in {64,128}, and BK=256"
        )
    if TOPK != 9:
        raise ValueError(
            "token-wave Stage 1 requires eight routed slots plus one shared slot"
        )
    if shared_bn not in (64, 128, 256):
        raise ValueError("token-wave shared BN must be 64, 128, or 256")

    routed_topk = TOPK - 1
    routes_per_phase = 4
    num_phases = routed_topk // routes_per_phase
    logical_chunks = BN // 32
    role_count = 2 * logical_chunks
    b_scale_groups = BN // 64
    output_lanes = BN // 16
    if role_batch_size not in (2, 4, 8):
        raise ValueError("token-wave role batch must be 2, 4, or 8")
    if role_count % role_batch_size != 0:
        raise ValueError(
            f"role count {role_count} must be divisible by batch {role_batch_size}"
        )
    kh_tile = BK // 2
    k_half = k_half_for(D_HIDDEN)
    k_tiles = k_tiles_total_for(D_HIDDEN, BK)
    n_out = 2 * D_INTER
    n_blocks = num_n_blocks_for(n_out, BN)
    shared_n_blocks = num_n_blocks_for(n_out, shared_bn)
    bq_bytes = bq_bytes_for(NE, n_out, D_HIDDEN)
    bscale_bytes = bscale_bytes_for(NE, n_out, D_HIDDEN)
    kbs_per_expert_dw = kbs_per_expert_dw_for(n_out, D_HIDDEN)
    kbs_stride_n0_dw = kbs_stride_n0_dw_for(D_HIDDEN)
    out_as_per_chunk_dw = kas_per_chunk_dw_for(D_INTER)

    # Keep the complete token A tile in LDS for this correctness-first body.
    # Only row zero is populated; the MFMA mapping ignores all other rows.
    a_tile_bytes = kh_tile if compact_a else BM * kh_tile
    a_bytes = k_tiles * a_tile_bytes
    a_scale_off = a_bytes
    a_scale_bytes = k_tiles * 256
    acc_off = a_scale_off + a_scale_bytes
    acc_bytes = routes_per_phase * role_count * 16 * 4
    wave_lds_bytes = acc_off + acc_bytes
    _, _, _, shared_body_lds_bytes = _bm_constants(
        BM, shared_bn, kh_tile, k_tiles
    )
    lds_bytes = max(wave_lds_bytes, shared_body_lds_bytes)

    rnt_tag = "nt" if routed_use_nt else "cached"
    snt_tag = "nt" if shared_use_nt else "cached"
    a_tag = "compacta" if compact_a else "fulla"
    name = (
        f"mxfp4_token_wave_shared_g1_h{D_HIDDEN}_i{D_INTER}"
        f"_ne{NE}_tk{TOPK}_bm{BM}_bn{BN}_r{rnt_tag}_s{snt_tag}"
        f"_{a_tag}_sbn{shared_bn}_rb{role_batch_size}_v3"
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
        routed_blocks = i32_ntok * fx.Int32(num_phases * n_blocks)
        max_m_blocks = num_routed + fx.Int32(1)

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
            slot = phase * fx.Int32(routes_per_phase) + wave
            raw_route = token * fx.Int32(TOPK) + slot
            expert = rocdl.readfirstlane(
                T.i32, _raw(_global_i32_at(arg_topk_ids, raw_route))
            )
            route = token * fx.Int32(routed_topk) + slot

            lds_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
            lds_base = _lds_ptr3(lds_base_i32, fx.Int32(0))
            hidden_base = _global_base_ptr1(arg_hidden)
            lib = lane & fx.Int32(3)
            lane_shr2_and3 = (lane >> fx.Int32(2)) & fx.Int32(3)

            # The four waves quantize disjoint K tiles. Lanes 0-15 in each
            # wave reproduce the proven row-zero inline-quant layout.
            for quant_pass in range_constexpr(k_tiles // routes_per_phase):
                kt = wave + fx.Int32(quant_pass * routes_per_phase)
                if lane < fx.Int32(16):
                    scale_accum = fx.Int32(0)
                    for b128_idx in range_constexpr(2):
                        hidden_byte = (
                            token * fx.Int32(D_HIDDEN * 2)
                            + kt * fx.Int32(BK * 2)
                            + fx.Int32(b128_idx * 256)
                            + lane_shr2_and3 * fx.Int32(64)
                            + lib * fx.Int32(16)
                        )
                        h_v = Vec(
                            llvm.load(
                                T.vec(4, T.i32),
                                _gep1(hidden_base, hidden_byte),
                                alignment=16,
                                invariant=True,
                            )
                        )
                        h_dw = [
                            fx.Int32(_raw(h_v[j]))
                            for j in range_constexpr(4)
                        ]
                        hm = [
                            h_dw[j] & fx.Int32(0x7FFF7FFF)
                            for j in range_constexpr(4)
                        ]
                        m01 = _pkmax_u16(hm[0], hm[1])
                        m23 = _pkmax_u16(hm[2], hm[3])
                        m0123 = _pkmax_u16(m01, m23)
                        lo = m0123 & fx.Int32(0xFFFF)
                        hi = (
                            m0123.shrui(fx.Int32(16))
                            & fx.Int32(0xFFFF)
                        )
                        local_amax = _umax_i32(lo, hi)
                        block_amax = _inline_dpp_quad_amax(local_amax)
                        e8 = _inline_e8m0(block_amax)
                        qscale = _raw(
                            fx.Float32(
                                _raw(e8 << fx.Int32(23)).bitcast(T.f32)
                            )
                        )
                        packed = _raw(fx.Int32(0))
                        for j in range_constexpr(4):
                            src_bf16x2 = _raw(
                                fx.Vector.from_elements(
                                    [h_dw[j]], fx.Int32
                                ).bitcast(fx.BFloat16)
                            )
                            packed = rocdl.cvt_scalef32_pk_fp4_bf16(
                                T.i32,
                                packed,
                                src_bf16x2,
                                qscale,
                                j,
                            )
                        kb_in_kt = (
                            fx.Int32(b128_idx * 4) + lane_shr2_and3
                        )
                        a_byte = (
                            kt * fx.Int32(a_tile_bytes)
                            + kb_in_kt * fx.Int32(16)
                            + lib * fx.Int32(4)
                        )
                        llvm.StoreOp(
                            packed,
                            _gep3(lds_base, a_byte),
                            alignment=4,
                        )
                        scale_accum = scale_accum | (
                            e8 << fx.Int32(b128_idx * 16)
                        )

                    if lib == fx.Int32(0):
                        scale_byte = (
                            fx.Int32(a_scale_off)
                            + kt * fx.Int32(256)
                            + lane_shr2_and3 * fx.Int32(64)
                        )
                        llvm.StoreOp(
                            _raw(scale_accum),
                            _gep3(lds_base, scale_byte),
                            alignment=4,
                        )
            gpu.barrier()

            bq_rsrc = _buffer_rsrc(arg_w1, fx.Index(bq_bytes))
            bscale_rsrc = _buffer_rsrc(
                arg_w1_scale, fx.Index(bscale_bytes)
            )
            b_aux = 2 if routed_use_nt else 0
            lane_div_16 = lane // fx.Int32(16)
            lane_mod_16 = lane % fx.Int32(16)
            logical_col = n_block * fx.Int32(BN // 2)
            b_cols = [
                fx.Int32(mw * D_INTER)
                + logical_col
                + fx.Int32(chunk * 16)
                for mw in range_constexpr(2)
                for chunk in range_constexpr(logical_chunks)
            ]
            b_load_s_base = [
                rocdl.readfirstlane(
                    T.i32,
                    (expert * fx.Int32(n_out) + col)
                    * fx.Int32(k_half),
                )
                for col in b_cols
            ]
            b_scale_s_base = [
                rocdl.readfirstlane(
                    T.i32,
                    (
                        expert * fx.Int32(kbs_per_expert_dw)
                        + (
                            n_block * fx.Int32(b_scale_groups)
                            + fx.Int32(scale_group)
                            + fx.Int32(mw * (n_out // 64))
                        )
                        * fx.Int32(kbs_stride_n0_dw)
                    )
                    * fx.Int32(4),
                )
                for mw in range_constexpr(2)
                for scale_group in range_constexpr(b_scale_groups)
            ]

            zero4 = Vec.filled(4, 0.0, fx.Float32)
            acc = [None] * role_count
            for kt in range_constexpr(k_tiles):
                mask = _lds_swizzle_mask(lane_mod_16)
                a_frag = [None, None]
                for half in range_constexpr(2):
                    if const_expr(compact_a):
                        a_byte = (
                            fx.Int32(kt * a_tile_bytes)
                            + lane_div_16 * fx.Int32(16)
                            + fx.Int32(half * 64)
                        )
                    else:
                        lds_col = (
                            lane_div_16 * fx.Int32(16)
                            + fx.Int32(half * 64)
                        ) ^ mask
                        a_byte = (
                            fx.Int32(kt * a_tile_bytes)
                            + lane_mod_16 * fx.Int32(kh_tile)
                            + lds_col
                        )
                    a_frag[half] = llvm.load(
                        T.vec(4, T.i32),
                        _gep3(lds_base, a_byte),
                        alignment=16,
                    )
                a_scale_byte = (
                    fx.Int32(a_scale_off + kt * 256)
                    + (
                        lane_div_16 * fx.Int32(16) + lane_mod_16
                    )
                    * fx.Int32(4)
                )
                a_scale = llvm.load(
                    T.i32,
                    _gep3(lds_base, a_scale_byte),
                    alignment=4,
                )

                v_b = (
                    lane_div_16 * fx.Int32(256)
                    + lane_mod_16 * fx.Int32(16)
                    + fx.Int32(kt * 2048)
                )
                v_scale = (
                    (
                        lane_div_16 * fx.Int32(16) + lane_mod_16
                    )
                    * fx.Int32(4)
                    + fx.Int32(kt * kBS_stride_k0_dw * 4)
                )
                b_scale = [
                    buffer_ops.buffer_load(
                        bscale_rsrc,
                        v_scale // fx.Int32(4),
                        vec_width=1,
                        dtype=T.i32,
                        soffset_bytes=b_scale_s_base[mw],
                    )
                    for mw in range_constexpr(2 * b_scale_groups)
                ]
                for role_batch in range_constexpr(
                    role_count // role_batch_size
                ):
                    b_frag = [
                        [None, None] for _ in range(role_batch_size)
                    ]
                    for local_role in range_constexpr(role_batch_size):
                        role = (
                            role_batch * role_batch_size + local_role
                        )
                        for half in range_constexpr(2):
                            frag = buffer_ops.buffer_load(
                                bq_rsrc,
                                (
                                    v_b + fx.Int32(half * 1024)
                                )
                                // fx.Int32(4),
                                vec_width=4,
                                dtype=T.i32,
                                cache_modifier=b_aux,
                                soffset_bytes=b_load_s_base[role],
                            )
                            b_frag[local_role][half] = Vec(frag)
                    rocdl.sched_barrier(0)

                    for local_role in range_constexpr(role_batch_size):
                        role = (
                            role_batch * role_batch_size + local_role
                        )
                        chunk = role % logical_chunks
                        mw = role // logical_chunks
                        in_b = chunk & 1
                        sb = b_scale[
                            mw * b_scale_groups + chunk // 2
                        ]
                        c = zero4 if kt == 0 else acc[role]
                        c = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.f32x4,
                            [
                                a_frag[0],
                                b_frag[local_role][0],
                                c,
                                4,
                                4,
                                0,
                                a_scale,
                                in_b,
                                sb,
                            ],
                        )
                        acc[role] = (
                            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                T.f32x4,
                                [
                                    a_frag[1],
                                    b_frag[local_role][1],
                                    c,
                                    4,
                                    4,
                                    2,
                                    a_scale,
                                    2 + in_b,
                                    sb,
                                ],
                            )
                        )

            # Preserve only row zero from this wave's MFMA results. Each quad
            # of four lanes quantizes one complete 32-value logical MX group.
            if lane_div_16 == fx.Int32(0):
                for role in range_constexpr(role_count):
                    acc_byte = (
                        fx.Int32(acc_off)
                        + wave * fx.Int32(role_count * 16 * 4)
                        + fx.Int32(role * 16 * 4)
                        + lane_mod_16 * fx.Int32(4)
                    )
                    llvm.StoreOp(
                        _raw(Vec(acc[role])[0]),
                        _gep3(lds_base, acc_byte),
                        alignment=4,
                    )
            gpu.barrier()

            if lane < fx.Int32(output_lanes):
                logical_start = lane * fx.Int32(8)
                gate_vals = [None] * 8
                up_vals = [None] * 8
                for elem in range_constexpr(8):
                    logical_idx = logical_start + fx.Int32(elem)
                    chunk = logical_idx // fx.Int32(16)
                    col = logical_idx - chunk * fx.Int32(16)
                    gate_byte = (
                        fx.Int32(acc_off)
                        + wave * fx.Int32(role_count * 16 * 4)
                        + chunk * fx.Int32(16 * 4)
                        + col * fx.Int32(4)
                    )
                    up_byte = gate_byte + fx.Int32(
                        logical_chunks * 16 * 4
                    )
                    gate_vals[elem] = fx.Float32(
                        llvm.load(
                            T.f32,
                            _gep3(lds_base, gate_byte),
                            alignment=4,
                        )
                    )
                    up_vals[elem] = fx.Float32(
                        llvm.load(
                            T.f32,
                            _gep3(lds_base, up_byte),
                            alignment=4,
                        )
                    )
                result = _silu_mul_batch(gate_vals, up_vals)
                local_max = _fabs_f32(result[0])
                for elem in range_constexpr(1, 8):
                    local_max = local_max.maximumf(
                        _fabs_f32(result[elem])
                    )
                amax_bits = _inline_dpp_quad_amax(
                    fx.Int32(_raw(local_max).bitcast(T.i32))
                )
                amax = fx.Float32(_raw(amax_bits).bitcast(T.f32))
                e8m0, qscale = _e8m0_from_amax(amax)
                packed = _raw(fx.Int32(0))
                for pair in range_constexpr(4):
                    packed = rocdl.cvt_scalef32_pk_fp4_f32(
                        T.i32,
                        packed,
                        _raw(result[2 * pair]),
                        _raw(result[2 * pair + 1]),
                        _raw(qscale),
                        pair,
                    )

                out_q_base = _global_base_ptr1(arg_inter_q)
                out_row = route * fx.Int32(BM)
                q_byte = (
                    out_row * fx.Int32(D_INTER // 2)
                    + n_block * fx.Int32(BN // 4)
                    + lane * fx.Int32(4)
                )
                llvm.StoreOp(
                    packed,
                    _gep1(out_q_base, q_byte),
                )

                if (lane & fx.Int32(3)) == fx.Int32(0):
                    local_scale_group = lane // fx.Int32(4)
                    scale_group = (
                        n_block * fx.Int32(b_scale_groups)
                        + local_scale_group
                    )
                    ku = scale_group >> fx.Int32(3)
                    ikxdl = (
                        scale_group >> fx.Int32(2)
                    ) & fx.Int32(1)
                    lane_grp = scale_group & fx.Int32(3)
                    scale_dword = (
                        route * fx.Int32(out_as_per_chunk_dw)
                        + ku * fx.Int32(64)
                        + lane_grp * fx.Int32(16)
                    )
                    scale_byte = (
                        scale_dword * fx.Int32(4)
                        + ikxdl * fx.Int32(2)
                    )
                    llvm.StoreOp(
                        arith.trunci(T.i8, _raw(e8m0)),
                        _gep1(
                            _global_base_ptr1(arg_inter_scale),
                            scale_byte,
                        ),
                    )
        else:
            n_block = bx_i32 - routed_blocks
            shared_m_block = num_routed
            tile = (
                shared_m_block * fx.Int32(shared_n_blocks) + n_block
            )
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
                BN=shared_bn,
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
            + fx.Index(shared_n_blocks)
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


def compile_mxfp4_token_wave_shared_stage2(
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
    role_batch_size=4,
):
    """Assign one route-private GEMM2 expert to each physical wave."""
    if BM != 16 or BN != 256 or BK != 256:
        raise ValueError(
            "token-wave Stage 2 requires BM=16 and BN=BK=256"
        )
    if TOPK != 9:
        raise ValueError(
            "token-wave Stage 2 requires eight routed slots plus one shared slot"
        )
    if role_batch_size not in (2, 4, 8):
        raise ValueError("token-wave Stage-2 role batch must be 2, 4, or 8")

    routed_topk = TOPK - 1
    routes_per_phase = 4
    num_phases = routed_topk // routes_per_phase
    role_count = BN // 16
    b_scale_groups = BN // 32
    if role_count % role_batch_size != 0:
        raise ValueError(
            f"Stage-2 role count {role_count} must be divisible by "
            f"batch {role_batch_size}"
        )

    kh_tile = BK // 2
    k_half = k_half_for(D_INTER)
    k_tiles = k_tiles_total_for(D_INTER, BK)
    n_blocks = num_n_blocks_for(D_HIDDEN, BN)
    bq_bytes = bq_bytes_for(NE, D_HIDDEN, D_INTER)
    bscale_bytes = bscale_bytes_for(NE, D_HIDDEN, D_INTER)
    kbs_per_expert_dw = kbs_per_expert_dw_for(
        D_HIDDEN, D_INTER
    )
    kbs_stride_n0_dw = kbs_stride_n0_dw_for(D_INTER)

    route_a_bytes = routes_per_phase * k_half
    route_acc_off = route_a_bytes
    route_acc_bytes = routes_per_phase * role_count * 16 * 4
    route_lds_bytes = route_acc_off + route_acc_bytes

    a_stages = kStages if k_tiles <= kStages else 3
    slot_bytes = saq_slot_bytes(BM, kh_tile)
    shared_lds_bytes = (
        lds_acc_bytes_for(BM, BN) + a_stages * slot_bytes
    )
    lds_bytes = max(route_lds_bytes, shared_lds_bytes)

    rnt_tag = "nt" if routed_use_nt else "cached"
    snt_tag = "nt" if shared_use_nt else "cached"
    name = (
        f"mxfp4_token_wave_shared_g2_h{D_HIDDEN}_i{D_INTER}"
        f"_ne{NE}_tk{TOPK}_bm{BM}_bn{BN}_r{rnt_tag}_s{snt_tag}"
        f"_rb{role_batch_size}_v1"
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
        routed_blocks = i32_ntok * fx.Int32(num_phases * n_blocks)
        max_m_blocks = num_routed + fx.Int32(1)
        aq_records = (
            fx.Int64(max_m_blocks) * fx.Int64(BM * k_half)
        )
        aq_rsrc = _buffer_rsrc(arg_inter_q, aq_records)

        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
        lds_base = _lds_ptr3(lds_base_i32, fx.Int32(0))

        if bx_i32 < routed_blocks:
            token_phase = bx_i32 // fx.Int32(n_blocks)
            n_block = bx_i32 - token_phase * fx.Int32(n_blocks)
            token = token_phase // fx.Int32(num_phases)
            phase = token_phase - token * fx.Int32(num_phases)
            slot = phase * fx.Int32(routes_per_phase) + wave
            raw_route = token * fx.Int32(TOPK) + slot
            expert = rocdl.readfirstlane(
                T.i32, _raw(_global_i32_at(arg_topk_ids, raw_route))
            )
            weight = fx.Float32(
                rocdl.readfirstlane(
                    T.f32,
                    llvm.load(
                        T.f32,
                        _gep1(
                            _global_base_ptr1(arg_topk_weights),
                            raw_route * fx.Int32(4),
                        ),
                        invariant=True,
                    ),
                )
            )
            route = token * fx.Int32(routed_topk) + slot

            # Load the one live FP4 row for this route into wave-private LDS.
            if lane < fx.Int32(16):
                q_byte = (
                    route * fx.Int32(BM * k_half)
                    + lane * fx.Int32(16)
                )
                q_v = llvm.load(
                    T.vec(4, T.i32),
                    _gep1(_global_base_ptr1(arg_inter_q), q_byte),
                    alignment=16,
                )
                lds_q_byte = (
                    wave * fx.Int32(k_half)
                    + lane * fx.Int32(16)
                )
                llvm.StoreOp(
                    q_v,
                    _gep3(lds_base, lds_q_byte),
                    alignment=16,
                )
            gpu.barrier()

            bq_rsrc = _buffer_rsrc(arg_w2, fx.Index(bq_bytes))
            bscale_rsrc = _buffer_rsrc(
                arg_w2_scale, fx.Index(bscale_bytes)
            )
            b_aux = 2 if routed_use_nt else 0
            lane_div_16 = lane // fx.Int32(16)
            lane_mod_16 = lane % fx.Int32(16)
            b_load_s_base = [
                rocdl.readfirstlane(
                    T.i32,
                    (
                        expert * fx.Int32(D_HIDDEN)
                        + n_block * fx.Int32(BN)
                        + fx.Int32(role * 16)
                    )
                    * fx.Int32(k_half),
                )
                for role in range_constexpr(role_count)
            ]
            b_scale_s_base = [
                rocdl.readfirstlane(
                    T.i32,
                    (
                        expert * fx.Int32(kbs_per_expert_dw)
                        + (
                            n_block * fx.Int32(b_scale_groups)
                            + fx.Int32(scale_group)
                        )
                        * fx.Int32(kbs_stride_n0_dw)
                    )
                    * fx.Int32(4),
                )
                for scale_group in range_constexpr(b_scale_groups)
            ]
            acc = [None] * role_count
            zero4 = Vec.filled(4, 0.0, fx.Float32)

            for kt in range_constexpr(k_tiles):
                a_frag = [None, None]
                for half in range_constexpr(2):
                    a_byte = (
                        wave * fx.Int32(k_half)
                        + fx.Int32(kt * kh_tile)
                        + lane_div_16 * fx.Int32(16)
                        + fx.Int32(half * 64)
                    )
                    a_frag[half] = llvm.load(
                        T.vec(4, T.i32),
                        _gep3(lds_base, a_byte),
                        alignment=16,
                    )
                a_scale_byte = (
                    route * fx.Int32(D_INTER)
                    + fx.Int32(kt * 256)
                    + (
                        lane_div_16 * fx.Int32(16) + lane_mod_16
                    )
                    * fx.Int32(4)
                )
                a_scale = llvm.load(
                    T.i32,
                    _gep1(
                        _global_base_ptr1(arg_inter_scale),
                        a_scale_byte,
                    ),
                    alignment=4,
                )

                v_b = (
                    lane_div_16 * fx.Int32(256)
                    + lane_mod_16 * fx.Int32(16)
                    + fx.Int32(kt * 2048)
                )
                v_scale = (
                    (
                        lane_div_16 * fx.Int32(16) + lane_mod_16
                    )
                    * fx.Int32(4)
                    + fx.Int32(kt * kBS_stride_k0_dw * 4)
                )
                b_scale = [
                    buffer_ops.buffer_load(
                        bscale_rsrc,
                        v_scale // fx.Int32(4),
                        vec_width=1,
                        dtype=T.i32,
                        soffset_bytes=b_scale_s_base[scale_group],
                    )
                    for scale_group in range_constexpr(
                        b_scale_groups
                    )
                ]

                for role_batch in range_constexpr(
                    role_count // role_batch_size
                ):
                    b_frag = [
                        [None, None] for _ in range(role_batch_size)
                    ]
                    for local_role in range_constexpr(role_batch_size):
                        role = (
                            role_batch * role_batch_size + local_role
                        )
                        for half in range_constexpr(2):
                            frag = buffer_ops.buffer_load(
                                bq_rsrc,
                                (
                                    v_b + fx.Int32(half * 1024)
                                )
                                // fx.Int32(4),
                                vec_width=4,
                                dtype=T.i32,
                                cache_modifier=b_aux,
                                soffset_bytes=b_load_s_base[role],
                            )
                            b_frag[local_role][half] = Vec(frag)
                    rocdl.sched_barrier(0)

                    for local_role in range_constexpr(role_batch_size):
                        role = (
                            role_batch * role_batch_size + local_role
                        )
                        in_b = role & 1
                        sb = b_scale[role // 2]
                        c = zero4 if kt == 0 else acc[role]
                        c = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.f32x4,
                            [
                                a_frag[0],
                                b_frag[local_role][0],
                                c,
                                4,
                                4,
                                0,
                                a_scale,
                                in_b,
                                sb,
                            ],
                        )
                        acc[role] = (
                            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                T.f32x4,
                                [
                                    a_frag[1],
                                    b_frag[local_role][1],
                                    c,
                                    4,
                                    4,
                                    2,
                                    a_scale,
                                    2 + in_b,
                                    sb,
                                ],
                            )
                        )

            if lane_div_16 == fx.Int32(0):
                for role in range_constexpr(role_count):
                    acc_byte = (
                        fx.Int32(route_acc_off)
                        + wave * fx.Int32(role_count * 16 * 4)
                        + fx.Int32(role * 16 * 4)
                        + lane_mod_16 * fx.Int32(4)
                    )
                    llvm.StoreOp(
                        _raw(Vec(acc[role])[0]),
                        _gep3(lds_base, acc_byte),
                        alignment=4,
                    )
            gpu.barrier()

            col_start = lane * fx.Int32(4)
            role = lane // fx.Int32(4)
            role_col = (lane & fx.Int32(3)) * fx.Int32(4)
            acc_byte = (
                fx.Int32(route_acc_off)
                + wave * fx.Int32(role_count * 16 * 4)
                + role * fx.Int32(16 * 4)
                + role_col * fx.Int32(4)
            )
            out_v = Vec(
                llvm.load(
                    T.vec(4, T.f32),
                    _gep3(lds_base, acc_byte),
                    alignment=16,
                )
            )
            out_base = _global_base_ptr1(arg_out)
            out_col = n_block * fx.Int32(BN) + col_start
            for pair in range_constexpr(2):
                weighted = Vec.from_elements(
                    [
                        out_v[2 * pair] * weight,
                        out_v[2 * pair + 1] * weight,
                    ],
                    fx.Float32,
                ).to(fx.BFloat16)
                out_byte = (
                    token * fx.Int32(D_HIDDEN * 2)
                    + (out_col + fx.Int32(pair * 2)) * fx.Int32(2)
                )
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    _gep1(out_base, out_byte),
                    _raw(weighted),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )
        else:
            n_block = bx_i32 - routed_blocks
            m_block = num_routed
            m_row = m_block * fx.Int32(BM)
            n_load_waves, rows_per_wave, k_subblocks = tiling(BM)
            if wave < fx.Int32(n_load_waves):
                for k_slot in range_constexpr(k_tiles):
                    for sub in range_constexpr(k_subblocks):
                        lds_row = (
                            wave * fx.Int32(rows_per_wave)
                            + fx.Int32(sub * 8)
                        )
                        car = m_row + lds_row + (lane // fx.Int32(8))
                        _issue_a_load_lds(
                            aq_rsrc,
                            lds_base_i32,
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
                direct_sequential_weight_one=True,
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
            * fx.Index(num_phases * n_blocks)
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
