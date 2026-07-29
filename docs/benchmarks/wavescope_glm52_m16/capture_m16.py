import ctypes
import os

import torch

# Repro harness for WaveScope/rocprofv3 ATT captures. Run from the repository
# root with PYTHONPATH set to the checkout. ATT_MODE selects the token-wave or
# forced f16in two-stage pipeline; rocprof's kernel filter chooses GEMM1/GEMM2.
import aiter
from aiter import dtypes
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1
from aiter.ops.flydsl.mxfp4_gemm2_kernels import flydsl_mxfp4_gemm2
from aiter.ops.flydsl.mxfp4_token_wave_shared_moe_kernels import (
    flydsl_mxfp4_token_wave_shared_moe,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils


M, H, INTER, E, TOPK = 16, 6144, 512, 257, 9
MODE = os.environ.get("ATT_MODE", "ours")
WARMUPS = int(os.environ.get("ATT_WARMUPS", "3"))

roctx = ctypes.CDLL("/opt/rocm/lib/librocprofiler-sdk-roctx.so")
roctx.roctxProfilerPause.argtypes = [ctypes.c_uint64]
roctx.roctxProfilerResume.argtypes = [ctypes.c_uint64]
roctx.roctxProfilerPause.restype = ctypes.c_int
roctx.roctxProfilerResume.restype = ctypes.c_int

torch.set_default_device("cuda")
torch.manual_seed(4)

x = torch.randn((M, H), dtype=torch.bfloat16)
w1 = torch.randn((E, 2 * INTER, H), dtype=torch.bfloat16)
w2 = torch.randn((E, H, INTER), dtype=torch.bfloat16)
score = torch.randn((M, E - 1), dtype=torch.bfloat16)
routed_weights, routed_ids = fused_topk(x, score, TOPK - 1, True)
topk_ids = torch.cat(
    (routed_ids, torch.full((M, 1), E - 1, dtype=torch.int32)),
    dim=1,
).contiguous()
topk_weights = torch.cat(
    (routed_weights, torch.ones((M, 1), dtype=routed_weights.dtype)),
    dim=1,
).contiguous()

quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
w1_q, w1_s = quant(w1, quant_dtype=dtypes.fp4x2)
w2_q, w2_s = quant(w2, quant_dtype=dtypes.fp4x2)
w1_q = w1_q.view(E, 2 * INTER, H // 2)
w2_q = w2_q.view(E, H, INTER // 2)
w1_kernel = shuffle_weight(w1_q, layout=(16, 16))
w2_kernel = shuffle_weight(w2_q, layout=(16, 16))
w1_s_kernel = fp4_utils.e8m0_shuffle(w1_s).view(torch.uint8)
w2_s_kernel = fp4_utils.e8m0_shuffle(w2_s).view(torch.uint8)

workspace_blocks = M * (TOPK - 1) + 1
workspace_q = torch.empty(
    (workspace_blocks * 16, INTER // 2), dtype=torch.uint8
)
workspace_scale = torch.empty(
    (workspace_blocks * INTER,), dtype=torch.uint8
)
workspace_out = torch.empty((M, H), dtype=torch.bfloat16)
metadata_rows = M * (TOPK - 1) * 16
metadata_blocks = M * (TOPK - 1)
sorted_token_ids_workspace = torch.empty(
    (metadata_rows,), dtype=torch.int32
)
sorted_weights_workspace = torch.empty(
    (metadata_rows,), dtype=torch.float32
)
expert_ids_workspace = torch.empty(
    (metadata_blocks,), dtype=torch.int32
)
counts_workspace = torch.empty(
    (metadata_blocks,), dtype=torch.int32
)

main_blocks = M * TOPK
main_inter_q = torch.empty(
    (main_blocks * 16, INTER // 2), dtype=torch.uint8
)
main_inter_scale = torch.empty(
    (main_blocks * INTER,), dtype=torch.uint8
)
dummy_q = torch.empty((1,), dtype=torch.uint8)
dummy_s = torch.empty((1,), dtype=torch.uint8)


def run_ours():
    return flydsl_mxfp4_token_wave_shared_moe(
        hidden_states=x,
        w1=w1_kernel,
        w1_scale=w1_s_kernel,
        w2=w2_kernel,
        w2_scale=w2_s_kernel,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        out=workspace_out,
        inter_q=workspace_q,
        inter_scale=workspace_scale,
        sorted_token_ids=sorted_token_ids_workspace,
        sorted_weights=sorted_weights_workspace,
        expert_ids=expert_ids_workspace,
        counts=counts_workspace,
    )


def run_f16in():
    (
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        main_out,
        m_indices,
        _reverse_sorted,
    ) = moe_sorting(
        topk_ids,
        topk_weights,
        E,
        H,
        torch.bfloat16,
        16,
        accumulate=True,
        output_aux=True,
    )
    flydsl_mxfp4_gemm1(
        a_quant=dummy_q,
        a_scale_sorted_shuffled=dummy_s,
        w1_u8=w1_kernel,
        w1_scale_u8=w1_s_kernel,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=num_valid_ids,
        m_indices=m_indices,
        inter_sorted_quant=main_inter_q,
        inter_sorted_shuffled_scale=main_inter_scale,
        hidden_states=x,
        n_tokens=M,
        BM=16,
        use_nt=True,
        inline_quant=True,
        NE=E,
        D_HIDDEN=H,
        D_INTER=INTER,
        topk=TOPK,
        BN=256,
        BK=256,
    )
    flydsl_mxfp4_gemm2(
        inter_sorted_quant=main_inter_q,
        inter_sorted_shuffled_scale=main_inter_scale,
        w2_u8=w2_kernel,
        w2_scale_u8=w2_s_kernel,
        sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=num_valid_ids,
        sorted_token_ids=sorted_ids,
        sorted_weights=sorted_weights,
        flat_out=main_out,
        M_logical=M,
        max_sorted=sorted_ids.numel(),
        BM=16,
        use_nt=False,
        atomic=True,
        mxfp4out=False,
        NE=E,
        D_HIDDEN=H,
        D_INTER=INTER,
        topk=TOPK,
        BN=256,
        BK=256,
    )
    return main_out


run = run_ours if MODE == "ours" else run_f16in
for _ in range(WARMUPS):
    run()
torch.cuda.synchronize()

resume_status = roctx.roctxProfilerResume(0)
run()
torch.cuda.synchronize()
pause_status = roctx.roctxProfilerPause(0)
print(
    f"ATT capture completed for mode={MODE}, "
    f"resume={resume_status}, pause={pause_status}"
)
