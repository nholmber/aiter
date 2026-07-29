# GLM-5.2 TP4 MXFP4 FlyDSL MoE

Benchmark and implementation notes for the GLM-5.2 TP4 MoE shape:

- Model dimension: 6144
- Intermediate dimension: 512
- Experts: 257
- Top-k: 9
- Activation: SiLU
- Input/weights: BF16 / MXFP4
- GPU: gfx950, 256 CUs
- Base AITER main commit: `4a1cc773f34cbfc74387259e51262556ee38edd0`

## Implemented paths

### Flat, M=1-4

Two GPU launches:

1. Route-direct GEMM1 with fused BF16-to-MXFP4 input quantization, SiLU
   activation, and FP4 intermediate quantization.
2. Route-direct GEMM2 with routed-weight atomic accumulation.

This path does not sort.

### Embedded sort, M=8-16

Two GPU launches:

1. A route-leader Stage 1 embeds expert compaction in LDS, performs fused
   BF16-to-MXFP4 input quantization, GEMM1, SiLU, and FP4 intermediate
   quantization. Compacted rows are scattered to route-private intermediate
   blocks.
2. Route-direct GEMM2 consumes the FP4 intermediate.

The Stage-1 N dispatch uses two groups for M up to 12 and four groups for
M=13-16.

## Forced latest-main `f16in` comparison

The upstream baseline was forced to:

- GEMM1: `flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt`
- GEMM2: `flydsl_mxmoe_g2_a4w4_16x256x256_atomic`

The baseline timing invokes, on every iteration:

1. `moe_sorting(..., block_size=16, accumulate=True, output_aux=True)`
2. `f16in` GEMM1
3. Atomic GEMM2

`AITER_MOE_SORT_BACKEND=auto` selected the adaptive MXFP4 auxiliary sort.
The sort emits sorted token/weight/expert arrays, cumsum, `m_indices`,
`reverse_sorted`, and zeros the BF16 atomic output buffer.

Each measurement used 10 warmups and 30 timed iterations in the same process.
The table delta is `(ours / main - 1)`.

### Normal routing

| M | Main `f16in` (us) | Ours (us) | Delta |
|---:|------------------:|----------:|------:|
| 1  | 24.352 | 30.626 | +25.76% |
| 2  | 27.713 | 31.918 | +15.17% |
| 4  | 36.910 | 41.224 | +11.69% |
| 8  | 69.626 | 59.043 | -15.20% |
| 12 | 79.414 | 73.924 | -6.91% |
| 16 | 85.191 | 99.326 | +16.59% |

### Shared-expert routing

One of the nine routes for every token targets expert 256.

| M | Main `f16in` (us) | Ours (us) | Delta |
|---:|------------------:|----------:|------:|
| 1  | 25.388 | 30.441 | +19.90% |
| 2  | 27.859 | 31.920 | +14.57% |
| 4  | 35.135 | 40.080 | +14.07% |
| 8  | 51.989 | 60.394 | +16.17% |
| 12 | 76.446 | 68.550 | -10.33% |
| 16 | 84.844 | 94.983 | +11.95% |

Cross-output normalized differences were approximately `9e-6` to `1.4e-5`.

### Isolated upstream sorting

The adaptive auxiliary sort measured:

- Normal routing: 2.66-3.24 us
- Shared-expert routing: 2.72-2.97 us

These isolated values are diagnostic. The full upstream timings above already
include the sorting launch.

## Why embedded sorting costs more than the isolated sort

The isolated adaptive sort is only about 3 us. Most overhead in the embedded
kernel comes from work duplicated inside GEMM workgroups:

1. The grid launches one candidate per route and N-dispatch group. Every
   candidate performs leader detection; nonleaders still execute global route
   reads and workgroup barriers.
2. Every leader scans the full route list and uses LDS atomics to build its
   local expert block. This work is repeated independently for each N group.
3. Fused input quantization is also repeated across N groups because each
   GEMM1 N tile owns a separate workgroup.
4. Most random-routing expert blocks contain only one or two live rows, but
   GEMM1 still executes a BM16 tile. The remaining rows are padding.
5. Route-private intermediates make Stage 2 simple and correct, but they give
   up cross-route expert reuse. At M=16, route-direct GEMM2 is a substantial
   part of the remaining gap.
6. Shared-expert M=8 creates a dense reusable expert block. Latest-main
   `f16in` benefits from that grouping, while the route-private Stage 2 still
   processes each route independently.

The extra occupancy from multiple N groups helps at normal M=8 and at M=12.
By M=16, memory traffic and repeated work dominate, so additional occupancy no
longer compensates for the embedded bookkeeping.

## Current dispatch conclusion

- M=1-4: latest-main `f16in` is faster than the new flat path.
- M=8 normal routing: embedded sort wins.
- M=8 shared-expert routing: latest-main `f16in` wins.
- M=12: embedded sort wins for both routing modes.
- M=16: latest-main `f16in` wins.

No production tuning CSV rows are enabled for the new kernels.
