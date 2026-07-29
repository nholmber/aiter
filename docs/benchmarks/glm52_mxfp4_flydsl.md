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

## Next experiment: deterministic shared-expert hybrid

GLM-5.2 routes the shared expert deterministically in the final top-k slot:

- Expert count: 257
- Shared expert ID: 256 (zero-based)
- Top-k: 9
- Shared slot: 8 (zero-based)
- Shared routed weight: exactly 1.0

This permits a specialized two-launch path with no general sorting:

1. The first eight routes remain sparse and route-direct.
2. The final route is removed from the sparse grid and processed as one
   grouped BM16 shared-expert block containing all M tokens.
3. Stage 2 similarly uses route-direct workgroups for the first eight routes
   and one grouped shared-expert block.
4. The shared Stage-2 epilogue specializes the routed weight to 1.0 and skips
   shared-weight loads and multiplies.

For M=8 and GEMM1 BN256, the intended Stage-1 grid is:

- Routed: `8 tokens * 8 routes * 4 N blocks = 256` workgroups
- Shared: `4 N blocks`
- Total: 260 workgroups

This should provide approximately one routed workgroup per CU without route
scans, leader detection, LDS sorting atomics, or duplicated shared-expert
GEMM2 work.

The primary hypotheses are:

- Shared-expert M=8 should recover the largest current regression.
- M=12/16 should benefit from eliminating general embedded-sort bookkeeping.
- M=1/2/4 will still require finer GEMM1 N tiling:
  - M=1: BN32 with K-wave4
  - M=2: BN64 with K-wave2
  - M=4: BN128 with K-wave1
  - M>=8: BN256 with K-wave1

The first implementation will keep BN256 and validate the hybrid dispatch
before introducing the smaller-N/K-wave variants.

## Deterministic shared-expert hybrid results

The BN256 hybrid was implemented with:

- Routed slots 0-7 dispatched route-directly.
- Shared slot 8 grouped into one BM16 block.
- Shared expert ID fixed to 256.
- Shared routed weight specialized to 1.0.
- Routed Stage-1 weights non-temporal.
- Shared Stage-1 and all Stage-2 weights cached.
- Routed workgroups scheduled before the four shared Stage-1 workgroups.

The table compares the forced main `f16in` pipeline, the previously selected
flat/embedded-sort path, and the shared hybrid in one process with identical
weights and shared-expert routes.

| M | Main `f16in` (us) | Previous (us) | Hybrid (us) | Hybrid vs main |
|---:|------------------:|--------------:|------------:|---------------:|
| 1  | 24.553 | 30.309 | 30.542 | +24.39% |
| 2  | 28.025 | 32.099 | 32.942 | +17.54% |
| 4  | 33.985 | 39.077 | 40.604 | +19.48% |
| 8  | 51.516 | 56.448 | 53.963 | +4.75% |
| 12 | 77.780 | 73.425 | 78.247 | +0.60% |
| 16 | 86.321 | 96.486 | 100.023 | +15.87% |

The hybrid improves the prior shared M=8 path by about 4.4%, but remains about
4.8% behind forced main `f16in`. It should not replace embedded sort at M=12
or M=16, where compacting duplicate routed experts is more valuable than only
special-casing the shared expert.

### M=8 stage breakdown

With the shared-weight=1 specialization:

- Stage 1: approximately 35.39 us
- Stage 2: approximately 16.24 us
- Two launches together: approximately 53.08 us

### Rejected hybrid variants

- Shared Stage-1 non-temporal loads regressed by roughly 5 us.
- Making all Stage-1 loads cached regressed by roughly 3.5 us.
- Non-temporal Stage-2 loads regressed.
- Scheduling shared workgroups before routed workgroups regressed the M=8
  total to roughly 58.4 us. The heavier shared blocks delayed completion of
  the routed wave; routed-first ordering restored approximately 53 us.

### Hybrid interpretation

The remaining M=8 gap is most likely routed-expert duplication. Main sorting
groups the approximately six-to-eight duplicate experts among the 64 routed
slots. The hybrid deliberately leaves those routes independent to avoid
general sorting overhead. Closing the final gap therefore requires either:

- A cheaper routed-only deduplication mechanism, or
- Smaller-N sparse GEMM1 tiles that make the route-direct work cheaper.

The next higher-leverage work remains the sparse N-tiling sequence:
BN128 for M=4, BN64/K-wave2 for M=2, and BN32/K-wave4 for M=1.

## BN128 sparse GEMM1 results

GEMM1 was generalized from BN256 to BN128 for separated gate/up weights.
GEMM2 remains BN256.

The BN128 implementation uses:

- Eight GEMM1 N blocks instead of four.
- Two 16-column MFMA J tiles per wave.
- Wave parity to select the low/high half of each 32-row weight/scale pack.
- An explicit `scf.if` with separate even/odd-wave MFMA instruction sequences,
  because `opselB` must be a compile-time immediate.
- BN-independent FP4 output and e8m0 scale addressing.

The first attempt passed wave parity as a runtime `opselB` and failed during
FlyDSL lowering. A Python `if` also failed to carry the accumulator SSA value
out of the branch. The explicit result-producing `scf.IfOp` is required.

### Normal routing

Same-process forced-main comparison:

| M | Main `f16in` (us) | Flat BN128 (us) | Delta |
|---:|------------------:|----------------:|------:|
| 1 | 24.427 | 19.703 | -19.34% |
| 2 | 28.214 | 22.935 | -18.72% |
| 4 | 36.415 | 33.528 | -7.93% |

### Shared-expert routing

| M | Main `f16in` (us) | Flat BN128 (us) | Hybrid BN128 (us) | Selected |
|---:|------------------:|----------------:|------------------:|:---------|
| 1 | 24.554 | 19.838 | 20.604 | Flat |
| 2 | 28.069 | 21.871 | 22.990 | Flat |
| 4 | 35.371 | 33.108 | 32.795 | Hybrid |

Normalized differences remained approximately `5e-6` to `1.3e-5`.

### Automatic low-M dispatch

- Flat M=1-4 defaults to GEMM1 BN128.
- Shared hybrid M=1-4 defaults to GEMM1 BN128.
- Shared production selection:
  - M=1-2: flat BN128
  - M=4: shared hybrid BN128
  - M=8: shared hybrid BN256
  - M=12-16: embedded sort BN256

BN128 already beats forced main at M=1/2/4, so BN64/K-wave2 and
BN32/K-wave4 are now optional follow-up optimizations rather than blockers.
