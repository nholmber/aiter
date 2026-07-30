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

### BN128 above M=4

Shared-hybrid BN128 was also tested at larger token counts:

| M | Hybrid BN128 (us) | Hybrid BN256 (us) | Selected path |
|---:|------------------:|------------------:|:--------------|
| 8 | 53.13 | 53.24 | Effectively tied; BN256 retained |
| 12 | 77.15 | 76.35 | BN256 |
| 16 | 98.82 | 103.14 | Embedded sort remains faster |

BN128 improves the hybrid at M=16 but does not beat the existing embedded-sort
path, and it provides no material M=8 benefit. Automatic BN128 selection
therefore remains limited to M<=4.

## BN64 and BN32 completeness check

BN64 was implemented for separated gate/up weights without K-wave splitting.
The four waves take fixed roles:

- Wave 0: gate columns 0-15
- Wave 1: up columns 0-15
- Wave 2: gate columns 16-31
- Wave 3: up columns 16-31

As with BN128, an explicit result-producing `scf.IfOp` selects the required
low/high 16-row MFMA `opselB` half.

### Shared-expert results

| M | Flat BN128 (us) | Flat BN64 (us) | Hybrid BN128 (us) | Hybrid BN64 (us) |
|---:|----------------:|---------------:|------------------:|-----------------:|
| 1 | 19.84 | 19.89 | 20.60 | 21.82 |
| 2 | 21.87 | 25.63 | 22.99 | 24.67 |

BN64 is effectively tied with BN128 for flat M=1 and clearly regresses M=2.
It is retained as a correctness-tested variant but is not selected
automatically.

### Why BN32 was not emitted as an independent workgroup tile

A raw BN32 gate/up tile produces only 16 logical SiLU-times-up values. MXFP4
requires one e8m0 scale for a complete group of 32 values. Two independent
BN32 workgroups cannot form that scale without cross-workgroup communication
or a subsequent reduction.

Pairing two BN32 subtiles inside one workgroup restores a complete 32-value
scale group, but that is equivalent to the tested BN64 workgroup shape.
Therefore BN32 was not retained as a separate kernel variant.

The selected low-M GEMM1 tile remains BN128 for M=1/2/4.

## Next M=8 experiment: one-wave routed deduplication

At shared M=8, the first eight routed slots contain exactly 64 route entries.
They fit in one CDNA wave. A specialized Stage-1 candidate workgroup can:

1. Load all 64 routed expert IDs with wave 0.
2. Ballot matches against the candidate expert.
3. Determine the leader from the lowest matching route.
4. Use `mbcnt` to assign deterministic compact slots.
5. Publish at most eight matching route IDs to an LDS list.
6. Run GEMM1 only for the leader candidate.

The other three waves wait at one workgroup barrier and then participate in
GEMM1. This removes the full-route scans and LDS sorting atomics used by the
general embedded-sort kernel while retaining duplicate routed-expert
compaction. The deterministic shared expert remains a separate grouped block.

This path is only intended for M<=8, where `(TOPK-1) * M <= 64`.

### One-wave dedup result

The wave-ballot implementation was numerically correct at shared M=8, but
regressed latency:

- Hybrid without routed dedup: approximately 52.5 us
- Hybrid with one-wave routed dedup: approximately 54.7 us
- Removing the route-list initialization barrier still measured approximately
  54.9 us and did not recover the regression.

The roughly seven duplicate routed expert tiles saved per dispatch did not
repay even one workgroup barrier plus ballot/list setup executed by all 256
routed candidate workgroups. The implementation is retained behind an explicit
option for experimentation, but routed deduplication is disabled by default.

## Next M=16 experiment: routed embedded sort plus grouped shared expert

The M=16 target should combine the two successful ideas instead of choosing
between them:

1. Apply embedded expert compaction only to routed slots 0-7.
2. Exclude deterministic shared slot 8 from candidate leader detection and
   route scans.
3. Process the shared expert as one grouped BM16 block in Stage 1.
4. Use route-direct Stage 2 for the first eight routes and one grouped
   shared-expert Stage-2 block with weight fixed to 1.0.

This preserves duplicate routed-expert reuse, which matters at M=12/16, while
removing duplicated shared-expert GEMM2 work and all shared-route sorting
bookkeeping.

For M=16, Stage 1 uses four N-dispatch groups:

- Routed candidates: `16 * 8 * 4 = 512` workgroups
- Shared grouped work: `4` workgroups

The existing shared-hybrid Stage 2 can consume the resulting layout directly:
`16 * 8 + 1` BM16 intermediate blocks.

### Combined-path results

| M | Main `f16in` (us) | Previous embedded sort (us) | Combined (us) |
|---:|------------------:|----------------------------:|--------------:|
| 12 | 77.78 | 73.43 | 68.24 |
| 16 | 86.32 | 96.49 | 93.47-94.03 |

The combined path is a clear M=12 win: about 12.3% faster than main and 7.1%
faster than the previous embedded-sort path. At M=16 it improves the in-tree
path by roughly 2.5-3 us, but remains about 8-9% behind main.

### M=16 dispatch and cache tuning

Routed Stage-1 N-dispatch groups:

- One group: 107.74 us
- Two groups: 93.47 us
- Four groups: 94.56 us

Two groups are selected for both M=12 and M=16.

Cache-policy experiments around the two-group recipe:

- Routed Stage 1 cached: approximately 106.50 us
- Shared Stage 1 non-temporal: approximately 98.83 us
- Routed Stage 2 non-temporal: approximately 96.65 us
- Shared Stage 2 non-temporal: approximately 95.60 us
- Both Stage-2 paths non-temporal: approximately 97.46 us

The selected policy remains routed Stage 1 non-temporal, shared Stage 1
cached, and both Stage-2 paths cached.

### M=16 stage breakdown

For the selected BN256/two-group combined path:

- Stage 1: approximately 63.05 us
- Stage 2: approximately 30.40 us

The remaining gap to main is now primarily Stage 2. Main sorting reduces the
128 routed rows to roughly 100-105 unique routed expert blocks, while the
combined route-direct Stage 2 still launches all 128 routed rows.

### Combined BN128 check

At M=16:

- BN256 with two routed N groups: approximately 93.47 us
- BN128 with two routed N groups: approximately 94.27 us
- BN128 with four routed N groups: approximately 93.07 us

BN128/four groups is marginally faster in one run, but the difference is
small enough to treat as a tie. BN256/two groups remains the simpler selected
configuration.

The next structural experiment is grouped routed Stage 2 that gathers
route-private FP4 rows and scales for duplicate routed experts, while retaining
the grouped deterministic shared-expert block.

## Sparse contiguous routed blocks plus grouped shared Stage 2

A second combined path makes routed row ordering deterministic in Stage 1 and
writes each leader expert to its sparse candidate BM16 block. Stage 1 publishes
count, expert, token, and weight metadata once. Stage 2 checks the count before
loading A, skips nonleaders, and runs the standard grouped routed epilogue.
The shared expert remains the final grouped block with weight fixed to 1.0.

### Results

| M | Main `f16in` (us) | Prior combined (us) | Sparse-grouped selected (us) |
|---:|------------------:|--------------------:|-----------------------------:|
| 12 | 77.78 | 68.24 | 67.84 |
| 16 | 86.32 | 93.47 | 91.21 |

M=12 uses Stage-1 BN256 with two N groups. M=16 uses Stage-1 BN128 with
four N groups.

At M=16, the initial BN256/four-group result was approximately 92.20 us.
BN128/four groups improved it to approximately 91.21 us. BN128 does not help
M=12:

- BN128/two groups: approximately 82.47 us
- BN128/four groups: approximately 71.84 us
- BN256/two groups: approximately 67.84 us

### M=16 stage profile

For BN256/four groups before the final BN128 switch:

- Stage 1: approximately 62.31 us
- Stage 2: approximately 28.11 us

Grouped routed Stage 2 reduces Stage-2 time by about 2.3 us versus the
route-direct combined path. The remaining gap is Stage 1 and the dynamic
multi-row routed epilogue overhead.

### Selected M=12/16 shared dispatch

- M=12: sparse-grouped, Stage-1 BN256, two N groups
- M=16: sparse-grouped, Stage-1 BN128, four N groups

### Stage-2 N-grouping result

Reusing one loaded A block across multiple output N tiles did not improve
M=16:

| Stage-2 N groups | Total latency (us) |
|-----------------:|-------------------:|
| 24 | 91.23 |
| 12 | 92.32 |
| 8 | 91.77 |
| 6 | 92.30 |
| 4 | 94.54 |

The saved A loads and count checks do not compensate for reduced N parallelism
and the larger serial body. The default remains one workgroup per output
N tile.

## Next structural M=16 idea: persistent fused-sort Stage 1

Candidate-local compaction is now the dominant Stage-1 overhead. A persistent
Stage-1 kernel could embed sorting once:

1. Launch at most one resident workgroup per CU.
2. One designated workgroup initializes workspace and builds routed expert
   metadata once.
3. Publish a ready flag after a device-scope fence.
4. Other resident workgroups wait for readiness.
5. All workgroups pull compact `(expert block, N tile)` tasks from a global
   atomic work queue.
6. Keep the deterministic shared expert as explicit grouped work.

This would approximate the separate adaptive-sort plus compact GEMM pipeline
while retaining two external launches. It also creates an opportunity to
quantize each token once into shared/global workspace instead of repeating
inline quantization across every routed expert/N tile.

The main risks are software grid synchronization, workspace epoch/reset
handling, and ensuring the designated sorter workgroup is resident to avoid
deadlock.

### Persistent block-0 sorter prototype result

A prototype was implemented with:

- An epoch-tagged global ready flag.
- Block 0 performing the complete routed sort.
- A fixed persistent worker grid.
- A global atomic GEMM1 task queue.
- Dense routed blocks consumed by the sparse-grouped Stage 2.

Results at shared M=16:

| Persistent workers | Latency (us) | Correct |
|-------------------:|-------------:|:--------|
| 256 | ~329 | Yes |
| 128 | ~230 | Yes |
| 64 | ~219 | No |

The software barrier did not deadlock at 128 or 256 workers, but the design is
far slower than the ~91 us sparse-grouped path. The dominant overheads are the
serialized block-0 sort, spin waiting, global task-queue atomics, and repeated
per-task workgroup barriers. Reducing the worker count increases the number of
statically unrolled queue passes and eventually failed to process the output
correctly at 64 workers.

The prototype code was removed. If persistence is revisited, it should use the
existing all-block arrival/last-arriver barrier pattern from
`moe_fused_route_quant_scatter.py`, where all workgroups participate in the
counting phase and the last arriver performs only the prefix/dispatch step.

### All-block last-arriver prototype result

The recommended all-block pattern was also implemented:

1. A zeroed workspace initializes two arrival/release barriers.
2. All blocks clear and count routed experts.
3. The last arriving block builds dense routed metadata.
4. All blocks consume GEMM1 tasks from an atomic queue.

Results at shared M=16:

| Workers | Latency (us) | Correct |
|--------:|-------------:|:--------|
| 256 | ~273 | Yes |
| 128 | ~217 | Yes |

This is faster than the block-0 sorter prototype but still more than 2x slower
than the ~91 us sparse-grouped kernel. The two grid barriers, metadata
construction, global queue atomics, and persistent task loop overwhelm the
approximately 3 us adaptive sort that the kernel is intended to eliminate.

The all-block persistent prototype code was removed. Persistence is not a
promising direction for this low-token shape.

The next fundamental experiment should accept the separate adaptive sort and
test a sorted BN128 `f16in` GEMM1. The three-launch structure may still beat the
current main BN256 pipeline because sorting is cheap and BN128 materially
improves low-M GEMM1.

### Sorted BN128 M=16 result

The separate adaptive-sort pipeline was tested with only GEMM1 changed from
BN256 to BN128:

- Main sorted BN256 `f16in`: approximately 88.57 us
- Sorted BN128 `f16in`: approximately 91.99 us

BN128 helps sparse route-direct layouts but regresses the dense sorted M=16
pipeline. Main's BN256 GEMM1 remains the correct tile once expert rows are
already compacted.

## Remaining longshot directions

### Production token buckets

AITER kernel selection rounds the runtime token count up to a power-of-two
bucket. Measurements at intermediate values are diagnostic and do not imply a
separate production kernel row. Follow-up tuning should report both:

- The actual runtime token count used by the benchmark.
- The power-of-two kernel bucket selected by AITER.

In particular, M=12 exercises the M=16 selection bucket. Comparisons must
force or report the kernel selected for that bucket rather than treating M=12
as an independently tuned production point.

### 1. Fused sort plus one-time input quantization

Quantize each BF16 token once, then reuse the FP4 activation and e8m0 scales
across all routed experts and GEMM1 N tiles. The route/quant preparation could
use the existing fused dynamic MXFP4 quant-and-sort path.

Advantages:

- Removes repeated BF16 loads and quantization arithmetic from every routed
  GEMM1 workgroup.
- Uses conventional compact GEMM1/GEMM2 kernels with no software grid barrier.

Risks:

- Adds a third external launch.
- Main selected `f16in` because separate quantization historically lost at
  smaller token counts; M=16 shared routing must be remeasured rather than
  inferred from the older normal-routing rows.

This is the lowest-risk remaining experiment.

#### Quant-once M=16 result

The complete conventional pipeline was measured with BM32 prequantized GEMM1:

- Main `f16in`: approximately 89.0 us
- Quant-once with cached GEMM1: approximately 116.7-119.3 us
- Quant-once with non-temporal GEMM1: approximately 103.8-106.1 us

The valid BM32 preparation path selected Opus sorting:

- BM32 sort: approximately 13.7 us
- BM32 sort plus MXFP4 quant/scale shuffle: approximately 18.5 us

Forcing the adaptive BM32 auxiliary sort in the direct prequantized wiring
still measured roughly 14.5 us and produced an incompatible layout/output in
that prototype. Quant-once is therefore not competitive for the M=16 bucket.

### 2. Token-centric multi-expert GEMM1

Dispatch one workgroup around a token/N tile instead of an expert/N tile.
Quantize the token once into LDS, then assign waves to different routed experts
and reuse the same A tile.

For M=8, small logical N tiles could still produce approximately one workgroup
per CU while eliminating eight repeated activation quantizations per token.

Risks:

- Every wave uses a different B expert, requiring a new weight/scale addressing
  and scheduling scheme.
- Four waves can process only four experts concurrently, requiring multiple
  expert phases for top-8 routing.
- Register pressure and long-lived LDS A tiles may reduce occupancy.

This is the highest-upside two-launch design, but requires a new GEMM1 body.

#### Proposed token-centric mapping

For each token:

- Two expert phases cover the eight routed experts.
- Four waves each own one expert within a phase.
- Sixteen logical-N chunks cover the 512 intermediate values at 32 values per
  chunk.
- Grid size is `M * 2 * 16`: 256 workgroups at M=8 and 512 at M=16.
- The token is quantized once into LDS and reused by all four expert waves.
- Each wave performs its own gate/up MFMA sequence and writes one
  route-private 32-value FP4 chunk plus one complete MX scale group.

This avoids the cross-workgroup FP4 scale problem that made BN32 invalid:
each wave emits a complete 32-value logical chunk (raw gate/up width 64).

#### Token-centric prototype result

A prototype reused the existing four-wave GEMM1 loop with:

- A different routed expert per physical wave.
- One shared quantized token row in LDS.
- A wave-local SiLU, 32-value amax reduction, FP4 packing, and scale store.

Performance at shared M=8 was promising at approximately 52.36 us, but the
output was incorrect:

- Wave 0 / route 0 was bit-exact.
- Waves 1-3 produced incorrect FP4 values.
- The second four-expert phase produced zero rows.
- End-to-end normalized difference was approximately 0.335.

Forcing all waves to the same expert did not fix waves 1-3, confirming that the
failure is in the existing GEMM body's physical-wave B/MFMA assumptions rather
than route/expert addressing. The incorrect prototype code was removed.

A viable token-centric implementation therefore needs a genuinely wave-native
MFMA loop and cannot safely reuse the current four-wave cooperative GEMM1 body.

#### Serial-route geometry stepping stone

Before another wave-native MFMA rewrite, validate the token/phase dispatch and
route-private output mapping with a correctness-first kernel:

- Grid: `M * 2 phases * 16 logical-N chunks`, plus 16 grouped shared-expert
  workgroups.
- Each routed workgroup invokes the existing correct BN64 GEMM1 body four
  times serially for the four routes in its phase.
- Routed intermediate blocks remain `token * 8 + slot`; the final block is the
  grouped expert-256 intermediate consumed by the existing shared-hybrid
  Stage 2.
- Input quantization is intentionally repeated. This prototype tests only
  whether consolidating dispatch around `(token, phase, logical-N)` is correct
  and remotely viable before factoring quantization outside the route loop.

The expected grids are 272 workgroups at M=8 and 528 at M=16. A useful result
would justify replacing the four serial bodies with a wave-native multi-expert
body; a large regression would stop that rewrite early.

The serial-route prototype compiled and produced correct shared-routing output:

- M=8 normalized difference: approximately `5.1e-6`
- M=16 normalized difference: approximately `5.3e-6`

Stable stage profiles were:

| M | Serial Stage 1 (us) | Shared-hybrid Stage 2 (us) | Sum (us) |
|---:|--------------------:|----------------------------:|---------:|
| 8  | 66.61 | 15.32 | 81.93 |
| 16 | 84.66 | 29.20 | 113.86 |

The M=8 end-to-end smoke measurement was approximately 84.0 us and M=16 was
approximately 114.0 us. These are intentionally not competitive: the routed
workgroup repeats the complete quantize/GEMM/epilogue body four times.

The useful result is correctness of the `M * 2 * 16` dispatch and the
route-private output layout. The remaining upside is concentrated entirely in
Stage 1, so the next implementation should retain this grid while replacing
the four serial bodies with one shared A quantization and four wave-native
expert bodies.

#### Wave-native shared-A implementation

The routed Stage 1 now uses the validated token/phase grid with genuinely
independent expert waves:

- The four waves quantize disjoint K tiles of one BF16 token into shared LDS.
- Each wave owns one routed expert and computes both gate and up columns.
- Four-role batches limit live B fragments while retaining enough outstanding
  memory operations to hide weight latency.
- Only the live MFMA row is retained, activated, quantized to FP4, and written
  to the original route-private intermediate block.
- The packed A row is stored compactly; the other 15 MFMA rows are deliberately
  undefined because their results are discarded.
- The deterministic shared expert remains a grouped BM16 block with expert ID
  256 and weight 1.0.

The routed width is bucket-specific:

- M<=8: BN64, cached W1 loads.
- M>8: BN128, non-temporal W1 loads.

The larger bucket also has a wave-native Stage 2. Its grid is
`M * 2 phases * 24 N blocks`, plus 24 grouped shared-expert blocks. Each wave
loads one route-private FP4 row, computes all 256 output columns in four-role
batches, and atomically accumulates its weighted BF16 result. M<=8 retains the
existing route-direct shared-hybrid Stage 2 because it is equally fast and has
a more stable accumulation order.

The selected same-process comparison against the forced latest-main path
(adaptive auxiliary sort + BM16 `f16in` GEMM1 + BM16 atomic GEMM2) is:

| Actual M | Token-wave (us) | Main `f16in` (us) | Delta |
|---------:|----------------:|------------------:|------:|
| 8  | 46.88 | 48.95 | -4.22% |
| 12 | 66.52 | 75.17 | -11.50% |
| 16 | 88.55 | 85.82 | +3.17% |

Normalized differences were approximately:

- M=8: `5.2e-6`
- M=12: `1.23e-5`
- M=16: `8.9e-6`

The M=12 row is diagnostic: AITER still treats it as the M=16 production
bucket. The same BN128/NT recipe is used at M=12 and M=16; it wins strongly at
the former but remains behind at the full bucket.

At M=16, a representative selected profile was:

- Token-wave Stage 1: approximately 61.2 us
- Token-wave Stage 2: approximately 28.6 us
- Forced-main sort: approximately 4.3 us
- Forced-main GEMM1: approximately 55.2 us
- Forced-main GEMM2: approximately 26.9 us

The remaining M=16 gap is routed duplicate reuse. Main pays for sorting but
reduces roughly 128 routed rows to the distinct routed experts, while the
token-centric path intentionally computes every route independently.

Important tuning and negative results:

- Compact A is neutral at M=8 and saves roughly 5 us in isolated M=16 Stage 1
  versus the full 49 KiB row-padded LDS representation.
- Cached W1 loads win at M=8; non-temporal W1 loads win in the real alternating
  GEMM1/GEMM2 workload for M=12/16.
- Shared-expert BN256 regresses Stage 1 by roughly 4 us; grouped shared BN64
  remains selected.
- Stage-1 BN128 role batches:
  - batch 2: approximately 94.5 us end to end
  - batch 4: approximately 89.1 us before the final Stage-2 change
  - batch 8: approximately 89.4 us
- Routed BN256 measured approximately 102.7 us and produced NaNs in part of
  the output; it is disabled.
- Wave-native Stage-2 cached batch 4 is selected. Batch 8 is slower and
  non-temporal W2 loads regress to approximately 98.4 us.
- Removing the non-temporal hint from FP4 intermediate stores was effectively
  neutral within run-to-run variance.

No tuning CSV row is enabled yet. A production integration should use the
token-wave path for the small/intermediate actual-M cases where it wins and
fall back to sorted `f16in` at the full M=16 bucket unless routed duplicate
grouping is added.

### WaveScope/ATT M=16 bottleneck analysis

A steady-state ATT and PMC comparison was captured for token-wave GEMM1/GEMM2
and forced-main `f16in` GEMM1/GEMM2. The complete report and WaveScope
annotation payloads are in
`docs/benchmarks/wavescope_glm52_m16/README.md`.

The main conclusions are:

- Token-wave GEMM1 spends 88.35% of active wave time in WAIT and has only one
  routed wave per SIMD because the routed grid is almost exactly one workgroup
  per CU.
- Four-role batches repeatedly expose 1.5-3.0k-cycle `vmcnt(7)` waits after
  only one MFMA pair. A rolling four-role prefetch is the highest-priority
  micro-optimization.
- Token-wave GEMM2 spends 95.1% of wave time in WAIT+STALL. Its initial eight
  narrow B-scale loads gate the first MFMA for approximately 2.95k cycles.
- Token-wave issues 26.6% more GEMM1 TCC requests and 18.9% more GEMM2 TCC
  requests than sorted `f16in`. This quantifies the irreducible duplicate-route
  cost at full M=16.
- Token-wave LDS conflict ratios are only 4.95%/1.61% for GEMM1/GEMM2, versus
  44.4%/63.7% for the baseline. LDS layout and atomics are not the next
  bottlenecks to pursue.

#### Deterministic compact-route output

Each token-wave route now computes its rank and first matching leader among the
128 routed rows using two ballot/popcount scans. Stage 1 writes FP4 output
directly to `leader_block * BM16 + rank` and publishes count/expert/token/weight
metadata. This allows the existing grouped routed Stage 2 to be reused without
an extra sort launch.

Selected measurements:

| Actual M | Compact/grouped (us) | Prior token-wave (us) | Main `f16in` (us) |
|---:|---:|---:|---:|
| 12 | 65.06 | 67.13 | 74.98 |
| 16 | 86.48 | 88.66 | 84.09 |

At M=16:

- Compact-route Stage 1: approximately 61.5 us.
- Grouped Stage 2: approximately 26.2 us.
- Grouped Stage-2 TCC requests: 1.447M, versus 1.758M for the prior
  route-private Stage 2 and 1.478M for forced `f16in`.

The remaining gap is entirely GEMM1. Compact-route GEMM1 still issues 3.429M
TCC requests versus 2.707M for sorted `f16in`.

ATT-inspired rolling prefetch, next-batch overlap, two-K-tile scale prefetch,
and half-major payload ordering were all implemented and rejected. None beat
the original four-role batch schedule; detailed negative results are recorded
in `docs/benchmarks/wavescope_glm52_m16/README.md`.

The final hybrid GEMM1 longshot was also implemented. Token-wave routes with
`count > 1` suppressed their W1 payload/MFMA work, while added candidate
workgroups ran grouped BM16 GEMM1 for duplicate experts. It was correct but
regressed:

- Four duplicate N groups: approximately 96.3 us.
- Two duplicate N groups: approximately 96.4 us.
- One duplicate N group: approximately 125.6 us.

Candidate scan overhead dominates at four groups; insufficient duplicate-work
parallelism dominates at one group. This closes the remaining no-extra-launch
duplicate-GEMM1 direction. Exact M=16 should use sorted `f16in`.

### 3. Multi-route wave-specialized workgroups

Pack independent route/N tasks into the four waves of one workgroup. Each wave
uses its own expert and B tile while sharing only dispatch and synchronization
infrastructure.

This reduces workgroup scheduling overhead but does not remove repeated
quantization unless combined with the token-centric design. It is less likely
to move M=16, where workgroup count is already sufficient.

### 4. Atomic-free shared-output initialization

Let the grouped shared expert write the initial output non-atomically, then add
routed experts atomically.

This requires a per-token producer/consumer ordering mechanism inside Stage 2.
Without a device-wide ordering primitive it risks the same residency and
spin-wait problems as the rejected persistent prototypes.

### 5. Shared-routing-specific retune of main `f16in`

Sweep the existing main kernel candidates under deterministic shared routing,
including GEMM1/GEMM2 BM, cache policy, XCD swizzle, atomic/reduce epilogues,
and persistent Stage 2.

This does not create a new fused kernel, but it is more likely to improve the
production M=16 result than further candidate-local sorting work.

#### Shared-routing `f16in` sweep result

At M=16 shared:

- BM16 atomic cached: approximately 88.54 us
- BM16 atomic non-temporal: approximately 91.51 us
- BM32 cshuffle: approximately 110.83 us
- Forced BM32 atomic combinations were numerically invalid in this wiring

Lower-level XCD swizzles produced only noise-level changes:

- Best observed: GEMM1 XCD=2 at approximately 88.33 us
- Baseline XCD=0: approximately 88.64 us

The existing BM16 atomic cached main configuration remains the best supported
`f16in` recipe under deterministic shared routing.

### Production-derived M=1–16 validation image

On July 29, 2026, the specialized low-M dispatch was packaged on top of the
unchanged production vLLM image:

`amdsiloai/vllm-private:vllm_69715823_aiter_4a1cc77_glm52_production_tp4_pr1_pr50008`

The derivative is:

- Tag: `nholmber/glm52-fused-moe:3f5972f5c`
- Image ID:
  `sha256:c4bc1fa9b0b6177eb93c87a53e21d502ec56cedd96b523abca9f345a0ec0fb83`
- Overlay size: approximately 802 KiB
- Runtime guard: `AITER_GLM52_FUSED_MOE=1`, actual M=1 through M=16

The public `aiter.fused_moe` dispatcher selects:

- M=1–2: flat BN128
- M=3–4: deterministic shared-hybrid BN128
- M=5–8: token-wave BN64
- M=9–16: compact token-wave BN128 plus grouped routed GEMM2

Built-image public-API validation on gfx950 GPU 4 produced:

| Actual M | Selected path | Normalized difference |
|---:|:---|---:|
| 1 | flat | `1.139e-5` |
| 2 | flat | `1.267e-5` |
| 3 | shared-hybrid | `5.748e-6` |
| 4 | shared-hybrid | `5.376e-6` |
| 5 | token-wave | `6.302e-6` |

The ASE configuration expanded all 22 requested experiments in dry-run mode.

#### TP4 coherence and GSM8K validation

After the existing `glm52-eval` workload drained, the derivative image was
started on GPUs 0–3 with:

- Tensor parallel size 4
- Async scheduling
- FP8 KV cache
- Production AITER shared-expert integration
- `AITER_GLM52_FUSED_MOE=1`
- Actual-M guard 1 through 16

The original production container was preserved in the stopped
`glm52-eval-baseline-20260729` container.

Coherence checks passed:

- A single request identified Paris as the capital of France.
- A single arithmetic request correctly derived `127 * 53 = 6731`.
- Sixteen synchronized short-answer requests passed `16/16` expected-answer
  checks in approximately 0.22 seconds wall time.

GSM8K used the same sample recipe as the existing production result:

- 100 examples
- 5-shot
- Concurrency 32
- `max_gen_toks=8192`
- Seed 44

| Image | Flexible extract | Strict match |
|:---|---:|---:|
| Production `pr1_pr50008` | `0.95` | `0.94` |
| Fused M=1–16 image | `0.95` | `0.95` |

The per-example outputs changed, as expected from the numerical path change.
For flexible extraction there were two gains and two losses relative to the
production sample, leaving the aggregate score unchanged. Strict extraction
had three gains and two losses, increasing the sample score by one point.

All 100 generated responses were non-empty. Their median length was 1,199
characters and the maximum was 10,810 characters. The longest response
remained coherent and received a correct score. There were no server
tracebacks, exceptions, NaNs, fatal errors, or failed API requests during the
run.

Artifacts:

`/home/nholmber/silo-tiger-oob-benchmark-configs-2/phantom-configs/mi355/results_lmeval/glm52_fused_moe_3f5972f5c_gsm8k`

The evaluation helper's final `GSM8K_SCORE` convenience line reports `0.0219`
because its grep selects the standard error. The aggregated lm-eval table and
results JSON are the source of truth for the `0.95` score.
