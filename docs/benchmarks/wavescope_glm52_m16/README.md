# GLM-5.2 M=16 WaveScope analysis

WaveScope/ATT analysis of the deterministic shared-expert GLM-5.2 TP4 shape:

- Hidden dimension: 6144
- Intermediate dimension: 512
- Experts: 257
- Top-k: 9
- Shared expert: ID 256, slot 8, weight 1.0
- GPU: gfx950 / MI355X

## Capture method

- WaveScope: `aghamari/WaveScope` v0.2.7, commit `3f49e996`
- rocprofv3: 1.1.0 from ROCm 7.2.3
- rocprof-trace-decoder: 0.1.6
- Physical GPU: 6
- ATT target CU/WGP: 1, shader engine 0
- Trace buffer: 96 MiB
- Captured dispatch: iteration 4 after three complete pipeline warmups
- PMC sidecar: TCC hit/miss, LDS active/conflict, active VALU, and GUI active

The workload harness is `capture_m16.py`. For example, from the repository
root inside the development container:

```bash
ATT_MODE=ours PYTHONPATH=/workspace/aiter \
  python docs/benchmarks/wavescope_glm52_m16/capture_m16.py
```

The ATT runs used `--kernel-iteration-range "[4]"` to select the invocation
after the three warmups in that script.

The decoded trace folders are stored outside the git repository:

| Trace | Decode folder |
|---|---|
| Selected compact-route GEMM1 | `/data/wavescope-att/ours-g1-compact-steady/ui_output_agent_48380_dispatch_203` |
| Selected grouped GEMM2 | `/data/wavescope-att/ours-g2-grouped-steady/ui_output_agent_56094_dispatch_204` |
| Superseded route-private GEMM1 | `/data/wavescope-att/ours-g1-steady/ui_output_agent_14178_dispatch_203` |
| Superseded token-wave GEMM2 | `/data/wavescope-att/ours-g2-steady/ui_output_agent_39561_dispatch_204` |
| Forced `f16in` GEMM1 | `/data/wavescope-att/f16in-g1-steady/ui_output_agent_49971_dispatch_207` |
| Forced `f16in` GEMM2 | `/data/wavescope-att/f16in-g2-steady/ui_output_agent_20702_dispatch_208` |

Each folder contains `annotations.json` and `pmc_counter_collection.csv`, so
WaveScope automatically shows both the agent findings and counter-backed
bottleneck rules.

## Dispatch resources

| Kernel | Workgroups | WG/CU | VGPR | LDS |
|---|---:|---:|---:|---:|
| Selected compact-route GEMM1 | 272 | 1.06 | 56 | 12 KiB |
| Forced `f16in` GEMM1 | 576 | 2.25 | 64 | 16 KiB |
| Selected grouped GEMM2 | 3096 | 12.09 | 48 | 20 KiB |
| Superseded token-wave GEMM2 | 792 | 3.09 | 64 | 20 KiB |
| Forced `f16in` GEMM2 | 3456 | 13.50 | 44 | 20 KiB |

The sorted grids include quickly rejected candidate blocks. In the GEMM1 trace,
the second `f16in` wave per SIMD exits after roughly 0.9k GPU cycles,
so the baseline does not retain two useful waves for the whole dispatch.

## ATT wave behavior

Percentages below include only active/long waves, excluding the short rejected
baseline candidates.

| Kernel | Active wave duration | Max overlapping waves/SIMD | EXEC | WAIT | STALL |
|---|---:|---:|---:|---:|---:|
| Selected compact-route GEMM1 | 113.0k cycles | 1 | 10.34% | 88.36% | 1.28% |
| Superseded route-private GEMM1 | 105.7k cycles | 1 | 10.54% | 88.35% | 1.08% |
| Forced `f16in` GEMM1 | 68.6k cycles | 2 briefly | 17.21% | 56.69% | 26.05% |
| Selected grouped GEMM2 | 20.3k cycles | 5 | 6.23% | 58.31% | 35.31% |
| Superseded token-wave GEMM2 | 48.8k cycles | 3 | 4.82% | 51.02% | 44.10% |
| Forced `f16in` GEMM2 | 18.9k cycles | 5 | 5.55% | 31.69% | 62.59% |

`WAIT` is explicit waitcnt/barrier time. `STALL` is instruction issue/pipeline
stall time. Token-wave GEMM1 has very little issue stall; its problem is that
the compiler reaches the next operand wait with almost no independent work
left.

## PMC comparison

| Kernel | TCC requests | TCC misses | L2 hit rate | LDS conflict ratio |
|---|---:|---:|---:|---:|
| Selected compact-route GEMM1 | 3.429M | 2.879M | 16.04% | 4.95% |
| Superseded route-private GEMM1 | 3.427M | 2.869M | 16.29% | 4.95% |
| Forced `f16in` GEMM1 | 2.707M | 2.657M | 1.83% | 44.44% |
| Selected grouped GEMM2 | 1.447M | 1.378M | 4.79% | 63.75% |
| Superseded token-wave GEMM2 | 1.758M | 1.509M | 14.12% | 1.61% |
| Forced `f16in` GEMM2 | 1.478M | 1.388M | 6.11% | 63.75% |

Compared with sorted `f16in`, the original route-private token-wave path
issues:

- 26.6% more GEMM1 TCC requests and 8.0% more misses.
- 18.9% more GEMM2 TCC requests and 8.8% more misses.

The route-private kernels have better cache hit rates and dramatically fewer
LDS conflicts, but still move more total data. The selected grouped GEMM2
reuses the baseline GEMM2 body and its LDS pattern in exchange for eliminating
duplicate-route traffic.

## Deterministic route compaction follow-up

Each routed wave now scans the 128 routed IDs, computes:

- `rank`: number of earlier routes targeting the same expert.
- `leader`: first routed index targeting that expert.
- `count`: total routed rows for that expert.

The wave writes its FP4 intermediate directly to
`leader_block * BM16 + rank`. Since a token cannot route twice to one expert
and M<=16, one expert can have at most 16 rows and always fits one BM16 block.
The separate GEMM2 launch can therefore reuse the existing grouped routed
kernel without a sort launch or device-wide synchronization.

Selected same-process measurements:

| Actual M | Compact/grouped | Original token-wave | Forced `f16in` |
|---:|---:|---:|---:|
| 12 | 65.06 us | 67.13 us | 74.98 us |
| 16 | 86.48 us | 88.66 us | 84.09 us |

At M=16, a representative profile is:

- Compact-route GEMM1: approximately 61.5 us.
- Grouped GEMM2: approximately 26.2 us.

The route scan lengthens the traced GEMM1 wave from 105.7k to 113.0k cycles,
but does not materially change device-level Stage-1 latency. Grouped Stage 2
recovers about 2 us, lowers active wave duration to 20.3k cycles, and removes
the route-private traffic excess:

- Grouped GEMM2 TCC requests: 1.447M.
- Forced `f16in` GEMM2 TCC requests: 1.478M.
- Superseded route-private GEMM2 requests: 1.758M.

The remaining full-M=16 gap is now entirely GEMM1.

## Primary bottlenecks

### 1. GEMM1 role-batch operand starvation

The four-role register batch avoids the register pressure of the eight-role
variant, but it is not software-pipelined.

Representative sequence:

1. Issue the batch's B payload loads.
2. Wait to consume one role.
3. Execute one MFMA pair.
4. Stall 1.5-3.0k cycles at `s_waitcnt vmcnt(7)` for the next role.

Examples:

- Instructions 4369, 3308, and 3923 each accumulate 8.7-9.1k wait cycles
  across the four traced waves.
- Similar `vmcnt(7)` blocks repeat through all 24 K tiles.

Because there is only one routed workgroup per CU, no other routed wave can
cover these waits.

### 2. Superseded route-private GEMM2 feed waits

The old token-wave GEMM2 loads one A scale and eight strided B-scale dwords
before the first role batch. Instruction 367 waits 2,952 cycles before the
first MFMA in the representative slow wave.

Afterward, the role batches repeatedly descend through `vmcnt(8..2)`, with
expensive `vmcnt(7)` and `vmcnt(3)` points. The MFMA instructions themselves
have very little stall. Grouped GEMM2 resolves this path and is now slightly
faster than forced `f16in` GEMM2.

### 3. Duplicate routed-expert GEMM1 traffic

This is the remaining structural full-M=16 gap. Token-wave GEMM1 deliberately
processes all 128 routed rows independently. Main pays roughly 4.3 us for
sorting, then executes only the distinct expert blocks.

The extra TCC requests remain even if wait overlap becomes perfect. This is why
the token-wave path wins at actual M=8/12 but remains behind at the full M=16
bucket.

## Secondary or disproven bottlenecks

- **GEMM1 LDS bank conflicts:** compact-route GEMM1 remains much cleaner than
  main at 4.95%.
- **Route-private GEMM2 atomics:** already secondary and no longer on the
  selected path.
- **Grouped GEMM2 LDS/barriers:** visible in ATT, but the complete stage is
  already slightly faster than forced `f16in` GEMM2.
- **MFMA execution:** the matrix instructions rarely stall directly. The feed
  path starves them.
- **Simply widening the live role batch:** batch 8 was slower than batch 4 due
  to register pressure; batch 2 exposed too little memory-level parallelism.

## Recommended experiments

### P0: hybrid unique/duplicate GEMM1

Route compaction solved Stage 2. The only remaining structural opportunity is a
single Stage-1 launch with two work classes:

- Unique routed experts keep the token-wave shared-A path.
- Duplicate routed experts are computed once as a grouped BM16 expert block.

Both classes can derive rank/count from the deterministic routed-ID scan, and
write the same compact intermediate layout. This is the only remaining
no-extra-launch design that removes the measured 26.7% excess GEMM1 traffic.

### P1: keep forced `f16in` as exact-M=16 fallback

If hybrid GEMM1 bookkeeping costs more than the duplicate work it removes,
forced sorted `f16in` remains the correct exact-M=16 production choice.

### P2: avoid further software-pipeline variants

The ATT-inspired scheduling experiments were implemented and measured:

- Per-role rolling prefetch: both stages approximately 96.3 us.
- Rolling GEMM1 only: approximately 94.8 us.
- Rolling GEMM2 only: approximately 90.4 us.
- Next-batch overlap after role 0: GEMM1 approximately 95.9 us; GEMM2
  approximately 89.3 us.
- Two-K-tile GEMM2 scale prefetch: approximately 88.8 us, neutral.
- Half-major payload load ordering: 88.61 us versus 88.64 us in a same-process
  A/B, a tie.

The per-role rolling ATT wave grew from 105.7k to 145.9k cycles and converted
the batch waits into serialized `vmcnt(1..4)` waits. The compiler kept VGPRs
flat, so the regression was scheduling/memory-level parallelism rather than
occupancy.

### P3: avoid LDS/atomic-first tuning

Do not spend the next iteration on compact-A layout, LDS swizzles, or atomic
epilogues. ATT and PMC show those are not the limiting resources in the
token-wave implementation.

## Annotation files

- `ours_g1_annotations.json`
- `ours_g2_annotations.json`
- `ours_g1_compact_annotations.json`
- `ours_g2_grouped_annotations.json`
- `f16in_g1_annotations.json`
- `f16in_g2_annotations.json`

These are copies of the `annotations.json` payloads installed in the decoded
trace folders.
