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
| Token-wave GEMM1 | `/data/wavescope-att/ours-g1-steady/ui_output_agent_14178_dispatch_203` |
| Token-wave GEMM2 | `/data/wavescope-att/ours-g2-steady/ui_output_agent_39561_dispatch_204` |
| Forced `f16in` GEMM1 | `/data/wavescope-att/f16in-g1-steady/ui_output_agent_49971_dispatch_207` |
| Forced `f16in` GEMM2 | `/data/wavescope-att/f16in-g2-steady/ui_output_agent_20702_dispatch_208` |

Each folder contains `annotations.json` and `pmc_counter_collection.csv`, so
WaveScope automatically shows both the agent findings and counter-backed
bottleneck rules.

## Dispatch resources

| Kernel | Workgroups | WG/CU | VGPR | LDS |
|---|---:|---:|---:|---:|
| Token-wave GEMM1 | 272 | 1.06 | 56 | 12 KiB |
| Forced `f16in` GEMM1 | 576 | 2.25 | 64 | 16 KiB |
| Token-wave GEMM2 | 792 | 3.09 | 64 | 20 KiB |
| Forced `f16in` GEMM2 | 3456 | 13.50 | 44 | 20 KiB |

The sorted grids include quickly rejected candidate blocks. In the GEMM1 trace,
the second `f16in` wave per SIMD exits after roughly 0.9k GPU cycles,
so the baseline does not retain two useful waves for the whole dispatch.

## ATT wave behavior

Percentages below include only active/long waves, excluding the short rejected
baseline candidates.

| Kernel | Active wave duration | Max overlapping waves/SIMD | EXEC | WAIT | STALL |
|---|---:|---:|---:|---:|---:|
| Token-wave GEMM1 | 105.7k cycles | 1 | 10.54% | 88.35% | 1.08% |
| Forced `f16in` GEMM1 | 68.6k cycles | 2 briefly | 17.21% | 56.69% | 26.05% |
| Token-wave GEMM2 | 48.8k cycles | 3 | 4.82% | 51.02% | 44.10% |
| Forced `f16in` GEMM2 | 18.9k cycles | 5 | 5.55% | 31.69% | 62.59% |

`WAIT` is explicit waitcnt/barrier time. `STALL` is instruction issue/pipeline
stall time. Token-wave GEMM1 has very little issue stall; its problem is that
the compiler reaches the next operand wait with almost no independent work
left.

## PMC comparison

| Kernel | TCC requests | TCC misses | L2 hit rate | LDS conflict ratio |
|---|---:|---:|---:|---:|
| Token-wave GEMM1 | 3.427M | 2.869M | 16.29% | 4.95% |
| Forced `f16in` GEMM1 | 2.707M | 2.657M | 1.83% | 44.44% |
| Token-wave GEMM2 | 1.758M | 1.509M | 14.12% | 1.61% |
| Forced `f16in` GEMM2 | 1.478M | 1.388M | 6.11% | 63.75% |

Compared with sorted `f16in`, token-wave issues:

- 26.6% more GEMM1 TCC requests and 8.0% more misses.
- 18.9% more GEMM2 TCC requests and 8.8% more misses.

The token-wave kernels have much better cache hit rates and dramatically fewer
LDS conflicts. They still lose at full M=16 because sorting eliminates
duplicate routed-expert work and therefore reduces total traffic.

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

- Instruction 3006: 2,500 dynamic wait cycles.
- Instruction 3207: 2,956 dynamic wait cycles.
- Similar blocks repeat through all 24 K tiles.

Because there is only one routed workgroup per CU, no other routed wave can
cover these waits.

### 2. GEMM2 scale fan-out and role-batch waits

GEMM2 loads one A scale and eight strided B-scale dwords before the first role
batch. Instruction 367 waits 2,952 cycles before the first MFMA in the
representative slow wave.

Afterward, the role batches repeatedly descend through `vmcnt(8..2)`, with
expensive `vmcnt(7)` and `vmcnt(3)` points. The MFMA instructions themselves
have very little stall; operand delivery is the limiting path.

### 3. Duplicate routed-expert traffic

This is the structural full-M=16 gap. Token-wave deliberately processes all
128 routed rows independently. Main pays roughly 4.3 us for sorting, then
executes only the distinct expert blocks.

The extra TCC requests remain even if wait overlap becomes perfect. This is why
the token-wave path wins at actual M=8/12 but remains behind at the full M=16
bucket.

## Secondary or disproven bottlenecks

- **LDS bank conflicts:** token-wave is already much better than main.
- **Atomic output:** token-wave GEMM2's routed atomic pair accounts for roughly
  9.1k aggregate stall cycles, small compared with more than 503k load/wait
  cycles.
- **Final GEMM2 barrier:** roughly 18.1k aggregate stall cycles; also secondary.
- **MFMA execution:** the matrix instructions rarely stall directly. The feed
  path starves them.
- **Simply widening the live role batch:** batch 8 was slower than batch 4 due
  to register pressure; batch 2 exposed too little memory-level parallelism.

## Recommended experiments

### P0: rolling four-role prefetch

Keep four role fragments live, but after MFMA consumes role `r`, immediately
reuse that role's registers to issue the payload for role `r+4`. This overlaps
the next batch with the remaining current-batch MFMAs without recreating the
eight-role register footprint.

Apply the same scheme to both GEMM1 and GEMM2.

### P1: prefetch GEMM2 scales across K tiles

GEMM2 has only two K tiles. Load the A/B scales for K=1 while computing K=0, or
load both tiles' scale operands before entering the role pipeline. The scale
state is small compared with the B payload fragments and directly targets the
2.95k-cycle first-MFMA wait.

### P2: duplicate-aware M=16 path

Even successful software pipelining cannot remove the 19-27% extra TCC request
count. Options are:

- Keep forced sorted `f16in` as the exact-M=16 fallback.
- Introduce a hybrid path that groups only duplicate routed experts while
  retaining token-wave processing for unique routes.
- Revisit a lightweight route histogram/compaction only if it avoids the
  software-grid synchronization problems of the rejected persistent kernels.

### P3: avoid LDS/atomic-first tuning

Do not spend the next iteration on compact-A layout, LDS swizzles, or atomic
epilogues. ATT and PMC show those are not the limiting resources in the
token-wave implementation.

## Annotation files

- `ours_g1_annotations.json`
- `ours_g2_annotations.json`
- `f16in_g1_annotations.json`
- `f16in_g2_annotations.json`

These are copies of the `annotations.json` payloads installed in the four
decoded trace folders.
