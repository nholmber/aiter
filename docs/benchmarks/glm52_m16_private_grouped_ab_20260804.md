# GLM-5.2 M=16 private-grouped model A/B

Date: August 4, 2026

This is the exact-production-base A/B for the route-private GEMM1 plus grouped
GEMM2 experiment.

Baseline:

```text
amdsiloai/vllm-private:vllm_69715823_aiter_4a1cc77_glm52_production_tp4_pr1_pr2_pr3_pr4_pr50008
```

Candidate:

```text
amdsiloai/vllm-private:vllm_69715823_aiter_4bec01e6a_glm52_m16_private_grouped_pr1_pr2_pr3_pr4_pr50008
```

The candidate has the complete baseline as its first 58 filesystem layers and
adds six overlay layers. The production `mxfp4_gemm1.py` and
`mxfp4_gemm2.py` checksums are identical in both images. Private copies are
imported only by the exact M=16 environment-gated path.

## Serving result

Both sides used TP4 on gfx950, concurrency 16, fused shared expert enabled,
`max_num_seqs=16`, the same random seeds, and the same prompt counts.

| Workload | Baseline output tok/s | Candidate output tok/s | Delta | Baseline TPOT | Candidate TPOT | TPOT delta |
|:--|--:|--:|--:|--:|--:|--:|
| 1024 / 1024 | 840.22 | 823.92 | -1.94% | 18.15 ms | 18.54 ms | +2.15% |
| 8192 / 1024 | 753.57 | 740.75 | -1.70% | 19.99 ms | 20.35 ms | +1.80% |
| 60000 / 600 | 175.07 | 175.58 | +0.29% | 77.87 ms | 77.36 ms | -0.65% |

The short and medium-context regressions are consistent. The 60k result is
effectively neutral because prefill and long-context attention dominate a
larger fraction of the request.

Databricks reporting timed out after the first candidate point. The 8192 and
60000 candidate measurements were therefore run directly against the same
already-loaded candidate server with the identical
`sglang.bench_serving` arguments. Raw logs remain under the ignored local
`docker/glm52_m16_private_grouped/logs/20260804_151325/` directory.

## Pure decode trace

Matched four-rank Kineto traces used the 1024 / 1024 workload with 32 prompts.
The analysis selects only scheduler groups with `prefill=0, decode=16`.

The mean wall time is contaminated by profiler stop/dump outliers. The robust
distribution shows a clear regression:

| Metric | Baseline | Candidate | Delta |
|:--|--:|--:|--:|
| Step p10 | 15.526 ms | 15.875 ms | +349 us / +2.25% |
| Step median | 15.699 ms | 16.091 ms | +392 us / +2.50% |
| Step p90 | 15.857 ms | 16.285 ms | +429 us / +2.70% |

MoE kernel time, averaged across four TP ranks and divided by 75 MoE layers:

| Component | Baseline | Candidate | Delta |
|:--|--:|--:|--:|
| GEMM1 | 54.982 us | 61.831 us | +6.849 us |
| GEMM2 | 28.759 us | 29.785 us | +1.025 us |
| Adaptive auxiliary/sort | 4.348 us | 0 | -4.348 us |
| Total | 88.090 us | 91.616 us | +3.526 us |

The private path therefore adds 264.4 us across the 75 MoE layers in each
M=16 decode step. This explains most of the 349–429 us robust scheduler-step
shift and agrees with the serving TPOT regression.

The candidate kernels are present in the traces:

```text
mxfp4_routed_compact_shared_g1_h6144_i512_ne257_tk9_bm16_rnt_scached_sep_ng4_private_v2
mxfp4_routed_compact_shared_g2_h6144_i512_ne257_tk9_bm16_rcached_scached_ng24_private_v2
```

This also confirms that the exact M=16 branch was selected. The ordinary
startup log contains generic selector lines for M=32, M=8, and lower tiers,
but no generic M=16 line because the opt-in branch returns before the tuned
selector.

## Interpretation

The isolated validator uses random routed scores. Its seed-41 route set has
about 101 unique routed experts and the candidate wins. Real GLM routing is
load-balanced and appears to have materially less expert reuse.

Stage 1 launches `M * (topk - 1) * 4 = 512` routed candidate workgroups.
Every candidate scans preceding routes to decide whether it is the expert
leader. With high route uniqueness, almost every candidate is a leader, so
the kernel gets little W1 reuse while retaining leader detection, metadata,
and route-private scatter overhead. The real-route G1 result is consequently
61.8 us/layer instead of the roughly 51–54 us seen on favorable synthetic
routes.

Holding the measured candidate GEMM2 at 29.785 us, GEMM1 must fall from
61.831 us to at most 58.305 us to reach MoE-path break-even. That is another
3.526 us, or 5.7%, on the real route distribution.

## Decision

Keep:

```text
AITER_GLM52_M16_PRIVATE_GROUPED=0
```

for production. The path remains useful as an opt-in experiment, but the
synthetic graph win does not transfer to real GLM-5.2 M=16 routing.

The next implementation should centralize or remove the per-candidate leader
scan, or retain the route-stable sorted/quant-once path. Promotion should be
based on real-route model traces, not random-route operator timing alone.

## Reproduction

Throughput:

```bash
./docker/glm52_m16_private_grouped/run_ab.sh perf
```

Trace analysis:

```bash
python docker/glm52_m16_private_grouped/analyze_profile.py \
  --workers 8 \
  --json-output /tmp/glm52_m16_private_profile_summary.json
```

Local trace directories:

```text
/tmp/traces_glm52_m16_baseline_20260804
/tmp/traces_glm52_m16_candidate_20260804
```
