# GLM-5.2 MXFP4 decode profiling A/B

Date: July 30, 2026

This report compares the production MXMOE image with the fused M=1–16 image
using the matched TP4 Kineto profiles in:

- `/tmp/traces_glm52_mxmoe_ab_20260730`
- `/tmp/traces_glm52_fused_moe_ab_20260730`

Both runs used 60,000 input tokens, 600 output tokens, GPUs 0–3, and dedicated
profiles at concurrency 1, 2, 4, 8, and 16. Results below average the four TP
ranks. Rank-to-rank mean step time differed by at most 6 us in every case.

The analysis uses full-duration `gpu_user_annotation` scheduler steps and only
pure decode groups (`prefill=0`, `decode=concurrency`). Kernel sums have small
ROCm timestamp overlap, so scheduler-step wall time is the primary end-to-end
metric. Kernel durations are used to explain the delta.

## Decode-step result

| Concurrency | Production MXMOE (ms) | Fused-MoE (ms) | Delta (us) | Delta | Implied token-rate delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 12.722 | 11.853 | -868.6 | -6.83% | +7.33% |
| 2 | 12.152 | 11.700 | -452.6 | -3.72% | +3.87% |
| 4 | 12.690 | 12.552 | -138.0 | -1.09% | +1.10% |
| 8 | 14.361 | 14.683 | +321.8 | +2.24% | -2.19% |
| 16 | 17.766 | 17.926 | +160.5 | +0.90% | -0.90% |

The crossover is unambiguous:

- Fused wins at M=1, M=2, and M=4.
- Production MXMOE wins at M=8 and M=16.

The M=4 distributions are separated despite the smaller win:

- Production p10/p90: 12.640 / 12.743 ms
- Fused p10/p90: 12.505 / 12.597 ms

The M=8 loss is similarly separated:

- Production p10/p90: 14.235 / 14.480 ms
- Fused p10/p90: 14.553 / 14.807 ms

## MoE path breakdown

There are 75 MoE layers per decode step. Values below are microseconds per MoE
layer.

| M | Production G1 | Production G2 | Production adaptive aux | Production total | Fused G1 | Fused G2 | Fused total | Fused path delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.956 | 6.208 | 13.194 | 30.357 | 17.601 | 4.899 | 22.500 | -7.858 |
| 2 | 20.497 | 7.313 | 4.052 | 31.862 | 18.443 | 6.987 | 25.429 | -6.432 |
| 4 | 22.363 | 10.855 | 4.113 | 37.331 | 23.591 | 11.403 | 34.994 | -2.337 |
| 8 | 31.230 | 17.393 | 4.053 | 52.676 | 37.775 | 17.992 | 55.767 | +3.091 |
| 16 | 53.698 | 28.146 | 4.228 | 86.072 | 59.359 | 28.868 | 88.227 | +2.155 |

At M=1, the production baseline is its tuned a4w4 recipe rather than the
MXMOE `f16in` recipe used at M=2 and above.

For M=2 and above, the traced kernel retains the internal
`moe_sort_quant::sort_quant_kernel` template name, but the `f16in` path invokes
it in sort-only mode. Its measured time includes the adaptive routing sort,
auxiliary index generation, padding metadata, and atomic-output zero
initialization. It does not include input quantization; G1 performs that
internally.

### M=1

Production launches five MoE kernels per layer:

- G1
- G2
- Two fused quant/sort launches
- One one-shot sorting launch

Flat fused-MoE launches only G1 and G2. Across the full step this removes 225
kernel launches and saves approximately 589 us of measured MoE kernel time.
The decode step improves by 869 us, so reduced launch/synchronization overhead
contributes materially beyond kernel duration.

### M=2

This is the cleanest fused win:

- Fused G1 is 2.05 us/layer faster.
- Fused G2 is 0.33 us/layer faster.
- Removing the adaptive auxiliary launch saves another 4.05 us/layer.

The MoE path saves 482 us per step and the decode step saves 453 us.

### M=4

The shared-hybrid G1 and G2 are individually slower than production by 1.23
and 0.55 us/layer. Removing 4.11 us/layer of adaptive auxiliary work still
yields a net 2.34 us/layer MoE win and a 138 us decode-step win.

### M=8

Token-wave G1 is the regression:

- G1: +6.55 us/layer
- G2: +0.60 us/layer
- Removed adaptive auxiliary work: -4.05 us/layer
- Net MoE regression: +3.09 us/layer, or +232 us/step
- Decode-step regression: +322 us

The MoE path explains 72% of the total step loss. To reach end-to-end
break-even with the current G2, token-wave G1 needs approximately another
4.3 us/layer reduction, from 37.8 us to about 33.5 us. That is roughly an
11% Stage-1 improvement.

### M=16

The trace reproduces the microbenchmark conclusion:

- G1: +5.66 us/layer
- G2: +0.72 us/layer
- Removed adaptive auxiliary work: -4.23 us/layer
- Net MoE regression: +2.16 us/layer, or +162 us/step
- Decode-step regression: +160 us

The MoE path accounts for essentially 100% of the decode-step loss. If G2 is
unchanged, compact token-wave G1 needs about 2.14 us/layer improvement to
break even end to end, reducing 59.36 us to approximately 57.22 us. This is a
3.6% Stage-1 target.

## Adaptive sort cost

The in-model adaptive auxiliary launch is nearly constant from M=2 through
M=16:

| M | Aux time per decode step | Time per MoE layer | Full-step share | MoE-path share |
|---:|---:|---:|---:|---:|
| 2 | 303.9 us | 4.052 us | 2.50% | 12.72% |
| 4 | 308.5 us | 4.113 us | 2.43% | 11.02% |
| 8 | 304.0 us | 4.053 us | 2.12% | 7.69% |
| 16 | 317.1 us | 4.228 us | 1.78% | 4.91% |

The prior isolated adaptive-sort microbenchmark measured approximately
2.7–3.2 us. The trace is higher because the production auxiliary launch also
emits the extra a4w4 indices and performs atomic-output zero initialization.

M=1 is not directly comparable. Its tuned production recipe spends 989.5 us
per step, or 13.19 us/layer, across two fused quant/sort launches and one
one-shot sorting launch per MoE layer.

## Tail-step caveat

Decode groups below the configured maximum concurrency occur after requests
finish. Their active sequences and context lengths are not matched between
the two runs, so raw wall times are not a valid kernel A/B.

For example, the concurrency-8 traces show an apparent 188 us fused wall-time
win at M=5, but the fused MoE path itself is 214 us slower:

| Actual M | Production MoE path (us/step) | Fused MoE path (us/step) | MoE delta |
|---:|---:|---:|---:|
| 5 | 4042.6 | 4256.4 | +213.8 |
| 6 | 4039.5 | 4264.9 | +225.4 |
| 7 | 3983.0 | 4208.4 | +225.4 |
| 8 | 3950.7 | 4182.6 | +231.9 |

The M=5 wall-time win therefore comes from unmatched non-MoE work, not the
fused kernel. A dedicated fixed-concurrency M=5 run is required before enabling
it.

M=9–15 tail groups have only 1–81 steps and the same context-matching issue.
They should not drive dispatch selection.

## Recommendation

For the current implementation:

```text
AITER_GLM52_FUSED_MOE_MIN_M=1
AITER_GLM52_FUSED_MOE_MAX_M=4
```

- Ship fused flat/shared-hybrid for M=1–4.
- Fall back to production MXMOE for M>=5.
- Do not enable token-wave M=8 or compact token-wave M=16 in production yet.
- If M=5 remains interesting, add a dedicated concurrency-5 profile rather
  than using the concurrency-8 tail.

Further M=8/M=16 work should focus almost entirely on Stage 1. Stage 2 is
within 0.6–0.7 us/layer of production, while Stage 1 is 5.7–6.5 us/layer
slower.

## Reproduction

The reusable parser is:

`docker/glm52_token_wave/profiling/analyze_decode_ab.py`

Run the four-rank summary with:

```bash
docker/glm52_token_wave/profiling/analyze_decode_ab.py \
  --ranks all \
  --workers 8 \
  --json-output /tmp/glm52_decode_ab_all_ranks.json
```
