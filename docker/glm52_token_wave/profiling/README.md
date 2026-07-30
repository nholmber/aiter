# GLM-5.2 fused-MoE versus production MXMOE profiling

These two inference-testing configurations are a matched TP4 A/B:

- Production baseline:
  `amdsiloai/vllm-private:vllm_69715823_aiter_4a1cc77_glm52_production_tp4_pr1_pr50008`
- Fused M=1–16:
  `amdsiloai/vllm-private:vllm_69715823_aiter_3f5972f5c_glm52_fused_moe_m1_m16`

The vLLM arguments, AITER integration flags, model, random workload, seed, and
profiling settings are identical. The fused configuration only adds the
M=1–16 dispatch guard and uses the derivative image.

## Coverage

Each configuration profiles 60,000 input tokens and 600 output tokens at
concurrency 1, 2, 4, 8, and 16. These powers of two directly cover the runtime
selection buckets:

| Concurrency | Fused path |
|---:|:---|
| 1–2 | Flat BN128 |
| 4 | Deterministic shared-hybrid BN128 |
| 8 | Token-wave BN64 |
| 16 | Compact token-wave BN128 plus grouped routed GEMM2 |

The production image selects its tuned a4w4 recipe at M=1. At M=2, 4, 8, and
16 it selects the embedded-input-quant MXMOE
`flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt` GEMM1 with the tuned atomic GEMM2.

`stop_between_runs: false` keeps one loaded server per image while the five
profiling points run. inference-testing collects each trace using its
benchmark-start timestamp.

## Run

```bash
docker/glm52_token_wave/profiling/run_ab.sh
```

The script uses a host `inference-testing` installation when available.
Otherwise it runs through `nholmber-inference-testing-amd-1`; override that
name with `INFERENCE_TESTING_CONTAINER`.

Validate both configurations without starting servers:

```bash
DRYRUN=1 docker/glm52_token_wave/profiling/run_ab.sh
```

Run either side independently:

```bash
inference-testing --platform amd \
  -c docker/glm52_token_wave/profiling/vllm_glm52_mxfp4_mxmoe_profile.yaml

inference-testing --platform amd \
  -c docker/glm52_token_wave/profiling/vllm_glm52_mxfp4_fused_moe_profile.yaml
```

Host trace directories:

- MXMOE: `/tmp/traces_glm52_mxmoe_ab_20260730`
- Fused-MoE: `/tmp/traces_glm52_fused_moe_ab_20260730`

Analyze pure decode steps across all four TP ranks:

```bash
docker/glm52_token_wave/profiling/analyze_decode_ab.py \
  --ranks all \
  --workers 8 \
  --json-output /tmp/glm52_decode_ab_all_ranks.json
```

The completed analysis and dispatch recommendation are in:

`docs/benchmarks/glm52_mxfp4_decode_profile_ab_20260730.md`

The configurations use GPUs 0–3 and port 8000. Both prior validation servers
are stopped, so those resources were free when these files were created.
