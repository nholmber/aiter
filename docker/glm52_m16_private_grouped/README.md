# GLM-5.2 TP4 M=16 private-grouped A/B image

Base image:

```text
amdsiloai/vllm-private:vllm_69715823_aiter_4a1cc77_glm52_production_tp4_pr1_pr2_pr3_pr4_pr50008
```

Candidate tag:

```text
amdsiloai/vllm-private:vllm_69715823_aiter_4bec01e6a_glm52_m16_private_grouped_pr1_pr2_pr3_pr4_pr50008
```

Build:

```bash
./docker/glm52_m16_private_grouped/build_image.sh
```

The image leaves the production generic GEMM1/GEMM2 modules unchanged. It
installs private copies used only when all of these hold:

- `AITER_GLM52_M16_PRIVATE_GROUPED=1`
- actual M is 16
- gfx950
- GLM-5.2 TP4 MXFP4 fused-shared-expert shape:
  `(H=6144, inter=512, experts=257, topk=9)`
- shared expert is expert 256 in the final slot with weight 1

Unset or set `AITER_GLM52_M16_PRIVATE_GROUPED=0` to use the unchanged base
selection inside the candidate image.

## Exact-base A/B result

The August 4, 2026 TP4 concurrency-16 model A/B does not support enabling this
path in production:

| Workload | Output-token throughput delta | TPOT delta |
|:--|--:|--:|
| 1024 / 1024 | -1.94% | +2.15% |
| 8192 / 1024 | -1.70% | +1.80% |
| 60000 / 600 | +0.29% | -0.65% |

Matched pure-decode traces show that real GLM routing makes the private MoE
path 3.526 us/layer slower after accounting for the removed adaptive
auxiliary launch. Keep the flag disabled by default.

Detailed report:

```text
docs/benchmarks/glm52_m16_private_grouped_ab_20260804.md
```

Machine-readable results:

```text
docker/glm52_m16_private_grouped/results/20260804_exact_base_c16.json
```

Run the throughput A/B:

```bash
./docker/glm52_m16_private_grouped/run_ab.sh perf
```

Capture bounded pure-decode profiles:

```bash
./docker/glm52_m16_private_grouped/run_ab.sh profile
```

Analyze matched four-rank traces:

```bash
python docker/glm52_m16_private_grouped/analyze_profile.py \
  --workers 8 \
  --json-output /tmp/glm52_m16_private_profile_summary.json
```
