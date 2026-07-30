# GLM-5.2 fused low-M e2e image

Build from the self-contained context:

```bash
docker build \
  -t nholmber/glm52-fused-moe:3f5972f5c \
  docker/glm52_token_wave
```

The derivative starts from:

`amdsiloai/vllm-private:vllm_69715823_aiter_4a1cc77_glm52_production_tp4_pr1_pr50008`

It overlays the specialized FlyDSL low-M kernels and enables the guarded
public `aiter.fused_moe` dispatch for actual M=1 through M=16:

- M=1–2: route-direct flat BN128
- M=3–4: deterministic shared-hybrid BN128
- M=5–8: token-wave BN64
- M=9–16: compact token-wave BN128 with grouped routed GEMM2

All other shapes and token counts retain the base image's production path.

ASE config:

`vllm_glm-5.2_tp4_mxfp4_token_wave_e2e.yaml`

Run it with the local inference-testing/ASE installation:

```bash
inference-testing \
  -c docker/glm52_token_wave/vllm_glm-5.2_tp4_mxfp4_token_wave_e2e.yaml
```

## Image

- Tag: `nholmber/glm52-fused-moe:3f5972f5c`
- Registry tag:
  `amdsiloai/vllm-private:vllm_69715823_aiter_3f5972f5c_glm52_fused_moe_m1_m16`
- Registry digest:
  `sha256:42b3dc81d2f5bb868799c26ee932c989addb0ca14cb1e5cecf50417e922a03dc`
- Image ID:
  `sha256:c4bc1fa9b0b6177eb93c87a53e21d502ec56cedd96b523abca9f345a0ec0fb83`
- Size: 36.3 GB
- Overlay layer: approximately 802 KiB
- AITER source revision: `3f5972f5c`
- Base image and its vLLM integration are unchanged.

## Profiling A/B

Matched PyTorch/Kineto profiling configurations for the production MXMOE
baseline and the fused M=1–16 image are in `profiling/`. They cover 60k/600
workloads at concurrency 1, 2, 4, 8, and 16.

See `profiling/README.md` or run:

```bash
docker/glm52_token_wave/profiling/run_ab.sh
```

## Validation

- AITER imports from `/tmp/aiter-main`.
- The patched public `aiter.fused_moe` dispatch is enabled for actual M=1–16.
- Flat, shared-hybrid, and token-wave modules import from the overlaid source
  tree.
- Built-image public-dispatch correctness checks on gfx950 GPU 4:

  | M | selected path | normalized difference |
  |---:|:---|---:|
  | 1 | flat | `1.139e-5` |
  | 2 | flat | `1.267e-5` |
  | 3 | shared-hybrid | `5.748e-6` |
  | 4 | shared-hybrid | `5.376e-6` |
  | 5 | token-wave | `6.302e-6` |

- ASE dry-run expanded all 22 benchmark experiments and retained the intended
  image tag and M=1–16 environment guard.

Four-GPU end-to-end validation was subsequently completed on GPUs 0–3:

- TP4 vLLM started with async scheduling, FP8 KV cache, the production shared
  expert integration, and the M=1–16 fused-MoE guard.
- Single-request checks returned `Paris` and the correct
  `127 * 53 = 6731` calculation.
- A synchronized 16-request coherence batch passed `16/16` checks in
  approximately 0.22 seconds wall time.
- A baseline-matched GSM8K sample used 100 examples, 5-shot prompting,
  concurrency 32, and `max_gen_toks=8192`:

  | Image | Flexible extract | Strict match |
  |:---|---:|---:|
  | Production base | `0.95` | `0.94` |
  | Fused M=1–16 | `0.95` | `0.95` |

- All 100 GSM8K responses were non-empty. The longest response was 10,810
  characters and remained coherent and correct.
- No server traceback, exception, NaN, fatal error, or failed API request was
  observed during the evaluation.

The GSM8K result artifacts are under:

`/home/nholmber/silo-tiger-oob-benchmark-configs-2/phantom-configs/mi355/results_lmeval/glm52_fused_moe_3f5972f5c_gsm8k`

The helper script's final `GSM8K_SCORE` line incorrectly captures the metric
standard error (`0.0219`). The aggregated lm-eval table and results JSON above
contain the actual `0.95` score.
