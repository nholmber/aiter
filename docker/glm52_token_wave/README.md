# GLM-5.2 fused low-M e2e image

Build from the self-contained context:

```bash
docker build \
  -t nholmber/glm52-fused-moe:47208fb1a \
  docker/glm52_token_wave
```

The derivative starts from:

`amdsiloai/vllm-private:vllm_69715823_aiter_4a1cc77_glm52_production_tp4_pr1_pr50008`

It overlays the specialized FlyDSL low-M kernels and enables the guarded
public `aiter.fused_moe` dispatch for actual M=1 through M=4:

- M=1: one-launch route-direct flat, 128-wide intermediate chunks
- M=2: two-launch route-direct flat BN128
- M=3–4: deterministic shared-hybrid BN128

All other shapes and token counts retain the base image's production path.

ASE config:

`vllm_glm-5.2_tp4_mxfp4_token_wave_e2e.yaml`

Run it with the local inference-testing/ASE installation:

```bash
inference-testing \
  -c docker/glm52_token_wave/vllm_glm-5.2_tp4_mxfp4_token_wave_e2e.yaml
```

## Image

- Tag: `nholmber/glm52-fused-moe:47208fb1a`
- Registry tag:
  `amdsiloai/vllm-private:vllm_69715823_aiter_47208fb1a_glm52_fused_moe_m1_m4_s1`
- Registry digest:
  `sha256:11c9bfc62533c3b0955197017f4092f3fa1271afe0f76e7b611b439b4c1ca7ba`
- Image ID:
  `sha256:b461b897892fde2537f1d6fa34f357837795aad01c48986d656705daa1ca27db`
- Size: 36.3 GB
- AITER source revision: `47208fb1a`
- Base image and its vLLM integration are unchanged.

## Profiling A/B

The completed PyTorch/Kineto profiling configurations for the production
MXMOE baseline and the earlier fused M=1–16 image are in `profiling/`. They
cover 60k/600 workloads at concurrency 1, 2, 4, 8, and 16 and motivated the
new M=4 cutoff.

See `profiling/README.md` or run:

```bash
docker/glm52_token_wave/profiling/run_ab.sh
```

## Validation

- AITER imports from `/tmp/aiter-main`.
- The patched public `aiter.fused_moe` dispatch is enabled for actual M=1–4.
- Single-stage flat, two-stage flat, and shared-hybrid modules import from the
  overlaid source tree.
- Built-image M=1 validation on gfx950 GPU 0:

  - Public dispatch selected the single-stage kernel.
  - Normalized difference versus the independent torch reference was
    approximately `3.1e-5`.
  - Five ordinary repeats and five graph replays remained below `3.6e-5`.

- Synthetic steady-state M=1 performance was approximately 17% faster than
  the two-stage flat FlyDSL path.

## Earlier M=1–16 image validation

The previous `3f5972f5c` M=1–16 image completed four-GPU coherence and GSM8K
validation on GPUs 0–3:

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
