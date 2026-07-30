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
- Image ID:
  `sha256:c4bc1fa9b0b6177eb93c87a53e21d502ec56cedd96b523abca9f345a0ec0fb83`
- Size: 36.3 GB
- Overlay layer: approximately 802 KiB
- AITER source revision: `3f5972f5c`
- Base image and its vLLM integration are unchanged.

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

The complete four-GPU server smoke test was not started during image creation
because an existing `glm52-eval` container was actively holding GPUs 0–3.
