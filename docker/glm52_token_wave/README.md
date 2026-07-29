# GLM-5.2 token-wave e2e image

Build from the self-contained context:

```bash
docker build \
  -t nholmber/glm52-token-wave:ff3b34254 \
  docker/glm52_token_wave
```

The derivative starts from:

`amdsiloai/vllm-private:vllm_69715823_aiter_4a1cc77_glm52_production_tp4_pr1_pr50008`

It overlays the FlyDSL token-wave kernels and enables the guarded public
`aiter.fused_moe` dispatch for actual M=5 through M=16. All other shapes and
token counts retain the base image's production path.

ASE config:

`vllm_glm-5.2_tp4_mxfp4_token_wave_e2e.yaml`

Run it with the local inference-testing/ASE installation:

```bash
inference-testing \
  -c docker/glm52_token_wave/vllm_glm-5.2_tp4_mxfp4_token_wave_e2e.yaml
```

## Built image

- Tag: `nholmber/glm52-token-wave:ff3b34254`
- Image ID: `sha256:5cebcd0514ac9616aebaf66d0ae6a97cf8aadeff658906412c53bb12f1d79156`
- Size: 36.3 GB
- Overlay layer: approximately 753 KiB

## Validation

- AITER imports from `/tmp/aiter-main`.
- The patched public `aiter.fused_moe` dispatch is enabled for actual M=5–16.
- The FlyDSL token-wave module imports from the overlaid source tree.
- An M=8 shared-routing public-API smoke test selected:
  - `mxfp4_token_wave_shared_g1_*_bn64_*`
  - `mxfp4_shared_hybrid_g2_*`
- Public-path normalized difference versus the torch reference was
  approximately `5.3e-6`.

The complete four-GPU server smoke test was not started during image creation
because an existing `glm52-eval` container was actively holding GPUs 0–3.
