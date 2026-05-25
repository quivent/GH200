# GH200 Inference — Raw PyTorch 2.x + torch.compile + Triton + FlashAttention + xFormers

**Agent**: research #06 of 20
**Date**: 2026-05-25
**Domain**: PyTorch / torch.compile / Triton / FlashAttention / xFormers / custom CUDA — for LLM + diffusion inference on NVIDIA GH200 Grace-Hopper (sm_90a, aarch64, 96 GB HBM3 + up to 480 GB LPDDR5X via NVLink-C2C).

---

## 0. Executive summary (TL;DR)

1. On GH200 in May 2026, the "raw PyTorch" stack that actually works in production is:
   **PyTorch 2.7 or 2.8 (cu128 aarch64 wheel) + Triton 3.7 + FlashAttention-3 (built from source against `flash-attention/hopper/`) + torchao for quantization + diffusers ≥ 0.34 / transformers ≥ 4.50.**
   Use FA3, not FA4 — FA4 (March 2026, CuTe-DSL) is Blackwell-first; on Hopper it is roughly *parity* with FA3, not a win, and FA3 remains the de-facto kernel.
2. **For LLM decode**: `torch.compile(model, mode="reduce-overhead", fullgraph=True)` with static KV-cache shapes is the right default. CUDA graphs come "for free" through `reduce-overhead`; vLLM's piecewise CUDA-graph path is what every serving engine has converged on.
3. **For diffusion (UNet/DiT)**: `pipe.transformer.compile_repeated_blocks(fullgraph=True)` (regional compile, diffusers ≥ 0.34) with `mode="max-autotune-no-cudagraphs"`, plus `torch.channels_last`, plus FA3 via SDPA. Regional compile cuts cold-start from ~67 s to ~10 s on Flux, with the same ~1.5× runtime speedup as full compile.
4. **FP8 is the live landmine.** vLLM/vllm-omni Issue #2728 (April 2026) documents catastrophic FP8 regressions on Flux.1-dev / Z-Image / Qwen-Image reproducible on **both H100 and B200** — i.e. it's the online-quant path, not the silicon. Combined with our local Wan2.x/GH200 observation (FP8 broken on this stack), **default to bf16** for diffusion. Use INT8/FP8 only where you have a known-good recipe (torchao `float8_dynamic_activation_float8_weight` on Flux.1-Dev *transformer* layers from the `huggingface/flux-fast` recipe is the closest to "trusted").
5. **GH200 vs H100 expectation**: GH200 SXM is an H100 die at 700W + 96 GB HBM3 (vs 80 GB H100 SXM) + LPDDR5X NVLink-C2C. For *per-step* compute-bound diffusion you should expect **H100 SXM numbers ±~5%**; the GH200 wins on (a) batched/longer-context bandwidth-bound LLM decode, (b) models that don't fit in 80 GB and would otherwise CPU-offload (Flux NF4, Wan2.2 14B fp16). NVIDIA's own MLPerf v4.1 result is "1.4× per-accelerator over H100 SXM" — a peak number, not typical for diffusion.

---

## 1. Install / version pins (GH200, aarch64)

### 1.1 PyTorch (the gotcha)
Default `pip install torch` on aarch64 gives a **CPU-only** wheel until PyTorch 2.11. For 2.7/2.8 you must point at the CUDA index:

```bash
# PyTorch 2.8 stable, cu128, aarch64 (works on GH200)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Or 2.7 (FlashAttention-3 betas were built against this for many users)
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

PyTorch 2.7 (2025-04-23) added Blackwell support and `torch.compile` improvements; PyTorch 2.8 (2025-08-06) added float8 training plumbing and broader stability. Source: PyTorch release notes ([2.7](https://pytorch.org/blog/pytorch-2-7/), [2.8 RC](https://dev-discuss.pytorch.org/t/pytorch-2-8-final-rc-available/3144)).

NVIDIA NGC monthly PyTorch container (`nvcr.io/nvidia/pytorch:25.04-py3` and later) is the path-of-least-resistance for GH200 — pre-built aarch64, includes Triton, cuDNN, NCCL, Transformer Engine. ([NGC PyTorch catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)).

### 1.2 Triton
Triton 3.7 publishes aarch64 wheels on PyPI:
```bash
pip install --upgrade pytorch-triton --index-url https://download.pytorch.org/whl/cu128
# or
pip install triton==3.7.0
```
([Triton on PyPI](https://pypi.org/project/triton/) – aarch64 wheels confirmed for cp310–cp314.)

### 1.3 FlashAttention-3 — build from source on GH200
There are no official prebuilt aarch64 wheels for FA3. The recipe verified on GH200 (sm_90/sm_90a, CUDA 12.8, PyTorch 2.7) is from [this gist (Dominoer, 2025)](https://gist.github.com/Dominoer/790091cc9e35c079c3251bf10492279e):

```bash
module load cuda/12.8 gcc/12.4.0
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(which nvcc)")")")"
export TORCH_CUDA_ARCH_LIST="9.0;9.0a"
export MAX_JOBS=6
# Hopper-only build (skip Ampere SM80 kernels — much faster build)
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_FP16=TRUE   # only if you only need bf16

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```
Notes:
- `setup.py install` inside `hopper/` builds the FA3 kernels (not FA2). Confirmed in the official repo README. ([Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention))
- Common GH200 failure (issue [#2036](https://github.com/Dao-AILab/flash-attention/issues/2036)) is an unrelated Ubuntu arm64 apt mirror miss for `libmysqlclient21` — not an FA bug.
- Community prebuilt wheels exist on HF Hub (`varunneal/flash-attention-3`, `varunneal/flash-attention-hopper`) but pin to specific torch/python; build from source is safer.

### 1.4 FlashAttention-4 — skip for now on GH200
FA4 landed publicly 2026-03-05 ([PyTorch blog](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/), [Lambda blog](https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet), [Modal reverse-engineer](https://modal.com/blog/reverse-engineer-flash-attention-4)). It's CuTe-DSL based and Blackwell-first: ~1,600 TFLOPS / 71% util on B200. On Hopper H200 the PyTorch blog reports only 1.30–1.65× speedups over Triton's flex backend (vs FA3's already-740-TFLOPS / 75%-util baseline), and crucially these are FlexAttention-with-FLASH-backend numbers, not raw FA3 vs FA4. **Practical reading**: on GH200, FA3 remains the workhorse in May 2026; install FA4 only if you need a feature it has and FA3 doesn't (e.g., FlexAttention score-mods).

### 1.5 xFormers
Still relevant for legacy diffusion code paths (memory-efficient attention with custom block-sparse masks). Build:
```bash
pip install -v --no-build-isolation --use-pep517 \
  "xformers @ git+https://github.com/facebookresearch/xformers.git@v0.0.30"
```
xFormers has deprecated its in-tree FA2 (defers to upstream Dao-AILab); on H100/GH200 it dispatches to FA2 or memory-efficient kernels. For diffusers ≥ 0.27, **SDPA + FA3 strictly dominates xformers**; keep xformers only if a third-party pipeline hard-codes `pipe.enable_xformers_memory_efficient_attention()`.

### 1.6 bitsandbytes / torchao
```bash
# bitsandbytes (NF4 for Flux): build from source on aarch64
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes && cmake -DCOMPUTE_BACKEND=cuda -S . && make -j6 && pip install -e .

# torchao (FP8 dynamic-activation, INT8, AWQ)
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 1.7 diffusers / transformers
```bash
pip install -U "diffusers>=0.35"  "transformers>=4.50" accelerate "huggingface_hub[hf_xet]"
```
diffusers **0.34** (regional compile API: `compile_repeated_blocks`) and **0.35** (Wan 2.2, Flux Kontext) are the relevant cuts. ([0.35.0 release](https://github.com/huggingface/diffusers/releases/tag/v0.35.0), [0.34.0 release](https://github.com/huggingface/diffusers/releases/tag/v0.34.0)).

---

## 2. SDPA backends and attention selection

PyTorch 2.x ships four SDPA backends, selectable per region:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel
with sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                  SDPBackend.CUDNN_ATTENTION,
                  SDPBackend.EFFICIENT_ATTENTION,
                  SDPBackend.MATH]):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```
([sdpa_kernel docs, PyTorch 2.12](https://docs.pytorch.org/docs/2.12/generated/torch.nn.attention.sdpa_kernel.html))

Hopper/GH200 priority list (what to actually do):

| Workload | First-choice backend | Notes |
|---|---|---|
| LLM prefill, bf16, causal | **FlashAttention-3** (via the `flash_attn_func` API, or SDPA `FLASH_ATTENTION` once PyTorch SDPA picks up FA3 — that lands progressively in 2.7/2.8) | Up to ~740 TFLOPS / 75% util fp16, ~1.5–2× over FA2 ([PyTorch blog on FA3](https://pytorch.org/blog/flashattention-3/)). |
| LLM decode (q_len=1) | cuDNN attention (`CUDNN_ATTENTION`) or FA3 decoding kernels | FA3 has tuned decode kernels; cuDNN is competitive and graph-friendly. |
| Diffusion U-Net cross-attn, head_dim=64 | FA3 (sm_90) | Falls back to mem-efficient on weird head dims. |
| DiT (Flux, Wan, Hunyuan), head_dim=128 | FA3 | Hopper's "happy path". |
| LoRA + small batch | FA3 with `is_causal=False` mask | |
| Score-modded attention (ALiBi, custom mask) | **FlexAttention** + FA3 kernel option | `flex_attention(..., kernel_options={"BACKEND":"FLASH"})` — only available with FA4 install on bleeding edge; on FA3 use the Triton backend. |

Do NOT use `enable_flash_sdp(False)` / `enable_math_sdp(True)` on GH200 unless you're debugging numerics — the math backend is 5–10× slower and burns HBM.

`torch.backends.cudnn.benchmark = True` is the right setting for **fixed-shape inference** (LLM decode, diffusion at a single resolution). It picks the best conv algo on first warm-up. Disable it if your shapes vary every call.

---

## 3. torch.compile — mode choices that matter

Three modes you'll actually use:

| mode | What it does | When to use |
|---|---|---|
| `"default"` | Inductor, no CUDA graphs | Debugging, breaking on graph breaks |
| `"reduce-overhead"` | Inductor + CUDA-graph capture for static-shape regions | **LLM inference default.** First call 30–90 s, then near-peak. Requires static shapes. ([Spheron 2026 guide](https://www.spheron.network/blog/torch-compile-cuda-graphs-llm-inference-pytorch-2-6/)) |
| `"max-autotune"` | Triton vs cuBLAS autotuning + CUDA graphs | Diffusion (Flux, SDXL). Compile time minutes-to-tens-of-minutes. Long-running servers. |
| `"max-autotune-no-cudagraphs"` | Autotune without graph capture | Diffusion + LoRA hot-swap; Wan2.2 (distributed ops break CUDA-graph capture); HunyuanVideo |

Empirical guidance from production recipes:

- **vLLM** uses `torch.compile` by default in V1 with "piecewise CUDA graphs" so that cascade-attention / un-graph-capturable ops can run eagerly while the rest is graph-captured. Disable with `--enforce-eager` for debugging. Compiled artifacts are cached under `~/.cache/vllm/torch_compile_cache` and are reusable across machines with matching env. ([vLLM torch.compile blog, 2025-08-20](https://vllm.ai/blog/2025-08-20-torch-compile))
- **HF flux-fast** uses `mode="max-autotune"` with `fullgraph=True` plus a set of Inductor flags (see §4). ([huggingface/flux-fast](https://github.com/huggingface/flux-fast))
- **HF diffusers official guide (2025-07-17)** recommends **regional compile** for diffusion authors: `pipe.transformer.compile_repeated_blocks(fullgraph=True)` cuts compile latency from 67.4 s → 9.6 s on Flux while keeping the ~1.5× runtime speedup. ([PyTorch blog](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/))
- **Dynamic shapes**: for diffusion with variable image sizes or batch sizes, use `dynamic=True` to avoid recompilation thrash. Without it, every new (H, W) triggers a fresh compile.

### 3.1 Canonical LLM inference snippet

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_3",   # transformers >= 4.50 with FA3 installed
).to("cuda").eval()

# Static KV cache is what makes reduce-overhead+CUDA-graph work
model.generation_config.cache_implementation = "static"
model.forward = torch.compile(
    model.forward, mode="reduce-overhead", fullgraph=True
)

# Warm-up to pay compile cost
_ = model.generate(**inputs, max_new_tokens=32)
```
HF transformers' `attn_implementation="flash_attention_3"` is the integration point as of `transformers` 4.50+ on Hopper. Use `"sdpa"` as the fallback and let PyTorch pick.

### 3.2 Canonical diffusion (Flux) snippet — H100/GH200

```python
import torch
from diffusers import FluxPipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.epilogue_fusion = False       # important on diffusion
torch._inductor.config.shape_padding = True

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")

# Memory layout: channels-last for both the DiT/UNet and VAE
pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

# Fuse QKV projections (single GEMM instead of three)
pipe.transformer.fuse_qkv_projections()
pipe.vae.fuse_qkv_projections()

# Regional compile (diffusers >= 0.34) — best cold-start/runtime trade
pipe.transformer.compile_repeated_blocks(fullgraph=True)
pipe.vae.decode = torch.compile(
    pipe.vae.decode, mode="max-autotune-no-cudagraphs", fullgraph=True
)

# Warm-up (compile happens here, ~10 s regional or ~5–10 min full max-autotune)
_ = pipe("warmup", num_inference_steps=2).images
```

### 3.3 Wan2.2 video — production recipe

From [morphicfilms/wan2.2_optimizations](https://github.com/morphicfilms/wan2.2_optimizations/blob/main/blog.md) on 8× H100:

```
Baseline (FA2)                              250.70 s (1280×720, 81 frames, 40 steps)
+ FA3                                       195.13 s   (-22%)
+ TF32                                      159.55 s   (-36%)
+ MagCache (E012K2R20)                      150.45 s   (-40%)
+ torch.compile (max-autotune-no-cudagraphs)109.81 s   (-56% / 2.28× over baseline)
```
Key constraint: `fullgraph=False` is required because Wan2.2 has "non compile-friendly distributed operations and dynamic slicing". On GH200 (single-chip), drop the sequence-parallelism step; remaining stack still applies.

Voltage Park (October 2025) reports per-step time on Wan2.2 dropping from **4.67 s → 1.51 s** (3.1×) using FA3 + batched forward + SageAttention + TeaCache on 8× H100. ([Voltage Park blog](https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations))

### 3.4 HunyuanVideo

From [ParaAttention's HunyuanVideo recipe](https://github.com/chengzeyi/ParaAttention/blob/main/doc/fastest_hunyuan_video.md):
- `torch.compile(mode="max-autotune-no-cudagraphs")` or `"max-autotune"` for the transformer
- `torch._inductor.config.reorder_for_compute_comm_overlap = True` when combined with FP8
- Stack: SageAttention + FP8 (torchao `float8_dynamic_activation_float8_weight` on the DiT, `float8_weight_only` on the text encoder) + First-Block-Cache + torch.compile
- **WARN**: their FP8 path is L20-validated, not GH200-validated. Expect numerics surprises.

---

## 4. Inductor flags that matter for diffusion

The four flags below come from huggingface/flux-fast and the diffusers benchmarking gist; they materially affect generated kernel selection on H100/GH200:

```python
import torch._inductor.config as cfg
cfg.conv_1x1_as_mm = True                          # treat 1×1 conv as GEMM (better on Hopper)
cfg.coordinate_descent_tuning = True               # extra autotuner pass
cfg.coordinate_descent_check_all_directions = True
cfg.epilogue_fusion = False                        # avoid fusing pointwise into matmul (helps FA paths)
cfg.shape_padding = True                           # pad odd shapes to tensor-core-friendly sizes
```
Source: [huggingface/flux-fast](https://github.com/huggingface/flux-fast), [Modal Flux example](https://modal.com/docs/examples/flux), [sayakpaul benchmarking gist](https://gist.github.com/sayakpaul/91fa328e949c71dc4420ebb50eb35ca3).

Cache locations (set these for autoscaling):
- `TORCHINDUCTOR_FX_GRAPH_CACHE=1`
- `TORCHINDUCTOR_CACHE_DIR=/path/persistent`
- `TRITON_CACHE_DIR=/path/persistent`
- `VLLM_DISABLE_COMPILE_CACHE=0` (default; keep it on)

---

## 5. Channels-last memory format

Universal win on Hopper for convolutional diffusion (SDXL UNet) and for the VAE in DiT pipelines:

```python
pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
```
Diffusers official guidance for SDXL ([HF docs](https://huggingface.co/docs/diffusers/main/en/optimization/fp16)). For pure DiTs (Flux, Wan, Hunyuan) the transformer benefits less (it's matmul-bound, not conv) but VAE still does — apply to VAE always, to the DiT cheaply (it's free if compatible).

---

## 6. Benchmark numbers (cite per row)

All numbers are H100 SXM unless otherwise stated. GH200 ≈ H100 for compute-bound diffusion (same SM count, same clocks at 700W); slightly faster on bandwidth-bound LLM decode due to NVLink-C2C.

### Diffusion — image
| Model | Config | Time / image | Source |
|---|---|---|---|
| Flux.1-Dev (1024², 28 steps) | bf16 baseline | 6.7 s | [PyTorch blog 2025-07-17](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/) |
| Flux.1-Dev | +`compile(fullgraph=True)` | 4.5 s (1.5×) | same |
| Flux.1-Dev | +`compile_repeated_blocks` (regional) | 4.5 s (1.5×, but 9.6 s compile vs 67 s) | same |
| Flux.1-Dev | +NF4 quant +compile | 5.0 s @ 15 GB | same |
| Flux.1-Dev | +torchao fp8dqrow (float8 dynamic) +compile | **2.97 s** | [diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao) |
| Flux.1-Dev (28 steps) | flux-fast (compile + FA3 + float8 + fused QKV + channels-last) | **~3.0 s (2.5×)** | [huggingface/flux-fast](https://github.com/huggingface/flux-fast) |
| Flux.1-Schnell (4 steps) | flux-fast | **~1.0 s** | same |
| Flux.1-Schnell | Modal recipe (torch 2.5, max-autotune, channels-last) | ~700 ms | [Modal Flux example](https://modal.com/docs/examples/flux) |
| Flux.1-Dev | Modal 3× recipe | 6.75 s → 2.25 s | [Modal Flux 3× faster](https://modal.com/blog/flux-3x-faster) |
| SDXL (30 steps, batch 1) | TensorRT FP8 (NOT torch) on H100 | 1.478 s | [Baseten](https://www.baseten.co/blog/40-faster-stable-diffusion-xl-inference-with-nvidia-tensorrt/) |

### Diffusion — video (Wan2.2, 1280×720, 81 frames, 40 steps, 8× H100)
| Stage | Time | Source |
|---|---|---|
| Baseline FA2 | 250.7 s | [morphicfilms](https://github.com/morphicfilms/wan2.2_optimizations/blob/main/blog.md) |
| + FA3 | 195.1 s | same |
| + TF32 | 159.6 s | same |
| + MagCache | 150.5 s | same |
| + torch.compile (max-autotune-no-cudagraphs) | **109.8 s** (2.28×) | same |

Per-step (Voltage Park, 8× H100): 4.67 s → 1.51 s (3.1×). ([Voltage Park](https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations))

### HunyuanVideo (single 720p 5-s clip)
- H100 reference: 60–90 s end-to-end; H200 fits batch-2 ⇒ ~½ cost/clip. ([Spheron 2026 best-GPU guide](https://www.spheron.network/blog/best-gpu-for-ai-inference-2026/))
- "Optimized workflow" on H100 SXM: 18.8 s, ~50% improvement vs baseline. ([instasd HunyuanVideo testing](https://www.instasd.com/post/hunyuanvideo-performance-testing-across-gpus))

### LLM (Llama 3.1 8B decode, batch 1, H100)
- `torch.compile(mode="reduce-overhead")` + CUDA graphs: **1.65×** at batch 1, **1.16×** at batch 32 over eager. ([Spheron PyTorch 2.6 LLM guide, 2026-04-27](https://www.spheron.network/blog/torch-compile-cuda-graphs-llm-inference-pytorch-2-6/))

### FlashAttention-3 vs FA2 (Hopper, raw)
- FP16/BF16: 1.5–2.0× over FA2, ~740 TFLOPS, 75% util on H100. FP8: ~1.2 PFLOPS (but see §7 on FP8 caution). ([PyTorch blog FA3](https://pytorch.org/blog/flashattention-3/))

### FlashAttention-4 (Blackwell-first)
- B200: ~1,600 TFLOPS / 71% util / up to 2.7× over predecessors.
- **Hopper H200**: 1.30–1.65× over Triton flex-FA backend — *not* the same speedup over FA3 itself. ([PyTorch blog FlexAttention+FA4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/), [Lambda](https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet))

### MLPerf v4.1 GH200 (NVIDIA official)
- GH200 delivers up to 1.4× per-accelerator vs H100 SXM across generative-AI MLPerf workloads (LLM + SDXL). ([NVIDIA blog](https://developer.nvidia.com/blog/nvidia-gh200-grace-hopper-superchip-delivers-outstanding-performance-in-mlperf-inference-v4-1/))

---

## 7. FP8 — flag and source

Our local observation: FP8 broken on the Wan/GH200 stack. This is consistent with multiple independent reports:

1. **vllm-omni Issue #2728 (April 2026)**: "fp8 online quantization produces catastrophic LPIPS regression across diffusion transformers (Z-Image, FLUX.1-dev, Qwen-Image)" — reproducible on **both H100 and B200**, indicating the online-quant calibration path, not the silicon. LPIPS 0.8–1.0 = perceptually unrelated to bf16 reference. Diffusion activations have much larger outliers than LLM activations; AdaLayerNorm / timestep embeddings / final projection need ignored_layers. ([Issue #2728](https://github.com/vllm-project/vllm-omni/issues/2728))
2. **kijai/ComfyUI-WanVideoWrapper Issue #1554 (2026)**: SageAttention noise artifacts on H100 Hopper, fine with SDPA. ([Issue #1554](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1554))
3. **thu-ml/SageAttention Issue #221**: Wan FP8 + SageAttention → black output. Workaround: FP16 + Sage, or FP8 alone without Sage. ([Issue #221](https://github.com/thu-ml/SageAttention/issues/221))

**Trusted FP8 recipes (use bf16 elsewhere)**:
- Flux.1-Dev *transformer linear layers only*, `float8_dynamic_activation_float8_weight` (torchao, calibrated by Sayak Paul's `flux-fast`). Yields ~1.54× speedup vs bf16 with claimed parity quality. ([flux-fast README](https://github.com/huggingface/flux-fast), [diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao))
- FP8 text encoders via `float8_weight_only` (lossy but tolerable).
- NF4 (bitsandbytes) on Flux is more conservative quality-wise than FP8 dynamic.

**For Wan2.x on GH200 default bf16 + FA3.** Quantization gains there come from INT8 SageAttention (separate from FP8 weights) — and even that occasionally regresses (see Issue #1554 above).

---

## 8. Custom Triton / CUDA kernels — what's worth writing

In May 2026 on GH200, the kernels worth hand-rolling are vanishingly few because:
- Attention: FA3 saturates 75% of fp16 flops on H100. You will not beat this.
- GEMM: cuBLASLt + Inductor autotune already pick optimal Hopper kernels with TMA + wgmma.
- Layernorm / RMSNorm / RoPE: HF transformers ships fused Triton kernels; Inductor codegens equivalent.

Worth writing custom Triton for:
- Custom score-mods that don't fit FlexAttention's API.
- Fused diffusion-specific ops (RoPE-2D + AdaLN-shift in a single pass).
- Sparse/block-sparse attention with novel patterns.

Register custom CUDA ops with `torch.library.custom_op` (PyTorch 2.6+) so they co-exist with `torch.compile`/Dynamo without graph breaks. ([Spheron 2026 PyTorch 2.6 guide](https://www.spheron.network/blog/torch-compile-cuda-graphs-llm-inference-pytorch-2-6/))

---

## 9. Recipe cheatsheet by workload

| Workload | Compile mode | Attention | Memory format | Quant | Compile API |
|---|---|---|---|---|---|
| LLM decode (Llama, Qwen, Mistral) | `reduce-overhead`, `fullgraph=True` | FA3 (`attn_implementation="flash_attention_3"`) | default | torchao INT8/FP8 dynamic OR bf16 | `model.forward = torch.compile(...)` + static KV cache |
| LLM prefill (long context) | `max-autotune` | FA3 | default | bf16 | same |
| SDXL UNet (1024²) | `max-autotune` or `max-autotune-no-cudagraphs` | SDPA → FA2/FA3 auto | channels_last on UNet+VAE | INT8 dynamic via torchao (safe) | `pipe.unet = torch.compile(...)` |
| Flux.1-Dev (1024²) | `max-autotune` | FA3 via SDPA | channels_last VAE; transformer free | FP8 dynamic on linears (flux-fast recipe) | `pipe.transformer.compile_repeated_blocks(fullgraph=True)` |
| Wan2.2 (1280×720) | `max-autotune-no-cudagraphs`, `fullgraph=False` | FA3 (+ optional SageAttention) | default | **bf16 (FP8 BROKEN on this stack)** | `--compile True --compile_mode "max-autotune-no-cudagraphs"` |
| HunyuanVideo | `max-autotune-no-cudagraphs` | SageAttention + FA3 | default | FP8 with caution; bf16 safe | ParaAttention recipe |
| Stable Video Diffusion | `max-autotune-no-cudagraphs` | FA3 | channels_last | bf16 | `pipe.unet = torch.compile(...)` |

---

## 10. Confidence & Gaps

### High confidence
- PyTorch 2.7/2.8 + Triton 3.7 + FA3 from-source is the working stack on GH200 aarch64. ([gist install recipe](https://gist.github.com/Dominoer/790091cc9e35c079c3251bf10492279e), official repo, PyTorch release notes).
- Regional compile (`compile_repeated_blocks`) is the diffusers-recommended default since 0.34 (2025). Numbers verified in the PyTorch blog.
- FP8 has real, documented breakage on diffusion transformers in vLLM-omni (April 2026 issue), independent of Hopper-vs-Blackwell. Default to bf16.
- FA3 is the right attention kernel for GH200 in May 2026; FA4 is Blackwell-first and not a meaningful Hopper win yet.
- Inductor flags from flux-fast/Modal are reproduced verbatim across 3+ independent sources.

### Medium confidence
- Direct GH200 (vs H100 SXM) diffusion benchmark numbers are scarce in public — I'm extrapolating from H100 SXM. The architectures are identical SM count/clocks, but TDP cap (700 W on GH200 NVL versions) and LPDDR5X-vs-HBM3-only memory mix can shift things. NVIDIA's MLPerf v4.1 "1.4× per accel" is best-case.
- FA3-via-PyTorch-SDPA dispatch: as of PyTorch 2.7/2.8, SDPA's `FLASH_ATTENTION` backend dispatches to FA2 in the upstream binary; FA3 is reached via `flash_attn_func` or via transformers `attn_implementation="flash_attention_3"` (transformers ≥ 4.50). Whether 2.8 fully wires FA3 into SDPA on Hopper is something I'd want to verify by reading the actual symbol PyTorch exports. Flagging.

### Gaps / unknowns
- No public Wan2.2 numbers on single GH200 (everything I found is 8× H100 or A100). A round-2 task: actually benchmark Wan2.2 on a single GH200 with the morphicfilms stack minus sequence parallelism.
- No reliable HunyuanVideo H100/H200 number — Spheron quotes 60–90 s/clip without methodology; ParaAttention's optimized numbers are on L20 PCIe.
- xFormers on aarch64: I confirmed build flags but did not verify it doesn't regress vs SDPA+FA3 on GH200 in 2026 — historical wisdom says xFormers is no longer worth it on Hopper, but I'd want a fresh measurement.
- FA4 on Hopper-specifically: claims of FA4 working on Hopper exist (PyTorch blog explicitly says "Hopper H200 — full backend support, 1.30–1.65× speedups"), but I haven't seen anyone post FA3 vs FA4 on H100 head-to-head with a third-party repro. Treat as "FA4 on Hopper ≈ FA3 on Hopper, ±10%" until proven otherwise.
- PyTorch 2.11+ promises CUDA-on-PyPI for aarch64 by default; release date TBD, post-May 2026.

### Recommended round-2 follow-ups
1. Single-GH200 measured numbers for Flux.1-Dev, SDXL, Wan2.2 5B/14B, HunyuanVideo. Compare to H100 SXM 80 GB. Note thermal/TDP cap.
2. PyTorch 2.8 SDPA → FA3 dispatch verification (read `aten::scaled_dot_product_attention` kernel registry, not just docs).
3. End-to-end FA3 vs FA4 micro-bench on H100 SM_90a (q_len = 1, 1k, 4k, 16k; head_dim = 64, 128).
4. Calibrated FP8 recipe for Wan2.2 — what's the minimal `ignored_layers` set that makes outputs not collapse?
5. Cross-check against Round 1's vLLM/SGLang/TensorRT-LLM domain reports (#01–#05) — overlap is intentional; flag where numbers disagree.

---

## Sources (chronological where dates known)

- 2024-07 — [FlashAttention-3 paper](https://arxiv.org/abs/2407.08608) / [PyTorch blog](https://pytorch.org/blog/flashattention-3/)
- 2024+ ongoing — [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- 2025-04-23 — [PyTorch 2.7 release](https://pytorch.org/blog/pytorch-2-7/)
- 2025-06-18 — [Modal: Run FLUX.1-dev 3× faster](https://modal.com/blog/flux-3x-faster)
- 2025-07-17 — [PyTorch blog: torch.compile and Diffusers hands-on](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)
- 2025-08-06 — [PyTorch 2.8 RC](https://dev-discuss.pytorch.org/t/pytorch-2-8-final-rc-available/3144)
- 2025-08-20 — [vLLM torch.compile blog](https://vllm.ai/blog/2025-08-20-torch-compile)
- 2025 — [Dominoer GH200 install gist (FA3 / Triton / xFormers / bitsandbytes)](https://gist.github.com/Dominoer/790091cc9e35c079c3251bf10492279e)
- 2025 — [huggingface/flux-fast repo](https://github.com/huggingface/flux-fast)
- 2025 — [diffusers 0.34 release (regional compile)](https://github.com/huggingface/diffusers/releases/tag/v0.34.0)
- 2025 — [diffusers 0.35 release (Wan 2.2, Flux Kontext)](https://github.com/huggingface/diffusers/releases/tag/v0.35.0)
- 2025 — [sayakpaul/diffusers-torchao](https://github.com/sayakpaul/diffusers-torchao)
- 2025-10 — [Voltage Park: Wan2.2 4.67s → 1.51s/step](https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations)
- 2025+ — [morphicfilms Wan2.2 optimizations](https://github.com/morphicfilms/wan2.2_optimizations/blob/main/blog.md)
- 2025+ — [chengzeyi/ParaAttention HunyuanVideo doc](https://github.com/chengzeyi/ParaAttention/blob/main/doc/fastest_hunyuan_video.md)
- 2026-03-05 — [PyTorch blog FlexAttention + FlashAttention-4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
- 2026-03 — [Lambda: FlashAttention-4 on Blackwell](https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet)
- 2026-03 — [Modal reverse-engineer FA4](https://modal.com/blog/reverse-engineer-flash-attention-4)
- 2026-04 — [vllm-omni Issue #2728 — FP8 catastrophic regression](https://github.com/vllm-project/vllm-omni/issues/2728)
- 2026-04-27 — [Spheron: torch.compile + CUDA graphs PyTorch 2.6 LLM guide](https://www.spheron.network/blog/torch-compile-cuda-graphs-llm-inference-pytorch-2-6/)
- 2026 — [Spheron: FlashAttention-4 on Blackwell guide](https://www.spheron.network/blog/flashattention-4-blackwell-gpu-cloud-guide/)
- NVIDIA — [MLPerf v4.1 GH200 results](https://developer.nvidia.com/blog/nvidia-gh200-grace-hopper-superchip-delivers-outstanding-performance-in-mlperf-inference-v4-1/)
- NVIDIA — [NGC PyTorch container catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- PyTorch docs — [sdpa_kernel context manager](https://docs.pytorch.org/docs/2.12/generated/torch.nn.attention.sdpa_kernel.html)
- HF docs — [Accelerate inference (channels-last, SDPA)](https://huggingface.co/docs/diffusers/main/en/optimization/fp16)
- [pytorch/pytorch Issue #160162 — aarch64 GPU wheel on PyPI](https://github.com/pytorch/pytorch/issues/160162)
- [Dao-AILab/flash-attention Issue #2036 — GH200 install failure](https://github.com/Dao-AILab/flash-attention/issues/2036)
- [thu-ml/SageAttention Issue #221 — Wan FP8 black output](https://github.com/thu-ml/SageAttention/issues/221)
- [kijai/ComfyUI-WanVideoWrapper Issue #1554 — SageAttention noise on Hopper](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1554)
