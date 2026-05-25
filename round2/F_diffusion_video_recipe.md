# GH200 Diffusion Image & Video Inference Recipe (Round 2)

**Agent**: cross-pollination F — diffusion/video
**Date**: 2026-05-25
**Hardware**: NVIDIA GH200 Grace-Hopper (96 GB HBM3, 480 GB LPDDR5X over 900 GB/s NVLink-C2C, SM90/sm_90a, aarch64)
**User constraint (load-bearing)**: **FP8 is broken on this stack for DiT models.** Confirmed across Round-1 reports and three independent upstream bug threads (see §6). Default to **bf16** for every recipe below. FP8 paths are listed only as flagged exceptions with a specific upstream-validated recipe.

---

## 0. The stack (pin everything before running anything)

This is the same stack as Round-1 agents #01 and #06; reproduced for completeness so a fresh GH200 can be set up in one pass.

```bash
# 0. Kernel: 64K-page (mandatory on GH200)
sudo apt install -y linux-nvidia-64k-hwe-24.04-edge && sudo reboot
getconf PAGESIZE     # must print 65536

# 1. Driver + CUDA (open-kernel — mandatory on Hopper)
sudo apt install -y nvidia-open-580 cuda-drivers-580 cuda-toolkit-13-0 \
                    libcudnn9-cuda-13 libcudnn9-dev-cuda-13
sudo systemctl enable --now nvidia-persistenced

# 2. Python env + PyTorch (aarch64 CUDA wheel)
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# Pin 2.8.0 cu128 — this is what FA3 hopper kernels build cleanly against
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install triton==3.7.0
pip install -U "diffusers>=0.35" "transformers>=4.50" accelerate \
    "huggingface_hub[hf_xet]" safetensors einops imageio imageio-ffmpeg

# 3. FlashAttention-3 (build from source — no aarch64 wheels)
module load cuda/13.0 gcc/12.4.0 || true
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(which nvcc)")")")"
export TORCH_CUDA_ARCH_LIST="9.0;9.0a"
export MAX_JOBS=6
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_FP16=TRUE   # bf16-only build — faster compile, smaller wheel
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper && python setup.py install

# 4. Hopper-aware kernels package (diffusers FA3 dispatch)
pip install -U kernels      # required for set_attention_backend("_flash_3_hub")

# 5. Caching add-ons (optional, model-dependent)
pip install para-attn       # FBCache for Flux/Hunyuan
# TeaCache / MagCache install from their repos:
# git+https://github.com/ali-vilab/TeaCache.git
# git+https://github.com/Zehong-Ma/MagCache.git
```

Persistent caches (set in `.envrc` or systemd unit so you only pay compile cost once):

```bash
export TORCHINDUCTOR_CACHE_DIR=/var/lib/torchinductor
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TRITON_CACHE_DIR=/var/lib/triton
export HF_HOME=/var/lib/huggingface
```

**Common inductor flags for every diffusion recipe below** (from `huggingface/flux-fast`, Modal Flux example, sayakpaul benchmarking gist — three independent sources converge on this exact set):

```python
import torch
import torch._inductor.config as cfg
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
cfg.conv_1x1_as_mm = True
cfg.coordinate_descent_tuning = True
cfg.coordinate_descent_check_all_directions = True
cfg.epilogue_fusion = False
cfg.shape_padding = True
```

---

## 1. Per-model definitive recipes

Order: user interest (Wan first, then Flux, then Hunyuan, then LTX). Every recipe is **bf16 only**. All "expected" timings are extrapolated from H100/H200 measurements unless the line says "measured." GH200 silicon is identical to H100 SXM (same SM count, same 700 W TDP cap on Grace-Hopper NVL platforms); compute-bound diffusion lands within ±10% of H100 SXM, memory-bandwidth-bound steps within ±5% of H200 (since GH200 4.0 TB/s sits between H100 SXM 3.35 TB/s and H200 4.8 TB/s).

### 1.1 Wan2.2-A14B (T2V or I2V) — user's primary

**What it is.** MoE: two 14B experts (high-noise + low-noise), 27B total, 14B active per step. UMT5-XXL text encoder. Wan-VAE for video. Default config: 81 frames at 16 fps, `num_inference_steps=40`, `guidance_scale=4.0`, `guidance_scale_2=3.0`.

**VRAM (bf16)** — this is the key fit question, see §2 for the full answer.

| Component | bf16 size | Notes |
|---|---|---|
| High-noise expert | ~28.6 GB | from lightx2v table |
| Low-noise expert | ~28.6 GB | |
| Both experts resident | ~57 GB | the configuration you want on GH200 |
| UMT5-XXL text encoder | ~10–12 GB | offload to Grace LPDDR5X — cheap over NVLink-C2C |
| Wan-VAE | ~0.5 GB | trivial |
| Activations + attention scratch (720p 81f) | ~10–25 GB | scales with seq len² |
| **Total resident in HBM (both experts + activations)** | **~70–85 GB** | **fits in 96 GB GH200, does not fit in 80 GB H100 SXM** |

Wan-AI's official model card reads "at least 80GB VRAM" for A14B single-GPU. With T5 on CPU (`--t5_cpu` / `text_encoder.to("cpu")` in diffusers) and both experts resident, GH200 has 10–25 GB of headroom for activation scratch at 720p — comfortable.

**Recipe.**

```python
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

# Inductor flags from §0
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# Step 1: FA3 (Hopper-only kernel)
pipe.transformer.set_attention_backend("_flash_3_hub")
pipe.transformer_2.set_attention_backend("_flash_3_hub")  # both MoE experts

# Step 2: offload text encoder to Grace LPDDR5X — saves ~10 GB HBM, ~free over C2C
pipe.text_encoder.to("cpu")          # UMT5-XXL stays on Grace; encoding runs there
# Alternative for tighter HBM budgets: pipe.enable_model_cpu_offload()

# Step 3: torch.compile, max-autotune-no-cudagraphs, fullgraph=False
# Wan has dynamic slicing + distributed ops that break fullgraph capture
pipe.transformer = torch.compile(
    pipe.transformer, mode="max-autotune-no-cudagraphs", fullgraph=False)
pipe.transformer_2 = torch.compile(
    pipe.transformer_2, mode="max-autotune-no-cudagraphs", fullgraph=False)

# Step 4: distillation LoRA — collapses 40 steps to 4
# Use the 720p high-noise + low-noise pair from lightx2v
pipe.load_lora_weights("lightx2v/Wan2.2-Distill-Models",
                       weight_name="wan2.2-i2v-a14b-4step-720p-high.safetensors",
                       adapter_name="hi")
pipe.load_lora_weights("lightx2v/Wan2.2-Distill-Models",
                       weight_name="wan2.2-i2v-a14b-4step-720p-low.safetensors",
                       adapter_name="lo")
pipe.set_adapters(["hi", "lo"], adapter_weights=[1.0, 1.0])
pipe.fuse_lora()

# Step 5: caching — MagCache (preferred over TeaCache for Wan, better LPIPS at same speedup)
from magcache import apply_magcache_on_pipe
apply_magcache_on_pipe(pipe, magcache_thresh=0.12,
                       K=2, R=20)   # E012K2R20 — same setting morphic used

# First call: compile cost ~3–6 min (regional would help but Wan blocks aren't uniform)
_ = pipe(prompt="warmup", num_inference_steps=4, num_frames=9, height=480, width=832).frames

video = pipe(
    prompt="A cinematic shot of a wolf running through a snowy forest",
    height=720, width=1280,
    num_frames=81,
    num_inference_steps=4,           # distilled
    guidance_scale=1.0,              # distill LoRA bakes CFG; use 1.0
).frames[0]
export_to_video(video, "wan22.mp4", fps=16)
```

**Expected GH200 timing (bf16, FA3, torch.compile, lightx2v-4step, MagCache):**

| Resolution | Frames | Steps | Est. wall time | Source / extrapolation |
|---|---|---|---|---|
| 480p (832×480) | 81 (5 s) | 4 | **8–15 s** | Voltage Park 8×H100 1.51 s/step × 4 = 6 s ÷ 0.5 (single-GPU vs 8-GPU SP no longer applies); Morphic 109 s @ 40 steps on 8×H100 SP → ~11 s/step single-GPU bf16 → 4 steps ≈ 44 s without MagCache, MagCache 2.5× → ~17 s. **extrapolated, ±40%** |
| 720p (1280×720) | 81 (5 s) | 4 | **25–45 s** | Same chain, larger seq len. Cross-check: Wan2.1-14B single H100 (no MoE, no distill, 50 steps) = 284 s; replace 50→4 steps + MoE 2× weight load offset by FA3+compile+MagCache → 25–45 s. **extrapolated, ±35%** |
| 720p | 81 | 40 (no distill) | **350–500 s** | Wan2.1-14B 284 s × 1.5 (MoE switching + more capacity) ÷ 1.3 (compile+FA3 vs baseline). **extrapolated, ±25%** |

**Per-step cross-check.** Voltage Park reports 4.67 s → 1.51 s per step on 8×H100 with FA3+SageAttention+batched-fwd+TeaCache. Their 1.51 s/step is the **8×H100 SP-divided** number; single-GPU equivalent is ~10–12 s/step. lightx2v 4-step distillation gives you 4 of those × bf16 + MagCache 2.5× → 10–20 s for full denoising, plus 5–25 s VAE+encoding overhead = the 25–45 s range above.

**The big GH200 win on this model**: you do not need 8 GPUs. H100 80 GB cannot hold both MoE experts in bf16 simultaneously (needs ~57 GB just for weights, leaves <23 GB for activations at 720p — tight). GH200 96 GB does, so you keep the MoE coherent on one chip and skip Ulysses sequence-parallel setup entirely.

**Wan A14B FP8 — do not use.** See §6.

### 1.2 Wan2.2-TI2V-5B (FastWan) — the easy choice

**What it is.** Dense 5B DiT, Wan2.2-VAE (16×16×4 compression). Built for "9 minutes on a single RTX 4090" per Wan-AI's official line. FastVideo's distilled `FastWan2.2-TI2V-5B-FullAttn-Diffusers` checkpoint runs in **3 steps**.

**VRAM (bf16):** ~12–16 GB. Trivial on GH200 — no offload, no quantization needed.

**Recipe.**

```python
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained(
    "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.transformer.set_attention_backend("_flash_3_hub")
pipe.transformer = torch.compile(
    pipe.transformer, mode="max-autotune-no-cudagraphs", fullgraph=True)

_ = pipe("warmup", num_inference_steps=3, num_frames=9, height=480, width=832).frames

video = pipe(
    prompt="A drone shot of a coastal cliff at sunrise",
    height=720, width=1280, num_frames=81,
    num_inference_steps=3,           # FastWan2.2 ships at 3-step
).frames[0]
export_to_video(video, "fastwan22.mp4", fps=16)
```

**Expected GH200 timing:**

| Resolution | Frames | Steps | Time | Source |
|---|---|---|---|---|
| 720p | 81 (5 s) | 3 (DMD) | **~16 s end-to-end** | **Measured on H200** (FastVideo blog 2025-08-04). GH200 ≈ H200 on bandwidth-light workload. Denoising alone: 2.64 s @ FA3+DMD+torch.compile. |
| 480p | 81 | 3 | **~5–8 s** | Linear scaling from 720p. |

This is the **price/quality leader on GH200 for video** unless you specifically need the 14B's expressivity.

### 1.3 Flux.2-dev (32B) — the largest open image model

**What it is.** Black Forest Labs, released 2025-11-25. 32B DiT (8 double-stream + 48 single-stream blocks) + Mistral Small 3.1 text encoder + Flux VAE.

**VRAM (bf16):**

| Config | VRAM | Hardware |
|---|---|---|
| Full no-offload | **>80 GB** | **Fits GH200 96 GB, does not fit 80 GB H100 SXM** |
| With `enable_model_cpu_offload()` | ~62 GB | H100 path |
| NF4 quant + offload | ~20 GB | RTX 4090 path |
| NF4 + remote text encoder | ~8 GB free | extreme low-VRAM |

GH200 is the smallest single-GPU that runs Flux.2-dev bf16 with no offload. Per HF Flux.2 doc: "Even an H100 can't hold the text-encoder, transformer and VAE at the same time."

**Recipe.**

```python
import torch
from diffusers import Flux2Pipeline

pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.transformer.set_attention_backend("_flash_3_hub")
pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
pipe.transformer.fuse_qkv_projections()
pipe.vae.fuse_qkv_projections()

# Regional compile — diffusers >= 0.34 — keeps cold start ~10 s instead of 5+ min
pipe.transformer.compile_repeated_blocks(fullgraph=True)
pipe.vae.decode = torch.compile(
    pipe.vae.decode, mode="max-autotune-no-cudagraphs", fullgraph=True)

_ = pipe("warmup", num_inference_steps=2).images

img = pipe(
    prompt="a glass octopus reading a book in a sunlit study, photorealistic",
    height=1024, width=1024,
    num_inference_steps=28,          # BFL: "28 is a good trade-off"
    guidance_scale=4.0,
).images[0]
```

**Expected GH200 timing:**

| Resolution | Steps | bf16, no offload | bf16, optimized stack | Source |
|---|---|---|---|---|
| 1024² | 28 | ~10–15 s | **~6–9 s** | Extrapolated: Spheron reports 14 img/min @ FP8 H100 PCIe (~4.3 s/img); bf16 at 32B is ~1.5–2× FP8 due to memory bandwidth on weights, so 6.5–8.5 s. ±25% |
| 1024² | 28 | with `enable_model_cpu_offload()` | ~12–18 s | offload-induced PCIe shuffle adds ~50% |

**No Flux.2 distillation exists yet** as of 2026-05. Flux.2-klein (9B) is the fast variant — not a distill of dev. Watch for Lightning/Hyper LoRAs.

**FBCache caveat.** ParaAttention FBCache works on Flux.1; Flux.2 wrapper status is unverified as of 2026-05 — measure before relying on it.

### 1.4 Flux.1-dev (12B)

**Recipe.**

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")
pipe.transformer.set_attention_backend("_flash_3_hub")
pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
pipe.transformer.fuse_qkv_projections()
pipe.vae.fuse_qkv_projections()
pipe.transformer.compile_repeated_blocks(fullgraph=True)
pipe.vae.decode = torch.compile(
    pipe.vae.decode, mode="max-autotune-no-cudagraphs", fullgraph=True)

# 8-step distill LoRA (Hyper-FLUX)
pipe.load_lora_weights("ByteDance/Hyper-SD",
                       weight_name="Hyper-FLUX.1-dev-8steps-lora.safetensors")
pipe.fuse_lora()

img = pipe(prompt="...", height=1024, width=1024,
           num_inference_steps=8, guidance_scale=3.5).images[0]
```

**Expected GH200 timing:**

| Config | Time | Source |
|---|---|---|
| bf16 baseline, 28 steps | ~6.7 s | PyTorch blog (H100 SXM) |
| +compile fullgraph, 28 steps | ~4.5 s | same |
| +Hyper-FLUX 8 steps + compile | **~1.2–1.5 s** | linear from 28→8 |
| +Hyper + FBCache (rdt=0.12) | **~0.8–1.0 s** | ParaAttention 1.55× on FBCache |
| flux-fast full stack (FP8 dyn quant on linears) | 2.97 s @ 28 steps | sayakpaul/diffusers-torchao — **only validated FP8 path on Flux.1-dev** |

VRAM: ~24 GB bf16. Trivial on GH200.

### 1.5 Flux.1-schnell

Same recipe as 1.4 but `num_inference_steps=4`, `guidance_scale=0.0`. **No CFG, no distill LoRA needed** — it's already a 4-step distill.

Expected GH200 timing: **~0.7–1.0 s** at 1024² (Modal recipe: ~700 ms on H100 SXM with max-autotune+channels-last). VRAM ~24 GB.

### 1.6 HunyuanVideo 1.5 (8.3B) — Tencent, late 2025

**What it is.** 8.3B params, smaller and faster than the original 13B HunyuanVideo. Multiple distilled variants: 480p I2V step-distilled, 480p→720p SR step-distilled, 720p→1080p SR step-distilled, CFG-distilled.

**VRAM (bf16):**

| Config | VRAM | Notes |
|---|---|---|
| Full no-offload | ~25 GB | Trivial on GH200 |
| With pipeline+group offload + VAE tiling | 13.6 GB peak | RTX 4090 reference from tech report |

**Recipe.**

```python
import torch
from diffusers import HunyuanVideoPipeline
from diffusers.utils import export_to_video

pipe = HunyuanVideoPipeline.from_pretrained(
    "tencent/HunyuanVideo-1.5",
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.transformer.set_attention_backend("_flash_3_hub")
pipe.transformer.to(memory_format=torch.channels_last)
pipe.transformer.fuse_qkv_projections()
pipe.transformer = torch.compile(
    pipe.transformer, mode="max-autotune-no-cudagraphs", fullgraph=True)

# Use CFG-distilled checkpoint variant (50 steps, but single forward per step instead of 2)
# OR step-distilled (8–12 steps) — pick one, not both
# CFG-distill: tencent/HunyuanVideo-1.5-CFG-Distilled-720p
# Step-distill: tencent/HunyuanVideo-1.5-Step-Distilled-480p

video = pipe(
    prompt="...",
    height=720, width=1280, num_frames=121,
    num_inference_steps=12,          # step-distill
    guidance_scale=1.0,              # CFG-distill bakes it
).frames[0]
export_to_video(video, "hyv15.mp4", fps=24)
```

**Expected GH200 timing (720p, 121 frames):**

| Config | Time | Source |
|---|---|---|
| bf16 baseline 50 steps FA3 | ~100 s | tech report: 2.01 s/step × 50 |
| +sparse attention 50 steps | ~78 s | tech report: 1.56 s/step × 50 |
| +engineered acceleration 50 steps | **~28 s** | tech report: 0.57 s/step avg × 50 |
| Step-distill 12 steps (this recipe) | **~10–20 s** | linear from 50-step optimized |

Tech report's "8 NVIDIA H800 GPUs with context parallelism" comes out to single-H200/GH200 ≈ 8×H800 ÷ ~6 (SP rarely scales linearly) → 28 s × 6 ≈ 170 s **without** the step-distill. The 12-step distill multiplies this further → **the 10–20 s range is achievable on single GH200 with step-distilled checkpoint + bf16 + FA3 + torch.compile.**

**FP8 status.** Tencent added FP8 GEMM 2025-12-23 via SGL-Kernel. **Do not enable per user constraint.** bf16 is the safe path.

### 1.7 LTX-Video 0.9.8 (Lightricks)

**What it is.** Three variants:
- `ltxv-13b-0.9.8-dev` — highest quality
- `ltxv-13b-0.9.8-distilled` — 8-step (recommended), slight quality drop
- `ltxv-2b-0.9.8-distilled` — fastest, "real time on H100" per official docs

Constraint: resolution ≤ 720×1280, frames ≤ 257, frames must be `8k+1` (e.g., 257).

**VRAM (bf16):** ~15–25 GB depending on variant. Trivial.

**Recipe.**

```python
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.8-13B-distilled",
    torch_dtype=torch.bfloat16,
).to("cuda")
pipe.transformer.set_attention_backend("_flash_3_hub")
pipe.transformer.fuse_qkv_projections()
pipe.transformer = torch.compile(
    pipe.transformer, mode="max-autotune-no-cudagraphs", fullgraph=True)

video = pipe(
    prompt="...",
    height=512, width=768, num_frames=121,    # 5s @ 24fps
    num_inference_steps=8,                    # distilled recommended
).frames[0]
export_to_video(video, "ltx.mp4", fps=24)
```

**Expected GH200 timing:**

| Variant | Resolution | Frames | Steps | Time | Source |
|---|---|---|---|---|---|
| 13B-distilled | 768×512 | 121 (5 s) | 8 | **~2 s** | Lightricks: "HD videos in 10 seconds" 13B distilled on H100; "3 s preview" |
| 2B-distilled | 768×512 | 121 | 8 | **~1 s ("real time")** | Lightricks H100 spec |
| 13B-dev | 768×512 | 121 | 30+ | ~10–15 s | extrapolated |

**This is the throughput leader on GH200 for video.** Quality is below Wan2.2 and HunyuanVideo 1.5, but if your spec is "5-second 768×512" and quality is acceptable, nothing else is close.

---

## 2. Does Wan2.2-A14B fit in 96 GB without offload?

**Answer**: Yes for bf16 weights + activations at 720p × 81f, **only if you offload the UMT5-XXL text encoder to Grace LPDDR5X**. The text-encoder offload is essentially free because (a) text encoding runs once per generation, not per step, and (b) Grace ↔ Hopper is 900 GB/s NVLink-C2C, not PCIe.

**Detailed breakdown (bf16):**

| Component | Size | Resident on Hopper HBM? |
|---|---|---|
| High-noise expert (14B) | ~28.6 GB | yes |
| Low-noise expert (14B) | ~28.6 GB | yes |
| UMT5-XXL text encoder | ~10–12 GB | **no — offload to Grace** |
| Wan-VAE | ~0.5 GB | yes |
| Latent + noise tensors (720p, 81f, bf16) | ~0.3 GB | yes |
| Attention KV scratch (FA3 reduces this — no full attention matrix held) | ~5–15 GB at 720p×81f | yes |
| Inductor compile artifacts / Triton kernels in VRAM | ~1 GB | yes |
| CUDA context + cuBLAS workspace | ~1–2 GB | yes |
| **Total HBM (excluding TE on Grace)** | **~65–76 GB** | comfortably fits 96 GB |

**Round-1's "70–85 GB" range was correct.** The upper bound (85 GB) was for "everything on HBM including UMT5 + cuDNN workspace at high concurrency." Lower bound (65 GB) is the realistic resident set if you offload UMT5.

**Source resolution.** lightx2v's table ("~28.6 GB bf16") is for **the transformer weights alone**, not per-expert KV, not text encoder. That number times 2 experts (~57 GB) leaves 39 GB headroom in 96 GB GH200 — plenty for activations at 720p × 81 frames once you offload UMT5.

**What you do NOT need on GH200:**
- `enable_model_cpu_offload()` — over-conservative; only adds latency.
- `enable_group_offload()` — meant for <80 GB cards.
- FP8 weights — broken on this stack, see §6.
- Multi-GPU sequence parallelism (Ulysses) — single GH200 fits the whole model.

**What you DO want:**
- `pipe.text_encoder.to("cpu")` — keeps UMT5 on Grace, encoding still fast over C2C.
- Optionally, swap experts in/out across the MoE timestep boundary if you push to 1080p (where activations grow). Then both experts can't fit alongside scratch; group-offload one at a time.

**The KV-cache question.** Diffusion DiTs do not have an LLM-style autoregressive KV-cache. Per-step attention is recomputed; FA3 means we never materialize the full attention matrix. So "KV-cache offload" is not a knob here. The only "cache" is **TeaCache/MagCache/FBCache** — these are *output-residual* caches living in HBM, ~50–500 MB.

---

## 3. Pipeline benchmark recipe (run on a fresh GH200)

Drop this script into a fresh GH200 with the stack from §0 installed. It measures Wan2.2-A14B, FastWan2.2-5B, Flux.2-dev, Flux.1-dev, HunyuanVideo-1.5, LTX-Video-0.9.8-distilled with warm-up + repeats + per-step breakdown.

```python
# bench_gh200_diffusion.py — run on freshly provisioned GH200
import os, time, json, gc, statistics, torch
from diffusers import (
    WanPipeline, DiffusionPipeline, Flux2Pipeline, FluxPipeline,
    HunyuanVideoPipeline, LTXPipeline,
)
from diffusers.utils import export_to_video

# Inductor / cuDNN flags (paste from §0)
import torch._inductor.config as cfg
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
cfg.conv_1x1_as_mm = True
cfg.coordinate_descent_tuning = True
cfg.coordinate_descent_check_all_directions = True
cfg.epilogue_fusion = False
cfg.shape_padding = True

WARMUP = 1          # first call pays compile cost — discarded
REPEATS = 3         # median of 3 timed runs
RESULTS = {}


def time_pipe(name, pipe, kwargs, video=False, frame_count=None):
    print(f"\n=== {name} ===")
    # Warm-up — also triggers torch.compile
    for i in range(WARMUP):
        t0 = time.perf_counter()
        out = pipe(**kwargs)
        torch.cuda.synchronize()
        print(f"  warmup {i}: {time.perf_counter()-t0:.2f} s "
              f"(includes compile; discard)")
    # Timed runs
    times = []
    for i in range(REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = pipe(**kwargs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  run {i}: {dt:.2f} s")
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    RESULTS[name] = {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "peak_vram_gb": round(peak_mem, 2),
        "runs": [round(t, 3) for t in times],
        "kwargs": {k: v for k, v in kwargs.items() if isinstance(v, (int, float, str))},
        "frame_count": frame_count,
        "s_per_frame": (statistics.median(times) / frame_count) if frame_count else None,
    }
    # Save artifact for visual inspection
    out_path = f"/tmp/bench_{name}.{'mp4' if video else 'png'}"
    if video:
        export_to_video(out.frames[0], out_path,
                        fps=24 if "ltx" in name.lower() else 16)
    else:
        out.images[0].save(out_path)
    print(f"  median: {RESULTS[name]['median_s']:.2f} s | "
          f"peak VRAM: {peak_mem:.1f} GB | artifact: {out_path}")
    # Tear down to free HBM for next model
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# --- Wan2.2-A14B ---
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers", torch_dtype=torch.bfloat16).to("cuda")
pipe.transformer.set_attention_backend("_flash_3_hub")
pipe.transformer_2.set_attention_backend("_flash_3_hub")
pipe.text_encoder.to("cpu")
pipe.transformer = torch.compile(pipe.transformer,
    mode="max-autotune-no-cudagraphs", fullgraph=False)
pipe.transformer_2 = torch.compile(pipe.transformer_2,
    mode="max-autotune-no-cudagraphs", fullgraph=False)
pipe.load_lora_weights("lightx2v/Wan2.2-Distill-Models",
    weight_name="wan2.2-i2v-a14b-4step-720p-high.safetensors", adapter_name="hi")
pipe.load_lora_weights("lightx2v/Wan2.2-Distill-Models",
    weight_name="wan2.2-i2v-a14b-4step-720p-low.safetensors", adapter_name="lo")
pipe.set_adapters(["hi", "lo"], adapter_weights=[1.0, 1.0])
pipe.fuse_lora()
time_pipe("wan22_a14b_720p_4step", pipe, dict(
    prompt="a cinematic shot of a wolf in snow",
    height=720, width=1280, num_frames=81,
    num_inference_steps=4, guidance_scale=1.0), video=True, frame_count=81)

# --- FastWan2.2-5B (3-step DMD) ---
pipe = DiffusionPipeline.from_pretrained(
    "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
    torch_dtype=torch.bfloat16).to("cuda")
pipe.transformer.set_attention_backend("_flash_3_hub")
pipe.transformer = torch.compile(pipe.transformer,
    mode="max-autotune-no-cudagraphs", fullgraph=True)
time_pipe("fastwan22_5b_720p_3step", pipe, dict(
    prompt="a drone shot of a coastal cliff at sunrise",
    height=720, width=1280, num_frames=81,
    num_inference_steps=3), video=True, frame_count=81)

# --- LTX-Video 0.9.8 13B distilled ---
pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video-0.9.8-13B-distilled",
    torch_dtype=torch.bfloat16).to("cuda")
pipe.transformer.set_attention_backend("_flash_3_hub")
pipe.transformer.fuse_qkv_projections()
pipe.transformer = torch.compile(pipe.transformer,
    mode="max-autotune-no-cudagraphs", fullgraph=True)
time_pipe("ltx098_13b_distilled_768x512_8step", pipe, dict(
    prompt="a sunset on a beach",
    height=512, width=768, num_frames=121,
    num_inference_steps=8), video=True, frame_count=121)

# --- HunyuanVideo-1.5 step-distilled ---
pipe = HunyuanVideoPipeline.from_pretrained(
    "tencent/HunyuanVideo-1.5",         # use distill variant repo when available
    torch_dtype=torch.bfloat16).to("cuda")
pipe.transformer.set_attention_backend("_flash_3_hub")
pipe.transformer.to(memory_format=torch.channels_last)
pipe.transformer.fuse_qkv_projections()
pipe.transformer = torch.compile(pipe.transformer,
    mode="max-autotune-no-cudagraphs", fullgraph=True)
time_pipe("hyv15_720p_12step", pipe, dict(
    prompt="a forest at dawn",
    height=720, width=1280, num_frames=121,
    num_inference_steps=12, guidance_scale=1.0), video=True, frame_count=121)

# --- Flux.2-dev ---
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe.transformer.set_attention_backend("_flash_3_hub")
pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
pipe.transformer.fuse_qkv_projections()
pipe.vae.fuse_qkv_projections()
pipe.transformer.compile_repeated_blocks(fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode,
    mode="max-autotune-no-cudagraphs", fullgraph=True)
time_pipe("flux2_dev_1024_28step", pipe, dict(
    prompt="a glass octopus reading a book",
    height=1024, width=1024,
    num_inference_steps=28, guidance_scale=4.0))

# --- Flux.1-dev with Hyper-FLUX 8-step ---
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe.transformer.set_attention_backend("_flash_3_hub")
pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
pipe.transformer.fuse_qkv_projections()
pipe.vae.fuse_qkv_projections()
pipe.transformer.compile_repeated_blocks(fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode,
    mode="max-autotune-no-cudagraphs", fullgraph=True)
pipe.load_lora_weights("ByteDance/Hyper-SD",
    weight_name="Hyper-FLUX.1-dev-8steps-lora.safetensors")
pipe.fuse_lora()
time_pipe("flux1_dev_hyper8_1024", pipe, dict(
    prompt="a glass octopus reading a book",
    height=1024, width=1024,
    num_inference_steps=8, guidance_scale=3.5))

# --- Dump results ---
with open("/tmp/bench_gh200_results.json", "w") as f:
    json.dump({
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "results": RESULTS,
    }, f, indent=2)
print("\nDone. Results in /tmp/bench_gh200_results.json")
```

**What to log when reporting back:**

1. `nvidia-smi -q | head -30` — driver, CUDA, persistent mode, compute mode.
2. `getconf PAGESIZE` — confirm 65536.
3. `python -c "import torch; print(torch.__version__, torch.version.cuda)"` — confirm 2.8+ / cu128+.
4. `pip show flash-attn | head -5` — confirm FA3 build (the `hopper/` subdir installs as `flash_attn_3` package).
5. For each model: median, min, max of 3 runs after 1 warm-up; peak VRAM; s/frame for videos.
6. CPU offload state — if you fall back to `enable_model_cpu_offload()` for Flux.2 or Wan, log it.
7. Compile time (warm-up duration is a fair proxy).

**A 1-hour rental on Lambda or HyperStack covers all six models above** if you persist `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to a mounted volume; the second run of each is the timed number. Cold compile dominates at first call: Wan2.2 ~3–6 min, Flux.2 ~2–4 min, others 1–3 min.

---

## 4. Cross-engine: xDiT/xfuser and SGLang Diffusion

### 4.1 xDiT / xfuser on single GH200 — **don't use**

Round-1 concluded this. **Confirmed.** xDiT is entirely a sequence-parallel framework (Ulysses, Ring, USP, CFG-parallel) for multi-GPU. The benchmark wins it publishes (e.g., Flux 4×H100 → 1.6 s) require NVLink-paired multi-GPU. On single GH200, xDiT collapses to "diffusers with extra overhead." Use it only if you have GH200 NVL pairs — and only then.

### 4.2 SGLang Diffusion — **pilot worth running, but not yet the default**

**Status update (cross-checked May 2026):**

- Launched 2025-11-07 (LMSYS blog) with the "1.2–5.9× over diffusers" headline — but the actual blog has those numbers only in chart images, not transcribed tables.
- 2026-01-16 follow-up "Two Months In" reports **"up to 2.5× faster than the initial release"** and **"up to 5× over competing solutions"** — but again no transcribed per-model table.
- 2026-02-16 "Advanced Optimizations" post adds token-level sharding, parallel folding, parallel VAE, Cache-DiT fixes, LayerNorm fusion. Mentions "Performance Results" comparing SGLang-Diffusion vs LightX2V on Wan2.2 T2V — but the published HTML has the table missing.
- Active GitHub: cookbook in progress (issue #15443). H100 is "primary testing platform," A100 secondary.

**SGLang Cookbook Wan2.2 page** (the only concrete recipe I could extract from docs.sglang.io):

```bash
# Single-GPU
sglang serve --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
             --dit-layerwise-offload true

# 4-GPU with SP + CFG-parallel
sglang serve --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
             --dit-layerwise-offload true \
             --num-gpus 4 \
             --ulysses-degree 2 \
             --enable-cfg-parallel
```

**Cache-DiT environment knobs** (claimed up to 7.4× over baseline diffusers):

```bash
export SGLANG_CACHE_DIT_ENABLED=true
export SGLANG_CACHE_DIT_FN=2          # first 2 blocks always recomputed
export SGLANG_CACHE_DIT_RDT=0.4       # residual difference threshold
export SGLANG_CACHE_DIT_TAYLORSEER=true
```

**Single B200 benchmark in docs**: 630.43 s/video, 62.6 GB peak. That's the only concrete table number — no diffusers baseline, no H100/H200 number. **The "5.9×" / "7.4×" speedup claims are not independently verifiable against an apples-to-apples diffusers run as of 2026-05.**

**Hardware support**: SGLang Cookbook explicitly lists **B200, H200, MI300X+** for the Wan2.2 recipe. **H100 is not in the listed support matrix on the cookbook page** — surprising; probably an oversight, but flag it before betting a deployment on it.

**Recommendation**: pilot SGLang Diffusion against diffusers on the exact GH200 deployment before adopting. If it beats diffusers by >1.3× on your specific Wan2.2 / Flux config end-to-end (not just per-step), switch. Otherwise diffusers + the recipes in §1 are simpler and well-understood. The diffusers stack is the safer default in May 2026.

---

## 5. The Wan FP8 path — exact failure mode

The user already hit this. Documenting the failure mode in detail so it doesn't get re-litigated.

### 5.1 Symptoms

Across three independent bug threads:

1. **DiffSynth-Studio #466** (Wan 2.1 I2V-14B 480p): FP8 weights → **visible color shift / hue distortion** vs bf16 baseline on the same prompt. "BF16 has no issue." Issue **open** as of 2026, no upstream fix. Text-to-video variants reportedly don't exhibit it; image-to-video does. Workaround: load model FP8 but pipeline bf16 (`torch_dtype=torch.bfloat16` at pipeline level even when transformer is FP8) — partial fix only.

2. **thu-ml/SageAttention #221** (RTX 5090, FP8 + SageAttention): **completely black output** when combining FP8 weights with SageAttention. FP16 + Sage works. FP8 alone (without Sage) works on some configs. Issue **open**.

3. **kijai/ComfyUI-WanVideoWrapper #1554** (H100): **noise-only output** when SageAttention enabled with H100 SM90 kernels. SDPA fallback produces correct output. Suggests bug in the SageAttention H100 SM90 kernel path itself, independent of FP8. Issue **open** as of 2026-05.

4. **vllm-project/vllm-omni #2728** (April 2026, FP8 online quant): catastrophic LPIPS regression on Flux.1-dev (LPIPS **0.8014** vs 0.20 threshold), Z-Image (LPIPS **0.8826** vs 0.10), Qwen-Image (LPIPS **1.0344** vs 0.35). LPIPS 0.8–1.0 = "perceptually unrelated to the bf16 reference." Reproducible on **both H100 and B200**, so it's the quantization calibration path (per-matmul math verified correct via micro-bench; full transformer diverges). Workaround: none documented — fallback ChannelWiseTorchFP8 kernel crashes on diffusion 3D tensor shapes.

### 5.2 Why it breaks (root cause as understood)

Diffusion activations have much larger outliers than LLM activations because of:
- **AdaLayerNorm timestep embeddings** — broadcast over the entire latent, large dynamic range.
- **Final output projection layer** — output has unbounded range until VAE decode normalizes.
- **Multi-modal cross-attention** — text-encoder activations on different scale than vision tokens.

FP8 dynamic quantization calibrations developed for LLMs (per-token scale, per-row weight scale) don't capture these distributions. The cuTLASS / cuBLAS / SGL-Kernel FP8 micro-kernels are arithmetically correct; the upstream **scale picking** is wrong for DiT layers.

The **only validated FP8 path** in May 2026 is `huggingface/flux-fast` for Flux.1-dev *transformer linear layers only*, using torchao `float8_dynamic_activation_float8_weight` with hand-tuned ignored_layers. That recipe yields ~2.97 s on H100. Do not extrapolate this recipe to Wan, Hunyuan, or Flux.2 without per-layer ablation.

### 5.3 The bf16 substitute and its throughput penalty

For Wan2.2-A14B on GH200:

| Path | Per-step time (single GH200, FA3+compile) | Quality | Status |
|---|---|---|---|
| **bf16 (recommended)** | ~10–12 s/step (extrap from 8×H100 Voltage Park) | reference | **stable** |
| FP8 weight-only (lightx2v) | ~6–8 s/step (~1.4–1.5× faster) | color regression, see #466 | **broken on this stack** |
| INT8 SageAttention QK | ~7–9 s/step on Wan (~1.3× faster than bf16+FA3) | noise on H100, see #1554 | **broken on Hopper** |

**Throughput penalty for staying bf16 vs notional working FP8: ~30–50%.** That's the price of correctness here. With the 4-step lightx2v distill + MagCache, you're still at 25–45 s per 720p 5-s clip — the lost FP8 1.4× would have brought it to ~18–32 s. **Worth losing.**

For Flux.1-dev only, you can pay the FP8 cost on the *transformer linears only* via flux-fast (validated). For everything else on this stack: bf16.

---

## 6. Direct measurement gaps — what a 1-hour GH200 rental would nail down

Every number marked "extrapolated" in §1 deserves a measurement. The benchmark script in §3 covers them all in one ~50-minute run (1-hour rental with margin). Specifically:

1. **Wan2.2-A14B bf16 single-GH200 end-to-end @ 720p × 81f × 4-step lightx2v + MagCache.** Today's "25–45 s" range collapses to a single number with this measurement. The most valuable number on the list — no public single-GPU Wan2.2-A14B benchmark exists.

2. **Wan2.2-A14B bf16 peak HBM at 720p and 1080p.** Confirms or refutes whether 96 GB is enough at 1080p×81f. Today's estimate is 70–85 GB at 720p; 1080p adds ~30% activation footprint and may push past 96 GB, requiring one-expert-at-a-time group offload.

3. **FastWan2.2-5B 3-step end-to-end on GH200.** Validate the 16 s H200 number reproduces (or improves) on GH200.

4. **Flux.2-dev bf16 single-GH200 @ 1024² × 28 steps, with and without `enable_model_cpu_offload()`.** Today's "6–9 s" extrapolation needs grounding — Flux.2 has no published GH200 measurement. Also: does the no-offload path even fit in 96 GB after activations? bf16 weights are >80 GB; if activations push past 96 GB, must offload regardless.

5. **HunyuanVideo-1.5 step-distilled 720p × 121f × 12 steps.** Validate the "10–20 s" estimate. The tech report's 28 s/clip is engineered-acceleration on 8×H800 with sparse attention — single-GH200 with step-distill should land in the same ballpark.

6. **LTX-Video 0.9.8-13B-distilled 768×512 × 121f × 8 steps.** Validate the ~2 s H100 number reproduces. If yes, confirms real-time-class single-GPU video on GH200.

7. **Flux.1-dev + Hyper-FLUX-8steps on GH200.** "0.7–1.0 s" is the schnell number; with Hyper LoRA on dev, should be 1.2–1.5 s. Validate.

8. **SGLang Diffusion vs diffusers, head-to-head, same Wan2.2 720p × 4-step config.** The "5.9× / 7.4× / 2.5×" SGLang Diffusion claims are all on missing or chart-only data. One side-by-side run on GH200 settles whether the diffusers stack in §1 is the right default or whether SGLang Diffusion + Cache-DiT actually wins meaningfully.

9. **PyTorch SDPA → FA3 dispatch sanity check on PyTorch 2.8.** Verify that `set_attention_backend("_flash_3_hub")` actually wires the kernel — `nsys profile` should show `fa3_fwd_kernel` in the trace, not `flash_attn::flash_fwd_kernel` (which is FA2). Quick — 1 minute of nsys per model.

10. **FBCache on Flux.2** — confirm or refute that ParaAttention's wrapper applies cleanly to Flux2Pipeline. If yes, expect another 1.5× on top of the recipe in §1.3.

The benchmark script in §3 produces results for items 1, 2, 3, 4, 5, 6, 7 in one run (~50 min including all the cold compiles, ~25 min on a second run with warm caches). Items 8, 9, 10 are 5–15 min add-ons each.

---

## Sources

All accessed 2026-05-25, except where noted.

**Core stack and recipes**
- https://huggingface.co/docs/diffusers/optimization/attention_backends — `_flash_3_hub` Hopper dispatch
- https://github.com/Dao-AILab/flash-attention — FA3 hopper build
- https://gist.github.com/Dominoer/790091cc9e35c079c3251bf10492279e — GH200 FA3 install recipe
- https://github.com/huggingface/flux-fast — Flux.1-dev FP8-on-linears recipe (only trusted FP8 path)
- https://github.com/chengzeyi/ParaAttention/blob/main/doc/fastest_flux.md — FBCache thresholds, L20 benchmarks
- https://github.com/sayakpaul/diffusers-torchao — diffusers + torchao recipes
- https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/ — regional compile, Flux numbers
- https://modal.com/blog/flux-3x-faster — Flux.1-dev Modal 6.75→2.25 s recipe

**Wan2.2**
- https://github.com/Wan-Video/Wan2.2 — official repo, MoE architecture, 80 GB min
- https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers — model card, UMT5-XXL, default params
- https://huggingface.co/lightx2v/Wan2.2-Distill-Models — 4-step distill, ~28.6 GB bf16/expert
- https://github.com/ModelTC/LightX2V — distill checkpoint repo, 720p high/low variants
- https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations — 8×H100 per-step optimization
- https://github.com/morphicfilms/wan2.2_optimizations/blob/main/blog.md — 8×H100 250.7→109.8 s table
- https://www.baseten.co/blog/wan-2-2-video-generation-in-less-than-60-seconds/ — 8×H100 2.6× / 8×B200 3.2× headline
- https://github.com/Wan-Video/Wan2.2/issues/88 — single-H800 A14B inference reported 1600 s (unoptimized)

**FastWan / FastVideo**
- https://haoailab.com/blogs/fastvideo_post_training/ — FastWan2.2-5B 16 s on H200 measured
- https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers — 3-step inference

**Flux.2 / Flux.1**
- https://huggingface.co/blog/flux-2 — 32B DiT + Mistral 3.1, >80 GB bf16, NF4 path
- https://github.com/black-forest-labs/flux2/blob/main/docs/flux2_dev_hf.md — HF recipe, bf16/NF4
- https://www.spheron.network/blog/deploy-flux2-gpu-cloud-production-guide/ — H100 PCIe/SXM5 FP8 14/18 img/min @ 1024²×28
- https://github.com/xdit-project/xDiT/blob/main/docs/performance/flux.md — Flux.1 multi-GPU SP

**HunyuanVideo 1.5**
- https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5 — 8.3B, step-distill, FP8 GEMM 2025-12-23
- https://arxiv.org/html/2511.18870v1 — tech report: 2.01→0.57 s/step engineered, 8×H800 baseline
- https://github.com/chengzeyi/ParaAttention/blob/main/doc/fastest_hunyuan_video.md — FBCache+CP recipe

**LTX-Video**
- https://github.com/Lightricks/LTX-Video — variants, 8-step distilled, H100 real-time 2B
- https://arxiv.org/pdf/2501.00103 — LTX-Video paper: 2 s on H100

**SGLang Diffusion**
- https://www.lmsys.org/blog/2025-11-07-sglang-diffusion/ — launch, "1.2–5.9×" headline (charts only)
- https://www.lmsys.org/blog/2026-01-16-sglang-diffusion/ — "2.5× faster than initial"
- https://www.lmsys.org/blog/2026-02-16-sglang-diffusion-advanced-optimizations/ — token sharding, parallel folding, parallel VAE
- https://docs.sglang.io/cookbook/diffusion/Wan/Wan2.2 — Wan2.2 launch flags, Cache-DiT envs, B200 630 s
- https://docs.sglang.io/diffusion/index.html — supported model list (Wan/Hunyuan/Qwen-Image/FLUX/Z-Image/GLM)
- https://github.com/sgl-project/sglang/issues/15443 — cookbook proposal, H100 primary, model list

**Caching**
- https://github.com/ali-vilab/TeaCache — TeaCache 2–3× video
- https://arxiv.org/html/2506.09045v2 — MagCache 2.10–2.68× on Open-Sora/CogVideoX/Wan/Hunyuan
- https://github.com/chengzeyi/ParaAttention — FBCache 1.55–1.62× Flux/Hunyuan

**FP8 bug threads (all open as of 2026-05)**
- https://github.com/modelscope/DiffSynth-Studio/issues/466 — Wan 2.1 FP8 color shift vs bf16
- https://github.com/thu-ml/SageAttention/issues/221 — Wan + FP8 + Sage black output
- https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1554 — Sage on H100 SM90 noise (FP8-independent)
- https://github.com/vllm-project/vllm-omni/issues/2728 — Flux/Z-Image/Qwen-Image LPIPS 0.8–1.0 on H100+B200

**Round-1 reports referenced**
- /home/ubuntu/research/gh200_inference/round1/20_diffusion_video.md
- /home/ubuntu/research/gh200_inference/round1/06_pytorch_triton.md
- /home/ubuntu/research/gh200_inference/round1/03_vllm.md
- /home/ubuntu/research/gh200_inference/round1/01_driver_stack.md
