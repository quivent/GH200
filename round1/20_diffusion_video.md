# Diffusion Image & Video Generation on GH200 — Round 1

**Agent**: #20 of 20
**Date**: 2026-05-25
**Hardware target**: NVIDIA GH200 Grace Hopper Superchip — 96 GB HBM3 on Hopper GPU + 432–480 GB LPDDR5X on Grace CPU, joined by 900 GB/s NVLink-C2C cache-coherent unified memory. SM90 architecture (same compute as H100 SXM); FlashAttention-3 supported.

**Hard constraint (user)**: FP8 is broken on this GH200/Wan stack. Always recommend bf16. Any FP8 claim is flagged inline with its source so the user can decide.

---

## 1. GH200 framing for diffusion workloads

Why GH200 matters for this domain (different from LLM serving):

1. **Video latents are huge.** Wan2.2-14B and HunyuanVideo at 720p × 121–129 frames push the latent + activations + KV/attention scratch well past a single H100's 80 GB. GH200's 96 GB HBM3 is the first single-GPU node that runs these in bf16 without aggressive offload.
2. **Coherent CPU offload is "free".** 900 GB/s NVLink-C2C makes Grace LPDDR5X look like a giant slow-tier VRAM. For 32B Flux.2-dev (needs >80 GB bf16 + Mistral 3.1 text encoder + VAE), CPU-offloading the text encoder and idle expert weights stays cheap. (`enable_model_cpu_offload()` / `enable_group_offload()` in diffusers).
3. **Single-GPU, no NVLink island.** xDiT/SGLang Diffusion sequence-parallel speedups are NVLink-bound — GH200 standalone gets no multi-GPU parallelism benefit. So the answer is "biggest fit-in-memory single-GPU pipeline + caching tricks (TeaCache, MagCache, FBCache) + step distillation," not sequence parallelism.
4. **No FP8.** Per user constraint. Hopper supports FP8 tensor cores but multiple recent reports show per-model FP8 quality regressions (LPIPS 0.8–1.0 on Z-Image / FLUX.1-dev / Qwen-Image, color shifts on Wan 2.1). bf16 stays the safe default. NVFP4 is Blackwell-only and not relevant here.

**Default attention backend on GH200**: `set_attention_backend("_flash_3_hub")` — diffusers' FlashAttention-3 dispatch, Hopper-only. Source: https://huggingface.co/docs/diffusers/optimization/attention_backends (accessed 2026-05). Fallback: SageAttention (`sage_hub`, INT8 QK + FP16 PV) — **note** known artifacts on H100 with Wan via ComfyUI-WanVideoWrapper (GitHub issue #1554); test before relying on it.

---

## 2. Per-model recommended pipeline

### 2.1 Image generation

#### FLUX.2-dev (32B DiT + Mistral 3.1 text encoder) — released 2025-11-25
- **Recommended**: `diffusers.Flux2Pipeline` + bf16 + `set_attention_backend("_flash_3_hub")` + `enable_model_cpu_offload()`.
- **VRAM at bf16**:
  - Full no-offload: **>80 GB** → does *not* fit on H100, *does* fit on GH200 (96 GB) with headroom for VAE decode. Source: https://huggingface.co/blog/flux-2
  - With CPU offload (tested on H100): ~62 GB. On GH200, leave offload off if pure latency matters.
- **Throughput (1024×1024, 28 steps, bf16)**: No published GH200 number. H100 reference: torch.compile + bf16 Flux.1-dev compiled gets ~700 ms/image (schnell variant) and ~3–5 s/image for dev at 28 steps. Flux.2-dev is ~2.7× the parameter count of Flux.1; expect **~10–15 s/image** on GH200 single-GPU bf16, 28 steps, without distillation. (Extrapolated — not measured.)
- **Speed-ups available**: `torch.compile(mode="max-autotune-no-cudagraphs")`, First Block Cache (FBCache rdt=0.12) ~1.55×, ParaAttention combination on Flux.1-dev: 26.36 s → 7.56 s on single L20 with FBCache + FP8 DQ (FP8 — do not apply on this stack). Source: https://github.com/chengzeyi/ParaAttention/blob/main/doc/fastest_flux.md
- **Distillation**: Flux.2-klein 9B (released 2026-01-15) is the fast variant. No Lightning/Hyper for Flux.2 yet (as of 2026-05). Flux.1-Lightning/Hyper-Flux1 LoRAs exist for the 12B model.

#### FLUX.1-dev (12B)
- **Recommended**: diffusers + bf16 + FA3 + torch.compile. With Hyper-FLUX-8steps or Flux-Schnell-distill LoRA → 8 steps.
- **VRAM**: ~24 GB bf16, fits easily.
- **Throughput**: ~3–5 s/image at 1024px, 28 steps, single H100/GH200 bf16. With FBCache: ~17 s → ~13 s with cache alone on L20 (extrapolate ~2× faster on GH200). Source: https://github.com/chengzeyi/ParaAttention/blob/main/doc/fastest_flux.md
- **xDiT**: 4×H100 → 1.6 s at 1024px with Ulysses-4 SP. Not relevant for single GH200. Source: https://github.com/xdit-project/xDiT/blob/main/docs/performance/flux.md

#### Stable Diffusion 3.5 Large (8B)
- **Recommended**: diffusers + bf16 + FA3 + torch.compile. bf16 TensorRT gives ~1.7× over PyTorch on SD3.5-Medium per NVIDIA. SD3.5-Large bf16 latency on H100 is typically 2–4 s for 1024×1024, 28 steps. Source: https://blogs.nvidia.com/blog/rtx-ai-garage-gtc-paris-tensorrt-rtx-nim-microservices/
- **VRAM**: ~16 GB bf16, fits easily on GH200.
- **Note**: NVIDIA cites FP8 TensorRT giving 2.3× over bf16 for SD3.5-Large. Per user constraint, do not deploy FP8 — keep bf16.

#### Nunchaku / SVDQuant
- Targets RTX 4090 / 5090 consumer cards (INT4). Not the optimal pipeline for GH200 — you have memory to spare; trading quality for VRAM reduction is unnecessary. **Skip.**
- (FYI: Hopper-series GPUs not explicitly listed as supported targets per Nunchaku repo, accessed 2026-05.)

---

### 2.2 Video generation

#### Wan2.2-T2V-A14B / I2V-A14B (MoE, 14B active / 27B total) — user-tested
- **Recommended**: `diffusers.WanPipeline` + bf16 + FA3 + lightx2v distill LoRA (4-step) + TeaCache or MagCache. Avoid FP8 (user constraint and Wan 2.1 BF16-vs-FP8 color regression issue #466 in DiffSynth-Studio confirms).
- **VRAM at bf16**:
  - Single expert weight ~28.6 GB (lightx2v table). Both experts loaded: ~57 GB. Plus T5 / UMT5 text encoder + VAE + activations → ~70–85 GB at 720p. **Fits in GH200's 96 GB; would not fit on H100.** Source: https://huggingface.co/lightx2v/Wan2.2-Distill-Models
  - With one expert offloaded to Grace via group offload, peak GPU residency drops to ~35 GB. Recommended when generating long clips at 720p.
- **Throughput (5-second, 81 frames, bf16)**:
  - Baseline 8×H100 Wan2.2: 250.7 s for 81-frame 720p (Flash Attention 2, no opts). Per-GPU equivalent ≈ 2000 s — but sequence-parallel doesn't divide cleanly.
  - Voltage Park optimized 8×H100: 187 s → 60 s at 40 steps (3.1× speedup) with batched fwd + Sage attention + TeaCache. Per step: 4.67 s → 1.51 s. Source: https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations
  - Wan2.1-14B single H100 reference (Wan2GP): 480p in 85 s, 720p in 284 s (50 steps). Source: https://www.blog.brightcoding.dev/2025/09/17/open-source-video-generation-for-low-vram-gpus-how-wan2gp-puts-cinematic-ai-in-reach-of-the-gpu-poor/
  - **Best GH200 single-GPU estimate (bf16, FA3, MagCache, lightx2v 4-step distill)**: 480p 5s ≈ **8–15 s**; 720p 5s ≈ **25–45 s**. Distill cuts steps 50 → 4; MagCache adds 2.0–2.7× on top of FA3 baseline. (Extrapolated — explicit GH200 numbers don't exist publicly.)
- **MagCache speedups confirmed**: 2.10×–2.68× on Open-Sora, CogVideoX, Wan 2.1, HunyuanVideo with single-sample calibration. Source: https://arxiv.org/html/2506.09045v2

#### Wan2.2-TI2V-5B (dense 5B) — the easy choice for GH200
- **Recommended**: diffusers + bf16 + FA3 + FastWan2.2-5B distilled checkpoint.
- **VRAM**: ~12–16 GB bf16; trivial on GH200.
- **Throughput**: FastWan2.2-5B **16 s end-to-end for 720p 5-second video on a single H200** (denoising 2.64 s with FA3 + DMD + torch.compile). GH200 ≈ H200 for this workload. Source: https://haoailab.com/blogs/fastvideo_post_training/ (2025-08-04)
- **This is the best price/quality option for a GH200 if you don't need the 14B's expressivity.**

#### Wan2.5 / Wan2.6
- **Status**: Wan2.5 (Oct 2025) is API-only via Alibaba Cloud — **no open weights**. Wan2.6 announced; not open at time of writing. Use Wan2.2 for any open-source GH200 deployment. Source: https://www.mindstudio.ai/blog/what-is-wan-2-5-video-open-source

#### HunyuanVideo (13B, original release)
- **Recommended**: diffusers `HunyuanVideoPipeline` + bf16 + FA3 + TeaCache.
- **VRAM**: ~25.6 GB bf16 for transformer alone; total ~60–70 GB with text encoders + VAE at 720p / 129 frames. Source: https://www.tauceti.blog/posts/turn-your-computer-into-ai-machine-comfyui-video-with-hunyuan/ — "HunyuanVideo at bf16 is 25.6 GB."
- **Throughput (1280×720, 129 frames, 50 steps)**:
  - xDiT single H100: **1904 s** (~32 min). 8×H100: 337 s. Source: https://github.com/xdit-project/xDiT/blob/main/docs/performance/hunyuanvideo.md
  - With ParaAttention FBCache: 3675 s → 2271 s on single L20 (1.62×); 8×L20 with FBCache+CP: 649 s. Source: https://github.com/chengzeyi/ParaAttention/blob/main/doc/fastest_hunyuan_video.md
  - **GH200 single-GPU bf16 + FA3 + TeaCache estimate**: 720p 129-frame ≈ **600–900 s**. With MagCache-fast (2.68×): ~250–400 s.

#### HunyuanVideo-1.5 (8.3B) — released late 2025, recommended over original
- **Recommended**: diffusers + bf16 + FA3 + step-distilled checkpoint (8 or 12 steps).
- **VRAM**: ~14 GB with offload; ~25 GB without. Fits comfortably on GH200.
- **Throughput**: H100 SXM optimized workflow: 18.8 s baseline (after optimizations, ~50% reduction). Step-distilled 480p I2V on RTX 4090: **75 s end-to-end**. GH200 should land **~10–20 s for 720p 5-second video with distillation**. Source: https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
- **FP8 note**: HunyuanVideo-1.5 added FP8 GEMM support 2025-12-23. **Do not enable per user constraint** — stay on bf16.

#### Mochi-1 (10B, Genmo) — 480p only
- **Recommended**: diffusers `MochiPipeline` + bf16 + FA3. Largely surpassed by HunyuanVideo-1.5 and Wan2.2-5B for new deployments.
- **VRAM**: ~42 GB at high quality. Fits on GH200.
- **Throughput**: ~90 s for 50 steps, single H100, 480p. Source: https://www.kdnuggets.com/top-5-open-source-video-generation-models
- **Verdict**: Keep for compatibility; not the default.

#### CogVideoX-5B
- **Recommended**: diffusers `CogVideoXPipeline` + bf16 + FA3. Trained in bf16, runs in bf16 natively.
- **VRAM**: ~12 GB bf16. Trivial.
- **Throughput**: ~90 s for 50 steps, single H100, 720×480, 6 seconds @ 8 fps. Source: https://huggingface.co/zai-org/CogVideoX-5b
- **MagCache**: 2.1× speedup. Source: https://arxiv.org/html/2506.09045v2

#### LTX-Video (Lightricks, 13B-dev / 13B-distilled / 2B-distilled, v0.9.8)
- **Recommended**: diffusers `LTXPipeline` + bf16 + FA3 + distilled checkpoint.
- **VRAM**: ~15–25 GB depending on variant. Trivial on GH200.
- **Throughput**: **2 seconds for 5-second 768×512 24fps on a single H100** (the standout real-time number). Distilled 0.9.6 is 15× faster than non-distilled. Source: https://arxiv.org/pdf/2501.00103 + https://github.com/Lightricks/LTX-Video
- **Verdict**: The fastest open-source video model. If 720p / 5-second is the spec and quality is acceptable, LTX-Video on GH200 is the throughput leader.

#### Open-Sora 2.0
- **Recommended**: diffusers / official repo + bf16 + sequence parallel for 768×768.
- **VRAM**:
  - 256×256, 1×H100: 52.5 GB
  - 768×768, 1×H100: 60.3 GB (near 80 GB ceiling — GH200's 96 GB is much safer)
  - Source: https://github.com/hpcaitech/Open-Sora
- **Throughput (50 steps)**:
  - 256×256: 60 s on 1× GPU; 34 s on 4× GPU.
  - 768×768: 1656 s on 1×H100; 466 s on 4×H100; 276 s on 8×H100.
  - **GH200 single-GPU 768×768 estimate**: ~1500 s baseline; with TeaCache/MagCache ~600–800 s.

---

## 3. Framework comparison — what to pick on GH200

| Framework | Best for | bf16 path? | GH200 notes |
|---|---|---|---|
| **diffusers** | Single-GPU, broadest model support, FA3 dispatch | Yes, default | **Primary recommendation.** `set_attention_backend("_flash_3_hub")` + `torch.compile(mode="max-autotune-no-cudagraphs")`. CPU offload uses NVLink-C2C effectively. |
| **ComfyUI + GGUF** | Memory-constrained / RTX 4090 territory | bf16 supported, GGUF Q8 ≈ bf16 quality | Not optimal on GH200 — you have memory to spare; Q8 GGUF only buys VRAM, not speed. Q8 mostly indistinguishable from bf16 per quality tests. Source: https://medium.com/@furkangozukara/bf16-vs-gguf-fp8-scaled-nvfp4-... |
| **xDiT / xfuser** | Multi-GPU sequence parallelism | bf16 | **Wasted on single GH200.** Single-node multi-GPU (H100 SXM with NVLink) is where it shines. Use only if you have 2× GH200 + NVL switch. |
| **SGLang Diffusion** | Production serving, supports CFG-parallel + USP, 1.2–5.9× speedup over diffusers | bf16 | Newer (2025-11-07). Worth piloting against diffusers baseline for Flux, Wan, Qwen-Image. Source: https://www.lmsys.org/blog/2025-11-07-sglang-diffusion/ |
| **Nunchaku / SVDQuant** | Consumer 4-bit (RTX 4090/5090) | INT4/NVFP4 (Blackwell) | **Don't use on GH200.** Built for VRAM-poor consumer cards. |
| **ParaAttention** | FBCache + Context Parallel on single GPU | bf16 | Useful — FBCache gives 1.55–1.62× on Flux/Hunyuan single-GPU with no parallelism. Source: https://github.com/chengzeyi/ParaAttention |
| **FastVideo / FastWan** | Sparse distillation for Wan family | bf16 | **Strong recommendation for Wan workloads.** 5-s video in 5 s (FastWan2.1-1.3B 480p H200) / 16 s (FastWan2.2-5B 720p H200). Source: https://haoailab.com/blogs/fastvideo_post_training/ |
| **lightx2v** | Wan2.2 distilled LoRAs (4-step), 2× ComfyUI | bf16, FP8, INT8 | Distill LoRA on top of bf16 base is the practical default. Source: https://huggingface.co/lightx2v/Wan2.2-Distill-Models |

---

## 4. Caching & distillation (stack on top of bf16 base)

### Caching (no retraining)
- **TeaCache** — timestep-embedding-aware caching. 2–3× on video diffusion. Source: https://github.com/ali-vilab/TeaCache
- **MagCache** — magnitude-aware, single-sample calibration. 2.10×–2.68× on Open-Sora, CogVideoX, Wan 2.1, HunyuanVideo. Better LPIPS than TeaCache at equal speedup. Source: https://arxiv.org/html/2506.09045v2 (June 2025)
- **First Block Cache (FBCache)** — ParaAttention. 1.55–1.62× on Flux / HunyuanVideo with no quality loss at rdt=0.08–0.12. Source: https://github.com/chengzeyi/ParaAttention
- **Stack order**: bf16 → FA3 → torch.compile → distillation LoRA → caching. Each provides ~1.5–3× independently.

### Step distillation
- **DMD2** — distribution matching. 4-step for SDXL/Pony/Illustrious. Source: https://www.diffus.me/models/dmd2-speed-lora-sdxl-pony-illustrious-dmd2-sdxl-4step-lora
- **lightx2v V2 Wan2.2** — 50 steps → 4 steps with LCM sampler. Source: https://huggingface.co/lightx2v/Wan2.2-Distill-Models
- **Hyper-FLUX-8steps / Flux-Lightning** — 8-step Flux.1. (No Flux.2 distill as of 2026-05.)
- **FastWan sparse distillation** — 1–4 steps. FastWan2.1-14B 720p denoising: 1746 s (FA2 baseline) → 37 s (FA3+DMD) → 13 s (VSA+DMD+torch.compile). Source: https://haoailab.com/blogs/fastvideo_post_training/
- **LCM / T2V-Turbo / MCM** — older 4–8-step pipelines, mostly superseded.
- **Self-Forcing** (LightX2V) — ODE-initialization based step distillation. Source: https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/step_distill.html

### CFG distillation
- HunyuanVideo-1.5 CFG-distilled variants → 2× at 50 steps. Source: https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
- Distill CFG into single forward pass — saves 50% of compute since classifier-free guidance doubles per-step cost otherwise.

---

## 5. Concrete GH200 recipe (single-GPU, bf16, no FP8)

```python
import torch
from diffusers import WanPipeline  # or Flux2Pipeline, HunyuanVideoPipeline, etc.
from diffusers.utils import export_to_video

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

# Hopper FA3 attention
pipe.transformer.set_attention_backend("_flash_3_hub")

# Compile main blocks (skip the VAE — it's small and benefits less)
pipe.transformer = torch.compile(
    pipe.transformer,
    mode="max-autotune-no-cudagraphs",
    fullgraph=True,
)

# Apply step distillation LoRA
pipe.load_lora_weights("lightx2v/Wan2.2-Distill-Models", weight_name="wan2.2-distill-4step.safetensors")
pipe.fuse_lora()

# Apply caching (use TeaCache or MagCache wrappers from their repos)
# from teacache import apply_teacache_on_pipe; apply_teacache_on_pipe(pipe, rel_l1_thresh=0.15)

# CPU offload only if you genuinely need >96 GB; otherwise skip — GH200 fits the model
# pipe.enable_model_cpu_offload()

video = pipe(
    prompt="...",
    height=720, width=1280,
    num_frames=81,
    num_inference_steps=4,           # distilled from 40–50
    guidance_scale=1.0,              # CFG-distilled, or use 5–6 if not
).frames[0]
export_to_video(video, "out.mp4", fps=16)
```

---

## 6. Throughput summary (single-GPU, bf16, GH200 estimates)

| Model | Resolution | Frames/Steps | Baseline | With FA3+compile+cache+distill | VRAM (bf16) |
|---|---|---|---|---|---|
| FLUX.1-dev | 1024² | 28 steps | ~3–5 s | ~0.7–1.5 s | ~24 GB |
| FLUX.2-dev | 1024² | 28 steps | ~10–15 s (est.) | ~5–8 s (est., no distill yet) | ~64 GB |
| SD 3.5 Large | 1024² | 28 steps | ~2–4 s | ~1–2 s | ~16 GB |
| Wan2.2-A14B | 720p×81f | 40–50 steps | ~250+ s | **~25–45 s** (4-step distill+MagCache) | ~70–85 GB |
| Wan2.2-5B (FastWan) | 720p×5s | 1–3 steps | — | **~16 s** (measured H200) | ~12–16 GB |
| Wan2.1-14B | 720p×5s | 50 steps | 284 s (H100) | ~80–110 s | ~70 GB |
| HunyuanVideo (13B) | 720p×129f | 50 steps | ~1900 s (H100, xDiT) | ~250–400 s | ~60–70 GB |
| HunyuanVideo-1.5 | 720p×121f | 8–12 steps (distill) | 18.8 s (H100 SXM) | **~10–20 s** | ~14–25 GB |
| Mochi-1 | 480p | 50 steps | ~90 s (H100) | ~35–45 s | ~42 GB |
| CogVideoX-5B | 720×480×6s | 50 steps | ~90 s (H100) | ~35–45 s | ~12 GB |
| LTX-Video 0.9.8-distilled | 768×512×5s | few-step | **~2 s** (H100) | **~1.5 s** | ~15–25 GB |
| Open-Sora 2.0 | 768² | 50 steps | 1656 s (1×H100) | ~600–800 s | ~60 GB |

*Estimates marked "est." extrapolate from H100 SXM since no public GH200 numbers for these workloads exist as of 2026-05.*

---

## 7. Per-domain recommendation

- **Fastest open image gen on GH200 (highest quality)**: Flux.2-dev bf16 + FA3 + torch.compile + (when a distill drops) — until then, Flux.1-dev + Hyper-FLUX-8steps.
- **Fastest open video on GH200 (real-time class)**: **LTX-Video 0.9.8-distilled** bf16 + FA3. ~2 s for 5-s 768×512.
- **Best video quality on GH200 (Wan family, user's tested stack)**: Wan2.2-A14B bf16 + lightx2v 4-step distill + MagCache + FA3. ~25–45 s for 720p 5-s clip. Bf16-only — do not enable FP8 weights (color regression + user reports).
- **Best video quality/cost on GH200 (smaller MoE-free)**: FastWan2.2-5B bf16 distilled. 16 s for 720p 5-s on H200 — GH200 should match.
- **Best HunyuanVideo path**: HunyuanVideo-1.5 (not the original 13B) — 8.3B, distilled variant, bf16, fits comfortably, ~10–20 s.

---

## 8. Confidence & Gaps

### High confidence
- **Bf16 is the right default on this stack.** Independent reports of FP8 quality regressions across Z-Image / FLUX.1-dev / Qwen-Image (vllm-omni #2728) and Wan 2.1 (DiffSynth-Studio #466) confirm the user's experience generalizes; bf16 on Hopper FA3 is the production-safe path.
- **GH200's 96 GB headroom unlocks Wan2.2-A14B + Flux.2-dev in bf16 single-GPU** — workloads that require offload or multi-GPU on H100. This is the single largest GH200 advantage in this domain.
- **FastWan2.2-5B and LTX-Video 0.9.8-distilled** are the real-time-class winners; both have explicit H100/H200 measurements (FastVideo blog, LTX-Video paper).
- **Framework hierarchy**: diffusers > SGLang Diffusion > ComfyUI for production single-GPU GH200; xDiT requires NVLink-paired multi-GPU.
- **Caching stack (MagCache > TeaCache > FBCache) and step distillation (lightx2v / DMD2 / FastWan)** are independently composable and multiplicative.

### Medium confidence
- **GH200 ≈ H100 SXM ≈ H200 for compute-bound workloads.** Same SM90 silicon, same 80 GB H100 SXM vs 96 GB GH200 HBM3 vs 141 GB H200 HBM3e. Memory bandwidth differs (H100 SXM 3.35 TB/s; GH200 4 TB/s; H200 4.8 TB/s) — Wan/Hunyuan are memory-bandwidth-bound at large activations so H200 > GH200 > H100 by a small margin. Extrapolations to GH200 from H100/H200 numbers should be accurate within ±15%.
- **Flux.2-dev throughput on GH200**: no published number anywhere. ~10–15 s/image bf16/28-step is my extrapolation from Flux.1 → Flux.2 parameter scaling.

### Low confidence / Gaps
- **No public GH200-specific diffusion benchmark.** Lambda Labs' GH200 diffusers tutorial only runs SD-v1.5 hello-world. The closest comparable hardware is H200 (FastVideo Hao Lab blog) and H100 SXM (ParaAttention, xDiT, Voltage Park).
- **SGLang Diffusion charts are images, not tables** — speedup figures (1.2–5.9× over diffusers) lack per-model breakdown in machine-readable form. Worth running directly on GH200 to validate.
- **SageAttention on Wan/Hopper has reported artifacts** (ComfyUI-WanVideoWrapper #1554) — diffusers' `_flash_3_hub` is safer than `sage_hub` for Wan as of mid-2026.
- **Flux.2 distillation status** — Flux.2-klein 9B is the only known fast variant; no Lightning/Hyper for the 32B-dev model yet.
- **Wan2.5 / 2.6** — closed weights, Alibaba Cloud API only. Cannot deploy on GH200.
- **xDiT on GH200 single GPU has no benefit** — it's a multi-GPU sequence-parallel framework. The published xDiT Flux benchmark (1.6 s at 1024px on 4×H100) won't reproduce on single GH200. To leverage parallelism you'd need GH200 NVL pairs.

### Open questions to resolve in round 2 / measurement
1. Measure Flux.2-dev bf16 latency on GH200 (`set_attention_backend("_flash_3_hub")` + torch.compile, with/without `enable_model_cpu_offload()`).
2. Measure Wan2.2-A14B bf16 + lightx2v-4-step + MagCache on single GH200 vs. 8×H100 SP baseline (cost/quality trade).
3. Validate FastWan2.2-5B H200 number (16 s @ 720p 5-s) reproduces on GH200.
4. Validate LTX-Video 0.9.8-distilled ~2 s 768×512 5-s on GH200 — should hold or improve given bandwidth.
5. SGLang Diffusion vs diffusers head-to-head on GH200 for Flux.2-dev and Wan2.2 — pick whichever wins by >1.3× as production default.

---

## Sources (all accessed 2026-05)

- https://huggingface.co/blog/flux-2 — Diffusers welcomes FLUX-2 (architecture, VRAM, recipes)
- https://github.com/black-forest-labs/flux2 — FLUX.2 official inference repo
- https://github.com/xdit-project/xDiT/blob/main/docs/performance/hunyuanvideo.md — HunyuanVideo H100/H20 numbers
- https://github.com/xdit-project/xDiT/blob/main/docs/performance/flux.md — Flux H100 numbers
- https://github.com/chengzeyi/ParaAttention/blob/main/doc/fastest_flux.md — Flux FBCache + FP8 DQ + CP
- https://github.com/chengzeyi/ParaAttention/blob/main/doc/fastest_hunyuan_video.md — Hunyuan FBCache + CP
- https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations — Wan2.2 8×H100 optimization
- https://morphic.com/blog/boosting-wan2-2-i2v-56-faster — Wan2.2 I2V sequence parallelism
- https://huggingface.co/lightx2v/Wan2.2-Distill-Models — Wan2.2 4-step distill, bf16/FP8/INT8 VRAM table
- https://haoailab.com/blogs/fastvideo_post_training/ — FastWan H100/H200 numbers (2025-08-04)
- https://github.com/hao-ai-lab/FastVideo — FastVideo unified framework
- https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5 — HunyuanVideo-1.5 specs, FP8 caveat
- https://huggingface.co/tencent/HunyuanVideo-1.5 — HunyuanVideo-1.5 model card
- https://github.com/Lightricks/LTX-Video — LTX-Video repo
- https://arxiv.org/pdf/2501.00103 — LTX-Video paper (2 s on H100)
- https://github.com/hpcaitech/Open-Sora — Open-Sora 2.0 H100 benchmarks
- https://huggingface.co/zai-org/CogVideoX-5b — CogVideoX-5b model card
- https://huggingface.co/genmo/mochi-1-preview — Mochi-1 model card
- https://github.com/ali-vilab/TeaCache — TeaCache
- https://arxiv.org/html/2506.09045v2 — MagCache paper (June 2025)
- https://github.com/thu-ml/SageAttention — SageAttention (Hopper since 2025-01-28)
- https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1554 — SageAttention artifacts on H100 with Wan
- https://huggingface.co/docs/diffusers/optimization/attention_backends — diffusers attention backends (_flash_3_hub)
- https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/ — torch.compile + diffusers
- https://github.com/sayakpaul/diffusers-torchao — diffusers+torchao recipes
- https://www.lmsys.org/blog/2025-11-07-sglang-diffusion/ — SGLang Diffusion launch (2025-11-07)
- https://github.com/nunchaku-ai/nunchaku — Nunchaku/SVDQuant (consumer-focused)
- https://hanlab.mit.edu/blog/svdquant-nvfp4 — SVDQuant NVFP4 (Blackwell)
- https://docs.lambda.ai/education/running-huggingface-diffusers-transformers-gh200/ — diffusers on GH200 setup
- https://www.spheron.network/blog/nvidia-gh200-guide/ — GH200 architecture
- https://www.clarifai.com/blog/nvidia-h100-vs.-gh200-choosing-the-right-gpu-for-your-ai-workloads — GH200 vs H100
- https://github.com/Wan-Video/Wan2.2 — Wan2.2 official repo
- https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers — Wan2.2 diffusers checkpoint
- https://github.com/deepbeepmeep/Wan2GP — Wan2GP (single-GPU Wan/Hunyuan/LTX/Flux)
- https://blog.salad.com/benchmarking-wan2-2/ — Wan2.2 cross-GPU benchmark (403 from our fetcher; cited from search snippet)
- https://github.com/vllm-project/vllm-omni/issues/2728 — FP8 LPIPS regression bug (H100 + B200)
- https://github.com/modelscope/DiffSynth-Studio/issues/466 — Wan2.1 FP8 color regression vs bf16
- https://github.com/huggingface/diffusers/issues/11359 — LTX-Video 0.9.6 distilled 15× speedup
- https://lightx2v-en.readthedocs.io/en/latest/method_tutorials/step_distill.html — Self-Forcing step distillation
- https://medium.com/@furkangozukara/bf16-vs-gguf-fp8-scaled-nvfp4-speed-quality-compared-comfyui-cuda-13-gains-flux-2-klein-9b-dd7f64edcd89 — Quantization quality comparison (BF16/GGUF Q8/FP8/NVFP4)
- https://www.mindstudio.ai/blog/what-is-wan-2-5-video-open-source — Wan2.5 closed weights status
- https://www.blog.brightcoding.dev/2025/09/17/open-source-video-generation-for-low-vram-gpus-how-wan2gp-puts-cinematic-ai-in-reach-of-the-gpu-poor/ — Wan2.1-14B H100 timing (85 s 480p, 284 s 720p)
- https://www.tauceti.blog/posts/turn-your-computer-into-ai-machine-comfyui-video-with-hunyuan/ — HunyuanVideo bf16 25.6 GB
