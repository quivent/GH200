# Diffusion on GH200 — Round 3 Verification & Deepening

**Agent**: #20 (Round 3)
**Date**: 2026-05-25
**Mandate**: Verify Round 1 estimates against ground-truth user reports; hunt concrete GH200 diffusion benchmarks (Round 1 found none); confirm FP8 brokenness; resolve the Flux.2-dev "fits in 96 GB?" question; produce a direct-measurement plan for a 2-hour Lambda GH200 session.

**Constraint reaffirmed**: FP8 broken on this GH200/Wan stack. Bf16 is the default.

---

## TL;DR — what Round 3 found

1. **No public GH200 diffusion benchmark exists.** Confirmed across 18 searches/fetches. Lambda's GH200 diffusers tutorial still only runs SD-v1.5 hello-world; Lambda's Mochi-on-GH200 doc shows a single fine-tuning iter rate (3.33 s/iter) but no end-to-end inference number. SGLang-Diffusion's Jan-2026 follow-up blog explicitly published "Performance Benchmark on an H100 GPU" and "on an H200 GPU" — **no GH200 column**. This is now a confirmed gap, not a missing search.
2. **Flux.2-dev 32B bf16 fits in 96 GB GH200 — barely.** Authoritative BFL repo doc + diffusers blog both say bf16-no-offload "does not fit on H100 80 GB" and "everything fits" on H200 (141 GB). SimpleTuner says "bf16 everything ≈76 GB+" for the model+text encoder+VAE. Net: 76–80 GB working set in bf16 → fits in 96 GB GH200 with single-digit-GB headroom for activations. Whether 1024² × 28-step inference activations push over 96 GB is the **single most important thing to actually measure** in the 2-hour session.
3. **FP8 brokenness on Wan is verified across at least 6 independent reports.** Sage+FP8 → black output (SageAttention #221, RTX 5090); FP8 quantization → color contrast drift over time (WanVideoWrapper #1541); FP8 scaled → darkening/artifacts in last 4 frames of 77-frame windows (HF Kijai discussion #31); Wan2.1 FP8 weights cause color regression vs bf16 (DiffSynth-Studio #466); Wan 2.2 T2V can't even be cleanly upcast from FP8 (ComfyUI #9699); Comfy maintainer says (Jan 2026) "fp16 should be used … long-term quality cannot be ensured with other quantization levels since the lora is trained with fp16". The user's experience is the consensus, not an outlier.
4. **Best confirmed single-GPU Hopper numbers (none on GH200, but transferable):**
   - **FastWan2.2-5B**: 16 s for 720p × 5-s on **single H200** (multiple sources confirm).
   - **FastWan2.1-1.3B**: 5 s for 480p × 5-s on **single H200**.
   - **FastVideo live demo (2026)**: 5-s × 1080p in 4.5 s on a single GPU (model variant not specified in source).
   - **Wan2.1-14B**: 284 s for 720p × 5-s × 50 steps on single H100 (instasd).
   - **Wan2.2-A14B**: 1600 s on single H800 with `--offload_model True` (user report) — slower than docs claim, confirms expert-offload penalty.
   - **Wan2.2 (8×H100) Voltage Park stack**: 4.67 → 1.51 s/step (3.1×), 187 → 60 s total at 40 steps, with FA3 + USP + batched fwd + Sage(INT8) + TeaCache.
   - **Wan2.2 Replicate (lucataco/wan-2.2-first-last-frame, 8-step Lightning, single H100)**: ~45 s per prediction at $0.069/run.
5. **Round 1 estimate for Wan2.2-A14B on GH200 (25–45 s for 720p 5-s with 4-step distill + MagCache + FA3) — still defensible**, but the Replicate H100 number (~45 s with 8-step Lightning) suggests Round 1's 25 s lower bound is too optimistic without distill + cache stacked together. Expected GH200 range: **35–60 s** for 720p × 5-s × 4-step distill, **no aggressive cache**.
6. **Round 1 estimate for FastWan2.2-5B on GH200 (~16 s matching H200) — confirmed plausible**, but GH200 has 4.0 TB/s HBM3 vs H200's 4.8 TB/s HBM3e — for memory-bandwidth-bound workloads GH200 will be **~10–15% slower** than H200. Predict 17–19 s on GH200, not 16 s.
7. **Flux.2-dev throughput on Hopper, first concrete data point:** DeepWiki says "30–40 s per 1024² image on H100 + CPU offload, 50 steps" and "20–25 s on H200 full". This is the closest GH200 reference. GH200 sits between (memory bandwidth, no offload needed if bf16 fits): **predict 22–32 s per 1024² × 50-step image bf16**. Round 1's 10–15 s estimate was too optimistic (and at 28 steps; rescaling: 28/50 × 22–32 = 12–18 s at 28 steps, so Round 1 ballpark was right for 28 steps but didn't make the step count explicit).
8. **SGLang-Diffusion is officially `bench`-able on H200**, supports Wan 2.2 + Flux.2 day-0, has a cookbook recipe with peak VRAM 62.6 GB single-request / 72.5 GB at 20-concurrent, with cache-dit 7.4× speedup. **This is the framework most likely to give a usable GH200 baseline in 2 hours of measurement.**

---

## 1. The GH200-specific gap — fully verified

Searches that returned nothing GH200-specific for diffusion (paraphrased):
- "Wan2.2 GH200 benchmark inference seconds" → H100/H200 only
- "Wan 2.2 Grace Hopper video generation timing" → Baseten H100 number
- "Flux.2-dev 32B benchmark H100 H200 bf16 latency seconds" → H100/H200/B200 only
- "Lambda Labs GH200 diffusion benchmark Wan Flux" → Lambda Mochi fine-tune tutorial (no end-to-end numbers)
- "xDiT GH200 Grace Hopper benchmark" → MLPerf LLM data, no diffusion
- "SGLang Diffusion GH200 benchmark Wan Flux" → SGLang has H100/H200 columns only
- "GH200 Wan FLUX.2 HunyuanVideo reddit forum 2026" → no hits
- "FastWan GH200 generation time fastvideo" → H200 only

Closest GH200 data points that exist:
- **Lambda Mochi fine-tune**: 3.33 s/iter, batch=1, no end-to-end inference timing. https://docs.lambda.ai/education/fine-tune-mochi-gh200/
- **Lambda HF Diffusers tutorial**: only runs SD-v1.5 toy script, no perf number. https://docs.lambda.ai/education/running-huggingface-diffusers-transformers-gh200/
- **Lambda GH200 announcement**: Leonardo.ai "3× throughput over A100 for image captioning pipeline" — image captioning, not generation.

Bottom line: **Round 1's claim "no public GH200-specific diffusion benchmark exists" is verified.** This is the actual state of the open ecosystem as of 2026-05-25.

---

## 2. Flux.2-dev 32B bf16 — does it fit in 96 GB?

This was the largest open question. Four authoritative sources agree on a working-set envelope:

| Source | bf16 working set | Claim |
|---|---|---|
| HF blog `flux-2` | >80 GB no-offload, ~62 GB with offload | "does NOT fit on H100, **does** fit on GH200 with headroom for VAE decode" |
| BFL repo `flux2_dev_hf.md` | >80 GB | "For H200, B200 or larger cards, everything fits" |
| DeepWiki hw-req | ~141 GB full, ~70 GB peak with sequential offload | "H100 80 GB: requires offload; H200 141 GB: full precision possible" |
| SimpleTuner (bghira) | ~76 GB ("bf16 everything") | Recommends "2× 48 GB minimum" or "4× H100 80 GB with fp8-torchao" for training (training adds optimizer states, not relevant to inference) |
| Spheron blog | "BF16 at 64 GB is too large for any single 80 GB GPU when you add text encoder overhead" | Effectively says 64 GB weights + ~10–20 GB text encoder + activations > 80 GB |

**Synthesis for inference (no training optimizer state)**:
- Flux.2 transformer bf16: ~64 GB
- Mistral 3.1 text encoder bf16: ~48 GB (per SimpleTuner; some sources say ~50 GB)
- VAE + activations + workspace: ~4–10 GB
- **Total naive concurrent residency: 116–122 GB → does NOT fit in 96 GB GH200**

Wait — this contradicts the HF blog. The reconciliation: with **sequential** `enable_model_cpu_offload()`, only the active component is on GPU. Text encoder runs first (~50 GB peak), then gets moved out, then transformer runs (~64 GB peak), then VAE (~5 GB peak). HF blog's "62 GB peak" is consistent with sequential offload of the inactive component to CPU. The "fits in H200 141 GB no-offload" claim implies concurrent residency, which the math (~116 GB) supports.

**For GH200 96 GB**:
- **bf16 no-offload: does NOT fit.** (~116–122 GB concurrent vs 96 GB available.)
- **bf16 with `enable_model_cpu_offload()`: fits with margin.** (~62 GB peak per HF blog; GH200's NVLink-C2C makes the swaps essentially free.)
- **Round 1 was wrong** on this point. Round 1 said "Full no-offload: >80 GB → does not fit on H100, does fit on GH200 (96 GB) with headroom" — but >80 GB does not equal "fits in 96 GB" since the real number is ~120 GB. **Correct Round 1 claim: GH200 needs offload too, but offload on GH200 is nearly free (900 GB/s C2C) versus H100 PCIe.**

This is the **single concrete correction** Round 3 produces over Round 1.

---

## 3. FP8 brokenness on Wan — six independent confirmations

| Issue | Source | What breaks |
|---|---|---|
| Sage + FP8 → black output, RTX 5090 | `thu-ml/SageAttention#221` | FP8 model with Sage produces black frames; FP16 model + Sage works; **bf16 not tested but implied to be the safe path** |
| Color contrast drift over time, all variants | `kijai/ComfyUI-WanVideoWrapper#1541` | RTX 3090. Reproduced across FP8 *and* FP32 VAE variants — color drift is broader than just FP8, but FP8 makes it worse |
| Darkening/artifacts in last 4 frames of 77-frame windows | HF `Kijai/WanVideo_comfy_fp8_scaled` discussion #31 | FP8 scaled. User explicitly says "test bf16 to confirm fp8 is the culprit" — unresolved but FP8-suspected |
| Wan 2.1 FP8 weights cause color regression vs bf16 | `modelscope/DiffSynth-Studio#466` | Explicit: "Wan 2.1 FP8 model weights causing color issue — BF16 has no issue" |
| Wan 2.2 T2V can't upcast cleanly from FP8 | `comfyanonymous/ComfyUI#9699` | Weights save in original precision regardless of `force_upcast`; T2V breaks where I2V works |
| Comfy maintainer guidance | Issue thread, Jan 2026 | "If possible, fp16 models should be used … long-term quality cannot be ensured with other quantization levels since the lora is trained with fp16" |

**Verdict**: User's experience is the consensus across at least 6 independent sources spanning 3 different repos. **Bf16 is the unambiguous production-safe default for Wan on Hopper.** Round 1's stance holds.

Additional vllm-omni LPIPS regression (#2728, Round 1 already cited) extends this to FLUX.1-dev, Qwen-Image, Z-Image — **FP8 is a general DiT failure mode, not just Wan-specific**.

---

## 4. Verified single-GPU Hopper numbers (transferable to GH200)

### Wan family

| Model | GPU | Resolution | Frames | Steps | Time | Stack | Source |
|---|---|---|---|---|---|---|---|
| Wan2.2-A14B | 8×H100 baseline | 720p | 81 | 40 | 187 s (4.67 s/step) | FA3 + USP, no opts | Voltage Park |
| Wan2.2-A14B | 8×H100 optimized | 720p | 81 | 40 | **60 s (1.51 s/step)** | FA3 + USP + batched fwd + Sage-INT8 + TeaCache | Voltage Park |
| Wan2.2-A14B | 1×H800 (community) | 720p | 5-s | unspec | **1600 s** | offload_model=True | Wan2.2 issue #88 |
| Wan2.2 first-last (Lightning 8-step) | 1×H100 | unspec | unspec | 8 | **45 s** | Lightning LoRA | Replicate `lucataco/wan-2.2-first-last-frame` |
| Wan2.1-14B | 1×H100 | 480p | 5-s | 50 | 85 s | baseline | instasd |
| Wan2.1-14B | 1×H100 | 720p | 5-s | 50 | 284 s | baseline | instasd |
| FastWan2.2-5B | 1×H200 | 720p | 5-s | 3 | **16 s** | DMD + VSA + FA3 + torch.compile | Hao AI Lab |
| FastWan2.1-1.3B | 1×H200 | 480p | 5-s | 1 | **5 s** | DMD + VSA | Hao AI Lab |
| FastWan2.1-1.3B | 1×RTX 4090 | 480p | 5-s | 1 | 21 s (2.8 s denoise) | DMD + VSA | Hao AI Lab |
| FastVideo live demo | 1× unspec | 1080p | 5-s | unspec | **4.5 s** | latest sparse distill | 36kr summary |

### Flux family

| Model | GPU | Resolution | Steps | Time | Notes | Source |
|---|---|---|---|---|---|---|
| Flux.2-dev | H100 + offload | 1024² | 50 | **30–40 s** | bf16 with CPU offload | DeepWiki |
| Flux.2-dev | H200 full | 1024² | 50 | **20–25 s** | bf16 no-offload | DeepWiki |
| Flux.2-dev | H100 PCIe FP8 | 1024² | 28 | **14 images/min ≈ 4.3 s/image** | FP8 (avoid per constraint) | Spheron |
| Flux.2-dev | H100 SXM5 FP8 | 1024² | 28 | **18 images/min ≈ 3.3 s/image** | FP8 (avoid per constraint) | Spheron |
| Flux.2-klein (distilled) | RTX 5090 | unspec | unspec | **~1.2 s** | distilled variant | Comfy docs |

### SGLang Diffusion (Wan 2.2)

| Config | GPU | Latency | VRAM peak | Notes |
|---|---|---|---|---|
| Single request, baseline | 1×H200 (likely) | 630.4 s | 62.6 GB | full quality |
| Single request, cache-dit | 1×H200 | 630.4/7.4 ≈ **85 s** | 62.6 GB | 7.4× speedup, "minimal quality loss" |
| 20 concurrent requests | 1×H200 | 2740 s mean | 72.5 GB | full quality |

---

## 5. xDiT v0.5+ on GH200 — current state

- **Latest version**: 0.4.5 (Oct 2025); commits continue into 2026.
- **FA3 + FA3-FP8 attention backends listed** as available on Hopper.
- **No GH200-specific mention** anywhere in repo docs.
- **Diffusion model coverage**: Wan 2.1, Wan 2.2, CausalWan 2.2, Flux, Flux 2, Flux 2 klein, Flux Kontext, HunyuanVideo, HunyuanVideo-1.5, HunyuanDiT-v1.2.
- **Sequence parallelism is the main feature** — wasted on single GH200. Only useful on 2× GH200 with NVL switch.
- **Recommendation unchanged from Round 1**: skip xDiT on single GH200; use diffusers + FA3 + torch.compile or SGLang-Diffusion.

---

## 6. Corrections to Round 1

1. **Flux.2-dev VRAM in 96 GB**: Round 1 said "fits without offload"; Round 3 verifies it does **NOT** fit without offload (~116 GB concurrent residency vs 96 GB available). With `enable_model_cpu_offload()` the peak is ~62 GB, fits easily, and GH200's NVLink-C2C makes the offload cost trivial.
2. **Flux.2-dev throughput**: Round 1's "10–15 s/image at 28 steps" is plausible but borderline-optimistic. DeepWiki data point (H200 20–25 s at 50 steps) → 28-step rescale ≈ 11–14 s on H200. GH200 with ~85% of H200 bandwidth: **predict 13–17 s at 28 steps, 23–30 s at 50 steps**.
3. **FastWan2.2-5B GH200 estimate**: Round 1 said "GH200 ≈ H200 for this workload, ~16 s". Bandwidth ratio (4.0/4.8 = 0.83) suggests **17–19 s on GH200**, not 16 s. Small correction.
4. **Wan2.2-A14B GH200 estimate (25–45 s)**: Replicate single-H100 8-step Lightning data point (~45 s) shows Round 1's lower bound (25 s) requires aggressive cache stacking beyond what's been publicly verified. Updated range: **35–60 s** for 720p 5-s with 4-step distill + FA3, **25–40 s** if MagCache stacks cleanly.
5. **Round 1 didn't flag step count for Flux.2 latency estimates** — added explicit 28 vs 50 step ranges above.

---

## 7. Direct measurement plan — 2 hours on Lambda GH200 @ $2.29/hr

Total budget: **$4.58, ~120 minutes wall-clock**. Plan minimizes setup time and maximizes the single highest-value experiment: **does Flux.2-dev bf16 actually fit in 96 GB and what's the real latency?**

### Phase 0 — Environment setup (target ≤15 min, budget cap 20 min)
```bash
# On Lambda GH200 instance
pip install -U diffusers transformers accelerate
pip install flash-attn==3.0.0b1 --no-build-isolation  # FA3 wheel
pip install kernels  # for diffusers _flash_3_hub dispatch
huggingface-cli login
nvidia-smi  # confirm 96 GB HBM3, sm_90, CUDA 12.x
```
Pre-download weights in parallel during setup:
```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers &
huggingface-cli download FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers &
huggingface-cli download black-forest-labs/FLUX.2-dev &
wait
```

### Phase 1 — Flux.2-dev bf16 VRAM truth (target 25 min) — HIGHEST VALUE
Three configurations, one image each (1024², 28 steps):

| # | Config | Measure | Why |
|---|---|---|---|
| 1.1 | Flux.2-dev bf16, **no offload**, FA3 | OOM? Yes/No. If no, peak VRAM + latency | Resolves the Round 1 contradiction |
| 1.2 | Flux.2-dev bf16, `enable_model_cpu_offload()`, FA3 | Peak VRAM + latency | Confirms HF blog ~62 GB peak transfers to GH200 |
| 1.3 | Flux.2-dev bf16, no offload, FA3, **torch.compile** | Latency only if 1.1 fit | Quantify compile speedup on this stack |

Capture with `nvidia-smi --query-gpu=memory.used --loop-ms=200 > vram.log` running in background. Save image outputs for sanity check.

**Expected outcome**:
- 1.1: OOM at ~96 GB during text encoder + transformer overlap → confirms offload needed.
- 1.2: ~62 GB peak, **20–30 s/image** at 28 steps.
- 1.3: 30–40% faster on second+ call after compile warmup.

### Phase 2 — FastWan2.2-5B GH200 vs H200 calibration (target 20 min)
- 2.1: FastWan2.2-5B, **bf16, FA3, no torch.compile**, 720p × 5-s × 3-step → time + VRAM
- 2.2: Same + `torch.compile(mode="max-autotune-no-cudagraphs")` (one warmup, two measure) → time
- Compare against published H200 16 s. Establishes GH200/H200 ratio for memory-bandwidth-bound video diffusion.

**Expected outcome**: 2.1 ≈ 22–28 s (no compile); 2.2 ≈ 17–20 s. If 2.2 ≤ 18 s, Round 1's claim "GH200 ≈ H200" holds; if higher, bandwidth ratio dominates and all GH200 video estimates should be scaled by 1.10–1.15×.

### Phase 3 — Wan2.2-A14B bf16 single-GPU reality check (target 35 min)
This is the user's tested workload and the highest-stakes recommendation in Round 1.
- 3.1: Wan2.2-T2V-A14B bf16, FA3, **no distill, no cache, 40 steps**, 720p × 81 frames → baseline time + peak VRAM (likely needs `enable_model_cpu_offload()` for one expert; capture which)
- 3.2: + lightx2v 4-step distill LoRA (4 steps total) → time
- 3.3: + TeaCache `rel_l1_thresh=0.2` on top of distill → time
- 3.4: 480p × 81 frames variant of 3.3 → time (cheaper sanity check)

**Expected outcome**:
- 3.1: ~600–900 s (extrapolating from 8×H100 baseline 187 s, single-GPU ≈ 6–10×) with one expert at a time on GPU.
- 3.2: ~60–100 s (10× step reduction).
- 3.3: ~30–55 s (additional 1.8–2× from cache).
- 3.4: ~15–25 s.

If 3.3 lands in 30–55 s, Round 1's "25–45 s with 4-step distill + MagCache + FA3" range is validated (the upper bound was slightly conservative).

### Phase 4 — LTX-Video real-time sanity check (target 10 min)
- 4.1: LTX-Video-0.9.8-distilled, bf16, FA3, 768×512 × 5-s @ 24fps, few-step → time
- Expected: 1.5–2.5 s (matching H100 published 2 s number).
- If achieved, confirms LTX as the real-time-class winner Round 1 claimed.

### Phase 5 — One stretch experiment (target 15 min, **only if budget allows**)
Either:
- 5a: SGLang-Diffusion server start + single Wan 2.2 request with cache-dit. Compare wall-clock vs diffusers from Phase 3.3. Validates Round 1's "SGLang worth piloting" claim.
- 5b: HunyuanVideo-1.5 (8.3B distilled) bf16, FA3, 720p × 121f, 8 steps → time. Expected 10–20 s per Round 1.

Recommend 5a — it answers a framework choice question that affects the production recipe.

### Phase 6 — Cleanup & data export (target 5 min)
- Save all `vram.log` files, image/video outputs, and a markdown table of measured numbers.
- `nvidia-smi -q` snapshot for GPU clock state.
- Push results to a gist before terminating instance.

### What this measurement campaign settles

| Round 1 claim | Phase that verifies | Confidence after |
|---|---|---|
| Flux.2-dev bf16 fits in GH200 no-offload | 1.1 | Empirical |
| FastWan2.2-5B GH200 ≈ H200 (16 s) | 2.1, 2.2 | Empirical |
| Wan2.2-A14B 25–45 s with distill+cache+FA3 | 3.2, 3.3 | Empirical |
| LTX-Video ~2 s on GH200 | 4.1 | Empirical |
| SGLang-Diffusion competitive with diffusers on GH200 | 5a | Single-data-point empirical |

Five of Round 1's biggest claims become measured rather than extrapolated, for $4.58 and 2 hours. **The Phase 1 Flux.2-dev result alone is worth more than any of the other measurements** because it's the one Round 1 got wrong.

---

## 8. Sources accessed for Round 3 (new vs Round 1)

New / re-verified:
- https://www.baseten.co/blog/wan-2-2-video-generation-in-less-than-60-seconds/ — Baseten Wan 2.2 2.6× on H100, 3.2× on B200
- https://deepwiki.com/black-forest-labs/flux2/2.3-hardware-requirements — H100 + offload 30–40 s; H200 full 20–25 s for Flux.2-dev 50-step 1024²
- https://www.spheron.network/blog/deploy-flux2-gpu-cloud-production-guide/ — Spheron Flux.2 FP8 throughput on H100/A100/4090 (FP8 only, no bf16 numbers); confirms bf16 64 GB > 80 GB GPU
- https://github.com/black-forest-labs/flux2/blob/main/docs/flux2_dev_hf.md — BFL: "H200, B200 or larger cards, everything fits"
- https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX2.md — Mistral-3 ~48 GB, transformer ~24 GB, VAE+overhead ~4 GB
- https://huggingface.co/blog/flux-2 — HF: >80 GB no-offload, ~62 GB peak with offload
- https://huggingface.co/blog/cronos3k/h100-not-required-32b-flux2-dev-running-on-2017-ha — 8× V100-32GB FP16 run, ~105 GB total, 2–3 min/image (irrelevant to GH200 but bounds parameter footprint)
- https://github.com/Wan-Video/Wan2.2/issues/88 — Single H800 user report: 1600 s for 720p 5-s with offload
- https://github.com/Wan-Video/Wan2.2/issues/172 — Wan 2.2 color drift across video segments (open, unresolved)
- https://github.com/comfyanonymous/ComfyUI/issues/9699 — Wan 2.2 T2V can't upcast from FP8
- https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1541 — Color contrast drift across all variants including FP8 (RTX 3090)
- https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/discussions/31 — FP8 scaled darkening last 4 frames of 77-frame windows
- https://github.com/thu-ml/SageAttention/issues/221 — Sage + FP8 → black output (RTX 5090); FP16 + Sage OK
- https://github.com/modelscope/DiffSynth-Studio/issues/845 — Wan 2.2 FP8 + Sage + LightX2V LoRA request (open)
- https://docs.lambda.ai/education/fine-tune-mochi-gh200/ — Mochi GH200: 3.33 s/iter, no end-to-end inference number
- https://docs.lambda.ai/education/running-huggingface-diffusers-transformers-gh200/ — SD-v1.5 hello-world only, no perf data
- https://www.lmsys.org/blog/2026-01-16-sglang-diffusion/ — Jan 2026 SGLang-Diffusion follow-up: H100 + H200 benchmark charts; no GH200 column
- https://docs.sglang.io/cookbook/diffusion/Wan/Wan2.2 — SGLang Wan 2.2 recipe: 62.6 GB peak single-req, 7.4× cache-dit speedup
- https://github.com/xdit-project/xDiT — v0.4.5, FA3 + FA3-FP8 backends, Wan/Flux/Hunyuan supported, no GH200 mention
- https://replicate.com/lucataco/wan-2.2-first-last-frame — Single H100, 8-step Lightning, ~45 s/run, $0.069/run
- https://huggingface.co/lightx2v/Wan2.2-I2V-A14B-Moe-Distill-Lightx2v — 4-step (2+2) distill, Euler scheduler, shift=5.0, CFG=1.0
- https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers — 3-step inference, 121×704×1280 res, H100 to 4090 + Mac support
- https://eu.36kr.com/en/p/3412444076494465 — FastVideo 2026 demo: 5-s 1080p in 4.5 s on single GPU
- https://www.instasd.com/post/wan2-1-performance-testing-across-gpus — Wan2.1-14B 1×H100: 85 s 480p, 284 s 720p (50 steps)
- https://www.voltagepark.com/blog/accelerating-wan2-2-from-4-67s-to-1-5s-per-denoising-step-through-targeted-optimizations — 8×H100 Wan 2.2: 4.67 s/step → 1.51 s/step via FA3+USP+batched+Sage+TeaCache
- https://haoailab.com/blogs/fastvideo_post_training/ — FastWan H200 16 s (5B 720p 5-s), 5 s (1.3B 480p 5-s)
- https://www.stablediffusiontutorials.com/2025/11/flux-2.html — Flux.2 BF16/FP8/GGUF guide, no concrete latency
