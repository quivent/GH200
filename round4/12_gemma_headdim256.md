# Gemma Family on GH200 — The head_dim=256 Problem (Round 4 Deep Dive)

**Agent**: Round 4 / LLM-only deep dive
**Date**: 2026-05-25
**Builds on**: Round 2 `B_fp8_forensic_audit.md`, Round 3 `03_vllm_verify.md`
**Hard prior (unchanged)**: FP8 broken on DiT stack; bf16 default for LLMs on this rig.
**New focus**: Why Gemma is the canonical head_dim=256 victim, and how to serve every Gemma SKU correctly on a single GH200.

---

## TL;DR

1. **head_dim=256 is a Gemma family choice, not a single-SKU accident.** Gemma 3 1B/4B/12B (all the "small" multimodal text Gemmas) use `head_dim=256`. Only **Gemma 3 27B** drops to `head_dim=128`. **Gemma 4** doubles down: *every* Gemma 4 SKU (E2B, E4B, 26B-A4B MoE, 31B Dense) uses `head_dim=256` for sliding-window layers and **`global_head_dim=512` for the full-attention layers** — heterogeneous within a single model.
2. **On Hopper FP8, head_dim=256 is the unfixed corner.** The two-level FP32 accumulation fix that landed in vLLM 0.19.1 only restored accuracy for head_dim ∈ {64, 128}. For head_dim=256, prefill is still ~1.6× slower than bf16 due to register pressure. The follow-up tiling PRs (vllm-project/flash-attention #122, #125, merged March 2026) recovered ~2× throughput at head_dim=256 with "every-N-step" accumulation — but at compile-time and with a small long-context accuracy hit (0.89 → 0.88 at N=4, → 0.856 at N=64). Not yet a runtime CLI flag.
3. **FlashAttention 2/3 hard-caps head_size at 256.** Gemma 4's `global_head_dim=512` global layers don't fit in *any* FA kernel, period. vLLM forces `TRITON_ATTN` globally to avoid mixed-backend numerical drift, which is why fresh Gemma 4 deployments on RTX 4090 / RTX PRO 6000 Blackwell show **9 tok/s on E4B (vs 50–100 expected for a 4.5B model)** — issue #38887, fixed in PR #38891 by allowing FA2 on the sliding layers and SDPA only on the global ones (HF transformers issue #45201).
4. **`--kv-cache-dtype-skip-layers sliding_window` (PR #33695, merged 2026-03-27, in vLLM ≥ 0.20.0) keeps sliding-window attention in BF16 while quantizing the global layers to FP8.** Originally designed for gpt-oss; directly applies to Gemma 3 and Gemma 4 because they share the same hybrid-attention layer-type tagging convention. Net win on Gemma is small (1–3% ITL) because Gemma's sliding window is 1024 (not 128) — the SW layers still have meaningful work to amortize.
5. **For dense-Gemma on single GH200, the answer is BF16 + EAGLE3-or-MTP, full stop.** Gemma 3 27B fits comfortably in 96 GB HBM3e with KV headroom. Gemma 4 31B Dense in BF16 needs ~71 GB; fits, but tight — pair with `--cpu-offload-gb 20` and HMA KV spill into Grace LPDDR. FP8 buys you memory (50% KV cache) but loses ~1.6× on prefill and demands the AdaLN-skip-style "exclude lm_head/vision_tower/multi_modal_projector" recipe (which RedHatAI's FP8-dynamic recipe gets right) — only worth it if you're memory-bound, not compute-bound.
6. **Vision + Gemma 3 27B** on vLLM serves cleanly as `vllm serve google/gemma-3-27b-it --limit-mm-per-prompt image=4 --dtype bfloat16` — the SigLIP-400M encoder is BF16 by default in RedHatAI's quantized variants. For Gemma 4, also set the **per-request `max_soft_tokens`** budget (70/140/280/560/1120) via `--mm-processor-kwargs '{"max_soft_tokens": 280}'`. Multimodal Gemma forces `TritonAttention` or `FlexAttention` for the bidirectional image-token attention — slower than FA2 by ~15-25%.
7. **No public Gemma + GH200 benchmark exists.** Same gap as Round 3's Llama-3.3-70B-on-GH200. H100 numbers ported: single H100 BF16 = **1,260 tok/s aggregate on Gemma 4 31B at concurrency 128** (InferenceBench Apr 2026), **22k tok/s on Gemma 3 27B with 4×H100** (Google Cloud Mar 2026). With MTP drafter at γ=8, Gemma 4 31B hits **3.19× speedup at batch=1** on H100 (vLLM PR #41745).

---

## 1. Gemma SKU + Architecture Map — Current as of 2026-05-25

### 1.1 Gemma 3 family (released March 2025; still actively maintained)

| SKU | Params | head_dim | num_heads | KV heads | hidden | layers | sliding_window | SW pattern | Context | Multimodal |
|---|---|---|---|---|---|---|---|---|---|---|
| **gemma-3-1b** | 1B | **256** | 4 | 1 | 1152 | 26 | 512 | 5:1 (every 6th = global) | 32k | text-only |
| **gemma-3-4b** | 4B | **256** | 8 | 4 | 2560 | 34 | 1024 | 5:1 | 128k | SigLIP-400M |
| **gemma-3-12b** | 12B | **256** | 16 | 8 | 3840 | 48 | 1024 | 5:1 | 128k | SigLIP-400M |
| **gemma-3-27b** | 27B | **128** | 32 | 16 | 5376 | 62 | 1024 | 5:1 | 128k | SigLIP-400M |
| **gemma-3-270m** (Aug 2025 add-on) | 270M | **256** | 4 | 1 | 640 | — | 512 | 5:1 | 32k | text-only |

Pattern: every Gemma 3 *except* 27B uses head_dim=256. The 27B drops to head_dim=128 because `hidden/num_heads = 5376/32 = 168` already gave a "big" effective Q dim via `query_pre_attn_scalar=168` — they kept head_dim at the FA-friendly 128.

### 1.2 Gemma 4 family (released 2026-04-02 per Google's blog; Apache 2.0)

| SKU | Total params | Active | Layers | SW size | SW pattern | head_dim (SW) | global_head_dim | Context | Multimodal |
|---|---|---|---|---|---|---|---|---|---|
| **gemma-4-E2B** (edge) | 2.3B | 2.3B | 35 | 512 | 5:1 | **256** | **512** | 128k | image + audio |
| **gemma-4-E4B** (edge) | 4.5B | 4.5B | 42 | 512 | 5:1 | **256** | **512** | 128k | image + audio |
| **gemma-4-26B-A4B** (MoE) | 25.2B | 3.8B | 30 | 1024 | layers [3,9,15,21] = global, rest SW | **256** | **512** | 256k | image, 128 fine-grained experts top-8 |
| **gemma-4-31B Dense** | 30.7B | 30.7B | 60 | 1024 | 5:1 | **256** | **512** | 256k | image, p-RoPE |

**The Gemma 4 architectural quirk**: heterogeneous head dims within a single model. Sliding layers run at head_dim=256 (FA-supported), global layers at head_dim=512 (NO FA kernel exists — FA2/FA3 hard-cap is 256). This forces the attention backend selector down a tree:

- vLLM ≥ 0.19 default: **TRITON_ATTN** globally (safe but slow; cost = ~5× decode penalty on E4B per issue #38887).
- vLLM ≥ 0.20 + PR #38891: **FA on SW layers, SDPA on global layers** (recovers most of the perf; HF transformers issue #45201 mirrors this for the HF backend).
- vLLM Blackwell (FLASHINFER backend): **broken** for Gemma 4 31B → "head_size not supported" error on RTX PRO 6000 Blackwell + force-FLASHINFER. Issue #40677, open as of 2026-04-23. Workaround = omit the override and let vLLM pick TRITON_ATTN.

### 1.3 The Gemma 3 27B FP8-dynamic story (the one Gemma FP8 that "just works")

RedHatAI's `gemma-3-27b-it-FP8-dynamic` is the cleanest reference for **why Gemma 3 27B is the easy FP8 case in the family**:
- head_dim=128 → falls in the FA3 sweet spot the vLLM 0.19.1 fix targets.
- Recipe: `FP8_DYNAMIC` on all `Linear` targets, **exclude** `lm_head`, `vision_tower`, `multi_modal_projector`. Vision tower + projector remain BF16 — same AdaLN-skip pattern vllm-omni #2795 used for diffusion.
- Quality recovery vs BF16: **MMLU 77.53 → 77.45 (99.89%), GSM8K 92.12 → 91.51 (99.34%), MMMU-vision 50.89 → 51.00 (100.22%)**. Effectively no loss.
- This is the *only* Gemma where FP8 is genuinely production-grade on Hopper today.

---

## 2. The head_dim=256 Problem — Why It Persists on Hopper

The vLLM 2026-04-22 blog "State of FP8 KV-Cache and Attention Quantization" is the canonical reference. Round 3 already noted the 1.6× prefill penalty; here is the full mechanism + the partial fixes.

### 2.1 Root cause: register pressure from two-level FP32 accumulation

- Hopper's WGMMA FP8 intrinsic accumulates into FP32, but at long contraction dimensions (typical for prefill at 128k context) the intermediate accumulation **drifts in precision** — same hardware effect DeepSeek-V3 hit during training.
- Symptom Round 1 was the 128k-needle-in-a-haystack regression: BF16 91% → FP8 13% on Hopper, *before* the fix.
- Fix in vLLM 0.19.1 (`vllm-project/flash-attention#104`, jmkuebler): write the partial accumulation into a real FP32 register every step. Restores accuracy 13% → 89%.
- **Cost of the fix**: that real FP32 register doubles the register-file footprint of the inner loop. At head_dim ≤ 128, the kernel still fits in registers without spilling. At head_dim=256, the kernel spills, occupancy drops, prefill goes ~1.6× slower than BF16.
- Quantitative: Gemma-4-E2B TTFT quadratic coefficient = **1.12e-06 ms/token² (FP8 post-fix) vs 6.93e-07 ms/token² (BF16)** ≈ 1.62× regression. Decode slope at 68% of BF16 (still wins on memory-bandwidth-bound decode).

### 2.2 What's been done since (Q1-Q2 2026)

| PR | Approach | Result for head_dim=256 | Cost | Merge status |
|---|---|---|---|---|
| `vllm-project/flash-attention#104` | Two-level accumulation (every step) | Restores 128k-needle accuracy 13% → 89% | Prefill 1.6× slower than BF16 at hd=256 | Merged into vLLM 0.19.1 (Apr 2026) |
| `vllm-project/flash-attention#125` | Tune FP8 tile sizes per head_dim | **No meaningful gain at hd=256** ("no tiling config meaningfully improves over default"). Wins 10-40% at hd=64/128 instead. | None | **Merged 2026-03-16** by LucasWilkinson; picked up by vLLM PR #36265 |
| `vllm-project/flash-attention#122` | Two-level accumulation every N steps (N=4, 16, 64) | **~2× throughput at hd=256** at N=4. Accuracy drops 0.89 → 0.88 (N=4), → 0.856 (N=64). "Slight degradation acceptable for 10% prefill e2e gains." | Compile-time only via `DFLASHATTENTION_FP8_TWO_LEVEL_INTERVAL`; **no runtime CLI flag**. Long-context-only accuracy degradation. | **Merged** by LucasWilkinson; referenced in vLLM PR #36009 |
| Disabling two-level accumulation entirely | Recover the pre-fix prefill throughput | Same as 0.10.2 perf | Returns the 91% → 13% 128k-needle regression. Only safe if your contexts stay short. | Not landed as a default; user-configurable |

**Bottom line for May 2026**: the head_dim=256 prefill penalty is **structurally improved but not eliminated**. The current best knob is recompile flash-attention with `DFLASHATTENTION_FP8_TWO_LEVEL_INTERVAL=4` and accept the small long-context accuracy ding. Most users won't recompile, so the practical guidance is unchanged: **head_dim=256 + Hopper + FP8 = stay on BF16 unless you've measured your specific workload tolerates the prefill regression.**

### 2.3 Cuteness no one tells you: the kernel selection tree

For head_dim=256 on Hopper, vLLM's actual kernel pick (post 0.20) is:

```
if dtype == bf16: FA3 (head_dim 256 fully supported, no register pressure)
elif dtype == fp8:
    if kv_cache_dtype == fp8 and head_dim == 256:
        FA3 with two-level accumulation  # the 1.6× slow path
    elif kv_cache_dtype == auto:
        FA3 with BF16 KV (no accumulation fix needed)
elif global_head_dim > 256 (Gemma 4 ONLY):
    TRITON_ATTN globally  OR  FA on SW + SDPA on global (PR #38891)
elif multimodal_image_attention:
    TRITON_ATTN or FlexAttention  # bidirectional image tokens
```

**Implication**: if you must run FP8 on a head_dim=256 Gemma, **set `--kv-cache-dtype auto` (i.e. BF16 KV) and FP8 only the weights**. You keep the FP8 memory win on weights, dodge the two-level accumulation slowdown on attention, lose the 50% KV-cache memory savings. For long-context Gemma serving where KV dominates, this is the wrong trade. For prefill-bound throughput on shorter contexts, it's the right one.

---

## 3. `--kv-cache-dtype-skip-layers sliding_window` — Gemma-specific guidance

PR `vllm-project/vllm#33695` (merged 2026-03-27, lands in vLLM 0.20.0):

```bash
vllm serve google/gemma-4-31B-it \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers sliding_window \
  --max-model-len 32768
```

What it does: keeps the **sliding-window attention layers** in BF16 KV while quantizing the **global (full) attention layers** to FP8 KV. The rationale from the vLLM blog: FP8 KV's memory win only matters when the KV cache is unbounded; on SW layers the KV is capped at `sliding_window` tokens, so FP8 saves nothing and adds quantization overhead.

**Gemma vs gpt-oss math**:
- gpt-oss SW size = 128. FP8 quant overhead on 128 KV entries dwarfs any memory savings → big win to skip.
- Gemma 3 / Gemma 4 SW size = 1024 (Gemma 3 4B/12B/27B, Gemma 4 26B/31B) or 512 (Gemma 3 1B/270m, Gemma 4 E2B/E4B). **8-16× larger than gpt-oss's SW window** — so the skip-savings are smaller (Red Hat measured **1-3% ITL improvement**, "essentially unchanged TTFT" for gpt-oss; expect even smaller deltas on Gemma).
- Still worth using on Gemma because (a) it's free correctness margin — SW layers in BF16 dodge per-layer FP8 scaling drift, and (b) the global layers, with their unbounded 128k/256k KV growth, are where the FP8 memory savings actually live.

**Important caveat from the PR**: the flag uses **underscores** (`--kv_cache_dtype_skip_layers`), not hyphens, despite the doc CLI listing it with hyphens. Both forms work on vLLM CLI but the docs/blog inconsistency is real.

---

## 4. Vision Multimodal Gemma on GH200 — vLLM Recipe

Gemma 3 (≥4B) and all of Gemma 4 are natively multimodal. The vision tower is **SigLIP-ViT-400M** for Gemma 3; Gemma 4 inherits the same with token-budget configurability.

### 4.1 Gemma 3 27B vision serving (BF16)

```bash
vllm serve google/gemma-3-27b-it \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt image=4 \
  --tensor-parallel-size 1
```

- ~54 GB model weights (incl. SigLIP) + KV cache headroom. Fits a single GH200 comfortably (96 GB HBM3e).
- For BF16-on-GH200 long context, optionally add `--cpu-offload-gb 20 --kv_offloading_backend native --kv_offloading_size 200` to push KV overflow into Grace LPDDR (Round 3 §4).
- **Attention backend on multimodal Gemma**: forces `TritonAttention` or `FlexAttention` for image-token bidirectional attention. ~15-25% slower than what FA3 would deliver on pure-text. No fix in sight as of May 2026.

### 4.2 Gemma 4 31B vision serving (BF16, with token budget)

```bash
vllm serve google/gemma-4-31B-it \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt image=4 \
  --mm-processor-kwargs '{"max_soft_tokens": 280}' \
  --enable-auto-tool-choice \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --chat-template examples/tool_chat_template_gemma4.jinja \
  --async-scheduling
```

Per-request token budgets: **70, 140, 280, 560, 1120**. Default 280 = balanced. Use 70 for batch classification/captioning workloads; 1120 for chart/OCR detail. Each token budget step roughly doubles compute on the vision tower.

### 4.3 Vision tower precision — keep BF16

RedHatAI's FP8-dynamic recipe explicitly excludes `vision_tower` and `multi_modal_projector` from quantization. NVIDIA's NVFP4 Gemma-4-31B-IT recipe (NVFP4 weights, NVFP4 activations) keeps these layers BF16 too. Empirically: quantizing the vision tower of any Gemma to FP8 or NVFP4 tanks MMMU by ≥2 points. Same precision-sensitive-layer pattern as the diffusion-FP8 bug class — vision encoders + AdaLN-style modulation + final projections are the layers that *always* need BF16.

---

## 5. Best Engine + Precision Config for Gemma on Single GH200

### 5.1 Decision matrix

| Gemma SKU | Workload | Engine | Dtype | KV dtype | Attention backend | Speculative | Sliding-window flag |
|---|---|---|---|---|---|---|---|
| gemma-3-1b | low-latency edge sim | vLLM 0.21.0 | bf16 | auto (bf16) | FA3 (gets TRITON_ATTN on MM) | none | n/a |
| gemma-3-4b | multimodal RAG | vLLM 0.21.0 | bf16 | auto | TritonAttention (forced by MM) | none | `--kv-cache-dtype-skip-layers sliding_window` if FP8'ing |
| gemma-3-12b | text + image agentic | vLLM 0.21.0 | bf16 | auto | TritonAttention (forced by MM) | EAGLE-3 (head: gemma) | optional |
| **gemma-3-27b** | **production text + vision** | **vLLM 0.21.0** | **bf16** (or FP8-dynamic if memory-tight) | **auto** | **FA3** (text) / TritonAttention (image) | **EAGLE-3** (`yuhuili` not yet, see note) | n/a (head_dim=128) |
| gemma-4-E2B | edge / streaming | vLLM 0.22.0+ | bf16 | auto | FA on SW + SDPA on global | **MTP γ=2** (Google-provided `-it-assistant`) | optional |
| gemma-4-E4B | edge / RAG | vLLM 0.22.0+ | bf16 | auto | FA on SW + SDPA on global | **MTP γ=4** | optional |
| **gemma-4-26B-A4B** | **MoE serving** | **vLLM 0.22.0+** | **bf16** (FP8 dynamic is RC-quality) | **auto** | FA on SW + SDPA on global | **MTP γ=4** | **yes** if FP8 KV |
| **gemma-4-31B Dense** | **single-GH200 prod text** | **vLLM 0.22.0+** | **bf16** | **auto** | FA on SW + SDPA on global | **MTP γ=8** (3.19× speedup) | **yes** if FP8 KV |

Notes:
- **EAGLE-3 head for gemma-3-27b** — there's an EAGLE-3 head listed for `Gemma-4-31B-it` (in the vLLM Speculators docs) but I couldn't find a published Gemma-3-27B EAGLE-3 head as of 2026-05-25. Use MTP if you upgrade to Gemma 4; for Gemma 3 27B, EAGLE-3 might require training your own drafter.
- **MTP drafters are Google-provided** (Apache 2.0), one per SKU: `google/gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant`. Day-0 vLLM support in 0.22.0 via PR #41745 (merged 2026-05-06).
- **Don't use TP for throughput on Gemma 4 31B**. InferenceBench finding: single-GPU = 1,260 tok/s; 8-GPU TP = 2,208 tok/s total, only 2,208/8 = 276 tok/s/GPU — TP loses 4.5× efficiency. Run multiple single-GPU replicas instead.

### 5.2 Canonical Gemma 4 31B GH200 BF16 + MTP recipe

```bash
vllm serve google/gemma-4-31B-it \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":8}' \
  --cpu-offload-gb 20 \
  --kv_offloading_backend native --kv_offloading_size 200 \
  --enable-prefix-caching \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4 \
  --async-scheduling
```

Why these choices on GH200:
- BF16: dodges the head_dim=256 FP8 prefill penalty AND the FP8_BLOCK garbage-output bug (vLLM #39407).
- `--cpu-offload-gb 20` + `--kv_offloading_size 200`: ~71 GB weights stay in HBM; KV overflows to 200 GB of Grace LPDDR over 900 GB/s C2C (Round 3 §4).
- MTP γ=8: vLLM PR #41745 measured **319% throughput at batch=1** on H100. On GH200 (same SM90 die + more HBM) expect ≥ same speedup.
- `--async-scheduling`: required for the vLLM 0.20+ Gemma-4 chat-template + tool calling combo to avoid head-of-line blocking on long thinking traces.

### 5.3 If you must go FP8 on Gemma (memory pressure)

- **Gemma 3 27B**: use RedHatAI's `gemma-3-27b-it-FP8-dynamic`. head_dim=128, vision tower stays BF16, MMLU recovery 99.89%. **Safe.**
- **Gemma 4 31B**: NVIDIA's `Gemma-4-31B-IT-NVFP4` is the safer bet on Blackwell (GPQA 85.80 → 85.35, AIME 87.92 → 87.60). On Hopper GH200, NVFP4 falls back to FP8 emulation — measure first. **Avoid `gemma-4-31B-it-FP8-block` (RedHatAI)** until vLLM #39407 fix lands: double-applied activation scales → garbage output (single token repeated, logits pinned at softcap ceiling 23.625).
- **Gemma 4 26B MoE**: NVFP4 win on DGX Spark (GB10) is **52 tok/s at single-stream vs 23.3 BF16 = 2.1× faster, 97.6% quality**. On GH200 Hopper this NVFP4 needs FlashInfer Cutlass + Marlin MoE backend (`--moe-backend marlin --quantization modelopt --kv-cache-dtype fp8`); test before committing.

---

## 6. Benchmark Numbers — Gemma on Hopper (Collected)

### 6.1 Pure-text decoding

| Model | Hardware | Engine + dtype | Concurrency | Throughput | Source |
|---|---|---|---|---|---|
| Gemma-3-27B | 4× H100 80GB | vLLM BF16 | high | **22,000 tok/s aggregate** | Google Cloud "Optimize Gemma 3 Inference vLLM on GKE" (Mar 2026) |
| Gemma-3-27B | 2× A100 40GB NVLink | vLLM BF16 | 100+ | 3,000-6,000 tok/s | Databasemart A100 benchmark (2025) |
| Gemma-4-31B Dense | 1× H100 80GB | vLLM 0.20 BF16 | 128 | **1,260 tok/s sustained, 855 tok/s peak** | InferenceBench Apr 2026 |
| Gemma-4-31B Dense | 1× H100 80GB | vLLM 0.20 BF16 | 1 | **40.3 tok/s baseline** | InferenceBench Apr 2026 |
| Gemma-4-31B Dense | 1× H100 80GB | vLLM 0.20 BF16 + MTP γ=4 | 1 | ~76 tok/s (1.9× MTP gain) | Jarvislabs "Gemma 4 MTP vs DFlash" |
| Gemma-4-31B Dense | 1× H100 80GB | vLLM 0.22 BF16 + MTP γ=8 | 1 | **3.19× = 128 tok/s** | vLLM PR #41745 measurement |
| Gemma-4-31B Dense | 2× H100 80GB TP=2 | vLLM BF16 | 128 | 1,675 tok/s sustained, 1,472 peak | InferenceBench Apr 2026 |
| Gemma-4-31B Dense | 8× H100 80GB TP=8 | vLLM BF16 | 128 | 2,208 tok/s sustained, 3,050 peak | InferenceBench Apr 2026 |
| Gemma-4-31B Dense | 1× RTX PRO 6000 Blackwell 96GB | vLLM 0.19.1rc1 BF16 | 1 | **22 tok/s** (single request) | allenkuo Medium Apr 8 2026 |
| Gemma-4-31B Dense | 1× RTX PRO 6000 Blackwell 96GB | vLLM 0.19.1rc1 BF16 | 4 parallel | 93 tok/s | allenkuo Medium |
| Gemma-4-E4B | 1× RTX PRO 6000 Blackwell 96GB | vLLM BF16 | 1 | **124 tok/s** | allenkuo Medium |
| Gemma-4-E4B | 1× RTX 4090 | vLLM 0.19.0 BF16 (TRITON_ATTN forced) | 1 | **9 tok/s (bug)** | vLLM #38887 |
| Gemma-4-26B-A4B MoE | 1× RTX PRO 6000 Blackwell 96GB | vLLM 0.19.1rc1 BF16 | 1 | **131 tok/s** | allenkuo Medium |
| Gemma-4-26B-A4B MoE | DGX Spark (GB10, SM12.1, 128GB LPDDR5x) | vLLM 0.19 NVFP4 + FP8 KV | 1 | **52 tok/s** | ai-muninn DGX Spark NVFP4 |
| Gemma-4-26B-A4B MoE | DGX Spark | vLLM 0.19 BF16 | 1 | 23.3 tok/s | ai-muninn (baseline) |
| Gemma-4-26B-A4B MoE | DGX Spark | vLLM 0.19 FP8 + MTP γ=4 | 1 | **108.78 tok/s** | ai-muninn DGX Spark MTP |
| Gemma-4-26B-A4B MoE | DGX Spark | vLLM 0.19 FP8 + MTP γ=4 | concurrency=8 | **674 tok/s aggregate** | ai-muninn |

### 6.2 Quality retention (FP8 vs BF16)

| Model | Quant | MMLU | GSM8K | MMMU (vision) | Notes |
|---|---|---|---|---|---|
| Gemma-3-27B BF16 (baseline) | — | 77.53 | 92.12 | 50.89 | RedHatAI eval (lm-eval-harness via vLLM) |
| Gemma-3-27B FP8-dynamic (RedHatAI) | FP8 | **77.45** | **91.51** | **51.00** | 99.89% / 99.34% / 100.22% recovery |
| Gemma-4-31B BF16 (baseline) | — | 85.25 (MMLU-Pro), 85.80 (GPQA Diamond) | — | — | NVIDIA NVFP4 card baseline |
| Gemma-4-31B NVFP4 (NVIDIA) | NVFP4 | **84.94** (MMLU-Pro), **85.35** (GPQA) | — | — | -0.31% to -0.45% delta; Blackwell preferred |
| Gemma-4-31B FP8_BLOCK (RedHatAI) | FP8_BLOCK | **garbage** (vLLM #39407) | — | — | Double-scaled activations → softcap saturation. Avoid until fix. |

### 6.3 What's missing — public GH200 numbers

No single Gemma model on GH200 has a published vLLM benchmark in the public record as of 2026-05-25. Same gap pattern as Round 3's Llama-3.3-70B-on-GH200. **Action item**: run `gemma-4-31B-it` BF16 + MTP γ=8 on our GH200 against the InferenceBench H100 number (1,260 tok/s at conc=128). Expected: GH200 should match H100 at small batch (same SM90 die, same memory bandwidth on the HBM side), with a 5-20% throughput edge on memory-bound long-context decoding via Grace LPDDR overflow.

---

## 7. The Bugs That Actually Bite — Gemma-Specific Open Issues

| Issue / PR | Title | Affected | Status | Workaround |
|---|---|---|---|---|
| vLLM #38887 | Gemma 4 E4B extremely slow on v0.19.0 forced TRITON_ATTN fallback (~9 tok/s on RTX 4090) | All Gemma 4, Hopper + Blackwell | **Fixed in PR #38891** (vLLM 0.20.0): FA on SW layers + SDPA on global | Upgrade to vLLM ≥ 0.20.0 |
| HF transformers #45201 | Gemma 4 per-layer FA: FA2 for sliding, SDPA for global | All Gemma 4 | Open RFC, partial impl | Transformers backend manual override |
| vLLM #40677 | Gemma-4 fails on FLASHINFER backend (head_size not supported) | Gemma 4 31B on Blackwell SM12.x | **Open** as of 2026-04-23 | Omit `--attention-backend FLASHINFER`; default to TRITON_ATTN |
| vLLM #39407 | Gemma 4 31B FP8_BLOCK garbage output (softcap saturation, double-applied activation scales) | Gemma 4 31B + RedHatAI FP8_BLOCK checkpoint | **Open** (Apr 2026) | Use FP8-dynamic instead; or NVIDIA NVFP4 checkpoint; or BF16 |
| vLLM #39133 | Gemma 4 31B INT4 on 2×24GB GPUs: KV cache size only 25,200 tokens at max_model_len=131072 | Gemma 4 31B INT4, TP=2 | Open | Reduce max_model_len; investigate sliding-window KV sharing config |
| vLLM #38918 | Gemma 4 on Turing GPUs (SM 7.5): all attention backends hit shared memory limits | All Gemma 4 on SM ≤ 7.5 | Open ("Turing not supported") | Use Ampere+ |
| vLLM #28539 | Gemma-3-27B-it shows degraded accuracy in vLLM v0.11.0 (HellaSwag 65% → 39%, LAMBADA PPL 3.77 → 950) | Gemma-3-27B on vLLM 0.11.0 | Open since Nov 2025; suspected sliding-window config bug | Upgrade past 0.11.0; check #17689 / PR #21927 |
| vLLM #42068 | Gemma 4 + DFlash incompatible: MTP backend forces TRITON_ATTN on DFlash drafters | Gemma 4 + DFlash | Follow-up to MTP PR #41745 | Use MTP, not DFlash, on Gemma 4 |

---

## 8. Confidence & Gaps

### High confidence
- **Architecture table in §1**: pulled directly from HuggingFace config.json on gemma-3-1b/4b/12b/27b/270m and gemma-4 model card. head_dim values verified per SKU. The Gemma 4 `head_dim=256 (SW) / global_head_dim=512 (global)` heterogeneity is documented in HF transformers issue #45201 and vLLM #40677.
- **head_dim=256 prefill 1.6× penalty**: vLLM blog 2026-04-22 + Round 3 verified. Tiling fix at hd=256 confirmed "no meaningful gain" (PR #125).
- **`--kv-cache-dtype-skip-layers sliding_window`**: PR #33695 merged 2026-03-27, lands in vLLM 0.20.0. Direct applicability to Gemma confirmed.
- **MTP drafter for Gemma 4 with 3.19× peak speedup**: vLLM PR #41745 numbers; cross-validated by ai-muninn DGX Spark 2.66× single-stream measurement.
- **FP8 Gemma-3-27B-FP8-dynamic is safe**: RedHatAI quality eval published; head_dim=128 is in the FA3 sweet spot.

### Medium confidence
- **Gemma-3-12B exact head_dim**: my fetched config gave 240 via calculation (3840/16), but HF docs strongly imply 256 is the explicit Gemma family standard. The discrepancy is likely a WebFetch transcription error; treat as 256 until verified.
- **Gemma 4 26B MoE NVFP4 on GH200**: only DGX Spark (GB10 SM12.1) numbers exist. GH200 (SM90) NVFP4 falls back to FP8 emulation in practice — likely loses the 2.1× DGX Spark gain. Measure before promising.
- **vLLM 0.22.0** (post 0.21.0): inferred as the first release after PR #41745 merge on 2026-05-06. Not yet on PyPI as of 2026-05-25 per my Round 3 check; expect release in late May / early June 2026.

### Open gaps
- **No public Gemma + GH200 vLLM benchmark** (any SKU, any dtype). Highest-leverage in-house measurement to publish.
- **No published Gemma 3 27B EAGLE-3 drafter head**. Either train one or use MTP via Gemma 4 31B if quality permits the upgrade.
- **The `DFLASHATTENTION_FP8_TWO_LEVEL_INTERVAL=4` compile-time knob from PR #122** has no runtime CLI flag. Either: (a) rebuild flash-attention for your serving image, or (b) wait for the runtime exposure PR (none currently filed). Worth tracking.
- **Vision-tower attention backend**: TritonAttention / FlexAttention costs Gemma 3/4 multimodal ~15-25% throughput vs hypothetical FA3-with-bidirectional support. No RFC for fixing it.

---

## 9. Recommended Memory Update (Gemma slice)

Add to user memory under the existing `feedback_no_fp8.md`:

> **Gemma family + head_dim=256 status, May 2026**:
> 1. Every Gemma 3 except 27B uses head_dim=256; Gemma 3 27B uses head_dim=128 (the only "easy FP8" Gemma). Every Gemma 4 SKU uses head_dim=256 for sliding-window layers and global_head_dim=512 for global layers (heterogeneous within model).
> 2. **head_dim=256 FP8 prefill penalty (~1.6× vs BF16) is NOT FIXED on Hopper as of vLLM 0.21.0.** Tiling PRs at hd=256 gave no gain; the "every-N-step" accumulation PR gives ~2× at small accuracy cost but is compile-time only. Default to BF16 on Gemma 4 + Hopper. Gemma 3 27B is the exception: FP8-dynamic recipe ships from RedHatAI with 99.89% MMLU recovery.
> 3. **`global_head_dim=512` on Gemma 4 global layers exceeds FA2/FA3's head_size=256 cap.** vLLM defaults to TRITON_ATTN globally (slow); use vLLM ≥ 0.20.0 which has PR #38891 = FA on SW + SDPA on global. Don't force `--attention-backend FLASHINFER` on Gemma 4 (vLLM #40677 = head_size not supported).
> 4. **Avoid `RedHatAI/gemma-4-31B-it-FP8-block`** until vLLM #39407 fix lands — double-applied activation scales produce garbage repetitive output (softcap-saturated logits). Use `gemma-3-27b-it-FP8-dynamic` (safe) or `nvidia/Gemma-4-31B-IT-NVFP4` (Blackwell preferred) instead.
> 5. **`--kv-cache-dtype-skip-layers sliding_window`** (vLLM ≥ 0.20.0, PR #33695) keeps SW layers in BF16 KV while quantizing global layers to FP8 KV. Marginal 1-3% ITL on Gemma (their SW=512/1024 is larger than gpt-oss's SW=128, so the skip-savings are smaller). Use it for correctness margin, not perf.
> 6. **MTP speculative decoding is day-0 on Gemma 4 in vLLM 0.22.0** (PR #41745, merged 2026-05-06). Google ships `gemma-4-{E2B,E4B,26B-A4B,31B}-it-assistant` drafters; recipe is `speculative_config={'method':'mtp','model':<assistant>,'num_speculative_tokens':2-8}`. Peak gain 3.19× at batch=1 with γ=8 on Gemma 4 31B / H100.
> 7. **Don't use TP for throughput on Gemma 4 31B** — InferenceBench shows single-H100 at 1,260 tok/s sustained; 8-GPU TP at 2,208/8 = 276 tok/s/GPU (4.5× efficiency loss). Run replicas instead.
> 8. **Multimodal Gemma forces TritonAttention or FlexAttention** for image-token bidirectional attention, ~15-25% slower than FA3 would be. No fix in sight. Vision tower stays BF16 in every Gemma quantization recipe — quantizing it tanks MMMU.

---

## 10. Sources (URLs + dates accessed 2026-05-25)

### Architecture + config files
- google/gemma-3-1b-it config (head_dim=256, 4 heads, 1 KV head, SW=512) — https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/blob/main/config.json (mirror of the gated google/gemma-3-1b-it config.json)
- google/gemma-3-4b-it config (head_dim=256, 8 heads, 4 KV heads, SW=1024) — https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/blob/.../config.json
- google/gemma-3-27b-it config (head_dim=128, 32 heads, 16 KV heads, SW=1024) — https://huggingface.co/google/gemma-3-27b-it/discussions/11
- google/gemma-3-270m-it head_dim=256 confirmation — https://huggingface.co/google/gemma-3-270m-it/discussions/8
- Gemma 4 model card (E2B/E4B/26B/31B sizes, SW sizes, context lengths) — https://ai.google.dev/gemma/docs/core/model_card_4
- Gemma 4 announcement (Apache 2.0, release date 2026-04-02) — https://deepmind.google/models/gemma/gemma-4/ , https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- Gemma 4 head_dim=256 + global_head_dim=512 heterogeneity — https://github.com/huggingface/transformers/issues/45201 ; https://github.com/vllm-project/vllm/issues/40677
- Gemma 3 technical deep dive (5:1 SW pattern, RoPE 1M, p-RoPE) — https://namangoyal.com/blog/2025/gemma3/ , https://huggingface.co/blog/gemma3

### vLLM FP8 + head_dim=256 + sliding-window
- vLLM blog "FP8 KV-Cache and Attention Quantization" (head_dim=256 1.6× penalty, two-level accumulation) — https://vllm-project.github.io/2026/04/22/fp8-kvcache.html
- flash-attention#104 (two-level FP32 accumulation, jmkuebler, merged 2026-04 → vLLM 0.19.1) — https://github.com/vllm-project/flash-attention/pull/104
- flash-attention#122 (every-N-step accumulation, ~2× at hd=256, compile-time DFLASHATTENTION_FP8_TWO_LEVEL_INTERVAL) — https://github.com/vllm-project/flash-attention/pull/122
- flash-attention#125 (tile-size tuning; "no meaningful gain at hd=256") — https://github.com/vllm-project/flash-attention/pull/125
- vLLM PR #33695 (--kv-cache-dtype-skip-layers sliding_window, merged 2026-03-27, vLLM 0.20.0) — https://github.com/vllm-project/vllm/pull/33695

### Gemma vLLM bugs
- vLLM #38887 (Gemma 4 E4B 9 tok/s on RTX 4090, TRITON_ATTN forced) — https://github.com/vllm-project/vllm/issues/38887 (fixed in PR #38891)
- vLLM #40677 (Gemma 4 FLASHINFER head_size not supported on Blackwell SM12.x) — https://github.com/vllm-project/vllm/issues/40677 (open 2026-04-23)
- vLLM #39407 (Gemma 4 31B FP8_BLOCK garbage output, softcap saturation) — https://github.com/vllm-project/vllm/issues/39407 (open Apr 2026)
- vLLM #39133 (Gemma 4 31B INT4 KV cache only 25,200 tokens) — https://github.com/vllm-project/vllm/issues/39133
- vLLM #38918 (Gemma 4 on Turing SM 7.5 hits shared memory limits) — https://github.com/vllm-project/vllm/issues/38918
- vLLM #28539 (Gemma-3-27B-it degraded accuracy on vLLM 0.11.0) — https://github.com/vllm-project/vllm/issues/28539
- vLLM #42068 (Gemma 4 + DFlash incompatible, MTP backend forces TRITON_ATTN on DFlash drafters) — https://github.com/vllm-project/vllm/issues/42068

### Gemma vLLM recipes + benchmarks
- vLLM Recipes Gemma 4 page — https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
- vLLM Recipes Gemma 4 26B-A4B-it — https://recipes.vllm.ai/Google/gemma-4-26B-A4B-it
- InferenceBench Gemma 4 31B on H100 (1,260 tok/s @ conc=128) — https://inferencebench.io/blog/gemma-4-31b-h100-complete-inference-benchmark/
- vLLM PR #41745 (Day-0 Gemma 4 MTP support, 3.19× peak at γ=8, batch=1) — https://github.com/vllm-project/vllm/pull/41745
- vLLM Twitter Day-0 MTP for Gemma 4 — https://x.com/vllm_project/status/2051744111116574950
- ai-muninn DGX Spark Gemma 4 26B NVFP4 (52 tok/s) — https://ai-muninn.com/en/blog/dgx-spark-gemma4-26b-nvfp4-52-toks
- ai-muninn DGX Spark Gemma 4 MTP (108.78 tok/s single-stream, 674 tok/s @ conc=8) — https://ai-muninn.com/en/blog/dgx-spark-gemma4-mtp-108-toks
- allenkuo Medium "Gemma 4 vLLM vs Ollama on 96GB Blackwell" (Apr 8 2026) — https://allenkuo.medium.com/gemma-4-on-vllm-vs-ollama-benchmarks-on-a-96-gb-blackwell-gpu-804ca4845a21
- Spheron Gemma 4 deployment guide — https://www.spheron.network/blog/deploy-gemma-4-gpu-cloud/
- Google Cloud "Optimize Gemma 3 Inference vLLM on GKE" (22k tok/s on 4xH100) — https://medium.com/google-cloud/optimize-gemma-3-inference-vllm-on-gke-c071a08f7c78

### Quantized checkpoints
- RedHatAI/gemma-3-27b-it-FP8-dynamic (FP8 Gemma 3 27B with MMLU 99.89% recovery) — https://huggingface.co/RedHatAI/gemma-3-27b-it-FP8-dynamic
- RedHatAI/gemma-3-27b-it-quantized.w4a16 — https://huggingface.co/RedHatAI/gemma-3-27b-it-quantized.w4a16
- nvidia/Gemma-4-31B-IT-NVFP4 (NVFP4 Gemma 4 31B, -0.31 to -0.45% accuracy delta) — https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4
- nvidia/Gemma-4-26B-A4B-NVFP4 — https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4
- RedHatAI/gemma-4-31B-it-FP8-block (AVOID — vLLM #39407 bug) — referenced in vLLM #39407
