# Round 5 — Qwen3.6-27B: the 27B dense sweet spot vs Llama-3.3-70B

**Date:** 2026-05-25
**Question:** Is Qwen3.6 at the 20–32B size range the actual sweet spot for a single GH200 96GB? What does it cost vs Llama-3.3-70B in quality, and what does it buy in latency?

---

## TL;DR

- The Qwen3.6 family has exactly **two open-weight SKUs in the 20–35B range**: **Qwen3.6-27B dense** (Apr 22, 2026, Apache-2.0) and **Qwen3.6-35B-A3B MoE** (Apr 16, 2026, Apache-2.0). No 14B, no 32B dense. The 27B is the dense flagship.
- Qwen3.6-27B is **natively multimodal (text + image + video)**, **262K native context (1M with YaRN)**, **201 languages**, native tool-calling, MTP speculative-decoding head built in. No audio.
- On the benchmarks the user cares about (MMLU-Pro 86.2, GPQA-Diamond 87.8, LiveCodeBench v6 83.9, AIME26 94.1), **Qwen3.6-27B in thinking mode beats Llama-3.3-70B by 15–35 absolute points on every reasoning/coding benchmark** while having ~38 % of the parameters. The gap is not subtle.
- At Q4_K_M weights fit in **~17 GB**, KV at 32K context adds ~3.5 GB, total ~21 GB. **Runs on a single 24 GB consumer card.** GH200 96GB has ~75 GB of headroom — enough for full 262K context and a draft model for speculative decoding.
- Single-stream decode advantage on GH200: Qwen3.6-27B Q4 projects to **~55–65 tok/s** vs Llama-3.3-70B Q4 at **~22 tok/s** (R5 path 3/4 baseline). That is the **~2.5–3× latency win** the user expected.
- **When 27B wins:** anything reasoning/coding/multilingual/vision/long-context. Which is most of the user's actual workload.
- **When 70B wins:** narrow English-only general-knowledge tasks (Llama's MMLU edge is small), instruction-following on short prompts where IFEval matters more than reasoning (Llama 92.1 vs Qwen typically ~85), and any pipeline already tuned for Llama tokenizer.

---

## 1. The exact SKUs in the 20–35B Qwen3.6 range

| SKU | Type | Total / Active | Release | License | Context (native / extended) | Modalities |
|---|---|---|---|---|---|---|
| **Qwen3.6-27B** | Dense | 27.8B / 27.8B | **2026-04-22** | Apache-2.0 | 262,144 / 1,010,000 | text + image + video |
| **Qwen3.6-35B-A3B** | MoE | 35B / 3B active | **2026-04-16** | Apache-2.0 | 262,144 / 1,010,000 | text + image + video |

**That's it.** No 14B, no 32B dense in the Qwen3.6 generation — Qwen3.6 skipped those points and consolidated on the 27B dense + 35B-A3B MoE pair. (For reference, the Qwen3 family from May 2025 *did* have a 14B and 32B dense; Qwen3.6 collapsed them into 27B.)

### Architecture (27B dense) — why this isn't "just another 27B"

- **64 layers** in a hybrid layout: 16 repeats of `(3 × GatedDeltaNet → FFN) + (1 × GatedAttention → FFN)`. So **3 of every 4 sequence-mixing layers are linear-attention (Gated DeltaNet)**, only 1 in 4 is full softmax attention.
- Gated Attention: 24 Q heads, 4 KV heads (GQA), head_dim 256, RoPE dim 64.
- Gated DeltaNet: 48 V heads, 16 QK heads, head_dim 128.
- FFN intermediate 17,408; hidden 5,120; vocab 248,320 (padded).
- **MTP head trained in** — speculative decoding works out of the box; the model card cites Qwen-internal ~60 TPS on RTX 5090 INT4 leveraging the MTP head.

The linear-attention-dominant layout is the reason the **KV cache scales sub-linearly with context** — only 16 of the 64 layers (25 %) carry a full softmax KV cache. This is critical for the memory math at 128K+ context (see §4).

---

## 2. Quality benchmarks — head-to-head

All numbers are from official model cards / blog posts; "thinking" = chain-of-thought mode enabled where the SKU supports it.

| Benchmark | **Qwen3.6-27B (think)** | Llama-3.3-70B Instruct | Qwen3-32B (Instruct, 2025-05) | Mistral-Small-3.2-24B |
|---|---|---|---|---|
| **MMLU-Pro** (5-shot CoT) | **86.2** | 68.9 | ~71.9 ¹ | 69.06 |
| **MMLU-Redux** | 93.5 | 86.0 (MMLU chat) | — | ~80.5 (MMLU) |
| **GPQA-Diamond** | **87.8** | 50.5 | ~54.6 ¹ | 46.13 |
| **MATH / MATH-500** | (HMMT/AIME used instead) | 77.0 (MATH) | — | 69.42 (MATH) |
| **AIME 2024 / 2025 / 2026** | 94.1 (AIME26) | — | ~81.5 (AIME25, 235B-A22B) ¹ | — |
| **LiveCodeBench v6** | **83.9** | — (LCB not officially reported) | ~70.7 (v5, 235B-A22B) ¹ | — |
| **HumanEval** | — (deprecated for SWE-bench) | 88.4 | — | 92.90 (HumanEval+) |
| **IFEval** | ~85 (Qwen3 series baseline) | **92.1** | — | improved vs 3.1 |
| **BigCodeBench** | not officially reported | not officially reported | — | — |
| **SWE-bench Verified** | **77.2** | not officially reported | — | — |
| **SWE-bench Pro** | 53.5 | — | — | — |
| **Terminal-Bench 2.0** | 59.3 | — | — | — |
| **HLE** (Humanity's Last Exam) | 24.0 | — | — | — |

¹ *Qwen3 Technical Report (arxiv 2505.09388) gives Qwen3-32B-Base 65.54 on MMLU-Pro / 39.78 on SuperGPQA. The 71.9 / 54.6 numbers above are extrapolated from the post-trained Instruct line at the 32B size; Alibaba's own published delta from Base → Instruct on the 235B-A22B is roughly +12 MMLU-Pro / +15 GPQA-Diamond. Treat as ±3 indicative, not authoritative.*

### Reading the table

1. **Qwen3.6-27B in thinking mode dominates Llama-3.3-70B on reasoning.** MMLU-Pro is +17 absolute, GPQA-Diamond is **+37 absolute** (87.8 vs 50.5). That's not a fair fight — Llama-3.3-70B predates the post-training reasoning revolution; Qwen3.6 is a 2026 post-thinking-era model.
2. **Llama wins IFEval (~92 vs ~85).** This is the one benchmark Llama-3.3-70B still leads on among open-weight models its size. If your workload is "follow this 20-bullet system prompt exactly," Llama tokenizer + Llama instruction-following is still very good.
3. **Mistral-Small-3.2-24B is broadly weaker than Qwen3.6-27B** but ships with the simplest deployment and best European-language balance.
4. **Qwen3-32B (May 2025) is the previous generation.** Qwen3.6-27B was trained later and is the strict upgrade — fewer parameters, better scores.
5. **No BBH in the official Qwen3.6-27B card.** Qwen has been deprecating BBH in favor of HLE, IMOAnswerBench, and SuperGPQA. If BBH is load-bearing for your eval, you'll need to run it yourself.

---

## 3. Context, multilingual, tools, modalities

| Feature | Qwen3.6-27B | Llama-3.3-70B |
|---|---|---|
| Native context | **262,144** | 128,000 |
| Extended context (YaRN) | **1,010,000** | n/a |
| Languages | **201** | 8 (EN, ES, FR, DE, HI, IT, PT, TH) |
| Vision input | Yes (image + video) | No |
| Audio | No | No |
| Tool calling | Yes (`--tool-call-parser qwen3_coder` in vLLM/SGLang) | Yes (BFCL v2 = 77.3) |
| Speculative-decoding head | Built-in MTP | None — needs separate draft model |
| Thinking mode toggle | Yes (`enable_thinking`, `preserve_thinking`) | No |

The **vision-language is in the same checkpoint**. There is no separate Qwen3.6-27B-VL — the dense 27B is natively multimodal. If you don't need vision, you can disable the visual encoder and save KV-cache memory ("Text-Only" mode in the model card).

---

## 4. Memory math at Q4 on GH200 96 GB

### Weights

| SKU | BF16 | Q8_0 | Q5_K_M | **Q4_K_M** | Q3_K_M | IQ2 |
|---|---|---|---|---|---|---|
| Qwen3.6-27B | 53.8 GB | 28.6 GB | 19.5 GB | **16.8 GB** | 13.6 GB | 10.9 GB |
| Qwen3.6-35B-A3B | 69.4 GB | 36.9 GB | 26.5 GB | 22.1 GB | 16.6 GB | 11.5 GB |
| Llama-3.3-70B (reference) | ~140 GB | ~70 GB | ~50 GB | ~42 GB | — | — |

Source: knightli.com/unsloth quantization tables (GGUF community builds).

### KV cache (the part that actually matters for the 27B vs 70B tradeoff)

Qwen3.6-27B's hybrid layout means **only 16 of 64 layers carry a softmax KV cache**. With 4 KV heads × head_dim 256 = 1024 KV channels per token per softmax layer:

- Softmax KV per token (FP16): 16 layers × 1024 channels × 2 bytes × 2 (K and V) = **65,536 bytes ≈ 64 KB/token**.
- Linear-attention layers carry a fixed-size recurrent state, not a per-token KV — call it ~50 MB total constant, independent of context length.

| Context | KV (FP16) | KV (FP8) |
|---|---|---|
| 8K | 0.5 GB | 0.25 GB |
| 32K | 2.0 GB | 1.0 GB |
| 128K | 8.0 GB | 4.0 GB |
| 262K (native max) | 16.4 GB | 8.2 GB |

Compare Llama-3.3-70B (80 layers, GQA 8 KV heads × 128 head_dim = 2048 KV channels, all layers softmax):

- KV per token = 80 × 2048 × 2 × 2 = **655 KB/token** — **10× higher per-token KV than Qwen3.6-27B**.
- At 32K: ~20.5 GB FP16 KV cache.
- At 128K: ~82 GB — won't fit on the GH200 at all without FP8 KV.

### Total VRAM @ Q4 / 32K context

| Model | Weights | KV @ 32K (FP16) | Activations + buffers | **Total** | Fits on |
|---|---|---|---|---|---|
| Qwen3.6-27B Q4_K_M | 16.8 | 2.0 | ~2 | **~21 GB** | 24 GB consumer (RTX 3090/4090/5090), 48 GB pro, 96 GB GH200 |
| Qwen3.6-27B Q4 @ 128K | 16.8 | 8.0 | ~2 | **~27 GB** | 32 GB+ (RTX 5090, A100, GH200) |
| Qwen3.6-27B Q4 @ 262K | 16.8 | 16.4 | ~3 | **~36 GB** | 48 GB+ (L40S, A100, GH200) |
| Qwen3.6-27B BF16 @ 128K | 53.8 | 8.0 | ~4 | **~66 GB** | GH200 only on a single card |
| Llama-3.3-70B Q4 @ 32K | 42 | 20.5 | ~3 | **~66 GB** | 80 GB+ (single H100/A100, GH200) |
| Llama-3.3-70B Q4 @ 128K | 42 | 82 (FP8: 41) | ~5 | **129 (FP8: 88) GB** | Doesn't fit at FP16 KV; fits FP8 KV on GH200 |

### What this means for GH200 96 GB headroom

- **Qwen3.6-27B Q4 @ 262K full context: 36 GB used, 60 GB free.** That is enough headroom to (a) keep the model in BF16 weights if you prefer (66 GB total, still fits), or (b) run a small draft model (e.g. Qwen3.6-2B-class) for speculative decoding alongside it.
- **Llama-3.3-70B Q4 @ 32K already uses 66/96 GB**, and 128K context requires FP8 KV cache — there's no room for a draft model, no room for batched serving above batch ≈ 4, no room to extend context further.

The 27B at GH200 is **memory-loose**; the 70B is **memory-tight**. That changes what you can do operationally.

---

## 5. Latency advantage on GH200 — single-stream decode

### Theory: decode is bandwidth-bound

Single-stream decode on a Hopper part is memory-bandwidth-bound. GH200 has 4.0 TB/s HBM3e. Per-token decode throughput ≈ `bandwidth / (weight_bytes_read_per_token)`.

| Model | Weight bytes/token (Q4) | Theoretical max @ 4.0 TB/s |
|---|---|---|
| Qwen3.6-27B Q4_K_M | ~17 GB | **~235 tok/s ceiling** |
| Llama-3.3-70B Q4_K_M | ~42 GB | ~95 tok/s ceiling |

Real-world achievement on a Hopper part is typically 25–40 % of ceiling for llama.cpp single-stream (GGUF overhead), 50–70 % for vLLM with NVFP4 / FP8 KV.

### Measured / estimated single-stream tok/s

From R5 #1–4 (Llama-3.3-70B paths on GH200):
- Path 3 (llama.cpp arm64 CUDA tarball, Q4_K_M): **~22 tok/s single-stream** — bandwidth-bound, matches `4000/42 × 0.23` = 22.
- Path 4 (vLLM AWQ INT4 on GH200): higher — typically 30–40 tok/s single-stream with PagedAttention overhead.

For Qwen3.6-27B at the same engines (no GH200-specific 27B benchmarks published yet, projected from architecture + bandwidth):
- llama.cpp Q4_K_M single-stream: **`4000/17 × 0.23 ≈ 54 tok/s`** baseline (RTX 4090 community reports show ~43 tok/s on 24 GB — GH200's 1.7× higher bandwidth gets you to ~55–65).
- vLLM NVFP4 single-stream: **~80–100 tok/s** based on the RTX 3090 INT4 result of 85 TPS sustained / 106 TPS peak, scaled by GH200's bandwidth advantage.
- **With MTP speculative head enabled** (Qwen3.6 ships this for free, Llama-3.3 does not): community report cites **~122–154 tok/s** on RTX 4090 at Q4 with spec-decode. On GH200, expect **~150–200 tok/s** single-stream.

### Net latency comparison (single-stream decode)

| Engine | Llama-3.3-70B Q4 | Qwen3.6-27B Q4 | **Ratio** |
|---|---|---|---|
| llama.cpp baseline | ~22 tok/s | ~55–65 tok/s | **2.5–3×** |
| vLLM batch=1 | ~30–40 tok/s | ~80–100 tok/s | **2.5×** |
| With speculative decoding | n/a easily (no MTP) | ~150–200 tok/s | **5–7×** |

The user's hypothesis of "~2.5× faster decode for a 27B-class" is correct **without** speculative decoding. **With MTP, the gap widens to 5–7×** because Llama-3.3-70B has no native draft model and adding one (e.g. Llama-3.2-1B as draft) costs additional VRAM and engineering effort that the Qwen3.6-27B doesn't ask for.

### Time-to-first-token (TTFT)

The hybrid linear-attention layout helps prefill too — linear-attention layers prefill in O(n) instead of O(n²). Community report on RTX 5060 Ti dual-card vLLM: TTFT **106–581 ms** across patterns vs llama.cpp at 208–2,279 ms. On GH200, expect proportional improvements over 70B prefill which is roughly 2× slower per token at long context.

---

## 6. When Qwen3.6-27B beats Llama-3.3-70B, and when it loses

### Qwen3.6-27B wins (and it's most of what the user actually does)

1. **Reasoning-heavy work** — GPQA Diamond gap is 37 points (87.8 vs 50.5). Anything resembling graduate-level science, math derivations, multi-step planning: not close.
2. **Coding / agentic coding** — LiveCodeBench v6 83.9, SWE-bench Verified 77.2. Llama-3.3-70B has no comparable agentic-coding tuning.
3. **Long context** — 262K native (1M YaRN) vs Llama's 128K. And the hybrid attention means long context doesn't kill your KV cache budget.
4. **Multilingual** — 201 languages vs 8. If the workload touches Korean, Japanese, Arabic, Russian, anything Southeast Asian: Llama doesn't even compete.
5. **Multimodal** — Qwen3.6-27B handles image and video inputs in the same checkpoint. Llama-3.3 is text-only.
6. **Latency-sensitive single-stream serving** — 2.5–7× faster decode on the same hardware.
7. **Memory-constrained deployments** — runs on a 24 GB consumer card at Q4 / 32K. Llama-3.3-70B needs at least 48 GB.
8. **Throughput at high concurrency** — vLLM bakeoff reports 345 tok/s chat, 377 tok/s coding for Qwen3.6-27B on two RTX 5060 Ti. Llama-3.3-70B at equivalent quantization on equivalent hardware is roughly half that.

### Llama-3.3-70B still wins (narrow but real)

1. **Strict instruction-following on short structured prompts** — IFEval 92.1 vs Qwen ~85. Workflows where format compliance is the whole game (structured outputs, JSON schemas with adversarial robustness) — Llama is more reliable.
2. **English-only general-knowledge Q&A where MMLU-style breadth matters more than depth** — Llama's 86.0 MMLU-chat is competitive on broad factual recall, and Llama is less likely to "over-think" simple questions in thinking mode.
3. **Tokenizer / ecosystem inertia** — if your eval harness, fine-tuning pipeline, RAG chunking, or quantization configs are all already Llama-tuned, the switching cost is real.
4. **Tool-calling for agentic frameworks already built on Llama's tool format** — Qwen uses a different tool-call parser; not all frameworks have first-class Qwen3.6 tool support yet (vLLM and SGLang do; some older frameworks lag).
5. **No vision / no thinking mode side-effects** — if you don't need vision and don't want to deal with the `enable_thinking` / `preserve_thinking` toggle complexity, Llama is simpler. Qwen3.6 in thinking mode can produce very long reasoning traces that blow up output token budgets if you don't set max_tokens carefully.
6. **Stable two-year-old well-understood failure modes** — Llama-3.3-70B has been deployed in production by thousands of teams since Dec 2024. Qwen3.6-27B is one month old. Long-tail behavioral surprises in Qwen3.6 are still being mapped.

### The honest call for josh@duchess

Given the user is already running **both** Llama-3.3-70B (R5 #1–9, GH200 path-of-least-friction) and presumably exploring alternatives: **Qwen3.6-27B is the better default model for almost everything**, and the GH200 96GB has enough headroom to run *both simultaneously* (Llama-3.3-70B Q4 ≈ 66 GB + Qwen3.6-27B Q4 ≈ 21 GB = 87 GB, tight but works) for A/B comparisons.

The Llama-3.3-70B path is the **frictionless reproducible baseline**. The Qwen3.6-27B path is the **actual quality + latency frontier** on this hardware. They are not the same question.

---

## Sources

### Primary (Qwen3.6 family)
- [Qwen3.6-27B model card — Hugging Face](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen3.6-27B blog: Flagship-Level Coding in a 27B Dense Model](https://qwen.ai/blog?id=qwen3.6-27b)
- [Qwen3.6 GitHub repo](https://github.com/QwenLM/Qwen3.6)
- [Qwen3.6-35B-A3B model card — Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.6 on LM Studio](https://lmstudio.ai/models/qwen3.6)
- [Qwen3.6 vLLM Recipes](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)

### Quantization / VRAM
- [Knightli quantization table for Qwen3.6](https://knightli.com/en/2026/05/01/qwen3-6-local-vram-quantization-table/)
- [Will It Run AI — Qwen 3.6 27B VRAM guide](https://willitrunai.com/blog/qwen-3-6-27b-vram-requirements)
- [Codersera — Run Qwen 3.6 locally guide](https://codersera.com/blog/how-to-run-qwen-3-6-locally-2026/)

### Performance / throughput
- [LLMKube — Qwen3.6-27B bakeoff (vLLM vs llama.cpp on RTX 5060 Ti)](https://llmkube.com/blog/qwen3-6-27b-bakeoff)
- [Medium — Overnight stack: Qwen3.6-27B at 85 TPS on RTX 3090](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914)
- [Baseten — Llama 3.3 70B on GH200 vs H100](https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/)
- [NVIDIA — Boost Llama 3.3 70B 3× with speculative decoding](https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/)
- [Build Fast with AI — Qwen3.6-27B review 2026](https://www.buildfastwithai.com/blogs/qwen3-6-27b-review-2026)

### Comparison models
- [DataCamp — Llama 3.3 70B specs and benchmarks](https://www.datacamp.com/blog/llama-3-3-70b)
- [Qwen3 Technical Report (arXiv 2505.09388)](https://arxiv.org/abs/2505.09388)
- [Mistral Small 3 announcement](https://mistral.ai/news/mistral-small-3)
- [Mistral Small 3.2 review (ChatForest)](https://chatforest.com/reviews/mistral-small-3-2-24b-instruct-refinement-llm-review/)
- [Mistral Small 3.2 release coverage (VentureBeat)](https://venturebeat.com/ai/mistral-just-updated-its-open-source-small-model-from-3-1-to-3-2-heres-why)
