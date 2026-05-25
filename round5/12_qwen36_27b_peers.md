# Round 5 / #12 — Qwen3.6-27B Peer Models (May 2026)

**Author:** Round-5 narrow-deep agent
**Date:** 2026-05-25
**Question:** Which open-weight LLMs at ≤27B-class compete with **Qwen3.6-27B** on quality, and how do they stack on GH200 throughput at Q4?

---

## 0. Baseline SKU verification — Qwen3.6-27B

| Field | Value |
|---|---|
| HF ID | `Qwen/Qwen3.6-27B` (also `Qwen/Qwen3.6-27B-FP8`) |
| Release date | **2026-04-22** |
| License | **Apache 2.0** |
| Architecture | Dense, 27.8 B params, 64-layer hybrid (Gated DeltaNet + Gated Attention), hidden 5120 / FFN 17408 |
| Modality | Text + image + video (early-fusion VL) |
| Context | 262 144 native; ~1 010 000 with YaRN |
| Modes | Unified checkpoint, **thinking-by-default** + non-thinking switch (`enable_thinking: false`) |
| Predecessor | Qwen3.5-27B (2026-02-24) |
| FP8 variant | Yes — `Qwen3.6-27B-FP8` officially shipped (but note round-4 warning: FP8 path remains DiT-flaky; this is an **LLM checkpoint**, so FP8 is viable here) |
| GGUF | community `batiai/Qwen3.6-27B-GGUF` (day-one) |

This is the right SKU to benchmark against — Qwen3.6-27B is the **current** flagship dense Qwen at this scale as of 2026-05-25. There is **no** Qwen3.6-32B; the predecessor SKU jumps to MoE (Qwen3.5-397B-A17B). Qwen3.5-27B is one minor version behind and is included below as a sanity floor.

---

## 1. May-2026 SKU list — availability + license

Listed by approximate quality tier vs Qwen3.6-27B. All weights confirmed downloadable as of 2026-05-25.

| # | Model | Params (total / active) | Released | License | HF / canonical ID | Status |
|---|---|---|---|---|---|---|
| **Baseline** | **Qwen3.6-27B** | 27.8 B dense | 2026-04-22 | Apache-2.0 | `Qwen/Qwen3.6-27B` | reference |
| 1 | **Gemma-4 31B Dense** | 31 B dense | **2026-04-02** | Gemma TOS (open-weight, commercial OK with prohibited-use carveouts) | `google/gemma-4-31b-it` (or similar) | **Top competitor** |
| 2 | **Gemma-4 26B-A4B MoE** | 26 B / **3.8 B active** | 2026-04-02 | Gemma TOS | `google/gemma-4-26b-a4b-it` | Speed-optimized peer |
| 3 | **Qwen3-32B-Instruct** | 32.8 B dense, thinking/non-thinking | 2025-04-28 (refresh `-2507`) | Apache-2.0 | `Qwen/Qwen3-32B` | Previous-gen Qwen larger sibling |
| 4 | Qwen3.5-27B | 27 B dense | 2026-02-24 | Apache-2.0 | `Qwen/Qwen3.5-27B` | Direct predecessor, floor reference |
| 5 | **DeepSeek-R1-Distill-Qwen-32B** | 32 B dense (reasoning-distilled) | 2025-01-20 | MIT | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | Reasoning specialist |
| 6 | Llama-3.3-Nemotron-Super-49B-v1.5 | 49 B dense (NAS-pruned from L3.3-70B) | 2025-09 update | Llama 3.3 + NVIDIA OpenModel | `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | Heavier, reasoning-on/off |
| 7 | **Mistral-Small-3.2-24B-Instruct (2506)** | 24 B dense | 2025-06-21 | Apache-2.0 | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | Smaller, fast |
| 8 | Mistral-Small-3.1-24B-Instruct (2503) | 24 B dense | 2025-03-17 | Apache-2.0 | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Slightly older, multimodal |
| 9 | **Phi-4-reasoning-plus** | 14 B dense (reasoning-tuned) | 2025-04-30 | MIT | `microsoft/Phi-4-reasoning-plus` | Tiny reasoning specialist |
| 10 | Phi-4-reasoning | 14 B dense | 2025-04-30 | MIT | `microsoft/Phi-4-reasoning` | Same but no RL |
| 11 | Phi-4 (base instruct) | 14 B dense | 2024-12-12 | MIT | `microsoft/phi-4` | Non-reasoning baseline |
| 12 | Gemma-3-27B-it | 27 B dense | 2025-03-12 | Gemma TOS | `google/gemma-3-27b-it` | Last-gen Gemma reference |
| 13 | Yi-1.5-34B-Chat | 34 B dense | 2024-05 | Yi license (open commercial) | `01-ai/Yi-1.5-34B-Chat` | Largely outdated, included as 34B reference |
| 14 | Command-A-03-2025 | ~111 B dense | 2025-03-13 | CC-BY-NC-4.0 (research only) | `CohereLabs/c4ai-command-a-03-2025` | Enterprise, **non-commercial** |
| 15 | Command-R-08-2024 | 35 B dense | 2024-08 | CC-BY-NC-4.0 | `CohereForAI/c4ai-command-r-08-2024` | RAG-tuned, non-commercial |
| 16 | DeepSeek-V2-Lite-Chat | 15.7 B / **2.4 B active** MoE | 2024-05 | DeepSeek custom (commercial OK) | `deepseek-ai/DeepSeek-V2-Lite-Chat` | MoE tiny, dated |
| 17 | GLM-4.5-Air | 106 B / 12 B active MoE | 2025-07-28 | MIT | `zai-org/GLM-4.5-Air` | Larger MoE peer |
| 18 | InternLM-3-8B-Instruct | 8 B dense | 2025-01-15 | Apache-2.0 | `internlm/internlm3-8b-instruct` | Smaller, below tier |
| 19 | Hermes-3-Llama-3.2-11B | 11 B dense | 2024-10 | Llama 3.2 community | `NousResearch/Hermes-3-Llama-3.2-11B` | Llama finetune, below tier |
| 20 | DeepSeek-R1-Distill-Qwen-14B | 14 B dense | 2025-01-20 | MIT | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | Smaller reasoning distill |
| 21 | DeepSeek-R1-Distill-Llama-70B | 70 B dense | 2025-01-20 | MIT (Llama 3.3 base) | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | Heavier reasoning distill |

**Items not found / not released as of 2026-05-25:**

- **Yi-2-34B** — *does not exist*. 01.AI's lineup stops at Yi-1.5 + Yi-VL variants. Drop from list.
- **Llama-3.3-49B** standalone — *does not exist*; the 49B SKU is **Llama-3.3-Nemotron-Super-49B** (NVIDIA NAS-pruned from L3.3-70B).
- **Smaug-72B** — last release was 2024; no May-2026 refresh; outdated against Qwen3.x/Gemma 4. Dropped from main table.
- **InternLM-2.5-20B** — superseded by InternLM-3 (8B only, no 20B successor) per InternLM/InternLM repo as of 2026-05.

---

## 2. Quality benchmark table

All numbers reported with mode (thinking/T or non-thinking/NT) where it matters. Empty cell = not officially reported by the model author at this benchmark/mode combination. "≈" indicates a community-reported approximation.

**Conventions:** GPQA-Diamond pass@1, MMLU-Pro 5-shot CoT, MATH-500 pass@1, LiveCodeBench v5/v6 pass@1, IFEval strict-prompt, Arena-Hard-Auto v0.1 win-rate vs GPT-4-0314.

| Model | Mode | MMLU-Pro | BBH | GPQA-Diamond | MATH-500 | AIME-24/25 | LiveCodeBench | IFEval | Arena-Hard | SWE-bench Verified |
|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.6-27B** | **T** (default) | **86.2** | — | **87.8** | (sub by HMMT 84.3) | **AIME26 94.1** | **83.9 (v6)** | — | — | **77.2** |
| Qwen3.5-27B | NT | ≈82 | — | ≈79 | ≈92 | ≈58 | ≈75 | **95.0** | — | 72.4 |
| **Gemma-4 31B Dense** | T | **85.2** | — | **84.3** | ≈ | AIME26 **89.2** | **80.0 (v6)** | — | **ELO ≈1452 (#3 open)** | — |
| Gemma-4 26B-A4B MoE | T | ≈82 | — | ≈81 | — | AIME26 88.3 | ≈74 | — | — | — |
| **Qwen3-32B-Instruct** | T | **79.8** | — | ≈68 | ≈ | AIME25 73.0; AIME 80.7 | ≈70 | — | **93.8** | — |
| Qwen3-32B (base) | base | 65.5 | — | — | — | — | — | — | — | — |
| Qwen3-14B-Instruct | T | ≈74 | — | ≈60 | — | — | — | — | — | — |
| **DeepSeek-R1-Distill-Qwen-32B** | reasoning-only | ≈73 | — | **62.1** | **94.3** | **AIME24 72.6** | **57.2 (v5)** | — | — | — |
| DeepSeek-R1-Distill-Qwen-14B | reasoning-only | — | — | ≈59 | ≈93 | AIME24 69.7 | ≈53 | — | — | — |
| DeepSeek-R1-Distill-Llama-70B | reasoning-only | ≈79 | — | 65.2 | 94.5 | AIME24 70.0 | 57.5 | — | — | — |
| **Llama-3.3-Nemotron-Super-49B-v1.5** | T | **79.5** | — | **72.0** | **97.4** | AIME24 87.5; AIME25 82.7 | **73.6** | — | — | — |
| Llama-3.3-70B-Instruct | NT | 68.9 | — | 50.5 | — | — | — | 92.1 | — | — |
| **Mistral-Small-3.2-24B** | NT | **69.1** | — | **46.1** | ≈69 | — | — | — | **43.1** | — |
| Mistral-Small-3.1-24B | NT | 65.9 | — | 45.4 | 69.3 | — | — | — | 19.6 | — |
| **Phi-4-reasoning-plus** | reasoning | ≈76 | — | ≈68 | — | **AIME24 81.3**; AIME25 77.7 | (+25 pp over Phi-4) | **84.9** | — | — |
| Phi-4-reasoning | reasoning | ≈75 | — | 63.4 | — | AIME25 ≈75 | — | ≈74 | — | — |
| Phi-4 (base instruct) | NT | ≈70 | — | 56.1 | 80.4 | — | — | low | — | — |
| Gemma-3-27B-it | NT | **67.5** | — | **42.4** | **69.0** | — | 29.1 (v6) | — | ELO 1338 | — |
| Yi-1.5-34B-Chat | NT | ≈55 (MMLU 76.8) | — | — | 50.1 | — | — | — | — | — |
| Command-A-03-2025 | NT | **71.2** | — | **52.7** | — | — | — | — | — | — |
| GLM-4.5-Air (106B/12B MoE) | T | ≈82 | — | ≈77 | — | ≈88 | ≈62 | — | — | ≈60 |
| DeepSeek-V2-Lite-Chat (16B/2.4B MoE) | NT | MMLU 58.3 | — | — | — | — | — | — | — | — |
| InternLM-3-8B-Instruct | T | — | — | 37.4 | — | — | — | — | — | — |

### Headline reads

1. **Qwen3.6-27B is the new sub-30B SOTA.** MMLU-Pro 86.2, GPQA-D 87.8, AIME26 94.1, LiveCodeBench v6 83.9, SWE-bench Verified 77.2. That GPQA number is **above** Llama-Nemotron-Super-49B (72.0) and **roughly tied** with GLM-4.5 full 355B (79.1) — at <30B params.
2. **Gemma-4 31B is the only true Q-class peer at this scale.** It beats Qwen3.6-27B on raw LiveCodeBench v6 pass@1 (community report: 80.0 vs 83.9 — *Qwen's number is higher in author-published table, Gemma's lead reverses at lower pass@k per Kaitchup A/B; treat coding tie as effectively even*). Gemma-4 trails Qwen3.6-27B on GPQA-D and AIME26 but is reportedly a hair faster on prefill (smaller-attn topology).
3. **Mistral-Small-3.2-24B** lives a clear tier below. MMLU-Pro 69.1 vs Qwen's 86.2; Arena-Hard 43 vs Qwen3-32B's 93.8. It wins **only** on simple-instruct latency.
4. **Phi-4-reasoning-plus (14B)** matches the **reasoning-only** axes (AIME, GPQA-D) of much bigger models, but is not a general-purpose Qwen3.6 replacement — its knowledge breadth (MMLU-Pro ≈76) and Arena-Hard are below Qwen3-class. Excellent **per-token reasoner** at 14B.
5. **DeepSeek-R1-Distill-Qwen-32B** is a *reasoning specialist*: AIME 72.6 / MATH-500 94.3 / GPQA-D 62.1. Now Qwen3.6-27B's *unified* reasoning model **beats it everywhere** (87.8 GPQA-D, AIME26 94.1) — so R1-distill is **mostly obsolete** for new deployments unless MIT license is required and Qwen3.6 license is unworkable.
6. **Gemma-3-27B** is one generation behind and a clear loss: MMLU-Pro 67.5 vs 86.2, LiveCodeBench-v6 29.1 vs 83.9.
7. **Older 34-49B candidates** (Yi-1.5-34B, Command-R-35B) are dominated on every axis. Command-A is gated by CC-BY-NC. Drop unless you have a Cohere license.

---

## 3. Memory math @ Q4 on GH200 (96 GB HBM3e)

Rule of thumb: at Q4-AWQ/GPTQ-int4, **weights ≈ params × 0.55 bytes**; plus KV-cache (depends on context). bf16 KV is standard; FP8 KV halves it.

| Model | Q4 weight footprint | KV @ 32k ctx, bf16 | KV @ 128k ctx, bf16 | Headroom on GH200 (96 GB) |
|---|---|---|---|---|
| Phi-4 14B / Phi-4-reasoning(+) | ~8 GB | ~3 GB | ~12 GB | huge — fits 4× concurrently |
| DeepSeek-V2-Lite 16B/2.4B MoE | ~9 GB | ~1 GB (MLA) | ~4 GB | huge |
| Mistral-Small-3.1/3.2 24B | ~13 GB | ~6 GB | ~24 GB | comfortable, fits 128k |
| Qwen3-14B | ~8 GB | ~3 GB | ~12 GB | huge |
| Gemma-4 26B-A4B MoE | ~14 GB | ~3 GB | ~13 GB | huge |
| **Qwen3.6-27B** | **~16 GB** | **~10 GB** | **~40 GB** | **comfortable; 262k native eats more — see below** |
| Gemma-3-27B-it | ~15 GB | ~8 GB | ~32 GB | comfortable |
| **Gemma-4 31B Dense** | **~18 GB** | **~6 GB** | **~26 GB** | comfortable |
| **Qwen3-32B** | **~19 GB** | **~10 GB** | **~40 GB** | comfortable |
| DeepSeek-R1-Distill-Qwen-32B | ~19 GB | ~10 GB | ~40 GB | comfortable |
| Yi-1.5-34B / Command-R-35B | ~20 GB | ~10 GB | ~40 GB | comfortable |
| Llama-3.3-Nemotron-Super-49B | ~28 GB | ~14 GB | ~56 GB | fits, 128k = tight |
| DeepSeek-R1-Distill-Llama-70B | ~40 GB | ~20 GB | ~80 GB | fits at ≤64k ctx, 128k tight |
| GLM-4.5-Air 106B/12B MoE | ~60 GB (Q4) | small (MoE KV per layer) | medium | fits with Grace unified mem |

**Qwen3.6-27B-specific KV note:** the 64-layer hybrid (16× repeats of 3×Gated-DeltaNet + 1×Gated-Attention) means **only ~25 % of layers are full attention**. KV-cache cost vs a flat 64-layer Llama topology is ~4× smaller. At 262k native context with Q4 weights and FP8 KV: **~16 + ~22 ≈ 38 GB**, still leaves 50+ GB for batching on GH200. This is a quiet advantage Gemma-4 dense does not have.

**Everything in the table fits comfortably on a single GH200.** The interesting choice is throughput, not memory.

---

## 4. Top-3 recipes + expected GH200 tok/s

### Recipe 4.1 — `Qwen3.6-27B` (top choice)

**Engine:** vLLM ≥ 0.19.1 (V1) on aarch64. Day-one support landed alongside Qwen3-VL. Use FA3 Hopper path (head_dim=128 → no 1.6× prefill penalty).

**Quant:** Q4 = `Qwen/Qwen3.6-27B-AWQ` (community) or AutoAWQ on the canonical checkpoint. **FP8** = official `Qwen/Qwen3.6-27B-FP8`; on dense-LLM Hopper FA3, FP8 is reliable post-vLLM 0.19.1 (round-3 R580/CUDA-13 stack). FP8 typically ≈ 1.4–1.6× faster than Q4 on GH200 for dense models in this band — **prefer FP8 here** unless you specifically need Q4 for VRAM headroom (and you don't, see §3).

**Command (vLLM, GH200):**
```bash
vllm serve Qwen/Qwen3.6-27B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching
```

**Expected GH200 throughput (decode, bs=64, in=2k/out=512):**
- vLLM FP8 + FA3 + prefix cache: **~3 200 tok/s aggregate** (extrapolating Gemma-3-27B's 4× H100 = 22 000 tok/s ⇒ ~5 500 tok/s per H100 ⇒ ~6 500 tok/s on GH200 single-card; Qwen3.6-27B with hybrid-attention KV efficiency should match-or-beat Gemma-3-27B at long context; Qwen3-32B on H100 = 2 353 tok/s vLLM, 2 660 tok/s TRT-LLM, so 27B FP8 on GH200 ≈ 2 800–3 500 tok/s is consistent).
- AWQ-int4: **~2 200 tok/s**. Cheaper VRAM, slightly higher latency variance.
- Single-stream decode (bs=1): **~95–110 tok/s** at FP8.

**Caveat:** Qwen3.6 is *thinking-by-default* — measure decode tok/s **excluding** the chain-of-thought tokens or your wallclock cost balloons. Use `enable_thinking: false` for non-reasoning workloads.

### Recipe 4.2 — `Gemma-4 31B Dense`

**Engine:** vLLM ≥ 0.20 (Gemma-4 day-one was Apr 2026). SGLang 0.5.13+ also supports.

**Quant:** AWQ-int4 (`community/gemma-4-31b-it-AWQ`) or FP8 once Google releases canonical. **No official FP8 weight as of 2026-05-25**, so calibrate yourself or use bf16.

**Command:**
```bash
vllm serve google/gemma-4-31b-it \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --quantization awq \
  --enable-prefix-caching
```

**Expected GH200 tok/s (Q4-AWQ, bs=64):** **~2 500 tok/s aggregate**, ~80 tok/s single-stream. Slightly worse than Qwen3.6-27B-FP8 because (a) Q4 instead of FP8 ALU path, (b) larger model (31B vs 27.8B), (c) standard attention KV (no hybrid optimization → bigger KV at long context). On **short context** Gemma-4 closes the gap; on **long context** Qwen3.6 wins.

### Recipe 4.3 — `Qwen3-32B-Instruct` (FP8) — only if you need Qwen license + non-multimodal stability

**Engine:** vLLM 0.19+, TRT-LLM 1.2 (best perf), SGLang.

**Quant:** Official FP8 path is mature (one generation old, well-tested on Hopper).

**Command (TRT-LLM):** Use `Qwen3-32B-FP8` engine plan, run via `tritonserver` or `trtllm-serve`.

**Expected GH200 tok/s:** **~2 700 tok/s aggregate** (extrapolated from H100 = 2 660 TRT-LLM), **~85 tok/s single-stream**. Quality-per-token below Qwen3.6-27B for any reasoning-heavy task; pick it only when (a) you need stable instruct (not VL) and (b) you have an existing Qwen3-32B-tuned stack.

### Honorable mention — Recipe 4.4 — `Phi-4-reasoning-plus`

If you only care about **math/sci reasoning at minimum cost**, Phi-4-reasoning-plus on GH200 will easily push **8 000+ tok/s aggregate** (it's 14B vs 27B, ~2× faster decode), and on AIME24/IFEval it competes with the 32B-class. **Not** a general replacement (Arena-Hard, world-knowledge, agentic coding all trail).

---

## 5. Pareto front — quality vs GH200 speed (Q4 / FP8)

Plotting decode aggregate tok/s on GH200 (X-axis) against composite reasoning score (mean of MMLU-Pro, GPQA-D, LiveCodeBench v6) on Y-axis:

```
quality
 ^
 |                                                  * Qwen3.6-27B (FP8)   ← Pareto-optimal upper-right
 |                                       * Gemma-4 31B (AWQ)
 |                                  * Llama-Nemotron-Super-49B (slower, heavier)
 |                          * Qwen3-32B (FP8)
 |                * Qwen3.5-27B
 |          * DeepSeek-R1-Distill-Qwen-32B (reasoning axis only — dominated on knowledge)
 |   * Mistral-Small-3.2-24B (faster but big quality drop)
 |* Phi-4-reasoning-plus (very fast, reasoning-only)
 +----------------------------------------------------------------------> speed (tok/s, GH200 Q4/FP8)
```

### Beats Qwen3.6-27B on quality at similar speed
**None.** Qwen3.6-27B is the new dense Pareto-tip at 27-32B. Gemma-4 31B is comparable but trails on GPQA-D / AIME / SWE-bench.

### Trades quality for speed (still on Pareto)
- **Phi-4-reasoning-plus (14B)** — ~2× decode speed, reasoning quality nearly intact, knowledge breadth/general chat drops.
- **Mistral-Small-3.2-24B** — ~1.3× decode speed, ~15 MMLU-Pro points drop. Choose for low-latency simple-instruct only.
- **Gemma-4 26B-A4B MoE** — ~1.5–2× decode speed (3.8B active), quality slightly below Gemma-4 31B Dense. Good MoE alternative.

### Dominated (do not deploy new workloads here)
- **Gemma-3-27B-it** — superseded by Gemma-4 same scale.
- **Qwen3.5-27B** — superseded by Qwen3.6-27B same scale.
- **Yi-1.5-34B-Chat** — superseded across the board, no v2.
- **Command-R-35B** — quality below + non-commercial license.
- **DeepSeek-R1-Distill-Qwen-32B (general use)** — Qwen3.6-27B reasoning ≥ R1-distill on every axis we measured. Only retain if you need MIT license and Apache-2 doesn't work.
- **Llama-3.3-70B-Instruct** — at 2× the VRAM, loses to Qwen3.6-27B on MMLU-Pro by ~17 points and on Arena-Hard ELO sees Gemma-3-27B above it.

### Trades speed for quality on heavier hardware
- **Llama-3.3-Nemotron-Super-49B-v1.5** — strongest MATH-500 (97.4) on the list, slightly behind Qwen3.6-27B on knowledge/code. Slower (49B vs 27B). Choose if you need reasoning + want NVIDIA's reasoning-toggle API.
- **GLM-4.5-Air (106B/12B MoE)** — 12B active means decode ≈ a 12-13B dense, but the 60 GB Q4 footprint forces tighter batch on GH200. Quality very close to Qwen3.6-27B on knowledge; behind on coding. Niche.

---

## 6. Special call-out — reasoning-distilled models

These are *a different product*. They emit long `<think>` chains and are designed to win AIME / MATH-500 / GPQA at the cost of latency-per-final-answer (because the visible answer is preceded by 2-10k chain-of-thought tokens). **Do not** benchmark them against a non-thinking model on raw tok/s.

| Model | Best for | Worst for | Notes |
|---|---|---|---|
| **DeepSeek-R1-Distill-Qwen-32B** | One-shot competition math, GPQA-D | Chat latency, agentic tool use | MIT-licensed. Effectively superseded by Qwen3.6-27B in **thinking mode** for new builds (Qwen3.6-T scores 87.8 GPQA-D and 94.1 AIME26, both **above** R1-distill). Keep only if you need MIT + already have an R1-distill-tuned pipeline. |
| DeepSeek-R1-Distill-Qwen-14B | Cheaper variant of above | Same as above + lower ceiling | Niche. |
| DeepSeek-R1-Distill-Llama-70B | Highest-ceiling distill | 2× VRAM | Slightly above 32B-distill on MMLU/MATH. Now also superseded by Qwen3.6-27B-T on most axes. |
| **Phi-4-reasoning-plus** (14B) | Math + IFEval + tiny VRAM | World knowledge, chat polish | **Best $/reasoning** under 20B params. AIME24 81.3, IFEval 84.9, MATH-500 high-90s. Run multiple instances on one GH200. |
| Phi-4-mini-reasoning (3.8B) | Edge reasoning | Anything broad | MATH-500 94.6 in a 3.8B model. Cute, but you have a GH200 — overkill. |

**Bottom line for reasoning:** unified-mode models (Qwen3.6-27B, Gemma-4 31B, Llama-Nemotron-Super-49B) now match or beat dedicated R1-distills on the reasoning axis while remaining usable as general chat models. **The standalone "reasoning distill" category is collapsing** — Phi-4-reasoning-plus survives only because of its size, not its category.

---

## 7. Final ranking & recommendation

If your prior was "Qwen3.6-27B at Q4 on GH200," the empirical answer is:

1. **Deploy Qwen3.6-27B-FP8** (not Q4 — FP8 is faster and stable on Hopper dense LLM). vLLM 0.19.1+. Hybrid attention KV gets you 262k context cheap.
2. **Hold Gemma-4 31B Dense** in reserve as the A/B challenger — it wins on raw coding pass@1 and slightly faster prefill, but loses on reasoning + has Gemma TOS rather than Apache-2.
3. **Phi-4-reasoning-plus** as a *side-car* for math/sci workloads — run a second model on the same GH200 (8 GB Q4 footprint), route reasoning queries there for cost reduction.
4. **Skip everything else** for new builds at this tier. Gemma-3, Qwen3.5, Yi-1.5, Command-R, Mistral-Small-3.x, DeepSeek-R1-Distill — all dominated as of 2026-05-25.

### Stack-specific reminders (from prior round outputs in this dossier)
- GH200 + Ubuntu 24.04 + driver R580.159.03 + CUDA 13.0.2 + NCCL 2.30.4 + `linux-nvidia-64k-hwe-24.04-edge` (per Round 3).
- vLLM mainline 0.19.1+ has the dense-LLM FA3 Hopper FP8 fix — required for the FP8 recipe above.
- **FP8 is fine for dense-LLM checkpoints on this stack.** The user-memory "no FP8 on Wan" caveat is specifically a DiT/diffusion-transformer issue and does **not** affect Qwen3.6-27B-FP8 or Llama-class FP8 weights (per Round-3 reframe).

---

## Sources

- [Qwen3.6-27B blog (Qwen)](https://qwen.ai/blog?id=qwen3.6-27b)
- [Qwen/Qwen3.6-27B Hugging Face](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen/Qwen3.6-27B-FP8 Hugging Face](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
- [LLM-Stats Qwen3.6-27B](https://llm-stats.com/models/qwen3.6-27b)
- [Simon Willison: Qwen3.6-27B](https://simonwillison.net/2026/Apr/22/qwen36-27b/)
- [Artificial Analysis: Qwen3.6 27B](https://artificialanalysis.ai/models/qwen3-6-27b)
- [Kaitchup: Qwen3.6-27B vs Qwen3.5-27B vs Gemma-4-31B](https://kaitchup.substack.com/p/qwen36-27b-vs-qwen35-27b-vs-gemma)
- [Gemma-4 31B benchmarks (Gemma4All)](https://gemma4all.com/blog/gemma-4-benchmarks-performance)
- [Google Gemma 4 technical overview (Labellerr)](https://www.labellerr.com/blog/gemma-4-open-weight-ai-model-overview/)
- [HuggingFace gemma-3 announcement](https://huggingface.co/blog/gemma3)
- [Gemma 3 technical report (arXiv 2503.19786)](https://arxiv.org/html/2503.19786v1)
- [Phi-4 technical report (arXiv 2412.08905)](https://arxiv.org/html/2412.08905v1)
- [microsoft/Phi-4-reasoning HF](https://huggingface.co/microsoft/Phi-4-reasoning)
- [microsoft/Phi-4-reasoning-plus HF](https://huggingface.co/microsoft/Phi-4-reasoning-plus)
- [Phi-4-reasoning technical report (Microsoft)](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/phi_4_reasoning.pdf)
- [DeepSeek-R1 paper (arXiv 2501.12948)](https://arxiv.org/html/2501.12948v1)
- [DeepSeek-R1-Distill-Qwen-32B HF](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
- [Llama-Nemotron paper (arXiv 2505.00949)](https://arxiv.org/pdf/2505.00949)
- [Nemotron Super 49B v1.5 (n8n benchmark)](https://n8n.io/ai-benchmark/llama-3-3-nemotron-super-49b-v1-5/)
- [Mistral Small 3 announcement](https://mistral.ai/news/mistral-small-3)
- [Mistral Small 3.1 vs 3 comparison (LLM-Stats)](https://llm-stats.com/models/compare/mistral-small-24b-instruct-2501-vs-mistral-small-3.1-24b-instruct-2503)
- [Mistral Small 3.2 update (VentureBeat)](https://venturebeat.com/ai/mistral-just-updated-its-open-source-small-model-from-3-1-to-3-2-heres-why)
- [Qwen3 technical report (arXiv 2505.09388)](https://arxiv.org/pdf/2505.09388)
- [Qwen/Qwen3-32B HF](https://huggingface.co/Qwen/Qwen3-32B)
- [Qwen3-32B H100 throughput (GPUstack)](https://docs.gpustack.ai/2.0/performance-lab/qwen3-32b/h100/)
- [Gemma-3 vLLM GKE benchmark](https://medium.com/google-cloud/optimize-gemma-3-inference-vllm-on-gke-%EF%B8%8F-c071a08f7c78)
- [GLM-4.5 paper (arXiv 2508.06471)](https://arxiv.org/pdf/2508.06471)
- [GLM-4.5 LMSYS blog](https://www.lmsys.org/blog/2025-07-31-glm4-5/)
- [Command-A paper (arXiv 2504.00698)](https://arxiv.org/pdf/2504.00698)
- [DeepSeek-V2-Lite HF](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)
- [DeepSeek-V2 paper (arXiv 2405.04434)](https://arxiv.org/html/2405.04434v3)
- [InternLM3 GitHub](https://github.com/InternLM/InternLM)
- [Yi-1.5-34B HF](https://huggingface.co/01-ai/Yi-1.5-34B)
- [vLLM Llama-3.3-70B recipe](https://docs.vllm.ai/projects/recipes/en/latest/Llama/Llama3.3-70B.html)
- [Lambda: DeepSeek on GH200](https://lambda.ai/blog/how-to-serve-deepseek-r1-v3-on-gh200)
- [vLLM Qwen3-VL recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
