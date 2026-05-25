# Quality Benchmark Anchor Table — Candidate Open-Weight Set (Round 5)

**Date:** 2026-05-25
**Target hardware:** NVIDIA GH200 (96 GB HBM3e, single-stream / batch-1 inference)
**Baseline #1:** Llama-3.3-70B-Instruct
**Baseline #2 (SKU verification):** **Qwen3.6-27B** — verified. Released **2026-04-21** by Qwen. Dense 27.8B parameters, hybrid thinking / non-thinking modes in one checkpoint, native multimodal (text + image), 262K native context (1.01M via YaRN), Apache-2.0. SKU exists at `Qwen/Qwen3.6-27B` on HF.

---

## Methodology & caveats

- **Number harvest:** every cell below was sourced from one of: vendor model card (HF / Mistral / Cohere / Meta), official tech report (arXiv / vendor PDF), Artificial Analysis (AA), llm-stats.com, vals.ai, lambda.ai leaderboard, or LMSYS-derived snapshots.
- **Thinking vs non-thinking:** for hybrid models (Qwen3-Next, Qwen3.6, Kimi K2 Thinking) **thinking-mode** numbers are used because that is how the model is normally evaluated on reasoning benchmarks. Non-thinking would be ~5–25 pts lower on AIME/GPQA.
- **AIME convention:** AIME25 reported when available; AIME24 substituted only where AIME25 is absent.
- **"—" = no public number found** (not zero; not available at the harvest date).
- **Question marks on cells** indicate vendor-reported only (not third-party replicated).
- **Llama-3.3-70B Arena Elo:** the model has been deprecated from active Arena tracking; historical snapshots placed it at ≈1257 (Dec-2024) when it launched, and the May-2026 frontier band is 1450–1561, so Llama-3.3-70B today would land far down the active board. Treat as the floor reference, not a competitive number.

---

## Table 1 — Quality benchmark anchor (model × benchmark)

Scores are percent unless stated. Bold = best in column among the 15 listed models.

| # | Model | Arena Elo (LMSYS) | MMLU-Pro | MMLU-Redux | BBH | GPQA-Diamond | MATH-500 | AIME-24 | AIME-25 | LiveCodeBench (v5/v6) | BigCodeBench-Hard | IFEval | Arena-Hard v2 | MixEval-Hard |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Llama-3.3-70B-Instruct** (baseline) | ~1257 (legacy, Dec-24) [HF, LMSYS hist.] | 68.9 [Meta card] | — | — | 50.5 [Meta card] | 77.0 (MATH CoT) [Meta card] | — | — | — | — | **92.1** [Meta card] | — | — |
| 2 | **Qwen3.6-27B** (dense, hybrid) | not yet rated | **86.2** [HF card] | **93.5** [HF card] | — | **87.8** [HF card] | — | — | — | **83.9** (v6) [HF card] | — | — | — (SWE-bench Verified 77.2) | — |
| 3 | **Qwen3-32B-Instruct** (w/ on-policy distill) | — | 76.5 (tech rpt) | 88.3 [Qwen3 TR] | — | 63.3 [Qwen3 TR] | 97.0 [Qwen3 TR] | 74.4 [Qwen3 TR] | 65.5 [Qwen3 TR] | 60.3 (v5) [Qwen3 TR] | — | ~85 [Qwen3 TR] | ~76 [Qwen3 TR] | — |
| 4 | **Qwen3-Next-80B-A3B-Thinking** | — | 82.7 [HF card] | 92.5 [HF card] | — | 77.2 [HF card] | — | — | **87.8** [HF card] | 68.7 (v6) [HF card] | — | 88.9 [HF card] | 62.3 [HF card] | — |
| 5 | **DeepSeek-V3.1** (671B / 37B active) | — | 81.2 [DS API changelog] | — | — | 68.4 [DS API changelog] | — | — | 59.4 [DS API changelog] | 56.4 [llm-stats] | — | — | — | — |
| 6 | **DeepSeek-R1-Distill-Llama-70B** | — | 71.2 [DS-R1 paper] | — | — | 65.2 [DS-R1 paper] | 94.5 [DS-R1 paper] | 70.0 [DS-R1 paper] | — | 57.5 [llm-stats] | — | — | — | — |
| 7 | **DeepSeek-R1-Distill-Qwen-32B** | — | ~71 (est.) [DS-R1 paper] | — | — | 62.1 [DS-R1 paper] | 94.3 [DS-R1 paper] | 72.6 [DS-R1 paper] | — | 57.2 [llm-stats] | — | — | — | — |
| 8 | **Mistral-Large-3** (675B MoE / 41B active, 2512) | — | 73.1 (also 85.5 8-lang MMLU) [Mistral card] | — | — | 43.9 [vals.ai] | — | — | low | 82.8 (LCB; vendor) [Mistral card] | — | — | — | — |
| 9 | **Mistral-Small-3.2-24B** (24B dense) | — | ~65 (MMLU 80.5) [Mistral card] | — | — | 46.1 [Mistral card] | — | — | 9.3 (Small 3.1) [Mistral card] | — | — | ~89 [Mistral card] | — | — |
| 10 | **Gemma-3-27B-it** | — | 67.5 [Google card] | — | — | 42.4 [Google card] | 89.0 [Google card] | — | 20.8 (AIME26) [Google card] | 29.1 (v6) [Google card] | — | 90.4 [llm-stats] | — | — |
| 11 | **GPT-OSS-120B** (5.1B active MoE) | — | **90.0** (high) [AA, OAI card] | — | — | 80.9 (w/ tools) [OAI card] | — | — | **97.9** (high) [OAI card] | 85.0 (vendor) [OAI card] | — | ~92 [OAI card] | — | — |
| 12 | **Llama-4-Maverick** (17B act / 400B total MoE) | — | 80.5 [Meta card] | — | — | 69.8 [Meta card] | — | — | — | 43.4 [llm-stats] | — | — | — | — |
| 13 | **Hermes-4-70B** (Nous; built on **Llama-3.1**-70B, NOT 3.3) | — | — (MMLU 87.2) [Nous tech rpt] | — | — | 70.5 [Nous tech rpt] | **96.3** [Nous tech rpt] | 81.9 [Nous tech rpt] | — | 61.3 [Nous tech rpt] | — | — | — | — |
| 14 | **Cohere Command A** (111B dense, 03-2025) | — | 71.2 [AA] | — | — | 52.7 [AA] | 81.9 [AA] | — | 13.0 [AA] | — | — | — (IFBench 36.5) [AA] | — | — |
| 15 | **Kimi K2 Thinking** (1T MoE / 32B active) | — | 84.6 (no-tools) [HF card] | **94.4** [HF card] | — | 84.5 [HF card] | — | — | **94.5** (no-tools; 100.0 heavy) [HF card] | **83.1** [HF card] | — | — | — | — |

### Sources by cell-cluster
- **Meta card (Llama-3.3-70B):** `https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct`
- **Qwen3.6-27B card:** `https://huggingface.co/Qwen/Qwen3.6-27B`
- **Qwen3 Technical Report:** arXiv 2505.09388
- **Qwen3-Next-80B-A3B-Thinking card:** `https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking`
- **DeepSeek-V3.1:** DeepSeek API changelog (`api-docs.deepseek.com/updates`)
- **DeepSeek-R1 paper:** arXiv 2501.12948 (contains R1-Distill-* numbers)
- **Mistral Large 3:** `https://docs.mistral.ai/models/mistral-large-3-25-12`, `mistralai/Mistral-Large-3-675B-Instruct-2512` HF
- **Mistral Small 3.2:** Mistral platform docs benchmark page
- **Gemma 3:** Google AI Studio card + apxml mirror
- **GPT-OSS-120B:** OpenAI model card PDF (2025-08-05), AA evaluations page
- **Llama-4-Maverick:** `https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct`
- **Hermes-4-70B:** Nous Research Hermes 4 Technical Report (arXiv 2508.18255)
- **Command A:** AA model page + Cohere arXiv tech rpt 2504.00698
- **Kimi K2 Thinking:** `https://huggingface.co/moonshotai/Kimi-K2-Thinking`
- **Aggregators:** `llm-stats.com/benchmarks/{mmlu-pro,gpqa,livecodebench,arena-hard-v2}`, `artificialanalysis.ai`, `vals.ai`

---

## Table 2 — GH200 deployment fit (Q4 weights, single-stream)

VRAM = Q4 weights + ~10–15 GB KV cache headroom at typical 8–32 K context. "No-offload Y" means model + working KV fits entirely in the GH200's 96 GB HBM3e (no Grace LPDDR offload required). tok/s estimates are **single-user, batch-1, output-decode** based on H100/H200 published numbers scaled by the GH200's wider memory-bandwidth advantage (~7.5x H100 effective at this batch size for memory-bound 70B-class — Lambda Labs). For MoE models the active-param size dominates decode bandwidth.

| Model | Total / Active params | Q4 weights VRAM | Fits GH200 96 GB no-offload? | Expected GH200 single-stream tok/s | License-clean for commercial? |
|---|---|---|---|---|---|
| **Llama-3.3-70B-Instruct** | 70B / 70B dense | ~40 GB | **Y** (with full 128K KV) | ~55–70 tok/s [Lambda GH200/Baseten] | Y (Llama 3.3 Community Lic., <700M MAU clause) |
| **Qwen3.6-27B** | 27.8B / 27.8B dense | ~17 GB | **Y** (huge KV headroom; supports 262K native) | ~120–150 tok/s (est.; 27B dense bw-bound on Hopper) | **Y — Apache-2.0** |
| **Qwen3-32B-Instruct** | 32B / 32B dense | ~20 GB | **Y** | ~100–130 tok/s | **Y — Apache-2.0** |
| **Qwen3-Next-80B-A3B-Thinking** | 80B / 3B active | ~45 GB | **Y** (linear-attn hybrid; small KV) | ~150–200+ tok/s (3B active, hybrid attn) | **Y — Apache-2.0** |
| **DeepSeek-V3.1** | 671B / 37B active MoE | ~380 GB (Q4) | **N** (needs ≥4–8× H200 class or aggressive Grace offload; even Q2_K is ~200 GB) | N/A on single GH200 96 GB; offload mode <10 tok/s | Y (MIT-style DS model lic.) |
| **DeepSeek-R1-Distill-Llama-70B** | 70B / 70B dense | ~40 GB | **Y** | ~50–65 tok/s (long thinking trace inflates wall-clock) | Y (MIT-style) |
| **DeepSeek-R1-Distill-Qwen-32B** | 32B / 32B dense | ~20 GB | **Y** | ~100–130 tok/s | Y (MIT-style) |
| **Mistral-Large-3** | 675B / 41B active MoE | ~360 GB (Q4) | **N** (single-GH200 infeasible; 2× H200 minimum) | N/A | **Y — Apache-2.0** |
| **Mistral-Small-3.2-24B** | 24B / 24B dense | ~14 GB | **Y** (massive KV headroom) | ~140–170 tok/s | **Y — Apache-2.0** |
| **Gemma-3-27B-it** | 27B / 27B dense | ~15 GB | **Y** | ~110–140 tok/s | **Conditional** — Gemma Terms of Use (not Apache); commercial OK but with use-restriction AUP. Gemma-4 moved to Apache-2.0; Gemma-3 did NOT |
| **GPT-OSS-120B** | 117B / 5.1B active MoE | ~63 GB (Q4) or 65 GB MXFP4 native | **Y** (tight: 96 − 63 = 33 GB KV headroom, fine ≤32K) | ~180–250 tok/s (5B active; vLLM Blackwell hits 500+, GH200 lower) | **Y — Apache-2.0** |
| **Llama-4-Maverick** | 400B / 17B active MoE (128 experts) | ~220 GB (Q4) | **N** on single GH200 96 GB; Scout-109B fits at Q4 (~55 GB) instead | N/A on 96 GB | Y (Llama-4 Community Lic.) |
| **Hermes-4-70B** | 70B / 70B dense (Llama-3.1 base) | ~40 GB | **Y** | ~50–65 tok/s (hybrid reasoning inflates trace length) | Y (Llama-3.1 Community Lic., inherits) |
| **Cohere Command A** | 111B / 111B dense | ~60 GB | **Y** (tight; ~30 GB KV headroom) | ~40–55 tok/s | **N — CC-BY-NC 4.0** (research only; commercial requires Cohere contract) |
| **Kimi K2 Thinking** | 1T / 32B active MoE | ~560 GB (Q4) — INT4-native QAT variant ~500 GB | **N** on single 96 GB; even Q2_K ~300 GB | N/A | Y (Modified MIT) |

**Source notes on VRAM/throughput:** Llama-3.3-70B Q4 ≈ 40 GB → fits H100 80 GB / GH200 96 GB with headroom (community measurements via bartowski GGUF, Lambda Labs blog, Baseten GH200 testing). MoE Q4 sizes computed as total_params × 0.5 bytes + ~10–15% group-scale overhead. Single-stream tok/s anchored to: Lambda GH200 7.6× H100 for memory-bound 70B (`lambda.ai/blog/putting-the-nvidia-gh200-grace-hopper-superchip-to-good-use`), Baseten 500+ tok/s GPT-OSS on Blackwell scaled down for Hopper, vLLM/SGLang Llama-3.3 70B recipe page.

---

## Substitution analysis: replace Llama-3.3-70B with **X**

Speed gain expressed as multiplier vs Llama-3.3-70B Q4 GH200 baseline (~60 tok/s nominal). Quality gain = mean-of-overlapping benchmarks (MMLU-Pro, GPQA-Diamond, IFEval where present). Negative numbers = regression.

### Top-3 proposals

**1. Replace Llama-3.3-70B with Qwen3.6-27B (dense, Apache-2.0)**
- **Quality:** MMLU-Pro 68.9 → **86.2** (+17.3 pts), GPQA-Diamond 50.5 → **87.8** (+37.3 pts), MATH/coding categories not even close (SWE-bench Verified 77.2). Even non-thinking mode dominates.
- **Speed:** ~60 tok/s → ~130 tok/s single-stream (~2.2×) — 27B dense is the sweet spot for Hopper-class memory bandwidth.
- **License:** Apache-2.0 (cleaner than Llama Community License).
- **VRAM headroom:** 17 GB weights leaves >75 GB for KV → can run the full 262K native context comfortably or batch-up multiple users.
- > **If you replace Llama-3.3-70B with Qwen3.6-27B, you gain roughly +20–37 pts on hard knowledge/reasoning benchmarks and ~2.2× single-stream throughput, on a fully Apache-2.0 license.**

**2. Replace Llama-3.3-70B with Qwen3-Next-80B-A3B-Thinking (MoE, 3B active, Apache-2.0)**
- **Quality:** MMLU-Pro 68.9 → 82.7 (+13.8), GPQA-Diamond 50.5 → 77.2 (+26.7), AIME25 effectively 0 → 87.8 (reasoning unlocked), IFEval 92.1 → 88.9 (−3.2 instruction-following regression). LiveCodeBench v6 68.7 vs no Llama number.
- **Speed:** ~60 tok/s → ~150–200 tok/s single-stream (≈3×) — 3B active + hybrid linear attention is exceptionally fast per token, but thinking-mode trace length (often 3–10× longer) means **wall-clock to final answer can be slower** despite higher tok/s. For latency-sensitive single-shot use, keep Llama 3.3 or pick #1.
- **License:** Apache-2.0.
- **VRAM:** ~45 GB weights, fits cleanly.
- > **If you replace Llama-3.3-70B with Qwen3-Next-80B-A3B-Thinking, you gain ~+14 to +27 pts on reasoning benchmarks and ~3× raw tok/s, but lose 3 pts of IFEval and pay 3–10× more output tokens per query in thinking mode.**

**3. Replace Llama-3.3-70B with GPT-OSS-120B (MoE, 5.1B active, Apache-2.0)**
- **Quality:** MMLU-Pro 68.9 → **90.0** (+21.1), GPQA-Diamond 50.5 → 80.9 w/tools (+30.4), AIME25 effectively 0 → **97.9** (high effort), IFEval roughly parity (~92).
- **Speed:** ~60 tok/s → ~200–250 tok/s on GH200 single-stream (≈3–4×) — only 5.1B active params.
- **License:** Apache-2.0.
- **VRAM:** 63 GB Q4 (or MXFP4 native ~65 GB) → fits, but KV headroom is only ~33 GB — practical context 32K, not 128K, without offload.
- **Caveat:** vendor numbers; lower at "low effort" reasoning setting; weaker than Qwen on agentic / SWE-bench style tasks per Clarifai head-to-head.
- > **If you replace Llama-3.3-70B with GPT-OSS-120B, you gain roughly +21 pts MMLU-Pro / +30 pts GPQA / unlock high-end math, and 3–4× throughput, but you sacrifice ~3× KV/context headroom (96 GB tighter than the 70B's 40 GB footprint).**

### Honorable mentions / explicit rejects
- **DeepSeek-V3.1, Mistral-Large-3, Kimi K2 Thinking, Llama-4-Maverick:** all **infeasible single-GH200** at Q4. Require multi-GPU or aggressive Grace LPDDR offload (tok/s collapses to single digits). Use only if the deployment is multi-node.
- **Cohere Command A:** clean fit (~60 GB) and decent quality (MMLU-Pro 71.2) but **CC-BY-NC license blocks commercial use** — disqualified unless a Cohere license is bought.
- **Gemma-3-27B-it:** dense, fits trivially, fast — but quality is *behind* Llama-3.3-70B on MMLU-Pro (67.5 vs 68.9) and barely ahead on IFEval. No reason to switch unless multimodal vision is mandatory; in that case prefer Qwen3.6-27B (also multimodal, far better scores).
- **Mistral-Small-3.2-24B:** the fastest fit (~150 tok/s) but quality regression vs Llama-3.3 on MMLU-Pro (~65 vs 68.9). Only pick if latency budget is brutal AND quality drop is acceptable.
- **Hermes-4-70B:** built on **Llama-3.1-70B base, not 3.3** — a sidegrade, not an upgrade in instruction-following, though Hermes adds hybrid reasoning. Picks up reasoning at cost of some general-chat polish. Useful as a Llama-3.3 *companion* (reasoning route) but not a strict replacement.
- **DeepSeek-R1-Distill-{Llama-70B, Qwen-32B}:** good reasoning lift (GPQA 65/62 vs Llama-3.3 50.5) but legacy now that Qwen3-Next and Qwen3.6 exist with cleaner training.

---

## Final recommendation (one-line)

> **Drop Llama-3.3-70B-Instruct, run Qwen3.6-27B as the new daily-driver baseline on a single GH200, and keep GPT-OSS-120B configured as the high-effort reasoning route for AIME/GPQA-heavy queries.**
> Both are Apache-2.0, both fit no-offload, together they cover the speed/quality Pareto frontier of the May-2026 open-weight set.
