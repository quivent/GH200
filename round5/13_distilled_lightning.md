# Round 5 / Doc 13 — Distillation & Speed Variants for GH200 Serving

**Date:** 2026-05-25
**Baselines:** Llama-3.3-70B-Instruct (dense reasoning), Qwen3.6-27B (dense efficient)
**Hardware target:** NVIDIA GH200 Grace Hopper Superchip (96 GB HBM3 / 144 GB HBM3e variants)
**Serving stacks:** vLLM 0.21+, SGLang, TensorRT-LLM 1.2+
**Reminder from MEMORY:** no FP8 on this stack — assume bf16 everywhere unless explicitly W4A16/INT8.

This document catalogs **quality-for-speed** variants that buy useful throughput on a single GH200
without crossing the FP8 bridge. Six sections, each ending with a recommendation.

---

## 1. EAGLE-3 / EAGLE-2 / Medusa heads — pretrained drafts for our candidate models

EAGLE-3 is the current production speculative-decoding technique (removed EAGLE-2's
feature-prediction constraint; trains a tiny 1–2 layer draft head that reuses target features).
EAGLE-2 dynamic-tree drafts are obsolete; EAGLE-3 wins everywhere it's been measured. Medusa is
older but still has a TensorRT-LLM optimized kernel path on Hopper.

### Pretrained EAGLE-3 heads available today

| Target model | EAGLE-3 head HF ID | Size | Released by | Engines |
|---|---|---|---|---|
| Llama-3.3-70B-Instruct | `lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge` | ~1 B | LMSYS / SpecForge (SpecBundle phase 1) | SGLang (primary), vLLM |
| Llama-3.3-70B-Instruct | `nvidia/Llama-3.3-70B-Instruct-Eagle3` | ~1 B | NVIDIA (TRT-LLM Model Optimizer) | TensorRT-LLM only, Blackwell-tuned |
| Llama-3.3-70B-Instruct | `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B` | ~1 B | EAGLE authors (Yuhui Li) | reference impl, vLLM, SGLang |
| Llama-3.1-8B-Instruct | `lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge` | ~0.25 B | LMSYS | SGLang, vLLM |
| Qwen3-32B | `RedHatAI/Qwen3-32B-speculator.eagle3` | 2 B | Red Hat AI (Speculators format) | vLLM, SGLang |
| GPT-OSS-20B / 120B | `RedHatAI/...` (Eagle3 collection) | — | Red Hat AI | vLLM |
| Mistral-Large-3-675B | `mistralai/Mistral-Large-3-675B-Instruct-2512` (uses `-Eagle` companion) | — | Mistral | vLLM |
| Mistral-Medium-3.5-128B | `mistralai/Mistral-Medium-3.5-128B-EAGLE` | — | Mistral | vLLM |

**Notable gap:** no public EAGLE-3 head for **Qwen3-72B** as of 2026-05-25 (only 32B from Red Hat).
SpecForge framework can train one in ~hours on 8×H100 but you'd be rolling your own.
Mistral-Large (older 123B) also has no public head — only Large-3-675B and Medium-3.5 do.

### Acceptance lengths (real numbers)

From `RedHatAI/Qwen3-32B-speculator.eagle3` measured on 2×A100, vLLM 0.11.0:

| Workload | k=1 | k=3 | k=5 |
|---|---|---|---|
| Coding (HumanEval) | 1.67 | 2.29 | 2.47 |
| Math (GSM8K) | 1.73 | **2.49** | 2.80 |
| Summarization (CNN/DM) | 1.62 | 2.15 | 2.27 |

From `nvidia/Llama-3.3-70B-Instruct-Eagle3` (TRT-LLM, B200, max_draft_len=3), MT-Bench mean
accepted tokens per draft step:
- Math 3.25, Coding 3.18, Extraction 2.60, STEM 2.53, Reasoning 2.40, Humanities 2.30, Roleplay 2.12, Writing 2.10.
- Reasoning/code workloads hit near-maximum (3/3 accepted on 70B target).

### Expected GH200 throughput uplift

EAGLE-3 published numbers (paper + SGLang internal):
- **Single-stream (batch 1–2):** 4.1×–6.5× at temperature 0 on academic benchmarks (Vicuna 13B, Llama-3.1-8B, Llama-3.3-70B).
- **SGLang H100 production:** 1.81× at batch 2, 1.38× at batch 64 (40% throughput at bs=64).
- **SGLang H200:** 1.8× decode at bs=1, 1.5× at bs=32 (MTP via EAGLE).
- GH200 has the same Hopper SM as H100/H200 plus NVLink-C2C to CPU — expect H200-class
  numbers for batch ≥ 2. Single-stream on GH200 you can expect ≈ **1.7–2.0× over bf16 baseline**
  for Llama-3.3-70B and Qwen3-32B; up to **2.5×** on math/code workloads where acceptance is highest.

### Medusa on Hopper

NVIDIA's Medusa path on TensorRT-LLM hit **268 tok/s/user on Llama-3.1-70B / H200**
(1.9× over base on HGX H200 with NVLink Switch). Older heads exist (e.g., `nvidia/Llama-3.1-8B-Medusa-FP8`
— FP8 only, not usable for us). For Llama-3.3-70B specifically, **EAGLE-3 supersedes Medusa**;
use EAGLE-3. The only reason to revisit Medusa would be if you're already locked into TRT-LLM
and need a kernel that's been battle-tested longer.

### Recommendation
- **Llama-3.3-70B:** ship `lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge` under SGLang
  (best out-of-box). vLLM works too. NVIDIA version only if you commit to TRT-LLM.
- **Qwen3-32B:** ship `RedHatAI/Qwen3-32B-speculator.eagle3` under vLLM; that's the only
  production-grade option.
- **Qwen3-72B / Mistral-Large-123B:** no head available; either skip spec-decode or train
  one with SpecForge (9.9× faster training than reference, ~8 H100-hours for 32B-scale).
- Tune `num_speculative_tokens` to **3** (Red Hat's sweet spot — 35.6% accept rate, 2.07 mean
  accepted tokens) for general workloads; bump to 5 for math/code only.

---

## 2. Distilled reasoning models (R1-distill family, OpenThinker, S1, Nemotron)

The 32B–70B distilled reasoners are the most important class for us: they punch at o1-mini level
on math/code while serving with the latency profile of a normal Llama/Qwen base.

### Models, identifiers, key benchmark numbers

| Model | HF ID | Base | AIME 2024 | MATH-500 | GPQA Diamond | LiveCodeBench |
|---|---|---|---|---|---|---|
| **Llama-3.3-70B-Instruct (baseline)** | `meta-llama/Llama-3.3-70B-Instruct` | own | ~16 (no CoT) | 77 | 50.5 | ~33 |
| **DeepSeek-R1-Distill-Llama-70B** | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | Llama-3.3-70B | 70.0 | 94.5 | 65.2 | 57.5 |
| **DeepSeek-R1-Distill-Qwen-32B** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | Qwen2.5-32B | **72.6** | 94.3 | 62.1 | 57.2 |
| **OpenThinker-32B** | `open-thoughts/OpenThinker-32B` | Qwen2.5-32B-Instruct | ~66 (loses to DeepSeek) | **90.6** | **61.6** | ~55 |
| **S1-32B** | `simplescaling/s1-32B` | Qwen2.5-32B-Instruct | 50 → **57 w/ budget forcing** | 93 | 59.6 | — |
| **Llama-3.3-Nemotron-Super-49B-v1.5** | `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | Llama-3.3-70B + NAS | **87.5** | **97.4** | 71.97 | 73.58 |
| **Llama-3.1-Nemotron-Ultra-253B** | `nvidia/Llama-3_1-Nemotron-Ultra-253B-v1` | Llama-3.1-405B → NAS distilled | 72.5 (reasoning on) | 97.0 | 76.0 | 66.3 |
| **Qwen3.6-27B (baseline)** | `Qwen/Qwen3.6-27B` | own | ~62 | 91 | ~58 | ~50 |

(Caveat: AIME 2024/2025 numbers above are pass@1 with reasoning enabled; baseline Llama
numbers are without long-CoT and are not directly comparable in serving — quoted for context.)

### Quality assessment vs the two baselines

- **DeepSeek-R1-Distill-Llama-70B**: same architecture and serving footprint as
  Llama-3.3-70B; +54 pts on AIME, +17.5 on MATH-500, +15 on GPQA. **Zero serving-side cost
  for the upgrade** — drop-in replacement. The only catch: longer outputs (CoT) eat tokens-per-query.
- **DeepSeek-R1-Distill-Qwen-32B**: beats the 70B distill on AIME, ties on MATH-500. Half the
  parameters → ~1.7× higher throughput. **Best Pareto point** for math/code reasoning under 96 GB.
- **OpenThinker-32B**: fully open data (OpenThoughts-114k). Edges DeepSeek on MATH-500, loses
  on AIME. Use if you care about reproducible training data.
- **S1-32B**: 1k-example finetune + "budget forcing" (append "Wait" to extend CoT). Sample-efficient
  research curiosity, **not a production choice** — DeepSeek-R1-Distill-Qwen-32B beats it on every
  axis.
- **Nemotron-Super-49B-v1.5**: NAS-distilled from Llama-3.3-70B-Instruct. Replaces some attention
  blocks → fits **single H200 / single GH200**. Hits **97.4 MATH-500 and 87.5 AIME-2024** — the
  highest open-weight scores on a model this small. Throughput per token ~1.3–1.5× a dense 70B.
  Caveat: TRT-LLM-only officially supported by NVIDIA NIM; vLLM works for the v1.5 release but
  with manual config. Known crash at >15 concurrent requests with "detailed thinking on".
- **Nemotron-Ultra-253B**: too big for single GH200 (needs 2× even at bf16 with KV); skip
  unless you have multi-node NVLink. Listed for completeness.

### Expected GH200 throughput

Distilled models inherit their base's serving footprint:
- DeepSeek-R1-Distill-Llama-70B: ~**40–50 tok/s/user** single-stream, **~800–1200 tok/s aggregate**
  at batch 16, bf16, vLLM, 96 GB GH200. (Same as Llama-3.3-70B-Instruct because architecture
  is identical.) Artificial Analysis shows median provider at 43.2 tok/s; SambaNova hits
  366.9 tok/s (custom hardware, not GH200).
- DeepSeek-R1-Distill-Qwen-32B: ~**70–90 tok/s/user** single-stream, **~2000+ tok/s aggregate**
  at batch 32 on GH200 bf16.
- Nemotron-Super-49B-v1.5: ~**55–70 tok/s/user**, fits with KV headroom on a 96 GB GH200 at
  modest context. v1.5-NVFP4 and v1.5-FP8 exist but we can't use them.

### Recommendation
- **For reasoning workloads**: ship **DeepSeek-R1-Distill-Qwen-32B** as the default. Add
  **Nemotron-Super-49B-v1.5** as the "premium reasoning" tier if you need MATH-500 ≥ 97.
- **For "smarter Llama-3.3-70B" with no infra change**: drop in
  **DeepSeek-R1-Distill-Llama-70B**. Identical serving config, +5–15 pts on every reasoning
  benchmark. The single best zero-friction upgrade in this entire document.

---

## 3. Lightning / Hyper / Turbo / SDXL-Turbo-style variants for LLMs

The diffusion-world step-distillation (Turbo, Lightning, Hyper-SD, LCM) **does not have a direct
LLM analog**. The reason: text generation is already autoregressive single-step-per-token; the
"step compression" idea targets multi-step denoising loops. No public LLM exists labeled "Turbo"
or "Lightning" with step-distillation semantics.

What does exist under similar marketing language:
- **Together AI's "Turbo" tier**: this is **INT8/FP8 quantization**, not distillation. Llama-3.1-70B-Turbo
  is just FP8 — unusable for us.
- **Groq "Turbo"**: hardware (LPU) not model.
- **Anthropic "Haiku-style" distillation**: closed-source.
- **Block-diffusion + EAGLE ("DFlash", 2026)**: hybrid that wraps a small diffusion block around a
  spec-decode loop. Claims 6× over base on academic benchmarks. Research-grade; no public
  weights for our targets yet.

The closest "step compression" analogs for LLMs are:
1. Speculative decoding (Section 1) — proper analog: do K target-model steps per draft check.
2. **Multi-Token Prediction (MTP)**: Qwen3-Next and DeepSeek-V3 train an MTP head natively, giving
   ~2× decode speedup without a separate spec-decode pipeline. Covered in Section 6.
3. **Lookahead decoding** (Jacobi-style): supported in TRT-LLM, no extra weights needed.
   ~1.4–1.8× on Llama-3.3-70B at batch 1.

### Recommendation
- There is no "Llama-3.3-70B-Lightning" to grab. The honest equivalents are EAGLE-3 (Section 1)
  and MTP-native models (Section 6). Anyone selling "Turbo" weights is selling quantization,
  which we've ruled out for FP8.

---

## 4. N-gram + dynamic speculative decoding in vLLM 0.21 — "free" speedup

vLLM 0.18.0+ moved n-gram drafting to GPU (was CPU-bound, ate the savings). vLLM 0.21 has
the async-scheduler-compatible GPU path and a draft-cost reduction PR cutting drafting time
38% (O(ngram_range·tokens) → O(tokens)).

### Exact configuration (vLLM 0.21+)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,      # n-gram window for prompt match
        "prompt_lookup_min": 2,      # optional, default 2
    },
    tensor_parallel_size=1,          # GH200 single-GPU
    dtype="bfloat16",
)
```

CLI equivalent:
```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'
```

### Expected gain on GH200

Red Hat published numbers (vLLM, GPT-OSS-120B, H200-PCIe-141GB — closest public analog to GH200):
- ShareGPT (conversational): **+20.7% throughput, +20.3% median latency, +12.4% ITL P95**
- MLPerf summarization (TP=2): +16.0% throughput, +12.4% median latency
- SWE-bench (code): **+20.5% throughput, +15.9% median latency, +17.5% ITL P95**, $0.85/M
  tokens cost reduction (-19.4%).

Note Red Hat used Eagle3 with 3 draft tokens for the *headline* numbers; the n-gram-only path
typically gives ~10–20% throughput when the workload has prompt repetition (RAG, structured
output, code completion in a known file). On open-ended chat without repetition: 0–5% gain,
sometimes net slowdown.

### Where n-gram is "free" on Llama-3.3-70B / Qwen3-32B
- **RAG / long-context summarization**: prompt has 5k–30k tokens, model frequently quotes back
  fragments → high n-gram hit rate. Expect **15–25% throughput**.
- **Tool use / structured output**: JSON keys repeat. Expect **10–20%**.
- **Code completion in-file**: function names repeat. Expect **15–25%**.
- **Open-ended chat / creative writing**: ~0%, sometimes -2%. Disable.

### Recommendation
- **Always enable n-gram spec-decode by default** for any serving endpoint that handles RAG,
  tool-use, or code workloads. It's a one-line config change with no extra weights.
- **Use n-gram instead of EAGLE-3** for very large batch sizes (≥64) where EAGLE-3's tree-attention
  overhead can eat the gains; n-gram has near-zero per-step cost.
- **Layer n-gram under EAGLE-3** is **not** supported as a stacked mode in vLLM 0.21 — pick one.

---

## 5. Prompt-distilled "small" models that punch above their weight

### Numbers (zero-shot or 5-shot per official reports)

| Model | HF ID | Params | MMLU | HellaSwag | GSM8K | Context |
|---|---|---|---|---|---|---|
| **SmolLM3-3B** | `HuggingFaceTB/SmolLM3-3B` | 3 B | 68.9 (Global MMLU) | 78.5 | — | 128k |
| Llama-3.2-3B-Instruct (ref) | `meta-llama/Llama-3.2-3B-Instruct` | 3 B | 66.5 | 76.2 | ~70 | 128k |
| Qwen3-4B (ref) | `Qwen/Qwen3-4B` | 4 B | 69.4 | 79.1 | ~85 | 128k |
| **Granite-4.1-3B-Instruct** | `ibm-granite/granite-4.1-3b-instruct` | 3 B | 67.0 | — | — | 512k |
| **Granite-4.1-8B-Instruct** | `ibm-granite/granite-4.1-8b-instruct` | 8 B | 73.8 | — | — | 512k |
| **Granite-4.1-30B-Instruct** | `ibm-granite/granite-4.1-30b-instruct` | 30 B | **80.2** | — | — | 512k |
| **OLMo-2-7B-Instruct** | `allenai/OLMo-2-1124-7B-Instruct` | 7 B | 63.7 | — | — | 4k |
| **OLMo-2-13B-Instruct** | `allenai/OLMo-2-1124-13B-Instruct` | 13 B | 67.5 | — | — | 4k |
| **OLMo-2-32B-Instruct** | `allenai/OLMo-2-0325-32B-Instruct` | 32 B | ~75 | — | — | 4k |
| Llama-3.3-70B (anchor) | — | 70 B | 81.9 | — | 92 | 128k |
| Qwen3.6-27B (anchor) | — | 27 B | ~80 | — | ~90 | 128k |

Granite-4.1 specifically: "Granite 4.1-30B performs within ~2 points of Llama-3.1-70B on
MMLU-Pro and GSM8K" per IBM's release (April 2026). At 30B with 512k context that's
extraordinary on a per-VRAM basis.

### Quality picture
- **SmolLM3-3B**: matches Qwen3-4B on knowledge benchmarks, 128k context, hybrid reasoning
  mode. The best 3B available. Targets edge / multiplexed serving.
- **Granite-4.1-30B**: 80.2 MMLU is the headline — beats most 30B-class models, **2 pts below
  Llama-3.1-70B** while serving at ~30B latency. 512k context is huge for RAG.
- **Granite-4.1-8B**: rivals Llama-3.1-8B-Instruct on OpenLLM Leaderboard v1+v2; 512k context
  makes it competitive with 13B-class models for long-doc workloads.
- **OLMo-2-32B**: first fully-open (data + code + weights) model to beat GPT-3.5-Turbo and
  GPT-4o-mini. Competitive with Llama-3.1-70B on reasoning at half the VRAM. Only 4k native
  context — disqualifier for many use cases.

### Expected GH200 throughput

Small dense models on a GH200 are I/O-bound and ridiculous:
- SmolLM3-3B bf16: ~**400–600 tok/s/user** single-stream, **10k+ tok/s** aggregate at batch 64.
- Granite-4.1-8B bf16: ~**250–350 tok/s/user**, **6k+ tok/s** aggregate.
- Granite-4.1-30B bf16: ~**90–120 tok/s/user**, **2k–3k tok/s** aggregate. Same neighborhood
  as Qwen3.6-27B but with much longer context budget.
- OLMo-2-32B: same as a Qwen2.5-32B (~80–100 tok/s/user single-stream).

### Recommendation
- **Granite-4.1-30B**: best quality-for-VRAM in the small dense class. If 80 MMLU is
  acceptable and you need 512k context, this beats Qwen3.6-27B for long-doc RAG.
- **SmolLM3-3B**: serve as the "fast tier" for chat completions, classification, routing,
  reranking. Multiplex hundreds of concurrent users per GH200.
- **OLMo-2** is academically important but the 4k context kills it for production.

---

## 6. MoE small-active variants — "feels like a small model, talks like a big one"

This is the most interesting class for GH200: small *active* parameter counts mean memory
bandwidth (the GH200's strength: 4.0 TB/s HBM3e) directly translates to throughput, while the
*total* parameter count gives big-model quality.

### Candidates

| Model | HF ID | Total | Active | MMLU | Quality reference |
|---|---|---|---|---|---|
| **GPT-OSS-20B** | `openai/gpt-oss-20b` | 20.9 B | **3.6 B** | **85.3** | matches o3-mini |
| GPT-OSS-120B | `openai/gpt-oss-120b` | 117 B | 5.1 B | ~88 | — |
| **Qwen3-Next-80B-A3B-Instruct** | `Qwen/Qwen3-Next-80B-A3B-Instruct` | 80 B | **3 B** | 78.5 | rivals Qwen3-235B |
| Qwen3-Next-80B-A3B-Thinking | `Qwen/Qwen3-Next-80B-A3B-Thinking` | 80 B | 3 B | — | 92 GSM8K |
| **DeepSeek-V2-Lite** | `deepseek-ai/DeepSeek-V2-Lite` | 15.7 B | **2.4 B** | ~60 | beats 7B dense |
| DeepSeek-Coder-V2-Lite-Instruct | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | 15.7 B | 2.4 B | — | HumanEval 81.1, Mean 86.4 |

### Quality vs the baselines
- **GPT-OSS-20B is the standout**: 85.3 MMLU and 89.3 on AIME-2025 from a model with **3.6 B
  active params**. That's **better MMLU than Llama-3.3-70B (81.9)** with ~5% the FLOPS per token.
  GPQA Diamond 68.8, MMLU-Pro 74.8, LiveCodeBench 77.7. Quality is consistently in the
  Qwen3.6-27B-or-better range.
- **Qwen3-Next-80B-A3B**: 78.5 MMLU (below Llama-3.3-70B), but **rivals Qwen3-235B-A22B
  quality at 10× the throughput for >32k context** because of hybrid attention + native MTP.
  Decode throughput at 4k context is 4× Qwen3-32B; at long context, 10×. Thinking variant
  hits 92 GSM8K.
- **DeepSeek-V2-Lite**: older (May 2024), only ~60 MMLU. Below both our baselines.
  Coder-V2-Lite is the right variant — Mean code score 86.4 across Python/Java/JS, 81.1 HumanEval.
  Useful as a **code-only** small-active model. Not a general-purpose pick.

### Expected GH200 throughput (the actual win)

Throughput on MoE small-active is **dominated by active-param read bandwidth**, not by total
parameter storage. The GH200's 4 TB/s HBM3e (or 3 TB/s on the 96 GB HBM3 variant) gives:

- **GPT-OSS-20B** (3.6 B active, bf16 = ~7 GB read per token):
  - Single-stream: ~**500–600 tok/s** theoretical, ~**350–450 tok/s** observed.
  - At batch 32: **8k–12k tok/s aggregate**. **Feels like a 7B model, talks like Llama-3.3-70B.**
- **Qwen3-Next-80B-A3B** (3 B active, bf16 ~6 GB/token, but expert routing overhead):
  - vLLM recipes documented to work on 4×H200/H20 due to *total* size (160 GB at bf16). On a
    96 GB GH200 you'll need INT8 quantization to fit (W8A16 OK, FP8 not allowed for us).
  - On a 144 GB GH200 HBM3e: fits bf16 with tight KV budget. Expect **~300–400 tok/s/user**,
    **6k–8k tok/s aggregate at batch 16**.
- **DeepSeek-V2-Lite** (15.7 B total bf16 = 31 GB, 2.4 B active):
  - Fits with massive headroom. Expect **~600+ tok/s/user**, **15k+ tok/s aggregate**.
  - Quality limits its usefulness.

### Recommendation
- **GPT-OSS-20B is the single highest-value pick in this entire round.** Quality at Llama-3.3-70B
  parity (or above on math/code), serving footprint of a 7B dense, easily fits on any GH200
  variant in bf16, native vLLM and SGLang support. **Add this as a serving tier.**
- **Qwen3-Next-80B-A3B-Instruct** is the right answer **only if you need 32k+ context
  throughput**. The 10× long-context decode win is real and unique. On 96 GB GH200 you need
  W8A16 quantization to fit; on 144 GB it's bf16-fitable. MTP head built-in (no separate
  EAGLE-3 needed).
- **DeepSeek-V2-Lite**: skip the general variant. Use **Coder-V2-Lite-Instruct** only if you
  have a dedicated code-completion endpoint.

---

## Summary table — pick list for GH200, bf16, no-FP8 constraint

| Tier | Workload | Model | HF ID | GH200 tok/s/user (est) | Notes |
|---|---|---|---|---|---|
| Premium reasoning | math/code, willing to wait | Nemotron-Super-49B-v1.5 + EAGLE-3 (none yet) | `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` | 55–70 | 97.4 MATH-500, TRT-LLM only |
| Premium reasoning, drop-in | math/code, want Llama infra | DeepSeek-R1-Distill-Llama-70B + EAGLE-3 | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` + `lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge` | 40–50 × 1.7–2.0 = **70–100** | best zero-friction upgrade |
| Balanced reasoning | math/code, throughput-conscious | DeepSeek-R1-Distill-Qwen-32B + EAGLE-3 head | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` + `RedHatAI/Qwen3-32B-speculator.eagle3` | 70–90 × ~1.7 = **120–150** | head trained on Qwen3-32B; check transfer to distill |
| MoE high-throughput | mixed, want big-model quality | GPT-OSS-20B + EAGLE-3 head (Red Hat) | `openai/gpt-oss-20b` + `RedHatAI/gpt-oss-20B-speculator.eagle3` | 350–450 × ~1.3 = **450–600** | best Pareto point overall |
| Long-context MoE | 32k+ context, throughput-critical | Qwen3-Next-80B-A3B-Instruct (native MTP) | `Qwen/Qwen3-Next-80B-A3B-Instruct` | 300–400 | needs W8A16 on 96 GB GH200 |
| Long-context dense | 512k RAG | Granite-4.1-30B-Instruct | `ibm-granite/granite-4.1-30b-instruct` | 90–120 | 80 MMLU, 512k context |
| Fast tier | chat, routing, rerank | SmolLM3-3B | `HuggingFaceTB/SmolLM3-3B` | 400–600 | 68.9 MMLU, 128k context |
| Free speedup overlay | any workload with prompt repetition | + n-gram speculative decoding | (vLLM config) | +10–25% throughput | RAG/tool/code workloads only |

## Key takeaways

1. **EAGLE-3 is the universal serving accelerator for dense models on GH200.** Heads exist for
   Llama-3.3-70B and Qwen3-32B; no public head for Qwen3-72B or Mistral-Large-123B.
   Expect **1.7–2.0× single-stream** on bf16 GH200 for reasoning workloads.
2. **DeepSeek-R1-Distill-Llama-70B is a free upgrade over Llama-3.3-70B-Instruct** for any
   reasoning-flavored use case. Same serving footprint, much better benchmarks.
3. **GPT-OSS-20B (MoE 3.6 B active) is the dark horse**: Llama-3.3-70B-class quality at
   sub-7B-dense serving cost. If you can only add one model to your stack, this is it.
4. **N-gram speculative decoding in vLLM 0.21 is a free 10–25% throughput** on RAG/tool/code —
   one config-flag change, no extra weights. Enable by default for those workloads, disable
   for open chat.
5. **Qwen3-Next-80B-A3B with native MTP** is the only viable option if you need *long-context*
   throughput at scale; bf16 just barely fits on a 144 GB GH200.
6. **"Lightning/Hyper/Turbo" LLM weights do not exist** in a meaningful sense — the diffusion
   step-distillation paradigm doesn't transfer. Anyone selling "Llama Turbo" is selling FP8
   quantization, which we cannot use.

## Sources

EAGLE-3 / Medusa / spec-decode:
- [lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge (HF)](https://huggingface.co/lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge)
- [nvidia/Llama-3.3-70B-Instruct-Eagle3 (HF)](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-Eagle3)
- [RedHatAI/Qwen3-32B-speculator.eagle3 (HF)](https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3)
- [EAGLE-3 paper (arXiv 2503.01840)](https://arxiv.org/html/2503.01840v1)
- [Speculators (Red Hat Developer, 2025-11)](https://developers.redhat.com/articles/2025/11/19/speculators-standardized-production-ready-speculative-decoding)
- [Boost Llama 3.3 70B 3x with Medusa (NVIDIA TRT-LLM)](https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/)
- [Low Latency Chapter 1: Medusa on H200 (NVIDIA)](https://developer.nvidia.com/blog/low-latency-inference-chapter-1-up-to-1-9x-higher-llama-3-1-performance-with-medusa-on-nvidia-hgx-h200-with-nvlink-switch/)
- [SGLang vs vLLM 2026 benchmarks](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison)
- [mistralai/Mistral-Medium-3.5-128B-EAGLE (HF)](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B-EAGLE)

Distilled reasoning:
- [deepseek-ai/DeepSeek-R1-Distill-Llama-70B (HF)](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
- [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B (HF)](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
- [R1 Distill Llama 70B - Artificial Analysis](https://artificialanalysis.ai/models/deepseek-r1-distill-llama-70b)
- [R1 Distill Qwen 32B - Artificial Analysis](https://artificialanalysis.ai/models/deepseek-r1-distill-qwen-32b)
- [OpenThinker-32B blog (Open-Thoughts)](https://www.open-thoughts.ai/blog/scale)
- [s1: Simple test-time scaling (arXiv 2501.19393)](https://arxiv.org/abs/2501.19393)
- [Llama-Nemotron paper (arXiv 2505.00949)](https://arxiv.org/html/2505.00949v1)
- [nvidia/Llama-3_3-Nemotron-Super-49B-v1_5 (HF)](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5)
- [nvidia/Llama-3_1-Nemotron-Ultra-253B-v1 (HF)](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1)
- [Nvidia Nemotron Ultra outperforms DeepSeek R1 (VentureBeat)](https://venturebeat.com/ai/nvidias-new-llama-3-1-nemotron-ultra-outperforms-deepseek-r1-at-half-the-size)

vLLM n-gram + serving:
- [vLLM N-Gram Speculation docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/)
- [Performance improvements with speculative decoding in vLLM for gpt-oss (Red Hat, 2026-04)](https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss)
- [vLLM v0.18 / v0.19 update notes (Fazm Blog)](https://fazm.ai/blog/vllm-update-april-2026)

Small dense models:
- [SmolLM3 blog (HuggingFace)](https://huggingface.co/blog/smollm3)
- [HuggingFaceTB/SmolLM3-3B (HF)](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
- [IBM Granite 4.1 review (SpecPicks 2026)](https://specpicks.com/reviews/ibm-granite-4-1-local-inference-benchmarks-2026)
- [OLMo 2 32B (Ai2 blog)](https://allenai.org/blog/olmo2-32B)
- [2 OLMo 2 Furious paper (arXiv 2501.00656)](https://arxiv.org/pdf/2501.00656)

MoE small-active:
- [gpt-oss model card (arXiv 2508.10925)](https://arxiv.org/html/2508.10925v1)
- [GPT-OSS-20B deployment analysis (arXiv 2508.16700)](https://arxiv.org/pdf/2508.16700)
- [Introducing gpt-oss (OpenAI)](https://openai.com/index/introducing-gpt-oss/)
- [Qwen/Qwen3-Next-80B-A3B-Instruct (HF)](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- [Qwen3-Next vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-Next.html)
- [deepseek-ai/DeepSeek-V2-Lite (HF)](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)
- [DeepSeek-V2 paper (arXiv 2405.04434)](https://arxiv.org/pdf/2405.04434)

GH200-specific:
- [How to serve DeepSeek-R1 & v3 on NVIDIA GH200 (Lambda Labs)](https://lambda.ai/blog/how-to-serve-deepseek-r1-v3-on-gh200)
