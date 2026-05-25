# Round 5 — Peers of Llama-3.3-70B-Instruct on GH200 96 GB

**Date:** 2026-05-25
**Target hardware:** 1× Lambda GH200 96 GB HBM3 + 480 GB LPDDR5X (Grace)
**Reference model:** `meta-llama/Llama-3.3-70B-Instruct` (released 2024-12-06)
**Scope:** open-weight peers at comparable quality, similar or smaller parameter footprint, runnable on a single GH200 at Q4 (GGUF) or W4A16/FP8 (vLLM/TRT-LLM).

---

## 1. May-2026 SKU list — confirmed available & licensed

Verified by HF model card + vendor blog as of 2026-05-25.

| Model | Params (total / active) | Released | License | Single-GH200? |
|---|---|---|---|---|
| **Llama-3.3-70B-Instruct** (baseline) | 70 B dense | 2024-12-06 | Llama 3.3 Community | Yes — FP8 ≈ 70 GB, Q4 ≈ 43 GB |
| **Llama-3.1-Nemotron-70B-Instruct** | 70 B dense | 2024-10 | Llama 3.1 Community + NVIDIA OMA | Yes (same footprint) |
| **Llama-3.3-Nemotron-Super-49B-v1.5** | 49 B dense (NAS-pruned from 70B) | 2025 H2 | NVIDIA OMA + Llama | Yes — FP8 ≈ 49 GB; designed single-H100/H200 |
| **DeepSeek-R1-Distill-Llama-70B** | 70 B dense | 2025-01 | MIT (weights) | Yes — same footprint as Llama-3.3 |
| **Hermes-4-70B** (Nous, hybrid-reasoning) | 70 B dense (Llama-3.1 base) | 2025-08-27 | Llama 3.1 Community | Yes — FP8 ckpt published |
| **Tülu-3-70B (DPO)** | 70 B dense (Llama-3.1 base) | 2024-11 | Llama 3.1 Community + ODC-BY data | Yes |
| **Qwen2.5-72B-Instruct** | 72 B dense | 2024-09 | Qwen Tongyi (commercial OK ≤100 M MAU) | Yes — FP8 ≈ 72 GB, Q4 ≈ 44 GB |
| **Qwen3-32B** | 32 B dense | 2025-04-28 | Apache 2.0 | Yes — FP8 ≈ 32 GB, ample headroom |
| **Qwen3-30B-A3B** (MoE) | 30 B / 3 B active | 2025-04-28 | Apache 2.0 | Yes — ~17 GB Q4 |
| **Qwen3-235B-A22B** (MoE) | 235 B / 22 B active | 2025-04-28 | Apache 2.0 | **Tight** — Q4 ≈ 130 GB; needs CPU-offload via Grace (works) |
| **Qwen3.6-27B** (dense, multimodal) | 27 B dense | 2026-04-22 | Apache 2.0 | Yes — easy |
| **Llama-4-Scout-17B-16E-Instruct** | 109 B / 17 B active MoE | 2025-04-05 | Llama 4 Community | Yes — INT4 fits 80 GB; FP8 ≈ 109 GB needs Grace offload |
| **Llama-4-Maverick-17B-128E-Instruct** | 400 B / 17 B active MoE | 2025-04-05 | Llama 4 Community | **No** at FP8 (≈400 GB); only via Grace CPU-RAM offload, slow |
| **Mistral-Large-3 (2512)** | 675 B / 41 B active MoE | 2025-12-02 | Apache 2.0 | **No** — needs multi-GPU; Grace-offload possible but 5–15 tok/s |
| **GPT-OSS-120B** | 117 B / 5.1 B active MoE | 2025-08-05 | Apache 2.0 | **Yes** — MXFP4 ≈ 60.8 GiB, fits comfortably |
| **DeepSeek-V3.1** | 671 B / 37 B active MoE | 2025 | DeepSeek License (commercial OK) | No — even Q4 ≈ 380 GB; Grace-offload only, very slow |
| **DeepSeek-V3-0324** | 671 B / 37 B active MoE | 2025-03 | DeepSeek License | Same as V3.1 |
| **Cohere Command-A (03-2025)** | 111 B dense | 2025-03 | CC-BY-NC 4.0 (research only) | Yes — FP8 ≈ 111 GB → Grace offload; INT4 ≈ 60 GB fits |
| **Cohere Command R+ (104B)** | 104 B dense | 2024-04 | CC-BY-NC 4.0 | Yes (same as Command-A roughly) |
| **Nemotron-Ultra-253B-v1** | 253 B dense (NAS-pruned) | 2025 | NVIDIA OMA + Llama | **No** — Q4 ≈ 150 GB; needs Grace offload or 2 GPUs |
| **Falcon-180B** | 180 B dense | 2023-09 | Falcon-180B TII License | Dominated; obsolete |

Notable phantoms (do not exist as released open-weight models in May 2026):
- **Qwen3-72B** — does NOT exist. Qwen3 dense ladder tops out at 32B; the only large Qwen3 is the 235B-A22B MoE.
- **Qwen3.6-72B** — does NOT exist. Largest Qwen3.6 dense is 27B; flagship is the closed Qwen3.6-Max API.
- **Hermes-4-Llama-3.3-70B** — Hermes 4 70B was built on **Llama-3.1-70B**, not 3.3.
- **Phi-4-70B** — does not exist. Phi-4 line tops at 14B dense + Phi-4-MoE (~42B).

---

## 2. Quality benchmark table

All values are reported by the model vendor unless flagged otherwise. Llama-3.3-70B numbers are the official Meta `eval_details.md` figures. *Italic* = reasoning mode (CoT/thinking enabled — comparisons to non-reasoning baselines are not strictly apples-to-apples).

| Model | MMLU-Pro | GPQA-Diamond | MATH-500 | AIME-24 | AIME-25 | LiveCodeBench | IFEval | Arena Hard |
|---|---|---|---|---|---|---|---|---|
| **Llama-3.3-70B-Instruct** (baseline) | **68.9** | **50.5** | **77.0** | — | — | — | **92.1** | ~72 |
| Llama-3.1-Nemotron-70B-Instruct | ~67 | ~48 | ~67 | — | — | — | 87 | **85.0** |
| Llama-3.3-Nemotron-Super-49B-v1.5 *(reasoning)* | *79.5* | *72.0* | *97.4* | *87.5* | *82.7* | *73.6* | — | — |
| DeepSeek-R1-Distill-Llama-70B *(reasoning)* | — (MMLU 90.8) | *71.5* | *94.5* | *70.0–86.7* | — | *57.5* | — | — |
| Hermes-4-70B *(reasoning)* | — (MMLU 88.4) | *66.1* | *95.5* | *73.5* | *67.5* | — | — | — |
| Tülu-3-70B (DPO) | ~55 | ~42 | 64 | — | — | — | 83 | — |
| Qwen2.5-72B-Instruct | 71.1 | 49.0 | 83.1 (MATH) | — | — | 55.5 (MultiPL-E) | 84.1 | **81.2** |
| Qwen3-32B *(thinking)* | *~70* | *~62* | *~94* | *81.4* | — | *~65* | — | — |
| Qwen3-30B-A3B *(thinking)* | *~63* | *~56* | *~87* | *70* | — | — | — | — |
| Qwen3-235B-A22B *(thinking)* | **82.8** | **70.0** | ~97 | ~85 | — | ~70 | — | **0.956 (#1)** |
| Qwen3.6-27B | **86.2** | (high) | (high) | — | — | (high) | — | — |
| Llama-4-Scout-17B-16E | 74.3 | 57.2 | — | — | — | — | — | — |
| Llama-4-Maverick-17B-128E | **80.5** | **69.8** | **95.0** | — | — | **43.4** | — | — |
| GPT-OSS-120B (high reasoning) | ~80 | ~74 | ~96 | ~88 | — | ~70 | — | — |
| Mistral-Large-3 (2512) | ~75 | ~62 | — | ~73 (14B variant 85% on AIME25) | — | — | — | — |
| Cohere Command-A (03-2025) | ~74 (MMLU ≈85, GPT-4o tier) | — | — | — | — | — | — | — |

Sources: Meta llama.com/models/llama-4, Qwen2.5 & Qwen3 technical reports (arXiv:2412.15115, 2505.09388), HF `meta-llama/Llama-3.3-70B-Instruct-evals`, NVIDIA build.nvidia.com Nemotron Super 49B v1.5 card, NousResearch Hermes-4 tech report (arXiv:2508.18255), DeepSeek-R1 paper (arXiv:2501.12948), OpenAI gpt-oss model card (2025-08-05), Mistral large-3 launch (mistral.ai/news/mistral-3), Artificial Analysis.

**Reading the table.** Llama-3.3-70B is a non-reasoning instruction model. Honest comparisons within that class are: Qwen2.5-72B (clearly above on MMLU-Pro / MATH / Arena), Llama-3.1-Nemotron-70B (better alignment, similar knowledge), Tülu-3-70B (slightly behind), Cohere Command-A (mixed, often ahead but restrictive license), Llama-4-Scout (above on STEM benchmarks but smaller active params), GPT-OSS-120B in `reasoning_effort=low` (roughly comparable). Reasoning models (Nemotron-Super-49B-v1.5, R1-Distill-70B, Hermes-4-70B with thinking, Qwen3-32B thinking) blow past Llama-3.3-70B on math/code/GPQA at the cost of 4–10× output tokens.

---

## 3. Memory math at Q4 (GGUF Q4_K_M) and AWQ / FP8

Rule of thumb for dense models: weights ≈ params × 0.62 bytes (Q4_K_M) ≈ params × 0.5 bytes (true 4-bit). FP8 ≈ params × 1.0 bytes. KV cache scales with `2 × n_layers × n_kv_heads × head_dim × ctx × bytes_per_elem`.

| Model | FP16 wts | FP8 wts | Q4_K_M GGUF | W4A16 (AWQ/GPTQ) | KV @ 32k ctx FP16 | Fits 96 GB GH200? |
|---|---|---|---|---|---|---|
| Llama-3.3-70B (80 L, 8 KV-head GQA) | 140 GB | 70 GB | **42.5 GB** | ~40 GB | 8.4 GB | Yes (Q4 + 50 GB ctx; FP8 + 20 GB ctx) |
| Llama-3.1-Nemotron-70B | 140 | 70 | 42.5 | 40 | 8.4 | Same as Llama-3.3 |
| Nemotron-Super-49B-v1.5 (NAS, variable FFN) | 98 | 49 | ~30 | ~28 | ~7 | Yes — large headroom |
| Qwen2.5-72B (80 L, 8 KV) | 145 | 72.5 | **43.5 GB** | ~41 | 8.4 | Yes |
| Qwen3-32B (64 L, 8 KV) | 64 | 32 | ~19 | ~18 | 6.7 | Easy |
| Qwen3-30B-A3B (MoE, 48 L) | 60 | 30 | ~17 | ~16 | 4 | Easy |
| Qwen3-235B-A22B (MoE) | 470 | 235 | ~130 | ~125 | ~12 | **No (HBM only)** — works via Grace LPDDR offload |
| Qwen3.6-27B | 54 | 27 | ~16 | ~15 | ~5 | Easy |
| Llama-4-Scout-109B (MoE, 17B active) | 218 | 109 | ~62 | ~55 (INT4 official) | ~9 | INT4 fits; FP8 needs Grace offload |
| Llama-4-Maverick-400B (MoE) | 800 | 400 | ~225 | ~210 | ~10 | **No** — Grace offload, slow |
| GPT-OSS-120B (MoE, MXFP4 native) | n/a | n/a | **60.8 GiB MXFP4** | n/a | ~7 | Yes — native MXFP4 designed for 80 GB GPUs |
| Mistral-Large-3 (675B MoE) | 1350 | 675 | ~380 | ~360 | — | **No** |
| DeepSeek-V3.1 (671B MoE) | 1342 | 671 | ~380 | ~360 | — | **No** |
| Cohere Command-A 111B | 222 | 111 | ~63 | ~60 | ~10 | INT4/Q4 fits; FP8 needs Grace offload |
| Nemotron-Ultra-253B (NAS) | 506 | 253 | ~150 | ~145 | — | **No** in HBM |
| Hermes-4-70B / R1-Distill-70B / Tülu-3-70B | 140 | 70 | 42.5 | 40 | 8.4 | Yes |

References for the headline numbers: Bartowski `Llama-3.3-70B-Instruct-GGUF` HF repo (Q4_K_M = 42.5 GB), Baseten GH200 blog (FP8 = 70 GB, 27 GB free for KV after weights), OpenAI gpt-oss model card (MXFP4 = 60.8 GiB), NVIDIA Llama-4 blog (Scout INT4 fits single H100).

KV-cache pressure matters. At 128k context, FP16 KV cache for Llama-3.3-70B is ~33 GB. That kills FP8-weights-on-96GB at long context — you need INT4 KV (vLLM `--kv-cache-dtype fp8`) or Q4 weights + FP16 KV.

---

## 4. Recommended engine + recipe + expected GH200 tok/s — top 3

GH200 Q4 reference for Llama-3.3-70B from prior rounds: ~30–45 tok/s single-stream Q4_K_M llama.cpp, ~50–80 tok/s FP8 vLLM batch=1, **scales to ~1000–1800 tok/s aggregate at batch=32 FP8 (vLLM)** — the Baseten/Lambda blog confirms GH200 is +32 % over H100 at the same workload.

### Top 1 — Llama-3.3-Nemotron-Super-49B-v1.5  (the headline upgrade)

**Why:** Same lineage as Llama-3.3-70B (post-trained from it), but NAS-pruned to 49 B with FFN-width search → ~30 % smaller HBM footprint, ~1.4–1.6× higher tok/s, and **substantially higher quality** on every reasoning benchmark (GPQA 72 vs 50, MATH 97 vs 77, AIME-24 87 vs n/a). License is NVIDIA OMA + Llama-3.3 Community — friendly. Single-H100/H200 was the explicit design target.

- Engine: **vLLM 0.7+** with `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8` (FP8 KV cache `--kv-cache-dtype fp8`).
- Recipe:
  ```
  vllm serve nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8 \
    --max-model-len 65536 --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 --port 8000
  ```
- Expected GH200 throughput: **single-stream ~110–150 tok/s; batch=32 aggregate ~2200–2800 tok/s** (extrapolating 1.4× from Llama-3.3-70B FP8 vLLM numbers given proportional memory bandwidth cut).
- Caveat: reasoning mode emits long `<think>…</think>` traces; expect 4–8× more tokens per turn. For chat-only use disable thinking (`detailed thinking off` system prompt).

### Top 2 — GPT-OSS-120B (MXFP4 native)

**Why:** Frontier-tier reasoning on MMLU-Pro/GPQA/MATH at only 5.1 B active params, native MXFP4 weights, Apache 2.0, fits comfortably (60.8 GiB) leaving ~30 GB HBM for KV at long context. Reported 325 tok/s output on commodity providers, **>500 tok/s achievable on H100/GH200 with vLLM** (Baseten result).

- Engine: **vLLM ≥ 0.6.5** with the official `vllm/vllm-openai` image. (TRT-LLM also has a recipe.)
- Recipe:
  ```
  vllm serve openai/gpt-oss-120b \
    --max-model-len 131072 --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.92 --port 8000 \
    --tool-call-parser openai --enable-auto-tool-choice
  ```
- Expected GH200 throughput: **single-stream ~250–450 tok/s; batch=32 aggregate ~5–8k tok/s** (MoE with 5.1B active is bandwidth-cheap once weights are resident; GH200's HBM3 hits ~4.0 TB/s).
- Caveat: tokenizer is OpenAI's o200k — different from Llama. Cannot drop-in replace if you have prompt templates tied to Llama.

### Top 3 — Qwen2.5-72B-Instruct (the like-for-like swap)

**Why:** The cleanest apples-to-apples upgrade. Non-reasoning instruction model, ~2 GB heavier than Llama-3.3-70B in Q4, but uniformly better on MMLU-Pro (71.1 vs 68.9), MATH (83 vs 77), HumanEval (86.6), Arena-Hard (81.2 vs ~72). License is Qwen Tongyi (commercial OK below 100 M MAU). Tokenizer is BPE 152k — different from Llama, no drop-in compatibility for raw token IDs but OpenAI-API endpoints are identical.

- Engine: **vLLM** with `Qwen/Qwen2.5-72B-Instruct` (FP16 needs >144 GB → use FP8 dynamic quant or AWQ).
- Recipe (FP8 dynamic):
  ```
  vllm serve Qwen/Qwen2.5-72B-Instruct \
    --quantization fp8 --max-model-len 32768 \
    --kv-cache-dtype fp8 --gpu-memory-utilization 0.92 --port 8000
  ```
- Or AWQ:
  ```
  vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
    --max-model-len 32768 --port 8000
  ```
- Expected GH200 throughput: **single-stream ~70–100 tok/s FP8, ~90–130 tok/s AWQ; batch=32 ~1.6–2.2k tok/s** (slightly slower than Llama-3.3-70B due to 72.7 B vs 70.6 B params and Qwen tokenizer dispatch overhead, but within 10 %).

---

## 5. Pareto front: what to swap, what to keep, what to avoid

The frontier is "quality vs single-stream tok/s on 1× GH200 Q4/FP8". Llama-3.3-70B Q4 ≈ (quality 68.9 MMLU-Pro, 45 tok/s). Numbers below are quality (MMLU-Pro or comparable) vs estimated single-stream tok/s on a 96 GB GH200 with the recommended quantization.

### Strictly dominates Llama-3.3-70B (better quality AND ≥ same speed)

1. **Llama-3.3-Nemotron-Super-49B-v1.5** — quality ↑↑ on every benchmark, ~1.4× faster (smaller). **The obvious upgrade if you accept reasoning mode / NVIDIA license.**
2. **GPT-OSS-120B** — quality ↑↑ (frontier-tier reasoning), tok/s ↑↑↑ (5.1 B active MoE on MXFP4). Drawback: different tokenizer; OpenAI o-style outputs.
3. **Qwen3-32B (thinking)** — Quality ≥ Llama-3.3 on most STEM benchmarks, ~1.5–2× faster (32 B dense), Apache 2.0. Worth it if you can tolerate Qwen tokenizer + CoT bloat.
4. **DeepSeek-R1-Distill-Llama-70B** — Same memory/speed as Llama-3.3-70B but quality ↑↑ on math/GPQA/MATH. Caveat: reasoning traces eat tokens.
5. **Qwen3.6-27B** — much smaller (~16 GB Q4 → ~3× faster) and MMLU-Pro 86.2 vs 68.9. The real wildcard. Apache 2.0, multimodal. Best raw-quality-per-watt option on this list.

### Better quality, slower (Pareto-acceptable for quality-sensitive workloads)

6. **Qwen2.5-72B-Instruct** — quality clearly above Llama-3.3-70B on knowledge/math/Arena; tok/s ~95 % of Llama-3.3 (slightly bigger). Like-for-like swap.
7. **Hermes-4-70B (reasoning)** — Same footprint, reasoning capability not present in Llama-3.3. Same speed minus token-bloat from thinking mode.
8. **Qwen3-235B-A22B (thinking)** — Best open-weight quality bar Mistral-Large-3. Tight fit on GH200 (130 GB Q4 needs Grace LPDDR offload, lands ~15–25 tok/s single-stream). Choose only if quality > latency.
9. **Llama-4-Scout INT4** — slightly above Llama-3.3 on MMLU-Pro and GPQA; tok/s should be 1.5–2× higher (17 B active MoE). Multimodal as a bonus.
10. **Llama-4-Maverick (FP8)** — Maverick's MMLU-Pro 80.5 / MATH 95 are excellent, but on a single GH200 you must offload to Grace LPDDR (≈250 GB needed); expect 10–25 tok/s single-stream. Worth it only if you must have Maverick quality on one GPU.

### Worse quality but much faster (Pareto-acceptable for latency-sensitive)

11. **Qwen3-30B-A3B (thinking)** — 3 B active MoE → expect 200–400 tok/s single-stream on GH200. Quality below Llama-3.3 on knowledge, but reasoning-strong. Great for high-throughput agent loops.
12. **Llama-3.1-8B / Qwen3-8B** — referenced only as the floor; out of scope.

### Dominated — don't use vs Llama-3.3-70B

13. **Llama-3.1-Nemotron-70B-Instruct** — Same footprint and tok/s as Llama-3.3-70B; Arena Hard wins (85 vs ~72) but MMLU-Pro/MATH/IFEval are equal or worse. Niche win for chat alignment only.
14. **Tülu-3-70B-DPO** — Same memory/speed, lower MMLU-Pro and MATH. Use only for fully-open-data compliance.
15. **Cohere Command R+ (104B)** — CC-BY-NC blocks commercial use; quality is at Llama-3.3 level on most non-RAG tasks; tok/s worse (bigger). Only attractive for its RAG/tool template heritage.
16. **Cohere Command-A (111B)** — Strong quality, but CC-BY-NC. INT4 fits but you lose ~30 % speed. Dominated by GPT-OSS-120B in the same memory budget and license-permissive.
17. **Falcon-180B** — Dominated on every axis. Quality < Llama-3.3-70B; 2.5× heavier. Skip.
18. **Mistral-Large-3 (675B MoE)** — Apache 2.0 and frontier-tier quality, but does not fit on one GH200 even at Q4. Grace LPDDR offload would give 5–12 tok/s. Need ≥2 GH200 or 1 GB200.
19. **DeepSeek-V3.1 / V3-0324 (671B MoE)** — Same story as Mistral-Large-3. Single-GH200 only via heavy LPDDR offload, single-digit tok/s. Not in Llama-3.3-peer category for this hardware.
20. **Nemotron-Ultra-253B** — Q4 ≈ 150 GB, doesn't fit HBM. Pareto-dominated by Nemotron-Super-49B-v1.5 on single GH200.

### Pareto frontier summary (single-GH200, 1-stream, Q4 / native quant)

```
quality
  ^
  | Qwen3-235B-A22B (thinking, with offload)         ← top-right but slow
  | Mistral-Large-3 (off-frontier on this HW)
  | Nemotron-Super-49B-v1.5 ★  (sweet spot)
  | GPT-OSS-120B  ★             (frontier high-speed)
  | Llama-4-Maverick (offload)
  | Llama-4-Scout INT4
  | Qwen2.5-72B
  | DeepSeek-R1-Distill-70B
  | Qwen3-32B (think)
  | Llama-3.3-70B (baseline)
  | Qwen3.6-27B  ★               (cheap + smart)
  | Qwen3-30B-A3B (think)        (cheap + fast)
  | Hermes-4-70B
  | Llama-3.1-Nemotron-70B
  | Tülu-3-70B
  | Falcon-180B          ← dominated
  +-------------------------------------------------> tok/s
```

**Three ★ recommendations for a Llama-3.3-70B replacement on 1× GH200:**

1. **Nemotron-Super-49B-v1.5** if you want "same family, better, faster" with no behavioural surprises.
2. **GPT-OSS-120B** if you want highest absolute throughput + frontier reasoning and don't mind a different tokenizer.
3. **Qwen3.6-27B** if your workload is small/medium reasoning and you can absorb a tokenizer swap — by far the best quality-per-tok/s on this hardware.

---

## Sources

- Meta Llama 3.3: <https://huggingface.co/datasets/meta-llama/Llama-3.3-70B-Instruct-evals>, eval_details on Meta GitHub.
- Llama 4 Scout/Maverick: <https://ai.meta.com/blog/llama-4-multimodal-intelligence/>, <https://www.llama.com/models/llama-4/>, <https://blog.vllm.ai/2025/04/05/llama4.html>, <https://developer.nvidia.com/blog/nvidia-accelerates-inference-on-meta-llama-4-scout-and-maverick/>, <https://blog.us.fixstars.com/running-llama-4-scout-on-a-single-nvidia-h100-using-int4-quantization/>.
- Qwen2.5 Tech Report: <https://arxiv.org/pdf/2412.15115>, <https://qwenlm.github.io/blog/qwen2.5-llm/>.
- Qwen3 Tech Report: <https://arxiv.org/pdf/2505.09388>.
- Qwen3.6-27B: <https://qwen.ai/blog?id=qwen3.6-27b>.
- DeepSeek-V3 / V3-0324 / V3.1: <https://huggingface.co/deepseek-ai/DeepSeek-V3-0324>, <https://api-docs.deepseek.com/updates>.
- DeepSeek-R1 paper: <https://arxiv.org/html/2501.12948v1>.
- Nemotron-Super-49B-v1.5: <https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5>, <https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5>, <https://developer.nvidia.com/blog/build-more-accurate-and-efficient-ai-agents-with-the-new-nvidia-llama-nemotron-super-v1-5/>.
- Nemotron-Ultra-253B: <https://aws.amazon.com/marketplace/pp/prodview-itx64tcpfgukg>.
- Llama-3.1-Nemotron-70B: <https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF>.
- Hermes-4-70B: <https://huggingface.co/NousResearch/Hermes-4-70B>, arXiv:2508.18255.
- Tülu-3: <https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B>, arXiv:2411.15124.
- GPT-OSS-120B: <https://openai.com/index/introducing-gpt-oss/>, <https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf>, <https://huggingface.co/openai/gpt-oss-120b>, <https://blog.vllm.ai/2025/08/05/gpt-oss.html>, <https://www.baseten.co/blog/sota-performance-for-gpt-oss-120b-on-nvidia-gpus/>.
- Mistral-Large-3 (2512): <https://mistral.ai/news/mistral-3>, <https://artificialanalysis.ai/models/mistral-large-3>.
- Cohere Command-A / Command R+: arXiv:2504.00698, <https://docs.cohere.com/docs/command-r-plus>, <https://huggingface.co/CohereLabs/c4ai-command-r-plus>.
- GH200 inference data: <https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/>, <https://lambda.ai/blog/partner-spotlight-testing-llama-3.3-70b-inference-performance-on-nvidia-gh200-with-baseten>, <https://docs.vllm.ai/projects/recipes/en/latest/Llama/Llama3.3-70B.html>, <https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/>.
- Quantization sizes: <https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF>.
- Artificial Analysis listings: <https://artificialanalysis.ai/models/llama-3-3-instruct-70b>, <https://artificialanalysis.ai/models/qwen2-5-72b-instruct>, <https://artificialanalysis.ai/models/gpt-oss-120b>, <https://artificialanalysis.ai/models/hermes-4-llama-3-1-70b>.
