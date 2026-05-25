# Apple Silicon for LLM/Diffusion Inference vs GH200

**Research agent #13 of 20 | Domain: Apple Silicon (Mac Studio, Mac Pro, MLX)**
**Date: 2026-05-25**

---

## TL;DR

- **M3 Ultra Mac Studio** is the only single-box workstation that fits DeepSeek-V3 671B (4-bit, ~405 GB) entirely in unified memory. At "ideal" small-context use it reaches ~20 tok/s generation; under realistic 8 K context the prompt prefill is the killer (9 tok/s prefill = ~14 minutes before first token on 8 K input).
- **M4 Ultra does not exist.** Apple skipped it. The next Ultra is **M5 Ultra**, expected in the 2026 Mac Studio refresh, but Apple has reportedly pushed shipment to October 2026 due to DRAM shortages.
- **CRITICAL purchase blocker (as of May 2026):** Apple **pulled the 512 GB option** in March 2026 due to global DRAM shortage. As of this writing the Mac Studio M3 Ultra caps at 256 GB (some reporting suggests configurations are now restricted further — 96 GB max with multi-month lead times). Anyone budgeting on the $9.5 k 512 GB unit may not be able to actually buy one.
- **Sweet spot:** 70 B–235 B-A22B class models at 4-bit MLX, single-user / few-user workloads, ~10–50 tok/s generation. Not a serving box.
- **Per-token cost** for solo use beats cloud GH200 on power alone (Awni Hannun's M2 Ultra calc: $0.20 / Mtok at 60 W); economics break down once you need batching, multi-user, or long-context prefill at scale.

---

## 1. Hardware reality check (May 2026)

| SKU | Status (May 2026) | Memory ceiling | Bandwidth | List price |
|---|---|---|---|---|
| Mac Studio M3 Ultra 512 GB | **Pulled March 2026** (DRAM shortage) | (was 512 GB) | 819 GB/s | (was $9,499) |
| Mac Studio M3 Ultra 256 GB | Available, 4–5 mo lead time | 256 GB | 819 GB/s | $5,599 + $2,000 upgrade ≈ $7,599 |
| Mac Studio M3 Ultra 96 GB | Available now | 96 GB | 819 GB/s | $3,999 |
| Mac Studio M4 Max | Available | 128 GB | 546 GB/s | $1,999+ |
| **M4 Ultra** | **Does not exist — Apple skipped** | — | — | — |
| Mac Pro M2 Ultra | Available (older) | 192 GB | 800 GB/s | $6,999+ |
| Mac Studio M5 Ultra | Expected Oct 2026 (delayed from mid-2026) | TBD (rumor: up to 512 GB if DRAM frees up) | Rumored ~ proportional to M5's +28 % over M4 | TBD |

Sources: Tom's Hardware (Mar 2026), Cult of Mac (Mar 2026), 9to5Mac (Apr 2026), Macworld 2026 roadmap. The 512 GB SKU was reviewed and benchmarked widely in 2025; in May 2026 it's effectively unobtainable new. **Used / refurb is the only path to 512 GB right now.**

**M3 Ultra silicon details:** 32-core CPU, 80-core GPU, 32-core Neural Engine, 819 GB/s unified memory bandwidth (often rounded to "800 GB/s" in press). Thunderbolt 5. 3 nm.

---

## 2. Throughput table — Apple Silicon LLM benchmarks

All numbers M3 Ultra 512 GB Mac Studio (where the 512 GB model was tested) running MLX-LM unless noted. "Prompt" = prefill tok/s, "Gen" = decode tok/s.

| Model | Quant | Framework | Context | Prompt tok/s | Gen tok/s | Source |
|---|---|---|---|---|---|---|
| Llama 3.3 70B | 4-bit | MLX-LM | small | ~150 (M3 Max extrap.) | ~15–20 | Awni Hannun / Hardware Corner |
| Llama 3.3 70B | 4-bit | MLX-LM | (M3 Max 64 GB) | — | ~10 | Awni Hannun (X, Dec 2024) |
| Llama 3.3 70B | 4-bit | MLX-LM | (M4 Max 128 GB) | — | ~12 | Ivan Fioravanti (X) |
| Llama 4 Scout (109B MoE) | 4-bit | MLX | 30 tok | 103 | 44 | Hardware Corner |
| Llama 4 Scout (109B MoE) | 4-bit | MLX | 10 k | 82 | 22 | Hardware Corner |
| Llama 4 Maverick (400B MoE) | 4-bit | MLX | 30 tok | 140 | 50 | Hardware Corner |
| Llama 4 Maverick (400B MoE) | 4-bit | MLX | 10 k | 117 | 25 | Hardware Corner |
| Qwen3-235B-A22B | 4-bit | MLX | small | — | **24** (124 GB RAM) | MacStories |
| Qwen3-235B-A22B | 4-bit | GGUF/llama.cpp | small | — | 16 (133 GB RAM) | MacStories |
| Qwen2.5-VL-72B | 4-bit | MLX | image+text | — | (~3.5 min/image) | MacStories |
| Qwen3-30B-A3B (on M4 Max) | 4-bit | MLX-LM | small | 1,754 | 113 | Awni Hannun gist |
| Qwen3-30B-A3B (on M4 Max) | 8-bit | MLX-LM | small | 1,719 | 83 | Awni Hannun gist |
| DeepSeek-V3 671B (0324) | 4-bit | MLX-LM | small | — | **>20** (peak 466 GB used at 16 k ctx) | Awni / Slashdot Mar 2025 |
| DeepSeek-V3 671B | 4-bit (q4_K_M) | llama.cpp | 8 k | **9.0** (= 888 s = 14.8 min prefill) | 6.2 | Hardware Corner |
| Gemma 3 27B | 4-bit | MLX | small | — | 41 | Alex Ziskind / Newport |

**Cross-model patterns:**
- **4-bit is the standard.** 6-bit is "near lossless" sweet spot per MLX community. 8-bit gives modest accuracy gain at ~30 % speed cost.
- **Generation scales with bandwidth (819 GB/s).** Prefill scales with FP16/INT8 compute (M3 Ultra ~26 TFLOPS fp16, vs GH200 ~990 TFLOPS fp16 / ~2 PFLOPS fp8).
- **MoE models punch up:** Maverick (400B) faster than Scout (109B) because of activation pattern.
- **MLX beats llama.cpp** by ~50 % on generation in current testing (Qwen3-235B: 24 vs 16 tok/s). Reported 4–5× advantage on DeepSeek prefill.

---

## 3. The prefill problem (load-bearing for any "serve users" plan)

The H100/GH200 advantage is not generation speed — it's prefill. Specifically:

- **GH200**: ~3 TB/s HBM3 bandwidth, ~2 PFLOPS FP8 / ~990 TFLOPS BF16 — prefill bottleneck is compute, and there's a lot of it.
- **M3 Ultra**: 819 GB/s, ~26 TFLOPS FP16. **Memory-refresh-rate analysis: H100 = 37.5 / s, M3 Ultra = 1.56 / s** (24× lower; matters for single-request batch=1).

Concrete: DeepSeek-V3 671B with 8 k prompt on M3 Ultra (llama.cpp): **14.8 minutes** before the first output token. With MLX it's better but still minutes-scale. On a GH200, same 8 k prefill = seconds.

**EXO Labs' workaround**: cluster M3 Ultra (fast generation) + DGX Spark (fast prefill) for ~2.8× speedup. With Llama 3.1 8B @ 8 k context: Spark prefill 1.47 s, M3 Ultra generation 0.85 s, combined 2.32 s total. Demonstrates that the Mac alone is structurally wrong for prefill-heavy workloads.

---

## 4. Cost comparison: M3 Ultra vs cloud GH200

### Cloud GH200 pricing (May 2026, surveyed)
| Provider | $/hr | Notes |
|---|---|---|
| Spheron | $1.88 | Dedicated, 99.99 % SLA |
| Lambda | $1.99 | 480 GB LPDDR5X + 96 GB HBM3 |
| Range | $1.50–$6.50 | Per ComputePrices / getdeploying |

Monthly 24/7: **$1,080–$4,680 per GH200**.

### Capex: Mac Studio M3 Ultra 512 GB (when available)
$9,499 list. Power ~200 W under load (TechRadar measured DeepSeek 671B inference at <200 W). Awni's M2 Ultra calc: 60 W × $0.18/kWh ⇒ ~$0.20 / Mtok at 15 tok/s. M3 Ultra scales similarly.

### Break-even
At Spheron's $1.88 / hr GH200 = $13,800 / yr 24/7. A $9.5 k Mac Studio "pays for itself" in **~7 months of 24/7 GH200 cloud cost** — *if* your workload tolerates M3 Ultra's prefill ceiling and single-stream nature.

### Per-token economics (single-user)
| System | Tok/s gen | Power | $/Mtok (electricity only) |
|---|---|---|---|
| M3 Ultra @ Llama 70B 4-bit | 15 | ~60–100 W | ~$0.20–$0.33 |
| Cloud GH200 @ Llama 70B FP8 | ~80 (with batching, hundreds) | (billed by hour) | At $1.88/hr ÷ 80 tok/s = $6.5/Mtok solo, ~$0.10–$1 with batching |

**Verdict:** For a single user generating tokens 8 hr / day, M3 Ultra is dramatically cheaper than cloud GH200. For serving 10+ concurrent users with long prefill, GH200 is cheaper per token because batching amortizes compute across requests.

---

## 5. Use-case fit matrix

| Use case | M3 Ultra rating | Why |
|---|---|---|
| Single dev / single user, 70B–235B, short-ish context | **Best in class** | $/perf, silent, no cloud egress, models fit |
| Run DeepSeek-V3 671B locally for tinkering | **Only single-box option** | 512 GB unified memory; nothing else does this for <$10 k |
| Long-context (>16 k) prefill-heavy work | **Bad** | Prefill is the bottleneck; minutes-scale TTFT |
| Serving 5+ concurrent users | **Bad** | vllm-mlx exists but throughput ceiling is far below GH200 batched |
| Diffusion (FLUX/SDXL) | **OK** | FLUX Schnell 30–50 s/image; ~10× slower than H100 |
| Fine-tuning 70B+ | **Marginal** | MLX LoRA works; full-precision FT is impractical |
| 24/7 always-on inference low-volume | **Great** | <200 W; tokens-per-joule champion |

---

## 6. Software stack snapshot

- **MLX / MLX-LM** (Apple's framework): Best generation throughput on M-series. Maturing prefill optimizations. 4/6/8-bit native quant.
- **llama.cpp Metal**: Broader model support, GGUF ecosystem. ~50 % slower generation than MLX on M3 Ultra currently; prefill 4–5× slower on DeepSeek.
- **Ollama**: Wraps llama.cpp; convenient but not optimal.
- **vllm-mlx** (community port, 2025): Adds continuous batching, paged KV-cache, prefix caching to MLX. Claims 21–87 % improvement over llama.cpp and 4.3× throughput scaling at 16 concurrent requests. Anthropic Messages API + OpenAI-compatible endpoints. **This is the closest thing to vLLM on Apple Silicon.**
- **EXO Labs**: Cluster N Macs via Thunderbolt 5 / RDMA. Day-0 RDMA-over-TB. Up to 3.2× speedup on 4 devices via tensor parallel.

---

## 7. MLX quantization gotchas (analog to the FP8/GH200 issue)

User note flags FP8 broken on this stack — same warning applies in reverse on Apple:

- **No native FP8 on M-series.** Quant options are integer (q4, q5, q6, q8) and bf16. No FP4/FP8 hardware acceleration. If your workflow requires FP8 weights, you can't run them on Apple Silicon without dequant.
- **MLX nvfp4 issue (open as of Mar 2026):** MLX uses signed E4M3 scales instead of NVIDIA's unsigned UE4M3, giving 137× less dynamic range. Saturates / clips on large activations. Issue #2962 in ml-explore/mlx. Don't use MLX nvfp4 for accuracy-critical work.
- **Uniform bit-width** in stock MLX quant — every layer gets same width. Attention heads need more bits than MLPs; uniform 4-bit hits attention softmax precision. Mitigation: 6-bit ("nice trade-off, minimal accuracy loss" per ml-explore/mlx-lm discussion #616).
- **3-bit:** Big quality drop-off from 4-bit. Avoid.
- **Long-context KV cache:** Major memory consumer at large contexts; PolarQuant / TurboQuant work is ongoing (issue #1060) but not yet stable.

---

## 8. Apple newsroom / authoritative specs

- **M3 Ultra** (Apple, Mar 5 2025): "up to 512 GB unified memory, 819 GB/s memory bandwidth, run LLMs with over 600 billion parameters directly on device." [apple.com/newsroom](https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/)
- That 600 B+ claim has been demonstrated in the wild by Awni Hannun running DeepSeek-V3 (671 B) in 4-bit at >20 tok/s.

---

## 9. Sources (with dates)

1. [Apple newsroom — M3 Ultra reveal (Mar 5 2025)](https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/)
2. [Apple newsroom — Mac Studio (Mar 2025)](https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/)
3. [Tom's Hardware — Apple pulls 512 GB option (Mar 2026)](https://www.tomshardware.com/tech-industry/apple-pulls-512-mac-studio-upgrade-option)
4. [9to5Mac — Mac Studio 4–5 month lead time (Apr 3 2026)](https://9to5mac.com/2026/04/03/mac-studio-delivery-4-5-months-out-for-top-ram-after-apple-dropped-512gb-option/)
5. [Macworld — M5 Mac Studio delayed to Oct 2026](https://www.macworld.com/article/2973459/2026-mac-studio-m5-release-date-specs-price-rumors.html)
6. [MacRumors — 512 GB disappears (Mar 5 2026)](https://www.macrumors.com/2026/03/05/mac-studio-no-512gb-ram-upgrade/)
7. [Hardware Corner — DeepSeek 671B llama.cpp on M3 Ultra (2025)](https://www.hardware-corner.net/mac-studio-m3-ultra-deepseek-llamacpp/)
8. [Hardware Corner — Llama 4 Scout/Maverick on M3 Ultra](https://www.hardware-corner.net/llama-4-on-mac-m3-ultra-speed/)
9. [Slashdot — DeepSeek-V3 at 20 tok/s on Mac Studio (Mar 25 2025)](https://apple.slashdot.org/story/25/03/25/2054214/deepseek-v3-now-runs-at-20-tokens-per-second-on-mac-studio)
10. [MacStories — Qwen3-235B-A22B / Qwen2.5-VL-72B on M3 Ultra](https://www.macstories.net/notes/notes-on-early-mac-studio-ai-benchmarks-with-qwen3-235b-a22b-and-qwen2-5-vl-72b/)
11. [Awni Hannun — MLX-LM benchmark gist (M4 Max Qwen3)](https://gist.github.com/awni/c1790e4c3a39be6e8f1c4afd42423d2d)
12. [Awni Hannun X — Llama 3 70B 4-bit cost/M2 Ultra (May 2024)](https://x.com/awnihannun/status/1786069640948719956)
13. [Awni Hannun X — Llama 3.3 70B on M3 Max 64 GB (Dec 2024)](https://x.com/awnihannun/status/1865187697146700273)
14. [Ivan Fioravanti X — Llama 3.3 70B on M4 Max](https://x.com/ivanfioravanti/status/1865237429780721853)
15. [EXO Labs — DGX Spark + M3 Ultra cluster (Oct 2025)](https://blog.exolabs.net/nvidia-dgx-spark/)
16. [Simon Willison — DGX Spark + Mac Studio coverage (Oct 16 2025)](https://simonwillison.net/2025/Oct/16/nvidia-dgx-spark-apple-mac-studio/)
17. [vllm-mlx GitHub — continuous batching on MLX](https://github.com/waybarrios/vllm-mlx)
18. [Spheron — GH200 pricing $1.88/hr](https://www.spheron.network/gpu-rental/gh200/)
19. [getdeploying — GH200 cloud pricing comparison 2026](https://getdeploying.com/gpus/nvidia-gh200)
20. [Lambda Labs pricing](https://lambda.ai/pricing)
21. [Production-Grade Local LLM Inference on Apple Silicon — comparative study arxiv 2511.05502](https://arxiv.org/pdf/2511.05502)
22. [ml-explore/mlx issue #2962 — nvfp4 problems](https://github.com/ml-explore/mlx/issues/2962)
23. [ml-explore/mlx-lm discussion #616 — 6-bit vs bf16 quality](https://github.com/ml-explore/mlx-lm/discussions/616)
24. [Billy Newport — M3 Ultra misses the mark for LLM inference (Medium)](https://medium.com/@billynewport/apples-m3-ultra-mac-studio-misses-the-mark-for-llm-inference-f57f1f10a56f)
25. [TechRadar — M3 Ultra runs DeepSeek R1 <200 W](https://www.techradar.com/pro/apple-mac-studio-m3-ultra-workstation-can-run-deepseek-r1-671b-ai-model-entirely-in-memory-using-less-than-200w-reviewer-finds)
26. [xenix.blog — Mac Studio 2025 vs NVIDIA Blackwell (May 2025)](https://xenix.blog/2025/05/05/mac-studio-m4-max-m3-ultra-vs-nvidia-blackwell-which-desktop-reigns-for-local-genai/)

---

## Confidence & Gaps

**High confidence:**
- M4 Ultra does not exist. Apple skipped it. M3 Ultra is the current Ultra (819 GB/s, 80-core GPU, up to 512 GB nominal).
- DeepSeek-V3 671B fits and runs at ~20 tok/s on 512 GB M3 Ultra in MLX 4-bit — multiple primary sources (Awni, EXO, Slashdot, TechRadar).
- 512 GB SKU pulled March 2026 — confirmed by Tom's Hardware, MacRumors, 9to5Mac, Cult of Mac.
- Prompt-prefill is the structural weakness vs GH200 — corroborated by EXO Labs cluster benchmarks (3.8× DGX Spark advantage in prefill).
- GH200 cloud pricing ~$1.88–$6.50/hr range — multiple aggregators agree.

**Medium confidence:**
- Specific Llama-3.3-70B MLX tok/s on M3 Ultra 512 GB — no clean primary benchmark surfaced. Numbers in table are extrapolated from M3 Max (10 tok/s) and M4 Max (12 tok/s) with Ultra's ~2× memory bandwidth advantage suggesting ~15–25 tok/s; Awni's M2 Ultra 4-bit 70B was 15 tok/s, M3 Ultra should be modestly faster.
- Qwen2.5-72B MLX specifically — community reports it works but no consolidated tok/s benchmark; Qwen3-235B-A22B at 24 tok/s gives the upper-bound feel.
- vllm-mlx batched throughput claims (4.3× at 16 concurrent) — single-source, lab numbers, not independently verified.

**Low confidence / gaps:**
- **Real-world availability of M3 Ultra 512 GB in May 2026.** Conflicting reports — one search said current configs cap at 96 GB; others suggest 256 GB still orderable with 4–5 month lead. Worth a direct Apple Store check before any procurement plan.
- M5 Ultra specs and timing — purely rumor / leak based; mid-2026 → Oct 2026 slip is the most-cited current expectation.
- Apple Silicon diffusion model throughput (FLUX/SDXL) — only rough numbers found (30–50 s per image, FLUX Schnell). Insufficient data to draw a strong $/image comparison vs GH200.
- No data on **prompt cache reuse** behavior for MLX (would substantially change the prefill-cost analysis for chat / multi-turn workloads).
- Power draw under sustained MLX inference — claims of <200 W exist but no rigorous wall-plug measurement at peak.

**FP8 / quant note (per user memory):** User's GH200 memory says "FP8 broken — always bf16." Apple Silicon has the inverse problem: **no FP8/FP4 hardware support at all**. MLX's experimental nvfp4 is itself broken (issue #2962). Practical Apple quant menu = q4 / q6 / q8 / bf16. Plan accordingly — anything model-shipped as FP8-native needs dequantization to run on M-series.
