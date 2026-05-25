# Mid-Size MoE Alternatives for GH200

Round 5 / Doc 14 — investigating Mixture-of-Experts replacements for
Llama-3.3-70B (dense) and Qwen3.6-27B (dense) on a single GH200 (96 GB HBM3
+ 480 GB Grace LPDDR5X, NVLink-C2C 900 GB/s).

Date: 2026-05-25

---

## TL;DR

The biggest GH200-specific insight from Round 1 was that the
`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` + `--n-cpu-moe` patch in llama.cpp
makes huge MoE models (DeepSeek V3.1 671B Q4_K_M, ~400 GB) decode at
**18.3 t/s** because *only the active expert subset moves over NVLink-C2C
per token*. That same trick generalizes to every model on this list.

For the **Llama-3.3-70B replacement** slot, the two serious candidates are:

| Model                                | Total / Active | Q4 size | MMLU (5-shot) | Fits HBM? | GH200 single-stream est. |
| ------------------------------------ | -------------: | ------: | ------------: | --------: | -----------------------: |
| **Llama-3.3-70B (baseline, dense)**  |        70 / 70 |  ~40 GB |          86.0 |       yes |          ~28–35 t/s (vLLM bf16 baseline from prior rounds) |
| **Mixtral 8x22B**                    |       141 / 39 |  ~80 GB |          77.3 |   tight   |                ~55–70 t/s |
| **Qwen3-Next-80B-A3B**               |        80 / 3  |  ~45 GB |    ~88 (MMLU-Redux) / 80 (MMLU-Pro) |       yes |    150–200 t/s (artificialanalysis: 168.7) |
| **GPT-OSS-120B (MXFP4)**             |       117 / 5.1 |  ~60 GB |          90.0 |       yes |   **209 t/s** (measured GH200, llama.cpp #18005) |

For the **Qwen3.6-27B replacement** slot, the strongest candidates are:

| Model                                | Total / Active | Q4 size | MMLU | Fits HBM? | GH200 single-stream est. |
| ------------------------------------ | -------------: | ------: | ---: | --------: | -----------------------: |
| **Qwen3.6-27B (baseline, dense)**    |        27 / 27 |  ~16 GB | ~82 / 81.7 MMLU-Pro |  yes  | ~60–80 t/s estimate |
| **Mixtral 8x7B**                     |       47 / 13  |  ~26 GB |          70.6 |       yes |                  ~120 t/s |
| **Qwen3-30B-A3B**                    |        30 / 3.3 |  ~17 GB |          83  |       yes |    **120–200 t/s** (RTX 4090 already 196 t/s) |
| **GPT-OSS-20B (MXFP4)**              |       21 / 3.6 |  ~12 GB |          85.3 |       yes |   **323 t/s** (measured GH200) |
| **Nemotron-3-Nano-30B-A3B**          |        30 / 3.5 |  ~17 GB |    78.3 MMLU-Pro |  yes  | ~3.3x Qwen3-30B → est. **400+ t/s** |
| **Phi-3.5-MoE**                      |        42 / 6.6 |  ~24 GB |          78.9 |       yes |                 ~80–120 t/s |
| **DeepSeek-V2-Lite**                 |        16 / 2.4 |  ~10 GB |          58.3 |       yes |              ~200–250 t/s |
| **OLMoE-7B-A1B**                     |         7 / 1  |   ~4 GB |          54  |       yes |              ~300–400 t/s |

**Bottom line**: **GPT-OSS-120B at MXFP4 is the strongest Llama-3.3-70B
replacement** on GH200 — it has measured **209 t/s decode** (5–7x faster
than dense 70B) and matches or beats Llama 3.3 on MMLU. **GPT-OSS-20B at
323 t/s** or **Nemotron-3-Nano-30B-A3B** (NVIDIA-optimized, 3.3x Qwen3-30B
throughput on H200) are the strongest 27B-class replacements.

---

## 1. Mixtral 8x22B (141B total / 39B active)

- **License**: Apache 2.0 (mistralai/Mixtral-8x22B-v0.1 and -Instruct-v0.1).
- **Release**: 2024-04-17.
- **Architecture**: 8 experts of ~22B params each, top-2 gating → 39B active.
- **Context**: 64K.

### Quality

- MMLU 5-shot: **77.3**.
- Below Llama-3.3-70B (86.0 MMLU) and below Qwen3-Next-80B-A3B-Thinking
  (92.5 MMLU-Redux).
- Strong at multilingual (FR/DE/ES/IT) and code; outperforms Llama-2-70B by
  large margins. Against Llama-3.3, **it loses on raw quality** — Mistral
  shipped Mixtral 8x22B before Llama-3 took the dense crown.

### Memory math at Q4 (Q4_K_M, ~4.8 bpw effective)

- BF16 size: **263 GB**.
- Q4_K_M GGUF: **~80 GB** (community quants typically 78–83 GB).
- **Fits in GH200 96 GB HBM3 but tight** (need to leave ≥10 GB for KV cache,
  activations, CUDA workspace at 8K ctx). Practical Q4_K_S (~73 GB) is the
  comfortable choice; Q4_K_M needs `--n-cpu-moe` to spill 1–2 expert layers
  to Grace LPDDR.

### Expected GH200 throughput

- Active path is 39B params at Q4 ≈ 22 GB/token of memory traffic.
- HBM3 effective bandwidth ~3 TB/s on GH200 → **~135 t/s ceiling** (memory-
  bound).
- With overhead, realistic single-stream **~55–70 t/s**. Batch=16: ~400–700
  t/s aggregate (vLLM/SGLang grouped-GEMM helps expert utilization).
- **No measured GH200 number was found in llama.cpp #18005** — the discussion
  benchmarks GPT-OSS and DeepSeek but not Mixtral. The estimate is from the
  active-param scaling rule (39B/Q4 ≈ 22 GB · 3 TB/s = 137 t/s upper bound,
  derated 50–60% for KV/attention/router overhead).

### Verdict

- **Skip** for the Llama-3.3-70B slot. Same memory footprint as a dense 70B
  at Q4, much weaker quality (77.3 vs 86.0 MMLU). The only reason to choose
  Mixtral 8x22B today is the Apache 2.0 license vs Llama's custom community
  license — but Qwen3-Next-80B-A3B-Instruct is Apache 2.0 too and is
  qualitatively superior.

---

## 2. Mixtral 8x7B (47B / 13B active)

- **License**: Apache 2.0.
- **Release**: 2023-12-11.
- **Context**: 32K.

### Quality

- MMLU 5-shot: **70.6** — matches Llama-2-70B (69.9) and GPT-3.5 (70.0).
- Far below Qwen3.6-27B (~82 MMLU / 81.7 MMLU-Pro) — 2.5 years of progress
  between them.

### Memory math at Q4

- BF16: ~88 GB. Q4_K_M GGUF: **~26 GB**.
- Fits comfortably in HBM3 with ~70 GB headroom for KV cache.

### Expected GH200 throughput

- Active path is 13B at Q4 ≈ 7.5 GB/token of expert traffic + 4 GB shared
  attention/embedding. ~12 GB/token total memory traffic → **~250 t/s
  ceiling** on HBM3.
- Realistic **~120 t/s single-stream**, batch=16 **~800–1200 t/s**.

### Verdict

- **Skip** for the Qwen3.6-27B slot. The quality gap is too large — a model
  that scores 70.6 MMLU cannot replace a 27B that scores ~82. Mixtral 8x7B
  is fast but its 2023-era quality is dated for 2026 production work.

---

## 3. Qwen3-Next-80B-A3B (80B / 3B active)

- **License**: Apache 2.0 (Qwen/Qwen3-Next-80B-A3B-Instruct, -Thinking).
- **Release**: Sept 2025 (-Instruct/-Thinking variants); -Coder-Next followup.
- **Architecture**: 3B active, hybrid attention + linear-attention layers,
  on par with Qwen3-235B-A22B-Instruct-2507 at 10x throughput for 32K+ ctx.
- **Context**: 65K (262K native, extendable).

### Quality

- MMLU-Redux: **92.5** (Thinking variant).
- MMLU-Pro: comparable to or better than Llama-3.3-70B (68.9) — Qwen3-Next
  beats it on GPQA and MMLU-Pro per the llm-stats head-to-head.
- AAI Intelligence Index: 27 (well above average for size).

### Memory math at Q4

- BF16: ~160 GB. Q4_K_M GGUF: ~45 GB; Q4_K_S ~42 GB.
- **Fits in HBM3 with ~50 GB headroom** for KV cache and activations.
  Unsloth's documentation reports ~46 GB RAM/UM needs at Q4_K_M.

### Expected GH200 throughput

- Active path is **3B at Q4 ≈ 1.7 GB/token** — extremely lightweight.
- Artificialanalysis API measurement (third-party hosted): **168.7 t/s** for
  the Thinking variant.
- On GH200 single-stream, expect **150–200 t/s** decode. Batch=16: **1500–
  2500 t/s** aggregate (small active path + grouped GEMM is the perfect
  pattern for GH200's SM count).
- Hybrid attention helps long-context dramatically; this is the model that
  will absolutely dominate at 32K+ context lengths.

### Verdict

- **Top pick for Llama-3.3-70B replacement.** Higher quality, Apache 2.0,
  ~5x faster decode, fits in HBM with room for KV cache. Only caveats:
  vLLM support for Qwen3-Next's hybrid attention was still maturing as of
  late 2025 — verify the kernel path on GH200 before committing.

---

## 4. Qwen3-30B-A3B (30.5B / 3.3B active)

- **License**: Apache 2.0.
- **Release**: 2025-04-29 (predecessor of Qwen3-Next).
- **Context**: 32K-128K depending on variant.

### Quality

- MMLU (Q6_K quant): **83%**.
- Roughly matches Qwen3.6-27B on knowledge benchmarks but with **10x
  faster decode**.
- Speculative decoding does NOT help (PR #19493 benchmarks: no net speedup
  on Ampere + A3B MoE — active path is already tiny so draft-then-verify
  doesn't pay off).

### Memory math at Q4

- BF16: ~61 GB. Q4_K_M GGUF: **~17 GB**.
- Fits trivially. Plenty of room for very long context KV cache.

### Expected GH200 throughput

- RTX 4090 (1 TB/s mem bw): **196 t/s**. RTX 3090: 136 t/s.
- GH200 (3 TB/s HBM): linearly scaled would be ~580 t/s; realistic
  single-stream **150–300 t/s** depending on the inference framework.
  At batch=16: 2000+ t/s aggregate.

### Verdict

- **Strong candidate for Qwen3.6-27B replacement.** Same vendor, same
  knowledge level, ~3–5x decode speedup. Apache 2.0. The natural drop-in.

---

## 5. DeepSeek-V2.5 (236B / 21B active) and DeepSeek-V2-Lite (16B / 2.4B active)

### DeepSeek-V2.5

- **License**: DeepSeek Model License (commercial use permitted).
- **Release**: 2024-09-05 (DeepSeek-V2.5), 2024-12 (V2.5-1210 update).
- **Architecture**: Multi-head Latent Attention (MLA) + DeepSeekMoE.
- **MMLU**: V2 base reports 78.1 / 79.2 (DeepSeek-V2 + V2-Chat / V2-Coder);
  V2.5 likely 80–81 (consolidation of V2-Chat + V2-Coder-V2). The publicly
  posted DeepSeek-V3 hits 88.5 by contrast.

#### Memory at Q4

- BF16: ~470 GB. Q4_K_M GGUF: ~135 GB → **does NOT fit in HBM**.
- **Requires the Round-1 unified-memory + n-cpu-moe trick.** Decode speed
  will be MoE-active-bound: 21B active at Q4 ≈ 12 GB/token. Memory traffic
  per token over Grace LPDDR (~400 GB/s effective per the verda data) →
  **~30–40 t/s** estimated decode.
- This places V2.5 between Mixtral 8x22B HBM-resident (~60 t/s) and
  DeepSeek V3.1 671B unified-memory (18.3 t/s measured).

#### Verdict
- **Niche**. Quality is below Llama-3.3-70B, requires unified-memory mode
  to run, and DeepSeek-V3 / V3.1-Terminus completely supersede it.
  Skip in favor of V3.1 (if 18.3 t/s is acceptable) or Qwen3-Next (if not).

### DeepSeek-V2-Lite

- **License**: DeepSeek Model License (commercial use permitted).
- **MMLU**: **58.3** — below Mistral-7B-Instruct-v0.2.
- **Memory at Q4**: ~10 GB. Trivially fits.
- **GH200 throughput**: 2.4B active at Q4 → ~1.4 GB/token. Estimated **200–
  250 t/s** decode single-stream.

#### Verdict
- Interesting research/proving-grounds model (it pioneered MLA at scale)
  but **MMLU 58 is unusable as a Qwen3.6-27B replacement** in 2026.

---

## 6. OLMoE-7B-A1B (7B / 1B active)

- **License**: Apache 2.0 (fully open: weights, data, training code).
- **MMLU**: ~54 (OLMoE-1B-7B-Instruct outperforms Llama-2-13B-Chat).
- **Memory at Q4**: ~4 GB.
- **GH200 throughput**: 1B active at Q4 ≈ 0.6 GB/token → **300–500 t/s**
  feasible.

#### Verdict
- **Research model**, not a production substitute. The quality gap to
  Qwen3.6-27B is enormous. The only reason to deploy OLMoE is if you need
  fully reproducible open-data training provenance.

---

## 7. Phi-3.5-MoE (42B / 6.6B active)

- **License**: MIT.
- **Release**: 2024-08.
- **Architecture**: 16 experts of 3.8B each, top-2 routing → 6.6B active.
- **Context**: 128K.

### Quality

- MMLU 5-shot: **78.9**. Beats GPT-4o-mini on MMLU 5-shot.
- Multilingual MMLU: 69.9 average.
- Strong reasoning/math relative to active param count.

### Memory at Q4

- BF16: ~84 GB. Q4_K_M: **~24 GB**. Fits comfortably.

### Expected GH200 throughput

- 6.6B active at Q4 ≈ 4 GB/token + 2 GB shared → **~500 t/s ceiling**.
- Realistic single-stream **80–120 t/s** (Phi-3.5-MoE has had less inference-
  engine tuning than Qwen/Mixtral; the routing pattern doesn't match
  TensorRT-LLM's grouped-GEMM kernel ideally).

### Verdict

- **Quality-competitive with Qwen3.6-27B (78.9 vs ~82)** at smaller active
  size. MIT license is friendlier than Apache 2.0 for some use cases. Worth
  a head-to-head benchmark if you care about coding/math specifically — Phi
  models punch above their MMLU score in those domains.
- **Caveat**: Microsoft has slowed Phi-MoE work since Phi-4 (dense)
  shipped; community kernel support lags Qwen.

---

## 8. GPT-OSS-20B and GPT-OSS-120B (OpenAI, MXFP4-native)

- **License**: Apache 2.0.
- **Release**: 2024-08 (OpenAI's first open-weight release).
- **Quantization**: Trained native MXFP4 (4-bit microscaling) on MoE weights.

### GPT-OSS-20B (21B total / 3.6B active)

- MMLU: **85.3** (some sources report 69; the 85 figure is the standard
  5-shot Instruct number).
- Layers: 24. Active params per token: 3.6B.
- **Native MXFP4 size: ~12 GB.** Fits trivially in HBM.

#### GH200 measurement (llama.cpp #18005)
- **PP2048: 9,682 t/s. TG32: 322.93 t/s.** This is real, run on actual GH200.
- This is the **fastest decode rate of any benchmarked model on GH200 in
  the Round-1 dataset.**

### GPT-OSS-120B (117B total / 5.1B active)

- MMLU: **90.0** (5-shot Instruct).
- Layers: 36. Active params per token: 5.1B.
- **Native MXFP4 size: ~60 GB.** Fits in 96 GB HBM3 with 30+ GB for KV cache.

#### GH200 measurement (llama.cpp #18005)
- **PP2048: 5,393 t/s. TG32: 209.01 t/s.** Real measurement.
- vLLM on H100 (single GPU): **1008 t/s offline-batch mode** per simplismart.
- vLLM on H100 reported 6095 t/s on Clarifai vs TRT-LLM 2139 t/s vs SGLang
  1814 t/s — vLLM has the most mature gpt-oss MXFP4 kernel path.

### Verdict (BOTH)

- **GPT-OSS-120B is the strongest measured Llama-3.3-70B replacement.**
  MMLU 90 vs 86, decode **209 t/s vs 28–35 t/s** (estimated dense 70B
  baseline at bf16) = **~6–7x speedup with higher quality**. Apache 2.0.
  Fits in HBM.
- **GPT-OSS-20B is the strongest measured Qwen3.6-27B replacement.**
  MMLU 85.3 vs ~82, decode **323 t/s vs ~60–80 t/s** (estimated dense
  27B baseline) = **~4–5x speedup with comparable quality**.
- The MXFP4 native format means no calibration-quality tradeoff that
  affects post-hoc Q4 quants. This is the cleanest production story.

---

## 9. Snowflake Arctic (480B / 17B active)

- **License**: Apache 2.0.
- **Release**: 2024-04. Confirmed still available on HuggingFace and
  Snowflake Cortex as of 2026.
- **Architecture**: 128 fine-grained experts + dense FFN base. Top-2 gating.
- **MMLU**: **67.3** (Arctic Instruct) vs DBRX Instruct 73.3.

### Memory at Q4

- BF16: ~960 GB. Q4_K_M: ~270 GB. **Does NOT fit in HBM.** Requires Grace
  unified-memory + n-cpu-moe (Round 1 trick).
- Active path 17B at Q4 ≈ 10 GB/token. Realistic decode ~25–35 t/s through
  Grace LPDDR (similar regime to DeepSeek-V2.5).

### Verdict

- **Skip.** Quality (67.3 MMLU) is below Mixtral 8x22B (77.3), below
  Llama-3.3-70B (86), below GPT-OSS-120B (90). Arctic was a 2024
  efficiency-research model; in 2026 it has no compelling use case unless
  you have specific SQL/enterprise prompts where its training data shines.

---

## 10. NVIDIA Nemotron-3-Nano-30B-A3B (30B / 3.5B active)

- **License**: NVIDIA Open Model License (commercial use OK; verify terms).
- **Release**: Late 2025.
- **Architecture**: **Hybrid Mamba-2 + Transformer + MoE** — 23 Mamba/MoE
  layers + 6 attention layers. 128 routed experts + 1 shared, top-5 gating.
  3.5B active params.
- **Context**: native 1M tokens.

### Quality

- MMLU-Pro: **78.3** (slightly below Qwen3-30B's 80.9 on the same bench).
- MATH: 82.88% (beats Qwen3-30B's 61.14 substantially).
- HumanEval: 78.05.
- RULER @ 64K: 87.5 (long-context strength).

### Memory at Q4

- BF16: ~60 GB. FP8: ~30 GB. Q4: ~17 GB. Fits.

### GH200/H200 throughput

- NVIDIA's own measurement on H200 (8K input / 16K output): **3.3x Qwen3-
  30B-A3B and 2.2x GPT-OSS-20B**.
- If GPT-OSS-20B is 323 t/s on GH200, Nemotron 3 Nano extrapolates to
  **~700 t/s decode** in the same llama.cpp regime. On vLLM/TRT-LLM with
  NVIDIA's native optimizations: expect even higher.
- This is the most aggressive inference-throughput model in the comparison.

### Verdict

- **Top pick for Qwen3.6-27B replacement** alongside GPT-OSS-20B.
- The hybrid Mamba+Transformer architecture is a structural advantage for
  long-context (1M ctx native). For agentic/long-context workloads this is
  the strongest open model in its weight class as of mid-2026.
- **Caveat**: licensing is NVIDIA Open Model License — slightly more
  restrictive than Apache 2.0 for some commercial uses. Read terms carefully.

---

## The Active-Parameter Speedup, Made Concrete

The whole MoE thesis on GH200:

```
decode tokens/sec  ≈  (peak memory bandwidth) / (active params × bytes/param + KV/activation overhead)
```

At Q4 (0.5 bytes/param) on GH200 HBM3 (3 TB/s effective):

| Active params | Memory traffic / token | Theoretical decode | Realistic decode |
| ------------: | --------------------: | -----------------: | ---------------: |
|         1B    |               0.5 GB  |        ~6000 t/s   |    300–500 t/s   |
|         3B    |               1.5 GB  |        ~2000 t/s   |    150–300 t/s   |
|         5B    |               2.5 GB  |        ~1200 t/s   |    150–250 t/s   |
|         7B    |               3.5 GB  |         ~850 t/s   |    100–150 t/s   |
|        13B    |               6.5 GB  |         ~460 t/s   |     80–120 t/s   |
|        22B    |              11   GB  |         ~270 t/s   |     50–80  t/s   |
|        39B    |              19.5 GB |         ~150 t/s   |     30–55  t/s   |
|        70B (dense) |          35   GB  |          ~85 t/s   |     20–35  t/s   |

Realistic = ~25–30% of theoretical (attention isn't 100% bandwidth-bound;
KV cache reads, expert routing overhead, kernel launch overhead all add).
GH200 numbers measured in Round 1 (GPT-OSS-20B 323 t/s, GPT-OSS-120B 209
t/s, DeepSeek V3.1 18.3 t/s via UM) all fit this model with active params
of 3.6B, 5.1B, and 37B-effective-via-LPDDR respectively.

**The Grace LPDDR trick (DeepSeek V3.1: 18.3 t/s with 671B model)
demonstrates that NVLink-C2C at 900 GB/s gives ~300–360 GB/s effective for
the active-expert path** — enough to keep MoE decode useful even when the
inactive experts live in Grace LPDDR. This is what makes GH200 uniquely
attractive for huge MoE.

---

## Ranked Recommendations

### Replace Llama-3.3-70B with:

1. **GPT-OSS-120B (MXFP4)** — measured 209 t/s, MMLU 90, fits HBM, Apache 2.0.
   The clear winner if its post-training style suits your prompts.
2. **Qwen3-Next-80B-A3B-Instruct** — est. 150–200 t/s, MMLU-Pro ≥ Llama-3.3,
   Apache 2.0, hybrid attention dominates at 32K+ ctx. Choose if you need
   Qwen's instruction-following style or long-context.
3. **DeepSeek V3.1-Terminus** (671B/37B active via UM) — measured 18.3 t/s,
   MMLU 88+. Choose only when you need V3.1-class reasoning specifically and
   can accept the slower decode.

Skip: Mixtral 8x22B (lower quality at same memory), Arctic (lower quality
at far higher memory), DeepSeek-V2.5 (superseded by V3.1).

### Replace Qwen3.6-27B with:

1. **GPT-OSS-20B (MXFP4)** — measured 323 t/s on GH200, MMLU 85.3, Apache 2.0.
2. **Nemotron-3-Nano-30B-A3B** — est. 400–700 t/s on GH200 via NVIDIA's
   3.3x-Qwen3-30B figure; strong code/math; 1M native context. Verify
   license terms.
3. **Qwen3-30B-A3B** — est. 150–300 t/s, MMLU 83, Apache 2.0, drop-in
   compatible with existing Qwen tokenizers/templates.

Skip: Mixtral 8x7B (MMLU 70 is dated), DeepSeek-V2-Lite (MMLU 58),
OLMoE (MMLU 54), Phi-3.5-MoE (only marginal speed advantage over Qwen3-30B
and weaker community kernel support).

---

## Open Questions (worth empirically verifying on actual GH200)

1. **vLLM Qwen3-Next hybrid-attention kernel path on GH200.** Was reported
   not-yet-mature as of late 2025. If broken, fall back to llama.cpp + UM.
2. **TensorRT-LLM MXFP4 path for GPT-OSS-120B on GH200 specifically.**
   Clarifai's numbers showed vLLM dominates TRT-LLM for gpt-oss on H100;
   verify TRT-LLM hasn't caught up before committing the stack.
3. **Nemotron-3-Nano-30B-A3B kernel maturity in llama.cpp.** Hybrid
   Mamba+Transformer+MoE is a new architecture; native CUDA Graph paths
   exist in NeMo but llama.cpp Mamba support is GGUF-experimental.
4. **DeepSeek V3.1-Terminus vs GPT-OSS-120B on actual prompt set.** 18.3 t/s
   vs 209 t/s is a 11x decode gap; only worth taking the V3.1 hit if your
   prompts demand its specific reasoning capabilities.
5. **Batch=16 numbers for all models.** Single-stream measurements above
   come from llama.cpp llama-bench; production batched throughput on
   vLLM/SGLang likely puts each model in the 1500–7000 t/s aggregate range
   depending on active-param count.

---

## Sources

- llama.cpp GH200 benchmark thread: https://github.com/ggml-org/llama.cpp/discussions/18005
- Mixtral 8x22B: https://mistral.ai/news/mixtral-8x22b ; https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1
- Mixtral 8x7B: https://mistral.ai/news/mixtral-of-experts
- Qwen3-Next-80B-A3B: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct ; https://artificialanalysis.ai/models/qwen3-next-80b-a3b-reasoning
- Qwen3-30B-A3B: https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507 ; https://artificialanalysis.ai/models/qwen3-30b-a3b-instruct
- DeepSeek-V2 / V2-Lite: https://github.com/deepseek-ai/DeepSeek-V2 ; https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite ; arxiv 2405.04434
- DeepSeek V2.5 changelog: https://api-docs.deepseek.com/updates
- GPT-OSS: https://huggingface.co/openai/gpt-oss-120b ; https://openai.com/index/introducing-gpt-oss/ ; https://docs.gpustack.ai/latest/performance-lab/gpt-oss-120b/h100/
- GPT-OSS benchmark spread: https://www.clarifai.com/blog/comparing-sglang-vllm-and-tensorrt-llm-with-gpt-oss-120b
- Phi-3.5-MoE: https://huggingface.co/microsoft/Phi-3.5-MoE-instruct
- OLMoE: https://arxiv.org/abs/2409.02060
- Snowflake Arctic: https://www.snowflake.com/en/blog/arctic-open-efficient-foundation-language-models-snowflake/
- Nemotron 3 Nano: https://research.nvidia.com/labs/nemotron/Nemotron-3/ ; https://llm-stats.com/models/nemotron-3-nano-30b-a3b ; https://medium.com/@leucopsis/a-technical-review-of-nvidias-nemotron-3-nano-30b-a3b-e91673f22df4
- Llama 3.3 70B: https://llm-stats.com/models/llama-3.3-70b-instruct
- Qwen3.6-27B: https://qwen.ai/blog?id=qwen3.6-27b ; https://localaimaster.com/models/qwen-3-6-27b
- Mixtral 8x22B Q4 sizing: https://huggingface.co/bartowski/Mixtral-8x22B-v0.1-GGUF ; https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1/discussions/2
- Qwen3-Next GGUF: https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF ; https://github.com/ggml-org/llama.cpp/discussions/22064
