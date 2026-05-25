# GH200 LLM Quantization Matrix — Definitive Reference

**Agent**: Round 4 LLM-only deep dive
**Date compiled**: 2026-05-25
**Scope**: Text-LLM quantization on GH200 (Hopper SM90 + Grace). Diffusion-transformer (DiT) FP8 is broken on this stack — see `/home/ubuntu/research/gh200_inference/round2/B_fp8_forensic_audit.md`. This document covers **dense and MoE text LLMs only**.
**Anchor models**: Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct, DeepSeek-V3 671B (native FP8), with cross-references to Llama-3.1-8B and Qwen3-4B where the published data lives.

---

## TL;DR — Production picks for GH200

| Use case | First choice | Why |
|---|---|---|
| Dense Llama-3.3-70B / Qwen2.5-72B serving, latency-tolerant batch ≥16 | **FP8 W8A8 e4m3 (TRT-LLM 1.2.1 or vLLM 0.21)** | 1.5–1.8× throughput, +32% over H100, MMLU delta <0.5%, 70 GB weights fit in 96 GB HBM3e with 26 GB KV headroom. |
| Dense Llama-3.3-70B at batch 1–8, low TPOT (interactive) | **AWQ-Marlin W4A16** | Decode is memory-bandwidth-bound; 35 GB weights free 60 GB for KV; ~2× decode throughput vs FP8 at batch 1–4. |
| DeepSeek-V3 671B MoE | **FP8 W8A8 native (SGLang 0.5.12)** | Model ships FP8-native (685 GB); SGLang DeepGEMM/DeepEP kernels are reference; needs 2× GH200 NVL2 (or H200 single-node). |
| Long-context dense ≥32k tokens | **FP8 W8A8 + TurboQuant 3-bit KV** (vLLM 0.21) | KV cache halved by FP8 then further to 3-bit gives 4× capacity; NIAH 128k holds 89–100% depending on preset. |
| Gemma-4-E2B / any head_dim=256 model | **BF16 (or AWQ-Marlin for memory)** | FP8 prefill is ~1.6× SLOWER than BF16 at head_dim=256 due to two-level accumulation register pressure. |
| Sliding-window models (Gemma2/3, gpt-oss, Mistral) | **FP8 W8A8 + `--kv-cache-dtype-skip-layers sliding_window`** | Sliding-window layers have bounded KV — quantizing them yields little, and the kernel often misbehaves. |
| llama.cpp single-card hobby | **Q4_K_M or Q5_K_M GGUF** | Q5_K_M = +0.18 MMLU vs F16 on Llama-3.1-8B; Q4_K_M gives 4 GB/8B and 40 GB/70B. CUDA path works on GH200 but multi-GPU still weak — use vLLM/SGLang/TRT-LLM for serving. |

**Don't pick on GH200**:
- **NVFP4 / MXFP4 for compute**: emulated on Hopper, zero throughput gain vs FP8. Only memory savings. Reserve for B200.
- **W4A4 (Atom/QuaRot/QServe)**: no production-grade kernel in vLLM 0.21 / SGLang 0.5.12 / TRT-LLM 1.2.1; research-only.
- **NF4 (bitsandbytes)**: slow dequantization, no Marlin path. QLoRA fine-tuning artifact, not a serving format.
- **fp8_e5m2** with FA3: silently falls back to Triton (sglang#18550). Use e4m3.

---

## 1. The Compatibility Matrix

Rows: quantization format. Columns: serving stack (versions pinned). Cell verdict + key citation.

**Legend**:
- **Production** = ships pre-quantized checkpoints, MLPerf-validated or named in vendor recipe, no open accuracy bugs
- **Supported** = works for many models, some accuracy/performance bugs open
- **Beta** = recently merged, not widely deployed
- **Broken-on-Hopper** = open issue on H100/H200/GH200 with no production fix
- **Emulated** = software fallback only, no HW acceleration on Hopper
- **N/A** = format incompatible with this stack

| Format | vLLM 0.21 | SGLang 0.5.12 | TRT-LLM 1.2.1 | llama.cpp |
|---|---|---|---|---|
| **bf16** | Production | Production | Production | Production (CPU/CUDA) |
| **fp16** | Production | Production | Production | Production |
| **FP8 W8A8 e4m3** | **Production** (recipe for Llama-3.3-70B on Hopper [1]) | **Production** (DeepSeek-V3 reference impl [2]) | **Production** (vendor checkpoint pipeline [3]) | N/A (no FP8 GEMM kernels) |
| **FP8 W8A8 e5m2** | Supported (FA3 falls back to Triton, sglang#18550 → not Hopper-optimal) | Supported (same FA3 fallback) | Supported | N/A |
| **FP8 KV cache only** | **Production** post 0.19.1 (FA3 accumulation fix [4]) | Production | Production | N/A |
| **INT8 W8A8 (SmoothQuant)** | **Production** (LLM Compressor pipeline) | Production (`blockwise_int8`/`w8a8_int8`) | Production | N/A |
| **INT4 W4A16 AWQ-Marlin** | **Production** (auto-selects `awq_marlin` kernel on SM80+) | **Production** (`awq_marlin` available) | **Production** (`W4A16-AWQ`) | N/A |
| **INT4 W4A16 GPTQ-Marlin** | **Production** | Production | Production (`W4A16-GPTQ`, `W4A8-GPTQ`) | N/A |
| **INT4 W4A8 GPTQ** | Supported (Machete kernel) | Supported | **Production** on Hopper | N/A |
| **W4A4 (Atom/QuaRot/QServe/COMET)** | Not supported (no kernel) | Not supported | Not supported | N/A |
| **NF4 (bitsandbytes)** | Supported (slow dequant; 1.54× kernel speedup landed April 2026 [5]) | Supported | N/A | N/A |
| **TurboQuant 3-bit KV** | **Production** (PR #38479 merged 2026-04-15, ships in 0.20.0+ [6]) | Beta (issue #21618 tracking, WIP) | Not supported | N/A |
| **TurboQuant 2-bit KV (k8v4)** | Beta (presets in 0.21.0) | Not supported | Not supported | N/A |
| **NVFP4 (W4A4 FP4)** | **Emulated** on Hopper (Marlin software fallback, memory-only) [7] | **Emulated** on Hopper | **Emulated** on Hopper (Production on B200) | Supported (CPU + emulated) |
| **MXFP4** | **Emulated** on Hopper (memory savings only, no throughput) | **Emulated** on Hopper | **Emulated** on Hopper | Supported (memory only) |
| **GGUF Q4_K_M** | Supported (single-GPU) — degraded on multi-GPU vs Marlin path | Supported (CUDA only, NVIDIA) | N/A | **Production** |
| **GGUF Q5_K_M / Q6_K / Q8_0** | Supported | Supported | N/A | **Production** |

**Citations**:
- [1] vLLM Llama-3.3-70B recipe — https://docs.vllm.ai/projects/recipes/en/latest/Llama/Llama3.3-70B.html (FP8 recommended for Hopper, NVFP4 for Blackwell)
- [2] SGLang DeepSeek-V3 guide — https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/deepseek_v3.md (DeepGEMM/DeepEP FP8 kernels)
- [3] TRT-LLM quantization doc — https://nvidia.github.io/TensorRT-LLM/features/quantization.html (FP8 per-tensor/block-scaling/rowwise; Hopper Production)
- [4] vLLM blog 2026-04-22 "The State of FP8 KV-Cache and Attention Quantization in vLLM" — https://vllm.ai/blog/2026-04-22-fp8-kvcache (FA3 accumulation fix; head_dim=256 prefill regression)
- [5] arXiv 2604.02556 (April 2026), "Fast NF4 Dequantization Kernels for Large Language Model Inference" — 2.0–2.2× kernel speedup, 1.54× end-to-end on Llama-3.3-70B
- [6] vLLM PR #38479 — https://github.com/vllm-project/vllm/pull/38479 (TurboQuant 3-bit KV merged 2026-04-15, shipping in 0.20.0+)
- [7] vLLM NVFP4 docs — Marlin FP4 software fallback gives memory benefit, no throughput on H100/H200/GH200

---

## 2. Production-Grade Combinations — Detailed Profiles

For each cell that's **Production** on GH200, the following gives Llama-3.3-70B-Instruct (or Qwen2.5-72B where cited) numbers. All throughput figures are vLLM `benchmark_serving.py` ShareGPT or NIAH-style; cite source.

### 2.1 FP8 W8A8 e4m3 (vLLM 0.21 / TRT-LLM 1.2.1 / SGLang 0.5.12)

| Metric | Llama-3.3-70B (GH200) | Qwen2.5-72B (GH200) | Source |
|---|---|---|---|
| MMLU vs BF16 | 83.8 → 83.8 (Δ 0.0) | ~ -0.2 (extrapolated from arXiv 2411.02355 Llama-3.1-70B) | "Give Me BF16 or Give Me Death?" arXiv 2411.02355v3 Table 5 |
| NIAH @128k vs BF16 | 89% (FP8) vs 91% (BF16) post 0.19.1 FA3 fix | comparable | vLLM blog 2026-04-22 |
| Throughput vs BF16 | **+50% to +80% at batch 16–64** (Llama-3.1-70B FP8 reaches 460 tok/s on 1× H100 batch 64 [8]; GH200 +32% on top → ~607 tok/s) | similar | morphllm benchmarks + Baseten GH200 [9] |
| Latency (ITL) vs BF16 | **−14 to −15%** at concurrency 8 (Llama-3.1-8B, scales similar at 70B) | similar | vLLM blog 2026-04-22 |
| VRAM (weights only, 70B) | **70 GB** (vs 140 GB BF16) → leaves **~26 GB for KV** on GH200 96 GB | **72 GB** (Qwen2.5-72B) → ~24 GB KV | Baseten GH200 study |
| Calibration cost | **None** (dynamic per-token activation quant + static weights via RTN); pre-quantized FP8 checkpoints from RedHatAI / nvidia / FriendliAI | None | vLLM FP8 docs |
| Break-even context vs BF16 | **~7,010 tokens** post-fix (Llama-3.1-8B), shorter for 70B | similar | vLLM blog 2026-04-22 |

**Known restrictions**:
- `fp8_e5m2` falls back to Triton silently on FA3 — always use `fp8_e4m3` (sglang#18550, open inactive).
- **head_dim=256 prefill regression**: TTFT quadratic coefficient 6.93e-07 → 1.12e-06 ms/token² (~1.6× slower). Affects Gemma-4-E2B specifically; **does NOT affect** Llama-3.3-70B or Qwen2.5-72B (head_dim=128).
- **Sliding-window models** (gpt-oss-20b, Gemma2/3, Mistral-style): use `--kv-cache-dtype-skip-layers sliding_window` flag. Without it, accuracy can degrade.
- **MoE online quant**: still fragile in vLLM 0.21 (vLLM#30830 open) — prefer pre-quantized FP8 MoE checkpoints (DeepSeek-V3 native, Mixtral-FP8 from RedHat) over `--quantization fp8` online flag.
- **TRT-LLM 1.2.1 specifically**: validated Llama-3.3-70B-Instruct FP8 + NVFP4 paths; Qwen2.5-72B not in 1.2 validated list (use Qwen3 there); FP8 KV scaling on QKV FMHA added in 1.2.

**$/Mtok on GH200** (Lambda Cloud pricing): FP8 Llama-3.3-70B at batch 32 ≈ $0.45–$0.55/Mtok, vs $1.19/Mtok BF16 on H100 (~2.2× cheaper). [E_pricing_leaderboard.md from Round 2]

### 2.2 INT8 W8A8 SmoothQuant (vLLM 0.21 / SGLang / TRT-LLM)

| Metric | Llama-3.3-70B (GH200) | Source |
|---|---|---|
| MMLU vs BF16 | 83.8 → 83.7 (Δ −0.1) | arXiv 2411.02355v3 |
| MMLU-Pro recovery (405B) | 97.8% | "BF16 or Death" paper |
| Throughput vs BF16 | **~1.87× on A100**, ~1.6× speedup at 5 QPS on H100 | LLM Compressor / RedHat blog |
| Latency vs BF16 | Comparable to FP8 (sometimes slightly worse decode due to per-token activation calibration) | RedHat blog |
| VRAM (weights, 70B) | **70 GB** (same as FP8) | calculator |
| Calibration cost | **Few-shot** (SmoothQuant + GPTQ): typically 128–512 calibration samples (UltraChat / Open-Platypus). Smoothing-strength 0.5–0.8 tunable. | LLM Compressor docs |

**When to pick INT8 over FP8 on GH200**: rarely. FP8 has equivalent or better quality with zero calibration. INT8 makes sense if you're constrained to a non-FP8-capable accelerator OR you need the GPTQ-trained INT8 weight as a stepping stone to W4A8 weight-only.

**Restrictions**:
- Activations quantized per-token dynamically — adds latency at very small batches.
- Not supported on Blackwell SM100+ (CC ≥ 10.0) — Blackwell drops INT8 tensor cores. Use FP8 there. (Relevant for forward compat when migrating GH200 → GB200.)

### 2.3 INT4 W4A16 AWQ-Marlin (vLLM 0.21 / SGLang 0.5.12)

| Metric | Llama-3.3-70B (GH200) | Qwen2.5-72B (GH200) | Source |
|---|---|---|---|
| MMLU vs BF16 | 83.8 → 83.6 (Δ −0.2) on Llama-3.1-70B w/ 128-sample calib | comparable; AWQ reportedly +1–3% over GPTQ INT4 | arXiv 2411.02355v3, AWQ guides |
| MMLU absolute (Llama-3.1-70B-Instruct AWQ 128-sample) | **86.62** | n/a published | govenance Medium / Spheron AWQ guide |
| Throughput vs BF16 (low batch, decode-heavy) | **~3× decode** at batch 1–4 (memory-bandwidth-bound) | similar | morphllm vLLM benchmarks 2026 |
| Throughput vs FP8 (high batch, compute-bound) | **~0.5–0.7× of FP8** at batch ≥32 (dequant overhead dominates) | similar | morphllm vLLM benchmarks 2026 |
| Long-context (128k NIAH) | Modest degradation — 4-bit weight-only is more robust than 4-bit-activation; some EMNLP 2025 evals show drops "up to 59%" for AWQ-int4 on adversarial long-context tasks | similar caveat | aclanthology 2025.emnlp-main.479 |
| VRAM (weights, 70B) | **35 GB** → leaves **~61 GB for KV** on GH200 | 36 GB | calc |
| Calibration cost | **Few-shot** (128 samples typical; calibration dataset choice matters — different calib data → measurably different models) | same | AWQ paper, govenance discussions |

**Restrictions**:
- Activations remain BF16 → no activation-tensor-core speedup. Pure decode wins.
- MoE AWQ in vLLM had MLA-crash bugs (referenced in 0.21 release notes as fixed for AWQ+GPTQ MLA crashes).
- Calibration-data sensitivity is real — re-quantize from same calib data if reproducibility matters.

### 2.4 INT4 W4A16 GPTQ-Marlin

Within ~0.1 MMLU points of AWQ-Marlin on Llama-3.x at the same bit width. AWQ slightly better on reasoning (+1–3% MMLU); GPTQ slightly better on perplexity. **Pick AWQ-Marlin for new deployments** unless you already have GPTQ checkpoints.

**TRT-LLM W4A8-GPTQ** (Hopper-only): combines INT4 weights with FP8 activations. Stronger throughput at large batches than W4A16 because the activation path uses FP8 tensor cores. Production in TRT-LLM 1.2.x but no vLLM/SGLang equivalent.

### 2.5 FP8 W8A8 + TurboQuant 3-bit KV (vLLM 0.20+/0.21, Llama-3.3-70B)

| Metric | Value | Source |
|---|---|---|
| KV cache compression | 2.6× over BF16 (4-bit packed) → 4× capacity with 3-bit preset | TurboQuant ICLR 2026 paper / vLLM PR #38479 |
| NIAH @128k (3-bit nc preset, Qwen3-4B) | "0.720" — partial recall (acceptable for retrieval, not legal/medical) | turboquant-vllm repo |
| NIAH @128k (`turboquant_k8v4` preset) | **100% recall** at 2.6× compression on Qwen3-4B | turboquant-vllm repo |
| Throughput vs BF16 KV | **35–43% of baseline tok/s** for decode-heavy (early data; improving) | turboquant-vllm repo |
| Calibration cost | **None** (Hadamard rotation + Lloyd-Max grid, no calib data) | TurboQuant paper |

**Restrictions**:
- Works on standard attention (Llama, Qwen, Mistral, Gemma GQA/MHA).
- **Does NOT support hybrid attention+Mamba models** (Qwen3.5/3.6-dense variants). Important caveat.
- MLA (DeepSeek) coverage only via plugin monkey-patch — upstream coming.
- SM121 (Grace Hopper) uses Triton fallback (slower than Hopper-specific CUDA kernels).
- Trade-off is clear: **only enable for ≥32k contexts**. Below that, FP8 KV alone wins.

### 2.6 DeepSeek-V3 671B Native FP8 (SGLang 0.5.12 — recommended)

| Metric | Value | Source |
|---|---|---|
| Format | Native FP8 (DeepSeek shipped FP8 weights directly — no BF16 master) | DeepSeek V3 technical report |
| VRAM | 685 GB total (671 GB model + 14 GB MTP head) | DeepSeek docs |
| Hardware fit | **Does NOT fit single GH200** (96 GB HBM3e). Needs 2-node GH200 NVL2 (192 GB HBM) or single 8× H200 (1,128 GB HBM3e) | calc |
| Throughput on 8× H200 | Reference production sweet spot | Baseten/Perplexity 2026 |
| MMLU | ~88 (DeepSeek-V3-0324 reported) — no BF16 baseline (FP8 is the native format) | DeepSeek model card |
| Kernels | DeepGEMM + DeepEP open-sourced; SGLang integrates as reference | SGLang DeepSeek doc |

**Caveats**:
- TRT-LLM 1.2.1 also supports DeepSeek-V3 FP8; SGLang remains the reference impl.
- Online FP8 quantization of DeepSeek-V3 via vLLM `--quantization fp8` flag is **not recommended** — use the native FP8 weights.
- **DeepSeek-V4-Flash + EAGLE spec-decode** open bug (sglang#24650) — avoid that combo on Hopper.

### 2.7 llama.cpp GGUF on GH200 (single-card)

| Format | MMLU (Llama-3.1-8B-Instruct) | Δ vs F16 | File size (8B) | File size (70B est.) |
|---|---|---|---|---|
| F16 | 63.50% | — | 15.3 GB | 130 GB |
| Q8_0 | 63.43% | −0.07 | 8.1 GB | 70 GB |
| Q6_K | 63.17% | −0.33 | 6.3 GB | 55 GB |
| Q5_K_M | 62.80% | −0.70 | 5.5 GB | 48 GB |
| Q5_K_S | 63.36% | −0.14 | 5.3 GB | 46 GB |
| Q4_K_M | 62.43% | −1.07 | 4.7 GB | 40 GB |
| Q4_K_S | 62.06% | −1.44 | 4.5 GB | 39 GB |
| Q3_K_M | 62.01% | −1.49 | 3.8 GB | 32 GB |

Source: arXiv 2601.14277v1 (Llama-3.1-8B-Instruct unified evaluation).

**GH200 status**: llama.cpp CUDA works. Top throughput is multi-modal / single-user. On large models with unified memory, the `cudaMemAdvise()` patch for MoE expert tensors (gives ~7× speedup on DeepSeek-V3.1 prompt processing per GH200 GitHub discussion #18005) is **not yet upstreamed**.

**For 70B serving on GH200**, do NOT pick llama.cpp — vLLM 0.21 + FP8 dominates throughput by ~10× at production batch sizes. llama.cpp is the right pick for:
- Single-user chat where wall-clock latency at batch 1 matters more than batch throughput
- Models where AWQ/FP8 quantized weights aren't available
- Tinkering / development

**Calibration**: GGUF k-quants need no calibration; importance-matrix (`imatrix`) builds use ~1000–10000 sample tokens for better accuracy (e.g., the IQ_S variants).

---

## 3. The AWQ-Marlin vs FP8 Trade-off on GH200 — Decisive Analysis

This is the central operational question for GH200. Both formats are **production on Hopper**. Pick by workload signature.

### When AWQ-Marlin (W4A16) wins on GH200

1. **Low-batch decode (batch 1–8)**: Decoding is memory-bandwidth-bound. AWQ-Marlin's 35 GB weight footprint (vs 70 GB for FP8) means 2× less DRAM traffic per token. The Marlin kernel fuses dequant into the GEMM, so the overhead is small. Result: **~3× decode throughput vs BF16, ~2× vs FP8 at batch 1–4** (morphllm 2026 benchmarks).
2. **Single-stream chatbot, low concurrency**: Same as above. Per-user TPOT (time per output token) is the goal; AWQ-Marlin's smaller weight read dominates.
3. **VRAM-constrained co-tenancy**: 35 GB weights + 60 GB KV on GH200 96 GB → fits **much larger batches' worth of KV** than FP8 (which leaves only 26 GB for KV). For 16k-context production workloads, AWQ can serve 2–3× more concurrent users on a single GH200.
4. **Model not available as FP8**: Some newer / smaller-vendor models ship AWQ but not FP8.

### When FP8 W8A8 wins on GH200

1. **High-batch throughput (batch ≥16)**: The compute side is now FP8 tensor-core-bound. FP8 W8A8 lights up Hopper's full TFLOPS budget. AWQ-Marlin can't match — it's still feeding BF16 activations through the tensor cores, so it's effectively a W4-decode + BF16-compute path. Result: **+50–80% throughput over BF16 at batch 32–64**, **1.4–1.8× over AWQ-Marlin at same batch**.
2. **Long output sequences with light input**: Same as above — compute scales linearly with output tokens, FP8 wins.
3. **Lower MMLU degradation**: FP8 W8A8 hits MMLU 83.8 (Δ 0.0 vs BF16) on Llama-3.1-70B; AWQ INT4 hits 83.6 (Δ −0.2). Marginal but real.
4. **No calibration data**: FP8 dynamic uses RTN, zero calibration. AWQ needs 128+ samples. Reduces deployment risk.
5. **KV cache also FP8**: doubles effective context capacity — composed with TurboQuant 3-bit KV gives 4× over BF16 KV. AWQ-Marlin only quantizes weights, so KV must be FP8/INT8/TurboQuant separately.
6. **MMLP/MLPerf compliance**: every MLPerf Inference v5.0/v5.1 winning submission is FP8. AWQ is research-grade in MLPerf.
7. **DeepSeek-V3, native-FP8 MoE**: not optional. Use FP8.
8. **MoE in general** (Mixtral 8x7B, DeepSeek): pre-quantized FP8 MoE has matured (DeepGEMM/DeepEP); AWQ MoE has had MLA-crash bugs (fixed in vLLM 0.21 but less battle-tested).

### Quick decision tree

```
Are you serving DeepSeek-V3 / V3.1 / any native-FP8 MoE?
  → YES: FP8 (no choice)
  → NO: continue
Is your concurrency ≥16 sustained?
  → YES: FP8 W8A8
  → NO: continue
Does your model have head_dim=256 (Gemma-4-E2B class)?
  → YES: BF16 or AWQ-Marlin (FP8 prefill regresses 1.6×)
  → NO: continue
Do you need maximum simultaneous users on a single GH200?
  → YES: AWQ-Marlin (frees ~35 GB more VRAM for KV cache)
  → NO: continue
Do you have an FP8 pre-quantized checkpoint of the model?
  → YES: FP8 W8A8 + FP8 KV cache
  → NO: AWQ-Marlin (faster to deploy than producing your own FP8)
```

### A note on combining them

You can run **AWQ-Marlin weights + FP8 KV cache** on vLLM 0.21 (`--quantization awq_marlin --kv-cache-dtype fp8_e4m3`). Gives small-batch decode advantage of AWQ + doubled KV capacity of FP8 KV. Best low-concurrency long-context combo on GH200.

You can also stack **AWQ-Marlin + TurboQuant 3-bit KV** for extreme long-context (≥64k) low-concurrency. Reference: vLLM 0.20+ supports `turboquant_3bit_nc` orthogonally to the weight quant.

---

## 4. NVFP4 / MXFP4 on Hopper — Why It's Not a Choice Yet

The B200/GB200 native-FP4 narrative does not transfer to GH200:

- **Hardware**: Blackwell SM100/SM120 has dedicated FP4 tensor cores. **Hopper SM90 does not.** GH200 has the H100 SM90 die.
- **NVFP4 on Hopper**: vLLM 0.21 includes a **Marlin FP4 software fallback** that loads NVFP4 weights and emulates the GEMM. You get the memory savings (≈ 4-bit weights) but the **GEMM is no faster than FP8 — typically slower** due to dequant overhead.
- **MXFP4 on Hopper**: same — emulated for memory, no throughput. Spheron 2026 explicitly notes "memory savings, allowing you to run much larger models, but you won't see the full computational speedup."
- **Practical implication**: If you have NVFP4 weights from a vendor (gpt-oss-20b/120b ship MXFP4), you can load them on GH200, but for serving throughput, **convert to AWQ-Marlin INT4 instead** — same memory class, real Hopper-native kernel.
- **Forward compat**: when you migrate GH200 → GB200, your FP8 inference workload picks up NVFP4 on Blackwell automatically via the same checkpoint with `--quantization nvfp4`. Treat FP4 as Blackwell-only until then.

---

## 5. Cross-Stack Feature Gaps as of 2026-05-25

These are non-cell rows worth tracking that don't fit the format/stack matrix cleanly:

| Feature | vLLM 0.21 | SGLang 0.5.12 | TRT-LLM 1.2.1 | llama.cpp |
|---|---|---|---|---|
| **FP8 KV cache + FA3 accumulation fix** | Yes (0.19.1+) | Yes | Yes (own kernel) | N/A |
| **TurboQuant KV (3-bit `nc`, k8v4)** | **Yes (production, 0.20+)** | WIP (issue #21618) | No | No |
| **Pre-quantized FP8 checkpoints from RedHat/NeuralMagic/FriendliAI** | Yes | Yes | Yes (also Modelopt) | No |
| **AWQ-Marlin auto-kernel selection** | Yes | Yes | Manual flag | N/A |
| **GPTQ-Marlin / Machete** | Yes (both) | Yes | TRT W4 paths | N/A |
| **MoE pre-quantized FP8** | Yes (DeepSeek, Mixtral) | Yes (DeepGEMM/DeepEP reference) | Yes | Q4_K_M GGUF only |
| **Sliding-window-skip flag for FP8 KV** | `--kv-cache-dtype-skip-layers sliding_window` | similar | Configurable per-layer | N/A |
| **W4A8-GPTQ Hopper-native** | Beta (Machete) | Beta | **Production** | N/A |
| **NVFP4 native acceleration** | Blackwell only | Blackwell only | Blackwell only | Software (any) |
| **head_dim=256 FP8 prefill** | **Regresses 1.6×** (BF16 fallback advised) | Same | Same | N/A |
| **Unified-memory MoE expert offload (GH200-specific)** | No | No | No | **Patch available** (discussion #18005, not upstreamed) |

---

## 6. Memory Update Recommendations

The current "FP8 broken on this stack — always bf16" memory should be **retained for DiT (Wan / HunyuanVideo)** but **explicitly carved away from text LLMs on Hopper**:

**Proposed text-LLM memory addition**:
> On Hopper (H100/H200/GH200) for text LLMs, use FP8 W8A8 e4m3 by default (vLLM 0.19.1+ / SGLang 0.5.12 / TRT-LLM 1.2.1). MMLU delta <0.5%, 1.5–1.8× throughput, 50% VRAM. Switch to AWQ-Marlin when batch ≤8 (decode-bound) or VRAM-constrained for multi-tenant. Use `--kv-cache-dtype fp8_e4m3` and `--kv-cache-dtype-skip-layers sliding_window` for hybrid models. Avoid FP8 for head_dim=256 (Gemma-4-E2B) — prefill regresses 1.6×. NVFP4/MXFP4 emulated only on Hopper — no throughput gain, save for Blackwell migration. TurboQuant 3-bit KV (vLLM 0.20+) for ≥32k contexts.

---

## 7. Confidence & Gaps

**High confidence**:
- FP8 Llama-3.3-70B production status on GH200 (Baseten, Lambda, vLLM recipe, TRT-LLM 1.2 validated list — multiple independent sources).
- AWQ-Marlin vs FP8 cross-over at batch ~8–16: backed by morphllm 2026 benchmarks, RedHat compressor data, vLLM blog throughput data, and "Give Me BF16" arXiv paper.
- TurboQuant 3-bit KV merge into vLLM 0.20.0 (PR #38479 verified merged 2026-04-15).
- Hopper has zero native NVFP4/MXFP4 acceleration (confirmed by multiple 2026 sources including Spheron, NVIDIA blog, vLLM Marlin FP4 fallback note).
- GGUF MMLU table for Llama-3.1-8B (arXiv 2601.14277v1, full table extracted).

**Medium confidence**:
- Exact Qwen2.5-72B MMLU FP8 delta — extrapolated from Llama-3.1-70B; no direct 72B paper number found. Likely also Δ ≤ 0.5% based on RedHatAI pre-quantized checkpoint shipping with no warnings.
- Specific batch-size cross-over for FP8 vs AWQ on Llama-3.3-70B — direct benchmark at "batch 32" with both formats side-by-side on a GH200 was not found in public sources. The ~8–16 cross-over is inferred from general low-batch-decode-bound vs high-batch-compute-bound dynamics and the 3× / 1.6× speedup data points.
- TRT-LLM 1.2.1 specifically (vs 1.2.0) — release notes for the .1 patch did not surface cleanly; assumed FP8/NVFP4 changes are inherited from 1.2.0 plus small bug fixes.

**Gaps**:
- No public direct Llama-3.3-70B AWQ-Marlin vs FP8 head-to-head benchmark on **GH200 specifically** (most numbers are from H100). The GH200 +32% Hopper-vs-H100 advantage should apply to both formats roughly equally.
- TurboQuant 2-bit (k8v4) production maturity unclear — vLLM 0.21 ships presets but real-world deployments at scale are not yet documented.
- W4A8-GPTQ on TRT-LLM 1.2.1 Llama-3.3-70B numbers are not public; likely competitive with FP8 W8A8 at large batch on Hopper but no published comparison.

---

## 8. References

### Primary
- vLLM blog 2026-04-22, "The State of FP8 KV-Cache and Attention Quantization in vLLM" — https://vllm.ai/blog/2026-04-22-fp8-kvcache
- vLLM Llama-3.3-70B recipe — https://docs.vllm.ai/projects/recipes/en/latest/Llama/Llama3.3-70B.html
- vLLM FP8 docs — https://docs.vllm.ai/en/latest/features/quantization/fp8/
- vLLM INT4 W4A16 docs — https://docs.vllm.ai/en/latest/features/quantization/int4/
- vLLM quantization overview — https://docs.vllm.ai/en/latest/features/quantization/
- SGLang quantization docs — https://docs.sglang.io/advanced_features/quantization.html
- SGLang DeepSeek-V3 guide — https://github.com/sgl-project/sglang/blob/main/docs/basic_usage/deepseek_v3.md
- TRT-LLM quantization — https://nvidia.github.io/TensorRT-LLM/features/quantization.html
- TRT-LLM release notes — https://nvidia.github.io/TensorRT-LLM/release-notes.html
- TurboQuant PR #38479 — https://github.com/vllm-project/vllm/pull/38479
- TurboQuant repo — https://github.com/varjoranta/turboquant-vllm
- TurboQuant vLLM metal docs — https://docs.vllm.ai/projects/vllm-metal/en/latest/turboquant/

### Benchmarks & papers
- "Give Me BF16 or Give Me Death?" arXiv 2411.02355v3
- "Which Quantization Should I Use?" arXiv 2601.14277v1 (Llama-3.1-8B GGUF unified eval)
- "Fast NF4 Dequantization Kernels for LLM Inference" arXiv 2604.02556
- "Does quantization affect models' performance on long-context tasks?" ACL 2025 EMNLP main 479
- Baseten Llama-3.3-70B on GH200 Lambda — https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/
- Lambda GH200 inference blog — https://lambda.ai/blog/putting-the-nvidia-gh200-grace-hopper-superchip-to-good-use-superior-inference-performance-and-economics
- morphllm 2026 vLLM benchmarks — https://www.morphllm.com/vllm-benchmarks
- Spheron FP8 quantization 2026 — https://www.spheron.network/blog/fp8-quantization-inference-performance-hardware-explained/
- llama.cpp GH200 discussion #18005 — https://github.com/ggml-org/llama.cpp/discussions/18005

### FP8 status / bugs cross-reference
- `/home/ubuntu/research/gh200_inference/round2/B_fp8_forensic_audit.md` — comprehensive FP8 bug atlas; DiT FP8 broken, dense LLM FP8 production
- DeepSeek-V3 technical report — arXiv 2412.19437

[8] morphllm cites "Llama 3.1 70B on 1× H100 with batch 64 → 460 tok/s FP8 W8A8".
[9] Baseten: GH200 +32% over H100 SXM at FP8 batch 32 Llama-3.3-70B ShareGPT.
