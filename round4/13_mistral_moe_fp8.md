# Mistral / Mixtral on GH200 — MoE FP8 Fragility, Quantization Routing, Function-Calling

**Agent**: Round 4, deep dive #13
**Date**: 2026-05-25
**Hard prior (unchanged)**: FP8 broken on DiT stack; bf16 default for diffusion. FP8 on **dense LLMs** is production-grade on Hopper post-vLLM 0.19.1. **MoE FP8 online quant remains fragile** — the central subject of this drill.

**Read first**:
- Round 2 `/home/ubuntu/research/gh200_inference/round2/B_fp8_forensic_audit.md` — Mixtral 8x7B FP8 was the original 2024 sore point (vLLM #5535); MoE online-quant remains the highest-risk class in May 2026 (vLLM #30830, #41022).
- Round 3 `/home/ubuntu/research/gh200_inference/round3/03_vllm_verify.md` — vLLM stable is **0.21.0 (2026-05-15)**; FA3 accumulation patch applies to `head_dim ≤ 128` only.

---

## TL;DR

1. **Mistral's lineup as of May 2026 is dominated by MoE, not dense.** Mistral Large 3 (Dec 2025) is **675B total / 41B active** granular-MoE — not 123B dense. The user-prompt phrase "Mistral-Large-3 (123B dense)" refers to either the legacy Mistral-Large-2-2407 (123B dense, retired 2025-03-30 but still on HF) or Mistral Medium 3.5 128B dense (Apr 2026, current). Both are covered here.

2. **MoE FP8 online quantization in vLLM is still broken in May 2026.** Two unfixed open bugs anchor the risk: **vLLM #41022** (online FP8 ignores `logical_widths` on `MergedColumnParallelLinear` — opened 2026-04-27, no fix) and **vLLM #30830** (`patched_weight_loader` corrupts FP8 MoE expert weights — opened 2025-12-17, still active). The rule "use pre-quantized FP8 checkpoints, never `--quantization fp8` at load time" is hard.

3. **Safe MoE quantization paths on Hopper in May 2026**: **(a)** pre-quantized FP8 checkpoint (RedHatAI / unsloth / mistralai-NVFP4 sibling) — production-grade for Mixtral / Mistral-Small-4; **(b)** AWQ-Marlin INT4 — production-grade for dense (Mistral-Large-2 / Medium-3.5), watch **vLLM #41511** for MoE; **(c)** W4A16 MoE at **TP=2 only** until #41511 fix lands; **(d)** BF16 if you have the HBM.

4. **Mixtral 8x22B does NOT fit single GH200.** 141 GB FP8 weights > 96 GB HBM. The viable single-GPU path is: (a) drop to AWQ INT4 (~80 GB) and offload KV to Grace LPDDR via `--kv_offloading_backend native --kv_offloading_size 256`, or (b) use 2×H200 SXM with TP=2. Single GH200 BF16 is impossible (~282 GB).

5. **Mistral Large 3 (675B/41B MoE) does NOT fit single GH200.** Even NVFP4 is ~340 GB. Requires **8×H200 FP8** or **4×B200 NVFP4** single-node. Off-table for any single-GPU rig.

6. **Mistral Medium 3.5 (128B dense)**: closest to "Mistral-Large-3 (123B dense)" reading. **Does not fit single GH200 at FP8** (~128 GB > 96 GB HBM). AWQ-INT4 + Grace-LPDDR offload is the only single-GH200 path. Multi-node default: 4×H100 FP8 or 2×H200 FP8.

7. **Function-calling**: both engines accept `--tool-call-parser mistral`. vLLM is the documented default in Mistral's own model cards (`mistralai/Devstral-2-123B-Instruct-2512`, `Mistral-Large-3-675B-Instruct-2512`, `Mistral-Small-4-119B-2603`). **SGLang adds compressed-FSM constrained decoding** — wins on structured-JSON tasks; lacks first-class native multimodal for some Mistral vision checkpoints. Practical recipe: vLLM for serving, SGLang for "must-emit-valid-JSON-on-first-token" workloads.

---

## 1. Mistral / Mixtral Family — Current SKU List (May 2026)

| SKU | HF model ID | Architecture | Total / Active | Context | Released | Status | Recipe doc |
|---|---|---|---|---|---|---|---|
| **Mistral Large 3** | `mistralai/Mistral-Large-3-675B-Instruct-2512` | Granular MoE + vision | **675B / 41B** (LM: 673B / 39B; vision encoder: 2.5B) | 256k | 2025-12-02 | **Flagship**, Apache 2.0 | [vLLM Recipes / Mistral-Large-3](https://docs.vllm.ai/projects/recipes/en/latest/Mistral/Mistral-Large-3.html) |
| **Mistral Large 3 NVFP4** | `mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4` | Same, NVFP4 weights | Same | <64k recommended | 2025-12-02 | Active | Same |
| **Mistral Medium 3.5** | `mistralai/Mistral-Medium-3.5-128B` | **Dense** transformer | 128B (all active) | 256k | 2026-04-30 | **Frontier-class** (multimodal, coding, agentic) | [Lushbinary self-host guide](https://lushbinary.com/blog/mistral-medium-3-5-self-hosting-vllm-sglang-guide/) |
| **Mistral Small 4** | `mistralai/Mistral-Small-4-119B-2603` | MoE (128 experts, top-4) | **119B / 6.5B** | 256k | 2026-03 | **Current Small**, hybrid (instruct+reasoning+coding+vision), `reasoning_effort` knob | [Mistral blog](https://mistral.ai/news/mistral-small-4) |
| **Mistral Small 3.2** | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | Dense | 24B | 128k | 2025-06 | Active, cheaper than Small 4 | [RedHatAI FP8](https://huggingface.co/RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic) |
| **Mistral Nemo** | `mistralai/Mistral-Nemo-Instruct-2407` | Dense | 12B | 128k | 2024-07 | **Still recommended** ("Only Nemo 12B Worth Running" — InsiderLLM 2026), Apache 2.0, NVIDIA collab | NIM card |
| **Codestral 2508** | API only (closed) | Dense | 22B (same as 2501) | 256k | 2025-08 | Current code/FIM model | [Mistral Codestral 25.08](https://mistral.ai/news/codestral-25-08) |
| **Devstral 2** | `mistralai/Devstral-2-123B-Instruct-2512` | **Dense** | 123B | (likely 128k) | 2025-12 | "Frontier code agents model" | HF card |
| **Devstral Small 2** | `mistralai/Devstral-Small-2-24B-Instruct-2512` | Dense | 24B | 128k | 2025-12 | Edge code model | HF card |
| **Ministral 3 (3B/8B/14B)** | `mistralai/Ministral-{3B,8B,14B}-Instruct-2512` | Dense | 3B / 8B / 14B | 128k | 2025-12 | Best-in-class small/vision | HF cards |
| **Magistral Medium 1.2** | `mistralai/Magistral-Medium-1.2` | Dense | (Medium-class) | 128k | 2025-09 | Reasoning-specialist | HF |
| **Voxtral series** | `mistralai/Voxtral-*` | Audio (TTS/transcription/audio-in) | various | — | 2025-2026 | Audio-only | NIM / Mistral docs |
| **Mistral-Large-2** (legacy) | `mistralai/Mistral-Large-Instruct-2407` | **Dense** | 123B | 128k | 2024-07 | **DEPRECATED 2025-03-30** (Mistral docs), still on HF under Apache 2.0; still loads in vLLM. Migrate to Medium 3.5 / Large 3. | — |
| **Mixtral 8x22B** (legacy) | `mistralai/Mixtral-8x22B-Instruct-v0.1` | MoE | **141B / 39B** | 64k | 2024-04 | **DEPRECATED 2025-03-30**, still on HF Apache 2.0, still loads in vLLM. Replacement: Mistral Small 4 (similar active param count). | — |
| **Mixtral 8x7B** (legacy) | `mistralai/Mixtral-8x7B-Instruct-v0.1` | MoE | **46.7B / 12.9B** | 32k | 2023-12 | Deprecated, still loads. Replacement: Ministral-14B or Small 3.2-24B. | — |
| **Mistral 7B** (legacy) | `mistralai/Mistral-7B-Instruct-v0.3` | Dense | 7B | 32k | 2024-05 | Deprecated, still loads. Replacement: Ministral-8B. | — |

### Sub-question — "Mistral-Large-3 (123B dense)"

This pairing in the prompt is inconsistent with public Mistral docs. **Mistral Large 3 = 675B/41B MoE, NOT 123B dense.** The 123B-dense generation in the Mistral lineup is **Mistral Large 2 (2407)** and now **Devstral 2 123B (2512)**. The current dense flagship is **Mistral Medium 3.5 128B**. I cover all three single-node-tractable 123B/128B candidates in §5.

---

## 2. MoE FP8 Fragility — Specific OPEN Bugs (May 2026)

Anchor cites: Round 2 §1.2 already listed vLLM #5535, #30830 and sglang #24650. Verifying status + adding two new bugs surfaced this round.

### 2.1 OPEN — vLLM #41022 — Online FP8 ignores `logical_widths` on MergedColumnParallelLinear

- **Title**: "Online FP8 quantization ignores logical_widths on MergedColumnParallelLinear"
- **Opened**: 2026-04-27 — **OPEN, no fix PR, no assignee**
- **Repro**: `vllm serve Qwen/Qwen3.5-35B-A3B --quantization fp8`
- **Root cause**: `MergedColumnParallelLinear` fuses multiple projections (QKV, gate_up, GDN in_proj_qkvz) into one weight matrix with `logical_widths` carrying per-shard sizes. The INT8 CUTLASS path properly uses `convert_to_channelwise` to respect these. Both `Fp8OnlineLinearMethod` and `Fp8PerTensorOnlineLinearMethod` apply `ops.scaled_fp8_quant` to the entire fused weight, producing one global scale instead of per-shard scales.
- **Affected models (named in issue)**:
  - **GDN models** (Qwen3.5, Qwen3.6) — recurrence amplifies precision loss into NaN.
  - **SwiGLU models — Llama, Mistral, Qwen, Gemma** — shared scaling reduces gate-shard resolution.
  - **Mistral implication**: any Mistral SwiGLU MoE (Mixtral 8x7B / 8x22B / Mistral-Small-4 119B / Mistral-Large-3 675B) hits this if you use `--quantization fp8` (online quant) instead of loading pre-quantized weights.
- **Practical rule**: **NEVER use `--quantization fp8` for Mistral/Mixtral MoE.** Always download pre-quantized FP8 checkpoint (RedHatAI, neuralmagic, unsloth).

### 2.2 OPEN — vLLM #30830 — `patched_weight_loader` MoE FP8 accuracy issue

- **Title**: "accuracy issue on MoE online fp8 quantization"
- **Opened**: 2025-12-17 — **OPEN**, no fix, no PR
- **Repro hardware**: 7×A100 80GB, CUDA 12.9, PyTorch 2.9
- **Repro model**: `Qwen/Qwen3-30B-A3B`
- **Root cause**: `patched_weight_loader` in FP8 quant path corrupts expert weights → incoherent output with repeated fragments.
- **Mistral implication**: same MoE expert-loading path is used by Mistral MoE. Online FP8 quant degrades both routing and expert outputs. Confirms #41022 from a different angle.

### 2.3 OPEN — sglang #24650 — DeepSeek-V4-Flash-FP8 + EAGLE RoPE crash

- **Title**: "DeepSeek-V4-Flash-FP8 with EAGLE crashes under benchmark load with RoPE tensor shape mismatch (expected 128 but got 126)"
- **Opened**: 2026-05-08 — **OPEN**, last-confirmed status as of fetch 2026-05-25, no fix PR visible
- **Repro**: 4×H200, TP=4, ISL=1000 OSL=1000, 64 concurrent
- **Root cause**: Fused RoPE in compressed attention indexer expects batch aligned to 128; under EAGLE the verification batch can be 126.
- **Mistral implication**: NOT directly Mistral, but tells you **EAGLE speculative decode with FP8-MoE on SGLang is currently risky on H200**. Mistral Large 3 ships an EAGLE draft (`Mistral-Large-3-675B-Instruct-2512-Eagle`); use it on **vLLM** ≥ 0.21, not SGLang, until #24650 is fixed.

### 2.4 OPEN — vLLM #41511 — W4A16 MoE TP>2 sharding bug

- **Title**: "compressed-tensors W4A16 MoE: weight_scale not sharded along K under tensor parallelism, kernel computes wrong group_size"
- **Opened**: 2026-05-02 — **OPEN**, no fix PR, no assignee
- **Affected**: DeepSeek-V4-Flash-W4A16, Intel/DeepSeek-V4-Flash-W4A16-AutoRound, and "any compressed-tensors W4A16 MoE model under TP > 2"
- **TP behavior table** (from issue body):

| TP | Kernel-believed `group_size` | Result |
|---|---|---|
| 1 | 128 (correct) | Works |
| 2 | 64 (wrong but template exists) | Works, output mathematically correct |
| 4 | 32 (below MIN_THREAD_K) | Crash "Invalid thread config" |
| 8 | 16 | Crash |

- **Workaround**: TP=2 only.
- **Mistral implication**: W4A16 (INT4) MoE for Mixtral 8x22B / Mistral-Small-4 / Mistral-Large-3 **must run at TP=2** until #41511 is fixed. For Mistral-Large-3 675B that means W4A16 is unusable single-node (needs TP=8). For Mixtral 8x22B on 2×H200 (TP=2) W4A16 works.

### 2.5 OPEN — vLLM #39000 — Gemma 4 MoE MXFP4 weight loader 2D/3D mismatch

- **Title**: "Gemma 4 MoE (26B-A4B) — runtime MXFP4 quantization crashes during weight loading in fused MoE layer"
- **Opened**: 2026-04-04 — **OPEN**
- **Root cause**: weight tensor loaded 2D `(out_features, in_features)` but MXFP4 branch of weight_loader expects 3D `(num_experts, out_features, in_features)`. IndexError at `fused_moe/layer.py:1073`.
- **Mistral implication**: cross-architecture proof that **fused MoE weight loaders in vLLM are not robust to mixed-precision online quant**. Touches every MoE family, not just Gemma. Avoid `--quantization mxfp4` for Mistral MoE.

### 2.6 CLOSED but historically informative — vLLM #5535 (Mixtral 8x7B FP8 slow)

- **Closed**, PR #5327 referenced as fix
- **Original bug** (June 2024): Mixtral 8x7B Instruct FP8 throughput 29 tok/s vs 88 tok/s BF16 on H100 — 67% perf regression. Fixed via dedicated fused-MoE FP8 kernel.
- **Status May 2026**: Long-resolved for **pre-quantized** Mixtral FP8 (RedHatAI/neuralmagic). The bug today is online quant (#41022/#30830), not the original kernel-perf issue.

### 2.7 Confidence summary on MoE FP8 status, May 2026

| Path | Status | Citation |
|---|---|---|
| **vLLM `--quantization fp8` on MoE** | **BROKEN** — silently corrupts expert weights | #41022 + #30830 OPEN |
| **vLLM `--quantization mxfp4` on MoE** | **BROKEN** at weight load | #39000 OPEN |
| **vLLM W4A16 MoE TP > 2** | **BROKEN** kernel crash | #41511 OPEN |
| **vLLM W4A16 MoE TP = 2** | **WORKS** | #41511 workaround |
| **vLLM pre-quantized FP8 MoE checkpoint** | **WORKS** (RedHatAI Mixtral-8x22B-FP8 ships, neuralmagic ditto) | Round 2 §1.2 |
| **vLLM NVFP4 MoE on B200** | **WORKS** (`Mistral-Large-3-NVFP4`, `Mistral-Small-4-NVFP4` ship) | Mistral HF cards |
| **vLLM NVFP4 MoE on H100/H200/GH200 (Marlin FP4 fallback)** | **WORKS for memory, NO speedup over FP8** | Mistral HF NVFP4 card |
| **SGLang FP8 MoE + EAGLE on H200** | **BROKEN under load** | sglang #24650 OPEN |
| **SGLang BF16 MoE** | **WORKS** | Default-stable |

---

## 3. Safe MoE Quantization Paths — Per Engine, Per Hardware

### 3.1 Decision matrix — May 2026

| MoE class | Hopper (H100/H200/GH200) safe path | Blackwell (B200/GB200) safe path | Notes |
|---|---|---|---|
| **Pre-quantized FP8 checkpoint** | ✅ Default. e.g. `RedHatAI/Mixtral-8x22B-Instruct-v0.1-FP8` | ✅ Same checkpoint works | Use this unless you have explicit calibrated re-quant data |
| **Online `--quantization fp8`** | ❌ AVOID (vLLM #41022, #30830) | ❌ Same bug present | Forces `MergedColumnParallelLinear` corruption |
| **AWQ INT4 Marlin** | ✅ For **dense Mistral** (Large-2, Medium-3.5, Devstral 2-123B, Small 3.2) — 741 tok/s on Marlin-AWQ per Jarvis Labs benchmarks | ✅ Works | NOT well-supported for MoE — AWQ kernels for fused-MoE incomplete |
| **GPTQ INT4 Marlin** | ✅ For dense Mistral | ✅ Works | 2.6× over GPTQ-non-Marlin |
| **W4A16 compressed-tensors MoE** | ⚠️ TP=2 only (#41511) | ⚠️ TP=2 only | Wait for fix before scaling out |
| **NVFP4 native** | ⚠️ Marlin-FP4 fallback — memory win only, no speedup | ✅ Native, sweet spot | `Mistral-Large-3-NVFP4`, `Mistral-Small-4-NVFP4` |
| **BF16** | ✅ Always safe (if it fits) | ✅ Always safe | Use when correctness > throughput |

### 3.2 vLLM-specific flags by path

```bash
# Pre-quantized FP8 (SAFE — recommended default for Mixtral 8x22B / Mistral-Small-4)
vllm serve RedHatAI/Mixtral-8x22B-Instruct-v0.1-FP8 \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --enable-auto-tool-choice --tool-call-parser mistral

# Online FP8 — DO NOT USE FOR MoE
# vllm serve mistralai/Mixtral-8x22B-Instruct-v0.1 --quantization fp8   # BROKEN

# AWQ INT4 Marlin — dense Mistral
vllm serve TheBloke/Mistral-Large-Instruct-2407-AWQ \
  --quantization awq_marlin \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice --tool-call-parser mistral

# W4A16 compressed-tensors MoE — TP=2 ONLY
vllm serve nm-testing/Mixtral-8x22B-Instruct-v0.1-quantized.w4a16 \
  --quantization compressed-tensors \
  --tensor-parallel-size 2   # NEVER 4 or 8 — #41511

# NVFP4 on Hopper (memory benefit only)
vllm serve mistralai/Mistral-Small-4-119B-2603-NVFP4 \
  --tensor-parallel-size 2 \
  --tool-call-parser mistral --enable-auto-tool-choice
```

### 3.3 SGLang-specific flags

```bash
# BF16 default (safest)
python -m sglang.launch_server \
  --model-path mistralai/Mixtral-8x22B-Instruct-v0.1 \
  --tp 2 \
  --tool-call-parser mistral

# FP8 KV cache + pre-quantized FP8 model
python -m sglang.launch_server \
  --model-path RedHatAI/Mixtral-8x22B-Instruct-v0.1-FP8 \
  --tp 2 \
  --kv-cache-dtype fp8_e4m3 \
  --tool-call-parser mistral
# (Avoid fp8_e5m2 — sglang #18550 silent Triton fallback)
```

SGLang does not currently provide its own MoE FP8 online-quant path — it consumes pre-quantized HF checkpoints and uses the same fused-MoE kernels under the hood. The #24650 EAGLE+FP8 crash is the only Mistral-relevant SGLang FP8 bug.

---

## 4. Mixtral 8x22B (141B / 39B Active) on Single GH200

### 4.1 Memory math

| Format | Weights | Activations + KV (256-batch, 8k ctx) | Total | Fits 96 GB GH200 HBM? |
|---|---|---|---|---|
| BF16 (2 B/p) | ~282 GB | ~30 GB | ~312 GB | **NO** |
| FP8 (1 B/p) | ~141 GB | ~30 GB | ~171 GB | **NO** |
| AWQ-INT4 (0.5 B/p) | ~70 GB | ~10 GB | ~80 GB | **YES with care** |
| W4A8 (0.5 B/p weights, 1 B/p act) | ~70 GB + 10 GB | ~80 GB | **YES marginal** |
| GGUF Q4_K_M | ~85 GB | ~10 GB | ~95 GB | **MARGINAL** |

**Verdict**: Single GH200 (96 GB HBM3) **only fits Mixtral 8x22B at INT4**. KV-offload to the 480 GB Grace LPDDR is mandatory for non-trivial context. BF16 and FP8 single-GH200 are off the table.

### 4.2 Single-GH200 INT4 recipe (recommended)

```bash
# AWQ-Marlin INT4 on single GH200 (Hopper SM90)
vllm serve mistral-community/Mixtral-8x22B-v0.1-4bit \
  --quantization awq_marlin \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --cpu-offload-gb 40 \
  --kv_offloading_backend native --kv_offloading_size 200 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --tool-call-parser mistral
```

Rationale:
- AWQ-Marlin INT4 kernel = ~10× speedup over un-Marlinized AWQ, 741 tok/s ceiling on dense; MoE numbers lower due to expert dispatch overhead but still the practical choice.
- `--cpu-offload-gb 40` spills weights to Grace LPDDR when HBM tight; `kv_offloading_size 200` reserves 200 GB LPDDR for KV overflow.
- Avoid TP — single GPU keeps you under #41511's TP > 2 threshold (which only bites W4A16 compressed-tensors, but conservative path is single-rank).

### 4.3 2×H200 SXM TP=2 FP8 recipe (cleaner production path if you have two GPUs)

```bash
# Pre-quantized FP8 Mixtral on 2× H200 SXM
vllm serve RedHatAI/Mixtral-8x22B-Instruct-v0.1-FP8 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --enable-chunked-prefill \
  --max-num-seqs 256 \
  --enable-auto-tool-choice --tool-call-parser mistral
```

99.54% accuracy recovery vs BF16 baseline per RedHatAI card. Mixtral 8x22B at 256k ctx requires bumping `--max-model-len` and accepting KV blow-up (default is 65k in the HF config — Mixtral's 64k window).

### 4.4 Benchmark anchor points (May 2026)

| Source | HW | Quant | Engine | tok/s aggregate |
|---|---|---|---|---|
| RedHatAI card | 4×A100 80GB | FP8 (pre-quant) | vLLM | "production" — no number |
| AMD ROCm blog | MI300X | BF16 | vLLM-ROCm | Listed but no single number |
| **GH200 single** | — | — | — | **No public number for Mixtral 8x22B on single GH200.** Highest-priority custom benchmark for our rig. |

Working estimate (interpolated from Round 3's Llama-3.3-70B GH200 working assumption of 50-100 tok/s aggregate BF16-with-offload, scaled by 39B/70B active and INT4/BF16): **single GH200 Mixtral 8x22B AWQ-INT4 ≈ 30-70 tok/s aggregate**. **Flag as estimate**, not measurement.

---

## 5. Mistral-Large-3 (and 123B-Dense Cousins) on GH200 — Recipes

The prompt's "Mistral-Large-3 (123B dense)" maps to three candidates. Treat each separately.

### 5.1 Mistral Large 3 — 675B / 41B granular MoE (Dec 2025)

| Format | Total weight | Single GH200 fit? | Single 8×H200 node fit? | Single 4×B200 node fit? |
|---|---|---|---|---|
| BF16 | ~1,350 GB | NO | NO (1,128 GB HBM) | NO |
| FP8 | ~675 GB | NO | **YES (8×H200 = 1,128 GB)** | NO (4×B200 = 768 GB OK marginal, prefers 8) |
| NVFP4 | ~340 GB | NO | YES (oversize) | **YES (NATIVE)** |

**Conclusion: Mistral-Large-3 is single-node-but-not-single-GPU.** Single GH200 cannot host it under any quantization. Multi-GH200 over NVLink-C2C / IB is the only path and is not a standard configuration.

**Standard FP8 recipe (8×H200 single node)**:

```bash
vllm serve mistralai/Mistral-Large-3-675B-Instruct-2512 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --tokenizer_mode mistral --config_format mistral --load_format mistral \
  --enable-auto-tool-choice --tool-call-parser mistral \
  --limit-mm-per-prompt '{"image": 10}' \
  --speculative_config '{
    "model": "mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle",
    "num_speculative_tokens": 3,
    "method": "eagle",
    "max_model_len": 16384
  }'
```

**NVFP4 recipe (4×B200 single node)**:

```bash
vllm serve mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4 \
  --tensor-parallel-size 4 \
  --max-model-len 65536  # NVFP4 perf drops > 64k per HF card
  --tokenizer_mode mistral --config_format mistral --load_format mistral \
  --enable-auto-tool-choice --tool-call-parser mistral
```

> **NVFP4 caveat from HF card (verified this round)**: "<32k tokens — no degradation vs FP8; 32k-64k tokens — minor degradation; >64k — subsequent performance drop, use FP8 weights instead." For long-context workloads, stay on FP8 even on Blackwell.

**Benchmarks** (Artificial Analysis, May 2026):
- Output speed: 51.7 tok/s
- TTFT: 1.09 s
- API pricing: $0.50/M input, $1.50/M output
- AA Intelligence Index: 23 (below average for its class)
- MMLU-Pro: 73.11%, MATH-500: 93.60% (cross-source)

### 5.2 Mistral Medium 3.5 — 128B dense (Apr 2026, current frontier-class)

Closest to the user's "Mistral-Large-3 (123B dense)" description.

| Format | Weight | Single GH200 (96 GB)? | 2×H200 SXM? | 4×H100 80GB? |
|---|---|---|---|---|
| BF16 | ~256 GB | NO | NO (282 GB > 282 GB ✗ marginal) | YES (320 GB) |
| FP8 | ~128 GB | **NO marginal** | YES (282 GB) | YES |
| INT4 / AWQ | ~64 GB | **YES with offload** | YES | YES |

**Single GH200 path** (the GH200-relevant question):

```bash
# AWQ-INT4 Marlin (only viable single-GH200 path for 128B dense)
vllm serve TheBloke/Mistral-Medium-3.5-128B-AWQ \   # hypothetical path; verify exists
  --quantization awq_marlin \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92 \
  --cpu-offload-gb 30 \
  --kv_offloading_backend native --kv_offloading_size 256 \
  --enable-prefix-caching --enable-chunked-prefill \
  --enable-auto-tool-choice --tool-call-parser mistral
```

Note: as of 2026-05-25, I did not verify a community-AWQ release of Medium 3.5 specifically. If absent, fall back to FP8 on **2×H200 SXM** or quantize yourself via `llm-compressor` (Mistral docs reference this path for Large 3).

**Standard FP8 recipe (4×H100 or 2×H200)**:

```bash
vllm serve mistralai/Mistral-Medium-3.5-128B \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --tool-call-parser mistral --enable-auto-tool-choice \
  --gpu_memory_utilization 0.8
```

**Benchmarks**: 77.6% SWE-Bench Verified (Lushbinary). Throughput numbers not published per-config; expected ~25-40 tok/s aggregate at TP=4 H100 FP8 batch=32, extrapolating from Llama-70B FP8 H100 references.

### 5.3 Mistral Large 2-2407 (legacy 123B dense)

For users who literally have Mistral-Large-2-Instruct deployed:

```bash
# AWQ-INT4 single GH200
vllm serve TheBloke/Mistral-Large-Instruct-2407-AWQ \
  --quantization awq_marlin \
  --max-model-len 32768 \
  --cpu-offload-gb 30 \
  --kv_offloading_backend native --kv_offloading_size 200 \
  --enable-auto-tool-choice --tool-call-parser mistral
```

Deprecated by Mistral (retired 2025-03-30) but Apache-2.0 weights still load. Migration target: Mistral Medium 3.5 128B.

### 5.4 Devstral 2 — 123B dense (Dec 2025)

```bash
# vLLM
vllm serve mistralai/Devstral-2-123B-Instruct-2512 \
  --tool-call-parser mistral --enable-auto-tool-choice \
  --tensor-parallel-size 8

# SGLang
python -m sglang.launch_server \
  --model-path mistralai/Devstral-2-123B-Instruct-2512 \
  --host 0.0.0.0 --port 30000 \
  --tp 8 --tool-call-parser mistral
```

Same memory footprint class as Mistral-Large-2. Single-GH200 requires AWQ-INT4 + KV offload; multi-GH200 single-node FP8 needs 4× / 8× units (not a stocked config).

---

## 6. Function-Calling and Structured Output — vLLM vs SGLang

### 6.1 Mistral's official tool-call parser support

Mistral's own model cards (Mistral-Large-3, Devstral 2, Mistral-Small-4, Devstral Small 2) ship **both** vLLM and SGLang launch commands using `--tool-call-parser mistral` and `--enable-auto-tool-choice` flags. This parser is on both engines (vLLM list per docs; SGLang tool_parser list `mistral` — confirmed in §6.4 below).

### 6.2 vLLM tool-calling specifics

- Flag set: `--enable-auto-tool-choice --tool-call-parser mistral`
- Parser pulls tool-call args from the raw model output AFTER generation completes.
- **No schema-level constraint applied during decoding** — args may occasionally be malformed (Mistral's own recommendation: retry on parse failure).
- Mistral recommends temperature ≤ 0.15 for production tool calls.
- **Speculative decoding compatible**: `--speculative_config` works alongside tool parser, but watch sglang #24650 if mirroring to SGLang on FP8-MoE+EAGLE.

### 6.3 SGLang tool-calling specifics

- Flag set: `--tool-call-parser mistral`
- Available parsers (full list confirmed via SGLang docs 2026-05): `deepseekv3, deepseekv31, deepseekv32, glm, gpt-oss, kimi_k2, llama3, llama4, mistral, pythonic, qwen, qwen3_coder, step3` (13 total).
- **Compressed-FSM constrained decoding**: SGLang can enforce JSON schemas / regex during generation (not post-hoc). This is the structural difference.
- Lower TTFT for structured tasks because it avoids retry loops.

### 6.4 Practical workflow recommendation

| Workload | Engine | Why |
|---|---|---|
| Open-ended tool calls (LLM decides which tool when) | **vLLM** | Better throughput, simpler stack, Mistral's documented default, EAGLE-3 spec-decode mature |
| Strict JSON schema / regex enforcement (e.g. extracting structured data, agent control planes that must NOT emit invalid JSON) | **SGLang** | Compressed-FSM constrained decoding produces valid output on first attempt |
| Multi-step agent loops with structured intermediate states | **SGLang** | RadixAttention + per-step schemas + cache reuse |
| Vision tool-calling (Pixtral / Mistral-Large-3 vision) | **vLLM** | Mistral's vision tokenizer / `--load_format mistral` first-class on vLLM; SGLang Pixtral/Mistral vision parsers historically less mature (verify if SGLang ships vision Mistral parser at your version) |
| FP8-MoE + EAGLE | **vLLM** | sglang #24650 OPEN on H200 |
| Reasoning models w/ `reasoning_effort` (Mistral-Small-4) | **vLLM ≥ 0.21** | 0.21 added "speculative decoding respects reasoning/thinking budgets" — direct match. SGLang has reasoning parser too but vLLM is the explicit reference. |
| Heavy concurrency tool-call serving (>200 concurrent) | **vLLM** | Red Hat 2026-04-16 EAGLE benchmark shows persistence; SGLang FA3 backend hasn't published equivalent at 200-concurrent yet |

**Hybrid pattern in the wild**: vLLM bulk inference + SGLang for the structured-output endpoint, behind a single router. This is what "many enterprises combine both" refers to in the May 2026 SGLang-vs-vLLM comparison literature.

---

## 7. Summary Recipes for the GH200 / Mistral User

### 7.1 Single GH200 (96 GB HBM + 480 GB Grace LPDDR)

| Model | Best path | Estimated tok/s |
|---|---|---|
| Mistral Nemo 12B | BF16 native, single GPU, no offload | ~80-150 tok/s aggregate (interpolated from H100 NIM 1.4× BF16) |
| Mistral Small 3.2 24B | BF16 or pre-quant FP8 (`RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic`) | ~60-100 tok/s |
| Codestral 22B | BF16 native | ~70-100 tok/s |
| Mistral Small 4 119B / 6.5B-active MoE | Pre-quant FP8 + KV-offload (marginal at 96 GB; recommend 2×H200 instead) | Marginal — needs ~120 GB FP8 weights, doesn't fit single GH200 cleanly |
| Mistral Large 2 / Devstral 2 / Medium 3.5 (123B-128B dense) | **AWQ-INT4 + KV-offload** (only viable single-GH200 path) | ~20-50 tok/s estimate |
| Mixtral 8x22B (141B/39B MoE) | **AWQ-INT4 + KV-offload** | ~30-70 tok/s estimate |
| Mistral Large 3 (675B/41B MoE) | **NOT VIABLE** on single GH200 — requires 8×H200 FP8 | n/a |

### 7.2 Hard rules (May 2026)

1. **NEVER `--quantization fp8` on Mistral/Mixtral MoE.** Use pre-quantized FP8 checkpoints only (RedHatAI, neuralmagic, unsloth). (vLLM #41022, #30830)
2. **NEVER `--quantization mxfp4` on Mistral MoE.** (vLLM #39000)
3. **NEVER W4A16 compressed-tensors MoE at TP > 2.** (vLLM #41511)
4. **NEVER FP8 + EAGLE on SGLang for MoE under load.** (sglang #24650). Use vLLM ≥ 0.21 instead.
5. **NEVER `fp8_e5m2` on SGLang FA3.** Use `fp8_e4m3`. (sglang #18550)
6. **ALWAYS pin `mistral_common >= 1.8.6`** before launching Mistral-3 series on vLLM. (Red Hat 2025-12 guide)
7. **ALWAYS test with the 128k-needle benchmark** before promoting any FP8 deployment that uses long context — even on dense Mistral. (Round 2 § FP8)

### 7.3 Memory update recommendation

The existing memory says "FP8 broken on this stack; always bf16." For Mistral specifically, refine to:

> **Mistral / Mixtral on GH200, May 2026:**
> - Mistral Large 3 (675B MoE) does not fit single GH200; needs 8×H200 FP8 or 4×B200 NVFP4 multi-GPU single-node.
> - Mistral Medium 3.5 (128B dense) and Mistral Large 2 (123B dense) need AWQ-INT4 + KV-offload to run single GH200; FP8 doesn't fit.
> - Mixtral 8x22B (141B/39B MoE) needs AWQ-INT4 + KV-offload single GH200, or 2×H200 SXM TP=2 with pre-quantized FP8.
> - Mistral Nemo 12B / Small 3.2 24B / Codestral 22B run BF16 native on single GH200.
> - For all MoE Mistral: **only pre-quantized FP8 checkpoints** — never online `--quantization fp8` (vLLM #41022, #30830 both OPEN).
> - W4A16 MoE: TP=2 cap (vLLM #41511).
> - SGLang FP8 + EAGLE + MoE: broken on H200 (sglang #24650). Use vLLM ≥ 0.21 for EAGLE on Mistral Large 3.
> - vLLM 0.21.0 is the recommended pin (FA4-MLA default, HMA KV offload, reasoning-budget-aware spec-decode, mistral_common ≥ 1.8.6 bundled).

---

## 8. Confidence and Gaps

### High confidence
- Mistral SKU list and HF model IDs (verified directly from Mistral docs, HF model cards, NVIDIA NIM build catalogs).
- vLLM open MoE FP8 bugs #41022, #30830, #41511, #39000 statuses (fetched directly from GitHub issues this round).
- sglang #24650 still OPEN as of 2026-05-25 fetch (issue body present, no closing PR linked in visible thread).
- Pre-quantized FP8 Mixtral 8x22B (RedHatAI / neuralmagic) is production-grade (99.54% accuracy recovery per RedHatAI card).
- Mistral-Large-3 (675B/41B MoE) cannot fit single GH200 — all three quantization formats exceed 96 GB.
- Mistral Medium 3.5 = 128B **dense**, current Mistral frontier non-MoE flagship.

### Medium confidence
- The "AWQ-Marlin INT4 single GH200" path for Medium 3.5 specifically — depends on whether `TheBloke/Mistral-Medium-3.5-128B-AWQ` or equivalent community AWQ release exists. If absent, must self-quantize via `llm-compressor`. Worth verifying before committing the recipe.
- Single-GH200 Mixtral 8x22B AWQ-INT4 tok/s estimate (30-70) is interpolated, not measured. Round-3 already flagged Llama-3.3-70B-bf16-GH200-vLLM-V1 as the same gap class.
- vLLM #41511 mentions DeepSeek-V4 and Qwen3 MoE explicitly; does the bug also affect compressed-tensors W4A16 **Mixtral**? Likely yes (same code path) but not specifically asserted in the issue body.

### Gaps
- No published Mixtral 8x22B on single GH200 throughput numbers under any quantization. Highest-leverage in-house benchmark.
- No published Mistral Medium 3.5 single-GH200 (or single-H200) recipe with INT4 + offload. Would need to validate via `llm-compressor` AWQ run + custom test.
- SGLang Mistral vision tool-calling parity with vLLM — undocumented at the level of "does the SGLang mistral parser handle Pixtral / Mistral-Large-3-vision tool calls identically?" Worth a controlled test.
- The Hugging Face NVFP4 Mistral checkpoints (Large-3-NVFP4, Small-4-NVFP4) on Hopper via Marlin-FP4 fallback — **memory benefit confirmed, no throughput benefit over FP8**. No published Hopper-vs-Blackwell apples-to-apples NVFP4 tok/s number for either model.

---

## 9. Sources (URLs, dated 2026-05-25)

### Mistral model lineup & specs
- [Mistral Large 3 announcement](https://mistral.ai/news/mistral-3) — 2025-12-02
- [Mistral Large 3 docs page](https://docs.mistral.ai/models/mistral-large-3-25-12)
- [vLLM Recipes — Mistral-Large-3](https://docs.vllm.ai/projects/recipes/en/latest/Mistral/Mistral-Large-3.html)
- [HF mistralai/Mistral-Large-3-675B-Instruct-2512](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512)
- [HF mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4)
- [HF mistralai/Mistral-Small-4-119B-2603](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) — MoE 128 experts top-4, 6.5B active
- [Mistral Small 4 announcement](https://mistral.ai/news/mistral-small-4)
- [Mistral docs — models overview](https://docs.mistral.ai/getting-started/models/) — current lineup, retirements
- [HF mistralai/Devstral-2-123B-Instruct-2512](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512) — 123B dense
- [HF mistralai/Mistral-Medium-3.5-128B](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B) — 128B dense
- [Lushbinary — Self-Host Mistral Medium 3.5 vLLM/SGLang](https://lushbinary.com/blog/mistral-medium-3-5-self-hosting-vllm-sglang-guide/)
- [Mistral retirement notice — Mixtral 8x22B](https://legal.mistral.ai/ai-governance/models/mixtral-8-22b) — retired 2025-03-30
- [HF mistralai/Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)
- [HF RedHatAI/Mixtral-8x22B-Instruct-v0.1-FP8](https://huggingface.co/RedHatAI/Mixtral-8x22B-Instruct-v0.1-FP8)
- [HF neuralmagic/Mixtral-8x22B-Instruct-v0.1-FP8](https://huggingface.co/neuralmagic/Mixtral-8x22B-Instruct-v0.1-FP8)
- [HF mistralai/Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
- [Mistral Codestral 25.08 announcement](https://mistral.ai/news/codestral-25-08)
- [Mistral NeMo announcement](https://mistral.ai/news/mistral-nemo)

### MoE FP8 fragility — OPEN bugs (verified 2026-05-25)
- [vLLM #41022 — Online FP8 ignores logical_widths on MergedColumnParallelLinear](https://github.com/vllm-project/vllm/issues/41022) — OPEN 2026-04-27
- [vLLM #30830 — MoE online FP8 patched_weight_loader accuracy issue](https://github.com/vllm-project/vllm/issues/30830) — OPEN 2025-12-17
- [vLLM #41511 — compressed-tensors W4A16 MoE TP>2 sharding crash](https://github.com/vllm-project/vllm/issues/41511) — OPEN 2026-05-02
- [vLLM #39000 — Gemma 4 MoE MXFP4 weight loader crash](https://github.com/vllm-project/vllm/issues/39000) — OPEN 2026-04-04
- [vLLM #38022 — Marlin MoE kernel fails with MXFP4 GPT-OSS](https://github.com/vllm-project/vllm/issues/38022)
- [vLLM #38912 — Gemma 4 MoE NVFP4 expert_params_mapping scale key suffixes](https://github.com/vllm-project/vllm/issues/38912)
- [vLLM #38999 — Gemma 4 MoE data-parallel crash](https://github.com/vllm-project/vllm/issues/38999)
- [vLLM #40591 — 0.19.1 regression Gemma 4 MoE packed experts](https://github.com/vllm-project/vllm/issues/40591)
- [sglang #24650 — DeepSeek-V4-Flash-FP8 + EAGLE RoPE crash](https://github.com/sgl-project/sglang/issues/24650) — OPEN 2026-05-08
- [sglang #25704 — DeepSeek-V4-Pro NVFP4 B200 single-node NaN](https://github.com/sgl-project/sglang/issues/25704)
- [sglang #25165 — DeepSeek V4 Flash main branch broken](https://github.com/sgl-project/sglang/issues/25165)
- [vLLM #5535 — Mixtral 8x7B FP8 slow H100](https://github.com/vllm-project/vllm/issues/5535) — CLOSED (PR #5327 fix; historical anchor only)

### Function calling / tool parsers
- [vLLM Tool Calling docs](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [SGLang Tool Parser docs](https://docs.sglang.io/advanced_features/tool_parser.html) — 13 parsers, `mistral` included
- [SGLang Tool & Function Calling docs](https://docs.sglang.ai/advanced_features/function_calling.html)
- [vLLM vs SGLang 2026 morphllm comparison](https://www.morphllm.com/comparisons/vllm-vs-sglang)
- [Red Hat — Mistral Large 3 / Ministral 3 vLLM Day-0 guide](https://developers.redhat.com/articles/2025/12/02/run-mistral-large-3-ministral-3-vllm-red-hat-ai) — 2025-12-02

### Quantization paths and benchmarks
- [vLLM Quantization overview](https://docs.vllm.ai/en/latest/features/quantization/)
- [vLLM INT4 W4A16 docs](https://docs.vllm.ai/en/latest/features/quantization/int4/)
- [vLLM FP8 W8A8 docs](https://docs.vllm.ai/en/latest/features/quantization/fp8/)
- [vLLM GPTQModel docs](https://docs.vllm.ai/en/latest/features/quantization/gptqmodel/)
- [llm-compressor (Red Hat / vLLM)](https://github.com/vllm-project/llm-compressor)
- [Jarvis Labs vLLM quantization guide & benchmarks](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks)
- [Spheron — FP4 quantization on Blackwell](https://www.spheron.network/blog/fp4-quantization-blackwell-gpu-cost/)
- [Marlin nearly-ideal speed for 4-bit models with vLLM](https://kaitchup.substack.com/p/marlin-nearly-ideal-inference-speed)
- ["Give Me BF16 or Give Me Death" accuracy/perf tradeoffs](https://arxiv.org/pdf/2411.02355)

### Hardware & deployment context
- [NVIDIA Developer — Mistral 3 + NVIDIA acceleration](https://developer.nvidia.com/blog/nvidia-accelerated-mistral-3-open-models-deliver-efficiency-accuracy-at-any-scale/)
- [Baseten Llama 3.3 70B GH200 inference perf](https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/)
- [NVIDIA blog — KV cache offload CPU-GPU memory sharing](https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/)
- [Artificial Analysis — Mistral Large 3](https://artificialanalysis.ai/models/mistral-large-3)
- [Mistral Large 3 vs Mistral Large 2 comparison](https://artificialanalysis.ai/models/comparisons/mistral-large-3-vs-mistral-large-2407)
- [Spheron — MoE inference optimization GPU cloud 2026](https://www.spheron.network/blog/moe-inference-optimization-gpu-cloud/)
- [InsiderLLM — Are Mistral Models Still Worth Running?](https://insiderllm.com/guides/mistral-mixtral-guide/) — "Only Nemo 12B" take
