# Round 4 — Qwen Family on GH200 (LLM-only deep dive)

**Date:** 2026-05-25
**Scope:** Single GH200 (96 GB HBM3 / 480 GB Grace LPDDR / NVLink-C2C 900 GB/s)
**Constraint reminder:** bf16 default on this stack — FP8 only where verified clean. SGLang Qwen3.6-27B-FP8 garbage bug (#23687) still open.

---

## 0. TL;DR table — Qwen LLMs as of May 2026

| Model | Release | Total / Active | Native ctx | Single GH200? | Recommended engine + precision |
|---|---|---|---|---|---|
| **Qwen2.5-72B-Instruct** | Sep 2024 (still relevant) | 72B dense | 128K (32K base + YaRN) | **Yes** — bf16 fits HBM with KV in Grace | vLLM 0.21+ / SGLang 0.5.x, **bf16** |
| **Qwen2.5-72B-Instruct-AWQ / GPTQ-Int4 / GPTQ-Int8** | Sep 2024 | 72B → ~37 GB / ~37 GB / ~75 GB | 128K | Yes, comfortably | vLLM, **AWQ-Marlin** (auto) |
| **Qwen2.5-VL-72B-Instruct** | Jan 2025 | 72B dense + vision | 128K | Yes (bf16) | vLLM Qwen2.5-VL recipe, bf16 (FP8-dynamic available from RedHatAI) |
| **Qwen2.5-7B / 14B-Instruct-1M** | Jan 2025 | 7B / 14B dense | **1M** | 7B yes; **14B no** (320 GB total VRAM officially required) | Custom vLLM `dev/dual-chunk-attn` branch + sparse attention |
| **"Qwen3-72B" dense** | **Does not exist.** Qwen3 dense series stops at 32B (0.6B/1.7B/4B/8B/14B/32B). | — | — | — | — |
| **Qwen3-235B-A22B-Instruct-2507 / -FP8** | Jul 2025; DCA July 2025 | 235B / 22B MoE | **256K native, 1M with DCA** | **No** — bf16 needs ~470 GB; FP8 needs ~250 GB. Multi-GPU only. | 8× H200 vLLM/SGLang. On single GH200: not feasible. |
| **Qwen3-VL-235B-A22B-Instruct / -Thinking (-FP8)** | Sep 23, 2025 | 235B / 22B + vision | 256K | **No** (4× H100 INT4 / 8× H100 FP8 official) | 8× H200, vLLM `--mm-encoder-tp-mode data --enable-expert-parallel --async-scheduling` |
| **Qwen3-VL-30B-A3B-Instruct (-FP8)** | Sep 2025 | 30B / 3B MoE + vision | 256K | **Yes** (FP8 ~32 GB; bf16 ~60 GB) | vLLM ≥ 0.11, single GH200 fine |
| **Qwen3-Next-80B-A3B-Instruct / -Thinking (-FP8)** | Sep 22, 2025 | 80B / 3B hybrid MoE (Gated DeltaNet + Gated Attention) | 256K native, 1M YaRN | **bf16 no** (160 GB needs TP=4); **FP8 ~80 GB barely fits 96 GB HBM but officially TP=4 recommended** | vLLM 0.10.2+, **be aware of FlashInfer bugs**, MTP `qwen3_next_mtp` |
| **Qwen3-Coder-480B-A35B-Instruct (-FP8)** | Jul 2025 | 480B / 35B MoE, 160 experts | 256K (1M extrap) | **No** (bf16 ~960 GB, FP8 ~250 GB; 8× H200 official) | 8× H200, vLLM `data-parallel-size 8 --enable-expert-parallel` |
| **Qwen3-Coder-30B-A3B-Instruct** | Jul 2025 | 30B / 3B MoE | 256K | **Yes** (FP8) | vLLM FP8, 600 tok/s at 6 conc on 1× H200 SXM (Millstone) |
| **Qwen3.5-27B / -FP8** | Feb 24, 2026 | 27B dense | 262K (GDN hybrid, 75% Gated DeltaNet layers) | **Yes** (FP8 ~31 GB; bf16 ~54 GB) | vLLM ≥ 0.17 (GDN kernel), FP8 |
| **Qwen3.5-35B-A3B / -FP8** | Feb 24, 2026 | 35B / 3B MoE GDN | 262K | **Yes** | vLLM, FP8 single H100/H200 |
| **Qwen3.5-122B-A10B / -FP8** | Feb 24, 2026 | 122B / 10B MoE GDN | 262K | bf16 doubtful; **FP8 ~120 GB fits with Grace KV** | vLLM, FP8, TP=2 safer on dual GH200 |
| **Qwen3.5-397B-A17B / -FP8 / -NVFP4** | Feb 16, 2026 | 397B / 17B MoE | 262K | **No** — needs 4× H100 INT4 / 8× H100 FP8 / 4× GB200 NVFP4 | data-parallel 8 + expert-parallel |
| **Qwen3.6-27B / -FP8** | Apr 22, 2026 (DENSE, with vision) | 27B dense | 262K | **Yes** (FP8 ~28 GB on H200 / GH200) | vLLM ≥ 0.19. **SGLang #23687 BROKEN.** |
| **Qwen3.6-35B-A3B / -FP8** | Apr 16, 2026 | 35B / 3B MoE GDN | 262K | **Yes** (FP8 ~37 GB) | vLLM single GH200, FP8, `--max-num-seqs 512` workaround |
| **Qwen3.5-Plus-20260420 (API)** | API only | Closed | 1M | n/a | OpenRouter / DashScope only |
| **"Qwen3.5-Turbo"** | **Does not exist** as an open-weight release; codersera lists Qwen3.5/3.6/3.7 timeline with no Turbo SKU. | — | — | — | — |

> Headline:
> - **Best single-GH200 LLM in the family today is Qwen3.6-27B-FP8** (vLLM) or **Qwen3.6-35B-A3B-FP8** (vLLM). Both fit in HBM with room for 256K context KV, both have working vLLM paths, both have an active SGLang FP8 bug (#23687 for the dense 27B specifically).
> - **For long-context (≥256K)**: Qwen2.5-7B-Instruct-1M is the only "true" 1M single-GH200 model with the published recipe; **there is NO Qwen2.5-72B-Instruct-1M**. The 72B 1M context is a community myth.
> - **For 70B-class dense bf16**: use Qwen2.5-72B-Instruct. Qwen3 has no 72B dense.

---

## 1. Latest Qwen LLM releases through May 2026

Confirmed release chronology (LLM only — VL/code/omni listed separately):

```
2024-09  Qwen2.5  family (0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B dense)
2025-01  Qwen2.5-{7B,14B}-Instruct-1M, Qwen2.5-VL-{3B,7B,72B}
2025-04  Qwen3 dense (0.6B/1.7B/4B/8B/14B/32B), MoE (30B-A3B, 235B-A22B), thinking/non-thinking unified
2025-07  Qwen3-235B-A22B-Instruct-2507 (256K native, 1M DCA), Qwen3-30B-A3B-2507, Qwen3-Coder-{480B-A35B, 30B-A3B}
2025-09-22  Qwen3-Next-80B-A3B-{Instruct,Thinking} + FP8 builds (hybrid GDN + Gated Attention + ultra-sparse MoE 512 experts)
2025-09-23  Qwen3-VL-{2B,4B,8B,32B} dense + {30B-A3B,235B-A22B} MoE with Instruct + Thinking + FP8 variants
2026-02-16  Qwen3.5-397B-A17B (flagship MoE with 75% GDN layers, 262K native)
2026-02-24  Qwen3.5-122B-A10B, Qwen3.5-35B-A3B, Qwen3.5-27B (dense)
2026-03-30  Qwen3.6 Plus API only, 1M native context
2026-04-16  Qwen3.6-35B-A3B (MoE, open weights + FP8)
2026-04-22  Qwen3.6-27B (dense + native vision) — beats Qwen3.5-397B-A17B on SWE-bench Verified (77.2 vs 76.2)
2026-04-20  Qwen3.5-Plus-20260420 (API on OpenRouter)
```

Distinctive features per family:

- **Qwen2.5**: classic dense Transformer, GQA, RoPE+YaRN. The reliable workhorse.
- **Qwen3**: unifies thinking + non-thinking modes; adds QK-Norm; DCA in the -2507 refreshes pushes to 1M tokens with ~3× speedup vs vanilla attention.
- **Qwen3-Next**: first Qwen with **Gated DeltaNet linear attention** interleaved with regular Gated Attention (3 GDN : 1 GA per block × 12 blocks), plus 512-expert MoE (10 routed + 1 shared). RULER 91.8% at 1M tokens via YaRN. **head_dim = 256** on the Gated Attention layers — relevant to FA3 FP8 register-pressure regression (see §2).
- **Qwen3.5 / Qwen3.6**: 75% of attention layers replaced by **GDN (Gated DeltaNet)**. Native 262K. vLLM ≥ 0.17 added GDN kernels.
- **Qwen3-VL**: Interleaved-MRoPE, DeepStack feature fusion, Text-Timestamp Alignment for video, native 256K interleaved text/image/video.
- **Qwen3-Coder**: 480B / 35B MoE (160 experts, 8 active) + 30B / 3B (128 experts, 8 active). Aggressive agentic-coding training.

---

## 2. Recommended engine + precision per Qwen on single GH200

### 2.1 Decision matrix (single GH200, May 2026)

| Workload | Recommended | Command sketch | Why |
|---|---|---|---|
| **70B-class dense, bf16, reliability-first** | Qwen2.5-72B-Instruct on **vLLM 0.21 bf16** | `vllm serve Qwen/Qwen2.5-72B-Instruct --max-model-len 32768 --enable-prefix-caching` (use 65536 with Grace KV offload) | 72B bf16 = 144 GB; doesn't fit 96 GB HBM. Either AWQ-Int4 or use `--kv-offloading-size 320` to push KV to Grace LPDDR. |
| **70B-class dense, single-GH200 HBM only** | Qwen2.5-72B-Instruct-**AWQ** (Marlin auto-kernel) | `vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --max-model-len 32768` | Fits 96 GB with room for KV. AWQ-Marlin path in vLLM handles 72B Qwen since 2024. |
| **70B-class dense, FP8** | None native. **Qwen2.5 has no official FP8-72B; RedHatAI ships VL-72B-FP8-dynamic.** | n/a for LLM-only | Use AWQ-Int4 or bf16+offload instead. |
| **Best single-GH200 quality, FP8** | **Qwen3.6-27B-FP8** dense (vLLM) | `vllm serve Qwen/Qwen3.6-27B-FP8 --max-model-len 262144 --reasoning-parser qwen3` | ~28 GB weights + ~256K KV easily fits HBM. Outperforms Qwen3.5-397B on SWE-bench Verified per vLLM recipes & buildfastwithai. **Avoid SGLang for this checkpoint** (issue #23687, open, unfixed, May 2026). |
| **Best single-GH200 MoE, FP8** | **Qwen3.6-35B-A3B-FP8** | `vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --max-model-len 262144 --reasoning-parser qwen3 --max-num-seqs 512` | ~37 GB FP8 + Mamba/GDN caches. Note `--max-num-seqs 512` works around the GDN-block cache vs vLLM default 1024 mismatch. |
| **80B hybrid, single-GH200, last resort** | Qwen3-Next-80B-A3B-Instruct-FP8 (recipe says TP=4) | `vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 --tensor-parallel-size 1 --enable-prefix-caching` (off-recipe) | FP8 ~80 GB just barely fits 96 GB HBM in isolation, but **vLLM recipe demands TP=4**. On single GH200 you will be one OOM away from death. Bugs open: #25378 startup fail, #31990 illegal memory on H200, #25760 FlashInfer + MTP fail, #26393 v0.11.0 fail, #34892 FP8 accuracy degradation with FlashInfer CUTLASS MoE. **Recommendation: do not run on single GH200 in May 2026.** Wait for 0.22. |
| **VL on single GH200, FP8** | **Qwen3-VL-30B-A3B-Instruct-FP8** | `vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 --gpu-memory-utilization 0.95 --max-num-batched-tokens 8192 --max-num-seqs 256` | ~3000 tok/s prompt, ~170 tok/s generation on H200 (HF discussion). vLLM ≥ 0.11. |
| **Coding agent, FP8, single H200/GH200** | **Qwen3-Coder-30B-A3B-Instruct-FP8** | per Millstone recipe (vLLM FP8) | Best published single-H200 SXM benchmark: **599.6 tok/s peak system @ 6 concurrent, 1K ctx; 79.9 tok/s/user @ 256K; TTFT 76 ms @ 1K** — see §3. |
| **Long context (256K-1M) MoE, single-GH200** | None of the 235B/397B/480B fits single GPU. **The only "single-GH200 + 1M" model is Qwen2.5-7B-Instruct-1M** via custom vLLM `dev/dual-chunk-attn` branch. | `git clone -b dev/dual-chunk-attn https://github.com/QwenLM/vllm` | 7B + 1M needs 120 GB total — borderline with Grace KV offload. Note `VLLM_ATTENTION_BACKEND=DUAL_CHUNK_FLASH_ATTN VLLM_USE_V1=0`. |
| **Long context, dense, single GH200, hassle-free** | **Qwen3.6-27B-FP8 @ 262K** | `vllm serve Qwen/Qwen3.6-27B-FP8 --max-model-len 262144` | Native, no DCA, no V0 fallback. |

### 2.2 Precision recommendations

- **bf16 default** for all dense ≤ 32B that fit HBM and for Qwen2.5-72B with AWQ alternative.
- **FP8 (vLLM 0.21+)** acceptable for: Qwen3.6-27B-FP8, Qwen3.6-35B-A3B-FP8, Qwen3.5-27B-FP8, Qwen3.5-35B-A3B-FP8, Qwen3-VL-30B-A3B-Instruct-FP8, Qwen3-Coder-30B-A3B-Instruct-FP8. **These have working vLLM paths.**
- **FP8 risky**: Qwen3-Next-80B-A3B-FP8 — six open vLLM/FlashInfer bugs. Use bf16 multi-GPU instead.
- **FP8 broken in SGLang**: Qwen3.6-27B-FP8 dense (#23687, opened 2026-04-25, still open May 2026). The bug is a string-mutation in `qwen3_5.py`'s `load_weights` that loses `weight_scale_inv` → dequantize-with-1.0 → token salad. Workarounds (triton backend, drop fp8_e4m3 KV cache, GA upgrade, TP adjust) all confirmed failed. **Use vLLM, not SGLang, for Qwen3.6-27B-FP8.**
- **AWQ-Marlin** is mature for Qwen2.5-72B and Qwen2.5-VL-72B (multiple HF AWQ checkpoints; vLLM auto-detects awq_marlin backend). For Qwen3.x family the official recommendation is the FP8 checkpoint, but QuantTrio publishes AWQ for Qwen3.6-27B.
- **GPTQ Int4/Int8** for Qwen2.5-72B: official `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4` and `-Int8` exist. For Qwen3 the focus shifted to FP8 + AWQ; GPTQ is older.

### 2.3 head_dim caveats relevant to Qwen

From Round 3 verification, vLLM 0.19.1 + FA3 fixed the FP8 prefill regression for `head_dim ∈ {64, 128}` but **NOT for head_dim = 256**. Qwen-specific implications:

- **Qwen2.5, Qwen3 dense, Qwen3.5/3.6 dense**: head_dim = 128 → ✅ FP8 prefill competitive.
- **Qwen3-Next Gated Attention layers: head_dim = 256** → ⚠️ ~1.6× prefill penalty under FP8. Combined with the FlashInfer-MoE bug list, **stay bf16 for Qwen3-Next**.

---

## 3. Published throughput numbers (Hopper) — what's actually citable

Single most useful number for sizing:

### Qwen3-Coder-30B-A3B-Instruct-FP8 on **1× H200 SXM 141GB**, vLLM (Millstone AI, 2026-02-03)

| Context | Concurrency | Output tok/s (system) | TTFT |
|---:|---:|---:|---:|
| 1K | 1 | 164.9 (per-user) | 76 ms |
| 1K | 6 | **599.6** | — |
| 8K | 1 | — | Peak prefill **45,004 tok/s** |
| 32K | 6 | 238.4 | — |
| 32K | 15 | — | ~8.1 s |
| 128K | 6 | 42.3 | — |
| 256K | 1 | 79.9 (per-user) | 47.6 s |
| 256K | 4 | — | Queue wait 67.4 s |

Capacity-by-use-case extrapolation from Millstone (1× H200 SXM, FP8):
- Code completion 1K ctx → ~42 concurrent
- Short-form chatbot 8K → 125+ concurrent
- Mid chatbot 32K → ~15 concurrent
- Long doc 64K → ~5 concurrent
- Coding assistant 96K → ~2 concurrent

This is the **most reliable single-Hopper Qwen number on public record** as of May 2026.

### Other concrete numbers found

- **Qwen3-VL-30B-A3B-Instruct-FP8 on H200, vLLM** (community HF discussion): ~3,000 tok/s prompt throughput, ~170 tok/s generation throughput.
- **Qwen3.6-35B-A3B-FP8 on Blackwell desktop** (allenkuo medium, Apr 2026): 208 tok/s decode (vLLM beats Ollama).
- **Qwen3.5-27B on B200 GKE** (Google Cloud blog): aggregate cluster headline "1M tok/s" — cluster-level, not per-GPU; for sizing assume B200 ≈ 1.5–1.8× H200 per the consensus from Round 2.
- **Llama-3.1-70B on GH200 vs H100 SXM** (Lambda, BF16+CPU offload): 4.33 vs 0.57 tok/s **at the same single-instance no-batching configuration with KV offload** — quoted 7.6× throughput and 8× cost/token. (This is the BF16 closest analog to running Qwen2.5-72B on GH200 — same regime: weights + offloaded KV.)
- **Qwen2.5-72B on 4× A6000 vLLM** (DatabaseMart): 449.51 tok/s aggregate. Not Hopper, not GH200, but useful upper-bound calibration: a single GH200 should beat 4× A6000 on this workload.
- **Qwen3-235B-A22B-FP8 on 8× H100** (vLLM ascend docs): request throughput 0.23 req/s, output 232.83 tok/s. Note **issue #25540: Qwen3-235B-A22B-Instruct-2507-FP8 is slower than bf16** on vLLM as of late 2025 — confirms the GPUstack observation. Implication: if you have 8 GPUs, you may be better off bf16 than FP8 for this model.
- **Qwen3-Coder-480B-A35B-Instruct-FP8 on 8× H200** (vLLM recipe published numbers): output 131.88 tok/s, total throughput 394.81 tok/s, mean TTFT 7.6 s — using `data-parallel-size 8 --enable-expert-parallel` (tensor parallelism failed, switch to DP).

**Gap that persists from Round 1+3:** No clean published Qwen2.5-72B or Qwen3.6-27B-FP8 single-GH200 tok/s number. The Millstone H200-SXM number is the closest proxy.

---

## 4. Quantization recipes — AWQ-Marlin & GPTQ for Qwen

### 4.1 AWQ availability on HuggingFace (verified)

Official `Qwen/...` checkpoints:
- `Qwen/Qwen2-72B-Instruct-AWQ`
- `Qwen/Qwen2.5-72B-Instruct-AWQ`
- `Qwen/Qwen2.5-VL-72B-Instruct-AWQ` (and 3B, 7B sizes)
- `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`

Community AWQ checkpoints worth pinning:
- `QuantTrio/Qwen3.6-27B-AWQ` — 4-bit AWQ for Qwen3.6-27B dense. Single-GH200 useful.
- `hec-ovi/vllm-awq4-qwen` GitHub demonstrates Qwen3.6-27B AWQ-INT4 at 24.8 tok/s single-stream on AMD Strix Halo iGPU — proves the AWQ-Int4 path works for Qwen3.6-27B across stacks.

### 4.2 GPTQ on HuggingFace (verified)

Official `Qwen/...`:
- `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4` ✓
- `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8` ✓
- `Qwen/Qwen2-72B-Instruct-GPTQ-Int8` ✓
- All Qwen2.5 dense sizes have both Int4 and Int8 GPTQ.

Community 2-bit AutoRound-GPTQ:
- `kaitchup/Qwen2.5-72B-Instruct-AutoRound-GPTQ-2bit` and `-4bit` — useful when KV cache is the binding constraint and you need every HBM byte for context.

For Qwen3-series: official recommendation switched to FP8 checkpoints. Community GPTQ ports exist but aren't first-class.

### 4.3 AWQ-Marlin maturity in vLLM (May 2026)

- vLLM auto-detects AWQ checkpoints and routes to **awq_marlin** kernel — Marlin is the high-performance AWQ path optimized for large batches. Hopper-supported since vLLM 0.5+.
- For Qwen2.5-72B-Instruct-AWQ on a single GH200: this is the recommended path if you want to keep all 96 GB HBM available with maximum KV cache. **Don't use vanilla AWQ kernel — Marlin is ~2-3× faster at high batch.**
- **AWQ-Marlin caveat**: requires the model checkpoint to be packed in the AWQ format Marlin understands (group_size 128, sym, zero_point). Official Qwen AWQ checkpoints are compatible.
- No known AWQ-Marlin regressions for Qwen2.5 family in 2026.

### 4.4 GPTQ-Marlin

Same story — vLLM routes GPTQ checkpoints to a Marlin kernel. The Int4 path works; Int8 GPTQ is slower per-token because Marlin Int8 wasn't optimized as aggressively. **For single GH200 with Qwen2.5-72B, prefer AWQ-Int4-Marlin over GPTQ-Int8 unless you specifically need GPTQ's grouping.**

---

## 5. Long context: Qwen2.5-72B-Instruct-1M on single GH200

**Short answer: no such model. The 1M variant of Qwen2.5 stops at 14B.**

Officially released `-1M` checkpoints:
- `Qwen/Qwen2.5-7B-Instruct-1M` (HuggingFace, public)
- `Qwen/Qwen2.5-14B-Instruct-1M` (HuggingFace, public)

There is **no** `Qwen/Qwen2.5-72B-Instruct-1M`. Anyone asking for it is conflating the 72B base with the 1M long-context extrapolation. The closest models to 1M-at-large-scale are:

- **Qwen3-235B-A22B-Instruct-2507** with DCA → 1M (requires 8 GPUs + ~1000 GB).
- **Qwen3-Next-80B-A3B-Thinking** with YaRN factor=4.0 → 1M (4× H200 official; 91.8% RULER@1M).
- **Qwen3-Coder-480B-A35B-Instruct** with extrapolation → 1M (8× H200).
- **Qwen3.6 Plus API** — 1M native, but closed-weights.

### 5.1 Does Qwen2.5-14B-Instruct-1M run on single GH200 with Grace KV offload?

Qwen team explicitly states **14B-1M needs ≥ 320 GB total VRAM** (multi-GPU). On single GH200 with the 480 GB Grace LPDDR exposed via `--kv-offloading-size`:

- 14B bf16 weights = 28 GB → fits HBM easily.
- KV at 1M tokens for 14B with GQA (8 KV heads, 128 head_dim, 80 layers ≈ Qwen2.5-14B config) ≈ ~250-300 GB **before** Qwen's sparse-attention/MInference reduction.
- With Qwen team's published 96.7% activation-memory reduction via chunked prefill + sparse attention, working-set could drop into the 30-50 GB range during prefill — within HBM if you forgo concurrency.

**Practical answer:** Qwen2.5-14B-Instruct-1M on single GH200 is **theoretically feasible** with Grace LPDDR as KV L2 — but **no published recipe exists**. The Qwen blog requires their custom `dev/dual-chunk-attn` vLLM branch, and that branch's KV offload path predates the vLLM 0.21 HMA rewrite. **You would have to merge the dual-chunk-attn branch against current main and validate** — concrete engineering work, not a published recipe.

For Qwen2.5-7B-Instruct-1M on single GH200: this **is** feasible. 7B bf16 = 14 GB; 1M KV with sparse attention is well under 96 GB. Use the Qwen custom-vLLM branch + `VLLM_ATTENTION_BACKEND=DUAL_CHUNK_FLASH_ATTN VLLM_USE_V1=0`.

### 5.2 Long-context alternative on single GH200 that actually works today

For 256K-class context on a single GH200, the clean, recipe-published path is:

```bash
vllm serve Qwen/Qwen3.6-27B-FP8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

Or:

```bash
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --max-num-seqs 512
```

Both ship in `recipes.vllm.ai` with explicit single-GPU H200 support and were the headline GH200-class single-GPU long-context models in Q2 2026.

---

## 6. Open bugs to track (Qwen-specific, May 2026)

| Engine | Issue | Affected | Status | Impact |
|---|---|---|---|---|
| SGLang | #23687 | Qwen3.6-27B-FP8 dense | **Open** since 2026-04-25, no PR | Garbage output. **Use vLLM instead.** |
| SGLang | #18550 | `fa3 + fp8_e5m2` KV cache | Open | Silent triton fallback (slower). |
| SGLang | #24650 | DeepSeek-V4-Flash-FP8 + EAGLE RoPE shape mismatch | Open | Not Qwen but adjacent. |
| vLLM | #25378 | Qwen3-Next-80B-A3B-Instruct-FP8 fails to start | EngineCore startup error w/ FlashInfer |
| vLLM | #31990 | [H200] Qwen3-Next-80B-A3B-Instruct-FP8 TP1 DP4 EP4 | CUDA illegal memory access |
| vLLM | #25760 | Qwen3-Next-80B-A3B-Thinking-FP8 + FlashInfer + qwen3_next_mtp spec decode | Open |
| vLLM | #25811 | Qwen3-Next-80B-A3B-Instruct on B200 + FlashInfer | Open |
| vLLM | #26393 | v0.11.0 + Qwen3-Next-80B-A3B-Instruct-FP8 | Open |
| vLLM | #33833 | Qwen3-Next + Docker 0.15+ + VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=1 | Compilation errors |
| vLLM | #34892 | Qwen3.5 FP8 accuracy degradation with FlashInfer CUTLASS MoE | Open |
| vLLM | #40124 | TurboQuant + Hybrid MoE (Qwen3.6-35B-A3B) broken on Ampere SM 80-86 | 13 patches submitted |
| vLLM | #25540 | Qwen3-235B-A22B-Instruct-2507-FP8 slower than bf16 | Performance regression |
| vLLM | #26720 | Qwen3-235B FP8 CUDA illegal memory access in 0.11.0 | Open |
| HF | discussion #35 on Qwen3-Next-80B-A3B-Instruct | FlashInfer 0.4 + nonsense | Workaround = Flash_attn backend, loses FP8 KV cache |

**Master rule for Qwen3-Next on Hopper FP8 in May 2026: don't.** Wait for vLLM 0.22+ or use bf16 multi-GPU.

---

## 7. GH200-specific configuration cheatsheet (Qwen)

```bash
# Best-default single-GH200 LLM-only Qwen serve, May 2026:
vllm serve Qwen/Qwen3.6-27B-FP8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.92

# With KV offload to Grace LPDDR (still cap at 400 GB for safety per Round 3):
vllm serve Qwen/Qwen2.5-72B-Instruct \
  --max-model-len 65536 \
  --enable-prefix-caching \
  --kv-offloading-backend native \
  --kv-offloading-size 320

# Long-context, smaller model:
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --max-model-len 262144 \
  --max-num-seqs 512 \
  --reasoning-parser qwen3

# AWQ-Marlin 72B on single GH200:
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
  --max-model-len 32768 \
  --enable-prefix-caching

# Coding agent (best single-H200/GH200 published bench):
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --max-model-len 262144 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Avoid on single GH200: Qwen3-Next-80B (six open bugs), Qwen3-VL-235B (won't fit), Qwen3-235B-A22B (won't fit), Qwen3-Coder-480B (won't fit), Qwen3.5-397B (won't fit).

Avoid on SGLang: Qwen3.6-27B-FP8 dense (#23687).

---

## 8. Open items / next-round follow-ups

1. **Run our own single-GH200 Qwen3.6-27B-FP8 tok/s benchmark** — no public number exists. Millstone H200 SXM is the closest proxy.
2. **Validate Qwen2.5-14B-Instruct-1M on single GH200** with Grace LPDDR KV offload — currently theoretical, no recipe. Would be a novel result.
3. **Watch for SGLang #23687 fix** — if/when it lands, re-enable SGLang Qwen3.6-FP8 path.
4. **Watch vLLM 0.22** for Qwen3-Next FP8 stability fixes — currently 6+ open bugs make single-GPU FP8 deployment unsafe.
5. **Investigate AWQ-Marlin on Qwen3.6-27B** via `QuantTrio/Qwen3.6-27B-AWQ` — if AWQ-Int4 quality matches FP8, you double available KV.

---

## Sources

Qwen blog & papers:
- [Qwen2.5-1M deployment blog (Qwen team)](https://qwenlm.github.io/blog/qwen2.5-1m/)
- [Qwen3-Coder blog](https://qwenlm.github.io/blog/qwen3-coder/)
- [Qwen3-VL Technical Report (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/pdf/2505.09388)
- [Qwen2.5 Technical Report (arXiv:2412.15115)](https://arxiv.org/pdf/2412.15115)
- [Qwen2.5-1M Technical Report (arXiv:2501.15383)](https://arxiv.org/pdf/2501.15383)
- [Qwen3-VL-235B-A22B-Thinking blog](https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef)
- [Qwen3.6-35B-A3B blog](https://qwen.ai/blog?id=qwen3.6-35b-a3b)
- [Qwen team on X — DCA 1M tokens](https://x.com/Alibaba_Qwen/status/1953760230141309354)

HuggingFace model cards:
- [Qwen/Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)
- [Qwen/Qwen3-235B-A22B-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8)
- [Qwen/Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking)
- [Qwen/Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)
- [Qwen/Qwen3-VL-30B-A3B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)
- [Qwen/Qwen3-Coder-480B-A35B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct)
- [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen/Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
- [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
- [Qwen/Qwen2.5-72B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ)
- [Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8)
- [Qwen/Qwen2.5-7B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M)
- [RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic)
- [QuantTrio/Qwen3.6-27B-AWQ](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ)

vLLM recipes & docs:
- [vLLM recipe: Qwen3.6-27B](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
- [vLLM recipe: Qwen3.6-35B-A3B](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B)
- [vLLM Qwen3.5 & Qwen3.6 usage guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [vLLM Qwen3-Next usage guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-Next.html)
- [vLLM Qwen3-VL usage guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- [vLLM Qwen3-Coder-480B usage guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-Coder-480B-A35B.html)

Benchmarks:
- [Millstone AI — Qwen3-Coder-30B-A3B-Instruct FP8 on 1× H200 SXM](https://www.millstoneai.com/inference-benchmark/qwen3-coder-30b-a3b-instruct-fp8-1x-h200-sxm)
- [Lambda — GH200 inference economics, Llama-3.1-70B BF16](https://lambda.ai/blog/putting-the-nvidia-gh200-grace-hopper-superchip-to-good-use-superior-inference-performance-and-economics)
- [Google Cloud — 1M tok/s Qwen3.5-27B on B200 GKE](https://medium.com/google-cloud/1-million-tokens-per-second-qwen-3-5-27b-on-gke-with-b200-gpus-161da5c1b592)
- [Qwen2.5 official speed benchmark](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html)
- [DatabaseMart 4×A6000 Qwen2.5-72B vLLM](https://www.databasemart.com/blog/vllm-gpu-benchmark-a6000-4)
- [GPUStack — Qwen3-235B-A22B on H100](https://docs.gpustack.ai/latest/performance-lab/qwen3-235b-a22b/h100/)
- [allenkuo medium — Qwen3.6-35B-A3B on Desktop Blackwell](https://allenkuo.medium.com/qwen3-6-35b-a3b-on-desktop-blackwell-the-first-time-vllm-beats-ollama-on-decode-f139f445f926)

Bugs & issues:
- [SGLang #23687 — Qwen3.6-27B-FP8 garbage output (OPEN)](https://github.com/sgl-project/sglang/issues/23687)
- [SGLang #18550 — fa3 + fp8_e5m2 silent triton fallback](https://github.com/sgl-project/sglang/issues/18550)
- [vLLM #25378 — Qwen3-Next-80B-A3B-Instruct-FP8 fails starting](https://github.com/vllm-project/vllm/issues/25378)
- [vLLM #31990 — [H200] Qwen3-Next-80B-A3B-Instruct-FP8 illegal memory](https://github.com/vllm-project/vllm/issues/31990)
- [vLLM #25760 — Qwen3-Next-Thinking-FP8 + flashinfer + mtp](https://github.com/vllm-project/vllm/issues/25760)
- [vLLM #25540 — Qwen3-235B-A22B-FP8 slower than bf16](https://github.com/vllm-project/vllm/issues/25540)
- [vLLM #34892 — Qwen3.5 FP8 accuracy degradation with FlashInfer CUTLASS MoE](https://github.com/vllm-project/vllm/issues/34892)
- [vLLM #40124 — TurboQuant + Qwen3.6-35B-A3B broken on Ampere](https://github.com/vllm-project/vllm/issues/40124)
- [vLLM PR #6139 — DualChunkAttention for Qwen2](https://github.com/vllm-project/vllm/pull/6139)

Third-party deployment guides:
- [Spheron — Deploy Qwen3.5 on GPU Cloud](https://www.spheron.network/blog/deploy-qwen-3-5-gpu-cloud/)
- [Spheron — Deploy Qwen3 GPU Cloud](https://www.spheron.network/blog/deploy-qwen3-gpu-cloud/)
- [codersera — Qwen 3.7 release & status](https://codersera.com/blog/qwen-3-7-release-date-whats-new-2026/)
- [buildfastwithai — Qwen3.6-27B review](https://www.buildfastwithai.com/blogs/qwen3-6-27b-review-2026)
- [SiliconFlow — Best Qwen3 models 2026](https://www.siliconflow.com/articles/en/the-best-qwen3-models-in-2025)
- [hec-ovi/vllm-awq4-qwen — AWQ-Int4 Qwen3.6-27B on Strix Halo](https://github.com/hec-ovi/vllm-awq4-qwen)
