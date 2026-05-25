# Qwen3.6-27B Q4 GGUF on GH200 96GB via llama.cpp — Round 5

**Date:** 2026-05-25
**Hardware:** 1× NVIDIA GH200 96GB (Hopper, SM 9.0), Lambda Ubuntu 24.04 aarch64
**Goal:** Highest tok/s + least friction for Qwen3.6-class ~27B Q4 GGUF via llama.cpp

---

## 0. Model identity — verified

**The user said "Qwen3.6-27B Q4 GGUF". That model exists and is the right target.**

### Qwen 20–32B–class lineup as of May 2026

| SKU | Type | Released | Status |
|---|---|---|---|
| **Qwen3.6-27B** | Dense, hybrid attn (Gated DeltaNet + Gated Attention), VLM | **2026-04-22** | **← This is the one** |
| Qwen3.6-35B-A3B | MoE (35B total / 3B active), VLM | 2026-04-16 | Sibling; smaller active, similar quality |
| Qwen3.6-Max-Preview | Closed-weights flagship | 2026 | API-only |
| Qwen3.6-Flash | Closed-weights, 1M ctx | 2026 | API-only |
| Qwen3-Next-80B-A3B | Hybrid MoE | late 2025 | Predecessor of 3.6 |
| Qwen3.5-397B-A17B | Huge MoE | 2025 | Previous flagship; Qwen3.6-27B beats it on coding |
| Qwen3-32B | Dense | 2025 | Older generation |
| Qwen3-30B-A3B | MoE | 2025 | Older generation |
| Qwen3-Next-A22B | (none with this name confirmed) | — | User probably misremembered Qwen3-Next-80B-A3B |

**There is no "Qwen3.6-32B" or "Qwen3.6-30B".** The dense Qwen3.6 SKU is exactly **27B**, and it's the flagship-quality open-weight model — it beats Qwen3.5-397B-A17B on SWE-bench Verified (77.2 vs 76.2), SWE-bench Pro, Terminal-Bench 2.0, and SkillsBench.

**Recommendation: Qwen3.6-27B is the correct target.** Quality-per-VRAM is best-in-class for ~17 GB Q4.

### Architecture (relevant for llama.cpp support)

- 27B params, 64 layers, hidden 5120, vocab 248,320 (padded)
- **Hybrid attention** — alternates Gated DeltaNet (linear) and Gated Attention blocks, layout `16 × (3 × (DeltaNet → FFN) → 1 × (GatedAttn → FFN))`
- Gated Attn: 24 Q heads, 4 KV heads (GQA 6:1), head_dim 256, partial RoPE (dim 64)
- Native context **262,144** tokens; YaRN-extensible to ~1M
- VLM (vision encoder); for text-only use, just skip image inputs
- **MTP-trained** (Multi-Token Prediction head) — enables self-speculative decoding

llama.cpp needed two PR series to land full support: **#19408** (Gated DeltaNet linear-attn operators, originally for Qwen3-Next) and **#22673** (MTP speculative decoding, merged 2026-05-16). Anything built from main after mid-May 2026 covers both.

---

## 1. Best GGUF source

### Recommended: `unsloth/Qwen3.6-27B-MTP-GGUF`

Why MTP version specifically: it carries the extra MTP head as draft tensors, so you get **self-speculative decoding (~1.4–2.2× speedup)** without loading a separate draft model. Unsloth's "Dynamic 2.0" iMatrix calibration is the current quality leader; their GGUFs match or beat bartowski's at the same bit budget on standard perplexity battery and downstream evals.

| Quant | File size | Notes |
|---|---|---|
| `UD-Q4_K_XL` | **17.9 GB** | Recommended Q4 — Q8_0 embed/output + dynamic per-layer bits |
| `Q4_K_M` | 17.1 GB | Standard Q4 |
| `Q5_K_M` | 19.8 GB | Quality bump, fits easily |
| `Q6_K` | 22.9 GB | Near-perfect; comfortable on 96 GB |
| `Q8_0` | 29.0 GB | Reference-quality |
| `BF16` | 53.8 GB | Full precision; runs on GH200 with room |

### Alternative: `bartowski/Qwen_Qwen3.6-27B-GGUF`

Non-MTP. Same iMatrix tradition, broader quant menu (IQ2/IQ3 ladder). Use **if** you don't need MTP and want the canonical bartowski lineage. Q4_K_M is 17.98 GB; Q4_K_L (Q8 embed/output) is 18.93 GB.

### Avoid

- `DavidAU/*-Heretic-Uncensored-...` — finetune, not base
- `HauhauCS/*-Uncensored-Aggressive` — finetune
- Random fp8 reposts — see Section 8

---

## 2. Memory footprint on GH200 96 GB

For `UD-Q4_K_XL` (17.9 GB weights) on a single GH200 with 96 GB HBM3:

| Component | At 32K ctx | At 128K ctx | At 262K ctx (native max) |
|---|---|---|---|
| Weights | 17.9 GB | 17.9 GB | 17.9 GB |
| KV cache (f16, full attn layers only) | ~1.0 GB | ~4.0 GB | ~8.2 GB |
| KV cache (Gated DeltaNet state, fixed) | ~0.5 GB | ~0.5 GB | ~0.5 GB |
| Compute buffer, CUDA graphs, scratch | ~3 GB | ~4 GB | ~6 GB |
| **Total resident** | **~22 GB** | **~26 GB** | **~33 GB** |

KV is small because only 16 of 64 layers use full attention (the rest are DeltaNet with fixed-size recurrent state). **You have 60+ GB of headroom even at full 262K context** — use it to crank quant (Q6_K or Q8_0) or run draft model alongside.

GH200's unified memory means CUDA can spill to LPDDR5X (480 GB) but you should never need to for this model; keep everything in HBM3.

---

## 3. Exact llama-server commands

### 3a. Max single-stream tok/s (recommended for interactive/coding)

```bash
# Build once
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-cli

# Run — MTP self-speculative, FA, q8_0 KV
./build/bin/llama-server \
  -hf unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL \
  -ngl 999 \
  -c 131072 \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0.75 \
  -np 1 \
  --host 0.0.0.0 --port 8080 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 1.5
```

Key flag rationale:
- `-DCMAKE_CUDA_ARCHITECTURES=90` — Hopper SM 9.0; do NOT use `native` on aarch64 (PTX detection can misfire — issue #8714)
- `-ngl 999` — all 64 layers on GPU; GH200 has the room
- `-fa on` — flash-attn CUDA, required for fast q8_0 V cache
- `-ctk/-ctv q8_0` — see Section 5; quality-neutral on Qwen3.6
- `--spec-type draft-mtp` — uses the built-in MTP head; no separate draft model
- `--spec-draft-n-max 3` — sweet spot per Unsloth/NVIDIA dev forum tests; n=5 only helps at very high acceptance
- `--spec-draft-p-min 0.75` — being *more* selective raises acceptance rate (counterintuitive but verified)
- `-np 1` — MTP is single-slot only as of May 2026
- Sampling: Qwen3.6 thinking-mode defaults from official model card

### 3b. Max throughput (batched serving, no MTP)

```bash
./build/bin/llama-server \
  -hf unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL \
  -ngl 999 \
  -c 65536 \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -b 4096 -ub 1024 \
  -np 8 \
  --host 0.0.0.0 --port 8080
```

`-np 8` is 8 parallel slots × 8K context each (= 65,536 total). MTP disabled — it tanks under concurrency (Dre Dyson + NVIDIA forum data show c=4 with MTP underperforms c=4 baseline). For throughput, **use vLLM or SGLang instead** — Section 9.

### 3c. Full 262K context, single-stream

```bash
./build/bin/llama-server \
  -hf unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL \
  -ngl 999 -c 262144 -fa on \
  -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
  -np 1
```

(Drop to `-ctk q4_0 -ctv q4_0` if you hit a wall — but on 96 GB you won't.)

### Sampling presets

| Task | temp | top-p | top-k | presence-pen |
|---|---|---|---|---|
| Thinking, general | 1.0 | 0.95 | 20 | 1.5 |
| Thinking, coding | 0.6 | 0.95 | 20 | 0.0 |
| Non-thinking (`--chat-template-kwargs '{"enable_thinking":false}'`) | 0.7 | 0.8 | 20 | 1.5 |

---

## 4. Expected tok/s on Hopper

No published GH200-specific number for Qwen3.6-27B + llama.cpp exists yet (May 2026). Extrapolating from comparable hardware:

| Hardware | Config | Reported tok/s | Source |
|---|---|---|---|
| RTX 6000 Ada (48 GB, Ada SM 8.9) | Q4_K_M + MTP | **~160** gen | Unsloth/HF discussion |
| H100 GB10 (DGX Spark) | Q4_K_M + MTP n=5, c1 | 28.3 (baseline 13.1) | NVIDIA Dev Forum |
| RTX 5090 (Blackwell, 32 GB) | Q4_K_M + MTP n=3 | 28.3 (c1) → 29.9 (c4) | Dre Dyson |
| RTX 4090 | Q4_K_M + Qwen3.5-4B draft | 43.2 mean / 67.1 peak | outsourc-e recipes |
| 2× RTX 3090 layer-split | Q8_0 + MTP | 25.7 → 55.9 (2.17× w/ MTP) | community |
| 2× RTX 5060 Ti 16 GB | Q4_K_M, c=64 | 94–133 (vLLM same h/w: 262–377) | LLMKube bakeoff |

**GH200 prediction: 75–110 tok/s single-stream with MTP at Q4, scaling to 140+ on coding patterns with high draft acceptance.** GH200's HBM3 bandwidth (~4 TB/s) is ~2.5× the H100 PCIe and ~6× the RTX 5090, and decode is memory-bandwidth-bound; expect roughly linear scaling vs RTX 5090's 29 tok/s baseline → ~75–90 tok/s baseline + the MTP multiplier. Prompt processing on Hopper is exceptional; expect 4000+ pp tok/s.

Tag the GH200 dev-forum thread is dormant — if Lambda lets you, run `llama-bench` and post results to llama.cpp discussion #18005.

---

## 5. KV cache types — quality vs memory

| `-ctk/-ctv` | Bits | KV size at 262K | Quality on Qwen3.6 |
|---|---|---|---|
| `f16` | 16 | ~16 GB | Reference |
| `bf16` | 16 | ~16 GB | Identical to f16 in practice |
| `q8_0` | 8 | ~8 GB | **Recommended.** PPL delta 0.002–0.05 vs f16. Modern Qwen handles q8_0 KV cleanly |
| `q5_1` | 5 | ~5 GB | Marginal degradation |
| `q4_1` | ~4.5 | ~4.5 GB | Some hits on long-context tasks |
| `q4_0` | 4 | ~4 GB | "Weird results" on older Qwen2; **Qwen3.6 dense handles it fine for ≤32K; degrades on hybrid layers** |
| `turbo3` (TurboQuant) | 3.25 | ~3.3 GB | Out-of-tree fork (spiritbuun/buun-llama-cpp for CUDA); 1–2% PPL vs f16; not needed on GH200 |

**On GH200 96 GB: use `-ctk q8_0 -ctv q8_0`.** Pure quality-neutral and you have memory to burn anyway. The only reason to go lower is to fit much longer contexts than 262K, which Qwen3.6 doesn't natively support without YaRN.

**Note on the outsourc-e recipes finding:** "q8 KV hangs at >32K" was on RTX 4090 — a VRAM/CUDA-graph artifact specific to 24 GB cards. Does not apply to GH200.

Flash attention (`-fa on`) is **required** for fast V cache reads when V is quantized. Don't skip it.

---

## 6. Q4_K_M vs Q5_K_M vs Q6_K — the 96 GB question

You have 96 GB. Q4 is leaving quality on the table.

| Quant | Size | Inference speed vs Q4 | Quality |
|---|---|---|---|
| `UD-Q4_K_XL` | 17.9 GB | 1.00× (baseline) | Standard Q4 + Q8 embed/output |
| `Q5_K_M` | 19.8 GB | ~0.92× | Measurably better on hard coding/math evals |
| `Q6_K` | 22.9 GB | ~0.80× | Near-perfect — PPL delta vs Q8 typically < 0.01 |
| `Q8_0` | 29.0 GB | ~0.65× | Reference-grade |

Memory bandwidth governs decode tok/s. Q6_K is ~28% bigger than Q4_K_XL → ~22% slower. On GH200 baseline ~80–100 tok/s, Q6_K still gives **~65–80 tok/s with MTP** — faster than Q4 on most consumer cards.

**Recommendation:** Default to **`UD-Q6_K_XL`** (25.6 GB) for production. Use `UD-Q4_K_XL` only if you need the absolute peak speed for autocomplete-style latency-critical work. The user's "Q4 GGUF" intent is fine; just point out the headroom.

---

## 7. Speculative decoding partner

**Use MTP — don't use a separate draft model.**

Qwen3.6-27B ships with an integrated MTP head. The Unsloth `Qwen3.6-27B-MTP-GGUF` builds bake that head into the GGUF. llama.cpp's `--spec-type draft-mtp` (PR #22673, merged 2026-05-16) uses it natively. This is strictly better than external draft models for Qwen3.6 because:

1. Same vocab guaranteed — no token-translation corruption (see Section 8)
2. ~92% draft acceptance rate at `--spec-draft-p-min 0.75` (NVIDIA forum data)
3. Zero extra VRAM for the draft model
4. 1.4–2.2× decode speedup vs baseline; ~2.17× on dense

**If you must use an external draft** (e.g., MTP build flaky for you):

- **Best:** `Qwen3.5-4B-Q4_K_M.gguf` — same tokenizer family as Qwen3.6, no translation needed. Verified in outsourc-e recipes at 43.2 tok/s on 4090.
- **OK:** `Qwen3-1.7B-Q4_K_M.gguf` — smaller draft, faster but lower acceptance.
- **Avoid:** `Qwen3-0.6B-Instruct` — too small; the discussion #22473 user actually saw regression (38→33 tok/s) with the 0.8B equivalent.

External draft invocation:
```bash
--model-draft ~/models/Qwen3.5-4B-Q4_K_M.gguf \
--draft-max 8 --draft-min 2 --draft-p-min 0.6 \
-ngld 999
```

**Do not use `ik_llama.cpp` with cross-vocab drafts** — it silently corrupts JSON/list output (outsourc-e A2 config). Stick with mainline llama.cpp + MTP or same-vocab Qwen3.5 draft.

---

## 8. The SGLang #23687 FP8 garbage bug — does it affect GGUF?

**No. It is strictly an SGLang FP8 weight-loader bug. Does not touch GGUF or llama.cpp.**

### What the bug is

In `sgl-project/sglang` issue [#23687](https://github.com/sgl-project/sglang/issues/23687), loading `Qwen/Qwen3.6-27B-FP8` (dense) with the official SGLang command produces token salad. Root cause: SGLang's `qwen3_5.py` loader has an init-order bug — `MergedColumnParallelLinear` doesn't expose `gate_up_proj.weight_scale_inv` as a registered parameter when `Qwen3_5ForConditionalGeneration` is instantiated for the 3.6 dense variant. The FP8 block-scaling factors silently default to 1.0, dequantization runs with no scale, garbage.

### Why it cannot affect llama.cpp / GGUF

GGUF is a completely independent quantization pipeline. The FP8 scale factors live inside the safetensors checkpoint and are consumed only by FP8-aware loaders (SGLang, vLLM with `quantization=fp8`). When you build a GGUF you start from the bf16 master (or quantize from f16/bf16 weights via `convert_hf_to_gguf.py` + `llama-quantize`), and the GGUF file embeds its own per-block scales for K-quants. There is no FP8 path through llama.cpp at all (llama.cpp has no E4M3 kernels). Q4_K_M's correctness is mathematically and code-pathically disjoint from the FP8 loader.

### Verification points

- Bartowski and Unsloth Q4_K_M files were both quantized from bf16 master, not from FP8
- The bug ticket explicitly scopes itself to SGLang loader code; same hardware loads Qwen3.5-27B-FP8 cleanly in the same SGLang
- vLLM has had related FP8 issues for other Qwen variants (vllm#38197) — also irrelevant to GGUF

**Caveat:** If you ever try `vllm serve Qwen/Qwen3.6-27B-FP8 ...` on GH200, you'll hit a different but related class of bugs. Stay on bf16 master + GGUF or bf16 in vLLM. (This matches your stack-level memory: no FP8 on this stack.)

---

## 9. When to drop llama.cpp

llama.cpp is the right pick for **single-stream interactive** workloads on GH200. For **high-concurrency serving** the LLMKube bakeoff shows vLLM beating llama.cpp 2.8–3.7× at c=64:

| Pattern | llama.cpp tok/s (c=64) | vLLM tok/s (c=64) |
|---|---|---|
| chat | 94 | 345 |
| coding | 133 | 377 |
| agentic | 72 | 262 |

For your GH200, options for high-throughput:
- **vLLM** with bf16 weights, `--max-model-len 262144`, no FP8
- **SGLang** with bf16, `--reasoning-parser qwen3`, MTP via `--speculative-algo NEXTN --speculative-num-steps 3` — but pin to a version that has the #23687 fix, or use bf16 (not FP8)

Stay on llama.cpp if: interactive, single-user, low ops complexity, want one binary, want MTP today.

---

## 10. Build gotchas on GH200 aarch64

- Use **CUDA 12.6+**; older toolchains hit `ptxas signal 11` on SM 9.0 (llama.cpp issue #8714).
- `-DCMAKE_CUDA_ARCHITECTURES=90` (not `native`, not `90a` unless you specifically need WMMA).
- `-DGGML_CUDA_FA_ALL_QUANTS=ON` enables flash-attn kernels for all KV quant types (q8_0, q4_0, etc.) — required to actually use `-ctk q8_0 -ctv q8_0` with `-fa on`.
- `huggingface-cli` works on aarch64 for downloading; or use `-hf user/repo:quant` syntax in llama-server (auto-downloads via libcurl).
- Do **not** rely on `cudaMallocManaged` for oversubscription; issue #5026 documents that the cuBLAS path fails. Keep within HBM3 (96 GB) — for Qwen3.6-27B you're nowhere near.
- `ai-dock/llama.cpp-cuda` has prebuilt aarch64 + CUDA tarballs if you don't want to compile; pick the `linux-arm64-cuda12` artifact.

---

## TL;DR for the impatient

```bash
# One-time
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server

# Run
./build/bin/llama-server \
  -hf unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q6_K_XL \
  -ngl 999 -c 131072 -fa on \
  -ctk q8_0 -ctv q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
  -np 1 --host 0.0.0.0 --port 8080
```

Expected: **75–110 tok/s single-stream**, full hybrid attention + MTP self-speculative, 128K context, 25 GB VRAM resident, 70 GB free for whatever else.

If user really wants Q4: swap `UD-Q6_K_XL` → `UD-Q4_K_XL`, expect ~95–140 tok/s.

---

## Sources

- [Qwen/Qwen3.6-27B (HF model card)](https://huggingface.co/Qwen/Qwen3.6-27B)
- [unsloth/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)
- [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF)
- [bartowski/Qwen_Qwen3.6-27B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.6-27B-GGUF)
- [Unsloth docs — Qwen3.6](https://unsloth.ai/docs/models/qwen3.6)
- [Simon Willison — Qwen3.6-27B](https://simonwillison.net/2026/Apr/22/qwen36-27b/)
- [SGLang Issue #23687 — FP8 garbage bug](https://github.com/sgl-project/sglang/issues/23687)
- [llama.cpp Discussion #18005 — GH200 performance](https://github.com/ggml-org/llama.cpp/discussions/18005)
- [llama.cpp Discussion #22473 — Speculative decoding for Qwen3.6-27B](https://github.com/ggml-org/llama.cpp/discussions/22473)
- [llama.cpp Discussion #20969 — TurboQuant KV cache](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [llama.cpp Issue #8714 — ptxas error on Grace Hopper](https://github.com/ggml-org/llama.cpp/issues/8714)
- [llama.cpp Issue #5026 — cudaMallocManaged oversubscription on GH200](https://github.com/ggml-org/llama.cpp/issues/5026)
- [outsourc-e/qwen36-4090-recipes — verified configs and silent-corruption bug](https://github.com/outsourc-e/qwen36-4090-recipes)
- [NVIDIA Dev Forum — MTP+llama.cpp on Qwen3.6-27B](https://forums.developer.nvidia.com/t/mtp-llama-cpp-a-look-at-qwen3-6-27b/370298)
- [Dre Dyson — MTP llama.cpp Qwen3.6-27B beginner guide](https://dredyson.com/mtp-llama-cpp-with-qwen3-6-27b-a-complete-beginners-step-by-step-guide-to-speculative-decoding-turboquant-and-running-multiple-models-on-limited-gpu-vram/)
- [LLMKube — Qwen3.6-27B llama.cpp vs vLLM bakeoff](https://llmkube.com/blog/qwen3-6-27b-bakeoff)
- [smcleod.net — K/V context quantization](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/)
- [Qwen3.6-27B blog (Qwen team)](https://qwen.ai/blog?id=qwen3.6-27b)
