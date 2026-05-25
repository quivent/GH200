# R5 #8 — Qwen3.6-27B 4-bit on SGLang 0.5.12 (GH200 96GB / aarch64)

**Date:** 2026-05-25  
**Target:** 1× NVIDIA GH200 96GB on Lambda Ubuntu 24.04 aarch64  
**Engine pin:** SGLang 0.5.12 (with sgl-kernel ≥ 0.3.21 for aarch64 wheel)  
**Model:** Qwen3.6-27B, 4-bit weight-only quantization  
**Hard rule:** Do not use FP8 — see SGLang #23687.

---

## TL;DR (read this first)

1. **Right SKU:** `Qwen/Qwen3.6-27B` (dense 27B, hybrid Gated DeltaNet + Gated Attention, 262K native context, MTP head trained-in). The base/AWQ checkpoints fit a 96GB GH200 comfortably; FP8 of the *same SKU* is the one that's broken in SGLang.
2. **Preferred 4-bit format on SGLang for Qwen3.6 dense:** **AWQ (group-128) via the `awq_marlin` path**. GPTQ-Pro-4bit also works through `gptq_marlin`, but the broader community and the Qwen Hub ecosystem ship AWQ first and SGLang's Marlin AWQ path is the better-tested fast path for *dense* Qwen3.6. Avoid `moe_wna16` — that's the MoE fallback and does not apply to the dense 27B.
3. **FP8 ban scope (SGLang #23687):** The bug is in `qwen3_5.py::load_weights` — `gate_proj.weight_scale_inv` gets repeatedly string-mutated into `gate_gate_up_proj.weight_scale_inv`, so FP8 scales never load and dequant runs with scale=1.0 → token salad. This is **FP8-block-128 only**; AWQ and GPTQ paths go through a completely different loader (`AWQMarlinLinearMethod` / `GPTQMarlinLinearMethod`) and `weight_scale_inv` is not involved. The issue thread contains zero reports of AWQ/GPTQ regressions on Qwen3.6-27B. 4-bit AWQ/GPTQ are **safe**.
4. **Expected single-stream throughput on GH200 96GB (BS=1, AWQ-INT4, MTP on):** ~80–110 tok/s with EAGLE/NEXTN speculative decoding at avg accept length 2.5–3.5. BS=1 BF16 baseline on same hardware would be ~35–45 tok/s, so 4-bit + spec ≈ **2.5–3× over BF16**.
5. **vs vLLM 0.19:** On Qwen3.6-27B 4-bit, **vLLM currently wins single-stream latency** (Marlin-Pro + qwen3_next_mtp spec is more mature). **SGLang wins on multi-turn / shared-prefix workloads** (RadixAttention / MambaRadixCache). For "tok/s + least friction" on GH200, SGLang is preferable when (a) you run many requests with overlapping system prompts, or (b) you want EAGLE-3 quality out of the box. Pick vLLM if you only ever issue independent single-shot prompts.
6. **JIT compile time on first aarch64 launch:** Plan for **3–8 minutes** of FlashInfer JIT on the first run (MTP/EAGLE adds ~1–2 minutes more). Subsequent launches: ~20–40 s as kernels are cached under `~/.cache/flashinfer`. Pre-warm by running one prefill before benchmarking.

---

## 1. SKU identification (cross-ref with R5 #6/#7)

Qwen3.6-27B (released 2026-04-22, Alibaba) is the canonical "Qwen3.6 27B" target. Key facts from the model card:

| Property | Value |
|---|---|
| Type | Dense, decoder-only, multimodal (text+vision+video) |
| Total params | 27B |
| Layers | 64 |
| Hidden | 5120 |
| FFN intermediate | 17,408 |
| Attention layout | 16 × (3× Gated-DeltaNet→FFN + 1× Gated-Attn→FFN) — the Qwen3-Next hybrid pattern, 3:1 linear:full |
| Gated-DeltaNet | 48 V heads / 16 QK heads, head dim 128 |
| Gated-Attention | 24 Q / 4 KV (GQA), head dim 256, partial RoPE (factor 0.25, dim 64) |
| Native context | 262,144 |
| YaRN-extended | 1,010,000 |
| MTP | Trained-in multi-step head (use for speculative decoding) |
| Vocab (padded) | 248,320 |

This is structurally a Qwen3-Next architecture marketed as Qwen3.6, which is why MambaRadixCache + EAGLE/NEXTN matter (see §3, §6). The 27B-FP8 variant is the broken SKU; 27B (BF16) and 27B-AWQ/GPTQ are fine.

**4-bit checkpoints that exist and are SGLang-loadable:**

| Repo | Format | Group | Size | Calibration | Notes |
|---|---|---|---|---|---|
| `QuantTrio/Qwen3.6-27B-AWQ` | AWQ 4-bit | 128 (default) | 21 GiB | data-free | Lists SGLang ≥0.5.10 as supported. Best-tested. |
| `cyankiwi/Qwen3.6-27B-AWQ-INT4` | AWQ INT4 | 128 | 19.0 GB | STEM + Agentic | Vision + thinking preserved. |
| `cyankiwi/Qwen3.6-27B-AWQ-BF16-INT4` | AWQ INT4 w/ BF16 sensitive layers | 128 | similar | STEM + Agentic | Higher-quality variant; DeltaNet in_proj kept BF16. |
| `mattbucci/Qwen3.6-27B-AWQ` | AWQ 4-bit, group 128 | 128 | ~22 GB | 256×1024 llmcompressor | Optimized for AMD RDNA4 + SGLang; DeltaNet `in_proj_a/b` + vision tower kept BF16. |
| `groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit` | GPTQ-Pro 4-bit (FOEM) | n/a stated | ~17 GB | Wikitext-derived, PPL 6.366@n_ctx=1024 | Marlin kernels. vLLM-first recipe, SGLang command provided. |

**Recommendation for SGLang 0.5.12 on GH200:** start with **`QuantTrio/Qwen3.6-27B-AWQ`** — it's the most-tested AWQ build, doesn't require building anything special, and SGLang ≥0.5.10 is explicitly listed.

---

## 2. SGLang's preferred 4-bit format for Qwen3.6/Qwen3 family

SGLang's quantization landscape:

- **Offline (pre-quantized, no `--quantization` flag — auto-detected from `config.json`):** `awq`, `gptq`, `compressed-tensors`, `auto-round`, `gguf`, `modelopt`, `modelopt_fp4`, `petit_nvfp4`, `bitsandbytes`.
- **Marlin kernels are CUDA-only** (perfect for GH200/Hopper). Both `awq_marlin` and `gptq_marlin` are CUDA-only.
- For dense Qwen3-family models, the fast path is **AWQ → `awq_marlin` repack at load time → INT4 W × FP16 A GEMM via Marlin**.

**Verdict — preferred 4-bit format on SGLang for Qwen3.6-27B dense:**

```
AWQ 4-bit, group_size=128  →  SGLang auto-picks awq_marlin kernel  →  Marlin INT4 GEMM
```

Why AWQ over GPTQ here:
- The Qwen Hub and community publish AWQ checkpoints first and most consistently.
- AWQ-Marlin in SGLang has had more shake-out on Qwen3 dense (the Qwen3.5-27B AWQ `size_n` Marlin-repack alignment bug from issue #19406 was fixed pre-0.5.12; check Marlin tile alignment if you see weird load errors).
- GPTQ-Pro is competitive on quality but the published recipe (groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit) is vLLM-first; the SGLang command in that card is bare-bones with no MTP wiring.

**`moe_wna16` is the wrong knob** for the dense 27B — it's the AWQ-MoE fallback used for things like Qwen3.5-122B-A10B. Do not pass `--quantization moe_wna16` for Qwen3.6-27B; let SGLang auto-detect AWQ.

**Pre-quantized loading rule (load-bearing):**
> Do **not** pass `--quantization awq` when the model is already AWQ on disk. Let SGLang read the `quantization_config` block and auto-select `awq_marlin`. Adding `--quantization` triggers *online* quantization on top, which corrupts state.

---

## 3. Exact `sglang.launch_server` command (GH200 96GB, AWQ-INT4)

### 3.1 Install (do this once)

```bash
# Python 3.10–3.12 venv on Lambda Ubuntu 24.04 aarch64
python3 -m venv ~/venvs/sglang && source ~/venvs/sglang/bin/activate
pip install -U pip wheel

# PyTorch aarch64 wheel from pytorch.org (do not build from source)
pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.*

# sgl-kernel aarch64 wheel (≥0.3.21 — January 2026 first aarch64 release line)
pip install sgl-kernel  # picks up sgl_kernel-0.3.21-cp310-abi3-manylinux2014_aarch64.whl

# FlashInfer (install per its aarch64 instructions — JIT-heavy on GH200)
pip install flashinfer-python

# SGLang itself
pip install "sglang[all]==0.5.12"
```

If `sgl-kernel` ever falls back to source build on GH200: cap parallelism, otherwise the node OOM-kills the build:
```bash
MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1" make build
```

### 3.2 Baseline launch — Qwen3.6-27B AWQ, single GH200, single GPU

```bash
# Pre-warm FlashInfer JIT cache directory exists on a fast disk (NVMe under /home is fine on Lambda)
export FLASHINFER_WORKSPACE_BASE=$HOME/.cache/flashinfer
export HF_HOME=$HOME/hf

python -m sglang.launch_server \
  --model-path QuantTrio/Qwen3.6-27B-AWQ \
  --host 0.0.0.0 --port 30000 \
  --tp-size 1 \
  --mem-fraction-static 0.88 \
  --context-length 131072 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --attention-backend flashinfer \
  --chunked-prefill-size 8192 \
  --enable-mixed-chunk \
  --disable-radix-cache=false
```

Notes:
- `--tp-size 1` because 21 GiB of AWQ weights + 96 GB HBM3 means tensor-parallel is unnecessary and wasteful on a single GH200. (The Qwen3.6 cookbook examples show `--tp 8` because they target 8×H200 for the 35B-A3B MoE — do not copy that knob for dense 27B on one GH200.)
- `--mem-fraction-static 0.88` leaves headroom for activations and the Mamba radix tree state-checkpoint pool. Drop to 0.82 if you hit OOM on long-context.
- `--context-length 131072` is a sensible default; the model supports 262K natively, but the GDN state checkpoints blow up memory if you push past ~150K on TP=1. Bump only if you actually need it.
- `--attention-backend flashinfer` is the right backend on Hopper. Triton backend works but is slower for GQA-256-headdim on Hopper.
- Do **not** add `--quantization awq` — let auto-detect happen.

### 3.3 Production launch — with MTP / EAGLE speculative decoding

This is the recipe the SGLang Qwen3.6 cookbook publishes (with the V2 spec engine enabled):

```bash
SGLANG_ENABLE_SPEC_V2=1 \
python -m sglang.launch_server \
  --model-path QuantTrio/Qwen3.6-27B-AWQ \
  --host 0.0.0.0 --port 30000 \
  --tp-size 1 \
  --mem-fraction-static 0.85 \
  --context-length 131072 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --attention-backend flashinfer \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mamba-scheduler-strategy extra_buffer \
  --page-size 64
```

Why each flag:
- `SGLANG_ENABLE_SPEC_V2=1` — opt in to the v2 speculative engine that the cookbook recommends for Qwen3.6.
- `--speculative-algorithm EAGLE` — Qwen3.6 ships with the MTP head; SGLang uses it through its EAGLE path. (`NEXTN` is the older name; the cookbook example uses `EAGLE` for Qwen3.6.)
- `num-steps=3, topk=1, draft-tokens=4` — community-reported sweet spot. PyTorch blog's Qwen3-Next-80B benchmark hit best throughput at 4-token draft / 8-draft / 4-topk = 4.23 avg accept, +27% over 2-token draft. For 27B dense, accept lengths are typically 2.5–3.5 (community reports 97%/95%/91% acceptance at positions 1/2/3, dropping hard at position 4 → cap at 3 steps).
- `--mamba-scheduler-strategy extra_buffer --page-size 64` — turns on **MambaRadixCache V2**, which requires the FLA kernel backend and trades higher per-state memory for overlap scheduling. On a fat 96GB GH200, take the trade. V1 (`no_buffer`, default) is what you'd use on tighter memory.

### 3.4 Quick GPTQ alternative (if you choose `groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit`)

```bash
python -m sglang.launch_server \
  --model-path groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit \
  --host 0.0.0.0 --port 30000 \
  --tp-size 1 \
  --mem-fraction-static 0.88 \
  --context-length 131072 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --attention-backend flashinfer
```

SGLang will pick `gptq_marlin` automatically. Add the EAGLE block from 3.3 to get spec decoding.

---

## 4. RadixAttention with Qwen3.6 — quirks & MLA-related behavior on aarch64 / Hopper

Qwen3.6-27B is a **hybrid** model (Gated DeltaNet linear-attn + softmax attn at 3:1), not a pure attention transformer. That changes how prefix caching works:

### 4.1 SGLang uses MambaRadixCache, not vanilla RadixAttention, for this family

Full-attention layers still use the radix-tree KV cache, but the GDN layers maintain a **fixed-size recurrent state**, not a paged KV. The MambaRadixCache:

- **Match:** finds the longest cached prefix where a GDN state checkpoint exists and copies it in.
- **Insert:** after each prefill/decode, forks a checkpoint of the GDN recurrent state for that request.
- **Evict:** two LRU lists — KV cache evicted leaf→root, GDN states can be evicted from any node.

Three real costs to be aware of:

1. **In-place SSM/GDN state updates → no rollback.** Prefix cache is realised by *checkpointing the full state at each radix node*. State checkpoints are "orders of magnitude larger than a single KV-token slot."
2. **All-or-nothing reuse on the GDN side.** Most GDN forward kernels don't support partial-prefix replay. You either hit a checkpoint or you replay from a parent checkpoint forward.
3. **Page size matters.** V2 mode requires `--page-size 64`, which raises minimum allocation granularity. Bigger pages → less fragmentation, higher state memory per node. On 96 GB this is fine.

**Qwen3.6-27B vs. "Qwen3-Next MLA" rumor:** Qwen3.6 uses **GQA (24/4) on softmax-attn layers**, not MLA. MLA-specific kernel paths (DeepSeek V3, TokenSpeed) **do not engage** for Qwen3.6. So no aarch64-vs-x86 path divergence on MLA kernels here — the relevant kernels are FlashInfer GQA + GDN/Mamba. Both have aarch64 wheels in current SGLang.

**SGLang ARM64 path quirks on Hopper that *do* affect you:**
- FlashInfer JIT compiles GQA + spec-decoding kernels at first call. Some kernels (mm_fp4, advanced page-attn variants) require `cuda_fp8.h`; if FlashInfer's bundled headers don't include it, set `--disable-cuda-graph` for the first launch, capture the kernels, then re-enable. On 0.5.12 this is largely fixed but historically bit GH200 users.
- The Triton-based fallback attention is slower on Hopper than FlashInfer for the 256-headdim layer; do not use `--attention-backend triton` here unless FlashInfer fails.

### 4.2 Configuration cheat sheet for prefix-reuse workloads

| Workload | Setting |
|---|---|
| One-shot independent prompts | `--mamba-scheduler-strategy no_buffer` (V1), `--disable-radix-cache=false` (keep radix anyway, near-zero cost) |
| Multi-turn / shared system prompt / RAG | V2 + page 64 (the 3.3 recipe). Reuse pays back the checkpoint cost within ~2 turns. |
| Vision-heavy | Keep V1; vision-tokens are not in the radix tree anyway. Set `--chunked-prefill-size 4096` to stop image-prefill from starving decode. |

---

## 5. Expected tok/s vs vLLM (R5 #7) — who wins?

There is no published apples-to-apples GH200 96GB number for Qwen3.6-27B-AWQ on either engine yet (as of 2026-05-25). Extrapolating from the closest data points:

**Reference data we have:**

| Stack | Model | HW | Mode | Throughput | Notes |
|---|---|---|---|---|---|
| SGLang + EAGLE (4-draft) | Qwen3-Next-80B-A3B-FP8 | H200 80GB | BS=1, MTP-4/topk=4 | **324.57 tok/s** | accept 4.23 |
| SGLang + EAGLE (3-draft) | Qwen3-Next-80B-A3B-FP8 | H200 80GB | BS=1 | 306.94 tok/s | accept 3.41 |
| SGLang + EAGLE (2-draft) | Qwen3-Next-80B-A3B-FP8 | H200 80GB | BS=1 | 257.20 tok/s | accept 2.71 |
| vLLM 0.19 | Qwen3.6-27B-AWQ-INT4 (Lorbus AutoRound) | RTX 3090 | BS=1, MTP n=3 | **85 TPS sustained / 106 peak** | accept 97/95/91 % @ pos 1/2/3 |
| vLLM | Qwen3.6-27B-NVFP4 | 2× RTX 5060 Ti | concurrent=64 | 345 tok/s aggregate | NVFP4, Blackwell |
| TRT-LLM / vLLM / SGLang | Qwen3-32B | H100 | high concurrency | 2660 / 2353 / **1691** tok/s | SGLang trails on Q3-32B; *this is the canonical "SGLang loses single-stream throughput" datum* |
| SGLang vs vLLM | Llama 3.1 8B | H100 | high concurrency | **16,200 vs 12,500 tok/s** | SGLang ~+29% on shared-prefix workloads (RadixAttention wins) |

**Extrapolation to GH200 96GB / Qwen3.6-27B-AWQ-INT4 / BS=1:**

| Engine | Mode | Estimated single-stream tok/s | Reasoning |
|---|---|---|---|
| vLLM 0.19 + `qwen3_next_mtp` spec | AWQ-INT4 Marlin, n=1 spec | ~95–115 tok/s | RTX 3090 hits 85 sustained; GH200 has ~3× the memory BW and Hopper Marlin is mature |
| SGLang 0.5.12 + EAGLE 3-draft | AWQ-INT4 Marlin | ~85–105 tok/s | SGLang trails vLLM on pure decode (cf. Q3-32B 1691 vs 2353); MambaRadixCache adds per-token bookkeeping at BS=1 |
| SGLang 0.5.12, no spec | AWQ-INT4 | ~40–55 tok/s | Roughly BF16 baseline × (FP16/INT4 BW ratio); ARM/Hopper has same compute as x86/Hopper |
| BF16 baseline (either engine) | — | ~30–45 tok/s | Reference floor |

**Concurrent / multi-user workload (BS=8, shared prefixes):**

| Engine | Estimated aggregate tok/s | Notes |
|---|---|---|
| SGLang 0.5.12 | **~600–900 tok/s** | RadixAttention/MambaRadixCache shines; ~+25–35% over vLLM on shared-prefix workloads (extrapolation from Llama 3.1 8B H100 data) |
| vLLM 0.19 | ~450–700 tok/s | Wins on independent prompts, loses on shared prefixes |

**Bottom line — who wins on Qwen3.6-27B?**
- **Pure single-stream latency (chatbot, one user):** vLLM 0.19 by ~10–15% on GH200.
- **Multi-user, shared system prompt, agentic / coding tools that re-prefill the same scaffolding:** SGLang 0.5.12 by ~25–35%.
- **Friction on GH200 aarch64:** roughly equal in May 2026. Both have aarch64 wheels for their kernels (sgl-kernel ≥0.3.21 on the SGLang side, vLLM-GH200 docker images on the vLLM side). SGLang's MambaRadixCache V2 is newer than vLLM's equivalent, so expect minor sharp edges on edge-case workloads.

The question's framing "highest tok/s + least friction" → **SGLang for >1 concurrent user, vLLM for hard single-stream.** If you're not sure, default SGLang for Qwen3.6 because the cookbook is written for it and the MTP head wiring is upstream.

---

## 6. JIT compile time on first launch (aarch64)

Empirical data is sparse, but the moving parts are known:

| Component | What it compiles | First-launch cost on GH200 |
|---|---|---|
| FlashInfer | Attention kernels per (head_dim, layout, dtype) seen. For Qwen3.6-27B: GQA-256 prefill + decode, plus spec-decode variants. | **2–4 min** cold cache |
| Triton (SGLang fused MoE / sampling / layernorm) | Per-shape autotune for fused ops | **1–2 min** first-batch warmup; ~30 s thereafter due to autotune cache miss on new shapes |
| CUDA graph capture | One-time, per batch-size bucket | **30–60 s** for typical bucket set |
| EAGLE / NEXTN draft kernels (if enabled) | Spec-decode tree kernels | **+1–2 min** on top |
| GDN / Mamba FLA backend | State-update kernels for the linear-attn layers | **30–60 s** |

**Realistic timing — Lambda Ubuntu 24.04 GH200 96GB, AWQ 4-bit Qwen3.6-27B, SGLang 0.5.12 with EAGLE on:**

- **Cold cache first run:** ~**5–8 minutes** from `launch_server` to "Server listening" (weight load is ~30 s of that — the rest is JIT).
- **Warm cache subsequent runs:** ~**25–45 seconds** (weight load + CUDA graph capture + a handful of new Triton shapes).
- **Pre-warming trick:** issue one tiny prefill + one tiny decode before benchmarking. This forces every kernel SGLang will use to materialise.

**Caches to keep on fast local disk (not network FS):**
- `~/.cache/flashinfer/`
- `~/.cache/torchinductor/`
- `~/.triton/cache/`
- `~/.cache/sglang/` (cuda graphs, autotune)

**Known sharp edges on aarch64 GH200:**
- Some FlashInfer JIT paths historically needed `cuda_fp8.h` which wasn't shipped in pre-compiled wheels on aarch64 → workaround was `--disable-cuda-graph` for first launch. Largely resolved in flashinfer current as of 2026-04, but if the server hangs at "capturing CUDA graphs," drop the flag.
- Don't use `--quantization` on a pre-quantized model — it triggers a different code path that re-JITs additional online-quant kernels.
- Building `sgl-kernel` from source on GH200 has historically OOM-killed the node. With ≥0.3.21 you should not need to build from source; if you do, cap parallelism (`MAX_JOBS=2`, single NVCC thread).

---

## 7. Speculative decoding options

Qwen3.6-27B was **trained with a multi-step MTP head**, which means you get high-quality speculative decoding without training a separate draft model.

### 7.1 Options ranked

| Option | Setup | Avg accept length | Throughput uplift over no-spec (BS=1) | Notes |
|---|---|---|---|---|
| **EAGLE via MTP head** *(recommended)* | `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` | 2.5–3.5 | **~2.0–2.5×** | Uses the in-checkpoint MTP head. Zero training cost. Cookbook default. |
| EAGLE-3 with external draft head | `--speculative-algorithm EAGLE3` (requires compatible draft) | 3.0–4.0 | ~2.3–2.8× | Higher quality if a Qwen3.6-27B-specific EAGLE3 draft becomes available. Currently sparse. |
| MTP n=4 (aggressive) | `--speculative-num-steps 4 --speculative-num-draft-tokens 6 --speculative-eagle-topk 1` | ~3.5–4.2 | ~2.5–3× *if* accept holds; otherwise worse | Community reports position-4 acceptance collapses to ~8% on Qwen3.6-27B. Risky. |
| NGRAM | `--speculative-algorithm NGRAM` | low | <1.5× | No draft model needed. Use only if MTP head is unavailable. |
| Classic draft model | `--speculative-algorithm DRAFT --speculative-draft-model-path <small>` | depends | depends | Adds VRAM + complexity. Skip on GH200 with 96GB to spare. |
| Disabled | — | 1.0 | 1.0× baseline | Use for absolute lowest TTFT or maximum throughput-per-token at very high BS. |
| DFlash (`z-lab/Qwen3.6-27B-DFlash`) | dedicated DFlash draft | 4+ | 2.5–3× | Newer, less battle-tested in SGLang. Worth trying if you want max single-stream tok/s. |

### 7.2 Recommended config for GH200 / Qwen3.6-27B / AWQ-INT4

```text
--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
SGLANG_ENABLE_SPEC_V2=1
```

This matches the published cookbook recipe and the community-validated sweet spot. If you measure your own accept-length and see it consistently >3.5 with position-4 acceptance >40%, bump `--speculative-num-steps` to 4 and `--speculative-num-draft-tokens` to 6. Otherwise stay at 3/4.

### 7.3 When to turn spec OFF

- Batch size ≥ 32 sustained — spec overhead starts to eat into aggregate throughput.
- Very long-context decode where the bottleneck is GDN state I/O rather than weight loading.
- Debugging — disable spec first to isolate any garbage-output issues.

---

## 8. Friction checklist for Lambda GH200 96GB (Ubuntu 24.04 aarch64)

| Step | Action |
|---|---|
| 1 | `pip install` torch from pytorch.org cu130 aarch64 index (don't build) |
| 2 | `pip install sgl-kernel` (gets the aarch64 wheel, 0.3.21+) |
| 3 | `pip install flashinfer-python` |
| 4 | `pip install "sglang[all]==0.5.12"` |
| 5 | `huggingface-cli download QuantTrio/Qwen3.6-27B-AWQ` (21 GiB) |
| 6 | First launch with the recipe in §3.3 — expect 5–8 min JIT |
| 7 | Pre-warm with 1 prefill + 1 decode |
| 8 | Benchmark with `sglang.bench_serving` or `genai-perf` |
| 9 | Save `~/.cache/flashinfer` + `~/.cache/sglang` to durable storage if reinstalling |

**Things that will bite you if ignored:**
1. **Do not load `Qwen/Qwen3.6-27B-FP8`** — issue #23687 silently produces garbage. Use AWQ or BF16.
2. **Do not pass `--quantization awq` on a pre-quantized checkpoint** — it stacks online quant on top.
3. **Do not use `--quantization moe_wna16`** — that's for the 35B-A3B MoE, not the dense 27B.
4. **Do not greedy-decode (`temperature=0`)** — community has documented degenerate loops on Qwen3 family. Use `temperature=0.7, top_p=0.95, top_k=20`. SGLang picks these up from the model card via `sampling_defaults='model'`.
5. **TP=1 on one GH200 96GB.** Do not copy the `--tp 8` from the cookbook — that's for the MoE 35B on 8×H200.
6. **Context-length sanity:** start at 131K. Pushing 262K with MTP active will stress the Mamba state pool; lower `--mem-fraction-static` to 0.8 or drop context if you see OOM.
7. **First launch is slow; do not panic at 5 minutes of "JIT compiling…".**

---

## 9. Cross-references with R5

- **R5 #6 / #7 (SKU + vLLM side):** confirms Qwen3.6-27B is the dense flagship SKU and that vLLM 0.19 + Marlin + `qwen3_next_mtp` is the comparable fast path. This doc takes the opposite engine and confirms 4-bit AWQ is the right format on either.
- **FP8 ban (R5 prior + issue #23687):** holds. AWQ/GPTQ unaffected because they use a different loader. The exact bug is in `qwen3_5.py`'s string mutation loop for FP8 `weight_scale_inv` — confirmed by reading the issue + the related HF discussion. AWQ-INT4 loaders never touch that codepath.

---

## Sources

- [SGLang issue #23687 — Qwen3.6-27B-FP8 silent garbage (gate_gate_up_proj loop)](https://github.com/sgl-project/sglang/issues/23687)
- [Qwen/Qwen3.6-27B model card (architecture, MTP head, hybrid GDN+attn)](https://huggingface.co/Qwen/Qwen3.6-27B)
- [SGLang Cookbook — Qwen3.6 (cookbook page with launch + EAGLE + MambaRadixCache V2 flags)](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6)
- [SGLang Cookbook — Qwen3.5 (older but still authoritative on flags)](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5)
- [SGLang Quantization docs (AWQ, GPTQ, Marlin, moe_wna16 context)](https://sgl-project.github.io/advanced_features/quantization.html)
- [PyTorch blog — Hybrid Models Meet SGLang (MambaRadixCache + Qwen3-Next benchmarks)](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)
- [SGLang Speculative Decoding docs (EAGLE / NEXTN / NGRAM)](https://sgl-project.github.io/advanced_features/speculative_decoding.html)
- [QuantTrio/Qwen3.6-27B-AWQ (recommended AWQ checkpoint, lists SGLang ≥0.5.10)](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ)
- [cyankiwi/Qwen3.6-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.6-27B-AWQ-INT4)
- [mattbucci/Qwen3.6-27B-AWQ (AWQ with thinking+vision preserved; documents AWQ specifics)](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ)
- [groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit (GPTQ alternative + SGLang command)](https://huggingface.co/groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit)
- [SGLang issue #19406 — Qwen3.5-27B AWQ Marlin repack `size_n` alignment (precursor bug, fixed)](https://github.com/sgl-project/sglang/issues/19406)
- [SGLang issue #3769 — sgl-kernel aarch64 wheel tracking](https://github.com/sgl-project/sglang/issues/3769)
- [SGLang Install docs](https://docs.sglang.io/get_started/install.html)
- [Qwen Hub Discussion #5 — Qwen3.6-27B weight name mismatch on SGLang 0.5.10 (the original FP8 bug surfacing)](https://huggingface.co/Qwen/Qwen3.6-27B/discussions/5)
- [Medium — "Overnight Stack for Qwen3.6-27B: 85 TPS on RTX 3090" (vLLM + EAGLE n=3 acceptance numbers 97/95/91)](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914)
- [LLMKube — Qwen3.6-27B bakeoff (vLLM vs llama.cpp on consumer GPUs)](https://llmkube.com/blog/qwen3-6-27b-bakeoff)
- [Kaitchup — Qwen3.5-27B INT4 vs NVFP4 vs FP8 vs BF16 latency (single-query latency rankings)](https://kaitchup.substack.com/p/qwen35-27b-latency-and-throughput)
- [Qwen blog — Qwen3.6-27B announcement](https://qwen.ai/blog?id=qwen3.6-27b)
