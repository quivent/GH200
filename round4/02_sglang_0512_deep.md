# SGLang on GH200 — Round 4 LLM-only Deep Dive

**Agent**: #04 / Round 4
**Date**: 2026-05-25
**Scope**: 8-question LLM-only drill on SGLang v0.5.12.post1 (LLM workloads only — no diffusion).
**Constraints carried forward**:
- FP8 is broken on this stack for diffusion / Wan; **bf16 is default for diffusion**.
- FP8 is **OK on dense Hopper LLMs with head_dim ≤ 128** (per the user's note for this round).
- Package renamed `sgl-kernel` → `sglang-kernel` in v0.5.10; **install `sglang-kernel >= 0.4.2.post2`** or use the Docker image.

---

## 0. TL;DR (executive summary)

1. **v0.5.12 / v0.5.12.post1** is current. Big-ticket additions for LLM serving are: HiCache + UnifiedRadixTree (with SWA), HiCache for DeepSeek V4, Mooncake **SSD incremental offload**, Speculative Decoding V2 with **Adaptive Spec V2**, EAGLE-3 + SWA, **W4A4 MegaMoE** + Marlin / FlashInfer W4A8 MoE on Hopper, FlashMLA hybrid attention single-kernel call, prometheus metric `sglang:get_loads_duration_seconds` and `fwd_occupancy`.
2. **RadixAttention** on GH200 inherits H100 traffic-shape numbers verbatim: **75–95 % hit rate on 60 %+ prefix-overlap workloads**, ~6.4× max prefix throughput vs vLLM. Each hit is worth ~1.43× more than on H100 because GH200 HBM3e is 4.8 TB/s vs 3.35 TB/s — but no first-party LMSYS GH200 hit-rate publication exists.
3. **HiCache L2 over NVLink-C2C** is *architecturally* the killer GH200 feature: 297 GB/s D2H (write to LPDDR) and 375 GB/s H2D (cache restore) — **5–6× cheaper than PCIe Gen5 H100 host offload**. LMSYS has not published a GH200 number; the closest benchmark is H800 + Mooncake. **Recommended sizing**: `--hicache-ratio 4–6` on a single GH200 144 GB SKU (uses 576–864 GB of the 480 GB-effective LPDDR — cap at ratio ~3 for the 96 GB SKU).
4. **PD disaggregation single-GH200** is rarely the right shape (prefill TP and decode TP would each be 1; you're better off running combined). **Two-GH200 P/D pair** is the canonical config — Mooncake + IB device + (optionally) GPU Staging Buffer for heterogeneous TP. Use `MC_INTRANODE_NVLINK=1` + `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK` if both GH200s sit on the same NVLink switch.
5. **DeepSeek V3.1 / V4 on Hopper SM90 / GH200**: pick `--attention-backend flashmla` (page=64, FP8 KV supported) or fallback to `fa3` (default). **W4A16 (Marlin / FlashInfer MXFP4)** is the landed Hopper path for V4. CutlassMLA is supported but FlashMLA is the V4 day-0 default. TileLang mHC kernels are general SM90/SM100; **TileLang ARM64 only matters for non-CUDA NPU back-ends**, not for GH200 (Hopper die is still CUDA).
6. **JIT compile on first launch (aarch64)** is **the single biggest operational gotcha**. Empirical numbers reported in the wild:
   - `sgl-flash-attn3` JIT on arm64 first request: minutes-scale (no Hugging Face arm cubin shipped).
   - DeepGEMM JIT: **10–20 minutes** without pre-warm; **~1 second** after `python -m sglang.compile_deep_gemm`. Reports of 23-minute prefill / 11-minute decode warmups w/o it.
   - FlashInfer JIT: **up to 172 seconds** per kernel without `flashinfer-jit-cache`.
   - **Pre-warm recipe** below — run *exactly* the same flags you'll serve with.
7. **Structured output**: XGrammar default; **< 40 µs / token overhead**; jump-forward kicks in automatically on fixed-string runs in the grammar. **3× vLLM** on guided JSON; SGLang 96–98.2 % JSON compliance vs vLLM 90–94 %. **XGrammar-2** (2026-05-04) compiles grammars 80× faster — useful for highly variable schemas.
8. **Speculative decoding**: **SGLang EAGLE-3 still leads vLLM 0.13 EAGLE-3** on like-for-like H100 / H200 single-stream (SGLang 373 tok/s vs baseline 158 = 2.36×; vLLM 1051 → 1150.5 = 1.09× on MLPerf, 2024 → 2574 = 1.27× on ShareGPT). vLLM's *Eagle3 absolute throughput numbers are good*; the **acceleration ratio is still SGLang's**, plus SGLang gets **Adaptive Spec V2** and **EAGLE-3 SWA** in 0.5.12 — neither is in vLLM 0.21 yet.

---

## 1. SGLang v0.5.12 / v0.5.12.post1 — every LLM-relevant flag / env / feature

**Source**: v0.5.12 release notes (GitHub, fetched 2026-05-25). v0.5.12 cut **2026-05-16**, .post1 **2026-05-23** (tagger Fridge003).

### 1.1 New launch flags

| Flag | Purpose | Notes |
|---|---|---|
| `--prefill-only-disable-kv-cache` | Skip KV pool allocation for prefill-only deployments. | Disaggregated prefill workers can now skip the KV pool entirely. |
| `--enable-strict-thinking` | Activates a two-phase reasoning grammar for "thinking" outputs. | Useful for o1-style / DeepSeek-R1-style models that have a `<think>...</think>` block. |
| `--lora-paths` | Multi-node deterministic LoRA-ID assignment. | Behavior change: stable IDs across nodes. |
| `--disable-cuda-graph` | Now usable for NPU deployments with MTP warmup fixes. | Behavior tightening; same flag, fewer gotchas. |
| `--random-input-len` (bench tool) | Variable input length in `send_one.py`. | For benchmark realism. |
| `--moe-runner-backend marlin\|flashinfer_mxfp4\|deepgemm\|...` | Selection of MoE runner backend, including the new Marlin / FlashInfer W4A8 MoE on Hopper. | DeepSeek V4 cookbook uses `marlin` on Hopper. |

### 1.2 New / renamed environment variables

| Env var | Meaning |
|---|---|
| `SGLANG_MAX_KV_CHUNK_CAPACITY` | Caps max KV cache chunk size (per the v0.5.12 notes). |
| `SGLANG_RADIX_FORCE_MISS` | Debug: force every radix lookup to miss. Use to A/B prefix-cache uplift. |
| `SGLANG_OPT_FP8_WO_A_GEMM` | Default-on FP8 weight-only A-GEMM optimization. Hopper FP8-friendly. |
| `SGLANG_USE_JIT_ALL_REDUCE` → **`SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2`** | Renamed. Custom AR v2 path. |
| `SGLANG_TRACE_LEVEL` | Startup trace verbosity. |
| `SGLANG_ENABLE_SPEC_V2=True` | Opt-in to Speculative Decoding V2 path (Adaptive Spec V2; EAGLE topk=1 only). |
| `SGLANG_DSV4_FP4_EXPERTS=0` | Force the **FP8-converted** DeepSeek-V4 path (`sgl-project/DeepSeek-V4-Flash-FP8`) over FP4 on Hopper. |
| `SGLANG_MOONCAKE_CUSTOM_MEM_POOL={NVLINK,INTRA_NODE_NVLINK,BAREX}` | Mooncake transport pool — **`INTRA_NODE_NVLINK`** is the GH200-pair sweet spot. |
| `MC_FORCE_MNNVL` | Enable NVL72 NVLink transport on Mooncake. |
| `MC_INTRANODE_NVLINK` | Enable intra-node NVLink for Mooncake. |
| `SGLANG_DISAGG_STAGING_BUFFER=1` + `_SIZE_MB=64` + `_POOL_SIZE_MB=4096` | GPU Staging Buffer for heterogeneous-TP PD (v0.5.10+). **Disabled by default**. **Not for MLA models** (DeepSeek V2/V3/V4). |
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=4–12` | Mooncake worker threads (default dynamic). |
| `SGLANG_DISAGGREGATION_QUEUE_SIZE=4` | Mooncake queues. |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=300` | Init timeout (s). |
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL=5.0` | Decode-side. |
| `SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=2` | Decode-side. |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT=300` | Decode-side. |
| `ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE=true` | Ascend NPU; irrelevant on GH200. |
| `TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800` | Workaround for long DeepGEMM v2 warmup. **Set this if you skip `compile_deep_gemm`.** |

### 1.3 New attention / KV-cache options

- **TokenSpeed MLA** prefill/decode kernels on Blackwell (SM100) with FP8 KV — Blackwell-only, not GH200.
- **FlashMLA**: refreshed interface; SWA + compressed KV now in a *single fused kernel call*. **Hopper SM90 requires head-count multiples of 64**; Blackwell SM100 requires 128. The kernel takes `k_cache` + `extra_k_cache` and indices, sharing metadata construction in the forward pass. (PR #24225, #25418.)
- **HiSparse FP8 KV cache** via `flashmla_kv` backend — offload inactive KV to host memory.
- **HiCache framework support for UnifiedRadixTree (with SWA)** — closes #23639.
- **HiCache for DeepSeek V4** with full SSD offload via Mooncake (PR #24691).
- **ShadowRadix prefix caching** with three heterogeneous KV pools (SWA, C4, C128).

### 1.4 New speculative decoding pieces

- **Adaptive Spec V2** — adaptive draft acceptance thresholds (set `SGLANG_ENABLE_SPEC_V2=True`).
- **EAGLE-3 + SWA** + newer drafters.
- **Kimi K2.5** EAGLE-3 MLA integration; **Gemma 3/4 + EAGLE-3**.
- Draft-extend refactor (`EagleDraftExtendInput`) — internal but breaks third-party draft model wrappers.

### 1.5 New quantization

- **W4A4 MegaMoE kernels** — faster, minor accuracy hit.
- **Marlin / FlashInfer W4A8 MoE on Hopper** — landed path for DeepSeek V4 on H100/H200/GH200.
- **DeepEP** swapped from community fork to `deepseek-ai/DeepEP@hybrid-ep`; CUDA 13 migration.
- **Fused SiLU + clamp + FP8 quant** kernel.
- **FlashInfer pinned at 0.6.11.post1** (relevant for FlashInfer-JIT-cache pin choice — see §6).

### 1.6 New observability

- Prometheus `sglang:get_loads_duration_seconds`.
- `fwd_occupancy` in `SchedulerStats`.
- Per-iteration metrics via ZMQ PUB — useful for high-rate dashboards without scraping HTTP.

### 1.7 Frontend / API

- `/v1/tokenize` supports chat-completion-style params.
- OpenAI `reasoning.enabled` mapped to `thinking` / `enable_thinking`.
- Tool-call parser hardened for DeepSeek V4 / Kimi K2.5 (bare-numeric IDs).
- Reasoning parser auto-detects from chat template.
- `repetition_penalty=0` now rejected in validation (was a silent footgun).
- Azure Blob (`az://`) connector for model load.

### 1.8 Behavior changes worth noting

- **CUDA 13 + Torch 2.11** default (v0.5.11+); GH200 wheel index path: `https://docs.sglang.io/whl/cu130`.
- **Piecewise CUDA graph default-on** (v0.5.10+) — improves memory overhead on models with complex control flow. Disable with `--disable-cuda-graph` only for debug.
- **`sgl-kernel` → `sglang-kernel`** package rename (v0.5.10+) — `sgl-kernel 0.3.21` is *archived*.

---

## 2. RadixAttention prefix-cache hit-rate — measurements + GH200 inference

**No published GH200 measurement exists** (verified Round 3, still true Round 4). Use H100 / H200 numbers as proxy — hit rate is a function of *traffic shape*, not memory bandwidth.

| Workload | Hit rate | Source |
|---|---|---|
| LLaVA-Next-34B multimodal | **52.4 %** | LMSYS 2024-01-17 |
| Vicuna-33B chat | **74.1 %** (→ 1.7× lower mean TTFT) | LMSYS 2024-01-17 |
| Few-shot | **85–95 %** | 2026 production guides |
| Multi-turn chat | **75–90 %** | 2026 production guides |
| Code analysis / RAG | **60–80 %** | 2026 production guides |
| 60 %+ prefix overlap | **75–95 %** | Spheron 2026 |
| Novita Qwen3-Coder-480B (HiCache + Mooncake) | **40 % → 80 %** with L3 | LMSYS HiCache blog |
| Unique-prompt traffic | 0 % | Spheron 2026 |
| H100 raw throughput edge prefix-heavy | **6.4× vLLM** | Particula 2026 |
| H100 single-turn unique prompts | **vLLM wins narrowly** (60 vs 52.7 tok/s in one Particula test) | Particula 2026 |

### Why GH200 hit rate equals H100 hit rate

RadixAttention is a software data structure over KV pages. Cache hit rate is **only** a function of (a) prefix overlap in your traffic, (b) tree size relative to working set, (c) eviction policy. None of these depend on memory bandwidth or aarch64 vs x86_64.

### Why GH200 hit *value* is ~1.43× H100

- H100 SXM HBM3 ≈ 3,350 GB/s. GH200 HBM3e ≈ 4,800 GB/s (96 GB SKU) or 4,900 GB/s (144 GB SKU).
- Decode is memory-bandwidth bound. Each KV-cache reuse saves memory traffic; on GH200 those saved bytes count for ~1.43× more decode tok/s than on H100, **until you become compute-bound** (rarely for LLM decode at non-tiny batch).

### Debug tip new in 0.5.12

Set `SGLANG_RADIX_FORCE_MISS=1` and re-run your benchmark to A/B exactly how much of your observed throughput is RadixAttention's contribution. Useful for capacity planning when you can't replay production traffic.

---

## 3. HiCache L2 over NVLink-C2C — design + sizing

### 3.1 Design recap (from `docs.sglang.io/.../hicache_design.html`)

- **L1** = GPU HBM (private to inference instance). Layout: `layer_first`.
- **L2** = host LPDDR / RAM (private). Layout: `page_first` for I/O efficiency.
- **L3** = distributed storage (Mooncake / 3FS / NIXL / HiCacheFile) — shared across instances.

Eviction / promotion policies:
- **`write_through`** — every L1 access immediately written to L2. Max coverage if bandwidth allows.
- **`write_through_selective`** — only promote pages above a hit-count threshold.
- **`write_back`** — promote at eviction time only. Lowest L2 bandwidth, lowest coverage.

I/O backends: `--hicache-io-backend {direct, kernel}`. `kernel` is GPU-assisted, **up to 3× higher transfer speed** than direct CUDA memcpy per LMSYS HiCache blog.

### 3.2 GH200-specific bandwidth (third-party measured)

| Path | GH200 NVLink-C2C | PCIe Gen5 x16 (H100) | Ratio |
|---|---|---|---|
| Host → Device (cache restore) | **~375 GB/s** (~83 % of 450 GB/s theoretical) | ~63 GB/s | **5.95×** |
| Device → Host (KV eviction) | **~297 GB/s** (~66 %) | ~63 GB/s | **4.71×** |
| DDR ↔ DDR via C2C | ~250 GB/s | n/a | — |
| C2C latency (ping-pong) | 150–400 ns | µs-scale | ~10× lower |

Sources: Verda blog (Jan 2026), arXiv 2408.11556v2.

### 3.3 Sizing recommendation for Grace's 480 GB LPDDR5x

- **480 GB LPDDR is *system memory*** — not "free" for KV cache. Reserve ≥ 64 GB for OS, page cache, NCCL buffers, and the Python heap on a single-GPU node.
- **`--hicache-ratio R`** sets host pool = R × HBM-side pool. Demonstrated value in LMSYS bench: **R = 2**.
- For GH200 **144 GB HBM3e** SKU: reasonable starting point is `--hicache-ratio 2` → 288 GB L2; push to `--hicache-ratio 3` → 432 GB L2 only if you're confident the rest of the host workload is small. Or use the absolute form: `--hicache-size 320` (GB per rank).
- For GH200 **96 GB HBM3e** SKU: ratio = 4 → 384 GB L2 fits inside 480 GB with headroom.
- **Layout**: `--hicache-mem-layout page_first` (default for L2); on NVLink-C2C `page_first_direct` skips an intermediate copy and is theoretically optimal — **needs in-house benchmark to confirm**.
- **Write policy**: start with `write_through_selective` (best balance), fall back to `write_back` only if D2H bandwidth becomes the bottleneck (visible as `device→host` time dominating in scheduler stats).

### 3.4 Expected hit-rate at default sizes (inferential)

With `--hicache-ratio 2`, L1+L2 ≈ 3× the GPU's KV budget. Empirical Novita result shows hit rate climbing **40 % → 80 %** when L3 is added on Qwen3-Coder-480B — for a single GH200 (no L3 needed unless the working set blows past 432 GB), expect:
- Multi-turn chat: 75–90 % at L1 + 88–95 % at L1+L2 (saturating).
- Long-context RAG (16k–128k contexts): 60–80 % → 80–92 % with L2.
- Document QA streaming: 30–50 % → 60–80 % with L2.

These are extrapolations; **no GH200 hit-rate measurement is published** as of 2026-05-25.

### 3.5 What the 0.5.12 release added on top

- HiCache framework support for **UnifiedRadixTree (with SWA)** — your KV cache layer can be sliding-window and still hit the L1/L2/L3 stack.
- **HiCache for DeepSeek V4** + SSD offload via Mooncake store.
- Stability fixes across **cascade eviction**, **tombstone replay**, **partial-match paths**.

---

## 4. PD disaggregation — single GH200 vs two-GH200 P/D pair

### 4.1 Single GH200 — *almost never* the right shape

If you have one GH200, prefer **unified prefill+decode** on the same process. PD disaggregation pays off when:
- Prefill and decode have wildly different optimal TP / batch.
- You want to scale prefill and decode capacity independently.
- Your traffic is bursty (long-input vs steady-decode).

On a *single* GH200 the only way to "disaggregate" is to colocate two processes sharing the GPU via MPS — which negates the point. **Skip PD on a single GH200.**

If you absolutely must (e.g., to validate config before scaling out):
```bash
# Prefill on GPU 0, decode on GPU 0 (MPS or naive sharing)
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill --port 30000 --base-gpu-id 0 \
  --dtype bfloat16 --mem-fraction-static 0.4

python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode --port 30001 --base-gpu-id 0 \
  --dtype bfloat16 --mem-fraction-static 0.4

python -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 8000
```
This is a *test rig*, not a production config.

### 4.2 Two-GH200 P/D pair — the canonical setup

Two physical GH200 nodes, each with one GPU, connected by InfiniBand (ConnectX-7 / ConnectX-8) or NVLink (NVL2 / NVL32 fabric).

**Recommended config (homogeneous TP=1 on each side):**

```bash
# === Prefill node (GH200 #0) ===
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=12
export SGLANG_DISAGGREGATION_QUEUE_SIZE=4
# Set ONLY if both GPUs sit on the same NVLink switch (NVL32 / NVL2):
export MC_INTRANODE_NVLINK=1
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK

python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0 \
  --prefill-only-disable-kv-cache \
  --dtype bfloat16 \
  --port 30000

# === Decode node (GH200 #1) ===
export SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL=5
export MC_INTRANODE_NVLINK=1
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK

python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0 \
  --enable-hierarchical-cache --hicache-ratio 2 \
  --dtype bfloat16 \
  --port 30001

# === Router (anywhere) ===
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://${PREFILL_IP}:30000 \
  --decode  http://${DECODE_IP}:30001 \
  --host 0.0.0.0 --port 8000
```

### 4.3 GPU Staging Buffer — **only if prefill TP ≠ decode TP**

The v0.5.10 GPU Staging Buffer batches scattered KV head slices into contiguous memory for bulk RDMA. **~1000× fewer RDMA requests on GQA models**, **2–5× throughput at high concurrency** when TP-sizes differ.

```bash
# Heterogeneous-TP only (e.g., prefill TP=2 across both GPUs, decode TP=1 on one)
export SGLANG_DISAGG_STAGING_BUFFER=1
export SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=64
export SGLANG_DISAGG_STAGING_POOL_SIZE_MB=4096
```

**Hard constraint**: docs say staging buffer is **not safe for MLA models** (DeepSeek V2/V3/V4). For homogeneous TP it's neutral-to-slightly-negative — leave off unless you measure a win.

### 4.4 Mooncake transport pool selection

| Pool | When to use |
|---|---|
| `INTRA_NODE_NVLINK` | Both GPUs on same NVLink switch (NVL2 / NVL32 / NVL72). **Best for two GH200s in same chassis or NVL2 pair.** |
| `NVLINK` | Cross-node NVLink (NVL72). |
| `BAREX` | RDMA over IB / RoCE fallback. **Default for two GH200s in different chassis.** |

### 4.5 Decode-side radix cache

v0.5.12 fully landed the decode-side radix cache for disaggregated workflows — decode workers no longer recompute everything from the KV blob; they store a small radix tree on their KV pages too. Enable with `--enable-hierarchical-cache` on the decode node (this is the same flag that enables HiCache L2; the radix-on-decode behavior follows automatically).

---

## 5. DeepSeek V3.1 / V4 on Hopper SM90 (GH200)

### 5.1 MLA backend matrix (from `attention_backend.html`)

| Backend | Page size | SM90 support | SM100 support | FP8 KV |
|---|---|---|---|---|
| FA3 (partial MLA) | n/a | ✅ default | ✅ | ❌ |
| FlashInfer MLA | 1 | ✅ | ✅ | ❌ |
| **FlashMLA** | **64** | ✅ | ✅ | ✅ |
| Cutlass MLA | 128 | ✅ | ✅ | ✅ |
| TRTLLM MLA | 32 / 64 | ✅ | ✅ default | ✅ |

**Recommendation on GH200 (Hopper SM90)**:
- **DeepSeek V3.1** (685B MoE): start with `--attention-backend fa3` (auto-selected); switch to `flashmla` if FP8 KV cache is needed. Cutlass MLA if page_size=128 fits your batch shape.
- **DeepSeek V4 / V4-Flash**: documented day-0 path is **`--moe-runner-backend marlin`** (W4A16) or `flashinfer_mxfp4`. The cookbook does *not* call out an aarch64 / GH200 caveat, but it doesn't promise GH200 either. **Empirical run required.**

### 5.2 DeepSeek V3.1 launch on a single GH200 144 GB SKU

V3.1 is 685B params — does *not* fit on one GH200 even at FP8. **TP across at least 4 GH200s** is the floor.

```bash
# 8 × GH200 144 GB (single NVLink switch / NVL8), V3.1 with DP attention
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --trust-remote-code \
  --tp 8 \
  --dp 8 --enable-dp-attention \
  --attention-backend flashmla \
  --moe-a2a-backend deepep \
  --dtype bfloat16 \
  --enable-hierarchical-cache --hicache-ratio 2 \
  --port 30000
```

Quoted V3.1 / Hopper numbers from `docs.sglang.io/basic_usage/deepseek_v3.html`:
- DP attention: **+1.9× decoding throughput** on high-batch.
- EAGLE MTP: **1.8× at batch 1, 1.5× at batch 32** on H200 TP=8.
- MLA optimizations: "up to 7× output throughput."

### 5.3 DeepSeek V4 / V4-Flash launch on GH200

Per the cookbook (no GH200-specific section — extrapolated from Hopper H200 guidance):

```bash
# Pre-converted FP8 path (recommended for H200/H100 — and by extension GH200)
SGLANG_DSV4_FP4_EXPERTS=0 \
python -m sglang.launch_server \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --trust-remote-code \
  --tp 4 \
  --attention-backend flashmla \
  --port 30000
```

The day-0 LMSYS blog reports **H200 Flash (285B): 266 → 240 tok/s** across 4K → 900K context (10 % drop). **No GH200 measurement exists**; expect roughly +25 % on GH200 144 GB on memory-bandwidth-bound decode (HBM3e 4.9 TB/s vs H200 4.8 TB/s is essentially a wash — gain comes from C2C HiCache, not raw HBM).

### 5.4 The TileLang ARM64 question

User asked specifically: does TileLang ARM64 apply to GH200?

**Answer: No.** TileLang's mHC kernels are a generic IR that targets GPU / HPU / NPU. On GH200 the GPU side is still the Hopper SM90 CUDA path — TileLang on GH200 generates CUDA, not ARM64. The "TileLang ARM64" backend exists for the CPU / NPU side and is only relevant on aarch64 NPU systems (e.g., Huawei Ascend on ARM hosts). On GH200, you use FlashMLA / Cutlass MLA / TRTLLM MLA exactly as on H100/H200; aarch64-ness of the host CPU doesn't change the GPU kernel path.

### 5.5 Open Hopper FP8 caveat reminder

User's Round-4 constraint allows FP8 for **dense Hopper LLMs with head_dim ≤ 128**. DeepSeek V3/V4 has MLA with head_dim 128 — qualifies. **But** Round-1 issues #24650 (V4-Flash-FP8 + EAGLE RoPE shape crash) and #18550 (fa3 + fp8_e5m2 silent triton fallback) are *still open* per Round-3 sweep. Run with FP8 deliberately and validate output; don't assume.

---

## 6. JIT compile time on first launch (aarch64) + pre-warm recipe

This is the **single biggest operational gotcha** on GH200. The Dockerfile literally documents it:
```
aarch64: kernels-community/sgl-flash-attn3 ships no arm variants; JIT-compile at runtime.
```

### 6.1 What gets JIT-compiled on arm64 first launch

1. **sgl-flash-attn3** kernels (no upstream arm cubin in `kernels-community/sgl-flash-attn3` — verified, no arm folder).
2. **FlashInfer** kernels (`flashinfer-python` without `flashinfer-jit-cache` / `flashinfer-cubin`). Reports of single-kernel compile at **172 seconds**; multi-worker deadlock bug #41865 noted on vLLM side.
3. **DeepGEMM** kernels (FP8 / W4A16 paths). Reports of **10–20 minute** total warmup; bug #9867 reports **23 min prefill / 11 min decode** node startup w/o pre-warm; bug #10733 tracks the speed-up effort.
4. **Torch compile** if `--enable-torch-compile` (this you control).
5. **CUDA graph capture** at first batch.

### 6.2 Measured JIT cost on x86_64 (use as aarch64 lower bound; aarch64 typically slower)

- DeepGEMM cold run: **606 tok/s** vs warm **1,376 tok/s** (≈ 56 % perf penalty during cold).
- DeepGEMM cache directory at warm state: **81 kernel directories** for a single model.
- DeepGEMM full warmup w/o pre-compile: **10–20 minutes**; **~1 second** if you ran `compile_deep_gemm` ahead.

### 6.3 Pre-warm recipe (run **before** opening to traffic)

```bash
# Step 1 — install pre-built kernel caches where available
pip install flashinfer-python flashinfer-cubin
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130

# Step 2 — pre-compile DeepGEMM with *same flags* as the launch you'll do
python -m sglang.compile_deep_gemm \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --tp 1 \
  --trust-remote-code

# Step 3 — boot the server. First request will still JIT sgl-flash-attn3 on arm64;
#         second request onward is fully warm.
python -m sglang.launch_server [...your flags...] &
SERVER_PID=$!

# Step 4 — synthetic warm-up request to trigger sgl-flash-attn3 JIT + CUDA graph capture
# Wait for /health to be ready (use until-loop, not sleep-loop)
until curl -sf http://localhost:30000/health > /dev/null; do sleep 5; done
curl -sf http://localhost:30000/generate -d '{
  "text": "Hello",
  "sampling_params": {"max_new_tokens": 8}
}' > /dev/null

# Now open to real traffic.
```

If you cannot pre-compile (cold container, no model on disk), set the safety net:
```bash
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800
```
This prevents NCCL c10d timeouts during the multi-minute first batch.

### 6.4 Long-term fix watch list

- v0.5.12 release notes mention **"aarch64 cubin handling improvements"** and **"ARM CPU phase-1A CI bootstrap"** — proper arm cubin distribution is coming but not here yet.
- LMSYS roadmap issue #12780 lists "JIT kernels" as in-progress.

---

## 7. Structured output (jump-forward decoding) — overhead and use cases

### 7.1 Overhead numbers

- **< 40 µs / token** XGrammar mask-overhead per LMSYS.
- **3× faster than vLLM** on guided JSON workloads (LMSYS Twitter, SqueezeBits).
- **96–98.2 % JSON compliance** vs vLLM's 90–94 % (Particula 2026).
- **3–10× faster JSON decoding** than other open-source guided-decoding stacks (LMSYS announcement).
- **XGrammar-2** (2026-05-04): **80× faster grammar compilation** for highly variable schemas (e.g., per-request JSON schemas in tool calling).

### 7.2 Jump-forward mechanism (LMSYS 2024-02-05 blog)

When the grammar FSM has a deterministic run of fixed tokens (e.g., the `"name":` literal in a JSON schema after `{`), SGLang **terminates the current decode and re-enqueues the request** with the fixed tokens appended. RadixAttention's "extend" primitive then re-uses the KV from the previous run, so no compute is wasted.

Best returns when:
- Output is mostly schema scaffolding with small variable bits (configs, structured records, IDs).
- Schema has long deterministic runs (e.g., type signatures, JSON keys, boolean enums).

### 7.3 Use-case fit

| Use case | Jump-forward gain | Notes |
|---|---|---|
| Tool calling / function calling | High | Function names + arg keys are fixed; values are variable. |
| Structured extraction (named entities → JSON) | High | Key names dominate token budget. |
| Code generation with type signatures | Medium | Class names / keywords fixed, bodies variable. |
| Open-ended narrative with JSON-wrapped metadata | High on metadata, neutral on narrative. |
| Pure free-form chat | Zero (grammar not engaged). |
| Regex-validated outputs (UUIDs, dates) | High | Deterministic format. |

### 7.4 Backends

- **XGrammar (default)** — JSON / regex / EBNF.
- `--grammar-backend outlines` — JSON / regex only.
- `--grammar-backend llguidance` — JSON / regex / EBNF.

Use cases with structural tags (function calling with multi-part outputs) work natively on XGrammar via the structural-tag API.

---

## 8. Speculative decoding — SGLang vs vLLM 0.21

### 8.1 SGLang state (May 2026)

- **EAGLE-3 single-stream on H100 / Llama-3.1-8B**: baseline 158.34 → EAGLE-2 244.10 → **EAGLE-3 373.25 tok/s** (2.36×).
- **DeepSeek MTP** via EAGLE on H200: **1.8× batch 1**, **1.5× batch 32**.
- **EAGLE-3 SWA + newer drafters** added in v0.5.12.
- **Adaptive Spec V2** (SPECv2 path, set `SGLANG_ENABLE_SPEC_V2=True`) — adaptive draft acceptance thresholds; topk=1 limitation.
- **Kimi K2.5 EAGLE-3 MLA**, **Gemma 3/4 + EAGLE-3** added in 0.5.12.
- Algorithms supported: EAGLE, EAGLE3, MTP, STANDALONE, NGRAM, DFLASH.
- Documented EAGLE-3 model: `jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B`.

### 8.2 vLLM 0.13 / 0.21 state (Red Hat blog 2026-04-16)

- **gpt-oss-120B** with EAGLE-3 on H200-PCIe-141GB:
  - ShareGPT: **2024 → 2574 tok/s peak (+27.2 %)**.
  - MLPerf TP=1: **1051.0 → 1150.5 tok/s (+9.5 %)**.
  - SWE-bench: **2620.89 → 3250.07 tok/s (+24 %)**.
- Acceptance rates: 2-draft 45.4 %, 3-draft 35.6 %, 4-draft 28.3 %. Recommended config: **3 draft tokens**.
- Cost savings: 19.4 % reduction in $/1M output tokens (SWE-bench).
- Concurrency tested: 1, 5, 25, 50, 100, 200. TP=1 and TP=2.

### 8.3 Head-to-head verdict

| Dimension | SGLang | vLLM |
|---|---|---|
| Single-stream EAGLE-3 speedup ratio | **2.36×** (Llama-3.1-8B) | **1.10–1.27×** (gpt-oss-120B; different model) |
| Algorithms breadth | EAGLE/EAGLE3/MTP/STANDALONE/NGRAM/DFLASH | EAGLE/EAGLE3/MTP |
| Adaptive draft thresholds | **Adaptive Spec V2** | Not in 0.21 |
| EAGLE-3 + SWA | **Yes (v0.5.12)** | Not documented |
| EAGLE-3 + MLA (Kimi K2.5) | **Yes (v0.5.12)** | Not documented |
| Production guidance | docs.sglang.io/.../speculative_decoding | docs.vllm.ai/.../spec_decode |
| OOM recovery recipe | Documented (mem-fraction-static, draft tree shrink) | Less explicit |

**Caveat**: the model is different in the two head-to-head bullets above (Llama-3.1-8B vs gpt-oss-120B). What's invariant: SGLang's speculative subsystem has more knobs (Adaptive Spec V2, EAGLE-3 SWA, EAGLE-3 + MLA on Kimi). vLLM 0.21's EAGLE-3 is a solid second; **for novel model architectures (MLA / SWA / MTP combinations), SGLang lands the feature first**.

**Verdict**: **SGLang ≥ vLLM 0.21** on speculative decoding, with the gap widest on (a) novel architectures and (b) topology configurations (SWA, MLA, MTP). For Llama-class dense models the gap is narrower but still in SGLang's favor.

---

## 9. Three full launch commands

### 9.1 Chat-bf16 (Llama-3.1-8B on single GH200, jump-forward + RadixAttention)

```bash
# Pre-warm (one-time, run before launch)
pip install flashinfer-python flashinfer-cubin
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130

# Launch
docker run --rm -d --gpus all --shm-size 32g --ipc=host --name sglang-chat \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/flashinfer:/root/.cache/flashinfer \
  -v ~/.deep_gemm:/root/.deep_gemm \
  -e HF_TOKEN=$HF_TOKEN \
  -e TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800 \
  lmsysorg/sglang:latest-cu130 \
  python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 30000 \
    --dtype bfloat16 \
    --mem-fraction-static 0.88 \
    --enable-torch-compile \
    --enable-hierarchical-cache --hicache-ratio 2 \
    --hicache-mem-layout page_first \
    --hicache-write-policy write_through_selective \
    --hicache-io-backend kernel \
    --grammar-backend xgrammar \
    --max-running-requests 256

# Warm up first request synchronously (triggers sgl-flash-attn3 JIT)
until curl -sf http://localhost:30000/health; do sleep 5; done
curl -sf http://localhost:30000/generate -d '{"text":"Hello","sampling_params":{"max_new_tokens":8}}' > /dev/null
echo "Server warm; open to traffic."
```

### 9.2 Batch-throughput (Llama-3.1-70B bf16 on single GH200 144 GB, EAGLE-3 + HiCache L2)

```bash
# One-time DeepGEMM pre-warm (skips the 10–20 min cold start later)
docker run --rm --gpus all --shm-size 32g --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.deep_gemm:/root/.deep_gemm \
  -e HF_TOKEN=$HF_TOKEN \
  lmsysorg/sglang:latest-cu130 \
  python3 -m sglang.compile_deep_gemm \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 1 --trust-remote-code

# Launch
docker run --rm -d --gpus all --shm-size 32g --ipc=host --name sglang-bt \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/flashinfer:/root/.cache/flashinfer \
  -v ~/.deep_gemm:/root/.deep_gemm \
  -e HF_TOKEN=$HF_TOKEN \
  -e TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800 \
  lmsysorg/sglang:latest-cu130 \
  python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --host 0.0.0.0 --port 30000 \
    --dtype bfloat16 \
    --mem-fraction-static 0.92 \
    --attention-backend fa3 \
    --enable-hierarchical-cache --hicache-ratio 3 \
    --hicache-mem-layout page_first \
    --hicache-write-policy write_through_selective \
    --hicache-io-backend kernel \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 8 \
    --max-running-requests 512 \
    --chunked-prefill-size 8192

# Warm up
until curl -sf http://localhost:30000/health; do sleep 5; done
curl -sf http://localhost:30000/generate -d '{"text":"Hello","sampling_params":{"max_new_tokens":8}}' > /dev/null
```

Notes:
- 70B bf16 ≈ 140 GB weights + KV — only fits on the **144 GB SKU**. On the 96 GB SKU you must INT4 quantize or shard across two GH200s.
- `--hicache-ratio 3` reserves ~432 GB of LPDDR — leave ~48 GB for OS. If you see OOM during boot, drop to ratio 2.
- EAGLE-3 with the listed draft model is a per-Llama-3.1 family choice; for Llama-3.3-70B swap the draft model accordingly.

### 9.3 P/D split (two GH200s, Llama-3.1-70B bf16, Mooncake over IB, optional NVL2 NVLink-C2C)

```bash
# === Per-node common ===
export HF_TOKEN=...
export MODEL=meta-llama/Llama-3.1-70B-Instruct
export PREFILL_IP=10.0.0.10
export DECODE_IP=10.0.0.11
# Set BOTH lines below ONLY if both GH200s are on a shared NVLink switch (NVL2/NVL32):
export MC_INTRANODE_NVLINK=1
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
# Otherwise, leave defaults (BAREX / RDMA over IB).

# === Node 0 (prefill, GH200 #0) ===
docker run --rm -d --gpus all --shm-size 32g --ipc=host --network host \
  --privileged --ulimit memlock=-1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.deep_gemm:/root/.deep_gemm \
  -e HF_TOKEN -e MC_INTRANODE_NVLINK -e SGLANG_MOONCAKE_CUSTOM_MEM_POOL \
  -e SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=12 \
  -e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 \
  -e TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800 \
  --name sglang-prefill \
  lmsysorg/sglang:latest-cu130 \
  python3 -m sglang.launch_server \
    --model-path $MODEL \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device mlx5_0 \
    --prefill-only-disable-kv-cache \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --mem-fraction-static 0.92 \
    --chunked-prefill-size 16384 \
    --host 0.0.0.0 --port 30000

# === Node 1 (decode, GH200 #1) ===
docker run --rm -d --gpus all --shm-size 32g --ipc=host --network host \
  --privileged --ulimit memlock=-1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.deep_gemm:/root/.deep_gemm \
  -e HF_TOKEN -e MC_INTRANODE_NVLINK -e SGLANG_MOONCAKE_CUSTOM_MEM_POOL \
  -e SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL=5 \
  -e SGLANG_DISAGGREGATION_WAITING_TIMEOUT=300 \
  -e TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800 \
  --name sglang-decode \
  lmsysorg/sglang:latest-cu130 \
  python3 -m sglang.launch_server \
    --model-path $MODEL \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device mlx5_0 \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --mem-fraction-static 0.88 \
    --enable-hierarchical-cache --hicache-ratio 3 \
    --hicache-mem-layout page_first \
    --hicache-write-policy write_through_selective \
    --hicache-io-backend kernel \
    --max-running-requests 512 \
    --host 0.0.0.0 --port 30001

# === Router (separate small node) ===
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://$PREFILL_IP:30000 \
  --decode  http://$DECODE_IP:30001 \
  --host 0.0.0.0 --port 8000

# === Synchronous warm-up before opening to traffic ===
for url in http://$PREFILL_IP:30000/health http://$DECODE_IP:30001/health; do
  until curl -sf "$url" > /dev/null; do sleep 5; done
done
curl -sf http://localhost:8000/v1/completions -d '{
  "model":"'$MODEL'","prompt":"Hello","max_tokens":8
}' > /dev/null
echo "PD pair warm."
```

Notes:
- **No GPU Staging Buffer** here because TP is homogeneous (1=1). Add the three `SGLANG_DISAGG_STAGING_*` env vars only if you change to heterogeneous TP, and **never** for MLA models.
- `--prefill-only-disable-kv-cache` on the prefill side is the v0.5.12 micro-win: skips the KV pool allocation entirely on the prefill node.
- IB device name (`mlx5_0`) depends on host — confirm with `ibstat`.
- For DeepSeek V3/V4 across two GH200s, swap to `--attention-backend flashmla` and add `--moe-a2a-backend deepep --enable-dp-attention --tp 2 --dp 2` (TP across both GH200s; this changes shape from a P/D split to a unified multi-node serve).

---

## 10. Confidence and gaps after Round 4

### High confidence
- v0.5.12 / v0.5.12.post1 feature list and new flags (release notes verified).
- Pre-warm recipe — DeepGEMM `compile_deep_gemm` cuts 10–20 min to ~1 sec (LMSYS docs + bug #4773 / #9867 / #10733).
- XGrammar overhead < 40 µs/token and 3× vLLM on JSON (LMSYS Twitter + SqueezeBits + Particula).
- SGLang ≥ vLLM 0.21 on speculative decoding ratio (SGLang 2.36× vs vLLM 1.1–1.27× on different models; SGLang has Adaptive Spec V2 + EAGLE-3 SWA + EAGLE-3 MLA).
- HiCache L2 design + flags + write policies (docs verified).
- PD config for two-GH200 pair (docs + release notes verified).

### Medium confidence
- GH200 HiCache L2 sizing (`--hicache-ratio 2–4` for 144 GB, 4 for 96 GB) — extrapolated from 480 GB LPDDR less OS overhead; not benchmarked.
- DeepSeek V4 day-0 numbers extrapolated to GH200 from H200 (HBM3e parity argument).
- `INTRA_NODE_NVLINK` Mooncake pool benefit on NVL2 GH200 pair — documented capability, no published benchmark.
- TileLang ARM64 backend irrelevant for GH200 — confirmed by reading: TileLang generates CUDA on the GPU side; aarch64 only matters for NPU/CPU.

### Gaps (still open after four rounds)
- **No first-party GH200 SGLang throughput benchmark** anywhere — LMSYS, NVIDIA, sgl-project, Baseten, Lambda.
- **No measured HiCache L2 bandwidth on GH200** — third-party C2C STREAM numbers (375 / 297 GB/s) are the only data points.
- **No measured first-launch JIT compile time on aarch64 GH200** — only x86 lower bounds available.
- **Open Hopper FP8 bugs** (#24650 V4-Flash-FP8 + EAGLE RoPE crash; #18550 fa3+fp8_e5m2 silent triton; #23687 Qwen3.6 FP8 weight_scale_inv dropped) still listed open at last check.
- **veRL+SGLang+GH200 slowdown** (Issue volcengine/verl#1208) unresolved.

### Watch list
- `flashinfer-jit-cache` arm64 wheel availability — currently x86_64-only as far as I can tell. Track `flashinfer.ai/whl/cu130` listing.
- v0.5.12 "ARM CPU phase-1A CI bootstrap" — proper arm64 CI is just landing.
- v0.5.13 (unreleased) — watch for arm cubin distribution for `sgl-flash-attn3`.

---

## Sources (Round 4, new beyond Rounds 1–3)

Release notes:
- [SGLang v0.5.12 release notes (GitHub)](https://github.com/sgl-project/sglang/releases/tag/v0.5.12)
- [SGLang v0.5.12.post1 tag](https://github.com/sgl-project/sglang/releases/tag/v0.5.12.post1)
- [Issue #23639 — UnifiedRadix HiCache for DeepSeek V4](https://github.com/sgl-project/sglang/issues/23639) (closed Apr 2026)
- [Issue #20415 — Unified Hybrid Radix Cache Refactor roadmap](https://github.com/sgl-project/sglang/issues/20415)
- [Issue #23602 — DeepSeek V4 Roadmap](https://github.com/sgl-project/sglang/issues/23602)
- [DeepSeek-V4 cookbook (SGLang docs)](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)
- [DeepSeek V3/V3.1/R1 usage (SGLang docs)](https://docs.sglang.io/basic_usage/deepseek_v3.html)
- [Attention Backend reference (SGLang docs)](https://sgl-project.github.io/advanced_features/attention_backend.html)

HiCache + Mooncake:
- [HiCache design (SGLang docs)](https://docs.sglang.io/advanced_features/hicache_design.html)
- [LMSYS HiCache blog 2025-09-10](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)
- [PD Disaggregation (SGLang docs)](https://docs.sglang.io/advanced_features/pd_disaggregation.html)
- [LMSYS DeepSeek-V4 day-0 blog](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)

Speculative decoding + structured output:
- [Speculative Decoding (SGLang docs)](https://docs.sglang.io/advanced_features/speculative_decoding.html)
- [Red Hat — vLLM EAGLE-3 on gpt-oss (2026-04-16)](https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss)
- [Structured Outputs (SGLang docs)](https://docs.sglang.io/advanced_features/structured_outputs.html)
- [LMSYS — XGrammar integration tweet](https://x.com/lmsysorg/status/1861880264567443958)
- [LMSYS — Compressed FSM jump-forward (2024-02-05)](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/)
- [SqueezeBits — Guided decoding vLLM vs SGLang](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang)

JIT compile / pre-warm:
- [SGLang Dockerfile aarch64 branch](https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile)
- [Issue #4773 — Speed up DeepGEMM JIT compilation](https://github.com/sgl-project/sglang/issues/4773)
- [Issue #9867 — Long DeepGEMM v2 warmup → NCCL timeout](https://github.com/sgl-project/sglang/issues/9867)
- [Issue #10733 — Speed up DeepGEMM v2 warmup feature](https://github.com/sgl-project/sglang/issues/10733)
- [Issue #5817 — deep_gemm precompile in shared fs](https://github.com/sgl-project/sglang/issues/5817)
- [FlashInfer Installation docs (jit-cache, cubin)](https://docs.flashinfer.ai/installation.html)
- [vLLM Issue #41865 — FlashInfer GDN JIT multi-worker deadlock](https://github.com/vllm-project/vllm/issues/41865)
- [kernels-community/sgl-flash-attn3 on Hugging Face](https://huggingface.co/kernels-community/sgl-flash-attn3)

Benchmarks (RadixAttention + comparison):
- [Particula — SGLang vs vLLM 2026](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison)
- [Spheron — SGLang production guide 2026](https://www.spheron.network/blog/sglang-production-deployment-guide/)
- [Modal — Low-latency Qwen 3.5 with SGLang](https://modal.com/docs/examples/sglang_low_latency)
- [Modal — Serverless Qwen 3-8B with SGLang Snapshots](https://modal.com/docs/examples/sglang_snapshot)

GH200 NVLink-C2C background (no new sources — same Verda + arXiv references from Round 3 already cited).

TileLang context:
- [TileLang FlashMLA Hopper doc](https://tilelang.com/deeplearning_operators/deepseek_mla.html)
