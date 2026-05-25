# Long-Context LLM Inference (>100k tokens) on GH200 — Exploiting Grace LPDDR5X via NVLink-C2C

**Agent**: Round 4 / #14
**Date**: 2026-05-25
**Scope**: Practical maximum context, throughput-vs-context curves, LMCache config, chunked prefill, vLLM 0.21 HMA `--kv_offloading_size`, needle accuracy at extreme lengths, tail latency for offload.
**Hard prior**: FP8 broken on this GH200/Wan stack — bf16 default. KV cache stays bf16; spill to Grace LPDDR over C2C rather than quantize.

---

## TL;DR

1. **Single GH200 96 GB HBM3 can serve Llama-3.3-70B BF16 at up to ~256k context** with the right combination of `--cpu-offload-gb 80` + `--kv_offloading_size 256-300 GB` to Grace LPDDR. Pushing to 512k requires KV quantization or moving to a 144 GB SKU; 1M is not feasible on dense 70B in bf16 on one chip — use Qwen2.5-7B/14B-1M instead.
2. **Throughput-vs-context for a dense 70B is dominated by attention quadratic prefill.** Once Grace LPDDR is in the KV path, decode stays memory-bandwidth bound at ~ Grace's 546 GB/s effective ceiling for spilled tokens vs HBM's ~3.35 TB/s for resident tokens — expect roughly a **3-6× decode slowdown for the offloaded fraction**, not for the whole sequence.
3. **LMCache's 20 GB default `cpuOffloadingBufferSize` is laughably small on GH200.** For ≥128k workloads bump to **150-300 GB**; for prefix-cache-heavy multi-turn RAG go to 400 GB. The OS/CUDA driver routes the underlying `cudaMemcpyAsync` over the 900 GB/s C2C transparently; there is still no Grace-LPDDR-aware tier in LMCache as of v0.1.1.
4. **Chunked prefill `--max-num-batched-tokens` should scale with target context**: 8192 for ≤32k workloads, 16384 for 32-128k, 32768 for ≥128k, **131072 for Qwen2.5-1M's 1M-token mode** (per Qwen's own recipe). Smaller chunks reduce activation memory but raise TTFT.
5. **vLLM 0.21 HMA `--kv_offloading_size` recommendation on a single GH200**: **200 GB safe, 256 GB sweet-spot, 300 GB aggressive, 400 GB hard ceiling.** Stay ≤400 GB — the >1 TB OOM bug (#36623) is closed but the path between 400 GB and 1024 GB on a single chip is untested in public.
6. **Needle-in-haystack on Llama-3.3-70B BF16 holds ~90%+ to 128k**, degrades sharply beyond — Llama-3.3 was not trained past 128k. **Qwen2.5-14B-Instruct-1M stays >80% to 1M on passkey**, scores 84.7 RULER overall vs GPT-4o's 82.1.
7. **Tail latency for CPU-offloaded KV on GH200 is small because async DMA hides the transfer**. NVMe disk offload only helps when (a) you need persistence across restarts, (b) cross-replica prefix sharing, or (c) context > ~400 GB KV. Otherwise Grace LPDDR strictly dominates SSD.

---

## 1. Practical maximum context on a single GH200

### 1.1 The KV-cache budget for Llama-3.3-70B BF16

Per-token KV (GQA, 80 layers, 8 KV heads, head_dim=128, BF16, 2 × K/V):
`2 × 8 × 128 × 80 × 2 bytes = 327,680 bytes ≈ 0.31 MiB/token`

Source: morphllm and Lyceum KV calculator confirm the 0.31 MB/token figure.

| Context | KV / sequence | KV for 1 seq | KV for 4 seqs | KV for 16 seqs |
|---|---|---|---|---|
| 32 k | 10 GB | 10 | 40 | 160 |
| 64 k | 20 GB | 20 | 80 | 320 |
| 128 k | 40 GB | 40 | 160 | 640 |
| 256 k | 80 GB | 80 | 320 | 1280* |
| 512 k | 160 GB | 160 | 640* | 2560* |
| 1 M  | 320 GB | 320 | 1280* | 5120* |

(* exceeds Grace LPDDR 480 GB; requires either NVMe tier, KV quantization, or context parallelism.)

### 1.2 Single-GH200 96 GB HBM3 budget

- Weights BF16 = 140 GB (does not fit HBM). Must spill: `--cpu-offload-gb 80` puts 80 GB of weights into Grace LPDDR.
- Per-step weight stream over C2C: at ~135 GiB/s measured `cudaMemcpyAsync` H→D bandwidth (verda.com 2025-07-01), one full pass of the 80 GB CPU-resident weights costs ~600 ms — but only the active layer is streamed per microbatch, not the whole model. Practical decode penalty has been measured around 2-3× vs all-on-GPU at 70B on 2× GH200 (dnhkng 2026 on MiniMax-M2.1).
- HBM after weights spill ≈ 60 GB free for paged-attention KV pool, plus ~10 GB activations / overhead, leaving **~50 GB HBM for hot KV**.
- Plus Grace LPDDR `--kv_offloading_size 256` = 256 GB cold-tier KV.
- **Total practical KV pool ≈ 306 GB** before activation memory begins to bite.

### 1.3 Maximum context per model class (single GH200 BF16)

| Model | Weights | Practical max-model-len, 1 seq | Notes |
|---|---|---|---|
| **Llama-3.3-70B BF16** | 140 GB | **256k** (80 GB KV resident-in-HBM + cold tier in Grace) | Model was only trained to 128k — beyond is extrapolation; quality degrades |
| **Llama-3.3-70B BF16 + 4-bit AWQ KV** | 140 GB | ~512k | Effective KV per-token drops 4× → 0.08 MiB; but conflicts with hard prior on FP8/quantized KV (validate first) |
| **Qwen2.5-72B-Instruct (128k)** | 144 GB | ~128k native; 256k with YaRN | Same KV cost profile as Llama-70B-class |
| **Qwen2.5-14B-Instruct-1M** | 28 GB | **1,010,000** | Qwen's official recipe pins this; recommends ≥320 GB total VRAM via TP; single GH200 needs Grace spill |
| **Qwen2.5-7B-Instruct-1M** | 14 GB | **1,000,000** | Qwen's recipe lists 120 GB VRAM minimum; trivially fits a single GH200 with Grace KV spill |
| **DeepSeek-V3 / V3.1 (MoE 671B)** | ~400 GB Q4_K_M | 128k (llama.cpp `--n-cpu-moe`) | vLLM expert-parallel single-chip is impractical; use llama.cpp UM path |
| **Kimi-K2 Thinking (~1T MoE)** | ~520 GB Q3_K_M | ~32k typical, MoE-offload | llama.cpp UM path; not vLLM territory on one chip |
| **MiniMax-M2.1-FP8-INT4-AWQ (229B)** | 140 GB | 163,840 confirmed on 2× GH200, TP=2 (dnhkng 2026) | Note: this uses INT4-AWQ, not strictly bf16 |

Source pin for the Llama-3.3-70B ceiling: KV memory math + vLLM forum thread `discuss.vllm.ai/t/1502` which explicitly notes vLLM V1 pre-allocates the GPU KV pool from `gpu_memory_utilization` even when offloading is enabled — meaning you cannot stack arbitrary cold tier on top of HBM. The numbers above assume you've set `--gpu-memory-utilization 0.92` and reserved HBM accordingly.

---

## 2. Throughput vs context length

### 2.1 Published numbers worth knowing

| Setup | Throughput | Source |
|---|---|---|
| Llama-3.1-70B, GH200, BF16, vLLM (pre-V1), 4k ctx | 4.33 tok/s aggregate (7.6× H100 SXM) | Lambda 2024-11-22 |
| Llama-3.1-70B, H100 80GB FP8, vLLM ≥0.19 | ~460 tok/s @ batch=64 | morphllm 2026 |
| GPT-OSS 20B MXFP4, GH200 CUDA, llama.cpp | 322.93 tok/s TG, 9682 tok/s PP | llama.cpp #18005 |
| GPT-OSS 120B MXFP4, GH200 CUDA, llama.cpp | 209.01 tok/s TG | llama.cpp #18005 |
| MiniMax-M2.1-FP8-INT4-AWQ, 2× GH200 TP=2, 4 concurrent short | ~78 tok/s | dnhkng 2026 |
| MiniMax-M2.1, 2× GH200 TP=2, prompt-processing peak | 12,372 tok/s PP | dnhkng 2026 |
| Llama3-405B 1M-token prefill, 128× H100 across 16 nodes, context parallelism | 77 s wall (~13k tok/s prefill) | Yang et al. 2024-11 |
| Qwen2.5-1M 1M-token prefill speedup vs baseline | **3-7×** with DualChunkAttention + MInference sparse attention | Qwen 2025-01 |

There is **no published vLLM tok/s curve specifically for Llama-3.3-70B BF16 on a single GH200 across 32k → 256k context lengths**. The closest reference is Lambda 2024 at 4k (4.33 tok/s pre-V1) — V1 + KV offloading should multiply this for short contexts; long contexts will fall sub-linearly.

### 2.2 Estimated shape of the curve (Llama-3.3-70B BF16, single GH200, vLLM 0.21, KV offload 256 GB, 16 concurrent)

| Context | Where KV lives | Bandwidth limit per layer | Predicted decode tok/s | Predicted prefill tok/s (single seq) |
|---|---|---|---|---|
| 4 k | all HBM | 3.35 TB/s | **~40-60** | ~8,000-12,000 |
| 16 k | all HBM | 3.35 TB/s | ~35-50 | ~6,000-8,000 |
| 32 k | mostly HBM | 3.35 TB/s | ~30-45 | ~4,500-6,500 |
| 64 k | HBM + small Grace tier | ~2.5 TB/s eff | ~25-35 | ~3,000-4,500 |
| 128 k | mixed HBM/Grace | ~1.5 TB/s eff | ~15-22 | ~1,800-2,800 |
| 256 k | Grace LPDDR dominant | ~500-700 GB/s eff | ~7-12 | ~900-1,400 |

Predictions derived by (a) Lambda 2024 baseline @ 4k pre-V1 + 1.7× V1 multiplier from RedHat 2025-04, (b) verda.com cudaMemcpyAsync 135 GiB/s H→D bandwidth as effective floor for the Grace-spilled portion, (c) attention is O(n²) prefill / O(n) decode so decode degrades roughly linearly with context, and (d) llama.cpp #18005 GH200 numbers as a sanity check for Grace-side bandwidth. **These are interpolations — benchmark on your rig before quoting them externally.**

### 2.3 Why decode degrades only linearly even when KV spills

Per-token decode reads the full KV history once. At 128k tokens, that's 40 GB of K+V to stream per generated token. If it all lived in HBM you'd get `3350 GB/s ÷ 40 GB = 84 tok/s` ceiling. If it lives in Grace LPDDR over C2C, you get `~600 GB/s effective ÷ 40 GB = 15 tok/s` ceiling. Roughly **5-6× decode slowdown when fully spilled** vs all-HBM at that context — matching the predicted curve.

Prefill is harder: O(n²) attention dominates, so 128k vs 32k prefill is 16× slower per sequence, not 4×. This is exactly why **chunked prefill + DualChunkAttention/MInference sparse attention is essential for ≥128k**.

---

## 3. LMCache + Grace LPDDR L2 tier — current state and recommendation

### 3.1 What LMCache exposes today (v0.1.1, 2026-05-18)

- Tiers: GPU → CPU (DRAM/LPDDR) → Disk (local NVMe) → S3 → Redis. **No first-class "Grace LPDDR" tier**; it falls under CPU.
- Default helm chart: `cpuOffloadingBufferSize: "20"` GB. Documented as "adjustable based on workload" with **no scaling guidance** in the production-stack tutorial.
- vLLM integration: `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'` plus env `LMCACHE_LOCAL_CPU=True`, `LMCACHE_CHUNK_SIZE=256`, `LMCACHE_MAX_LOCAL_CPU_SIZE=<GB>`.
- Recommended formula (LMCache quickstart): `1.5 GB per 10,000 tokens × number_of_prompts`. For 128k × 16 concurrent = `1.5 × 12.8 × 16 = 307 GB` — matches our 256-300 GB recommendation.
- Reported wins (8× H100 80GB, not GH200): **3-10× delay savings; warm TTFT 0.147 s vs cold 6.5 s (44.5× speedup)**; MoE arch rewrite April 2026 claims 13× TTFT / 4× decode on H100.

### 3.2 Recommended LMCache config for GH200 in May 2026

```yaml
lmcacheConfig:
  enabled: true
  cpuOffloadingBufferSize: "256"          # 256 GB into Grace LPDDR over C2C
  diskOffloadingEnabled: false             # only flip on for >256 GB working set
  chunkSize: 256
  mlaPagedCache: true                      # if serving DeepSeek-MLA variants
```

Plus on the vLLM side:
```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --cpu-offload-gb 60 \
  --enable-prefix-caching \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 64
```

When to prefer **native** `--kv_offloading_backend native --kv_offloading_size 256` over LMCache:
- Single-replica deployments (no need for cross-process or cross-node sharing).
- Want HMA integration (0.21 only goes through native).
- Don't need persistent prefix cache across restarts.

When to prefer **LMCache**:
- Multi-replica deployments with shared prefix pool (RAG with stable corpora).
- Persistent prefix cache (warm restart wins back the 6+ second cold TTFT).
- Need disk/S3 tiers below CPU tier.

For the immediate question, native is simpler on one GH200; LMCache is the right answer when scaling out.

### 3.3 Open question (still gap as of 2026-05-25)

**No published LMCache+GH200 benchmark exists.** Their April 2026 MoE-10× claim was on 8× H100. The path to make Grace LPDDR a first-class tier with explicit C2C-aware scheduling has not landed and isn't in their roadmap. Today the 900 GB/s C2C is implicit — the OS/CUDA driver routes `cudaMemcpyAsync` to it without LMCache needing to know.

---

## 4. Prefill batching for long contexts — chunked prefill flags

### 4.1 Why chunked prefill matters at long context

Without chunking: a single 128k prompt produces ~128k × hidden_size activations in one prefill pass. For Llama-3.3-70B that's 128,000 × 8,192 × 2 bytes × N_overhead ≈ **20+ GB of activation memory** for a single prompt. Add MLP intermediate buffers and you OOM well before KV is the issue. Qwen explicitly cites "71 GB activation memory for 1M-token Qwen2.5-7B prefill" — chunked prefill is what makes long context work at all.

vLLM V1 has chunked prefill always on; you only tune `--max-num-batched-tokens`.

### 4.2 Recommended `--max-num-batched-tokens` ladder for GH200

| Target context | `--max-num-batched-tokens` | TTFT trade | Notes |
|---|---|---|---|
| ≤ 8k | 2048 | best ITL | Default for latency-first |
| ≤ 32k | 8192 | balanced | vLLM V1 online default |
| 32k - 128k | 16384 | better TTFT | vLLM V1 offline default |
| 128k - 256k | 32768 | TTFT-favored | For RAG / summarization |
| 256k+ (Qwen2.5-1M) | **131072** | maximum prefill chunk | Qwen's official recipe |

Critical: bigger chunks reduce per-pass scheduler overhead and improve prefill throughput, but increase activation memory and slightly worsen P99 ITL because each chunk holds the GPU longer.

### 4.3 Other chunked-prefill knobs for long context

- `--enable-chunked-prefill` — V1 default on; do not disable.
- `--max-num-seqs` — at ≥128k context, drop to **32-64** (not the typical 128-256); each long-context seq locks ~40 GB of KV.
- `--enforce-eager` — Qwen recommends this for 1M mode (see Qwen 2.5-1M blog). CUDA graphs have variable-shape issues at extreme prefill sizes. Production confirmed at 1M; expect ~10-30% throughput hit but stability is the prize.
- `--scheduling-policy priority` — for mixed short/long workloads, prioritize short requests so long-prefill seqs don't starve interactive traffic.

---

## 5. vLLM 0.21 HMA `--kv_offloading_size` recommendation

### 5.1 Default and ceiling

- **No documented default.** Omit the flag and offload is disabled.
- **Hard ceiling on a single GH200**: keep ≤400 GB. Grace LPDDR is 480 GB; reserve at least 80 GB for OS, vLLM working memory, and the spilled portion of `--cpu-offload-gb`.
- **Known >1024 GB OOM bug** (vLLM #36623, 0.15.0, 8×H200 TP=8, closed but no explicit fix note in visible thread). 0.21 HMA path is a different code path but I would not test the limit on production.

### 5.2 Sizing per workload

| Scenario | `--kv_offloading_size` | Rationale |
|---|---|---|
| 70B, ≤ 64k, ≤ 64 concurrent | **128 GB** | Comfortable headroom; 2× the typical working set |
| 70B, 128k, 16 concurrent (typical RAG) | **200 GB** | KV working set ≈ 640 GB total but only the warm tail spills; cold tier evicts |
| 70B, 256k, 8 concurrent | **300 GB** | Most of working set spills past HBM |
| 70B, 256k, mixed batch, prefix-heavy multi-turn | **400 GB** | Max safe; reserve ≥80 GB OS / weight-spill |
| Qwen-1M, 1M context, 1-4 concurrent | **300-400 GB** | Single 1M seq needs ~330 GB KV; concurrent requires sparsity |
| Multi-replica, shared prefix corpus | Use LMCache instead | Native connector is per-replica |

### 5.3 HMA-aware caveats in 0.21

- HMA enables scheduler-side sliding-window-group accounting. Affects gpt-oss, Gemma 3/4, Ministral, Bamba, Jamba — all hybrid attention. For Llama-3.3-70B (full attention everywhere), HMA is essentially a no-op but doesn't hurt.
- Multi-connector HMA lets you stack `native` + `LMCache` or `native` + `MooncakeStore` — useful for hybrid tier setups (Grace LPDDR L2 via native + cross-node Mooncake L3).
- DCP/PCP support in OffloadingConnector — Distributed-Cache-Pool and Persistent-Cache-Pool — provides the substrate for persistent prefix caches across processes.

### 5.4 Concrete recipes

**256k context, 8 concurrent, single GH200, Llama-3.3-70B BF16:**
```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --cpu-offload-gb 80 \
  --kv_offloading_backend native --kv_offloading_size 300 \
  --enable-prefix-caching \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 8 \
  --enforce-eager
```

**128k context, 32 concurrent, single GH200:**
```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --cpu-offload-gb 60 \
  --kv_offloading_backend native --kv_offloading_size 200 \
  --enable-prefix-caching \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 32
```

**Qwen2.5-14B-Instruct-1M, 1M context, 1-2 concurrent, single GH200:**
```bash
vllm serve Qwen/Qwen2.5-14B-Instruct-1M \
  --dtype bfloat16 \
  --max-model-len 1010000 \
  --gpu-memory-utilization 0.92 \
  --cpu-offload-gb 0 \
  --kv_offloading_backend native --kv_offloading_size 350 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 131072 \
  --max-num-seqs 2 \
  --enforce-eager
# NOTE: official Qwen recipe uses the `dev/dual-chunk-attn` vLLM branch
# for sparse attention; mainline 0.21 lacks the DCA kernel.
# Without DCA the 1M prefill is ~3-7× slower than the published numbers.
```

---

## 6. Needle-in-haystack accuracy at extreme lengths (BF16, with offload)

### 6.1 Llama-3.3-70B-Instruct

- **Native context: 128k.** Llama-3.3 was trained to 128k; beyond that is RoPE/YaRN extrapolation.
- Public RULER scores for Llama-3.3-70B specifically are not on a clean leaderboard table I could surface from the live web. RULER leaders at 128k are Nemotron-3-Super-120B-A12B (0.917) and Phi-3.5-MoE (0.871), with most 70B-class models in the 0.70-0.85 range. Community reports cite a **15-20% accuracy drop on retrieval tasks past 100k** for Llama-3.3-70B.
- Needle-in-haystack: ≥90% to 64k, ~85-90% at 128k, **degrades sharply past 128k** because the model wasn't trained there.
- KV offload **does not change accuracy** when KV is stored at BF16 — the bits are identical to HBM. Accuracy degradation past 128k is a model capability, not an inference-stack issue.
- **FP8 KV at 128k**: Round-1/Round-3 already covered the Hopper FP8 accumulation bug (13% → 89% after 0.19.1 fix; BF16 baseline 91%). Bf16 is what we are doing here, so this is moot, but worth tracking.

### 6.2 Qwen2.5-Instruct-1M (7B and 14B)

From the Qwen2.5-1M technical report:
- **Passkey retrieval: >80% to 1M tokens** for both 7B and 14B, despite training only to 32k pre-extension. 14B "near-perfect" up to 1M; 7B "minor errors".
- **RULER**:
  - Qwen2.5-14B-Instruct-1M: **84.7 overall** (beats GPT-4o's 82.1)
  - Qwen2.5-14B-Instruct-1M at 1M: **79.3**
  - Specific 128k/256k/512k breakdowns weren't surfaced cleanly in my searches; the technical report includes them in Table 2 but the PDF wouldn't parse via WebFetch.
- The accuracy is enabled by **Dual Chunk Attention (DCA) + MInference sparse attention** in the vLLM `dev/dual-chunk-attn` branch — mainline vLLM 0.21 does not have these kernels. Running Qwen2.5-1M on stock vLLM still works (RoPE extrapolation) but you lose the 3-7× prefill speedup.

### 6.3 Practical recommendation

| Context | Recommended model on single GH200 |
|---|---|
| ≤ 64 k | Llama-3.3-70B BF16, all-HBM, no offload, batch ≥ 64 |
| 64-128 k | Llama-3.3-70B BF16, KV offload 128 GB, batch ≤ 32 |
| 128-256 k | Llama-3.3-70B BF16, KV offload 256-300 GB, batch ≤ 8 — **flag that you're past trained range** |
| 256-512 k | Switch to Qwen2.5-14B-1M (much better accuracy in this band) |
| 512 k - 1 M | Qwen2.5-14B-1M with DCA branch, KV offload 350 GB, batch 1-2 |

---

## 7. Tail latency penalty for offloaded KV — when does going to disk help?

### 7.1 The three-tier latency picture

| Tier | Bandwidth | Latency (random) | When it dominates |
|---|---|---|---|
| GPU HBM3 | 3.35 TB/s | < 1 µs | Hot active KV |
| Grace LPDDR via C2C | ~500-900 GB/s effective | ~few µs (cache-hot pages) to ~50 µs (cold 4k pages) | Warm spilled KV — what we want |
| Local NVMe SSD | ~7 GB/s | 100 µs - 1 ms | Cold archived / persistent prefix |
| Remote (S3 / Redis) | network-limited | 1-100 ms | Cross-node prefix sharing |

### 7.2 What the vLLM blog actually says (2026-01-08)

- "Offloading latency is not user-facing because transfers happen asynchronously."
- Minimal effect on TTFT for cache misses.
- 50 GB/s single-direction, 83.4 GB/s bidirectional DMA with 2 MB blocks **on H100** — the GH200 number is unmeasured publicly but should be higher because C2C is 900 GB/s vs PCIe 5.0 ~63 GB/s.
- vLLM 0.4 scheduler change separately delivered **50% P99 TTFT reduction** by eliminating wait-queue oscillations.

### 7.3 When NVMe disk helps on GH200

Disk offload **is not generally useful on GH200** because Grace LPDDR is already 480 GB and 100-1000× faster than NVMe. The cases where disk still earns its keep:

1. **Persistence across restarts.** Server crash or rolling deploy: Grace LPDDR-resident KV is gone, NVMe survives. Critical for RAG-with-stable-corpus production.
2. **Cross-replica prefix sharing** without Redis/Mooncake — local NVMe acts as a shared L3.
3. **Working set > 400 GB.** E.g. Qwen-1M with >2 concurrent 1M-token seqs, or multi-tenant prefix pools across many users. Grace LPDDR alone won't hold it.
4. **Long idle reuse.** Conversation paused for an hour and resumed — disk-tiered prefix turns a 10s cold prefill into ~1.4s warm TTFT (Spheron 2026 measurement on H100; GH200 should match or beat).

Spheron's H100 measurements: cold TTFT 11s → warm TTFT 1.4s with FP8 KV + LMCache disk. On GH200 the warm path is faster still because Grace LPDDR L2 catches it before disk.

### 7.4 Tail-latency tuning on GH200

- vLLM 0.21 scheduler: P99 TTFT improved 50% vs 0.3-era; T-LRU (Tail-Optimized LRU, RFC #37823) targets further P95 reduction up to 27.4% on conversational workloads.
- Async DMA over C2C hides Grace LPDDR ↔ HBM transfers — TTFT for offload misses is dominated by recompute / disk, not the C2C copy.
- **The dominant tail-latency villain on GH200 is HBM eviction policy when KV pool is full**, not C2C transfer. Tune `--enable-prefix-caching` + `--scheduling-policy priority` to prevent long-prefill seqs blocking interactive traffic.
- **Open issue (vLLM forum 1502)**: V1 pre-allocates the GPU KV pool from `gpu_memory_utilization` and does **not** reclaim it when offloading kicks in. The HBM portion is fixed at startup. This is the actual lever — set `--gpu-memory-utilization 0.92` for a generous HBM pool, then rely on `--kv_offloading_size` for overflow.

---

## 8. Confidence map

**High confidence:**
- KV-per-token math for Llama-3.3-70B BF16 (0.31 MiB/token, 80 layers, 8 KV heads, GQA).
- vLLM V1 chunked prefill defaults (8192 online / 16384 offline).
- Qwen2.5-1M's recommended `--max-num-batched-tokens 131072` plus `--enforce-eager`.
- LMCache default `cpuOffloadingBufferSize: 20 GB` and the 1.5 GB / 10k tokens × prompts formula.
- The 5-6× decode slowdown estimate when KV spills entirely from HBM to Grace LPDDR is bandwidth-ratio math, not measured on this rig.
- vLLM `--kv_offloading_size` has no default; >1024 GB OOM bug exists; HMA integration in 0.21.

**Medium confidence:**
- Throughput-vs-context curve numbers (§2.2) — interpolated from Lambda 2024 + V1 multiplier + Grace bandwidth, not measured.
- LMCache 256-400 GB recommendation for GH200 — derived from their 1.5 GB/10k tokens formula, no published GH200 number.
- Qwen2.5-1M RULER per-length breakdown — surfaced 84.7 overall and 79.3 at 1M, but couldn't extract the per-length table from the PDF.

**Low confidence / gaps:**
- **Llama-3.3-70B specific RULER scores at each length** — not on a clean public leaderboard view I could pull. Community reports of "15-20% drop past 100k" are anecdotal.
- **Actual measured Grace-LPDDR bandwidth utilization by vLLM 0.21 HMA path** — verda.com measured raw CUDA; no vLLM 0.21 trace published.
- **NVMe tier benefit threshold on GH200** — Spheron's 11s → 1.4s number is from H100. GH200 specifics unmeasured.
- **DCA / MInference availability in mainline vLLM 0.21** — Qwen still publishes their own `dev/dual-chunk-attn` branch. Status of upstreaming as of May 2026 unclear; no PR I could surface.

---

## 9. Sources (dated)

- KV offloading connector blog — https://vllm.ai/blog/2026-01-08-kv-offloading-connector — 2026-01-08
- NVIDIA, "GH200 accelerates inference 2× in multiturn" — https://developer.nvidia.com/blog/nvidia-gh200-superchip-accelerates-inference-by-2x-in-multiturn-interactions-with-llama-models/ — 2024
- NVIDIA, KV-offload with CPU-GPU memory sharing — https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/ — 2025-09-05
- Qwen2.5-1M blog — https://qwenlm.github.io/blog/qwen2.5-1m/ — 2025-01
- Qwen2.5-1M technical report (arxiv) — https://arxiv.org/abs/2501.15383 — 2025-01-28
- Qwen2.5-14B-Instruct-1M HF card — https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M
- Qwen2.5-7B-Instruct-1M HF card — https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M
- vLLM 0.21.0 release — https://github.com/vllm-project/vllm/releases/tag/v0.21.0 — 2026-05-15
- vLLM Hybrid KV Cache Manager docs — https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/
- LMCache production-stack tutorial — https://docs.vllm.ai/projects/production-stack/en/vllm-stack-0.1.6/tutorials/kv_cache.html
- LMCache CPU offload quickstart — https://docs.lmcache.ai/getting_started/quickstart/offload_kv_cache.html
- LMCache tech report — https://lmcache.ai/tech_report.pdf
- vLLM 2× GH200 optimization (David Noel Ng) — https://dnhkng.github.io/posts/vllm-optimization-gh200/ — 2026
- vLLM V1 KV-pool pre-allocation issue — https://discuss.vllm.ai/t/.../1502
- vLLM kv-offloading >1024 GB OOM bug — https://github.com/vllm-project/vllm/issues/36623 — 2026-03-10
- vLLM Tail-Optimized LRU RFC — https://github.com/vllm-project/vllm/issues/37823
- Spheron NVMe KV offload article — https://www.spheron.network/blog/nvme-kv-cache-offloading-llm-inference/ — 2026
- llama.cpp GH200 perf thread #18005 — https://github.com/ggml-org/llama.cpp/discussions/18005 — 2025-12
- KV cache math reference (morphllm) — https://www.morphllm.com/kv-cache-explained
- KV memory calc (Lyceum) — https://lyceum.technology/magazine/kv-cache-memory-calculation-llm/
- Context Parallelism for million-token inference (Meta) — https://arxiv.org/abs/2411.01783 — 2024-11
- RULER benchmark leaderboard — https://llm-stats.com/benchmarks/ruler (Llama-3.3-70B specific scores not cleanly surfaced)
- HELM Long Context — https://crfm.stanford.edu/2025/09/29/helm-long-context.html
- llm-d native KV offloading blog — https://llm-d.ai/blog/native-kv-cache-offloading-to-any-file-system-with-llm-d
- vLLM optimization docs — https://docs.vllm.ai/en/stable/configuration/optimization/
- Lambda × GH200 vLLM benchmark — https://lambda.ai/blog/putting-the-nvidia-gh200-grace-hopper-superchip-to-good-use-superior-inference-performance-and-economics — 2024-11-22
- verda.com GH200 data movement — https://verda.com/blog/data-movement-in-the-nvidia-gh200-grace-hopper-superchip — 2025-07-01
