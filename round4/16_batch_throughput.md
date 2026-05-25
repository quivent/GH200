# High-throughput batch LLM serving on GH200 — chunked prefill, continuous batching, prefix-cache tuning

**Agent**: #16 / Round 4 (LLM-only deep dive)
**Date**: 2026-05-25
**Builds on**: `round1/03_vllm.md`, `round3/03_vllm_verify.md`
**Hard prior (unchanged)**: bf16 on this GH200 stack — FP8 is **not** in scope here.
**Target workload**: Llama-3.3-70B-Instruct BF16, single GH200 96 GB HBM3 + 480 GB Grace LPDDR5X, throughput-first.

---

## TL;DR

| Knob | vLLM 0.21 | SGLang 0.5.x | TRT-LLM 1.2.x (PyTorch backend) |
|---|---|---|---|
| Concurrency cap | `--max-num-seqs 256-512` (default **1024**; cap to match real load) | `--max-running-requests` (no default; tune via #queue-req 50-500) | `--max_batch_size 512` (default 2048; **512 is the empirical sweet spot for 70B**) |
| Per-step token budget | `--max-num-batched-tokens 8192-16384` (default ~8192 V1) | `--chunked-prefill-size 8192` (default; lower to 4096/2048 only on OOM) | `--max_num_tokens 2048-8192` (default 8192; **2048 best in NVIDIA's 70B sweep**) |
| Context | `--max-model-len 32768-131072` (cap to real workload) | `--context-length` (model-derived) | `--max_seq_len` |
| KV mem fraction | `--gpu-memory-utilization 0.90-0.95` | `--mem-fraction-static 0.88-0.92` | `kv_cache_free_gpu_memory_fraction 0.90` (default) |
| Scheduler / stepping | `--async-scheduling` (default; **replaces `--num-scheduler-steps`** which is V0-only and obsolete) | `--schedule-policy lpm` (default; switch to `fcfs` only when no shared prefixes) | In-flight batching always on; no `num_scheduler_steps` analog |
| Prefix caching | always on in V1; no flag to enable, only to disable | `--enable-prefix-caching` (default on; uses RadixAttention) | `--enable_block_reuse` (Paged KV with block reuse) |
| Chunked prefill | always on in V1 (no flag) | always on; size = chunked_prefill_size | `--enable_chunked_context` (paged context attention) |

For Llama-3.3-70B BF16 on a single GH200, the no-thought-required throughput config is:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --cpu-offload-gb 60 \
  --kv_offloading_backend native --kv_offloading_size 200 \
  --enable-prefix-caching
```

This will saturate the H100 die at concurrency ≥ 64. Expect order-of-magnitude **3,000-5,000 aggregate output tok/s** at 100-200 concurrent ShareGPT-like requests (extrapolated from NVIDIA NIM 2×H100 fp8 TP2 numbers below — single GH200 BF16 is competitive because the 900 GB/s NVLink-C2C absorbs the larger BF16 weight footprint via offload).

---

## 1. vLLM 0.21 — optimal flags for max-throughput Llama-3.3-70B BF16 on single GH200

### 1.1 `--max-num-seqs` (default 1024 in V1; **256-512 for 70B on single GH200**)

- **V1 default raised from 256 (V0) to 1024**. On a single GH200 with a 70B BF16 model, 1024 is too aggressive — KV per sequence is large and you will preempt.
- Per the vLLM optimization docs: "If actual concurrency is smaller, setting this to a smaller number matching the max concurrency may improve performance and improve per-user latencies."
- For 70B BF16, **start at 256**. Sweep 128 / 256 / 512. The same recipe page at `docs.vllm.ai/projects/recipes/.../Llama3.3-70B` for Hopper says "defaults to a large value on high-memory GPUs; adjust downward if actual concurrency is lower."
- **Single GH200 specific**: because we offload weights to Grace LPDDR (60-80 GB out of ~140 GB at BF16), HBM-resident KV headroom is large — 256 is comfortably feasible. Going to 512 only helps if you actually have ≥256 in-flight requests.

### 1.2 `--max-num-batched-tokens` (default ~8192 V1; **8192-16384 for throughput**)

- The vLLM optimization page recommendation: **">8192, especially for smaller models on large GPUs"**. Smaller values (2048) prioritize ITL; larger values prioritize TTFT and aggregate throughput.
- Empirical sweep recommendation from the community: **sweep 8k → 64k**. One report (ShareGPT-style) showed peak at 98,304 with consistent gains as long as memory allows.
- For 70B BF16 on GH200, **8192 is the safe throughput default**. Going to 16384 yields marginal gains per vLLM recipe page; going past 32k starts to eat KV budget aggressively.
- Interacts with chunked prefill: a 32k-token prompt at `max-num-batched-tokens=8192` splits into 4 chunks, each scheduled alongside decodes. This is the V1 design point.

### 1.3 `--max-model-len` (**cap to real workload**)

- Default = model's native max (131k for Llama-3.3). Each unused token of headroom costs KV reservation per active sequence.
- Throughput rule: set `max-model-len` to `~max(prompt+output)` observed in production. For chat: 8k-16k. For RAG/code: 32k. Long-context summarization: 131k with offload.
- On GH200, the long-context tail is what `--kv_offloading_size` is for — keeps `max-model-len` flexible without crushing HBM.

### 1.4 `--gpu-memory-utilization` (default 0.90; **0.90-0.95**)

- "Increasing this percentage allocates more GPU memory for KV cache." Primary lever for reducing preemption (`vllm:num_preemptions_total`).
- On GH200 there is no other tenant on the GPU, so **0.92 is the right default**. Bump to 0.95 only after confirming no OOM under load.
- Below 0.90 is throughput-leaving territory unless you have driver overhead concerns.

### 1.5 `--num-scheduler-steps` — **OBSOLETE in V1**

- Old V0 multi-step scheduling (`--num-scheduler-steps 8` ran 8 decode passes before a GPU-CPU sync).
- **V1 replaces this with `--async-scheduling`** (default on in 0.19+). vLLM forum: "with V1 architecture, `num-scheduler-steps` and `enable-chunked-prefill` are not needed anymore" — chunked prefill and multistep are always active with token-budget dynamically split.
- **Do not pass `--num-scheduler-steps` to V1.** If you see a 2026 tutorial recommending it, the tutorial is for V0 and is wrong on V1.
- The Red Hat 0.8.1 V1 writeup confirms V1 matches V0+multi-step at the upper bound and beats it everywhere else, without manual tuning.

### 1.6 Other 0.21 throughput knobs worth knowing

- `--async-scheduling` — default on; do not disable except to debug.
- `--enforce-eager` — disables CUDA graphs, halves throughput. Debug-only.
- `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'` — default in V1; full graph for uniform decode, piecewise for mixed batches.
- `-O2` — default optimization level. `-O3` is more aggressive but slower startup; `-O0` is fastest startup, no JIT.
- `--cpu-offload-gb 60-80` — spills weights into Grace LPDDR over C2C. Without this, 70B BF16 (~140 GB) does not fit in 96 GB HBM.
- `--kv_offloading_backend native --kv_offloading_size 200` — second-tier KV offload into Grace LPDDR (0.21 routes through Hybrid Memory Allocator).

---

## 2. SGLang 0.5.x equivalents

### 2.1 `--max-running-requests`

- **No documented default** in the tuning guide; SGLang sizes from `--mem-fraction-static` and KV cache pool.
- Decrease if you see OOM during decode. Target `#queue-req` in the 50-500 range (per `docs.sglang.io/advanced_features/hyperparameter_tuning.html`).
- Functionally equivalent to vLLM's `--max-num-seqs`.

### 2.2 `--chunked-prefill-size` (default **8192**)

- Confirmed default 8192 in recent server logs (e.g. `chunked_prefill_size=8192, max_prefill_tokens=16384`).
- If OOM during prefill, drop to 4096 or 2048 — saves memory, slows long-prompt prefill.
- **Activation memory is proportional to `chunked_prefill_size`** and CUDA-graph memory is proportional to `cuda_graph_max_bs`. SGLang's internal estimation formula: `reserved = chunked_prefill_size * 1.5 + cuda_graph_max_bs * 2`.
- The "piecewise CUDA graph max tokens is set to chunked-prefill-size by default" — these knobs co-vary.

### 2.3 `--schedule-policy` (default **lpm**)

- `lpm` (longest prefix match) — default; best for workloads with shared prefixes (system prompts, few-shot, chat). Higher scheduling overhead.
- `fcfs` (first-come-first-serve) — use when **no** shared prefixes exist, OR when shared-prefix requests are dispatched together. Lower scheduling overhead.
- Rule of thumb: chat / RAG / few-shot → leave at `lpm`. Pure-streaming generation with no prefix overlap → `fcfs`.

### 2.4 `--schedule-conservativeness` (recommended range **0.3 - 1.3**)

- Decrease toward 0.3 if `token_usage < 0.9` and server is conservative (under-utilizing KV cache).
- Increase toward 1.3 if seeing frequent OOM at high `token_usage`.

### 2.5 Throughput tip: prefer DP over TP when memory allows

- SGLang tuning guide: "**Data parallelism (`--dp-size`) is preferred for throughput** when GPU memory allows." This is consistent with the general principle (§6) that TP communication eats throughput; on multi-GH200 (NVL2), if the model fits per node, run replicas.

### 2.6 Memory split

- `--mem-fraction-static` — fraction of GPU memory reserved for static model weights + buffers. Lower if KV cache pool is too small; raise if OOM. No documented default in the tuning guide.

### 2.7 Concrete SGLang launch for 70B on single GH200 (extrapolated)

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --tp-size 1 \
  --chunked-prefill-size 8192 \
  --max-running-requests 256 \
  --schedule-policy lpm \
  --mem-fraction-static 0.88 \
  --enable-metrics
```

(Note: SGLang's CPU-offload story for weights is less mature than vLLM's `--cpu-offload-gb`; 70B BF16 single-GPU may force the SGLang user toward FP8 or a smaller model. See round3 SGLang investigation for that side. For this doc the constraint is **bf16 only** so vLLM's offload path is the more natural fit for the single-GH200 70B-BF16 case.)

---

## 3. TRT-LLM 1.2.1 PyTorch backend in-flight batching config

### 3.1 PyTorch backend status

- Default since TensorRT-LLM 1.0 (per release notes). `trtllm-serve --backend pytorch ...` — **no engine build step required**.
- 1.1 added: guided decoding ↔ spec-decode integration.
- 1.2 added: DGX Spark beta, FP16/FP8/NVFP4 validation across more models.
- 1.2.1 (May 2026): patch release; no major architectural changes — pin for stability.

### 3.2 In-flight batching (IFB)

- **Always on by default** in 1.x. Documented in NVIDIA's TRT-LLM blog: "in-flight batching activated by default as a runtime configuration parameter."
- Pipeline-parallel performance was improved in 1.x when IFB is on.
- No CLI flag to disable; it is the runtime model.

### 3.3 Tuning sweet spot for Llama-3.3-70B (NVIDIA's documented sweep)

The TRT-LLM "Tuning Max Batch Size and Max Num Tokens" guide ran a 4×H100 sweep on Llama-3.3-70B (FP8). It is the only published Llama-3.3-70B sweep on Hopper with these knobs:

| `max_batch_size` | tok/s |
|---|---|
| 64 | 1,944 (bottlenecked) |
| **512** | **2,467 (sweet spot)** |
| 2048 (default) | 2,044 |

At `max_batch_size=512`, sweeping `max_num_tokens`:

| `max_num_tokens` | tok/s |
|---|---|
| **2048** | **2,474 (best)** |
| 8192 | 2,467 |
| 16384 | 2,461 |

Tuned baseline → tuned: **+58% throughput (1,564 → 2,474 tok/s)**.

The takeaway is contrarian relative to vLLM/SGLang folk wisdom: **for 70B-class models on Hopper, `max_num_tokens=2048` matches or beats 8k/16k** under TRT-LLM's scheduler. The vLLM analog is to consider sweeping `--max-num-batched-tokens` **down** from 8192 to 4096/2048 for ITL-sensitive workloads, not just up.

### 3.4 KV cache config

- `kv_cache_free_gpu_memory_fraction` — default **0.90**. Bump to 0.95 if no OOM observed.
- `enable_block_reuse: true` for prefix caching with paged KV.
- For PyTorch backend, perf statistics are beta; enable iteration stats via YAML (`enable_iter_perf_stats: true`).
- **Known issue (NVIDIA dev-forum)**: TRT-LLM PyTorch memory fraction has been observed silently capped at 0.14 despite explicit configuration — watch for that.

### 3.5 Trtllm-serve launch (PyTorch backend, single GH200, 70B BF16)

```bash
trtllm-serve serve meta-llama/Llama-3.3-70B-Instruct \
  --backend pytorch \
  --max_batch_size 512 \
  --max_num_tokens 2048 \
  --max_seq_len 32768 \
  --kv_cache_free_gpu_memory_fraction 0.90
```

Gap: NVIDIA's published sweep is FP8 4×H100. Single GH200 BF16 PyTorch-backend TRT-LLM numbers are not in the public record.

---

## 4. Prefix-cache hit-rate measurement — instrumentation and expectations

### 4.1 vLLM V1 metric names (changed from V0)

V1 replaced the V0 hit-rate gauge with **counters**:

| Metric | Type | Meaning |
|---|---|---|
| `vllm:prefix_cache_queries` (a.k.a. `_total`) | Counter | total prefix-cache lookups |
| `vllm:prefix_cache_hits` (a.k.a. `_total`) | Counter | of those, how many hit |
| `vllm:kv_cache_usage_perc` | Gauge | fraction of KV blocks in use (0-1) |
| `vllm:num_requests_running` | Gauge | currently processing |
| `vllm:num_requests_waiting` | Gauge | queued (TTFT inflator if > 0) |
| `vllm:num_preemptions_total` | Counter | KV evictions; high = need to drop max-num-seqs |
| `vllm:gpu_prefix_cache_hit_rate` | Gauge (PR #12592, V1) | direct hit-rate gauge, available in recent V1 |

**Prometheus query for live hit rate:**

```promql
rate(vllm:prefix_cache_hits_total[1m]) / rate(vllm:prefix_cache_queries_total[1m])
```

Use a 1-minute rate window so the ratio is responsive but not noisy. Endpoint: `http://<host>:8000/metrics`. (Deprecated in V1: `vllm:num_requests_swapped`, `vllm:cpu_cache_usage_perc`, `vllm:cpu_prefix_cache_hit_rate` — the V0 swap subsystem is replaced.)

### 4.2 SGLang metrics

| Metric | Type |
|---|---|
| `sglang:cache_hit_rate` (a.k.a. `sglang_cache_hit_rate`) | Gauge — direct ratio 0.0-1.0 |
| `sglang:cached_tokens_total` | Counter |
| `sglang:token_usage` | Gauge — KV utilization |
| `sglang:num_running_reqs` | Gauge |
| `sglang:num_queue_reqs` | Gauge |
| `sglang:num_used_tokens` | Gauge |
| `sgl_router_cache_hits_total` / `_misses_total` | Counters (router-level) |

Enable with `--enable-metrics`; default endpoint `:30000/metrics`. Quick check: `curl host:30000/metrics | grep sglang_cache_hit_rate`.

### 4.3 Expected hit rates by workload

From multiple production reports (SGLang RadixAttention data, generalizable across engines):

| Workload | Expected hit rate |
|---|---|
| Few-shot classification (fixed system prompt + few examples) | **85-95%** |
| Multi-turn chat (long shared system + history) | **75-90%** |
| Code-analysis assistant | **60-80%** |
| Mixed production traffic | **50-70%** |
| RAG with templated query envelope | **50-80%** |
| Pure-streaming generation, no shared prefix | **0-5%** (RadixAttention contributes nothing) |

**Healthy minimum: > 0.3 for any chat-shaped workload.** Below 0.3 = either workload truly has no shared structure, or system-prompt drift (a single character mismatch in a templated prompt is a full miss). The kuncoro.io SGLang guide flags this explicitly.

### 4.4 Interpretation rubric (KV usage × hit rate)

From the vLLM forum diagnostic guide:

| KV usage | Hit rate | Interpretation |
|---|---|---|
| High (>90%) | High (>50%) | Cache is **earning** — best case |
| High | Low | Workload doesn't reuse — RadixAttention is overhead; consider `--no-enable-prefix-caching` if confirmed |
| Low | High | Plenty of headroom; you can raise `--max-num-seqs` |
| Low | Low | Bottleneck is elsewhere (NCCL, prefill, etc.) — caching is not the issue |

Example from real logs cited in the vLLM forum: usage 92.8-99.8%, hit rate saturated at 55-62% — that's the "earning, near saturation" mode where bumping KV budget (raise GPU memory util / offload size) helps.

---

## 5. CUDA-graph capture sizes — automatic vs manual

### 5.1 Default automatic detection (vLLM V1)

When `cudagraph_capture_sizes` is not specified:

- **Default size list**: `[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_cudagraph_capture_size + 1, 16))`
- **`max_cudagraph_capture_size` default**: `min(max_num_seqs * 2, 512)` — avoids OOM in tight memory and avoids capturing >512 graphs (startup time bloat).
- **`cudagraph_mode` default**: `FULL_AND_PIECEWISE` (full CUDA graph for uniform-decode batches, piecewise for mixed prefill+decode batches; piecewise leaves attention eager).

For `--max-num-seqs 256` (our recommendation): `max_cudagraph_capture_size = min(512, 512) = 512`. Generated sizes: `[1, 2, 4, 8, 16, 24, ..., 248, 256, 272, ..., 496, 512]` — roughly 50 captured graphs. Capture takes seconds at startup, then near-zero overhead at runtime.

### 5.2 When to manually override

- **Long startup tax**: If you have `max-num-seqs 1024` and don't care about batches > 256, override:
  `--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256]}'`
- **Hybrid models / TOKENSPEED-MLA / DeepSeek-R1**: 0.21 adds backend-specific capture quirks; check the backend docs.
- **Spec decode**: vLLM 0.21 supports independent drafter attention backend selection; capture sizes apply per-model.
- **Debug**: `--enforce-eager` disables CUDA graphs entirely. **Halves throughput**; debug-only.

### 5.3 SGLang

- `cuda_graph_max_bs` — caps captured batch sizes. Memory cost ~2× per captured batch.
- Default piecewise CUDA graph max tokens = `chunked_prefill_size` (8192). So if you raise `chunked_prefill_size`, CUDA-graph memory grows.
- Disable with `--disable-cuda-graph` only if hitting capture-time JIT bugs (e.g. flashinfer JIT issues seen in SGLang #5389).

### 5.4 TRT-LLM

- CUDA graphs are part of the engine compilation. PyTorch backend uses torch.compile-style graph capture under the hood; tuned via `AutoTuner` (1.1+ generalizes tactic selection).

---

## 6. NCCL all-reduce overhead — when does TP=1 beat TP>1 on a single GH200?

### 6.1 The single-die GH200 case

- **A single GH200 superchip contains one Hopper H100 die.** TP within one die is not a thing — `--tensor-parallel-size 1` is the only option for a 1× GH200 deployment. There is no inter-GPU all-reduce because there is no second GPU.
- The `--disable-custom-all-reduce` flag is irrelevant at TP=1 (no all-reduce path is invoked). This is the bog-standard correct configuration.
- **NCCL is not in the hot path at TP=1.** All collectives are no-ops.

### 6.2 The interesting case: GH200 NVL2 (2-superchip) or two GH200s

If you do have two GH200 dies linked over NVLink (NVL2 / NVLink-C2C aggregate), then TP=2 becomes viable. Scaling efficiency depends on interconnect:

| Interconnect | Per-GPU scaling efficiency at TP=2 |
|---|---|
| **NVLink (H100 SXM, GH200 NVL2)** | **~0.92×** → 1.84× combined throughput |
| PCIe Gen5 | 0.70-0.78× → 1.4-1.56× combined |

NVLink-equipped TP=2 retains ~92% per-GPU. For 70B BF16 (140 GB weights) on 2× GH200 you can drop the offload entirely — fits on aggregate 192 GB HBM — and pick up the latency win.

### 6.3 When TP=1 beats TP>1 even with NVLink

- **If the model fits on one GPU.** Single-GPU TP=1 is always faster per-GPU than TP=2 for the same model, because all-reduce after every layer costs cycles.
- **High-throughput batch serving with many independent requests**: replicating (DP=2, TP=1 per replica) beats sharding (TP=2 across one engine). SGLang docs make this explicit. vLLM same principle.
- **Threshold rule**: choose TP only when the model's parameters + KV at target context don't fit on one die's HBM after offload. For 70B BF16 on a 96 GB GH200 with `--cpu-offload-gb 60-80`, **TP=1 wins**.

### 6.4 The TRT-LLM 70B 4×H100 sweep — counter-data point

NVIDIA's tuned TRT-LLM Llama-3.3-70B sweep used 4×H100 (TP=4) and got 2,474 tok/s tuned. NIM's 2×H100 fp8 TP2 hits 3,322 tok/s at 100 concurrency (single workload). These are different engines and workloads — not directly comparable — but the magnitude says: TP scales for 70B because the model doesn't fit on one H100 80GB at FP8 (it does at FP8 on 96 GB GH200). **GH200's 96 GB + Grace offload is the architectural reason TP=1 is competitive on GH200 when TP=2-4 was forced on H100.**

---

## 7. Throughput curves: tok/s vs concurrency

### 7.1 NVIDIA NIM Llama-3.3-70B reference numbers (closest published to GH200 case)

**Workload**: 500 input / 2000 output tokens. NIM uses TRT-LLM under the hood.

#### 2× H100 80G fp8 TP2

| Concurrency | TTFT (ms) | ITL (ms) | Throughput (tok/s) |
|---|---|---|---|
| 1 | 47.77 | 18.91 | 52.85 |
| 5 | 126.53 | 19.04 | 261.83 |
| 25 | 147.63 | 21.69 | 1,149.05 |
| 50 | 181.64 | 23.97 | 2,078.43 |
| 100 | 229.77 | 29.98 | 3,322.87 |
| 150 | 296.74 | 37.85 | 3,948.26 |
| 200 | 4,867.89 | 40.72 | 4,624.14 |

**Knee at concurrency ≈ 100-150.** TTFT explodes at 200 (queueing dominates).

#### 2× H200 141GB fp8 TP2

| Concurrency | TTFT (ms) | ITL (ms) | Throughput (tok/s) |
|---|---|---|---|
| 1 | 46.38 | 16.79 | 59.51 |
| 5 | 121.67 | 16.36 | 304.63 |
| 25 | 169.85 | 18.64 | 1,335.79 |
| 50 | 190.84 | 19.79 | 2,515.16 |
| 100 | 238.18 | 26.06 | 3,820.26 |
| 150 | 302.88 | 33.98 | 4,395.18 |
| 200 | 357.10 | 37.06 | 5,370.23 |
| 250 | 576.85 | 39.99 | 6,202.02 |

**H200 holds linear scaling further** because more HBM = more KV slots = less queueing. Peak ~6.2k tok/s at 250 concurrency. Knee shifts past 200.

### 7.2 Extrapolation to single GH200 BF16 vLLM V1 (interpolated, not measured)

There is **no published number** for Llama-3.3-70B BF16 vLLM V1 on a single GH200 (re-confirmed for round 4 — gap noted in round 3). Best we can do is interpolation:

| Concurrency | Estimated tok/s on 1× GH200 BF16 vLLM 0.21 |
|---|---|
| 1 | 35-50 (single-stream, weights via C2C add latency) |
| 4 | 130-180 |
| 16 | 500-700 |
| 64 | 1,800-2,400 |
| 256 | **3,500-5,000** (knee) |

Rationale: GH200 is one Hopper die (= ½ of 2×H100 NIM number); BF16 is ~1.5× more memory-bound than FP8 for decode; but C2C offload pays back via larger KV headroom. The single GH200 should beat single H100 by ~7× per the 2024 Lambda data, putting it close to 2×H100 in fp8 terms despite BF16. **Flag these as extrapolations** — round 2-3 noted this is the highest-leverage benchmark to actually run on our rig.

### 7.3 Engine-to-engine comparison at fixed Llama-3.3-70B 1×H100 (Spheron 2026)

| Concurrency | vLLM tok/s | TRT-LLM tok/s | SGLang tok/s |
|---|---|---|---|
| 1 | 120 | 130 | 125 |
| 10 | 650 | 710 | 680 |
| 50 | 1,850 | 2,100 | 1,920 |
| 100 | 2,400 | 2,780 | 2,460 |

Engine ranking on dense 70B: **TRT-LLM ≥ SGLang ≥ vLLM**, gap 5-13%. SGLang's RadixAttention edge appears mainly on prefix-heavy workloads (16,200 vs 12,500 tok/s on smaller 8B+ models, ~29% gap). On 70B dense the gap narrows to 3-5% (Spheron, particula.tech).

### 7.4 What concurrency is "saturated"?

Heuristic for 70B on a single Hopper die:

- **< 16 concurrent**: decode is memory-bound on weight reads; throughput scales nearly linearly with concurrency.
- **16-64 concurrent**: continuous batching kicks in; sub-linear but still strong scaling.
- **64-256 concurrent**: knee region; throughput plateaus.
- **> 256 concurrent**: queueing dominates TTFT (see 200-concurrency TTFT explosion in NIM 2×H100 fp8 table — 4,867 ms TTFT vs 297 ms at 150). Latency unacceptable; throughput marginal gain only.

For throughput-first deployments, **target 128-200 concurrent in-flight** as the operating point. Tune `--max-num-seqs` to match, set load-balancer admission control above that.

---

## 8. Spec decode interactions with high-concurrency throughput

(Cross-link to round3 §6.) Red Hat's 2026-04-16 gpt-oss EAGLE-3 benchmark on H200 PCIe shows **9.5%-20.5% throughput gain persisting through 200 concurrent requests** with `num_speculative_tokens=2` or `3` — directly contradicting the earlier folklore that spec-decode decays above 16 concurrent. **EAGLE-3 stays on at high throughput on Hopper.** For Llama-3.3-70B, use `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B` as the drafter.

---

## 9. Concrete observability stack to deploy alongside the serving engine

```yaml
# Prometheus scrape config (excerpt)
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['gh200-host:8000']
  - job_name: 'sglang'
    static_configs:
      - targets: ['gh200-host:30000']
```

**Grafana dashboards worth shipping:**

1. `rate(vllm:prefix_cache_hits_total[1m]) / rate(vllm:prefix_cache_queries_total[1m])` — target > 0.3 for chat.
2. `vllm:kv_cache_usage_perc` — target 0.7-0.95; > 0.98 sustained → preemption coming.
3. `vllm:num_requests_running` — should hit `--max-num-seqs` under load.
4. `vllm:num_requests_waiting` — > 0 sustained → admission control or scale-up needed.
5. `rate(vllm:num_preemptions_total[5m])` — > 0 sustained → reduce `--max-num-seqs` OR increase `--gpu-memory-utilization` / offload size.
6. P50/P95 of `vllm:time_to_first_token_seconds` and `vllm:time_per_output_token_seconds`.

Red Hat's "5 steps to triage vLLM performance" (2026-03-09) codifies this rubric.

---

## 10. Confidence map

**High confidence:**
- `--num-scheduler-steps` is V0-only; V1 uses `--async-scheduling` and the flag is obsolete.
- V1 `--max-num-seqs` default is 1024; for 70B BF16 on single GH200 cap to 256-512.
- Default `--max-num-batched-tokens` ~8192; throughput recommendation is ≥ 8192, sweep up to 64k.
- TRT-LLM 70B sweet spot on Hopper: `max_batch_size=512`, `max_num_tokens=2048` (NVIDIA's own sweep, 4×H100 FP8).
- V1 prefix cache metrics are counters (`prefix_cache_queries_total` / `prefix_cache_hits_total`); compute rate ratio in Prometheus.
- TP=1 is correct for a single GH200; TP>1 only makes sense across 2+ GH200s and even then DP+TP=1 often wins.
- NIM 2×H100 / 2×H200 throughput curves are the closest published proxy for GH200 (table §7.1).
- SGLang defaults: `chunked_prefill_size=8192`, `schedule-policy=lpm`.

**Medium confidence:**
- Specific tok/s extrapolation for single GH200 BF16 (§7.2 — interpolated, not measured).
- TRT-LLM PyTorch backend `kv_cache_free_gpu_memory_fraction` silent-cap-at-0.14 bug (NVIDIA dev-forum) — reproduction conditions unclear.
- vLLM 0.21 HMA interactions with offload+prefix-cache at very high concurrency — no published GH200 data.

**Low confidence / gaps:**
- **Llama-3.3-70B BF16 vLLM V1 on 1× GH200 tok/s at concurrency 1/4/16/64/256 — not in public record.** Persistent gap from rounds 1-3. Action item: benchmark in-house with `vllm bench throughput` against ShareGPT, publish numbers.
- TRT-LLM 1.2.1 PyTorch backend numbers on GH200 specifically — also not published.
- SGLang on GH200 with BF16 70B — the constraint of "no FP8" plus SGLang's weaker CPU-offload story may make it impractical without quantization. Round 3 SGLang doc has more.

---

## 11. Sources consulted (dated)

- vLLM Optimization & Tuning — https://docs.vllm.ai/en/stable/configuration/optimization/
- vLLM Llama-3.3-70B recipe (Hopper) — https://docs.vllm.ai/projects/recipes/en/latest/Llama/Llama3.3-70B.html
- vLLM CUDA Graphs design — https://docs.vllm.ai/en/stable/design/cuda_graphs/
- vLLM Engine Arguments — https://docs.vllm.ai/en/stable/configuration/engine_args/
- vLLM 0.21.0 release — https://github.com/vllm-project/vllm/releases/tag/v0.21.0 — 2026-05-15
- vLLM PR #12592 (GPU prefix cache hit rate gauge) — https://github.com/vllm-project/vllm/pull/12592
- vLLM forum "GPU KV cache usage & prefix cache hit rate" — https://discuss.vllm.ai/t/gpu-kv-cache-usage-prefix-cache-hit-rate/1366
- vLLM issue #16886 "deciding max-num-seqs and max-num-batched-tokens" — https://github.com/vllm-project/vllm/issues/16886
- vLLM forum "perf diff V0 num-scheduler-steps vs V1" — https://discuss.vllm.ai/t/what-is-the-perf-difference-between-v0-engine-num-scheduler-steps-vs-v1-engine/718
- vLLM "Inside vLLM" deep dive — https://www.aleksagordic.com/blog/vllm
- vLLM V1 metrics doc — https://docs.vllm.ai/en/v0.8.5/design/v1/metrics.html
- vLLM async_scheduler API doc — https://docs.vllm.ai/en/stable/api/vllm/v1/core/sched/async_scheduler/
- Red Hat "5 steps to triage vLLM performance" — https://developers.redhat.com/articles/2026/03/09/5-steps-triage-vllm-performance — 2026-03-09
- Red Hat "vLLM 0.8.1 V1 engine" — https://developers.redhat.com/articles/2025/04/28/performance-boosts-vllm-081-switching-v1-engine
- Red Hat "EAGLE-3 spec decode gpt-oss" — https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss — 2026-04-16
- SGLang Hyperparameter Tuning — https://docs.sglang.io/advanced_features/hyperparameter_tuning.html
- SGLang Server Arguments — https://docs.sglang.io/advanced_features/server_arguments.html
- SGLang Prometheus metrics guide — https://kuncoro.io/blog/sglang-prometheus-metrics-guide/
- SGLang Production Metrics — https://docs.sglang.io/references/production_metrics.html
- SGLang Production Deployment / RadixAttention — https://www.spheron.network/blog/sglang-production-deployment-guide/
- SGLang server_args.py source — https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py
- TRT-LLM tuning max_batch_size / max_num_tokens — https://nvidia.github.io/TensorRT-LLM/1.1.0rc2/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html
- TRT-LLM release notes — https://nvidia.github.io/TensorRT-LLM/release-notes.html
- TRT-LLM trtllm-serve docs — https://nvidia.github.io/TensorRT-LLM/1.0.0rc2/commands/trtllm-serve.html
- TRT-LLM PyTorch memory fraction issue (NVIDIA dev forum) — https://forums.developer.nvidia.com/t/tensorrt-llm-pytorch-memory-fraction-automatically-limited-to-0-14-despite-kv-cache-configuration/354807
- NVIDIA "Boost Llama 3.3 70B with TRT-LLM spec decode" — https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/
- NVIDIA NIM Llama-3.3-70B benchmark dashboard — https://docs.nvidia.com/nim/benchmarking/llm/latest/performance.html
- Spheron "vLLM vs TRT-LLM vs SGLang H100 benchmarks 2026" — https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/
- Particula "SGLang vs vLLM 2026" — https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison
- Cerebrium Llama-3.1 70B benchmark — https://cerebrium.ai/blog/benchmarking-vllm-sglang-tensorrt-for-llama-3-1-api
- morphllm vLLM 2026 benchmarks — https://www.morphllm.com/vllm-benchmarks
- ROCm vLLM V1 performance optimization — https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- CROZ "Tuning vLLM Vol. 1" — https://croz.net/run-your-own-ai-at-scale-vol-1-tuning-vllm/
- Lambda Labs GH200 vLLM 2024 baseline — https://lambda.ai/blog/putting-the-nvidia-gh200-grace-hopper-superchip-to-good-use-superior-inference-performance-and-economics — 2024-11-22
