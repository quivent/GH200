# vLLM 0.21.0 on GH200 — Round 4 LLM-Only Deep Dive

**Agent**: #01 / Round 4 (LLM-only)
**Date**: 2026-05-25
**Builds on**: round1/03_vllm.md, round3/03_vllm_verify.md
**Hard prior (unchanged)**: FP8 broken on DiT/Wan; **bf16 is the default for diffusion**. For dense Hopper LLMs FP8 is acceptable only when `head_dim ≤ 128` per the vLLM 0.19.1+ Flash-Attention #104 fix. This round is LLM-only — no diffusion.

This round drills past Round 3's verification into actionable specifics: exact flag syntax, kernel-level mitigations, mind-the-gap PR landings, and three full launch commands ready to paste.

---

## 0. Top-line LLM-only verdicts (TL;DR)

1. **Pin `vllm==0.21.0` (2026-05-15).** No 0.21.x patches yet; 0.22 not shipped. Default attention backend on SM90 (GH200) is `FLASH_ATTN` (FA3); `--attention-backend` is the new flag (the `VLLM_ATTENTION_BACKEND` env var was deprecated at 0.13).
2. **`head_dim=256` mitigation has shipped.** Flash-attention PR #122 (merged 2026-04-24) landed the "two-level accumulation every N steps" variant. With `N=4` head_dim=256 prefill goes from **328 → 748 TFLOPs/s (2.28×)** while the 128k-needle score only drops 0.89 → 0.88. PR #125 (merged 2026-03-16) tuned tile sizes for `head_dim ∈ {64,128,192}`. **This means: as of 0.21 the 1.6× penalty is fixable for head_dim=256 by enabling N-step accum.** It is a *compile-time* constant (`DFLASHATTENTION_FP8_TWO_LEVEL_INTERVAL`), not a runtime flag — so you either build vllm-flash-attention with `N=4` or accept the stock build's `N=1` (= full accuracy, slow prefill). Confirm by inspecting `vllm/attention/backends/flash_attn.py` after install.
3. **EAGLE-3 for Llama-3.3-70B is one JSON line.** Use `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B` with `method=eagle3, num_speculative_tokens=3, draft_tensor_parallel_size=1`. Red Hat's 9.5-20.5% throughput gain (concurrency 1→200) is on gpt-oss-120B MoE, **not** Llama-3.3-70B — replicability is "high confidence yes for low-medium concurrency, unmeasured >100" for dense 70B. Sweet spot stays at 2-3 draft tokens.
4. **KV-offload safe size on GH200 = 256 GB.** The OOM at >1024 GB on issue #36623 is a **PyTorch CachingHostAllocator power-of-2 rounding bug** — set 1025 → allocator reserves 2048 GB. For 480 GB Grace LPDDR the safe ceiling is therefore **anything ≤ 256 GB** (next round-up = 256 ≤ 480, fits). 257-511 rounds to 512 ≤ 480 = OOM. 512 GB exact = boundary case, avoid.
5. **MooncakeStoreConnector is overkill for a single GH200.** It's designed for multi-instance KV pools over RDMA/TCP. On a single node the native KV-offload connector wins on simplicity; Mooncake's value (3.8× throughput, 46× lower P50 TTFT in vLLM's 2026-05-06 agentic blog) requires ≥ 12 GB200s and a shared store.
6. **The async scheduler is on by default in 0.21** and doesn't need a flag. Set `--async-scheduling false` only to debug.

---

## 1. vLLM 0.21.0 — every LLM-relevant new flag and what it actually does

Source: GitHub release notes for v0.21.0 (367 commits / 202 contributors) and v0.20.0 (752 commits / 320 contributors).

### 1.1 Hybrid Memory Allocator (HMA) + KV offload integration (0.21.0)

The KV-offload subsystem now flows through the **HMA**. Practical implications:

| Subfeature | What it changes | CLI exposure |
|---|---|---|
| **Scheduler-side sliding-window group support** | KV-offload manager knows about sliding-window layer groups (gpt-oss, gemma 2/3/4, Ministral, Bamba, Jamba); doesn't evict wrong layer's KV | Implicit. Works automatically if model has hybrid attention |
| **Multi-connector HMA** | Multiple `KVConnector` instances can co-exist (e.g. native + LMCache) | `--kv-transfer-config` JSON now accepts `MultiConnector` |
| **Per-job store completion** | Offload writes acknowledged per request, not per batch — fewer stalls | Implicit |
| **DCP/PCP support in OffloadingConnector** | KV offload now compatible with `--decode-context-parallel-size` and `--prefill-context-parallel-size` | Combine: `--decode-context-parallel-size N --kv_offloading_backend native` |
| **MooncakeStoreConnector** | Distributed KV pool over Mooncake's transfer engine | `--kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true}}'` |

### 1.2 Speculative decoding — reasoning/thinking budget aware (0.21.0)

Spec decode now respects `extra_body={"thinking_budget": N}` so reasoning models (Llama-3.3-Thinking, Qwen3-Thinking) generate correct draft trees. **No new CLI flag** — it's automatic when the model exposes a thinking budget. Independent drafter attention backend selection landed at the same time:

```json
{
  "method": "eagle3",
  "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
  "num_speculative_tokens": 3,
  "draft_tensor_parallel_size": 1,
  "draft_attention_backend": "FLASH_ATTN"   // optional override; defaults to target's
}
```

### 1.3 TOKENSPEED_MLA backend (0.21.0)

**Blackwell-only.** Purpose-built for Kimi-K2.5/K2.6 and DeepSeek-R1/R2 prefill+decode on SM100+. Not available on Hopper/GH200 (the issue #40608 thread uses `TRITON_MLA` on RTX 6000 Pro; no Hopper code path). Activate with `--attention-backend TOKENSPEED_MLA` — but on GH200 this resolves to a build failure. Mention only because the Round-3 doc flagged it; **for our GH200 deployment it is not in scope.**

### 1.4 TurboQuant — 2-bit KV cache (added 0.20.0, refined 0.21.0)

| Aspect | Detail |
|---|---|
| What | 2-bit KV cache compression with **4× capacity** vs bf16 |
| Backend integration | FA3 and FA4 prefill paths supported in 0.20.0 |
| CLI | `--kv-cache-dtype turboquant_int2` (or via `--attention-backend TURBOQUANT`) |
| Hopper status | Works on SM90. Quality vs bf16 not yet independently benchmarked at 128k context |
| Hybrid quantization | `--kv-cache-quant-config '{"mode":"hybrid"}'` — keeps critical layers at higher precision |

**Recommendation:** Don't enable TurboQuant on GH200 yet. The native KV offload to 480 GB Grace LPDDR already gives you 5× the bf16 KV capacity of an 80 GB H100, so the 4× compression of TurboQuant is the *other* leg of the trade-off — pick offload first, quant only if you also need >1M context on a single die.

### 1.5 Other 0.20/0.21 LLM-relevant changes

- **FA4 default for MLA prefill** on SM90+ with paged-KV (0.20.0). For Llama-3.3-70B this only matters if you use MLA-style models — Llama uses GQA, so FA3 path applies.
- **`--attention-config.flash_attn_version`** sub-arg: explicitly pick FA2/FA3/FA4. Default on Hopper = FA3. **FA4 is not available on Hopper** — confirmed via vLLM forum thread "How to apply FA4 on B200?" — Blackwell SM100+ only.
- **CUDA-graph memory profiling on by default** (0.20.0) — adds ~1-2% startup time, no runtime hit.
- **FlashInfer top-k/top-p sampler on by default** (0.21.0). Env var `VLLM_USE_FLASHINFER_SAMPLER=1` (default). Disable only to debug.
- **C++20 build requirement** (0.21.0). Affects source builds: need gcc 13+ or clang 17+. PyPI wheels for aarch64 are pre-built.
- **Transformers v4 deprecated**, v5 only (0.20.0+). Pin `transformers>=5.0.0` in your env. Older HF model code that uses `transformers.PreTrainedModel` may need updates.
- **vLLM IR foundation** (0.20.0): `rms_norm` operator on the new IR. Not user-visible yet; affects future kernel fusion plumbing.
- **`async-scheduling: true`** is in the Llama-3.3-70B official YAML config. Default in V1 is "on" but the recipe sets it explicitly.
- **`no-enable-prefix-caching: true`** in the official Llama-3.3-70B Hopper recipe. **Counterintuitive — they explicitly disable prefix caching for the throughput benchmark** because the benchmark workload has no cache reuse. For RAG/Claude-Code-style serving, keep it ON. For ShareGPT-style microbenchmarks, off.

---

## 2. The head_dim=256 prefill penalty — exact 0.21 mitigation

### 2.1 What's still in the box at vllm-flash-attention HEAD (May 2026)

Three PRs form the mitigation stack:

| PR | Date | What | Status | Affects head_dim 256? |
|---|---|---|---|---|
| **#104** (jmkuebler) | 2026-Q1 | Two-level FP32 accumulation for SM90 FP8 fwd | **Merged**; bundled in vLLM 0.19.1 | Yes — *introduces* the 1.6× slowdown |
| **#125** (jmkuebler) | 2026-03-16 | FP8 tile-size tuning to reduce register spill | Merged | **No — author: "did not find a tiling config that meaningfully improves over the default for headdim 256"** |
| **#122** (PatrykSaffer) | 2026-04-24 | Two-level accumulation **every N steps** | Merged | **Yes — 2.28× speedup at N=4 with -1 pt accuracy** |

### 2.2 How to actually turn on the N-step mitigation

PR #122 ships a **compile-time constant**, not a runtime env var:

```
DFLASHATTENTION_FP8_TWO_LEVEL_INTERVAL = N   // default 1 (every step = full accum)
```

- Default in the upstream wheel: **N=1** (every step) → 89% on 128k-needle, full 1.6× penalty preserved.
- Author-recommended for ≥10% prefill gain: **N=4** → 88% needle, ≈2.28× faster at head_dim=256, neutral or positive at head_dim ∈ {64, 128}.
- Validation envelope: N=4 matches cutlass/deepseek convention; GSM8K & MMLU show no measurable accuracy drop.
- N=64 = 85.6% needle → too aggressive for long-context.

**Practical recipe for GH200 if you serve Gemma-4-E2B / Gemma-3 / any head_dim=256 model FP8:**

```bash
# Build vllm-flash-attention with the N-step mitigation
git clone https://github.com/vllm-project/flash-attention.git vfa
cd vfa
git checkout main   # PR #122 already merged
# Edit src/flash_attn/csrc/.../fp8_fwd_kernel.h
# Change: #define DFLASHATTENTION_FP8_TWO_LEVEL_INTERVAL 1  →  4
MAX_JOBS=64 pip install -e . --no-build-isolation
# Then re-link vllm to this build
```

**For Llama-3.3-70B / Qwen2.5-72B / Qwen3-72B (head_dim=128): not needed.** Stock 0.21 wheel is correct; the 1.6× penalty doesn't apply.

### 2.3 If you can't rebuild

Stay on bf16 for head_dim=256 prefill-heavy workloads. The vLLM FP8 blog explicitly recommends "extensive accuracy testing on the relevant workloads" if you disable two-level accum entirely (set N=∞) — not worth it for a few percent over the N=4 patch.

---

## 3. EAGLE-3 — full configuration matrix for Llama-3.3-70B and Qwen-class

### 3.1 Confirmed pretrained drafts (on Hugging Face as of 2026-05-25)

| Target | Draft head | Hopper / GH200 compatible | vLLM min version |
|---|---|---|---|
| **meta-llama/Llama-3.3-70B-Instruct** | `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B` | Yes (head_dim=128) | 0.19.0 |
| meta-llama/Llama-3.1-8B-Instruct | `RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3` | Yes | 0.19.0 |
| meta-llama/Llama-4-Maverick-17B-128E | yuhuili eagle3 head | Yes | 0.20.0 |
| Qwen/Qwen3-8B | `yuhuili/EAGLE3-Qwen3-8B` | Yes | 0.20.0 |
| **Qwen/Qwen2.5-72B-Instruct** | **No official pretrained EAGLE-3 head as of 2026-05-25** | — | — |
| **Qwen/Qwen3-72B** (if released) | **No public EAGLE-3 head yet** | — | — |
| openai/gpt-oss-20b | `nvidia/gpt-oss-20b-Eagle3-v2` | Yes | 0.19.0 |
| openai/gpt-oss-120b | `nvidia/gpt-oss-120b-Eagle3-v2` | Yes | 0.19.0 |
| google/gemma-4-31b-it | yuhuili eagle3 head | Yes (head_dim 256 → see §2) | 0.20.0 |
| cohereai/* | (Cohere Eagle, new arch in 0.21) | Yes | 0.21.0 |

**Llama-3.3-70B exact config:**
```json
{
  "method": "eagle3",
  "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
  "num_speculative_tokens": 3,
  "draft_tensor_parallel_size": 1
}
```

**Qwen2.5-72B / Qwen3-72B fallback:** No EAGLE-3 head exists publicly. Use **n-gram** instead. Llama-3.1-8B-as-draft is the only other option for Qwen2.5-72B but it's a different tokenizer, won't work. See §6.

### 3.2 Red Hat 9.5-20.5% gain — replicability matrix

| Their setup | Our setup | Replicable? |
|---|---|---|
| H200-PCIe-141GB, TP=1, gpt-oss-120B (MoE) MXFP4 | GH200 96 GB HBM, TP=1, Llama-3.3-70B bf16 | **Partial**: model class differs (MoE vs dense), HBM smaller, dtype different. Expect a *smaller* relative gain on dense bf16 (no MoE all-to-all bottleneck to hide behind) |
| Concurrency 1-200, three datasets (MLPerf/ShareGPT/SWE-bench) | Concurrency 1-200, our workload | Yes — sweep this range |
| `num_speculative_tokens` = 3 (2 and 4 also tested) | Start at 3 | Yes |
| vLLM 0.13.0 | vLLM 0.21.0 | Newer is faster, but spec-decode framework is the same |

**Practical: expect 10-20% throughput gain at concurrency 1-50, narrowing to ~5-10% at 100-200 on a single GH200 with Llama-3.3-70B bf16.** Below 10% gain at concurrency 200 means the GPU is already saturated and you should turn EAGLE-3 off.

### 3.3 P-EAGLE (parallel drafting EAGLE) — newer, AWS-published

vLLM 0.16+ supports `"parallel_drafting": true` for EAGLE-3. Per AWS 2026 blog: +1.10× to +1.36× over autoregressive EAGLE-3 on Qwen3-Coder 30B at concurrency=1, **diminishing to 1.05-1.25× at c=64**. Drafts available for gpt-oss-20b/120b and Qwen3-Coder 30B — **not Llama-3.3-70B yet.** Watch for `amazon/Llama-3.3-70B-P-EAGLE` to appear.

---

## 4. KV-offload + HMA on GH200 — sizing rules and why

### 4.1 The 400 GB / 256 GB / 1024 GB story

From issue #36623 comment by `xiejibing`:
> "Not a bug, it's a memory allocation strategy in pytorch ... [CachingHostAllocator] will round up the allocation to the nearest power of two. When > 1024 GB, it will allocate 2048 GB memory which exceeds the memory limit of the pod."

**Implication for single-node GH200** (480 GB Grace LPDDR, no pod limit but Linux still bounds you to physical RAM minus kernel/userspace headroom ≈ 440-460 GB usable):

| `--kv_offloading_size` GB | PyTorch rounds to | Fits in 480 GB Grace? |
|---|---|---|
| 64 | 64 | Yes (Round-1 community starting size) |
| 128 | 128 | Yes |
| **256** | **256** | **Yes — recommended GH200 default** |
| 257-511 | 512 | **No (OOM at allocation)** |
| 512 | 512 | Borderline; depends on what else is using LPDDR — avoid |
| 1024 | 1024 | No |
| 1025 | 2048 | Definitely no |

**The "stay under 400 GB" rule from issue #36623 is multi-GPU specific** (8× H200 pod with 1872 GB total) but the underlying power-of-2 rounding bug is universal. On a single GH200 the safe values are exactly the powers of two: 64, 128, **256**. **Recommended: 256 GB on a single GH200.** Don't set 300.

This rule **does** apply to single-node. The pod size limit doesn't, but the allocator behavior does.

### 4.2 Recommended GH200 sizing recipe

Single GH200, Llama-3.3-70B bf16 (~140 GB weights), max-model-len=131072, max-num-seqs=128:

```text
HBM 96 GB:
  ~ 60 GB active weights (rest spilled)
  ~ 28 GB active KV (≈ 800 KB/token × 128 seqs × 256 active tok avg)
  ~  8 GB workspace, CUDA graphs, activations

Grace 480 GB LPDDR:
  ~ 80 GB cpu-offload-gb (weight spill)
  ~ 256 GB kv_offloading_size (KV spill, power-of-2 safe)
  ~ 64 GB swap-space (preempted KV blocks)
  ~ 80 GB headroom (OS, Mooncake/LMCache process if used, HF cache)
```

If max-model-len > 256k or max-num-seqs > 256, you'll hit the 256 GB power-of-2 ceiling — at that point the answer is **not** "set 384" (rounds to 512, OOM) but **add LMCache as L3 over NVMe** or scale to 2× GH200 NVL2 with TP=2.

### 4.3 Block size and DMA mechanics (unchanged from Round 1, confirmed)

- vLLM uses `cudaMemcpyAsync` for offload — the *GPU's* DMA engine, not CPU-side memcpy.
- Block size grew from kilobytes (0.11.0) to **0.5-2 MB (0.12.0+)** to better saturate DMA.
- Measured bidirectional throughput on H100 PCIe = **83.4 GB/s** (vLLM 2026-01-08 blog).
- On GH200 the same code path rides NVLink-C2C at **900 GB/s** with zero vLLM changes. **Unbenchmarked publicly** — still the open follow-up from Round 3.

---

## 5. MooncakeStoreConnector — when it's a net win on GH200

### 5.1 What Mooncake gives you

| Feature | Native KV offload | MooncakeStoreConnector |
|---|---|---|
| Storage | Local CPU memory only | Distributed pool (CPU + SSD + NVMe + RDMA across nodes) |
| Cross-instance prefix cache hits | No | Yes (hash-based) |
| Setup complexity | Trivial (one flag) | Mooncake master + config file (`MOONCAKE_CONFIG_PATH`) |
| Single-node viability | First-class | Works in `kv_role=kv_both`, uses ZMQ IPC for local lookup |
| Production benchmark | Single-node H100 | **3.8× throughput / 46× lower P50 TTFT** on 12× GB200, 1.7% → 92.2% cache hit (agentic) |

### 5.2 Decision rule for GH200 deployments

- **Single GH200 serving one workload type**: use native KV offload. Mooncake adds setup pain with no benefit.
- **Single GH200 serving multi-instance traffic with cross-request cache reuse** (agentic, multi-turn): consider Mooncake if you'll later scale to multiple GH200s and want the config to carry forward.
- **2+ GH200s sharing a KV pool**: Mooncake or LMCache. Mooncake's 2026-05-06 vLLM blog is the production-ready endorsement.
- **Long-context retrieval / RAG with shared prefix prompts across many users**: Mooncake. Cross-instance hits are where it shines.

### 5.3 Sample config (single-node `kv_both` mode)

```bash
export MOONCAKE_CONFIG_PATH=/etc/mooncake/master.json
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true,"global_segment_size":68719476736}}'
```

`global_segment_size` is bytes contributed per vLLM worker; 64 GiB = `68719476736`. On a single GH200 with 480 GB LPDDR, contributing 64-128 GB per worker is reasonable. **RDMA is optional** (single-node uses local IPC) — confirmed from PR #40900.

---

## 6. PagedAttention v2 / V1-engine async scheduler — every flag that matters

### 6.1 V1 SchedulerConfig defaults (0.21)

| Flag | Default | Throughput-relevant? |
|---|---|---|
| `--max-num-batched-tokens` | **2048** | Yes — major lever. 8192 is the official Llama-3.3-70B recipe; 16384 for batch throughput; 2048 for ITL-sensitive |
| `--max-num-seqs` | **128** | Yes — GH200 has KV headroom for 256+. Lambda's 7.6× win came from large `max-num-seqs` |
| `--async-scheduling` | **on (default in V1)** | Yes — disabling costs 5-15% throughput. Only disable to debug |
| `--enable-chunked-prefill` | **on (V1 default)** | Yes — required for balanced TTFT/ITL |
| `--scheduling-policy` | `fcfs` | Marginal — set to `priority` for production with priority queues |
| `--long-prefill-token-threshold` | `0` (disabled) | Set to e.g. 8192 to deprioritize very-long prefills so short requests don't starve |
| `--max-num-partial-prefills` | `1` | Increase to 2-4 to allow multiple long prefills to chunk concurrently |
| `--max-long-partial-prefills` | `1` | Cap on how many long prefills can run in parallel |
| `--enable-prefix-caching` | **`False`** by default per stable docs; **V1 design says always on** | **Set explicitly.** For RAG/multi-turn/Claude-Code-style serving: `--enable-prefix-caching`. For one-shot benchmark: `--no-enable-prefix-caching` |
| `--prefix-caching-hash-algo` | `sha256` | Switch to `xxhash` for ~5% lower hash overhead on prefix-heavy workloads |
| `--gpu-memory-utilization` | **0.92** (current stable default) | 0.95 fine on GH200 since Grace LPDDR catches overflow |
| `--kv-cache-dtype` | `auto` (= model dtype) | Don't change to fp8 without 128k-needle test |
| `--cpu-offload-gb` | `0` | Yes — main lever for fitting 70B bf16 weights on 96 GB HBM |
| `--swap-space` | 4 GB | Bump to 64 GB on GH200; Grace LPDDR makes swap nearly free |
| `--kv-offloading-backend` | `native` (when size set) | `lmcache` is the other option |
| `--kv-offloading-size` | None (disabled) | **Set to 256** on GH200; see §4 |
| `--decode-context-parallel-size` | 1 | Don't enable on single GH200; multi-GPU only |
| `--prefill-context-parallel-size` | 1 | Same |
| `--attention-backend` | auto (= FLASH_ATTN on SM90) | Override only for FlashInfer A/B tests |
| `--attention-config.flash_attn_version` | auto (=3 on SM90) | FA4 not available on Hopper — don't try to force it |

### 6.2 PagedAttention v2 / "what's actually called PA2"

PagedAttention v2 is **not a separate kernel mode you flag.** It's the V1-engine evolution of paged attention that supports chunked prefill + prefix caching concurrently with the same block-table representation. There's no `--paged-attention-v2` flag — it's just what V1 does. The original V0 PagedAttention is gone.

### 6.3 Throughput levers ranked >20% impact

In order of measured impact for Llama-3.3-70B on Hopper-class:

1. **bf16 vs FP8** weights (when applicable) — up to 1.5-2× memory & decode throughput; FP8 needs head_dim ≤ 128
2. **`max-num-seqs` 128 → 256-512** — typically 1.3-1.7× aggregate throughput on GH200 (more batching room thanks to KV offload)
3. **EAGLE-3 spec decode** — 1.10-1.21× per Red Hat (gpt-oss); 1.10-1.20× expected on dense 70B at concurrency ≤ 100
4. **`max-num-batched-tokens` 2048 → 8192-16384** — 1.2-1.4× prefill throughput, trades 1.5-2× higher ITL spikes
5. **Prefix caching ON for RAG workloads** — 5-50× TTFT improvement at high cache-hit rates; unmeasurable at 0% hit
6. **Async scheduling ON** — 1.05-1.15× (default in V1)
7. **`--compilation-config -O3`** — startup time ↑ (~3× slower) but +5-10% steady-state throughput
8. **TurboQuant 2-bit KV** — 4× KV capacity, throughput-neutral on Hopper

### 6.4 The Hopper Llama-3.3-70B official compilation config

Per the vllm-ai-recipes repo, **the Blackwell recipe includes**:

```yaml
compilation-config: '{"pass_config":{"fuse_allreduce_rms":true,"fuse_attn_quant":true,"eliminate_noops":true}}'
```

**The Hopper recipe does NOT include this** — these fused FlashInfer kernels (fuse_allreduce_rms = AllReduce+RMSNorm fusion, fuse_attn_quant = attention+FP8 quant fusion) are Blackwell-only. **Do not copy the Blackwell compilation-config to a GH200 launch — it will silently be ignored or error.**

For GH200, the **Llama-3.3-70B Hopper YAML** as published:

```yaml
kv-cache-dtype: fp8           # remember: fp8 only safe at head_dim ≤ 128 with vLLM ≥ 0.19.1; Llama is 128
async-scheduling: true
no-enable-prefix-caching: true
max-num-batched-tokens: 8192
```

The `no-enable-prefix-caching: true` is **for benchmark reproduction**, not production. Flip it for real workloads.

---

## 7. Speculative decoding — which method for Llama-3.3-70B vs Qwen2.5-72B vs Qwen3-72B

### 7.1 Decision matrix

| Target model | Best method (May 2026) | Why | Fallback |
|---|---|---|---|
| **Llama-3.3-70B-Instruct** | **EAGLE-3** (`yuhuili/EAGLE3-LLaMA3.3-Instruct-70B`) | Pretrained head exists, vLLM ≥0.19, 1.1-1.21× gain through c=200 (extrapolating from gpt-oss) | n-gram for RAG/code completion |
| Llama-3.3-Thinking-70B | EAGLE-3 + thinking-budget aware (0.21+) | 0.21 specifically fixed spec decode for reasoning models | n-gram |
| **Qwen2.5-72B-Instruct** | **n-gram** | **No public EAGLE-3 head for Qwen2.5-72B** as of 2026-05-25 | Suffix decoding (no draft) |
| **Qwen3-72B** (if shipped) | n-gram or wait for EAGLE-3 head | Same gap as Qwen2.5 | n-gram |
| Qwen3-8B | EAGLE-3 (`yuhuili/EAGLE3-Qwen3-8B`) | Pretrained head exists | n-gram |
| Qwen3-Coder-30B | **P-EAGLE** (`amazon/Qwen3-Coder-30B-A3B-Instruct-P-EAGLE`) | AWS-published P-EAGLE head; +10-25% over autoregressive EAGLE-3 | EAGLE-3 |
| DeepSeek-V3 / R1 | **MTP** (native) | DeepSeek's MTP head trained jointly; ~1.8× at >80% acceptance | EAGLE-3 |
| gpt-oss-20b/120b | EAGLE-3 (NVIDIA-published heads) | Red Hat verified 9.5-20.7% | n-gram |

### 7.2 n-gram (the Qwen2.5-72B / Qwen3-72B answer)

n-gram speculative decoding **moved CPU→GPU in vLLM 0.18.0** (PR #29184) and is now compatible with the async scheduler. This significantly cut overhead — n-gram used to be useless above a few QPS because of CPU-GPU sync stalls; it's now a viable default.

Configuration:

```json
{
  "method": "ngram",
  "num_speculative_tokens": 5,
  "prompt_lookup_min": 2,
  "prompt_lookup_max": 5
}
```

**Caveat:** issue #40875 (vLLM 2026-05) — `prompt_lookup_min=2` (the default) **corrupts tool-call output on Qwen3-class models with structured output**. Config-only fix: set `prompt_lookup_min=8`. **For Qwen2.5-72B with structured/tool-calling workloads, set `prompt_lookup_min: 8`.**

n-gram performance: 1.2-1.5× speedup on prompts with high overlap (RAG, summarization, code completion). 1.0× (neutral) on creative generation. Don't bother on dialogue/chat.

### 7.3 MEDUSA — still in the codebase, low priority

MEDUSA proposer class still exists (`vllm.v1.spec_decode.medusa`), not deprecated as of 0.21. But no major pretrained heads being published for Llama-3.3 or Qwen — EAGLE-3 has eaten the ecosystem. **Skip MEDUSA for new deployments.**

### 7.4 Suffix decoding (zero-draft alternative)

vLLM 0.20+ ships **suffix decoding** as a method=spec option:

```json
{
  "method": "suffix",
  "suffix_decoding_max_tree_depth": 24,
  "suffix_decoding_max_cached_requests": 10000,
  "suffix_decoding_max_spec_factor": 1.0,
  "suffix_decoding_min_token_prob": 0.1
}
```

No draft model, dynamic speculation depth. Useful when you can't get an EAGLE-3 head trained for your specific fine-tune. Quality varies — benchmark vs n-gram before committing.

---

## 8. Three full launch commands (paste-ready)

All assume `vllm==0.21.0`, `torch==2.11.*`, CUDA 13.0+, single GH200 (1× Hopper die + 480 GB Grace LPDDR), Llama-3.3-70B-Instruct bf16 (no FP8 silently — the hard prior).

### 8.1 Interactive serving (low latency, modest concurrency, Claude-Code-style)

Goal: TTFT < 500 ms, ITL < 50 ms, up to ~16-32 concurrent users, 32k context, prefix caching ON for multi-turn reuse.

```bash
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_SLEEP_WHEN_IDLE=0          # reduce post-idle latency spikes
export VLLM_ALLREDUCE_USE_SYMM_MEM=1   # default; matters only at TP>1

vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.92 \
  --cpu-offload-gb 60 \
  --kv-offloading-backend native \
  --kv-offloading-size 128 \
  --swap-space 64 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --kv-cache-dtype auto \
  --async-scheduling \
  --scheduling-policy fcfs \
  --speculative-config '{"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.3-Instruct-70B","num_speculative_tokens":3,"draft_tensor_parallel_size":1}' \
  --attention-backend FLASH_ATTN \
  --port 8000
```

Expected on GH200 (interpolated, not measured): ~35-60 tok/s per request at low concurrency, 600-1200 tok/s aggregate at concurrency 16.

### 8.2 Batch throughput (offline scoring, ShareGPT-style, max tok/s)

Goal: maximize aggregate tokens/sec, latency irrelevant. No prefix caching (random prompts), large batches.

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.95 \
  --cpu-offload-gb 40 \
  --kv-offloading-backend native \
  --kv-offloading-size 64 \
  --swap-space 32 \
  --no-enable-prefix-caching \
  --kv-cache-dtype auto \
  --async-scheduling \
  --max-num-partial-prefills 4 \
  --max-long-partial-prefills 2 \
  --long-prefill-token-threshold 4096 \
  --compilation-config '{"level":3}' \
  --attention-backend FLASH_ATTN \
  --port 8000
```

Notes:
- Prefix cache OFF: matches the official Hopper Llama-3.3-70B benchmark recipe.
- `kv-offloading-size` only 64 GB because max-model-len=8192 limits active KV; no need to spill much.
- `compilation-config level=3`: -O3 trades 2-3× slower startup for ~5-10% steady-state throughput.
- EAGLE-3 **off** intentionally — at concurrency 512 gains decay; verify on your workload before re-adding.

Expected on GH200: ~1500-2200 tok/s aggregate (interpolated from Lambda 2024 × V1 1.7× × Llama-3.3 vs 3.1 parity).

### 8.3 Long context (128k, RAG, document summarization)

Goal: 128k context, moderate concurrency (16-64), full prefix caching for RAG retrieval reuse, aggressive KV offload to Grace.

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --cpu-offload-gb 80 \
  --kv-offloading-backend native \
  --kv-offloading-size 256 \
  --swap-space 64 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo xxhash \
  --kv-cache-dtype auto \
  --async-scheduling \
  --long-prefill-token-threshold 16384 \
  --max-num-partial-prefills 2 \
  --speculative-config '{"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.3-Instruct-70B","num_speculative_tokens":3,"draft_tensor_parallel_size":1}' \
  --attention-backend FLASH_ATTN \
  --port 8000
```

Notes:
- `kv-offloading-size 256` = exactly the power-of-2 ceiling under 480 GB Grace (see §4). **Do not bump to 300/384.**
- `long-prefill-token-threshold 16384`: chunks documents > 16k so short follow-ups don't starve while a 128k doc is prefilling.
- `cpu-offload-gb 80`: weights spill to Grace, leaves ~16 GB HBM for active KV plus workspace.
- bf16 KV (not FP8): with 256 GB Grace KV offload available, you don't *need* FP8 KV — and the FP8 head_dim ≤ 128 caveat means it would also work here, but bf16 is the conservative default for long-context accuracy.

Expected on GH200: TTFT 3-8 s at 128k cold prompts, ~25-40 tok/s decode per concurrent user.

---

## 9. What's still unmeasured / open follow-ups

1. **Llama-3.3-70B bf16 vLLM 0.21 tok/s on a single GH200** — public number does not exist. Benchmark in-house.
2. **EAGLE-3 gain for dense 70B at concurrency 100-200** — Red Hat data is gpt-oss MoE. We expect partial transfer.
3. **NVLink-C2C effective throughput for `--kv-offloading-backend native`** — vLLM's published number is 83.4 GB/s on H100 PCIe. GH200 should ride C2C at 900 GB/s. Unmeasured publicly. Run a `cudaMemcpyAsync` micro-benchmark to verify.
4. **Qwen2.5-72B / Qwen3-72B EAGLE-3** — no pretrained head. If we deploy Qwen, train one with SpecForge (cited 2026-03-20 arxiv) or use n-gram.
5. **TOKENSPEED_MLA on Hopper** — not in scope; Blackwell-only as of 0.21. Track for future GH200 Blackwell-successor.
6. **TurboQuant 2-bit KV quality** at 128k on Llama-3.3-70B — needle test before production.

---

## 10. Sources (this round)

- vLLM 0.21.0 release — https://github.com/vllm-project/vllm/releases/tag/v0.21.0 — 2026-05-15
- vLLM 0.20.0 release — https://github.com/vllm-project/vllm/releases/tag/v0.20.0 — 2026-04-27
- vLLM FP8 KV-cache blog — https://vllm-project.github.io/2026/04/22/fp8-kvcache.html — 2026-04-22
- vLLM Mooncake agentic serving blog — https://vllm.ai/blog/2026-05-06-mooncake-store — 2026-05-06
- vLLM KV-offload connector blog — https://vllm.ai/blog/2026-01-08-kv-offloading-connector — 2026-01-08
- vLLM v0.18.0 release (n-gram CPU→GPU, PR #29184) — https://github.com/vllm-project/vllm/releases/tag/v0.18.0
- vllm-flash-attention PR #104 (two-level accum) — https://github.com/vllm-project/flash-attention/pull/104
- vllm-flash-attention PR #122 (N-step accum, head_dim=256 2.28× fix) — https://github.com/vllm-project/flash-attention/pull/122 — merged 2026-04-24
- vllm-flash-attention PR #125 (FP8 tile-size tuning) — https://github.com/vllm-project/flash-attention/pull/125 — merged 2026-03-16
- vLLM Llama-3.3-70B recipe — https://docs.vllm.ai/projects/recipes/en/latest/Llama/Llama3.3-70B.html
- vLLM EAGLE feature docs — https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/
- vLLM Speculative Decoding overview — https://docs.vllm.ai/en/latest/features/speculative_decoding/
- vLLM Attention Backends design — https://docs.vllm.ai/en/latest/design/attention_backends/
- vLLM Engine Args — https://docs.vllm.ai/en/stable/configuration/engine_args/
- vLLM SchedulerConfig API — https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/
- vLLM Context Parallel Deployment (DCP/PCP) — https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/
- vLLM torch.compile design — https://docs.vllm.ai/en/latest/design/torch_compile/
- vLLM Environment Variables — https://docs.vllm.ai/en/stable/configuration/env_vars/
- vLLM issue #36623 (OOM at kv_offloading_size > 1024) — https://github.com/vllm-project/vllm/issues/36623 (CachingHostAllocator power-of-2 root cause)
- vLLM issue #40875 (n-gram prompt_lookup_min default corrupts Qwen3 tool calls) — https://github.com/vllm-project/vllm/issues/40875
- vLLM issue #40608 (Kimi K2.5/K2.6 MLA + EAGLE on Blackwell) — https://github.com/vllm-project/vllm/issues/40608
- vLLM PR #40900 (MooncakeStoreConnector) — https://github.com/vllm-project/vllm/pull/40900
- Red Hat, gpt-oss EAGLE-3 perf — https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss — 2026-04-16
- AWS P-EAGLE blog — https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/
- LightSeek TokenSpeed announcement — https://lightseek.org/blog/lightseek-tokenspeed.html
- David Noel Ng, 2× GH200 Claude Code optimization — https://dnhkng.github.io/posts/vllm-optimization-gh200/
- vLLM forum, FA4 on B200 — https://discuss.vllm.ai/t/how-to-apply-fa4-on-b200/2133
- vLLM-AI-recipes Llama3.3-70B (Hopper/Blackwell YAML diff) — https://github.com/michelabboud/vllm-ai-recipes/blob/main/Llama/Llama3.3-70B.md
- HF: yuhuili/EAGLE3-LLaMA3.3-Instruct-70B — https://huggingface.co/yuhuili/EAGLE3-LLaMA3.3-Instruct-70B
