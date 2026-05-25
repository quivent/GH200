# TensorRT-LLM 1.2.1 on GH200 — Round 4 LLM-Only Deep Dive (Agent 02)

**Date compiled:** 2026-05-25
**Scope:** LLM-only. DiT/diffusion is out of scope (FP8 still broken on that stack per team policy).
**Prior docs:**
- `/home/ubuntu/research/gh200_inference/round1/02_tensorrt_llm.md`
- `/home/ubuntu/research/gh200_inference/round3/02_trtllm_verify.md`

**Reaffirmed constraints from Round 3:**
- v1.2.1 (2026-04-20) is the latest stable; v1.3.0rc15 (2026-05-21) is pre-release only.
- #4218 (Qwen FP8 KV gibberish) closed *without fix*; #8974 (AutoDeploy FP8/NVFP4) closed *as abandoned*; Llama-3.x PP+FP8 hang still patched only via env-var workaround.
- BF16 default for diffusion; for dense Hopper LLMs FP8 is OK head_dim ≤ 128 but ships with caveats.

---

## Q1. TRT-LLM 1.2.1 PyTorch backend — exact LLM API config for Llama-3.3-70B BF16 on a single GH200

### The constructor parameter list (verified against `llm-api/reference.html`, 2026-05-25)

Confirmed PyTorch-backend `LLM(...)` parameters (with documented defaults where present):

| Parameter | Default | Notes |
|---|---|---|
| `model` | required | HF name or local path |
| `dtype` | `'auto'` | use `'bfloat16'` explicit |
| `tensor_parallel_size` | 1 | GH200 single = 1; GH200-NVL2 = 2 |
| `pipeline_parallel_size` | 1 | **avoid PP+FP8 on Llama (hang)** |
| `context_parallel_size` | 1 | for very long ISL workloads |
| `max_batch_size` | 2048 | the *PyTorch backend* default is generous; tune down |
| `max_input_len` | 1024 | raise to 8192–32768 for long-context |
| `max_seq_len` | None | derived from `max_input_len + max_num_tokens` if unset |
| `max_num_tokens` | 8192 | per-iteration batched token budget |
| `max_beam_width` | 1 | beam=1 is the only fully-tested path |
| `enable_chunked_prefill` | **False** | **default is OFF in v1.2.x** — turn ON for ISL-mixed traffic |
| `attn_backend` | `'TRTLLM'` | other choice: `'FLASHINFER'` |
| `cuda_graph_config` | None | enable for low-latency single-batch |
| `kv_cache_config` | `KvCacheConfig()` | see below |
| `scheduler_config` | None | continuous batching is implicit; no flag needed |
| `speculative_config` | None | see Q3 for Eagle3 |
| `lora_config` | None | see Q5 |
| `quant_config` | None | leave None for bf16 |

### `KvCacheConfig` fields (verified against `features/kvcache.html`)

| Field | Default | Recommended GH200 BF16 70B |
|---|---|---|
| `dtype` | `'auto'` | `'auto'` (resolves to bf16) |
| `free_gpu_memory_fraction` | **0.9** | **0.85** (leave 15% for activations/cuda graphs) |
| `enable_block_reuse` | **True** | True (prefix-cache wins +30% on shared system prompts — see Q4) |
| `enable_partial_reuse` | True | True |
| `host_cache_size` | **0 bytes** | **200 × 1024³ (200 GB)** — offload to Grace LPDDR5X |
| `secondary_offload_min_priority` | 35 | leave default |
| `copy_on_partial_reuse` | (unspecified) | leave default |
| `max_attention_window` | None | None (use full context) |

**Critical default to know:** `free_gpu_memory_fraction = 0.9` is high enough that adding CUDA graphs or speculative decoding can OOM. Drop to 0.85 if you enable either.

### Recommended LLM-API config (matches or beats vLLM 0.21 on GH200-96 BF16)

```python
# trtllm-1.2.1 PyTorch backend, single GH200-96GB, Llama-3.3-70B BF16
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, CudaGraphConfig

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    dtype="bfloat16",
    tensor_parallel_size=1,
    # batch / token budgets — tuned for 96 GB HBM3 single-GH200
    max_batch_size=64,           # 70B BF16 weights ≈ 140 GB → can't fit on 96 GB alone
                                 # (so this config assumes weights stored once with KV
                                 #  spilling to Grace; if you fit 70B BF16 on a single
                                 #  96 GB GPU you actually need NVL2/144G or quantize)
    max_num_tokens=8192,
    max_seq_len=16384,
    enable_chunked_prefill=True, # NOT default — turn ON
    attn_backend="TRTLLM",
    kv_cache_config=KvCacheConfig(
        free_gpu_memory_fraction=0.85,
        enable_block_reuse=True,         # default; explicit for the record
        host_cache_size=200 * 1024**3,   # 200 GB offload to Grace LPDDR5X
    ),
    cuda_graph_config=CudaGraphConfig(
        max_batch_size=64,
        enable_padding=True,
    ),
)
```

**Memory reality check for 70B BF16 on a single GH200-96:** 70B × 2 bytes/param ≈ **140 GB** for weights alone. You **cannot fit Llama-3.3-70B BF16 weights resident on a single GH200-96GB GPU**. Options:
1. **GH200-144G HBM3e** (the only single-GPU SKU that fits 70B BF16 weights plus a usable KV pool).
2. **GH200-NVL2** (two-GPU NVLink-C2C package, TP=2).
3. **Single GH200-96 + W4A16 AWQ** (quantize weights to int4 → ~40 GB, keep bf16 activations and KV).
4. **Single GH200-96 + FP8 weights** — accepted today by NVIDIA's NIM matrix for Llama-3.3-70B on GH200, but inherits the FP8 risk surface from Round 3.

For an honest "match-or-beat vLLM 0.21" comparison on a single GH200-96GB, use **W4A16 AWQ + BF16 KV** as the apples-to-apples target (vLLM's Llama-3.3-70B recipe on Hopper recommends FP8 + FP8 KV; we're substituting bf16-default and absorbing the memory cost via AWQ weights).

### vLLM 0.21 baseline we're matching (verified from vLLM Llama-3.3-70B recipe)

vLLM Hopper recipe (verbatim from docs.vllm.ai):
```yaml
kv-cache-dtype: fp8
async-scheduling: true
no-enable-prefix-caching: true   # NOTE: vLLM disables prefix caching for 70B
max-num-batched-tokens: 8192
```
Throughput tuning: "Set TP to 1 and increase BS to the maximum possible value" for max throughput; "TP=2, BS=128" for balanced.

### TRT-LLM 1.2.1 config to match-or-beat that

```bash
trtllm-serve meta-llama/Llama-3.3-70B-Instruct \
    --backend pytorch \
    --tp_size 1 \
    --max_batch_size 128 \
    --max_num_tokens 8192 \
    --kv_cache_free_gpu_memory_fraction 0.85 \
    --kv_cache_dtype auto \
    --enable_chunked_prefill \
    --extra_llm_api_options config.yml \
    --host 0.0.0.0 --port 8000
```

`config.yml`:
```yaml
enable_attention_dp: false
enable_autotuner: false
cuda_graph_config:
  max_batch_size: 128
  enable_padding: true
kv_cache_config:
  enable_block_reuse: true        # we WANT prefix caching ON (vLLM turns it off for 70B)
  free_gpu_memory_fraction: 0.85
  host_cache_size: 214748364800   # 200 GB offload to Grace
```

**Why this should beat the vLLM recipe at high concurrency:**
1. vLLM's Llama-3.3-70B recipe explicitly **disables prefix caching** (`no-enable-prefix-caching: true`). The SqueezeBits direct comparison shows TRT-LLM gains **+34.7% throughput / +20.9% TPOT** with prefix-cache ON when shared prefixes exist (vLLM only gains 13.3%/9.8% in the same scenario, presumably because their cache walker has higher overhead). If your traffic has a shared system prompt, TRT-LLM with `enable_block_reuse=True` is the clear pick.
2. `host_cache_size` 200 GB to Grace LPDDR5X is **a GH200-only capability** vLLM does not currently match. NVLink-C2C 450 GB/s makes this practical; no other inference stack has a clean equivalent.
3. CUDA graphs in the PyTorch backend reduce kernel-launch overhead — important at decode-step granularity where 70B is memory-bound.

Sources: NVIDIA LLM API reference https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html ; KV cache features https://nvidia.github.io/TensorRT-LLM/features/kvcache.html ; vLLM recipe https://docs.vllm.ai/projects/recipes/en/latest/Llama/Llama3.3-70B.html ; SqueezeBits prefix caching benchmark https://blog.squeezebits.com/vllm-vs-tensorrtllm-12-automatic-prefix-caching-38189

---

## Q2. Engine-compile path (`trtllm-build`) — when worth it?

### Build cost (verified)
- **Llama-3 70B BF16 single-GPU engine on H100/GH200:** community reports **30–90 min wall-clock** (varies with `--workers`, `--max_batch_size`, plugins).
- **Llama-3.1 405B FP8:** "hours" of babysitting; requires the target GPUs *during build*, not just at inference.
- **Engine is hardware-specific:** H100-built engine **does not run** on GH200 (different SM90 codegen path). GH200-96 engine may not run on GH200-144 depending on memory-dependent plugin choices.

### Runtime gain
NVIDIA has not published a head-to-head PyTorch-backend-vs-compiled-engine number for Llama-3.3-70B on any Hopper variant. The closest signals:
- TRT-LLM GH issue #8564 (closed without comment from maintainers in the public thread) asked exactly this question; NVIDIA did not respond on-thread with a number.
- General community consensus (Synthetic Futures, Spheron, Introl, BentoML 2026 guides): **compiled engine wins 5–15% peak throughput** over PyTorch backend at matched concurrency on a stable workload; cold-start is the big win on the PyTorch side (loads HF weights in 60–90 s vs 30–90 min compile).
- One Spheron data point (H100 production, not GH200): "production deployments report 4× throughput vs native PyTorch" — but this compares TRT-LLM **engine** vs raw HF Transformers, not TRT-LLM PyTorch *backend* vs TRT-LLM compiled engine. Don't conflate the two. The 4× number is misquoted in most secondary sources.

### Decision matrix

| Scenario | Engine build worth it? |
|---|---|
| Single fixed model, prod traffic, max-throughput priority | **Yes** — 5–15% wins are real money over a year |
| Iterating model choices / draft models for spec-decode | **No** — rebuild tax kills iteration speed |
| Multi-LoRA serving | **No** — see Q5; PyTorch backend handles adapters cleanly, engine path requires pre-baking |
| Frequent model swaps / R&D | **No** |
| KV cache reuse with shared system prompts | Either — both backends support `enable_block_reuse` |
| Chunked prefill | Either; default OFF in 1.2.x — turn it on explicitly |

### Operational tax to remember
1. Build needs the target GPU. Plan capacity.
2. Engine pin to a specific TRT-LLM version. Rebuild required when upgrading. **v1.3.0 GA still hasn't shipped after 15 RCs over 4 months** — upgrade risk is real.
3. PyTorch backend was made stable+default in v1.0 (Sep 2025) and has been the canonical entry point for ~8 months. NVIDIA's investment is going there, not into the legacy engine path. Engine path is *not* deprecated, but it's no longer the lead horse.

**Recommendation:** Use the PyTorch backend by default. Reach for `trtllm-build` only when (a) you have profiled and confirmed a >10% bottleneck that profiling traces to TRT-engine-only kernels, AND (b) the model/quant config is stable enough to amortize ~1 hour of build time per release cycle.

Sources: https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html ; https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/ ; GH issue #8564 https://github.com/NVIDIA/TensorRT-LLM/issues/8564 ; https://introl.com/blog/tensorrt-llm-optimization-nvidia-inference-stack-guide ; BentoML tuning guide https://www.bentoml.com/blog/tuning-tensor-rt-llm-for-optimal-serving-with-bentoml

---

## Q3. EAGLE-3 + TRT-LLM — vendor-claimed 1.6–2.6× for Llama-3.3-70B at low concurrency

### What the "1.6–2.6×" claim actually maps to (verified)

The 1.6–2.6× speedup figure for "Llama-3.3-70B + EAGLE3 at low concurrency" cannot be sourced **exactly as stated** from any NVIDIA-published Llama-3.3-70B benchmark. The closest numerically-anchored sources are:

1. **NVIDIA dev blog (17 Dec 2024, H200, FP8, batch=1):** Llama-3.3-70B + draft model (NOT EAGLE3, draft-target):
   - Llama-3.2-1B draft → 70B target: 51.14 → **181.74 tok/s = 3.55×**
   - Llama-3.2-3B draft → 70B target: 51.14 → **161.53 tok/s = 3.16×**
   - Llama-3.1-8B draft → 70B target: 51.14 → **134.38 tok/s = 2.63×**
   - This is the **2.63×** number that gets quoted as "EAGLE3-like" but it's actually a different spec-decode mode (draft-target).
2. **EAGLE-3 paper (arXiv 2503.01840):** reports **4.1–6.5× at temperature 0** on academic benchmarks for various target models including Llama-3.3-70B. These are paper numbers, not NVIDIA TRT-LLM measurements.
3. **NVIDIA HF model card (`nvidia/Llama-3.3-70B-Instruct-Eagle3`):** publishes **acceptance rates only** (Writing 2.10, Math 3.25, Coding 3.18, etc., for `max_draft_len=3`). **No tok/s numbers, no speedup numbers.**
4. **Llama-4 Maverick + EAGLE3 (NVIDIA tech blog 6):** "over 1000 tokens/sec per user on 8×B200." This is Blackwell, not GH200, and Llama-4, not Llama-3.3. Note the doc explicitly warns: *"This configuration is optimized for minimum latency. When increasing the concurrency of requests, the TPS per user degrades rapidly."*

**Verdict on the "1.6–2.6× at low concurrency" claim:** the numerical range is consistent with what you'd see for EAGLE3+Llama-3.3-70B at acceptance rates 2.1–3.2 (MT-Bench numbers from the HF card) using `max_draft_len=3`. The math: speedup ≈ acceptance_length / (1 + draft_overhead) ≈ 2.1–3.2 / ~1.2 ≈ **1.7–2.7×**. **The vendor claim is a reasonable engineering estimate but is not a single benchmark number you can point to in NVIDIA's literature.**

### Confirmed recipe (verbatim from the NVIDIA HF model card)

**Command:**
```bash
trtllm-serve <Llama-3.3-70B checkpoint> \
  --host 0.0.0.0 --port 8000 \
  --backend pytorch \
  --max_batch_size 32 \
  --tp_size 8 \
  --extra_llm_api_options extra-llm-api-config.yml
```

**`extra-llm-api-config.yml`:**
```yaml
enable_attention_dp: false
enable_autotuner: false
cuda_graph_config:
  max_batch_size: 32
  enable_padding: true
speculative_config:
  decoding_type: Eagle
  max_draft_len: 3
  speculative_model_dir: <nvidia/Llama-3.3-70B-Instruct-Eagle3 checkpoint>
  eagle3_layers_to_capture: [-1]   # final hidden state post-LayerNorm, pre-LMHead
kv_cache_config:
  enable_block_reuse: false        # NOTE: NVIDIA recipe DISABLES prefix caching with Eagle3
```

**Important nuances:**
- `tp_size=8` in the NVIDIA recipe is for an **8×H200 deployment**, not GH200. On a single GH200, set `tp_size=1` and `max_batch_size` proportionally lower (8 or 16).
- **`kv_cache_config.enable_block_reuse: false`** — the NVIDIA recipe disables KV cache reuse when running EAGLE3. The recipe doesn't explain why; the likely reason is that Eagle3's hidden-state caching interacts with block reuse in ways not yet exercised. v1.3.0rc15 release notes explicitly add "Hidden-state reuse across CUDA graph captures" for EAGLE3 — implying the integration story is *still maturing*. **If you're paying for EAGLE3 you may have to give up block reuse.** Test both configurations on your traffic before committing.
- PyTorch backend supports **only EAGLE3** of the EAGLE family (not legacy EAGLE/EAGLE2). Verified verbatim from `features/speculative-decoding.html`: *"The PyTorch backend supports only Eagle3."*

### Programmatic LLM API equivalent

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import Eagle3DecodingConfig, KvCacheConfig

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    dtype="bfloat16",
    tensor_parallel_size=1,
    max_batch_size=16,
    speculative_config=Eagle3DecodingConfig(
        max_draft_len=3,
        speculative_model="nvidia/Llama-3.3-70B-Instruct-Eagle3",
    ),
    kv_cache_config=KvCacheConfig(
        enable_block_reuse=False,        # per NVIDIA recipe
        free_gpu_memory_fraction=0.80,   # lower than the no-spec config; spec adds memory
    ),
)
```

### Hopper-vs-Blackwell EAGLE3 caveat
The 1.6–2.6× range applies cleanly to bf16 on Hopper. NVIDIA's own showcase numbers for EAGLE3 (the 1000+ tok/s/user figure) are on Blackwell B200 with FP8 + FP4 paths that don't exist on Hopper SM90. **Don't expect Blackwell-level EAGLE3 wins on GH200.**

Sources: https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-Eagle3 ; https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/ ; https://nvidia.github.io/TensorRT-LLM/features/speculative-decoding.html ; https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog6_Llama4_maverick_eagle_guide.html ; EAGLE-3 paper https://arxiv.org/pdf/2503.01840

---

## Q4. In-flight batching + KV cache reuse — published deltas for BF16 70B

### In-flight batching (continuous batching)
- **Default and not configurable** in TRT-LLM PyTorch backend since v1.0. C++ batch manager open-sourced in v0.19. No flag to disable — IFB is implicit.
- The Orca-era "36.9× over FasterTransformer on GPT-3 175B" number from the literature is the dramatic-but-misleading public figure; modern frameworks all do IFB by default and the meaningful comparison is between modern IFB implementations.
- **No NVIDIA-published BF16 70B in-flight-batching delta exists.** The IFB-vs-static comparison is treated as "obviously won" and skipped in benchmark publications.

### KV cache reuse / prefix caching — the only numerically-anchored data we have

**SqueezeBits direct vLLM-vs-TRT-LLM benchmark** (1× A100-SXM 80GB, Llama-3.1-**8B**-Instruct, BF16, max_batch_size=128, ISL 1K/2K/4K, OSL 1K, 256 requests, Dynamic-Sonnet shared-prefix dataset):

| Engine | Throughput delta | TPOT delta |
|---|---|---|
| TRT-LLM v0.15.0 (block reuse ON) | **+34.7%** | **−20.9%** |
| vLLM v0.6.3 (prefix caching ON) | +13.3% | −9.8% |

Random (no shared prefix) dataset:
- vLLM: −36.7% throughput regression with prefix caching ON (real overhead when the cache doesn't hit)
- TRT-LLM: "relatively minor overhead" (no exact number given)

**Caveat:** This is 8B BF16 on A100, not 70B on GH200. Extrapolating: the prefix-caching win at 70B should be *larger* in tok/s terms (decode is bandwidth-bound; cache hits skip more bandwidth-hungry attention) but the percentage delta is roughly the same shape. Also note the SqueezeBits benchmark used TRT-LLM v0.15.0 (Sep 2024) — the v1.2.1 prefix-caching code has been heavily refactored since.

**NVIDIA "5× faster TTFT" claim** (https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/):
- Specifically describes the "early reuse" optimization: KV blocks become available for reuse *before* the producing request completes (vs the default behavior where blocks stay locked until termination).
- "Up to 5×" applies to *time-to-first-token* in shared-system-prompt enterprise chatbot scenarios with traffic surges. Not a general throughput number.
- Same blog: reducing KV cache block size from 64 tokens to 8 tokens gives "up to 7% TTFT improvement on LLAMA70B on H100 in multi-user environments." This is the closest published Llama-70B number for the KV-reuse feature.

### Chunked prefill — no published Llama-70B BF16 benchmark exists

The NVIDIA chunked-prefill blog (https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/) explains the mechanism but **publishes zero benchmark numbers**. **Chunked prefill is still default OFF in v1.2.x** (verified against the LLM API reference) despite Round 1 calling it default-on. Turn it on explicitly for any workload with mixed ISL.

### Net delta estimate for GH200 BF16 70B (engineering estimate, not measured)

| Feature | Estimated delta vs no-feature baseline |
|---|---|
| In-flight batching (continuous batching) | Implicit / always on; vs static batching ≈ 2–5× at concurrency ≥ 16 |
| Chunked prefill | +5–15% throughput on mixed-ISL workloads; **+30–50% TTFT** on long-ISL requests behind short ones |
| KV cache reuse — shared system prompts | **+20–35% throughput, +20% TPOT improvement** (SqueezeBits 8B extrapolated) |
| KV cache reuse — random prompts | +0–2% (basically a wash); no regression like vLLM |
| KV cache early reuse | up to 5× TTFT on burst traffic with shared prompts |
| Host-cache offload (200 GB to Grace LPDDR5X via NVLink-C2C) | enables 4–10× larger effective KV pool; specific tok/s gain not published |

**The single highest-leverage feature on GH200** for 70B-class is the host-cache offload to Grace LPDDR5X. No vLLM equivalent exists; Baseten's "GH200 +32% over H100" delta on Llama-3.3-70B (Feb 2025) is largely attributed to this mechanism by their own write-up.

Sources: https://blog.squeezebits.com/vllm-vs-tensorrtllm-12-automatic-prefix-caching-38189 ; https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/ ; https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/ ; https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/

---

## Q5. Multi-LoRA serving on TRT-LLM 1.2.1 — does it actually work with PyTorch backend on aarch64?

### Functional status

**Yes, multi-LoRA works on the PyTorch backend.** Verified from `features/lora.html`:
- *"Load and apply multiple LoRA adapters simultaneously"* — explicit doc quote.
- LoraConfig fields confirmed: `lora_dir`, `lora_target_modules`, `max_lora_rank`, `max_loras`, `max_cpu_loras`, `lora_ckpt_source`.
- HuggingFace and NeMo adapter formats supported.

### Serving API

Via `trtllm-serve` OpenAI-compatible endpoint:
```python
response = client.completions.create(
    model="/path/to/base/model",
    prompt="…",
    extra_body={
        "lora_request": {
            "lora_name": "lora-example-0",
            "lora_int_id": 0,
            "lora_path": "/path/to/lora_adapter",
        }
    }
)
```
This API mirrors vLLM's adapter-request pattern closely. Adapters can be hot-swapped per-request inside an in-flight batch.

### aarch64 specifics — caveat list

| Concern | Status |
|---|---|
| Multi-LoRA documented as PyTorch-backend feature | Yes, verified |
| Tested on aarch64 / GH200 in NVIDIA's CI | **No public evidence either way.** NVIDIA's CI matrix on the public GH Actions side is x86_64-dominated. |
| Listed in NIM 2.0.5 GH200 support matrix | Yes — "LoRA verified across all TP levels" per the NIM docs |
| Known aarch64-specific LoRA bugs | None found in the public issue tracker as of 2026-05-25 |
| Engine-build path multi-LoRA on aarch64 | Painful — requires pre-baking adapter set into engine; not recommended on aarch64 where build infra is rougher |

**Operational recommendation:** Multi-LoRA on aarch64/GH200 should work via the PyTorch backend + `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`. Treat the first deployment as a smoke test — load 2 adapters, hot-swap per request, verify the responses differ correctly. There is no proof-of-correctness benchmark or NVIDIA blog post specifically demonstrating multi-LoRA on aarch64 GH200. If correctness matters, build a small accuracy regression test (5 prompts × 2 adapters × N=10 runs) before production.

Sources: https://nvidia.github.io/TensorRT-LLM/features/lora.html ; https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html ; https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html

---

## Q6. NGC container manifest — does `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` have an arm64 manifest?

### What we can verify from outside the cluster
- **NGC catalog page** (https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release) renders client-side; WebFetch returns only the page title and a banner image. We **cannot read the manifest table directly** via WebFetch from this environment.
- Search results confirm community users have run `nvcr.io/nvidia/tensorrt-llm/release:1.1.0rc0` and `1.2.0` and `1.3.0rc10` on **aarch64 / GH200** in production, strongly implying the multi-arch manifest exists.
- v1.3.0rc13 release notes explicitly add "LLM_SBSA_WHEEL_DOCKER_IMAGE" and "Restored missing aarch64 library" — telling us NVIDIA acknowledges the SBSA path and is actively maintaining it.
- v1.3.0rc15 (the latest RC) release notes include: *"SBSA wheel image support for aarch64/GH200"* — confirming SBSA aarch64 is a first-class build target as of late May 2026.
- The TRT-LLM installation doc (https://nvidia.github.io/TensorRT-LLM/installation/installation-guide.html) names PyTorch NGC Container as the supported install path on SBSA, implying the matching TRT-LLM release container also has SBSA support.

### What we cannot verify from outside
- The exact set of platforms in the **`1.2.1` tag's manifest list** (vs the newer 1.3.0rc15). It's *possible* — though not consistent with the community-reported pattern — that 1.2.1 lacks an aarch64 manifest and only the 1.3.0rc series has the full SBSA story.

### Verification command to run on the actual GH200 host
```bash
docker manifest inspect nvcr.io/nvidia/tensorrt-llm/release:1.2.1 \
  | jq '.manifests[] | {platform: .platform, digest: .digest}'
# Expect at minimum:
#   {"platform":{"architecture":"amd64","os":"linux"}, ...}
#   {"platform":{"architecture":"arm64","os":"linux"}, ...}
```

Or just attempt the pull from a GH200 — Docker auto-selects the right manifest:
```bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.2.1
docker run --rm nvcr.io/nvidia/tensorrt-llm/release:1.2.1 uname -m
# Should print: aarch64
```

### Confidence call
**Medium-high** that `release:1.2.1` carries an arm64/SBSA manifest. **High** that `release:1.3.0rc15` does (explicit release note). If you want zero-risk, use `:1.3.0rc15` on aarch64 and accept the pre-release status; otherwise default to `:1.2.1` and verify with the commands above before committing.

Sources: NVIDIA TRT-LLM release notes ; NGC catalog page (title only retrievable) ; forum thread https://forums.developer.nvidia.com/t/arm64-gh200-llm-engine-issues/339136 ; v1.3.0rc15 release notes

---

## Q7. `trtllm-serve` OpenAI-compat endpoint — feature parity with vLLM as of May 2026

### Endpoints confirmed (from `commands/trtllm-serve/trtllm-serve.html`)

| Endpoint | trtllm-serve | vLLM | Notes |
|---|---|---|---|
| `/v1/models` | ✓ | ✓ | parity |
| `/v1/completions` | ✓ | ✓ | parity |
| `/v1/chat/completions` | ✓ | ✓ | parity |
| `/v1/embeddings` | ✗ | ✓ | **vLLM-only** |
| `/v1/audio/*` | partial (multimodal beta) | ✓ (some) | both fragmented |
| `/health` | ✓ | ✓ | parity |
| `/metrics` (Prometheus) | ✓ | ✓ | parity |
| `/version` | ✓ | ✓ | parity |

### Feature parity matrix

| Feature | trtllm-serve 1.2.1 | vLLM 0.10+ | Notes |
|---|---|---|---|
| Streaming SSE | ✓ | ✓ | parity |
| Function/tool calling | ✓ via `--tool_parser` | ✓ | TRT-LLM added the flag in 1.x series; per-model parser plugins |
| Reasoning models (deepseek-r1, qwen3) | ✓ via `--reasoning_parser` | ✓ | parity |
| Structured outputs (JSON Schema) | ✓ via XGrammar-2 integration | ✓ via XGrammar/Outlines/lm-format-enforcer | Both use XGrammar-2 as of 2026 — see MLC blog |
| Multi-LoRA per-request | ✓ via `extra_body.lora_request` | ✓ via `extra_body.lora_request` | parity |
| Speculative decoding (server-side) | ✓ EAGLE3 only on PyTorch backend | ✓ EAGLE/Medusa/draft-target | **vLLM has wider spec-decode coverage; TRT-LLM has the SOTA EAGLE3 path** |
| KV cache reuse / prefix caching | ✓ `enable_block_reuse` | ✓ `--enable-prefix-caching` | parity (TRT-LLM's gain percentage is higher per SqueezeBits) |
| Chunked prefill | ✓ `--enable_chunked_prefill` (default OFF) | ✓ default ON | vLLM defaults more aggressively |
| Disaggregated serving | ✓ via KV cache connector + transceiver v2 | ✓ via P/D split | parity |
| Multimodal (image/video/audio) | ✓ beta (Gemma4, Kimi K2.5, Llama4) | ✓ | both supported, models differ |
| Logprobs in streaming | ✓ in completions; streaming logprobs less polished | ✓ | vLLM slightly ahead on streaming logprobs |
| Token healing | not documented | partial | edge case |
| OpenAI Harmony Response Format | ✓ via XGrammar-2 Structural Tag | ✓ via XGrammar-2 | parity |
| KV cache offload to host memory | ✓ `host_cache_size` (Grace LPDDR5X on GH200) | partial (CPU swap) | **TRT-LLM advantage on GH200** |

### Net assessment
**Feature parity is approximately 90%.** The places trtllm-serve trails vLLM:
1. **No native `/v1/embeddings` endpoint** — if you need an OpenAI-compat embeddings server, you still need a separate vLLM or NIM embedding container.
2. **Narrower speculative-decoding coverage on the PyTorch backend** (EAGLE3 only). vLLM supports EAGLE, Medusa, draft-target, and lookahead at the serving layer.
3. **Chunked prefill default OFF in v1.2.x** — vLLM has it default ON. Easy to fix with the flag, but easy to forget.

Places trtllm-serve leads vLLM:
1. **Host-cache offload to Grace LPDDR5X** is a clean GH200-specific advantage.
2. **EAGLE3 + reasoning parser combo** has tighter integration in TRT-LLM (NVIDIA-published EAGLE3 weights, dedicated `eagle3_layers_to_capture` config).
3. **Prefix-caching throughput gain is larger** per SqueezeBits (TRT-LLM +34.7% vs vLLM +13.3% in shared-prefix benchmark).

Sources: https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html ; https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation ; https://docs.vllm.ai/en/stable/serving/openai_compatible_server/

---

## Q8. NIM availability for headline LLMs on aarch64 — still amd64-only?

### Direct verification

**NVIDIA's developer forum has an open complaint thread** ("Missing official native ARM64 NIM images for essential AI models" — https://forums.developer.nvidia.com/t/missing-official-native-arm64-nim-images-for-essential-ai-models/350681) **last active 20 March 2026** with **no official NVIDIA roadmap response and no timeline commitment**. User-named models lacking arm64 NIM:
- Llama 2 70B (primary complaint)
- BioNeMo NIMs (most)
- NMT NIM (`riva-translate-1.6B`)
- GLM-5 (NVFP4 variant availability questioned)
- Various embedding models including `llama-3.2-nv-embedqa-1b-v2`

NVIDIA staff responses on-thread (the closest to official guidance) suggest **using TRT-LLM, vLLM, or SGLang directly** as alternatives to NIM on arm64 — effectively conceding that the official NIM path is not yet there for arm64.

### A separate forum thread

**"NIM does not support llama-3.1-8b-instruct and llama-3.1-70b-instruct on GH200 On-Prem deployment"** (https://forums.developer.nvidia.com/t/nim-does-not-support-llama-3-1-8b-instruct-and-llama-3-1-70b-instruct-on-gh200-on-prem-deployment/312542). Quote: *"The requested image's platform (linux/amd64) does not match the detected host platform (linux/arm64/v8)."* This is the exact failure mode users hit when pulling Llama 3.1 NIM images on GH200.

### What IS available on arm64 NIM (subset)
- A small set of LLM NIMs *are* listed against the GH200 hardware in the NIM support matrix (Llama-3.1-8B, Llama-3.1-70B, Llama-3.3-70B, Llama-3.3-Nemotron-Super-49B-v1.5, Nemotron-3-Nano, Nemotron-3-Super-120B-a12b, Starcoder2-7B, GPT-OSS-20B/120B per Round 1).
- **The catch:** the NIM support matrix lists *hardware compatibility* (which GPUs run the model after the container starts), not *container architecture* (whether the image manifest has linux/arm64). The forum complaints suggest the *image* is still linux/amd64 for many of these listings, even though the matrix says GH200 is "supported." Users get the platform-mismatch error at `docker pull` time.
- BioNeMo OpenFold3 has a dedicated "Prerequisites for arm64/aarch64 DGX Systems" doc page — confirming at least *one* biology NIM has a proper arm64 image.

### Net answer

**As of May 2026, the NIM catalog is still predominantly amd64-only.** A small subset of LLM and BioNeMo NIMs have arm64 images; the bulk do not. NVIDIA has acknowledged the gap in forum threads but has not published a public arm64-NIM roadmap or timeline.

**For GH200 LLM deployment, the practical recommendation remains: skip NIM, use TRT-LLM 1.2.1 directly via `nvcr.io/nvidia/tensorrt-llm/release:1.2.1`** (which does have arm64 support per Q6) **and run the open-source serving stack.** You give up NVIDIA's "validated profile + ModelOpt-packaged weights" convenience but you get a working aarch64 path.

Sources: https://forums.developer.nvidia.com/t/missing-official-native-arm64-nim-images-for-essential-ai-models/350681 ; https://forums.developer.nvidia.com/t/nim-does-not-support-llama-3-1-8b-instruct-and-llama-3-1-70b-instruct-on-gh200-on-prem-deployment/312542 ; https://docs.nvidia.com/nim/bionemo/openfold3/1.2.0/prerequisites-arm64-with-dgx-stack.html ; https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html

---

## Round 4 Net Recommendations Update (delta from Round 3)

1. **Default LLM API config for Llama-3.3-70B on GH200 (BF16):** Use `enable_chunked_prefill=True` explicitly (it is default OFF in v1.2.x, contrary to Round 1's claim), `kv_cache.free_gpu_memory_fraction=0.85`, `kv_cache.host_cache_size=200GB`, `enable_block_reuse=True`. **70B BF16 weights do not fit on a single GH200-96GB GPU** — you need GH200-144G HBM3e, GH200-NVL2, or weight quantization (W4A16 AWQ recommended).

2. **Engine-build path:** Skip unless you have a profiled, measurable >10% gap on a stable model. PyTorch backend is canonical and the build tax is real (30–90 min for 70B, target GPU required, version-pinned).

3. **EAGLE3 recipe:** Use the NVIDIA HF model card config verbatim (`max_draft_len=3`, `eagle3_layers_to_capture=[-1]`). The "1.6–2.6× at low concurrency" claim is consistent with the published MT-Bench acceptance rates (2.1–3.2 token average) but is not a single NVIDIA-published Llama-3.3-70B + EAGLE3 + GH200 benchmark number. Expect **1.7–2.7× at batch=1**, degrading rapidly with concurrency. NVIDIA's recipe **disables block reuse** when EAGLE3 is on — test both with your traffic.

4. **In-flight batching + KV cache reuse deltas:** No published BF16 70B numbers. Best engineering estimates: IFB ~2–5× over static batching at concurrency ≥ 16, prefix caching +20–35% throughput on shared-prompt workloads (per SqueezeBits 8B data extrapolated), early reuse up to 5× TTFT on burst traffic.

5. **Multi-LoRA on PyTorch backend on aarch64:** Documented as working. No NVIDIA-published proof on GH200 specifically. Treat first deployment as a correctness smoke test.

6. **NGC container 1.2.1 arm64 manifest:** Medium-high confidence it exists. Verify with `docker manifest inspect` on the actual GH200 host before committing. v1.3.0rc15 has it explicitly per release notes.

7. **`trtllm-serve` ≈ 90% feature-parity with vLLM 0.21.** Gaps: no embeddings endpoint, narrower spec-decode coverage, chunked-prefill default OFF. Wins: host-cache offload to Grace, tighter EAGLE3 integration, larger prefix-cache throughput gain.

8. **NIM on aarch64 is still amd64-dominant.** Skip NIM for GH200 LLM deployment in May 2026; use the raw TRT-LLM 1.2.1 release container.

---

## Updated source index (verified 2026-05-25)

New / updated in Round 4:
- LLM API reference https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html
- KV cache features https://nvidia.github.io/TensorRT-LLM/features/kvcache.html
- LoRA features https://nvidia.github.io/TensorRT-LLM/features/lora.html
- Speculative decoding features https://nvidia.github.io/TensorRT-LLM/features/speculative-decoding.html
- trtllm-serve CLI https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html
- Llama4 Maverick + EAGLE3 tech blog https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog6_Llama4_maverick_eagle_guide.html
- NVIDIA HF EAGLE3 card https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-Eagle3
- NVIDIA 3.6× spec-decode blog https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/
- NVIDIA 70B 3× spec-decode blog https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/
- NVIDIA 5× TTFT KV-reuse blog https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/
- NVIDIA chunked prefill blog https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/
- NVIDIA new KV reuse optimizations blog https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/
- SqueezeBits prefix-caching vLLM-vs-TRT-LLM https://blog.squeezebits.com/vllm-vs-tensorrtllm-12-automatic-prefix-caching-38189
- EAGLE-3 paper https://arxiv.org/pdf/2503.01840
- XGrammar-2 launch blog https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation
- TRT-LLM GH issue #8564 (PyTorch vs engine perf) https://github.com/NVIDIA/TensorRT-LLM/issues/8564
- vLLM Llama-3.3-70B recipe https://docs.vllm.ai/projects/recipes/en/latest/Llama/Llama3.3-70B.html
- NIM LLM support matrix https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html
- Missing arm64 NIM forum thread https://forums.developer.nvidia.com/t/missing-official-native-arm64-nim-images-for-essential-ai-models/350681
- NIM GH200 platform-mismatch thread https://forums.developer.nvidia.com/t/nim-does-not-support-llama-3-1-8b-instruct-and-llama-3-1-70b-instruct-on-gh200-on-prem-deployment/312542
- BioNeMo OpenFold3 arm64 prereqs (one of the few arm64 NIMs) https://docs.nvidia.com/nim/bionemo/openfold3/1.2.0/prerequisites-arm64-with-dgx-stack.html
- BentoML TRT-LLM tuning guide https://www.bentoml.com/blog/tuning-tensor-rt-llm-for-optimal-serving-with-bentoml
- Introl TRT-LLM optimization guide https://introl.com/blog/tensorrt-llm-optimization-nvidia-inference-stack-guide

---

## Confidence calibration

### High confidence
- LLM API constructor parameter list and KvCacheConfig defaults (direct from official docs)
- EAGLE3 recipe (verbatim from NVIDIA HF model card)
- PyTorch backend supports only EAGLE3 of the EAGLE family (verbatim doc quote)
- trtllm-serve endpoint surface (verbatim from CLI docs)
- NIM catalog is mostly amd64-only (multiple confirming forum threads, no NVIDIA refutation)
- SqueezeBits prefix-caching numbers (+34.7% / −20.9%)
- NVIDIA 70B + draft-target speedup numbers (51.14 → 181.74 tok/s, etc., verbatim from Dec 2024 blog)
- v1.2.1 release content (KV cache corruption fix + xgrammar/flashinfer upgrade)
- v1.3.0rc15 has SBSA wheel image support per release notes

### Medium confidence
- `release:1.2.1` carries a linux/arm64 manifest — strong indirect evidence; requires `docker manifest inspect` on the host to confirm
- The "1.6–2.6×" EAGLE3 + Llama-3.3-70B claim maps to the published acceptance-rate math (2.1–3.2 / ~1.2 = 1.7–2.7×) but is not a single NVIDIA-published benchmark
- PyTorch backend ~5–15% slower than compiled engine on a tuned workload — consensus across multiple secondary sources, no first-party NVIDIA confirmation
- KV-cache-reuse 70B benchmark deltas extrapolated from 8B SqueezeBits numbers — directional only

### Low confidence / still unknown
- Whether the v1.2.1 NGC release container has the *same* SBSA quality as v1.3.0rc13+ (release notes only explicitly cite 1.3.0rc13 for the "restored aarch64 library" fix; 1.2.1 may carry older SBSA wheels)
- Real measured tok/s for Llama-3.3-70B BF16 on a single GH200-96 or GH200-144 (no public number exists — confirmed unchanged from Round 3)
- Multi-LoRA on aarch64 correctness at scale (no public test; needs smoke test on actual hardware)
- Whether `enable_block_reuse: false` is *required* with EAGLE3 in v1.2.1 or just NVIDIA's conservative default (v1.3.0rc15 added "hidden-state reuse across CUDA graph captures" suggesting the integration is still maturing)
