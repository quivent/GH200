# GH200 Inference — Round 4, Agent #05: Modular MAX Deep Dive (LLM-only)

**Author**: Research Agent #05, Round 4
**Date**: 2026-05-25
**Scope**: Drill the seven open questions on Modular MAX for **LLM inference on GH200 (BF16)** — no diffusion/video.
**Stack constraint**: FP8 broken on this GH200 stack; BF16 mandated for diffusion; FP8 OK for dense Hopper LLMs (per round-1/round-3 note).

---

## TL;DR — answers to the seven questions

1. **MAX 26.3 changelog (LLM bits)**: multi-GPU via `max.experimental.sharding` (DeviceMesh + DTensor) shipped; heterogeneous TP-prefill / DP-decode in MLA KV transfer; KV cache moved to `max.pipelines.kv_cache`; new server flags `--allow-unsupported-logprobs`, `--allow-extra-request-fields`, `--tool-parser none`, `--reasoning-parser none`; MiniMax-M2 / Qwen3-MoE multi-GPU; NVFP4 matmul that "beats FlashInfer on B200". No GH200-specific LLM line items.
2. **Mojo 1.0 Beta**: real, shipped 2026-05-07 alongside MAX 26.3 (`v1.0.0b1`). New `TileTensor` (compile-time-checked memory layouts), `HopperMatmulSM90Kernel` family. Expanded GPU targets list (Apple Metal M5, AMD MI250X, NVIDIA B300 sm_103a) — **GH200 sm_90 was already covered**; no GH200 line in the 1.0b1 changelog itself. Direct relevance to GH200 LLM inference is **medium** (better kernel authoring path; Mojo source is how you'd hand-tune a sm_90 GEMM).
3. **Real published throughput on Hopper**:
   - **MAX vs vLLM on H100 SXM5 80GB, Llama-3.1-8B-Instruct FP8** (Spheron, 2026-05-13): MAX **2,150 tok/s** vs vLLM 1,850 vs SGLang 1,920 vs TRT-LLM 2,100 at 50 concurrent. MAX is **~16% over vLLM, ~2% over TRT-LLM** on that workload.
   - **MAX vs vLLM on A100, Llama-3.1-8B** (Vast.ai): MAX **89.9 tok/s** vs vLLM 75.9 (single-request streaming). 48.5% TTFT win.
   - **MAX vs vLLM on Hopper-class H20, Qwen3-14B** (Modular issue #6134, **still OPEN**): MAX **3,261.57 tok/s** vs vLLM **4,260.80 tok/s** — **MAX is 23% SLOWER on H20**. Mean TTFT 133 s (MAX) vs 82 s (vLLM). Comment in the issue: "MAX performs better than vLLM on A100, A40, L40S; performs **worse** than vLLM on H20 and L20."
   - **No published Llama-3.3-70B / Qwen2.5-72B / Mistral-Large numbers for MAX on H100/H200/GH200/B200**. Modular's own benchmark page is a tutorial, not a results table; no GH200 entries.
4. **MAX vs Triton + TRT-LLM**: MAX wins on small-dense-model latency and developer ergonomics (single binary, OpenAI endpoint out of the box, no compile step) on A100-class hardware. **TRT-LLM still beats MAX on H100 at high concurrency for Llama-3.1-8B-FP8** (2,780 vs 2,760 at 100 concurrent — essentially tied; TRT-LLM is barely ahead but has 25× longer cold start). **Triton is irrelevant as a direct comparison** — it's a model server that hosts TRT-LLM/vLLM backends. The real choice is MAX vs (Triton hosting TRT-LLM). MAX wins on cold start (~8 min vs ~28 min for TRT-LLM), wins on operational simplicity, ties on peak throughput.
5. **`max serve` OpenAI endpoint**: production-grade in the conventional sense — function calling, structured output, prefix caching, speculative decoding, FP8 KV cache, KV offload to host/disk, LMCache integration, batched `/v1/batches` API since 25.5, Prometheus + OTel metrics since 26.2, OpenAI Responses API (`/v1/responses`) for image/video since 26.2. **What it lacks vs Dynamo / llm-d**: no first-party P/D disaggregation, no KV-cache-aware router, no NIXL transport. Single-replica or DP-replica only.
6. **`whl.modular.com/nightly` aarch64**: index URL is valid and current (relocated 2026-01-07 from `dl.modular.com`). **However**, Modular's `packages` doc explicitly lists supported Linux as **Ubuntu 22.04 LTS x86-64 OR AWS Graviton2/3** — there is **no explicit aarch64 Grace Hopper line** for the pip wheel. The advertised arm64 path is **the Docker container** (`modular/max-nvidia-full` and `max-nvidia-base` — both confirmed multi-arch `linux/amd64 + linux/arm64` on Docker Hub, latest stable `26.3.0`, nightly tag rebuilt ~4h ago at fetch time). **Practical guidance: use the container on GH200, not pip.**
7. **BF16 on ARM CPU devices in MAX graphs (26.3) — hybrid CPU+GPU LLM inference?**: The 26.3 changelog line is *"Added support for the `bfloat16` data type on ARM CPU devices in MAX graphs."* That enables **ARM CPU as a compute device for BF16 tensor ops in a graph** — relevant to CPU embedding pre/post-processing, BF16 activations on Grace, and **mixing CPU and GPU stages in a single graph**. It is **NOT a published feature for splitting an LLM's KV cache between Grace LPDDR5X and Hopper HBM3e** — that's the CUDA UM / unified-memory story (separate), exploited by the MAX KV cache offload (`--kv-cache-format`, host/disk spill) that has been in MAX since 26.2. The two features compose, but no Modular blog post markets MAX as a "Grace-aware hybrid" engine. **In practice, you get the unified-memory benefit free from the CUDA driver; MAX 26.3's ARM-CPU-BF16 is a separate kernel-level enabler with weaker LLM relevance.**

**Bottom line**: MAX on GH200 is **functionally viable** (arm64 container exists, BF16 path is real, OpenAI endpoint works) and **performance-credible on Hopper for dense small LLMs** (8B beats vLLM by 15-20%). But there is **zero published 70B GH200 number from Modular**, an **open Hopper-class perf regression on H20** (issue #6134), and **no P/D disaggregation**. **For GH200 BF16 70B production, Dynamo+vLLM remains the lower-risk choice; MAX is the right pick only if you've measured your specific workload and the cold-start / ergonomics win matters more than P/D.**

---

## 1. MAX 26.3 LLM-relevant features (full enumeration)

**Source**: https://docs.modular.com/max/changelog/ , https://www.modular.com/blog/modular-26-3-mojo-1-0-beta-max-video-gen-and-more
**Date shipped**: 2026-05-07 (stable). Nightly tag rebuilt continuously.

### Multi-GPU / distributed inference (new in 26.3)
- `max.experimental.sharding` — `DeviceMesh` + placement primitives + DTensor. **This is MAX's first first-class TP/EP/PP API.** Previously, multi-GPU was bolted on per-model.
- Gemma 3 ModuleV3 is the first model on this path.
- MiniMax-M2 / MiniMax-M2.7: **MoE on 4× H100** with FP8, DP+EP execution, **auto overlap scheduling**, device-graph capture.
- **Two-phase prefill execution** under overlap scheduler when in "distributed-inference prefill role" — partial step toward P/D (still single-engine, not separate processes).
- **Heterogeneous TP-prefill / DP-decode in MLA KV transfer** (e.g., tp4 prefill → DP decode pool). Relevant for DeepSeek-V3.2 / Kimi-style MLA models. **Not a full P/D split**, just a transfer layout option inside one engine.
- Qwen3-MoE: TP + FP8 support.

### Serving / OpenAI endpoint
- `--allow-unsupported-logprobs` — pass through logprob requests even on unsupported architectures.
- `--allow-extra-request-fields` — relax strict OpenAI schema validation (better compatibility with client libraries that ship extra fields).
- `--tool-parser none` and `--reasoning-parser none` — disable tool-call parsing / chain-of-thought block extraction when not needed.
- Two-phase prefill exposed under overlap scheduler.

### KV cache
- KV cache module relocated `max.kv_cache` → `max.pipelines.kv_cache` (breaking import path).
- Carry-over from 26.2: KV offload spill GPU→CPU→disk; FP8 KV (`--kv-cache-format float8_e4m3fn`); LMCache cross-instance sharing; CUDA graph auto-enabled for Llama with `max_batch_size`.

### Kernels / quantization
- NVFP4 grouped matmul: "now outperforms FlashInfer on B200" for Kimi K2.5 decoding and prefill (Modular blog; not on Hopper).
- NVFP4 support extended to Gemma 4.
- MXFP4 added for MiniMax-M2.
- Removed `use_blocking_impl` / `single_thread_blocking_override` knobs from custom-op infra.

### Models added in 26.3
- Qwen3, Qwen3-VL, Qwen3-MoE
- MiniMax-M2, MiniMax-M2.7
- Gemma 4 (ModuleV2), Gemma 3 ModuleV3
- Step-3.5-Flash, Qwen-Image, Z-Image
- Mamba state-space models
- FLUX.2 Klein (diffusion — out of LLM scope)
- Wan 2.1/2.2 video (out of LLM scope, plus FP8 broken)

### Python API (LLM-impacting)
- `InferenceSession.load_all()` returns `dict[str, Model]` keyed by `sym_name`; accepts `max.graph.Module`.
- `Graph.empty_module()` removed → use `Module()` directly.
- `auto_cast=True` weight-load flag: automatic float32 ↔ bfloat16 conversion (handy when source checkpoint is fp32 and you want BF16 on GH200).
- `CPUMetricsCollector` now a context manager with `get_stats()` — Grace-CPU instrumentation hook.
- `max.experimental.nn.Module.compile()` emits build/compile logs and wraps in tracer spans.

### NOT in 26.3 (would matter if it were)
- No P/D disaggregation (separate processes, NIXL transport).
- No KV-cache-aware router.
- No GH200-specific blog post or benchmark page.
- No DeepSeek-V4 in stable (only via dev preview alongside Dynamo 1.2-dev).
- No mention of NVL2 or multi-superchip topology.

---

## 2. Mojo 1.0 Beta — relevance to GH200 LLM inference

**Source**: https://mojolang.org/releases/v1.0.0b1/ , https://docs.modular.com/mojo/kernels/linalg/matmul/gpu/sm90/matmul_kernels/HopperMatmulSM90Kernel/

### What 1.0b1 changes (2026-05-07)
- **`TileTensor`** — successor to `LayoutTensor`. Memory layout (swizzles, strides, indexing) becomes a compile-time property of the tensor type → type checker catches GPU memory access bugs. Added `bitcast`, `flat_load()` / `flat_store()`, new `tile()` overload, `load()` / `load_linear()` default to `invariant=True` for immutable tensors.
- **Closure unification** + **conditional trait conformance** + `fn` keyword deprecation → easier kernel composition.
- **GPU target expansion**: Apple Metal (with print() + M5 MMA), AMD MI250X, NVIDIA **B300 (sm_103a)**. **Hopper sm_90 / GH200 was already in-tree before 1.0b1.**
- Public `HopperMatmulSM90Kernel` family: software-pipelined matmul, TMA loads, wgmma — the building blocks of a hand-rolled Llama BF16 GEMM on GH200.
- Mojo docs split off to `mojolang.org`; `docs.modular.com` is now MAX-only.

### GH200 LLM relevance — honest assessment
- **Direct relevance: medium-low.** Mojo 1.0b1 doesn't add new GH200 hardware-feature support. Hopper sm_90 codegen was already there.
- **Indirect relevance: medium.** If you ever need to **write a custom kernel** for a GH200 attention / matmul variant that MAX's bundled kernels don't cover (e.g., a custom MoE router, an unusual head-dim, a fused RoPE+attention), Mojo 1.0b1 is the language you'd reach for — and TileTensor makes that safer.
- **For straight LLM inference on standard architectures (Llama-3.x, Qwen-2.5/3, Mistral, DeepSeek-V3.x), you never touch Mojo.** You run `max serve` and the Hopper kernels are picked up from the package.
- **Caveat**: Mojo is **still beta**. The "Path to Mojo 1.0" plan targets final 1.0 in **late summer 2026**, so anything you commit in Mojo today might need updates by GA.

---

## 3. Published throughput on Hopper — what actually exists

### MAX vs vLLM vs SGLang vs TRT-LLM on H100 SXM5 (Llama-3.1-8B-Instruct FP8)

**Source**: https://www.spheron.network/blog/modular-max-mojo-gpu-cloud-llm-inference/
**Date**: 2026-05-13. Single H100 SXM5 80GB bare metal. MAX = `modular/max-openai-api:latest` (May 2026 stable). vLLM v0.18.0, SGLang v0.5.9, TRT-LLM v1.2.0.

| Concurrency | MAX | vLLM | SGLang | TRT-LLM |
|---|---|---|---|---|
| 1 req | 128 tok/s | 120 | 125 | 130 |
| 10 req | 740 | 650 | 680 | 710 |
| **50 req** | **2,150** | 1,850 | 1,920 | 2,100 |
| 100 req | 2,760 | 2,400 | 2,460 | **2,780** |

| Concurrency | MAX p50 TTFT | MAX p95 | vLLM p50 | vLLM p95 |
|---|---|---|---|---|
| 1 | 28 ms | 42 | 35 | 55 |
| 10 | 68 | 108 | 82 | 130 |
| 50 | 105 | 195 | 120 | 210 |
| 100 | 195 | 380 | 230 | 410 |

| Engine | Cold start (uncached) | Warm |
|---|---|---|
| MAX | ~8 min | ~65 s |
| vLLM | ~62 s (no compile) | ~62 s |
| TRT-LLM | **~28 min** | n/a |
| SGLang | ~58 s | ~58 s |

**Takeaway**: MAX is the **throughput leader at 50-concurrent** for Llama-8B-FP8 on H100 (+16% over vLLM, +2% over TRT-LLM), **TTFT leader** at all concurrencies, **cold-start middle of the pack** (8 min vs 28 min for TRT-LLM, 1 min for vLLM). This is FP8 — which is **legal for dense Hopper LLMs per the stack note** but **not for diffusion**.

### MAX vs vLLM on A100, Llama-3.1-8B-Instruct (Vast.ai)

**Source**: https://vast.ai/article/modular-max-vs-vllm-performance-comparison-on-vast-ai
- MAX: 89.9 tok/s (avg), 0.111s TTFT, 6.699s total
- vLLM: 75.9 tok/s (avg), 0.215s TTFT, 7.410s total
- Delta: **+18.5% throughput, +48.5% TTFT for MAX**. Sequential single-request methodology, 5 prompts. Versions not specified. Precision not specified.

### MAX 25.6 official: 3,860 tok/s on A100 (ShareGPTv3, Llama-3.1)

**Source**: Modular's MAX 24.6 / 25.6-era marketing.
- Llama-3.1 (size unspecified, likely 8B), A100, MAX-only NVIDIA kernels, >95% GPU utilization → **3,860 output tok/s on ShareGPTv3**.
- This is the "MAX vs vLLM 2×" headline number.

### MAX vs vLLM on H20 (Qwen3-14B) — **PERF REGRESSION, open**

**Source**: https://github.com/modular/modular/issues/6134 (open, labels: bug, max)

| Engine | Req throughput | Output tok/s | Mean TTFT | Mean TPOT |
|---|---|---|---|---|
| MAX | 16.03 req/s | **3,261.57 tok/s** | 133,134.59 ms | 153.10 ms |
| vLLM | 21.34 req/s | **4,260.80 tok/s** | 82,502.25 ms | 241.03 ms |

- **MAX is 23% SLOWER on H20** for Qwen3-14B.
- Reporter comment: *"MAX performs better than vLLM on A100, A40 and L40S, but performs worse than vLLM on H20 and L20."*
- **H20 is Hopper-class** (sm_90, China-export-compliant Hopper SKU). This is the closest analog to GH200 architecture in the issue tracker, and it's a regression. **GH200 is not directly mentioned.** TPOT being lower for MAX (153 vs 241 ms) suggests it's a prefill or batching pathology, not pure kernel speed.
- Issue is **still open as of 2026-05-25**.

### GH200 / GB200: zero published MAX numbers

- Modular's official benchmark page is a how-to-run-benchmarks tutorial; no GH200 entries.
- No Modular blog post in 2026 mentions GH200 by name in a throughput context.
- The 25.7 announcement (Nov 2025) says GH200/GB200 BF16 *works*; the 26.3 announcement says the same about ARM-CPU BF16 — neither attaches a tok/s number.
- The Modular MAX marquee benchmarks for 26.x target B200 (NVFP4 vs FlashInfer for Kimi K2.5) and H100 (the Spheron numbers above).
- Closest GH200 LLM benchmark from any vendor: **Baseten's GH200 Llama-3.3-70B test (Feb 2025)** — but that used **TRT-LLM FP8**, not MAX, and reported only "+32% vs H100" without absolute tok/s.

### Llama-3.3-70B / Qwen2.5-72B / Mistral-Large on MAX: nothing found

Direct searches for all three model families crossed with MAX/Modular returned **no Modular-published numbers**. Searches included Spheron, Vast.ai, Baseten, Modular blog, GitHub issues. The 70B-class Modular benchmark gap is real and is **the single biggest evidence gap for the GH200 BF16 production decision**.

---

## 4. MAX vs Triton + TRT-LLM — when is MAX worth picking?

### The comparison is asymmetric

- **Triton** is a model server. Its LLM throughput is set by the backend (TRT-LLM, vLLM, Python).
- **TRT-LLM** is an engine. It runs inside Triton (or standalone) and produces the actual kernels.
- **MAX** is **engine + server in one binary**.

The honest comparison is **`max serve` vs `tritonserver --backend tensorrtllm`** (the typical Hopper LLM stack).

### Verdict matrix (May 2026)

| Need | Pick MAX if... | Pick Triton+TRT-LLM if... |
|---|---|---|
| **Cold start** | You restart often; can't tolerate 28-min engine builds | You build once, redeploy rarely |
| **Peak throughput @ 100c on H100 FP8 (Llama-8B)** | Within ~1% (essentially tied, MAX 2,760 vs TRT-LLM 2,780) — pick either | Same; tied |
| **TTFT under load** | You care; MAX wins at all concurrency levels in the Spheron H100 test | You can pay TTFT for TPOT |
| **Operational simplicity** | You want one binary, one config, one OpenAI endpoint | You can run a multi-process model_repo + plan files |
| **Model diversity** | You serve 5+ architectures; MAX's model zoo (Qwen3, Gemma 4, MiniMax, Kimi, DeepSeek-V3.2) is up to date | You're locked to Llama / Mistral / Qwen-2.x |
| **70B / 72B / Mistral-Large in production** | You can measure MAX yourself — no published number | TRT-LLM has mature 70B engine recipes, validated for years |
| **GH200 arm64** | Container is multi-arch (verified); BF16 path works | Triton SBSA arm64 is also published; both work |
| **P/D disaggregation** | Not yet — wait for MAX or front it with Dynamo | TRT-LLM via Dynamo has full P/D |
| **KV-aware routing** | Not in MAX | Not in Triton either; Dynamo for that |
| **Speculative decoding** | Yes, in MAX 26.2 (EAGLE/MTP, typical acceptance, chunked prefill compatible) | Yes, TRT-LLM has more knobs |
| **Structured output / tool calls** | Yes, llguidance + native parser flags | Yes via TRT-LLM |
| **AMD support** | Yes (`modular/max-amd`), MI300X/MI325X/MI355X | Triton has limited AMD support |

### When MAX is the **wrong** choice

- You need **production-validated 70B numbers today**. MAX has none on GH200.
- You hit an H20/L20-class kernel pathology (issue #6134) and your hardware is one of those SKUs.
- You need **disaggregated P/D** at multi-node scale: use Dynamo+vLLM/TRT-LLM.
- You're locked into TRT-LLM `.engine` artifacts upstream (vendor pipeline, NIM, etc.).

### When MAX is the **right** choice

- Single-GH200 or dual-GH200-NVL2 BF16 deployment, dense 7B-13B-class LLM.
- AMD + NVIDIA mixed fleet — one container/codebase.
- Cold-start sensitivity (autoscaling, spot, evict-prone workloads).
- Latest model architectures (Kimi K2.5, MiniMax-M2, Qwen3-MoE) where TRT-LLM may lag.

---

## 5. `max serve` OpenAI endpoint — production-grade?

### Verdict: **yes, production-grade for serving** — but **not production-grade for multi-node orchestration**.

### What `max serve` does well (verified in changelog 25.5 → 26.3)

- **OpenAI-compatible endpoints**: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/batches` (since 25.5), `/v1/responses` (since 26.2 — for image+video gen). Streaming. Function calling. Structured output via llguidance (switched from XGrammar in 25.5).
- **Performance features**:
  - Prefix caching with accelerated Mojo token hashing (9-12× faster than Python `hash()`, since 25.5).
  - Speculative decoding (EAGLE/MTP, typical-acceptance, chunked-prefill compatible, since 26.2).
  - CUDA graph auto-enabled for Llama (26.2).
  - Overlap scheduling auto-enabled for select architectures.
  - FP8 KV cache (`--kv-cache-format float8_e4m3fn`).
  - KV cache spill GPU→CPU→disk; LMCache integration for cross-instance KV sharing.
- **Multi-GPU**: TP (since 25.x), DP (`--data-parallel-degree`, since 25.6 for Llama), PP, EP (MoE), since 26.3 the unified `max.experimental.sharding` DeviceMesh API.
- **Observability**:
  - Prometheus / OTel: `maxserve.num_requests_queued` gauge added 26.2; full metrics catalog.
  - `MAX_SERVE_SCHEDULER_STATS_LOG_INTERVAL_S` periodic batch metrics logging (since 25.6).
  - `/health` readiness endpoint (since 25.5).
- **Scheduling knobs**:
  - `MAX_SERVE_BATCH_PRIORITY=prefill_first|decode_first|balanced|per_replica` (26.2).
  - `--max-batch-size`, `--max-batch-input-tokens`, `--max-batch-total-tokens` (renamed in 26.1).
  - `--kvcache-ce-watermark` (CE scheduled if <95% full, 26.1).
- **Container**: `modular/max-nvidia-full:26.3.0` (production-ready GPU image, ~1.8 GB compressed amd64 / ~1.8 GB arm64). `modular/max-nvidia-base:26.3.0` (lightweight, needs host NVIDIA driver, ~930 MB amd64 / ~780 MB arm64).
- **Deployment templates**: AWS, GCP, Azure IaC; supports g5/A10G, g2/L4, ND-MI300X-v5.

### What it doesn't have (production gaps for multi-node)

- **No first-party disaggregated prefill/decode** (separate processes, KV-cache transport between them). The 26.3 "two-phase prefill under overlap scheduler" is an in-process scheduling tweak, not P/D.
- **No KV-cache-aware request router** (Dynamo's flagship feature).
- **No NIXL integration** (no UCX RDMA KV transport).
- **No native LMCache server** — MAX is an LMCache *client*, not the cache itself.
- **No Helm chart published by Modular** (Round-4 search returned only generic Helm/observability tutorials — Modular doesn't ship a chart). You roll your own K8s manifests.
- **No first-party rate limiting / auth** — front with an API gateway (Envoy, Kong, NGINX) like you would with vLLM.
- **Single-process model**: `max serve` is one process per replica. Horizontal scaling = run N replicas behind a load balancer; there's no node-aware scheduler.

### Production-grade rating

| Dimension | Rating |
|---|---|
| API surface (OpenAI) | **A** — full + extras (`/v1/batches`, `/v1/responses`) |
| Throughput / latency on H100 dense | **A-** (per Spheron numbers) |
| Throughput on Hopper-China SKUs (H20/L20) | **C** (issue #6134, open) |
| Multi-GPU single-node | **B** — DeviceMesh shipped 26.3 but new |
| Multi-node / P/D | **D** — not supported first-party |
| Observability | **B+** — Prometheus + OTel + scheduler logs |
| Container ecosystem | **B+** — multi-arch Docker, IaC templates, no Helm |
| Security / auth | **D** — bring your own gateway |
| Documentation | **B** — changelog is exhaustive; benchmark page is thin |

For **single-node GH200 BF16 dense LLM serving**: **production-grade**. For **multi-node GH200 cluster, P/D, KV-router**: **not yet** — Dynamo wins.

---

## 6. `whl.modular.com/nightly` aarch64 install path — verified

### Status of the index URL

- New nightly wheel index location: `https://whl.modular.com/nightly/simple/` (relocated 2026-01-07 from `dl.modular.com`).
- Internally it redirects to `us-central1-python.pkg.dev/distribution-prod-0a7c7b3/nightly/simple/` (GCP Artifact Registry). Live as of fetch.

### Supported platforms per Modular docs

Per https://docs.modular.com/max/packages/ , the **explicitly supported** Linux platforms are:
- Ubuntu 22.04 LTS x86-64
- **AWS Graviton2/3** (ARM64 CPU-only; not Grace Hopper)

The docs **do not list aarch64 with NVIDIA GPU (GH200)** as a tested pip-install target.

### What this means for GH200

1. **`pip install modular` MAY work on GH200 Ubuntu 24.04 aarch64**, but **it's not on the supported matrix**. Wheels for `aarch64` exist in the index (Graviton path), and they're often the same wheels used on GH200 — but you're off-paved-road.
2. **The supported aarch64 GH200 path is the Docker container**: `modular/max-nvidia-full:26.3.0` ships a `linux/arm64` manifest (confirmed 2026-05-25 on Docker Hub). All 19+ tags are multi-arch. `nightly` tag is rebuilt continuously.
3. **Round-1's "container arm64 availability not documented"** was wrong — the manifests have been there. Round-3's verification step (`docker manifest inspect`) confirms it.

### Recommended install commands for GH200 (May 2026)

```bash
# OPTION A (recommended): pull the multi-arch container, runs natively on arm64
docker pull modular/max-nvidia-full:26.3.0   # 1.8 GB compressed arm64
docker run --gpus all -p 8000:8000 \
  modular/max-nvidia-full:26.3.0 \
  max serve --model-path meta-llama/Llama-3.3-70B-Instruct --dtype bfloat16

# OPTION B (off-paved-road but often works): pip install on the host
uv pip install modular --index https://whl.modular.com/nightly/simple/ --prerelease allow
# or stable:
uv pip install modular
```

`pixi add modular` also works for a Conda-style environment but is less common in production.

---

## 7. BF16 on ARM CPU devices in MAX graphs — hybrid CPU+GPU Grace?

### The literal feature (MAX 26.3 changelog)

> *"Added support for the `bfloat16` data type on ARM CPU devices in MAX graphs."*

This is a **type-system + kernel** change: BF16 tensors can now live on an ARM CPU as a MAX graph device (`Device("cpu")` on an aarch64 host), with native BF16 ops compiled by Mojo for the ARM SVE/NEON path.

### What this enables (real)

- **CPU pre/post-processing in BF16**: tokenizer outputs, embedding-table lookups, RoPE bookkeeping done on Grace CPU in BF16 without bouncing through float32.
- **CPU embedding fallback** when GPU memory is tight: BF16 embedding matrices on Grace LPDDR5X (480 GB) addressed by GPU via NVLink-C2C at 900 GB/s.
- **Mixed-device graphs**: a single MAX graph can have some ops on `Device("gpu:0")` (Hopper HBM) and some on `Device("cpu")` (Grace LPDDR), with automatic transfers.
- **Verified equivalence path**: BF16 graph numerics match between ARM CPU and NVIDIA GPU execution — easier model debugging on Grace before pushing to the GPU.

### What this does NOT enable (the marketing temptation)

- **It is NOT a Grace-Hopper-aware hybrid LLM scheduler.** No paper / blog / changelog line says "MAX 26.3 runs Llama-70B prefill on Grace and decode on Hopper" or similar. That hybrid pattern is **research-grade** (see HALI, GLAKE, Kandinsky papers); MAX hasn't claimed it.
- **It does NOT change the KV-cache-offload story.** KV-offload to host RAM has been in MAX since 26.2 via the standard CUDA UM / cudaHostRegister path. ARM-CPU-BF16 is orthogonal to KV-offload (KV blocks aren't BF16-CPU-op'd; they're staged to GPU).
- **It does NOT make Grace LPDDR5X a first-class GPU memory tier in MAX scheduling.** The 480 GB of LPDDR5X is exposed via CUDA UM page migration, not via MAX's `Device("cpu")` abstraction.

### Hybrid CPU+GPU LLM inference exploiting Grace — current reality

- **What works today** (independent of MAX 26.3):
  - CUDA Unified Memory page migration at 900 GB/s C2C — useful for KV cache that overflows HBM (96-144 GB) into LPDDR5X (480 GB).
  - NVIDIA's published "2× KV-offload speedup vs x86-H100" (Llama 3 70B multi-turn) — TRT-LLM, not MAX.
- **What MAX 26.3 adds**: BF16 CPU ops in graphs — useful for **embedding** workloads (vector DBs / RAG) and **pre/post-processing pipelines**, modestly relevant to LLM token-gen.
- **Net LLM impact**: **small-to-moderate**. It's table-stakes for a serious GH200 inference stack but not differentiating. Don't expect a "2× from BF16-on-ARM" headline.

---

## Synthesis — MAX on GH200 vs vLLM/TRT-LLM, May 2026

### Where MAX wins

1. **Single-node H100 FP8 Llama-8B**: +16% vs vLLM, +2% vs TRT-LLM at 50 concurrent (Spheron). TTFT leader at every concurrency.
2. **A100 small LLM**: +18-48% vs vLLM (Vast.ai, Modular's own ShareGPTv3 number 3,860 tok/s).
3. **Cold start**: 8 min (MAX) vs 28 min (TRT-LLM) for Llama-8B engine build.
4. **Container ergonomics**: one image, OpenAI endpoint, multi-arch (verified).
5. **Latest model coverage**: Kimi K2.5, MiniMax-M2, Qwen3-MoE, DeepSeek-V3.2, Mamba — out of the box.
6. **Multi-vendor**: AMD MI300X/325X/355X in the same codebase.
7. **AMD + NVIDIA mixed fleet** — single container vs separate vLLM/TRT-LLM stacks.

### Where MAX loses or is silent

1. **No published 70B GH200 BF16 number.** This is the single biggest gap for the stack's primary use case.
2. **Hopper-China SKUs (H20, L20)**: 23% slower than vLLM on Qwen3-14B (issue #6134, **open**). GH200 perf risk by analogy is unverified but plausible.
3. **P/D disaggregation**: not supported first-party. Dynamo is the only mature path here on Hopper.
4. **KV-cache-aware router**: not supported. Dynamo / llm-d only.
5. **No GH200-specific benchmarks from Modular** — you must measure in-house.
6. **Pip aarch64 GH200 is off the supported matrix.** Container path works; pip is gray-zone.
7. **Mojo 1.0 still beta** through summer 2026 — any custom kernel work is on shifting ground.

### Recommendation for GH200 BF16 LLM production (May 2026)

| Workload | Recommendation |
|---|---|
| **Single GH200, Llama-3.3-70B BF16, OpenAI endpoint, ≤100 concurrent** | **MAX 26.3 container or vLLM aarch64.** Benchmark both on your traffic before committing. MAX has the cold-start + ergonomics edge; vLLM has the broader community-validated path on GH200. |
| **Dual GH200 NVL2, Llama-3.3-70B BF16, P/D split** | **Dynamo 1.1.1 + vLLM-runtime** (multi-arch confirmed). MAX cannot do P/D today. |
| **Multi-node GH200 cluster, KV-router** | **Dynamo 1.1.1.** Same as above. |
| **Mixed NVIDIA + AMD fleet** | **MAX 26.3.** Single binary advantage. |
| **Fast iteration / dev loop / many model variants** | **MAX 26.3.** Cold start + model zoo. |
| **Production 70B serving where you cannot afford a kernel regression** | **vLLM aarch64 build or TRT-LLM via Triton SBSA.** More community testing on GH200. |

### Direct answer to "is MAX worth picking on GH200 vs vLLM/TRT-LLM?"

**Conditionally yes**, but the conditions are narrower than Modular's marketing suggests:

- If your model is **dense 7B-13B** and your traffic is **interactive (low-concurrency, TTFT-sensitive)** — MAX likely wins. Verified on H100.
- If your model is **70B** — **no public evidence MAX wins on GH200**, and the H20 regression is a yellow flag. Benchmark before committing.
- If you need **P/D, KV-router, or multi-node orchestration** — MAX is the wrong tool; pick Dynamo.
- If you need **the latest open models (Kimi K2.5, MiniMax-M2, Qwen3-MoE) on day one** — MAX is the right tool; community engines lag by weeks/months.

**Not a compelling story on GH200 specifically.** It's a compelling story on H100 dense small LLMs and a viable story on GH200 — but the GH200 evidence base is thin enough that you should treat MAX as one of two candidates (the other being vLLM aarch64 / Dynamo), not the default.

---

## Sources

1. https://docs.modular.com/max/changelog/ — MAX 25.5 → 26.3 changelog (full)
2. https://www.modular.com/blog/modular-26-3-mojo-1-0-beta-max-video-gen-and-more — MAX 26.3 + Mojo 1.0 Beta announcement (2026-05-07)
3. https://mojolang.org/releases/v1.0.0b1/ — Mojo 1.0.0b1 release notes
4. https://docs.modular.com/mojo/kernels/linalg/matmul/gpu/sm90/matmul_kernels/HopperMatmulSM90Kernel/ — HopperMatmulSM90Kernel docs
5. https://docs.modular.com/max/serve/ — MAX serve features
6. https://docs.modular.com/max/tutorials/max-serve-local-to-cloud/ — local-to-cloud tutorial
7. https://docs.modular.com/max/deploy/local-to-cloud/ — deployment guide (AWS/GCP/Azure)
8. https://docs.modular.com/max/deploy/benchmark/ — benchmark tutorial (no GH200 numbers)
9. https://docs.modular.com/max/packages/ — supported platforms (no aarch64 GH200 listed)
10. https://docs.modular.com/max/container/ — container index
11. https://hub.docker.com/r/modular/max-nvidia-full/tags — multi-arch arm64 manifests verified
12. https://hub.docker.com/r/modular/max-nvidia-base/tags — multi-arch arm64 verified
13. https://forum.modular.com/t/change-in-location-of-our-nightly-wheel-index/2583 — whl.modular.com relocated 2026-01-07
14. https://forum.modular.com/t/pip-install-modular-now-available/1382 — pip install announcement (2025-05-06)
15. https://github.com/modular/modular/releases — release tags
16. https://github.com/modular/modular/issues/6134 — **OPEN**: MAX slower than vLLM on H20 / L20 with Qwen3-14B
17. https://github.com/modular/modular/issues/5867 — Closed, references GH200 only in passing (InternVL3 unrelated bug)
18. https://www.spheron.network/blog/modular-max-mojo-gpu-cloud-llm-inference/ — **MAX vs vLLM vs SGLang vs TRT-LLM on H100 SXM5 Llama-8B FP8, May 2026**
19. https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/ — H100 Llama-3.3-70B FP8 (no MAX, points to #18)
20. https://vast.ai/article/modular-max-vs-vllm-performance-comparison-on-vast-ai — MAX vs vLLM A100 Llama-8B
21. https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/ — GH200 Llama-3.3-70B (TRT-LLM, not MAX) +32% vs H100
22. https://developer.nvidia.com/blog/nvidia-gh200-superchip-accelerates-inference-by-2x-in-multiturn-interactions-with-llama-models/ — NVIDIA GH200 + KV offload (TRT-LLM, not MAX)
23. https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/ — GH200 KV offload 2× faster
24. https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience — 25.7 GH200/GB200 BF16 announcement (Nov 2025)
25. https://www.modular.com/open-source/max — MAX overview (no Hopper benchmark)
26. https://www.modular.com/max/solutions/rag-cag — MAX RAG/CAG marketing
27. https://docs.modular.com/max/intro/ — MAX intro

---

## Confidence

**High confidence**:
- `modular/max-nvidia-full` and `max-nvidia-base` are multi-arch (`linux/amd64` + `linux/arm64`) on Docker Hub — verified by fetching tag pages.
- MAX 26.3 (2026-05-07) shipped multi-GPU sharding, ARM-CPU BF16 in graphs, two-phase prefill under overlap scheduler, new server flags.
- MAX beats vLLM at 50-concurrent Llama-8B FP8 on H100 (+16%) per Spheron 2026-05-13.
- MAX is open issue #6134 SLOWER than vLLM on H20 (Hopper-China) for Qwen3-14B by 23%.
- MAX has no first-party P/D disaggregation, no KV-router, no NIXL.
- No Modular-published GH200 LLM throughput numbers exist.
- Mojo 1.0 Beta (`v1.0.0b1`) shipped 2026-05-07; final 1.0 targeted "late summer 2026".

**Medium confidence**:
- Whether `pip install modular` on Ubuntu 24.04 aarch64 GH200 actually works — wheels for aarch64 exist (Graviton path) and **likely** work on GH200, but it's off the supported matrix. Container is the supported path.
- Whether MAX's H20 regression (#6134) extends to GH200 (sm_90 standard Hopper) — issue specifically calls out H20/L20, and GH200 is a different SKU; could be unrelated, but it's the closest signal we have.
- Whether MAX 26.3's TP-prefill / DP-decode MLA path is mature enough for production DeepSeek-V3.2 serving on GH200. Changelog says it's added; no benchmark.

**Low confidence / gaps**:
- Llama-3.3-70B / Qwen2.5-72B / Mistral-Large performance on MAX-on-GH200. **No data exists.** Must be measured in-house.
- Whether MAX's BF16-on-ARM-CPU feature meaningfully accelerates any real LLM serving workload, vs. being a kernel-level enabler. No published end-to-end LLM benchmark exploits it.
- Whether Modular intends to ship P/D in 26.4/26.5 — no roadmap commitment found.

**Recommended Round 5 follow-ups**:
1. Pull `modular/max-nvidia-full:26.3.0` on the actual GH200, run `max serve --model-path meta-llama/Llama-3.3-70B-Instruct --dtype bfloat16`, and measure ShareGPTv3 throughput vs the same workload on vLLM aarch64 image. **This is the only way to resolve the 70B GH200 gap.**
2. Repeat the issue #6134 methodology (Qwen3-14B benchmark serving) on GH200 to see if the H20 pathology reproduces on standard Hopper.
3. Test whether the `--kv-cache-format float8_e4m3fn` flag is stable for KV cache on this stack (FP8 broken constraint applies to weights/activations on DiT; KV-cache FP8 on dense Hopper LLMs may be OK — verify).
4. Investigate whether `max.experimental.sharding` DeviceMesh handles 2× GH200 in an NVL2 chassis correctly (NVLink between GPU-0 and GPU-1 at 900 GB/s); compare to MAX's auto-DP path.
