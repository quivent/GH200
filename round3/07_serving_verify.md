# GH200 Inference — Round 3, Agent #07: Serving Frameworks Verification

**Author**: Research Agent #07, Round 3 (verification + deepening)
**Date**: 2026-05-25
**Scope**: Drill on six items left ambiguous in Round 1 (`07_serving_frameworks.md`).
**Stack constraint**: FP8 broken on this GH200 stack; BF16 default.

---

## TL;DR — corrections to Round 1

1. **Dynamo current version**: latest stable is **1.1.1 (2026-05-09)**, plus **1.1.0 (2026-05-04)**. There is **no 1.2.x stable** — only experimental `v1.2.0-deepseek-v4-dev.*` tags (most recent `dev.3` on 2026-05-09). Round 1 was correct; confirmed against the GitHub releases page.
2. **Dynamo arm64 multi-arch reach is BROADER than Round 1 said**. Per the `Release Artifacts` page, in 1.1.1 the **vLLM, SGLang, AND TensorRT-LLM** standard runtimes all ship as `AMD64/ARM64` multi-arch. The CUDA-12.9 vs CUDA-13 SGLang split is a **dev-only** artifact (visible in `v1.2.0-deepseek-v4-dev.2`); the 1.1.x stable line is uniformly multi-arch. The Round-1 "SGLang arm64-only on CUDA-13" claim was based on dev-branch artifacts and overstates the constraint. EFA variants of all three runtimes remain amd64-only.
3. **No published GH200 NVL2 P/D reference architecture exists.** The official GH200 NVL2 Enterprise RA (NVIDIA blog) is `2-2-3-400` (2 sockets / 2 GPUs / 3 NICs / 400 Gbps) for **single-node** LLM, RAG, recommender, GNN — it does **not** prescribe prefill/decode disaggregation across the two superchips. Dynamo's own recipe tree has **zero GH200 recipes**: `recipes/llama-3-70b/vllm/disagg-single-node` targets **8× H100/H200** (not GH200), and `disagg-multi-node` targets **16× H100/H200** across two nodes. GH200 P/D deployment on this stack will be a **port**, not a pull-down.
4. **NIXL does NOT exploit NVLink-C2C between Grace CPU and Hopper GPU as a transport.** The 900 GB/s number is the GPU↔CPU coherent link **inside one superchip**, used by the CUDA driver/UVM for unified-memory page migration — not by NIXL. NIXL backends are UCX (RDMA/IB/RoCE/TCP, optionally CUDA-IPC and NVLink for same-process peers), GDS (storage), Mooncake, POSIX, Libfabric, plus OBJ/S3. **Between GH200 nodes, NIXL uses InfiniBand or RoCE** (RDMA), not the C2C bus. The C2C bus does help indirectly because KV blocks living in pinned host LPDDR are still GPU-accessible at C2C bandwidth before/after the IB transfer, but the wire-format transport is still IB.
5. **llm-d is on v0.7.0 (2026-05-12)**, not v0.2/v0.3 as Round 1 implied. v0.4 (Dec 2025), v0.5 (Feb 2026), v0.6 (Mar 2026), v0.7 (May 2026). v0.7 added a **GB200 image variant**. **ARM64/GH200 issue #281 (opened 2025-09-30) is still OPEN with no 2026 activity** — confirmed not delivered.
6. **Modular MAX 26.3 (2026-05-07) explicitly added `bfloat16` on ARM CPU devices in MAX graphs**, which is the kernel-level enabler for GH200/GB200 BF16 — confirmed. **However, no published MAX→GH200 throughput numbers exist** in Modular's blog or changelog. MAX benchmark posts target H100/H200/B200/AMD; GH200 perf is implied by the BF16-on-ARM-host work but unquantified.

---

## 1. Dynamo current version — verified

**Sources:** https://github.com/ai-dynamo/dynamo/releases, https://docs.nvidia.com/dynamo/dev/resources/release-artifacts

| Tag | Date | Status | Notes |
|---|---|---|---|
| **1.1.1** | 2026-05-09 | **Stable (current)** | TRT-LLM scheduler/KV-reuse + chunked-prefill deadlock fix |
| 1.1.0 | 2026-05-04 | Stable | 896 PRs / 113 contributors. Resilient KV routing, Anthropic Messages API, Mocker offline replay, multimodal embedding cache |
| 1.0.x | 2026-03-16 (GA) | Maintenance | GA at GTC March 2026 |
| `1.2.0-deepseek-v4-dev.3` | 2026-05-09 | **Dev preview only** | vLLM 0.20.1, GB200 DSv4-Flash/Pro recipes, multi-stream pre-attention GEMM |
| `1.2.0-deepseek-v4-dev.2` | 2026-05-01 | Dev preview | First DSv4 native vLLM support |
| `1.2.0-sglang-deepseek-v4-dev.1` | 2026-04-25 | Dev preview | SGLang DSv4 recipe |

**Verdict:** Round 1's "1.1.1 / 1.2-dev" naming is accurate. There is no 1.2 GA. No 1.3 even on the dev branch.

---

## 2. Dynamo on GH200 aarch64 — multi-arch manifest verified

**Sources:** https://docs.nvidia.com/dynamo/dev/resources/support-matrix, https://docs.nvidia.com/dynamo/dev/resources/release-artifacts

### Architecture support (Dynamo 1.1.1)

| Container | amd64 | arm64 | Notes |
|---|---|---|---|
| vLLM Runtime (standard) | yes | **yes** | Multi-arch confirmed |
| vLLM Runtime (CUDA-13) | yes | **yes** | Multi-arch confirmed |
| vLLM Runtime (EFA) | yes | no | AWS-only; amd64 |
| SGLang Runtime (standard) | yes | **yes** | Multi-arch confirmed (corrects Round 1) |
| SGLang Runtime (CUDA-13) | yes | **yes** | Multi-arch confirmed (corrects Round 1) |
| TensorRT-LLM Runtime (standard) | yes | **yes** | Multi-arch confirmed |
| TensorRT-LLM Runtime (EFA) | yes | no | AWS-only; amd64 |
| vLLM-Omni (multimodal) | yes | no | Skipped on arm64 build; multimodal not expected to work on arm64 for v0.8.1 CUDA-13 dev tags |

### GPU architecture support

Listed by **architecture family**, not SKU: Blackwell, Hopper, Ada Lovelace, Ampere. **GH200 is Hopper-class (sm_90)** so it works in practice — but it is **not enumerated by SKU** in the matrix. The matrix also requires **Ubuntu 24.04 on arm64** (not 22.04).

### Net for GH200

All three runtimes (vLLM / SGLang / TRT-LLM) ship arm64 manifests on the standard tags in 1.1.1. **Round 1 was overly pessimistic about SGLang**. The remaining caveats — vLLM-Omni and other multimodal paths being amd64-only — stand.

---

## 3. GH200 NVL2 P/D reference architecture — does NOT exist

**Sources:**
- https://developer.nvidia.com/blog/simplify-system-memory-management-with-the-latest-nvidia-gh200-nvl2-enterprise-ra/
- https://github.com/ai-dynamo/dynamo/blob/main/recipes/README.md
- https://run-ai-docs.nvidia.com/self-hosted/tutorials/inference-tutorials/disaggregated-dynamo

### What the official GH200 NVL2 RA says

- Configuration: **2-2-3-400** = 2 CPUs / 2 GPUs / 3 NICs / 400 Gbps east-west per GPU
- Interconnects:
  - **CPU-CPU NVLink-C2C: 600 GB/s** (between the two Grace CPUs)
  - **GPU-GPU NVLink: 900 GB/s** (between the two Hopper GPUs)
  - **GPU-CPU NVLink-C2C: 900 GB/s** (within each superchip)
- Workloads framed: single-node LLM, RAG, recommenders, GNNs. **Prefill/decode disaggregation is NOT in the published RA.**
- Networking platform: Spectrum-X Ethernet for AI.

### What Dynamo's recipes target

| Recipe | Target hardware |
|---|---|
| `llama-3-70b/vllm/agg/...` | 4–8× H100/H200, single node |
| `llama-3-70b/vllm/disagg-single-node/...` | **8× H100/H200**, prefill+decode on one node |
| `llama-3-70b/vllm/disagg-multi-node/...` | **16× H100/H200**, 8 per node × 2 nodes |
| `deepseek-v4/...` | **GB200** (24-32× GB200) |
| `qwen3-*/...`, `gpt-oss-120b/...`, `glm-5-nvfp4/...` | GB200 / B200 |
| `nemotron-3-super-fp8/...` | H200 |

**No `gh200/` or `grace-hopper/` recipe directory exists.** The closest analog is `llama-3-70b/vllm/disagg-multi-node`, which is 16× H100/H200 — i.e., **two 8-GPU HGX boxes, not two GH200 superchips**.

### Practical GH200 NVL2 P/D mapping (derived, not published)

To run a P/D pair on a GH200 NVL2 chassis (2 GH200s, 1 prefill + 1 decode), the realistic adaptation is:

- **Prefill** = GH200 #0 (96-144 GB HBM3e + 480 GB LPDDR5X)
- **Decode** = GH200 #1
- **KV transport** = NIXL/UCX over GPU-GPU NVLink (900 GB/s) **if both workers run in the same process or use CUDA IPC**, else over the 400 Gbps Spectrum-X NIC fabric (50 GB/s).
- The 900 GB/s intra-chassis NVLink path is **the entire reason to choose NVL2 over two single-GH200 boxes** — but exploiting it requires same-process / CUDA-IPC config (see §4) and is **not the K8s-pod default**.

### Closest published precedent

`disagg-single-node` (8× H100/H200 in one HGX) and `disagg-multi-node` (2× HGX over IB) bracket the GH200 NVL2 case. A GH200 NVL2 P/D deployment would inherit the `disagg-single-node` style (NVLink between GPUs) with 2 GPUs instead of 8 and an arm64 image. **No NVIDIA-published example as of 2026-05-25.**

---

## 4. NIXL transport — what it actually uses

**Sources:**
- https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md
- https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/
- https://docs.vllm.ai/en/stable/features/nixl_connector_usage/
- https://docs.nvidia.com/dynamo/dev/kubernetes-deployment/deployment-guide/disagg-communication
- https://github.com/ai-dynamo/nixl/issues/1614
- https://github.com/ai-dynamo/dynamo/issues/493 (NVLINK disabled in default container, closed not-planned)
- https://github.com/openucx/ucx/discussions/9896 (UCX NVLink usage)

### Backends present in the repo (May 2026)

| Backend | What it transports | Same-node | Cross-node |
|---|---|---|---|
| **UCX** (default) | RoCE, InfiniBand, GPUDirect RDMA, TCP, **NVLink (via CUDA IPC inside the UCX `cuda_ipc` TL)**, shared memory | yes | yes |
| **GDS** (Magnum IO GPUDirect Storage) | VRAM ↔ persistent FS | n/a | n/a |
| **POSIX** | VRAM ↔ file (no GDS) | n/a | n/a |
| **Mooncake** | KV-store backed transfers | yes | yes |
| **Libfabric** | OFI providers (Slingshot/CXI, EFA, Neuron OFI as of 1.0.x/1.1.0) | yes | yes |
| **OBJ / S3** (dev) | Object stores (incl. Dell ObjectScale S3-over-RDMA, 1.1.0) | n/a | n/a |

### NVLink-C2C: explicit answer

The 900 GB/s **NVLink-C2C** in GH200 is the **on-package CPU↔GPU coherent link** (not a network). It is exploited by:
- CUDA driver / unified memory page migration (transparent)
- CUDA host pointer registration (`cudaHostRegister`)
- pinned-host KV cache that the GPU can stream from at C2C bandwidth

It is **not** a NIXL backend, and it is **not** what's between two GH200 superchips on an NVL2 board. The link between two GH200 GPUs on NVL2 is regular **GPU-GPU NVLink (900 GB/s)** — same number, different link.

### What NIXL uses between GH200 nodes

- **Default**: UCX over **InfiniBand / RoCE**. Dynamo's K8s disagg-communication guide states *"NVLink cannot be used between Kubernetes pods… requires same process and direct memory access within a single process namespace, which K8s pod isolation violates"* and recommends **IB RDMA (20-50 GB/s, ~1 µs)** or **RoCE (10-25 GB/s)**; **EFA libfabric on AWS at ~9.6 GB/s**; TCP fallback shows ~98 s TTFT vs 200-500 ms with RDMA (~200-500× degradation).
- **Same-node, same-process**: UCX can use **CUDA IPC over NVLink** if `UCX_TLS` includes `cuda_ipc` and the two endpoints share the CUDA context.
- **Cross-node CUDA IPC over NVLink (MNNVL)**: a new mode, enabled by `UCX_CUDA_IPC_ENABLE_MNNVL=y`. This is the only way to ride the NVL Switch fabric between physically separate GH200/GB200 servers in a rack-scale NVL36/NVL72 system. **Currently buggy**: NIXL issue #1614 reports `NIXL_ERR_BACKEND` reproducibly on GB200 cross-node with `UCX_TLS=cuda_copy,cuda_ipc,tcp` + MNNVL on vllm 0.19 + `nixl_cu13` (still open).
- **Default container ships with NVLINK disabled in UCX** (Dynamo issue #493, closed not-planned). You must override `UCX_TLS` explicitly to enable `cuda_ipc`.

### Verdict on the original question

> Does NIXL use the 900 GB/s NVLink-C2C between GH200 nodes via NVL fabric, or just IB?

**Neither, exactly.** NVLink-C2C is intra-superchip and not a NIXL transport. NIXL between GH200 *nodes* uses **IB / RoCE** by default. The NVL switch fabric (multi-node NVLink) is reachable via UCX `cuda_ipc` + MNNVL, but this is opt-in, broken on GB200 today (issue #1614), and disabled by default in the Dynamo container (issue #493).

### Round-1 statement to correct

Round 1 said "NIXL KV-cache transport: direct GPU↔GPU over NVLink, InfiniBand/UCX, or TCP fallback. Non-blocking." This implies NVLink is a primary transport. **Accurate qualifier**: NVLink is only used for same-process intra-node peers via UCX `cuda_ipc`; pod-isolated K8s deployments fall back to IB/RoCE.

---

## 5. llm-d — May 2026 status, GH200 fitness

**Sources:**
- https://github.com/llm-d/llm-d/releases
- https://llm-d.ai/blog/llm-d-v0.4-achieve-sota-inference-across-accelerators
- https://llm-d.ai/blog/llm-d-v0.3-expanded-hardware-faster-perf-and-igw-ga
- https://github.com/llm-d/llm-d/issues/281

### Release history

| Version | Date | Major changes |
|---|---|---|
| v0.3 | 2025-10 | Wider well-lit paths, expanded hardware, IGW GA |
| v0.4 | 2025-12 | 40% latency reduction on DeepSeek V3.1 on H200; Intel XPU + Google TPU disagg; CPU memory-tiered prefix cache offload; 50% P99 TTFT improvement |
| v0.5 | 2026-02-04 | Gateway API v1.4.0 + Istio 1.28.1; vLLM v0.14.1; Workload Variant Autoscaler → core |
| v0.6 | 2026-03-03 | vLLM v0.17.1; multi-cloud infra |
| **v0.7** | **2026-05-12** | **CUDA 13.0.2 (NVIDIA driver 580+ required); GB200 image variant; default deployment shifted to standalone mode with generic proxy** |

### GH200 / ARM fitness

- **GB200 has a dedicated image variant in v0.7** — that's Blackwell + Grace, not GH200 directly.
- **Issue #281 "Expand Hardware Expansion with ARM support"** (filed 2025-09-30 for GH200+GB200) is **still OPEN with no 2026 activity**. ARM64 has not been delivered.
- Multi-accelerator: XPU (Intel), HPU (Habana), ROCm (AMD), TPU (Google) variants exist. No arm64 variant.
- **Net**: llm-d is K8s-native and increasingly polished, but **not GH200-ready today**. Round 1's "not GH200 today; on roadmap" was correct; if anything, the roadmap status is *less* clear than Round 1 suggested — the issue has aged 8 months without a commit.

### Performance benchmarks (for context, not GH200)

- v0.3/v0.4 validated 3.1k tok/s/B200 decode on Llama 3 70B; 50k out-tok/s on 16×16 B200 P/D topology.
- v0.4 cut DeepSeek V3.1 H200 per-output-token latency by 40%, P99 TTFT by 50%.

---

## 6. Modular MAX 26.x — GH200 BF16 throughput

**Sources:**
- https://docs.modular.com/max/changelog/
- https://www.modular.com/blog
- https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience (Nov 2025)

### What's confirmed in the 26.x changelog

| Version | Date | GH200-relevant change |
|---|---|---|
| 26.1 | 2026-01-29 | `accelerator_count()` returns non-zero on Apple silicon; broader ARM GPU support (not GH200-specific) |
| 26.2 | 2026-03-19 | NVIDIA PTX compiler upgrade requires driver 580+; DGX Spark / Jetson Thor support; Blackwell sm_100 kernel optimizations (SnapMLA, TMA im2col); GH200 not called out |
| **26.3** | **2026-05-07** | **"Added support for the `bfloat16` data type on ARM CPU devices in MAX graphs"**; NVFP4 grouped matmul beats FlashInfer on B200 for Kimi K2.5; GH200/GB200 BF16-on-ARM-host context retained from 25.7 |

### Published throughput numbers — what's there and what isn't

**Available** (NOT GH200):
- MAX vs vLLM 0.8 on Sonnet: +12% (25.2 era, H100/H200)
- MAX 25.6→25.7 Qwen2.5-VL: +30-80% throughput; >2× vLLM on vision models
- MAX 26.3 NVFP4 GEMM: beats FlashInfer on B200 across Kimi K2.5 shapes

**NOT available**:
- **No published Modular GH200 BF16 throughput** for Llama-3-70B or any LLM
- No MAX benchmark blog post mentions Grace Hopper by name
- Modular's official benchmark page (`docs.modular.com/max/deploy/benchmark/`) targets H100/H200/B200/MI300X/MI325X/MI355X — **GH200 is absent from the marquee list**

### Container manifest status

- Round 1 flagged "arm64 image availability not explicitly documented" — **still true as of 2026-05-25**. The `modular/max-nvidia-full` and `max-nvidia-base` tags don't have a published arm64 manifest in Modular's container docs. Install via `uv pip install modular --index https://whl.modular.com/nightly/simple/` is the verified arm64 path; container deployment on GH200 needs `docker manifest inspect` verification before relying on it.
- Modular open-sourced MAX + Mojo (2025); Mojo 1.0 Beta shipped with 26.3 on 2026-05-07.

### Verdict

The BF16-on-ARM-host kernel work is real and aligns perfectly with this stack's BF16-only constraint. **But Modular has not published GH200 throughput numbers**, so any GH200 performance claim against Modular requires you to benchmark in-house. Treat MAX as **functionally validated for GH200 BF16** (per 25.7 + 26.3 release notes) but **performance-unverified** for this specific platform.

---

## Consolidated corrections to Round 1

| Round 1 claim | Round 3 verification |
|---|---|
| "Dynamo SGLang CUDA-13 arm64-only" | **Wrong for stable 1.1.x**. SGLang is multi-arch (amd64+arm64) on standard and CUDA-13 tags. The split is dev-branch-only. |
| "NIXL KV-cache transport: direct GPU↔GPU over NVLink, IB/UCX, or TCP fallback" | **Misleading**. NVLink is only used same-process via UCX `cuda_ipc`. Default K8s pod path is IB/RoCE. NVLink-C2C is not a NIXL transport. |
| "GH200 NVL2 reference architecture for P/D" | **Does not exist**. Official NVL2 RA is single-node multi-workload; Dynamo has zero GH200 recipes. |
| "llm-d… ARM/GH200 still requested (issue #281), not flagship" | **Confirmed and aged**. Issue #281 has had no 2026 activity. llm-d is on v0.7 with GB200 image variant but no arm64. |
| "Modular MAX 25.7 added BF16 on ARM host" | **Confirmed and extended**. 26.3 explicitly added bfloat16 on ARM CPU devices in MAX graphs. But no GH200 throughput numbers published. |
| "Dynamo support matrix lists Blackwell/Hopper/Ada/Ampere" | Confirmed. **GH200 still not enumerated by SKU**; relies on Hopper-architecture coverage. ARM64 requires Ubuntu 24.04. |

---

## High-confidence operational guidance for this GH200 stack (May 2026)

1. **Dynamo 1.1.1 on GH200 NVL2 is viable today** via vLLM or TRT-LLM backend, both with arm64 manifests. Force BF16. Expect to build your own P/D recipe (no published one). KV transport: NIXL over NVLink intra-chassis if you can keep prefill+decode in the same K8s pod / same process; else over the 400 Gbps Spectrum-X NIC fabric.
2. **Same-process P/D with NVLink** requires explicit `UCX_TLS=cuda_copy,cuda_ipc,rc,ud,tcp` and may require `UCX_CUDA_IPC_ENABLE_MNNVL=y` for cross-node NVL. **Test before committing** — issue #1614 shows this path is buggy on GB200; GH200 status unknown.
3. **llm-d is not an option** for GH200 today. Defer.
4. **Modular MAX 26.3 is the BF16-aligned choice** if you want a non-vLLM engine and can install via pip on arm64. Throughput on GH200 will need in-house measurement.
5. **Triton 26.04 SBSA** remains the safe single-node multi-model server (Round 1 verdict unchanged).
6. **NIM still skip** on GH200 for headline LLMs (Round 1 verdict unchanged).

---

## Sources (Round 3)

1. https://github.com/ai-dynamo/dynamo/releases — Dynamo release history
2. https://docs.nvidia.com/dynamo/dev/resources/support-matrix — GPU + arch support
3. https://docs.nvidia.com/dynamo/dev/resources/release-artifacts — per-container arch matrix
4. https://github.com/ai-dynamo/dynamo/tree/main/recipes — recipe inventory (no GH200)
5. https://github.com/ai-dynamo/dynamo/blob/main/recipes/README.md — recipe target hardware
6. https://run-ai-docs.nvidia.com/self-hosted/tutorials/inference-tutorials/disaggregated-dynamo — Run:ai Dynamo P/D tutorial (Llama-3.3-70B-FP8 on MNNVL)
7. https://developer.nvidia.com/blog/simplify-system-memory-management-with-the-latest-nvidia-gh200-nvl2-enterprise-ra/ — official GH200 NVL2 RA (2-2-3-400)
8. https://github.com/ai-dynamo/nixl — NIXL repo / backends
9. https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md — backend list
10. https://github.com/ai-dynamo/nixl/releases — NIXL 1.0/1.0.1/1.1.0 (May 2026)
11. https://github.com/ai-dynamo/nixl/issues/1614 — GB200 cross-node MNNVL NIXL_ERR_BACKEND (open)
12. https://github.com/ai-dynamo/dynamo/issues/493 — default container disables NVLINK on UCX (closed not-planned)
13. https://docs.nvidia.com/dynamo/dev/kubernetes-deployment/deployment-guide/disagg-communication — disagg transport guide ("NVLink cannot be used between K8s pods")
14. https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/ — NIXL technical blog
15. https://docs.vllm.ai/en/stable/features/nixl_connector_usage/ — vLLM NIXL connector
16. https://github.com/openucx/ucx/discussions/9896 — UCX `cuda_ipc`/NVLink
17. https://github.com/llm-d/llm-d/releases — llm-d v0.5/v0.6/v0.7 (2026)
18. https://llm-d.ai/blog/llm-d-v0.4-achieve-sota-inference-across-accelerators — v0.4 multi-accelerator
19. https://github.com/llm-d/llm-d/issues/281 — ARM64/GH200 request (still open)
20. https://docs.modular.com/max/changelog/ — MAX 26.1/26.2/26.3 changelogs
21. https://www.modular.com/blog — recent posts (no GH200 perf)

---

## Confidence

**High**:
- Dynamo 1.1.x is multi-arch on vLLM/SGLang/TRT-LLM standard tags (per release-artifacts page).
- GH200 NVL2 P/D reference architecture does not exist as a published artifact.
- NIXL default backend is UCX; NVLink path is opt-in, disabled in default container, broken cross-node on GB200 today.
- llm-d v0.7 released 2026-05-12; ARM64 issue #281 still open.
- Modular MAX 26.3 added BF16 on ARM CPU in graphs; no GH200 perf published.

**Medium**:
- Whether NIXL same-process NVLink path actually works on GH200 NVL2 (intra-chassis, two superchips) — the GB200 cross-node bug (#1614) is suggestive of broader MNNVL fragility, but GH200 NVL2 same-node has not been tested by NVIDIA in a published recipe.
- Whether MAX containers (`modular/max-nvidia-full`) have arm64 manifests by 2026-05 — Modular blog doesn't say; pip install is the verified path.

**Low / gaps**:
- Exact GH200 BF16 tok/s under Dynamo P/D vs monolithic. NVIDIA's published P/D ratios (7×, 30×) are FP8 Blackwell — not applicable. No third-party GH200-specific P/D benchmarks surfaced.
- Whether the `disagg-multi-node` recipe (16× H100/H200 across 2 nodes) can be ported to 2× GH200 NVL2 chassis (4 GPUs across 2 nodes) with minimal config changes, or whether tensor-parallel assumptions baked into the recipe break.
