# SGLang on GH200 — Round 3 Verification + Deepening

**Agent**: #04 / Round 3
**Date**: 2026-05-25
**Scope**: Verify Round-1 claims; deepen on HiCache-L2-over-C2C, Mooncake-on-GH200, GH200 user reports, RadixAttention hit-rate measurements.
**Constraint carried forward**: FP8 broken on this stack — bf16 default.

---

## 0. What changed since Round 1

| Round-1 claim | Round-3 verification | Delta |
|---|---|---|
| "SGLang v0.5.12.post1 is current" | **Confirmed**. v0.5.12 released **2026-05-16**, v0.5.11 released 2026-05-05, v0.5.10/.post1 released 2026-04-06/09. | OK. |
| "`sgl-kernel >= 0.3.21` ships official aarch64 wheels" | **Half-stale.** `sgl-kernel` was **renamed to `sglang-kernel` in v0.5.10** (kernel consolidation). Current is `sglang-kernel 0.4.2.post2`. `sgl-kernel 0.3.21` (Jan 14 2026, with aarch64 wheel) is now the last release on the **archived** `sgl-kernel` PyPI project — that page literally reads "This project has been archived. No new releases are expected." Install guides that pin `sgl-kernel` will break. | **Update**: pin `sglang-kernel>=0.4.1` (April 3 2026, aarch64 wheel confirmed) or use the Docker image. |
| "`lmsysorg/sglang:latest-cu130` arm64 manifest is live" | **Confirmed live as of 2026-05-25.** `latest`, `latest-cu130`, `latest-cu129` all point to `v0.5.12.post1` and carry **multi-arch (linux/amd64 + linux/arm64) manifests**. Nightly tags `nightly-dev-cu13-20260525-*` are live today. arm64 image sizes: `latest-cu130` 11.92 GB, `latest-cu129` 15.66 GB. | OK. Note the older era of explicit `-arm64` suffixed tags (e.g., `v0.5.4.post2-cu129-arm64`) is gone — current tags are multi-arch. |
| "RadixAttention 50–99% hit rate" | **Confirmed.** Multiple 2026 sources reaffirm: 75–95% on 60%+ prefix-overlap workloads; 50–99% range across benchmarks. No new GH200-specific hit-rate measurement found. | OK. |
| "CUDA 13 / Torch 2.9.1 is default" | **Outdated.** v0.5.11 release notes call out **"CUDA 13 + Torch 2.11" as default**. Update install instructions accordingly. | **Update**: torch == 2.11 for CUDA 13 path. |
| "Default piecewise CUDA graph" | **New in v0.5.10**: "Piecewise CUDA Graph Enabled by Default" — reduces memory overhead, improves throughput on models with complex control flow. Affects launch tuning. | New flag default. |
| "No public SGLang-on-GH200 throughput numbers from LMSYS" | **Still true** as of 2026-05-25. The 2026-02-20 GB300 NVL72 blog uses **H200 as the baseline** (not GH200), and includes a calibration caveat: "in the absence of latency constraint H200 can achieve similar throughput" to GB300. | Gap persists. |
| "FP8 broken on stack" | **Reinforced.** v0.5.12 added "HiSparse FP8 KV cache via flashmla_kv backend" and "Fused SiLU+clamp+FP8 quant kernel", but DeepSeek-V4 roadmap (#23602) still lists Hopper FP8 issues; tooling moved to W4A16 / W4A4 / NVFP4 for newest models. **Stay bf16** on GH200 absent your own validation. | OK. |
| "Multi-GH200 NVL32 + SGLang supported but unbenchmarked" | **Confirmed.** DeepSeek V4 roadmap lists Hopper (SM90) and Blackwell (SM100) as targets; GB200 / GB300 NVL72 are the documented multi-node targets. **GH200 NVL32 + SGLang remains undocumented in benchmarks.** A separate bug (#23743) tracks "DeepSeek V4 Flash GB200 serving fixes" — Grace-Blackwell, not Grace-Hopper. | Gap persists. |

**Two genuinely new things since Round 1**:
1. **GPU Staging Buffer for PD Disaggregation** (shipped in v0.5.10): gathers scattered head slices into contiguous memory for bulk RDMA, **~1000× fewer RDMA requests on GQA models**, 2–5× throughput at high concurrency with heterogeneous TP. Round-1 had only generic env-var advice; this is now the canonical Mooncake path.
2. **Elastic EP for Partial Failure Tolerance** (v0.5.10): NIXL-EP integration redistributes expert weights when a GPU fails — no full restart. Mostly relevant to multi-node DeepSeek MoE; not GH200-specific but useful in NVL32-class deployments.

---

## 1. SGLang stable release verification (drill #1)

**Verified via**:
- `https://github.com/sgl-project/sglang/releases` (full tag history May 2026)
- v0.5.11 release notes (PyPI/GitHub)
- NVIDIA SGLang Release Notes PDF (RN-08516-001_v26.04, April 2026)

**Result**:

| Version | Date | Notes |
|---|---|---|
| **v0.5.12.post1** | 2026-05-16 (post1 within days) | **Current stable.** DeepSeek V4, HiCache+UnifiedRadixTree+SWA, HiCache+DeepSeek-V4 SSD offload, W4A4 MegaMoE, Marlin/FlashInfer W4A8 MoE on Hopper, NVFP4 hot-reload, TP16 on H100/H20, Mooncake "incremental transfer + SSD offload", aarch64 cubin handling improvements, ARM CPU phase-1A CI bootstrap |
| v0.5.11 | 2026-05-05 | NVFP4 KV cache, ModelOpt diffusion FP8, DeepSeek-R1-0528-w4a8 + DeepEP LL FP8 dispatch, CUDA 13 + Torch 2.11 default, `sgl-kernel` → 0.4.1.post1 → 0.4.2 |
| v0.5.10.post1 | 2026-04-09 | bugfix |
| v0.5.10 | 2026-04-06 | Elastic EP, GPU Staging Buffer for PD, **package rename `sgl-kernel` → `sglang-kernel` 0.4.1**, Native MLX backend for Apple Silicon, piecewise CUDA graph default, HiSparse, HybridCacheController, Mooncake 0.3.9 → 0.3.10.post1 |
| v0.5.10rc0 | 2026-03-28 | pre-release |

**Recommendation**: Pin `sglang==0.5.12.post1` and `sglang-kernel>=0.4.2` (or rely on the Docker image and forget the wheels).

---

## 2. Docker arm64 manifest verification (drill #2)

**Verified via** Docker Hub tag listing (`hub.docker.com/r/lmsysorg/sglang/tags`, fetched 2026-05-25).

**Result**: **Live, multi-arch.**

| Tag | Points to | linux/amd64 | linux/arm64 | Notes |
|---|---|---|---|---|
| `latest` / `latest-cu130` | `v0.5.12.post1-cu130` | ✅ | ✅ (11.92 GB) | Default CUDA 13 |
| `latest-cu129` | `v0.5.12.post1-cu129` | ✅ | ✅ (15.66 GB) | CUDA 12.9 fallback |
| `nightly-dev-cu13-20260525-ed179bf9` | HEAD | ✅ | ✅ | Today's nightly |
| `-cu130-runtime` / `-cu129-runtime` | smaller runtime images | ✅ | ✅ | Strip dev tools |
| `-rocm720-mi35x`, `-rocm700-mi30x` | AMD | N/A | N/A | ROCm variants (no aarch64) |

**Practical note**: The older era of explicit `-arm64` suffixed tags (e.g., `v0.5.4.post2-cu129-arm64`, `gemma4-mtp-arm64-cu13`) still exists historically. Going forward (v0.5.10+), the official path is the **multi-arch latest/cu130/cu129 manifests** — Docker pulls the arm64 layer automatically on GH200.

---

## 3. HiCache L2 over NVLink-C2C (drill #3)

**This is the biggest gap I tried to close and couldn't fully.**

### What LMSYS / Mooncake publish
- LMSYS 2025-09-10 HiCache blog: "up to 6× throughput, up to 80% reduction in TTFT", "up to 2× higher throughput in typical deployments" from layout opt + zero-copy CPU↔GPU. **No concrete bandwidth numbers. No GH200, no NVLink-C2C, no aarch64.**
- Mooncake × SGLang HiCache design doc (`kvcache-ai.github.io/Mooncake/design/hicache-design.html`): defines L1=GPU, L2=host, L3=Mooncake/SSD; transports = RDMA + zero-copy; GPU-assisted I/O kernels claim "up to 3× higher transfer speed" — **no GH200, no NVLink-C2C mention.**
- HiCache benchmark page (`...performance/sglang-hicache-benchmark-results-v1.html`): hardware is **A10 cluster (3×2 A10 + 100 GbE eRDMA) and 8×H800 (mlx5 RDMA)**. No GH200. Graphs only, no tabular numbers extractable. Models: Qwen3-14B (A10), Qwen3-235B-A22B-Instruct-2507 (H800).
- v0.5.12 adds **"HiCache framework support for UnifiedRadixTree (with SWA)"** and **"HiCache for DeepSeek V4"** with SSD offload — but again the recipe targets H100/H200/B200, not GH200.

### What we *can* infer (from third-party Grace-Hopper memory papers)

| Metric | Value | Source |
|---|---|---|
| NVLink-C2C theoretical (each direction) | 450 GB/s | NVIDIA spec |
| Host→Device practical (STREAM) | **~375 GB/s** (~83% of peak) | `verda.com/blog/data-movement-in-the-nvidia-gh200-grace-hopper-superchip` (Jan 2026) |
| Device→Host practical (STREAM) | **~297 GB/s** (~66% of peak) | same |
| Hopper accessing local HBM (peak %) | 93% read, 84% write | arXiv 2408.11556v2 |
| Grace accessing peer HBM (peak %) | 53% read, 64% write | arXiv 2408.11556v2 |
| Cross-C2C latency (ping-pong) | 150–400 ns | arXiv 2408.11556v2 |
| DDR↔DDR bidirectional copy via C2C | ~250 GB/s | arXiv 2408.11556v2 |
| DDR→HBM unidirectional copy via C2C | ~450 GB/s (near peak) | arXiv 2408.11556v2 |

**Net inferential conclusion for SGLang HiCache L2 on GH200**:
The KV-eviction path is almost entirely **device→host** writes (decode evicts KV to LPDDR5x), which is the *slower* direction at ~297 GB/s practical. The cache-restore path (host→device) is faster at ~375 GB/s. By comparison, a PCIe Gen5 x16 H100 offload tops out at ~63 GB/s for the same operation. So **HiCache L2 on GH200 is theoretically 4.7–6× the PCIe-attached H100 baseline**, but this is an architectural inference, not a measured SGLang number.

**Round-3 action item that wasn't possible from the web**: run the SGLang HiCache benchmark suite on a GH200 in-house. The published runs (A10 + H800) literally do not exist on Grace-Hopper as of 2026-05-25.

---

## 4. Mooncake transport on GH200 (drill #4)

**Verified via**:
- `docs.sglang.io/advanced_features/pd_disaggregation.html`
- `kvcache-ai.github.io/Mooncake/` index
- v0.5.10 / v0.5.11 / v0.5.12 release notes

**Result**: **No GH200-specific Mooncake deployment guide exists**. Mooncake docs cover:
- Generic RDMA over ConnectX/MLX5 NICs
- AMD ROCm variant (8.0/12.0 tutorials on `rocm.docs.amd.com`)
- NVIDIA Dynamo integration (`docs.nvidia.com/dynamo/latest/backends/sglang/sglang-disaggregation.html`)
- GB200 NVL72 (the Blackwell side)

**Two new since Round 1**:
1. **GPU Staging Buffer for PD Disaggregation** (v0.5.10): gather scattered KV head slices into contiguous memory for bulk RDMA, then scatter on decode side. **~1000× fewer RDMA requests on GQA models**, 2–5× throughput at high concurrency, matches homogeneous-TP within ~5%. Enable when prefill TP ≠ decode TP with Mooncake.
2. **RDMA-based P2P weight transfer using Mooncake TransferEngine** (April 2026): **7× faster weight updates for 1T-param Kimi-K2** (53 s → 7.2 s). For RL workflows, not inference itself.

**GH200 angle (inferred, not documented)**:
- GH200 typically ships with ConnectX-7 or BlueField-3 NICs (200–400 Gbit/s).
- Mooncake will use GPUDirect RDMA automatically; aarch64 RDMA stack is mature.
- The *interesting* feature for GH200 is that Mooncake's **intra-node transport** can use NVLink-C2C for prefill↔decode on the same node, but I find no doc that calls this out as a GH200-tuned path. SGLang's intra-node Mooncake transport is generic NVLink/NVL-switch + IntraNode + TCP fallback.

---

## 5. SGLang-on-GH200 user reports (drill #5)

**Searched**: GitHub issues/discussions, Reddit, Twitter, blogs, NVIDIA developer forums.

**Found**:

| Source | Date | Concrete numbers | Verdict |
|---|---|---|---|
| Baseten/Lambda blog | 2025-02-07 | GH200 96GB vs H100 80GB on Llama-3.3-70B FP8 ShareGPT batch=32: **GH200 +32% throughput**. Engine: **TRT-LLM** (SGLang used only as bench harness). | Already in Round 1. Not a SGLang-runtime number. |
| GitHub Discussion #13303 / Issue #13304 | 2025-11 | 4× GH200 server, sgl-kernel 0.1.0 no aarch64 wheel, source build OOMs the node. | Historical failure; resolved by sglang-kernel 0.4.x + Docker. |
| veRL Issue #1208 (volcengine/verl) | 2025-04-22 | veRL+SGLang on GH200 aarch64 cluster: **significantly slower than veRL+vLLM**, despite less memory use. TP=2, gpu_memory_utilization=0.6, 1024×1024. "standalone SGLang works as well as veRL+vLLM" — integration bug, not SGLang itself. Specific tok/s **not** disclosed. | veRL-side integration issue, not core SGLang. Worth knowing if you run RL training on GH200. |
| llama.cpp Discussion #18005 | 2026 | Not SGLang, but useful GH200 calibration: GPT-OSS-20B **322.9 tok/s**, GPT-OSS-120B **209.0 tok/s**, DeepSeek V3.1 (671B) **18.3 tok/s** (with `cudaMemAdvise` keeping experts in LPDDR), Kimi K2 Thinking (1026B) **21.6 tok/s**. Single-stream decode. | Best-available "real numbers on GH200" data point, but llama.cpp not SGLang. |
| LMSYS LLM blogs (HiCache, GB300 NVL72, DeepSeek V4) | various 2026 | **None measure GH200.** Baselines are H200, not GH200. | Gap. |
| Spheron, Particula, Inference.net, Techsy 2026 H100/H200 benchmarks | various 2026 | Repeat the H100/H200 numbers from Round 1. No GH200 entries. | No new info. |
| NVIDIA SGLang Release Notes (NGC), April 2026 (RN-08516-001_v26.04) | 2026-04 | Lists supported hardware. **Hopper SM90 supported; no GH200-specific section.** | Implies GH200 is "Hopper" generically. |

**Verdict**: As of 2026-05-25 there is **still no first-party SGLang-on-GH200 throughput publication**. The closest concrete GH200 numbers are from llama.cpp, a fundamentally different runtime.

---

## 6. RadixAttention hit-rate measurements (drill #6)

**Verified via**: LMSYS 2024-01-17 blog (foundational), Spheron 2026 production guide, Inference.net 2026 guide, Particula 2026 comparison, runpod.io 2026 production guide.

**Result**: The published hit-rate ranges are stable across 2026 sources — **no separate GH200 measurement**, and the H100 numbers are unchanged from Round 1.

| Workload | Hit rate | Source |
|---|---|---|
| LLaVA-Next-34B (multimodal) | 52.4% | LMSYS blog |
| Vicuna-33B (chat) | 74.1% (→ 1.7× lower mean TTFT) | LMSYS blog |
| Few-shot prompting | 85–95% | Multiple 2026 guides |
| Multi-turn chat | 75–90% | Multiple 2026 guides |
| Code analysis / RAG | 60–80% | Multiple 2026 guides |
| 60%+ prefix-overlap workloads | **75–95%** | Spheron 2026 |
| Unique-prompt workloads | 0% (no benefit) | Spheron 2026 |
| Throughput edge on prefix-heavy | **10–20% over vLLM** at H100 (29% raw aggregate) | Particula/Techsy 2026 |

**Why GH200 hit rates would not differ from H100 in principle**: RadixAttention is a software data structure over KV pages. Hit rate is a function of traffic shape, not memory bandwidth. **Hit rate ports 1:1 from H100 to GH200.** What *does* change on GH200 is the *value* of each hit:
- H100 SXM HBM3 ≈ 3,350 GB/s — saved decode work hits HBM bandwidth.
- GH200 HBM3e ≈ 4,800 GB/s (96 GB SKU) or 4,900 GB/s (144 GB SKU) — **~1.43× the bandwidth** of H100. Each cache hit is worth ~1.43× more in decode tok/s, until you become compute-bound.
- HiCache L2 cost on GH200 (~300–375 GB/s) is **5–6× cheaper** than PCIe-attached H100 host offload (~63 GB/s). So extending hit rate via L2 has a much better cost/benefit on GH200 than on H100. (Inferred; unmeasured.)

---

## 7. Updated install recipe (replaces Round 1 §2.2)

If you must install on bare metal aarch64 in May 2026 (NGC PyTorch container, slurm node):

```bash
# CUDA 13 default for v0.5.11+; torch 2.11
pip install --upgrade pip uv

# Kernel is now `sglang-kernel` (not `sgl-kernel` — that PyPI project is archived)
uv pip install sglang-kernel \
  --index-url https://docs.sglang.io/whl/cu130

uv pip install sglang==0.5.12.post1
```

Wheels resolve to `sglang_kernel-0.4.2.post2-cp310-abi3-manylinux2014_aarch64.whl` (or `0.4.1` if you pin earlier).

**Equivalent Docker (recommended on GH200)**:
```bash
docker pull lmsysorg/sglang:latest-cu130   # multi-arch, picks arm64 on GH200
```

---

## 8. Updated confidence & gap matrix

### High confidence (verified Round 3)
- `lmsysorg/sglang:latest-cu130` multi-arch manifest live (arm64 layer 11.92 GB).
- v0.5.12.post1 is the current stable as of 2026-05-25.
- `sgl-kernel` PyPI project **archived**; current kernel is `sglang-kernel 0.4.2.post2`.
- Default toolchain is CUDA 13 + Torch 2.11 (v0.5.11+).
- Piecewise CUDA graph default-on (v0.5.10+).
- NVLink-C2C practical bandwidth: ~375 GB/s H2D, ~297 GB/s D2H (third-party measured).
- RadixAttention hit-rate ranges unchanged from Round 1; ports to GH200 unchanged in principle.
- No SGLang-on-GH200 throughput publication exists from LMSYS / NVIDIA / sgl-project as of 2026-05-25.

### Medium confidence
- Multi-arch Docker pulls arm64 layer without explicit selection on GH200 (standard Docker behavior; confirmed by manifest inspection).
- HiCache L2 cost on GH200 is ~5–6× cheaper than PCIe H100 host offload (architectural inference).
- GH200 hit rate per request = H100 hit rate per request (traffic-shape dependent only).

### Gaps (still open, two rounds in)
- **No published SGLang HiCache L2 throughput measurement on GH200.** Best inference: 300–375 GB/s effective L2.
- **No GB200 / GH200 distinction in DeepSeek V4 fixes** — bug #23743 explicitly targets GB200 (Blackwell), not GH200 (Hopper). Whether the same TileLang ARM64 fixes help GH200 is unclear.
- **No SGLang-on-NVL32** benchmark.
- veRL+SGLang+GH200 has a documented integration slowdown (Issue #1208). Cause unknown.
- v0.5.12's "ARM CPU phase-1A CI bootstrap" suggests proper aarch64 CI is *just now* coming online — confidence in arm64 build correctness will rise across the next few releases.

### Open bugs that affect GH200 deployment (May 2026)
- **#23743** — DeepSeek V4 Flash GB200 serving fixes (Grace-Blackwell; not directly GH200, but the TileLang ARM64 fixes share lineage).
- **#7905** — Slow Model Loading on GB200 (~30 min for DeepSeek-R1-FP4, ~50% timeout rate). Watch for GH200 parallel.
- **#23177** — GLM 5.1 queue piling up (general Hopper).
- **Round-1 FP8 watch list still active** (#24650, #18550, #23687).

---

## 9. Sources (new in Round 3, beyond Round 1)

Release verification:
- [SGLang releases page (live tag listing)](https://github.com/sgl-project/sglang/releases) — v0.5.12 (2026-05-16), v0.5.11 (2026-05-05), v0.5.10 (2026-04-06)
- [v0.5.12 release notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.12)
- [v0.5.11 release notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.11) — CUDA 13 + Torch 2.11 default
- [v0.5.10 release notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.10) — `sgl-kernel` → `sglang-kernel`, piecewise CUDA graph default, Elastic EP, GPU Staging Buffer
- [NVIDIA SGLang Release Notes PDF, April 2026 (RN-08516-001_v26.04)](https://docs.nvidia.com/deeplearning/frameworks/pdf/SGLang-Release-Notes.pdf)
- [sgl-kernel on PyPI (archived 0.3.21)](https://pypi.org/project/sgl-kernel/)
- [SGLang Kernel Wheel Index repo (sgl-project/whl)](https://github.com/sgl-project/whl)

GH200 user reports / bugs:
- [Discussion #13303 — GH200 install discussion](https://github.com/sgl-project/sglang/discussions/13303)
- [Issue #13304 — sgl-kernel aarch64 install failure on GH200](https://github.com/sgl-project/sglang/issues/13304)
- [veRL Issue #1208 — veRL-SGLang slower than expected on GH200](https://github.com/volcengine/verl/issues/1208)
- [Issue #23743 — DeepSeek V4 Flash GB200 serving fixes](https://github.com/sgl-project/sglang/issues/23743)
- [Issue #7905 — Slow Model Loading on GB200](https://github.com/sgl-project/sglang/issues/7905)
- [Issue #18458 — Tracking SGLang supported architectures (model archs, not hardware)](https://github.com/sgl-project/sglang/issues/18458)
- [Issue #23602 — DeepSeek V4 Roadmap](https://github.com/sgl-project/sglang/issues/23602)
- [Issue #17130 — SGLang × NVIDIA Collaboration Roadmap 2026 Q1](https://github.com/sgl-project/sglang/issues/17130)

NVLink-C2C measured bandwidth (closest proxy for HiCache L2):
- [Verda blog — MEMCPY analysis on GH200 (Jan 2026)](https://verda.com/blog/data-movement-in-the-nvidia-gh200-grace-hopper-superchip) — 375 GB/s H2D, 297 GB/s D2H STREAM
- [arXiv 2408.11556v2 — Data Movement in Tightly Coupled Heterogeneous Systems (GH200 case study)](https://arxiv.org/html/2408.11556v2) — 93/84% local-HBM utilization, 60% cross-C2C cap, 150–400 ns latency
- [HPI — Towards Memory Disaggregation via NVLink-C2C (paywalled / 403'd)](https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/rabl/publications/2025/hcds25-Werner-GraceHopper.pdf) — referenced but inaccessible
- [llama.cpp Discussion #18005 — GH200 perf with `cudaMemAdvise` for MoE experts](https://github.com/ggml-org/llama.cpp/discussions/18005) — concrete GH200 tok/s for OSS/DeepSeek/Kimi (non-SGLang baseline)

HiCache + Mooncake (verified no GH200 numbers):
- [LMSYS HiCache blog 2025-09-10](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)
- [Mooncake × SGLang HiCache design](https://kvcache-ai.github.io/Mooncake/design/hicache-design.html)
- [Mooncake HiCache benchmark v1 (A10 + H800, no GH200)](https://kvcache-ai.github.io/Mooncake/performance/sglang-hicache-benchmark-results-v1.html)
- [LMSYS GB300 NVL72 InferenceXv2 blog, 2026-02-20](https://www.lmsys.org/blog/2026-02-20-gb300-inferencex/) — uses H200 (not GH200) as baseline; notes "in the absence of latency constraint H200 can achieve similar throughput"

RadixAttention hit-rate verification (2026 production guides):
- [Spheron — SGLang Production Deployment Guide 2026](https://www.spheron.network/blog/sglang-production-deployment-guide/)
- [Particula — SGLang vs vLLM 2026](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison)
- [Inference.net — Complete SGLang guide](https://inference.net/content/sglang-complete-guide/)
- [Runpod — SGLang in production](https://www.runpod.io/articles/guides/blog-sglang-production-llm-pipelines)
- [Techsy — vLLM vs SGLang H100 2026](https://techsy.io/en/blog/vllm-vs-sglang)
- [LearnOpenCV — Serving SGLang production server](https://learnopencv.com/sglang-a-production-server/)

Mooncake / disaggregation (verified, no GH200):
- [SGLang PD Disaggregation docs](https://docs.sglang.ai/advanced_features/pd_disaggregation.html)
- [Mooncake project index](https://kvcache-ai.github.io/Mooncake/)
- [NVIDIA Dynamo — SGLang Disaggregated Serving](https://docs.nvidia.com/dynamo/latest/backends/sglang/sglang-disaggregation.html)
- [ROCm — SGLang distributed inference with Mooncake](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/sglang-distributed.html)
