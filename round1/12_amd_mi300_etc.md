# Agent 12 — AMD Instinct MI300X / MI325X / MI350X / MI355X as GH200 Inference Alternatives

Date: 2026-05-25
Author: Research agent #12 of 20
Scope: AMD CDNA3 (MI300X / MI325X) and CDNA4 (MI350X / MI355X) accelerators for LLM inference vs Hopper GH200. Software stack (ROCm 7.x, vLLM-ROCm, SGLang, AITER, Quark, hipBLASLt), real production gaps, cloud pricing, $/Mtok comparisons.

---

## 1. Hardware lineup (May 2026)

| GPU | Arch | HBM | BW | TDP | Native dtypes | Status |
|---|---|---|---|---|---|---|
| MI300X | CDNA3 (gfx942) | 192 GB HBM3 | 5.3 TB/s | 750 W | BF16/FP16, FP8 (E4M3/E5M2), INT8 | Shipping since late 2023; broadly available |
| MI325X | CDNA3 (gfx942) | 256 GB HBM3e | 6.0 TB/s | 1000 W | Same as MI300X | Shipping; ~1.3x compute, 1.3x BW over MI300X |
| MI350X | CDNA4 | 288 GB HBM3e | 8.0 TB/s | 1000 W (air) | + native FP6/FP4 (MXFP4/MXFP6) | Shipping Q1 2026 |
| MI355X | CDNA4 | 288 GB HBM3e | 8.0 TB/s | 1400 W (DLC) | Same as MI350X | Shipping Q1 2026; ~20% perf over MI350X |

Key CDNA4 improvement: per-CU FP8 throughput doubled (4096 → 8192 FLOPS/clock) and native FP4/FP6 added (CDNA3 had no FP4). FP6 and FP4 share a datapath, so FP6 ≈ FP4 throughput. MI355X peak ≈ 2.5 PFLOPS FP8 (dense), ~5 PFLOPS FP4. Source: ISSCC 2026 disclosure ([Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/inside-the-instinct-mi355x), 2026-02).

**Memory vs GH200**: GH200 carries 96 GB or 144 GB HBM3e (single-package Grace Hopper). MI300X = 1.33–2x memory; MI355X = 2–3x memory. This is AMD's structural advantage.

---

## 2. Software stack reality

### 2.1 vLLM-ROCm (the primary serving path)

- vLLM has first-class ROCm wheels for ROCm 7.0/7.2.1; nightly Docker images target MI300X/MI325X/MI350X/MI355X (`rocm/vllm-dev`).
- Attention backends (per [vLLM blog 2026-02-27](https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html)):
  - **ROCM_AITER_FA / ROCM_AITER_MLA** — production-recommended. AITER (AI Tensor Engine for ROCm) ships hand-tuned assembly kernels for decode. Delivers 2.7–4.4x throughput over legacy ROCM_ATTN.
  - **TRITON_ATTN / TRITON_MLA** — fallback for non-supported shapes and Radeon GPUs.
  - **ROCM_ATTN** — legacy HIP paged attention; only supports limited KV head sizes (Qwen3-235B falls back to Triton).
- vLLM AMD CI pass rate: 37% (Nov 2025) → 93% (Jan 2026). Dedicated AMD CI pipeline shipped Dec 29, 2025. Source: [ROCm Blog vLLM-Omni](https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html).
- Disaggregated prefill/decode: experimental in vLLM upstream; AMD prototype landed in 2025-Q3, but **not yet on parity with NVIDIA Dynamo** at rack scale. MI355X cited as "single-node only" (4–8 GPU) in production today — [SemiAnalysis InferenceX Kimi K2.5 article](https://inferencex.semianalysis.com/blog/mi355x-kimi-k2-5-vllm-aiter-7x-speedup) (March 2026).

### 2.2 SGLang-ROCm
- Official ROCm support, but per SemiAnalysis Q2 2025: "AMD SGLang CI coverage less than 10% parity with NVIDIA"; "25% of tested models fail accuracy tests on AMD." Status has improved by May 2026 but no public claim of full parity. [SemiAnalysis benchmark](https://newsletter.semianalysis.com/p/amd-vs-nvidia-inference-benchmark-who-wins-performance-cost-per-million-tokens).
- DeepSeek R1 FP4 SGLang reference path exists on ROCm 7.0 RC1, but FP4 GEMM kernels still "have room for improvement" (SemiAnalysis InferenceMAX, 2026).

### 2.3 TensorRT-LLM equivalent
- AMD has **no direct TRT-LLM equivalent**. The closest analog stack is AITER + hipBLASLt + Composable Kernel + Quark (quantization toolkit). It's modular, not a single tightly-optimized serving runtime.
- The practical implication: H200/B200 with TRT-LLM consistently beats MI325X/MI355X with vLLM on $/Mtok at high interactivity, even when AMD beats NVIDIA-with-vLLM. AMD has no answer to TRT-LLM's kernel polish.

### 2.4 FP8 — the open wound (critical for the comparison vs GH200)

The GH200 FP8 path on this team's stack is broken. **AMD's FP8 has different but real production gaps**, not parity:

- **MI300X FP8 slower than BF16 on MoE models**: [vLLM issue #31475](https://github.com/vllm-project/vllm/issues/31475), still OPEN as of investigation. GLM-4.7: FP8 ~30 TPS vs BF16 ~60 TPS (2x slowdown). MiniMax-M2.1: FP8 ~45 TPS vs BF16 ~58 TPS. Root cause: scale handling / kernel fusion gaps / dispatch overhead dominate any FP8 advantage.
- **torch._scaled_mm catastrophically slow**: [PyTorch issue #143465](https://github.com/pytorch/pytorch/issues/143465) — MI300X reaches ~100 TFLOP/s vs ~1200 TFLOP/s on H100 (12x gap on the same primitive). Practical FP8 GEMM on MI300X peaks ~990 TFLOP/s through optimized paths, still 22% behind H100.
- **FP4 misconfig crashes on MI300X**: [vLLM issue #34641](https://github.com/vllm-project/vllm/issues/34641) — default `VLLM_ROCM_USE_AITER_FP4BMM=True` crashes on gfx942 (MI300X has no FP4 hardware). Filed Jan 2026, recently fixed. Symptom of MI300X / MI355X feature-flag confusion.
- **MI355X FP4 kernels immature**: SemiAnalysis InferenceMAX: "B200 significantly outperforms MI355X across all three workload types [on Llama 70B FP4] — AMD's FP4 kernels have room for improvement." MI355X hits ~67% of B200 throughput on Kimi K2.5 (2,687 vs 4,021 tok/s/GPU).

Net assessment: **FP8 works on MI325X and on MI355X for dense Llama-class models, but it is NOT a drop-in win**. Production teams report needing weeks of tuning, and for MoE workloads BF16 remains the safer choice on MI300X. FP4 on MI355X is the bleeding edge — works for select shapes (gpt-oss-120b MXFP4, DeepSeek-R1 MXFP4 via Quark) but is not yet equivalent to NVIDIA's FP4 maturity on B200.

---

## 3. Performance: tokens/sec and MLPerf

### MLPerf Inference v5.1 (Sep 2025) — [AMD blog](https://www.amd.com/en/blogs/2025/accelerating-generative-ai-how-instinct-gpus-delivered.html)
- Llama 2 70B Server (FP8): MI355X delivered **2.7x tok/s vs MI325X**.
- 4-node MI355X vs 4-node MI325X: **3.4x faster**.
- First AMD MLPerf submission with MXFP4 (MI355X).
- MangoBoost third-party MI355X: 648k tok/s aggregate on Llama 2 70B.

### MLPerf Inference v6.0 (early 2026) — [AMD blog](https://www.amd.com/en/blogs/2026/amd-delivers-breakthrough-mlperf-inference-6-0-results.html)
- First AMD submission >1M tok/s multi-node (Llama 2 70B, 11 nodes).
- gpt-oss-120b, 12 nodes / 94 GPUs: >1M tok/s with >90% scaling efficiency.

### Real workload throughput (MI300X, 8x platform)
- DeepSeek-R1 with expert parallelism: **21,000 output tok/s** ([Moreh 2025-11-13](https://moreh.io/technical-report/21k-output-tokens-per-second-deepseek-inference-on-amd-instinct-mi300x-gpus-with-expert-parallelism-251113/)).
- Llama 3.1 405B FP8 on 8x MI300X: deployable (fits 810 GB on 1.5 TB HBM) — Oracle OCI production path documented.

### AMD's own claims (verify before relying)
- MI355X up to 1.3x B200 on Llama 3.1 405B (vLLM); 1.2x B200 on DeepSeek-R1 (SGLang). Caveat: this is at AMD-selected batch sizes.

### Independent (SemiAnalysis InferenceMAX, continuous 2026)
- MI325X-vLLM beats H200-vLLM on $/Mtok across nearly all interactivity levels on Llama 3.3 70B FP8.
- H200-**TRT-LLM** beats MI325X-vLLM on $/Mtok at high interactivity. AMD has no TRT-LLM-class runtime.
- MI355X-vLLM beats B200-vLLM on $/Mtok for gpt-oss-120b FP4 below ~225 tok/s/user; B200-TRT-LLM wins above that.
- MI300X beats H100 on $/perf-TCO on GPT-OSS 120B MX4 across the interactivity range.
- For DeepSeek-V3 670B FP8: H200 wins low-latency; MI325X competitive only in 25–35s latency window.

---

## 4. Cloud availability and pricing (May 2026)

Sources: [getdeploying.com/gpus/amd-mi300x](https://getdeploying.com/gpus/amd-mi300x), [/amd-mi325x](https://getdeploying.com/gpus/amd-mi325x), [/amd-mi355x](https://getdeploying.com/gpus/amd-mi355x), Hot Aisle, Crusoe, Vultr.

### MI300X (mature market — 9+ providers)
| Provider | Config | $/GPU/hr | Terms |
|---|---|---|---|
| DigitalOcean | 1x | $1.99 | on-demand |
| Hot Aisle | 1/2/4x VM | $1.99 | on-demand, per-minute |
| Hot Aisle | 8x bare metal | $3.39 | 1-month min |
| Crusoe | 8x | $3.45 (on-demand), $0.95 (spot) | |
| Vultr | 8x | $1.85 | 24-mo reserved |
| Microsoft Azure | 8x | $7.86 (OD), $1.45 (spot) | |
| Oracle Cloud | 8x | $6.00 | on-demand |
| TensorWave | 8x | quote | custom |

### MI325X (3 providers)
| Provider | $/GPU/hr | Terms |
|---|---|---|
| Vultr | $2.00 | 36-mo reserved |
| DigitalOcean | $2.10 | 12-mo reserved |
| TensorWave | $2.25 | custom |

### MI355X (5 providers, mostly reserved/quote-only)
| Provider | $/GPU/hr | Terms |
|---|---|---|
| Vultr | $2.29 | 36-mo reserved |
| Oracle Cloud | $8.60 | on-demand |
| TensorWave | quote | custom |
| Crusoe | quote | on-demand |
| Hot Aisle | quote | waitlist |

Spread: $2.29–$8.60 (avg $5.45). Spot/on-demand short-term MI355X is essentially not available below ~$5/hr.

### Reference: NVIDIA comparables
- **H200**: $1.45 (spot) – $13.78 (Azure OD); typical Lambda/Runpod $3–4; OD median ~$4.07; 31 providers.
- **GH200**: 3 providers only (CoreWeave $6.50, Vultr, Lambda); floor ~$1.99 spot. Limited supply.

### The "AMD rental tax"
SemiAnalysis flagged: "only a handful of providers offer short-term AMD GPU rentals" vs "100+ Neocloud providers for NVIDIA." Result: AMD on-demand rates are artificially elevated relative to hardware capability. Reserved pricing (Vultr 24–36 mo) is where AMD's TCO advantage materializes; on-demand it largely evaporates.

---

## 5. $/Mtok comparison — the bottom line

Using SemiAnalysis InferenceMAX framing + current pricing:

| Workload | Best AMD | Best NVIDIA | Winner |
|---|---|---|---|
| Llama 3.3 70B FP8, low-interactivity reasoning | MI300X / MI325X vLLM | H200 vLLM | AMD on $/Mtok (memory bandwidth wins) |
| Llama 3.3 70B FP8, high-interactivity chat | MI325X vLLM | H200 TRT-LLM | NVIDIA (no AMD TRT-LLM equivalent) |
| DeepSeek-V3/R1 670B FP8 | MI325X vLLM/SGLang | H200 / GH200 TRT-LLM | NVIDIA at most latencies; AMD wins memory fit on fewer GPUs |
| gpt-oss-120b MXFP4 summarization, <225 tok/s/user | MI355X vLLM | B200 vLLM | AMD |
| gpt-oss-120b MXFP4, >225 tok/s/user | MI355X vLLM | B200 TRT-LLM | NVIDIA |
| Llama 70B FP4 | MI355X vLLM | B200 TRT-LLM | NVIDIA decisively |
| Very-large-context single-GPU serving | MI300X / MI325X / MI355X | H200 / GH200 | AMD (memory) |

Reference point from SemiAnalysis: H200 at ~75 tok/s/user interactivity is ~$1.56/Mtok; GB200 NVL72 ~$0.10/Mtok (a real gap AMD can't close at rack scale yet because MI400 UALoE72 hasn't shipped).

---

## 6. Where AMD wins decisively (vs GH200 specifically)

1. **Memory headroom**: GH200 = 96/144 GB; MI300X = 192 GB; MI355X = 288 GB. Llama 405B FP8 fits on 8× MI300X (1.5 TB) with comfortable KV-cache room; on GH200 you fight for KV cache. Long-context, multi-tenant serving favors AMD.
2. **Single-tenant TCO at reserved pricing**: Vultr MI300X 24-mo @ $1.85/GPU/hr is structurally cheaper than GH200 on CoreWeave $6.50/hr. Even with software overhead, $/Mtok wins for memory-bound workloads.
3. **Single-node fit reduces inter-GPU traffic**: a 70B model that needs TP=4/8 on H100 can run TP=2 on MI300X — fewer NCCL/RCCL all-reduces, lower latency for small batches in the right setup.
4. **No NVLink-style lock-in at the rack**: Infinity Fabric is intra-node-only on CDNA3/4 (rack-scale UALink is MI400-class, late 2026); but for single-node 8-GPU inference, this is irrelevant.
5. **Public roadmap is credible**: MI400 (UALoE72) and MI500 (UAL256) are explicitly targeted at the GB200/GB300 NVL72 rack-scale class.

## 7. Where AMD loses

1. **FP8 path is uneven** — MI300X FP8 slower than BF16 on MoE; torch._scaled_mm bug still open; MoE production safe path is BF16, costing memory and bandwidth advantage.
2. **No TRT-LLM equivalent** — AITER + hipBLASLt is modular and improving, but every high-interactivity benchmark goes to H200/B200-TRT-LLM.
3. **FP4 immaturity** — MI355X FP4 kernels lag B200 substantially on Llama 70B; Kimi K2.5 best case is ~67% of B200.
4. **Cloud-availability tax** — short-term on-demand AMD prices are inflated by thin provider count.
5. **Disaggregated prefill/decode** at rack scale is months behind NVIDIA Dynamo.
6. **SGLang accuracy/CI gaps** — 25% of tested models failed accuracy on AMD per SemiAnalysis 2025; improved since but no public parity claim.
7. **Customer-engineering burden** — Reports of "6 months tuning to hit 95% of H100 throughput." Real cost to small teams.

---

## 8. Recommendation for the GH200 alternative question

For inference workloads where the team's GH200 FP8 path is broken and BF16 fits:

- **If memory-bound and reserved-pricing-acceptable**: MI300X (Vultr 24-mo $1.85/hr) or MI325X ($2.00/hr) is a defensible swap. Use vLLM-ROCm + AITER, run BF16 first, then attempt FP8 selectively per model.
- **If you can wait + want bleeding-edge perf**: MI355X for FP4-tolerant models (gpt-oss, DeepSeek-R1 MXFP4 via Quark), but accept that B200/GB200 will beat it on $/Mtok at high interactivity.
- **Do NOT assume AMD FP8 = drop-in fix for broken NVIDIA FP8**. The bug shapes differ but both exist. On MoE specifically, plan for BF16.
- **For multi-node / disaggregated serving at scale**: AMD is not yet competitive with Dynamo + GB200. Wait for MI400 UALoE72.

---

## Sources & dates

- [vLLM ROCm Attention Backend Blog](https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html), 2026-02-27
- [SemiAnalysis: AMD vs NVIDIA Inference Benchmark](https://newsletter.semianalysis.com/p/amd-vs-nvidia-inference-benchmark-who-wins-performance-cost-per-million-tokens), 2025-Q2 (still primary reference)
- [SemiAnalysis InferenceMAX Open-Source Benchmarking](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference), 2026 ongoing
- [SemiAnalysis InferenceX — MI355X Kimi K2.5 vLLM AITER 7x Speedup](https://inferencex.semianalysis.com/blog/mi355x-kimi-k2-5-vllm-aiter-7x-speedup), 2026-03
- [AMD MLPerf Inference v5.1 blog](https://www.amd.com/en/blogs/2025/accelerating-generative-ai-how-instinct-gpus-delivered.html), 2025-09
- [AMD MLPerf Inference 6.0 blog](https://www.amd.com/en/blogs/2026/amd-delivers-breakthrough-mlperf-inference-6-0-results.html), 2026
- [Tom's Hardware ISSCC 2026 MI355X deep dive](https://www.tomshardware.com/tech-industry/semiconductors/inside-the-instinct-mi355x), 2026-02
- [vLLM issue #31475 — MI300X FP8 slower than BF16](https://github.com/vllm-project/vllm/issues/31475), open
- [vLLM issue #34641 — FP4 default crashes MI300X](https://github.com/vllm-project/vllm/issues/34641), Jan 2026
- [PyTorch issue #143465 — MI300X torch._scaled_mm 12x slower than H100](https://github.com/pytorch/pytorch/issues/143465)
- [Moreh: 21k tok/s DeepSeek on 8x MI300X](https://moreh.io/technical-report/21k-output-tokens-per-second-deepseek-inference-on-amd-instinct-mi300x-gpus-with-expert-parallelism-251113/), 2025-11
- [ROCm vLLM-Omni blog](https://rocm.blogs.amd.com/software-tools-optimization/vllm-omni/README.html)
- [AITER on GitHub](https://github.com/ROCm/aiter)
- [getdeploying MI300X cloud pricing](https://getdeploying.com/gpus/amd-mi300x), May 2026 snapshot
- [getdeploying MI325X cloud pricing](https://getdeploying.com/gpus/amd-mi325x), May 2026
- [getdeploying MI355X cloud pricing](https://getdeploying.com/gpus/amd-mi355x), May 2026
- [Hot Aisle pricing](https://hotaisle.xyz/pricing/)
- [Crusoe MI355X](https://www.crusoe.ai/cloud/gpus/amd-mi355x)
- [AMD MI350X datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi350x-platform-brochure.pdf)
- [Spheron MI350X vs B200](https://www.spheron.network/blog/amd-mi350x-vs-nvidia-b200/)

---

## Confidence & Gaps

**High confidence**:
- Hardware specs (multiple corroborating AMD datasheets + ISSCC disclosure).
- FP8 production gaps on MI300X for MoE (open GitHub issues, reproducible).
- Cloud pricing snapshots May 2026 (multiple aggregator sources agree).
- AITER as production attention backend on MI300X/MI325X/MI355X.
- AMD wins memory capacity vs GH200 (just arithmetic).

**Medium confidence**:
- $/Mtok cross-points (SemiAnalysis InferenceMAX numbers shift with each nightly run; the qualitative directions are stable but precise crossover interactivity levels move).
- MI355X vs B200 FP4 gap (reported as ~33% Kimi K2.5; varies by model/shape; AMD claims parity-or-better on Llama 405B, B200 wins on Llama 70B FP4).
- SGLang AMD CI pass-rate today — most recent public number is Q2 2025 (10% parity); improved since (vLLM CI went 37→93%) but no current public SGLang figure.

**Low confidence / gaps**:
- TensorWave, Hot Aisle MI355X exact $/hr (quote-only; I have no firm number).
- Real disaggregated-prefill production performance on MI355X — articles cite single-node-only, but I lack a multi-node MI355X disaggregated benchmark to quantify the gap vs Dynamo/GB200.
- MI400 timing and whether late-2026 MI400 UALoE72 numbers will be available before GH200 contracts need to renew.
- Whether the MI300X FP8 MoE issue has been fixed in a ROCm/vLLM nightly newer than the cited GitHub issue snapshot. Status still listed as "in progress" — I did not find a closing PR.
- DigitalOcean MI325X actual provisioned capacity and waitlist time.
