# NVIDIA Blackwell as a GH200 Alternative for Inference — Round 1

**Agent 11 / 20**  •  **Date: 2026-05-25**  •  **Scope: B100, B200, B300/Ultra, GB200/GB300 NVL36/NVL72**

---

## TL;DR

- **B200 is the workhorse**: widely rentable on 20+ clouds in May 2026, $4.29–$14.24/GPU/hr on-demand, ~$2.65–$4.86 reserved. Hardware itself is sold out through mid-2026 (3.6M unit backlog) — the available *cloud capacity* is what was pre-bought by hyperscalers and CoreWeave-class providers.
- **GB200 NVL72** is available at CoreWeave ($10.50/GPU/hr), Oracle (~$16), Azure ND GB200 v6 ($108.16/hr per 4-GPU instance, ~$27/GPU/hr), and on waitlist at Nebius/Lambda. Lambda + Pegatron deployment is in flight but not GA as of March 2026.
- **B300 / Blackwell Ultra** shipped Jan 2026; available on CoreWeave, Lambda, FluidStack, Crusoe at $4.50–$5.80/GPU/hr on-demand, ~$3.40 reserved. **GB300 NVL72** delivers 45% higher DeepSeek-R1 throughput than GB200 NVL72 (MLPerf v5.1).
- **B100 was effectively cancelled** (KeyBanc analyst, design flaw on the original respin) — do not plan around it. Replaced by B200A for mid-range HGX form factors.
- **FP4/FP8 software status (vs GH200's FP8 problem)**:
  - **TRT-LLM 1.0** (PyTorch-stable, NVFP4 kernels) is production-grade for Blackwell.
  - **vLLM v0.20.x** has FlashAttention-4 default on SM100, MRV2 gives +56% throughput on GB200, FP4 dense + MoE supported.
  - **SGLang 0.5.6** hits 907 tok/s/GPU on DeepSeek-R1 NVFP4 (B200), 1.79x over 0.5.5.
  - **Real landmines** still exist: FA4 head_dim limits, FlashInfer MoE strict-inequality bug, vision-encoder SM100 crash in vLLM, ptxas C7907 hitting Triton autotuner. Not "broken stack" — but not zero-friction either.
- **Bottom line vs GH200**: For batch inference of Llama-3.3-70B or DeepSeek-V3/R1, **B200 delivers ~3-4× throughput at ~3× the price** = roughly **3× better $/token**. GB200 NVL72 wins decisively for large MoE (DeepSeek-V3 671B) — ~15× throughput vs H200 at ~$0.10 vs $1.56 per million tokens. GH200 remains the cheap single-GPU option at $1.99/hr (Lambda), but loses badly on throughput/$ for production scale.

---

## 1. Hardware Specs (May 2026)

| GPU | HBM | BW | FP4 Dense (TFLOPS) | FP8 (TFLOPS) | Notes |
|---|---|---|---|---|---|
| **GH200** (Grace Hopper) | 141 GB HBM3e | 4.0 TB/s | n/a (Hopper, no FP4 TC) | ~2,000 (sparse FP8) | Reference baseline |
| **H200** (Hopper) | 141 GB HBM3e | 4.8 TB/s | n/a | ~2,000 | Reference baseline |
| **B100** | **CANCELLED** | — | — | — | Replaced by B200A |
| **B200** SXM6 | **192 GB HBM3e** | **8.0 TB/s** | 9,000 dense / 18,000 sparse | ~4,500 dense | SM100, 5th-gen TC |
| **B300** / Blackwell Ultra | 288 GB HBM3e | ~8 TB/s | ~12,500 dense (NVFP4) | ~6,250 | Shipped Jan 2026 |
| **GB200 NVL72** (rack) | **13.4 TB HBM3e** | 130 TB/s NVLink5 | 720,000 dense / 1,440,000 sparse | 720,000 | 72× B200 + 36× Grace, ~120 kW |
| **GB300 NVL72** (rack) | ~20 TB HBM3e | 130 TB/s NVLink5 | ~1,000,000 dense | ~1,000,000 | 72× B300 + 36× Grace, ~120 kW |

Sources: [NVIDIA GB200 NVL72 page](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) (FP4 1,440/720 PFLOPS), [Spheron B200 specs](https://www.spheron.network/blog/nvidia-b200-complete-guide/), [Tweaktown B100 cancelled](https://www.tweaktown.com/news/100083/analyst-nvidia-has-effectively-canceled-b100-ai-gpu-over-design-flaw-b200a-to-replace-it/index.html), [Spheron B300 guide](https://www.spheron.network/blog/nvidia-b300-blackwell-ultra-guide/).

---

## 2. Cloud Availability Matrix (May 2026)

### B200 — Widely Available

| Provider | $/GPU/hr On-Demand | Reserved | Notes |
|---|---|---|---|
| Spheron (spot) | $2.12 | — | Cheapest, spot |
| Vultr | — | **$2.89 (36mo)** | Lowest reserved tracked |
| Packet.ai | $3.75 | — | Waitlist |
| Lyceum | $4.29 | — | |
| Cirrascale | — | $4.86 (12mo) | |
| Verda | $4.89 | — | |
| UpCloud | $5.22 | — | |
| Lambda Labs | $5.29–$6.99 | — | |
| Nebius | $5.50 | — | |
| Runpod | $5.89 | — | Available |
| CoreWeave | $8.60–$10.50 (NVL) | Contract | 8 of 9 instances in stock |
| Oracle Cloud | $14.00 (8x) → ~$17.85/GPU | Contract | Bare metal |
| AWS p6-b200.48xlarge | **$113.93/hr (8×B200)** = $14.24/GPU | $89.72/hr 1yr (21% off) | us-east-1, us-west-2, us-east-2, GovCloud-W (May 6, 2026 GA) |

Sources: [getdeploying B200](https://getdeploying.com/gpus/nvidia-b200), [Spheron pricing 2026](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/), [AWS GovCloud announcement May 6 2026](https://aws.amazon.com/about-aws/whats-new/2026/05/amazon-ec2-p6-b200-aws-govcloud/), [Vantage p6-b200](https://instances.vantage.sh/aws/ec2/p6-b200.48xlarge).

### GB200 NVL72 — Tight Supply, 9 Providers

| Provider | $/GPU/hr | Notes |
|---|---|---|
| **CoreWeave** | **$10.50** | 4-GPU compute instances; first-mover |
| Oracle Cloud | $16.00 | OCI bare metal |
| **Azure ND GB200 v6** | $27.04 (= $108.16/hr per 4-GPU VM) | GA, 19 regions including East US, West US 2/3, UK South, Germany W Central |
| Lyceum / Nebius / CUDO / Gcore / FluidStack / Crusoe | On request | Custom contracts; **Nebius on waitlist** |

Lambda Labs: GB200 NVL72 **not yet GA** as of March 2026 (Pegatron partnership deploying); Lambda *does* offer GB300 NVL72 clusters.

Sources: [getdeploying GB200](https://getdeploying.com/gpus/nvidia-gb200), [Azure ND GB200 v6 GA](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/accelerating-the-intelligence-age-with-azure-ai-infrastructure-and-the-ga-of-nd-/4394575), [Lambda + Pegatron DCD](https://www.datacenterdynamics.com/en/news/lambda-partners-with-pegatron-to-deploy-nvidia-gb200-nvl72-rack/).

### B300 / GB300 NVL72 — Newer, Available

- B300 single-GPU: $4.50–$5.80/hr on-demand, ~$3.40 reserved across CoreWeave, Lambda, FluidStack, Crusoe.
- B300 spot via Spheron: $2.45/hr (cheaper than H200 typical).
- GB300 NVL72 full rack: ~$3-4M capex; cloud at CoreWeave (first deploy July 2025, GA Aug 19 2025), Microsoft, Oracle.

Sources: [Spheron B300](https://www.spheron.network/blog/nvidia-b300-blackwell-ultra-guide/), [CoreWeave GB300 NVL72 blog](https://www.coreweave.com/blog/coreweave-leads-the-way-with-first-nvidia-gb300-nvl72-deployment).

### Supply Reality Check
- **B200 + GB200 silicon sold out through mid-2026** with ~3.6M unit backlog. CoWoS packaging + HBM are the bottleneck. ([FinancialContent / TokenRing Dec 2025](https://markets.financialcontent.com/stocks/article/tokenring-2025-12-29-nvidias-blackwell-dynasty-b200-and-gb200-sold-out-through-mid-2026-as-backlog-hits-36-million-units))
- Cloud capacity for B200 is reasonably available because hyperscalers + CoreWeave pre-bought; new direct purchase orders are 6+ months out.

---

## 3. Inference Performance (Llama-3.3-70B, DeepSeek-V3/R1)

### Llama-3.3-70B (1K/1K)
- **B200**: ~10,000 tok/sec/GPU @ 50 TPS/user (NVFP4, InferenceMAX v1) — **>4× per-GPU vs H200**.
- vLLM measured **3.7× throughput vs Hopper** at iso-latency on Llama-3.3-70B 1K/8K.
- B200 on Llama-3.3-70B FP4 serving: **~$0.17/M tokens** vs $0.50/M on H200.
- MLPerf-cited: B200 hits 11,264 tok/s on Llama-2-70B; full GB200 NVL72 server: ~60,000 tok/s per server.

### DeepSeek-V3 / R1 (671B MoE)
- **GB200 NVL72** with SGLang PD + WideEP: **26,156 tok/s/GPU prefill, 13,386 tok/s/GPU decode** with FP8 attention + NVFP4 MoE → 3.8× / 4.8× over H100.
- **GB200 NVL72** on DeepSeek-R1 (8K/1K, InferenceMAX v1): **10,000 TPS/GPU**, 15× vs H200, **$0.10/M tokens vs $1.56 H200**.
- **B200 single-node** SGLang 0.5.6 on DeepSeek-R1 NVFP4: 907 tok/s/GPU @ concurrency 4; 5,145 tok/s/GPU @ concurrency 128.
- **GB300 NVL72 (MLPerf v5.1)** on DeepSeek-R1: **5,842 tok/s/GPU offline, 2,907 tok/s/GPU server** — 45% higher than GB200 NVL72 offline; ~5× over Hopper.

### Llama-3.1-8B (MLPerf v5.1, GB300 NVL72)
- Offline: **18,370 tok/s/GPU**, Server: 16,099, Interactive: 15,284.

Sources: [vLLM InferenceMAX blog](https://vllm.ai/blog/blackwell-inferencemax), [NVIDIA Blackwell InferenceMAX](https://developer.nvidia.com/blog/nvidia-blackwell-leads-on-new-semianalysis-inferencemax-benchmarks/), [LMSYS GB200 Part II](https://www.lmsys.org/blog/2025-09-25-gb200-part-2/), [SemiAnalysis SGLang 0.5.6](https://inferencex.semianalysis.com/blog/sglang-0-5-6-b200-deepseek-r1-fp4-up-to-1-8x), [NVIDIA Blackwell Ultra MLPerf](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/), [Tom's Hardware GB300 +45%](https://www.tomshardware.com/pc-components/gpus/nvidia-claims-software-and-hardware-upgrades-allow-blackwell-ultra-gb300-to-dominate-mlperf-benchmarks-touts-45-percent-deepseek-r-1-inference-throughput-increase-over-gb200).

---

## 4. Price / Performance vs GH200

Reference: GH200 at **$1.99/hr (Lambda)** to $6.50/hr (CoreWeave single-GPU) — see [getdeploying GH200](https://getdeploying.com/gpus/nvidia-gh200).

| Scenario | GH200 | B200 | GB200 NVL72 |
|---|---|---|---|
| **$/hr per GPU (typical)** | $1.99–$6.50 | $5.29–$8.60 | $10.50–$27 |
| **Llama-70B tok/s/GPU** | ~1,500 (FP8, est. from H200 +est. memory benefit) | ~10,000 (NVFP4) | ~10,000+ (in MoE/large-batch contexts) |
| **$/M tokens (Llama-70B)** | ~$0.50–$1.20 | **~$0.17** | $0.10–$0.30 in serving |
| **DeepSeek-V3 671B viable?** | No (memory + interconnect) | Tight on 8x | **Yes, native** |

**Key takeaway**: GH200 remains attractive only when **(a)** you need a single-GPU dev box, **(b)** Lambda pricing applies, or **(c)** you specifically need Grace CPU large-memory KV-cache offload. For batch production inference at scale, **B200 is ~3× better $/token, GB200 NVL72 is ~10-15× better $/token for large MoE**.

---

## 5. Software Stack Readiness (FP4/FP8) — The Honest Assessment

User's prior pain: FP8 broken on GH200/Wan. **Is FP4/FP8 on Blackwell production-ready in May 2026?** Answer: **yes for datacenter SM100, with caveats**.

### What Works (Production Path)
- **TensorRT-LLM 1.0** — PyTorch architecture stable, LLM API stable, native NVFP4 kernels, ModelOpt PTQ workflow. Llama-4 @ 40K tok/s, gpt-oss @ 60K tok/s/GPU on B200. ([NVIDIA TRT-LLM releases](https://github.com/NVIDIA/TensorRT-LLM/releases))
- **vLLM v0.20.x** — FlashAttention-4 default on SM100/SM103, MRV2 +56% on GB200, FP4 dense + MoE via FlashInfer (`VLLM_USE_FLASHINFER_MOE_FP4=1`), FP8 KV-cache, FP8 GEMMs for attention projections. ([vLLM Blackwell InferenceMAX](https://vllm.ai/blog/blackwell-inferencemax))
- **SGLang 0.5.6** — DeepSeek-V3/R1 NVFP4 native, DeepGEMM enabled by default on Blackwell, piecewise CUDA graphs + JIT kernels. ([SGLang DeepSeek-V3.2 docs](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2))
- **NVFP4 accuracy**: ≤1% degradation on DeepSeek-R1-0528 vs FP8 (some benchmarks +2%); ~95-98% BF16 recovery on Llama 7B-14B; full 70B class typically <2% loss. ([NVIDIA NVFP4 QAD report](https://research.nvidia.com/labs/nemotron/files/NVFP4-QAD-Report.pdf))
- **Pre-quantized NVFP4 models**: Llama-3.3-70B, Llama-3.1-{8B,405B}, DeepSeek-V3.2/R1/R1-0528, Llama-4-Scout 17B-16E on HuggingFace from `nvidia/*-NVFP4`.

### Known Landmines (Real, Documented)
1. **FlashAttention-4 head_dim restrictions** on SM100: only `head_dim ∈ [8,128] div 8` plus DeepSeek's (192,128) asymmetric exception. Non-standard architectures may fall back. ([FA repo](https://github.com/Dao-AILab/flash-attention))
2. **vLLM vision encoder crash on SM100**: `_vllm_fa2_C.varlen_fwd` compiled SM80-only, hard crash for multimodal models during KV profiling. ([vLLM #38411](https://github.com/vllm-project/vllm/issues/38411))
3. **FlashInfer MoE strict-inequality bug** on SM100 for specific DeepSeek-V3 routing configs (`<` should be `<=`). Patchable but ships in some versions.
4. **Triton autotuner / ptxas C7907 internal compiler error** eliminates `num_warps=4/8` configs for some backward kernels on SM100a → 38.7× slowdown observed in Mamba3 backward. ([state-spaces/mamba #904](https://github.com/state-spaces/mamba/issues/904))
5. **CUDA version pinning fragility** — vLLM stable PyPI vs Blackwell CUDA 12.8: `ImportError: libcudart.so.12` is the canonical failure. Use vLLM's own nightly index. (Allen Kuo's Apr 2026 Medium post.)
6. **WSL2 + Blackwell CUDA-graph reboots** (Kernel-Power Event 41) until WSL2 ≥ 2.7.0 with dxgkrnl fix. Workaround `--enforce-eager` costs 40% decode.
7. **SM120 (consumer 50-series Blackwell, DGX Spark)** is *not* supported by FA4/FlashMLA. Doesn't affect B200/B300 datacenter parts but matters if you confuse the two.
8. **AWQ 4-bit can be slower than BF16** on Blackwell (B200's 8 TB/s makes dequant overhead exceed bandwidth savings on 9B-class models). Use NVFP4, not AWQ.

### Driver / CUDA Story
- **R580** is the current Long-Term Support branch (EOL Aug 2028), CUDA 13+ support, the production default for B200/B300.
- R570 (production) EOLs **June 2026** — migrate now.
- R575 already EOL — migrate urgently.
- **CUDA ≥ 12.8 minimum** for Blackwell; vLLM binaries compiled with 12.8, 12.9, 13.0. cuDNN 9.7+ (Apr 2026) adds FP8 conv on Blackwell + fused GQA attention graph patterns.

Sources: [NVIDIA datacenter driver matrix](https://docs.nvidia.com/datacenter/tesla/drivers/supported-drivers-and-cuda-toolkit-versions.html), [vLLM GPU install](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/), [vLLM landmines Medium](https://allenkuo.medium.com/vllm-or-ollama-on-blackwell-benchmarks-landmines-and-what-agents-actually-need-5dc539bb28ef).

---

## 6. NVLink5 Fabric & Rack-Scale Inference

GB200 NVL72 binds 72 B200 GPUs through 5th-gen NVLink at **130 TB/s aggregate all-to-all bandwidth**, with 13.4 TB unified HBM3e visible as one device. This is the *only* configuration where DeepSeek-V3 671B fits comfortably with WideEP (Expert Parallelism at scale) for low-latency serving. vLLM's WideEP + LMSYS's PD-disaggregation paper show **3.8× prefill / 4.8× decode** over H100 baselines on the full model.

GB300 NVL72 extends the same fabric with higher per-GPU compute (Blackwell Ultra) and more HBM — observed 45% higher throughput on DeepSeek-R1 in MLPerf v5.1.

If your model is **>200B params** or you want serving latency targets that 8× H200 can't hit, **NVL72 is the answer, full stop**. For everything ≤ 70B dense, 8× B200 SXM is cheaper.

---

## 7. Recommendation Synthesis

**Replace GH200 with:**
- **B200 (8× SXM node)** for Llama-3.3-70B and similar dense models: ~3× $/token improvement, available on 20+ clouds, mature TRT-LLM + vLLM + SGLang.
- **GB200 NVL72** for DeepSeek-V3/R1 or 200B+ MoE production: ~10-15× $/token improvement, available CoreWeave/Oracle/Azure today.
- **GB300 NVL72** if you can wait/get allocation: another 25-45% throughput uplift on reasoning workloads.
- **Skip B100** — cancelled.

**Do not assume**: FP4 is universally a win. Validate NVFP4 vs FP8 on *your* model — quality holds for Llama/DeepSeek/Qwen ≥ 7B per published QAD work, but RL-tuned models and small models can degrade. FP8 is still the safe default in 2026 if you don't have time to validate.

**Stack starting points**:
- TRT-LLM 1.0 + ModelOpt → if you want NVIDIA's blessed path and willing to do engine builds.
- vLLM v0.20.x + FlashInfer FP4 MoE → if you want OSS, fast iteration, broad model support.
- SGLang 0.5.6+ → if your model is DeepSeek family or you need WideEP at NVL72 scale.

---

## Confidence & Gaps

**High confidence**:
- B200/GB200/B300/GB300 specs and feature support (multiple NVIDIA + cloud + benchmark sources cross-confirm).
- Cloud pricing ranges (cross-referenced getdeploying, Spheron, Vantage, official cloud provider pages).
- MLPerf v5.1 numbers (NVIDIA + Tom's Hardware + MLCommons all consistent).
- vLLM / TRT-LLM / SGLang FP4 support status and the specific landmines (each landmine has a real GitHub issue or Medium post backing it).
- B100 cancelled, supply-constrained mid-2026.

**Medium confidence**:
- Exact GH200 per-token costs (less recent benchmarking — GH200 was deprecated focus by mid-2025, so 2026 numbers are sparse). Used H200 as baseline proxy where needed.
- Spot pricing (volatile, snapshot-only). Some quoted spot prices may not be available at request time.
- GB300 cloud pricing — newer, less stable price discovery.

**Gaps / Unknowns**:
- **Llama-3.3-70B specific tok/s on GH200** — couldn't find a direct MLPerf-style number. Inferred ~1,500 tok/s/GPU FP8 from H200 references. Other agents may have better data.
- **DGX B300 / B300 SXM detailed FP4 TFLOPS** — NVIDIA hasn't published the same level of detail as B200; estimated ~12.5 PFLOPS dense NVFP4 from press claims.
- **Lambda GB200 NVL72 GA date** — Pegatron deployment in flight Mar 2026, no confirmed cloud GA as of May 2026.
- **Real AWS Capacity Block prices for p6-b200 in May 2026** — pricing scheduled to update again July 2026; current numbers may shift.
- **Production telemetry on the FlashInfer MoE bug** — issue exists but fix landing status across vLLM/SGLang/TRT-LLM versions not fully tracked.
- **FP4 stability under sustained 24/7 production load** — NVIDIA's quoted accuracy numbers are from PTQ benchmarks, not multi-month deployments. The "FP8 broken on GH200/Wan" experience suggests caution; need long-running burn-in before betting a product on NVFP4 for a *specific* model.

**Recommended follow-ups** for other agents in the cohort:
- Direct GH200 vs B200 Llama-3.3-70B head-to-head benchmark at fixed batch/concurrency.
- Real-world FP4 quality regression report on the specific target model (e.g., the user's Wan workload if relevant).
- AWS Capacity Block vs CoreWeave on-demand long-term TCO for steady 8× B200 inference.
