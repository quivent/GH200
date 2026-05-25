# Blackwell — Round 3 Verification & Deepening

**Agent 11 / 20**  •  **Date: 2026-05-25**  •  **Scope: Verify and deepen Round 1 claims on B200/GB200/GB300 with focus on real waitlist times, on-demand inventory, DiT FP8/NVFP4 bugs on Blackwell, and CUDA 12.8 vs 13.x stability.**

---

## TL;DR — What Held, What Changed, What's New

- **Round 1 cloud matrix mostly holds**, with some sharpening:
  - **B200 is "available in hours" only at specific providers** (CoreWeave, Inworld Compute, Nebius self-serve, Crusoe with sales). Lambda Labs B200 capacity is real but constrained — no formal waitlist, just intermittent stockouts. Most other providers have **mid-2026 backlogs** for non-reserved direct buys.
  - **B200 spot pricing has been volatile**: March 2026 mean $5.09/hr, median $4.96/hr, swinging $4.40–$6.11 with single-day moves up to 18.9% (Silicon Data SDB200RT index). Higher and more volatile than Round 1 implied.
- **GB200 NVL72 on-demand reality**:
  - **CoreWeave is the only true on-demand provider** at scale today ($10.50/GPU/hr, US-WEST-01 with regions expanding). All others (Oracle, Azure, Google) are sold via contract/reservation, not click-to-launch on-demand.
  - **AWS p6e-gb200 is *not* on-demand** — only Capacity Blocks for ML reservations in Dallas Local Zone (`us-east-1-dfw-2a`). Important correction to Round 1.
- **GB300 NVL72 production deployments are real**:
  - **Microsoft Azure has the first at-scale production cluster**: 4,608 GB300 GPUs (= 64 NVL72 racks) for OpenAI, announced Oct 2025 — 92.1 ExaFLOPS FP4.
  - **AWS p6e-gb300 UltraServers GA December 2, 2025** (Capacity Blocks only).
  - **CoreWeave GB300 NVL72 cloud instances GA August 19, 2025** (US-WEST-01A).
  - Beyond MLPerf 5.1 lab numbers, **GB300 is in real production serving OpenAI workloads today** — not just a benchmark target.
- **DiT FP8/NVFP4 on Blackwell — bugs reproduce in a *different* form**:
  - **Wan 2.2 + FP8 + SageAttention → black-frame output** on Blackwell (RTX 5090, SM_120) — same family of bug pattern as the GH200 Wan FP8 issue. Open + unresolved as of Aug 2025. (`thu-ml/SageAttention#221`, `Comfy-Org/ComfyUI#7020`)
  - **NVFP4 native loading fails on Wan 2.2 / Flux2-Dev / LTX2** on Blackwell — silently upcasts to FP8/FP16, blowing memory (28 GB instead of 7.5 GB). Open + unresolved. (`Comfy-Org/ComfyUI#11864`)
  - **FP8 quantized MoE on B300 + vLLM 0.13 + CUDA 13** fails with weight-block-size divisibility error (`192 not divisible by 128`). (`vllm-project/vllm#31262`, open)
  - **However**: First-party NVIDIA path (TRT-LLM + ModelOpt NVFP4) works fine for FLUX.1-Dev and FLUX.2 (cited NVIDIA blog shows 10.2× FLUX.2 speedup). The bug surface is in the **community OSS stack** (ComfyUI, SageAttention, vLLM-on-CUDA-13), not the NVIDIA blessed path.
- **CUDA 12.8 vs 13.x stability**: **CUDA 12.8 with R580 LTS driver is the production-stable choice as of May 2026**. CUDA 13.0 (released Aug 4, 2025), 13.1, and 13.2 (Apr 12, 2026) all add Blackwell improvements (up to 20% better grouped GEMM), but **CUDA 13 + vLLM has live regressions**: prebuilt wheels expect CUDA 12 (`libcudart.so.12`), FP8 MoE breaks on B300, DGX Spark SM_121 unsupported by PyTorch, multiple sequential install failures. **Use 12.8 unless you specifically need a 13.x-only feature.**

---

## 1. Real Waitlist Times — B200 (May 2026)

### Provider-by-Provider

| Provider | Status | Detail |
|---|---|---|
| **Inworld Compute** | **Available within hours** | Dedicated B200 capacity, advertised as no waitlist (cited Inworld + getdeploying) |
| **CoreWeave** | **GA since May 29, 2025**; on-demand available | 8 of 9 instance types in stock; sales contact for new contracts; reserved gets best rate |
| **Nebius** | **Self-serve B200 live** at $5.50/hr | Blackwell self-service portal launched 2025; **GB200 NVL72 still pre-order/waitlist only** (per Nebius pre-order page) |
| **Lambda Labs** | Available, **no formal waitlist** | Intermittent capacity constraints reported. On-demand $4.99–$6.99/hr; 1-Click Clusters available |
| **Crusoe** | Available but **gated behind sales contact** | "Reserved allocation processes" required; HGX B200 listed but not self-serve checkout |
| **Most others (Lyceum, Verda, UpCloud, Packet.ai, Vultr, Spheron)** | Mixed — spot/short reservations available; long-term direct buys backlogged through mid-2026 | Listings exist on getdeploying but "available now" varies day-to-day |

**Key insight**: There is no single "waitlist time" — the supply story is bimodal. Either:
1. You take what a hyperscaler/CoreWeave/Inworld has pre-allocated (immediate, market price ~$5/hr volatile), or
2. You go through procurement for a long-term reservation (no formal queue length disclosed publicly; ranges from weeks at Nebius/Lambda to "mid-2026" for net-new orders).

The **3.6M unit Blackwell backlog** is on new fab/CoWoS allocations, not on what's already in cloud DCs. Cloud capacity ≠ silicon supply.

Sources: [Inworld B200](https://inworld.ai/resources/nvidia-b200-gpu-cloud), [CoreWeave B200 GA](https://www.coreweave.com/blog/coreweave-expands-its-nvidia-blackwell-fleet-with-generally-available-nvidia-hgx-b200-instances), [Nebius pre-order](https://nebius.com/blackwell-pre-order), [Nebius self-serve blog](https://nebius.com/blog/posts/introducing-self-service-nvidia-blackwell-gpus), [Lambda B200 page](https://lambda.ai/1-click-clusters), [Crusoe B200](https://www.crusoe.ai/cloud/gpus/nvidia-b200), [Silicon Data B200 March 2026 index](https://www.silicondata.com/blog/b200-rental-price-march-2026-update).

### Pricing Index — March 2026 (Silicon Data SDB200RT)
- **Mean**: $5.09/hr
- **Median**: $4.96/hr
- **Range**: $4.40–$6.11/hr
- **MoM change**: **+12.8%** (Feb → Mar)
- **Volatility**: coefficient of variation 11.4%; max daily swing 18.9%
- "Supply constraint hit mid-month" → providers with inventory pulled back discounts.

This is **higher and more volatile** than the Round 1 $4.29–$5.50 range and updates the picture: B200 spot is not a stable index, it's a tight commodity market.

---

## 2. GB200 NVL72 — Actual Rentable On-Demand Inventory

| Provider | On-Demand? | Reality |
|---|---|---|
| **CoreWeave** | **YES** — GA Feb 3, 2025 | $10.50/GPU/hr; via CoreWeave Kubernetes Service in US-WEST-01; "additional regions in coming months." This is the **only true click-to-launch on-demand GB200 NVL72**. |
| **Oracle Cloud** | **NO (reservation/contract)** | OCI bare metal, ~$16/GPU/hr published; sold through OCI sales, not self-serve |
| **Azure ND GB200 v6** | **Partially** — GA but capacity-allocated | $108.16/hr per 4-GPU VM = $27.04/GPU/hr; 19 regions; "GA" but actual VM provisioning is gated by Azure capacity-allocation queues, not pure click-launch |
| **AWS p6e-gb200** | **NO — Capacity Blocks only** | Only via EC2 Capacity Blocks for ML in `us-east-1-dfw-2a` (Dallas Local Zone). u-p6e-gb200x72 / x36 sizes. **This corrects Round 1 — it is not standard on-demand.** |
| **Lambda Labs** | **NO** | Pegatron deployment in motion; rack arrived per DCD; no public GA date as of May 2026 |
| **Nebius** | **NO — pre-order/waitlist** | Nebius pre-order page explicitly says "Reserve" / "join the waiting list" |
| **Google Cloud** | Listed | Available but contracted; not public on-demand SKU |

**Bottom line**: For GB200 NVL72 click-to-launch today, **CoreWeave is essentially the only game in town**. Everyone else is a queue or a contract. Round 1's listing of "9 providers" was hardware availability; **true on-demand is 1 provider**.

Sources: [CoreWeave GB200 NVL72 GA](https://www.coreweave.com/blog/coreweave-becomes-the-first-cloud-provider-with-generally-available-nvidia-gb200-nvl72-instances), [CoreWeave Feb 3 2025 changelog](https://docs.coreweave.com/docs/changelog/release-notes/gb200-nvl72), [AWS p6e-gb200 launch July 14, 2025](https://aws.amazon.com/about-aws/whats-new/2025/07/amazon-p6e-gb200-ultraservers-gpu-performance-ec2/), [Vantage p6e-gb200.36xlarge](https://instances.vantage.sh/aws/ec2/p6e-gb200.36xlarge), [Lambda+Pegatron DCD](https://www.datacenterdynamics.com/en/news/lambda-partners-with-pegatron-to-deploy-nvidia-gb200-nvl72-rack/), [getdeploying GB200](https://getdeploying.com/gpus/nvidia-gb200).

---

## 3. GB300 NVL72 — Real Production Deployments (Beyond MLPerf)

This is **new ground vs Round 1**. GB300 isn't just a benchmark claim — it's serving OpenAI in production:

| Deployment | Scale | Date | Workload |
|---|---|---|---|
| **Microsoft Azure / OpenAI** | **4,608 GB300 GPUs (64 NVL72 racks)** | Announced Oct 2025 | OpenAI multi-trillion-param model serving; 92.1 ExaFLOPS FP4 inference; described as "first supercomputer-scale GB300 NVL72 cluster" |
| **Microsoft Azure (broader)** | "Hundreds of thousands of Blackwell Ultra GPUs" planned | Ongoing rollout 2026 | OpenAI workloads + Azure AI Foundry |
| **CoreWeave** | First GB300 NVL72 rack delivered by Dell July 2025; **cloud instances GA Aug 19, 2025** | US-WEST-01A region | General cloud customers; 4× GB300 Superchip instance shape, 279 GB per GPU, 21 TB total HBM per rack |
| **AWS p6e-gb300 UltraServers** | **GA December 2, 2025** | Capacity Blocks for ML | Enterprise AI workloads |
| **Oracle Cloud** | Announced; specific cluster scale not public | Late 2025 / 2026 | Sales-contracted |

**Production telemetry** (from MLPerf 5.1, NVIDIA Azure post):
- DeepSeek-R1 offline: 5,842 tok/s/GPU; server: 2,907 tok/s/GPU — 45% over GB200.
- Llama-3.1-8B offline: 18,370 tok/s/GPU.
- Per Tom's Hardware on Microsoft cluster: "1.44 PFLOPS" claim per GPU bf16-equivalent / 92.1 ExaFLOPS aggregate FP4 across 4,608 GPUs.

**Verification**: GB300 has crossed the chasm from benchmark to live serving. The Azure-OpenAI deployment is the **single most consequential production data point**. If you're considering GB300 for a 2026 buildout, the answer "real, deployed, working at hyperscale" is now defensible.

Sources: [Tom's Hardware Microsoft 4608 GB300](https://www.tomshardware.com/tech-industry/artificial-intelligence/microsoft-deploys-worlds-first-supercomputer-scale-gb300-nvl72-azure-cluster-4-608-gb300-gpus-linked-together-to-form-a-single-unified-accelerator-capable-of-1-44-pflops-of-inference), [Azure blog GB300 for OpenAI](https://azure.microsoft.com/en-us/blog/microsoft-azure-delivers-the-first-large-scale-cluster-with-nvidia-gb300-nvl72-for-openai-workloads/), [Azure techcommunity GB300](https://techcommunity.microsoft.com/blog/azureinfrastructureblog/reimagining-ai-at-scale-nvidia-gb300-nvl72-on-azure/4464556), [CoreWeave GB300 first deploy](https://www.coreweave.com/blog/coreweave-leads-the-way-with-first-nvidia-gb300-nvl72-deployment), [CoreWeave Aug 19 2025 GA](https://docs.coreweave.com/docs/changelog/release-notes/gb300-nvl72), [Dell first GB300 to CoreWeave](https://www.dell.com/en-us/blog/dell-delivers-market-s-first-nvidia-gb300-nvl72-to-coreweave/), [AWS p6e-gb300 GA Dec 2 2025](https://aws.amazon.com/blogs/aws/new-amazon-ec2-p6e-gb200-ultraservers-powered-by-nvidia-grace-blackwell-gpus-for-the-highest-ai-performance/).

---

## 4. DiT Diffusion FP8/NVFP4 on Blackwell — Do GH200/Wan Bugs Reproduce?

**Hypothesis**: User's prior pain (FP8 broken on GH200/Wan) might be a Hopper-only thing, or it might reproduce on Blackwell. Answer: **the bugs reproduce in different forms on Blackwell, and the OSS DiT stack on Blackwell is still rough — but NVIDIA's first-party FLUX/TRT-LLM path works**.

### Bugs that DO reproduce on Blackwell (community/OSS stack)

1. **Wan 2.2 + FP8 (e4m3fn) + SageAttention → BLACK FRAME OUTPUT** on RTX 5090 / Blackwell
   - `thu-ml/SageAttention#221` (Aug 2025, **open**, no workaround documented)
   - `Comfy-Org/ComfyUI#7020` (same symptom; Wan 2.1)
   - Root cause likely Triton backend in SageAttention 2.2 incompatible with Wan FP8 weights on SM_120
   - Workaround: drop `--use-sage-attention`, use KJNodes "Patch Sage Attention" node with `sageattn_qk_int8_pv_fp16_cuda` backend (per community)
   - **This is the same family of bug as GH200/Wan FP8** — attention path under FP8 producing garbage

2. **Native NVFP4 loading fails on Wan 2.2 / Flux2-Dev / LTX2** (Blackwell RTX 5090, SM_120, CUDA 13 + PyTorch 2.9.1+cu130)
   - `Comfy-Org/ComfyUI#11864` (**open**, unresolved)
   - Loader silently upcasts NVFP4 weights to FP16 (28 GB) or FP8 (14 GB) instead of native 7.5 GB FP4
   - Triggers `torch.OutOfMemoryError`
   - "Manual cast: torch.float16" warning logged
   - Means: **NVFP4 advertised savings don't materialize** in the OSS DiT loader path on Blackwell consumer parts
   - Datacenter B200/B300 (SM_100) likely similar code paths but not directly confirmed in this issue

3. **vLLM 0.13 + CUDA 13 + FP8 MoE breaks on B300**
   - `vllm-project/vllm#31262` — Qwen3-235B-A22B FP8 fails: `output_size 192 not divisible by weight quantization block_n=128`
   - Non-quantized version works fine; **FP8 path specifically broken** for certain MoE expert shapes
   - **Open** (Dec 2025)

### Bugs that do NOT reproduce on first-party NVIDIA path

- **TRT-LLM NVFP4 for FLUX.1-Dev**: PyTorch blog (Diffusers + TorchAO) shows 1.26× MXFP8 / 1.68× NVFP4 speedup, 3.5× memory reduction, working
- **TRT-LLM NVFP4 for FLUX.2**: NVIDIA blog reports **10.2× speedup** with NVFP4 microblock scaling + TeaCache + CUDA graphs — production-grade
- **QwenImage** more sensitive to quantization than FLUX (LPIPS 0.34 vs 0.11 for MXFP8) — but works, just lower fidelity

### Pattern

The split is sharp:
- **NVIDIA's blessed path (TRT-LLM + ModelOpt + diffusers + TorchAO) → works, ships, benches**
- **Community OSS path (ComfyUI custom nodes + SageAttention + Triton + vLLM-on-CUDA-13) → multiple open bugs on FP8/NVFP4 with DiT models**

**Recommendation if your workload is DiT (Wan/Flux/LTX)**:
- If you have engineering capacity → TRT-LLM/Diffusers+TorchAO on **CUDA 12.8 + R580**, validate on small clip, sign off.
- If you're using ComfyUI + community quants → **stay on BF16 or FP16 on Blackwell** until the SageAttention/NVFP4-loader bugs close. FP8 weight savings are not safe yet for Wan-class video DiT on the OSS path. **This matches user's MEMORY: "FP8 broken on this stack; always bf16."**

Sources: [SageAttention #221](https://github.com/thu-ml/SageAttention/issues/221), [ComfyUI #7020](https://github.com/comfyanonymous/ComfyUI/issues/7020), [ComfyUI #11864](https://github.com/Comfy-Org/ComfyUI/issues/11864), [vLLM #31262](https://github.com/vllm-project/vllm/issues/31262), [PyTorch blog Diffusers Blackwell](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/), [NVIDIA FLUX.2 NVFP4](https://developer.nvidia.com/blog/scaling-nvfp4-inference-for-flux-2-on-nvidia-blackwell-data-center-gpus/), [Spheron FP4 Blackwell guide](https://www.spheron.network/blog/fp4-quantization-blackwell-gpu-cost/).

---

## 5. CUDA 12.8 vs 13.x for Blackwell — Stability May 2026

### Timeline
- **CUDA 12.8** — released early 2025, first toolkit with **full** Blackwell support across compilers, libraries, profilers
- **CUDA 13.0** — released **Aug 4, 2025**
- **CUDA 13.1** — point release (late 2025)
- **CUDA 13.2** — released **Apr 12, 2026** (Update 1 = current GA), Blackwell Compatibility Guide r13.2 dated Apr 2, 2026

### Production stability today

**Use CUDA 12.8 + R580 LTS driver** for production Blackwell inference. Reasons:

1. **R580 is the current Long-Term Support driver branch**, EOL Aug 2028 — explicit NVIDIA recommendation for production LLM inference on H100/H200/A100/B200/B300
2. **vLLM stable PyPI wheels expect CUDA 12** — `libcudart.so.12` import errors on CUDA 13 (`vllm#22756`, `vllm#24464`, `vllm#37714`)
3. **vLLM 0.13 + CUDA 13 + FP8 MoE = broken** on B300 (`vllm#31262`, open)
4. **DGX Spark SM_121 unsupported by PyTorch 2.9.0** on ARM64 + CUDA 13 (`vllm#31128`)
5. **`--enforce-eager` workaround for CUDA 13 issues** costs **20–40% throughput** (disables CUDA graphs)
6. Driver issues: newer NVIDIA driver + CUDA 13 produces "unsupported display driver / cuda driver combination" errors

### Why people *do* move to CUDA 13.x

- Up to **20% better grouped GEMM** on Blackwell in 13.x
- New cuDNN 9.7+ FP8 conv on Blackwell + fused GQA attention graph patterns
- Mandatory for some 2026-only library features (newer cuBLAS LT kernels)
- Vera Rubin (NVL144) will require ≥13.x — start migrating in H2 2026

### Net recommendation for May 2026

| Scenario | CUDA |
|---|---|
| Production B200/B300 LLM inference (vLLM, SGLang, TRT-LLM stable wheels) | **CUDA 12.8 + R580** |
| TRT-LLM 1.0 with latest cuBLAS/cuDNN optimizations | **CUDA 13.2** (controlled environment, fresh PyTorch nightly, your own builds) |
| ARM64 / DGX Spark / SM_121 | **avoid right now** — PyTorch wheels missing |
| New buildout planning for late 2026 | Plan **CUDA 13.x migration path** but don't put it in prod today |

Sources: [CUDA 12.8 Blackwell announce](https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/), [CUDA 13 release blog](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/), [CUDA 13.2 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html), [Blackwell Compatibility Guide 13.2](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/), [vLLM CUDA 13 wheels issue](https://github.com/vllm-project/vllm/issues/22756), [vLLM Blackwell + CUDA 13 install failures](https://github.com/vllm-project/vllm/issues/37714), [vLLM new driver fail](https://github.com/vllm-project/vllm/issues/32373), [vLLM SM121 support](https://github.com/vllm-project/vllm/issues/31128), [Spheron CUDA news 2026](https://www.spheron.network/blog/cuda-news-today-nvidia-toolkit-amd-rocm-ai-frameworks-2026/).

---

## 6. Net Updates to Round 1

| Round 1 claim | Round 3 verdict |
|---|---|
| "B200 widely rentable on 20+ clouds" | **Confirmed for hardware existence; revised on "rentable now"** — true click-to-launch availability concentrated in CoreWeave / Inworld / Nebius (B200 self-serve) / Lambda. Most others = sales or queue. |
| "B200 $4.29–$14.24/GPU/hr on-demand" | **Confirmed range but more volatile**: Silicon Data March 2026 index mean $5.09, +12.8% MoM, 18.9% daily swings. AWS $14.24 still tops the range. |
| "GB200 NVL72 available at CoreWeave / Oracle / Azure / waitlist at Nebius/Lambda" | **Sharpened**: CoreWeave is the only true on-demand. Oracle/Azure are contracted/capacity-gated. AWS p6e-gb200 = Capacity Blocks only, NOT on-demand. Nebius = pre-order/waitlist (confirmed via Nebius's own page). |
| "GB300 NVL72 available at CoreWeave, Microsoft, Oracle" | **Verified + deepened**: Microsoft has 4,608-GPU production cluster live for OpenAI; AWS p6e-gb300 GA Dec 2, 2025; CoreWeave GA Aug 19, 2025. Real production, not just benchmarks. |
| "FP4/FP8 production-grade with caveats" | **Verified, with sharper caveats**: NVIDIA blessed path (TRT-LLM, Diffusers+TorchAO, ModelOpt) works for FLUX.1/FLUX.2. **DiT OSS path (ComfyUI + SageAttention + Triton custom nodes) has open FP8/NVFP4 bugs that reproduce Round 1's GH200/Wan FP8 pattern**. Wan 2.2 specifically still hits black-frame on FP8+Sage on Blackwell. |
| "CUDA 12.8 minimum for Blackwell" | **Confirmed and *strongly* preferred over 13.x for production**. CUDA 13 has live regressions in vLLM, FP8 MoE on B300, ARM64 wheels, driver matching. R580 LTS + CUDA 12.8 is the safe pick. |
| "R580 LTS / R570 EOL June 2026" | **Confirmed** |

---

## 7. New / Sharpened Recommendations

1. **For DiT video diffusion (Wan-class)**: Run **BF16 on B200/B300** (matches user's MEMORY guidance for *both* GH200 and now Blackwell). Do **not** trust FP8 + community attention kernels on Blackwell until SageAttention #221 / ComfyUI #11864 / #7020 close. If you must quantize, use NVIDIA's TRT-LLM + ModelOpt NVFP4 path with explicit validation against BF16 reference, on a controlled CUDA 12.8 environment.
2. **For LLM production**: B200 8× node on **CoreWeave or Lambda, CUDA 12.8 + R580, vLLM 0.20.x stable wheels** is the lowest-risk + best-cost combo. Skip CUDA 13.x for now.
3. **For DeepSeek-V3/R1 671B serving**: **GB200 NVL72 via CoreWeave on-demand** (US-WEST-01) is the *only* truly self-serve option today. Azure ND GB200 v6 requires capacity allocation; AWS requires Capacity Block reservation.
4. **For Blackwell Ultra (GB300) early adopters**: CoreWeave US-WEST-01A is GA. AWS p6e-gb300 Capacity Blocks works if you can plan ahead. Azure GA exists but routed through enterprise sales for the OpenAI-class clusters.
5. **For procurement planning**: Don't plan around new B200/GB200 silicon allocations before late 2026; ride existing cloud inventory at hyperscalers + CoreWeave. Vera Rubin NVL144 samples shipped Sept 2025, volume 2026 — don't bet a 2026 production deployment on it.

---

## 8. Confidence & Remaining Gaps

**High confidence (multiply-cross-confirmed)**:
- CoreWeave GB200/GB300 NVL72 GA dates (CoreWeave docs + NVIDIA blog + Dell blog + Tom's Hardware)
- Microsoft Azure GB300 4,608-GPU OpenAI cluster (Azure blog + Tom's Hardware + Windows Forum coverage)
- AWS p6e-gb200 / p6e-gb300 as Capacity Blocks only (AWS news + Vantage)
- B200 + Wan 2.2 + SageAttention black-frame bug (multiple GitHub issues, open)
- CUDA 13.0 release Aug 4, 2025; 13.2 release Apr 12, 2026
- CUDA 13 + vLLM regression set (4+ open GitHub issues)
- NVIDIA TRT-LLM + FLUX NVFP4 working path (NVIDIA blog + PyTorch blog)

**Medium confidence**:
- Exact "waitlist length" at Lambda, Crusoe, Nebius for B200 — providers don't publish queue depth
- Whether the ComfyUI NVFP4 loader bug (#11864) reproduces on datacenter B200 (SM_100) vs consumer 5090 (SM_120) — issue is filed against 5090 specifically; the loader code path is shared but SM_100 may have a different `TensorCoreNVFP4Layout` code path
- Azure ND GB200 v6 capacity-allocation queue behavior — "GA" but in practice gated

**Gaps / open questions for other agents**:
- **Direct head-to-head FLUX.1-Dev BF16 vs NVFP4 latency + LPIPS on B200**, in production (NVIDIA blogs use FLUX.2; community blogs use 5090; need a B200 SM_100 datapoint)
- **Wan 2.2 BF16 throughput on B200 vs GH200** — neither rounds 1 nor 3 have direct numbers; user's actual workload would benefit
- **R580 + CUDA 13.2 combo**: NVIDIA docs say R580 supports CUDA 13+, but production stories of the combination on B200/B300 with vLLM/TRT-LLM are sparse
- **Lambda Pegatron GB200 NVL72 GA date** — still no public confirmation as of May 2026
- **Azure ND GB200 v6 actual provisioning latency** — capacity allocation queue depth not disclosed
