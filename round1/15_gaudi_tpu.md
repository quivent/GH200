# Agent 15 — Intel Gaudi 3 and Google TPU as Alternatives to GH200 for Inference

**Date:** 2026-05-25
**Scope:** Intel Gaudi 3 (Habana) and Google TPU v5e/v5p/v6e (Trillium)/v7 (Ironwood) as inference alternatives to NVIDIA GH200, in May 2026.
**Method:** ~12 WebSearch queries + WebFetches against Google Cloud, Intel/IBM, vLLM, Spheron, MLCommons.

---

## 1. Executive Summary

Two real alternatives to GH200 for open-weights inference exist in May 2026, but with very different access models and ecosystem risks.

- **Intel Gaudi 3 (HL-325L OAM, 128GB HBM2e):** Hardware shipping and rentable on IBM Cloud at ~$60/hr per 8-card node (~$7.50/card-hr list, ~$3.75 promo) and on Intel Tiber AI Cloud (mostly custom-quote). **Headline problem in May 2026:** Intel announced the Gaudi product line is being wound down — Falcon Shores cancelled to internal-only, Crescent Island (inference GPU) entering customer test H2 2026, Jaguar Shores (rack-scale, HBM4, "Gaudi" brand carry-over) targeting 2027. SynapseAI open-source contributions have ceased. The `HabanaAI/vllm-fork` is being EOL'd; the upstream `vllm-project/vllm-gaudi` plugin is the path forward and is production-ready as of v0.15.1 (Feb 2026). Software is functional but lagging upstream vLLM, with no MLPerf Inference v5.1 submission visible.
- **Google TPU v6e / v7 (Ironwood):** Ironwood is GA as of early 2026 (Google Cloud Next, Las Vegas). v7 has 192 GiB HBM3e at 7.37 TB/s and 4,614 FP8 TFLOPS/chip — meaningfully ahead of GH200 on memory and competitive on compute. **Headline problem:** Ironwood requires GKE or TPU Cluster Director under an "All Capacity mode reservation" via account team — i.e., not truly on-demand for arbitrary users. Trillium (v6e) is on-demand at ~$2.70–$4.50/chip-hr depending on region/source. vLLM-TPU (`tpu-inference`) was unified across JAX and PyTorch in late 2025; Llama 3.1-70B v6e-8 reached ~2.1× the Feb-2025 prototype, but no third-party Llama 3.3-70B benchmark on Ironwood has surfaced publicly.

**Bottom line:** TPU v6e/v7 is the technically stronger alternative if you can live with GCP lock-in, JAX/XLA path, and reservation-style access. Gaudi 3 makes sense only if (a) IBM Cloud or specific sovereign-cloud regions are mandatory or (b) the IBM 50% promo (GPU4YOU, expires 30 Jun 2026) brings $/token below GH200 for your specific workload — and you accept that you are buying into a sunsetting product.

---

## 2. Hardware Side-by-Side

| Spec | NVIDIA GH200 (96 GB) | Intel Gaudi 3 (HL-325L OAM) | Google TPU v6e (Trillium) | Google TPU v7 (Ironwood) |
|---|---|---|---|---|
| HBM capacity | 96 GB HBM3 (also 144 GB HBM3e SKU) | 128 GB HBM2e | 32 GB HBM | **192 GiB HBM3e** |
| HBM bandwidth | ~4 TB/s (HBM3) / ~4.8 TB/s (HBM3e) | 3.7 TB/s | ~1.6 TB/s | **7.37 TB/s** |
| FP8 dense TFLOPS | ~3,958 (H200 die) | 1,678 (OAM) / 1,835 (PCIe) | ~918 | **4,614** |
| BF16 TFLOPS | ~1,979 | 1,678 | ~459 | 2,307 |
| Interconnect | NVLink 4 @ 900 GB/s + Grace-Hopper coherent | 24× 200 GbE RoCE on-package (~1.2 TB/s aggregate) | ICI (4D torus) | ICI 1,200 GB/s bidir per chip (3D torus) |
| Pod scale | NVL32/72 (Hopper) | OAM tray of 8 | up to 256 chips/slice | up to 9,216 chips (pod) |
| Single-chip 70B fit | Yes (BF16 ~140 GB → needs 2× GH200) | Yes (Llama 4 Maverick on 1× 8-card node) | No (needs slice) | **Yes (70B BF16 fits on 1 chip)** |
| Framework first-class | CUDA, TensorRT-LLM, vLLM, PyTorch | SynapseAI + PyTorch via bridge, vllm-gaudi plugin | JAX/MaxText, PyTorch/XLA, vLLM-TPU | **JAX only (no TF, no PT/XLA listed in v7 docs)** |

Sources: Intel Gaudi 3 datasheet ([cdrdv2-public.intel.com 845118 PDF](https://cdrdv2-public.intel.com/845118/gaudi-3-ai-accelerator-30-3-30.pdf)), TPU7x docs ([docs.cloud.google.com/tpu/docs/tpu7x](https://docs.cloud.google.com/tpu/docs/tpu7x)), Trillium GA blog ([cloud.google.com/blog Trillium GA](https://cloud.google.com/blog/products/compute/trillium-tpu-is-ga)).

---

## 3. Intel Gaudi 3 — Detailed Findings

### 3.1 Product status & Intel strategy (critical context)

- **Falcon Shores cancelled** as a shipping product (became internal-only) — Tom's Hardware, Notebookcheck reporting from 2025 carried through to 2026 roadmap analyses.
- **"Intel officially concluded the Gaudi product line"** per PC Outlet and tomshardware roadmap pieces. Gaudi 3 is the final standalone Habana-lineage accelerator.
- **Crescent Island** = inference-focused GPU, customer testing H2 2026.
- **Jaguar Shores** = 2027, rack-scale with HBM4 + silicon photonics; carries the "Gaudi" brand name forward but is a different design (not Habana TPC architecture).
- **SynapseAI open-source driver:** Phoronix reported Intel quietly stopped accepting patches; long-promised upstream kernel driver delayed through multiple Habana-team layoffs.

This is a **buy decision into a wind-down product**. The chip itself is fine; what is decaying is the team and the software cadence.

Sources: [PC Outlet — Intel Ends Gaudi Line](https://pcoutlet.com/software/ai/intel-ends-gaudi-line-what-jaguar-shores-means-for-the-future-of-ai-hardware), [Tom's Hardware — Falcon Shores cancelled](https://www.tomshardware.com/tech-industry/artificial-intelligence/intel-cancels-falcon-shores-gpu-for-ai-workloads-jaguar-shores-to-be-successor), [Tom's Hardware — Intel roadmap 2026-2028](https://www.tomshardware.com/tech-industry/semiconductors/intel-chip-roadmap-2026-2028), [Phoronix — Intel-SynapseAI-Stops](https://www.phoronix.com/news/Intel-SynapseAI-Stops).

### 3.2 Software stack — vLLM-Gaudi maturity

- `HabanaAI/vllm-fork` (the original fork) **reaches EOL in v1.24.0**; Gaudi software 1.23.0 release notes mark it deprecated. Migration path is the upstream community plugin `vllm-project/vllm-gaudi`.
- `vllm-gaudi` plugin: production-ready in v1.23.0; v0.15.1 (Feb 2026) added Granite 4.0-h, Qwen3-VL (dense + MoE), and Llama 4 stability fixes. v0.13.0 (Jan 2026) added experimental dynamic quantization for MatMul + KV cache.
- **Known gaps as of May 2026:**
  - Model Runner v2 (MRV2): in upstream vLLM, **not yet ported to HPU**.
  - **SGLang has no HPU backend** (RadixAttention scheduler missing). For workloads where SGLang's prefix sharing matters, Gaudi 3 is out.
  - No public MLPerf Inference v5.1 submission for Gaudi 3 on Llama 2/3 70B (v5.1 — Sept 2025; subsequent v6 cycle no submissions reported either).

Sources: [vllm-gaudi GitHub](https://github.com/vllm-project/vllm-gaudi), [vLLM Gaudi plugin release notes](https://docs.vllm.ai/projects/gaudi/en/latest/release_notes.html), [Habana docs 1.23.0 release notes](https://docs.habana.ai/en/latest/Release_Notes/GAUDI_Release_Notes.html).

### 3.3 Cloud availability and pricing

| Provider | Status | Headline price | Regions |
|---|---|---|---|
| **IBM Cloud (VPC Virtual Servers)** | GA — first cloud, since 2025 | **~$60/hr per 8× Gaudi 3 node** (vs ~$85/hr NVIDIA equivalent IBM quoted). 50% off first 6 months with promo **GPU4YOU** (expires 30 Jun 2026). | Frankfurt, Washington DC, Dallas |
| **Intel Tiber AI Cloud** | Early access / "coming soon" wording persists; pricing is custom-quote for clusters | Not publicly listed | Multi-region |
| **AWS / Azure / GCP** | **Not available** | — | — |
| **Dell / Supermicro / Gigabyte on-prem** | GA | List ~$125k/server (8× OAM) approximations from industry analysis | n/a |

Sources: [IBM Newsroom — Intel + IBM Gaudi 3 GA](https://newsroom.ibm.com/blog-intel-and-ibm-announce-the-availability-of-intel-gaudi-3-ai-accelerators-on-ibm-cloud), [DCD — Intel Gaudi 3 IBM Cloud](https://www.datacenterdynamics.com/en/news/intels-gaudi-3-ai-chips-now-available-on-ibm-cloud/), [Intel Tiber AI Cloud pricing](https://ai.cloud.intel.com/pricing/), [Intel Gaudi 3 Cost-Effective Choice infographic](https://www.intel.com/content/www/us/en/content-details/850774/bringing-cost-effective-choice-to-ai-innovation-with-intel-gaudi-3-ai-accelerators-on-ibm-cloud-infographic.html).

### 3.4 Throughput vs GH200/H200 (Llama 3 / 3.3 70B FP8)

From Spheron's May 2026 analysis (explicitly **projections** — no MLPerf submission and no first-party measured numbers from Intel for Llama 3.3 70B FP8 at multiple batch sizes):

| Batch | Gaudi 3 (tok/s) | H200 (tok/s) | Gaudi 3 / H200 |
|---|---|---|---|
| 1 | ~520 | ~680 | 0.76× |
| 8 | ~1,350 | ~1,800 | 0.75× |
| 32 | ~1,850 | ~2,500 | 0.74× |
| 128 | ~2,100 | ~3,200 | 0.66× |

**DCD reported** (Llama 3.1 405B, first independent test): H200 outperformed Gaudi 3 **by 9×**. This is on 405B which is much harder for Gaudi due to limited interconnect aggregate vs NVLink; it's not representative of 70B but does show the upper bound of Gaudi 3's weakness on large-model multi-card scaling.

**Break-even for cost-per-token vs H200:** Gaudi 3 needs to be priced below ~$3.12/hr/card at batch-32 throughput to win on $/M-tokens against H200 at $4.22/hr.
- IBM list ~$7.50/card-hr → **loses on $/token** to H200 at list.
- IBM with GPU4YOU promo → ~$3.75/card-hr → **borderline**, still loses on raw TCO except for memory-bound workloads.

Sources: [Spheron — Gaudi 3 vs H200/B200 LLM inference 2026](https://www.spheron.network/blog/intel-gaudi-3-vs-nvidia-h200-b200-llm-inference-2026/), [DCD — H200 9× Gaudi 3 on 405B](https://www.datacenterdynamics.com/en/news/nvidia-h200-outperforms-intel-gaudi-3-by-factor-of-nine-across-first-llama-31-405b-benchmark-test-exclusive/), [Intel Llama 4 on Gaudi 3 community blog](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Deploying-Llama-4-Scout-and-Maverick-Models-on-Intel-Gaudi-3/post/1699547).

---

## 4. Google TPU — Detailed Findings

### 4.1 Generation map (May 2026)

| Gen | HBM | FP8 TFLOPS | Status | Pricing |
|---|---|---|---|---|
| v5e | 16 GB | ~393 | GA, on-demand | $1.20/chip-hr on-demand, $0.54 1y, $0.39 3y |
| v5p | 95 GB | ~459 | GA, mostly via reservation | $4.20/chip-hr (training-tier) |
| v6e (Trillium) | 32 GB | ~918 | **GA, on-demand & GKE** | ~$2.70/chip-hr on-demand; ~$0.55-$1.89 with 1y/3y CUD (sources vary) |
| v7 (Ironwood) | **192 GiB** HBM3e | **4,614** | **GA early 2026 — reservation only via account team** | Not publicly listed |

Notable Ironwood requirement from official docs: "**Deployment mandates Google Kubernetes Engine (GKE) or TPU Cluster Director via All Capacity mode reservation.**" This means **you cannot spin one up like a GH200 on Lambda** — you need an account-team relationship.

Sources: [TPU7x docs (Ironwood)](https://docs.cloud.google.com/tpu/docs/tpu7x), [TPU v6e docs](https://docs.cloud.google.com/tpu/docs/v6e), [Google Cloud TPU pricing page](https://cloud.google.com/tpu/pricing), [Awesome Agents TPU v7 summary](https://awesomeagents.ai/hardware/google-tpu-v7-ironwood/), [Ironwood Google Blog](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/ironwood-tpu-age-of-inference/).

### 4.2 Software stack — vLLM-TPU, JAX/MaxText, SAX

- **vLLM-TPU (`vllm-project/tpu-inference`)**: October 2025 redesign — unified JAX + PyTorch under single XLA lowering path. Llama 3.1-8B v6e-1 throughput improved **3.6×**, Llama 3.1-70B v6e-8 improved **2.1×** vs Feb 2025 prototype, and **~5×** total vs original. PyTorch/XLA path retained but JAX is the optimization target.
- **MaxText (AI-Hypercomputer/maxtext)**: Active in 2026 — Kimi-K2 and DeepSeek-V3.2 added April 2026. This is the JAX-native reference inference/training stack for TPU.
- **SAX (Saxml)**: Google's internal serving framework, open-sourced, used for production multi-host serving on TPU pods. Less community traction than vLLM, but the canonical large-pod path.
- **Ironwood docs explicitly say "Only JAX framework is supported; TensorFlow is not."** No mention of PyTorch/XLA support tier in the v7 docs as of May 2026 — that's a meaningful regression from Trillium for PyTorch-first teams. (PyTorch/XLA likely still works but is not first-class per the v7 page.)

Sources: [vLLM TPU blog (Oct 2025)](https://blog.vllm.ai/2025/10/16/vllm-tpu.html), [vllm-project/tpu-inference GitHub](https://github.com/vllm-project/tpu-inference), [MaxText GitHub](https://github.com/AI-Hypercomputer/maxtext).

### 4.3 Pricing — concrete figures

Different sources report different numbers because Google publishes per-chip-hour but bills per VM-host and runs frequent regional promotions. Triangulating:

- **TPU v6e on-demand**: $2.70/chip-hr (most-cited 2026 number); some 2026 reports say $4.20-$4.50 (likely older list pricing pre-Trillium-GA). One outlier $1.375/hr appears to be a regional special.
- **TPU v6e 3-year CUD**: $0.55-$0.39/chip-hr (60%+ off on-demand).
- **TPU v7 (Ironwood)**: **No published per-hour price**. Anthropic's deal — 400k chips Broadcom finished racks + 600k via GCP rental — implies large-customer pricing only; Zephyr/SemiAnalysis estimates put Google's capital cost at ~$15k/v7 chip, suggesting on-demand (if it ever appears) could be ~$6-9/chip-hr at typical 3-year cloud margin.

Sources: [Google Cloud TPU pricing](https://cloud.google.com/tpu/pricing), [Spheron — TPU v6e vs B200 2026](https://www.spheron.network/blog/google-tpu-trillium-v6-vs-nvidia-b200-llm-inference/), [SemiAnalysis — TPUv7 The 900lb Gorilla](https://newsletter.semianalysis.com/p/tpuv7-google-takes-a-swing-at-the), [Anthropic + Google announcement](https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services), [DCD — Anthropic 1M TPU deal](https://www.datacenterdynamics.com/en/news/google-and-anthropic-confirm-massive-1gw-cloud-deal-with-up-to-one-million-google-tpus/).

### 4.4 Third-party Llama 3.3-70B benchmarks on TPU v6e / v7

**Hard finding: there is no public third-party Llama-3.3-70B benchmark on TPU v7 Ironwood as of May 25, 2026.** Google has published comparative claims (4× v6e, 10× v5p per chip for "training and inference"), but no Artificial Analysis, MLPerf, or independent harness has posted a tokens/sec figure on Ironwood for Llama-3.3-70B.

Closest concrete numbers:
- **TPU v5p, Llama-7B BF16, batch 128**: ~6,200 tok/s on a 4-chip slice — comparable to a single H100 80GB via vLLM at ~5,800 tok/s. (Source: easecloud.io 2026 piece — itself secondary.)
- **TPU v6e-8, Llama 3.1-70B**: post-Oct-2025 vLLM-TPU = 2.1× the Feb-2025 prototype. Absolute numbers not given in the blog.
- **TPU v6e + Trillium Llama 2-70B**: Google blog claims ~2× v5e inference throughput.

Compared to GH200 Llama 3.3-70B numbers from Baseten/Lambda (GH200 = +32% vs H100, fits 70B on a single 96GB chip), the lack of Ironwood public numbers is striking — and is itself a signal.

Sources: [Baseten — Llama 3.3 70B on GH200 Lambda](https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/), [easecloud.io — TPU v5p LLM benchmarks](https://blog.easecloud.io/ai-cloud/llm-throughput-with-google-tpu-v5p/), [vLLM TPU blog](https://blog.vllm.ai/2025/10/16/vllm-tpu.html).

---

## 5. GH200 Baseline (anchor for comparison)

- **Hardware:** 96GB HBM3 (or 144 GB HBM3e variant), Hopper compute (FP8 ~3.96 PFLOPS), Grace ARM CPU coherently attached at 900 GB/s. Llama 3.3-70B fits on a single chip in BF16-equivalent.
- **Pricing (May 2026):** Lambda $1.99/hr (on-demand single-GPU), CoreWeave $6.50/hr (enterprise SLA), market range $1.99-$6.50/hr; market-average ~$3.59/hr, down ~3% YoY.
- **Ecosystem:** First-class CUDA, TensorRT-LLM, vLLM, SGLang, FlashAttention-3, etc. Zero migration cost from H100.
- **Baseten:** GH200 +32% vs H100 on Llama-3.3-70B serving on Lambda Cloud.

Sources: [GetDeploying — GH200 cloud pricing](https://getdeploying.com/gpus/nvidia-gh200), [Spheron — GPU Cloud Pricing 2026](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/), [Baseten GH200 Llama 3.3 blog](https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/).

---

## 6. Side-by-Side Verdict Table

| Dimension | GH200 | Gaudi 3 | TPU v6e | TPU v7 Ironwood |
|---|---|---|---|---|
| Open-cloud on-demand | **Yes** (Lambda $1.99-$6.50) | Yes (IBM ~$7.50/card, ~$3.75 w/promo) | **Yes** ($2.70/chip-hr) | **No** — reservation only via account team |
| Llama 3.3-70B fits 1 chip | Yes | Yes (and 405B fits 1 node) | No (slice needed) | **Yes** |
| MLPerf v5.1 Llama-2-70B submission | Yes (NV) | **No** | n/a | n/a |
| Public third-party Llama 3.3-70B tok/s | **Yes** | Projections only | Indirect/relative | **No** |
| vLLM first-class | Yes (upstream) | Plugin, lags upstream | Plugin, unified Oct 2025 | Plugin (JAX path) |
| SGLang | Yes | **No HPU backend** | Yes (limited) | Unknown |
| Vendor commitment to follow-on | Yes (Blackwell, Vera Rubin) | **No** — Gaudi line ended | Yes (v8 previewed at TSMC 2nm) | Yes |
| Migration cost from CUDA | None | Medium (vLLM-Gaudi works but quirky) | Medium-high (JAX-first) | High (JAX-only per v7 docs) |
| Best for | Standard answer | IBM sovereign-cloud / promo-window arbitrage | Open-cloud TPU users on Trillium today | Anthropic-class committed contracts |

---

## 7. When to Pick Each

**Pick GH200** if: you want zero migration friction, on-demand from multiple providers, broad SGLang/vLLM/TRT-LLM support. Default answer for May 2026.

**Pick Gaudi 3** only if: (a) you are already on IBM Cloud and the GPU4YOU promo (50% off through 30 Jun 2026) makes ~$3.75/card-hr beat H200/GH200 cost-per-token on your specific workload **AND** (b) you accept that you are buying into a 2024-launched product whose follow-ons (Crescent Island H2-2026, Jaguar Shores 2027) are different chips on different software paths.

**Pick TPU v6e (Trillium)** if: you are willing to invest in JAX/MaxText or vLLM-TPU, want lower $/token with 1-3y commit ($0.55-$1.89/chip-hr), and your model fits well in a slice (8-16 chip slices are typical for 70B).

**Pick TPU v7 (Ironwood)** if: you have an account-team relationship with GCP, are willing to take an All Capacity reservation, and need 70B-405B+ models on a JAX-first stack — and you are okay being a relatively early benchmark publisher because public Llama-3.3-70B numbers don't exist yet.

---

## 8. Confidence & Gaps

### High confidence
- Gaudi product line is ending; Crescent Island / Jaguar Shores are the announced succession plan. Multiple independent sources (Tom's Hardware, PC Outlet, Notebookcheck, Phoronix) align.
- Ironwood is GA in 2026 but **not on-demand** in the same sense as GH200/Trillium — reservation only per official Google Cloud docs.
- vLLM `HabanaAI/vllm-fork` is being EOL'd in v1.24.0 in favor of community `vllm-gaudi` plugin.
- IBM Cloud Gaudi 3 list ~$60/hr per 8-card node and the GPU4YOU 50%-off-6-months promo (expires 30 Jun 2026).
- TPU v7 specs: 192 GiB HBM3e, 7.37 TB/s, 4,614 FP8 TFLOPS, JAX-only per docs.

### Medium confidence
- **Gaudi 3 Llama 3.3-70B throughput numbers** — the batch-1/8/32/128 figures (520/1350/1850/2100 tok/s) are **Spheron projections**, not measured. Intel publishes Llama 4 Maverick on 1×8-card node but not Llama-3.3-70B with batch-sweeps in a directly comparable format.
- **TPU v6e on-demand price** — sources cluster at $2.70/chip-hr but two outliers ($4.20-$4.50 and $1.375) suggest region/SKU confusion. Verify on `cloud.google.com/tpu/pricing` for your region.
- **TPU v6e CUD pricing** — $0.39-$0.55 (3y) and $0.55-$1.89 (1y) ranges came from secondary blogs; Google's official table may have shifted.

### Gaps / unknowns
- **No public MLPerf submission for Gaudi 3** on Llama 2/3 70B inference in v5.1 (Sept 2025) or v6 cycle — this is unusual for a chip that's been on the market 18+ months and is a red flag.
- **No public third-party Llama-3.3-70B tokens/sec on Ironwood** as of May 2026. Google's "4× v6e per chip" claim is unverified by independent harness.
- **Intel Tiber AI Cloud Gaudi 3 on-demand pricing** — never publicly listed; everything is custom quote.
- **PyTorch/XLA support tier on Ironwood** unclear — v7 docs say JAX-only; whether PT/XLA "works but is slow" or "is unsupported" matters a lot for PyTorch-first teams.
- **AWS Gaudi 3 instances** — none. AWS hosts Trainium/Inferentia; if AWS is a hard requirement, Gaudi 3 is unavailable.

### Bias note
- Spheron is a GPU marketplace and may bias toward GPU-favorable framings. Intel and IBM materials are first-party and bias toward Gaudi 3 cost wins. Google's Ironwood blogs cite only relative numbers vs prior TPU generations, never vs NVIDIA. None of the central throughput claims here are MLPerf-style apples-to-apples.

---

**Output file:** `/home/ubuntu/research/gh200_inference/round1/15_gaudi_tpu.md`
