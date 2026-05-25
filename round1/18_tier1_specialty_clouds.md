# Tier-1 Specialty GPU Clouds — Monthly Inference Rentals (Round 1, Agent 18)

**Date:** 2026-05-25
**Scope:** CoreWeave, Crusoe, Together AI, Nebius AI, Cudo Compute, Civo, Oracle OCI, Yotta Shakti — H100 / H200 / GH200 / B200 (and GB200/GB300 where present) for sustained monthly inference workloads.
**Method:** WebSearch (15+ queries) + WebFetch of each provider's own pricing pages + third-party benchmark/rating sources (SemiAnalysis ClusterMAX 2.0, getdeploying, computeprices, spheron, thundercompute), press coverage of 2025-2026 hyperscaler reservations.

---

## 1. Headline Pricing Table (per-GPU-hour, USD, May 2026)

| Provider | H100 (on-demand) | H100 (reserved) | H200 (on-demand) | H200 (reserved) | GH200 | B200 (on-demand) | B200 (reserved) | GB200/GB300 |
|---|---|---|---|---|---|---|---|---|
| **CoreWeave** | $4.25–$4.76 PCIe/HGX (classic) ; ~$2.44 (current pub) | up to 60% off (sales) | ~$2.58 (pub) ; $6.31 in HGX 8x | sales | $6.50 | $8.60 (SXM) / $10.50 NVL | sales (15–30% off for 12-mo) | GB200 NVL72 ~$10.50/GPU-hr ; GB300 NVL72 GA Aug 2025 |
| **Crusoe** | $3.90/GPU-hr HGX | spot + reserved (sales) | $4.29/GPU-hr HGX | sales (multi-month) | n/a (no GH200 SKU) | "Contact sales" | sales | GB200 contact sales |
| **Together AI** | $5.49 HGX cluster ; $6.49 dedicated endpoint | $4.99 (7–30d) / $4.49 (31–90d) / $3.99 (91–180d) | $6.79 HGX | $5.95 / $4.99 / $4.55 (same tiers) | n/a | $9.95 HGX ; $11.95 dedicated | $9.65 / $9.35 / $9.09 | GB200/GB300 NVL72: sales |
| **Nebius** | $2.95 on-demand ; $1.25 preempt | as low as $2.30 for 3+ mo (35% off) | $3.50 on-demand ; $1.45 preempt | ~$2.30/hr at 3+ mo | n/a (no public GH200) | $5.50 on-demand ; $2.90 preempt | up to 35% off / 20% Blackwell pre-order | GB200 NVL72 pre-order |
| **Cudo Compute** | ~$1.80–$2.45/GPU-hr (16-GPU cluster ~$30–50/hr) | up to 57% off w/ 1-yr or 3-yr commit | not publicly listed (sales) | sales | n/a | sales | sales | n/a publicly |
| **Civo** | ~$2.00–$3.00/GPU-hr range (mid-market) | 36-mo commit (e.g. L40S $0.89 vs $1.29 on-demand) | yes (listed at getdeploying) | sales | n/a | yes (B200) | sales | n/a |
| **Oracle OCI** | $10/GPU-hr list (BM.GPU.H100.8 = $80/hr) | 15–45% off w/ 1-yr UCC ; 25–60% off w/ 3-yr / $5M+ | $10/GPU-hr (BM.GPU.H200.8 = $80/hr) | same UCC bands | n/a (uses H200 + GB200) | yes via OCI Supercluster | UCC discounted | GB200 NVL72 GA, 131,072 GPU cluster ; pricing via sales |
| **Yotta Shakti** (India) | ₹351/GPU-hr bare metal ; ₹192,113/mo for 1xH100 dedicated VM (~$2,300) ; SLURM cluster ₹386 ; K8s ₹404 | govt IndiaAI: ₹115.85–₹150/hr | not on public pricing yet | sales | n/a | HGX B200 (640 servers being deployed via Gorilla) | sales | n/a publicly |

Notes on conversions: at 730 hr/month — H100 at $2.44/hr ≈ $1,780/mo; H200 at $2.30/hr (Nebius 3-mo commit) ≈ $1,680/mo; H100 reserved (Together 91-180d) at $3.99/hr ≈ $2,910/mo; OCI H200 list at $7,300/GPU/mo (before commit discount). Yotta ₹192,113/mo at ~₹83.5/USD ≈ $2,300/H100/mo full-VM (cheapest dedicated tier on the list once you exclude government IndiaAI rate).

---

## 2. Provider Deep-Dives

### CoreWeave — Platinum (sole ClusterMAX 2.0 Platinum, both years)
- **Tier rating:** Only Platinum provider per SemiAnalysis ClusterMAX 2.0 (2026). NVIDIA's first "Elite" cloud services provider; NVIDIA Exemplar Cloud certified for inference.
- **Pricing structure (Mar-2026 redesign):** New "Capacity Plans" framework: Reservations (always-on, up to 60% off), Flex Reservations (peak-guarantee with lower holding fee + full rate only on burn), On-Demand, Spot (new). 12-month commitments cited at 15-30% discount; multi-year deals push to 60% off.
- **Fleet:** GA: H100 HGX, H200 HGX, GH200, B200 HGX, GB200 NVL72 (first cloud to GA, late 2024), GB300 NVL72 (first to GA, Aug 19, 2025). Approx 45,000+ GPUs across DCs, multi-billion-GPU footprint via the OpenAI deal.
- **Networking:** Quantum-2 NDR InfiniBand, 3,200 Gbps per H200/B200 node (8x 400G ConnectX-7), rail-optimized topology with SHARP in-network reductions, topology-aware scheduling. H200 clusters scale to ~42,000 GPUs.
- **Storage:** VAST Data signed $1.17B exclusivity deal (Nov 2025) — now CoreWeave's primary data platform. Still offers WEKA, DDN, IBM Spectrum Scale, Pure Storage as dedicated tiers. Proprietary "LOTA" (Local Object Transfer Accelerator) for caching to local NVMe (kills thundering-herd for inference cold-starts).
- **Orchestration:** Kubernetes-native bare-metal (no hypervisor); SUNK = Slurm-on-Kubernetes; Run:ai (NVIDIA) supported.
- **Reputation:** Hyperscaler-class. OpenAI committed ~$22.4B across three contracts (Mar/May/Sep 2025 — $11.9B + $4B + $6.5B). MLPerf Inference v6.0 (Apr 2026) leadership on GB200/GB300; CoreWeave inference 8-10x faster cold-start than "generalized cloud" (45-70s vs 270-390s).
- **Fit for monthly inference:** Excellent for serious inference at scale (>=100 GPU). Premium pricing, but the LOTA + Kubernetes + Run:ai stack is purpose-built for disaggregated inference. Reservations require sales engagement; minimum commit is effectively contract-driven (no self-serve monthly reserved on H200/B200).
- **Caveat:** HyperFRAME Q1 2026 review flags power/networking/activation timing as the limiting factor on capacity; backlog ($55B) means new monthly customers compete with hyperscaler reservations.

### Crusoe — Gold (ClusterMAX 2.0)
- **Tier rating:** Gold. Same band as Nebius, Oracle, Azure, Fluidstack.
- **Pricing:** H100 HGX $3.90/GPU-hr; H200 HGX $4.29/GPU-hr; B200 + GB200 NVL72 contact-sales. Three modes: on-demand, spot, reserved.
- **Fleet:** H100, H200, B200 (HGX), GB200 NVL72. Stargate Abilene TX campus operated by Crusoe is building 1.2 GW (8 buildings) for the Oracle/OpenAI contract — first two buildings live in Sep 2025; remaining six expected mid-2026. Expansion to 2.0 GW was canceled in Mar 2026; NVIDIA reportedly placed a $150M deposit to backfill with Meta capacity.
- **Networking:** InfiniBand HDR or NDR; 1,600-3,200 Gbps depending on SKU/cluster.
- **Storage:** Persistent disks $0.08/GiB/mo; shared disks $0.07/GiB/mo. Doesn't publicly headline a VAST/WEKA partnership (more vertical/integrated).
- **Orchestration:** Crusoe Managed Kubernetes (CMK, K8s 1.31); curated ubuntu22.04-nvidia-slurm VM image for Slurm.
- **Reputation:** 99.98% uptime claim; Windsurf endorsement; <6 min support response, 24-hr resolution, 99.5% SLA; named to Fast Company Most Innovative Companies 2026. Renewable-powered (stranded gas + grid renewables).
- **Fit for monthly inference:** Strong second-choice to CoreWeave for H100/H200 production inference. Reserved capacity requires sales but B200 reservations are scarce given Stargate priority. Watch for spot pricing for elastic inference.

### Together AI — Gold/Silver overlap
- **Tier rating:** Listed Gold in one source, Silver in another in ClusterMAX 2.0 (right at the boundary).
- **Pricing (clear tiered table — best public reserved pricing transparency in this set):**
  - HGX H100 cluster: on-demand $5.49 → $4.99 (7-30d) → $4.49 (31-90d) → $3.99 (91-180d)
  - HGX H200 cluster: $6.79 → $5.95 → $4.99 → $4.55
  - HGX B200 cluster: $9.95 → $9.65 → $9.35 → $9.09
  - Dedicated endpoints: 1xH100 $6.49/hr, 1xB200 $11.95/hr (single-tenant, pay-per-token also available)
- **Networking:** Per their docs: InfiniBand multi-node, "real-time scaling", persistent storage.
- **Storage:** Persistent volumes + shared storage via Kubernetes; not VAST/WEKA specifically advertised on public pages.
- **Orchestration:** K8s-native (every cluster runs Kubernetes); Slurm-on-K8s via Slinky (SchedMD/NVIDIA layer). Self-serve "Instant GPU Clusters" launched Sep 2025.
- **Reputation:** Strong for inference because of dual-mode (serverless per-token AND dedicated hourly); breakeven for moving from serverless to dedicated is ~130k tokens/min (~2xH100) or >50% utilization.
- **Fit for monthly inference:** Best pricing transparency for monthly. 91-180 day reservation at $4.55/hr H200 = ~$3,320/GPU/mo. Min reservation is 6 days. GB200/GB300 NVL72 only via sales.

### Nebius AI Cloud — Gold (ClusterMAX 2.0)
- **Tier rating:** Gold, alongside Oracle and Azure.
- **Pricing:** Lowest published on-demand among Gold-tier neoclouds.
  - H100 HGX: $2.95 on-demand / $1.25 preemptible
  - H200 HGX: $3.50 on-demand / $1.45 preemptible / **$2.30 at 3+ mo commitment (35% off)**
  - B200 HGX: $5.50 on-demand / $2.90 preemptible. Up to 20% off via Blackwell pre-order.
  - GB200 NVL72: pre-order
- **Networking:** Custom + Quantum-2 InfiniBand on HGX nodes; not publicly speced as 3,200 Gbps but consistent w/ HGX standard.
- **Storage:** Premium tier powered by **WEKA** (announced Jun 2025); also offers **VAST** integration (Q2 2025). Up to 1 TB/s read throughput for shared FS; 2 GB/s per GPU object storage.
- **Orchestration:** Managed Kubernetes (MK8s) + **Soperator** (open-source K8s operator for Slurm — Nebius authored). USDC payment, $25 minimum first payment, suitable for self-serve onboarding.
- **Reputation:** Listed at semianalysis as one of the top providers in the rest-of-world tier; backed by ex-Yandex team and substantial Nvidia GPU supply contracts.
- **Fit for monthly inference:** **Best price-performance for self-serve monthly H200 on a Tier-1 cloud.** $2.30/hr × 730 = $1,679/GPU/mo for 3+mo H200 commit. Preemptible $1.45 ($1,058/mo) is unbeatable for fault-tolerant batch/inference replicas. WEKA tier is a real differentiator.

### Cudo Compute — outside ClusterMAX top tiers but enterprise-positioned
- **Tier rating:** Not in Platinum/Gold/Silver; positions itself as "enterprise-grade" via NVIDIA Cloud Partner status, ISO 27001 facilities in NA/EU/UK/MENA.
- **Pricing:** Less transparent. H100 cited at ~$1.80-$2.45/GPU-hr; 16-GPU cluster $30-50/hr. 1-yr or 3-yr commitment saves up to 57%. H200 not in public price list — sales-only.
- **Fleet:** H100, H200, A100 80GB, L40S, A800. B200/GB200 not publicly advertised.
- **Networking:** Multi-tier (varies by site); not publicly speced at 3,200 Gbps.
- **Storage / Orchestration:** Both on-demand and reserved clusters; less mature K8s/Slurm story than the Gold-tier set.
- **Reputation:** 24/7 monitoring, NVIDIA escalation paths, 40,000+ GPUs operated historically by team. Reviews are sparse — limited independent enterprise references.
- **Fit for monthly inference:** Workable for budget-conscious, simpler inference workloads at smaller scale. Not where I'd run an Anthropic-scale endpoint. Best leverage is the 1-3 year commitment discount if you have predictable demand and want price floor.

### Civo — sovereign-cloud focus, mid-market
- **Tier rating:** Not in ClusterMAX top tiers.
- **Pricing:** H100 in the $2-3/GPU-hr band; L40S $1.29 on-demand → $0.89 at 36-mo commit. H200 listed at getdeploying among 31 providers.
- **Fleet:** A100, H100, H200, B200 (per Civo 2026 guide); L40S in volume.
- **Networking:** Not publicly headline InfiniBand specs; positioned as K8s-native rather than HPC-fabric.
- **Storage / Orchestration:** **Kubernetes-native is the unique sell** — CivoStack Enterprise + FlexCore appliance, Kubeflow-as-a-Service, identical platform between public cloud and on-prem ("FlexCore in under 2 hours"). Strong for sovereign + portable deployment.
- **Reputation:** Mid-market, dev-friendly; less referenced for >100 GPU jobs.
- **Fit for monthly inference:** Niche — best when you want portable K8s deployment, sovereignty/UK-EU residency, or hybrid public/private. Not the choice for max-perf B200/GB200 multi-rack inference.

### Oracle Cloud Infrastructure (OCI) — Gold (ClusterMAX 2.0), bare-metal champion
- **Tier rating:** Gold. Real hyperscaler reach + cluster networking that competes with neoclouds.
- **Pricing:** $10/GPU-hr list for both BM.GPU.H100.8 and BM.GPU.H200.8 ⇒ $80/hr per 8-GPU bare-metal node. NOT the "cheap" rep many think — list price is high; the discount comes from Universal Cloud Credits (UCC):
  - 1-year UCC: **15–45% off** based on commit size / negotiation
  - 3+ year UCC, $5M+: **25–60% off**
  - Free first 10 TB egress; **RDMA + block + network bandwidth are not charged extra** (real cost differentiator vs AWS/GCP).
- **Fleet:** A100, L40S, H100 (BM + VMs), H200 (BM.GPU.H200.8 + supercluster up to 65,536), GB200 NVL72 supercluster (GA, scaling to 131,072 GPUs, "zettascale"). 660 MW expansion announced early 2025; Stargate Abilene 4.5 GW under 15-yr lease for OpenAI ($30B/yr).
- **Networking:** **RoCEv2 (RDMA over Converged Ethernet)** with NVIDIA ConnectX-7 NICs — 3,200 Gbps cluster, 400 Gbps front-end, 1.7 µs latency on 64B packets. GB200 NVL72 clusters use NVIDIA Quantum-2 InfiniBand switches.
- **Storage:** File Storage with Lustre, NVMe SSDs, block + object. Block + bandwidth bundled into compute price.
- **Orchestration:** OKE (Oracle K8s), Slurm-on-OCI well documented. Not as Kubernetes-pure as CoreWeave but strong HPC integration.
- **Reputation:** Has been the "secret cheap" hyperscaler for bare-metal GPU compute since 2023. OpenAI Stargate's primary infra partner ($30B/yr deal; 4.5 GW). Concerns from April 2026 Register article about support quality declining w/ AI obsession and job cuts.
- **Fit for monthly inference:** Best for **large committed inference workloads ($1M+/yr) where you negotiate UCC**. List price is uncompetitive; committed-use price is excellent and pairs with no-egress-charge model. Bare-metal H200 means no hypervisor overhead → real inference latency benefit.

### Yotta Shakti Cloud (India) — sovereign/regional, NVIDIA Reference Platform NCP
- **Tier rating:** Not in ClusterMAX (regional). One of only 5 Reference Platform NVIDIA Cloud Partners globally; first NCP in APAC.
- **Pricing (publicly listed — rare):**
  - H100 AI Workspace VM dedicated monthly: ₹192,113 (1x), ₹392,959 (2x), ₹785,919 (4x). At ₹83.5/USD ≈ $2,300/$4,710/$9,420 per month respectively for full H100 dedicated VMs.
  - H100 bare metal: ₹351/GPU/hr (~$4.20/hr) on 8x HGX H100 (640 GB).
  - SLURM cluster: ₹386/GPU/hr (~$4.62/hr).
  - Kubernetes cluster: ₹404/GPU/hr (~$4.84/hr).
  - AI Lab slices: ₹27,000/mo (10 GB) → ₹1,504,000/mo (8x80 GB).
  - **IndiaAI Mission discounted rate (government scheme):** ₹115.85/hr (standard) and ₹150/hr (high-precision) — Yotta agreed 15% L1 match.
  - Storage: ₹9.9/GB/mo high-speed parallel; ₹3.6/GB/mo object; ₹9.5/GB/mo block tier 1.
  - **InfiniBand interconnect line item: ₹16,060/GPU** (one-time / monthly per config).
- **Fleet:** 16,000+ H100; ~20,000 Blackwell Ultra (B200 + GB200) being deployed via Gorilla Technology ($500M, 5-yr). 640 HGX B200 servers being installed.
- **Networking:** InfiniBand (specs not publicly speced at Gbps); deployed at Navi Mumbai + Greater Noida DCs, Tier IV.
- **Storage / Orchestration:** Full-stack platform (bare-metal, VM, serverless GPU, AI lab, model endpoints); SLURM + K8s offered as productized clusters with explicit per-hour rate.
- **Reputation:** India sovereign AI flagship; powers India AI Mission; on NVIDIA DGX Cloud Lepton. Strong for India data-residency mandates.
- **Fit for monthly inference:** Only relevant if you need APAC/India data residency or are an IndiaAI Mission beneficiary (then unbeatable rates). H100 dedicated VM monthly is ~$2,300 which is competitive with US neoclouds. B200 supply is real and growing.

---

## 3. ClusterMAX 2.0 Quick Reference (SemiAnalysis, 2026)

| Tier | Providers |
|---|---|
| **Platinum** | CoreWeave (only) |
| **Gold** | Nebius, Oracle, Azure, Crusoe, Fluidstack, Together AI (boundary), LeptonAI |
| **Silver** | Google Cloud, AWS, Together AI (boundary), Lambda |
| **Bronze** | DataCrunch, TensorWave, others |

This is the most authoritative third-party benchmark for fabric quality, multi-node NCCL throughput, storage performance, and inference fit.

---

## 4. Fit-for-Monthly-Inference Ranking (sustained workload, May 2026)

Assuming the workload is sustained inference on H200 or B200, 1-3 month commitment, ~50-200 GPU footprint, needing solid networking and storage:

1. **CoreWeave** — best stack (LOTA, K8s, VAST, Quantum-2, GB300 first to GA). Premium price. Capacity contested.
2. **Nebius** — best $/performance for self-serve monthly H200 ($2.30/hr 3+mo = $1,679/GPU/mo). WEKA + VAST. Soperator. Lowest published rate that still has Gold-tier fabric.
3. **Together AI** — best **transparent** reserved pricing in the set. 91-180 day tiering published. Dual serverless + dedicated mode. K8s-native + Slinky.
4. **Crusoe** — solid Gold-tier. H200 $4.29 on-demand; reserved by sales. Tight capacity due to Stargate priority. Strongest renewables story.
5. **Oracle OCI** — winner if you can negotiate a 1-3 yr UCC ($1M+). RoCEv2 fabric, bare-metal, no egress, no separate RDMA charge. Awful list price; great net price for big committed deals.
6. **Yotta Shakti** — only choice for India residency; pricing actually competitive for H100 dedicated VM ($2,300/mo) and IndiaAI-subsidized rates are extreme.
7. **Cudo Compute** — workable for budget multi-month; less mature fabric story; H200 sales-only.
8. **Civo** — K8s-portable sovereign cloud — niche fit, not raw inference performance king.

---

## 5. Contract Flexibility / Minimum Commits Summary

| Provider | Smallest reserved unit | Self-serve monthly? | Sales required for B200? |
|---|---|---|---|
| CoreWeave | Capacity Plan / Reservation contract (custom) | No — sales-driven | Yes |
| Crusoe | Reserved contract (custom) | On-demand only self-serve | Yes |
| Together AI | **6 days** minimum reserved | **Yes — 7d, 31d, 91d tiers self-serve** | No (B200 on tiers) |
| Nebius | Multi-month commitment (3+ mo cited) | Largely yes; sales for big | Pre-order |
| Cudo Compute | 1-yr / 3-yr commits cited | Limited — sales for H200 | Yes |
| Civo | Up to 36-mo commits | Self-serve K8s | Sales |
| Oracle OCI | UCC 1-yr or 3-yr | On-demand yes; commits via sales | Yes |
| Yotta Shakti | Monthly VM publicly listed | Yes (monthly tier published) | Sales for B200 |

---

## 6. Sources (with dates where visible)

- [CoreWeave Classic Pricing](https://www.coreweave.com/pricing/classic) — accessed May 2026
- [CoreWeave Capacity Plans](https://www.coreweave.com/coreweave-capacity-plans) — announced Mar 10, 2026
- [CoreWeave HGX H100/H200 product page](https://www.coreweave.com/products/hgx-h100-h200)
- [CoreWeave – ClusterMAX Platinum announcement](https://www.coreweave.com/news/coreweave-achieves-semianalysis-platinum-clustermax-tm-rating-for-the-second-consecutive-ranking-remaining-the-industrys-sole-platinum-provider)
- [CoreWeave Expands Agreement with OpenAI by up to $6.5B](https://www.coreweave.com/news/coreweave-expands-agreement-with-openai-by-up-to-6-5b) — Sep 2025
- [CNBC: OpenAI to pay CoreWeave $11.9B over 5 years](https://www.cnbc.com/2025/03/10/openai-to-pay-coreweave-11point9-billion-over-five-years-for-ai-tech.html) — Mar 10, 2025
- [BlocksAndFiles: VAST $1.17B CoreWeave deal](https://blocksandfiles.com/2025/11/06/vasts-1-17-billion-coreweave-deal/) — Nov 2025
- [CoreWeave Inference Benchmarks vs. generalized cloud](https://www.coreweave.com/blog/inference-benchmarks-coreweaves-versus-popular-generalized-cloud)
- [CoreWeave MLPerf Inference v6.0 results](https://www.coreweave.com/news/coreweave-delivers-leading-inference-performance-in-mlperf-r-benchmark) — Apr 2026
- [HyperFRAME Research Q1 2026 review](https://hyperframeresearch.com/2026/05/11/coreweave-reaches-a-new-scale-threshold-but-can-the-ai-neocloud-sustain-long-tail-demand/) — May 11 2026
- [Crusoe Cloud Pricing](https://www.crusoe.ai/cloud/pricing) — accessed May 2026
- [Crusoe Cloud H200 page](https://www.crusoe.ai/cloud/gpus/nvidia-h200)
- [DCD: Oracle/OpenAI drop Abilene expansion; Meta/Crusoe](https://www.datacenterdynamics.com/en/news/oracleopenai-drop-plans-to-expand-flagship-abilene-stargate-site-meta-in-talks-to-pick-up-crusoe-capacity-with-nvidias-help/) — Mar 2026
- [Crusoe customer stories / reputation](https://www.crusoe.ai/resources/customers)
- [eesel: Crusoe AI review](https://www.eesel.ai/blog/crusoe-ai-review)
- [Together AI Pricing](https://www.together.ai/pricing) — accessed May 2026
- [Together AI GPU Clusters](https://www.together.ai/gpu-clusters)
- [Together AI: on-demand dedicated endpoints](https://www.together.ai/blog/on-demand-dedicated-endpoints)
- [Together AI multi-tenant cluster design](https://www.together.ai/blog/multi-tenant-gpu-cluster-design-for-ai-native-teams)
- [Together AI self-service launch — DCD](https://www.datacenterdynamics.com/en/news/together-ai-launches-self-service-ai-clusters/) — Sep 2025
- [Nebius Prices](https://nebius.com/prices) — accessed May 2026
- [Nebius H200 page](https://nebius.com/h200)
- [Nebius Blackwell pre-order](https://nebius.com/blackwell-pre-order)
- [Nebius Q2 2025 updates (WEKA, VAST integrations)](https://nebius.com/blog/posts/q2-2025-nebius-ai-cloud-updates)
- [WEKA × Nebius partnership PR](https://www.weka.io/company/weka-newsroom/press-releases/weka-and-nebius-partner-to-catalyze-ai-innovation-with-ultra-high-performance-cloud-infrastructure-solution/) — Jun 2025
- [Nebius Soperator (Slurm-on-K8s)](https://github.com/nebius/soperator)
- [CUDO Compute pricing](https://www.cudocompute.com/pricing)
- [CUDO Compute economics blog](https://www.cudocompute.com/blog/what-does-it-cost-to-rent-cloud-gpus)
- [CUDO Compute enterprise](https://www.cudocompute.com/enterprise)
- [Civo 2026 on-demand A100/H100/B200 guide](https://www.civo.com/blog/on-demand-nvidia-a100-h100-b200-cloud-instances)
- [Civo L40S product page](https://www.civo.com/cloud-gpu/nvidia-l40s-gpu)
- [Oracle OCI GPU](https://www.oracle.com/cloud/compute/gpu/)
- [Oracle blog: OCI Supercluster H200 GA](https://blogs.oracle.com/cloud-infrastructure/now-ga-largest-ai-supercomputer-oci-nvidia-h200)
- [Oracle blog: GB200 / Blackwell supercluster](https://blogs.oracle.com/cloud-infrastructure/supercluster-nvidia-blackwell-dedicated-alloy)
- [Oracle blog: RoCE + ConnectX](https://blogs.oracle.com/cloud-infrastructure/oci-accelerates-hpc-ai-db-roce-nvidia-connectx)
- [UpperEdge: Oracle UCC discounting basics](https://upperedge.com/oracle/oracle-universal-cloud-credits-ucc-licensing-and-discounting-basics/)
- [OracleLicensingExperts: OCI Universal Credits 2026 net pricing](https://oraclelicensingexperts.com/blog/oracle-oci-universal-credits-net-pricing/)
- [Register: Oracle AI obsession, support concerns](https://www.theregister.com/2026/04/13/oracle_job_losses_ai_impact_support/) — Apr 13 2026
- [Yotta Shakti Cloud pricing](https://shakticloud.ai/pricing/) — accessed May 2026
- [NVIDIA case study: Yotta Shakti sovereign AI](https://www.nvidia.com/en-in/customer-stories/yotta-built-india-sovereign-ai-infrastructure-shakti-cloud/)
- [CRN Asia: Gorilla deploys 5,000 Blackwell GPUs for Yotta](https://www.crnasia.com/india/news/2026/gorilla-to-deploy-over-5-000-gpus-for-yotta-in-500-million-india-ai-infrastructure-deal)
- [SemiAnalysis ClusterMAX 2.0](https://newsletter.semianalysis.com/p/clustermax-20-the-industry-standard)
- [SemiAnalysis ClusterMAX 1.0](https://newsletter.semianalysis.com/p/the-gpu-cloud-clustermax-rating-system-how-to-rent-gpus)
- [Lyceum: CoreWeave vs Lambda](https://lyceum.technology/magazine/coreweave-vs-lambda-gpu-cloud/)
- [Northflank: Crusoe alternatives 2026](https://northflank.com/blog/crusoe-alternatives)
- [Thundercompute: CoreWeave pricing guide May 2026](https://www.thundercompute.com/blog/coreweave-gpu-pricing-review)
- [Thundercompute: H200 price comparison May 2026](https://www.thundercompute.com/blog/nvidia-h200-pricing)
- [getdeploying: H100 / H200 / B200 / GB200 cloud pricing (43+ providers)](https://getdeploying.com/gpus/nvidia-h100)
- [Spheron: GPU Cloud Pricing 2026](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/)

---

## 7. Confidence & Gaps

### Confidence (High)
- **Publicly-listed on-demand rates** for H100/H200/B200 across CoreWeave (classic), Crusoe, Together AI, Nebius, Yotta. These are pulled directly from provider pages and corroborated by 3rd party trackers in May 2026.
- **Together AI's reserved tier ladder** (7-30d, 31-90d, 91-180d) — only provider in this set with fully public tiered reserved pricing for H100/H200/B200.
- **Nebius commitment pricing** $2.30/hr H200 at 3+ mo (35% off) and preemptible rates.
- **CoreWeave ClusterMAX Platinum status** (two years running per SemiAnalysis).
- **OCI list pricing** at $10/GPU-hr for H100/H200 BM shapes; RoCEv2 fabric specs.
- **Yotta Shakti monthly VM pricing in INR** (published).
- **Networking specs** (3,200 Gbps Quantum-2 InfiniBand on CoreWeave; RoCEv2 ConnectX-7 on OCI; InfiniBand HDR/NDR on Crusoe).
- **Customer/contract signals** — CoreWeave–OpenAI $22.4B aggregate; OCI–OpenAI $30B/yr; Stargate Abilene 4.5 GW status.

### Confidence (Medium)
- **CoreWeave reserved % discounts** ("up to 60%" / "15-30% for 12-mo") — published in derivative sources but actual contract pricing is sales-driven and non-disclosed.
- **OCI UCC discount bands** (15-45% 1-yr, 25-60% 3-yr) — from a 2026 published guide; actual discounts vary heavily by negotiation, customer size, and competitive context.
- **Cudo Compute reputation** — fewer independent benchmarks; relies on the provider's own claims and limited reviews.
- **Civo H100 pricing** — quoted as "$2-3 range" rather than from a definitive public price card I could pull.
- **Yotta B200 / GB200 pricing** — fleet being deployed (640 HGX B200 servers via Gorilla) but no public price card yet; "sales-only".

### Gaps (need follow-up)
- **CoreWeave's actual 12-month reserved $/GPU-hr for H200 and B200** — not public; need a sales quote for round 2.
- **Crusoe B200/GB200 monthly reserved price points** — "contact sales" everywhere.
- **Nebius B200 long-term commit price** — pre-order discounts cited (20%) but stable 6-mo / 12-mo monthly equivalents not pinned down.
- **Yotta H200/B200 pricing** — not yet on public pricing page.
- **Cudo H200/B200 pricing** — public price card absent; needs sales touch.
- **OCI net price for H200 supercluster with 3-yr UCC at the $5M+ tier** — would materially change the OCI vs neocloud math; only available via Oracle account exec.
- **GH200** specifically: CoreWeave is the only provider in this set with a clear public on-demand SKU/price ($6.50). Crusoe, Together, Nebius, Cudo, Civo, OCI, Yotta do not publicly list GH200 — GH200 appears to be deprioritized fleet in favor of H200 and Blackwell across the Tier-1 specialty cloud set as of May 2026.
- **Inference-specific SLAs / latency commitments** (vs raw uptime) are rarely published; would need direct sales engagement for serious comparison.
- **Egress and ingress fees for inference workloads** — only OCI is unambiguous (10 TB free + bundled RDMA + no extra network charges). Need to verify each other provider's network billing model for a high-throughput inference endpoint.
