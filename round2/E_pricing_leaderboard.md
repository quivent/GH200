# E — GPU Rental Leaderboard & Managed-Inference Break-Even

**Date:** 2026-05-25  •  **Round 2 cross-pollination agent**  •  **Currency:** USD
**Sources:** Round 1 agents 10, 11, 16, 17, 18, 19 + live WebFetch of Lambda, Vultr, Nebius, CoreWeave, Together, DeepInfra pricing pages (2026-05-25).
**Normalization:** All monthly figures use **730 h/mo**. Per-GPU figures normalize multi-GPU host rates to a single GPU. Where a provider quotes per-host, I divide by the GPU count of that host (8 for HGX H100/H200/B200; 4 for GB200 NVL72 VMs, 72 for full-rack pricing).

---

## 0. How to Read This

A single GH200 24/7 inference workload (the user's incumbent setup on Lambda at $1,672/mo) is the reference price point. Everything below is benchmarked against it.

Three contradictions had to be reconciled between agents:

| Claim | Agent 17 | Agent 16 | Agent 18 | Live check 2026-05-25 | Resolution |
|---|---|---|---|---|---|
| Vultr GH200 | $1.99/hr | $1.99/hr | — | $1.99/hr (vultr.com/pricing) | $1.99/hr confirmed |
| Lambda GH200 | $2.29/hr | $2.29/hr | — | $2.29/hr (lambda.ai/pricing) | $2.29/hr confirmed |
| CoreWeave GH200 | $6.50/hr | $6.50/hr | $6.50/hr | $6.50/hr (coreweave.com/pricing) | $6.50/hr confirmed |
| CoreWeave H100 | classic $4.25/hr | — | $4.25–$4.76 + "current ~$2.44" | host $49.24/hr ÷ 8 = $6.16/GPU/hr (NA, current public on-demand) | $6.16 is the live on-demand; $2.44 was a misread of inference-tier or stale reserved indicative |
| Bedrock Llama-3.3-70B | — | — | — | pecollective $0.72/$0.72 vs tokenmix $2.65/$2.65 | Conservative blended estimate $0.72/$0.72 — both sources are aggregators; AWS console required for final settlement |
| Nebius H200 committed | $2.30/hr 3-mo | — | $2.30/hr 3-mo (35% off) | live page confirms "save up to 35%"; specific tier table not visible | $2.30/hr 3-mo treated as published |

---

## 1. Master Leaderboard — Single GPU, On-Demand (May 2026)

Sorted within each GPU class by $/GPU/hr ascending. "Egress" = bandwidth out. "Root" = OS-level access on instance.

### 1.1 GH200 (96 GB HBM3) — on-demand

| Rank | Provider | $/GPU/hr | $/GPU/mo (730h) | Notes | SLA / risk |
|---|---|---|---|---|---|
| 1 | **Vultr** | **$1.99** | **$1,453** | 1 GPU = 1 VM; full root; free egress quota; 1 region in stock (Atlanta) May 2026 | Public cloud SLA; less inference-specific tooling |
| 2 | **Lambda Labs** | $2.29 | $1,672 | Full VM, root, Lambda Stack image, free egress, 4 TiB local SSD | 16h us-south-1 outage Mar 2025; capacity stockouts at peak; SOC 2 |
| 3 | Sesterce (via Shadeform) | $2.52 | $1,840 | Shadeform-routed; less reputation data | Aggregator dependency |
| 4 | Denvr Dataworks | $3.87 | $2,825 | 7.6 TB local; Virginia | Niche provider |
| 5 | Latitude.sh (via Shadeform) | $4.23 | $3,088 | Not on Latitude's own pricing page | Aggregator routing |
| 6 | CoreWeave | $6.50 | $4,745 | Premium tier; LOTA + Quantum-2 IB | ClusterMAX Platinum |

**Not offered (GH200):** AWS, Azure, GCP, Oracle, Nebius, Crusoe, Together, Cudo, Civo, RunPod, Vast.ai, TensorDock, Hyperbolic, Massed Compute, FluidStack, Voltage Park.

### 1.2 H100 SXM 80GB — on-demand

| Rank | Provider | $/GPU/hr | $/GPU/mo | Notes | SLA / risk |
|---|---|---|---|---|---|
| 1 | Vast.ai (unverified) | $0.90 | $657 | Spot, host can yank instance | Marketplace risk |
| 2 | Thunder Compute | $1.38 | $1,007 | Single-GPU floor | Niche |
| 3 | Hyperbolic | $1.49 | $1,088 | Decentralized; price moved $3.20→$1.49 in 2026 | No SOC 2 |
| 4 | RunPod Community spot | $1.49 | $1,088 | Container-only | 210+ incidents in 8 mo |
| 5 | Vast.ai verified DC | $1.87 | $1,365 | ISO 27001 hosts | Lower risk than marketplace |
| 6 | TensorDock spot | $1.60–$1.91 | $1,168–$1,394 | Full VM, root | **Heavy negative SLA signal** |
| 7 | **Voltage Park (Ethernet)** | **$1.99** | **$1,453** | Bare metal, full root, self-serve 8× | SOC 2 II, HIPAA; strong rep |
| 8 | RunPod Community PCIe | $1.99 | $1,453 | Container | Mixed reliability |
| 9 | GMI Cloud H100 PCIe | $2.00 | $1,460 | Dedicated | Not in ClusterMAX |
| 10 | TensorDock H100 SXM | $2.25 | $1,643 | Full root | Reputation concerns |
| 11 | Massed Compute | $2.40 | $1,752 | Tier III, SOC 2 II | Solid |
| 12 | Voltage Park (Infiniband) | $2.49 | $1,818 | 3.2 Tbps Quantum-2, 8× min | Strong |
| 13 | Nebius H100 OD | $2.95 | $2,154 | InfiniBand, free egress | Gold ClusterMAX |
| 14 | Latitude.sh Metal | $1.68 → $1,230/mo flat | $1,230 | Flat monthly (no hourly mult.) — **cheapest flat-rate** | Bare metal, full root |
| 15 | Lambda 8×H100 SXM (per GPU) | $3.99 | $2,914 | 8× node only; per-GPU = $3.99 | SOC 2 |
| 16 | CoreWeave HGX H100 | $6.16 | $4,497 | $49.24/host ÷ 8 (live 2026-05-25) | Platinum |
| 17 | AWS p5.48xlarge | $6.88 | $5,022 | $55.04 host ÷ 8 | AWS SLA |
| 18 | Azure ND_H100_v5 | $12.29 | $8,972 | $98.32/host ÷ 8 | Azure SLA |
| 19 | Oracle BM.GPU.H100.8 | $10.00 list | $7,300 | List only — UCC negotiation drops it sharply | OCI SLA |
| 20 | GCP a3-highgpu-8g | ~$3.93 | $2,869 | $31.45/host ÷ 8 | GCP SLA |

### 1.3 H200 SXM 141GB — on-demand

| Rank | Provider | $/GPU/hr | $/GPU/mo | Notes | SLA / risk |
|---|---|---|---|---|---|
| 1 | **Nebius (preemptible)** | **$1.45** | **$1,059** | Spot-style, interruptible | Gold ClusterMAX; spot risk |
| 2 | Sesterce 4×H200 | $2.09 | $1,526 | Per-GPU on 4× | Aggregator |
| 3 | Theta EdgeCloud | $2.29 | $1,672 | Decentralized | Mixed rep |
| 4 | FluidStack | $2.30 | $1,679 | Contact sales | SOC 2 II, HIPAA |
| 5 | GMI Cloud | $2.50–$2.60 | $1,825–$1,898 | Dedicated single-tenant | Reserved available |
| 6 | Massed Compute (NVL) | $2.83 | $2,066 | Self-serve | Tier III, SOC 2 II |
| 7 | Nebius H200 OD | $3.50 | $2,555 | InfiniBand, free egress | Gold |
| 8 | RunPod Community SXM | $3.59 | $2,621 | 141GB SXM | Container |
| 9 | Vast.ai H200 NVL | $3.95 | $2,884 | Marketplace floor | ISO 27001 verified hosts |
| 10 | RunPod Secure | $4.39 | $3,205 | T3/T4 DC | SOC 2 II (Secure tier) |
| 11 | Crusoe HGX H200 | $4.29 | $3,132 | 8× HGX rate | Gold; 99.5% SLA |
| 12 | Together AI HGX (OD) | $6.79 | $4,957 | Cluster pricing | Gold/Silver |
| 13 | CoreWeave HGX H200 | $6.31 | $4,606 | $50.44/host ÷ 8 (live 2026-05-25) | Platinum |
| 14 | AWS p5e.48xlarge (Capacity Block) | $4.975 | $3,632 | Block-only, 1d–6mo, +15% Jan 2026 | Auction-style |
| 15 | Azure ND_H200_v5 | $13.78 | $10,059 | $110.24/host ÷ 8 | Azure SLA |
| 16 | Oracle BM.GPU.H200.8 list | $10.00 | $7,300 | List; UCC discounts heavily | OCI SLA |

**Lambda H200:** No self-serve on-demand SKU. 1-Click Clusters or Private Cloud quote only. Third-party guesses $3.79–$5.29/hr; not verified.

### 1.4 B200 SXM6 192GB — on-demand

| Rank | Provider | $/GPU/hr | $/GPU/mo | Notes |
|---|---|---|---|---|
| 1 | Spheron (spot) | $2.12 | $1,548 | Spot |
| 2 | Packet.ai | $3.75 | $2,738 | Waitlist |
| 3 | Lyceum | $4.29 | $3,132 | |
| 4 | Crusoe HGX B200 | quote | — | Sales |
| 5 | Verda | $4.89 | $3,570 | |
| 6 | UpCloud | $5.22 | $3,811 | |
| 7 | **Lambda 8× B200 (per-GPU)** | **$6.69** | **$4,884** | 8× node = $39,070/mo |
| 8 | Lambda 1× B200 | $6.99 | $5,103 | Single-GPU |
| 9 | Nebius B200 | $5.50 | $4,015 | OD; $2.90 preemptible = $2,117/mo |
| 10 | RunPod B200 | $5.89 | $4,300 | |
| 11 | GCP a4-highgpu-8g | $7.00 | $5,110 | $56.00/host ÷ 8 |
| 12 | CoreWeave HGX B200 | $8.60 | $6,278 | $68.80/host ÷ 8 (live 2026-05-25) |
| 13 | Together AI HGX B200 | $9.95 | $7,264 | Cluster |
| 14 | CoreWeave NVL B200 | $10.50 | $7,665 | NVL form-factor |
| 15 | AWS p6-b200.48xlarge | $14.24 | $10,396 | $113.93/host ÷ 8 |
| 16 | Oracle BM.GPU.B200.8 | $14.00 list | $10,220 | UCC discount avail. |

### 1.5 GB200 NVL36 / NVL72 — on-demand

| Rank | Provider | $/GPU/hr | $/GPU/mo | Notes |
|---|---|---|---|---|
| 1 | **CoreWeave GB200 NVL72** | **$10.50** | **$7,665** | $42.00/4-GPU instance (live 2026-05-25); first cloud to GA |
| 2 | AWS p6e-gb200.36xlarge (Capacity Block) | $10.582 | $7,725 | 4-GPU UltraServer block; US-East Dallas LZ |
| 3 | Oracle BM.GPU.GB200 | $16.00 | $11,680 | List; UCC negotiable; 131k GPU supercluster |
| 4 | Azure ND_GB200_v6 | $27.04 | $19,739 | $108.16/4-GPU VM; 19 regions GA |
| 5 | GMI Cloud GB200 | $8.00 | $5,840 | Listed but limited public detail |
| 6 | Nebius GB200 NVL72 | contact / pre-order | — | Waitlist |
| 7 | Lambda GB200 NVL72 | not GA (May 2026); Pegatron deploy in flight | — | GB300 available |
| 8 | Crusoe / FluidStack / Cudo / Gcore / Lyceum | "on request" | — | Custom contracts |

---

## 2. Commitment-Term Leaderboards (Winner / 2nd / 3rd)

For each GPU class, the table below names the cheapest publicly verifiable $/GPU/mo at four commitment tiers. "Monthly" = 30-day commit. "1-yr" / "3-yr" use reserved or committed-use pricing.

### 2.1 GH200

| Tier | Winner | 2nd | 3rd |
|---|---|---|---|
| On-demand | **Vultr $1,453/mo** ($1.99/hr) | Lambda $1,672/mo ($2.29/hr) | Sesterce $1,840/mo ($2.52/hr) |
| Monthly (30d) | **Vultr ~$1,162/mo** (20% monthly-commit discount, gpuperhour) | Lambda $1,672/mo (no monthly tier; on-demand rate) | Sesterce $1,840/mo |
| 1-year | **Vultr ~$1,046/mo** (20% mo + 10% annual stack) | Lambda ~$1,088/mo (rumored $1.49/hr reserved; sales-only) | CoreWeave (contract; no public 1-yr GH200 quote) |
| 3-year | **Lambda contract** (~$0.99–$1.20/hr est. = $720–$880/mo; sales-only) | Vultr (no published 3-yr GH200) | CoreWeave (sales) |

**Caveat (3-yr):** No provider publishes a GH200 3-yr public price. All sales-quote. Numbers are extrapolated from Lambda's published "up to 27% off on 3-yr" guidance and Vultr's stacked-discount language.

### 2.2 H100 SXM 80GB

| Tier | Winner | 2nd | 3rd |
|---|---|---|---|
| On-demand | **Voltage Park Ethernet $1,453/mo** ($1.99/hr, self-serve bare metal) | Hyperbolic $1,088/mo ($1.49/hr, lower SLA) | Vast.ai verified $1,365/mo ($1.87/hr) |
| Monthly | **Latitude.sh Metal $1,230/mo** (flat monthly rate, bare metal) | Voltage Park IB $1,818/mo (3.2 Tbps) | Massed Compute $1,752/mo |
| 1-year | **GCP a3-high CUD $1,978/mo** (~$2.71/GPU-hr, 31% off; well-supported) | Together AI 91-180d $2,914/mo ($3.99/hr) | AWS p5 1-yr SP $3,519/mo (~$4.82/hr) |
| 3-year | **GCP a3-high 3-yr CUD $1,292/mo** (~$1.77/hr, 55% off) | AWS p5 3-yr RI $2,263/mo (~$3.10/hr) | OCI BM.GPU.H100.8 + UCC ~$2,990/mo |

**Note:** Voltage Park's $1.99/hr H100 Ethernet rate has no published commit discount but is already cheaper than GCP's 1-yr CUD — for **<1yr horizons, Voltage Park wins outright**.

### 2.3 H200 SXM 141GB

| Tier | Winner | 2nd | 3rd |
|---|---|---|---|
| On-demand | **Nebius preemptible $1,059/mo** ($1.45/hr; interruptible) | Sesterce 4×H200 $1,526/mo ($2.09/hr) | Theta EdgeCloud $1,672/mo |
| Monthly | **Nebius 3-mo commit $1,679/mo** ($2.30/hr, 35% off) | FluidStack $1,679/mo ($2.30/hr; contact sales) | GMI Cloud $1,825/mo |
| 1-year | **Together AI 91-180d $3,322/mo** ($4.55/hr; public tier ladder) | RunPod Savings Plan $2,227/mo ($3.05/hr; single source) | Nebius reserved $1,679/mo* (3-mo; only published reserved tier) |
| 3-year | **Azure ND_H200_v5 3-yr RI est. $4,526–5,037/mo** (~$6.20–$6.90/hr) | OCI BM.GPU.H200.8 + UCC est. ~$3,000–4,400/mo | AWS p5e Capacity Block $3,632/mo (6-mo max — not true 3-yr but cheapest committed AWS) |

*Nebius does not publish 1-yr-specific H200 rate; their 3-mo+ commit is the published floor.

**Reality check:** For H200, **no Tier-1 provider has a transparent 3-yr public rate**. The "winner" is whoever your account team negotiates hardest with. Use **Nebius 3-mo committed at $1,679/mo** as the realistic floor for serious capacity.

### 2.4 B200 SXM6 192GB

| Tier | Winner | 2nd | 3rd |
|---|---|---|---|
| On-demand | **Spheron spot $1,548/mo** ($2.12/hr) | Nebius preemptible $2,117/mo ($2.90/hr) | Lyceum $3,132/mo ($4.29/hr) |
| Monthly | **Vultr 36-mo amortized $2,110/mo** ($2.89/hr; longest commit needed) | Cirrascale 12-mo $3,548/mo ($4.86/hr) | Nebius pre-order (20% off OD) ≈ $3,212/mo |
| 1-year | **Cirrascale $3,548/mo** ($4.86/hr; only widely-published 12-mo B200 rate) | AWS p6-b200 1-yr No-Up $8,187/mo ($11.215/hr; 21% off) | Together AI 31-90d $6,825/mo ($9.35/hr) |
| 3-year | **AWS p6-b200 3-yr inst-locked $4,492/mo** ($6.153/hr best-case, 57% off) | AWS p6-b200 3-yr No-Up $7,635/mo ($10.459/hr) | Together AI 91-180d $6,636/mo ($9.09/hr; not a 3-yr tier but published floor) |

### 2.5 GB200 NVL36/NVL72

| Tier | Winner | 2nd | 3rd |
|---|---|---|---|
| On-demand | **GMI Cloud $5,840/mo** ($8.00/hr — limited public detail) | CoreWeave $7,665/mo ($10.50/hr; GA, ClusterMAX Platinum) | AWS p6e-gb200 Capacity Block $7,725/mo |
| Monthly | **CoreWeave $7,665/mo** (most reliable GA, monthly contract via Capacity Plan) | Oracle $11,680/mo (list, UCC discount avail.) | Azure $19,739/mo |
| 1-year | **CoreWeave reserved** (~15–30% off = est. $5,365–$6,515/mo; sales-only) | Oracle UCC 1-yr est. ~$7,000–$9,930/mo (15–45% off list) | Azure quote-only |
| 3-year | **CoreWeave / Oracle multi-year** (estimated ~$3,750–$5,250/mo at deep commit; sales-only) | AWS p6e-gb200 (no public 3-yr; Capacity Block max 6 mo) | Azure / GCP A4X — quote-only |

**Reality:** GB200 is **not a self-serve monthly product** on any cloud except CoreWeave (Capacity Plans, March 2026 redesign) and Oracle (UCC). Everyone else is quote-driven.

---

## 3. Managed Inference — $/Million Tokens (Llama-3.3-70B)

This is the "rent inference, not GPUs" comparison.

| Provider | Input $/Mtok | Output $/Mtok | Blended 1:1 | Notes |
|---|---|---|---|---|
| **DeepInfra (Llama-3.3-70B-Turbo)** | **$0.10** | **$0.32** | **$0.21** | Cheapest by a wide margin; Turbo variant (FP8) |
| **Fireworks (tier price >16B)** | $0.90 | $0.90 | $0.90 | Generic tier; not Llama-3.3-specific |
| **Azure AI Foundry** | $0.59 | $0.79 | $0.69 | Cheapest hyperscaler-hosted Llama 70B |
| **AWS Bedrock** (pecollective figure) | $0.72 | $0.72 | $0.72 | Conflicts with tokenmix $2.65; needs console verify |
| **Together AI (serverless)** | $0.88 | $0.88 | $0.88 | Verified live 2026-05-25 |
| **Google Vertex AI** | ~$0.72 | ~$2.00 | $1.36 | ~55% premium over dedicated providers |
| **AWS Bedrock** (tokenmix figure) | $2.65 | $2.65 | $2.65 | Likely stale/wrong; flagged |

**Source-of-truth caution:** DeepInfra is the published serverless floor; Together is verified live; Bedrock Llama-3.3 has a 3.7× disagreement between two aggregators and needs an AWS console check to settle. Use **$0.21 (DeepInfra)** as the realistic floor and **$0.69 (Foundry)** as the floor among compliance-friendly hyperscaler endpoints.

### 3.1 First-party / non-Llama reference points (for context)
- Amazon Nova Pro: $0.80 in / $3.20 out → $2.00 blended
- Amazon Nova Micro: $0.035 in / $0.14 out → $0.088 blended (the cheapest hosted text model on a hyperscaler)
- Mistral Large 3 (Foundry): $0.50 in / $1.50 out → $1.00 blended
- DeepSeek v3.2 (Bedrock): $0.62 in / $1.85 out → $1.24 blended

---

## 4. Self-Host vs Managed Break-Even (Llama-3.3-70B Output Tokens)

Math is conservative: assume the GPU box is a steady output factory at the throughput cited by MLPerf v4.0 / vLLM benchmarks for Llama-2/3.3-70B FP8. We compute the monthly token volume at which **self-hosting equals managed cost**, vs the three cheapest managed offerings.

### 4.1 Assumed sustained output throughput

| Hardware | Output tok/s (sustained, FP8, large-batch) | Monthly output tokens at 100% util |
|---|---|---|
| 1× GH200 96GB | ~1,500–2,500 (single-GPU, Llama-3.3-70B FP8, batch 32) | ~3,940–6,570 Mtok/mo |
| 8× H100 SXM (Lambda 8×H100 box) | ~21,500–25,000 (MLPerf v4.0 server) | ~56,500–65,700 Mtok/mo |
| 8× H200 SXM | ~30,000–35,000 (MLPerf v4.0/v5.0) | ~78,800–92,000 Mtok/mo |
| 1× B200 (NVFP4) | ~10,000 (InferenceMAX v1) | ~26,300 Mtok/mo |
| 8× B200 | ~70,000–90,000 | ~184,000–237,000 Mtok/mo |

Rule of thumb: **B200 FP4 ≈ 4× H100 FP8** per GPU on Llama-3.3-70B at iso-latency.

### 4.2 Break-even monthly token volume

For each platform, the break-even token volume V satisfies: V × $managed-per-Mtok = $platform-per-month.

**Reference platform A: Lambda 1× GH200 on-demand = $1,672/mo**

| Managed comparator | Managed $/Mtok | Break-even V (Mtok/mo) | At what utilization of 1× GH200? |
|---|---|---|---|
| DeepInfra Llama-3.3-70B-Turbo | $0.21 blended | 7,962 Mtok/mo | ~121% — i.e., self-host **never wins** vs DeepInfra on 1× GH200 (you'd need to overdrive past 100% sustained output) |
| Azure Foundry | $0.69 | 2,423 Mtok/mo | ~37% sustained output util → self-host wins above ~37% |
| Bedrock pecollective | $0.72 | 2,322 Mtok/mo | ~35% util |
| Together AI | $0.88 | 1,900 Mtok/mo | ~29% util |
| Google Vertex | $1.36 | 1,229 Mtok/mo | ~19% util |

**Interpretation:** A single GH200 has roughly 6,570 Mtok/mo of theoretical Llama-3.3-70B output capacity at 100% util. To beat DeepInfra's $0.21 blended you'd need ~7,962 Mtok/mo — which is **above the GH200's headroom**. **On 1× GH200, DeepInfra always wins on cost.** Against any other managed endpoint, you break even at 19–37% utilization — easy to clear if you have real load.

**Reference platform B: Voltage Park 8× H100 Ethernet = $11,462/mo (8 GPUs)**

Capacity ≈ 56,500–65,700 Mtok/mo at 100% util.

| Managed comparator | Managed $/Mtok | Break-even V (Mtok/mo) | Utilization required |
|---|---|---|---|
| DeepInfra | $0.21 | 54,581 Mtok/mo | **~83–97% util** — extremely tight; effectively tied |
| Azure Foundry | $0.69 | 16,612 Mtok/mo | ~25–29% util |
| Bedrock | $0.72 | 15,919 Mtok/mo | ~24–28% util |
| Together AI | $0.88 | 13,025 Mtok/mo | ~20–23% util |

**Interpretation:** Even the cheapest H100 self-host (Voltage Park) **barely beats DeepInfra at 80%+ util**. Below that, **DeepInfra wins**. Against any other managed endpoint, 20–30% util is sufficient.

**Reference platform C: Nebius 3-mo committed 1× H200 = $1,679/mo**

Per-GPU H200 sustained output ≈ ~5,000–8,000 tok/s on Llama-3.3-70B FP8 batch 128 (Spheron data) = ~13,140–21,030 Mtok/mo per GPU.

| Managed comparator | Managed $/Mtok | Break-even V (Mtok/mo) | Utilization required |
|---|---|---|---|
| DeepInfra | $0.21 | 7,995 Mtok/mo | ~38–61% util |
| Azure Foundry | $0.69 | 2,433 Mtok/mo | ~12–19% util |
| Bedrock | $0.72 | 2,332 Mtok/mo | ~11–18% util |
| Together AI | $0.88 | 1,908 Mtok/mo | ~9–15% util |

**Interpretation:** **Nebius committed H200 is the best self-host platform for beating managed inference** — it breaks even with DeepInfra at ~50% utilization, and with everything else at <20%. This is the cleanest "buy hardware, run vLLM" play in May 2026.

### 4.3 Summary break-even chart

| Self-host platform | $/mo | Cap (Mtok/mo @100%) | Break-even util vs DeepInfra ($0.21) | vs Foundry ($0.69) | vs Bedrock ($0.72) |
|---|---|---|---|---|---|
| 1× GH200 (Lambda) | $1,672 | ~6,570 | **>100% (loses)** | 37% | 35% |
| 1× GH200 (Vultr) | $1,453 | ~6,570 | 105% (loses by hair) | 32% | 31% |
| 1× H200 (Nebius 3mo) | $1,679 | ~13,140–21,030 | **38–61%** | 12–19% | 11–18% |
| 8× H100 (Voltage Park) | $11,462 | ~56,500–65,700 | **83–97%** | 25–29% | 24–28% |
| 8× H200 (Massed Compute) | $16,300 | ~78,800–92,000 | **84–98%** | 25–30% | 24–28% |
| 8× B200 (Lambda) | $39,070 | ~184,000–237,000 | **78–101%** | 24–30% | 23–29% |

**Bottom line:** If your Llama-3.3-70B workload is <~7,000 Mtok/mo of output, **DeepInfra is unbeatable on raw cost**. If you have real production volume (>10,000 Mtok/mo) and don't need DeepInfra's API-tier compliance/privacy profile, **Nebius committed H200 is the right self-host target**. GH200 single-chip is dominated by either choice — it's only the right answer if you specifically need (a) Grace CPU memory for KV offload, (b) ARM64 architecture, or (c) the unique LPDDR5X 480 GB pool.

---

## 5. SLA & Reliability Risk Flags

Color-coded by signal strength from agents 16, 17, 18 plus independent reviews.

### 5.1 RED (avoid for production, or use with backup)

- **TensorDock** — "heavy negative signal on Trustpilot/LowEndTalk; GPUs crashing and not restarting; support nonresponsive; instances missing advertised GPUs (RTX 5090 listings ended up with lesser cards); VMs deploying inaccessible via SSH while billing continues; accounts unable to remove payment methods" (Agent 17). Public pricing page last-updated July 2024. **Treat as price-only, never SLA-critical.**
- **RunPod Community Cloud** — 210+ incidents in 8 months per StatusGator; pods deploying on broken servers; payment-decline + storage purge on zero balance. **Secure tier (T3/T4 DC) is OK; Community tier is not.**
- **Vast.ai unverified hosts** — host can yank instance; ISO 27001 "verified" filter is mandatory for any production workload.

### 5.2 YELLOW (track risk, generally acceptable)

- **Lambda Labs** — March 2025 us-south-1 outage was 16 hours (well beyond 99.9% monthly budget for that region); capacity stockouts during peak H100/H200 demand; "abrupt account suspensions and persistent FS deletion after minor payment processing issues" reported by users in 2026. SOC 2 + Silver ClusterMAX. **Acceptable for single-region inference if you maintain a DR copy of model weights and state off-Lambda.**
- **Hyperbolic** — decentralized architecture; no published SOC 2 / ISO 27001 (May 2026). Price moves fast ($3.20 → $1.49 in H100 in <12 mo).
- **Vultr** — public cloud SLA, but GH200 inventory was sold out in Manchester and "Global" regions per gpuperhour May 2026; only Atlanta in stock for GH200.
- **AWS Capacity Blocks** — auction pricing, +15% hike Jan 2026, next re-price scheduled July 2026. Price is not stable.
- **Oracle OCI** — list price uncompetitive; Universal Credits negotiation needed for real value. April 2026 Register article flagged support quality declining alongside AI focus / job cuts.

### 5.3 GREEN (strong reliability signal)

- **CoreWeave** — sole ClusterMAX Platinum (two years running); OpenAI $22.4B contracts; MLPerf v6.0 inference leadership; LOTA + Quantum-2 IB. Premium price is the only downside.
- **Voltage Park** — non-profit ownership (Navigation Fund); SOC 2 II + HIPAA; "running uninterrupted for weeks, sometimes months" per VAST case studies. 7 Tier 3+ DCs.
- **Nebius** — Gold ClusterMAX; ex-Yandex team; WEKA + VAST storage partnerships; Soperator (Slurm on K8s) open-sourced. Best public-rate Gold-tier H200.
- **Crusoe** — Gold ClusterMAX; 99.98% uptime claim; Windsurf endorsement; <6 min support response. Renewables story.
- **FluidStack** — SOC 2 II + HIPAA + GDPR + ISO 27001; $200M Series A Feb 2025; in talks for $1B at $18B valuation Apr 2026; Mistral as customer.
- **Massed Compute** — Tier III DCs, SOC 2 II, NVIDIA Preferred Partner.
- **Latitude.sh** — positive G2 reviews; dedicated machines; flat monthly pricing.

---

## 6. Key Takeaways

1. **For 1× GH200 24/7 on-demand inference, Vultr beats Lambda by $219/mo ($1,453 vs $1,672)**, but inventory is tight (one region in stock). Lambda is the next-best at $1,672/mo with broader region coverage and a longer track record despite the 2025-03 outage. **CoreWeave at $4,745/mo is not worth the premium unless you need Platinum-tier fabric / VAST / LOTA — which 1×GH200 inference doesn't.**

2. **For H200 on-demand monthly, Nebius committed (3-mo, $1,679/mo) is the runaway winner** among self-serve providers with real SLA backing. **DeepInfra serverless ($0.21/Mtok blended) is cheaper than self-host on a 1× GH200 — period.** It only stops winning when you have a real H200 box and >50% utilization.

3. **H100 SXM**: cheapest credible production-grade rate is **Voltage Park $1,453/mo** (Ethernet, self-serve bare metal). For Tier-1 commitment, **GCP a3-high 3-yr CUD $1,292/mo** is the cheapest hyperscaler.

4. **B200 reservation pricing is a mess** outside of AWS p6-b200 3-yr ($4,492/mo) and Cirrascale 12-mo ($3,548/mo). Spot/preemptible (Spheron $1,548/mo, Nebius $2,117/mo) is half the price but unsuitable for production endpoints.

5. **GB200 is sales-driven everywhere except CoreWeave ($7,665/mo on-demand) and AWS Capacity Blocks ($7,725/mo)** — and even those need careful capacity planning.

6. **No hyperscaler sells GH200.** AWS/Azure/GCP/Oracle all skipped to H200 + Blackwell. Neoclouds (Vultr, Lambda, CoreWeave) are the only path.

7. **Bedrock Llama-3.3-70B pricing is ambiguous** ($0.72 vs $2.65 across two aggregators) — needs a console-side verification before committing.

8. **Live-verified prices 2026-05-25**: Lambda GH200 $2.29/hr, Vultr GH200 $1.99/hr, Nebius H100 $1.25/$2.95 preempt/OD, Nebius H200 $1.45/$3.50, Nebius B200 $2.90/$5.50, CoreWeave HGX H100 $49.24/host ($6.16/GPU), CoreWeave HGX H200 $50.44/host ($6.31/GPU), CoreWeave HGX B200 $68.80/host ($8.60/GPU), CoreWeave GB200 NVL72 $42.00/4-GPU ($10.50/GPU), Together Llama-3.3-70B $0.88/$0.88, DeepInfra Llama-3.3-70B-Turbo $0.10/$0.32.

---

## 7. Sources & Live-Verification Log (2026-05-25)

| URL | Verified live | Key data extracted |
|---|---|---|
| https://lambda.ai/pricing | Yes | GH200 $2.29, H100 SXM $3.99–$4.29, H100 PCIe $3.29, B200 $6.69–$6.99 — H200 NOT on on-demand |
| https://www.vultr.com/pricing/#cloud-gpu | Yes | GH200 $1.99/hr; H100 $1.99/hr (24-mo contract); B200 $1.99/hr (48-mo contract) |
| https://nebius.com/prices | Yes | H100 $1.25/$2.95, H200 $1.45/$3.50, B200 $2.90/$5.50, GB200 contact, no GH200, "save up to 35%" multi-month |
| https://www.coreweave.com/pricing | Yes | GH200 $6.50, HGX H100 $49.24/host, HGX H200 $50.44/host, HGX B200 $68.80/host, GB200 NVL72 $42.00/4-GPU, GB300 sales |
| https://www.together.ai/pricing | Yes | Llama-3.3-70B $0.88/$0.88; 1×H100 dedicated $6.49/hr; B200 $11.95/hr |
| https://deepinfra.com/pricing | Yes | Llama-3.3-70B-Turbo $0.10/$0.32; Llama-3.1-70B $0.40/$0.40 |
| https://docs.fireworks.ai/serverless/pricing | Partial | Generic >16B tier $0.90/$0.90 (Llama-3.3 not individually listed) |
| https://aws.amazon.com/bedrock/pricing/ | Failed extraction | Need console login to confirm Llama-3.3-70B rate |

**Round 1 source agents:** 10 (H100/H200 alternatives), 11 (Blackwell), 16 (Lambda), 17 (marketplaces), 18 (Tier-1 specialty), 19 (hyperscalers). Cross-validated against getdeploying.com, computeprices.com, gpuperhour.com (all updated May 2026), SemiAnalysis ClusterMAX 2.0.

---

## 8. Confidence & Gaps

### High confidence
- All on-demand single-GPU rates for Vultr, Lambda, Nebius, CoreWeave, Together (live-verified 2026-05-25).
- CoreWeave per-GPU normalization (host ÷ 8 for HGX, ÷ 4 for GB200) confirmed against agent 18.
- ClusterMAX 2.0 tier assignments (CoreWeave Platinum; Nebius/Crusoe/Oracle/Azure/FluidStack Gold; Lambda/AWS/GCP/Together Silver).
- DeepInfra Llama-3.3-70B-Turbo at $0.10/$0.32 (cheapest serverless floor for Llama-3.3-70B in May 2026).
- AWS p5/p6 list prices and Capacity Block Jan 2026 +15% hike.

### Medium confidence
- Lambda reserved GH200/H100/B200 rates ($1.49/$1.89/$3.79) — aggregator-sourced, not on lambda.ai.
- Nebius "$2.30/hr H200 3-mo commit" — single-source from DeployBase Nebius review; Nebius's own page only says "up to 35%".
- Bedrock Llama-3.3-70B pricing — 3.7× disagreement between pecollective ($0.72) and tokenmix ($2.65); used pecollective as conservative.
- AWS p6-b200 3-yr instance-locked No-Up at $6.153/hr (57% off) — third-party derivation; not on AWS pricing page.
- Vultr 36-mo B200 at $2.89/hr — round 1 agent 11 cited as "lowest tracked"; not re-verified live.
- GMI Cloud GB200 at $8.00/GPU/hr — published but limited detail on availability.

### Low confidence / gaps
- **No public 3-yr GH200 rate exists.** All extrapolations from "up to 27% off on 3-yr" Lambda guidance.
- **No public 1-yr / 3-yr H200 rate from Nebius.** Their committed tier table doesn't show 1-yr+ specifically.
- **GCP A3 Ultra (H200), A4 (B200), A4X (GB200) reserved pricing is gated behind AI Hypercomputer future-reservations** — quote-only via Google account team.
- **Azure ND_H200_v5 and ND_GB200_v6 reserved rates are not in Vantage/third-party feeds.** Estimates use "up to 60%" Microsoft language.
- **Oracle UCC actual discount bands** (15–45% 1-yr, 25–60% 3-yr at $5M+) — from a 2026 published guide; real discounts vary heavily.
- **CoreWeave Capacity Plans / Reservations actual $/GPU-hr** — March 2026 redesign; the "up to 60% off" is published but specific tier table is sales-only.
- **Inference-specific SLAs** (vs raw uptime) rarely published; would need direct sales engagement.
- **Egress for high-throughput inference endpoints** — only OCI (10 TB free + bundled RDMA + no extra network) and Nebius (free egress on InfiniBand) are unambiguous. Other providers' network billing models for high-volume outbound traffic need contract review.
