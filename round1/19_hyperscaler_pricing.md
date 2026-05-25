# Hyperscaler GPU Reserved Pricing + Managed Inference (Round 1, Agent #19)

**Date of snapshot:** 2026-05-25
**Scope:** AWS, Google Cloud, Azure, Oracle OCI — H100 / H200 / GH200 / B200 / GB200 monthly equivalents; managed inference $/Mtok for open-weights models on Bedrock, Vertex, Foundry.

All monthly figures use **730 hours/month** unless otherwise noted. "Per-GPU" figures = full host hourly rate / 8 GPUs (4 GPUs for GB200 ND-v6 / p6e-gb200 / a4x). Reserved prices use the cheapest payment option each cloud publishes (typically All-Upfront for AWS/Azure; resource-based CUD for GCP).

---

## 1. Effective Monthly $/GPU — Hyperscalers, May 2026

### 1a. AWS (EC2)

| Shape | GPU | On-Demand $/GPU-hr | 1-yr RI/SP $/GPU-hr | 3-yr RI/SP $/GPU-hr | Capacity Block $/GPU-hr | On-Demand $/GPU-mo | 3-yr RI $/GPU-mo |
|---|---|---|---|---|---|---|---|
| p5.48xlarge | 8× H100 80GB | $6.88 | ~$4.82 (Savings Plan) | ~$3.10 (3yr RI, instance-locked best case 57%) | $4.326 (US-East/West-Oregon) | $5,022 | $2,263 |
| p5e.48xlarge | 8× H200 141GB | ~$8.18 (derived from $39.799/hr Capacity Block, no public OD) | n/a (RI not published) | n/a | $4.975 (US-East) / $6.219 (US-W-N.Cal) | — | — |
| p5en.48xlarge | 8× H200 141GB Sapphire | ~$8.55 (derived) | n/a | n/a | $5.721 (US-East) / $5.201 (EU/APAC) | — | — |
| p6-b200.48xlarge | 8× B200 180GB | $14.242 | $11.215 (1-yr No-Up, 21% off) | $10.459 (3-yr No-Up, 27%) — best case **$6.153** (3-yr inst-locked, 57%) | $10.296 (US-East) / $12.870 (GovCloud) | $10,396 | $4,492 (inst-locked) |
| p6e-gb200.36xlarge | 4× GB200 (UltraServer block) | n/a OD | n/a | n/a | $10.582 (US-East Dallas LZ) | — | — |

Notes:
- p5/p5e/p5en/p6 **on-demand 3-yr RI hourly numbers above are 3rd-party derivations**; AWS does not publish standard RI rates for p5e/p5en/p6 — they're sold primarily via **Capacity Blocks for ML** (1 day to 6 months, dynamic supply/demand pricing). Capacity Blocks were raised ~15% in Jan 2026 (Ohio p5e went $34.608 → $39.799/hr; p5en $36.184 → $41.612/hr). Next scheduled re-price: **July 2026**.
- p5.48xlarge effective monthly: **On-Demand $39,629 → 3-yr RI ~$17,834** (full 8-GPU host).
- p6-b200 best published reserved tier (instance-family-locked 3-yr No-Upfront) is **$49.219/hr full host → ~$4,492 $/GPU-mo at 730h**.

### 1b. Google Cloud

| Shape | GPU | On-Demand $/GPU-hr | 1-yr CUD $/GPU-hr | 3-yr CUD $/GPU-hr | On-Demand $/GPU-mo | 3-yr CUD $/GPU-mo |
|---|---|---|---|---|---|---|
| a3-highgpu-8g | 8× H100 80GB | ~$3.93 ($31.45/hr host) | ~$2.71 (~31% off) | ~$1.77 (~55% off) | $2,869 | $1,292 |
| a3-megagpu-8g | 8× H100 Mega 80GB | ~$11.07 ($88.55/hr OD) — Vantage shows $9.46/hr base; tokenmix uses $85.61/hr 1-yr | $10.70/hr host = $1.34/GPU-hr (3% off vs OD) ⚠ likely 3rd-party error | $10.40/hr host = $1.30/GPU-hr (6% off) ⚠ | — | — |
| a3-ultragpu-8g | 8× H200 141GB | $12.266/hr host = **$1.533/GPU-hr** (third-party Vantage). Other source: ~$10.60/GPU-hr OD. Conflicting. | Reserved only via AI Hypercomputer **future reservations** — not publicly listed | Same — by negotiation | (~$1,119 if Vantage right) or (~$7,738 if higher #) | n/a published |
| a4-highgpu-8g | 8× B200 192GB | $56.00/hr host = **$7.00/GPU-hr** | by negotiation (AI Hypercomputer) | by negotiation | $5,110 | n/a |
| a4x-highgpu-4g | 4× GB200 Superchip | not publicly listed (reserve-only) | by negotiation | by negotiation | — | — |

Notes:
- **A3 Ultra (H200), A4 (B200), A4X (GB200)** all gate reserved capacity behind Google's **AI Hypercomputer future-reservations** workflow — no self-serve 1/3-yr CUD button. Standard CUD math (37% / 55%) does **not** apply.
- A3-mega "1-yr/3-yr" rates in the Vantage feed (3% / 6% off) appear to reflect only the sustained-use auto-discount, not a true commit — treat with caution.
- A3-high (H100) is the only Google GPU machine with reliable, published CUD pricing.

### 1c. Azure

| Shape | GPU | On-Demand $/GPU-hr | 1-yr RI $/GPU-hr | 3-yr RI $/GPU-hr | On-Demand $/GPU-mo | 3-yr RI $/GPU-mo |
|---|---|---|---|---|---|---|
| Standard_ND96isr_H100_v5 | 8× H100 80GB | $12.29 ($98.32/hr host) | $7.93 (~35% off) | $5.47 (~55% off) | $8,972 | $3,993 |
| Standard_ND96isr_H200_v5 | 8× H200 141GB | $13.78 ($110.24/hr host) | n/a (Vantage shows N/A; Microsoft says 1-3 yr RI available, ~30-50% off → est. ~$8.00–$9.65/GPU-hr 1-yr; ~$6.20–$6.90/GPU-hr 3-yr) | est. ~$6.20–$6.90 | $10,059 | est. $4,526–5,037 |
| Standard_ND128isr_GB200_v6 | 4× GB200 (2× Grace + 4× Blackwell per VM; 18 VMs/rack = NVL72) | not publicly listed | not publicly listed | not publicly listed | — | — |

Notes:
- Azure H200 reserved rates aren't published in third-party feeds; estimates use Microsoft's stated "up to 60%" reserved discount language.
- ND GB200-v6 is **GA but quote-only / reserve-only** as of March 2026 — no spot, no self-serve PAYG; capacity is sold via committed AI infrastructure deals.

### 1d. Oracle Cloud Infrastructure (OCI)

| Shape | GPU | On-Demand $/GPU-hr | Universal Credit / Commit discount | $/GPU-mo (OD) |
|---|---|---|---|---|
| BM.GPU.H100.8 | 8× H100 80GB | **$10.00** (host $80.00/hr) — Oracle posted rate. Third-party reports as low as **$4.10/hr/GPU** ($32.77/hr host = $23,922/mo) for committed/Universal-Credits buyers. | Annual Universal Credits 25-33% off; large commits negotiable | $7,300 (list) / $2,990 (commit) |
| BM.GPU.H200.8 | 8× H200 141GB | $10.00 | Same as H100 — Oracle held parity per Oct-2025 GA announcement | $7,300 (list) |
| BM.GPU.B200.8 | 8× B200 192GB | $14.00 (third-party listing) | commit discount available | $10,220 |
| BM.GPU.GB200 / NVL72 supercluster | GB200 Superchip | $16.00 per GPU | committed only | $11,680 |
| BM.GPU.A100-v2.8 | 8× A100 80GB | ~$4.00 | — | $2,920 |

Notes:
- OCI's headline "much cheaper than AWS" claim derives from **annual Universal Credit** discounting + bare-metal (no hypervisor overhead), not the list price. List H100 at $10/GPU-hr is **higher** than AWS p5 on-demand. With commit, easecloud.io reports BM.GPU.H100.8 effective at **$32.77/hr host ($23,922/mo)** vs AWS $39,629/mo on-demand and ~$29,533 on Azure — a 17-22% saving vs the others **only when you commit**.
- Oracle does not publish a 1-yr vs 3-yr RI table the way AWS/Azure do; commits are bundled into Universal Credit deals (typically annual, prepaid).

---

## 2. Cross-Cloud Summary — Cheapest Reserved $/GPU/Month (730 hr)

| GPU | Cheapest Hyperscaler (3-yr or best commit) | $/GPU-mo |
|---|---|---|
| **H100 80GB** | GCP a3-high 3-yr CUD | ~$1,292 |
| H100 80GB (runner-up) | Azure ND H100 v5 3-yr RI | ~$3,993 |
| H100 80GB (runner-up) | AWS p5 3-yr SP/RI | ~$2,263 |
| H100 80GB (runner-up) | OCI H100.8 w/ Universal Credits | ~$2,990 |
| **H200 141GB** | Azure ND H200 v5 est. 3-yr RI | ~$4,526–5,037 |
| H200 141GB | AWS p5e Capacity Block (US-East) | $4.975/hr = $3,632/mo, **6-mo max term** (not 1/3-yr) |
| H200 141GB | OCI H200.8 list | ~$7,300 |
| **B200 192GB** | AWS p6-b200 3-yr inst-family-locked No-Up | ~$4,492 |
| B200 192GB | GCP a4-high OD | ~$5,110 (no public reserved) |
| B200 192GB | OCI B200.8 list | ~$10,220 |
| **GB200 Superchip** | AWS p6e-gb200 Capacity Block | $10.582/hr = ~$7,725/mo, **block-only** |
| GB200 | Azure ND-GB200-v6 | quote-only |
| GB200 | GCP A4X | quote-only |
| GB200 | OCI GB200 supercluster | ~$11,680 |
| **GH200 Grace Hopper** | **No hyperscaler offers GH200 self-serve in 2026** — see "Confidence & Gaps" |

---

## 3. Managed Inference $/Mtok — "Rent inference, not GPUs"

### 3a. AWS Bedrock (open-weights + Nova; us-east-1, May 2026)

| Model | Input $/Mtok | Output $/Mtok | Notes |
|---|---|---|---|
| Llama 3.3 8B | $0.22 | $0.22 | Meta open-weights |
| Llama 3.3 70B | $0.72 (pecollective) / $2.65 (tokenmix) | $0.72 / $2.65 | **Source conflict** — pecollective shows $0.72 in/out flat; tokenmix shows $2.65. Recheck. |
| Llama 4 Scout (17B/109B MoE) | $0.17 | $0.66 (estimated) | 10M ctx |
| Llama 4 Maverick (17B/400B MoE) | $0.24 | $0.97 | 1M ctx |
| Mistral 7B | $0.15 | $0.20 | |
| Mistral Large 2 | $3.00 | $9.00 | |
| Mistral Large 3 | $0.50 | $1.50 | (newest, 2026) |
| Ministral 3B | $0.10 | $0.10 | |
| DeepSeek v3.2 | $0.62 | $1.85 | Standard tier |
| Amazon Nova Micro | $0.035 | $0.14 | 128K ctx, first-party |
| Amazon Nova Lite | $0.06 | $0.24 | 300K |
| Amazon Nova Pro | $0.80 | $3.20 | 300K |
| Amazon Nova Premier | $2.00 | $8.00 | 1M |

Discounts: **Batch inference = 50% off on-demand**. **Provisioned Throughput** (capacity-reserved endpoints) ~15-25% (1-mo) / 30-40% (6-mo) cheaper than on-demand for high-utilization. Cross-region inference adds **+10% surcharge**.

### 3b. Google Vertex AI Model Garden

| Model | Input $/Mtok | Output $/Mtok | Notes |
|---|---|---|---|
| Llama 3.3 70B | $1.36 blended (or ~$0.72/$2.00 split per artificialanalysis) | ~$2.00 | ~55% premium over dedicated inference providers |
| Mistral Large 2 | $2.00 | $6.00 | ~8% premium vs Mistral La Plateforme |
| Mistral Small | ~$0.20 | ~$0.60 | |

(Vertex's Gemini 1.5/2.x and Anthropic Claude are also on the same metered API but were excluded — only open-weights asked.)

### 3c. Azure AI Foundry (serverless API, "Models-as-a-Service" billed via Azure Marketplace)

| Model | Input $/Mtok | Output $/Mtok | Notes |
|---|---|---|---|
| Llama 3.3 70B | $0.59 | $0.79 | cheapest hyperscaler-hosted Llama 70B by a wide margin |
| Llama 3.3 8B | ~$0.18 | ~$0.24 | indicative |
| Mistral Large 3 (preview) | $0.50 | $1.50 | Apache 2.0, 41B active / 675B MoE |
| Phi-4-mini (Microsoft) | indicative | indicative | first-party |

Notes:
- All Foundry serverless-API rates are set by the model provider (Mistral, Meta) and billed through Azure Marketplace — **not** part of the Azure compute reservation discount tree.
- Azure also offers **Provisioned Throughput Units (PTU)** for committed-capacity endpoints — analogous to Bedrock Provisioned Throughput; not priced per token.

### 3d. Cross-cloud cheapest hosted-Llama-3.3-70B per Mtok

| Provider | Input | Output | Blended (1:1) |
|---|---|---|---|
| Azure AI Foundry | $0.59 | $0.79 | $0.69 |
| AWS Bedrock | $0.72 | $0.72 | $0.72 (per pecollective) — **needs reverify** |
| Google Vertex | ~$0.72 | ~$2.00 | $1.36 blended (per tokenmix) |
| **Reference: dedicated inference (Together, Fireworks, Groq)** | $0.60–0.88 | $0.60–0.88 | ~$0.7 |

Bedrock Nova Pro ($0.80/$3.20) and Mistral Large 3 ($0.50/$1.50) on Foundry are the two best in-class first-party + open-source offerings if cross-region data residency or Marketplace SaaS billing is required.

---

## 4. Break-Even Math: Managed Endpoint vs Rent the GPU

**Quick estimate for Llama 3.3 70B on H100:**

- One 8×H100 box (AWS p5 3-yr RI) ≈ $17,834/mo = $24.43/hr.
- vLLM/SGLang serving Llama 3.3 70B (FP8/bf16) on 8×H100 ≈ ~5,000–10,000 output tok/s aggregate (8×80GB allows full bf16 70B; throughput is empirical — see other agents).
- 10k tok/s × 3600s × 730h = **2.63 × 10¹⁰ output tokens / month** = 26,280 Mtok/mo.
- $17,834 / 26,280 Mtok = **$0.68 per Mtok output** at full saturation.

Cross-checks:
- Azure Foundry Llama 3.3 70B output = $0.79/Mtok — break-even when own-GPU throughput is ~22k Mtok/mo (~85% utilization).
- Bedrock pecollective rate $0.72 → break-even at ~92% utilization vs AWS 3-yr RI; **almost never wins** on cost if you can drive >80% util.
- Vertex $2.00/Mtok output → managed is ~3× the unit cost of self-hosting at 80% util.

**Take-away:** at ≥70% utilization, self-hosting on **3-yr RI hyperscaler GPUs** is competitive with managed endpoints; Azure Foundry is the only managed offering that's hard to beat below ~50% utilization. AWS Bedrock pricing is at-or-above self-host cost; Vertex is significantly more expensive than self-host.

---

## 5. Sources (all fetched 2026-05-25)

- AWS Capacity Blocks for ML pricing — https://aws.amazon.com/ec2/capacityblocks/pricing/
- AWS p5/p6 instance pages — https://aws.amazon.com/ec2/instance-types/p5/ , https://aws.amazon.com/ec2/instance-types/p6/
- AWS p5.48xlarge Vantage — https://instances.vantage.sh/aws/ec2/p5.48xlarge
- AWS p6-b200.48xlarge Vantage / DoiT — https://instances.vantage.sh/aws/ec2/p6-b200.48xlarge , https://compute.doit.com/spot/us-east-1/p6-b200.48xlarge
- AWS Capacity Block price hike Jan 2026 — https://www.theregister.com/2026/01/05/aws_price_increase/ , https://www.networkworld.com/article/4113150/aws-hikes-prices-for-ec2-capacity-blocks-amid-soaring-gpu-demand.html , https://www.datacenterdynamics.com/en/news/aws-quietly-increases-prices-for-h200-ec2-instances-by-15/
- AWS H100 pricing 2026 (Spheron) — https://www.spheron.network/blog/aws-h100-pricing-2026/
- AWS p6-b200 blog — https://aws.amazon.com/blogs/aws/new-amazon-ec2-p6-b200-instances-powered-by-nvidia-blackwell-gpus-to-accelerate-ai-innovations/
- GCP Accelerator-optimized VM pricing — https://cloud.google.com/products/compute/pricing/accelerator-optimized
- GCP GPU pricing — https://cloud.google.com/compute/gpus-pricing
- GCP a3-megagpu-8g Holori / CloudPrice / Vantage — https://calculator.holori.com/gcp/vm/a3-megagpu-8g , https://cloudprice.net/gcp/compute/instances/a3-megagpu-8g , https://instances.vantage.sh/gcp/a3-megagpu-8g
- GCP a3-ultragpu-8g CloudPrice — https://cloudprice.net/gcp/compute/instances/a3-ultragpu-8g
- GCP a4-highgpu-8g CloudPrice — https://cloudprice.net/gcp/compute/instances/a4-highgpu-8g
- GCP A4X blog — https://cloud.google.com/blog/products/compute/new-a4x-vms-powered-by-nvidia-gb200-gpus
- GCP A4 blog — https://cloud.google.com/blog/products/compute/introducing-a4-vms-powered-by-nvidia-b200-gpu-aka-blackwell
- GCP A3 Ultra CUD documentation — https://docs.cloud.google.com/compute/docs/instances/committed-use-discounts-overview
- Azure H100 pricing (Spheron) — https://www.spheron.network/blog/azure-h100-pricing/
- Azure ND96isr_H100_v5 Vantage — https://instances.vantage.sh/azure/vm/nd96isrh100-v5
- Azure ND96isr_H200_v5 Vantage — https://instances.vantage.sh/azure/vm/nd96isrh200-v5
- Azure ND-GB200-v6 docs — https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nd-gb200-v6-series
- Azure ND GB200 v6 GA announcement — https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/accelerating-the-intelligence-age-with-azure-ai-infrastructure-and-the-ga-of-nd-/4394575
- OCI GPU compute — https://www.oracle.com/cloud/compute/gpu/
- OCI ComputePrices — https://computeprices.com/providers/oracle
- OCI bare-metal H100 LLM blog (easecloud) — https://blog.easecloud.io/ai-cloud/optimal-oci-gpu-shape-for-llms/
- OCI H200 GA announcement — https://blogs.oracle.com/cloud-infrastructure/now-ga-largest-ai-supercomputer-oci-nvidia-h200
- H100/H200 multi-provider comparisons — https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison , https://www.thundercompute.com/blog/nvidia-h100-pricing , https://www.thundercompute.com/blog/nvidia-h200-pricing , https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/
- AWS Bedrock pricing — https://aws.amazon.com/bedrock/pricing/
- AWS Bedrock pricing analysis (TokenMix) — https://tokenmix.ai/blog/aws-bedrock-pricing
- AWS Bedrock pricing analysis (pecollective) — https://pecollective.com/tools/aws-bedrock-pricing/
- Vertex AI pricing — https://cloud.google.com/vertex-ai/generative-ai/pricing , https://tokenmix.ai/blog/vertex-ai-pricing
- Azure AI Foundry models pricing — https://azure.microsoft.com/en-us/pricing/details/ai-foundry-models/mistral-ai/ , https://azure.microsoft.com/en-us/pricing/details/ai-foundry/
- Foundry pricing analysis — https://www.wrvishnu.com/azure-ai-foundry-pricing-2026/

---

## 6. Confidence & Gaps

### Confidence

- **High** on AWS p5/p6 list, Capacity Block pricing (multiple confirming sources, including AWS's own page).
- **High** on Azure ND H100 v5 reserved structure ($98.32 OD → ~$43.79 3-yr RI host; Spheron + Vantage agree).
- **Medium-High** on GCP a3-high (H100) and a4-high (B200) on-demand list rates.
- **Medium** on managed-inference token pricing (multiple third-party aggregators agree on order-of-magnitude, but per-Mtok rates for Llama 3.3 70B disagreed by ~3× between sources — flagged in Section 3a).

### Gaps / unresolved

1. **AWS p5e/p5en/p6-b200 standard 1-yr / 3-yr RI rates are not published cleanly anywhere.** AWS sells these instances primarily via Capacity Blocks (max 6-mo term, dynamic pricing). The "3-yr 57% off best-case" number for p6-b200 ($49.22/hr full-host instance-locked) came from one third-party source (cloudprice/economize) and could not be cross-verified against AWS's own pricing page.
2. **Google Cloud A3 Ultra (H200), A4 (B200), A4X (GB200) reserved pricing is gated behind AI Hypercomputer future-reservations — quote-only via account team.** Standard CUD discount math (37% / 55%) does not apply. I documented this but could not surface actual negotiated rates.
3. **Azure ND H200 v5 reserved rates are not in Vantage / third-party feeds.** I gave estimates (~30-50% off OD) using Microsoft's "up to 60%" language but did not confirm with a hard quote.
4. **Azure ND GB200 v6 pricing is entirely quote-only.** Announced GA, but no per-hour price published anywhere I could find.
5. **Oracle OCI pricing is conflicting between sources.** ComputePrices.com posts $10/GPU/hr for H100, H200; easecloud.io reports the effective committed rate at $4.10/GPU/hr ($32.77/hr 8-GPU host). The ~2.5× gap likely reflects list-vs-Universal-Credit-commit but I could not confirm against Oracle's own price list (https://www.oracle.com/cloud/price-list/ returned HTTP 403 to WebFetch).
6. **GH200 is essentially absent from hyperscaler menus.** AWS, Azure, OCI do not publish a GH200 SKU. Google had/has GH200 access via Project Cumulus / experimental but no general-availability SKU with public pricing was found. **For GH200 specifically, neoclouds (Lambda, Crusoe, Nebius, CoreWeave) are the path** — out of scope for this agent but flagged.
7. **Bedrock Llama 3.3 70B per-Mtok rate conflict** — pecollective.com says $0.72 in/out; tokenmix.ai says $2.65. AWS's own Bedrock pricing page only surfaced legacy Llama 2 rates to WebFetch. Worth a manual recheck on the live Bedrock console.
8. **Spot / preemptible pricing not included** — out of scope for "reserved monthly" framing but available for all four clouds (typically 60-80% off on-demand) if relevant for a follow-up.

### Recommended next steps for downstream agents
- Cross-verify the Bedrock Llama 3.3 70B rate from the live AWS console (login required).
- Pull GCP A3 Ultra / A4 quotes from a real Google Cloud account team — they will not be in any public feed.
- Pull a real Oracle Universal Credits H100/H200 quote to settle the $10 vs $4.10/GPU-hr gap.
- For GH200 specifically, see neocloud agents (Lambda Labs / CoreWeave / Crusoe / Nebius round-1 reports) — none of the four hyperscalers currently rents GH200 on a publicly-priced monthly contract.
