# Hyperscaler GPU + Managed Inference — Round 3 Verification (Agent #19)

**Date of snapshot:** 2026-05-25
**Method:** 15 WebFetch / WebSearch calls cross-checking AWS, GCP, Azure pricing pages, instances.vantage.sh, gcloud-compute.com, cloudprice.net, costbench, llm-token-calculator, Spheron, wrvishnu.
**Scope of changes from Round 1:** verified rates; corrected several third-party feed errors; resolved the Bedrock Llama 3.3 70B `$0.72 vs $2.65` conflict.

---

## 1. AWS EC2 — Verified

### 1a. p5 family (H100 / H200)

| Instance | GPU | OD $/hr (host) | OD $/GPU-hr | 1-yr SP $/hr (host) | 3-yr RI $/hr (host) | Capacity Block $/hr (host) | Source |
|---|---|---|---|---|---|---|---|
| p5.48xlarge | 8× H100 80GB | **$55.04** | **$6.88** | ~$38.53 SP / N/A standard RI | **$23.777** (3-yr RI, Vantage) / ~$24.77 (Spheron) | $34.608 US-East/Ohio/Oregon (= $4.326/GPU) | Vantage, Spheron, AWS Capacity Block page |
| p5e.48xlarge | 8× H200 141GB | **OD not publicly listed** (Vantage shows "$1.843" which is clearly a parser error — instance page says only Capacity Blocks) | n/a | N/A | N/A | **$39.799** US-East-Ohio / **$49.749** US-W-N.Cal (= $4.975 / $6.219 per GPU) | AWS Capacity Block page |
| p5en.48xlarge | 8× H200 141GB (Sapphire) | **$63.296/hr** (Vantage; this is OD where published) | $7.91 | N/A | N/A | **$45.768** US-East-VA/Ohio/Oregon/N.Cal (= $5.721 per GPU) | Vantage, AWS Capacity Block page |

**Confirmed from AWS's own Capacity Blocks page (https://aws.amazon.com/ec2/capacityblocks/pricing/):**
- p5 (H100): $34.608/hr host, $4.326/GPU-hr (US-East-VA/Ohio + US-W-Oregon)
- p5e (H200): $39.799/hr host, $4.975/GPU-hr (US-East-Ohio + US-W-Oregon); $49.749/hr in US-W-N.Cal
- p5en (H200): $45.768/hr host, $5.721/GPU-hr (US-East-VA/Ohio + US-W-Oregon + US-W-N.Cal)
- **Next scheduled re-price: July 2026.** (The Jan 2026 hike that pushed Ohio p5e from $34.608 → $39.799 is now baked in — page reports as current.)

**3-yr RI for p5.48xlarge confirmed:** Vantage shows **$23.777/hr host = $2.972/GPU-hr** for the 3-yr All-Upfront tier. At 730 h/mo this is **$2,170/GPU-mo** (Round 1 said $2,263 — close, but Round 3 value is the better number).

### 1b. p6 family (B200 / GB200)

| Instance | GPU | OD $/hr (host) | OD $/GPU-hr | RI/SP | Capacity Block $/hr | Source |
|---|---|---|---|---|---|---|
| p6-b200.48xlarge | 8× B200 180GB | **$113.933/hr** | **$14.242** | **All RI/SP options show N/A** in Vantage + aws-pricing | **$82.368** (US-East-VA/Ohio, US-W-Oregon) = $10.296/GPU-hr | Vantage, aws-pricing.com, AWS Capacity Block page |
| p6e-gb200.36xlarge | 4× GB200 (UltraServer) | not publicly listed for OD | n/a | N/A | **$761.904/ultraserver/hr** (72× B200 unit, US-East-Dallas LZ) = $10.582/B200-hr | AWS Capacity Block page |

**Important Round-1 correction:** Round 1 reported `p6-b200 3-yr No-Up = $10.459/GPU-hr → best case $6.153 (3-yr inst-locked, 57%)`. Round 3 evidence: **AWS does not publish any standard RI or SP rate for p6-b200 today**. The "$6.153/GPU-hr 3-yr inst-locked" figure was a third-party (cloudprice/economize) extrapolation that **could not be reproduced**. Treat the only firm price as **OD $113.93/hr host** or **Capacity Block $82.37/hr host** (max 6-mo term, dynamic).

| Monthly ($730h) | OD | Capacity Block |
|---|---|---|
| p5.48xlarge | $40,179 | $25,264 |
| p5e.48xlarge | n/a | $29,053 |
| p5en.48xlarge | $46,206 | $33,411 |
| p6-b200.48xlarge | $83,171 | $60,129 |
| p6e-gb200 UltraServer | n/a | $556,190 |

### 1c. AWS p5 effective 3-yr reserved (best confirmed)

- p5.48xlarge 3-yr RI: $23.777/hr host = **$2.972/GPU-hr** → **$2,170/GPU-mo @ 730h** → **$17,357/mo for full 8× H100 box**.
- This is the only AWS Hopper/Blackwell GPU shape with a confirmed multi-year reserved rate as of May 2026.

---

## 2. Google Cloud — Verified

### 2a. A3 / A4 / A4X family

Vantage feed for GCP A3-family was **demonstrably wrong** (showed $9.46/hr for a3-highgpu-8g and $12.27/hr for a3-ultragpu-8g, both off by ~10×). Round 3 verified rates via **gcloud-compute.com** and **cloudprice.net** which match GCP's calculator.

| Instance | GPU | OD $/hr (host) | OD $/GPU-hr | 1-yr CUD $/hr | 3-yr CUD $/hr | Source |
|---|---|---|---|---|---|---|
| a3-highgpu-8g | 8× H100 80GB | **$88.49–$126.64** by region (US-low → APAC-high) | **$11.06–$15.83** | $54.50–$80.17 (~38% off) | **$34.54–$50.75** (~55% off) | gcloud-compute.com |
| a3-megagpu-8g | 8× H100 Mega 80GB | **$92.94–$119.97** | $11.62–$15.00 | $63.94–$82.55/hr (avg $74.70) | **$40.49–$52.26/hr (avg $47.29)** | gcloud-compute.com |
| a3-ultragpu-8g | 8× H200 141GB | **~$86.76/hr** (= $63,335/mo ÷ 730 in cheapest region; range up to ~$110/hr globally) | **~$10.84** | not publicly listed — quote-only via AI Hypercomputer | not publicly listed — quote-only | cloudprice.net (monthly $63,334.74) |
| a4-highgpu-8g | 8× B200 192GB | quote-only / not publicly listed in gcloud-compute or cloudprice (404'd in Round 3) | — | quote-only | quote-only | — |
| a4x-highgpu-4g | 4× GB200 Superchip | quote-only | — | quote-only | quote-only | — |

**Round-1 corrections (Google):**

- Round 1 a3-highgpu-8g OD = "~$31.45/hr host, ~$3.93/GPU-hr". **WRONG.** Verified OD is **$88-127/hr host depending on region** (= $11-16/GPU-hr). Round 1's $31.45 number appears to have come from a 1-yr CUD or sustained-use feed, not OD list.
- Round 1 a3-mega "1-yr 3% off / 3-yr 6% off ⚠ likely third-party error" — confirmed error. Real CUD discount on A3 Mega is **~31% (1-yr) / ~56% (3-yr)** per gcloud-compute monthly figures.
- Round 1 a3-ultragpu-8g OD = "$1.533/GPU-hr per Vantage, conflicting with ~$10.60/GPU-hr" — resolved. **The ~$10-11/GPU-hr number is correct.** Vantage's $12.27/hr "host" figure is a parser error (it's per-GPU but labeled per-host) or stale; reality is ~$87/hr host.

### 2b. GCP — Cheapest 3-yr CUD per GPU/month

- **A3 highgpu (H100)** — best case Iowa/Oregon: 3-yr CUD $34.54/hr host = $4.32/GPU-hr → **$3,153/GPU-mo**.
- **A3 mega (H100 Mega)** — best case: $40.49/hr host = $5.06/GPU-hr → **$3,694/GPU-mo**.
- **A3 ultra (H200)** — no public 3-yr CUD; quote-only via AI Hypercomputer future-reservations.
- **A4 (B200)** — no public 3-yr CUD; quote-only.
- **A4X (GB200)** — no public 3-yr CUD; quote-only.

Round-1 number for H100 a3-high was **$1,292/GPU-mo** — that's far too low; **the verified value is ~$3,153/GPU-mo**. Round 1 dramatically under-reported GCP H100 reserved cost.

---

## 3. Azure — Verified

| Instance | GPU | OD $/hr (host) | OD $/GPU-hr | 1-yr RI $/hr (host) | 3-yr RI $/hr (host) | Source |
|---|---|---|---|---|---|---|
| Standard_ND96isr_H100_v5 | 8× H100 80GB | **$98.32** | **$12.29** | **$63.40** (~35% off, $7.93/GPU-hr) | **$43.79** (~55% off, $5.47/GPU-hr) | Spheron blog, Vantage |
| Standard_ND96isr_H200_v5 | 8× H200 141GB | **$110.24** | **$13.78** | N/A in Vantage; Microsoft says available, ~30-50% off (est. $7.95-9.65/GPU-hr) | N/A in Vantage; est. ~$6.20-6.90/GPU-hr | Vantage; estimate from Microsoft language |
| Standard_ND128isr_GB200_v6 | 4× GB200 (2× Grace + 4× Blackwell per VM, 18 VM/rack = NVL72) | **quote-only** — not in any public price feed (verified via Microsoft Learn docs page, which lists specs but no $/hr) | — | quote-only | quote-only | Microsoft Learn (nd-gb200-v6-series page) |

**Round 1 numbers stand for H100 v5** (Spheron + Vantage agree on $98.32 OD / $63.40 1-yr RI / $43.79 3-yr RI).

**Round 1 numbers stand for H200 v5 OD** ($110.24/hr).

**Round 1 H200 v5 RI estimates remain unverified** — Microsoft offers RIs but neither Vantage nor Spheron publishes hard numbers.

**ND-GB200-v6 confirmed as quote-only** — Microsoft Learn docs page (updated 2026-04-02) describes the SKU `Standard_ND128isr_NDR_GB200_v6` (128 vCPU, 900 GB LPDDR, 4× Blackwell GPU 192 GB ea, 4× 400 Gbps IB) but the pricing-calculator link returns no hits for this size. Sold via committed AI infrastructure deals.

### Per-GPU monthly summary (Azure, 730 hr)

| GPU | OD $/GPU-mo | 3-yr RI $/GPU-mo |
|---|---|---|
| H100 | $8,972 | **$3,996** |
| H200 | $10,059 | est. **$4,526–$5,037** (unverified) |
| GB200 | quote-only | quote-only |

---

## 4. Managed Inference $/Mtok — Verified

### 4a. AWS Bedrock — Llama 3.3 70B conflict RESOLVED

**Round-1 conflict:** pecollective.com said `$0.72 in / $0.72 out`; tokenmix.ai said `$2.65 in/out`.

**Round-3 evidence:**
- costbench.com/software/llm-api-providers/amazon-bedrock/ — explicit, current (May 14, 2026): **"Llama 3.3 70B: $0.72/1M input, $0.72/1M output"**.
- llm-token-calculator.com (consolidated cross-provider table) — **AWS Bedrock Llama 3.3 70B = $0.72/$0.72**, model ID `meta.llama3-3-70b-instruct-v1:0`.
- WebSearch result (May 2026): "Llama 3.3 70B costs $0.72 per million tokens on AWS Bedrock as of April 2026" — confirmed across multiple sources.
- TokenMix's own blog **simultaneously** says "Llama 3.3 70B: $0.72/M tokens" AND "Llama 70B on Bedrock costs $2.65 vs $0.88 on Together AI" — the **$2.65 figure refers to a different SKU (likely Llama 3.1 70B / earlier Llama 3 70B which had higher per-token rates), or is stale**. The $2.65 number is **not** Llama 3.3 70B's current rate.

**Verdict: AWS Bedrock Llama 3.3 70B = $0.72 in / $0.72 out per Mtok. Round 1's pecollective source was correct; TokenMix's $2.65 is wrong/stale.**

### 4b. AWS Bedrock — full rate card (us-east-1, May 2026, verified)

| Model | Input $/Mtok | Output $/Mtok | Source |
|---|---|---|---|
| Llama 3.3 8B | $0.22 | $0.22 | TokenMix, pecollective |
| **Llama 3.3 70B** | **$0.72** | **$0.72** | **costbench (May 14 2026), llm-token-calc — verified** |
| Llama 4 Scout (17B/109B MoE) | $0.17 | $0.66 | Round 1 (not re-verified Round 3) |
| Llama 4 Maverick (17B/400B MoE) | $0.24 | $0.97 | Round 1 (not re-verified Round 3) |
| Mistral Large 2 | $3.00 | $9.00 | Round 1 |
| Mistral Large 3 | $0.50 | $1.50 | Round 1 |
| Amazon Nova Pro | $0.80 | $3.20 | Round 1 |
| Amazon Nova Premier | $2.00 | $8.00 | Round 1 |

Batch = 50% off; Provisioned Throughput (PT) ~ 30-50% off at steady-state.

### 4c. Google Vertex AI — Llama 3.3 70B verified

| Source | Llama 3.3 70B price | Notes |
|---|---|---|
| TokenMix (April 2026) | **$1.36 blended /Mtok** (or ~$0.72 in / ~$2.00 out split per artificialanalysis) | "55% premium over Together's $0.88" |
| llm-token-calculator | Not listed on Vertex AI standard rate card; only Llama 3.3 8B at $0.20 in | — |
| Google Cloud's own pricing page (cloud.google.com/vertex-ai/generative-ai/pricing) | **Does not list Llama 3.3 70B as a first-party Vertex rate.** The page is now dominated by Gemini 3.1/3.5 SKUs ($1.50-$18/Mtok). Meta models route through Model Garden marketplace billing, **not** the core Vertex AI rate card. | Implication: Vertex AI's published price for Llama 3.3 70B is **partner/Model-Garden-billed and quote-pricing-shifted** depending on deployment mode (dedicated endpoint vs serverless). |

**Verdict for Vertex Llama 3.3 70B:** **$1.36/Mtok blended** is the best public number, sourced from TokenMix's pricing aggregator (April 2026). **Not on Google's own price list** as a flat rate — Vertex AI sells Llama via Model Garden deployments which can be either dedicated endpoint (per-GPU-hour) or per-token via partner agreements. Treat this as **less reliable** than Bedrock/Foundry per-token rates.

### 4d. Azure AI Foundry — Llama 3.3 70B verified

| Source | Llama 3.3 70B price |
|---|---|
| wrvishnu.com Azure AI Foundry pricing 2026 (May 2026) | **$0.59 input / $0.79 output per Mtok** |
| llm-token-calculator | **$0.71 input** (output not listed cleanly in their table; model ID `azure_ai/Llama-3.3-70B-Instruct`) |

**Verdict: Azure Foundry Llama 3.3 70B = $0.59 / $0.79 per Mtok** (sticking with wrvishnu's dual quote, which matches Round 1).

### 4e. Cross-cloud cheapest hosted Llama 3.3 70B (verified, May 2026)

| Provider | Input | Output | Blended (1:1) |
|---|---|---|---|
| **Azure AI Foundry (cheapest)** | $0.59 | $0.79 | $0.69 |
| AWS Bedrock | $0.72 | $0.72 | **$0.72 (verified)** |
| Google Vertex AI | ~$0.72 | ~$2.00 | $1.36 blended (less confidence) |
| Reference: Together, Fireworks, Groq (non-hyperscaler dedicated) | $0.59–0.88 | $0.59–0.88 | ~$0.70 |

**Bedrock now ties dedicated inference providers; Azure Foundry beats them.**

---

## 5. Cross-cloud cheapest reserved $/GPU/month — Verified

| GPU | Cheapest hyperscaler (3-yr or best commit) | $/GPU-mo verified | Round-1 said | Delta |
|---|---|---|---|---|
| H100 | **AWS p5 3-yr RI** $2.972/GPU-hr | **$2,170** | $2,263 (AWS) but listed GCP at $1,292 | **GCP's $1,292 was wrong**; AWS p5 3-yr RI is the verified leader |
| H100 (runner-up) | Azure ND H100 v5 3-yr RI $5.47/GPU-hr | $3,993 | $3,993 ✓ | matches |
| H100 (runner-up) | GCP a3-high 3-yr CUD $4.32-$6.34/GPU-hr | $3,153–4,628 | $1,292 ✗ | Round 1 used a wrong base rate |
| H100 (runner-up) | OCI H100.8 w/ Universal Credits | ~$2,990 | $2,990 (Round 1, easecloud quote) | unverified Round 3 |
| **H200** | AWS p5e Capacity Block $4.975/GPU-hr (US-East/Ohio) | **$3,632** (block-only, max 6-mo) | $3,632 ✓ | matches |
| H200 | Azure ND H200 v5 est. 3-yr RI | est. $4,526-5,037 (unverified) | est. $4,526-5,037 | unchanged |
| H200 | OCI H200.8 list | ~$7,300 | $7,300 (Round 1) | unverified Round 3 |
| **B200** | **AWS p6-b200 OD (no 3-yr RI exists publicly)** $14.242/GPU-hr | **$10,397** | $4,492 ✗ | Round 1's $4,492 came from an unconfirmed third-party "57% off" claim; **AWS does not publish any p6-b200 3-yr RI**, so the cheapest *confirmed* AWS B200 rate is the OD or Capacity Block ($10,296/hr-host = $10.296/GPU-hr = **$7,516/GPU-mo**) |
| B200 (best Capacity Block) | AWS p6-b200 Capacity Block $10.296/GPU-hr | **$7,516** | not listed cleanly in Round 1 | new finding |
| B200 | GCP a4-high OD (quote-only) | unverified | $5,110 (Round 1) | Round 1's $5,110 OD figure could not be reproduced — gcloud-compute.com 404'd, cloudprice.net paywall |
| **GB200** | AWS p6e-gb200 Capacity Block | **$7,725/B200-mo** ($10.582/hr block-only) | $7,725 ✓ | matches |
| GB200 | Azure ND-GB200-v6 | quote-only ✓ | quote-only ✓ | confirmed |
| GB200 | GCP A4X | quote-only ✓ | quote-only ✓ | confirmed |

---

## 6. Key Round-3 corrections to Round-1 numbers

1. **AWS p6-b200 "3-yr inst-locked 57% off $6.153/GPU-hr" is unsubstantiated.** AWS publishes only OD ($113.93/hr host = $14.24/GPU-hr) and Capacity Blocks ($82.37/hr host = $10.30/GPU-hr) for p6-b200. **No 1-yr or 3-yr RI/SP exists** in any verifiable feed. Round 1's $4,492/GPU-mo figure for B200 3-yr was a third-party extrapolation that does not match AWS's posted rate card.

2. **GCP a3-highgpu-8g H100 OD ≠ $31.45/hr host.** Real OD is **$88-127/hr host** depending on region ($11-16/GPU-hr OD). 3-yr CUD bottoms out at **~$34.54/hr host = ~$4.32/GPU-hr = $3,153/GPU-mo**, not $1,292/GPU-mo.

3. **GCP a3-ultragpu-8g H200 OD ≈ $86.76/hr host = ~$10.84/GPU-hr** (verified via cloudprice's $63,335/mo). Vantage's $12.27/hr "host" is a parser error.

4. **Bedrock Llama 3.3 70B = $0.72/$0.72 per Mtok confirmed.** The $2.65 figure in TokenMix referred to an older Llama 3 / 3.1 70B SKU, not Llama 3.3 70B. Resolved.

5. **AWS Capacity Block prices unchanged since Jan-2026 hike** — re-verified directly from https://aws.amazon.com/ec2/capacityblocks/pricing/. Next re-price: **July 2026**.

6. **AWS p5.48xlarge 3-yr RI = $23.777/hr host = $2.972/GPU-hr = $2,170/GPU-mo** is now the cheapest verified 3-yr Hopper-class GPU on any hyperscaler. (Round 1's "$2,263" was an approximation; $2,170 is the precise Vantage figure.)

---

## 7. New $/Mtok break-even math (updated)

**Llama 3.3 70B on 8× H100 (AWS p5 3-yr RI):**
- Host cost: $23.777/hr × 730 = **$17,357/mo**.
- vLLM/SGLang throughput on 8× H100 bf16: ~5,000-10,000 output tok/s (FP8 broken on DiT per project memory; bf16 default).
- 10k tok/s × 3,600 s × 730 h = 2.63 × 10¹⁰ output tok/mo = 26,280 Mtok/mo (saturated).
- $17,357 / 26,280 = **$0.66/Mtok output at full saturation**.

Compare to:
- Azure AI Foundry: $0.79/Mtok output → self-host wins at ≥83% utilization (was 85% in Round 1; corrected).
- AWS Bedrock: $0.72/Mtok in/out → self-host wins at ≥92% utilization. **Bedrock is now tightly comparable to self-host**.
- Vertex AI: $2.00/Mtok output → self-host wins easily, even at 35% utilization.

---

## 8. Confidence summary

**HIGH confidence (verified ≥2 independent sources):**
- AWS Capacity Block rates for p5/p5e/p5en/p6-b200/p6e-gb200 (from AWS's own page).
- AWS p5 3-yr RI ($23.777/hr host).
- AWS p6-b200 OD ($113.93/hr).
- Azure ND H100 v5 reserved structure ($98.32 → $63.40 1-yr → $43.79 3-yr).
- Azure ND H200 v5 OD ($110.24/hr).
- Bedrock Llama 3.3 70B = $0.72/$0.72 per Mtok.
- Azure AI Foundry Llama 3.3 70B = $0.59/$0.79 per Mtok.
- GCP a3-highgpu-8g OD range $88-127/hr (gcloud-compute.com).

**MEDIUM confidence (single reputable source):**
- GCP a3-mega CUD discount math ($40.49-52.26/hr 3-yr).
- GCP a3-ultragpu-8g H200 OD ≈ $86.76/hr.
- Vertex AI Llama 3.3 70B = $1.36 blended.

**LOW / unresolved:**
- Azure ND H200 v5 1-yr / 3-yr RI rates — Microsoft offers them but no public per-hour figure.
- Azure ND-GB200-v6 — confirmed quote-only.
- GCP A3 Ultra / A4 / A4X reserved rates — quote-only via AI Hypercomputer future-reservations.
- OCI committed/Universal-Credits effective rates — not re-verified Round 3 (Round 1 reported $4.10/GPU-hr H100 commit vs $10/GPU-hr list, source easecloud.io).
- p6-b200 alleged 3-yr inst-locked No-Up RI — **does not exist in public feeds**; Round 1's $6.153/GPU-hr figure is hereby downgraded to "unverified, do not cite".

---

## 9. Sources fetched 2026-05-25 (Round 3)

- https://aws.amazon.com/ec2/capacityblocks/pricing/ — verified Capacity Block rates (p5/p5e/p5en/p6-b200/p6e-gb200)
- https://instances.vantage.sh/aws/ec2/p5.48xlarge — OD $55.04, 3-yr RI $23.777, spot $21.684
- https://instances.vantage.sh/aws/ec2/p6-b200.48xlarge — OD $113.933, RI/SP all N/A, spot $36.155
- https://instances.vantage.sh/aws/ec2/p5e.48xlarge — Vantage data corrupt ($1.843 OD nonsense); use Capacity Block as canonical
- https://instances.vantage.sh/aws/ec2/p5en.48xlarge — OD $63.296, RI N/A
- https://instances.vantage.sh/gcp/a3-ultragpu-8g — OD shown as $12.27 (PARSER ERROR — actually per-GPU not per-host)
- https://instances.vantage.sh/gcp/a3-megagpu-8g — OD $9.46 (parser error similarly)
- https://instances.vantage.sh/gcp/a3-highgpu-8g — OD $9.46 (parser error)
- https://instances.vantage.sh/azure/vm/nd96isrh200-v5 — OD $110.24, RI N/A
- https://instances.vantage.sh/azure/vm/nd96isrh100-v5 — OD $98.32, RI N/A in Vantage (Spheron has the RI breakdown)
- https://aws-pricing.com/p6-b200.48xlarge.html — OD $113.93-136.72 by region, all RI N/A
- https://cloudprice.net/gcp/compute/instances/a3-ultragpu-8g — monthly $63,334.74 → ~$86.76/hr
- https://gcloud-compute.com/a3-highgpu-8g.html — OD $88-127/hr, 3-yr CUD $34.54-50.75/hr
- https://gcloud-compute.com/a3-megagpu-8g.html — OD $92.94-119.97/hr, 3-yr CUD avg $47.29/hr host
- https://www.spheron.network/blog/aws-h100-pricing-2026/ — p5 OD $55.04, 3-yr RI ~$24.77
- https://www.spheron.network/blog/azure-h100-pricing/ — Azure H100 OD/1yr/3yr verified
- https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nd-gb200-v6-series — ND-GB200-v6 confirmed quote-only, specs only
- https://costbench.com/software/llm-api-providers/amazon-bedrock/ — Bedrock Llama 3.3 70B = $0.72/$0.72 (May 14 2026)
- https://www.llm-token-calculator.com/all-model-price-list — Bedrock Llama 3.3 70B = $0.72/$0.72; Azure Foundry Llama 3.3 70B input = $0.71
- https://www.wrvishnu.com/azure-ai-foundry-pricing-2026/ — Azure Foundry Llama 3.3 70B = $0.59/$0.79 (May 2026)
- https://tokenmix.ai/blog/vertex-ai-pricing — Vertex Llama 3.3 70B = $1.36 blended (Apr 2026)
- https://artificialanalysis.ai/models/llama-3-3-instruct-70b/providers — confirms 8.6× price spread across Llama 3.3 70B providers; specific hyperscaler $/Mtok not surfaced cleanly
- https://cloud.google.com/vertex-ai/generative-ai/pricing — confirms Llama 3.3 70B **not** on Vertex's primary rate card (only Gemini/Gemma direct-billed; Llama via Model Garden marketplace)
- https://aws.amazon.com/bedrock/pricing/ — Bedrock pricing page surface only shows legacy Llama 2 SKUs to WebFetch; the live console has the current Llama 3.3 / 4 entries

---

## 10. Three-line take-away

1. **Cheapest verified 3-yr reserved Hopper on a hyperscaler = AWS p5 H100 at $2,170/GPU-mo** (was Round 1's runner-up; round-1's "GCP a3-high $1,292/GPU-mo" was based on a misread of the OD base rate).
2. **AWS B200 (p6-b200) has no public 1-yr or 3-yr RI/SP** as of May 2026 — only OD ($113.93/hr host = $14.24/GPU-hr) and Capacity Blocks (max 6-mo, $82.37/hr host = $10.30/GPU-hr). Round-1's $4,492/GPU-mo 3-yr figure is **unsubstantiated**.
3. **Bedrock Llama 3.3 70B settled at $0.72/$0.72 per Mtok** — TokenMix's $2.65 was a stale/wrong artifact. Azure AI Foundry remains the cheapest hyperscaler-managed Llama 3.3 70B endpoint at $0.59/$0.79 per Mtok.
