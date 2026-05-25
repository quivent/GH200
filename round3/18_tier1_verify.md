# Tier-1 Specialty Clouds — Round 3 Verification (Agent 18)

**Date:** 2026-05-25
**Method:** 14 WebFetch + 3 WebSearch directly against provider pricing pages and SemiAnalysis sources. Cross-checked Round 1 numbers; corrections noted in bold.
**Verdict at a glance:** Round 1 specialty-cloud pricing table holds up well. Five numbers verified exactly; one tier rating corrected (Together AI is **Gold**, not boundary); CoreWeave per-node vs per-GPU pricing clarified; ClusterMAX 2.0 publish date pinned to **Nov 6, 2025**.

---

## 1. Verified Pricing Table (per-GPU-hour, USD)

| Item | Round-1 claim | Verified value | Source | Status |
|---|---|---|---|---|
| **CoreWeave GH200** on-demand | $6.50/hr | **$6.50/hr** (1-GPU SKU) | coreweave.com/pricing | ✅ Exact match |
| **CoreWeave HGX H100** on-demand | $4.25 PCIe / $4.76 HGX | PCIe **$4.25/hr**; HGX H100 **$49.24/hr per 8-GPU node** (= **$6.16/GPU**, listed as "Inference" rate per GPU) | coreweave.com/pricing/classic + /pricing | ⚠️ Clarified — Round 1 conflated the per-node and per-GPU columns |
| **CoreWeave HGX H200** on-demand | ~$2.58 (pub) / $6.31 HGX 8x | **$50.44/hr per 8-GPU node** (= **$6.31/GPU** Inference rate). $2.58 figure not on current page | coreweave.com/pricing | ⚠️ The $2.58 number is stale / from a derivative tracker |
| **CoreWeave HGX B200** on-demand | $8.60 SXM / $10.50 NVL | **$68.80/hr per 8-GPU node** (= **$8.60/GPU**); GB200 NVL72 (4-GPU) **$42.00/hr** (= **$10.50/GPU**) | coreweave.com/pricing | ✅ Confirmed (per-GPU equivalents exact) |
| **CoreWeave Reserved discount** | up to 60% off | **"up to 60% discount over on-demand"** confirmed on page | coreweave.com/pricing | ✅ Confirmed |
| **CoreWeave Capacity Plans** modes | Reservations, Flex Reservations, On-Demand, Spot | All four confirmed; Spot has **7-minute interrupt notice** | coreweave.com/coreweave-capacity-plans | ✅ Confirmed + extra detail |
| **CoreWeave HGX 2-yr contract floor** | not in R1 | **"starting at $2.23/hr" on 2-year HGX contract** (newly surfaced) | coreweave.com/products/hgx-h100-h200 | 🆕 New data point |
| **Together AI HGX H100** on-demand | $5.49 | **$5.49/hr** | together.ai/pricing | ✅ Exact match |
| **Together AI HGX H100** reserved (7-30 / 31-90 / 91-180d) | $4.99 / $4.49 / $3.99 | **$4.99 / $4.49 / $3.99** | together.ai/pricing | ✅ Exact match |
| **Together AI HGX H200** on-demand | $6.79 | **$6.79/hr** | together.ai/pricing | ✅ Exact match |
| **Together AI HGX H200** reserved (7-30 / 31-90 / 91-180d) | $5.95 / $4.99 / $4.55 | **$5.95 / $4.99 / $4.55** | together.ai/pricing | ✅ Exact match — $4.55 confirmed |
| **Together AI HGX B200** on-demand | $9.95 | **$9.95/hr** | together.ai/pricing | ✅ Exact match |
| **Together AI HGX B200** reserved (7-30 / 31-90 / 91-180d) | $9.65 / $9.35 / $9.09 | **$9.65 / $9.35 / $9.09** | together.ai/pricing | ✅ Exact match |
| **Together AI dedicated endpoint** 1xH100 | $6.49 | **$6.49/hr** | together.ai/pricing | ✅ Exact match |
| **Together AI dedicated endpoint** 1xB200 | $11.95 | **$11.95/hr** | together.ai/pricing | ✅ Exact match |
| **Together AI 181+ day pricing** | not in R1 | "Contact for pricing" on all SKUs at 181+d | together.ai/pricing | 🆕 Detail |
| **Together AI min reservation** | 6 days | **"above 6 days"** confirmed | together.ai/pricing | ✅ Confirmed |
| **OCI BM.GPU.H100.8** list | $80/hr per node = $10/GPU-hr | **$10/GPU-hr** confirmed | computeprices.com (May 24 2026) + oraclelicensingexperts.com cite + WebSearch | ✅ Confirmed |
| **OCI BM.GPU.H200.8** list | $80/hr per node = $10/GPU-hr | **$10/GPU-hr** confirmed (same price as H100 — H200 is *not* priced higher) | computeprices.com + WebSearch | ✅ Confirmed |
| **OCI BM.GPU.B200** list | yes via supercluster | **$14/GPU-hr** | computeprices.com | 🆕 New data point |
| **OCI BM.GPU.GB200.4** list | NVL72 GA, pricing via sales | **$16/GPU-hr** publicly tracked | computeprices.com | 🆕 New data point |
| **OCI UCC 1-year discount** | 15–45% off | Cited in derivative source; oracle.com/cloud/compute/gpu/ and price-list returned 403 Forbidden — cannot independently re-verify from Oracle directly today | oraclelicensingexperts.com (R1 source) | ⚠️ Medium confidence — Oracle's own pages blocked WebFetch |
| **OCI UCC 3-year, $5M+ discount** | 25–60% off | Same — only derivative source confirms | oraclelicensingexperts.com | ⚠️ Medium confidence |
| **OCI shape: H200 networking** | 3,200 Gbps RoCEv2 | Docs confirm **8 x 400 Gbps RDMA = 3,200 Gbps** on BM.GPU.H200.8; ConnectX-7; RoCEv2 | docs.oracle.com computeshapes | ✅ Confirmed |
| **OCI shape: H200 spec** | n/a in R1 | **8x H200 141 GB, 1128 GB total GPU mem, 8x 3.84 TB NVMe, Intel SR 8480+ 2x56c** | docs.oracle.com computeshapes | 🆕 Detail |
| **OCI shape: GB200.4** | n/a in R1 | **4x B200 192 GB, 768 GB GPU mem, 2x 72-core NVIDIA Grace, 960 GB CPU mem, 4x 7.68 TB NVMe** | docs.oracle.com computeshapes | 🆕 Detail |
| **Crusoe H100 HGX** | $3.90 | **$3.90/GPU-hr** | crusoe.ai/cloud/pricing | ✅ Exact match |
| **Crusoe H200 HGX** | $4.29 | **$4.29/GPU-hr** | crusoe.ai/cloud/pricing | ✅ Exact match |
| **Crusoe B200 / GB200 NVL72** | contact sales | "Contact sales" — confirmed | crusoe.ai/cloud/pricing | ✅ Confirmed |
| **Crusoe MI300X** (not in R1) | n/a | **$3.45/GPU-hr** (AMD MI300X 192 GB) | crusoe.ai/cloud/pricing | 🆕 New data point |
| **Crusoe persistent disk** | $0.08/GiB/mo | **$0.08/GiB/mo** | crusoe.ai/cloud/pricing | ✅ Confirmed |
| **Crusoe shared disk** | $0.07/GiB/mo | **$0.07/GiB/mo** | crusoe.ai/cloud/pricing | ✅ Confirmed |
| **Crusoe billing** | n/a | **Per-minute billing, no setup fee** | crusoe.ai/cloud/pricing | 🆕 Detail |
| **Nebius H100 HGX** on-demand | $2.95 | **$2.95/GPU-hr** | nebius.com/prices | ✅ Exact match |
| **Nebius H100 HGX** preemptible | $1.25 | **$1.25/GPU-hr** | nebius.com/prices | ✅ Exact match |
| **Nebius H200 HGX** on-demand | $3.50 | **"from $3.50/GPU-hr"** | nebius.com/h200, nebius.com/prices | ✅ Exact match |
| **Nebius H200 HGX** preemptible | $1.45 | **$1.45/GPU-hr** | nebius.com/prices | ✅ Confirmed |
| **Nebius H200 HGX** 3+ mo commit | $2.30 / 35% off | **"as low as $2.30/hr"** at 3+ mo commit, **"~34% savings"** stated (Round 1 said 35%) | nebius.com/h200 | ✅ Very close — actual figure is 34%, not 35% |
| **Nebius B200 HGX** on-demand | $5.50 | **$5.50/GPU-hr** | nebius.com/prices | ✅ Exact match |
| **Nebius B200 HGX** preemptible | $2.90 | **$2.90/GPU-hr** | nebius.com/prices | ✅ Exact match |
| **Nebius GB200 NVL72** | pre-order | "Contact us" — no public price; pre-order form only | nebius.com/blackwell-pre-order, nebius.com/prices | ✅ Confirmed |
| **Nebius commit headline** | up to 35% off | **"Save up to 35% on on-demand rates when you reserve large-scale clusters for a multi-month period"** | nebius.com/prices | ✅ Confirmed exact language |

---

## 2. ClusterMAX 2.0 — Verified Tier Rankings

**Published: November 6, 2025** by SemiAnalysis. Covers **84 reviewed providers** out of **209 in their market view** (up from 26 in 1.0). No ClusterMAX 2.1 or 3.0 published yet as of 2026-05-25.

| Tier | Verified providers | Source |
|---|---|---|
| **Platinum** | **CoreWeave (sole member)** — "no other provider came particularly close" | semianalysis ClusterMAX 2.0 article |
| **Gold** | **Nebius, Oracle, Azure, Crusoe, Fluidstack, Together AI** | semianalysis + together.ai/blog/clustermax-gold (Together's own confirmation) |
| **Silver** | **Google Cloud, AWS, Lambda, GMO Internet (GMO GPU Cloud)** | semianalysis + GMO press release |
| **Bronze / Underperform / others** | Not yet fully extracted (84 providers total; full list behind $500 paywall) | semianalysis subscription |

**Correction vs Round 1:**
- Round 1 listed Together AI as "Gold/Silver boundary." That is **wrong** — Together AI is officially **Gold** per SemiAnalysis and per Together's own press release (together.ai/blog/clustermax-gold).
- Round 1 listed "LeptonAI" in Gold; I could **not independently confirm** LeptonAI's tier from the sources accessible without subscription.
- Round 1 listed DataCrunch / TensorWave as Bronze; also unconfirmed in accessible sources.

---

## 3. CoreWeave Pricing Structure — Clarification (most important correction)

CoreWeave's `/pricing` page presents three columns: **On-Demand (per-node), Spot (per-node), and "Inference" (per-GPU)**. Round 1 numbers blended them without flagging.

Verified per-node and per-GPU rates (NA region):

| GPU | On-Demand (per node) | Spot NA (per node) | "Inference" (per GPU) |
|---|---|---|---|
| H100 HGX 8x | $49.24/hr | $19.71/hr | **$6.16/hr/GPU** |
| H200 HGX 8x | $50.44/hr | $20.93/hr | **$6.31/hr/GPU** |
| B200 HGX 8x | $68.80/hr | $34.11/hr | **$8.60/hr/GPU** |
| GB200 NVL72 (4 GPU/instance) | $42.00/hr | N/A | **$10.50/hr/GPU** |
| GH200 (1 GPU) | **$6.50/hr/GPU** | N/A | $6.50/hr/GPU |
| H100 PCIe (single, classic) | $4.25/hr/GPU | n/a | n/a |

GH200 is sold as a **1-GPU SKU at $6.50/hr** — this is verified exact.

**Reserved capacity:** "up to 60% discount over on-demand" — confirmed on page. Specific committed-use tier prices remain non-public (sales-only). HGX H100/H200 has been quoted as **"starting at $2.23/hr" on a 2-year contract** on the HGX product page (`coreweave.com/products/hgx-h100-h200`) — new data point not in Round 1.

---

## 4. Provider Quick-Verify Recap

### CoreWeave
- GH200 **$6.50/hr** ✅ (1-GPU SKU)
- Reserved up to 60% off ✅
- Capacity Plans (Mar 2026): Reservations, Flex Reservations, On-Demand, Spot ✅
- HGX 2-yr floor $2.23/hr 🆕

### Together AI
- **Every** Round-1 pricing tier matches the live page exactly (H100/H200/B200 × on-demand + 3 reserved tiers + dedicated endpoints).
- **H200 91-180d = $4.55/hr ✅ verified.**
- Min reservation: 6 days ✅.
- ClusterMAX tier: **Gold** (Round 1 said "boundary" — should be definitively Gold).

### OCI
- BM.GPU.H100.8 and BM.GPU.H200.8 both **$10/GPU-hr** list ✅ (via computeprices.com tracker, last updated 2026-05-24; oracle.com/cloud/compute/gpu/ and /cloud/price-list/ returned 403 to WebFetch).
- BM.GPU.B200 **$14/GPU-hr** list 🆕
- BM.GPU.GB200.4 **$16/GPU-hr** list 🆕
- Hardware specs (8x 400G RDMA = 3,200 Gbps RoCEv2, ConnectX-7) confirmed via docs.oracle.com.
- UCC 1-yr / 3-yr discount bands (15-45% / 25-60%): only derivative source confirms — Oracle's own pages blocked direct re-verification.

### Crusoe
- H100 **$3.90/GPU-hr** ✅
- H200 **$4.29/GPU-hr** ✅
- B200, GB200 NVL72: contact sales ✅
- Storage rates ✅
- AMD MI300X **$3.45/GPU-hr** newly noted 🆕

### Nebius
- H100 on-demand **$2.95**, preempt **$1.25** ✅
- H200 on-demand **$3.50**, preempt **$1.45**, 3+ mo commit **$2.30** ✅
- ⚠️ Page states "~34% savings" not 35% — minor numeric correction.
- B200 on-demand **$5.50**, preempt **$2.90** ✅
- GB200 NVL72: pre-order via sales ✅
- "Up to 35%" reserve discount headline confirmed for large multi-month clusters ✅

### SemiAnalysis ClusterMAX 2.0
- Published **Nov 6, 2025** ✅
- CoreWeave sole Platinum ✅
- Gold tier: Nebius, Oracle, Azure, Crusoe, Fluidstack, Together AI ✅
- No 2.1 or 3.0 yet.

---

## 5. What Didn't Re-Verify (limitations of this round)

1. **Oracle's own pricing pages** (`oracle.com/cloud/compute/gpu/`, `oracle.com/cloud/price-list/`, `oraclelicensingexperts.com/blog/oracle-cloud-gpu-skus-pricing/`, and the H200 supercluster blog) all returned **HTTP 403** to WebFetch. OCI's $10/GPU-hr H100/H200 list price had to be confirmed via the third-party computeprices.com tracker and a WebSearch summary. UCC discount bands (15-45% / 25-60%) remain only as a Round-1 derivative claim; I could not re-verify from Oracle today.
2. **CoreWeave reserved $/GPU-hr for H200 and B200** at specific commitment lengths (12-mo, 24-mo) are still non-public; the only published anchor is "up to 60% off" and the new "starting at $2.23/hr on a 2-year HGX contract." Specific 12-month H200/B200 rates require a sales quote.
3. **Crusoe B200 / GB200 NVL72** pricing: still "contact sales" — no public number to verify.
4. **Nebius B200 long-term commit** pricing: pre-order page has no public discount %; only the general "up to 35%" 3+mo headline applies.
5. **ClusterMAX 2.0 full Bronze / Underperform tier listings**: behind a $500 SemiAnalysis paywall; surfaced articles only confirm Platinum/Gold/Silver top entries plus a few specific Silver additions (e.g., GMO Internet).
6. **Together AI `/gpu-clusters` page** returned **different (lower) numbers** in one WebFetch ($3.79 H200 on-demand, $2.09 reserved) than `/pricing` ($6.79 / $5.95 / $4.99 / $4.55). A re-fetch of `/pricing` reconfirmed the higher tiers. The lower numbers appear to be either model hallucination from cached content or an unrelated "GPU Clusters" product page rendering. **Treat `/pricing` as authoritative.**

---

## 6. Net Corrections to Round 1

| # | Correction |
|---|---|
| 1 | **Together AI ClusterMAX tier**: Gold, not "Gold/Silver boundary." Confirmed by Together's own announcement and SemiAnalysis. |
| 2 | **CoreWeave H100/H200/B200 on-demand**: the per-node price ($49.24, $50.44, $68.80) and per-GPU "Inference" price ($6.16, $6.31, $8.60) need to be reported as distinct rows, not blended. Round 1 mixed two different SKU columns. |
| 3 | **Nebius H200 3+mo commit discount**: page says "**~34%**" savings (giving $2.30/hr from $3.50/hr base), not 35%. The headline "up to 35%" applies to the most aggressive large-cluster commits. |
| 4 | **CoreWeave HGX 2-yr contract**: "starting at $2.23/hr" newly surfaced — material new anchor for committed inference pricing math. |
| 5 | **OCI B200 / GB200**: $14 / $16 per GPU-hr list prices newly confirmed via computeprices tracker — Round 1 left these as "yes via supercluster, UCC discounted." |
| 6 | **CoreWeave Spot interruption notice**: 7-minute notice (newly surfaced from Capacity Plans page). |
| 7 | **Together AI 181+ day reserved tier**: "Contact for pricing" (was not explicit in Round 1's $3.99 H100 / $4.55 H200 floor — those numbers cap at 91-180d). |

---

## 7. Confidence Summary

- **High confidence (directly re-verified on provider pages May 25 2026):** CoreWeave GH200 $6.50/hr, full Together AI cluster price ladder, Crusoe H100/H200 + storage, Nebius full H100/H200/B200 + preempt ladder, ClusterMAX 2.0 publish date and top-tier composition.
- **Medium confidence (verified via third-party tracker / search, provider page blocked):** OCI H100/H200 list $10/GPU-hr, OCI B200 $14, GB200 $16. OCI UCC discount bands.
- **Sales-only / not public:** CoreWeave 12-mo H200/B200 rates; Crusoe B200/GB200; Nebius B200 long-term commit; Together AI 181+ d tier; OCI net-effective price at $5M+ commits.

---

## 8. Sources (re-fetched in this round)

- [coreweave.com/pricing](https://www.coreweave.com/pricing) — verified GH200 $6.50, per-node H100/H200/B200/GB200 rates, "Inference" per-GPU column, 60% reserved discount
- [coreweave.com/pricing/classic](https://www.coreweave.com/pricing/classic) — verified H100 PCIe $4.25
- [coreweave.com/coreweave-capacity-plans](https://www.coreweave.com/coreweave-capacity-plans) — verified Reservations/Flex/On-Demand/Spot mode definitions; 7-min spot interrupt
- [coreweave.com/products/hgx-h100-h200](https://www.coreweave.com/products/hgx-h100-h200) — "$2.23/hr starting" 2-yr HGX contract
- [together.ai/pricing](https://www.together.ai/pricing) — verified entire H100/H200/B200 cluster ladder + dedicated endpoints + 6-day min reservation
- [together.ai/blog/clustermax-gold](https://www.together.ai/blog/clustermax-gold) — confirms Together AI Gold tier
- [crusoe.ai/cloud/pricing](https://www.crusoe.ai/cloud/pricing) — verified H100 $3.90, H200 $4.29, MI300X $3.45, storage, B200/GB200 contact-sales
- [nebius.com/prices](https://nebius.com/prices) — verified all H100/H200/B200 on-demand + preempt rates, GB200 contact us, 35% commit headline
- [nebius.com/h200](https://nebius.com/h200) — verified H200 3+mo commit $2.30, "~34% savings"
- [nebius.com/blackwell-pre-order](https://nebius.com/blackwell-pre-order) — pre-order form, no public discount %
- [docs.oracle.com/.../computeshapes.htm](https://docs.oracle.com/en-us/iaas/Content/Compute/References/computeshapes.htm) — verified BM.GPU.H100.8, BM.GPU.H200.8, BM.GPU.GB200.4 hardware specs
- [computeprices.com/providers/oracle](https://computeprices.com/providers/oracle) — verified OCI H100 $10, H200 $10, B200 $14, GB200 $16 (data 2026-05-24)
- [getdeploying.com/gpus/nvidia-h200](https://getdeploying.com/gpus/nvidia-h200) — cross-checked H200 prices across providers
- [newsletter.semianalysis.com/p/clustermax-20-the-industry-standard](https://newsletter.semianalysis.com/p/clustermax-20-the-industry-standard) — ClusterMAX 2.0 tier list
- WebSearch: "OCI BM.GPU.H200.8 price per hour 2026 $10 GPU-hr" — confirms H200 = $10/GPU-hr list price
- WebSearch: "ClusterMAX 2.0 Together AI Silver Gold tier" — confirms Gold

**Sources that returned HTTP 403 in this round (could not re-fetch):** oracle.com/cloud/compute/gpu/, oracle.com/cloud/compute/pricing/, oracle.com/cloud/price-list/, oraclelicensingexperts.com, blogs.oracle.com.
