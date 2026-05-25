# Lambda Labs — Monthly Inference Pricing & Stack Survey (May 2026)

Research agent #16 of 20. Compiled 2026-05-25.
Scope: Lambda Cloud (public, 1-Click Clusters, Private Cloud / reserved). The user is currently on Lambda, so this is the incumbent-baseline file.

---

## 1. Pricing at a Glance (May 2026, USD)

All Lambda public pricing is per-GPU-per-hour. Monthly = hourly × 730. Lambda bills compute hourly in 1-minute increments; 1-Click Clusters in weekly increments; filesystems in 1-hour increments.

### 1.1 On-Demand (self-serve, no commitment)

| Config | Per-GPU $/hr | Node $/hr | Node $/mo (730h) | vCPU | RAM | Local SSD |
|---|---|---|---|---|---|---|
| 1× GH200 96GB (Grace-Hopper) | $2.29 | $2.29 | $1,672 | 64 | 432 GiB | 4 TiB |
| 1× H100 SXM 80GB | $4.29 | $4.29 | $3,132 | 26 | 225 GiB | 2.75 TiB |
| 1× H100 PCIe 80GB | $3.29 | $3.29 | $2,402 | 26 | 225 GiB | 1 TiB |
| 8× H100 SXM 80GB | $3.99 | $31.92 | $23,302 | 208 | 1,800 GiB | 22 TiB |
| 8× B200 SXM6 192GB | $6.69 | $53.52 | $39,070 | 208 | 2,900 GiB | 22 TiB |
| 1× B200 SXM6 192GB | $6.99 | $6.99 | $5,103 | 26 | 360 GiB | 2.75 TiB |
| 8× A100 SXM 80GB | $2.79 | $22.32 | $16,294 | 240 | 1,800 GiB | 19.5 TiB |
| 1× A100 SXM 40GB | $1.99 | $1.99 | $1,453 | 30 | 220 GiB | 512 GB |
| 1× A10 24GB | $1.29 | $1.29 | $942 | 30 | 226 GiB | 1.3 TiB |
| 1× A6000 48GB | $1.09 | $1.09 | $796 | 14 | 100 GiB | 512 GB |
| 8× V100 32GB | $0.79 | $6.32 | $4,614 | 88 | 448 GiB | 5.8 TiB |

Source: https://lambda.ai/pricing (fetched 2026-05-25); cross-checked with https://lambda.ai/instances and https://computeprices.com/providers/lambda (last updated 2026-05-24).

**Important — H200 is NOT in the on-demand SKU list.** Lambda's H200 capacity is sold only through 1-Click Clusters or Private Cloud (custom). The "1× H200 at $3.79/hr" figure that circulates in third-party blogs (thundercompute.com/blog/nvidia-h200-pricing, May 2026) is not visible on Lambda's own pricing page — treat with caution; it is likely an aggregator's normalized number derived from cluster pricing.

### 1.2 1-Click Clusters (multi-node, InfiniBand, 2-week min)

Self-serve reservation. Weekly billing. 2-week minimum commitment. Payment due within 10 days of invoice or reservation is forfeited.

| GPU type | Cluster size | Term | $/GPU/hr | Per-month equivalent (730h, full cluster) |
|---|---|---|---|---|
| HGX H100 | 16 GPUs | 2 wk – 1 yr | $6.16 | $71,949 / mo |
| HGX H100 | 64 GPUs | 2 wk – 1 yr | $5.85 | $273,312 / mo |
| HGX H100 | 256 GPUs | 2 wk – 1 yr | $5.54 | $1,034,847 / mo |
| HGX B200 | 16 GPUs | 2 wk – 1 yr | $9.86 | $115,165 / mo |
| HGX B200 | 64 GPUs | 2 wk – 1 yr | $9.36 | $437,395 / mo |
| HGX B200 | 256 GPUs | 2 wk – 1 yr | $8.87 | $1,656,762 / mo |
| HGX H100 / B200 | 16+ | 1 yr+ | Contact sales | n/a |

Source: https://lambda.ai/1-click-clusters (fetched 2026-05-25).

Note the gap: 1-Click Cluster H100 starts at $6.16/GPU/hr (16-GPU pod) while 8× H100 on-demand is $3.99/GPU/hr. The cluster premium pays for non-blocking Quantum-2 400 Gb/s InfiniBand (3,200 Gb/s/node), rail-optimized topology, and shared FS. If you only need 8 GPUs in one box, the on-demand 8× node is dramatically cheaper.

### 1.3 Private Cloud / Reserved (custom contract)

Not self-serve. Quote-only via sales. Public hints from third-party reporting:

- **GH200 reserved**: ~$1.49/GPU/hr (vs $2.29 on-demand) — ~35% discount. Source: gpuperhour.com/providers/lambda-labs and getdeploying.com aggregator, May 2026; not confirmed on Lambda's own page.
- **H100 SXM reserved**: ~$1.89/GPU/hr (1-year) reported by multiple aggregators (synpixcloud.com, spheron.network May 2026); ~37% discount from $2.99–$3.99 on-demand.
- **B200 reserved**: ~$3.79/GPU/hr (vs $6.69 on-demand) — ~43% discount per third-party reporting.
- **H200**: no public list price. Requires custom 1-year+ reservation. Third-party guesses are $4.99–$5.29/GPU/hr range.
- **Generic Lambda guidance**: 19–42% off on-demand for 1-year, up to ~27% on 3-year (per checkthat.ai/brands/lambda-labs/pricing, 2026).

**These reserved numbers are NOT on lambda.ai/pricing as of May 2026** — Lambda removed published reserved rates and routes all reserved deals through sales. The $1.49/$1.89/$3.79 figures come from aggregators and may be stale.

### 1.4 Monthly TCO scenarios (compute only, on-demand)

| Workload | $/month |
|---|---|
| 1× GH200, 24/7 | $1,672 |
| 8× H100 SXM, 24/7 | $23,302 |
| 8× B200 SXM, 24/7 | $39,070 |
| 16× H100 1-Click (smallest cluster) | $71,949 |
| 16× B200 1-Click | $115,165 |

For 1× GH200, the user's apparent target, **on-demand at $2.29/hr = $1,672/mo** is the visible rate; **~$1.49/hr ≈ $1,088/mo** is the rumored 1-year-reserved rate but is sales-quoted only.

---

## 2. Storage, Filesystems, Egress

- **Persistent filesystem**: $0.20/GiB/month, billed per hour, no minimum duration. Continues to bill even when unmounted. Source: lambda.ai/blog/persistent-storage-for-lambda-cloud-is-expanding (orig. 2023, rate still cited in 2026 reports including thundercompute.com vs-lambda blog and gpucloudlist.com 2026 review).
- **Regional binding**: filesystem in `us-west-1` cannot attach to an instance in `us-east-1`. Per-region storage. (Source: docs.lambda.ai/public-cloud/filesystems/.)
- **Capacity**: most regions support up to 8 EB per filesystem; `us-south-1` (Texas) capped at 10 TB per filesystem.
- **Ingress / Egress**: **free**. No bandwidth fees on Lambda storage or compute. Confirmed on lambda.ai/1-click-clusters and the storage blog.
- **Local SSD on the instance**: free, included; 4 TiB on a 1× GH200, 22 TiB on an 8× H100 / 8× B200 node. Ephemeral — wiped on termination.

Practical monthly storage cost for typical inference setups:
- 1 TiB model weights + cache: $200/mo
- 10 TiB dataset: $2,000/mo
- 100 TiB: $20,000/mo (start asking for volume discount)

---

## 3. SLA, Uptime, Reliability

- **Marketed SLA**: 99.9% uptime SLA (cited in multiple 2026 reviews — gpucloudlist.com, runpod.io/articles/alternatives/lambda-labs, promptquorum.com). The 99.9% number is **not a self-serve published doc** like AWS's SLA page; it appears in enterprise contracts and SOC 2 reports. There is no public SLA URL on lambda.ai. A 2024 deeptalk.lambda.ai forum thread asked for a written SLA and got a "contact sales" answer.
- **Status page**: https://status.lambda.ai/ — currently "fully operational" as of fetch on 2026-05-25. Page tracks 10 components across 2 groups (Frontend, Infrastructure).
- **Recent notable incidents**:
  - **2025-03-25/26 us-south-1 network outage** — ~16 hours. Root cause: routine UPS maintenance disrupted cooling power, networking hardware throttled/powered down. Multiple hypervisors went down. Resolved 2025-03-26 20:37 with hard reboots. Source: status.lambda.ai/incidents/01JQ8F9JF7SS94Y9AKCR4448CV. **This is the canonical "Lambda outage" cited by users — 16h regional downtime is well above 99.9% SLA budget for that month in that region.**
  - **2026-03-24** Global Firewall Rulesets page not loading (control plane only).
  - **2026-04-03** Non-admin users unable to launch instances (control plane; existing instances unaffected).
- **Compliance**: SOC 2, GDPR, ISO 27001 (per gpuperhour.com/providers/lambda-labs, 2026).
- **Reputation**: 2026 reviews call out (a) capacity stockouts during peak demand for H100/H200, (b) "abrupt account suspensions and persistent FS deletion after minor payment processing issues" (thundercompute.com/blog/lambda-labs-vs-thunder-compute, 2026). The capacity issue is consistently flagged: a user can hit "temporarily unavailable" mid-project when scaling.

---

## 4. Lambda Stack & OS Images (May 2026)

Lambda Stack is the apt-repo + image bundle that gives you driver/CUDA/cuDNN/framework parity across workstation and cloud.

### 4.1 Versions shipped in May 2026 (from lambda.ai/lambda-stack-deep-learning-software, fetched 2026-05-25)

| Component | Version |
|---|---|
| CUDA Toolkit | 12.8.93 (12.8 Update 1) |
| PyTorch | 2.7.0 |
| TensorFlow | 2.19.0 |
| Keras | 3.10.0 |
| NCCL | 2.26.2 |
| NVIDIA Container Toolkit | 1.18.1 |
| cuDNN | **9.x** (page doesn't pin a number; 2026 reviews confirm cuDNN 9 line) |
| NVIDIA Driver | **580.xx series** (per dasroot.net Jan 2026 + community refs; not pinned on Lambda's marketing page) |
| Linux kernel | **HWE 6.8 on Ubuntu 22.04 / generic 6.11+ on Ubuntu 24.04** (inferred from upstream HWE; Lambda doesn't publish a kernel pin) |

### 4.2 Ubuntu version support

- **Supported by Lambda Stack apt repo (May 2026)**: Ubuntu **20.04** (Focal), **22.04** (Jammy), **24.04** (Noble). Confirmed by github.com/lambdal/lambda-stack-dockerfiles README ("only current LTS versions are supported: 20.04, 22.04, and 24.04") and lambda.ai/lambda-stack-deep-learning-software.
- **Recovery images (workstation)** dated 20241108 cover Focal + Jammy only. (docs.lambda.ai/education/linux-usage/lambda-stack-and-recovery-images/)
- **Cloud default image**: Ubuntu 22.04 with Lambda Stack remains the default boot image on most instance launches in May 2026. Ubuntu 24.04 is selectable on newer SKUs (GH200, B200 nodes).
- **Ubuntu 26.04 (Resolute Raccoon)**: released April 2026 with Linux 7.0 (ubuntuhandbook.org/2026/03, phoronix Ubuntu 26.04 LTS coverage). **Not available on Lambda yet as of 2026-05-25.** No 26.04 image in the console, no apt-repo target in lambda-stack-dockerfiles. Expect a 3–6 month lag for Lambda Stack on 26.04 based on historical cadence (Noble took ~5 months post-release to land in Lambda Stack).

### 4.3 Practical gotchas for the image

- Lambda Stack pins **one** CUDA major.minor at a time. You cannot apt-install CUDA 12.6 alongside 12.8; you switch versions or use containers. Source: deeptalk.lambda.ai/t/possible-to-upgrade-cuda-in-the-lambda-stack/4785.
- For GH200 (aarch64 Grace), Lambda Stack ships arm64 builds of PyTorch / CUDA. Confirmed Lambda Stack supports the GH200 arm64 image. Some third-party wheels (e.g., older Triton, certain xformers builds) may not have arm64 binaries.
- The container toolkit is bundled, so docker/NVIDIA-runtime works out of the box on cloud images.

---

## 5. Contract Terms & Gotchas

1. **1-Click Clusters: 2-week minimum, pay-or-forfeit within 10 days** of invoice. No automatic renewal; reservation is granted only after Lambda approval.
2. **Reserved/Private Cloud: non-cancelable, non-refundable** (per checkthat.ai). 1-year and 3-year terms typical. No public price list — sales quote per deal.
3. **Refunds are credit-only**, expire 12 months. (docs.lambda.ai/public-cloud/billing/)
4. **No spot pricing**, no preemptible tier — all instances are guaranteed-uptime hourly. (gpuperhour.com/providers/lambda-labs)
5. **No published written SLA** with service-credit math like AWS — 99.9% is in marketing copy, but enforcement = contract-only.
6. **Capacity is the real risk**: H100/H200 routinely sell out at peak. The user's 1× GH200 is a less-contested SKU (only 3 providers carry GH200 — Lambda, Vultr, CoreWeave) but capacity isn't infinite.
7. **Payment-failure FS deletion**: multiple 2026 reports cite filesystems being deleted after a card decline. Keep a backup outside Lambda for critical state.
8. **Regions**: Lambda uses AWS-style names (us-south-1 Texas, us-east-1, us-west-*). New 2026 datacenters: Kansas City (24 MW, early 2026), DFW-04 (425k sq ft, completing 2026), Chicago (EdgeConneX, 23 MW, 2026), Houston ECL TerraSite-TX1. (datacenterdynamics.com Lambda coverage, lambda.ai/blog/lambda-edgeconnex-dc-pr).

---

## 6. Comparison: What's your $/mo for 1× GH200 vs alternatives?

| Provider | 1× GH200 on-demand $/hr | $/mo (730h) |
|---|---|---|
| **Lambda** | $2.29 | $1,672 |
| Vultr | $1.99 | $1,453 |
| CoreWeave | $6.50 | $4,745 |

Source: getdeploying.com/gpus/nvidia-gh200 (updated 2026-05-24). Lambda is mid-market on GH200 — Vultr is cheaper, CoreWeave is enterprise-priced.

---

## 7. Quick-Reference Source URLs (with fetch dates)

- https://lambda.ai/pricing (2026-05-25)
- https://lambda.ai/1-click-clusters (2026-05-25)
- https://lambda.ai/instances (2026-05-25)
- https://lambda.ai/lambda-stack-deep-learning-software (2026-05-25)
- https://lambda.ai/blog/persistent-storage-for-lambda-cloud-is-expanding (orig. Sep 2023; rate confirmed 2026)
- https://docs.lambda.ai/public-cloud/filesystems/ (2026-05-25)
- https://docs.lambda.ai/public-cloud/billing/ (2026-05-25)
- https://docs.lambda.ai/education/linux-usage/lambda-stack-and-recovery-images/ (2026-05-25)
- https://status.lambda.ai/ (2026-05-25)
- https://status.lambda.ai/incidents/01JQ8F9JF7SS94Y9AKCR4448CV (2025-03 us-south-1 incident)
- https://github.com/lambdal/lambda-stack-dockerfiles (2026-05-25)
- https://computeprices.com/providers/lambda (updated 2026-05-24)
- https://getdeploying.com/gpus/nvidia-gh200 (updated 2026-05-24)
- https://getdeploying.com/lambda-labs
- https://gpuperhour.com/providers/lambda-labs (2026)
- https://www.thundercompute.com/blog/nvidia-h200-pricing (May 2026)
- https://www.thundercompute.com/blog/lambda-labs-vs-thunder-compute (2026)
- https://www.spheron.network/blog/lambda-cloud-h100-pricing-2026/
- https://www.synpixcloud.com/blog/lambda-labs-gpu-pricing-2026
- https://www.gpucloudlist.com/en/blog/lambda-labs-review-2026
- https://www.businesswire.com/news/home/20250318150707/en/ (B200 GA, March 2025)
- https://lambda.ai/blog/be-first-scale-fast-blackwell-clusters-now-live-on-lambda

---

## 8. Confidence & Gaps

### High confidence
- On-demand prices (1× GH200 $2.29/hr, 8× H100 $3.99/hr/GPU, 8× B200 $6.69/hr/GPU): scraped directly from lambda.ai/pricing today, cross-confirmed by 2 aggregators with same-day timestamps.
- 1-Click Cluster pricing ($6.16–$5.54 H100, $9.86–$8.87 B200): on lambda.ai/1-click-clusters today.
- Persistent storage $0.20/GiB/mo, no egress: consistent across 3+ 2026 sources.
- Lambda Stack package versions (CUDA 12.8.93, PyTorch 2.7.0, TF 2.19.0, NCCL 2.26.2): direct from lambda.ai/lambda-stack-deep-learning-software.
- Ubuntu 26.04 NOT yet supported on Lambda: confirmed by GitHub repo and docs.
- 2-week 1-Click min commitment, 10-day pay-or-forfeit: from docs.lambda.ai/public-cloud/billing.

### Medium confidence
- Reserved rates ($1.49 GH200, $1.89 H100, $3.79 B200): from third-party aggregators (gpuperhour, synpixcloud). Lambda removed reserved pricing from their public page; sales-quote only. **Treat as indicative, not contractual.**
- 99.9% SLA figure: cited in multiple 2026 reviews but no public written SLA URL. Enterprise contract only.
- Driver 580.xx and kernel 6.8/6.11: inferred from upstream Ubuntu HWE pinning + community references; Lambda doesn't publish a pinned version table.
- cuDNN version (9.x): consistent with CUDA 12.8 era and 2026 reviews, but lambda.ai's stack page does not enumerate a specific cuDNN micro-version.

### Gaps / unknowns
- **No firm H200 list price.** Lambda does not sell H200 self-serve. The $3.79/hr H200 number in thundercompute's blog and the $4.99–$5.29/hr range elsewhere are not visible on lambda.ai; this is a known unknown that requires direct sales contact.
- **3-year reserved rates** for any SKU: no aggregator has numbers; sales only.
- **Volume / annual storage discounts**: $0.20/GiB is the catalog rate; 100+ TiB customers may negotiate, but no public tier.
- **Regional pricing differentiation**: Lambda does not appear to charge differently per region for compute, but storage availability and instance availability vary by region (Utah was last to get persistent storage; us-south-1 has 10 TB filesystem cap).
- **Lambda Stack on Ubuntu 26.04 / kernel 7.0**: not available. ETA unknown; historical cadence suggests Q3-Q4 2026.
- **Exact 1× GH200 reserved monthly price**: best estimate $1,088/mo (≈ $1.49 × 730), unconfirmed.
- **Egress on 1-Click Clusters**: marketed as no-egress-fee; not verified against an enterprise contract.
- **SLA service credit math**: unknown — no published formula.

### Decision-relevant notes for the user
- If the workload is 1× GH200 24/7 inference, **on-demand Lambda = $1,672/mo**. To beat that, either (a) negotiate reserved (~$1,088/mo target, sales call required), or (b) consider Vultr at $1,453/mo on-demand (cheaper but different SLA/stack profile — out of scope for this file).
- Lambda's 16h us-south-1 outage in March 2025 is the worst recent event on record. If single-region single-host is unacceptable, you need multi-region or off-Lambda DR.
- Lambda Stack on the GH200 is mature (arm64 PyTorch 2.7, CUDA 12.8). No image upgrade needed for current workloads.
- Ubuntu 26.04 is not an option here yet; if you need kernel 7.0 features, you stay on 24.04 + HWE on Lambda for now.
