# Marketplace & Secondary GPU Clouds — Monthly Inference Rentals

**Research agent #17 of 20** — Round 1
**Date:** 2026-05-25
**Scope:** RunPod, Vast.ai, TensorDock, Hyperbolic, Massed Compute, Nebius, Latitude.sh, FluidStack, Voltage Park, Atlantic.Net, GMI Cloud, Shadeform (aggregator), plus relevant cross-listings (Vultr, Lambda for GH200).

All monthly costs computed at **720 hours** unless otherwise noted. Quoted hourly rates are public list as of May 2026; reservation/committed rates may not always reflect what sales will offer.

---

## 1. Headline Findings

- **GH200 is scarce on marketplaces.** Only three direct providers: Vultr ($1.99/hr), Lambda Labs ($2.29/hr), CoreWeave ($6.50/hr). Sesterce, Denvr, Latitude.sh also listed through Shadeform-style aggregators at $2.52–$4.23/hr. **RunPod, Vast.ai, TensorDock, Hyperbolic, Massed Compute, FluidStack, Voltage Park, Atlantic.Net do NOT list GH200 publicly** as of May 2026.
- **Cheapest GH200 (1×) monthly:** Vultr at ~$1,433/mo on-demand; Vultr offers a documented 20% monthly-commit / +10% annual-commit discount, dropping effective rate to roughly **~$1,030/mo with annual commit**.
- **Cheapest H200 (1×) monthly:** Nebius preemptible at $1.45/hr (~$1,044/mo) or Nebius committed 3+ months at $2.30/hr (~$1,656/mo). FluidStack's listed $2.30/hr via getdeploying matches this floor but requires contact-sales.
- **Cheapest 8×H200 monthly:** Massed Compute at $22.64/hr (~$16,300/mo) — fully self-serve, no quote required. Crusoe lists $4.29/hr per GPU on 8×H200 (~$24,710/mo). Sesterce shows $2.09/hr per GPU on a 4×H200 instance.
- **Cheapest 8×H100 monthly:** Voltage Park at $1.99/hr per GPU on the Ethernet variant (~$11,462/mo for 8 GPUs, self-serve) — same provider with 3.2 Tbps Infiniband is $2.49/hr (~$14,342/mo). Vultr lists 8×H100 at $23.92/hr (~$17,222/mo).
- **Container vs bare metal split:** RunPod, Vast.ai, Hyperbolic, Massed Compute (VM tier) are **container-only** with limited root inside the container; Voltage Park, Latitude.sh, FluidStack, Nebius give **bare metal or full VM** with proper root and OS image flexibility.

---

## 2. Pricing Table — Sorted by Monthly Cost

### 2.1 Single GH200 (per GPU, on-demand list)

| Provider | $/hr | $/mo (720h) | Notes | Source |
|---|---|---|---|---|
| Vultr | $1.99 | $1,433 | 96GB; only 1 region (Atlanta) in stock May 2026, others sold out; 20% monthly + 10% annual commit discount available | [getdeploying GH200](https://getdeploying.com/gpus/nvidia-gh200), [gpuperhour GH200](https://gpuperhour.com/rent/gh200) |
| Lambda Labs | $2.29 | $1,649 | Documented availability constraints during peaks | [computeprices GH200](https://computeprices.com/gpus/gh200) |
| Sesterce | $2.52 | $1,814 | Via Shadeform API | [computeprices GH200](https://computeprices.com/gpus/gh200) |
| Denvr Dataworks | $3.87 | $2,786 | 7.6 TB local storage, Virginia | [gpuperhour GH200](https://gpuperhour.com/rent/gh200) |
| Latitude.sh | $4.23 | $3,046 | Via Shadeform; not on Latitude's own public pricing page | [computeprices GH200](https://computeprices.com/gpus/gh200) |
| CoreWeave | $6.50 | $4,680 | Premium tier; managed cluster context | [getdeploying GH200](https://getdeploying.com/gpus/nvidia-gh200) |

**RunPod, Vast.ai, TensorDock, Hyperbolic, Massed Compute, FluidStack, Voltage Park, Atlantic.Net, GMI Cloud, Nebius** — no GH200 listed publicly.

### 2.2 Single H200 (per GPU)

| Provider | $/hr | $/mo (720h) | Tier / Notes | Source |
|---|---|---|---|---|
| Nebius (preemptible) | $1.45 | $1,044 | Spot-style, interruptible | [Nebius pricing](https://nebius.com/prices) |
| Sesterce (4×H200) | $2.09 | $1,505 | Per-GPU rate on 4× instance | [getdeploying H200](https://getdeploying.com/gpus/nvidia-h200) |
| Theta EdgeCloud | $2.29 | $1,649 | Decentralized | [getdeploying H200](https://getdeploying.com/gpus/nvidia-h200) |
| FluidStack | $2.30 | $1,656 | Listed rate; contact-sales required | [getdeploying H200](https://getdeploying.com/gpus/nvidia-h200) |
| Nebius (3+ month commit) | $2.30 | $1,656 | "Lowest publicly found H200 committed rate" | [DeployBase Nebius](https://deploybase.ai/articles/nebius-gpu-cloud-pricing-complete-guide-vs-hr-for-every-gpu) |
| GMI Cloud | $2.50–$2.60 | $1,800–$1,872 | Reserved tier available | [GMI Cloud pricing](https://www.gmicloud.ai/en/pricing) |
| Massed Compute (H200 NVL) | $2.83 | $2,038 | Self-serve, no commits | [Massed Compute pricing](https://vm.massedcompute.com/pricing) |
| RunPod (1-year commit) | $3.05 | $2,196 | Savings Plan tier | [runpodreferral blog](https://runpodreferral.com/blog/runpod-vs-lambda-labs) |
| Lyceum | $3.19 | $2,297 | | [getdeploying H200](https://getdeploying.com/gpus/nvidia-h200) |
| Verda | $3.39 | $2,441 | Via Shadeform | [getdeploying H200](https://getdeploying.com/gpus/nvidia-h200) |
| DigitalOcean | $3.44 | $2,477 | | [getdeploying H200](https://getdeploying.com/gpus/nvidia-h200) |
| Civo | $3.49 | $2,513 | | [getdeploying H200](https://getdeploying.com/gpus/nvidia-h200) |
| Hyperstack | $3.50 | $2,520 | | [getdeploying H200](https://getdeploying.com/gpus/nvidia-h200) |
| Nebius (on-demand) | $3.50 | $2,520 | | [Nebius pricing](https://nebius.com/prices) |
| RunPod Community | $3.59 | $2,585 | 141GB SXM | [RunPod H200](https://www.runpod.io/gpu-models/h200) |
| Vast.ai H200 NVL | $3.95 | $2,844 | Marketplace floor | [vast.ai H200](https://vast.ai/pricing/gpu/H200) |
| Oblivus | $3.99 | $2,873 | | [getdeploying H200](https://getdeploying.com/gpus/nvidia-h200) |
| RunPod Secure | $4.39 | $3,161 | T3/T4 datacenter | [RunPod H200](https://www.runpod.io/gpu-models/h200) |
| Together | $4.99 | $3,593 | | [getdeploying H200](https://getdeploying.com/gpus/nvidia-h200) |
| CoreWeave | $6.31 | $4,543 | | [getdeploying H200](https://getdeploying.com/gpus/nvidia-h200) |

### 2.3 Single H100 (per GPU)

| Provider | $/hr | $/mo (720h) | Notes | Source |
|---|---|---|---|---|
| Vast.ai (unverified) | $0.90 | $648 | Interruptible/hobbyist hosts | [intuitionlabs H100](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison) |
| Nebius (preemptible) | $1.25 | $900 | | [Nebius pricing](https://nebius.com/prices) |
| RunPod Community spot | $1.49 | $1,073 | | [runpod.io/pricing](https://www.runpod.io/pricing) |
| Hyperbolic H100 SXM | $1.49 | $1,073 | Quoted in Apr 2026 | [Hyperbolic on-demand](https://www.hyperbolic.ai/blog/introducing-hyperbolic-on-demand-cloud) |
| Latitude.sh H100 VM | $1.57 | $1,149/mo flat | Listed as $1,149/mo flat-rate by Latitude | [Latitude pricing](https://www.latitude.sh/pricing) |
| Latitude.sh H100 Metal | $1.68 | $1,230/mo flat | Bare metal | [Latitude pricing](https://www.latitude.sh/pricing) |
| TensorDock spot | $1.60–$1.91 | $1,152–$1,375 | | [TensorDock H100](https://www.tensordock.com/gpu-h100.html) |
| Vast.ai verified DC | $1.87 | $1,346 | ISO27001 hosts | [intuitionlabs H100](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison) |
| Voltage Park Ethernet | $1.99 | $1,433 | Self-serve, 8× HGX node | [VP pricing](https://www.voltagepark.com/pricing) |
| RunPod Community PCIe | $1.99 | $1,433 | | [RunPod H100 PCIe](https://www.runpod.io/gpu-models/h100-pcie) |
| GMI Cloud H100 PCIe | $2.00 | $1,440 | | [GMI Cloud pricing](https://www.gmicloud.ai/en/pricing) |
| TensorDock H100 SXM | $2.25 | $1,620 | | [TensorDock H100](https://www.tensordock.com/gpu-h100.html) |
| Shadeform/Verda H100 SXM | $2.26 | $1,627 | | [Shadeform instances](https://shadeform.com/directory/instances) |
| Massed Compute H100 | $2.40 | $1,728 | Self-serve | [Massed Compute pricing](https://vm.massedcompute.com/pricing) |
| Voltage Park Infiniband | $2.49 | $1,793 | 3.2 Tbps, 8×min | [VP pricing](https://www.voltagepark.com/pricing) |
| RunPod H100 NVL | $2.59 | $1,865 | 94GB variant | [runpod.io/pricing](https://www.runpod.io/pricing) |
| RunPod H100 SXM | $2.69 | $1,937 | Community | [runpod.io/pricing](https://www.runpod.io/pricing) |
| RunPod Secure H100 PCIe | $2.89 | $2,081 | T3/T4 | [RunPod H100 PCIe](https://www.runpod.io/gpu-models/h100-pcie) |
| Nebius H100 on-demand | $2.95 | $2,124 | | [Nebius pricing](https://nebius.com/prices) |
| Hyperbolic H100 (older list) | $3.20 | $2,304 | | [costbench Hyperbolic](https://costbench.com/software/ai-gpu-cloud/hyperbolic/) |
| Latitude.sh published H100 | $3.37 | $2,426 | from review article | [Spheron Latitude alternatives](https://www.spheron.network/blog/latitude-alternatives/) |

### 2.4 8× Clusters (full node price/mo)

| Provider | GPU | Total $/hr | $/mo (720h) | Notes | Source |
|---|---|---|---|---|---|
| Voltage Park 8×H100 Eth | H100 | $15.92 | $11,462 | $1.99/hr × 8, self-serve | [VP pricing](https://www.voltagepark.com/pricing) |
| Voltage Park 8×H100 IB | H100 | $19.92 | $14,342 | 3.2 Tbps Infiniband | [VP pricing](https://www.voltagepark.com/pricing) |
| Massed Compute 8×H100 | H100 | quote | — | "available upon request" | [Massed Compute pricing](https://vm.massedcompute.com/pricing) |
| Vultr 8×H100 | H100 | $23.92 | $17,222 | | [getdeploying H100](https://getdeploying.com/gpus/nvidia-h100) |
| Massed Compute 8×H200 NVL | H200 | $22.64 | $16,300 | Self-serve | [Massed Compute pricing](https://vm.massedcompute.com/pricing) |
| Crusoe 8×H200 | H200 | $34.32 | $24,710 | $4.29/GPU/hr | [getdeploying H200](https://getdeploying.com/gpus/nvidia-h200) |
| RunPod H200 SXM cluster | H200 | $4.31 × 8 = $34.48 | $24,826 | Per-GPU cluster rate | [runpod.io/pricing](https://www.runpod.io/pricing) |

---

## 3. Per-Provider Detail

### 3.1 RunPod (Secure + Community + Reserved)
- **Public list (May 2026):** H100 PCIe $1.99 (community)/$2.89 (secure), H100 SXM $2.69, H100 NVL $2.59, H200 SXM $3.59/$4.39, H200 cluster $4.31/GPU. **GH200 not offered.**
- **Reservation:** "Savings Plan" — H200 at $3.05/hr on 1-year commit; rest is sales-quote.
- **Storage:** Container disk $0.10/GB/mo; network volume $0.05–$0.07/GB/mo standard, $0.14/GB/mo high-perf.
- **Root/OS:** Container-only (Docker). User chooses public image; effectively root inside container, but no kernel modules. Ubuntu 20/22/24 templates standard.
- **Reliability:** 99.9% claim but StatusGator/uptime page shows 210+ incidents in 8 months, ~34 min outage on 2026-05-08; Community Cloud depends on third-party hosts. SOC2 Type II achieved Oct 2025 for Secure tier.
- **Gotchas:** Payment declines on prepaid cards (Stripe minimums); Mar 2026 Reddit report of recurring payment failure; storage purged on zero balance.
- Sources: [RunPod pricing](https://www.runpod.io/pricing), [costbench](https://costbench.com/software/ai-gpu-cloud/runpod/), [statusgator](https://statusgator.com/services/runpod), [Medium review](https://medium.com/@dankdarkhorse/runpod-io-the-gpu-cloud-that-accidentally-got-it-right-617ab468eed6).

### 3.2 Vast.ai
- **Public list:** H100 SXM from $1.49/hr (marketplace floor), $1.87/hr at verified datacenter hosts; H100 NVL $1.69/hr; H200 from $3.95/hr; **no GH200**.
- **Tiers:** On-demand (per-second), Interruptible (50%+ cheaper), Reserved (1/3/6 month, "up to 50% off").
- **Root/OS:** Docker container model. User picks any Docker image (Ubuntu 20/22/24 supported). Root inside container; host kernel fixed. Build args allow custom Python/CUDA base.
- **Reliability:** ISO 27001 verified hosts in "Secure Cloud" filter; unverified marketplace hosts have documented disconnection risk. Clean platform-level security record since 2018.
- **Gotchas:** Cheaper unverified hosts can yank instances; SSH key apply issues reported; pricing fluctuates per supply/demand; no fixed monthly contract beyond 6-month reserved.
- Sources: [vast.ai pricing](https://vast.ai/pricing), [vast.ai H100-SXM](https://vast.ai/pricing/gpu/H100-SXM), [Trustpilot reviews](https://www.trustpilot.com/review/vast.ai), [vast docker docs](https://vast.ai/docs/renting/edit-docker-image-and-configuration).

### 3.3 TensorDock
- **Public list:** H100 SXM5 from $2.25/hr on-demand, $1.91/hr spot. **No H200 or GH200 on public pricing page** (page last-updated date Jul 2024 per the cloud-gpus.html footer).
- **Monthly:** No native monthly plan; pay-as-you-go from deposited balance.
- **Root/OS:** VM-based with full root access. Ubuntu 22/24 images standard.
- **Reliability:** **Heavy negative signal on Trustpilot/LowEndTalk** — reports of GPUs crashing and not restarting, support nonresponsive, instances missing advertised GPUs (RTX 5090 listings ended up with lesser cards), VMs deploying inaccessible via SSH while billing continues, accounts unable to remove payment methods.
- **Gotchas:** Reputation has degraded materially in 2024–2025; treat as a price-only option. No 8× cluster pricing public.
- Sources: [TensorDock H100](https://www.tensordock.com/gpu-h100.html), [LowEndTalk thread](https://lowendtalk.com/discussion/205285/is-tensordock-reliable-for-cloud-gpus), [Trustpilot](https://www.trustpilot.com/review/tensordock.com).

### 3.4 Hyperbolic
- **Public list:** Per Hyperbolic's own blog: H100 SXM $1.49/hr, H200 $2.15/hr. Other comparison sites quote $3.20/hr H100 SXM — large discrepancy suggesting list price moved down in 2026. **No GH200.**
- **Tiers:** On-demand (multi-tenant), Reserved (isolated dedicated capacity, contact sales). OpenAI-compatible inference API also offered.
- **Root/OS:** SSH access to instances with prebuilt images. Container-style multi-tenant on the on-demand tier.
- **Reliability:** Decentralized "Hyper-dOS" model — trust/consistency tradeoffs vs centralized clouds. No published SOC2/ISO at agent date.
- **Gotchas:** Decentralized architecture means hardware/network heterogeneity. Price list moves fast.
- Sources: [Hyperbolic on-demand blog](https://www.hyperbolic.ai/blog/introducing-hyperbolic-on-demand-cloud), [thundercompute Hyperbolic comparison](https://www.thundercompute.com/blog/thunder-compute-vs-hyperbolic-gpu-cloud), [Northflank alternatives](https://northflank.com/blog/hyperbolic-ai-alternatives).

### 3.5 Massed Compute
- **Public list (self-serve, on-demand):** H100 80GB $2.40/hr; 2× $4.80; 4× $9.60; **8× contact sales**. H200 NVL $2.83/hr; 2× $5.66; 4× $11.32; **8× $22.64/hr ($16,300/mo)** — one of very few self-serve 8×H200 prices. A100 80GB $1.25/hr / 8× $10/hr.
- **Tiers:** VM (desktop-style) + bare metal. No monthly subscription published; commits via "contact us".
- **Root/OS:** Full root on bare metal; VM tier has desktop GUI option. Ubuntu/CentOS images.
- **Reliability:** Tier III datacenters; SOC 2 Type II; positive reputation re: stability and support. NVIDIA Preferred Partner.
- **Gotchas:** No GH200; 8×H100 not self-serve; rates are list, not negotiated.
- Sources: [Massed Compute pricing](https://vm.massedcompute.com/pricing), [koonka review](https://koonka.ai/massed-compute-review-is-this-cloud-gpu-provider-worth-it/), [CIOReview profile](https://www.cioreview.com/massed-compute-2025).

### 3.6 Nebius (ex-Yandex)
- **Public list:** H100 $1.25 preemptible / $2.95 on-demand. H200 $1.45 preemptible / $3.50 on-demand. Committed 3+ months as low as $2.30/hr (H200). Up to 35% off on-demand for large reserved clusters. **No GH200.**
- **Networking:** InfiniBand up to 3.2 Tbit/s on multi-node setups; free egress.
- **Storage:** $0.0147–$0.118/GiB/mo depending on tier.
- **Root/OS:** Full VM with root, image flexibility implied but not confirmed on the prices page.
- **Reliability:** Gartner Peer Insights positive; users report stable training on H200; NBIS publicly traded since 2024 spinoff. Lags T1 hyperscalers on SLA commitments.
- **Gotchas:** GH200 not offered; H200 inventory new as of early 2026.
- Sources: [Nebius pricing](https://nebius.com/prices), [Nebius H200](https://nebius.com/h200), [DeployBase Nebius review](https://deploybase.ai/articles/nebius-review-2026-pricing-performance-pros-cons), [Spheron Nebius alternatives](https://www.spheron.network/blog/nebius-alternatives/).

### 3.7 Latitude.sh
- **Public list:** H100 80GB Metal $1.68/hr → **$1,230/mo flat**; H100 80GB VM $1.57/hr → **$1,149/mo flat**. **H200 not on public page** as of May 2026; **GH200 listed via Shadeform aggregator at $4.23/hr** but not advertised on Latitude's own page. H100 SXM not offered (only PCIe).
- **Storage/Networking:** 2× 3.8TB NVMe + 2×10 Gbps (Metal); 500 GB NVMe + 10 Gbps (VM); 20 TB free egress.
- **Root/OS:** Bare metal with full root, no hypervisor; Ubuntu/Debian/RHEL images, custom ISO supported.
- **Reliability:** Positive G2 reviews; dedicated machines no noisy neighbors. Provision takes 3–10 min (vs 30–60s VM).
- **Gotchas:** Only H100 PCIe + RTX 6000 Ada + L40S in catalog. No H100 SXM. Premium vs marketplaces.
- Sources: [Latitude pricing](https://www.latitude.sh/pricing), [Latitude features](https://www.latitude.sh/features), [BestGPUCloud review](https://www.bestgpucloud.com/en/blog/latitude-sh-review-2026), [Spheron Latitude alternatives](https://www.spheron.network/blog/latitude-alternatives/).

### 3.8 FluidStack
- **Public list:** **No published pricing.** Everything is "Request Pricing." Listed by getdeploying at $2.30/hr for H200 (likely reserved indicative).
- **Tiers:** Reserved clusters (256–10k+ GPUs, ≥30 days); On-demand (8–4k+ GPUs); Private Cloud.
- **Networking:** InfiniBand fabric, 95%+ of theoretical perf validated.
- **Compliance:** SOC 2 Type 2, HIPAA, GDPR, ISO 27001. 99% uptime, 15-min response.
- **Reliability:** Strong; closed $200M Series A Feb 2025; in talks for $1B at $18B valuation Apr 2026. Real enterprise customer base (Mistral, etc.).
- **Gotchas:** Narrow self-serve; 30-day minimum reservation is the floor; "access to best configs often requires going through sales." No public 1×GPU monthly rate. No GH200 listed.
- Sources: [FluidStack site](https://www.fluidstack.io/), [getdeploying FluidStack](https://getdeploying.com/fluidstack), [Sacra equity research](https://sacra.com/c/fluidstack/), [SlashDot reviews](https://slashdot.org/software/p/FluidStack/).

### 3.9 Voltage Park
- **Public list:** H100 Ethernet $1.99/hr (1–1016 GPUs, self-serve, 15-min provisioning). H100 Infiniband (3.2 Tbps Quantum-2) $2.49/hr (8–1016 GPUs). **H200/GH200/B200/B300 on term contracts only — contact sales.**
- **Tiers:** Self-serve on-demand (no minimum term) + long-term reserved (6+ months, 32–8000+ GPUs).
- **Compliance:** SOC 2 Type II, HIPAA, BGP multihoming, Palo Alto firewalls.
- **Root/OS:** Bare-metal access; Dell PowerEdge XE9680 nodes with 8× H100 SXM5, 1TB RAM, dual Xeon. Ubuntu images standard with full root.
- **Reliability:** Strong reputation — VAST Data case studies show workloads "running uninterrupted for weeks, sometimes months." Non-profit ownership (Navigation Fund). 7 Tier 3+ DCs.
- **Gotchas:** H200/Blackwell only via term contracts (not self-serve as of May 2026). No public Reddit reliability complaints — but also less marketplace-style buzz.
- Sources: [Voltage Park pricing](https://www.voltagepark.com/pricing), [VP H100 product](https://www.voltagepark.com/product/cloud-h100), [VAST blog](https://www.vastdata.com/blog/a-closer-look-at-voltage-parks-billion-dollar-ai-infrastructure).

### 3.10 Atlantic.Net
- **Public list:** Catalog includes L40S and **H100 NVL** (dual-GPU 94GB HBM3). **Per-GPU hourly/monthly pricing NOT public** — billed via configurator/sales. No H200, no GH200, no H100 SXM.
- **Tiers:** Dedicated GPU servers + "GPU as a Service." Reserved 1-yr / 3-yr terms.
- **Root/OS:** Full root, OS choice (Ubuntu/CentOS/Windows).
- **Reliability:** 100% uptime SLA claimed; 24/7 support included. Long-running hosting biz (since 1994), more conservative customer base.
- **Gotchas:** Niche for AI; thin catalog vs marketplace clouds; opaque on AI GPU pricing.
- Sources: [Atlantic.Net GPU cloud](https://www.atlantic.net/gpu-server-hosting/gpu-cloud-hosting/), [Atlantic.Net guide](https://www.atlantic.net/gpu-server-hosting/a-guide-to-gpu-server-hosting-options/).

### 3.11 GMI Cloud
- **Public list:** H100 PCIe from $2.00/GPU-hr, H100 SXM $2.40, H200 $2.50–$2.60, B200 $4.00, GB200 $8.00 (no GH200 listed; GB200 ≠ GH200).
- **Tiers:** On-demand + Reserved with commitment discounts; "Usage-Adaptive" transition supported.
- **Compliance/Hardware:** Dedicated single-tenant; transparent no-throttle claim.
- **Gotchas:** No public monthly rate (hourly only); networking and root access details not public.
- Sources: [GMI Cloud pricing](https://www.gmicloud.ai/en/pricing), [GMI A100/H100/H200 comparison](https://www.gmicloud.ai/en/blog/gpu-cloud-pricing-a100-h100-h200).

### 3.12 Shadeform (aggregator)
- **Model:** No markup over direct provider rate; unified API/console across 21+ providers (Lambda, Nebius, Crusoe, Voltage Park, Verda, Sesterce, Denvr, Latitude.sh, etc.).
- **GPU breadth:** H100, H200, **GH200**, A100, L40S, RTX 4090/5090, B200, B300, GB200.
- **Visible aggregator rates:** Verda H100 SXM $2.26, B200 $5.63, B300 $6.59, Lambda B200 $5.29. Confirms GH200 routing through Lambda ($2.29) and Latitude.sh ($4.23) and Sesterce ($2.52).
- **Gotchas:** Final reliability/SLA depends on underlying provider; Shadeform sits in front of payment/auth.
- Sources: [Shadeform directory](https://shadeform.com/directory/instances), [Shadeform prices](https://shadeform.com/prices), [aimultiple comparison](https://aimultiple.com/gpu-marketplace).

---

## 4. Capability Matrix

| Provider | GH200 | H100 8× monthly | H200 8× monthly | Root model | InfiniBand | Self-serve 8× | Persistent storage | Compliance |
|---|---|---|---|---|---|---|---|---|
| RunPod | No | sales quote | $24,826 (cluster rate) | Container (Docker) | Limited | Yes (community) | Yes ($0.05–0.14/GB/mo) | SOC2 II (Oct 2025, Secure tier) |
| Vast.ai | No | varies / spot | varies / spot | Container (Docker) | Some hosts | Yes | Yes (host-dependent) | ISO 27001 (verified hosts) |
| TensorDock | No | not public | not public | VM, full root | Limited | No | Yes | Reputation concerns |
| Hyperbolic | No | not public | not public | SSH, prebuilt images | Limited | Yes | Yes | None published |
| Massed Compute | No | "contact us" | $16,300 (self-serve) | VM + bare metal | Available | 8×H200 yes; 8×H100 no | Yes | SOC 2 Type II |
| Nebius | No | quote | quote | VM, full root | 3.2 Tbit/s | Yes | $0.0147–0.118/GiB/mo | Lags T1 SLAs |
| Latitude.sh | via Shadeform | not public | not public | Bare metal, full root | Limited | Yes | NVMe, 20TB free egress | Yes |
| FluidStack | No | contact sales (≥30d) | contact sales (≥30d) | Bare metal | InfiniBand fabric | Sales | Yes | SOC2 II, HIPAA, GDPR, ISO 27001 |
| Voltage Park | No | $11,462 (Eth) / $14,342 (IB) | sales (term only) | Bare metal, full root | 3.2 Tbps Quantum-2 | Yes (H100 only) | Yes | SOC2 II, HIPAA |
| Atlantic.Net | No | not public | not offered | VM, full root | No | No | Yes | Long-running enterprise |
| GMI Cloud | No | not public | not public | Dedicated GPU | Yes (claim) | Yes | Not public | Not published |
| Vultr (cross-ref) | $1,433/mo | $17,222 | not offered | VM/bare metal, full root | Yes | Yes | Yes | Multiple |
| Lambda Labs (cross-ref) | $1,649/mo | quote | quote | VM, full root | Yes (Quantum-2) | Yes | Yes | SOC2 |

---

## 5. Ubuntu 24 / 26 Image Flexibility

- **Full custom OS / image:** Voltage Park, Latitude.sh, Nebius, FluidStack, Massed Compute (BM tier), Atlantic.Net, GMI Cloud, Vultr — bare metal or VM, can install Ubuntu 24.04 LTS or Ubuntu 26.04 (Resolute Raccoon, GA Apr 2026).
- **Docker container with Ubuntu base:** RunPod, Vast.ai, Hyperbolic — pick any Ubuntu-based Docker image (24.04 widely cached on hosts; 26.04 supported but slower first-pull). Kernel is host-fixed; no `modprobe`, no custom NIC drivers.
- **TensorDock:** VM with full root, Ubuntu 22/24 standard.

Ubuntu 26.04 LTS GA'd late April 2026; provider templates may lag — verify before relying on it. Source: [Ubuntu cloud-images](https://cloud-images.ubuntu.com/resolute/).

---

## 6. Gotchas Cross-Cut

- **Overbooking / inventory stockouts:** Vultr's GH200 sold out in Manchester and "Global" regions per gpuperhour tracker May 2026. Lambda Labs documents "occasional availability constraints during peak demand." RunPod Community Cloud reports of pods deploying on broken servers.
- **Sudden price changes:** Vast.ai pricing is supply/demand floating — quoted price is not contract. Hyperbolic public list moved from $3.20 → $1.49 for H100 SXM over 2025–2026. TensorDock's last-updated date on cloud-gpus.html still reads "July 24, 2024."
- **Payment / billing risk:** RunPod card declines on prepaid Stripe minimums; storage purged on zero balance. TensorDock complaints of inability to remove payment methods and unresponsive refunds. Vast.ai requires prepaid balance.
- **GH200 supply is genuinely thin:** Only 14 GH200 instances tracked across 3 direct providers as of May 25 2026 (gpuperhour). Lead time to buy new is 2–4 weeks. Expect monthly availability commits to be hard to get below ~$1,400/mo.
- **Self-serve vs sales-quote split:** RunPod, Vast, TensorDock, Hyperbolic, Massed Compute (≤4× tier), Voltage Park (H100), Vultr, Latitude.sh = self-serve. FluidStack, Voltage Park (H200/B200), GMI Cloud (commits), Nebius (large reserved), Massed Compute (8×H100, bare metal), RunPod (savings plan H100) = sales-quote required.
- **Container-only blocks some inference stacks:** RunPod/Vast/Hyperbolic can't run anything that needs kernel modules (custom NIC drivers, fabric-attached storage clients). Inference-only on stock CUDA stacks is fine.

---

## 7. Recommended Shortlist for GH200 Monthly Inference

1. **Vultr GH200** at $1,433/mo on-demand, ~$1,030/mo annual-committed — cheapest by a wide margin, but stockouts in 2/3 regions. Full root, real VM.
2. **Lambda Labs GH200** at $1,649/mo — reputable, SOC2, real engineering.
3. **Sesterce / Denvr** via Shadeform $1,814–$2,786/mo — alternative supply, less reputation data.
4. **CoreWeave $4,680/mo** — premium, enterprise-grade, but more than 3× the floor.

For **H200 monthly inference** specifically, the best secondary-cloud picks are **Nebius committed ($1,656/mo with 3-mo commit)**, **Massed Compute self-serve 8× ($16,300/mo for 8 GPUs)**, and **RunPod savings plan ($2,196/mo on 1-year)** depending on commitment tolerance.

---

## 8. Confidence & Gaps

**High confidence:**
- GH200 marketplace landscape (only Vultr/Lambda/CoreWeave directly + Sesterce/Denvr/Latitude.sh via aggregator). Pricing cross-confirmed across getdeploying, gpuperhour, computeprices, all updated within May 2026.
- RunPod & Voltage Park & Nebius & Latitude.sh public hourly rates (fetched from own pricing pages).
- Massed Compute self-serve hourly across 1×/2×/4×/8× — fetched direct.
- Reliability red flags on TensorDock (multiple independent negative reviews) and reliability strength of Voltage Park, FluidStack, Nebius.

**Medium confidence:**
- Hyperbolic H100/H200 list price — $1.49/$2.15 from Hyperbolic's own blog conflicts with $3.20 quoted on Thunder/costbench comparisons. Real rate likely now in $1.49–$2.15 range as of Apr–May 2026, but list may be dynamic.
- FluidStack $2.30/hr H200 — listed by getdeploying as published, but FluidStack's own page is contact-sales; cannot verify whether this is reserved-30d or true on-demand.
- Vast.ai H200 floor of $3.95/hr — quoted on vast.ai/pricing/gpu/H200 landing page text but actual marketplace prices were not fetchable from the page body.
- RunPod Savings Plan 1-year H200 at $3.05/hr — single source (runpodreferral.com); not on RunPod docs page.

**Gaps not closed:**
- True 8×H100 / 8×H200 reserved monthly contract pricing on FluidStack, GMI Cloud, RunPod Secure, Voltage Park H200 — all require sales contact, not in public scope.
- Atlantic.Net H100 NVL specific monthly price — fully behind configurator/sales.
- Hyperbolic compliance posture (no SOC2/ISO published as of agent date).
- Whether Vast.ai "Reserved 1/3/6 month" tier has firm SLA or remains marketplace-supply-dependent.
- Ubuntu 26.04 image availability per provider — generic in this report (Ubuntu 26.04 GA'd Apr 2026 but template lag is provider-specific).
- Reddit r/LocalLLaMA-specific reliability threads — search returned no direct hits; relied on Trustpilot/SlashDot/LowEndTalk/independent reviews instead.

**Date stamp:** All prices reflect listings updated between Apr 27 2026 and May 25 2026 per source pages. Marketplace pricing (Vast, RunPod Community, Hyperbolic spot) fluctuates daily.
