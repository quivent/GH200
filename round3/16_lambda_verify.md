# Lambda Labs — Round 3 Verification & Deepening (May 2026)

Round 3 agent #16. Verifying round1/16_lambda_pricing.md against fresh fetches 2026-05-25.
FP8 broken on this stack; bf16 default — context only.

---

## TL;DR — what changed, what was confirmed, what's wrong

| Claim from R1 | R3 status | Evidence |
|---|---|---|
| 1× GH200 on-demand $2.29/hr | CONFIRMED | lambda.ai/pricing direct fetch 2026-05-25 |
| 1× H100 SXM $4.29/hr, 8× $3.99/hr/GPU | CONFIRMED | lambda.ai/pricing |
| 1× B200 $6.99/hr, 8× $6.69/hr/GPU | CONFIRMED | lambda.ai/pricing |
| H100 PCIe 1× $3.29/hr | CONFIRMED | lambda.ai/pricing |
| **H200 not on Lambda's self-serve page** | CONFIRMED stronger | Direct re-fetch of /pricing and /instances — zero mention of H200 or 141GB. getdeploying H200 page (31 providers) does NOT list Lambda. |
| Third-party blogs cite Lambda H200 at $3.79/hr | DEBUNKED as stale/fabricated | thundercompute blog (data dated 2026-05-12) cites lambda.ai/instances as source — but /instances page contains no H200 SKU. Likely scraped a stale cluster price or a hallucinated entry. Do not use $3.79/hr H200 figure. |
| 1-Click Cluster H100 16/64/256 = $6.16/$5.85/$5.54 per GPU/hr | CONFIRMED | lambda.ai/1-click-clusters |
| 1-Click Cluster B200 = $9.86/$9.36/$8.87 per GPU/hr | CONFIRMED | lambda.ai/1-click-clusters |
| Reserved GH200 ~$1.49/hr | NOT VERIFIABLE — aggregators have removed this number | getdeploying.com/gpus/nvidia-gh200 shows "no reserved pricing displayed"; computeprices.com/providers/lambda shows on-demand only; gpuperhour.com/providers/lambda-labs no longer lists reserved rates. The $1.49 figure originated in 2024–2025 aggregator data and has no live anchor in May 2026. |
| Reserved H100 SXM ~$1.89/hr | PARTIAL — replaced with new estimates | spheron.network May 2026 post explicitly estimates reserved H100 SXM ~$3.19/hr (1yr), ~$2.95/hr (3yr) using historical 20%/26% discounts off the $4.29 on-demand. The $1.89 figure is stale (it traces to the era when H100 SXM on-demand was $2.49/hr). |
| Reserved B200 ~$3.79/hr | UNCONFIRMED — no live aggregator number | One search result mentions "$3.79/hr reserved B200" but cannot be traced to a sourced page. |
| Persistent FS $0.20/GiB/mo, no egress | CONFIRMED | getdeploying/lambda-labs, docs.lambda.ai/public-cloud/billing |
| 1-Click 2-week min, 10-day payment | CONFIRMED | docs.lambda.ai/public-cloud/billing |
| Lambda Stack: CUDA 12.8.93, PyTorch 2.7.0, TF 2.19.0, NCCL 2.26.2 | CONFIRMED | lambda.ai/lambda-stack-deep-learning-software direct fetch |
| Lambda Stack adds JAX 0.6.0 | NEW DATAPOINT | lambda.ai/lambda-stack-deep-learning-software (R1 didn't mention JAX) |
| Driver 580.xx | LIKELY WRONG — evidence points to driver 570 branch | Dockerfile master at github.com/lambdal/lambda-stack-dockerfiles uses `libcuda1-dummy Provides: libcuda1 (= 570)` — the dummy declares driver branch 570, not 580. Repo last pushed 2025-03-11 — Dockerfile is 14 months stale, but it matches CUDA 12.8 (CUDA 12.8 GA driver = 570.xx). Driver 580.xx would be CUDA 13.x. |
| Ubuntu 26.04 not on Lambda yet | CONFIRMED | github.com/lambdal/lambda-stack-dockerfiles README still lists "only current LTS: 20.04, 22.04, 24.04" |

---

## 1. On-demand pricing — direct verification

Direct fetch of https://lambda.ai/pricing 2026-05-25 confirms every line item from R1 to the cent, plus fills in the multi-GPU 2× and 4× tiers R1 collapsed:

| SKU | 1× | 2× | 4× | 8× | $/GPU/hr (full node) |
|---|---|---|---|---|---|
| B200 SXM6 180GB | $6.99 | $6.89 | $6.79 | $6.69 | $6.69 |
| H100 SXM 80GB | $4.29 | $4.19 | $4.09 | $3.99 | $3.99 |
| H100 PCIe 80GB | $3.29 | — | — | — | $3.29 |
| A100 SXM 80GB | — | — | — | $2.79 | $2.79 |
| A100 SXM 40GB | $1.99 | — | — | — | $1.99 |
| A100 PCIe 40GB | $1.99 | $1.99 | $1.99 | — | $1.99 |
| GH200 96GB | $2.29 | — | — | — | $2.29 (only sold as 1×) |
| A10 24GB | $1.29 | — | — | — | $1.29 |
| A6000 48GB | $1.09 | $1.09 | $1.09 | — | $1.09 |
| V100 16GB | — | — | — | $0.79 | $0.79 |
| Quadro RTX 6000 24GB | $0.69 | — | — | — | $0.69 |

**R1 note correction**: R1 said B200 had 192GB VRAM. Lambda's own page lists 180GB. The 192GB figure appears in some R1 entries (e.g., "1× B200 SXM6 192GB $6.99") — this is wrong on Lambda's marketing copy. Other Lambda pages say "180GB". The official B200 spec is 180GB HBM3e (NVIDIA confirms). 192GB shows up only in some Lambda aggregator copy. **Correct value: 180GB per Lambda's own pricing page.**

**Multi-GPU sliding scale** (R1 didn't capture): 1× → 8× is a ~$0.30/hr per-GPU discount. Trivial — not the same as the cluster discount.

---

## 2. H200 reality check

R1 said: "H200 NOT in on-demand SKU list; third-party blogs say $3.79/hr — treat with caution."

R3 verdict: **even stronger — Lambda does not sell H200 self-serve at all in May 2026.**

Evidence:
- lambda.ai/pricing: no H200.
- lambda.ai/instances: no H200, no 141GB.
- lambda.ai/1-click-clusters: only H100 and B200 — no H200, no GB200, no GH200.
- getdeploying.com/gpus/nvidia-h200 (31 providers tracked): Lambda Labs **not in the list**. Providers shown: Theta EdgeCloud $2.29, FluidStack $2.30, Sesterce $2.09, Lyceum $3.19, Koyeb $3.00, DigitalOcean $3.44, Verda $3.39, Hyperstack $3.50, Nebius $3.50, Civo $3.49, Runpod $3.59–$4.39, Together $4.99, CoreWeave $6.31, Crusoe $4.29, Oblivus $3.99. Range $1.21–$13.78.
- thundercompute.com/blog/nvidia-h200-pricing dated 2026-05-12 lists "Lambda Cloud H200 $3.79/GPU/hr, minute-billed, no commitment" citing `lambda.ai/instances` — but that page contains no H200. The $3.79 figure is either (a) a stale 2025 H200 ad Lambda has since pulled, (b) the H200 1-Click Cluster sales-quoted rate confused with on-demand, or (c) an aggregator confabulation. **Do not rely on $3.79/hr Lambda H200.**

If you need H200 capacity for less than enterprise contract, **Lambda is not the path** as of 2026-05-25. Sesterce $2.09 or Theta EdgeCloud $2.29 are the cheapest list-price H200s tracked. Lambda would require a Private Cloud / Supercluster sales engagement.

---

## 3. Lambda Public Cloud regions & current stock

Direct fetch of docs.lambda.ai/public-cloud/regions/:

**14 regions in May 2026** (R1 didn't enumerate the international ones):
- US: us-east-1 (Virginia), us-east-2 (Washington DC), us-midwest-1 (Illinois), us-south-1 (Texas), us-south-2 (North Texas), us-south-3 (Central Texas), us-west-1 (California), us-west-2 (Arizona), us-west-3 (Utah)
- International: asia-northeast-1 (Tokyo), asia-northeast-2 (Osaka), asia-south-1 (India), europe-central-1 (Germany), me-west-1 (Israel)

Docs explicitly state "Not every instance type will be available in every region." No per-SKU region matrix is published. Lambda forces you to open the launch console and inspect the region selector per-SKU.

**Stock signal**: shadeform aggregator (last update May 24 2026) reports "4 of 76 instance configs in stock across 15 regions" — i.e., ~95% of Lambda capacity is sold out at any given snapshot. Most flagship SKUs (B200, H100 SXM 8×) show "Sold Out" in their UI; some legacy 8× V100 in Texas is "Available". Real-time GH200 stock not visible without an authenticated console login. R1 was right to flag capacity as the real risk.

**Status page**: status.lambda.ai shows "all systems operational" as of 2026-05-25. No incidents recorded in May 2026. Same components inventory R1 captured. R1's note about the March 2025 us-south-1 16-hour outage remains the most recent material event.

---

## 4. Lambda Stack apt repo — what's actually shipped

### 4.1 Confirmed package versions (lambda.ai/lambda-stack-deep-learning-software, fetched 2026-05-25)

| Component | Version | Confidence |
|---|---|---|
| CUDA Toolkit | **12.8.93** (12.8 Update 1) | HIGH — page lists `12.8.93~12.8.1` |
| PyTorch | **2.7.0** (`2.7.0+ds`) | HIGH |
| TensorFlow | **2.19.0** | HIGH |
| Keras | **3.10.0** | HIGH |
| **JAX** | **0.6.0** | HIGH (R1 missed JAX) |
| NCCL | **2.26.2** | HIGH |
| NVIDIA Container Toolkit | **1.18.1** | HIGH |
| cuDNN | not pinned on the page | MEDIUM (likely cuDNN 9.x given CUDA 12.8) |
| NVIDIA Driver | not pinned on the page | **REVISED: 570.xx branch, not 580.xx** |

### 4.2 Driver version — corrected

R1 said driver 580.xx. R3 evidence says **570.xx**:
- github.com/lambdal/lambda-stack-dockerfiles/Dockerfile (master, fetched 2026-05-25) declares:
  ```
  Package: libcuda1-dummy
  Version: 12.8
  Provides: libcuda1 (= 570) , libcuda-12.8-1 , libnvidia-ml1 (= 570)
  ```
  The dummy `Provides:` clauses pin the driver compatibility line to the 570 branch.
- CUDA 12.8 GA driver is 570.xx (NVIDIA's CUDA Toolkit 12.8 launched with driver 570.86.x in Jan 2025). Driver 580.xx ships with CUDA 13.x.
- Lambda Stack pins CUDA 12.8.93 (Update 1, mid-2025). That maps to driver 570.xxx, not 580.xx.

R1's "580.xx" was wrong — likely a hallucination from a 2025 community thread or an overshoot.

### 4.3 Repo staleness signal

`github.com/lambdal/lambda-stack-dockerfiles`:
- Default branch: `master`
- `pushed_at`: **2025-03-11T20:53:26Z** — i.e., the public Dockerfiles repo has not been updated in **14 months**.
- `updated_at`: 2026-04-01 (metadata-only touch).
- Only 3 files: `Dockerfile`, `README.md`, `LICENSE`. Build is parameterized by `UBUNTU_VERSION` build-arg; same Dockerfile for all supported LTS.
- README still says: "only current LTS versions are supported: 20.04, 22.04, and 24.04."

Implication: the *container* recipe is frozen since March 2025; the *host* apt repo continues to update (PyTorch 2.7.0 came out April 2025, so the apt repo did get refreshed even while the Dockerfile didn't). If you want Lambda Stack inside a custom container, you're building on a 14-month-old base — fine for inference, but don't expect bleeding-edge.

### 4.4 `apt-cache policy lambda-stack-cuda` — what to actually expect on a fresh 24.04

Searched May 2026 blog posts and GitHub gists; no fresh community capture is published. Best reconstruction from the canonical install line and the version pins on lambda.ai:

```
$ apt-cache policy lambda-stack-cuda
lambda-stack-cuda:
  Installed: (none) | <version-string>
  Candidate: 0.1.18~<distro-noble>   (likely; this is the meta-package version,
                                      not the CUDA version it pulls)
  Version table:
     0.1.18~noble1  500
        500 https://lambdalabs.com/static/misc/lambda-stack-repo  noble/main amd64 Packages
```

The meta-package version (0.1.x) is a Lambda-internal versioning scheme; it does not match the CUDA version. The CUDA toolkit pulled is 12.8.93 (Update 1), the driver is 570.xxx. No firm community confirmation of the exact meta-package version string was found in May 2026 sources — this is the gap to close by SSHing into a fresh 24.04 GH200 instance.

### 4.5 Ubuntu 26.04 status

Still not on Lambda. README and `apt` repo target only Focal/Jammy/Noble. Ubuntu 26.04 LTS (April 2026) is not in any Lambda apt repo, no boot image, no Dockerfile target. Expect Q3-Q4 2026 based on Noble's 5-month lag.

---

## 5. Reserved & Private Cloud — the data has thinned

R1's $1.49 GH200 / $1.89 H100 SXM / $3.79 B200 reserved numbers came from 2024–2025 aggregator scrapes. In May 2026, **those numbers have been removed from the aggregator pages**:

- **getdeploying.com/gpus/nvidia-gh200** (last updated 2026-05-24): displays "no reserved pricing"; shows Lambda GH200 only at $2.29/hr on-demand. 3 providers total tracked.
- **getdeploying.com/lambda-labs** (last updated 2026-05-24): on-demand table only; no reserved column.
- **computeprices.com/providers/lambda** (updated 2026-05-24): on-demand table only.
- **gpuperhour.com/providers/lambda-labs**: lists 11 GPUs $0.69–$5.85/hr (note: $5.85 is the *1-Click Cluster 64-GPU H100 rate*, not B200 on-demand — gpuperhour appears to be mixing cluster and on-demand). "Reserved instances accepted" mentioned but no specific rate published.
- **spheron.network 2026 blog post**: explicitly estimates Lambda reserved H100 SXM at $3.19/hr (1yr) and $2.95/hr (3yr) — applying historical ~20%/~26% discounts to the current $4.29 on-demand. Labeled as estimate, not source-quoted.
- **checkthat.ai 2026**: states "Contact sales for Reserved Clusters & Private Cloud," lists no rates. Cites one Voltron Data case study: 32× A100 reserved cluster $292,934/yr (= $1.04/GPU/hr) vs $673,468/yr on AWS ($2.40/GPU/hr) — a 57% AWS-relative discount. That's the only hard reserved A100 datapoint visible.

**Net**: Lambda's reserved pricing transparency degraded between 2025 and 2026. The R1 reserved table (GH200 $1.49, H100 $1.89, B200 $3.79) is unverifiable today. Apply Lambda's historical ~20% (1yr) / ~26% (3yr) discount to current on-demand for a defensible estimate:

| SKU | On-demand | 1yr reserved est. | 3yr reserved est. |
|---|---|---|---|
| GH200 1× | $2.29 | ~$1.83 | ~$1.69 |
| H100 SXM 1× | $4.29 | ~$3.43 | ~$3.17 |
| H100 SXM 8× | $3.99 | ~$3.19 | ~$2.95 |
| B200 1× | $6.99 | ~$5.59 | ~$5.17 |
| B200 8× | $6.69 | ~$5.35 | ~$4.95 |

These are estimates, not quotes. The R1 "$1.49 GH200" was a 35% discount; the 2026 discount band per the aggregator commentary is 19–27%. The R1 number was either (a) an old 2024 promotion, (b) a 3-year multi-GPU bulk deal, or (c) aggregator hallucination. **For decision purposes, plan on ~$1.83/hr GH200 reserved, not $1.49/hr, unless you have a sales quote.**

### Private Cloud minimum commit

- lambda.ai/service/gpu-cloud/private-cloud explicitly states: "from 4,000 to 165,000+ NVIDIA GPUs."
- This is **single-tenant Supercluster** — not a small reserved deal. Floor is 4,000 GPUs.
- No published minimum dollar commit or contract length. Multi-year contracts are standard per checkthat.ai 2026 ("1-year, 2-year, or 3-year").
- The 1× GH200 user case is **wildly below** the Private Cloud floor. Private Cloud is irrelevant; reserved 1-Click Cluster is the only way to commit, and 1-Click Clusters start at 16 GPUs.
- For a single GH200, **the only Lambda paths are on-demand at $2.29/hr or a sales-quoted long-term reservation** — there is no productized 1-GPU reserved tier.

---

## 6. Cross-check with R1's data table

R1 monthly cost table corrections:

| Scenario | R1 figure | R3 figure | Δ |
|---|---|---|---|
| 1× GH200 24/7 on-demand | $1,672 | $1,672 | confirmed |
| 1× GH200 24/7 reserved | $1,088 (@$1.49) | **~$1,336 (@$1.83 est.)** | +$248/mo more realistic |
| 8× H100 SXM 24/7 on-demand | $23,302 | $23,302 | confirmed |
| 8× B200 SXM 24/7 on-demand | $39,070 | $39,070 | confirmed (note B200 is 180GB not 192GB) |
| 16× H100 1-Click (smallest cluster) | $71,949 | $71,949 | confirmed |
| 16× B200 1-Click | $115,165 | $115,165 | confirmed |

R1's reserved monthly estimate was optimistic by ~$250/mo on the GH200 case. Bake the more conservative ~$1,336/mo into the decision; the lower number is contingent on a sales conversation that may not yield.

---

## 7. New gotcha — B200 VRAM discrepancy

R1 said "1× B200 SXM6 192GB $6.99/hr". Lambda's pricing page says **180GB**. NVIDIA's B200 spec is 180GB HBM3e (192GB is the B200 NVL / B300 line). Practical impact: minor — but if you're capacity-planning a model that requires >180GB VRAM, you can't fit it on a Lambda B200 SXM.

---

## 8. Confidence ladder (updated)

### HIGH (direct fetch today)
- All on-demand $/hr per SKU (GH200 $2.29, H100 SXM $3.99–$4.29, B200 $6.69–$6.99, H100 PCIe $3.29, A100/A10/A6000/V100/RTX6000 — see §1 table)
- 1-Click Cluster rates ($6.16–$5.54 H100, $9.86–$8.87 B200)
- 1-Click 2-week min, 10-day pay window, weekly billing
- Persistent FS $0.20/GiB/mo, free egress
- Lambda Stack package versions (CUDA 12.8.93, PyTorch 2.7.0, TF 2.19.0, NCCL 2.26.2, JAX 0.6.0, NVCT 1.18.1)
- H200 not self-serve on Lambda
- Lambda has 14 regions
- Status page clean for May 2026
- Dockerfile repo stale since 2025-03-11

### MEDIUM
- Driver version 570.xx (inferred from Dockerfile dummy + CUDA 12.8 mapping; not stated on marketing page)
- cuDNN 9.x (inferred from CUDA 12.8 era)
- Reserved estimates at ~20%/~26% discount (range supported by multiple 2026 sources; specific numbers not contractual)
- B200 VRAM 180GB (per Lambda's own page; R1's 192GB likely wrong)
- 1× GH200 reserved monthly ~$1,336/mo (was $1,088/mo in R1)

### LOW / gaps
- Exact `apt-cache policy lambda-stack-cuda` output on fresh 24.04 — no May-2026 community capture; need a live SSH to confirm
- Exact GH200 stock per-region right now — Lambda does not expose this without authenticated console
- B200 / GH200 / H200 reserved firm quotes — sales-only, no public anchor
- Private Cloud minimum commit dollar value — sales-only
- Ubuntu 26.04 / kernel 7.0 ETA on Lambda Stack — unknown
- SLA service-credit math — unpublished

---

## 9. Decision-relevant updates vs R1

1. **Reserved GH200 budget should be ~$1,336/mo, not $1,088/mo.** The $1.49/hr R1 number is unverifiable in May 2026; aggregators have removed it. Plan on ~$1.83/hr est. for a 1-year reservation negotiated through sales.

2. **H200 is not a Lambda option for self-serve workloads.** If H200 is a hard requirement, look elsewhere (Sesterce $2.09, Theta EdgeCloud $2.29, Nebius $3.50 — all from getdeploying H200 listing). The "$3.79/hr Lambda H200" blog citation is dead.

3. **Driver is 570.xx, not 580.xx.** Confirm before running any kernel-module-sensitive code. CUDA 12.8 → driver 570.xx. If a workload needs driver ≥580 (e.g., for CUDA 13 features), Lambda Stack does not provide it as of 2026-05-25.

4. **B200 is 180GB VRAM.** Not 192GB. Affects capacity planning only on the edge.

5. **Lambda Stack containerization is on a 14-month-old Dockerfile.** Host apt repo is current; containers are not. Build your own if you need anything fresher than CUDA 12.8 / driver 570 inside a container.

6. **No private cloud option for 1× GH200.** Supercluster floor is 4,000 GPUs. Single-GPU paths are on-demand or 1-Click Cluster (16+ GPU min, GH200 not even offered in 1-Click — only H100/B200). A single-GH200 reservation requires a custom sales conversation and there is no productized tier.

---

## 10. Fetched-today source URLs

Direct successful fetches 2026-05-25:
- https://lambda.ai/pricing
- https://lambda.ai/1-click-clusters
- https://lambda.ai/instances
- https://lambda.ai/service/gpu-cloud
- https://lambda.ai/service/gpu-cloud/private-cloud
- https://lambda.ai/lambda-stack-deep-learning-software
- https://docs.lambda.ai/public-cloud/regions/
- https://docs.lambda.ai/public-cloud/billing/
- https://status.lambda.ai/
- https://getdeploying.com/gpus/nvidia-gh200 (last update 2026-05-24)
- https://getdeploying.com/gpus/nvidia-h200
- https://getdeploying.com/lambda-labs (last update 2026-05-24)
- https://computeprices.com/providers/lambda (last update 2026-05-24)
- https://gpuperhour.com/providers/lambda-labs
- https://www.thundercompute.com/blog/nvidia-h200-pricing (data dated 2026-05-12)
- https://www.spheron.network/blog/lambda-cloud-h100-pricing-2026
- https://checkthat.ai/brands/lambda-labs/pricing
- https://github.com/lambdal/lambda-stack-dockerfiles (master branch, Dockerfile + README, pushed 2025-03-11)

Failed/redirected fetches:
- https://cloud.lambda.ai/instances — 403 (authenticated)
- https://shadeform.com/directory/instances — empty content, requires JS
- https://docs.lambda.ai/public-cloud/lambda-stack-images/ — 404 (page moved or removed)
- https://jarvislabs.ai/blog/h200-price — no Lambda mention

---

## 11. What to update in the round1 file

If round1/16_lambda_pricing.md gets revised:

1. **§1.1 table — B200 VRAM**: change "192GB" → "180GB" (Lambda's own page says 180GB).
2. **§1.3 reserved rates**: flag the $1.49/$1.89/$3.79 numbers as stale; replace with on-demand × 0.80 (1yr) / × 0.74 (3yr) estimates and note "no live aggregator confirmation as of 2026-05-25."
3. **§1.4 reserved monthly**: GH200 reserved est. ~$1,336/mo, not $1,088/mo.
4. **§4.1 driver**: change "580.xx" → "570.xx branch" with evidence from libcuda1-dummy Provides clause in Dockerfile.
5. **§4.1 add JAX 0.6.0** to the Lambda Stack table.
6. **§5 H200**: state explicitly that Lambda is not on getdeploying.com/gpus/nvidia-h200 in May 2026 — Lambda is not a viable H200 self-serve provider.
7. **Region count**: 14 regions, not unspecified — list them.
8. **Private Cloud floor**: 4,000 GPUs (per lambda.ai/service/gpu-cloud/private-cloud).
9. **lambda-stack-dockerfiles repo**: Dockerfile last touched 2025-03-11; host apt repo is current but Dockerfile is 14 months stale.
