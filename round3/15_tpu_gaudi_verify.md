# Agent 15 Round 3 — TPU v6e/v7 Ironwood + Gaudi 3 Verification & Deepening

**Date:** 2026-05-25
**Scope:** Verify and update Round 1 claims (`/home/ubuntu/research/gh200_inference/round1/15_gaudi_tpu.md`) on five drill points: TPU v6e on-demand pricing, TPU v7 Ironwood availability, vLLM-TPU benchmarks, Gaudi 3 vs H200 independent tests, IBM GPU4YOU promo status.
**Method:** 11 WebSearch/WebFetch calls against official Google Cloud docs, IBM blog, vLLM repos, DCD, Spheron, Constellation Research, Tom's Hardware.

---

## TL;DR — What Changed vs Round 1

| Claim from Round 1 | Round 3 verdict | Update |
|---|---|---|
| Ironwood is GA but "reservation only via account team" | **PARTIALLY WRONG — needs correction** | Ironwood is **GA as of April 22, 2026 (Google Cloud Next, Las Vegas)**. The TPU7x training docs **explicitly show `--on-demand` as a supported provisioning flag** alongside `--reservation` and `--spot`. GKE is mandatory (no Compute Engine TPU API path), but the on-demand gate has been lifted. The TPU7x marketing page still says "contact your account team" for access — but the underlying GKE/XPK docs document on-demand/spot/reservation as first-class flags. **Reservation is no longer the only path.** |
| TPU v6e on-demand $2.70/chip-hr | **CONFIRMED** | $2.70/chip-hr remains the most-cited 2026 on-demand number. 1y commit $1.89, 3y commit $1.22 (some sources $0.39 as floor). $4.20-$4.50 outliers in Round 1 appear to be stale list-price extrapolations. |
| TPU v7 Ironwood "not publicly listed" pricing | **CONFIRMED** | Google has not published a chip-hour rate. SemiAnalysis/Zephyr estimate $15k capex → analyst guess ~$6-10/chip-hr on-demand. No leak found. |
| Gaudi 3 line being wound down, Falcon Shores cancelled, Crescent Island H2-2026, Jaguar Shores 2027 | **CONFIRMED — no new news** | No update reversing the wind-down narrative. |
| IBM GPU4YOU 50% promo, expires 30 Jun 2026 | **CONFIRMED with one correction** | IBM community blog says **"valid now until July 1st, 2026 – or until supplies last."** Effectively same date (June 30 vs July 1) but the official wording is July 1. Restricted to new IBM Cloud customers OR existing customers new to GPU services. |
| No public MLPerf submission for Gaudi 3 | **CONFIRMED & STRENGTHENED** | Intel did not submit Gaudi 3 to MLPerf Inference v6.0 (data center LLM tracks). Intel's v6 participation was Xeon 6 / Arc Pro only. |
| Spheron Llama 3.3-70B Gaudi 3 numbers are projections | **CONFIRMED, plus new independent data point** | New: **Aquatron + Los Alamos National Laboratory** Supermicro 8-card test (March 2026) — H200 9-9.5× Gaudi 3 on Llama 3.1 **405B** (H200 ~25-26 tok/s vs Gaudi 3 ~2.7 tok/s). Counter-data point: **Signal65** independent vLLM tests show Gaudi 3 beating H100 in 3 of 4 Llama-8B configs and matching/beating H200 on Mixtral-8x7B by up to 20%. The independent picture is workload-dependent. |
| vLLM-TPU unified Oct 2025; Llama 3.1-70B v6e-8 = 2.1× Feb-2025 prototype | **CONFIRMED with new version** | Latest `vllm-project/tpu-inference` release: **v0.20.0, May 21, 2026**. Adds Ironwood (v7x) as a recommended TPU generation alongside v5e and v6e. Llama 3.3-70B-Instruct now in the support matrix (unit/correctness/perf passing) but no absolute tok/s number published. |
| vllm-gaudi plugin production-ready at v0.15.1 (Feb 2026) | **CONFIRMED with new version** | **v0.16.0** has shipped (based on vLLM v0.16.0 / Gaudi Software v1.23.0). Adds Qwen3-VL, DeepSeek OCR, MiniMax-M2, Ovis, Mistral-Large-3, Hunyuan V1. Notable: changelog entry "Missing updates for Llama4 on main" — Llama 4 support is **lagging** in v0.16.0. |

---

## 1. TPU v6e Trillium Current On-Demand Pricing

**Verified:** Round 1 number ($2.70/chip-hour on-demand) holds in May 2026.

Triangulated rates from multiple sources:
- **On-demand:** $2.70/chip-hr (most-cited 2026 number, awesomeagents.ai, Spheron migration guide, Introl).
- **1-year CUD:** ~$1.89/chip-hr.
- **3-year CUD:** ~$1.22/chip-hr, with floors reported as low as **$0.39/chip-hr** at scale commitments.
- The Round-1 outliers ($4.20-$4.50) traced back to a "v5e + 10% uplift" speculative extrapolation predating Trillium GA pricing — they should be deprecated.
- Google's official `cloud.google.com/tpu/pricing` page rendered as truncated/unfetchable for both WebFetch and Google's own cached snapshot, so the table itself was not directly verified — but all secondary sources converge on $2.70.

**Note on regional variance:** TPU v6e pricing is per-chip-hour but billed per-VM-host; regional promotions exist. The $1.375/chip-hr Round-1 outlier was likely a temporary regional special.

Sources: [Spheron TPU Trillium v6 vs B200 (2026)](https://www.spheron.network/blog/google-tpu-trillium-v6-vs-nvidia-b200-llm-inference/), [Introl — Google TPU vs NVIDIA GPU](https://introl.com/blog/google-tpu-vs-nvidia-gpu-infrastructure-decision-framework-2025), [Awesome Agents — TPU v6e Trillium](https://awesomeagents.ai/hardware/google-tpu-v6e-trillium/), [Google Cloud TPU Pricing](https://cloud.google.com/tpu/pricing).

---

## 2. TPU v7 Ironwood — Updated Availability Picture

**Material correction to Round 1.**

### Confirmed
- **GA date:** April 22, 2026 at Google Cloud Next, Las Vegas. Cloud TPU release notes confirm a TPU7x GA entry on **March 31, 2026** (production track) with the public announcement at Next on April 22.
- **GKE is mandatory.** Compute Engine TPU API is not the path for Ironwood — must go through GKE Standard ≥ v1.34.0-gke.2201000 or GKE Autopilot ≥ v1.34.1-gke.3084001, or TPU Cluster Director.
- **Framework support:** JAX is the named first-class framework. TensorFlow explicitly not supported. PyTorch path runs via vLLM-TPU's torchax → JAX lowering, not native PT/XLA.

### Corrections to Round 1
Round 1 said Ironwood is **"reservation only via account team."** That overstated the gate. The TPU7x training documentation explicitly documents three provisioning modes via `xpk cluster create`:

```
--on-demand        (default)
--reservation=RESERVATION_NAME
--spot
```

The "contact your account team" phrasing on the TPU7x marketing page is a **quota request gating mechanism**, not a fundamental reservation-only restriction. Once quota is granted in your project, on-demand and spot provisioning work like any other TPU. This is a meaningful softening — it means Ironwood is approaching peer-with-Trillium accessibility for users who can get past the quota request.

Customer adoption list (still small, all enterprise-relationship):
- **Anthropic:** up to 1M chips, "$10B in finished racks from Broadcom" (first phase 400k Ironwood units) + GCP rental; over 1 GW capacity in 2026.
- **Lightricks:** image/video generation production workloads.
- **Essential AI:** frontier model development.

### Still no public pricing
- No on-demand `$/chip-hour` posted publicly as of 2026-05-25.
- SemiAnalysis estimates Google's capex at ~$15k/v7 chip → analyst back-of-envelope ~$6-10/chip-hr on-demand at typical 3-year cloud margin. **This remains speculation.**

Sources: [Cloud TPU release notes](https://docs.cloud.google.com/tpu/docs/release-notes), [TPU7x docs](https://docs.cloud.google.com/tpu/docs/tpu7x), [Train a model using TPU7x](https://docs.cloud.google.com/tpu/docs/tpu7x-training), [Ironwood in GKE](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/tpu-ironwood), [Ironwood GA blog](https://cloud.google.com/blog/products/compute/ironwood-tpus-and-new-axion-based-vms-for-your-ai-workloads), [SemiAnalysis TPUv7](https://newsletter.semianalysis.com/p/tpuv7-google-takes-a-swing-at-the), [Anthropic + GCP](https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services).

---

## 3. vLLM-TPU — Current Version and Llama 3.3-70B on v6e-8

**Verified with version bump.**

### Latest release
- **`vllm-project/tpu-inference` v0.20.0, May 21, 2026** (verified via GitHub releases page).
- Adds **v7x (Ironwood)** as a recommended TPU generation alongside v5e and v6e.
- Adds KV cache offload to host memory, Int4-CompressedTensors MoE, w4afp8 linear layers, Eagle3 speculative decoding improvements, embedding model support (Qwen3-Embedding-8B).
- Support matrix lists Llama 3.1-8B-Instruct and Llama 3.3-70B-Instruct as passing unit/correctness/perf tests.

### Llama 3.3-70B on TPU v6e-8 — still no public absolute tok/s number
Even with the v0.20.0 release in May 2026, there is no public absolute tokens/sec figure for **Llama 3.3-70B on v6e-8** from vLLM/Google/Artificial Analysis. The October 2025 vLLM blog only gives the **2.1× speedup vs the Feb-2025 prototype** for Llama 3.1-70B on v6e-8. Recent secondary writeups give general guidance ("300-400 tok/s/chip on v6e at batch 8 for 70B-class models") but these are unsourced.

This is the same gap as Round 1, and v0.20.0 didn't close it. **The MLPerf-style number for Llama 3.3-70B on v6e-8 still doesn't exist publicly.**

### What the v0.20.0 release does close
- **PyTorch model definitions run on TPU without code changes** via torchax → JAX/XLA. ~20% throughput uplift over the prior PT/XLA path.
- Multi-host disaggregated serving over DCN is supported, which matters for >v6e-8 slices.
- TPU7x (Ironwood) is now first-class in vLLM-TPU — meaning at least the runtime path exists for Ironwood, even if the chip remains quota-gated.

Sources: [vllm-project/tpu-inference releases](https://github.com/vllm-project/tpu-inference/releases), [vllm-project/tpu-inference repo](https://github.com/vllm-project/tpu-inference), [vLLM TPU blog Oct 2025](https://vllm.ai/blog/2025-10-16-vllm-tpu).

---

## 4. Gaudi 3 — Independent Benchmarks Since Round 1 Cutoff

**Material additions vs Round 1.**

Two independent benchmark sources have published since Round 1 was compiled. They tell **opposing stories** depending on model size and workload:

### Negative for Gaudi 3 (large dense models)
**Aquatron + Los Alamos National Laboratory (March 2026)** — claimed world-first 16-bit precision Llama 3.1 **405B** validation benchmark.
- Brand-new Supermicro servers, 8× H200 vs 8× Gaudi 3.
- Three workloads: AI research, history, coding.
- H200: ~25-26 tok/s consistent across workloads.
- Gaudi 3: ~2.7 tok/s.
- **H200 = 9-9.5× Gaudi 3** across all three.
- Aquatron's framing: "H200 could handle 9× more inference requests at same latency, or reduce inference cost by similar factor."
- This contradicts Intel's July 2024 sales claim of "1.5× faster than H200."

The 405B caveat: This is a 16-bit precision test (so 810GB of weights), which requires tight inter-card bandwidth. Gaudi 3's 200GbE-RoCE-based interconnect is structurally weaker than NVLink for this scale of model. Result is **not directly transferable to 70B** but is the upper-bound of Gaudi 3's interconnect weakness.

### Positive for Gaudi 3 (smaller/MoE models)
**Signal65 (Futurum Group) — IBM Cloud testing on vLLM** (mid-late 2025, still the most recent independent published).
- **Llama 8B:** Gaudi 3 beat H100 in 3 of 4 configurations.
- **Mixtral-8x7B-Instruct:** Wouldn't fit on a single H100; Gaudi 3 was "highly competitive with H200" — up to **20% more tok/s than H200** at balanced batches.
- **Llama 405B (cloud setup):** also tested but Signal65's framing was less favorable.
- Methodology: vLLM only, no extra tuning, varying input/output token shapes and batch sizes.

### MLPerf v6.0
**Intel did not submit Gaudi 3 to MLPerf Inference v6.0 data center LLM tracks.** Intel's v6 participation was limited to Xeon 6 and Arc Pro GPU. This continues the v5.1 pattern. For a chip that has been on the market 18+ months and is sold on price-performance claims, the continued no-show is a strong negative signal.

### Net read
- **Below 70B dense / for MoE-heavy workloads with 128GB-fit benefits:** Gaudi 3 is competitive and the IBM 50%-off promo can make it cheaper per token.
- **At 70B and especially 405B dense:** Gaudi 3 loses materially; the 9× gap on 405B is the headline number, even if it doesn't directly extrapolate to 70B.
- **Llama 3.3-70B specifically:** Still **no published independent measurement** from either side — Spheron's projections (520/1350/1850/2100 tok/s at batch 1/8/32/128) remain unverified.

Sources: [DCD — H200 9× Gaudi 3 Llama 3.1 405B](https://www.datacenterdynamics.com/en/news/nvidia-h200-outperforms-intel-gaudi-3-by-factor-of-nine-across-first-llama-31-405b-benchmark-test-exclusive/), [BW Businessworld — H200 vs Gaudi 3 9× independent](https://www.businessworld.in/article/nvidia-h200-chips-outperform-intels-gaudi-3-by-nine-times-in-independent-tests-558375), [Signal65 — Gaudi 3 on IBM Cloud](https://signal65.com/research/ai/intel-gaudi-3-accelerates-ai-at-scale-on-ibm-cloud/), [Techstrong.ai — Signal65 David vs Goliath](https://techstrong.ai/sponsored-content/david-vs-goliath-in-ai-hardware-signal65s-independent-tests-reveal-gaudi-3s-surprising-edge/), [Spheron Gaudi 3 vs H200/B200 2026](https://www.spheron.network/blog/intel-gaudi-3-vs-nvidia-h200-b200-llm-inference-2026/).

---

## 5. IBM Cloud Gaudi 3 GPU4YOU Promo — Active Status

**Confirmed active. Minor wording correction.**

- **Promo code:** `GPU4YOU`.
- **Discount:** 50% off hardware costs (not other IBM Cloud services) for first 6 months. After 6 months, reverts to standard pricing.
- **Expiration:** Official IBM Community blog wording is **"valid now until July 1st, 2026 – or until supplies last."** Round 1 said "30 Jun 2026" — effectively the same window, but IBM's own language is "July 1st." Either way, **~5 weeks remaining from 2026-05-25**.
- **Eligible hardware:**
  - Intel Gaudi 3 AI Accelerators
  - NVIDIA L40s GPUs (24-core configs, all IBM Cloud data centers)
  - NVIDIA A100 GPUs (24-core configs, all IBM Cloud data centers)
- **Eligibility:** **New IBM Cloud customers OR existing customers new to GPU services.** This is a material restriction — existing IBM Cloud GPU customers cannot stack this with their current usage.
- **Stacking:** Cannot be combined with any other offers or discounts.

### Effective Gaudi 3 cost during the promo window
- IBM list: ~$60/hr per 8-card node → ~$7.50/card-hr.
- With GPU4YOU: ~$3.75/card-hr.
- Round 1's break-even-vs-H200 threshold at batch-32 Llama 3.3-70B was ~$3.12/card-hr. **Even with the 50% promo, Gaudi 3 is still 20% above the break-even on Spheron's projected numbers**, and the Aquatron 405B finding makes the case harder still.
- **Window closes July 1, 2026**, after which Gaudi 3 reverts to the full ~$7.50/card-hr where it loses materially on $/token vs H200 and GH200.

Sources: [IBM Community blog — GPU4YOU promo](https://community.ibm.com/community/user/blogs/chris-carter/2025/05/12/ibm-cloud-exclusive-gpu-promotion), [IBM Cloud Gaudi 3 product page](https://www.ibm.com/products/gpu-ai-accelerator/intel-gaudi3), [IBM Newsroom — Gaudi 3 GA](https://newsroom.ibm.com/blog-intel-and-ibm-announce-the-availability-of-intel-gaudi-3-ai-accelerators-on-ibm-cloud).

---

## 6. Revised Decision Matrix (May 2026)

| Dimension | GH200 (96/144 GB) | Gaudi 3 | TPU v6e Trillium | TPU v7 Ironwood |
|---|---|---|---|---|
| Public on-demand $/hr | **Yes** ($1.99-$6.50 Lambda/CoreWeave) | Yes ($7.50/card list, $3.75 w/GPU4YOU until July 1) | **Yes** ($2.70/chip-hr) | **Yes via GKE** (no public $/hr; quota-gated) |
| Reservation gate | None | None | None | Quota request via account team (then on-demand/spot/reservation flags work) |
| Llama 3.3-70B fits 1 chip | Yes (BF16 needs 2×96GB or 1×144GB) | Yes (1× 8-card node trivially) | No (slice needed) | Yes (192 GiB) |
| Published Llama 3.3-70B tok/s | Yes (Baseten on Lambda) | **Projections only** | **None** (2.1× relative gain only) | **None** |
| MLPerf v6.0 Llama-2/3-70B submission | Yes (NV) | **None** | n/a | n/a |
| Independent benchmark vs H200 | n/a | **9× worse on 405B (LANL/Aquatron); 20% better on Mixtral (Signal65)** | n/a | n/a |
| vLLM first-class | Yes (upstream) | Yes (vllm-gaudi plugin v0.16.0; Llama 4 lagging) | Yes (vllm-project/tpu-inference v0.20.0, May 2026) | Yes (v0.20.0 lists v7x as recommended) |
| SGLang | Yes | **No HPU backend** | Limited | Unknown |
| PyTorch first-class | Yes | Yes (via plugin) | Yes (torchax→JAX, ~20% better than old PT/XLA) | Effectively via vLLM-TPU torchax path; native PT/XLA not documented |
| Vendor follow-on commitment | Yes (Blackwell, Vera Rubin) | **No — Gaudi line ended** | Yes | Yes (v8 Sunfish training + v8 Zebrafish inference, TSMC 2nm, late 2027) |
| Migration cost from CUDA | None | Medium (works but quirky) | Medium-high (JAX-first) | Medium-high (JAX-first, GKE-only) |

---

## 7. Updated Pick Recommendations

**Pick GH200** — Default. Still the right answer for most teams: on-demand from multiple providers, full vLLM/SGLang/TRT-LLM, no migration cost, MLPerf-validated.

**Pick Gaudi 3** — Only if **all three** apply:
1. You are already on IBM Cloud or sovereign-cloud-locked to a Gaudi 3 region.
2. Your workload is small dense (≤70B) or MoE (Mixtral-class) where Signal65 shows Gaudi 3 competitive.
3. You can lock in usage **before July 1, 2026** to get the GPU4YOU 50% promo, and are a new IBM GPU customer.

If you wait past July 1, Gaudi 3's $/token loses to H200/GH200 on most workloads, and you're buying into an EOL product line (Crescent Island H2-2026 / Jaguar Shores 2027 are different chips, different software).

**Pick TPU v6e (Trillium)** — If you want truly on-demand TPU at $2.70/chip-hr, are willing to invest in JAX/MaxText (or take the ~20% hit on vLLM-TPU torchax), and your model fits a v6e-8 slice. Good for 70B serving in steady state at 3y CUD pricing ($1.22-$0.39/chip-hr).

**Pick TPU v7 Ironwood** — Now actually viable for users who can request quota:
- GKE-mandatory, JAX-first, no published per-chip $/hr.
- vLLM-TPU v0.20.0 makes the runtime path real.
- 192 GiB HBM3e + 4,614 FP8 TFLOPS makes single-chip 70B trivial and 405B fits a small slice.
- **Big remaining unknown: no public absolute Llama 3.3-70B tok/s number from anyone.** If you commit, you're an early benchmark publisher.

---

## 8. Confidence & Gaps (Round 3)

### High confidence (verified by 2+ independent sources)
- TPU v6e on-demand at $2.70/chip-hr (multiple secondary, no first-party table fetch).
- Ironwood GA on March 31, 2026 (TPU release notes) and announced at Cloud Next on April 22, 2026.
- Ironwood **supports `--on-demand`, `--spot`, `--reservation` flags via XPK on GKE** — this is a correction to Round 1's "reservation only" framing.
- IBM GPU4YOU active, expires July 1, 2026, restricted to new GPU customers.
- vllm-project/tpu-inference v0.20.0 (May 21, 2026) lists v7x Ironwood as supported.
- Gaudi 3 absent from MLPerf v6.0 data center LLM submissions.
- Aquatron + LANL independent test: H200 9× Gaudi 3 on Llama 3.1-405B.
- Signal65 independent test: Gaudi 3 beat H100 on Llama-8B (3/4 configs), Gaudi 3 up to 20% better than H200 on Mixtral-8x7B.

### Medium confidence
- Ironwood pricing estimate $6-10/chip-hr — SemiAnalysis/Zephyr extrapolation, not published.
- TPU v6e CUD floor at $0.39/chip-hr — secondary blog claim, may be a special-deal floor.
- Aquatron 405B numbers are 16-bit precision Supermicro 8-card; methodology not yet peer-published in a venue like MLCommons.

### Open gaps (unchanged from Round 1)
- **No public Llama 3.3-70B tok/s on TPU v6e-8** with vLLM-TPU v0.20.0 (a number that's been missing for 7+ months).
- **No public Llama 3.3-70B tok/s on Ironwood** from anyone.
- **No MLPerf submission for Gaudi 3** continues to be the loudest silence in the sector.
- **No published per-chip $/hr for Ironwood** — best guesses come from supply-chain analysts.
- **PT/XLA support tier on Ironwood unclear** — docs are JAX-only; vLLM-TPU torchax path implies PyTorch works but won't be first-class.

### Bias notes
- Aquatron is a data center operator selling NVIDIA hardware racks; their result favors NVIDIA. LANL adds independence weight, but the test framing is from Aquatron.
- Signal65 is paid by Intel for some engagements; their methodology disclosure helps but is not MLPerf-grade.
- Spheron is a GPU marketplace and biases toward GPU-favorable framings.
- Google Ironwood blogs cite only TPU-vs-TPU performance multipliers, never head-to-head vs NVIDIA in absolute terms — itself a signal that head-to-head doesn't favor them on every workload.

---

**Output file:** `/home/ubuntu/research/gh200_inference/round3/15_tpu_gaudi_verify.md`
