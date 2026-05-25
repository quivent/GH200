# Gap Verification Roadmap — Round 2 Cross-Pollination

**Agent:** Round-2 / G (gap-verification)
**Date:** 2026-05-25
**Inputs:** All 20 Round-1 reports in `/home/ubuntu/research/gh200_inference/round1/`
**Purpose:** Turn the dossier's "what we don't yet know" into a ranked, actionable punch-list for Round 3 (web research, vendor sales calls) and the optional hands-on measurement budget.

A "high-impact" gap is one whose resolution materially changes the user's decision: which engine to install on the GH200, whether to swap GH200 for a different platform, whether to renew Lambda or sign a multi-month commit elsewhere, or whether to trust a quoted price.

---

## 0. Glossary of remediation types

| Tag | Meaning | Typical cost / time |
|---|---|---|
| **MEASURE** | Resolvable with ~1 hour of GH200 wall time on Lambda on-demand ($2.29/hr) or a swap to a 1× H100 ($3.29/hr) / 1× H200 ($1.45/hr Nebius preempt). Estimate ≤$5 per experiment. |
| **WEB-R3** | Resolvable by a focused Round-3 web-research pass (search + fetch + version-pin verification). No hardware needed. |
| **SALES** | Only resolvable by emailing a vendor account team (price, SLA, reservation availability). Hours-to-days latency, no $ cost but commits the user to a sales conversation. |
| **UNRESOLVABLE** | Cannot be settled in a one-week research window — requires multi-month production telemetry, unreleased product, NDA data, or third-party customer cooperation. |

---

## 1. Top-15 high-impact gaps (ranked by decision impact)

### Rank 1 — Wan 2.2 on GH200: exact bf16 timing vs H100 baseline
- **Why it matters:** This is *the* load-bearing claim for keeping GH200. Round-1/20 estimates Wan2.2-A14B 720p 5-s at 25–45 s; no public source actually ran it. If the real number is >90 s, the GH200 economic story collapses; if it is ~25 s and an H100 needs offload, the "only GH200 fits" recommendation crystallises.
- **Source of gap:** Reports 06 §10, 20 §8, "no public Wan2.2 numbers on single GH200."
- **Remediation type:** **MEASURE.**
- **Exact experiment:**
  1. `lambda-cloud-cli` spin a 1× GH200 ($2.29/hr) and a 1× H100 SXM ($4.29/hr) in parallel.
  2. On each: `git clone github.com/Wan-Video/Wan2.2 && diffusers main` + the morphicfilms optimization stack; set `set_attention_backend("_flash_3_hub")`, `torch.compile(mode="max-autotune-no-cudagraphs")`, lightx2v-4-step + MagCache rdt=0.12.
  3. Run 5× 720p 5-s clips bf16; record wall-clock per clip and peak VRAM (`nvidia-smi --query-gpu=memory.used --format=csv -l 1`).
  4. Repeat with `enable_model_cpu_offload()` on the H100 (it shouldn't fit without).
- **Cost:** ~$15 total (≤2 GH200-hr + ≤2 H100-hr including warm-up).
- **Settles:** the GH200-vs-H100 narrative for the user's flagship diffusion workload.

### Rank 2 — Does the post-April-2026 vLLM FP8 fix actually work on GH200?
- **Why it matters:** The user-memory constraint says "no FP8 on this stack." Round-1/01–03 and 10 report that the FA3 FP8 accumulation bug *was* fixed in vLLM 0.19.1 on Hopper, but every public repro is on H100, not GH200 specifically. If the fix transfers, the user can opt-in FP8 at head_dim≤128 for ~1.5–2× decode speedup; if not, the rule stays "always bf16."
- **Source of gap:** Reports 01 §13, 03 §11, 10 §11, 20 §8.
- **Remediation type:** **MEASURE.**
- **Exact experiment:**
  1. 1× GH200 Lambda, install vLLM ≥0.19.1 ARM wheel.
  2. Serve `meta-llama/Llama-3.1-8B-Instruct` and `meta-llama/Llama-3.1-70B-Instruct` twice each — once `--kv-cache-dtype auto` (bf16), once `--kv-cache-dtype fp8`.
  3. Run the 128k-token needle-in-a-haystack harness (vLLM blog 2026-04-22 published it); record accuracy delta. Cross-check on `Qwen2.5-Coder-7B` — the original gibberish-output failure mode (GH issue #4218).
  4. Throughput sweep: ShareGPT replay at concurrencies 1/16/64.
- **Cost:** ~$10 (≤4 GH200-hr).
- **Settles:** whether the user-memory rule should be updated to "FP8 OK on dense LLMs ≥ vLLM 0.19.1, still broken on diffusion."

### Rank 3 — Cheapest H200 self-serve monthly: Nebius preemptible reality
- **Why it matters:** Report 17 names Nebius preemptible H200 at $1.45/hr (~$1,044/mo) — *cheaper than GH200 anywhere*. If real and usable for inference (preemption windows tolerable, IB intact), the platform recommendation shifts from "stay on GH200" to "move workload to H200 Nebius preemptible + checkpointing."
- **Source of gap:** Reports 17 §8, 18 §7. Preemption frequency and minimum-lifetime not published.
- **Remediation type:** **MEASURE + SALES** (combination).
- **Exact experiment / queries:**
  1. **Web (Round 3):** fetch `nebius.com/prices` + `docs.nebius.com/compute/instances/spot` for current preemption notice period and median lifetime.
  2. **Measure:** spin 1× H200 preemptible on Nebius for 4 hours, run a serving load (vLLM Llama-3.3-70B FP8 ShareGPT replay), record any preemption events.
  3. **SALES:** ask Nebius account team for committed 6/12-month H200 quote with documented SLA.
- **Cost:** ~$6 hardware + 1 sales email.
- **Settles:** primary alternative to Lambda GH200.

### Rank 4 — Lambda 1× GH200 reserved monthly price
- **Why it matters:** Report 16 estimates ~$1,088/mo at $1.49/hr × 730h but says Lambda removed reserved pricing from the public page — sales-only. If real, this is ~35% below on-demand and competitive with Vultr $1,433/mo on-demand. Single most consequential pricing question for the incumbent platform.
- **Source of gap:** Report 16 §8 "exact 1× GH200 reserved monthly price... unconfirmed."
- **Remediation type:** **SALES.**
- **Exact query:** email `sales@lambda.ai` (or use the in-console request form): "Quote 1× GH200 96GB on a 6-month and 12-month commit, monthly billing, Lambda Stack 24.04 image, us-south-1 or us-east region. Please confirm $/hr, monthly equivalent, SLA, and whether termination-for-convenience fees apply."
- **Cost:** $0.
- **Settles:** the renew-vs-switch decision baseline.

### Rank 5 — CoreWeave 12-month reserved H200 / B200 $/GPU-hr
- **Why it matters:** Report 18 §7 names CoreWeave as the only Platinum-tier neocloud for GH200/H200/B200, but every reserved rate is "sales only" with "up to 60% off" marketing. If 12-month H200 reserved is ~$2.00/GPU-hr, CoreWeave Platinum SLA at parity with Nebius preemptible is the better choice for a production user. If $4+/GPU-hr, it's not.
- **Source of gap:** Report 18 §7, Report 19 §6 (hyperscaler GH200 absent).
- **Remediation type:** **SALES.**
- **Exact query:** "CoreWeave Capacity Plan — Reservation tier — 12-month — 1× H200 141GB and 8× B200 — VAST + Quantum-2 fabric — please share $/GPU-hr and minimum-commit dollar amount."
- **Cost:** $0.
- **Settles:** scale-up alternative when load grows past single GH200.

### Rank 6 — Does Wan FP8 fail on H100 the same way it fails on GH200?
- **Why it matters:** If Wan FP8 fails on H100 too, the user's "FP8 broken on Wan" rule is hardware-independent (it's a DiT/diffusion-transformer issue) and the conclusion "stay on GH200 for Wan because bf16 fits" is correct on any Hopper. If Wan FP8 works on H100 but not GH200, that is a GH200-specific bug that changes the buy decision.
- **Source of gap:** Report 10 §11 (explicitly proposes this experiment), Reports 01 §8, 20 §8 hint at DiT-class FP8 issues across architectures.
- **Remediation type:** **MEASURE.**
- **Exact experiment:**
  1. Lambda 1× H100 SXM ($4.29/hr). Same Wan2.2-A14B pipeline used in Rank 1.
  2. Run once bf16, once FP8 (`torchao` FP8 dynamic quant, ComfyUI's `fp8_scaled` checkpoint).
  3. Compute LPIPS between bf16 and FP8 outputs on a fixed 8-prompt set.
  4. If LPIPS > 0.5 on H100, the bug is portable across all Hopper → user memory rule confirmed as DiT-FP8 rule, not GH200 rule.
- **Cost:** ~$8 (≤2 H100-hr).
- **Settles:** the portable-vs-hardware-specific question that determines whether FP8 evaluation is ever worth re-trying on alternative hardware.

### Rank 7 — Llama-3.3-70B on GH200: real bf16 vLLM 0.19 throughput numbers
- **Why it matters:** Multiple reports cite the same single Baseten/Lambda data point (Feb 2025, TRT-LLM FP8, +32% vs H100). vLLM 0.19 (April 2026) is the production stack today. Without a current bf16 vLLM number, the user cannot defend a renewal decision against "just use H200 Nebius."
- **Source of gap:** Reports 03 §11, 04 §9, 02 §12.
- **Remediation type:** **MEASURE.**
- **Exact experiment:**
  1. 1× GH200 Lambda. `pip install vllm==0.19.1` (or 0.19.x stable ARM wheel).
  2. `vllm serve meta-llama/Llama-3.3-70B-Instruct --dtype bfloat16 --max-model-len 8192 --cpu-offload-gb 32`.
  3. `vllm bench serve` with ShareGPT, concurrencies 1, 8, 32, 64, 128. Report tok/s and p50/p99 TTFT/ITL.
  4. Repeat on 1× H100 SXM with `--cpu-offload-gb 0` (it won't fit without offload — that *is* the result).
- **Cost:** ~$15 (~3 hr GH200 + ~2 hr H100).
- **Settles:** the GH200 inference-economics headline for the user's most likely model.

### Rank 8 — NIM aarch64 container availability (single docker-manifest check)
- **Why it matters:** Reports 02 §9, 07 §1 disagree on whether NIM ships arm64. The NVIDIA NIM matrix lists GH200 as supported; the actual `docker manifest inspect` does not show `linux/arm64` for headline Llama-3.1-70B images. If arm64 manifests have landed since the report date, NIM is back on the GH200 menu and licensing economics shift; if not, GH200 remains a vLLM/SGLang/TRT-LLM-direct deployment.
- **Source of gap:** Reports 02 §12, 07 (Round-2 follow-up #1).
- **Remediation type:** **WEB-R3** (no hardware needed — pure manifest pull from any Linux box).
- **Exact query:**
  ```
  docker manifest inspect nvcr.io/nim/meta/llama-3.1-70b-instruct:latest \
    | jq '.manifests[] | {arch:.platform.architecture, os:.platform.os}'
  ```
  Repeat for `llama-3.3-70b-instruct`, `nemotron-3-super-120b-a12b`, `nv-ingest`. If `linux/arm64` appears, NIM is usable on GH200.
- **Cost:** $0 (any box with `docker` + NGC token).
- **Settles:** whether NIM is a 2026 GH200 deployment option.

### Rank 9 — vLLM aarch64 NGC monthly tag for May/June 2026
- **Why it matters:** Report 03 §11 flagged that the monthly NGC tag (`vllm:26.05-py3`) wasn't surfaced. NGC monthly tags are the supported install path — if it ships, "just docker pull" replaces the manual-build install for GH200. Saves 1–3 days of install pain per user.
- **Source of gap:** Report 03 §11.
- **Remediation type:** **WEB-R3.**
- **Exact query:** fetch `https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm/tags`, verify arm64 manifest for 26.05 and 26.06 tags via `docker manifest inspect nvcr.io/nvidia/vllm:26.05-py3`.
- **Cost:** $0.
- **Settles:** zero-friction install vs. build-from-source.

### Rank 10 — Real Wan2.2 FP8/INT8 quality calibration recipe
- **Why it matters:** Report 06 §10 #4 ("calibrated FP8 recipe for Wan2.2 — what's the minimal `ignored_layers` set that makes outputs not collapse?"). If a calibrated FP8 path exists at LPIPS < 0.1 vs bf16, the user can ship FP8 Wan and recover ~2× throughput. If even calibrated FP8 fails on Wan's DiT, the bf16 rule is permanent.
- **Source of gap:** Reports 06 §10, 20 §8, vllm-omni #2728.
- **Remediation type:** **MEASURE** (this is the only remediation; no vendor has published a recipe).
- **Exact experiment:**
  1. 1× GH200, `torchao.quantization.float8_dynamic_activation_float8_weight()`.
  2. Apply to Wan2.2-A14B DiT blocks in groups; measure LPIPS per group on a fixed 8-prompt eval set.
  3. Bisect to find the minimum `ignored_layers` set that keeps LPIPS < 0.1.
  4. Open-source the resulting allowlist (helps the community + closes vllm-omni #2728 with data).
- **Cost:** ~$20 (≤8 GH200-hr — calibration runs are slower than inference).
- **Settles:** whether Wan FP8 is ever recoverable on Hopper.

### Rank 11 — TensorDock reliability: is the recent "+8 weeks of negative reviews" still ongoing in May 2026?
- **Why it matters:** Report 17 §8 flags TensorDock as having "multiple independent negative reviews" but doesn't pin the exact failure modes (billing? availability? data loss?). If TensorDock is in fact stable for GPU rentals at $2.29/hr H100, it materially expands the supplier list. If reliability is still bad, scratching them from consideration is justified.
- **Source of gap:** Report 17 §8.
- **Remediation type:** **WEB-R3.**
- **Exact query:** Trustpilot, LowEndTalk, r/LocalLLaMA for "TensorDock 2026"; fetch reviews dated after 2026-01-01; tabulate complaint categories. Also check tensordock.com/status uptime for last 90 days.
- **Cost:** $0.
- **Settles:** one additional H100 supplier on the menu (or not).

### Rank 12 — Lambda Stack on Ubuntu 26.04 ETA
- **Why it matters:** Report 08 recommends staying on 24.04 — but if Lambda publishes a 26.04 image in Q3 2026, the user can plan a controlled upgrade window. Without an ETA the decision is "wait indefinitely."
- **Source of gap:** Reports 08 §6, 16 §8.
- **Remediation type:** **SALES** (and partially WEB-R3 — check the `lambdal/lambda-stack-dockerfiles` repo monthly).
- **Exact query / experiment:**
  1. Open issue in `github.com/lambdal/lambda-stack-dockerfiles`: "Tracking — Ubuntu 26.04 Lambda Stack image plans?"
  2. Email Lambda support: "ETA for Lambda Stack on Ubuntu 26.04 GH200 image?"
- **Cost:** $0.
- **Settles:** the 26.04 upgrade calendar.

### Rank 13 — Bedrock Llama-3.3-70B per-Mtok actual rate
- **Why it matters:** Report 19 §6 flags a 3.7× discrepancy in published rates ($0.72 vs $2.65 per Mtok). For a managed-inference-as-fallback strategy this matters: at $0.72/Mtok Bedrock is competitive with self-serve GH200 ($1.50–$2.00/Mtok); at $2.65/Mtok it's much worse and rules out the fallback.
- **Source of gap:** Report 19 §6.
- **Remediation type:** **WEB-R3** (live AWS console required — counts as a "verify in live UI" task for Round 3).
- **Exact query:** log into AWS Bedrock console → Model access → Meta Llama 3.3 70B → "Pricing." Snapshot the canonical $/1M input + $/1M output rate.
- **Cost:** $0 (account access only).
- **Settles:** managed-fallback unit economics.

### Rank 14 — NCCL multi-node GH200 regression status (issue #1272 still open?)
- **Why it matters:** Reports 01 §6 and 09 §6 both flag NCCL 2.20+ multi-node bandwidth regression on GH200 (~24% drop). For single-node use, irrelevant; for any scale-out plan (e.g., 8× GH200 for DeepSeek-V3 671B FP8), this is a 1/4 of throughput on the table.
- **Source of gap:** Reports 01 §13, 09 (Confidence/Gaps).
- **Remediation type:** **WEB-R3.**
- **Exact query:** check `github.com/NVIDIA/nccl/issues/1272` status as of 2026-05-25; also pull NCCL 2.30.4 release notes searching for "GH200" or "sendrecv_perf"; check if a fix PR has been merged or if a workaround (env var) is documented.
- **Cost:** $0.
- **Settles:** whether multi-node GH200 is viable for DeepSeek-class deployments.

### Rank 15 — TPU v6e on-demand actual price (which of three sources is correct?)
- **Why it matters:** Report 15 §8 reports v6e Trillium prices clustering at $2.70/chip-hr with outliers at $1.375 and $4.20. A 3× ambiguity on the *only* live JAX alternative to GH200 makes the TPU comparison undecidable. If $1.375 is real (likely region-specific), TPU v6e undercuts GH200 dramatically for JAX-tolerant workloads.
- **Source of gap:** Report 15 §8.
- **Remediation type:** **WEB-R3.**
- **Exact query:** fetch `cloud.google.com/tpu/pricing` for v6e on-demand pricing per region (us-central1, us-east5, europe-west4, asia-northeast1). Also check `cloud.google.com/spot-vms/pricing` for v6e Spot rates. Document the region with the lowest verified rate.
- **Cost:** $0.
- **Settles:** TPU v6e as a real GH200 alternative.

---

## 2. Honourable mentions (lower decision impact — only chase if cycles permit)

| # | Gap | Type | Why lower priority |
|---|---|---|---|
| 16 | Quantitative `init_on_alloc=0` / `iommu.passthrough=1` delta on GH200 inference throughput | MEASURE | NVIDIA already recommends them; size of win unlikely to change the recommendation. |
| 17 | Exact CUDA-13 cuDNN minor version packaged in NVIDIA's arm64-sbsa apt repo (Report 01 §10) | WEB-R3 | One `apt list` on a fresh GH200; bookkeeping not decision-changing. |
| 18 | M3 Ultra 512GB real availability (refurb/used market) | WEB-R3 | Apple Silicon is already sub-optimal vs GH200 for serving (Report 13). |
| 19 | HiCache L2 throughput over NVLink-C2C on GH200 (SGLang) | MEASURE | Useful, but no decision hinges on it until SGLang is chosen as the engine. |
| 20 | TPU v7 (Ironwood) third-party Llama-3.3-70B benchmark | WEB-R3 | Ironwood is reservation-only; cannot deploy on-demand even if the number is good. |
| 21 | Cerebras CS-3 self-host price | SALES | $2–3M/chip rumour — buying hardware is not on the user's table. |
| 22 | d-Matrix Corsair vendor latency claims (60K TPS @ 1ms Llama 8B) | UNRESOLVABLE | No independent AA benchmark; vendor-only marketing. |
| 23 | MI300X FP8 MoE fix landing status in latest ROCm nightly | WEB-R3 | AMD is already 2nd-tier in this analysis; doesn't unlock a switch. |
| 24 | FP4 production stability on B200 over 3+ months | UNRESOLVABLE | Requires sustained production telemetry not yet in public domain. |
| 25 | Etched Sohu real performance | UNRESOLVABLE | Chip hasn't shipped 23 months post-announcement. |

---

## 3. Cross-report contradictions flagged for Round 3 verification

These are cases where two Round-1 agents stated factually incompatible things. Round 3 must pick a winner before synthesis.

| # | Topic | Agent A says | Agent B says | Resolution path |
|---|---|---|---|---|
| **C1** | **vLLM FP8 fix on Hopper applies to GH200** | Report 01 §8 / 03 §11: FA3 FP8 accumulation bug fixed in vLLM 0.19.1, "Hopper-generation" includes GH200. | Report 02 §5: "No public May-2026 source declares FP8 as universally fixed for the GH200/Hopper-SM90 path." | **MEASURE Rank 2.** Only a GH200-specific needle test settles it. |
| **C2** | **Wan FP8 — DiT-class bug vs GH200-specific** | Reports 06 §10, 20 §8: framing is "FP8 broken on DiT regardless of arch" (corroborated H100 + B200 in vllm-omni #2728). | User memory + Round 1 framing in early agents: "FP8 broken on this GH200/Wan stack." | **MEASURE Rank 6.** Reproduce Wan FP8 on H100; if it fails too, update user memory to "DiT-class FP8 issue, applies to all Hopper." |
| **C3** | **H200 marketplace floor price** | Report 17 §8: Nebius preemptible H200 $1.45/hr. | Report 18 §7: "Nebius committed 3+ months $2.30/hr" listed as floor for production. | **WEB-R3 Rank 3.** Both can be true (preempt vs commit) but the user needs to know preemption frequency to choose. |
| **C4** | **OCI H100/H200 effective rate** | Report 19 §6: $10/GPU/hr list — only verified. | Report 19 §6 (same agent): easecloud.io reports $4.10/GPU/hr effective with UCC commit. | **SALES** — need an actual Oracle account quote. List-vs-UCC gap is 2.5× and unbridgeable from public sources. |
| **C5** | **TPU v6e on-demand** | Report 15 §8: $2.70/chip-hr cluster, with $1.375 outlier. | Same agent flags $4.20–$4.50 as additional outlier. | **WEB-R3 Rank 15.** Live fetch of cloud.google.com pricing per region. |
| **C6** | **Lambda H200 self-serve hourly** | Report 16: "Lambda does not sell H200 self-serve." | Report 17 §8: aggregators (thundercompute) list $3.79/hr 1× H200 attributed to Lambda. | **WEB-R3.** Fetch lambda.ai/pricing directly. If H200 not in on-demand SKU list, aggregator is wrong; Round 1/16 wins. |
| **C7** | **GCP A3 Ultra (H200) on-demand $/GPU-hr** | Report 19 §1b: Vantage shows $1.533/GPU-hr — third-party. | Same report: another source quotes ~$10.60/GPU-hr OD. ~7× gap. | **WEB-R3.** Fetch cloud.google.com/compute/gpus-pricing for a3-ultragpu-8g; live SKU lookup. |
| **C8** | **B100 status** | Report 11: B100 effectively cancelled per KeyBanc analyst. | Reports 11, 12 elsewhere reference B100 specs in tables. | **Documentation hygiene only.** B100 is dead; tables should drop the row in Round 3 synthesis. |
| **C9** | **TGI archived date** | Report 05: TGI archived 2026-03-21. | Report 07: TGI mentioned as adjacent serving framework (no archive flag in this report). | **WEB-R3.** Check github.com/huggingface/text-generation-inference last commit/archive marker. Report 05 likely correct. |
| **C10** | **NCCL latest stable** | Report 01 §6: NCCL 2.30.4 (2025-04-22 — note year/date is from 2025). | Report 09 §6: refers to NCCL 2.30.3 in env-var docs. | **WEB-R3.** Fetch github.com/NVIDIA/nccl/releases; 2.30.4-1 should be current. Confirm both year-tagged release date and minor version. |

---

## 4. Suggested measurement-budget plan

If the user authorises a **$100 hands-on measurement budget** (≈40 GH200-hr or equivalent), the best ROI is:

| Order | Experiment | Resolves | Est. cost |
|---|---|---|---|
| 1 | **Rank 1** — Wan2.2-A14B 720p single GH200 vs single H100 | The GH200 retention case | $15 |
| 2 | **Rank 7** — Llama-3.3-70B vLLM 0.19 bf16 GH200 throughput sweep | The LLM-economics case | $15 |
| 3 | **Rank 2** — vLLM FP8 needle test on GH200 | The user-memory rule | $10 |
| 4 | **Rank 6** — Wan FP8 reproduction on H100 (DiT-vs-GH200) | Updates the user memory scope | $8 |
| 5 | **Rank 10** — Wan FP8 calibration recipe (`torchao` + bisect) | Whether FP8 Wan is ever recoverable | $20 |
| 6 | **Rank 3** (measure part) — Nebius H200 preemptible 4-hr serve test | Migration alternative | $6 |
| **Total** | | | **~$74** |

Remaining budget reserves for re-runs / surprise re-measurements after Round 3 publishes verified vendor quotes.

---

## 5. Round-3 SALES email queue (parallel-dispatchable)

Five emails the user can send in one batch — each unblocks a high-impact gap:

1. **Lambda Labs** — "Quote 1× GH200 96GB on 6-mo and 12-mo commit; us-south-1 or us-east; Lambda Stack 24.04. Include SLA." (Rank 4)
2. **CoreWeave** — "Reservation tier — 12-mo — 1× H200 and 8× B200; Quantum-2 fabric. $/GPU-hr + minimum commit." (Rank 5)
3. **Nebius** — "Committed 6-mo H200 quote + preemptible H200 SLA / preemption-frequency documentation." (Rank 3)
4. **Oracle OCI** — "BM.GPU.H100.8 and BM.GPU.H200.8 — actual UCC-discounted $/GPU-hr at 1-yr and 3-yr commit at $1M and $5M tiers." (C4)
5. **Lambda Labs support** — "ETA for Lambda Stack on Ubuntu 26.04 GH200 image." (Rank 12)

All five are free, deliverable to vendor inbox within an hour, and should round-trip within a week. After answers land, Round 4 (synthesis) has the unit-economics inputs it needs.

---

## 6. What we explicitly chose NOT to chase

To keep Round 3 focused, the following gaps are de-prioritised:

- **Hyperscaler GH200 SKUs.** Report 19 §6 confirmed they don't sell GH200; nothing to chase.
- **Specialty-chip self-host pricing** (Cerebras, d-Matrix, Furiosa, Rebellions). User is not in the buy-hardware lane.
- **Apple Silicon FLUX/SDXL throughput.** Report 13 conclusively shows Apple loses on prefill; tighter Apple numbers won't flip the decision.
- **Gaudi 3.** Report 15 declares the product line sunsetting; no point in better numbers.
- **Etched Sohu.** Vapor.
- **AWS Bedrock per-Mtok dispute beyond Llama 3.3 70B** — single live-console check (Rank 13) covers the load-bearing model.
- **Vast.ai marketplace minute-by-minute pricing.** Volatile; not a sustained-monthly play.

---

## 7. Files this depends on / produces

- **Inputs:** `/home/ubuntu/research/gh200_inference/round1/01_driver_stack.md` … `20_diffusion_video.md`.
- **Output:** this file (`/home/ubuntu/research/gh200_inference/round2/G_gap_verification_roadmap.md`).
- **Downstream consumers:** Round 3 web-research and adversarial-validation agents, Round 4 synthesis. Use Section 1 as the work-queue and Section 3 as the contradiction-resolution checklist.
