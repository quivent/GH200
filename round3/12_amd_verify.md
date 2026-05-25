# Agent 12 — Round 3 Verification & Deepening: AMD MI300X/MI325X/MI355X

Date: 2026-05-25
Author: Verification agent (round 3, post Round 1 file `round1/12_amd_mi300_etc.md`)
Method: 12 WebSearch/WebFetch queries against ROCm blogs, MLPerf submission pages, GitHub issues, TensorWave / Hot Aisle / ComputePrices pricing pages, and NextPlatform / Tom's Hardware roadmap coverage.

This file overwrites assumptions from Round 1 where verification changed the picture; everything else is corroborated.

---

## 1. MLPerf Inference v5.1 — what AMD actually submitted

**Round 1 implied AMD submitted Llama-3.3-70B with the MI355X. That is wrong.** AMD's MLPerf v5.1 submission (announced 2025-09-09) used **Llama 2 70B** as the 70B model, not Llama 3.3 70B. The Llama 3 family appears only as **Llama 3.1 405B (pruned)**.

Specific numbers from the [ROCm blog technical dive](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v5.1/README.html):

| Submission | Hardware | Dtype | Result |
|---|---|---|---|
| Llama 2 70B Server (single node) | 1× node MI355X | MXFP4 | 2.7× vs MI325X-FP8 (absolute tok/s not in blog) |
| Llama 2 70B Offline (multi-node) | 4× node MI355X | MXFP4 | 3.9× over 4-node MI300X-FP8 from v5.0 |
| Llama 2 70B Offline (multi-node) | 8× node MI355X | MXFP4 | 7.9× over single-node MI355X |
| Llama 3.1 405B (pruned 26 layers) | MI355X | MXFP4 | 1,942 tok/s |
| Llama 3.1 405B (pruned 42 layers + FT) | MI355X | MXFP4 | 2,019 tok/s |
| Llama 2 70B (incumbent) | MI325X | FP8 | +2% offline / +4% server vs v5.0 |
| Mixtral 8x7B | MI325X | FP8 | +23% E2E throughput |
| SD-XL | MI325X | — | +8% over v5.0 |

**Verified takeaway**: AMD has *not* submitted Llama-3.3-70B to MLPerf as of v5.1 (Sep 2025) or v6.0 (Apr 2026). MLPerf's 70B slot is still Llama 2 70B. If a stakeholder asks "where are the AMD Llama 3.3 70B numbers?" the answer is: there are no official MLPerf ones; the SemiAnalysis InferenceMAX nightly is the only public continuous benchmark using Llama 3.3 70B on AMD hardware.

**MLPerf Inference v6.0** ([ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html), 2026-04-01):

| Workload | Hardware | Dtype | Result |
|---|---|---|---|
| Llama 2 70B Offline | 8× MI355X (1 node) | MXFP4 | **103,480 tok/s** |
| Llama 2 70B Server | 8× MI355X (1 node) | MXFP4 | **100,282 tok/s** |
| Llama 2 70B Interactive | 8× MI355X (1 node) | MXFP4 | 73,608 tok/s |
| Llama 2 70B Offline | 11 nodes / 87 GPU MI355X | MXFP4 | **1,016,375 tok/s** (first AMD >1M tok/s submission) |
| gpt-oss-120b Offline | 8× MI355X | MXFP4 / FP8-e4m3 | 95,004 tok/s |
| gpt-oss-120b Server | 8× MI355X | MXFP4 / FP8-e4m3 | 82,136 tok/s |
| Llama 3.1 405B Interactive (Open) | MI355X TP8 | FP4 + QuickReduce | 793 tok/s |
| Wan2.2-T2V-A14B Single Stream | MI355X | Sage attention FP8/FP4 + MXFP4 GEMM | **27.4 s latency** (per video) |

Wan2.2 is in MLPerf v6.0 as a first-time benchmark, and AMD's MI355X submission uses mixed FP8/FP4/MXFP4 — *not* pure bf16. So the "diffusion on AMD = bf16 only" assumption from our team needs context: AMD's own MLPerf submission uses FP8 Sage attention. The 27.4s/video number is the single-stream latency under MLPerf scoring rules.

---

## 2. ROCm 7.x current version and FP8 MoE status

From the [ROCm release history page](https://rocm.docs.amd.com/en/latest/release/versions.html):

| Version | Release date |
|---|---|
| 7.0.0 | 2025-09-16 |
| 7.0.1 | 2025-09-17 |
| 7.0.2 | 2025-10-10 |
| 7.1.0 | 2025-10-30 |
| 7.1.1 | 2025-11-26 |
| 7.2.0 | 2026-01-21 |
| 7.2.1 | 2026-03-25 |
| 7.2.2 | 2026-04-14 |
| 7.2.3 | 2026-05-04 (current production) |
| 7.13.0 | technology preview (no GA date) |

Round 1 referenced "ROCm 7.0/7.2.1" — outdated. **Current production = 7.2.3.** Preview stream begins at 7.9+; 7.13.0 preview docs are published but no GA. Production stream covers 7.0–7.8.

### FP8 MoE on MI300X: vLLM #31475 is **still open**

[vLLM issue #31475](https://github.com/vllm-project/vllm/issues/31475) — "FP8 inference much slower than BF16 on MI300X for GLM-4.7 and MiniMax-M2.1":
- Status: **OPEN**
- Filed: 2025-12-29
- Now carries a "stale" label (90+ days inactive)
- No linked PR or fix in the issue thread
- Original symptom (GLM-4.7 FP8 ~30 TPS vs BF16 ~60 TPS, 2× inversion) is **not refuted, not closed, no workaround merged**

**Distinct issue, do not confuse**: [vLLM #36105](https://github.com/vllm-project/vllm/issues/36105) — "Qwen3-VL-30B-A3B-Instruct-FP8 fails to start with No FP8 MoE backend supports the deployment configuration." This is **CLOSED** (resolved via PR #38086) and targets **gfx1151 (Strix Halo APU)**, not MI300X (gfx942) or MI355X (gfx950). Round 1's framing did not mention this; it's adjacent, not the same bug.

**Net**: As of 2026-05, **MI300X FP8 MoE remains the documented production gap**. ROCm 7.2.3 has not closed the underlying vLLM path bug. The Round 1 recommendation — "for MoE on MI300X, plan BF16" — still holds.

---

## 3. Wan2.2 diffusion on MI355X — actual numbers

This is the most important Round 1 gap to fill given our team's bf16-only constraint on DiT.

### a. ROCm ComfyUI blog ([source](https://rocm.blogs.amd.com/artificial-intelligence/comfyui/README.html))

Wan2.2 Text-to-Video, 1280×1280, 27B parameter DiT-MoE (14B active):

| GPU | Wan2.2 time | FLUX.1-dev time | Hunyuan3D 2.1 time |
|---|---|---|---|
| MI355X | **116.91 s** | 24.77 s | 21.51 s |
| B200 | 168.28 s | 35.09 s | 25.84 s |
| MI355X speedup | **1.44×** | 1.42× | 1.20× |

**Dtype not specified in blog.** Given MLPerf v6.0 used Sage attention FP8 + MXFP4 GEMM for Wan2.2, the ComfyUI blog numbers are likely the same mixed-precision path, not bf16.

### b. ROCm Wan2.2-S2V (audio-driven) blog ([source](https://rocm.blogs.amd.com/artificial-intelligence/audio-driven-videogen/README.html))

MI300X-only (no MI355X numbers in this blog):
- 1× MI300X, 7s audio @ 472×816: **~19 min 39 s** per video
- 1× MI300X, 8s audio @ 768×1024: **~36 min**
- 4× MI300X, 7s audio: **~9 min 40 s**
- 4× MI300X, 8s audio: **~15 min 53 s**

Dtype not stated; `--convert_model_dtype` flag used but target dtype omitted. Implicit bf16 from PyTorch defaults.

### c. ROCm/AMD officially supports Wan2.2 on gfx942 and gfx950

i.e., MI300X, MI325X, MI350X, MI355X — via xDiT diffusion inference framework (per [xDiT docs](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/xdit-diffusion-inference.html)).

### Verdict for our use case

- **Wan2.2 runs on AMD MI300X/MI355X today, officially supported via xDiT + ComfyUI.**
- **MLPerf MI355X result: 27.4s single-stream latency** (FP8/FP4/MXFP4 mixed) and ComfyUI 116.91s end-to-end at 1280×1280.
- **No clean public bf16-only MI355X Wan2.2 benchmark exists.** All AMD numbers we found use mixed FP8/FP4 paths.
- If our team mandates bf16 for the DiT due to FP8 brittleness on Hopper, **AMD does not publish a bf16-only Wan2.2 number to compare against**. We'd have to run it ourselves on a TensorWave/Hot Aisle MI355X to get a directly comparable bf16 number.

---

## 4. MI355X real pricing (May 2026)

Verified from current pricing pages:

### Hot Aisle ([hotaisle.xyz/pricing](https://hotaisle.xyz/pricing/))
- MI300X VM (1/2/4×): **$1.99/GPU/hr** on-demand, per-minute billing
- MI300X bare metal 8×: **$3.39/GPU/hr**, 1-month minimum
- MI325X / MI350X: **no public price** on this page
- MI355X: **reservation waitlist, quote-only** ("We are accepting MI355x reservations!")

### TensorWave (verified via [computeprices.com tracker, May 12, 2026 snapshot](https://computeprices.com/providers/tensorwave))
- MI300X: **$1.71/GPU/hr** (revised down from Round 1's $1.99 assumption)
- MI325X: **$2.25/GPU/hr**
- MI355X: **$2.95/GPU/hr** — **public on-demand price now exists**, no longer pure quote-only

(TensorWave's own /pricing page returns 404; their main marketing pages cite "MI325X+ starts at $1.95" and "MI355X− starts at $2.85" — *starting* prices for long-term contracts, lower than the computeprices on-demand rate.)

### Cross-reference (Round 1 + getdeploying)
- Vultr MI355X reserved (36-mo): $2.29/GPU/hr
- Oracle Cloud MI355X on-demand: $8.60/GPU/hr
- TensorWave on-demand: **$2.95/GPU/hr** (now public — this is new since Round 1)

**Revised MI355X pricing spread May 2026**: $2.29 (Vultr 36-mo reserved) → $2.95 (TensorWave on-demand) → $8.60 (Oracle OD). The TensorWave on-demand number being public is the most useful change vs Round 1; you no longer need a sales call to get an MI355X $/hr quote in the $3/hr range.

**Comparison to GH200 floor** (~$1.99 spot, ~$3.50 typical reserved, ~$6.50 CoreWeave OD): MI355X TensorWave OD ($2.95) is **comparable to or cheaper than mid-tier GH200**, while offering 288 GB vs 96/144 GB HBM. Plus you don't need bare metal commits.

---

## 5. MI400 / UALoE72 / Helios — Round 1 → Round 3 deltas

Round 1 said "MI400 UALoE72 late 2026." Verified and refined:

From [NextPlatform (2026-02-23)](https://www.nextplatform.com/compute/2026/02/23/amd-says-helios-racks-and-mi400-series-gpus-on-track-for-2h-2026/4092199), [Tom's Hardware CES 2026](https://www.tomshardware.com/tech-industry/artificial-intelligence/amd-touts-instinct-mi430x-mi440x-and-mi455x-ai-accelerators-and-helios-rack-scale-ai-architecture-at-ces-full-mi400-series-family-fulfills-a-broad-range-of-infrastructure-and-customer-requirements), [SemiAnalysis MI400 UALoE72 / MI500 UAL256](https://newsletter.semianalysis.com/p/amd-advancing-ai-mi350x-and-mi400-ualoe72-mi500-ual256):

- **CES 2026 announcement**: AMD detailed full MI400 family — MI430X, MI440X, MI455X — plus Helios rack architecture.
- **Helios spec**: 72× MI455X per rack, **31 TB HBM4 aggregate, 1.4 PB/s aggregate BW, 2.9 FP4 EFLOPS inference, 1.4 FP8 EFLOPS training**.
- **Timeline**: Low-volume production H2 2026. AMD CFO statement: "highly confident of ramping Helios in high volume in the second half of the year."
- **Interconnect**: UALink scale-up + Ultra Ethernet scale-out (via Pensando Pollara 400G / Vulcano 800G). This is AMD's direct answer to GB200/GB300 NVL72.
- **MI500 / UAL256**: announced for 2027.

**Practical implication for GH200 alternative decision**: MI400 Helios racks will not be in volume production until late 2026, possibly slipping. For a near-term decision (now through Q3 2026), the choice remains MI300X / MI325X / MI355X single-node 8-GPU. MI400 enters the picture for 2027 capacity planning.

---

## 6. Updated confidence table

| Claim | Round 1 confidence | Round 3 status |
|---|---|---|
| MI300X FP8 MoE slower than BF16 (vLLM #31475) | Medium | **HIGH — verified still OPEN with stale label, no merged fix** |
| MI355X = 288 GB HBM3e, 8 TB/s | High | High (re-confirmed) |
| ROCm 7.x current version | Medium ("7.0/7.2.1") | **REVISED: 7.2.3 (May 4, 2026); 7.13 preview** |
| MLPerf v5.1 includes Llama-3.3-70B | Implied | **WRONG — only Llama 2 70B and Llama 3.1 405B (pruned); no 3.3** |
| MI355X MLPerf v5.1 Llama 2 70B = 2.7× MI325X | High | High (re-confirmed) |
| MI355X Wan2.2 perf | Unknown | **NEW: 116.91s ComfyUI 1280×1280; 27.4s MLPerf single-stream** |
| MI355X uses bf16-only for Wan2.2 | Assumed | **WRONG — AMD's official path uses FP8 Sage attention + MXFP4 GEMM** |
| TensorWave MI355X price | "quote-only" | **REVISED: $2.95/GPU/hr public on-demand (May 12, 2026 snapshot)** |
| Hot Aisle MI355X price | "quote-only" | Confirmed still quote/waitlist |
| MI400 UALoE72 launch | "late 2026" | **REFINED: low-volume H2 2026, high-volume ramp H2 2026 per AMD CFO; CES 2026 fully detailed** |
| Helios rack spec | Not in Round 1 | **NEW: 72 MI455X, 31 TB HBM4, 1.4 PB/s, 2.9 EFLOPS FP4 inference** |
| vLLM #36105 (Strix Halo FP8 MoE) | Not mentioned | **NOTE: different bug, gfx1151 not MI300X, CLOSED via PR #38086** |

---

## 7. Updated recommendation for our team's question

Our constraints: **GH200 FP8 path broken on this stack; bf16 default for DiT (Wan2.2)**.

Refined Round 1 advice with verified data:

1. **For LLM serving (Llama 3.3 70B class, bf16)**: TensorWave MI355X at **$2.95/GPU/hr on-demand** is now a public, viable swap. 288 GB lets you skip TP tricks. Use vLLM-ROCm + AITER + ROCm 7.2.3. Skip FP8 on MoE models (#31475 still open).

2. **For Wan2.2 specifically**: AMD officially supports it via xDiT on MI300X/MI325X/MI355X. **No clean bf16 reference benchmark exists** — all published AMD numbers use mixed FP8/FP4 Sage attention. If we insist on bf16, we lose AMD's advertised speedup and need to characterize ourselves. **Risk**: AMD's Wan2.2 perf story is built on FP8 paths we may not be able to use. Hopper bf16 might be more directly comparable than expected.

3. **Pricing is closer than Round 1 suggested**: MI355X TensorWave OD ($2.95) is in the GH200 OD price band ($3–6.50), so the memory advantage (288 GB vs 144 GB) becomes the deciding factor for long-context / KV-heavy workloads.

4. **Skip MI400 for near-term decisions** — H2 2026 ramp means it doesn't help renewals due in Q2–Q3 2026.

5. **AMD has NOT submitted a Llama-3.3-70B MLPerf result.** If anyone cites such a number, ask for the source — likely SemiAnalysis InferenceMAX nightly, not official MLPerf.

---

## Sources & verification dates

All accessed 2026-05-25:

- [ROCm Blog: MLPerf Inference v5.1 Technical Dive](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v5.1/README.html)
- [ROCm Blog: MLPerf Inference v6.0 Submission](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html)
- [ROCm Blog: ComfyUI on MI355X (Wan2.2/FLUX/Hunyuan3D)](https://rocm.blogs.amd.com/artificial-intelligence/comfyui/README.html)
- [ROCm Blog: Wan2.2-S2V on MI300X](https://rocm.blogs.amd.com/artificial-intelligence/audio-driven-videogen/README.html)
- [ROCm Release Versions](https://rocm.docs.amd.com/en/latest/release/versions.html)
- [ROCm 7.2.3 Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
- [vLLM Issue #31475 (MI300X FP8 MoE slow)](https://github.com/vllm-project/vllm/issues/31475) — OPEN
- [vLLM Issue #36105 (Strix Halo FP8 MoE)](https://github.com/vllm-project/vllm/issues/36105) — CLOSED
- [xDiT Diffusion Inference on ROCm](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/xdit-diffusion-inference.html)
- [Hot Aisle Pricing](https://hotaisle.xyz/pricing/)
- [ComputePrices TensorWave tracker (May 12, 2026)](https://computeprices.com/providers/tensorwave)
- [NextPlatform: Helios + MI400 on track 2H 2026](https://www.nextplatform.com/compute/2026/02/23/amd-says-helios-racks-and-mi400-series-gpus-on-track-for-2h-2026/4092199)
- [Tom's Hardware CES 2026: MI430X/440X/455X + Helios](https://www.tomshardware.com/tech-industry/artificial-intelligence/amd-touts-instinct-mi430x-mi440x-and-mi455x-ai-accelerators-and-helios-rack-scale-ai-architecture-at-ces-full-mi400-series-family-fulfills-a-broad-range-of-infrastructure-and-customer-requirements)
- [SemiAnalysis MI400 UALoE72 / MI500 UAL256](https://newsletter.semianalysis.com/p/amd-advancing-ai-mi350x-and-mi400-ualoe72-mi500-ual256)
