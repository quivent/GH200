# H100 / H200 as Alternatives to GH200 for Inference

**Agent #10 of 20 — GH200 alternatives research**
**Date:** 2026-05-25
**Scope:** NVIDIA H100 (SXM5 80GB, PCIe NVL 188GB), H200 (SXM 141GB, NVL 282GB/564GB) vs GH200 96GB HBM3 / 144GB HBM3e Superchip.

---

## 1. TL;DR

- **H200 SXM (141GB HBM3e, 4.8 TB/s)** is the most direct, lower-risk replacement for GH200 96GB. It has *more* HBM than the original GH200 96GB variant, identical Hopper compute, and lives in the standard 8-GPU NVSwitch HGX form factor — which means real all-to-all bandwidth and zero ARM/Grace dependency. The GH200's only remaining edge is the 480GB LPDDR5X behind the 900 GB/s NVLink-C2C — useful only for KV offload at extreme contexts.
- **H100 SXM 80GB** is *not* a clean swap-in for GH200 96GB at the 70B FP8 weight tier — it fits the weights (~70 GB) but starves the KV cache, capping batch/context. Use it for ≤30B dense models, MoE shards, or as a 2×H100 pair for 70B BF16.
- **FP8 reliability:** H100/H200 FP8 path is **production-grade** for dense Llama-class models (vLLM blog April 2026 documents and *fixes* the FA3 accumulation bug; FriendliAI / NVIDIA NIM / Red Hat AI ship pre-quantized FP8 70B routinely; MMLU delta typically <0.5%). FP8 issues that *do* exist are MoE-specific (vLLM #30830, TRT-LLM autotuner crash on Qwen3-Next, MoE online-quant accuracy regressions) and **affect GH200 and H100/H200 equally** — they are not Hopper-arch problems, they are MoE-stack problems. The "FP8 broken on Wan/GH200 stack" finding does **not** generalize to H100/H200 dense LLM inference.
- **Pricing floor (May 2026, on-demand single GPU/hr):**
  - H100 SXM5 80GB: $1.38 (Thunder) – $1.99 (RunPod) – $3.99 (Lambda) – $6.16 (CoreWeave) – $11.06 (GCP)
  - H200 SXM 141GB: $2.09 (Sesterce) – $2.29 (Vast / Theta) – $3.79 (Lambda / RunPod NVL) – $4.39 (RunPod SXM) – $6.31 (CoreWeave) – $10.60 (Azure)
  - GH200 96GB: $1.99 (Vultr) – $3.19 (Lambda PC) – $5.99 (Lambda Clusters) – $6.50 ceiling
- **On-prem capex (May 2026):** ~$30–40K per H100 80GB chip; ~$30–40K per H200 chip; 8×H100 HGX node ~$350–450K; 8×H200 HGX node ~$315K board / $400–500K full server. GH200 single-chip systems hard to source in volume; NVL2 nodes ship from HPE/Supermicro but allocation is constrained, with primary cloud channels being Lambda / Vultr / CoreWeave.

---

## 2. Side-by-side Specification Table

| Property | H100 SXM5 80GB | H100 NVL (PCIe pair, 188GB) | H200 SXM 141GB | H200 NVL (PCIe, 141GB; 2-way 282GB; 4-way 564GB) | GH200 96GB HBM3 | GH200 144GB HBM3e (NVL2 variant) |
|---|---|---|---|---|---|---|
| HBM | 80 GB HBM3 | 2×94 GB HBM3 = 188 GB | 141 GB HBM3e | 141 GB HBM3e per card | 96 GB HBM3 | 144 GB HBM3e |
| Bandwidth | 3.35 TB/s | 2×3.9 TB/s = 7.8 TB/s aggregate | 4.8 TB/s | 4.8 TB/s per card | ~4.0 TB/s | ~4.9 TB/s |
| FP8 TFLOPS (sparse) | 3,958 | 2×3,341 = 6,682 | 3,958 | 3,341 per card | 3,958 | 3,958 |
| CPU memory pool over NVLink-C2C | none (PCIe Gen5, ~64 GB/s) | none | none | none | 480 GB LPDDR5X @ 900 GB/s | 480 GB LPDDR5X @ 900 GB/s |
| Multi-GPU fabric | NVSwitch 4 (900 GB/s/GPU, 8-way all-to-all) | 3×NVLink-4 bridges (2-way only) | NVSwitch 4 (900 GB/s/GPU, 8-way all-to-all) | 2-way (900 GB/s) or 4-way (1.8 TB/s, 564 GB combined) | Single chip; NVL2 = 2 chips @ 900 GB/s | NVL2 (2 chips) up to 256 via DGX GH200 |
| TDP | 700 W | 700–800 W combined | 700 W | 600 W per card | 900 W (chip module) | 1,000 W |
| Form factor | HGX 8-GPU server | PCIe dual-slot bridged pair | HGX 8-GPU server | PCIe 2-way / 4-way bridged | Custom Grace+Hopper module | NVL2 mainboard |

Sources: NVIDIA H100 / H200 product pages, PNY H200 NVL datasheet, TechPowerUp H200 NVL spec, Spheron / NADDOD topology write-ups (linked below).

---

## 3. Inference Throughput — Llama 2/3 70B (Single-node, FP8 unless noted)

| Hardware | Workload | Throughput | Source / MLPerf round |
|---|---|---|---|
| 8×H100 SXM | Llama 2 70B Offline | 22,290 tok/s | MLPerf Inference v4.0 (Lambda / NVIDIA) |
| 8×H100 SXM | Llama 2 70B Server | 21,504 tok/s | MLPerf Inference v4.0 |
| 8×H200 SXM | Llama 2 70B Offline | 31,712 tok/s (+42%) | MLPerf Inference v4.0; CoreWeave v5.0 reported ~33,000 tok/s |
| 8×H200 SXM | Llama 2 70B Server | 29,526 tok/s (+37%) | MLPerf Inference v4.0 |
| GH200 NVL2 (Supermicro/HPE) | Llama 2 70B Offline | "Outstanding" per NVIDIA blog; specific number not released in v4.1 article; Red Hat OpenShift submission ran on dual-GH200 144GB | MLPerf Inference v4.1 (NVIDIA / Red Hat / Supermicro) |
| 1×GH200 96GB (vLLM, FP8) | Llama 3.3 70B, batch 32 | +32% over 1×H100 80GB at same batch; gain came mostly from larger KV headroom (Baseten / Lambda) | Baseten 2024 blog |
| 1×GH200 96GB (vLLM, BF16 + CPU offload) | Llama 70B BF16 single GPU | 4.33 tok/s vs 0.57 tok/s on 1×H100 80GB; ≈7.6× — because BF16 70B does not fit on 80 GB and offloads over PCIe vs NVLink-C2C | Lambda "Putting GH200 to good use" blog |
| 1×H200 (vLLM/SemiAnalysis ref data, FP8) | Llama 3.3 70B | b=1 ~680 tok/s · b=8 ~1,800 · b=32 ~2,500 · b=128 ~3,200 | Spheron Gaudi-3 vs H200/B200 (2026) |
| 1×H100 80GB (TensorRT-LLM, FP8) | Llama 70B | 250–300 tok/s single stream; ~24K tok/s aggregate at large batch | NVIDIA dev blog, Perplexity Hub |

**Single-chip GH200 vs 1×H100 numbers are dramatic only in BF16/no-quant**, where the H100's 80 GB cannot hold the weights and falls back to CPU offload over PCIe at 64 GB/s; the GH200 hits CPU memory at 900 GB/s NVLink-C2C, so it isn't a fair compute comparison — it's a memory-fabric comparison. **In FP8 the gap collapses to ~30%** (Baseten Llama-3.3 70B test).

**Node-level GH200 NVL is what you should care about**, not single GH200 vs single H100. At 8-GPU scale, H200 HGX already gives you 1,128 GB of HBM3e behind a real switched NVLink fabric, which is enough for Llama-3.1-405B FP8, DeepSeek V3/R1 671B FP8, and any 70B-class model with effectively unbounded KV cache — none of which fit on one GH200 96GB.

---

## 4. Memory Headroom — Why H200 Beats GH200 96GB for Most Inference

| Model | FP8 weights | BF16 weights | Comfortable on 1 GPU? |
|---|---|---|---|
| Llama-3.1/3.3 70B | ~70 GB | ~140 GB | H100 80GB: weights only, ~10 GB for KV → small batches. **GH200 96GB: weights + 26 GB KV → OK at batch ≤32 short context.** **H200 141GB: weights + 71 GB KV → comfortable at batch 128 / 32k ctx.** |
| Qwen2.5-72B | ~72 GB | ~144 GB | H100 80GB: tight (weights barely fit). GH200 96GB: identical envelope to Llama 70B FP8. **H200 141GB: comfortable.** |
| Llama-3.1 405B | ~405 GB | ~810 GB | Single chip: none. **Best on 8×H200 (1,128 GB).** 8×H100 needs INT4 or tensor-parallel across two nodes. GH200 NVL2 (2 chips, 288 GB HBM) needs 4× nodes networked. |
| DeepSeek V3 / R1 671B (FP8 native) | ~671 GB | n/a (released FP8) | 8×H100 80GB = 640 GB → **does not fit, requires 2-node deployment**. 8×H200 = 1,128 GB → **single node** with 450+ GB KV headroom. Lambda/Baseten and Perplexity confirm single-node 8×H200 is the deployment sweet spot. |

The structural point: **the GH200 96GB variant has *less* HBM than an H200, so for pure GPU-resident inference, GH200 96GB is strictly worse than H200 141GB.** The GH200's 480 GB LPDDR5X CPU pool only helps if (a) your workload deliberately offloads KV cache to host, and (b) your inference engine supports that (vLLM has unified-memory offload patches; SGLang/TRT-LLM less mature). The GH200 NVL2 144GB HBM3e variant edges H200 141GB only by 3 GB per chip — negligible — and at +50% cost / +280W TDP.

---

## 5. NVLink Topology — Where GH200 Loses to H200 HGX for Multi-GPU

| Topology | All-to-all BW per GPU | Total HBM in domain | Useful for |
|---|---|---|---|
| 8×H100 / 8×H200 HGX (NVSwitch Gen4) | 900 GB/s | 8×80 / 8×141 = 640 / 1,128 GB | tensor-parallel 70B–405B FP8, DeepSeek V3 single node |
| H100 NVL / H200 NVL 2-way bridge | 900 GB/s pair only | 188 GB / 282 GB | air-cooled enterprise PCIe rack, no all-to-all |
| H200 NVL 4-way (new in 2025) | 1.8 TB/s aggregate, 564 GB combined | 4×141 = 564 GB | rack-scale enterprise inference without HGX |
| GH200 single chip | n/a (single chip) | 96 / 144 GB | self-contained small/medium inference |
| GH200 NVL2 (2 chips) | 900 GB/s NVLink-C2C | 192 / 288 GB HBM + 960 GB LPDDR | KV-offload-heavy long context |
| DGX GH200 (256 chips, NVLink Switch) | 900 GB/s mesh | 24–36 TB unified | hyperscale, not retail-available |

**For Llama-3.1-405B and DeepSeek V3/R1 FP8, the 8×H200 HGX node is currently the cheapest single-server deployment in the market.** No GH200 configuration short of DGX GH200 matches it without crossing a node boundary.

---

## 6. Cloud Pricing — May 2026

### H100 SXM5 80GB on-demand $/GPU/hr
| Provider | Price | Notes |
|---|---|---|
| Thunder Compute | $1.38 | floor, single-GPU |
| Hyperbolic | $1.50 | marketplace |
| Vast.ai | ~$1.53 | community |
| RunPod | $1.99 | community pod |
| Lambda | $3.78–$3.99 | 1-GPU on-demand |
| CoreWeave | $6.16 | normalized from 8-GPU HGX |
| AWS p5 | $6.88 | |
| Azure NC40 | $8.59 | |
| GCP a3-highgpu | $11.06 | |

### H200 SXM 141GB on-demand $/GPU/hr
| Provider | Price | Notes |
|---|---|---|
| Sesterce | $2.09 | floor |
| Vast.ai / Theta EdgeCloud / FluidStack | $2.29–$2.30 | |
| Koyeb | $3.00 | |
| Lambda | $3.79 | minute-billed |
| RunPod (Community / Secure NVL / Secure SXM) | $3.59 / $3.79 / $4.39 | |
| Crusoe | $4.29 | |
| Together | $4.99 | |
| CoreWeave | $6.31 | 8-GPU node normalized |
| Oracle / Azure | $10.00 / $10.60 | enterprise |

### GH200 96GB / 144GB on-demand $/GPU/hr
| Provider | Price | Notes |
|---|---|---|
| Vultr | $1.99 | market floor |
| Lambda Public Cloud | $3.19 | |
| Lambda Cloud Clusters | from $5.99 | |
| CoreWeave / specialized | up to $6.50 | |

### Per-Million-Token Headline Numbers (Llama-3.3 70B, FP8 unless noted)
| Configuration | $/M-tok | Provider / source |
|---|---|---|
| H100 SXM5 8×, FP8 vLLM | ~$1.15 | Spheron 2026 bench (3-node, 5,600 tok/s aggregate) |
| H100 SXM5 1×, FP8 (NVIDIA dev blog) | ~$0.71 on-demand BF16 was $1.19 | NVIDIA TRT-LLM technical blog |
| H200 SXM5 1×, FP8, batch 32, $4.22/hr | ~$0.47 | Spheron Gaudi-3 vs H200/B200 (2026) |
| H200 SXM5 8×, FP16 vLLM, 3,600 tok/s | $2.28 | Spheron 2026 |
| H200 spot (Q8/INT8) $0.33/hr | ~$0.042 | GPU Tracker 2026 |
| GH200 1× vLLM Llama-3.3 70B FP8 batch 32 | not published in $/M; Baseten reports +32% throughput vs 1×H100 at $3.19/hr ≈ same hourly | Baseten / Lambda blog |
| Together AI hosted Llama-3.3 70B | $0.88 blended | Artificial Analysis |
| Fireworks AI hosted Llama-3.3 70B | $0.90 | TokenMix |
| DeepInfra Turbo FP8 hosted | $0.12 blended ($0.10 in / $0.32 out) | Artificial Analysis |
| Hyperbolic hosted | $0.40 | Artificial Analysis |

**Implication:** at retail cloud prices, the GH200 96GB at $1.99–$3.19/hr is roughly *neutral* to H100 at $1.38–$3.99/hr on $/Mtok for 70B FP8, and **clearly worse than H200 at $2.09–$3.79/hr** for any workload that benefits from KV headroom. The only place GH200 wins decisively on $/Mtok is BF16 70B *without* quantization where the unified memory offload eats H100's PCIe penalty (Lambda blog: $0.02 vs $0.16 / 1K tokens, BF16, vLLM CPU offload). Once you use FP8 — which is the production default in 2026 — that advantage evaporates.

---

## 7. FP8 Reliability — the Critical Question

**Constraint from the project:** FP8 broken on GH200/Wan stack as tested. Does this generalize to H100/H200?

**Answer: No, FP8 on H100/H200 is production-grade for dense Llama-class models.**

Evidence:
1. **vLLM FP8 KV-cache blog, April 22 2026** — the team documented and *fixed* the FA3 accumulation bug ("On Hopper GPUs, the FP8 Flash Attention 3 kernel suffered from accumulation precision loss at long contexts. On a 128k needle-in-a-haystack task, FP8 accuracy dropped from 91% to 13%"). With the two-level accumulation fix, **FP8 reaches 97–98% of BF16 baseline AUC up to 128k**. Recommended Llama-70B serving command is `vllm serve … --kv-cache-dtype fp8`.
2. **Quality data:** MMLU delta FP8-vs-BF16 is <0.5% for Llama-3 70B; Llama-3.1 405B fully recovers accuracy at FP8 across Eleuther / Open LLM Leaderboard benchmarks (NVIDIA Nemotron, Spheron FP8 quantization explainer 2026).
3. **MLPerf Inference v4.0/v5.0/v5.1 winning submissions** for Llama-2 70B on H100/H200/GH200 are **all FP8** — these are accuracy-constrained submissions, not just speed runs. The accuracy gate would reject a broken FP8 path.
4. **Pre-quantized weights** — FriendliAI, RedHatAI, Neural Magic, NVIDIA NIM all ship `*-FP8`/`*-FP8-dynamic` checkpoints for Llama 3.x 8B/70B and Qwen2.5-VL-72B with documented loss numbers.

**Where FP8 *is* fragile** (and these affect *all* Hopper SKUs equally, not just GH200):
- **MoE online FP8 quantization** — vLLM issue #30830 (`patched_weight_loader` bug), TensorRT-LLM 1.3.0rc7 autotuner IndexError on Qwen3-Next-80B FP8/NVFP4, FP8 Scout illegal-memory-access (now fixed). MoE is the hard case.
- **Diffusion transformers** — vllm-omni issue #2728: catastrophic LPIPS regression on Z-Image, FLUX.1-dev, Qwen-Image with FP8 online quant; reproducible on H100 and B200. This matches the Wan video-diffusion FP8 issue the project found on GH200 — it is *not* a GH200-arch problem, it's a **DiT + online FP8 quantization** problem that crosses Hopper *and* Blackwell.
- **NIM:** NVIDIA explicitly declined to release some FP8 NIM profiles due to accuracy degradation; the BF16 GH200-480GB profile shipped instead. They ship FP8 only when validated per model.

**Practical recommendation for the project:**
- For text-only Llama 3.x / Qwen2.5 / DeepSeek-V3 inference on H100 or H200, **FP8 is safe** in vLLM ≥0.9 or TRT-LLM ≥0.13 with FA3 fixes; expect 1.4–1.8× throughput, <1% quality delta.
- For Wan / DiT video diffusion the FP8 path is broken across the entire Hopper/Blackwell stack today, **regardless of GH200 vs H100/H200**. Stay on BF16 for Wan on any of these. The choice of H100/H200/GH200 does not unlock FP8 for Wan.

---

## 8. On-Prem Availability (May 2026)

- **H100 SXM**: in steady supply via channel (Supermicro, Dell, HPE, Lenovo). Single chip $25–40K; 8-GPU HGX node $350–450K. NVIDIA confirmed production through Q2 2027, driver/security patch support to 2029. **Easiest to source.**
- **H200 SXM**: in supply; single chip $30–40K (~15–20% premium over H100). 8-GPU HGX node ~$315K (board) / $400–500K full server. January 2026 China export-control loosening drove some demand surge but supply is stable. **Recommended on-prem option for new builds.**
- **H200 NVL** (PCIe, 2-way or 4-way): broadly available through HPE / Dell / Supermicro reference architectures (NVIDIA enterprise RA published Q4 2025). Targets air-cooled enterprise racks; 4-way NVL gives you 564 GB at 1.8 TB/s bridge BW without a liquid-cooled HGX chassis.
- **GH200**: single-chip systems limited; Supermicro/HPE GH200 NVL2 nodes available but allocation-gated. **3 cloud providers only** (CoreWeave, Vultr, Lambda) — this is the supply tell. If you are buying in 2026, NVIDIA is steering enterprise customers to **GB200 NVL72** for greenfield and **H200 NVL / HGX H200** for retrofits. GH200 is in a soft sunset.

---

## 9. Inference-Quality Tradeoffs (FP8/BF16)

| Setup | Quality | Throughput | $/Mtok | Notes |
|---|---|---|---|---|
| 1×H100 80GB FP8 | Lossless (<0.5% MMLU) for Llama 70B | Medium (small KV envelope) | Mid | KV cache squeezed → small batch / short context |
| 1×H200 141GB FP8 | Lossless | High (large KV) | Low | Best $/Mtok for 70B single-GPU |
| 8×H200 FP8 | Lossless | Very high; supports 405B / DeepSeek V3 | Lowest at scale | Production default 2026 |
| 1×GH200 96GB FP8 | Lossless | High (a bit better than H100 at small ctx, *worse than* H200 at large ctx) | Mid | KV envelope ~26 GB — between H100 and H200 |
| 1×GH200 144GB FP8 | Lossless | Comparable to H200 | Mid–high (price premium) | NVL2 variant; rarely worth it over H200 |
| H100 NVL 188GB (PCIe, BF16 70B fits) | Lossless | Lower than SXM5 (no NVSwitch) | Mid | Enterprise air-cooled PCIe |
| H100/H200 BF16 (no quant) | Reference quality | Halved | 2× | For accuracy-critical eval or stiff alignment |

---

## 10. Recommendation Matrix

| If your workload is… | Pick |
|---|---|
| Llama-3.3 70B / Qwen2.5-72B FP8, latency-sensitive single stream | 1× **H200 SXM** ($2.09–$3.79/hr) — single-GPU, no offload, KV headroom |
| Llama-3.3 70B FP8, max throughput batch serving | 8× **H200 HGX** — switched NVLink, ~3,600+ tok/s aggregate |
| Llama-3.1 405B FP8 | 8× **H200 HGX** (single node, 1,128 GB) — fits cleanly |
| DeepSeek-V3 / R1 671B FP8 | 8× **H200 HGX** (single node) — avoids 2-node H100 networking overhead |
| Very long context (≥128k) with KV-offload-aware engine | **GH200 NVL2** if you have vLLM unified-memory offload paths working; otherwise 8× H200 |
| BF16 70B without quantization on 1 GPU | **GH200 96GB** (the one place it wins) |
| Cost floor, willing to tolerate marketplace risk | H100 SXM at Thunder/Vast ($1.38–$1.53/hr) FP8 70B |
| On-prem new build 2026 | **HGX H200** server or H200 NVL 4-way rack |
| Wan / DiT video diffusion | None of FP8 paths are reliable. BF16 on any Hopper, including H100 SXM. **GH200 specifically gives you nothing extra here.** |

---

## 11. Confidence & Gaps

**High confidence:**
- H100/H200/GH200 hardware specs, MLPerf v4.0/v4.1/v5.0 Llama-2 70B results, cloud pricing tiers, NVLink topology differences. Multiple primary sources (NVIDIA, MLCommons, vendor docs) corroborate.
- FP8 on dense Llama is production-grade on H100/H200 with current vLLM/TRT-LLM. vLLM blog April 2026 is the strongest evidence — they document the bug *and* the fix, and ship the fix.
- GH200 96GB has *less* HBM than H200 141GB. This is a hard architectural fact.

**Medium confidence:**
- Exact $/Mtok numbers for H100/H200 self-served — Spheron 2026 benchmark used as primary; numbers vary materially with input/output length ratios, batch size, KV cache reuse. Single-source for some figures.
- H200 single-GPU Llama-3.3 70B FP8 throughput points (680 / 1,800 / 2,500 / 3,200 tok/s at b=1/8/32/128) — Spheron-derived, no independent MLPerf-level confirmation at those exact batch sizes.
- GH200 NVL2 MLPerf v4.1 specific tok/s numbers — NVIDIA blog says "outstanding," but I could not extract a clean single number from the v4.1 article in this round; Red Hat OpenShift dual-GH200 submission is referenced but token counts not pulled.

**Gaps / open questions:**
- Reserved-pricing rates (1-year, 3-year) for H100 and H200 — none of the pricing sources surveyed publish those publicly; they require quoted enterprise contracts. Cloud pricing comparison sites explicitly excluded reserved instances.
- Direct head-to-head MLPerf submission of **GH200 NVL2 vs HGX H200** on Llama-2 70B Server scenario at the same software version — does not exist in v4.1 or v5.0 to my reading. The comparisons are usually indirect (8×H200 vs 1×GH200 vs 8×H100).
- Verified FP8 numerical reliability on Wan2.x specifically — only inferred from cross-arch DiT FP8 regressions reported in vllm-omni. A direct Wan-on-H100-FP8 reproduction would settle whether the project's "FP8 broken" finding is portable from GH200 to H100/H200 or in fact resolved on standard Hopper. **Best initial test: load Wan in FP8 on a Lambda 1×H100 instance ($1.99–$3.99/hr) and reproduce the GH200 failure mode; if it fails, the bug is in the DiT FP8 path, not GH200-specific, and the user's "no FP8 on this stack" rule is correct for all Hopper.**
- H200 NVL 4-way real-world inference benchmarks at 564 GB — NVIDIA Enterprise RA published but third-party benchmarks scarce as of May 2026.

---

## Sources (URLs)

- NVIDIA H100 product page — https://www.nvidia.com/en-us/data-center/h100/
- NVIDIA H200 product page — https://www.nvidia.com/en-us/data-center/h200/
- NVIDIA H200 NVL launch blog — https://blogs.nvidia.com/blog/hopper-h200-nvl/ (2024-11)
- NVIDIA Enterprise RA for H200 NVL — https://developer.nvidia.com/blog/deploying-nvidia-h200-nvl-at-scale-with-new-enterprise-reference-architecture/
- NVIDIA MLPerf Inference v4.1 GH200 blog — https://developer.nvidia.com/blog/nvidia-gh200-grace-hopper-superchip-delivers-outstanding-performance-in-mlperf-inference-v4-1/
- NVIDIA Blackwell MLPerf v5.0 blog — https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/
- NVIDIA TRT-LLM H200 launch — https://nvidia.github.io/TensorRT-LLM/blogs/H200launch.html
- NVIDIA TRT-LLM perf overview — https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html
- MLCommons MLPerf Inference v5.0 release — https://mlcommons.org/2025/04/mlperf-inference-v5-0-results/
- HPCwire MLPerf Inference v5.1 — https://www.hpcwire.com/2025/09/10/mlperf-inference-v5-1-results-land-with-new-benchmarks-and-record-participation/
- CoreWeave MLPerf v5.0 H200/GB200 — https://www.coreweave.com/blog/coreweave-delivers-breakthrough-ai-performance-with-nvidia-gb200-and-h200-gpus-in-mlperf-inference-v5-0
- Lambda MLPerf v5.1 — https://lambda.ai/blog/lambda-mlperf-inference-v5.1
- Lambda GH200 economics blog — https://lambda.ai/blog/putting-the-nvidia-gh200-grace-hopper-superchip-to-good-use-superior-inference-performance-and-economics
- Baseten Llama-3.3 70B on GH200 (Lambda Cloud) — https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/
- Baseten evaluating H200 for LLM inference — https://www.baseten.co/blog/evaluating-nvidia-h200-gpus-for-llm-inference/
- Baseten DeepSeek model guide — https://www.baseten.co/resources/guide/the-complete-deepseek-model-guide/
- Baseten multi-node DeepSeek-R1 — https://www.baseten.co/blog/how-multi-node-inference-works-llms-deepseek-r1/
- Perplexity Hub H100 Llama-2 70B — https://www.perplexity.ai/hub/blog/turbocharging-llama-2-70b-with-nvidia-h100
- Perplexity multi-node DeepSeek — https://www.perplexity.ai/hub/blog/lower-latency-and-higher-throughput-with-multi-node-deepseek-deployment
- vLLM FP8 KV-cache blog Apr 2026 — https://vllm-project.github.io/2026/04/22/fp8-kvcache.html  (mirror: https://vllm.ai/blog/fp8-kvcache)
- vLLM FP8 W8A8 docs — https://docs.vllm.ai/en/latest/features/quantization/fp8/
- vLLM issue #10142 (H100 FP8 fallback warning) — https://github.com/vllm-project/vllm/issues/10142
- vLLM issue #5535 (slow Mixtral 8x7B FP8 on H100) — https://github.com/vllm-project/vllm/issues/5535
- vLLM issue #30830 (MoE online FP8 accuracy) — https://github.com/vllm-project/vllm/issues/30830
- vllm-omni issue #2728 (FP8 DiT regression across Z-Image/FLUX/Qwen-Image, H100 + B200) — https://github.com/vllm-project/vllm-omni/issues/2728
- TensorRT-LLM release notes — https://nvidia.github.io/TensorRT-LLM/release-notes.html
- TensorRT-LLM issue #12230 (FP8 autotuner crash, Qwen3-Next) — https://github.com/NVIDIA/TensorRT-LLM/issues/12230
- ThunderCompute H100 pricing May 2026 — https://www.thundercompute.com/blog/nvidia-h100-pricing
- ThunderCompute H200 pricing May 2026 — https://www.thundercompute.com/blog/nvidia-h200-pricing
- GetDeploying H100 cloud pricing — https://getdeploying.com/gpus/nvidia-h100
- GetDeploying H200 cloud pricing — https://getdeploying.com/gpus/nvidia-h200
- GetDeploying GH200 cloud pricing — https://getdeploying.com/gpus/nvidia-gh200
- Spheron H100 vs H200 cost-per-token 2026 — https://www.spheron.network/blog/gpu-cost-per-token-benchmark-llm-inference-2026/
- Spheron Gaudi 3 vs H200/B200 2026 — https://www.spheron.network/blog/intel-gaudi-3-vs-nvidia-h200-b200-llm-inference-2026/
- Spheron GH200 architecture guide — https://www.spheron.network/blog/nvidia-gh200-guide/
- Spheron FP8 quantization 2026 — https://www.spheron.network/blog/fp8-quantization-inference-performance-hardware-explained/
- Spheron H100 specs 2026 — https://www.spheron.network/blog/nvidia-h100-specs/
- Spheron vLLM vs TRT-LLM vs SGLang benchmarks 2026 — https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/
- Artificial Analysis Llama-3.3 70B providers — https://artificialanalysis.ai/models/llama-3-3-instruct-70b/providers
- Yotta Labs H100 vs H200 — https://www.yottalabs.ai/post/h100-vs-h200-performance-memory-cost-and-inference-benchmarks-2026
- Introl cost per token analysis — https://introl.com/blog/cost-per-token-llm-inference-optimization
- Introl H100 vs H200 upgrade guide — https://introl.com/blog/h200-vs-h100-gpu-upgrade-decision-guide
- Introl inference unit economics — https://introl.com/blog/inference-unit-economics-true-cost-per-million-tokens-guide
- IntuitionLabs NVIDIA AI GPU pricing guide — https://intuitionlabs.ai/articles/nvidia-ai-gpu-pricing-guide
- IntuitionLabs H100 rental comparison — https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison
- GPUPerHour GH200 vs H200 NVL — https://gpuperhour.com/compare/gh200-vs-h200-nvl
- Uvation H200 Llama-2 70B — https://uvation.com/articles/why-llama2-70b-runs-better-on-h200
- Uvation H200 NVL AI inference — https://uvation.com/articles/h200-nvl-ai-inference-benchmarks
- Uvation H100 NVL — https://epoka.com/blogs/news/nvidia-h100-nvl-gpu-specs-performance-and-ai-inference-power
- TechPowerUp H200 NVL specs — https://www.techpowerup.com/gpu-specs/h200-nvl.c4254
- PNY H200 NVL datasheet — https://www.pny.com/file%20library/company/support/linecards/data-center-gpus/h200-nvl-datasheet.pdf
- Red Hat MLPerf v5.0 GH200 OpenShift — https://www.redhat.com/en/blog/mlperf-inference-v50-results
- Supermicro GH200/H200 system guidance whitepaper — https://www.supermicro.com/white_paper/white_paper_Supermicro_System_Guidance_for_GH200_and_H200.pdf
- Fluence GH200 explainer — https://www.fluence.network/blog/nvidia-gh200/
- Vultr GH200 cloud — https://www.vultr.com/products/cloud-gpu/nvidia-gh200/
- RunPod H200 — https://www.runpod.io/gpu-models/h200
- Verda DeepSeek V3 on H200 — https://verda.com/blog/deepseek-v3-llm-nvidia-h200-gpu-inference-benchmarking
- dstack H200 vs MI300X DeepSeek R1 — https://dstack.ai/blog/h200-mi300x-deepskeek-benchmark/
- GPUStack H200 DeepSeek-R1 — https://docs.gpustack.ai/2.0/performance-lab/deepseek-r1/h200/
- NADDOD H100 GH200 GB200 SuperPod — https://naddod.medium.com/from-h100-gh200-to-gb200-how-nvidia-builds-ai-supercomputers-with-superpod-a8bfa0e702fa
- Fibermall NVLink/NVSwitch evolution — https://www.fibermall.com/blog/nvidia-nvlink-and-nvswitch-evolution.htm
- Fibermall GH200 chip analysis — https://www.fibermall.com/blog/analysis-nvidia-gh200-chip-servers.htm
- LambdaLabsML/vllm-gh200-vs-h100 (repo) — https://github.com/LambdaLabsML/vllm-gh200-vs-h100
