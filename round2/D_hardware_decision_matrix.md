# D — Hardware Decision Matrix: GH200 Alternatives by Workload

**Round 2 cross-pollination agent**
**Date:** 2026-05-25
**Inputs:** Round-1 reports 10–15 (alternatives) + 20 (diffusion/video)
**Hard constraints honored:** bf16-default for diffusion (FP8 broken on this stack for Wan/DiT — see [vllm-omni #2728](https://github.com/vllm-project/vllm-omni/issues/2728), [DiffSynth-Studio #466](https://github.com/modelscope/DiffSynth-Studio/issues/466)). FP8 OK for dense Llama-class on Hopper/Blackwell. No FP8 anywhere on GH200/Wan.

---

## 0. Workload definitions

| Code | Workload | What "best fit" means |
|---|---|---|
| **A** | Dense 70B interactive (Llama 3.3, Qwen2.5-72B, batch≈1, single user) | Low TTFT (<300 ms) + high per-user TPS |
| **B** | Dense 70B batch throughput (batched serving, b≥32) | tok/s/$ at high concurrency |
| **C** | MoE DeepSeek-V3/R1 671B FP8/FP4 | Fit + decode tok/s/$ at scale |
| **D** | Long context 128k+ (KV-cache-heavy, RAG over docs) | KV headroom and prefill TPS |
| **E** | Video diffusion Wan2.2-A14B 720p (bf16-only) | Fit 27B MoE in bf16 + s/clip |
| **F** | Image diffusion Flux.2-dev (32B DiT, bf16-only) | Fit + s/image |
| **G** | RAG / cheap-$/Mtok serving (commodity 7B–70B blended) | Lowest blended $/Mtok |

---

## 1. Master Decision Matrix

Columns A–G as defined above. **BEST** = top-tier for that workload. **VIABLE** = production-deployable but not optimal. **POOR** = works but uncompetitive on cost/perf. **N/S** = not supported / broken / not buyable. Last column: best-published `$/Mtok` reference for the platform with citation tag.

| Platform | A · 70B interactive | B · 70B batch | C · DSV3 671B | D · 128k context | E · Wan2.2 video (bf16) | F · Flux.2-dev image (bf16) | G · RAG / cheap-Mtok | Best $/Mtok | Ref |
|---|---|---|---|---|---|---|---|---|---|
| **GH200 96GB** | VIABLE — 1× chip fits 70B FP8 w/ ~26GB KV; +32% over H100, slower than H200 [R10§3] | POOR — single chip; no NVSwitch all-to-all [R10§5] | N/S — 96GB << 671GB; multi-chip NVL2 still falls short of 8× node [R10§4] | BEST — only platform with 480GB LPDDR5X @ 900 GB/s NVLink-C2C for KV offload [R10§1] | VIABLE — 96GB fits Wan2.2-A14B bf16 (~70–85 GB w/ T5+VAE) on 1 chip; H100 cannot [R20§2.2] | BEST — Flux.2-dev bf16 >80GB doesn't fit on H100 80GB; GH200 96GB does w/ VAE headroom [R20§2.1] | POOR — $1.99–$6.50/hr but no FP8 throughput advantage at scale [R10§6] | ~$0.50–$1.20 [R11§4] | self-hosted single-GPU est. |
| **H200 SXM 141GB** | BEST — 1× chip fits 70B FP8 + 71GB KV; 680/1,800/2,500 tok/s @ b=1/8/32 [R10§3] | BEST — 8× HGX = 31,712 tok/s offline MLPerf v4.0; lossless FP8 [R10§3] | BEST — 8× HGX = 1,128 GB single-node fit; preferred over 2-node H100 deploys [R10§4] | VIABLE — 141 GB HBM gives biggest KV envelope on Hopper but no CPU pool [R10§4] | VIABLE — same SM90 as H100/GH200, ~10–20% faster than H100 on memory-bound DiT; bf16 fits via offload [R20§8] | VIABLE — 141 GB fits Flux.2-dev w/o offload (better than GH200) [R20§2.1] | VIABLE — $2.09–$3.79/hr on-demand floor [R10§6] | $0.47 (b=32 self-host); $0.12 hosted Turbo [R10§6] / [DeepInfra-verified] |
| **H100 SXM 80GB** | VIABLE — 70B FP8 fits but ~10 GB KV → small batch only; 250–300 tok/s single stream [R10§3] | VIABLE — 8× HGX = 22,290 tok/s MLPerf v4.0; needs 2-node for 405B [R10§3] | POOR — 8× = 640 GB; DeepSeek-V3 671B *doesn't fit* → forced 2-node networking [R10§4] | POOR — KV starvation at 128k unless quantized; KV-cache FP8 fixed in vLLM Apr 2026 helps [R10§7] | VIABLE — 80GB tight for Wan2.2-A14B bf16; ~250 s baseline 8×H100 → 60 s w/ Sage+TeaCache [R20§2.2] | POOR — Flux.2-dev bf16 *doesn't fit* on 80GB w/o offload [R20§2.1] | VIABLE — floor $1.38–$1.99/hr (Thunder/RunPod) [R10§6] | ~$0.71 self-host FP8; $1.15 batched [R10§6] |
| **B200 SXM 192GB** | BEST — NVFP4 ~10,000 tok/s/GPU @ 50 TPS/user; 3.7× Hopper at iso-latency [R11§3] | BEST — InferenceMAX: ~$0.17/Mtok vs H200 $0.50 on Llama-3.3-70B [verified InferenceMAX] | VIABLE — single-node 8× B200 (1.5 TB) fits DSV3 FP4; better $/tok than HGX H200 [R11§3] | VIABLE — 192GB/GPU = best Blackwell single-node KV; FA4 fixed for most head_dim [R11§5] | VIABLE bf16 — same DiT FP8 regression bug present (cross-arch [#2728]); bf16 path works [R20§1] | VIABLE — 192GB fits Flux.2-dev bf16 easily; same FP8/NVFP4 DiT caveat [R20§1] | BEST — $4.29–$5.89/hr on-demand; ~$0.17/Mtok 70B [R11§4] | $0.17 (Llama-3.3-70B NVFP4) [InferenceMAX-verified] |
| **GB200 NVL72** | VIABLE — fast but rack-scale is overkill for batch=1; ~$10.50–$27/GPU/hr [R11§2] | BEST — 60,000 tok/s per server (Llama-2-70B MLPerf) [R11§3] | BEST — 13.4 TB HBM3e unified; DSV3 native WideEP w/ SGLang [R11§3] | BEST — unified HBM domain across 72 GPUs eliminates KV partitioning [R11§6] | VIABLE bf16 — same DiT FP8 bug across Blackwell [R20§1]; bf16 fits trivially | VIABLE — overkill for 32B image gen | BEST — **$0.10/Mtok DeepSeek-R1** (15× H200's $1.56) [InferenceMAX-verified] | $0.10 (DeepSeek-R1) [InferenceMAX-verified] |
| **GB300 NVL72** | VIABLE — Blackwell Ultra; ~20 TB HBM; MLPerf v5.1 DSR1 +45% vs GB200 [R11§3] | BEST — 5,842 tok/s/GPU offline DSR1 MLPerf v5.1; Llama 8B @ 18,370 tok/s/GPU [R11§3] | BEST — +45% throughput vs GB200 NVL72 on DSR1 [R11§3] | BEST — biggest unified HBM pool in market today [R11§1] | VIABLE bf16 — same DiT caveat | VIABLE — fits easily | BEST — newest, best $/Mtok at scale; CoreWeave/Lambda/MSFT/Oracle [R11§2] | Inferred better than GB200 $0.10 by ~30%; no clean number published |
| **MI300X 192GB** | POOR — FP8 *slower than BF16* on MoE (vLLM #31475 OPEN); torch._scaled_mm 12× slower than H100 [R12§2.4] | VIABLE — MI325X-vLLM beats H200-vLLM on $/Mtok at low-interactivity; loses at high [R12§5] | VIABLE — 8× MI300X (1.5 TB) fits DSV3 FP8 w/ EP; Moreh got 21k tok/s [R12§3] | BEST — 192 GB single-chip = biggest single-GPU KV pool (2× H200) [R12§6] | VIABLE bf16 — no FP8 caveat needed; ROCm AITER backend production [R12§2.1] | VIABLE bf16 — 192 GB fits Flux.2-dev easily; diffusers AMD path less mature than CUDA | VIABLE — Vultr 24-mo $1.85/hr is structural $/Mtok win at reserved [R12§4] | ~$0.20–$0.40 reserved (Vultr/DigitalOcean) [R12§5] |
| **MI325X 256GB** | POOR — same FP8 stack issues as MI300X; no TRT-LLM equivalent [R12§2.3] | VIABLE — beats H200-vLLM; loses to H200-TRT-LLM at high interactivity [R12§5] | VIABLE — 256GB × 8 = 2 TB; comfortable DSV3 FP8 fit [R12§6] | BEST — 256 GB single-chip; KV champion among CDNA3 [R12§1] | VIABLE bf16 — same ROCm caveats | VIABLE bf16 — large memory; AMD diffusers path is the friction point | VIABLE — $2.00/hr 36-mo (Vultr); 3 providers only [R12§4] | ~$0.25–$0.50 [R12§5] |
| **MI355X 288GB** | VIABLE — FP4 works for select shapes; ~67% of B200 on Kimi K2.5 [R12§2.4] | VIABLE — MI355X-vLLM beats B200-vLLM <225 tok/s/user on gpt-oss-120B FP4; loses above [R12§5] | VIABLE — DSR1 MXFP4 via Quark, but B200/GB200 still beat on $/tok at high interactivity [R12§5] | BEST — 288 GB HBM3e + 8 TB/s; biggest single-chip in market [R12§1] | VIABLE bf16 — CDNA4 native FP4/FP6, but diffusion path is unproven; stay bf16 | VIABLE bf16 — 288 GB ample for Flux.2; AMD diffusers maturity is the gating factor | VIABLE — $2.29/hr 36-mo Vultr; on-demand $5–8.60 [R12§4] | ~$0.30 reserved; ~$0.60 OD [R12§5] |
| **TPU v6e Trillium** | POOR — 32 GB HBM/chip → 70B needs slice; not single-chip single-stream [R15§2] | VIABLE — v6e-8 = 256 GB slice; vLLM-TPU 2.1× the Feb-2025 prototype on Llama 70B [R15§4.2] | POOR — fits only with large slice (16+ chips); ICI bandwidth lower than NVLink5 [R15§2] | VIABLE — slice scaling on JAX/XLA; reasonable KV with slicing | N/S — no diffusers TPU first-class path for Wan; xDiT/SGLang-Diffusion CUDA-only [R20§3] | N/S — no production Flux.2 path on TPU; PT/XLA possible but unsupported [R20§3] | BEST — $0.39–$0.55/chip-hr 3-yr CUD; $2.70 on-demand [R15§4.3] | Inferred ~$0.10–$0.30 at 3-yr CUD; no clean public number |
| **TPU v7 Ironwood** | BEST — 192 GiB HBM3e/chip = 70B BF16 fits on **1 chip** [R15§2] | VIABLE — 4,614 FP8 TFLOPS/chip; no public Llama 70B TPS benchmark as of May 2026 [R15§4.4] | VIABLE — slice can fit, but reservation-only access limits ad-hoc use [R15§4.1] | BEST — 192 GiB/chip × pod scale; matches Blackwell on KV per chip | N/S — JAX-only per v7 docs; no diffusers path | N/S — same | BEST — Anthropic-scale economics; no public $/Mtok | unpublished; estimated $0.05–$0.15 at hyperscale [R15§4.3] |
| **Cerebras CS-3 / API** | BEST — 1,927.7 TPS on gpt-oss-120B (AA-verified May 2026); ~2,100 on 70B [verified] | POOR — TPS/user is king but high TTFT (1.5–3 s on big models) hurts batched cost story [R14§ladder] | VIABLE — 405B record 969 TPS; SambaNova fits 405B too, but Cerebras has Code Pro/Max sold out [R14§AA-table] | VIABLE — SRAM-resident architecture; long context not the strong suit due to fixed wafer | N/S — vision/diffusion not on Cerebras API | N/S — same | VIABLE — $0.39/Mtok blended gpt-oss-120B (AA); $0.60 70B | $0.39 (gpt-oss-120B) [AA-verified]; $0.60 (70B blended) [R14] |
| **Groq API / LPX** | BEST — **TTFT 180 ms** P50 on Llama 3.3 70B (verified May 2026 measurement); 322.9 TPS [verified AA] | VIABLE — strong per-user TPS but token cost mid-pack; $0.59/$0.79 in/out 70B [R14§prices] | POOR — Groq mostly serves up to ~70B / Maverick scale; not DSV3 671B native | POOR — small per-chip SRAM = limited KV; not the 128k+ play | N/S — text-only LPU | N/S — text-only | BEST — $0.59/$0.79 in/out 70B; sub-200 ms TTFT [R14§prices] | $0.59 in / $0.79 out (Llama 3.3 70B) [Groq pricing-verified] |
| **SambaNova SN40L / API** | VIABLE — 794 TPS on Maverick (AA); TTFT 4+ s hurts interactive feel [R14] | VIABLE — record 200 TPS on 405B; agentic-pitched throughput [R14] | BEST — only API hitting 132 TPS on Llama 3.1 405B native 16-bit [R14] | VIABLE — RDU + DDR-tiered memory supports very long context (SN50 → 10M ctx) [R14§developments] | N/S — text/agentic only | N/S — text only | VIABLE — $0.60/$1.20 70B; $5/$10 405B [R14] | $0.60 in (70B) [R14§prices] |
| **d-Matrix Corsair** | UNVERIFIED — vendor claim 1 ms/token (Llama 3 8B); 2 ms/token (70B). **Zero AA / third-party benchmark** [R14§gaps] — flag as unproven | UNVERIFIED — 30,000 TPS/rack Llama 70B claim; no independent rep | N/S — no DSV3 path published | UNVERIFIED — 256 GB capacity-mem story plausible but unvalidated | N/S — text/LLM only | N/S — text only | UNVERIFIED — vendor 3× perf/TCO claim; no public $/Mtok | UNKNOWN — no public $/Mtok; vendor-claim 3× TCO [R14§unverified] |
| **Tenstorrent Galaxy Blackhole** | VIABLE — 255 TPS DSR1 671B (EE Times independent); vendor "Blitz Mode" 350+ [verified] | VIABLE — 23 PFLOPS BFP8/server; software realizes ~40–60% of theoretical on LLMs [R14§unverified] | VIABLE — 1 TB DRAM @ 16 TB/s fits DSR1 671B on 1 Galaxy ($110k self-host) [R14§ladder] | VIABLE — large DRAM pool good for KV at modest BW | N/S — no diffusion path | N/S | VIABLE — vendor claim ~$6/Mtok DSR1 vs GB300 ~$30 [Wccftech-verified] | ~$6/Mtok (DSR1, vendor) [Wccftech-verified, not AA] |
| **Apple M3 Ultra 512GB** | BEST (single-user only) — Llama 70B 4-bit MLX ~15–20 tok/s; <200 W; $0.20/Mtok electricity [R13§4] | POOR — no batching at GH200-class throughput; vllm-mlx claims 4.3× at 16 concurrent but unverified [R13§6] | VIABLE — only single-box DSV3 671B 4-bit (>20 tok/s gen at small ctx); 14-min prefill at 8k kills serving [R13§3] | POOR — prefill bottleneck (819 GB/s, ~26 TFLOPS FP16); 24× slower memory-refresh-rate than H100 [R13§3] | POOR — no Wan2.2 path with quality; MLX FLUX Schnell 30–50s/image, ~10× H100 [R13§5] | POOR — no Flux.2 distill; bf16 inference 10× slower than GH200 | POOR — single-stream economics only; no batching ceiling | $0.20/Mtok electricity-only single user [R13§4] |

---

## 2. Confidence-flagged claim audit

| Claim | Round-1 source | Verified | Confidence | Notes |
|---|---|---|---|---|
| Cerebras 1,927.7 TPS gpt-oss-120B | [R14§AA-table] | YES — AA live page accessed 2026-05-25 | HIGH | Exact number reproduced on artificialanalysis.ai 72-h window |
| Groq Llama-3.3-70B TTFT ~120 ms P50 | [R14§Groq-row] | PARTIAL — direct benchmark May 17 2026 = **180 ms** P50; AA reports 0.92 s blended (includes routing) | MEDIUM | The ≤200 ms class is real; 120 ms specific number not reproducible. **Use 180 ms** as the citable figure. |
| Cerebras Maverick 2,522 TPS vs Blackwell 1,038 | [R14§AA-table] | YES — Cerebras press release + AA, May 28 2025 (still standing) | HIGH | Multi-source corroboration |
| d-Matrix Corsair 1–2 ms/token Llama 70B | [R14§gaps] | NO — vendor-claim only. **No AA, no third-party, no MLPerf** | LOW | **DO NOT use in customer-facing decision until validated** |
| Tenstorrent Galaxy 350 TPS DSR1 671B | [R14§ladder] | PARTIAL — vendor "Blitz Mode"; **EE Times independent test = 255 TPS** for short prompts | MEDIUM | Use 255 TPS as defensible third-party number; 350 only under specific config |
| GB200 NVL72 DSR1 $0.10/Mtok vs H200 $1.56 | [R11§3] | YES — InferenceMAX v1 / NVIDIA blog + multiple secondary | HIGH | The 15× delta is the headline market datapoint of 2026 |
| B200 Llama-3.3-70B ~$0.17/Mtok NVFP4 | [R11§3] | YES — SemiAnalysis InferenceMAX / NVIDIA blog | HIGH | Apples-to-apples with H200 $0.50 same harness |
| DeepInfra Llama-3.3-70B Turbo FP8 $0.12 blended | [R10§6] | YES — DeepInfra pricing page + AA, May 2026 | HIGH | $0.10 in / $0.32 out → $0.12 blended at common 4:1 ratio |
| M3 Ultra 512GB DeepSeek-V3 671B >20 tok/s | [R13§2] | YES — Awni Hannun + VentureBeat + Slashdot, March 2025 | HIGH | But only at small context; 8k prefill = 14 min (llama.cpp) |
| TPU v7 Ironwood Llama-70B TPS | [R15§4.4] | NO — **no public third-party benchmark exists** | LOW | Google publishes only relative vs prior TPU; treat all comparisons as unverified |
| FP8 broken on Wan/GH200 stack | user constraint + [R20§1] | YES — [vllm-omni #2728], [DiffSynth-Studio #466], cross-arch H100 + B200 reproduction | HIGH | Generalizes across all Hopper + Blackwell for DiT — *not* GH200-specific |

---

## 3. Top-3 ranked recommendations per workload

### A — Dense 70B interactive (Llama 3.3, batch≈1, latency-sensitive)
1. **Groq API** — TTFT 180 ms P50 on Llama 3.3 70B is the lowest in market; sustained 322.9 TPS; $0.59/$0.79 in/out [verified AA + Groq blog]. The pragmatic answer for any chat-style GH200 replacement.
2. **H200 SXM 141GB** — self-host single chip fits 70B FP8 + 71 GB KV comfortably; lossless quality; 680 tok/s b=1; $2.09–$3.79/hr on-demand from 6+ providers [R10§3, R10§6]. Best self-host single-stream.
3. **B200 SXM 192GB** — if you can tolerate NVFP4 (validate per-model); 3.7× Hopper at iso-latency on InferenceMAX; broad cloud availability [R11§3]. Use TRT-LLM 1.0 or vLLM v0.20.x.

### B — Dense 70B batch throughput
1. **B200 (8× SXM node)** — ~$0.17/Mtok on Llama-3.3-70B NVFP4 vs H200 $0.50 (InferenceMAX-verified). 3× $/tok improvement. Mature TRT-LLM + vLLM + SGLang [R11§3].
2. **GB200 NVL72** — wins at top-end concurrency where NVLink5 + WideEP unlock throughput; rack-scale only [R11§6]. Use only if you saturate 8× B200.
3. **8× H200 HGX** — lossless FP8 reference path; 31,712 tok/s MLPerf v4.0; production default for Hopper-only buyers [R10§3].

### C — MoE DeepSeek-V3 / R1 671B
1. **GB200 NVL72** — **$0.10/Mtok DeepSeek-R1** at 75 TPS/user (15× cheaper than H200). 13.4 TB unified HBM = native WideEP. The decisive win [R11§3, InferenceMAX-verified].
2. **GB300 NVL72** — +45% over GB200 on DSR1 MLPerf v5.1; same fabric, more HBM [R11§3]. Pick if you can get allocation.
3. **8× H200 HGX** — single-node fit (1,128 GB); avoids 2-node H100 networking; the cheapest non-Blackwell option [R10§4]. AMD MI325X is a viable 4th option for memory-bound deployments at reserved Vultr pricing.

### D — Long context (128k+)
1. **GH200 NVL2 (+ vLLM KV offload)** — only platform with 480 GB LPDDR5X CPU pool at 900 GB/s NVLink-C2C. *If* your engine supports it (vLLM unified-memory offload patches, not SGLang/TRT-LLM yet) this is the cheapest 128k+ option [R10§4]. This is GH200's one structural moat.
2. **GB200/GB300 NVL72** — unified 13.4–20 TB HBM across 72 GPUs eliminates KV partitioning entirely; the expensive answer [R11§6].
3. **MI355X 288GB** — biggest single-chip HBM in market; 8 TB/s. Best on-prem long-context per chip; bf16 mandatory for serious context [R12§1].

### E — Video diffusion Wan2.2-A14B 720p (bf16 only, per user constraint)
1. **GH200 96GB** — 96 GB fits Wan2.2 14B-MoE bf16 with both experts + T5 + VAE (~70–85 GB); H100 80GB *cannot* without offload. Optional Grace offload via NVLink-C2C is "free" for the idle expert [R20§2.2]. **This is GH200's single best workload fit.**
2. **H200 SXM 141GB** — bigger HBM, ~10–20% faster on memory-bound DiT than GH200; same SM90, same diffusers path; better single-GPU prefill margin [R20§8].
3. **8× H100 HGX (sequence-parallel)** — Voltage Park optimized 187 → 60 s for 720p 81-frame w/ Sage+TeaCache+batched fwd [R20§2.2]. Only worth it if you have the cluster already.

### F — Image diffusion Flux.2-dev (32B DiT, bf16 only)
1. **GH200 96GB** — Flux.2-dev bf16 needs >80 GB; H100 80GB *cannot* fit without offload; GH200 96GB does w/ VAE headroom. ~10–15 s/image extrapolated [R20§2.1].
2. **H200 SXM 141GB** — 141 GB removes any fit concern; same compute envelope; minor BW edge over GH200.
3. **B200 / MI355X** — 192–288 GB easily fits Flux.2-dev bf16; both have DiT-bf16 paths; AMD diffusers stack is rougher; NVFP4 *not* recommended for DiT per [#2728].

### G — RAG / cheap-$/Mtok serving (commodity 7B–70B)
1. **GB200 NVL72 hosted via CoreWeave/Azure** — $0.10/Mtok DSR1 is the absolute floor; for Llama-3.3-70B drops to $0.17 [InferenceMAX-verified].
2. **DeepInfra Turbo FP8 (Llama 3.3 70B)** — **$0.12 blended** end-customer pricing (verified DeepInfra + AA, May 2026). If you're a token consumer, not an operator, this beats every self-host option below GB200 scale.
3. **TPU v6e Trillium 3-yr CUD** — $0.39–$0.55/chip-hr with vLLM-TPU; lowest hyperscaler $/Mtok at commitment scale [R15§4.3]. Pair w/ MaxText for production.

---

## 4. The "what should we actually do?" summary

| If your workload is dominated by… | GH200 verdict | Pick instead |
|---|---|---|
| Wan2.2 video diffusion (bf16) | **KEEP** — GH200 96 GB is the cheapest single-GPU bf16 fit; H100 fails; H200/B200 work but cost more | GH200 stays the right tool here |
| Flux.2-dev image gen (bf16) | **KEEP** — same memory-fit story as Wan2.2 | GH200 stays |
| Llama 3.3 70B interactive | **DROP** GH200 → too pricey for what it delivers | **Groq API** ($0.59/$0.79, 180ms) or **H200** ($2.09/hr) |
| DeepSeek-V3 / R1 | **DROP** GH200 → 96 GB doesn't fit native | **GB200 NVL72** ($0.10/Mtok) |
| 128k+ context with KV offload | **KEEP** GH200 NVL2 — only real moat | GH200 NVL2 if vLLM offload supported |
| Cheap RAG tokens | **DROP** GH200 — economics are mid-pack | **DeepInfra Turbo** ($0.12 blended) |

---

## 5. Confidence per cell (HIGH / MED / LOW)

Same row order; HIGH = primary source + recent + reproducible. MED = single secondary or extrapolated. LOW = vendor-only / no independent verification.

| Platform | A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|---|
| GH200 96GB | MED (Baseten+Lambda blogs; no MLPerf single-chip 70B FP8 number) | HIGH (NVLink topology fact) | HIGH (arithmetic of 96GB << 671GB) | MED (CPU pool real; engine support partial) | HIGH (Wan2.2 bf16 fit math is concrete) | HIGH (Flux.2 bf16 fit math is concrete) | HIGH (pricing verified across 3 providers) |
| H200 SXM 141GB | HIGH (MLPerf v4.0 + Spheron + Baseten) | HIGH (MLPerf v4.0) | HIGH (1,128 GB single-node, Verda + dstack benchmarks) | MED (KV envelope; no native CPU pool) | MED (extrapolated from H100; same SM90) | HIGH (141 GB fits) | HIGH (multi-source) |
| H100 SXM 80GB | HIGH (TRT-LLM blog + Perplexity Hub) | HIGH (MLPerf v4.0) | HIGH (640 GB single-node fact) | LOW (depends on quant choices) | HIGH (Voltage Park measured) | HIGH (does not fit fact) | HIGH (pricing) |
| B200 SXM 192GB | HIGH (InferenceMAX) | HIGH (InferenceMAX-verified) | MED (DSV3 fits 8× but B200 NVFP4-MoE kernels still maturing) | MED (192 GB/GPU; FA4 caveats) | MED bf16 OK / NVFP4 BROKEN for DiT (#2728) | MED (same caveat) | HIGH (InferenceMAX) |
| GB200 NVL72 | MED (overkill for batch=1) | HIGH (MLPerf + InferenceMAX) | HIGH ($0.10/Mtok verified) | HIGH (unified HBM fact) | MED bf16 only | MED | HIGH (verified) |
| GB300 NVL72 | MED (limited public benchmarks at batch=1) | HIGH (MLPerf v5.1) | HIGH (MLPerf v5.1 +45%) | HIGH | MED | MED | MED (newer; less price discovery) |
| MI300X | MED (FP8/MoE issues open) | MED (SemiAnalysis-verified at low interactivity) | MED (Moreh 21k tok/s) | HIGH (192 GB fact) | MED bf16 (less validation than CUDA) | MED bf16 | HIGH (Vultr reserved) |
| MI325X | MED (same FP8 caveats) | MED | MED (2 TB single-node fits) | HIGH | MED | MED | MED (3 providers only) |
| MI355X | MED (FP4 immature; ~67% of B200 on Kimi) | MED | MED | HIGH (288 GB fact) | LOW (no Wan benchmarks on MI355X) | LOW | MED |
| TPU v6e | HIGH (small HBM/chip fact) | MED (vLLM-TPU blog: 2.1× prototype) | MED (slice requirement) | MED | HIGH (no first-class diffusers path) | HIGH | HIGH (pricing verified) |
| TPU v7 Ironwood | MED (HBM fit fact; no public TPS) | LOW (zero public Llama 70B numbers) | MED | HIGH (specs) | HIGH (no path) | HIGH (no path) | LOW (no published $/Mtok) |
| Cerebras CS-3 / API | HIGH (AA-verified) | LOW (TTFT hurts batched economics) | MED (405B 969 TPS record verified) | LOW (SRAM architecture) | HIGH (no diffusion) | HIGH | HIGH ($0.39 Mtok AA-verified) |
| Groq API / LPX | HIGH (Groq blog + AA: 180 ms TTFT May 17 2026) | MED (per-user TPS good; batched $/tok mid-pack) | HIGH (not in their model menu) | HIGH (small SRAM) | HIGH (no diffusion) | HIGH | HIGH (pricing verified) |
| SambaNova | MED (AA verified on Maverick) | MED | HIGH (405B record) | MED (SN50 10M ctx claim unverified) | HIGH | HIGH | MED |
| d-Matrix Corsair | **LOW** (vendor-only 1–2 ms/token) | **LOW** (vendor-only 30k tok/s/rack) | LOW (no DSV3 path published) | LOW | HIGH (no path) | HIGH | LOW (no public $/Mtok) |
| Tenstorrent Galaxy | MED (EE Times 255 TPS independent; Tenstorrent 350 Blitz Mode) | MED (vendor 5× TCO vs GB300; software 40–60% theoretical) | MED (255 TPS DSR1 671B independent) | MED (1 TB DRAM @ 16 TB/s) | HIGH (no path) | HIGH | MED (vendor $6/Mtok DSR1) |
| M3 Ultra 512GB | HIGH single-user (multiple primary sources) | HIGH (no batching capacity) | MED (works at small ctx; 14-min prefill at 8k is documented) | HIGH (prefill is the wall) | HIGH (no production Wan path) | HIGH | HIGH (electricity math) |

---

## Sources (key citations not already inline)

- Artificial Analysis gpt-oss-120B providers — https://artificialanalysis.ai/models/gpt-oss-120b/providers (accessed 2026-05-25)
- Artificial Analysis Llama-3.3-70B providers — https://artificialanalysis.ai/models/llama-3-3-instruct-70b/providers (accessed 2026-05-25)
- Groq Llama-3.3-70B speed benchmark — https://groq.com/blog/new-ai-inference-speed-benchmark-for-llama-3-3-70b-powered-by-groq (180 ms TTFT, 209 samples, May 17 2026)
- NVIDIA Blackwell on InferenceMAX — https://developer.nvidia.com/blog/nvidia-blackwell-leads-on-new-semianalysis-inferencemax-benchmarks/
- vLLM × InferenceMAX — https://blog.vllm.ai/2025/10/09/blackwell-inferencemax.html
- Tenstorrent Galaxy independent test (EE Times 255 TPS) — https://www.eetimes.com/tenstorrent-unveils-next-gen-servers-for-fast-tokens-no-disaggregation-needed/
- Tenstorrent vendor Blitz Mode 350 TPS — https://wccftech.com/tenstorrent-vows-to-crush-everyone-galaxy-blackhole-hits-350-tokens-on-deepseek-r1-undercut-nvidia-gb300-ai-tco/
- DeepInfra Llama-3.3-70B Turbo pricing — https://deepinfra.com/meta-llama/Llama-3.3-70B-Instruct-Turbo/api
- d-Matrix Corsair perf claims (vendor) — https://www.d-matrix.ai/announcements/d-matrix-unveils-corsair-the-worlds-most-efficient-ai-computing-platform-for-inference-in-datacenters/
- vllm-omni DiT FP8 regression — https://github.com/vllm-project/vllm-omni/issues/2728
- DiffSynth-Studio Wan 2.1 FP8 color regression — https://github.com/modelscope/DiffSynth-Studio/issues/466
- Cerebras Maverick benchmark — https://www.cerebras.ai/press-release/maverick
- M3 Ultra DeepSeek-V3 20 tok/s — https://venturebeat.com/ai/deepseek-v3-now-runs-at-20-tokens-per-second-on-mac-studio-and-thats-a-nightmare-for-openai

Round-1 source files referenced as [R10], [R11], [R12], [R13], [R14], [R15], [R20]:
- /home/ubuntu/research/gh200_inference/round1/10_h100_h200_alternatives.md
- /home/ubuntu/research/gh200_inference/round1/11_blackwell.md
- /home/ubuntu/research/gh200_inference/round1/12_amd_mi300_etc.md
- /home/ubuntu/research/gh200_inference/round1/13_apple_silicon.md
- /home/ubuntu/research/gh200_inference/round1/14_specialty_chips.md
- /home/ubuntu/research/gh200_inference/round1/15_gaudi_tpu.md
- /home/ubuntu/research/gh200_inference/round1/20_diffusion_video.md

---

## End — Confidence per cell

All matrix cells carry an explicit HIGH/MED/LOW grade in §5. Decision-grade cells (the bold winners in §3) are all HIGH or MED-HIGH. **LOW-confidence cells** (d-Matrix everywhere except "no diffusion path"; TPU v7 Llama 70B TPS; M3 Ultra batched throughput; Tenstorrent single-user TPS for non-DSR1 models; AMD diffusers maturity for Wan/Flux) are flagged as such and **should not drive procurement decisions without direct measurement**.
