# Specialty Inference Accelerators — Round 1, Agent 14
**Date:** 2026-05-25
**Domain:** Non-GPU inference silicon (Groq, Cerebras, SambaNova, Tenstorrent, Etched, Graphcore, d-Matrix, Rebellions, Furiosa)
**Question framing:** Real availability today (self-host or API), per-token cost, low-batch interactive latency, independent verification.

---

## TL;DR

- **Production-grade, buy-tokens-today:** Groq, Cerebras, SambaNova are the only three with mature APIs, independent Artificial Analysis benchmarks, and public pricing. All three beat Blackwell on Llama-class throughput per user.
- **Production hardware, limited models / earlier stack:** Tenstorrent (Galaxy Blackhole GA April 2026), Furiosa (RNGD mass-prod since H2 2025), d-Matrix (Corsair shipping to early-access), Rebellions (Rebel100 in deployment via SKT/KT in Korea).
- **Vapor / unshipped:** Etched Sohu (still no customer ships ~23 months post-announcement); Graphcore IPU absorbed into SoftBank's Stargate effort, no merchant API.
- **For low-batch, single-user interactive latency** (the case you actually care about for a chat-style GH200 replacement), **Cerebras and Groq are still the latency kings** — Cerebras hits 1,900–3,000 tok/s/user on gpt-oss-120B and Llama 4 Maverick; Groq holds P50 TTFT ~120 ms on Llama 3.3 70B with 394 TPS sustained.
- **Watch:** Nvidia announced **Groq 3 LPX** at GTC 2026 (March 16, 2026) as a 128-LPU rack co-deployed with Vera Rubin — Groq's IP is now also inside Nvidia's product line after the $20B Dec 2025 license deal.

---

## Vendor-by-vendor table

| Vendor | Product | Status (May 2026) | Self-host? | API? | Headline perf (independent where noted) | Price (API) | TTFT / latency | Source dates |
|---|---|---|---|---|---|---|---|---|
| **Groq** | LPU / GroqCloud / GroqRack | **GA** — API mature since 2024, GroqRack for enterprise/sovereign. Nvidia licensed IP for $20B (Dec 2025); founder moved to Nvidia. Independent co. continues under new CEO Simon Edwards. | Yes (GroqRack, 64–576+ LPUs/rack, sales-only) | Yes | Llama 3.1 8B 840 TPS; Llama 3.3 70B 394 TPS; gpt-oss-120B 500 TPS; Llama 4 Maverick 549 TPS (Artificial Analysis) | $0.05 in/$0.08 out (8B); $0.59/$0.79 (70B); $0.15/$0.60 (gpt-oss-120B); 50% off w/ batch+cache | **P50 TTFT ~120 ms, P95 280 ms on 70B**; sub-200 ms widely cited | groq.com/pricing (May 2026); AA leaderboard April 2026 |
| **Cerebras** | WSE-3 / CS-3 / Cerebras Inference + Cerebras Code | **GA + IPO'd May 14 2026 (CBRS, $95B mcap)**. Sold out into 2027 per CFO. | Yes (CS-3, ~$2–3M/chip est., never officially priced) | Yes | **gpt-oss-120B 1,927 TPS @ 1.53s TTFT (AA verified, May 2026)**; self-claim 3,000 TPS. Llama 4 Maverick 2,522 TPS (AA, beat Blackwell 1,038). Llama 3.3 70B ~2,100 TPS. Llama 3.1 405B record 969 TPS. | $0.39/M (gpt-oss-120B blended, AA); ~$0.10/M (8B), $0.60/M (70B); $50/mo Code Pro & $200/mo Max **both sold out** | TTFT 1.5–3 s on big models (longer prefill, but per-user output is fastest in the industry) | cerebras.ai, AA April 2026, CNBC May 14/15 2026 |
| **SambaNova** | SN40L + **SN50** (Feb 24 2026) | **GA on SN40L**, SN50 ships H2 2026. Intel acquisition talks **collapsed Q1 2026**; raised $350M Series E w/ Intel as 9% strategic investor. | Yes (SambaRack, 16-chip system) | Yes (SambaNova Cloud) | gpt-oss-120B **697 TPS @ 4.45s TTFT** (AA); Llama 4 Maverick 794 TPS (AA, beat Groq); Llama 3.1 405B 132 TPS at native 16-bit; record 200 TPS on 405B. SN50 self-claim 5× Blackwell B200 / 3× throughput for agentic; 10T param / 10M context support. | $0.10/$0.20 (8B); $0.60/$1.20 (70B); $5/$10 (405B) | Higher TTFT than Cerebras/Groq (4+ s on big MoE); good for high-throughput agentic | sambanova.ai Feb 24 2026; AA |
| **Tenstorrent** | Wormhole (mature) / **Blackhole p100/p150/Galaxy GA April 28 2026** / Quasar (roadmap) | **GA on Galaxy Blackhole**. Open-source SW stack (TT-Metal). Quasar (4nm chiplet, 32 Tensix NEO cores) — 2025 tapeout, 2026 product. | **Yes — sells silicon directly**: p100 $999, p150 $1,399, QuietBox workstation pre-order, **Galaxy server $110K**, supercluster 4× $440K | Via partners (Cirrascale, Equinix, Japan ai&) | Galaxy: 23 PFLOPS BFP8/system, 32 BH chips, 1 TB DRAM @ 16 TB/s, 56× 800G Ethernet. **DeepSeek R1-0528 671B at 308 TPS/user, claimed 350 w/ Blitz Mode, 500 on roadmap.** Self-claim 5× lower TCO vs Nvidia GB300. Independent: 40–60% TFLOPS realization on real LLMs (vs 60–80% Nvidia). | N/A (silicon only) | N/A measured; favorable for self-hosters | tenstorrent.com, theregister.com April 28 2026, hpcwire May 1 2026 |
| **Furiosa AI** | **RNGD (mass production since Sept 2025)** | **GA**. 180W TDP, 512 TFLOPS FP8, 48 GB HBM3, 1.5 TB/s. NXT RNGD Server (4U, 8 cards). LG AI Research production deployment on EXAONE 3.5 32B. Bloomberg Jan 2026: $500M pre-IPO round in motion. **Rejected Meta acquisition offer.** | **Yes — sells PCIe cards & 4U servers** (no public price; contact sales) | No public cloud API | LG EXAONE 3.5 32B: 60 TPS @ 4K context / 50 TPS @ 32K (batch=1, 4 cards). 2.25× perf/watt vs prior GPU. Llama 3.3 70B: 1,500 TPS at b64, 1,100 TPS at b32 (SDK 2025.3, Feb 14 2026). 2-card LLaMA 3.1-70B viable. | N/A | Low-power 150–180W envelope is the standout, not TTFT | furiosa.ai, koreaherald, networkworld |
| **d-Matrix** | **Corsair (sampling/early-access, broad GA Q2 2025)** / **Raptor 2026** | Corsair shipping to select customers via Supermicro, GigaIO SuperNODE, Gimlet Cloud (H2 2026). $275M Series C @ $2B (Nov 2025). Raptor (4nm, 3DIMC stacked DRAM, 150 TB/s on-chip BW) targeting 2026 commercial. | Yes (Corsair PCIe + JetStream NICs + Aviator SW; pricing private) | No first-party API; via partners | **Self-claim:** Llama3 8B 60,000 TPS @ 1 ms/token (single server); Llama3 70B 30,000 TPS @ 2 ms/token (single rack). 2,400 TFLOPS INT8/card, 2 GB perf mem + 256 GB capacity mem. 3× perf/TCO, 3× energy. **No independent AA benchmark.** | N/A | **Sub-2 ms/token claimed** is the lowest in the field if real — purpose-built for interactive batch=1 | d-matrix.ai, servethehome (Hot Chips 2025), siliconangle Nov 2025 |
| **Rebellions** (KR) | ATOM (gen-1, production) / **REBEL-Quad → Rebel100 (mass-prod 2026)** | **GA progressing.** ISSCC 2026: first quad-chiplet AI accel w/ UCIe-Advanced. Pre-IPO $400M round @ $2.34B (Mar 2026). SKT + Arm collab Apr 2026 for sovereign-AI deployment; KT investor. | Yes via RebelCard, RebelPOD, RebelRack | Korea-only / sovereign cloud (SKT) | Rebel100: 4× 320mm² NPU dies, 144 GB HBM3E, 4.8 TB/s, 512 MB SRAM. RebelRack 16 PFLOPS FP8 / 8 PFLOPS FP16. Self-claim: parity w/ Nvidia H200 at lower power. **No public AA benchmark.** | N/A | N/A measured | tomshardware (ISSCC 2026), nextplatform Apr 2026, businesswire Apr 10 2026 |
| **Etched** | **Sohu (transformer-only ASIC)** | **VAPOR as of May 2026.** Announced June 2024. $500M raise / $5B valuation (Stripes, Thiel). **No customer ships, no third-party benchmark, ~23 months post-announce.** Manifold prediction market trading well under 50% for "ships within a year." | N/A | N/A | **Self-claim only**: 500,000 TPS Llama 70B on 8×Sohu server; 1 server = 160 H100. 144 GB HBM3E, TSMC 4nm reticle-limit die. **Nothing independently verified.** | N/A | N/A | etched.com, tomshardware, manifold.markets |
| **Graphcore** | IPU (Bow, MK2) → **Izanagi (next-gen, ARM+IPU)** | Acquired by SoftBank July 2024; **additional $457M injection April 2026**. Repositioned away from merchant silicon — Izanagi targets US-Japan Stargate deployment for OpenAI workloads. **No merchant API, no buyable hardware** in normal channels in 2026. | No (captive to SoftBank/Stargate) | No | N/A (not selling tokens) | N/A | N/A | hpcwire BigDATAwire, eetimes, cnbc May 12 2026 |

---

## Independent (Artificial Analysis) verification — pulled May 25, 2026

From artificialanalysis.ai/models/gpt-oss-120b/providers (72-hour rolling window):

| Provider | TPS | TTFT | $/M tok |
|---|---|---|---|
| Cerebras | **1,927.7** | 1.53 s | $0.39 |
| Fireworks (Blackwell) | 717.9 | — | — |
| SambaNova | 697.4 | 4.45 s | — |
| Nebius Fast | 710.1 | — | — |
| Groq | (in benchmark, mid-pack on gpt-oss-120B at ~500–700 range) | — | $0.15 in / $0.60 out |
| DeepInfra (GPU) | — | 0.49 s | **$0.05** (cheapest) |

For Llama 4 Maverick (May 2025, still standing as 2026 ref): Cerebras 2,522 / SambaNova 794 / Groq 549 / Amazon 290 / Google 125 / Azure 54 TPS. Nvidia Blackwell DGX B200 self-reported 1,038 TPS on same model.

---

## Latency analysis for your use case (low-batch interactive, "GH200 replacement")

If the workload is **single-user chat / agentic at batch ≈ 1**, ranked best-fit:

1. **Cerebras** — fastest per-user generation, but TTFT 1.5–3 s on big models. Best for "wait a beat, then firehose." Sold out into 2027.
2. **Groq** — best TTFT in the field (P50 ~120 ms), 394 TPS on 70B, predictable P95/P50 ratio of 2.3×. Cheap. The pragmatic GH200-substitute for chat.
3. **d-Matrix Corsair** — IF the 1–2 ms/token claim survives independent benchmarking, this is purpose-designed for batch-1 interactive. No AA verification yet. Worth a direct evaluation if your timeline allows.
4. **SambaNova** — high TPS but TTFT 4+ s drags interactive feel. Strong choice for 405B+ where others can't fit.
5. **Tenstorrent Galaxy Blackhole** — best **self-host** economics ($110K/system for 23 PFLOPS) but software maturity behind; expect 40–60% of theoretical throughput on real LLMs in early 2026.
6. **Furiosa RNGD** — interesting at the edge / regulated deployments (180W, air-cooled, 2-card 70B). Modest per-user TPS.
7. **Rebellions Rebel100** — credible silicon but go-to-market is Korea-first / sovereign; non-trivial for a US buyer in May 2026.
8. **Etched Sohu** — not real yet.
9. **Graphcore** — not addressable.

---

## Per-token cost summary (where disclosed)

| Provider | Llama-class 70B (input/output $/M) |
|---|---|
| Groq | 0.59 / 0.79 |
| Cerebras | ~0.60 (blended) |
| SambaNova | 0.60 / 1.20 |
| Commodity 8×H100 vLLM (reference) | ~1.90 blended |
| Nvidia Blackwell B100/B200/B300 (per-tok) | ~0.25 (one source) |

Specialty hardware is **3–10× cheaper per token than commodity H100** on the open-weight Llama line, with **higher speed**.

---

## Notable May 2026 developments (relevant context)

- **Cerebras IPO** May 14, 2026: priced $185, opened $350, closed +68% at $311. Mcap ~$95B. Filed with $26.6B target, blew through it. CFO confirms "sold out into 2027."
- **Nvidia Groq 3 LPX** unveiled at GTC 2026 (March 16): 128-LPU rack co-deployed with Vera Rubin NVL72. Claims 35× throughput/MW. This is Nvidia owning Groq's IP after the Dec 2025 $20B license. Groq Inc. continues independently under CEO Simon Edwards.
- **SambaNova SN50** launched Feb 24 2026 (ships H2 2026): 64 GB HBM + 432 MB SRAM + 256 GB–2 TB DDR5 tiered memory; supports 10T-param / 10M-context. SambaRack uses 16 chips at 20 kW air-cooled.
- **Tenstorrent Galaxy Blackhole GA** April 28, 2026: 23 PFLOPS BFP8, 350 TPS DeepSeek R1, $110K start.
- **d-Matrix Raptor** taping out on 4nm + 3DIMC 3D-stacked DRAM (joint with Alchip), claimed 150 TB/s on-chip bandwidth, 10× HBM4. Commercial in 2026.
- **Rebellions** pre-IPO $400M @ $2.34B March 30, 2026; RebelRack + RebelPOD launched April 2026 with SKT/Arm partnership.
- **Furiosa RNGD** in mass production; 4,000 units delivered from TSMC/ASUS first wave; ramping to 2,000–3,000/mo by end-2026.

---

## Confidence & Gaps

**High confidence:**
- Cerebras, Groq, SambaNova performance numbers (Artificial Analysis-verified, multiple corroborating sources, recent dates).
- Cerebras IPO event (May 14 2026, CNBC primary).
- Nvidia-Groq deal and Groq 3 LPX (multiple GTC 2026 sources).
- Tenstorrent Galaxy GA, pricing ($110K), DeepSeek R1 350 TPS claim (theregister, hpcwire April 28 / May 1 2026).
- Etched Sohu still not shipping — confirmed by multiple skeptical sources including Manifold market.

**Medium confidence:**
- d-Matrix Corsair perf claims (60K TPS Llama3-8B @ 1ms, 30K TPS Llama3-70B @ 2ms) — **vendor-reported only, no AA verification**. Latency claims are dramatic and need direct benchmarking before relying on them.
- SambaNova SN50 5× Blackwell perf — vendor claim, not yet AA-verified (chip ships H2 2026).
- Tenstorrent Blackhole real-world throughput: independent sources warn 40–60% of theoretical TFLOPS on LLMs vs 60–80% for Nvidia (software maturity gap).

**Low confidence / gaps:**
- **Self-host pricing for Cerebras CS-3** — never officially disclosed; $2–3M/chip estimate is industry rumor. If you want to actually buy hardware, you must contact sales.
- **d-Matrix Corsair card price** — not public.
- **Furiosa RNGD per-card price** — not public.
- **Rebellions per-card and rack price** — not public; primarily Korean sovereign-cloud channel.
- **Etched Sohu real performance** — zero independent data exists. Treat all claims as marketing.
- **Graphcore Izanagi** — referenced for Stargate deployment but no specs, no merchant availability.

**Methodological caveats:**
- Artificial Analysis polls API endpoints every 8 hours; reported numbers are 72-hour averages and can shift with provider config changes.
- All TPS-per-user numbers assume the provider's published model configuration (precision, MoE expert count). Cross-vendor comparisons are valid only when the model and precision are identical (gpt-oss-120B and Llama 4 Maverick are good apples-to-apples; many other benchmarks are not).
- For an actual GH200-replacement decision, **the only defensible path is to run your own workload** on Groq Cloud, Cerebras Cloud, and one of (Tenstorrent Galaxy / d-Matrix Corsair early-access / Furiosa NXT) and measure.

---

## Source URLs (with access dates, all May 25 2026)

**Groq:**
- https://groq.com/pricing (May 2026 model list & per-tok prices)
- https://sacra.com/c/groq/ (equity research Feb 2026)
- https://www.voiceflow.com/blog/groq (Nvidia $20B deal context)
- https://spectrum.ieee.org/nvidia-groq-3 (Groq 3 LPX GTC 2026)
- https://the-decoder.com/gtc-2026-with-groq-3-lpx-nvidia-adds-dedicated-inference-hardware-to-its-platform-for-the-first-time/

**Cerebras:**
- https://artificialanalysis.ai/models/gpt-oss-120b/providers (live benchmark, accessed May 25 2026)
- https://www.cerebras.ai/press-release/maverick (Llama 4 Maverick 2,522 TPS vs Blackwell 1,038)
- https://www.cerebras.ai/blog/cerebras-launches-openai-s-gpt-oss-120b-at-a-blistering-3-000-tokens-sec
- https://www.cnbc.com/2026/05/14/cerebras-cbrs-stock-trade-nasdaq-ipo.html
- https://www.cnbc.com/2026/05/15/cerebras-stock-ipo-debut-ai.html
- https://www.cerebras.ai/pricing

**SambaNova:**
- https://sambanova.ai/blog/introducing-the-sn50-rdu-purpose-built-for-agentic-inference (Feb 24 2026)
- https://www.hpcwire.com/2026/02/24/sambanova-eyes-10-trillion-parameter-models-for-agentic-ai-with-new-chip/
- https://www.cnbc.com/2026/02/24/intel-partners-with-sambanova-after-buyout-talks-reportedly-failed.html
- https://x.com/ArtificialAnlys/status/1833502830713593933 (AA on 405B 132 TPS)

**Tenstorrent:**
- https://www.theregister.com/2026/04/28/tenstorrents_galaxy_blackhole_ai_servers_ga/
- https://www.hpcwire.com/aiwire/2026/05/01/tenstorrent-announces-general-availability-of-galaxy-blackhole-ai-system/
- https://wccftech.com/tenstorrent-vows-to-crush-everyone-galaxy-blackhole-hits-350-tokens-on-deepseek-r1-undercut-nvidia-gb300-ai-tco/
- https://www.spheron.network/blog/tenstorrent-vs-nvidia-open-source-ai-hardware/

**Furiosa:**
- https://furiosa.ai/blog/rngd-enters-mass-production-the-high-performance-ai-accelerator-for-any-data-center
- https://furiosa.ai/blog/furiosaai-sdk-2025-3-boosts-rngd-performance-with-multichip-scaling-and-more (Feb 14 2026 SDK)
- https://www.koreaherald.com/article/10708877
- https://www.networkworld.com/article/4029832/ai-chip-startup-furiosaai-strikes-deal-with-lg-with-enterprise-customers-in-mind.html

**d-Matrix:**
- https://www.d-matrix.ai/announcements/d-matrix-unveils-corsair-the-worlds-most-efficient-ai-computing-platform-for-inference-in-datacenters/
- https://www.servethehome.com/d-matrix-corsair-in-memory-computing-for-ai-inference-at-hot-chips-2025/
- https://www.d-matrix.ai/announcements/d-matrix-and-alchip-announce-collaboration-on-worlds-first-3d-dram-solution-to-supercharge-ai-inference/
- https://siliconangle.com/2025/11/12/chip-startup-d-matrix-raises-275m-speed-inference-memory-compute/

**Rebellions:**
- https://www.tomshardware.com/tech-industry/semiconductors/isscc-2026-rebellions-ucie-rebel-100
- https://rebellions.ai/newsroom/rebellions-closes-400-million-pre-ipo-and-launches-rebelrack-and-rebelpod-to-accelerate-global-expansion/
- https://www.nextplatform.com/compute/2026/04/06/rebellions-ai-rings-up-the-money-to-rack-up-ai-inference-systems/5214660
- https://www.businesswire.com/news/home/20260410578210/en/Rebellions-Collaborates-with-SK-Telecom-and-Arm-Targeting-Sovereign-AI-and-Telecom-Infrastructure (Apr 10 2026)

**Etched:**
- https://awesomeagents.ai/hardware/etched-sohu/
- https://manifold.markets/ahalekelly/will-the-sohu-ai-chip-ship-to-custo
- https://www.jonpeddie.com/news/welcome-to-the-club-etched/
- https://theaiworld.org/news/etcheds-500m-sohu-chip-takes-aim-at-nvidia

**Graphcore:**
- https://www.hpcwire.com/bigdatawire/this-just-in/softbank-expands-ai-portfolio-with-graphcore-acquisition/
- https://www.cnbc.com/2026/05/12/softbank-graphcore-ai-chip-investment.html
- https://www.tekedia.com/softbank-deepens-ai-hardware-push-with-fresh-457-million-injection-into-graphcore/

**General benchmark / price reference:**
- https://artificialanalysis.ai/leaderboards/models
- https://www.digitalapplied.com/blog/ai-inference-providers-pricing-matrix-q2-2026
- https://www.cloudzero.com/blog/groq-pricing/
- https://introl.com/blog/inference-unit-economics-true-cost-per-million-tokens-guide
