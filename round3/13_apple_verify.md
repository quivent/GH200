# Apple Silicon — Round 3 Verification & Deepening

**Date: 2026-05-25** | Verifies round1/13_apple_silicon.md against live sources
**Stack note:** FP8 broken on DiT/GH200; bf16 default. Apple has *no* FP8 hardware at all (separate problem).

---

## TL;DR of corrections

1. **The 512 GB SKU is still gone.** Pulled Mar 5 2026. Confirmed by MacRumors, Tom's Hardware, 9to5Mac, MacSparky, MacObserver. Round1 correct.
2. **NEW since round1: 256 GB also gone, as of May 5 2026.** Round1 said "256 GB available, 4–5 mo lead time" — that is now wrong. MacRumors May 5 2026 + PC Guide + MacObserver: Apple removed the 256 GB M3 Ultra config from the online store. **Today (May 25 2026) the M3 Ultra Mac Studio caps at 96 GB.** Higher configs are not orderable from apple.com. Tim Cook (earnings call): "the Mac mini and Mac Studio may take several months to reach supply demand balance."
3. **M5 Ultra: no official Apple announcement.** Pure leak/analyst territory. October 2026 is the most-cited target (down from original H1-2026 / WWDC 2026 expectation), with DRAM shortage as the cited cause of the slip.
4. **vllm-mlx v0.3.0 released May 9 2026.** Active. Claims 4.3× aggregate scaling at 16 concurrent. Documented benchmarks are M4 Max only (Qwen3-0.6B 417.9 tok/s, Llama-3.2-3B 205.6 tok/s). **No clean M3 Ultra multi-user numbers from primary source.** Real-world test (purplemaia labs) on M3 Ultra reports vllm-mlx OOM-crashed at 4 concurrent with MiniMax M2.1 5-bit; LM Studio was their working path. So the 4.3×@16 claim does *not* yet generalize to large MoE models on M3 Ultra.
5. **EXO Labs is open-source, not a commercial product.** EXO 1.0 shipped Nov 2025, RDMA-over-Thunderbolt-5 day-0, Apache-2.0. No enterprise/SLA tier surfaced. DeepSeek V3.1 671B at 24-26 tok/s across a TB5 mesh of Macs is the headline demo.
6. **DiT data on Apple Silicon is thin.**
   - **Flux.2 Klein on M3 Ultra: no primary benchmark surfaced.** Closest is "MacBook M3 Max → 14.23 s per 1024² with 6-bit Metal impl" (extrapolation only; M3 Ultra would be faster but not measured). Flux.2 Dev MLX: ~30 min per 1024² image per the flux-2-swift-mlx repo (any M-series), Klein 4B ~26 s, Klein 9B ~62 s in their CLI examples (machine unspecified).
   - **Wan 2.2 on M3 Ultra: zero primary benchmark.** mlx-video (Blaizzy) supports Wan2.2 T2V-14B / TI2V-5B / I2V-14B but publishes no perf numbers. Only public Wan 2.2 Mac test found was M3 Max 36 GB MacBook Pro via ComfyUI — author didn't post timings, asked the community to.

The hardware-availability finding (#2) is the load-bearing change. Anyone planning a procurement off round1's table will be blocked at the store.

---

## 1. Mac Studio availability — direct configurator check

| SKU | Status May 25 2026 | Source |
|---|---|---|
| M3 Ultra 512 GB | **Removed Mar 5 2026** | [MacRumors](https://www.macrumors.com/2026/03/05/mac-studio-no-512gb-ram-upgrade/), [Tom's Hardware](https://www.tomshardware.com/tech-industry/apple-pulls-512-mac-studio-upgrade-option) |
| M3 Ultra 256 GB | **Removed early May 2026** | [MacRumors May 5 2026](https://www.macrumors.com/2026/05/05/apple-mac-studio-mac-mini-ram-cuts/), [PC Guide](https://www.pcguide.com/news/mac-studio-m3-ultra-loses-512gb-256gb-configs-while-mac-mini-price-jumps-200-as-cheapest-model-removed/), MacObserver |
| M3 Ultra 128 GB | Removed (round1 didn't list this; now also gone per MacRumors May 5) | MacRumors |
| **M3 Ultra 96 GB** | **Available — now the maximum** | Multiple confirmations |
| M4 Max higher tiers | Also cut May 2026 (specific tier not enumerated in coverage) | MacRumors May 5 |
| M4 Max 36 / 64 GB | Available | Apple Shop |

Apple shop page itself was uninformative (couldn't fetch full configurator state with WebFetch — only shows top-of-page summaries: "M3 Ultra, 96 GB" and "M4 Max, 36 GB"). The press coverage is the citable source.

**Lead time:** "9–10 weeks" for both M3 Ultra and M4 Max Mac Studio orders (vs round1's "4–5 months" — likely slightly improved because the highest configs are no longer offered, so the remaining 96 GB SKU ships faster).

**Implication for your stack:**
- DeepSeek-V3 671B local (round1's headline use case for 512 GB Macs) is **no longer a buy-new option from Apple.** Refurb / used / channel-only.
- Llama 70B 4-bit (~40 GB) still fits 96 GB comfortably. Qwen3-235B-A22B 4-bit (~125 GB at quant + ~10 GB context) **does not fit** 96 GB. Round1 had Qwen3-235B as a sweet-spot use case — that requires the now-discontinued 256 GB SKU minimum.
- Practical 2026 Apple ceiling for new-purchase LLM work is ~70 B class 4-bit, not 235 B / 671 B.

**Pricing note:** Apple raised the 96 GB upgrade pricing along with the cuts; the cheap M3 Ultra entry is now more expensive than what round1 lists. Tom's Hardware reported the 256 GB upgrade jumped $400 just before its own removal.

---

## 2. M5 Ultra release timing

| Claim | Status | Source |
|---|---|---|
| Apple has officially announced M5 Ultra | **NO** | All coverage (Macworld, MacRumors guide, Geeky Gadgets, TechRepublic) explicitly says no official date |
| M5 Ultra Mac Studio originally expected | H1 2026 / WWDC 2026 (Gurman Nov 2025) | [MacRumors Nov 4 2025](https://www.macrumors.com/2025/11/04/mac-studio-m5-ultra-2026/) |
| Current expected slip | **October 2026** | [Macworld](https://www.macworld.com/article/2973459/2026-mac-studio-m5-release-date-specs-price-rumors.html), Geeky Gadgets, TechRepublic |
| Cause of slip | Global DRAM shortage (same root cause as the 512/256 GB cuts) | Multiple |
| Memory ceiling rumor | "Up to 512 GB if DRAM frees up" — speculative | Macworld |

Round1 had this directionally right; no correction needed beyond noting that the slip-to-October consensus has hardened over April–May 2026.

**Procurement implication:** if you need >96 GB unified memory from Apple, your options are (a) refurb/used 512 GB M3 Ultra, or (b) wait for M5 Ultra (likely Oct 2026, no guarantee of 512 GB return). Both are bad if you need it now.

---

## 3. vllm-mlx — current state

| Item | Value | Source |
|---|---|---|
| Current version | **v0.3.0** | github.com/waybarrios/vllm-mlx |
| Release date | May 9 2026 | repo |
| Commits | 483 on main | repo |
| Headline benchmark | "400+ tok/s" (M4 Max) | repo README |
| Decode (M4 Max 128 GB) | Qwen3-0.6B 417.9 tok/s @ 0.7 GB; Llama-3.2-3B 205.6 tok/s @ 1.8 GB | repo |
| Concurrency scaling claim | 4.3× aggregate @ 16 concurrent (general) | repo / arxiv 2601.19139 |
| API surface | OpenAI + Anthropic Messages | repo |
| MCP tool calling | Supported | repo |

**M3 Ultra-specific real-world tests:**
- purplemaia labs ran MiniMax M2.1 (MoE) MLX 5-bit on 512 GB M3 Ultra. **vllm-mlx OOM-crashed at 4 concurrent.** They fell back to LM Studio 0.4 (which added concurrent inference in this period). Optimal concurrency: 3–4; 6 concurrent caused swap thrashing. Total throughput claim: "~50 % higher generation than LM Studio single-thread" — but they couldn't get absolute numbers from vllm-mlx on that model.
- MACGPU 2026 framework bench compared all three (vllm-mlx / Ollama / llama.cpp Metal) on **M4 Pro 64 GB** with DeepSeek-V3 Q4. vllm-mlx single-user 42 tok/s; 32-concurrent 1150 tok/s; TTFT 120 ms. Ollama wins TTFT (45 ms) and single-user (58 tok/s); vllm-mlx wins concurrent aggregate.

**Verdict:** vllm-mlx is real, actively developed, and the right tool for many-concurrent text models that fit comfortably. It is **not** a free pass to serve large MoE models concurrently on M3 Ultra — at the upper memory/parameter end, OOM/swap is the bottleneck (and that bottleneck just got worse with the 256/512 GB SKU removals).

---

## 4. EXO Labs — production-grade or experimental?

**Verdict: experimental / open-source community project. Not a commercial product.**

- License: Apache-2.0
- Distribution: GitHub (github.com/exo-explore/exo)
- Latest milestone: EXO 1.0, mid-Nov 2025
- Key tech: RDMA-over-Thunderbolt-5 with day-0 macOS 26.2 support. Latency: 300 µs → 3 µs (claimed); EXO site claims **8 µs** measured with TB5+RDMA.
- Hardware support: any TB5 Mac (M4 Pro Mini, M4 Max Studio, M4 Max MBP, M3 Ultra Studio)
- Headline demo: DeepSeek V3.1 671B at **24–26 tok/s** across a TB5-meshed Mac cluster (>700 GB combined memory needed to load)
- macOS dependency: RDMA-over-TB5 requires macOS 26.2 with recovery-mode enablement
- Pricing tier / SLA: none surfaced
- Production case studies: none named on exolabs.net (claims "real hardware, real clusters" but no customer names)

This matters now that the 512 GB single-box Mac is gone: clustering is the only path to >96 GB Apple inference for new buyers. EXO is the only credible OSS option but you're betting your stack on an unfunded-looking community project with no SLA. For an org that needs >96 GB on Apple, the realistic shapes are:
- 4× M3 Ultra 96 GB clustered (≈ 384 GB total minus overhead) via EXO + TB5
- OR refurb 512 GB single-box
- OR just use cloud GH200

---

## 5. DiT image/video on Apple Silicon

This is the weakest-evidence section in round1, and verification confirms it: **the public benchmark surface for DiT on M3 Ultra is very thin.**

### Flux.2

| Datapoint | Value | Source |
|---|---|---|
| Flux.2 Klein release | Jan 2026 | bfl.ai, fal |
| Flux.2 Klein on RTX 4090 | <0.5 s per 1024² (4 steps) | bfl.ai |
| Flux.2 Klein on MacBook M3 Max (Metal, 6-bit) | **14.23 s** per 1024² | flux2klein.ai writeup (community Metal impl, not BFL official) |
| Flux.2 Klein on M3 Ultra | **No primary benchmark found** | — |
| Flux.2 Dev on Mac (MLX swift impl) | ~30 min per 1024² | flux-2-swift-mlx README |
| Flux.2 Klein 4B CLI ref | ~26 s | flux-2-swift-mlx README (hardware not stated) |
| Flux.2 Klein 9B CLI ref | ~62 s | same |
| flux-2-swift-mlx last release | v2.4.0, Mar 25 2026 | repo |
| Flux.1 dev on M3 Ultra (older Flux) | 98.56 s first image, 88.84 s subsequent | MacRumors forum (round1 cited similar) |

**Estimate (extrapolation, not measured):** Flux.2 Klein on M3 Ultra likely lands around **5–10 s per 1024²** based on M3 Max → M3 Ultra GPU/bandwidth scaling. This is an order of magnitude slower than RTX 4090's <0.5 s and ~50× slower than GH200's expected sub-100ms per image at FP8 (which is broken anyway, so call it ~few-hundred ms BF16).

### Wan 2.2

| Datapoint | Value | Source |
|---|---|---|
| MLX support | Yes, via Blaizzy/mlx-video (T2V-14B / TI2V-5B / I2V-14B) | github.com/Blaizzy/mlx-video |
| Published M3 Ultra benchmark | **None found** | — |
| Closest data point | Wan 2.2 on M3 Max 36 GB MacBook via ComfyUI — author published no timings | medium.com/@ttio2tech_28094 |
| Reference number (different model) | Wan 1.3B on M2 Ultra 125 GB: "~3 seconds [to generate]" (unclear: per frame? total clip?) | search snippet, unverified |

**Honest verdict:** if you need to forecast Wan 2.2 throughput on Apple Silicon for the project plan, **you do not yet have a defensible number from public sources.** The MLX video stack exists; the benchmarks haven't been published. For decision-making, treat Wan 2.2 on M3 Ultra as "works in principle, expect minutes-per-clip, validate before committing."

---

## 6. Net corrections to round1

| Round1 claim | Status | Correction |
|---|---|---|
| 512 GB SKU pulled Mar 2026 | **Confirmed** | — |
| 256 GB still orderable with 4–5 mo lead | **WRONG as of May 5 2026** | 256 GB also pulled. 96 GB is the new max. Lead time ~9–10 weeks. |
| 96 GB available now | Confirmed | Still true; now the ceiling |
| Qwen3-235B-A22B as a sweet-spot use case | **Now unbuyable new** | Needs ≥125 GB; doesn't fit 96 GB |
| DeepSeek-V3 671B as headline use case | **Now unbuyable new** | Needs 512 GB; refurb only |
| M4 Ultra doesn't exist | Confirmed | — |
| M5 Ultra mid-2026 → Oct 2026 slip | Confirmed | — |
| vllm-mlx ~4.3× @ 16 concurrent | Confirmed *as a claim*, not verified on M3 Ultra with large MoE | M3 Ultra real-world large-MoE concurrency capped at 3–4 (purplemaia) |
| EXO Labs RDMA over TB5 | Confirmed | EXO 1.0 shipped Nov 2025, open-source only |
| Apple has no FP8/FP4 hardware | Confirmed | — |
| MLX nvfp4 dynamic range issue | Confirmed (referenced — not re-checked this round) | — |

---

## 7. New high-confidence findings (additions to round1)

- **Apple Mac Studio supply is actively contracting.** Two RAM-tier removals in two months (Mar, May 2026). Trajectory points toward more cuts before M5 Ultra ships in Oct 2026 (which itself depends on DRAM availability). Procurement timing now matters a lot.
- **The "buy a 512 GB Mac Studio for $9.5k" thesis from 2025 is dead.** That price point isn't on Apple's store. Refurb pricing has not collapsed (DRAM shortage props it up).
- **EXO Labs is the only credible Apple-cluster software**, and it's an unfunded-looking OSS project. Stack stability risk if you bet a production deployment on it.
- **vllm-mlx hit a real wall on large MoE on M3 Ultra** (MiniMax M2.1 OOM at 4 concurrent). Plan for LM Studio fallback or smaller models if you need concurrency on M-series.
- **DiT benchmark surface for Apple Silicon is genuinely missing in May 2026.** mlx-video / flux-2-swift-mlx are active projects but haven't published a clean numbers table. For Wan 2.2 / Flux.2 production planning on M3 Ultra, you'll need to benchmark yourself.

---

## 8. Sources (added this round)

- [MacRumors May 5 2026 — Apple cuts more Mac Studio/Mini RAM options](https://www.macrumors.com/2026/05/05/apple-mac-studio-mac-mini-ram-cuts/)
- [PC Guide May 2026 — M3 Ultra loses 256GB & 512GB](https://www.pcguide.com/news/mac-studio-m3-ultra-loses-512gb-256gb-configs-while-mac-mini-price-jumps-200-as-cheapest-model-removed/)
- [MacObserver — Apple removes 256GB M3 Ultra from store](https://www.macobserver.com/news/apple-removes-256gb-m3-ultra-mac-studio-model-from-online-store/) (403 on fetch; cited via search summary)
- [MacRumors Mar 5 2026 — 512GB disappears](https://www.macrumors.com/2026/03/05/mac-studio-no-512gb-ram-upgrade/)
- [Tom's Hardware — Apple pulls 512GB upgrade](https://www.tomshardware.com/tech-industry/apple-pulls-512-mac-studio-upgrade-option)
- [Macworld — M5 Mac Studio Oct 2026 slip](https://www.macworld.com/article/2973459/2026-mac-studio-m5-release-date-specs-price-rumors.html)
- [MacRumors Nov 4 2025 — M5 Ultra in 2026](https://www.macrumors.com/2025/11/04/mac-studio-m5-ultra-2026/)
- [github.com/waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) (v0.3.0, May 9 2026)
- [purplemaia labs — MiniMax M2.1 on Mac Studio](https://blog.labs.purplemaia.org/from-benchmarks-to-builders-running-minimax-m2-1-on-our-mac-studio/)
- [purplemaia labs — vllm-metal vs vllm-mlx](https://blog.labs.purplemaia.org/two-paths-to-vllm-on-apple-silicon-vllm-metal-vs-vllm-mlx/)
- [MACGPU 2026 framework benchmark](https://macgpu.com/en/blog/2026-mac-inference-framework-vllm-mlx-ollama-llamacpp-benchmark.html)
- [arxiv 2601.19139 — Native LLM/MLLM at Scale on Apple Silicon](https://arxiv.org/pdf/2601.19139)
- [github.com/exo-explore/exo](https://github.com/exo-explore/exo)
- [exolabs.net](https://exolabs.net/)
- [Hacker News — macOS 26.2 RDMA over TB](https://news.ycombinator.com/item?id=46248644)
- [github.com/Blaizzy/mlx-video](https://github.com/Blaizzy/mlx-video)
- [github.com/VincentGourbin/flux-2-swift-mlx](https://github.com/VincentGourbin/flux-2-swift-mlx)
- [bfl.ai Flux.2 Klein](https://bfl.ai/models/flux-2-klein)
- [flux2klein.ai — M3 Max Metal 6-bit Klein benchmark](https://flux2klein.ai/blog/flux-2-klein-ultimate-guide-fastest-ai-image-generator)
- [Hostbor M3 Ultra DeepSeek test](https://hostbor.com/mac-studio-m3-ultra-tested/) (LLM-only, no Flux/Wan)
- [Tech-Practice — Wan 2.2 on M3 Max MBP](https://medium.com/@ttio2tech_28094/macbook-macmini-run-wan-2-2-generating-videos-dd0e32eb91b3)

---

## 9. Confidence & remaining gaps

**High confidence:**
- M3 Ultra ceiling is 96 GB on apple.com as of May 25 2026. (Multiple consistent press reports dated May 5 + no contradicting evidence.)
- 512 GB SKU is unavailable new. (Confirmed Mar 2026, still confirmed May 2026.)
- M5 Ultra has no official Apple date.
- vllm-mlx v0.3.0 released May 9 2026; M4 Max benchmarks are the documented set.
- EXO Labs is OSS-only, no commercial tier surfaced.

**Medium confidence:**
- 9–10 week lead time on the 96 GB SKU (one source).
- Tim Cook quote on supply balance "several months" — paraphrased through MacRumors; not verified against earnings transcript this round.
- vllm-mlx 4.3× @ 16 concurrent — claim from arxiv / repo, only the M4 Pro DeepSeek-V3 reproduction in the wild (32-user 1150 tok/s aggregate at MACGPU).

**Low confidence / open gaps:**
- **Wan 2.2 on M3 Ultra:** zero primary benchmark. Critical gap if video-DiT is in scope.
- **Flux.2 Klein on M3 Ultra:** no measured number; only M3 Max community Metal impl (14.23 s) as anchor.
- **Whether refurb / channel pricing on 512 GB M3 Ultras has spiked.** Did not check this round.
- **Concrete TTFT / prefill numbers for vllm-mlx at large context (>16k) on M3 Ultra.** Not published.
- **Apple's published delivery dates by SKU today** — couldn't get the live configurator via WebFetch (Apple's shop page returns top-of-page summaries only). Press-sourced 9–10 weeks is the best available.
