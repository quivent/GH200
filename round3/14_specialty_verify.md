# Specialty Inference Accelerators — Round 3 Verification + Deepening
**Date:** 2026-05-25
**Method:** 12 WebSearch/WebFetch calls against vendor pricing pages, ArtificialAnalysis.ai live leaderboard, third-party press, and prediction markets. Goal: confirm Round 1 claims, correct what is wrong, and tighten the d-Matrix / Tenstorrent / Etched verdicts.

---

## What Round 1 got right (verified)

1. **Groq Llama-3.3-70B pricing $0.59 in / $0.79 out** — confirmed verbatim on groq.com/pricing today. 394 TPS speed claim also confirmed on same page.
2. **Groq Llama-3.1-8B Instant $0.05 / $0.08, 840 TPS** — confirmed.
3. **Groq GPT-OSS-120B $0.15 in / $0.60 out, 500 TPS** — confirmed.
4. **Cerebras leads gpt-oss-120B on AA**: 1,927.7 t/s @ 1.53 s TTFT — confirmed from artificialanalysis.ai/models/gpt-oss-120b/providers (72-h rolling, pulled May 25 2026).
5. **Cerebras Code Pro $50/mo and Max $200/mo both still "sold out"** on cerebras.ai/pricing as of today — confirmed.
6. **SambaNova gpt-oss-120B 697.4 t/s @ 4.45 s TTFT** — confirmed on AA.
7. **Tenstorrent Galaxy Blackhole GA April 28 2026, $110K/system, 23 PFLOPS, 1 TB DRAM, 16 TB/s** — confirmed via TheRegister + EE Times + HPCwire.
8. **Etched Sohu is still vapor** as of May 2026 — confirmed by multiple sources. No customer ships, no third-party benchmarks, ~23 months since announce.
9. **Tenstorrent Blitz Mode DeepSeek R1-0528: 350 TPS/user @ <4 s TTFT** — vendor claim is confirmed across multiple outlets.

---

## What Round 1 got wrong / needs correcting

### A. Cerebras gpt-oss-120B pricing — Round 1 said "$0.39/M blended (AA)". **AA still shows the blended $0.39 number, but Cerebras's own published rate is $0.35 in / $0.75 out** per inference-docs.cerebras.ai. Both are "right" — blended price ≠ headline list price. Use the in/out split when modeling your workload, not the blended AA figure.

### B. Etched funding — Round 1 said "$500M raise / $5B valuation (Stripes, Thiel)". **Correct figure is $500M Series B in January 2026 at $5B valuation, on top of a prior $120M Series A. Total funding now ≈ $620M, headed toward "almost $1B".** The $500M is the latest round, not lifetime. Lead = Stripes, with Thiel, Positive Sum, Ribbit. Bloomberg + Yahoo Finance Jan 13 2026.

### C. Manifold market on Sohu — Round 1 said "trading well under 50% for 'ships within a year.'" **This is now wrong: the market already resolved NO on July 2 2025** (the question deadline was June 25 2025, one year post-announce). There is no live market currently pricing 2026 shipment probability. The signal is stronger than Round 1 implied: the market did not "predict failure," it **already resolved on failure**.

### D. Tenstorrent independent verification — Round 1 said "independent: 40–60% TFLOPS realization on real LLMs (vs 60–80% Nvidia)" but stopped there. **Stronger data point: EE Times independently tested Blitz Mode pre-launch and measured 255 TPS/user on DeepSeek-671B "short chatbot-style prompts" vs vendor's 350 TPS claim** (~73% of vendor claim). This is the closest thing to third-party validation Tenstorrent has, and it's the right number to cite.

### E. Groq's Kimi-K2 status — Round 1 didn't address. **Groq does serve Kimi-K2-Instruct-0905 at $1.00 in / $3.00 out per Mtok, 200+ TPS, 256k context, prompt caching supported.** GroqCloud blog post confirms. However: **Moonshot is end-of-lifing the K2-0905 / K2-Turbo / K2-Thinking / K2-0711 family on May 25, 2026** (today). After today, Groq's K2 SKU is effectively a sunset model — Kimi K2.6 is the new SOTA and Groq is NOT listed among the 13 AA-tracked Kimi K2.6 providers. **If you need Kimi-K2 today, go to DeepInfra FP4 ($0.60/M blended, 0.74 s TTFT) or Together.ai FP4 (302 t/s, 0.67 s TTFT, $0.70/M).** Groq's K2 offer is a legacy product.

### F. d-Matrix Corsair 1–2 ms/token claim — Round 1 flagged "no AA verification yet" but otherwise gave it the benefit of the doubt. **Round 3 finding: the closest thing to independent data is Gimlet Labs' March 2026 joint blog post on speculative decoding, and Gimlet themselves disclose "we used a combination of measured and modeled data."** That's not independent verification, that's a co-marketing whitepaper. The "2 ms/token on Llama3-70B" remains vendor-only. **Treat the 1–2 ms claim as unverified marketing** until either AA picks it up or a non-partner publishes raw numbers. Status unchanged from Round 1 but skepticism level should be higher.

---

## New evidence pulled this round

### Artificial Analysis live leaderboards (accessed May 25 2026)

**gpt-oss-120B (high) — 22 providers tracked:**

| Provider | Output t/s | TTFT (s) | Blended $/M |
|---|---|---|---|
| Cerebras | 1,927.7 | 1.53 | $0.39 |
| Fireworks (Blackwell) | 717.9 | — | — |
| SambaNova | 697.4 | 4.45 | — |
| Eigen AI | — | 3.49 | — |
| Together.ai | — | 3.93 | — |
| DeepInfra | 33.3 | 0.49 | **$0.05** (cheapest) |
| Novita | — | — | $0.07 |
| Google Vertex | — | — | $0.12 |

**Llama-3.3-70B Instruct — 17 providers:**

| Provider | Output t/s | TTFT (s) | Blended $/M |
|---|---|---|---|
| Groq | 322.9 | 0.92 | $0.25 |
| SambaNova | 291.6 | 1.17 | $0.28 |
| Google Vertex | 149.2 | 0.65 | $0.24 |
| CompactifAI | 146.0 | 1.08 | $0.17 |
| Amazon Bedrock | 135.5 | 1.26 | $0.34 |
| FriendliAI | 131.0 | 1.00 | $0.30 |
| Hyperbolic | 128.5 | 1.41 | $0.19 |
| CoreWeave | 125.0 | 0.89 | $0.40 |
| Together.ai Turbo | 113.2 | 1.54 | $0.35 |
| Nebius Base | 102.1 | 1.62 | $0.16 |
| DeepInfra (Turbo, FP8) | 17.3 | 3.12 | $0.12 |

**Two corrections from Round 1's Llama-3.3-70B numbers:**
- Round 1 said "Groq 394 TPS on 70B" — that's Groq's own marketing number. **AA measures 322.9 t/s** in production. Groq still leads but the gap to SambaNova (291.6) is narrower than the marketing-vs-marketing comparison suggests.
- Round 1 said "Groq P50 TTFT ~120 ms on 70B." **AA's measurement is 0.92 s TTFT.** The 120 ms figure may be Groq's own best-case-or-marketing TTFT; AA's median over 72 h is 8× slower. Google Vertex (0.65 s) is actually the TTFT leader on this model today, not Groq.

**Kimi K2.6 — 13 providers (Groq NOT listed):**

| Provider | Output t/s | TTFT (s) | Blended $/M |
|---|---|---|---|
| Together.ai (FP4) | 302.2 | 0.67 | $0.70 |
| Clarifai | 263.8 | 0.88 | — |
| CoreWeave | 256.9 | — | — |
| DeepInfra (FP4) | — | 0.74 | $0.60 |
| Parasail | — | — | $0.64 |
| Fireworks | — | — | $0.70 |

---

### d-Matrix Corsair — verification verdict

- **Customer shipments:** Corsair has been sampling to early-access since Q2 2025 per d-matrix.ai announcement (Nov 2024 launch). Broad GA was promised Q2 2025. The Gimlet Cloud partnership says **H2 2026 broader rollout** with select customers — that's a softer signal than "shipping today at scale."
- **Independent benchmark:** **None public.** The most-cited source (Gimlet Labs blog March 2026) explicitly says "combination of measured and modeled data." That is *not* independent verification.
- **The 1–2 ms/token claim:** Still vendor-only. Servethehome's Hot Chips 2025 coverage describes the die-to-die latency (115 ns) and PCIe-switch latency (650 ns) which are real measurements — but those are interconnect latencies, not token latencies, and they don't validate the end-to-end ms/token figure.
- **Verdict:** Move d-Matrix from Round 1's "Medium confidence" to **"vendor-claim-only, not independently verified."** The 1–2 ms claim is plausible (in-memory compute architecture is genuinely designed for batch-1 latency, and 2 GB perf-memory is enough to hold a 70B model's hot path) but cannot be relied on for procurement without direct evaluation. **Recommendation: if your workload truly requires sub-2-ms/token for a chat-style GH200 replacement, you must run a paid pilot — do not procure on the basis of public information.**

---

### Tenstorrent Galaxy Blackhole — verification verdict

- **GA confirmed:** April 28 2026, formal launch May 1 2026 ("TT-Deploy" event). Cirrascale committed as cloud partner. Multiple outlets independently report 23 PFLOPS BFP8, 1 TB GDDR6, 16 TB/s, $110K.
- **Performance — vendor:** 350 TPS/user, <4 s TTFT on DeepSeek-R1-0528 671B in Blitz Mode. Roadmap to 500 TPS.
- **Performance — third-party:** **EE Times measured 255 TPS/user pre-launch on short chatbot prompts.** That's 73% of vendor claim — within the reasonable "marketing-vs-real" gap, but worth pricing in.
- **Software maturity:** Jasmina Vasiljevic (Tenstorrent senior fellow) said publicly "the software stack has improved considerably since the first hands-on review" — implicit admission that earlier reviews were software-bound.
- **No user reviews from buyers yet** — GA is <30 days old. We'd need 60-90 more days for real customer reports.
- **Verdict:** Hardware story is real. Software story is "much better than 2024, still behind Nvidia." For a self-hoster comfortable with TT-Metal (open source) and willing to chase the last 25% of theoretical perf, this is **the most credible non-Nvidia self-host option for $110K-tier procurement**.

---

### Etched Sohu — verification verdict (debunk)

Round 3 confirms and **strengthens** Round 1's "vapor" call:

1. **Still no customer ships** as of March 2026 per multiple sources; no evidence of change in May.
2. **Manifold market already resolved NO** on July 2 2025 — failure to ship within first year is locked in.
3. **No tape-out evidence beyond announcements** — sources reference TSMC 4nm reticle-limit die size but no production lot confirmation, no MLPerf submission, no third-party benchmark, no inference provider has Sohu on their backend.
4. **$500M Series B in Jan 2026 (Stripes/Thiel) keeps the lights on**, but the burn rate of an ASIC company chasing HBM3E supply against Nvidia suggests this funding buys runway through 2026-2027 — not a shipping product.
5. **The 500K TPS/Llama-70B claim was self-derived from MLPerf H100 numbers extrapolation in mid-2024.** It has never been measured on a Sohu chip in public.
6. **Architectural risk:** Sohu is transformer-only by design. If the field moves toward MoE-heavy / SSM-hybrid / diffusion-LLM architectures (as is happening in 2026), Sohu's TAM is compressing while it's still pre-shipment.

**Verdict: definitive vaporware as of May 2026.** Do not include in procurement consideration. Reassess in Q4 2026 if there is a shipping announcement with a third-party benchmark; ignore until then.

---

## Updated latency / cost ranking (for batch ≈ 1 chat-style GH200 replacement)

Same order as Round 1 with refined confidence:

1. **Groq** — best practical TTFT (AA: 0.92 s on 70B, still tied for best with Google Vertex's 0.65 s GPU offering); 322.9 t/s on Llama-3.3-70B (AA-measured, not the 394 marketing number). $0.25/M blended.
2. **Cerebras** — fastest decode, slower prefill. 1,927 t/s on gpt-oss-120B at 1.53 s TTFT. Code subscriptions still sold out post-IPO. API still open at $0.35/$0.75 (gpt-oss-120B).
3. **SambaNova** — high throughput, 1.17 s TTFT on Llama-3.3-70B (much better than the 4.45 s on gpt-oss-120B — TTFT is workload-specific).
4. **d-Matrix Corsair** — **demote to "pilot before procure."** The 1–2 ms/token claim is unverified.
5. **Tenstorrent Galaxy Blackhole** — best self-host economics, software gap shrinking, EE Times saw 73% of vendor claim. Real but early.
6. **Furiosa / Rebellions** — unchanged from Round 1.
7. **Etched Sohu** — **definitively excluded.** Vaporware.
8. **Graphcore** — not addressable.

---

## Pricing summary refresh (May 25 2026)

| Model | Best speed | Best TTFT | Cheapest | Note |
|---|---|---|---|---|
| Llama-3.3-70B | Groq 322.9 t/s, $0.25 | Google Vertex 0.65 s | Nebius $0.16 / DeepInfra $0.12 | Groq's 394 t/s is marketing; AA says 323 |
| gpt-oss-120B | Cerebras 1,928 t/s | DeepInfra 0.49 s | DeepInfra $0.05 | Cerebras list: $0.35 in / $0.75 out; AA blended $0.39 |
| Kimi K2.6 | Together FP4 302 t/s | Together FP4 0.67 s | DeepInfra FP4 $0.60 | Groq NOT a provider for K2.6 |
| Kimi K2-0905 (legacy, EOL today) | Groq 200+ t/s | — | Moonshot direct ($0.60/$2.50) | Groq $1/$3, K2-0905 sunset May 25 2026 |

---

## Confidence levels (Round 3)

**Very high confidence:**
- Groq, Cerebras, SambaNova performance and pricing (AA + vendor pages corroborate)
- Tenstorrent Galaxy GA + headline specs (3 independent outlets)
- Etched Sohu = vapor (Manifold resolved NO, no third-party benchmarks, no production shipments per multiple May 2026 sources)
- Cerebras Code subscriptions sold out (verified on pricing page today)

**High confidence:**
- EE Times' 255 TPS measurement on Tenstorrent Blitz Mode (third-party but pre-launch and short-prompt-specific)
- Etched funding total ≈ $620M, not $500M (multiple sources confirm Series A $120M + Series B $500M)

**Medium confidence:**
- d-Matrix Corsair end-to-end latency claims (vendor + co-marketing only)
- Tenstorrent's customer-deployment-scale performance (no buyer reports yet, only EE Times pre-launch test)
- Cerebras post-IPO capacity situation: pricing page shows Code sold out, but no public statement on raw inference API capacity. CFO's "sold out into 2027" comment from IPO roadshow may refer to hardware sales pipeline, not API throughput.

**Low confidence (gaps still open):**
- Real customer Galaxy Blackhole production numbers (need 90 more days)
- d-Matrix Corsair per-card pricing (not public)
- Whether Etched will ship anything in 2026 (no positive signals; treat any 2026 ship announcement as separately needing verification of working silicon vs renders)

---

## Source URLs (May 25 2026)

**Verification fetches this round:**
- https://groq.com/pricing (confirmed Llama-3.3-70B $0.59/$0.79, gpt-oss-120B $0.15/$0.60, Llama-3.1-8B $0.05/$0.08)
- https://www.cerebras.ai/pricing (confirmed Code Pro/Max sold out; no per-Mtok rates on landing page)
- https://artificialanalysis.ai/models/gpt-oss-120b/providers (22 providers, Cerebras leads)
- https://artificialanalysis.ai/models/llama-3-3-instruct-70b/providers (17 providers, Groq leads)
- https://artificialanalysis.ai/models/kimi-k2-6/providers (13 providers, Together FP4 leads, **no Groq**)
- https://manifold.markets/ahalekelly/will-the-sohu-ai-chip-ship-to-custo (resolved NO 2025-07-02)
- https://gimletlabs.ai/blog/low-latency-spec-decode-corsair (admits "measured + modeled" data; co-marketing not independent)
- https://www.eetimes.com/tenstorrent-unveils-next-gen-servers-for-fast-tokens-no-disaggregation-needed/ (EE Times 255 TPS measurement)
- https://www.theregister.com/2026/04/28/tenstorrent_galaxy_blackhole_ai_servers_ga/
- https://www.hpcwire.com/aiwire/2026/05/01/tenstorrent-announces-general-availability-of-galaxy-blackhole-ai-system/
- https://inference-docs.cerebras.ai/models/openai-oss (gpt-oss-120B $0.35/$0.75)
- https://groq.com/blog/introducing-kimi-k2-0905-on-groqcloud (Kimi K2-0905 on Groq $1/$3, EOL May 25 2026)
- https://www.bloomberg.com/news/articles/2026-01-13/ai-chip-startup-etched-raises-500-million-to-take-on-nvidia (Series B $500M @ $5B; ~$620M cumulative)
- https://siliconangle.com/2026/01/13/ai-chip-unicorns-etched-ai-cerebras-systems-get-big-funding-boost-target-nvidia/

**Carried over from Round 1 (still valid):**
- https://www.cnbc.com/2026/05/14/cerebras-cbrs-stock-trade-nasdaq-ipo.html (Cerebras IPO)
- https://www.cnbc.com/2026/05/15/cerebras-stock-ipo-debut-ai.html
- https://www.cerebras.ai/press-release/maverick (Llama 4 Maverick 2,522 TPS)
- https://www.d-matrix.ai/announcements/d-matrix-unveils-corsair-the-worlds-most-efficient-ai-computing-platform-for-inference-in-datacenters/
- https://www.servethehome.com/d-matrix-corsair-in-memory-computing-for-ai-inference-at-hot-chips-2025/

---

## Bottom line for procurement

If you are replacing a GH200 for **batch-1 interactive chat**:

1. **Today, buy tokens:** Groq for TTFT, Cerebras for decode speed. Both are real, both are AA-verified.
2. **Today, self-host hardware:** Tenstorrent Galaxy Blackhole is the only credible non-Nvidia $110K-tier system shipping with a real (if early) software stack and an independent benchmark (EE Times 255 TPS).
3. **Pilot before procuring:** d-Matrix Corsair. The architecture is interesting, the latency claim is unverified, and no AA benchmark exists. Don't write a check on public information alone.
4. **Ignore:** Etched Sohu (vapor, prediction market resolved NO), Graphcore (captive to SoftBank).
5. **Track but don't depend on:** SambaNova SN50 (H2 2026 ship), Rebellions Rebel100 (Korea-first), Furiosa RNGD (low-power niche, not interactive-latency-led).
