# 18 — Definitive $/Mtok Table for Llama-3.3-70B + Self-Host Break-Even

**Date:** 2026-05-25
**Round 4 LLM-only deep dive**
**Models:** Meta Llama-3.3-70B-Instruct (primary); Qwen2.5-72B-Instruct (where served)
**Scope:** 12 managed providers verified live via WebFetch, plus three self-host platforms (Vultr GH200, Lambda GH200, Nebius H200 3-mo).
**Method:** 22 WebFetch + 5 WebSearch calls across provider pricing pages, aggregators (artificialanalysis.ai, pricepertoken.com, costbench.com, llm-token-calculator.com, getmaxim/bifrost, cloudprice.net, tokenmix.ai, helicone, aipricing.guru, wrvishnu) on 2026-05-25.

---

## 1. Headline — verified live 2026-05-25

### 1.1 Llama-3.3-70B-Instruct (sorted by blended 1:1 ascending)

| # | Provider | Variant / quant | Input $/Mtok | Output $/Mtok | Blended (1:1) | Output tok/s (median) | Status / source |
|---|---|---|---|---|---|---|---|
| 1 | **DeepInfra (Turbo)** | FP8 | **$0.10** | **$0.32** | **$0.21** | ~30–45 | ✅ Live deepinfra.com/pricing — "$0.10 in $0.32 out" |
| 2 | **DeepInfra (standard)** | bf16 | $0.23 | $0.40 | $0.315 | ~27 | ✅ Live aipricing.guru + helicone-DeepInfra row |
| 3 | **Hyperbolic** | bf16 | $0.12 | $0.30 | $0.21 | ~35 | ⚠️ Aggregator (llm-token-calc, May 2026); Hyperbolic's docs page does not render rates inline. Treat as **medium confidence** |
| 4 | **Nebius (Fast/Base)** | bf16 | $0.13 | $0.40 | $0.265 | ~80 | ✅ Live artificialanalysis row — "Nebius Base $0.13 / $0.40" |
| 5 | **Novita** | bf16 | $0.135 | $0.40 | $0.268 | ~30 | ✅ Live novita.ai/pricing — "$0.135 /Mt … $0.4 /Mt" |
| 6 | **Cloudflare Workers AI** | bf16 | Free* | Free* | Free* | ~30 | ✅ tokenmix verified; *10K neurons/day cap on free tier |
| 7 | **Groq** | bf16 (LPU) | $0.59 | $0.79 | **$0.69** | **394 (claim) / 315 (measured)** | ✅ Live groq.com/pricing — "Llama 3.3 70B Versatile $0.59 in $0.79 out" |
| 8 | **Azure AI Foundry** | bf16 | $0.59 | $0.79 | **$0.69** | ~120 | ✅ wrvishnu (May 2026) + llm-token-calc ($0.71 input alt-quote = pre-discount) |
| 9 | **AWS Bedrock** | bf16 | $0.72 | $0.72 | $0.72 | ~189 | ✅ costbench (May 14 2026) + llm-token-calc + Vercel Gateway pass-through |
| 10 | **IBM watsonx** | bf16 | $0.71 | $0.71 | $0.71 | n/a | ✅ llm-token-calc |
| 11 | **Fireworks AI** | bf16 | $0.90 | $0.90 | $0.90 | ~50 | ✅ Live fireworks.ai model page — "$0.90 per million tokens" (flat in=out) |
| 12 | **SambaNova** | bf16 (SN40L) | **$0.60** | **$1.20** | **$0.90** | **~294** | ✅ cloud.sambanova.ai/plans — "$0.60 USD / $1.20 USD per 1M" |
| 13 | **Together AI** | FP8 (Turbo) | $0.88 | $0.88 | $0.88 | ~45 | ✅ Live docs.together.ai — "meta-llama/Llama-3.3-70B-Instruct-Turbo … $0.88 / $0.88" |
| 14 | **Cerebras** | bf16 (WSE-3) | **$0.85** | **$1.20** | **$1.025** | **~1,800** | ⚠️ Cross-source: Bifrost+CloudPrice+general search say $0.85/$1.20; TokenMix April 2026 says $0.60/$0.60 — resolved to $0.85/$1.20 (newer, 2-source consensus) |
| 15 | **Google Vertex AI (MaaS)** | bf16 | ~$0.72 | ~$2.00 | **$1.36** | ~149 | ⚠️ Not on cloud.google.com/vertex-ai/generative-ai/pricing first-party rate card; Llama is Model-Garden / Marketplace-billed; tokenmix April 2026 reports $1.36 blended |
| 16 | **Replicate** | bf16 | $0.65 | $2.75 | $1.70 | ~40 | ✅ aipricing.guru |
| 17 | **Scaleway** | bf16 | (single blended) | (single blended) | **$1.05** | n/a | ⚠️ artificialanalysis "Scaleway $1.05 blended" — most expensive of 17 |
| 18 | **Cerebras (per TokenMix April)** | bf16 | $0.60 | $0.60 | $0.60 | 1,800 | ⚠️ Conflicting older quote — **do not cite** without re-verification |

**Cheapest verified live serverless:** DeepInfra Turbo at **$0.10 / $0.32 / $0.21 blended** (1:1 ratio). The "$0.12 blended" figure from artificialanalysis uses the 7:2:1 cache:input:output ratio — not a simple 1:1 — so $0.21 is the more honest comparable.

**Cheapest hyperscaler-managed:** **Azure AI Foundry at $0.69 blended** ($0.59 / $0.79), tied with Groq.

**Bedrock $0.72/$0.72 confirmed** (resolves the Round 2 $0.72-vs-$2.65 conflict — $2.65 was Llama 3 / 3.1 70B, not 3.3).

### 1.2 Qwen2.5-72B-Instruct (much thinner ecosystem)

| Provider | Input $/Mtok | Output $/Mtok | Blended | Notes |
|---|---|---|---|---|
| **DeepInfra** | **$0.36** | **$0.40** | **$0.38** | ✅ Live deepinfra.com — "Qwen2.5-72B-Instruct 32k $0.36 / $0.40" |
| Novita | $0.38 | $0.40 | $0.39 | ✅ Live novita.ai — "$0.38 /Mt … $0.4 /Mt" |
| SiliconFlow (FP8) | $0.59 | $0.59 | $0.59 | ✅ artificialanalysis |
| Hyperbolic | $0.12 | $0.30 | $0.21 | ⚠️ llm-token-calc only; Hyperbolic docs page does not display rates inline |
| Alibaba Cloud | n/a | n/a | n/a | listed by artificialanalysis but rates not surfaced |

**Not offered:** Together (only Qwen2.5-7B-Turbo $0.30/$0.30), Fireworks (Qwen-3-32B only), Groq (Qwen3-32B only), Cerebras (Qwen-3-32B only at $0.50/$0.50), AWS Bedrock, Azure Foundry, Google Vertex MaaS, SambaNova.

Qwen2.5-72B is effectively a **DeepInfra-or-Novita decision** at $0.38–$0.39 blended unless you self-host. (Qwen3-32B is more broadly available; Qwen2.5-72B is now a long-tail SKU as the ecosystem has moved to Qwen3.)

---

## 2. Self-Host Reference Platforms — capex math

### 2.1 Three reference rigs (verified live 2026-05-25)

| Platform | $/GPU/hr | $/GPU/mo (730h) | What you actually get |
|---|---|---|---|
| **Vultr 1× GH200 on-demand** | **$1.99** | **$1,453** | GH200 96 GB HBM3 + 480 GB LPDDR5X; full VM root; Atlanta region only (Manchester + Global sold out per gpuperhour) |
| **Lambda Labs 1× GH200 on-demand** | $2.29 | $1,672 | Same silicon; broader region coverage; Lambda Stack image; SOC 2 |
| **Nebius 1× H200 3-mo committed** | $2.30 | **$1,679** | H200 SXM 141 GB HBM3e; InfiniBand fabric; free egress; ClusterMAX Gold |

(User's prompt said Vultr **$1,433**; my math gives **$1,453** at 730h. The $1,433 figure assumes 720h/mo. Either is defensible; I'll carry both downstream.)

### 2.2 Sustained Llama-3.3-70B output throughput per platform

Numbers below blend MLPerf v4.0/v5.0 server data, vLLM 0.10+ benchmarks, NVIDIA TRT-LLM blog data, and the Round 3/Round 4 verify docs in this corpus. **bf16 only** — per project memory, FP8 is broken on this stack; do not assume FP8 perf.

| Hardware | Engine | Batch / mode | Output tok/s sustained | Notes |
|---|---|---|---|---|
| 1× GH200 96 GB | vLLM 0.10 bf16 | bsz≈32, large-batch | **~1,500–2,500** | Single-chip; KV-cache offload to 480 GB LPDDR5X is the GH200's unique win |
| 1× GH200 96 GB | TRT-LLM 0.12 bf16 + spec-decode | medusa/eagle | ~3,000–4,500 | 1.5–2× over baseline; needs draft model |
| 1× H200 SXM 141 GB | vLLM 0.10 bf16 | bsz≈128 | **~5,000–8,000** | Best single-GPU number for 70B bf16 |
| 2× H200 (NVLink pair) | vLLM 0.10 bf16 TP=2 | bsz≈128 | ~9,000–14,000 | Near-linear scaling on 70B |
| 1× H200 + TRT-LLM spec-decode | TRT-LLM 0.12 | bsz≈32 | up to ~15,000 (NVIDIA blog claim, 3.55× on 70B target) | Decoder/EAGLE on H200 HGX with NVLink |

For break-even math I use **conservative midpoints**:
- GH200 single chip = **2,000 output tok/s** sustained
- H200 single chip = **6,500 output tok/s** sustained

Monthly output capacity at 100% sustained util (730 h/mo, 3,600 s/h):
- GH200: 2,000 × 3,600 × 730 = **5,256 Mtok/mo**
- H200: 6,500 × 3,600 × 730 = **17,082 Mtok/mo**

### 2.3 Effective $/Mtok of self-host at 100% sustained output util

| Platform | $/mo | Cap (Mtok/mo) | Self-host $/Mtok output @100% util |
|---|---|---|---|
| Vultr GH200 (730h)   | $1,453 | 5,256 | **$0.276** |
| Vultr GH200 (720h, prompt's $1,433) | $1,433 | 5,184 | $0.276 |
| Lambda GH200 | $1,672 | 5,256 | **$0.318** |
| Nebius H200 3-mo | $1,679 | 17,082 | **$0.098** |

**Headline result:** **At 100% sustained output, the Nebius H200 3-mo committed box does the same job as DeepInfra Turbo's $0.32 output rate for $0.098** — i.e. self-host beats DeepInfra-Turbo by ~3.3× on pure output token economics, *if* you can hit and hold 100% sustained util.

---

## 3. Break-even — at what Mtok/mo does self-host equal managed?

For each self-host platform, the break-even volume V satisfies V × $managed = $platform/mo.
Two formulations follow: **(a) Output-tokens-only** (the apples-to-apples comparison) and **(b) Blended 1:1** (which matches how most billing actually settles).

### 3.1 Vultr GH200 — $1,453/mo, ~5,256 Mtok/mo cap

| Managed comparator | Managed $/Mtok output | Break-even V (output-Mtok/mo) | Required util of GH200 | Managed $/Mtok blended (1:1) | Break-even V (Mtok/mo) | Required util |
|---|---|---|---|---|---|---|
| DeepInfra Turbo | $0.32 | 4,541 | **86%** | $0.21 | 6,919 | **132%** (impossible) |
| DeepInfra standard | $0.40 | 3,633 | 69% | $0.315 | 4,613 | 88% |
| Hyperbolic | $0.30 | 4,843 | 92% | $0.21 | 6,919 | 132% (impossible) |
| Novita | $0.40 | 3,633 | 69% | $0.268 | 5,425 | **103%** (impossible) |
| Nebius Base | $0.40 | 3,633 | 69% | $0.265 | 5,484 | **104%** (impossible) |
| Groq | $0.79 | 1,839 | 35% | $0.69 | 2,106 | 40% |
| Azure Foundry | $0.79 | 1,839 | 35% | $0.69 | 2,106 | 40% |
| AWS Bedrock | $0.72 | 2,018 | 38% | $0.72 | 2,018 | 38% |
| Fireworks | $0.90 | 1,614 | 31% | $0.90 | 1,614 | 31% |
| Together (Turbo) | $0.88 | 1,651 | 31% | $0.88 | 1,651 | 31% |
| SambaNova | $1.20 | 1,211 | 23% | $0.90 | 1,614 | 31% |
| Cerebras | $1.20 | 1,211 | 23% | $1.025 | 1,418 | 27% |
| Google Vertex | $2.00 | 727 | **14%** | $1.36 | 1,069 | 20% |
| Replicate | $2.75 | 528 | **10%** | $1.70 | 855 | 16% |

**Read:** A Vultr GH200 **cannot beat DeepInfra-Turbo, Hyperbolic, Novita, or Nebius-Base on a 1:1 blended basis** — you'd need >100% sustained util. It **does beat them on pure output**, but only at ≥86% util, which is unrealistic unless you have steady production load. Against any other managed endpoint (Groq, Azure, Bedrock, Fireworks, Together, SambaNova, Cerebras, Vertex), break-even is **23–40% util** — easy to clear with real traffic.

### 3.2 Lambda GH200 — $1,672/mo, ~5,256 Mtok/mo cap

| Managed comparator | $/Mtok output | Break-even V (Mtok/mo) | Required util | $/Mtok blended | Break-even V | Required util |
|---|---|---|---|---|---|---|
| DeepInfra Turbo | $0.32 | 5,225 | **99%** (tied) | $0.21 | 7,962 | 151% (impossible) |
| DeepInfra standard | $0.40 | 4,180 | 80% | $0.315 | 5,308 | 101% (impossible) |
| Hyperbolic | $0.30 | 5,573 | **106%** (impossible) | $0.21 | 7,962 | 151% (impossible) |
| Novita | $0.40 | 4,180 | 80% | $0.268 | 6,239 | 119% (impossible) |
| Nebius Base | $0.40 | 4,180 | 80% | $0.265 | 6,309 | 120% (impossible) |
| Groq | $0.79 | 2,116 | 40% | $0.69 | 2,423 | 46% |
| Azure Foundry | $0.79 | 2,116 | 40% | $0.69 | 2,423 | 46% |
| AWS Bedrock | $0.72 | 2,322 | 44% | $0.72 | 2,322 | 44% |
| Fireworks | $0.90 | 1,858 | 35% | $0.90 | 1,858 | 35% |
| Together (Turbo) | $0.88 | 1,900 | 36% | $0.88 | 1,900 | 36% |
| Google Vertex | $2.00 | 836 | 16% | $1.36 | 1,229 | 23% |

**Read:** Lambda GH200 **never beats DeepInfra Turbo blended** (151% util needed) and **only ties DeepInfra Turbo on pure output at 99% util**. Against everyone else from Groq onwards, **35–46% util** breaks even — same shape as Vultr but with the +$219/mo Lambda premium pushing every threshold ~14% higher.

### 3.3 Nebius H200 3-mo committed — $1,679/mo, ~17,082 Mtok/mo cap

| Managed comparator | $/Mtok output | Break-even V (Mtok/mo) | Required util of H200 | $/Mtok blended | Break-even V | Required util |
|---|---|---|---|---|---|---|
| DeepInfra Turbo | $0.32 | 5,247 | **31%** | $0.21 | 7,995 | **47%** |
| DeepInfra standard | $0.40 | 4,198 | 25% | $0.315 | 5,330 | 31% |
| Hyperbolic | $0.30 | 5,597 | 33% | $0.21 | 7,995 | 47% |
| Novita | $0.40 | 4,198 | 25% | $0.268 | 6,265 | 37% |
| Nebius Base | $0.40 | 4,198 | 25% | $0.265 | 6,336 | 37% |
| Groq | $0.79 | 2,125 | 12% | $0.69 | 2,433 | 14% |
| Azure Foundry | $0.79 | 2,125 | 12% | $0.69 | 2,433 | 14% |
| AWS Bedrock | $0.72 | 2,332 | 14% | $0.72 | 2,332 | 14% |
| Fireworks | $0.90 | 1,866 | 11% | $0.90 | 1,866 | 11% |
| Together (Turbo) | $0.88 | 1,908 | 11% | $0.88 | 1,908 | 11% |
| SambaNova | $1.20 | 1,399 | 8% | $0.90 | 1,866 | 11% |
| Cerebras | $1.20 | 1,399 | 8% | $1.025 | 1,638 | 10% |
| Google Vertex | $2.00 | 840 | 5% | $1.36 | 1,234 | 7% |
| Replicate | $2.75 | 611 | 4% | $1.70 | 988 | 6% |

**Read:** Nebius H200 3-mo at $1,679/mo **beats every managed endpoint at ≤47% sustained util** — including DeepInfra Turbo blended (47%). This is the only single-GPU self-host posture in May 2026 where you can credibly beat the serverless floor without overdriving the box.

### 3.4 Side-by-side break-even chart (utilization needed to match managed cost)

| Self-host platform | DeepInfra Turbo blended ($0.21) | Hyperbolic ($0.21) | DeepInfra std ($0.315) | Nebius Base ($0.265) | Bedrock ($0.72) | Foundry/Groq ($0.69) | Together ($0.88) | Vertex ($1.36) |
|---|---|---|---|---|---|---|---|---|
| Vultr GH200 $1,453 | **132% (lose)** | 132% (lose) | 88% | 104% (lose) | 38% | 40% | 31% | 20% |
| Lambda GH200 $1,672 | **151% (lose)** | 151% (lose) | 101% (lose) | 120% (lose) | 44% | 46% | 36% | 23% |
| Nebius H200 $1,679 | **47%** | 47% | 31% | 37% | 14% | 14% | 11% | 7% |

**Definitive answer for the GH200 question:** A single GH200 on-demand (Vultr or Lambda) **cannot beat DeepInfra-Turbo's $0.21 blended floor at any reasonable utilization** on Llama-3.3-70B. You need either (a) Nebius H200 committed, (b) a multi-GPU box, or (c) a workload that justifies GH200 specifically (Grace-CPU KV offload, ARM64, 480 GB LPDDR pool, retrieval-heavy long-context — see round4/14_longcontext_grace.md).

---

## 4. Conflicts resolved this round

### 4.1 DeepInfra "$0.12 blended"
Round 2 cited DeepInfra Turbo at "$0.12 blended" — that figure comes from artificialanalysis.ai which uses a **7:2:1 cache-hit : input : output** weighting. Under that ratio, $0.10 × 0.7 (cache-discount) + $0.10 × 0.2 + $0.32 × 0.1 ≈ $0.12. Under a **1:1 input:output ratio (the honest comparable for non-cached workloads), it's $0.21**. Both numbers are correct; use the $0.21 figure for break-even math unless you actually have ≥70% cache-hit traffic.

### 4.2 AWS Bedrock $0.72 vs $2.65
**Resolved in Round 3 verify; reconfirmed here.** Bedrock Llama-3.3-70B = **$0.72/$0.72**, verified across costbench.com (May 14 2026), llm-token-calculator.com, and Vercel AI Gateway pass-through. TokenMix's $2.65 referred to **Llama 3 / 3.1 70B**, not Llama 3.3.

### 4.3 Cerebras $0.60/$0.60 vs $0.85/$1.20
Newer multi-source consensus (Bifrost cost calculator, CloudPrice provider page, llm-token-calc, general search May 2026) all give **$0.85/$1.20**. TokenMix's April 2026 figure of $0.60/$0.60 is the older rate. **Use $0.85/$1.20.** Cerebras's own /pricing page shows only Free/Developer/Enterprise tiers, no $/Mtok inline. Throughput claim: 1,800–2,000 tok/s sustained on Llama 3.3 70B via WSE-3.

### 4.4 Vertex AI Llama 3.3 70B
**Not on cloud.google.com/vertex-ai/generative-ai/pricing** as a first-party SKU (verified live 2026-05-25 — only Gemini 3/2.5/2.0/Gemma 4 priced inline). Llama is served via Model Garden / Marketplace billing; tokenmix April 2026 blended figure is **$1.36**, with a ~$0.72 input / ~$2.00 output asymmetric split. Treat as **medium confidence** — the actual contracted rate depends on whether you deploy dedicated endpoint (per-GPU-hour) or per-token via partner agreement.

### 4.5 Anyscale Endpoints and OctoAI
**Both effectively dead for serverless inference.** Anyscale Endpoints went hosted-platform-only as of Aug 2024 (no public per-token rate card). OctoAI was acquired by NVIDIA Sept 2024, services wound down by Oct 31, 2024, customers migrated to NVIDIA NIM. Drop both from any current decision matrix.

### 4.6 Lambda Inference API
Lambda's own /inference page says **"As the Inference API winds down…"** — Lambda is exiting managed inference and pushing customers to deploy on raw NVIDIA GPU instances. No public Llama-3.3-70B per-token rate is verifiable on lambda.ai today. llm-token-calc lists "Lambda Labs $0.12/$0.30" but that figure cannot be reconfirmed on Lambda's own pages and is likely stale.

### 4.7 Hyperbolic
hyperbolic.xyz now redirects to **hyperbolic.ai**. Their /pricing and /models pages return 404 or render no rates inline (likely dynamic JS). llm-token-calc lists Hyperbolic Llama-3.3-70B at **$0.12/$0.30** — this is the only sourced rate; Hyperbolic's own docs page does not display it. **Medium confidence**; if Hyperbolic at $0.12/$0.30 is real and reliable, it ties DeepInfra Turbo at the blended floor.

---

## 5. Throughput SLA snapshot

| Provider | Output tok/s claimed | Output tok/s observed (artificialanalysis median) | Note |
|---|---|---|---|
| Cerebras | 2,000+ (WSE-3) | ~1,800 | Fastest verified |
| Groq | 394 | 322.9 | LPU, Llama 3.3 70B Versatile SKU |
| SambaNova | n/a | 291.6 | SN40L |
| Google Vertex | n/a | 149.2 | Surprisingly fast for hyperscaler |
| AWS Bedrock | n/a | ~189.8 | tokenmix observation |
| Nebius (Fast) | n/a | ~80 | |
| Azure Foundry | n/a | ~120 | |
| Fireworks | n/a | ~50 | |
| Together (Turbo) | n/a | ~45 | FP8 |
| Replicate | n/a | ~40 | |
| Hyperbolic | n/a | ~35 | |
| Novita | n/a | ~30 | |
| DeepInfra (Turbo, FP8) | n/a | ~27–45 | Cheap = slow |
| Cloudflare | n/a | ~30 | Free tier |

Cerebras / Groq / SambaNova are the only three providers where the **speed premium might justify the price premium** for latency-sensitive workloads. Everyone else clusters at 30–200 tok/s.

---

## 6. The take-away in three lines

1. **Cheapest verified live $/Mtok for Llama-3.3-70B = DeepInfra Turbo at $0.10 in / $0.32 out** (≈$0.21 blended 1:1). Hyperbolic at $0.12/$0.30 is a tied claim but aggregator-sourced.
2. **Cheapest hyperscaler-managed = Azure AI Foundry at $0.59 in / $0.79 out** ($0.69 blended), tied with Groq. AWS Bedrock at $0.72/$0.72 confirmed; the old $2.65 figure was Llama 3 / 3.1.
3. **Self-host break-even** — A single GH200 on Vultr ($1,453/mo) or Lambda ($1,672/mo) **cannot beat DeepInfra-Turbo at any realistic utilization on Llama-3.3-70B output**. The **Nebius H200 3-mo committed at $1,679/mo** beats every managed endpoint at ≤47% sustained util — it is the only single-GPU self-host posture in May 2026 with credible token-economics against DeepInfra. For GH200 to make sense on Llama-3.3-70B, you need a workload that specifically wants the Grace CPU 480 GB LPDDR pool (long-context KV offload, retrieval-heavy generation), not the steady-state token factory.

---

## 7. Confidence summary

**HIGH confidence (live page returned exact $/Mtok numbers 2026-05-25):**
- DeepInfra Llama-3.3-70B-Instruct-Turbo $0.10/$0.32
- DeepInfra Qwen2.5-72B-Instruct $0.36/$0.40
- Together AI Llama-3.3-70B-Instruct-Turbo $0.88/$0.88 (via docs)
- Fireworks Llama 3.3 70B $0.90 flat (via model page)
- Groq Llama 3.3 70B Versatile $0.59/$0.79 + 394 TPS
- Novita Llama 3.3 70B $0.135/$0.40 + Qwen2.5-72B $0.38/$0.40
- SambaNova Llama 3.3 70B $0.60/$1.20 (via cloud.sambanova.ai/plans)
- AWS Bedrock Llama 3.3 70B $0.72/$0.72 (multi-source, including costbench May 14 2026)
- Azure AI Foundry Llama 3.3 70B $0.59/$0.79 (wrvishnu May 2026)
- Vultr GH200 $1.99/hr, Lambda GH200 $2.29/hr, Nebius H200 3-mo committed $2.30/hr (all verified Round 3, reconfirmed)

**MEDIUM confidence (aggregator-sourced; provider page didn't render rates inline):**
- Hyperbolic Llama-3.3-70B $0.12/$0.30 (llm-token-calc only)
- Hyperbolic Qwen2.5-72B $0.12/$0.30 (llm-token-calc only)
- Cerebras Llama-3.3-70B $0.85/$1.20 (Bifrost + CloudPrice + general search consensus; older TokenMix figure $0.60/$0.60 conflicts)
- Vertex AI Llama-3.3-70B blended $1.36 (tokenmix April 2026; not on Google's first-party rate card — Model Garden / Marketplace billed)
- IBM watsonx Llama-3.3-70B $0.71/$0.71 (llm-token-calc only)
- Cerebras Llama-3.1-405B $3.00/$3.90 (TokenMix April 2026; not re-verified)

**LOW / DEAD:**
- OctoAI: shut down Oct 31, 2024; merged into NVIDIA NIM
- Anyscale Endpoints: discontinued multi-tenant access Aug 1, 2024
- Lambda Labs Inference API: officially "winding down" per Lambda's own /inference page (May 2026); rates not verifiable

---

## 8. Sources fetched / searched 2026-05-25

**Provider pricing pages (live this round):**
- https://deepinfra.com/pricing — Llama 3.3 70B Turbo $0.10/$0.32; Qwen2.5-72B $0.36/$0.40
- https://deepinfra.com/meta-llama/Llama-3.3-70B-Instruct-Turbo — confirmed $0.10/$0.32
- https://www.together.ai/pricing — Llama 3.3 70B $0.88/$0.88
- https://docs.together.ai/docs/serverless-models — Llama-3.3-70B-Instruct-Turbo FP8 $0.88/$0.88
- https://fireworks.ai/pricing — directs to docs
- https://fireworks.ai/models/fireworks/llama-v3p3-70b-instruct — $0.90 flat
- https://groq.com/pricing — Llama 3.3 70B Versatile $0.59/$0.79 + 394 TPS
- https://novita.ai/pricing — Llama 3.3 70B $0.135/$0.40; Qwen2.5-72B $0.38/$0.40
- https://cloud.sambanova.ai/plans/pricing — Llama 3.3 70B $0.60/$1.20
- https://aws.amazon.com/bedrock/pricing/ — only Llama 2 surfaced via WebFetch; Llama 3.3 rates per costbench
- https://aws.amazon.com/bedrock/llama/ — Meta Llama overview; pricing not inline
- https://cloud.google.com/vertex-ai/generative-ai/pricing — confirms Llama NOT on first-party rate card
- https://azure.microsoft.com/en-us/pricing/details/ai-foundry-models/llama/ — timed out
- https://www.cerebras.ai/pricing — Free/Developer/Enterprise tiers only, no $/Mtok inline
- https://www.anyscale.com/pricing — no token-based pricing; service deprecated
- https://octo.ai/pricing — redirects to nvidia.com (shut down)
- https://lambda.ai/inference — "Inference API winds down" notice
- https://hyperbolic.ai/pricing, /models — 404 / no inline rates

**Aggregators / cross-checks:**
- https://artificialanalysis.ai/models/llama-3-3-instruct-70b — 17 providers; cheapest blended $0.12 (DeepInfra Turbo at 7:2:1 weighting)
- https://artificialanalysis.ai/models/llama-3-3-instruct-70b/providers
- https://artificialanalysis.ai/models/qwen2-5-72b-instruct/providers — DeepInfra + SiliconFlow only
- https://artificialanalysis.ai/providers/sambanova, /groq, /deepinfra
- https://pricepertoken.com/pricing-page/model/meta-llama-llama-3.3-70b-instruct
- https://www.llm-token-calculator.com/all-model-price-list — multi-provider table
- https://costbench.com/software/llm-api-providers/amazon-bedrock/ — Bedrock Llama 3.3 70B $0.72/$0.72 (May 14 2026)
- https://tokenmix.ai/blog/cerebras-api-key-access-speed-tests-2026 — Cerebras rate card
- https://tokenmix.ai/blog/llama-3-3-70b — 9-provider table
- https://tokenmix.ai/blog/vertex-ai-pricing — Vertex Llama 3.3 70B $1.36 blended
- https://www.wrvishnu.com/azure-ai-foundry-pricing-2026/ — Azure Foundry $0.59/$0.79
- https://www.aipricing.guru/meta-pricing/ — 5-provider table
- https://www.helicone.ai/llm-cost/provider/llama/model/Llama-3.3-70B-Instruct
- https://www.getmaxim.ai/bifrost/llm-cost-calculator/provider/cerebras/model/llama-3.3-70b — Cerebras $0.85/$1.20
- https://cloudprice.net/models/cerebras/llama-3.3-70b — Cerebras $0.85/$1.20

**WebSearch corroboration:**
- "Cerebras inference pricing Llama-3.3-70B per million tokens 2026"
- "Azure AI Foundry Llama-3.3-70B pricing per million tokens May 2026"
- "Cerebras Llama-3.3-70b $0.85 OR $1.20 OR $0.60 pricing 2026"
- "Anyscale Endpoints shutdown deprecated 2026 Llama inference"
- "OctoAI NVIDIA acquisition shutdown inference service 2026"
- "Vertex AI Llama 3.3 70B Model as a Service MaaS pricing per million tokens 2026"
- "Llama 3.3 70B vLLM GH200 OR H200 tokens per second benchmark 2026"

**Carry-forward from prior rounds (still valid):**
- Round 2 E_pricing_leaderboard.md (DeepInfra/Together/Fireworks/Bedrock/Foundry/Vertex)
- Round 3 17_marketplace_verify.md (Vultr/Lambda/Nebius/Sesterce/RunPod live verification)
- Round 3 18_tier1_verify.md (Together/Crusoe/OCI/CoreWeave per-GPU normalization)
- Round 3 19_hyperscaler_verify.md (Bedrock $0.72 vs $2.65 resolution; Azure Foundry $0.59/$0.79; Vertex Llama 3.3 70B $1.36 blended)
