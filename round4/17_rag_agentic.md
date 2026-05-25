# RAG / Agentic / Tool-Use Workloads on GH200 — Round 4 Deep Dive

**Agent**: #17 / Round 4
**Date**: 2026-05-25
**Scope**: RadixAttention hit rates in RAG/agentic traffic, structured output cost (XGrammar / Outlines / LLGuidance), Llama-3.3-70B tool calling, embeddings co-location, multi-LoRA serving, long-prefix system prompts on a single GH200.
**Constraint carried forward**: FP8 broken on this GH200/Wan stack — recipes default to **bf16**. Quoted FP8 numbers are flagged with the source's stack.

---

## 0. Executive summary

| Question | Short answer |
|---|---|
| 1. RadixAttention hit rates on RAG/agentic | **75–95% on prefix-heavy traffic** (multi-turn chat, RAG, ReAct/CrewAI). Production-measured 50–74% on mixed Chatbot-Arena traffic. Agent frameworks with stable system prompt + tool schemas (LangChain, LlamaIndex, CrewAI) typically land 60–80% in practice; Claude Code-style agent teams report 85–97% per worker / 97.2% aggregate. |
| 2. Structured output overhead | **XGrammar ≤ 40 μs/token** mask check; near-zero steady-state overhead with cached FSM. **SGLang ≈ 3× vLLM** on structured workloads (SGLang overlaps mask gen with GPU step; vLLM v0.10 stalls > batch 8). **Jump-forward** in SGLang skips deterministic spans entirely — pure win on fixed JSON keys. |
| 3. Llama-3.3-70B + tool grammar on 1× GH200 | **Just barely fits in bf16 on 144 GB SKU; doesn't fit on 96 GB SKU.** Use **SGLang** with `--tool-call-parser llama3 --grammar-backend xgrammar`, `tool_choice="required"`, schema-constrained `function.parameters`. On 96 GB SKU run **W8A8-INT8** (or AWQ-INT4) instead — INT8 path is clean on Hopper; **avoid FP8**. |
| 4. Embeddings co-located on GH200 | **Yes — strongly recommended**. NVIDIA RAG blueprint already does this; co-location cuts p99 1.8 s → 190 ms (no network hop). GH200 vs A100: **2.7× embed gen, 2.9× index build, 3.3× vector search**. Recommended embedder: BGE-M3 / Qwen3-Embedding-0.6B/4B. Engine: TEI (aarch64 supported as of Apr-2026, PRs #827/#840), or just use SGLang's `--is-embedding` flag on the same server for one less moving part. |
| 5. Multi-LoRA serving | Both engines support hundreds of adapters. **vLLM**: `--enable-lora --max-loras N --max-lora-rank R --max-cpu-loras M`. **SGLang**: `--lora-paths ... --max-loras-per-batch N --max-lora-rank R --lora-backend csgmv` (csgmv 20–80% faster than triton at high concurrency). Expect **15–50% steady-state throughput penalty** vs base model; padding rank to max hurts. SGLang's S-LoRA/Punica path scales to thousands of adapters with small overhead. |
| 6. Long-prefix system prompts (10k–50k tok) | Combined prefix cache + chunked prefill: **TTFT 4.3 s → 0.6 s** on 10k-token prompt repeat (vLLM, llm-d). **Known gotcha**: when chunked prefill is enabled, **only the first chunk uses prefix cache** in vLLM V1 — re-tune `max_num_batched_tokens` so the full cached prefix lands in one prefill. SGLang RadixAttention sidesteps this; dynamic chunked prefill (PP) cuts 1M-token TTFT 81%. |

**Single most actionable insight**: A single 144 GB GH200 running SGLang bf16 with Llama-3.3-70B + `--tool-call-parser llama3` + XGrammar + co-located `Qwen3-Embedding-0.6B` (`--is-embedding` on a second `launch_server` instance) is the **simplest production RAG/agentic stack** you can stand up. Two processes, one chip, prefix cache + jump-forward in front, embeddings have zero network latency.

---

## 1. RadixAttention hit rates on RAG and agentic traffic

### 1.1 Published / anecdotal numbers (2024–2026)

| Workload class | Hit rate | Source |
|---|---|---|
| LLaVA-Next-34B (multimodal mixed) | **52.4%** | LMSYS 2024-01-17 blog (first publication) |
| Vicuna-33B chat | **74.1%** (→ 1.7× lower mean TTFT) | LMSYS 2024-01-17 blog |
| Chatbot Arena production (sustained) | **> 50%** | LMSYS production deployment |
| Multi-turn chat | **75–90%** | Spheron / Particula / Runpod 2026 |
| Few-shot prompting | **85–95%** | LMSYS + multiple 2026 guides |
| Code analysis / RAG | **60–80%** | Spheron 2026 |
| 60%+ prefix-overlap workloads (mixed) | **75–95%** | Spheron 2026 |
| Voice cloning (reference audio KV reuse) | **avg 86.4%** | Spheron 2026 |
| LangChain / LlamaIndex agent frameworks | **40–70%** typical, often > 60% | n1n.ai, Particula 2026 |
| Custom agent with consistent system prompt | **20–40% compute reduction** | Particula 2026 |
| Claude-Code single agent | **85–97% per worker** | NVIDIA Dynamo agentic blog 2026 |
| Claude-Code 4-Opus team aggregate | **97.2% aggregate** (11.7× read/write ratio) | NVIDIA Dynamo agentic blog 2026 |
| NeMo Agent Toolkit custom routing | **4× lower p50 TTFT, 1.5× higher p50 tok/s** | NVIDIA Dynamo |
| Unique-prompt workloads | **0%** (no benefit) | Spheron 2026 |
| TVCACHE (tool-value cache, paper) | **up to 70%** | arXiv 2602.10986 |

### 1.2 Why agentic traffic hits hard

The pattern that makes RadixAttention shine:

```
[stable system prompt + tool schemas + few-shot ex.] | [retrieved docs] | [user msg]
                  <— 4k–30k tokens, identical —>         <— variable —>     <— variable —>
                                  HOT PREFIX
```

Agent frameworks (LangChain, LlamaIndex, CrewAI, AutoGen, ReAct) emit a fixed system block and tool catalog on **every** step in a session and across sessions for a tenant. RadixAttention auto-detects the shared prefix and reuses the KV. Hit rate per request ≈ length(shared prefix) / length(total prompt). For a 16k system + 2k user = **~89% theoretical hit** — and the radix tree's LRU policy keeps the prefix resident across thousands of concurrent sessions.

### 1.3 GH200 angle

**Hit rate is software, ports 1:1 from H100 to GH200** — it depends on traffic shape, not memory bandwidth. What changes on GH200 is the *value* of each hit:

- HBM3e ≈ 4.8 TB/s vs H100 SXM5 3.35 TB/s = **1.43×** — each cache hit translates to ~1.43× more decode tok/s, until you become compute-bound.
- HiCache L2 (host LPDDR over NVLink-C2C) is ~297 GB/s D2H / ~375 GB/s H2D — **5–6× cheaper** than PCIe-attached H100 host offload (~63 GB/s). So you can *extend* the working set into 100s of GB of LPDDR cheaply, getting hit rates closer to 95% even on heterogeneous tenants.

Practical recipe on GH200 144 GB:

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --enable-hierarchical-cache \
  --hicache-ratio 1.5 \
  --chunked-prefill-size 8192 \
  --tool-call-parser llama3 \
  --grammar-backend xgrammar \
  --max-running-requests 256
```

`--hicache-ratio 1.5` means total cache budget = 1.5× the GPU-only budget (the extra 0.5× lives in LPDDR).

---

## 2. Structured output — SGLang jump-forward vs vLLM xgrammar / outlines

### 2.1 The three production grammar backends

| Backend | Constraints | Notes |
|---|---|---|
| **XGrammar** (default in SGLang & vLLM since early 2026) | JSON Schema, regex, EBNF | Bitwise-mask FSM. **≤ 40 μs/token** overhead. XGrammar-2 (2026-05-04) is **80× faster compilation**. |
| **Outlines** | JSON Schema, regex | Faster *cold start* on simple schemas; slower per-token on complex ones. Tool-choice support not full on SGLang. |
| **LLGuidance** | JSON Schema, regex, EBNF | **Beats XGrammar on highly-dynamic schemas** (changes per request) — wins on Github_easy-style benchmarks. |

Switch backends in SGLang with `--grammar-backend {xgrammar,outlines,llguidance}` (default xgrammar). vLLM uses `--guided-decoding-backend`.

### 2.2 Throughput cost — concrete data

**SqueezeBits benchmark, Qwen3-8B & Qwen3-32B (TP=2) on H100 80GB, vLLM 0.10.0 vs SGLang 0.5.0rc0** (the canonical 2026 comparison):

- **Static / repetitive schema (book-info task)**: XGrammar > LLGuidance on **both** engines. SGLang ≈ 3× vLLM throughput. vLLM shows a **significant throughput drop at batch ≥ 8** with guided decoding; SGLang's overlapped mask gen mitigates this.
- **Dynamic schema (Github_easy)**: LLGuidance > XGrammar on **both** engines. Both engines stay closer to baseline because schema reuse doesn't help.
- **Steady-state overhead with cached FSM**: near-zero TPOT impact. Schema compile cost is 50–200 ms on first request; cached afterward (with XGrammar-2 the compile is now ~ms).

### 2.3 Jump-forward

SGLang's **jump-forward decoding** detects spans of the grammar where only one continuation is valid — e.g., the fixed `"name":` keys of a JSON object — and **emits them without running the model**. Numbers:

- For a JSON object with N fixed keys and M variable values, you skip roughly the key-length tokens entirely.
- On a `{"function":{"name":"...", "arguments":{...}}}` tool-call envelope, jump-forward typically eliminates **30–60%** of fixed-template tokens. Direct latency saving, no quality cost.

vLLM does not have an equivalent. TRT-LLM has a related "speculative decoding for guided generation" path but it is not as plug-and-play.

### 2.4 Practical recommendation (GH200, bf16)

```bash
# SGLang — best default for tool calling
--grammar-backend xgrammar                  # default, keep
# per-request:
"sampling_params": {
  "json_schema": "<schema>",                # or "regex": "...", or "ebnf": "..."
  "temperature": 0.0
}
# OR for tool calls:
tool_choice="required"                       # forces a tool, with parser + xgrammar
```

If your schemas change every request (LLM-of-the-day generates tool defs), benchmark with `--grammar-backend llguidance`. Otherwise stay on XGrammar.

---

## 3. Tool calling — Llama-3.3-70B on a single GH200

### 3.1 Memory math first

| Model | bf16 weight | KV per 1k tok @ batch 1 | Fits on 96 GB? | Fits on 144 GB? |
|---|---|---|---|---|
| Llama-3.1-8B | 16 GB | 0.13 GB | ✅ plenty of headroom | ✅ trivial |
| Llama-3.3-70B (bf16) | ~140 GB | 0.31 GB | **NO** | **YES, tight** |
| Llama-3.3-70B INT8 W8A8 | ~70 GB | 0.31 GB | ✅ tight | ✅ |
| Llama-3.3-70B AWQ INT4 | ~35 GB | 0.31 GB | ✅ plenty | ✅ |

**Reality check**: on a **96 GB GH200** you cannot run Llama-3.3-70B bf16. Use W8A8 INT8 (`--quantization w8a8_int8`) or AWQ INT4 (pre-quantized model). Do **not** use FP8 on this stack (known broken; see Round 1 §6 / Round 3).

On a **144 GB GH200** bf16 fits, but `--mem-fraction-static 0.92` and a constrained `--chunked-prefill-size 4096` are advisable to leave headroom for the radix tree + activations.

### 3.2 SGLang recipe (the recommended path)

```bash
# Single 144 GB GH200, bf16
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --mem-fraction-static 0.92 \
  --tool-call-parser llama3 \
  --grammar-backend xgrammar \
  --enable-hierarchical-cache \
  --hicache-ratio 1.3 \
  --chunked-prefill-size 4096 \
  --max-running-requests 128 \
  --host 0.0.0.0 --port 30000
```

Client (OpenAI-compatible):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

tools = [{
  "type": "function",
  "function": {
    "name": "search_docs",
    "description": "Retrieve from the corpus",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "top_k":  {"type": "integer", "minimum": 1, "maximum": 20}
      },
      "required": ["query"]
    }
  }
}]

resp = client.chat.completions.create(
  model="meta-llama/Llama-3.3-70B-Instruct",
  messages=[{"role":"user","content":"Find papers on RadixAttention"}],
  tools=tools,
  tool_choice="required",          # forces a tool call; XGrammar enforces schema
  temperature=0.0
)
```

**Parser values for other model families** (use whichever model you actually deploy):

| Parser | Model family |
|---|---|
| `llama3` | Llama 3.1 / 3.2 / 3.3 (text-style tool calls) |
| `llama4` | Llama 4 |
| `pythonic` | Llama 3.2 / 3.3 / Llama-4 (Python-list output, **supports parallel tools natively**) |
| `qwen` | Qwen 2.5 / 3 (replaces deprecated `qwen25`) |
| `qwen3_coder` | Qwen3-Coder |
| `hermes` | Hermes-style `<tool_call>...</tool_call>` (also works for Qwen 2.5) |
| `mistral` | Mistral variants |
| `deepseekv3` / `deepseekv31` / `deepseekv32` | DeepSeek V3 / V3.1 / V3.2 |
| `kimi_k2` | Kimi-K2-Instruct |
| `glm` | GLM series |
| `gpt-oss` | GPT-OSS (filters out analysis channels) |
| `step3` | Step-3 |

### 3.3 vLLM recipe (equivalent)

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --chat-template examples/tool_chat_template_llama3.2_json.jinja \
  --guided-decoding-backend xgrammar
```

Known limitation per vLLM docs: **parallel tool calls are not supported for Llama-3** with the `llama3_json` parser. Llama may also "generate parameters in incorrect format, such as array serialized as string". For parallel tools, switch to a model that emits pythonic output (Llama 3.2/3.3 + SGLang `pythonic` parser is the easiest fix).

### 3.4 Tool calling + grammar caveats

- `tool_choice="auto"` does NOT enforce schema at decoding time — parser extracts from raw text post-hoc, may fail.
- `tool_choice="required"` + XGrammar **does** enforce schema. **Always prefer this for production** unless you specifically want the model to choose whether to call a tool.
- For function-routing (multiple tools), XGrammar can enforce the union schema `{type:"object", oneOf:[<tool0>,<tool1>,...]}`.

---

## 4. Embeddings on GH200 — co-locate or not?

### 4.1 The case for co-location

**Cited number, Spheron 2026**: collapsing embedder + vector store + LLM onto one GPU server cut **p99 1.8 s → 190 ms** (vs split with network hops). NVIDIA GH200 RAG blueprint already co-locates these on the same chip and reports vs A100:

- **2.7× embedding generation** (Sentence Transformer Paraphrase Multilingual MPNet, batch 1024, 85M vectors / 768 dim)
- **2.9× index build** (FAISS / RAFT IVF-PQ)
- **3.3× vector search time** (RAFT IVF-PQ, 10k queries over 85M vectors)
- **5.7× Llama-2-70B FP8 inference** (input 2048, output 128, batch 64)
- **End-to-end RAG: 200 sample queries in 0.6 s** (embed + search + LLM)
- Software: TensorRT-LLM + Triton + NeMo + RAFT (their stack; FP8 — not transferable to our bf16 stack).

(Source: NVIDIA developer blog, "Deploying RAG on GH200".)

GH200 has 96 GB or 144 GB HBM3e + 480 GB LPDDR5x reachable via NVLink-C2C at ~300–375 GB/s practical. That LPDDR is the ideal place to park a large vector index (FAISS-CPU style index) without leaving the chip. RAFT's GPU IVF-PQ keeps the hot index in HBM.

### 4.2 Engine choices for the embedder

| Engine | aarch64/GH200 status (May 2026) | Notes |
|---|---|---|
| **SGLang `--is-embedding`** | Native (same multi-arch Docker image as the LLM) | Supports E5-Mistral, GTE-Qwen2, **Qwen3-Embedding (0.6B/4B/8B)**, **BGE (bge-large-en-v1.5)**, GME, CLIP. BGE requires `--attention-backend triton` or `torch_native`. **Easiest path on GH200** because you reuse the same container, same wheel, same observability. |
| **HuggingFace TEI** | **aarch64 PRs #827/#840 landed late 2025/early 2026**; Docker images officially advertise Blackwell aarch64 (DGX Spark). Grace-Hopper not explicitly listed but the H100 sm90 image works on GH200 in practice. | Best raw embed throughput. Use `--dtype float16` on Hopper. **BGE-M3 ≈ 90k tok/s** on H100 PCIe, batch 512 (baseten ref) — extrapolated ~130k tok/s on GH200 by bandwidth ratio. |
| **NVIDIA NIM (Nemotron-Embed)** | Container-based, supports Hopper | Used by NVIDIA RAG blueprint; horizontal pod autoscaling integrated. Closed source, NGC license. |
| **vLLM embedding mode** | Supported (`--task embed`) | Underperforms TEI/BEI for pure embedding. Use only if you already run vLLM for the LLM and don't want a second engine. |

### 4.3 Throughput data (closest published, no first-party GH200)

| Engine | Model | GPU | Batch | Throughput |
|---|---|---|---|---|
| TEI fp16 | BGE-M3 | H100 PCIe | 512 | **~90,000 tok/s** |
| TEI fp16 | BGE-M3 | A100 80GB PCIe | 512 | ~60,000 tok/s |
| TEI fp16 | Qwen3-Embedding-0.6B | A100 80GB PCIe | 512 | ~55,000 tok/s |
| TEI fp16 | Qwen3-Embedding-0.6B | RTX 4090 | 512 | ~20,000 tok/s |
| TEI fp16 | BGE Reranker v2-m3 | A100 80GB PCIe | 64 | ~800 pairs/s |
| BEI (Baseten) | (proprietary) | **B200** | — | **3.3× vLLM, 3.6× TEI@H100** |
| Spheron production estimate | Stella_en_1.5B_v5 | H100 SXM | (token) | "tens of thousands of sentences/sec" |
| NVIDIA GH200 RAG blog | Paraphrase MPNet Base v2 | GH200 96GB | 1024 | 2.7× A100 (no abs tok/s) |

**GH200 single-instance projection** (BGE-M3 fp16): scale H100 by HBM-bandwidth ratio (1.43×) → **~130,000 tok/s** with TEI on a GH200 144 GB. Treat as ±25%. For RAG ingest at this rate, BGE-M3 takes < 1 hour per 500M tokens.

### 4.4 Latency targets (Spheron 2026 colocated stack)

| Stage | Target latency |
|---|---|
| Embedding (single query, batch 1) | **< 5 ms** |
| FAISS-GPU search over 1M vectors | **< 3 ms** |
| LLM TTFT (70B, short prompt, prefix-cache hit) | **< 80 ms** |
| Full single-hop RAG | **< 120 ms** |
| Two-hop agentic RAG (ReAct, 1 tool call) | **< 200 ms** |

On GH200 the embedder + vector index can share the LPDDR domain with the LLM's HiCache L2; no PCIe round-trip.

### 4.5 Recommended embedding co-location pattern on GH200 (single chip)

```bash
# Process A: LLM
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --mem-fraction-static 0.78 \              # leave room for the embedder
  --port 30000

# Process B: Embedder (same Docker image, second instance)
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-Embedding-0.6B \
  --is-embedding \
  --dtype bfloat16 \
  --mem-fraction-static 0.10 \
  --port 30001

# Process C (host-side, optional): FAISS / RAFT index in LPDDR via Grace
```

Total: ~115 GB HBM in use (70B + 1.5 GB embedder + KV/radix + activations) on the 144 GB SKU. Embedder can also run on **CPU** (Grace 72-core Neoverse V2) for very low embedding QPS, freeing all 144 GB for the LLM.

---

## 5. Multi-LoRA serving for per-tenant fine-tunes

### 5.1 vLLM

```bash
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --enable-lora \
  --max-loras 8 \
  --max-lora-rank 64 \
  --max-cpu-loras 64 \
  --lora-modules \
      tenant0=hf-user/tenant0-lora \
      tenant1=hf-user/tenant1-lora \
      tenant2='{"name":"tenant2","path":"s3://bucket/lora2","base_model_name":"meta-llama/Meta-Llama-3.1-8B-Instruct"}' \
  --enable-prefix-caching
```

Key flags:
- `--max-loras N` — max **concurrent** adapters in a single forward pass batch. Higher = more memory pressure (LoRA weights + LoRA-aware GEMM scratch).
- `--max-lora-rank R` — pad-to-max rank across all loaded adapters. **Set to your actual max** — don't pick 256 if all adapters are rank 16. Higher rank → wasted memory and slower GEMM.
- `--max-cpu-loras M` — CPU-side cache; loaded on demand via API or pre-loaded at startup.
- `--lora-target-modules` — optional, restrict LoRA application (e.g., `qkv_proj,o_proj`).
- `--fully-sharded-loras` — for TP > 1 with large ranks.

Dynamic loading: POST `/v1/load_lora_adapter` and `/v1/unload_lora_adapter`. Set `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`.

### 5.2 SGLang

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --enable-lora \
  --lora-paths \
      tenant0=hf-user/tenant0-lora \
      tenant1=hf-user/tenant1-lora \
      '{"lora_name":"tenant2","lora_path":"s3://bucket/lora2","pinned":true}' \
  --max-loras-per-batch 2 \
  --max-lora-rank 64 \
  --lora-target-modules all \
  --lora-backend csgmv \
  --enable-lora-overlap-loading \
  --max-loaded-loras 4 \
  --lora-eviction-policy lru
```

Key flags:
- `--max-loras-per-batch N` (default 8): max adapters per batch.
- `--lora-backend csgmv` (default; the Punica-derived path): **20–80% lower latency vs `triton` at high concurrency**.
- `--enable-lora-overlap-loading`: async H2D for adapter weights — useful if you swap adapters between batches.
- `--max-loaded-loras M`: must be ≤ `2 × max-loras-per-batch` when overlap is on.
- `--lora-eviction-policy {lru,fifo}`: governs CPU cache eviction.
- `pinned: true` in the JSON path keeps an adapter on GPU permanently (limit: `max-loras-per-batch - 1` to prevent starvation).

Dynamic: `/load_lora_adapter` and `/unload_lora_adapter` REST endpoints.

### 5.3 Throughput expectations (multi-LoRA penalty)

| Source | Setup | Penalty |
|---|---|---|
| vLLM A100 issue #10062 | Single LoRA, A100 40 GB | **Up to 50% throughput drop** vs base |
| SqueezeBits vLLM 0.6.3 vs TRT-LLM 0.14, A100 80 GB | Llama-3.1-8B, 16 LoRAs, rank 16–256 | vLLM 23.9–47.0% throughput degradation; TRT-LLM 14.1–18.6% (Q/V only) |
| vLLM 0.15.0 blog 2026-02-26 | GPT-OSS-20B, rank 32, 8 adapters in parallel | After fix: **+454% OTPS, -87% TTFT** vs earlier vLLM. Llama-3.3-70B and Qwen3-32B benefit similarly; **Qwen3-32B OTPS +99%** from kernel tuning. |
| S-LoRA paper | Thousands of LoRAs, single GPU | Up to **4× throughput vs naive vLLM**, 30× vs PEFT |
| Punica paper | 4 LoRA popularity skews | **12× vs SOTA** at the time |

**Rule of thumb**: with vLLM ≥ 0.15.0 or SGLang csgmv at low rank (≤ 32), expect **5–15% steady-state penalty**. At rank 128–256, expect **20–40%**. Padding all adapters to max rank is the biggest avoidable cost — use distinct `max-lora-rank` per deployment if you can.

### 5.4 GH200 multi-LoRA pattern

GH200's extra LPDDR is **ideal** for adapter caching: parking 100s of adapters in 480 GB of LPDDR and pulling them over NVLink-C2C is much faster than PCIe to host RAM. Set:

```bash
--max-cpu-loras 256      # vLLM
--max-loaded-loras 64    # SGLang (limited by 2× max-loras-per-batch with overlap)
```

Adapter swap-in latency over NVLink-C2C at 375 GB/s for a typical 25 MB LoRA: ~66 μs — invisible vs prefill.

---

## 6. Long-prefix system prompts (10k–50k tokens)

### 6.1 The combined recipe — prefix cache + chunked prefill

Both vLLM and SGLang now enable **chunked prefill** and **automatic prefix caching** by default. For long-prefix workloads (system prompt + tool catalog + few-shot of 10k–50k tokens), you want:

```bash
# vLLM
--enable-prefix-caching \
--enable-chunked-prefill \
--max-num-batched-tokens 16384 \      # higher = better TTFT for long prompts
--max-num-seqs 256

# SGLang (most of these are default in 0.5.12)
--chunked-prefill-size 8192 \         # default 8192; tune down if TTFT spikes
--enable-hierarchical-cache \         # extends radix to LPDDR
--hicache-ratio 1.5
```

### 6.2 Measured TTFT wins

| Source | Setup | Result |
|---|---|---|
| llm-d (Qwen3-32B, 8× H100 cluster, ~10k-tok prompts) | Single-instance | **TTFT 4.3 s → 0.6 s** with cache hit (**~78% reduction**) |
| llm-d (distributed scheduling, 16 H100, customer-context 6k + query 1.2k) | Precise scheduling vs random | **P90 TTFT 0.542 s**, **170× faster than random**; output **8,730 tok/s** (**25%** above approximate-scheduling, **2×** vs cache-blind) |
| Generic provider study (10k-tok system prompts, agentic) | Across OpenAI/Anthropic/Google | **41–80% API cost reduction, 13–31% TTFT improvement** |
| SGLang PP long-context blog (Qwen3-235B-A22B-FP8, 1M tok) | PP1 TP4 → PP8 TP4 | **TTFT 55.5 s → 10.5 s (81.1% reduction)** with dynamic chunked prefill |
| SGLang PP long-context blog (DeepSeek-V3.1, 1M tok) | PP1 TP8 → PP4 TP8 | **TTFT 48.5 s → 15.5 s (67.9% reduction)** |
| GH200 (raw worst-case calibration) | Qwen3.5-122B, 64k-tok cold prefill | **419 s TTFT** (no cache) — illustrates why prefix caching is non-negotiable |

### 6.3 The vLLM gotcha you must know

> When chunked prefill is enabled, a long prompt is split into multiple chunks, and **only the first chunk leverages prefix caching during execution**, which may result in suboptimal cache utilization. (vLLM forum / docs)

Mitigation in vLLM V1:
- Bump `--max-num-batched-tokens` so the full cached prefix lands in **one prefill chunk**. For a 20k-token system prompt, set `--max-num-batched-tokens 24576`.
- For prompts > 32k, the trade-off shifts: now you want chunking to interleave with decode for TTFT-on-other-requests, but pay the suboptimal-cache cost.
- vLLM bug #8223 documents the poor-TTFT case when both flags are on naively.

**SGLang does not have this limitation** — RadixAttention checks for prefix matches at *radix-tree node* granularity, independent of the prefill chunker. **For workloads that combine "very long system prompt" + "many concurrent users", SGLang is the safer choice.**

### 6.4 Long-prefix cost model on GH200

For a stable 30k-token system prompt shared across 100 tenants:

| Strategy | Prefill cost per request | TTFT @ batch 1 | Notes |
|---|---|---|---|
| No prefix cache | 30k × per-token prefill | ~2–4 s on 70B bf16 | Baseline |
| Prefix cache, hit | ~1 chunk (8k) miss + match | **~0.3–0.6 s** | First user pays full cost; subsequent ~free |
| Prefix cache + HiCache L2 | Same as above, but cache resists eviction over hours | **~0.3–0.6 s sustained** | 480 GB LPDDR keeps prefixes resident |
| Prefix cache + jump-forward (tool calling) | + ~30–60% saved tokens on JSON envelope | TTFT same; **lower TPOT** | Compounds with prefix cache |

Cost compounding: For Llama-3.3-70B bf16 on a 144 GB GH200 with HiCache, a 30k-token shared system prompt across 100 concurrent agent sessions gets your prefill bill **down by ~95–99%**, and tool-call envelopes lose ~40% of their token cost via jump-forward. **This is the regime where SGLang wins by multiples, not percentages, against vLLM.**

---

## 7. Confidence & gaps

### High confidence
- Hit-rate ranges (RAG 60–80%, multi-turn 75–90%, few-shot 85–95%) are stable across LMSYS + 5 independent 2026 production guides.
- XGrammar ≤ 40 μs/token overhead, SGLang ≈ 3× vLLM on structured workloads, XGrammar-2 80× faster compile.
- SGLang tool-call parser names and Llama-3.3-70B recipe (cookbook page is canonical).
- Llama-3.3-70B bf16 weight budget vs GH200 96/144 GB SKU.
- vLLM and SGLang multi-LoRA flag lists.
- vLLM chunked-prefill + prefix-cache "only first chunk caches" gotcha (docs + bug #8223).
- llm-d TTFT 4.3 s → 0.6 s, P90 0.542 s under precise scheduling.
- NVIDIA GH200 RAG blueprint speedups vs A100 (2.7×/2.9×/3.3×/5.7×).

### Medium confidence
- Multi-LoRA throughput penalty is 5–15% at low rank / 20–40% at high rank — based on the SqueezeBits A100 study + vLLM 0.15.0 blog; not directly measured on GH200.
- GH200 embedder throughput extrapolation (BGE-M3 ~130k tok/s) is bandwidth-scaled from H100, not measured.
- TEI aarch64/GH200 viability — PRs landed, official docs list "Blackwell aarch64 (DGX Spark)" explicitly but not GH200; in practice the Hopper image runs on GH200 via the multi-arch / amd64 emulation path. Best confirmed by user reports rather than docs.

### Gaps (still open)
- **No first-party SGLang-on-GH200 prefix-cache hit rate measurement.** All RadixAttention numbers are from H100/H800 deployments.
- **No published GH200 multi-LoRA benchmark** for either engine. Penalty inferred from A100/H100.
- **HiCache L2 on GH200 with long prefixes** — strong architectural fit (~300 GB/s L2 vs ~63 GB/s PCIe on H100), no measured number.
- **TEI on aarch64 + GH200** — PRs merged; no published throughput.
- **Jump-forward exact savings** on Llama-3.3 tool calls — qualitative ("30–60%"); needs a microbench on JSON-envelope token shapes.

### Watch list
- vLLM 0.18.x / 0.19.x release notes (Apr 2026) — likely further multi-LoRA kernel work after the Feb 2026 blog.
- SGLang v0.5.13+ — XGrammar-2 native integration may further reduce compile cost.
- HuggingFace TEI v1.9+ — track explicit Grace-Hopper hardware listing.

---

## 8. Sources

### Round-4-new (RAG / agentic / structured / LoRA / embeddings)

Cache hit rates & agentic:
- [LMSYS RadixAttention launch blog, 2024-01-17](https://www.lmsys.org/blog/2024-01-17-sglang/) — 52.4% / 74.1% production hit rates
- [Spheron SGLang production deployment guide 2026](https://www.spheron.network/blog/sglang-production-deployment-guide/) — 75–95% hit-rate range, voice cloning 86.4%
- [Particula SGLang vs vLLM 2026](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison) — agent framework hit rates
- [n1n.ai vLLM vs SGLang vs LMDeploy 2026](https://explore.n1n.ai/blog/vllm-vs-sglang-vs-lmdeploy-fastest-inference-2026-2026-03-05)
- [Runpod SGLang in production guide 2026](https://www.runpod.io/articles/guides/blog-sglang-production-llm-pipelines)
- [NVIDIA Dynamo agentic inference blog (Claude Code cache rates)](https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/) — 85–97% per-worker, 97.2% aggregate
- [arXiv 2602.10986 — TVCACHE stateful tool-value cache](https://arxiv.org/pdf/2602.10986) — 70% hit, 6.9× tool latency
- [arXiv 2507.07400 — KVFlow workflow-aware prefix caching](https://arxiv.org/pdf/2507.07400)
- [arXiv 2511.02230 — CacheTTL multi-turn agent scheduling](https://arxiv.org/html/2511.02230v4)
- [arXiv 2601.06007 — Don't Break the Cache, long-horizon agentic prompt caching](https://arxiv.org/pdf/2601.06007)

Structured output / grammar:
- [SqueezeBits guided decoding bench, vLLM 0.10 vs SGLang 0.5.0rc0 on H100](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang) — XGrammar/LLGuidance/static vs dynamic schema
- [SGLang structured outputs docs](https://docs.sglang.io/advanced_features/structured_outputs.html)
- [ChatForest SGLang structured generation review 2026](https://chatforest.com/reviews/sglang-structured-generation-llm-serving/) — 40 μs/token, 3× vLLM, XGrammar-2 80× compile speedup
- [vLLM structured outputs docs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [XGrammar paper, arXiv 2411.15100](https://arxiv.org/pdf/2411.15100)

Tool calling:
- [SGLang Llama-3.3-70B cookbook](https://docs.sglang.io/cookbook/autoregressive/Llama/Llama3.3-70B) — `--tool-call-parser llama3` recipe
- [SGLang tool parser docs](https://docs.sglang.io/advanced_features/tool_parser.html) — complete parser list (llama3/llama4/pythonic/qwen/hermes/mistral/deepseekv3*/kimi_k2/glm/gpt-oss/step3)
- [vLLM tool calling docs](https://docs.vllm.ai/en/latest/features/tool_calling/) — `--enable-auto-tool-choice --tool-call-parser llama3_json`
- [vLLM Llama tool parser API](https://docs.vllm.ai/en/latest/api/vllm/tool_parsers/llama_tool_parser/)

Multi-LoRA:
- [SGLang LoRA serving docs](https://sgl-project.github.io/advanced_features/lora.html) — full flag set, csgmv backend, overlap loading
- [vLLM LoRA adapters docs](https://docs.vllm.ai/en/stable/features/lora/)
- [vLLM blog: multi-LoRA on SageMaker / Bedrock 2026-02-26](https://vllm.ai/blog/2026-02-26-multi-lora) — vLLM 0.15.0: +454% OTPS, -87% TTFT on GPT-OSS-20B; Qwen3-32B +99% OTPS
- [SqueezeBits vLLM vs TRT-LLM multi-LoRA bench](https://blog.squeezebits.com/37065) — Llama-3.1-8B + 16 LoRAs on A100, rank scaling penalty
- [vLLM A100 LoRA throughput issue #10062](https://github.com/vllm-project/vllm/issues/10062) — up to 50% drop
- [S-LoRA paper, arXiv 2311.03285](https://arxiv.org/pdf/2311.03285)
- [Punica paper, arXiv 2310.18547 / MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/file/054de805fcceb78a201f5e9d53c85908-Paper-Conference.pdf)
- [arXiv 2510.23346 — Block-Diagonal LoRA for TP serving](https://arxiv.org/pdf/2510.23346)

Embeddings + co-location:
- [NVIDIA GH200 RAG blog](https://developer.nvidia.com/blog/deploying-retrieval-augmented-generation-applications-on-nvidia-gh200-delivers-accelerated-performance/) — 2.7×/2.9×/3.3×/5.7× vs A100; 200 queries / 0.6 s end-to-end
- [Spheron agentic RAG GPU infra 2026](https://www.spheron.network/blog/agentic-rag-gpu-infrastructure-guide/) — colocation latency targets, p99 1.8s→190ms
- [SGLang embedding models doc](https://sgl-project.github.io/supported_models/retrieval_ranking/embedding_models.html) — `--is-embedding`, BGE/Qwen3/E5/GTE/CLIP
- [HuggingFace TEI README](https://github.com/huggingface/text-embeddings-inference/blob/main/README.md)
- [TEI supported models / hardware](https://github.com/huggingface/text-embeddings-inference/blob/main/docs/source/en/supported_models.md)
- [TEI ARM64 issue #769 (PRs #827 / #840)](https://github.com/huggingface/text-embeddings-inference/issues/769)
- [Baseten high-performance embedding guide](https://www.baseten.co/resources/guide/high-performance-embedding-model-inference/) — BGE-M3 on H100 ~90k tok/s, BEI vs vLLM vs TEI
- [Spheron TEI on GPU cloud 2026](https://www.spheron.network/blog/self-host-embedding-reranker-tei-gpu-cloud/) — BGE-M3 / Qwen3-Embedding throughput tables
- [NVIDIA RAG Blueprint Kubernetes autoscaling](https://developer.nvidia.com/blog/enabling-horizontal-autoscaling-of-enterprise-rag-components-on-kubernetes/)

Long-prefix / chunked prefill / prefix cache:
- [llm-d KV-cache wins blog](https://llm-d.ai/blog/kvcache-wins-you-can-see) — 4.3 s → 0.6 s, P90 0.542 s, 8,730 tok/s
- [vLLM automatic prefix caching design](https://docs.vllm.ai/en/stable/design/prefix_caching/) — 100–200 ns/token (~6 ms for 50k)
- [vLLM forum — prefix cache + chunked prefill interaction](https://discuss.vllm.ai/t/should-vllm-consider-prefix-caching-when-chunked-prefill-is-enabled/903)
- [vLLM bug #8223 — poor TTFT combining both](https://github.com/vllm-project/vllm/issues/8223)
- [SGLang pipeline parallelism long context blog 2026-01-15](https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/) — Qwen3-235B/DeepSeek-V3.1 1M-tok TTFT 81%/68% reduction
- [SGLang chunked prefill issue #20018](https://github.com/sgl-project/sglang/issues/20018) — chunked_prefill_size semantics
- [Inside vLLM: anatomy of high-throughput LLM serving](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)

Round-1 / Round-3 reused (already cited there, listed here for traceability):
- LMSYS HiCache blog 2025-09-10
- Mooncake × SGLang HiCache design + benchmark
- Spheron vLLM-vs-TRT-vs-SGLang H100 bench 2026-03-23
- Verda GH200 data-movement blog Jan 2026 (NVLink-C2C 375/297 GB/s)
- arXiv 2408.11556v2 — GH200 data movement
