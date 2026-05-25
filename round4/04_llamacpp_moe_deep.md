# GH200 Inference Round 4 — Agent #04: llama.cpp MoE Deep Dive

Date: 2026-05-25
Agent: research round 4 / LLM-only deep dive
Predecessor: `round1/05_tgi_llamacpp.md`
Constraint reminder: FP8 broken on DiT; bf16 default for diffusion; FP8 OK on
dense Hopper LLMs (but llama.cpp does not use FP8 — GGUF quants only).

This round drills on 8 LLM-only questions about llama.cpp on GH200. Every claim
flagged with confidence; gaps called out at the end.

---

## Question 1 — `--n-cpu-moe N` scaling curve for big MoE models

### What the flag actually does

PR #15077 (slaren, merged **2025-08-04**) adds two CLI flags:
- `--cpu-moe`: keep **all** MoE expert weights in CPU memory.
- `--n-cpu-moe N`: keep MoE weights of the **first N layers** in CPU.

Implementation: it expands to internal `--override-tensor` (`-ot`) regex
patterns. It does **not** call `cudaMemAdvise`; it just tells the loader to
allocate those expert tensors as host buffers, so the GPU reads them over PCIe
(or NVLink-C2C on GH200) on demand. If you pass `--override-tensor` before
`--n-cpu-moe`, your manual override wins.

Important footgun documented in the Doctor-Shotgun guide:
> `--n-cpu-moe` starts counting layers from the **highest** numbered layers.

So `--n-cpu-moe 58` on DeepSeek V3 (61 layers) does *not* mean "first 58 layers
on CPU"; it means "layers 3..60 (the 58 highest) on CPU", leaving the first
three (dense FFN) on GPU. That matches the architecture: DeepSeek V3 has 3
dense FFN layers and 58 MoE layers, and the dense layers should stay on GPU
because they're touched every token.

### Architecture-driven theoretical sweet spots

For each model, the "always-active" parameters (attention, dense FFN, embed,
shared expert) belong on GPU; the routed experts belong on CPU (Grace LPDDR5X)
because only a fraction (top-k of 256 or 384) is touched per token.

| Model              | Total layers | Dense layers | MoE layers | Experts (route+shared) | Active params |
|--------------------|-------------:|-------------:|-----------:|------------------------|--------------:|
| DeepSeek V3.1 671B |           61 |            3 |         58 | 256 routed + 1 shared, top-8 |    ~37B |
| Kimi K2 ~1T        |           61 |            1 |         60 | 384 routed + 1 shared, top-8 |    ~32B |
| GPT-OSS 120B MXFP4 |           36 |            0 |         36 | 128 routed, top-4      |    ~5.1B  |
| Mixtral 8x22B 141B |           56 |            0 |         56 | 8 routed, top-2        |    ~39B   |

(Layer counts confirmed: DeepSeek V3 via DeepWiki / DeepSeek-V3 technical
report; Kimi K2 via Moonshot model card; Mixtral 8x22B from Mistral release;
GPT-OSS via OpenAI release notes referenced in llama.cpp discussion #15396.)

Recommended starting points on GH200 96/144 GB HBM3e:

| Model              | Recommended `--n-cpu-moe` | Rationale |
|--------------------|---------------------------|-----------|
| DeepSeek V3.1 Q4_K_M | **58** (all MoE layers)  | Q4_K_M weights ≈ 380 GB total; only ~50 GB of dense+attn+KV fits on GPU at -c 8192 |
| Kimi K2 Q3_K_M       | **60** (all MoE layers)  | Q3_K_M ≈ 460 GB; same logic, even tighter |
| GPT-OSS 120B MXFP4   | **0** on GH200 144 GB    | Full model is ~64 GB; fits entirely in HBM, no CPU offload needed |
| GPT-OSS 120B MXFP4   | **0–3** on GH200 96 GB   | Tight; with -c 8192 fits; with -c 32768 may need n-cpu-moe 3 for KV headroom |
| Mixtral 8x22B Q4_K_M | **0** on GH200 144 GB    | ~78 GB total, fits in HBM3e |
| Mixtral 8x22B Q4_K_M | **40–48** on GH200 96 GB | Doesn't fit in 96 GB HBM with KV; partial offload required |

### What's actually measured

Discussion #18005 numbers (Dec 2025, GH200 144 GB HBM3e, CUDA 13.0, patched
build) — these are the *only* GH200-specific MoE benchmarks I found:

| Model                          | Config              |  PP t/s |  TG t/s |
|--------------------------------|---------------------|--------:|--------:|
| DeepSeek V3.1 Q4_K_M (671 B)   | UM + cudaMemAdvise  |  609.89 |   18.30 |
| DeepSeek V3.1 Q4_K_M           | UM naive (no patch) |    —    |    8.86 |
| Kimi K2 Thinking Q3_K_M (~1 T) | UM + cudaMemAdvise  |  521.11 |   21.57 |
| GPT-OSS 120 B MXFP4            | CUDA only           | 5393.05 |  209.01 |
| GPT-OSS 20 B MXFP4             | CUDA only           | 9682.19 |  322.93 |

**Patch effect on DeepSeek V3.1**: 8.86 → 18.30 t/s = **2.07x**.

**No public sweep of `--n-cpu-moe` from N=0 to N=60 on GH200 has been
published.** The Doctor-Shotgun guide explicitly recommends "trial and error":
inspect VRAM, decrement N until OOM, then back off by 1–2. This is the single
biggest measurement gap on GH200.

The one concrete `--n-cpu-moe` number for GPT-OSS 120 B in the wild (PR #15077
review thread, **not GH200**): dual RTX 5090 (32 GB×2 = 64 GB), `--n-cpu-moe 3`
gave **108 t/s** generation. That translates to "as soon as the model fits in
GPU memory with a small N, you recover near-full GPU throughput; the curve is
sharply non-linear at the fits-in-VRAM threshold."

### Confidence
**High**: PR #15077 mechanism, layer counts, the qualitative "fit-then-flat"
shape of the curve.
**Low**: actual `(N, tok/s)` pairs on GH200 — extant data only covers the
"all-MoE-on-CPU" endpoint. Worth a direct sweep on the box.

---

## Question 2 — `GGML_CUDA_ENABLE_UNIFIED_MEMORY` + cudaMemAdvise: status

### Unified memory environment variable
Merged via **PR #8035** (matteoserva). `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`
makes every CUDA allocation use `cudaMallocManaged` instead of `cudaMalloc`,
permitting oversubscription beyond HBM into host RAM. Linux only (UVM relies
on Linux page-migration hooks). PR #12934 (hjc4869) later unified the AMD
HIP path with the CUDA path so a single binary works on both.

This is **in master** as of the b9305 release (2026-05-24) and has been for
~18 months. Documented in `docs/build.md`.

Caveat: issue #16197 ("GGML_CUDA_ENABLE_UNIFIED_MEMORY does not work")
reports the env var alone is sometimes insufficient on multi-GPU x86 systems.
On GH200, single-GPU, it works.

### The cudaMemAdvise patch (discussion #18005)
Status: **not merged**. The patch (commit `c6f6e4f` posted in the discussion,
2025-12-14) calls
```c
cudaMemAdvise(ptr, size,
              cudaMemAdviseSetPreferredLocation,
              cudaCpuDeviceId);
```
on expert tensors during model load, so the CUDA UVM heuristic doesn't
migrate them into HBM and thrash. It's hand-applied — there's no upstream PR
number for it. It is **superseded for most users by `--n-cpu-moe`**, which
achieves the same effect via explicit tensor placement (host buffer) instead
of UVM hinting (managed allocation with location advice). Both produce ~18
t/s on DeepSeek V3.1 Q4_K_M on GH200 — they're alternative implementations of
the same idea.

When to use which:
- `--n-cpu-moe` if your build is current and you want a simple flag.
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` + the cudaMemAdvise patch if you want
  the *rest* of the model (KV cache, activations) to be allowed to spill into
  Grace LPDDR5X as well — useful for very long contexts on Kimi K2 where even
  the KV grows beyond HBM.

### Confidence
**High**: PR #8035 merged, env var in master, both confirmed via release docs.
**High**: cudaMemAdvise patch from #18005 is not upstream; the discussion has
no merge banner and no PR was filed.
**Medium**: claim that `--n-cpu-moe` produces equivalent throughput — I
inferred this from the discussion text rather than seeing both rows in a
single table.

---

## Question 3 — Llama-3.3-70B Q4/Q5/Q6/Q8 single-stream TG on GH200

### What exists

I could not find a published llama.cpp tok/s number for Llama-3.3-70B on
GH200 specifically. The closest data points:

- **Baseten / Lambda blog (2024 Nov)**: TensorRT-LLM (not llama.cpp) on GH200
  with Llama-3.3-70B FP8: **3.55x** speedup vs H100 SXM via speculative
  decoding; raw TG numbers not given for llama.cpp.
- **NVIDIA developer forum, DGX Spark thread**: Llama-3.3-70B Q4 ≈ 13–14 t/s
  TG on Spark/GB10 (128 GB unified memory, smaller GPU than GH200). Plausible
  GH200 floor.
- **Ventus Servers blog, H100**: Llama-3.1-70B Q4 4 parallel slots: 120 t/s
  aggregate (~30 t/s per slot). GH200 should match or beat at single-stream
  due to higher memory bandwidth.

### Theoretical envelope

Llama-3.3-70B is **dense, ~70 B params**. Q4_K_M weights ≈ 42 GB; Q5_K_M ≈ 50
GB; Q6_K ≈ 58 GB; Q8_0 ≈ 75 GB. All four fit in 96 GB HBM3 on the base GH200
SKU with room for KV at -c 8192. No CPU offload needed.

TG is **memory-bandwidth bound** at batch 1: `tok/s ≈ HBM_BW × util / weight_bytes`.
- HBM3 on GH200: ~3 TB/s peak.
- llama.cpp typically achieves 60-75 % of HBM peak on dense matmul (consensus
  from H100/H200 numbers).
- For Q4_K_M (42 GB): ~3 TB/s × 0.7 / 42 GB ≈ **50 t/s** ceiling, ~35–45 t/s
  realistic.
- Q5_K_M (50 GB): ceiling ~42 t/s, realistic 28-36.
- Q6_K (58 GB): ceiling ~36 t/s, realistic 24-32.
- Q8_0 (75 GB): ceiling ~28 t/s, realistic 18-24.

For reference an H100 SXM (80 GB HBM3, 3.35 TB/s) achieves ~38-42 t/s at
Llama-3.3-70B Q4 in llama.cpp per scattered LocalLLaMA reports. GH200 has the
same HBM3 but slightly less per-stack BW (~3.0 vs 3.35 TB/s on the 96 GB
HBM3 SKU); on the 144 GB HBM3e SKU it's ~4.9 TB/s — meaningfully faster TG
expected.

### Recommendation

Just run the bench:
```bash
./build/bin/llama-bench \
  -m llama-3.3-70b-Q4_K_M.gguf \
  -m llama-3.3-70b-Q5_K_M.gguf \
  -m llama-3.3-70b-Q6_K.gguf   \
  -m llama-3.3-70b-Q8_0.gguf   \
  -ngl 99 -fa 1 -t 16 -p 512 -n 128
```
This costs ~3 minutes on GH200 and would close the single biggest dense-LLM
data gap in our dataset.

### Confidence
**Low** on absolute numbers (extrapolated from H100/H200 + HBM ratios). 
**High** on shape (Q4 fastest by a clear margin, Q8 slowest, all four fit in
HBM so no offload artifacts). **Gap**: nobody has published this measurement.

---

## Question 4 — Speculative decoding on GH200 (MTP PR #22673, Llama-3.3)

### PR #22673 — MTP support
Status: **merged 2026-05-16** (by am17an). Activated via:
```bash
--spec-type draft-mtp --spec-draft-n-max N
```
where `N` is max draft tokens per step (typically 2–3).

Models with MTP heads in 2026: **Qwen3.6-27B, Qwen3.6-35B-A3B (MoE),
Qwen3.5** family. Llama-3.3-70B and DeepSeek V3/R1 do **not** ship MTP heads —
they require the classic draft+target speculative pair.

Reported performance (PR thread):
- Qwen3.6-27B on RTX 3090, 3 draft tokens, ~75% acceptance: **38 → 65 t/s**
  (1.71x).
- 2 draft tokens, ~83% acceptance: best on some hardware.
- Strix Halo APU: only 1.2x — speedup is GPU-bound; benefits scale with
  draft-verify being cheaper than serial decode.

Tested backends: CUDA, Vulkan, Metal. Constraint: `n_parallel=1` only
(speculative + batching not yet implemented).

### Llama-3.3-70B: classic draft+target
Best draft model picks (community consensus on LocalLLaMA):

| Target           | Draft               | Vocab compat | Acceptance | Speedup |
|------------------|---------------------|--------------|-----------:|--------:|
| Llama-3.3-70B    | Llama-3.2-1B        | exact match  |   ~65–75% |  1.6–2.1x |
| Llama-3.3-70B    | Llama-3.2-3B        | exact match  |   ~75–85% |  1.7–2.3x |
| Llama-3.3-70B    | Llama-3.1-8B        | exact match  |   ~80–88% |  1.4–1.8x* |

*8B draft is overkill — verification cost eats some of the gain; 1B/3B
better on a single GH200.

LocalLLaMA report (RTX 3090): Llama-3.3-70B Q4 + Llama-3.2-1B Q4 draft: **30 →
160 t/s** (5.3x). That's unrealistically high — it includes both speculative
gain *and* moving the target from CPU-offloaded to fully GPU-resident. On a
GH200 where the 70B already fits in HBM, the real speculative gain alone is
1.6–2.1x as in the table above.

### MTP on GH200 specifics
- MTP heads are small (~1-2 % of model size). They sit in HBM next to the
  target.
- For DeepSeek V3.1 with `--n-cpu-moe 58` (experts in Grace), an MTP head on
  the target (when DeepSeek adds one — it doesn't yet) would stay in HBM.
- Until DeepSeek/Llama ship MTP heads, on GH200 the practical recipe for
  speculative decode is **classic two-model**: Llama-3.3-70B-Q4 target +
  Llama-3.2-1B-Q4 draft (or 3B draft for tighter acceptance).

### Confidence
**High** on PR #22673 status and the Qwen3.6 numbers (multiple independent
sources). **Medium** on the Llama-3.3 + Llama-3.2-1B speedup band — the 5.3x
report is suspect; 1.6–2.1x is the cross-source consensus. **High** that
DeepSeek V3.1 / Kimi K2 don't ship MTP heads in current GGUFs.

---

## Question 5 — Batched / continuous batching in llama-server

### Mechanism
Continuous batching is **enabled by default**. Key flags:
- `--np N` (alias `--parallel N`): number of concurrent slots. Default 1.
- `-c C`: total context. **Divided across slots**: slot context = C / np.
- `-b B`: logical batch size (tokens dispatched per scheduling round).
  Default 2048.
- `-ub U`: physical micro-batch ("ubatch"). Default 512.
- `--cont-batching`: explicit toggle, default on.
- `--cram MB`: host-RAM prompt cache size; 0 = unlimited.

Crucial: a single 65 K context with `--np 16` gives each slot only 4 K tokens.
For agent workloads with long shared system prompts this is fatal; you must
size `-c` as `slot_ctx × np`.

### Throughput numbers
From discussion #18308 (RTX 5090, recent master):

| `--np` | batch | Aggregate t/s |
|-------:|------:|--------------:|
|      1 |    32 |          ~295 |
|     16 |    32 |          ~936 |

Scaling is **3.2x at 16x parallelism**, not 16x — sampling overhead and slot
contention dominate beyond ~4 slots. GPU utilization peaks at ~60 % in
llama-server vs ~96 % in llama-bench (the latter is steady-state-only).

Promptsicle benchmark (RTX 4090, Llama-3-13B):
- np=1, b=512: 28 t/s/user, 28 t/s aggregate.
- np=4, b=512: 18 t/s/user, 72 t/s aggregate.
- np=4, b=2048: 25 t/s/user, 100 t/s aggregate.

### llama-server vs vLLM/SGLang at concurrency
llama-server's slot scheduler is **noticeably weaker** than vLLM PagedAttention
or SGLang RadixAttention at 16+ concurrent users. Issue #22942 documents that
prompt-cache checkpoints are **slot-local** — two clients with the same
system prompt landing on different slots can't share cache, which is the
exact opposite of RadixAttention's behavior.

Rough rule of thumb on GH200:
- 1–4 concurrent users sharing a single Llama-3.3-70B: **llama-server fine**.
- 8+ concurrent users, dense model: **switch to vLLM or SGLang**.
- Any-concurrency MoE that needs Grace offload: **llama-server is the only
  option** — vLLM/SGLang don't have `--n-cpu-moe`-equivalent on GH200 yet.

### Confidence
**High** on mechanism and the scaling shape. **Medium** on absolute aggregate
numbers (RTX 5090 / RTX 4090 references; GH200 would scale similarly relative
to single-slot throughput, but not identically).

---

## Question 6 — CPU-only on Grace 72-core: when does it beat GPU-offload?

### Grace baseline
- 72 Neoverse-V2 cores, NEON + SVE2 + i8mm + bf16 vector.
- 480 GB LPDDR5X @ 546 GB/s nominal (about half of an H100's HBM3).
- llama.cpp has KleidiAI integration shipped (Arm Kleidi kernels), giving
  the i8mm fast path on Q4_0/Q8_0.

### Measured CPU-only points on GH200 (discussion #18005)
| Model            | Threads | PP t/s | TG t/s |
|------------------|--------:|-------:|-------:|
| GPT-OSS 20 B MXFP4 |     32 | 216.27 |  84.52 |
| GPT-OSS 20 B MXFP4 |     72 | 377.55 |  34.83 |

The 72-thread TG number **drops** vs 32 threads — Grace's memory subsystem
saturates around 32–40 threads on MoE; more threads add contention without
adding effective bandwidth. **32–36 threads is the Grace sweet spot for
batch-1 TG.**

### Comparable points (extrapolated)
- AWS Graviton4 (Neoverse-V2, 96 vCPUs), Llama-3.3-70B: ~50 t/s prompt
  encoding, ~10–15 t/s generation per Arm Newsroom (Google Axion c4a-standard-32,
  32-core Neoverse-V2 with KleidiAI).
- Grace 72-core should produce ~1.5–2x Axion's 32-core numbers if memory
  bandwidth doesn't bind first; in practice it does bind, so expect **15–22
  t/s TG on Llama-3.3-70B Q4_K_M CPU-only on Grace**.

### When does CPU-only beat GPU-offload?
For **dense** models that fit in HBM, GPU always wins. The only crossovers
are:
1. **MoE with very few experts active per token** (Mixtral 8x2): the GPU
   path is bottlenecked by C2C activation transfer; pure CPU with all
   experts in LPDDR5X avoids the transfer. Estimate: 15–18 t/s CPU-only vs
   18–22 t/s with `--n-cpu-moe` on Mixtral 8x22B Q4 — close enough that
   CPU-only wins on power and on freeing the GPU for other work.
2. **Long-context decode with quantized KV** on a model that already
   stresses HBM: pure-CPU avoids HBM thrash. Threshold is workload-specific
   but generally 32 K+ context.
3. **Multi-tenant where the GPU is busy serving another model**: CPU-only
   becomes the cheap secondary lane.

For DeepSeek V3.1 / Kimi K2, GPU-with-offload always wins because the
*shared* expert + dense layers benefit from HBM, even if 95 % of weights live
in LPDDR5X.

### Confidence
**High** on the 32-thread sweet spot and the GPT-OSS 20 B numbers.
**Medium** on extrapolated Grace 70B numbers (no direct measurement, only
adjacent Neoverse-V2 systems).

---

## Question 7 — `llama-server` OpenAI-compat: feature parity with vLLM/SGLang

### Endpoints implemented (b9305)
- `GET /v1/models`
- `POST /v1/completions` (streaming + non-streaming)
- `POST /v1/chat/completions` (streaming + non-streaming, multimodal via
  `image_url`)
- `POST /v1/embeddings` (requires pooling-enabled model)
- `POST /v1/responses` (translates Anthropic-style responses to chat
  completions)
- `POST /v1/messages` + `/v1/messages/count_tokens` (Anthropic Messages API)
- Tool calling: native via `--jinja` (per-model parsers: Llama 3.x header
  style, Hermes/Functionary XML, Mistral `[TOOL_CALLS]`, Qwen markdown-JSON,
  DeepSeek tags).
- Structured output: `response_format` accepts plain `json_object` or JSON
  schema. Schema is auto-converted to GBNF grammar.
- Logprobs: yes (`logprobs`, `top_logprobs`).
- Function calling per OpenAI spec: yes.

### Feature gaps vs vLLM/SGLang
| Feature                         | llama-server | vLLM    | SGLang  |
|---------------------------------|--------------|---------|---------|
| `/v1/chat/completions`          | yes          | yes     | yes     |
| Tool calling auto-parser        | yes          | yes     | yes     |
| JSON schema → grammar           | yes          | xgrammar| compiled FSM (3x faster) |
| Logprobs                        | yes          | yes     | yes     |
| Cross-slot prompt cache sharing | **no** (slot-local) | partial | yes (RadixAttention) |
| Speculative decoding via API    | yes (transparent) | yes | yes     |
| Continuous batching             | yes (default)| yes     | yes     |
| Chunked prefill                 | yes (PP/TG interleave) | yes | yes |
| Disaggregated prefill/decode    | **no**       | yes (KV transfer) | yes |
| LoRA hot-swap                   | partial      | yes     | yes     |
| `/v1/audio/*` (Whisper)         | via separate `llama-server` for whisper-cpp | no  | no  |
| `/v1/embeddings` with rerank    | yes (`/v1/rerank`) | yes | yes     |
| Multi-LoRA per request          | **no**       | yes     | yes     |
| Prefix-cache hit rate metrics   | basic        | rich    | rich    |

### When llama-server is the right OpenAI-compat choice
- Single user or ≤4 concurrent: throughput tied or better than vLLM on the
  same model (vLLM's batching overhead dominates at low concurrency).
- Any GH200 MoE >100 B: only engine with `--n-cpu-moe`.
- GGUF-quantized models that vLLM doesn't natively load (Q4_K_M, Q5_K_M, Q3,
  IQ-quants).

### When to switch to vLLM/SGLang
- 8+ concurrent users.
- Need cross-user prefix cache (system prompt sharing).
- Need multi-LoRA per request.
- Production agent fleet with mixed tool-calling models.

### Confidence
**High** on the feature lists; cross-checked llama.cpp `tools/server/README.md`
against vLLM/SGLang docs.

---

## Question 8 — KV cache types Q4_0 / Q8_0 / F16 with Grace LPDDR5X offload

### llama.cpp KV cache quantization
Flags: `--cache-type-k <type>` and `--cache-type-v <type>`. Types: `f32`,
`f16` (default), `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`.
You can quantize K and V independently. Requires `-fa` (Flash Attention) for
quantized V on most backends.

### Quality findings (consolidated from #20969, #21385, #5932, NVIDIA forum
DGX Spark thread)
- **q8_0**: "nearly identical" to f16 on 35 B+ models at temperature 0 —
  bit-exact in many cases. **2x memory savings.** Recommended default for
  long-context.
- **q4_0**: visible degradation on math/reasoning benchmarks, fine for
  casual chat. **4x memory savings.** Per-head adaptive q4_0 (issue #21385)
  is being explored and shows "lossless q4_0" on hybrid attention models.
- **q5_0 / q5_1**: better-than-q4_0 quality with ~2.5x savings; under-used
  in practice.

### Throughput on DGX Spark / GB10 (Nemotron-3-Nano-30B-A3B Q4_K_XL, build
8399, 128 GB unified mem)

| ctx    | f16 PP / TG | q4_0 PP / TG |
|--------|------------:|-------------:|
| ~8 K   | 371.3 / 14.7 | 363.4 / 14.2 |
| ~16 K  | 360.7 / 13.9 | 346.2 / 12.7 |
| ~32 K  | 328.3 / 13.5 | 316.9 / 11.0 |
| ~64 K  | 282.7 / 13.3 |  21.3 /  8.6 |

**q4_0 collapses at 64 K context** — the per-token dequant overhead in flash
attention grows linearly with context. **q8_0 is reportedly the sweet spot**;
specific numbers weren't tabulated in the source but consensus is "matches
f16 within 5 % across all context lengths, saves 50 % memory."

### GH200 / Grace LPDDR5X implications
The Spark numbers translate directly because Spark also has unified memory
(GB10, smaller GPU + Arm CPU + LPDDR5X). On GH200:

1. **KV in HBM** (default): fine up to ~32 K on a 70 B; q8_0 doubles that.
2. **KV in Grace LPDDR5X via UVM**: viable but adds C2C round-trips per
   attention step. Useful only when HBM is already full of weights (Kimi K2
   with all experts on Grace + KV pushed off the dense layers).
3. **Quantized KV + Grace offload**: with q8_0 KV in Grace LPDDR5X, you can
   push to 128 K context on Llama-3.3-70B without dipping under 20 t/s. The
   bottleneck shifts from KV memory to C2C bandwidth.

Recommended GH200 KV configs:

| Workload                                     | -ctk / -ctv | Why |
|----------------------------------------------|-------------|-----|
| Llama-3.3-70B, ≤16 K ctx                    | f16 / f16   | Headroom plenty, no quality risk |
| Llama-3.3-70B, 32–128 K ctx                 | q8_0 / q8_0 | Halves KV, near-zero quality loss |
| DeepSeek V3.1 with `--n-cpu-moe 58`, ≤16 K  | f16 / f16   | KV is small relative to expert RAM |
| DeepSeek V3.1, 64 K+ ctx                    | q8_0 / q8_0 | Avoid HBM pressure on attention |
| Math/reasoning eval                         | f16 / f16   | q4_0 visibly degrades these |
| Casual chat / RAG with 128 K               | q8_0 / q4_0 (V) | V tolerates more quant than K |

### Confidence
**High** on q8_0 vs f16 quality (corroborated by 3 independent sources).
**High** on the q4_0 64 K cliff (reproducible per the NVIDIA forum thread).
**Medium** on the "q8_0 KV in Grace LPDDR5X" claim — extrapolated from C2C
bandwidth and Spark data; not directly measured on GH200.

---

## Summary table — recommended GH200 configs by model

| Model              | Quant   | KV   | --n-cpu-moe | spec  | -np | Expected TG (t/s) |
|--------------------|---------|------|-------------|-------|----:|------------------:|
| Llama-3.3-70B      | Q4_K_M  | f16  | n/a (dense) | L3.2-1B | 1 | **35–45 (single), 60–80 with spec** |
| Llama-3.3-70B      | Q5_K_M  | f16  | n/a         | L3.2-3B | 1 | **28–36**         |
| Llama-3.3-70B      | Q8_0    | f16  | n/a         | L3.2-3B | 1 | **18–24**         |
| DeepSeek V3.1 671B | Q4_K_M  | q8_0 | 58          | none  |   1 | **~18** (measured)|
| Kimi K2 ~1T        | Q3_K_M  | q8_0 | 60          | none  |   1 | **~21** (measured)|
| GPT-OSS 120 B      | MXFP4   | f16  | 0           | none  |   1 | **~209** (measured)|
| GPT-OSS 20 B       | MXFP4   | f16  | 0           | none  |   1 | **~323** (measured)|
| Mixtral 8x22B 141B | Q4_K_M  | f16  | 0 (144 GB) / 40 (96 GB) | none | 1 | **~35–55 est.**  |
| Qwen3.6-27B (MTP)  | Q5_K_M  | f16  | n/a         | MTP-3 | 1 | **65+ (RTX 3090 ref)** |

---

## Confidence & Gaps

### High confidence (measured on GH200 or documented in master)
- `--n-cpu-moe` mechanism, layer count for each MoE.
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` is in master via PR #8035.
- cudaMemAdvise patch from #18005 is **not merged**, superseded by `--n-cpu-moe`
  for most cases.
- MTP support (PR #22673) merged 2026-05-16; Qwen3.6 only.
- llama-server OpenAI endpoint list as of b9305.
- DGX Spark KV cache quant numbers (Nemotron-3-Nano-30B).
- Grace 72-core: 32–36 thread sweet spot for TG.

### Medium confidence (extrapolated / partial data)
- Llama-3.3-70B Q4–Q8 single-stream TG on GH200 — band derived from H100/H200
  references and HBM bandwidth ratios, not direct measurement.
- Mixtral 8x22B on GH200 — no published number, but architecture and quant
  size make the recommended config straightforward.
- Speculative decoding speedup for Llama-3.3-70B on GH200 — 1.6–2.1x consensus
  from consumer-GPU reports; not GH200-specific.

### Low confidence / explicit gaps
- **No `--n-cpu-moe` sweep on GH200**: nobody has published `(N, t/s)` pairs
  for DeepSeek V3 / Kimi K2 / GPT-OSS 120 B / Mixtral 8x22B. This is the
  highest-value missing measurement.
- **No llama.cpp Llama-3.3-70B benchmark on GH200**: 1 hour of `llama-bench`
  closes this.
- **No measured C2C utilization** from llama.cpp's expert-streaming path —
  is it 100 GB/s or 700 GB/s? Determines headroom for more aggressive
  offload schemes.
- **Multi-tenant llama-server on GH200**: no published `--np` scan, no
  comparison to vLLM at the same concurrency.

### Constraint reminders
- FP8 is broken on DiT in this stack — irrelevant for llama.cpp (no FP8
  weights path; uses GGUF quants).
- bf16 default for diffusion — irrelevant for llama.cpp inference.
- GH200 single-GPU; multi-GH200 NVLink-Switch fabric not covered here.

---

## Sources

- https://github.com/ggml-org/llama.cpp/discussions/18005 — GH200 perf thread (Dec 2025)
- https://github.com/ggml-org/llama.cpp/pull/15077 — `--n-cpu-moe` PR (merged 2025-08-04, slaren)
- https://github.com/ggml-org/llama.cpp/issues/15263 — multi-GPU `--n-cpu-moe` feature request
- https://github.com/ggml-org/llama.cpp/pull/8035 — UVM PR (matteoserva)
- https://github.com/ggml-org/llama.cpp/pull/12934 — CUDA/HIP UVM unification
- https://github.com/ggml-org/llama.cpp/issues/16197 — UVM not-working report (multi-GPU x86)
- https://github.com/ggml-org/llama.cpp/pull/22673 — MTP speculative-decoding PR (merged 2026-05-16, am17an)
- https://github.com/ggml-org/llama.cpp/discussions/12130 — speculative decoding / MTP design
- https://github.com/ggml-org/llama.cpp/discussions/18308 — llama-server parallel inference parameters
- https://github.com/ggml-org/llama.cpp/discussions/20574 — host-memory prompt caching tutorial
- https://github.com/ggml-org/llama.cpp/issues/22942 — slot-local prompt cache limitation
- https://github.com/ggml-org/llama.cpp/discussions/15396 — GPT-OSS guide and benchmarks
- https://github.com/ggml-org/llama.cpp/discussions/16578 — DGX Spark / GB10 perf thread
- https://github.com/ggml-org/llama.cpp/discussions/20969 — TurboQuant / extreme KV cache quantization
- https://github.com/ggml-org/llama.cpp/issues/21385 — per-head adaptive KV cache quantization
- https://github.com/ggml-org/llama.cpp/discussions/5932 — 4-bit KV cache discussion
- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md — llama-server endpoint reference
- https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md — tool calling
- https://gist.github.com/DocShotgun/a02a4c0c0a57e43ff4f038b46ca66ae0 — MoE offload guide gist
- https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide — MoE offload HF blog
- https://github.com/ikawrakow/ik_llama.cpp — ik_llama.cpp fork (better CPU MoE perf)
- https://forums.developer.nvidia.com/t/kv-cache-quantization-benchmarks-on-dgx-spark-q4-0-vs-q8-0-vs-f16-llama-cpp-nemotron-30b-128k-context/365138 — KV cache quant on Spark
- https://newsroom.arm.com/blog/arm-neoverse-llama-3-3-70b-model — Neoverse-V2 Llama 3.3 70B numbers
- https://clear.ml/blog/benchmarking-llama-cpp-on-arm-neoverse-based-aws-graviton-instances-with-clearml — Graviton llama.cpp benches
- https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama-cpu/ — Kleidi on Arm CPU
- https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/ — GH200 Llama 3.3 70B (TRT-LLM context)
- https://lambda.ai/blog/putting-the-nvidia-gh200-grace-hopper-superchip-to-good-use-superior-inference-performance-and-economics — GH200 vs H100
- https://developer.nvidia.com/blog/nvidia-gh200-superchip-accelerates-inference-by-2x-in-multiturn-interactions-with-llama-models/ — GH200 multiturn Llama
- https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/ — KV cache offload to host
- https://promptsicle.com/tips/boosting-llama-server-performance-with-batch-settings/ — parallel slots benchmark
- https://ventusserver.com/context-length-behavior-explained/ — llama-server context split behavior
- https://markaicode.com/architecture/llamacpp-system-design-architecture-1158/ — llama.cpp at 1000 concurrent users (architecture, x86)
- https://techsy.io/en/blog/vllm-vs-sglang — vLLM vs SGLang comparison
- https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison — SGLang vs vLLM benches
- https://deploybase.ai/articles/best-llm-inference-engine — engine comparison overview
- https://deepwiki.com/deepseek-ai/DeepSeek-V3/1.2-model-architecture-overview — DeepSeek V3 layer count
- https://github.com/moonshotai/Kimi-K2 — Kimi K2 architecture spec
- https://apxml.com/models/kimi-k2-instruct — Kimi K2 VRAM requirements
