# Low-Latency Interactive LLM Inference on GH200 — TTFT, ITL, and Speculative Decoding

**Agent**: #15 / Round 4
**Date**: 2026-05-25
**Builds on**: `/home/ubuntu/research/gh200_inference/round3/03_vllm_verify.md` (vLLM 0.21.0, EAGLE-3 9.5-20.5% throughput gain to 200 concurrent on H200, head_dim=256 1.6× prefill penalty unchanged)
**Hard priors carried over**: FP8 broken on this DiT stack; bf16 default for LLMs; FA4 default MLA prefill since vLLM 0.20.0.

---

## TL;DR for the impatient operator

For a single GH200 running Llama-3.3-70B Instruct, bf16, vLLM 0.21.0, on interactive workloads:

1. **Pin `cudagraph_mode=FULL_AND_PIECEWISE`** (default in V1 at `-O2`). Don't override.
2. **Lower `max_num_batched_tokens` to 2048-4096** for interactive priority (ITL-optimal). Default chunked-prefill prioritizes decode after the chunk, so this is the lever that decides "how late am I willing to be on TTFT to keep ITL tight."
3. **Enable EAGLE-3** with `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B` and `num_speculative_tokens=3`. Red Hat measured **10.5-10.8% TTFT P95 and 6.6-17.5% ITL P95** improvements on gpt-oss; comparable on Llama-3.3-70B per NVIDIA TRT-LLM (3.55× throughput).
4. **Run `nvidia-smi -pm 1`** at boot. Persistence mode keeps the driver resident — a cold first request without it loses ~50-200 ms on driver init.
5. **Use `--scheduling-policy priority`** if you mix interactive + batch on one GPU. The flag exists since 0.6+. Pass `priority=0` for chats, `priority=10` for batch. Lower number = earlier execution. Preempts running batch back to waiting queue.
6. **Self-host beats Groq for interactive when you can hit TTFT ≤ ~1.0s P50 at the concurrency you actually run.** Groq's TTFT is 0.92s P50, output 322-394 t/s, $0.59/$0.79. Cerebras is 0.3s end-to-end for 500 tokens at 2,500 t/s, $0.85/$1.20. You won't beat them on TTFT — you can beat them on **cost per output token**, on **privacy**, and on **mixed-workload control** if your TTFT is in the same order of magnitude (~1-2s P50).

---

## 1. TTFT minimization on GH200 — concrete levers

### 1.1 Chunked prefill: the central lever

Chunked prefill is on by default in vLLM V1. The scheduler **prioritizes decodes** within an iteration; remaining `max_num_batched_tokens` budget is filled with prefill chunks. This is the only knob that simultaneously moves TTFT and ITL — and they move in opposite directions.

| `max_num_batched_tokens` | Effect on TTFT | Effect on ITL | When to use |
|---|---|---|---|
| 2048 | Worse (longer prefill, more iterations to first token) | Best (fewer prefills slowing decodes per step) | Interactive chat, streaming-priority |
| 4096-8192 | Balanced | Balanced | Mixed workload (the default-ish sweet spot) |
| 16384 | Best for single short prompt | Worse (decodes share iteration with bigger prefill chunks) | Single-user low-concurrency RAG with short contexts |
| 32k-64k | Best TTFT for long-prompt | Worst | Offline / throughput / disable chunked prefill territory |
| `= max_model_len` | V0-equivalent behavior | Worst | Don't, unless reproducing old benchmarks |

**Default in vLLM V1 docs:** "keep between 8k and 16k for interactive low-TTFT workloads." Round-1 advice of 8192 holds; for streaming chat (ITL-dominant) **drop to 2048-4096**.

### 1.2 Speculative prefill (vllm-mlx, March 2026)

A new technique landed in `vllm-mlx` PR #180 (merged 2026-03-21): **draft-assisted sparse prefill**. Uses a draft model to identify which prefill tokens are "important" and skips the rest. Reported **7.66× TTFT improvement and 7× E2E QPS on Llama-3.1-405B-Instruct-FP8**.

Status as of 2026-05-25:
- Production-implemented only in the vllm-mlx fork (Apple Silicon focus).
- Not yet in upstream vLLM (issue #39060 still open).
- Not validated on CUDA / GH200.
- **Watch this; don't depend on it.**

### 1.3 Scheduler priority for mixed workloads

`--scheduling-policy priority` (vLLM 0.6+, fully stabilized in 0.20+):
- **FCFS (default)**: requests served in arrival order.
- **priority**: sorted by integer priority; lower = earlier. Ties broken by arrival time.
- **Preemption**: if a higher-priority request lands while a lower-priority one is running, the running request is **forcefully evicted to the waiting queue**. (More aggressive than mere queue reordering.)

Per-request: set `priority` field in the OpenAI API `extra_body` or directly via the offline `LLM.generate(..., priority=N)`. Convention:
- 0-5: interactive chat / streaming
- 10: batch processing
- 50+: best-effort background

**RFC #30256** (still open) proposes "SLA-tiered scheduling" with per-request TTFT/ITL SLO targets driving admission control. Not landed in 0.21.0. Use the integer-priority knob until it ships.

### 1.4 NVL fabric / C2C warmup

GH200 has a 900 GB/s NVLink-C2C between Grace CPU and Hopper GPU, exposing a unified address space. The **first** memcpy across C2C after a long idle has higher latency than steady-state due to driver page-table state and SMs being parked.

Warmup recipe (run once at boot, after `vllm serve` starts):
1. `nvidia-smi -pm 1` — driver persistence. Skipping this costs **50-200 ms on the cold first request** because the driver tears down between client disconnects.
2. `nvidia-smi --auto-boost-default=0 ; nvidia-smi -lgc <max_clock>` — pin clocks. SM clock variation can cause **ITL P99 spikes of 5-15 ms** on otherwise idle GPUs as DVFS ramps up.
3. **Synthetic prefill warmup**: send 3-5 dummy requests of varying length (128, 1024, 8192, max_model_len/4) **before** opening the public endpoint. This pre-populates CUDA graph captures across the bucketed input sizes; otherwise the first real request of each bucket pays the capture cost (≈200-800 ms per bucket on 70B).
4. **KV-offload warmup**: if using `--kv_offloading_size`, send one request that exceeds in-HBM KV. This walks the path through Grace LPDDR once and primes the C2C TLB / unified memory page table. Skipping this is a documented ITL spike on the first overflow.

For the GH200 NVL32 large-system case (NVLink Switch between 32 GH200s), NVIDIA documented **3× TTFT improvement for 122,880-token Llama-3.1-405B queries** vs 8×H200 baseline — but that's a fabric-scale result, irrelevant to single-superchip.

For **single-superchip multi-turn** with KV-cache offload: NVIDIA measured **2× TTFT speedup of GH200 vs x86-H100** at 7168 input sequence length, **1.2× at 1024 ISL**. The C2C transparently absorbs KV reload; PCIe-attached H100 thrashes.

### 1.5 Other TTFT knobs worth knowing

- `--enable-prefix-caching` (default ON in V1): caches KV across requests sharing a prompt prefix. Cache hits cut TTFT by **30-99%** depending on cache hit rate. For chat with system prompts, this is essentially free TTFT.
- `--enforce-eager` (default OFF): **don't use** for low latency. Disables CUDA graphs. Listed only because the flag name confuses people.
- **Disaggregated prefill** (`disagg_prefill`): documented as **experimental** in 0.21. Splits prefill onto a separate worker. Helps in large-cluster deployments; **noise on single GH200**.

---

## 2. ITL (inter-token latency) — the decode-side fight

### 2.1 CUDA graphs in vLLM V1

vLLM V1 ships five `cudagraph_mode` values in `CompilationConfig`:

| Mode | Captures | Memory | Capture time | Best for |
|---|---|---|---|---|
| `NONE` | Nothing (eager) | Lowest | None | Debug only |
| `PIECEWISE` | Non-attention ops; attention stays eager | Low | Fastest | Models with custom attention; the V0 default |
| `FULL` | Entire forward pass (decode and prefill) | Highest | Slow | Static-shape batches |
| `FULL_DECODE_ONLY` | Full graph for decode batches; prefill eager | Medium | Medium | Throughput-tilted serving |
| `FULL_AND_PIECEWISE` | Full graph for decode; piecewise for prefill / mixed | Highest | Slowest | **Default for V1 `-O2`. Best for low ITL on small/MoE models. Use this for Llama-3.3-70B interactive.** |

CLI: `--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'` or just rely on `-O2` default.

**Practical impact:** vs `PIECEWISE`, `FULL_AND_PIECEWISE` typically cuts ITL by **10-25%** on 70B-class models at moderate batch (16-64). At batch=1 the gain narrows to ~5-15% but is still net positive. The cost is memory (multiple buckets captured) and ~10-30s extra capture time at startup.

Pin the capture set explicitly to avoid recapturing under load:
```bash
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,192,256]}'
```

### 2.2 Persistence mode / pinned memory

- `nvidia-smi -pm 1`: covered above. **Mandatory** for low-latency services.
- **Pinned-memory KV-offload path**: vLLM's `--kv_offloading_backend native` uses pinned host memory by default for the DRAM/LPDDR tier on Linux. On GH200, "host memory" is Grace LPDDR via unified memory addressing; the pinned-memory `cudaMemcpyAsync` rides C2C at ~600-700 GB/s effective. No flag needed; verify with `cudaHostAlloc` traces if suspicious.
- **CPU thread affinity**: vLLM 0.21 doc recommends `2 + N_workers` physical cores reserved. On Grace (72-core), default OS scheduling is generally fine; if you see ITL jitter, pin the API server / worker threads with `taskset -c 0-7` (API) and let workers float on cores 8-71. Document numbers don't exist; this is operator folklore.

### 2.3 Sampling-path latency

- **FlashInfer top-k/top-p sampler enabled by default in 0.21.0.** Replaces the older Torch path; reduces decode-step sampling overhead by 1-3 ms per token at batch=1. **Don't disable unless you're chasing a sampler bug.**
- **TurboQuant 2-bit KV** (0.20+): 4× KV capacity at modest quality cost. Reduces effective decode memory bandwidth → **lower ITL on KV-bound regimes** (long contexts). For Llama-3.3-70B at 128k context: enables longer windows in HBM, but adds dequant cost per step. Net ITL impact is positive only above ~16k context. Skip for short-context chat.

### 2.4 The ITL floor on GH200 for Llama-3.3-70B bf16

No public-record measured number. Working estimate:
- Memory bandwidth: HBM3e = 4.8 TB/s on GH200 SXM/96G (4.8 TB/s on the 144G variant too).
- Bytes per token decode: 70B × 2 bytes = 140 GB activations-equivalent + KV reads ≈ 145 GB at batch=1, 100k context.
- Theoretical lower bound: 145 / 4800 = **30 ms/token = 33 t/s decode at batch=1**.
- Real-world with vLLM 0.21 + EAGLE-3 + CUDA graphs: probably **40-55 ms/token = 18-25 t/s** at batch=1, 10-20 ms/token with EAGLE-3 accepting 3 tokens. Anchor against Groq's 322 t/s and Cerebras's 1800-2500 t/s for perspective — these architectures don't fight HBM the same way.

**Action item carried over from Round 3:** measure this in-house. The bound is interpolation, not measurement.

---

## 3. Speculative decoding head-to-head: Llama-3.3-70B at batch=1 vs batch=16

### 3.1 The methods on the table

| Method | What it is | Requires extra training? | Lossless? | Memory cost |
|---|---|---|---|---|
| **Vanilla** | Standard autoregressive decode | No | — | — |
| **EAGLE-3** | Lightweight transformer-decoder head (~0.3-0.7B for 70B target) trained on target's hidden states; predicts a tree of candidate tokens | **Yes** (1-2 days, 8-16 H100/A100) | Yes (verify against target) | ~1-3 GB head weights + tree buffers |
| **MEDUSA** | N parallel LM heads on top of frozen target; predicts N positions ahead | Yes (lighter than EAGLE — single-GPU possible via self-distillation) | Yes | ~0.5-1.5 GB N heads |
| **Lookahead** | Algorithmic — uses Jacobi-iteration trick; no separate model | **No** | Yes | Tiny (lookahead window buffer) |
| **n-gram / prompt-lookup** | Greedy match in recent context / prompt for next-token candidates | **No** | Yes | Tiny (n-gram index) |

### 3.2 Speedup matrix (gathered from EAGLE-3 paper, Red Hat 2026-04-16, NVIDIA TRT-LLM 2024, Premai 2026, vLLM Speculators docs)

| Method | 70B batch=1 | 70B batch=16 | 70B batch=64-200 | Notes |
|---|---|---|---|---|
| Vanilla | 1.0× | 1.0× | 1.0× | Baseline |
| EAGLE-3 | **4.0-6.5×** | 2.5-4.0× | **1.10-1.27×** (Red Hat measured on gpt-oss-20B, comparable on 70B) | Persists through 200 concurrent — Round-3 finding |
| EAGLE-2 | 3.0-4.5× | 2.0-3.0× | 1.05-1.15× | 1.4× slower than EAGLE-3 |
| EAGLE-1 | 2.5-3.5× | 1.7-2.5× | ~1.05× | 1.6× slower than EAGLE-3 |
| MEDUSA | 1.8-2.4× | 1.4-1.8× | ~1.0× (no benefit) | EAGLE is 1.5-1.6× faster than MEDUSA on same target |
| Lookahead | 1.4-1.8× | 1.2-1.4× | ~1.0× | EAGLE is 2× faster than Lookahead on same target; helps most in code completion (up to 4× there) |
| n-gram / prompt-lookup | 1.2-2.0× (highly workload-dependent) | 1.0-1.3× | ~1.0× | **Best when prompt has lots of repeated structure** (RAG, code-with-context, structured outputs). Zero benefit on free-form chat. |

**Practical recommendation by batch / workload:**

| Workload | Batch=1 chat | Batch=16 mixed | Batch=100-200 batch | RAG with long shared context |
|---|---|---|---|---|
| **Best choice** | EAGLE-3 | EAGLE-3 | EAGLE-3 (if you have it; otherwise vanilla) | EAGLE-3 + prefix-caching; n-gram as fallback if no head |
| **Why** | 4-6× decode speedup, max bang for batch=1 memory-bound regime | Still dominant; ~3× | Diminishing returns but still 10-27% throughput gain | Prefix-caching does heavy lifting on TTFT; EAGLE-3 still helps decode |
| **Fallback if no EAGLE-3 head** | n-gram if prompt structure repeats; MEDUSA if you have it trained; otherwise vanilla | n-gram on structured; vanilla otherwise | vanilla (spec-decode loses anyway) | n-gram (acceptance up due to overlap with prompt) |

`num_speculative_tokens=3` is the global sweet spot. Red Hat's measured acceptance rates:
- 2 draft tokens: 45.4% accept rate, 1,160.6 tok/s
- **3 draft tokens: 35.6% accept rate, 1,171.7 tok/s ← +1.0% vs 2**
- 4 draft tokens: 28.3% accept rate, 1,078.4 tok/s ← **-8.0% vs 3**

### 3.3 Coverage — what hardened, ready-to-use EAGLE-3 heads exist

As of 2026-05-25 (Hugging Face `yuhuili/`, `RedHatAI/`, `nvidia/`, `lmsys/`, `AngelSlim/` collections):

| Target model | EAGLE-3 head HF ID | Provenance |
|---|---|---|
| **Llama-3.3-70B-Instruct** | `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B` | SafeAILab official |
| Llama-3.3-70B (alt) | `nvidia/Llama-3.3-70B-Instruct-Eagle3` | NVIDIA |
| Llama-3.1-8B-Instruct | `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` | SafeAILab official |
| Llama-3.1-70B-Instruct | `yuhuili/EAGLE3-LLaMA3.1-Instruct-70B` | SafeAILab |
| Llama-4-Maverick-17B-128E | trained by SpecForge, in `lmsys/eagle-3` | SGLang team |
| Llama-4-Scout | trained by SpecForge | SGLang team |
| Qwen3-8B | community heads on HF | various |
| Qwen3-32B | `RedHatAI/Qwen3-32B-speculator.eagle3` | Red Hat |
| gpt-oss-20B | available | (referenced in Red Hat 2026-04-16) |
| Gemma-4-31B-it | available (added 0.20) | community |
| Cohere | new "Cohere Eagle" arch (added in 0.21) | Cohere |

If your target model isn't here: train one. See §7.

---

## 4. Continuous-batching prioritization for mixed interactive + batch

Concrete pattern for a single GH200 serving both:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --cpu-offload-gb 80 \
  --kv_offloading_backend native --kv_offloading_size 256 \
  --enable-prefix-caching \
  --max-num-seqs 192 \
  --max-num-batched-tokens 4096 \
  --scheduling-policy priority \
  --speculative-config '{"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.3-Instruct-70B","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,192]}'
```

Then in client code:
- Interactive endpoint sets `priority=0` per request.
- Batch endpoint sets `priority=10`.
- Optional: a "premium tier" sets `priority=-5` to override even other interactive.

Behavioral expectations (from vLLM RFC #6077 / production reports — no published GH200 numbers):
- A batch request currently running gets **preempted to waiting** when a `priority=0` request arrives, if HBM is full.
- Its KV cache is either recomputed or (with `--swap-space` configured) swapped to CPU. On GH200 with C2C, the swap-to-LPDDR path is fast (~600 GB/s effective), so preemption cost is sub-second for typical KV.
- TTFT for the high-priority request approaches what it would be on an idle GPU; batch tail latency stretches.
- **Open caveat:** preemption thrashing if priorities are too varied. Empirical rule: keep priority levels to 2-3 tiers, not a continuous spectrum.

**Note (Round-3 carryover):** speculative decoding in 0.21+ respects "thinking budgets" — for reasoning models (Llama-3.3 + thinking, Qwen-3 + thinking) the spec-decode now correctly counts reasoning tokens. If you serve a reasoning model under priority scheduling, this matters for fair tier accounting.

---

## 5. GH200 self-host vs Groq / Cerebras — when does it win?

### 5.1 Provider numbers (current as of 2026-05-25, Llama-3.3-70B-Instruct)

| Provider | TTFT P50 | Output speed | Input $/M | Output $/M | Blended $/M | Notes |
|---|---|---|---|---|---|---|
| **Groq** | **0.92s** | 322-394 t/s | $0.59 | $0.79 | ~$0.65 | LPU; batch API + prompt caching = 25% effective |
| **Cerebras** | ~0.3s end-to-end for 500-token output | **1,800-2,500 t/s** | $0.85 | $1.20 | ~$0.95 | Wafer-scale CS-3 |
| Google Vertex | 0.65s | 149 t/s | (varies) | (varies) | — | Lowest TTFT P50 in AA's list |
| CoreWeave | 0.89s | — | — | — | — | H100/H200 GPU cloud, vLLM-class |
| FriendliAI | 1.00s | — | — | — | — | Specialized inference SW on GPU |
| SambaNova | — | 291.6 t/s | — | — | — | Reconfigurable dataflow |
| DeepInfra (Turbo, FP8) | — | — | $0.10 | — | $0.12 | Cheapest by ~5× |
| Nebius Base | — | — | — | — | $0.16 | |
| **Aggregate median** | **1.63s** | **86.9 t/s** | — | — | **$0.60** | Per artificialanalysis.ai |

### 5.2 Where the self-hosted GH200 has to land to beat each

**Cost per output token at 100% utilization** (1× GH200 ≈ $3.00-4.00/hr on-demand at Lambda/Vultr/Crusoe as of May 2026; bare-metal lease ~$1.50-2.00/hr):

- At decode speed 50 t/s effective (vanilla bf16) at typical Llama-3.3-70B GH200 estimate: 180k tokens/hr → **$22/M output tokens** on-demand, **$11/M** bare-metal. **Loses to everyone on cost** at low utilization.
- At decode 50 t/s × 64 concurrent users sharing a GPU: ~3.2M tokens/hr → **$0.94/M** on-demand, **$0.47/M** bare-metal. **Beats Cerebras, ties Groq.**
- With EAGLE-3 at 1.3× throughput (high-concurrency tail) and prefix-cache hits: **$0.36-0.72/M**. **Beats Groq and Cerebras** if you can sustain 64+ concurrent.
- At batch=1 (single-user dedicated): **never beats them on cost.** Self-host wins only on (a) privacy, (b) data residency, (c) custom model, (d) batch jobs that don't need realtime.

**TTFT crossover:**

| Comparison | GH200 wins TTFT when... |
|---|---|
| vs Groq (0.92s P50) | GH200 P50 ≤ ~0.9s. Achievable for prompts ≤ 4096 tokens with prefix-caching on warm caches. **Hard for cold long prompts (32k+).** |
| vs Cerebras (~0.3s) | **Not achievable at any setting.** Cerebras dominates TTFT. Don't try to compete here. |
| vs Vertex (0.65s) | GH200 P50 ≤ ~0.65s — borderline; warm cache + prefix hit + short prompt only. |

**Practical verdict:** self-host GH200 wins for interactive workloads when:
1. You serve at **moderate-to-high concurrency** (16+ active users per GPU) — the per-token cost drops below provider rates.
2. Your TTFT requirement is **≤ 2s P50, not ≤ 0.5s P50** — you'll match Groq's order of magnitude but not Cerebras's.
3. You have **continuous demand** — provider APIs auto-scale; a parked GH200 is pure burn.
4. You have **mixed workloads** — priority scheduling lets one GPU serve chat + batch + agents from one pool.
5. You need **data control** / on-prem.

Self-host **loses** for spiky low-volume interactive demand: Groq's pay-per-token plus their TTFT will dominate.

---

## 6. EAGLE-3 head training for custom models — cost and recipe

### 6.1 Reference numbers

| Model size | Hardware | Wall-clock | Dataset | Source |
|---|---|---|---|---|
| LLaMA2-Chat 7B / 13B (EAGLE-1) | 4× A100-40G | 1-2 days | ShareGPT subset | SafeAILab README |
| LLaMA2-Chat 70B (EAGLE-1) | 4× A100-40G | 1-2 days | ShareGPT subset | SafeAILab README |
| LLaMA-3 70B (EAGLE-3) | 16× A100 | ~2 weeks | ShareGPT + curated | Community report cited in Premai/E2E |
| Llama 4 Maverick / Scout (EAGLE-3, SpecForge) | 8+ GPU (FSDP+TP, unspecified count/type) | unspecified, "several days" expected | 320K samples from ShareGPT + UltraChat (~12 TB raw) | LMSYS SpecForge blog 2025-07-25 |
| Llama-3 8B drafter (EAGLE-3, SpecForge) | 8 GPU (unspecified) | "few hours" | 1,000-5,000 domain examples for finetune; 320K for from-scratch | ROCm AI Dev Hub tutorial |

**Rough cost envelope for a from-scratch EAGLE-3 head on a 70B target:**
- Compute: 16× H100 × ~10 days × ~$2.50/H100-hour on-demand = **~$9,600** on-demand.
- Bare-metal lease (e.g., Lambda 8×H100 + 8×H100): 16× H100 × 240 hrs × $1.20/hr = **~$4,600**.
- Plus engineering time to prepare ShareGPT+UltraChat, run validation, debug acceptance — assume 2 engineer-weeks.

**Reduced-cost path: finetune from an existing EAGLE-3 head.** If a head exists for a related target (e.g., Llama-3.3-70B head as starting point for a Llama-3.3-70B fine-tune of yours), SpecForge can do **domain-adaptation finetune in hours on 2-4 A100s** with 1k-5k domain examples. This is the realistic path for most teams.

### 6.2 Steps (with SpecForge, the recommended path per SafeAILab and SGLang)

1. **Get the target model frozen** — Llama-3.3-70B-Instruct or your custom finetune.
2. **Generate training data**:
   - Use ShareGPT (120K conversations) + UltraChat (200K) = 320K samples — the default SpecForge corpus.
   - **Or** sample from your target model on your domain prompts — important when the model has been domain-finetuned; ShareGPT/UltraChat were generated by GPT-3.5/4 and don't match a Llama-3.3 distribution exactly. The training-time test in EAGLE-3 helps but isn't a substitute.
3. **Hyperparameters** (defaults from SpecForge configs, infer from EAGLE repo):
   - Optimizer: AdamW
   - Learning rate: ~3e-4 (cosine schedule, warmup ~2%)
   - Batch size: 4-8 per GPU with grad accumulation to global batch ~64-128
   - Training-time test (TTT) toggled on (EAGLE-3 feature)
   - Epochs: 2-4 over the 320K sample set
   - Mixed precision: bf16
4. **Train** on 8-16 H100s via SGLang's SpecForge. FSDP + TP for memory.
5. **Validate** acceptance length on a held-out set (MT-Bench, HumanEval); EAGLE-3 targets accept length ~3.5-5.0 tokens per draft round.
6. **Upload** the head; serve via vLLM `--speculative-config method=eagle3 model=<your-head>`.

**Gotchas:**
- Don't train on `--enforce-eager` mode; CUDA-graph mismatches confuse the TTT loss.
- Validate acceptance on **your** workload distribution, not just MT-Bench. A head trained on ShareGPT can have low acceptance on, e.g., legal or code-with-comments prompts.
- Head architecture is small (~0.3-0.7B params for 70B target) but training loss can plateau if you under-regularize — SafeAILab issue #286 documents loss-non-decreasing scenarios; fix by lowering LR or increasing TTT noise.

---

## 7. Updated confidence map

**High confidence:**
- `--scheduling-policy priority` exists, preempts running requests, integer-priority lower=earlier. [vLLM PR #5958, RFC #6077, docs]
- `cudagraph_mode=FULL_AND_PIECEWISE` is V1 `-O2` default; best for low-ITL 70B interactive. [vLLM compilation config docs, PR #20059, PR #16072]
- EAGLE-3 with `num_speculative_tokens=3` is the sweet spot; persists through 200 concurrent on H200. [Red Hat 2026-04-16, EAGLE-3 paper]
- Groq Llama-3.3-70B = $0.59 in / $0.79 out, TTFT 0.92s, 322-394 t/s. [groq.com/pricing, artificialanalysis.ai]
- Cerebras Llama-3.3-70B = $0.85 in / $1.20 out, 1,800-2,500 t/s, ~0.3s for 500 tokens. [cerebras.ai/blog, pricepertoken.com]
- Aggregate median TTFT for Llama-3.3-70B across providers = 1.63s, $0.60 blended. [artificialanalysis.ai]
- EAGLE-3 vs MEDUSA: EAGLE is 1.5-1.6× faster on same target. [SafeAILab README, EAGLE-3 paper]
- EAGLE-3 vs Lookahead: EAGLE is ~2× faster. [SafeAILab README]
- `nvidia-smi -pm 1` persistence mode required for stable low-latency. [NVIDIA driver docs]
- GH200 multi-turn TTFT 2× speedup vs x86-H100 with KV-offload at 7168 ISL. [NVIDIA blog]

**Medium confidence:**
- Cerebras specific TTFT P50 number (~0.3s) is end-to-end for 500-token output rather than purely first-token — TokenMix reports it; artificialanalysis doesn't list separate P50 for Cerebras on Llama-3.3-70B in the providers table.
- EAGLE-3 batch=1 speedup of 4-6× for 70B comes from the EAGLE-3 paper claims; Red Hat's 9.5-20.5% concurrency-200 number is much smaller because target is compute-bound at high batch. Both can be true and aren't contradictory.
- Exact MEDUSA training cost on 70B (claimed "single-GPU self-distillation" — likely smaller models, scaling unclear).

**Low confidence / open:**
- **No GH200-measured TTFT P50 number for Llama-3.3-70B bf16 vLLM 0.21 exists publicly.** Estimated 0.5-1.5s P50 for prompts ≤ 4k with warm caches; **measure this in-house**.
- **No GH200-measured ITL P50 number** for the same configuration. Estimated 40-55 ms/token vanilla, ~15-25 ms/token with EAGLE-3.
- SpecForge exact GPU-hours for a 70B EAGLE-3 head from scratch — community reports ~2 weeks on 16× A100/H100 but no LMSYS-published canonical number.
- Speculative prefill (7.66× TTFT) — only in `vllm-mlx`; not validated on CUDA.

---

## 8. Recommended starting config for our GH200 interactive deployment

```bash
# Pre-boot system tuning
nvidia-smi -pm 1
nvidia-smi --auto-boost-default=0

# Launch
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --cpu-offload-gb 80 \
  --kv_offloading_backend native --kv_offloading_size 256 \
  --enable-prefix-caching \
  --max-num-seqs 192 \
  --max-num-batched-tokens 4096 \
  --scheduling-policy priority \
  --speculative-config '{"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.3-Instruct-70B","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,192]}'

# Post-launch warmup (do before opening the public endpoint)
for L in 128 1024 8192 16384; do
  curl -sS -X POST http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"meta-llama/Llama-3.3-70B-Instruct\",\"prompt\":\"$(python -c "print('test '*$L)")\",\"max_tokens\":16,\"priority\":0}" \
    > /dev/null
done
```

Then measure with `vllm bench serve` against ShareGPT:
- 1 concurrent — single-user TTFT and ITL
- 16 concurrent — typical chat
- 64 concurrent — heavy chat
- 192 concurrent — saturation

Compare against Groq's 0.92s/322 t/s and report.

---

## 9. Sources consulted this round

- vLLM optimization docs — https://docs.vllm.ai/en/stable/configuration/optimization/
- vLLM compilation config (cudagraph_mode) — https://docs.vllm.ai/en/stable/api/vllm/config/compilation/
- vLLM CUDA graphs design — https://docs.vllm.ai/en/stable/design/cuda_graphs/
- vLLM full cudagraph PR #16072 — https://github.com/vllm-project/vllm/pull/16072
- vLLM full cudagraph + attention PR #20059 — https://github.com/vllm-project/vllm/pull/20059
- vLLM priority scheduling RFC #6077 — https://github.com/vllm-project/vllm/issues/6077
- vLLM priority scheduling PR #5958 — https://github.com/vllm-project/vllm/pull/5958
- vLLM SLA-tiered scheduling RFC #30256 — https://github.com/vllm-project/vllm/issues/30256
- vLLM scheduler config — https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/
- vLLM speculative prefill issue #39060 — https://github.com/vllm-project/vllm/issues/39060
- vLLM ROCm V1 optimization — https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- vLLM scheduling explainer (Audrey Wang) — https://audreywongkg.medium.com/understanding-vllm-scheduling-token-budgets-chunked-prefill-and-policies-2c879e3980e3
- vLLM priority scheduling explainer (HackerNoon) — https://hackernoon.com/how-vllm-prioritizes-a-subset-of-requests
- Red Hat 5-steps performance triage — https://developers.redhat.com/articles/2026/03/09/5-steps-triage-vllm-performance
- Red Hat EAGLE-3 gpt-oss benchmark — https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss
- EAGLE-3 paper — https://arxiv.org/html/2503.01840v1
- SafeAILab EAGLE GitHub — https://github.com/SafeAILab/EAGLE
- yuhuili/EAGLE3-LLaMA3.3-Instruct-70B — https://huggingface.co/yuhuili/EAGLE3-LLaMA3.3-Instruct-70B
- nvidia/Llama-3.3-70B-Instruct-Eagle3 — https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-Eagle3
- RedHatAI/Qwen3-32B-speculator.eagle3 — https://huggingface.co/RedHatAI/Qwen3-32B-speculator.eagle3
- AngelSlim EAGLE3 collection — https://huggingface.co/collections/AngelSlim/eagle3
- LMSYS SpecForge blog — https://www.lmsys.org/blog/2025-07-25-spec-forge/
- ROCm SpecForge tutorial — https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/SpecForge_SGlang.html
- Premai speculative decoding 2026 — https://blog.premai.io/speculative-decoding-2-3x-faster-llm-inference-2026/
- E2E Networks EAGLE guide — https://www.e2enetworks.com/blog/Accelerating_LLM_Inference_with_EAGLE
- NVIDIA TRT-LLM Llama-3.3-70B 3× spec decode — https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/
- NVIDIA GH200 multi-turn 2× — https://developer.nvidia.com/blog/nvidia-gh200-superchip-accelerates-inference-by-2x-in-multiturn-interactions-with-llama-models/
- NVIDIA GH200 NVL32 TTFT — https://developer.nvidia.com/blog/low-latency-inference-chapter-2-blackwell-is-coming-nvidia-gh200-nvl32-with-nvlink-switch-gives-signs-of-big-leap-in-time-to-first-token-performance/
- NVIDIA persistence mode docs — https://docs.nvidia.com/deploy/driver-persistence/persistence-mode-legacy.html
- Groq pricing — https://groq.com/pricing
- Groq Llama-3.3-70B benchmark — https://groq.com/blog/new-ai-inference-speed-benchmark-for-llama-3-3-70b-powered-by-groq
- Cerebras inference 3× — https://www.cerebras.ai/blog/cerebras-inference-3x-faster
- Cerebras pricing — https://www.cerebras.ai/pricing
- Cerebras docs pricing — https://inference-docs.cerebras.ai/support/pricing
- Artificial Analysis Llama-3.3-70B — https://artificialanalysis.ai/models/llama-3-3-instruct-70b
- Artificial Analysis providers — https://artificialanalysis.ai/models/llama-3-3-instruct-70b/providers
- Artificial Analysis Cerebras — https://artificialanalysis.ai/providers/cerebras
- morphllm vLLM benchmarks — https://www.morphllm.com/vllm-benchmarks
- morphllm LLM API comparison — https://www.morphllm.com/llm-api
- TokenMix Cerebras tests — https://tokenmix.ai/blog/cerebras-api-key-access-speed-tests-2026
- Snowflake Arctic Inference speculative — https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/
- Lookahead decoding paper — https://arxiv.org/pdf/2402.02057
- Continuous batching analysis — https://tianpan.co/blog/2026-04-09-continuous-batching-llm-inference
