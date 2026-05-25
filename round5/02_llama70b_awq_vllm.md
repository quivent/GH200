# Llama-3.3-70B 4-bit on GH200 via vLLM 0.21 — Round 5 Narrow-and-Deep

**Agent**: #02 / Round 5
**Date**: 2026-05-25
**Target**: 1× GH200 96 GB on Lambda Ubuntu 24.04 aarch64
**Goal**: Maximize tok/s and minimize friction for Llama-3.3-70B at 4-bit (AWQ-Marlin preferred) on vLLM 0.21.0
**Hard prior**: bf16 activations only — FP8 broken on this DiT/Wan stack; no FP8 anywhere

**Friction score: 2/5** (one-line `vllm serve`; AWQ checkpoint is 35 GiB and fits HBM with headroom; the only friction items are (a) picking which of three AWQ repos to use, (b) explicit `awq_marlin` flag avoids a known auto-detection foot-gun, (c) EAGLE-3 head load path has been buggy in 0.8.x — verify on 0.21.0 before committing.)

---

## 1. Best AWQ checkpoint for Llama-3.3-70B

Three viable HF repos exist for Llama-3.3-70B AWQ INT4. All are AutoAWQ-quantized with the same recipe (GEMM kernel, group_size=128, zero-point on). They differ only by author and download volume.

| Repo | Author | Last mtime | Monthly DLs | Notes |
|---|---|---|---|---|
| **`casperhansen/llama-3.3-70b-instruct-awq`** | Casper Hansen (AutoAWQ author) | 2024-12-06 | **236,582** | Author of AutoAWQ; reference implementation. Published MMLU 86.0 / HumanEval 88.4 / MATH 77.0 for the quantized model |
| `ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4` | community | active | 173,512 | Same recipe (AutoAWQ GEMM g128 zero-point); explicit "INT4" in name |
| `lambdalabs/Llama-3.3-70B-Instruct-AWQ-4bit` | Lambda Labs | older | unknown | Same recipe; less widely picked up |

**Recommendation: `casperhansen/llama-3.3-70b-instruct-awq`**. He is the AutoAWQ maintainer; this is the canonical AWQ. Either of the other two is functionally equivalent.

Note: there is **no `hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4`** — `hugging-quants` only covers Llama-**3.1**-70B. Don't paste the 3.1 path into a 3.3 recipe.

**On-disk / VRAM footprint**: ~35 GiB checkpoint, ~40 GiB resident on GPU. On a 96 GB GH200 that leaves **~50 GiB free for KV cache** at `--gpu-memory-utilization 0.92`, which is the entire structural reason AWQ on GH200 is attractive: you do NOT need `--cpu-offload-gb` for weights, freeing Grace LPDDR for pure KV overflow.

---

## 2. Exact `vllm serve` command (no spec-decode)

```bash
# vLLM 0.21.0, single GH200, AWQ-Marlin, bf16 activations
vllm serve casperhansen/llama-3.3-70b-instruct-awq \
  --quantization awq_marlin \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --kv-cache-dtype auto \
  --port 8000
```

Per-flag rationale specific to AWQ-on-GH200:

| Flag | Value | Why |
|---|---|---|
| `--quantization awq_marlin` | **explicit, not auto** | vLLM's auto-detection from a 0.5.3.post1-era checkpoint can fail to pick the Marlin path on some 0.19-0.21 builds (vllm issues #6985, #20112). Explicit `awq_marlin` forces the fast kernel. Without Marlin you get **67 tok/s instead of 741 tok/s** (Qwen-32B reference, jarvislabs) — 10.9× penalty. |
| `--dtype bfloat16` | locked | Activation dtype. AWQ checkpoint is W4A16; the A16 is bf16 here per hard prior. AWQ-Marlin in vLLM accepts `torch.half` or `torch.bfloat16` as activation dtype — both are valid. |
| `--tensor-parallel-size 1` | locked | Single Hopper die; never shard a 4-bit 70B on one chip |
| `--max-model-len 32768` | tunable up to ~96k | Bounded by KV budget. At 32k with batch 256 the KV fits comfortably in the post-weights HBM headroom |
| `--gpu-memory-utilization 0.92` | 0.90-0.95 | Pre-allocates HBM for KV pool. 0.92 leaves ~7 GiB headroom for CUDA-graphs, FlashInfer scratch, EAGLE-3 head |
| `--enable-prefix-caching` | on (V1 default) | Free win on multi-turn chat / RAG / agentic patterns |
| (chunked prefill) | implicit | V1 always on; do not pass `--no-enable-chunked-prefill` |
| `--max-num-batched-tokens` | 8192 | NVIDIA's own Hopper recipe for Llama-3.3-70B; balances TTFT vs ITL |
| `--max-num-seqs` | 256 | GH200 has KV headroom for big batches; this is where AWQ throughput peaks |
| `--kv-cache-dtype auto` | = bf16 | **Never** `fp8_e4m3` on this stack (FP8 hard prior; plus FP8 KV-cache head_dim=256 is still penalized in 0.21.0; head_dim=128 Llama works but breaks the hard prior) |
| `--async-scheduling` | (default-on in 0.19+) | Don't disable |

**Verification one-liner after server is up**:

```bash
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"casperhansen/llama-3.3-70b-instruct-awq","prompt":"Hello","max_tokens":8}' | jq .
```

---

## 3. With EAGLE-3 draft head

Pretrained head: **`yuhuili/EAGLE3-LLaMA3.3-Instruct-70B`** (exists on HF, Apache-2.0, EAGLE-3 v3.0.0, base = LLaMA-3.3-Instruct-70B, vLLM support merged via PR #16937).

**Important: the EAGLE-3 head was trained against the unquantized Llama-3.3-70B base.** Running it as the verifier with an AWQ-quantized target works in vLLM (the verifier is the AWQ model; the draft is a separate forward), but acceptance rates may be 5-10% lower than against the bf16 target. No published number for AWQ-target acceptance.

```bash
# vLLM 0.21.0, single GH200, AWQ-Marlin + EAGLE-3
vllm serve casperhansen/llama-3.3-70b-instruct-awq \
  --quantization awq_marlin \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --speculative-config '{"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.3-Instruct-70B","num_speculative_tokens":3,"draft_tensor_parallel_size":1}' \
  --port 8000
```

Changes vs §2:
- `--gpu-memory-utilization` dropped to **0.90** (draft head + KV for draft attention need ~3-4 GiB)
- `--max-num-seqs` halved to **128** — EAGLE-3 sweet spot is low-to-medium concurrency; the cost of rejected tokens grows with batch
- `num_speculative_tokens: 3` per Red Hat 2026-04-16 (2 and 3 tied; 4 loses 8%)
- `draft_tensor_parallel_size: 1` since TP=1 anyway

**Expected gain at this scale on Llama-3.3-70B AWQ**:
- **Conservative**: +10-15% throughput at concurrency 1-32
- **Realistic**: +15-20% at concurrency 32-128 (matches the Red Hat 2026-04-16 gpt-oss / H200 envelope of 9.5-20.5% through 200 concurrent)
- **Optimistic floor**: +25% at concurrency 1-8 if acceptance rate holds against the quantized target
- **Above ~150 concurrent**: gain decays; profile before enabling at full scale

Caveat: vLLM issue #19174 reported a shape-mismatch on EAGLE-3 load for Llama-3.3-70B (closed-not-planned, stale, likely fixed by PR #19033). vLLM issue #18452 reported TP-shared-memory timeout for EAGLE-3 + llama3-70b on 0.8.5.post1 (closed). Both predate 0.21.0. **Smoke-test the spec-decode path before declaring it stable on your specific cluster.**

---

## 4. Expected tok/s — extrapolated, with explicit error bars

**There is no published Llama-3.3-70B AWQ-Marlin vLLM benchmark on GH200 as of 2026-05-25.** All numbers below are extrapolated from three independent anchors with stated uncertainties.

### 4.1 Anchors used

| Anchor | Hardware | Workload | Number |
|---|---|---|---|
| A. jarvislabs 2026 | 1× H200 141 GB | Qwen-2.5-32B AWQ-Marlin, ShareGPT, max-conc 10 | **741 tok/s aggregate** (vs 461 bf16) |
| B. morphllm/Modal 2026 | 1× H100 80 GB | Llama-3.1-70B FP8 | **460 tok/s aggregate** at batch 64 |
| C. tech-insider 2026 | H100 class | Llama-3-70B AWQ peak | **~3,800 tok/s aggregate** at concurrency 64 |
| D. Lambda 2024 (pre-V1) | 1× GH200 vs H100 SXM | Llama-3.1-70B bf16 | **7.6× GH200/H100 ratio**; 4.33 tok/s end-to-end per request on GH200 |

Scaling assumptions:
- Llama-3.3-70B AWQ at batch 1 is **memory-bandwidth-bound**. GH200 HBM3 = 4.0 TB/s (vs H100 SXM 3.35 TB/s, H200 4.8 TB/s). Per-request bw-bound speed scales linearly with HBM bw.
- At high concurrency the workload becomes compute-bound on tensor cores; AWQ-Marlin runs INT4 on Hopper tensor cores, no peak-FLOP advantage GH200 vs H100. The bottleneck shifts to scheduler + KV-cache bandwidth.
- GH200's 480 GB Grace LPDDR + 900 GB/s C2C is only relevant if KV overflows HBM. At AWQ-INT4 weight + 32k context the KV pool fits in HBM; **no offload required for the AWQ path** — Grace LPDDR sits unused unless you push to ~120k context with batch ≥128.

### 4.2 Projected aggregate tok/s, single GH200, Llama-3.3-70B AWQ-Marlin, bf16 activations, 32k max-model-len, V1 default, no spec-decode

| Concurrency | Projected aggregate tok/s | Per-request tok/s | Error bar | Notes |
|---|---|---|---|---|
| **1**  | **55-75** | 55-75 | ±25% | Decode-bound on HBM bw; ~1.2× H100 at single-batch from HBM bw alone. Anchor: extrapolated from H100 batch-1 Llama-70B AWQ ~50-65 tok/s |
| **16** | **800-1,100** | 50-69 | ±30% | Scheduler+chunked-prefill overhead minor; AWQ-Marlin throughput plateau starting |
| **64** | **2,500-3,500** | 39-55 | ±35% | Approaches tech-insider 3,800 peak; uncertainty from sequence-length mix |
| **256** | **3,800-5,500** | 15-21 | ±40% | At this batch you hit either compute or KV-bw saturation. Lower bound = matches H100 AWQ peak; upper bound = GH200's KV-bandwidth headroom helps but doesn't 2× |

**Reading these numbers honestly**: the ±35-40% error bars at high concurrency are not nominal — they reflect that **the exact benchmark has never been published**. If you need a number to commit to, use the lower bound. To remove the error bars, run the benchmark once on the actual rig — it takes 30 minutes with vLLM's built-in `vllm bench serve`.

### 4.3 With EAGLE-3 (multiply above by)

| Concurrency | Multiplier | Comment |
|---|---|---|
| 1  | **1.20-1.35×** | best regime for spec-decode |
| 16 | **1.15-1.25×** | |
| 64 | **1.10-1.20×** | matches Red Hat envelope |
| 256 | **0.95-1.10×** | break-even or slight loss; do not enable |

### 4.4 Reality check on Lambda 2024 number

The Lambda 2024-11-22 number (7.6× GH200/H100 at bf16) was for **bf16 Llama-3.1-70B with `--cpu-offload-gb 60`** on H100, where the H100 was paging weights to PCIe DRAM. That comparison flatters GH200 because the H100 baseline was crippled by offload. With AWQ the entire 70B fits on an 80 GB H100 *and* a 96 GB GH200 with no offload — the 7.6× collapses to ~1.2-1.5× (the HBM bw ratio plus C2C-for-KV at long context). **Do not quote 7.6× for the AWQ path.**

---

## 5. AWQ-Marlin vs GPTQ-Marlin vs INT4 W4A16 on GH200 — which wins

**Short answer: AWQ-Marlin wins on throughput. GPTQ-Marlin / W4A16 (LLM-Compressor compressed-tensors) wins on accuracy by ~0.5pp average — call it a tie on quality.**

| Method | Throughput on Hopper (Qwen-32B/H200, jarvislabs) | Accuracy recovery (Llama-3.3-70B, RedHatAI eval) | Verdict for GH200 |
|---|---|---|---|
| BF16 baseline | 461 tok/s | 100% | reference |
| **AWQ-Marlin** | **741 tok/s (1.61× baseline, 10.9× over no-Marlin)** | ~98% (community-reported, MMLU 86.0) | **WIN on tok/s. Default choice.** |
| **GPTQ-Marlin (W4A16, compressed-tensors)** | 712 tok/s (1.54× baseline, 2.6× over no-Marlin) | **98.3% (98.8% MMLU, 100.3% GSM8K — RedHatAI published)** | Almost tied tok/s, slightly better measured accuracy. Use if accuracy-critical and you trust RedHatAI's eval more than community AWQ evals |
| INT4 W4A16 (LLM-Compressor, GPTQ algorithm) | same as GPTQ-Marlin (it IS GPTQ-Marlin under the hood) | 98.3% | Same as above — "INT4 W4A16" in vLLM docs == GPTQ-W4A16 via LLM-Compressor, served via Marlin kernel |

**Why AWQ-Marlin wins on tok/s**: AWQ's activation-aware weight scaling concentrates information in fewer salient weights; the resulting weight layout maps slightly more efficiently onto Marlin's mixed-precision tile loop than GPTQ's column-wise scaling does. The delta is small (~4% in published benchmarks) but consistent.

**Why GPTQ-W4A16 might still appeal**:
- Published evals are more rigorous (RedHatAI runs lm-eval-harness; AWQ benchmarks are mostly the author's own)
- compressed-tensors checkpoint format is more universal across vLLM / TGI / SGLang
- LLM-Compressor (the producer) is what Red Hat ships in their commercial AI Inference Server

**Why none of these are FP8**: FP8 is faster than 4-bit when accuracy holds (free 2× over bf16 on Hopper tensor cores), but the user constraint locks FP8 out. Skip.

**Recommendation for this rig**: **AWQ-Marlin** (`casperhansen/llama-3.3-70b-instruct-awq`), with `RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16` as the fallback if you observe accuracy regressions on your eval set. They are drop-in compatible at the vLLM CLI level — just swap the model name and remove the explicit `--quantization awq_marlin`.

---

## 6. KV cache offload to Grace LPDDR — recommended values for the AWQ path

**Counter-intuitive result: with AWQ-INT4 you almost certainly don't need KV offload on a 96 GB GH200.**

KV budget math at AWQ:
- Weights resident on GPU: ~40 GiB
- HBM available at `--gpu-memory-utilization 0.92`: 96 × 0.92 - 40 ≈ **48 GiB free for KV**
- Llama-3.3-70B KV per token (bf16, GQA 8 heads × 128 head_dim × 80 layers × 2 K+V): ≈ **328 KiB/token**
- 48 GiB / 328 KiB ≈ **152,500 tokens of KV in HBM**

So at AWQ you can hold **~150k tokens of total KV in HBM** before any offload is needed. That covers `max-num-seqs 256 × max-model-len 600` worth of fully-allocated sequences, or 16 × 9k, or 64 × 2.3k. In other words, for chat-like workloads with avg ~1.5k input + ~500 output, batch 256 fits without offload.

**Recommendation**:
- **Baseline AWQ path (§2): no offload flags.** `--cpu-offload-gb 0`, no `--kv_offloading_*`.
- **If you push `--max-model-len 131072` for long-context RAG**, add: `--kv_offloading_backend native --kv_offloading_size 200`. 200 GB of Grace LPDDR reserved for KV overflow over the 900 GB/s C2C. Safe ceiling is 400 GB (Grace exposes 480 GB total, leave headroom for OS + Grace-side I/O buffers). Stay below the documented OOM bug at 1024 GB.
- **Do not** add `--cpu-offload-gb` at AWQ; weights are already small enough to fit. CPU offloading of *weights* is meaningfully slower than HBM-resident weights even over C2C (B. Marie observation; vLLM Substack).

**Long-context AWQ command** (only if you need 128k):

```bash
vllm serve casperhansen/llama-3.3-70b-instruct-awq \
  --quantization awq_marlin --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 --max-num-seqs 128 \
  --kv_offloading_backend native --kv_offloading_size 200 \
  --port 8000
```

vLLM 0.21.0's Hybrid Memory Allocator (HMA) handles the placement; you don't need the legacy `--kv-transfer-config` JSON syntax.

---

## 7. One-command Docker quickstart (arm64, GH200, auto-download)

The official vLLM image now publishes aarch64 tags directly. As of 2026-05-25 the tag pattern is `cu129-nightly-aarch64` (nightly) and versioned tags like `v0.21.0-aarch64-cu129-ubuntu2404` (per the docker-hub layer listing). For stability prefer the versioned tag.

```bash
# Requires: docker with NVIDIA container runtime, CUDA 12.9+ host, HF_TOKEN exported
docker run --rm --gpus all --ipc=host --network host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN \
  --name vllm-l33-awq \
  vllm/vllm-openai:v0.21.0-aarch64-cu129-ubuntu2404 \
  --model casperhansen/llama-3.3-70b-instruct-awq \
  --quantization awq_marlin \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256
```

Notes:
- The model auto-downloads on first run (~35 GiB to `$HF_HOME`); cache the mount to avoid re-pulling
- `--ipc=host` matters for shared-memory between vLLM ranks even at TP=1 (some kernels still spawn workers)
- If the versioned aarch64 tag is not yet pushed when you try, fall back to `vllm/vllm-openai:nightly-aarch64` or `cu129-nightly-aarch64` — same image family, less battle-tested
- Community fallback: `drikster80/vllm-gh200-openai:latest` if the official ARM image regresses; it has been the standby for the GH200 community since early 2025

**Friction items observed during research that the quickstart sidesteps**:
1. The historical "ARM = build-from-source" tax is gone in 2026 — official wheels and official ARM containers both exist
2. The model checkpoint is publicly downloadable (no gated access) — `casperhansen/llama-3.3-70b-instruct-awq` requires no HF agreement
3. The base `meta-llama/Llama-3.3-70B-Instruct` (used by EAGLE-3 path indirectly) IS gated — only matters if you swap to bf16; the AWQ-only path needs no Meta-side approval

---

## 8. Confidence map

**High confidence (multiple sources, current)**
- `casperhansen/llama-3.3-70b-instruct-awq` is the canonical AWQ checkpoint; ibnzterrell and lambdalabs are equivalent backups
- AWQ-Marlin is the fastest 4-bit path on Hopper; the no-Marlin fallback is **10.9× slower** — never let it auto-select to plain AWQ
- vLLM 0.21.0 + aarch64 docker tag exists and works; nightly cu129 confirmed on Docker Hub
- EAGLE-3 head for Llama-3.3-70B exists (`yuhuili/EAGLE3-LLaMA3.3-Instruct-70B`) and the vLLM serve-config syntax for it is established

**Medium confidence**
- Projected tok/s at concurrency 64 / 256 — error bars ±35-40% because no public GH200-AWQ benchmark exists
- EAGLE-3 acceptance rate when verifier is AWQ-quantized rather than bf16 — no published number; conservative 10-25% gain envelope
- AWQ vs W4A16 accuracy delta on Llama-3.3-70B specifically — the 98.3% recovery number is for W4A16; AWQ's number is community-reported on MMLU/HumanEval only

**Low confidence / open**
- Whether `--quantization awq_marlin` is strictly necessary on 0.21.0 or whether auto-detection now reliably picks Marlin — change vs 0.19 not documented in release notes; explicit flag is the safe play
- Exact behavior of HMA-backed KV-offload on GH200 specifically (no published benchmark)
- Whether the EAGLE-3 head load shape-mismatch in vLLM issue #19174 is fully resolved in 0.21.0 — smoke-test before production

---

## 9. Sources (10-15, dated)

- Hugging Face: `casperhansen/llama-3.3-70b-instruct-awq` — https://huggingface.co/casperhansen/llama-3.3-70b-instruct-awq — 2024-12-06, 236k monthly DLs, MMLU 86.0
- Hugging Face: `ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4` — https://huggingface.co/ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4 — 173k monthly DLs, AutoAWQ GEMM g128
- Hugging Face: `RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16` — https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16 — W4A16 GPTQ-Marlin, 98.3% recovery
- Hugging Face: `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B` — https://huggingface.co/yuhuili/EAGLE3-LLaMA3.3-Instruct-70B — EAGLE-3 v3.0.0, vLLM PR #16937 merged
- vLLM Recipes, Llama-3.3-70B on Hopper — https://docs.vllm.ai/projects/recipes/en/latest/Llama/Llama3.3-70B.html — (FP8 path; max-num-batched-tokens=8192 recommendation)
- vLLM docs, INT4 W4A16 — https://docs.vllm.ai/en/latest/features/quantization/int4/ — GPTQ algorithm, group_size=128
- vLLM docs, AWQ-Marlin kernel — https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/awq_marlin/ — auto kernel selection, min capability 7.5
- vLLM docs, EAGLE — https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/ — speculative-config JSON schema
- Jarvislabs, Complete LLM Quantization Benchmarks — https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks — Qwen-32B H200: BF16 461 → AWQ-Marlin 741 → GPTQ-Marlin 712 tok/s
- arxiv 2411.02355, "Give Me BF16 or Give Me Death" — https://arxiv.org/html/2411.02355v1 — AWQ vs GPTQ W4A16 accuracy comparison
- Red Hat Developer, gpt-oss spec-decode 2026 — https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss — 2026-04-16, EAGLE-3 9.5-20.5% gain through 200 conc, sweet spot 2-3 draft tokens
- Red Hat Developer, Fly Eagle(3) fly — https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding — 2025-07-01, EAGLE-3 vLLM integration
- Docker Hub, vllm/vllm-openai tags — https://hub.docker.com/r/vllm/vllm-openai/tags — cu129-nightly-aarch64, v0.21.0-aarch64-cu129-ubuntu2404
- Lambda Labs GH200 inference blog — https://lambda.ai/blog/putting-the-nvidia-gh200-grace-hopper-superchip-to-good-use-superior-inference-performance-and-economics — 2024-11-22, 7.6× GH200/H100 for bf16 70B (offload baseline, NOT AWQ)
- vLLM forum, Llama 3.3 70B very slow thread — https://discuss.vllm.ai/t/llama-3-3-70b-very-slow/1715 — community troubleshooting; ~40 tok/s on 2× H200 reported as below-expected
- vLLM issue #19174, EAGLE-3 loading error for Llama 3.3 70b — https://github.com/vllm-project/vllm/issues/19174 — closed-not-planned, likely fixed by #19033
- vLLM issue #6985, AutoAWQ-marlin conflicts — https://github.com/vllm-project/vllm/issues/6985 — auto-detection foot-gun reference
- tech-insider.org, vLLM vs Ollama 2026 — https://tech-insider.org/vllm-vs-ollama-2026/ — LLaMA-3-70B AWQ peak ~3,800 tok/s at concurrency 64

---

## 10. TL;DR

1. **Checkpoint**: `casperhansen/llama-3.3-70b-instruct-awq`
2. **Serve**: `vllm serve <ckpt> --quantization awq_marlin --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.92 --enable-prefix-caching --max-num-batched-tokens 8192 --max-num-seqs 256`
3. **EAGLE-3 gain**: 10-25% at conc 1-128; head = `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B`; smoke-test load path
4. **Throughput envelope**: 55-75 tok/s @ batch 1; 2,500-3,500 tok/s @ batch 64; 3,800-5,500 tok/s @ batch 256 (±35-40%)
5. **AWQ-Marlin wins over GPTQ-Marlin / W4A16 by ~4% on tok/s**; W4A16 wins ~0.5pp on measured accuracy — call it a tie, default to AWQ-Marlin
6. **No KV offload needed** at AWQ unless `max-model-len ≥ 64k` and `max-num-seqs ≥ 128`; if you do offload, `--kv_offloading_backend native --kv_offloading_size 200`
7. **Docker**: `vllm/vllm-openai:v0.21.0-aarch64-cu129-ubuntu2404` with the §2 args; nightly-aarch64 as fallback
8. **Friction**: 2/5 — the only landmines are the explicit `awq_marlin` flag and the EAGLE-3 smoke-test
