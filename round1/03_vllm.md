# vLLM on GH200 — Round 1 Research

**Agent**: #03 / 20
**Date**: 2026-05-25
**Domain**: vLLM Grace-Hopper deployment, V1 engine, FA-3, KV offload to LPDDR5X, speculative decoding, quantization

---

## 1. Versions of record (May 2026)

| Channel | Latest tag | Date | Notes |
|---|---|---|---|
| PyPI `vllm` | 0.19.0 (0.19.1rc0 shipping) | 2026-04-03 | aarch64 wheels now feasible since PyTorch 2.11 |
| PyPI `vllm` (older stable) | 0.18.0 | 2026-03 | gRPC serving, GPU n-gram, FlexKV |
| NGC `nvcr.io/nvidia/vllm` | `26.01-py3` (vLLM 0.13.0) | 2026-01-31 | CUDA 13.1, FlashInfer 0.6.0 stable, TE 2.11 |
| Community GH200 image | `drikster80/vllm-gh200-openai:latest` | ~12 months old (stale) | aarch64+H100, builds FA/FlashInfer from source |
| Community GH200 image | `ghcr.io/abacusai/gh200-llm/llm-train-serve:latest` | active | XFormers + FA included |

vLLM V1 engine has been the default since v0.8.0 (Jan 2025). V0 is legacy in 2026 and should not be used for new deployments.

---

## 2. Install on GH200 (aarch64)

### 2.1 Preferred path — PyPI wheels (post-PyTorch 2.11, April 2026)

Starting with **PyTorch 2.11.0 (April 2026)**, default `pip install torch` on aarch64 Linux pulls a CUDA-enabled wheel. This finally resolves the years-long "vLLM on GH200 = build from source" tax.

```bash
# Assumes Ubuntu 22.04/24.04, CUDA 12.8+ or 13.x on host, Python 3.11
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio          # 2.11+, aarch64 CUDA wheel from default PyPI
pip install vllm                                   # 0.19.x; pulls aarch64 wheel
python -c "import torch, vllm; print(torch.cuda.is_available(), vllm.__version__)"
```

Caveat: vLLM aarch64 wheel coverage is recent. Some intermediate "fork" tags (e.g. 0.10.1-gptoss) never had ARM wheels; user reports on `discuss.vllm.ai` confirm only `x86_64` wheels for those. For the mainline 0.19.x release the wheel is now published; if it is missing for a given build, fall back to NGC container or source.

### 2.2 NGC container (most reliable, May 2026)

```bash
docker pull nvcr.io/nvidia/vllm:26.01-py3   # vLLM 0.13.0, CUDA 13.1, FlashInfer 0.6.0
# Newer monthly tags (26.02, 26.03, 26.04) ship newer vLLM; check catalog.ngc.nvidia.com.
docker run --gpus all --ipc=host --network=host \
  -v $HF_HOME:/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:26.01-py3 \
  vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000
```

Known issue against NGC 26.01: async-scheduler + long context can hit CUDA-graph crashes / warp exceptions. If you see them, try `--enforce-eager` or upgrade to a later NGC monthly. Source: NVIDIA dev-forum thread on `vllm:26.01-py3`.

### 2.3 Source build (only if wheels truly unavailable)

```bash
# CUDA 12.8+ toolchain on host, gcc 12+, ninja
pip install --index-url https://download.pytorch.org/whl/nightly/cu128 \
    torch torchvision torchaudio
git clone https://github.com/vllm-project/vllm && cd vllm
python use_existing_torch.py            # strips torch deps so the wheel build accepts your torch
pip install -r requirements/build.txt
MAX_JOBS=64 pip install -e . --no-build-isolation
```

For older releases use the documented Dockerfile build: `docker build --platform linux/arm64 --build-arg max_jobs=66 -f docker/Dockerfile .` Build takes ~25 min on a Grace 72-core, image ~6.9 GB.

---

## 3. Optimal launch flags for GH200

GH200 = 1× Hopper H100 die (96 GB HBM3) + 1× Grace 72-core ARM + 480 GB LPDDR5X over **NVLink-C2C at 900 GB/s** (≈ 7× PCIe Gen5). The optimal config is fundamentally different from H100 PCIe.

### 3.1 Baseline 70B-class (Llama-3.3-70B, Qwen2.5-72B) — bf16

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --cpu-offload-gb 60 \
  --swap-space 64 \
  --kv-cache-dtype auto
```

Per-flag rationale (May-2026 defaults in parens):

| Flag | Recommendation | Why for GH200 |
|---|---|---|
| `--tensor-parallel-size` | **1** for ≤72B bf16, 2+ only on GH200 NVL2 | Single Hopper die — don't shard unless you must |
| `--dtype` | **bfloat16** | FP8 has Hopper-specific long-context accuracy issues (see §5); bf16 is the safe default. Hard constraint from prior runs on this stack |
| `--gpu-memory-utilization` | 0.90-0.95 | Pre-allocates HBM for KV. 0.95 is aggressive but fine since Grace LPDDR is overflow |
| `--max-model-len` | 32k - 128k | Bounded by KV budget. Long context favors offload (see §4) |
| `--enable-prefix-caching` | **on** (V1 default since 0.8) | <1% throughput hit at 0% hit, multiplies at high hit |
| `--enable-chunked-prefill` | **on** (V1 default, do not disable) | Required for balanced TTFT/ITL; V1 scheduler is built around it |
| `--max-num-batched-tokens` | 8192 (throughput) / 2048 (ITL) | Higher = better TTFT and aggregate throughput, lower = better ITL |
| `--max-num-seqs` | 256-512 | GH200 has KV headroom for big batches; this is where the 7.6× win comes from |
| `--cpu-offload-gb` | **20-100** (key for GH200) | Spills weights to Grace LPDDR over C2C; frees HBM for more KV |
| `--swap-space` | 32-128 GB | CPU swap for preempted KV blocks; Grace LPDDR makes this nearly free |
| `--kv-cache-dtype` | `auto` (= bf16) | **Do not** set `fp8_e4m3` blindly — see §5 |
| `--async-scheduling` | **on by default** in 0.19 | Zero-bubble overlap between scheduling and GPU exec |

### 3.2 Optional flags worth knowing

- `--scheduling-policy {fcfs|priority}` (V1 new)
- `--logprobs-mode {raw_logprobs|processed_logprobs|raw_logits|processed_logits}` (V1)
- `-O2` (default optimization level; `-O3` more aggressive, `-O0` fastest startup)
- `--enable-expert-parallel` for DeepSeek-V3 / Qwen3-MoE
- `--speculative-config '{...}'` for EAGLE-3 / n-gram / MTP (see §6)

### 3.3 Newer KV-offload syntax (0.14+)

V1's preferred form replaces the older `--kv-transfer-config` JSON:

```bash
--kv_offloading_backend native --kv_offloading_size 80
```

Older equivalent (still works):
```bash
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks":<N>}}'
```

---

## 4. NVLink-C2C and KV offload to Grace LPDDR5X — what vLLM actually exploits

This is the central GH200 question. Findings:

1. **vLLM does NOT have a dedicated "NVLink-C2C-aware" code path.** The 2026-01-08 vLLM blog on the KV-offloading connector evaluates only H100 PCIe. The transfer mechanism is `cudaMemcpyAsync` via on-GPU DMA — same code path everywhere. Measured throughput in that post: 83.4 GB/s bidirectional with 2 MB blocks.
2. **It still wins big on GH200 — because the underlying CUDA copy rides C2C at 900 GB/s on Grace Hopper systems.** No vLLM code change needed; the OS/driver routes it. You get the bandwidth for free.
3. **Three offload mechanisms in vLLM, in increasing order of sophistication:**
   - `--cpu-offload-gb N` — pins N GB of model **weights** on CPU, streams per-layer into GPU. Fully transparent in V1 since 0.8.3. Best for fitting weights when HBM is tight.
   - `--swap-space N` — CPU-side swap for **preempted KV blocks**. Block-level eviction by scheduler.
   - `--kv_offloading_backend native --kv_offloading_size N` (0.14+) — proper **KV-cache offload connector** with async DMA, block-level placement, prefix-cache aware. This is what you want for long-context serving.
4. **LMCache + GH200**: For external multi-node CPU sharing or persistent prefix-cache pools, LMCache's `LMCacheConnectorV1` plugs into V1. It's the production-grade story; vLLM's native connector is leaner.
5. **NVIDIA's own GH200 KV-offload demo** (developer.nvidia.com, 2025-09-05) uses **RAPIDS Memory Manager (RMM)** with `managed_memory=True` and `change_current_allocator`, not vLLM directly. This shows raw unified-memory access for arbitrary PyTorch code; vLLM does not currently call RMM by default.

**Practical recipe for 70B on a single GH200 96 GB HBM with long context:**

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --cpu-offload-gb 80 \
  --kv_offloading_backend native --kv_offloading_size 200 \
  --enable-prefix-caching --max-num-seqs 128
```

This pushes ~80 GB of weights into Grace LPDDR over C2C and dedicates ~200 GB of LPDDR to KV overflow, keeping HBM almost fully usable for active KV.

---

## 5. FP8 on GH200 — FLAGGED, prefer bf16

Hard prior: prior FP8 runs on this GH200/Wan stack were broken. The 2026-04-22 vLLM blog ("The State of FP8 KV-Cache and Attention Quantization in vLLM", vllm.ai/blog/2026-04-22-fp8-kvcache) confirms the situation is messy on Hopper:

**Documented Hopper FP8 bug.** On long context, FP8 Flash-Attention-3 on Hopper accumulated into FP32 imprecisely. On a 128k needle-in-a-haystack test, accuracy collapsed from **91% (bf16) to 13% (FP8)**.

**Fix shipped.** A "two-level FP32 accumulation" patch (flash-attention#104) recovers accuracy to **89%**. Fixed in vLLM 0.19.1 timeframe. Break-even point where FP8 beats bf16 dropped from ~25k tokens (v0.10.2) to ~7k tokens (v0.19.1) for Llama-3.1-8B.

**New Hopper-specific cost from the fix.** For models with `head_dim = 256`, prefill (TTFT) is now ~**1.6× slower** than bf16 because of register pressure. For `head_dim ∈ {64, 128}` (Llama, Qwen, most), prefill is competitive.

**Recommendation for GH200 in May 2026:**
- **Default: bf16.** This matches the user constraint and is unconditionally correct.
- **If you must use FP8:** require vLLM ≥ 0.19.1, target decode-heavy workloads with ≥ ~7k context, and check `head_dim`. Skip sliding-window layers via `--kv-cache-dtype-skip-layers sliding_window` for hybrid-attention models (e.g. gpt-oss).
- **Never** use FP8 silently — log the version and run a 128k-needle sanity check before committing to it.

Cite: vLLM blog 2026-04-22.

---

## 6. Speculative decoding — what to enable

vLLM 0.19 made speculative decoding the default acceleration layer; the async scheduler now zero-bubble overlaps spec-decode rejection sampling with the next prefill.

Configuration (V1, JSON):

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct --dtype bfloat16 \
  --speculative-config '{"method":"eagle3","model":"<eagle3-draft-for-llama3.3-70b>","num_speculative_tokens":3}'
```

| Method | Speedup (low concurrency) | Notes |
|---|---|---|
| **EAGLE-3** | **1.6× - 2.6×** on 70B | Reuses target's internal features; lightweight draft head; best quality/perf balance. RedHat blog (2025-07) reports 1.6× on Llama-3.3-70B at low rate; falls off above ~16 concurrent |
| MEDUSA | 1.5× - 2× | Multiple draft heads; older approach, mostly superseded by EAGLE-3 |
| n-gram | 1.2× - 1.5× | Pure prompt-overlap; great for summarization/RAG/code-completion; **moved CPU→GPU in v0.18** |
| MTP (DeepSeek-V3 native) | ~1.8× at >80% acceptance | Use for DeepSeek-V3/V3.2 only |
| PARD / MLP | 1.5× - 2× | Newer; check vLLM recipes for support |

Concurrency interaction: speculative decoding pays best at **low-to-medium concurrency (1-200 simultaneous requests)**. At very high concurrency the GPU is already saturated; gains decay. RedHat April-2026 gpt-oss benchmark on H200: 20.7% throughput gain on ShareGPT, sweet spot at 2-3 draft tokens.

`num_speculative_tokens`: start at **3**, ablate 2/3/5. Higher is not better — verification cost rises.

---

## 7. Quantization matrix (Hopper / GH200, May 2026)

| Method | Hopper status | Use when |
|---|---|---|
| **bf16** | First-class | Default. Always works. Recommended baseline |
| **FP8 W8A8** (e4m3) | Conditional — needs vLLM ≥ 0.19.1 to be safe at long context; see §5 | Decode-heavy, head_dim ≤ 128, ≥ 7k ctx. Flag in production |
| **FP8 KV-cache** (`--kv-cache-dtype fp8_e4m3`) | Same caveat — accuracy fix lives in 0.19.1 | Doubles effective KV-cache capacity. Validate on your workload |
| **INT8 W8A8** | Stable | Cross-vendor fallback (AMD, A100). On Hopper FP8 generally wins because tensor cores are native |
| **AWQ-4** (W4A16, Marlin kernel) | Excellent — fastest 4-bit on Hopper | VRAM-bound. Reports of 10.9× speedup, ~741 tok/s vs bf16 baseline on small models |
| **GPTQ-4** (Marlin) | Stable, slightly behind AWQ-Marlin | Legacy checkpoints |
| **bitsandbytes 4-bit** | Works but **slated for deprecation** (RFC #39583, 2026-04) | Avoid for new deployments |
| **GGUF** | Works but deprecation RFC same as bnb | Avoid |

GH200's KV-offload-to-LPDDR usually eliminates the need to quantize KV — you can stay at bf16 KV and just spill to Grace memory. This is the major architectural advantage.

---

## 8. V1 vs V0 — what changed and why it matters

| Aspect | V0 (legacy) | V1 (default since 0.8) |
|---|---|---|
| Throughput | baseline | **1.7×** average |
| Chunked prefill | opt-in, model-dependent | **always on** |
| Prefix caching | optional, <1% hit-rate hit | **always on**, recomputed-only when logprobs requested |
| Scheduler | strict prefill/decode separation | unified budget `{request_id: num_tokens}`, FCFS or priority |
| Async overlap | none | scheduler runs in parallel with GPU step |
| Spec decode + async | sequential | zero-bubble overlap (v0.19+) |
| Long context | rough | "significant improvements" — V1 was designed for it |

For Qwen2.5-1.5B at high QPS, V1 has consistently lower latency than V0 (GitHub issue #17540). For 70B on GH200 the differential is even larger because V0's prefill/decode scheduling never balanced well with KV offload.

**There is no production reason to run V0 on GH200 in 2026.**

---

## 9. Benchmark numbers (GH200-specific, cited)

| Workload | Hardware | vLLM number | Source / Date |
|---|---|---|---|
| Llama-3.1-70B, no quant, max_seq=4096, TP=1, 60 GB CPU offload | 1× GH200 vs 1× H100 SXM | **7.6× higher throughput** on GH200; cost/token 8× lower (GH200 $0.02 vs H100 $0.16) | Lambda Labs blog, 2024-11-22 |
| Llama-3.1-70B, no quant | 1× GH200 | 4.33 tok/s end-to-end | same, vs 0.57 on H100 SXM |
| Llama-3.3-70B FP8, batch=32, ShareGPT | 1× GH200 96 GB vs H100 | **+32%** throughput on GH200 (TensorRT-LLM, but architecture story applies to vLLM) | Baseten blog |
| MoE / DeepSeek-V3, Wide-EP | 8× H200 | **2.2k tok/s/GPU** with Wide-EP (DP+EP combo) | vLLM blog 2025-12-17 |
| gpt-oss-120B + EAGLE-3, TP=1 | H200 141 GB PCIe | +20.7% on ShareGPT vs no-spec | RedHat 2026-04-16 |
| Llama-3.1-8B | 1× H100 | ~12,500 tok/s | morphllm 2026 benchmarks |
| FP8 break-even (Llama-3.1-8B) | Hopper | 7,010 tokens (post-fix) vs 24,889 (pre-fix) | vLLM blog 2026-04-22 |

**GH200-specific 2026 Llama-3.3-70B vLLM tok/s numbers do not exist in the public record I could find.** The 2024 Lambda Labs data is the canonical reference (different model version but same class). LLM-Inference-Bench (arxiv 2411.00136) includes GH200 vLLM throughput sweeps; the dashboard is the place to grab exact numbers for a specific (model, batch, seq-len) tuple.

---

## 10. Edge cases & traps on GH200

1. **aarch64 wheel coverage gaps.** Mainline `vllm` is fine; experimental forks (`vllm-gptoss`, model-specific tags) often skip ARM. Workaround: NGC container or source build with `use_existing_torch.py`.
2. **NGC 26.01-py3 async-scheduler bug** with long context — disable with `--no-async-scheduling` or upgrade to a later monthly tag.
3. **Single 70B on 96 GB HBM bf16** is *tight*: 140 GB weights + KV doesn't fit on-die. You **must** use `--cpu-offload-gb` (spill to Grace). With FP8 weights (70 GB) it fits with ~27 GB headroom, but see §5 about FP8.
4. **Tensor-parallel on a single GH200 die is pointless** (`--tensor-parallel-size 1`). On GH200 NVL2 (2-superchip) you can use TP=2 over NVLink, but EP is usually better for MoE.
5. **Performance regression report**: user on `discuss.vllm.ai/t/1320` reported 11-15 tok/s on a community ARM build of vLLM (gpt-oss model) vs 150 tok/s on llama.cpp same model. Suggests ARM-specific tuning is not fully landed on every release. Always benchmark before committing.
6. **`--enforce-eager`** disables CUDA graphs — use only to debug crashes. It will roughly halve throughput.
7. **Prefix caching + chunked prefill interaction**: only the first chunk hits the cache. For prompt-heavy/cache-heavy workloads, consider larger `--max-num-batched-tokens` so prompts fit in fewer chunks.
8. **Speculative decoding above ~200 concurrent requests** typically degrades throughput — the GPU is already saturated and rejection sampling becomes overhead. Profile before enabling at scale.
9. **DeepSeek-V3 / Qwen3-MoE**: use `--enable-expert-parallel` and DeepEP all-to-all kernels. Single GH200 fits ~37B-active 671B-total MoE only with heavy CPU offload; multi-GH200 is more realistic.
10. **bitsandbytes / GGUF**: avoid — RFC #39583 (April 2026) proposes deprecation, low usage (~0.5% / 0.1%).

---

## 11. Confidence & Gaps

**High confidence:**
- vLLM V1 is the only sane choice in 2026; V0 is legacy.
- bf16 is the correct default on GH200; FP8 has a documented Hopper accumulation bug fixed in 0.19.1 but introduces a head_dim=256 prefill penalty.
- `--cpu-offload-gb` works in V1 and is the primary lever for fitting large bf16 models on a single GH200.
- The 900 GB/s NVLink-C2C is exploited by vLLM **passively** through `cudaMemcpyAsync` — no special vLLM code path, but you get the bandwidth for free.
- aarch64 wheel install is much simpler since PyTorch 2.11 (April 2026).

**Medium confidence:**
- Exact tok/s numbers for Llama-3.3-70B on GH200 with vLLM 0.19. Public benchmarks are mostly TensorRT-LLM or older vLLM (0.6-0.10). The Lambda 2024 number (7.6× vs H100) is the canonical reference but predates V1 entirely; current V1 ratio likely higher.
- Whether vLLM 0.19+ auto-detects GH200 and tunes anything. No clear evidence of GH200-specific autotuning; you must set offload sizes manually.
- EAGLE-3 speedup for Llama-3.3-70B at GH200's typical batch sizes — RedHat's 1.6× number was at low concurrency on different hardware.

**Gaps / open questions:**
- **No published vLLM blog covers GH200 specifically.** All KV-offload benchmarks in vLLM blogs use H100. Need to run our own benchmark to confirm the 900 GB/s C2C actually delivers in production with V1's offload connector.
- **NGC monthly tag for May 2026** (`vllm:26.05-py3`?) not surfaced in search — check `catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm` directly.
- **FP8 status on GH200 specifically** (not just Hopper-class) — blog is about H100/B200. GH200 uses the same Hopper die so the fix should apply, but worth a sanity-check needle test.
- **GH200 NVL2 / NVL32 cluster behavior** — limited public vLLM data on multi-superchip configs; expert-parallel scaling is the open question for DeepSeek-V3.
- **LMCache + Grace LPDDR**: LMCache exposes a Connector V1 to vLLM but I did not find a public benchmark on GH200; recommended as a follow-up investigation.

---

## 12. Source URLs (dated)

- vLLM blog, "FP8 KV-Cache and Attention Quantization" — https://vllm.ai/blog/2026-04-22-fp8-kvcache — 2026-04-22 (FLAGGED reference for FP8)
- vLLM blog, "KV Offloading Connector" — https://vllm.ai/blog/2026-01-08-kv-offloading-connector — 2026-01-08
- vLLM blog, "vLLM V1: A Major Upgrade" — https://blog.vllm.ai/2025/01/27/v1-alpha-release.html — 2025-01-27
- vLLM blog, "Large Scale Serving: DeepSeek @ 2.2k tok/s/H200" — https://blog.vllm.ai/2025/12/17/large-scale-serving.html — 2025-12-17
- vLLM blog, "Triton Attention Backend Deep Dive" — https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive — 2026-03-04
- vLLM V1 guide — https://docs.vllm.ai/en/stable/usage/v1_guide/
- vLLM Optimization & Tuning — https://docs.vllm.ai/en/stable/configuration/optimization/
- vLLM FP8 quantization docs — https://docs.vllm.ai/en/latest/features/quantization/fp8/ — 2026-05-15
- vLLM Expert Parallel Deployment — https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- vLLM v0.19.0 release — https://github.com/vllm-project/vllm/releases/tag/v0.19.0 — 2026-04-03
- vLLM April 2026 release writeup — https://fazm.ai/blog/vllm-update-april-2026 — 2026-04
- PyTorch blog, aarch64 wheels — https://pytorch.org/blog/vllm-and-pytorch-work-together-to-improve-the-developer-experience-on-aarch64/ — 2026-04
- Lambda Labs, GH200 Llama-3.1-70B vLLM benchmark — https://lambda.ai/blog/putting-the-nvidia-gh200-grace-hopper-superchip-to-good-use-superior-inference-performance-and-economics — 2024-11-22
- Baseten, Llama-3.3-70B on GH200 (Lambda Cloud) — https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/
- NVIDIA, GH200 KV-offload with CPU-GPU memory sharing — https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/ — 2025-09-05
- vLLM aarch64 wheel forum thread — https://discuss.vllm.ai/t/unable-to-use-vllm-0-10-1-gptoss-on-gh200-aarch64-source-for-custom-wheel-not-available/1320
- NGC vLLM container catalog — https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
- NGC 26.01-py3 forum announcement — https://forums.developer.nvidia.com/t/new-ngc-vllm-container-image-vllm-26-01-py3/359213 — 2026-01-31
- drikster80 GH200 image — https://hub.docker.com/r/drikster80/vllm-gh200-openai
- abacusai gh200-llm — https://github.com/abacusai/gh200-llm
- vLLM speculative decoding docs — https://docs.vllm.ai/en/latest/features/speculative_decoding/
- RedHat, EAGLE-3 vLLM — https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding — 2025-07-01
- RedHat, gpt-oss spec decode 2026 — https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss — 2026-04-16
- RedHat, vLLM 0.8.1 V1 engine — https://developers.redhat.com/articles/2025/04/28/performance-boosts-vllm-081-switching-v1-engine
- morphllm 2026 vLLM benchmarks — https://www.morphllm.com/vllm-benchmarks
- vLLM Quantization overview — https://docs.vllm.ai/en/latest/features/quantization/
- vLLM BitsAndBytes deprecation RFC — https://github.com/vllm-project/vllm/issues/39583 — 2026-04
- LMCache KV offload quickstart — https://docs.lmcache.ai/getting_started/quickstart/offload_kv_cache.html
- LLM-Inference-Bench paper — https://arxiv.org/pdf/2411.00136 (GH200 vLLM throughput sweep dashboard)
- vLLM V1 "--cpu-offload-gb" thread — https://discuss.vllm.ai/t/the-new-v1-way-to-cpu-offload-gb/432
