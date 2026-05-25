# SGLang on NVIDIA GH200 (Grace Hopper) — Round 1 Research

**Agent**: #04 / 20
**Date**: 2026-05-25
**Domain**: SGLang inference server on GH200 aarch64 superchip
**Constraint carried forward**: FP8 is broken on this GH200/Wan stack — recommend bf16. Any FP8 claim is flagged with source.

---

## 1. TL;DR (executive summary)

- **SGLang now works on GH200 aarch64**, but only with recent versions (>= v0.5.x). The old `sgl-kernel==0.1.0` story was the blocker that bit early adopters; current `sgl-kernel >= 0.3.12` ships official aarch64 wheels, and `0.3.21` (released 2026-01-14) is current.
- **Use the official multi-arch Docker image** (`lmsysorg/sglang:latest-cu130` or `latest-cu129`) — Docker Hub publishes arm64 manifests for v0.5.12.post1. This is by far the path of least pain on GH200.
- **Caveat on aarch64**: SGLang's Dockerfile JIT-compiles a subset of CUDA kernels (sgl-flash-attn3) at first launch on arm64 because the upstream ships no arm cubins. Expect a slow first start. (See Dockerfile comments.)
- **RadixAttention** is SGLang's headline feature and the reason to pick it over vLLM/TRT-LLM: 50–99% cache hit rates measured on chat/RAG workloads, up to 6.4× throughput vs vLLM on prefix-heavy traffic.
- **PD disaggregation is production-ready** in May 2026 (Mooncake transport recommended, NIXL supported, ASCEND for Huawei). Multi-node DeepSeek setups use `--disaggregation-mode prefill|decode` + router.
- **Speculative decoding**: EAGLE/EAGLE3/MTP/DFLASH/NGRAM all supported. On Llama 3.1 8B / H100, EAGLE-3 gets 158 → 373 tok/s (2.36×) single-stream. On H200, DeepSeek MTP yields 1.8× decode at batch 1 / 1.5× at batch 32.
- **Quantization on GH200**: bf16 is the safe default. FP8 (modelopt_fp8) is *advertised* for Hopper/sm90, but the user's prior experience says FP8 is broken on this stack — and GitHub has open Hopper FP8 bugs (RoPE shape mismatch with DeepSeek-V4-Flash-FP8, FlashAttention3 + fp8_e5m2 silent triton fallback). **Stay on bf16.**
- **No published end-to-end GH200 benchmark from LMSYS** for SGLang. Closest data point: Baseten/Lambda Feb-2025 showed *GH200 vs H100* 32% throughput edge on Llama-3.3-70B FP8 — but they used TRT-LLM as the engine and only SGLang's bench tool. Treat SGLang-on-GH200 throughput as inferred from H100/H200 numbers + memory-bandwidth uplift.

---

## 2. Install on GH200 aarch64

### 2.1 Preferred path — Docker (multi-arch image)

GH200 host is aarch64 + sm_90 + (typically) 64K-page kernel. Docker Hub publishes arm64 manifests:

```bash
# CUDA 13 (default in v0.5.12+) — recommended
docker pull lmsysorg/sglang:latest-cu130

# CUDA 12.9 (use if you need CUDA 12 ecosystem compatibility)
docker pull lmsysorg/sglang:latest-cu129

# Pinned production tag
docker pull lmsysorg/sglang:v0.5.12.post1-cu130
```

Launch:

```bash
docker run --gpus all --shm-size 32g --ipc=host \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  lmsysorg/sglang:latest-cu130 \
  python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 30000 \
    --dtype bfloat16
```

Expected aarch64 oddity: the Dockerfile comment states `aarch64: kernels-community/sgl-flash-attn3 ships no arm variants; JIT-compile at runtime. Remove this branch once arm cubins are published upstream.` First request after a cold start will trigger JIT compile.

### 2.2 pip install on bare metal aarch64

If you must install on bare metal (NGC PyTorch container, slurm node, etc.):

```bash
# Required: torch == 2.9.1 (per sgl-kernel 0.3.21 metadata), CUDA 13 toolkit
pip install --upgrade pip uv

# Install kernel first from the SGLang wheel index — picks aarch64 wheel automatically
uv pip install sgl-kernel \
  --index-url https://docs.sglang.io/whl/cu130

uv pip install sglang==0.5.12
```

The wheel resolves to `sgl_kernel-0.3.21-cp310-abi3-manylinux2014_aarch64.whl` (per PyPI metadata).

### 2.3 The historical failure mode you may still hit

Pre-`sgl-kernel 0.3.12` had no aarch64 wheel. The classic GH200 failure (issue #13304, discussion #13303):

- `sgl-kernel==0.1.0` no aarch64 wheel → build from source → OOM-kills the compute node even with `MAX_JOBS=2`.
- `flashinfer-python` wants `nvidia-cudnn-frontend>=1.13.0`, NGC PyTorch 25.06 ships 1.12.0.

**Fix**: pin `sgl-kernel>=0.3.12` (you get 0.3.21 today), pin to SGLang v0.5.x, and prefer the Docker image to avoid the cudnn-frontend conflict entirely.

### 2.4 Verify install

```bash
python3 -c "import sglang, sgl_kernel; print(sglang.__version__, sgl_kernel.__version__)"
# Expect: 0.5.12 0.3.21 (or newer)
nvidia-smi  # confirm GH200 sm_90 visible
```

---

## 3. Launch configs — copy/paste recipes

### 3.1 Single GH200, Llama-3.1-8B bf16 (the safe default)

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 30000 \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --enable-torch-compile \
  --max-running-requests 256
```

Why these flags on GH200:
- `--dtype bfloat16`: avoids the broken FP8 path.
- `--mem-fraction-static 0.88`: GH200 has 96 GB HBM3e (96 GB SKU) or 144 GB HBM3e (144 GB SKU) plus the LPDDR5x via NVLink-C2C — leave headroom for the radix tree and activations.
- `--enable-torch-compile`: now stable with the V2 overlap scheduler.

### 3.2 Llama-3.1-70B bf16 on a single GH200 144 GB

70B in bf16 needs ~140 GB weight + KV — fits *only* on the 144 GB SKU and tightly. Most users will offload via HiCache (L2 = host LPDDR over NVLink-C2C) or shard across 2 GH200s.

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --dtype bfloat16 \
  --mem-fraction-static 0.92 \
  --enable-hierarchical-cache \
  --hicache-ratio 1.5
```

### 3.3 Multi-GH200 over NVLink switch (NVL32 / 2-node TP=16)

```bash
# On node 0 (rank 0)
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --dtype bfloat16 \
  --tp-size 16 \
  --nnodes 2 --node-rank 0 \
  --dist-init-addr ${MASTER_IP}:5000 \
  --enable-dp-attention \
  --moe-a2a-backend deepep

# On node 1 (rank 1) — same command, --node-rank 1
```

NVL32 (32 GH200 via NVLink switch) gives a single 19.5 TB unified memory domain and was designed precisely for AllReduce-heavy TP — but SGLang's collective layer is plain NCCL; no special NVL32 codepath. You get the NVLink switch bandwidth automatically.

### 3.4 PD-disaggregated (prefill/decode split) — production deployment

Per `docs.sglang.io/advanced_features/pd_disaggregation.html`:

```bash
# Prefill worker on GH200 #0
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0 \
  --dtype bfloat16 \
  --port 30000

# Decode worker on GH200 #1
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0 \
  --dtype bfloat16 \
  --port 30001

# Router
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://${PREFILL_IP}:30000 \
  --decode  http://${DECODE_IP}:30001 \
  --host 0.0.0.0 --port 8000
```

KV transfer: Mooncake (recommended; NVLink + IntraNode + TCP) or NIXL (UCX/LIBFABRIC). Both use GPUDirect RDMA on GH200's ConnectX NICs. The May-2026 0.5.12 release adds a **decode-side radix cache** specifically for the disaggregated workflow.

For heterogeneous TP (e.g., TP=8 prefill + TP=16 decode), set:
```bash
export SGLANG_DISAGG_STAGING_BUFFER=1
export SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=64
export SGLANG_DISAGG_STAGING_POOL_SIZE_MB=4096
```
Docs claim 2–5× throughput vs the per-token-slice default. **Not safe for MLA models** (DeepSeek V2/V3/V4) per the docs.

### 3.5 Speculative decoding (EAGLE-3) on bf16

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path lmsys/EAGLE3-Llama-3.1-8B \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 8
```

### 3.6 Structured output (XGrammar + jump-forward)

XGrammar is the default backend in 0.5.x. Send a request:

```bash
curl http://localhost:30000/generate -d '{
  "text": "Output a JSON product spec for a laptop.",
  "sampling_params": {
    "json_schema": "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}}}",
    "max_new_tokens": 256
  }
}'
```

Jump-forward kicks in automatically for fixed strings in the grammar. Reported overhead < 40 μs/token (XGrammar paper).

---

## 4. Benchmark numbers (with citations)

### 4.1 H100 reference (closest published, since no clean GH200/SGLang numbers exist)

**Spheron, 2026-03-23 — Llama-3.3-70B FP8 on H100 SXM5 80GB** (vLLM 0.18.0 / TRT-LLM 1.2.0 / **SGLang 0.5.9**):

| Concurrency | vLLM tok/s | TRT-LLM tok/s | SGLang tok/s |
|---|---|---|---|
| 1   | 120  | 130  | 125 |
| 10  | 650  | 710  | 680 |
| 50  | 1,850 | 2,100 | 1,920 |
| 100 | 2,400 | 2,780 | 2,460 |

TTFT p50 at conc=100: vLLM 740 ms / TRT-LLM 680 ms / SGLang 710 ms. Cold start: vLLM 62s, **TRT-LLM 28 min compile + 90s reload**, SGLang 58s.

Note this is FP8 — not directly usable for our bf16 GH200 target, but ordering (TRT-LLM > SGLang > vLLM by ~5–15%) holds.

**Multiple-source aggregate, Llama-3.1-8B on H100 80GB**:
- SGLang ~16,200 tok/s vs vLLM ~12,500 tok/s aggregate throughput (~29% lead).
- Prefix-heavy (RAG / multi-turn): SGLang up to **6.4× vLLM**.
- DeepSeek-V3 specifically: SGLang **3.1× vLLM** (MLA backends: FA3 / FlashInfer / FlashMLA / CutlassMLA).

### 4.2 H200 numbers (a closer proxy to GH200 — same Hopper chip, more HBM3e)

- DeepSeek MTP via EAGLE on H200: **1.8× decode at batch=1, 1.5× at batch=32**.
- DeepSeek-V4-Flash 285B single-batch decode on H200: **266 → 240 tok/s** across 4K → 900K context. (Source: LMSYS DeepSeek-V4 day-0 blog, 2026-04-25.)
- Flash Compressor kernel: up to 80% of peak memory bandwidth on H200. Lightning TopK: 100 μs → 15 μs at bs=1.

### 4.3 EAGLE/EAGLE-3 single-stream speedups (H100, Llama-3.1-8B)

From `docs.sglang.io/advanced_features/speculative_decoding.html`:

| Mode | tok/s |
|---|---|
| Baseline | 158.34 |
| EAGLE-2  | 244.10 (+54%) |
| EAGLE-3  | 373.25 (+136%) |

### 4.4 GH200 — what we actually have

- **Baseten / Lambda, 2025-02-07**: Llama-3.3-70B FP8 on GH200 96GB vs H100 80GB, ShareGPT bench, batch 32. GH200 **+32% throughput vs H100**. Engine: TRT-LLM. (SGLang was used only as the *bench harness*, not the runtime.)
- **No public LMSYS or sgl-project benchmark for SGLang-on-GH200** has been published in any form I could find. This is a gap.
- Extrapolated estimate (bf16, Llama-3.1-8B, single GH200 144GB, batch 100):
  - SGLang H100 baseline ≈ 2,460 tok/s @ FP8 → bf16 ≈ 60–70% of FP8 → ~1,500–1,700 tok/s
  - GH200 HBM3e bandwidth ≈ 4.8 TB/s vs H100 SXM 3.35 TB/s = 1.43×
  - Estimate: **~2,200–2,500 tok/s bf16 @ conc=100**. Treat as ±25%.

### 4.5 HiCache + Mooncake (no GH200 numbers, A10/H800 only)

`kvcache-ai.github.io/Mooncake/performance/sglang-hicache-benchmark-results-v1.html` shows pre-populated Mooncake L3 keeps TTFT flat across rounds while L2-only blows up once the conversation history exceeds host RAM. Concrete numbers are not extractable from the page rendering. Qwen3-235B-A22B on 8×H800 with 760 GB Mooncake pool.

---

## 5. RadixAttention behaviour notes

Source: LMSYS blog 2024-01-17 + production deployment guides 2026.

- **Mechanism**: GPU-side radix tree shared across all concurrent requests. Each node is a token-span with its KV; LRU-evicted under pressure.
- **Zero-config**: caching is automatic; SGLang detects prefix sharing from actual traffic.
- **Hit rates in production**:
  - LLaVA-Next-34B: **52.4%**
  - Vicuna-33B: **74.1%** → 1.7× lower TTFT mean
  - Few-shot: 85–95%
  - Multi-turn chat: 75–90%
  - Code analysis: 60–80%
  - Mixed prod: 50–70%
- **2026 extension — HiCache** (LMSYS 2025-09-10): L1=GPU HBM, L2=host RAM, L3=Mooncake/Mooncake-store/SSD-offload. On GH200, L2 is *especially* attractive: the LPDDR5x is on-package via NVLink-C2C at ~900 GB/s, so eviction to host costs much less than over PCIe. Enable with `--enable-hierarchical-cache --hicache-ratio 1.5`.
- **Multimodal**: EPD (Encode-Prefill-Decode) Disaggregation announced 2025-12-23 — decouples vision encoders from LM nodes, Mooncake RDMA zero-copy transfer of embeddings.

---

## 6. Quantization on GH200 — bf16 is the answer

| Method | Flag | GH200 (sm_90) status |
|---|---|---|
| **bf16** | `--dtype bfloat16` | **Recommended.** Stable, accurate, full performance. |
| FP8 (online) | `--quantization fp8` | Advertised. **Carrying user constraint: broken on this stack.** Multiple open Hopper bugs in 2026: RoPE shape mismatch with DeepSeek-V4-Flash-FP8 + EAGLE (issue #24650); `fa3 + fp8_e5m2` silent fallback to slower triton (issue #18550); Qwen3.6 FP8 weight_scale_inv silently dropped (issue #23687). **Source: GitHub issue list, May 2026.** |
| ModelOpt FP8 | `--quantization modelopt_fp8` | "Optimized for Hopper+ architectures" (docs). Same FP8 cautions. |
| AWQ INT4 | (auto-detected from model) | Works on Hopper. Use offline pre-quantized model (e.g. `Meta-Llama-3.1-8B-Instruct-AWQ-INT4`). |
| GPTQ INT4 | (auto-detected) | Works on Hopper. |
| INT8 W8A8 | `--quantization w8a8_int8` | Works on Hopper via sgl-kernel CUTLASS int8 path. |
| FP4 / modelopt_fp4 | `--quantization modelopt_fp4` | **Blackwell only (sm_100+). N/A on GH200.** |
| GGUF | (auto-detected) | Works but slower than native paths. |

**Net recommendation for GH200**: bf16 unless you have a known-good pre-quantized INT4 (AWQ) or INT8 W8A8 model and have validated correctness. Avoid FP8 until you have isolated the user's reported breakage.

---

## 7. Multi-GH200 / NVLink switch

- Single GH200 has no need for tensor parallel (one GPU). TP is only relevant in multi-GH200 (NVL32 / 2-node NVLink) deployments.
- SGLang multi-node uses standard `--tp-size`, `--nnodes`, `--node-rank`, `--dist-init-addr` over NCCL (PR #550 lineage). No NVL32-specific codepath; you inherit NVLink switch bandwidth automatically.
- NVL32 (32× GH200, 19.5 TB unified memory) is positioned by NVIDIA as 2× HGX H100 for inference — but the public SGLang launches I see in 2026 (DeepSeek V3/V4 cookbooks) target B200/H200/GB200, not GH200 NVL32. **The GH200 NVL32 + SGLang combination is documented as supported but not benchmarked publicly.**
- DeepEP (`--moe-a2a-backend deepep`) is the all-to-all backend for MoE; GH200 is supported.

---

## 8. Position vs vLLM and TRT-LLM (May 2026)

| Aspect | SGLang | vLLM | TRT-LLM |
|---|---|---|---|
| Raw throughput H100 70B-FP8 conc=100 | 2,460 tok/s | 2,400 | **2,780** |
| Prefix-heavy (RAG/multi-turn) | **6.4× vLLM** | baseline | comparable |
| DeepSeek-V3/V4 | **3.1× vLLM**, official rec. | works | works |
| Cold start | 58s | 62s | **28 min compile** + 90s |
| Structured output | **~3× vLLM** (XGrammar+jumpfwd) | XGrammar | proprietary |
| PD disaggregation | **Mature** (Mooncake/NIXL) | newer | yes (Dynamo) |
| Spec decoding variants | EAGLE/EAGLE3/MTP/DFLASH/NGRAM/SpecV2 | EAGLE/MTP | EAGLE/MTP |
| aarch64 / GH200 | Multi-arch Docker, JIT kernels first run | Multi-arch Docker | NGC container |
| Model-update flexibility | High | **Highest** | Low (recompile) |

**When to pick SGLang on GH200**: chat / RAG / multi-turn / agentic workloads (any shared-prefix pattern); DeepSeek; structured outputs / tool calling; mixed traffic with bursty long contexts (HiCache helps).
**When to skip**: single-prompt batch jobs with no prefix sharing (vLLM is simpler, throughput is within 5%) or you need the absolute fastest throughput on one frozen model and can afford TRT-LLM's compile-time tax.

---

## 9. Confidence & Gaps

### High confidence
- Install path on GH200 aarch64 with Docker `latest-cu130` works (multi-arch manifest confirmed on Docker Hub, May 2026).
- `sgl-kernel >= 0.3.12` has official aarch64 wheels (PyPI metadata).
- RadixAttention numbers and cache-hit ranges are well-documented across LMSYS blog + production guides.
- PD disaggregation is GA in 0.5.x with Mooncake/NIXL; CLI flags above are exactly as in `docs.sglang.io`.
- bf16 is the safe path on GH200; user's FP8-broken constraint is corroborated by multiple open Hopper FP8 bugs in May 2026.
- EAGLE/EAGLE-3 single-stream numbers (158 → 373 tok/s on H100/Llama-8B) are from official SGLang docs.

### Medium confidence
- GH200 vs H100 +32% throughput figure — Baseten/Lambda blog, but the engine was TRT-LLM and the workload was FP8. Should not be projected onto SGLang-bf16 without validation.
- HiCache benefit on GH200 specifically — strong theoretical fit because NVLink-C2C makes host-memory L2 cheap, but I found no published GH200-specific HiCache benchmark.
- Multi-GH200 NVL32 + SGLang — claimed to work, no public benchmark.

### Gaps (need round 2)
- **No public SGLang benchmark on GH200**: not from LMSYS, not from NVIDIA, not from any cloud provider. This is the single biggest gap. Recommend round-2 agents run an in-house bench: Llama-3.1-8B bf16 and Llama-3.1-70B bf16, ShareGPT replay, 1/10/50/100/200 concurrency, prefix-shared and unique-prompt variants.
- **Exact bf16 vs FP8 delta on GH200** for SGLang — unknown. Until FP8 is fixed, irrelevant; but worth measuring once FP8 issue is root-caused.
- **GH200 first-launch JIT compile time** for sgl-flash-attn3 on aarch64 — known to happen, not quantified anywhere I found. Need empirical measurement (and a warm-cache strategy for production).
- **HiCache L2 throughput over NVLink-C2C** on GH200 — would expect 800–900 GB/s effective, but no measurement.
- **NVL32-specific scaling efficiency** under SGLang TP=16 / TP=32 — unbenchmarked publicly.
- **EAGLE3 + GH200** specific numbers (only H100 data publicly).

### Watch list (active bugs as of May 2026)
- Issue #24650: DeepSeek-V4-Flash-FP8 + EAGLE → RoPE shape mismatch crash under load on Hopper.
- Issue #18550: `--kv-cache-dtype fp8_e5m2 --attention-backend fa3` silently falls back to triton.
- Issue #23687: Qwen3.6-27B-FP8 silent garbage output (weight_scale_inv dropped).
- Issue #14327: cuda 13 sgl-kernel wheel inconsistent version string fail-to-install (relevant if pinning cu130).

---

## Sources

Primary docs:
- [SGLang docs — Install](https://docs.sglang.io/get_started/install.html)
- [SGLang docs — PD Disaggregation](https://docs.sglang.io/advanced_features/pd_disaggregation.html)
- [SGLang docs — Speculative Decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html)
- [SGLang docs — Quantization](https://docs.sglang.io/advanced_features/quantization.html)
- [SGLang docs — Structured Outputs](https://docs.sglang.io/advanced_features/structured_outputs.html)
- [SGLang docs — HiCache Design](https://docs.sglang.io/advanced_features/hicache_design.html)
- [SGLang GitHub README](https://github.com/sgl-project/sglang)
- [SGLang Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile)
- [sgl-kernel on PyPI](https://pypi.org/project/sgl-kernel/) — 0.3.21 released 2026-01-14, aarch64 wheel
- [SGLang wheel index](https://github.com/sgl-project/whl) — PEP 503 index, cu118/cu124/cu128/cu129/cu130/rocm
- [lmsysorg/sglang Docker Hub tags](https://hub.docker.com/r/lmsysorg/sglang/tags) — confirms arm64 manifests for v0.5.12.post1

GH200 / aarch64 specific issues:
- [Issue #13304 — install failure on GH200 (sgl-kernel 0.1.0 no aarch64)](https://github.com/sgl-project/sglang/issues/13304)
- [Discussion #13303 — GH200 install discussion](https://github.com/sgl-project/sglang/discussions/13303)
- [Issue #3769 — sgl-kernel for aarch64 tracking](https://github.com/sgl-project/sglang/issues/3769)
- [Issue #14327 — cu13 sgl-kernel version string inconsistency](https://github.com/sgl-project/sglang/issues/14327)

FP8 / Hopper bug evidence (supports user's "FP8 broken" constraint):
- [Issue #24650 — DeepSeek-V4-Flash-FP8 RoPE shape crash with EAGLE](https://github.com/sgl-project/sglang/issues/24650)
- [Issue #18550 — fa3 + fp8_e5m2 silent triton fallback](https://github.com/sgl-project/sglang/issues/18550)
- [Issue #23687 — Qwen3.6 FP8 weight_scale_inv silently dropped](https://github.com/sgl-project/sglang/issues/23687)

Benchmarks:
- [Spheron — vLLM vs TRT-LLM vs SGLang on H100, 2026-03-23](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)
- [Particula — SGLang vs vLLM 2026](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison)
- [LMSYS — RadixAttention & SGLang launch blog, 2024-01-17](https://www.lmsys.org/blog/2024-01-17-sglang/)
- [LMSYS — SGLang HiCache blog, 2025-09-10](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)
- [LMSYS — DeepSeek-V4 day-0 with SGLang, 2026-04-25](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [Mooncake × SGLang HiCache benchmark](https://kvcache-ai.github.io/Mooncake/performance/sglang-hicache-benchmark-results-v1.html)
- [Baseten/Lambda — Llama-3.3-70B FP8 on GH200, 2025-02-07](https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/)
- [HPC-AI — SGLang DeepSeek speculative decoding tutorial 1.4×](https://company.hpc-ai.com/blog/sglang-speculative-decoding-tutorial)

NVIDIA references:
- [NVIDIA — SGLang Release Notes index](https://docs.nvidia.com/deeplearning/frameworks/sglang-release-notes/index.html) — releases 26.01 through 26.04
- [NVIDIA — GH200 NVL32 low-latency inference blog](https://developer.nvidia.com/blog/low-latency-inference-chapter-2-blackwell-is-coming-nvidia-gh200-nvl32-with-nvlink-switch-gives-signs-of-big-leap-in-time-to-first-token-performance/)
- [NVIDIA — Introducing GH200 NVL32](https://developer.nvidia.com/blog/one-giant-superchip-for-llms-recommenders-and-gnns-introducing-nvidia-gh200-nvl32/)

Misc:
- [XGrammar GitHub](https://github.com/mlc-ai/xgrammar)
- [SqueezeBits — Guided decoding perf vLLM vs SGLang](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang)
