# Llama-3.3-70B 4-bit on SGLang 0.5.12 / GH200 96GB — Round 5

**Agent**: #03 (Round 5 narrow & deep)
**Date**: 2026-05-25
**Target**: 1× GH200 96GB on Lambda Ubuntu 24.04 aarch64, `lmsysorg/sglang:latest-cu130`
**Carried constraints**: FP8 broken on this stack → 4-bit (W4A16) instead of bf16 for 70B; use subagents in parallel.
**Friction estimate**: **3 / 5**. Docker path is smooth; first-launch JIT and EAGLE-3-with-AWQ are the remaining sharp edges.

---

## 0. TL;DR

1. **Format**: pick **AWQ-INT4** as the checkpoint; SGLang auto-promotes it to **`awq_marlin`** on Hopper (sm90). That fused path is the documented fast lane (Marlin/FlashInfer W4A8 MoE kernels on Hopper landed in v0.5.12; W4A16 awq_marlin has shipped since 0.5.x). **GPTQ-INT4 also works** (auto-promotes to `gptq_marlin`) — both formats land on Marlin/Machete on H100/H200/GH200, so the practical perf delta is small. AWQ is recommended on quality grounds (~95% retention vs GPTQ ~90% on most evals).
2. **Best HF checkpoint**: `ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4` (~35 GiB weights, w_bit=4 / group=128 / GEMM / zero_point=True). GPTQ alternative: `shuyuej/Llama-3.3-70B-Instruct-GPTQ`.
3. **70B at 4-bit fits TP=1 on a 96 GB GH200** (35 GiB weights + ~50 GiB usable KV at mem_fraction 0.88).
4. **First-launch JIT on aarch64 is the friction**: Docker layer skips the `sgl-flash-attn3` cubin download (no arm variant upstream), so first cold start JIT-compiles those kernels. Plan for **5–10 min cold start** for the SGLang stack itself; add **+15–25 min** if you enable `--enable-torch-compile` (inductor autotune). Pre-warm by mounting a persistent `/root/.cache/{huggingface,torch,torchinductor_root,flashinfer}` volume and run once.
5. **EAGLE-3 + AWQ is real but not bulletproof**: the canonical 70B draft `lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge` was trained against the bf16 target. Running EAGLE-3 against a 4-bit target is supported by `--speculative-draft-model-path` + `--speculative-draft-quantization unquant`, but issue #4351 (AWQ-target + EAGLE shape mismatch in CUDA-graph capture) is documented and **the workaround is `--disable-cuda-graph`** at the cost of ~30–40% decode throughput. Best move: enable EAGLE-3 only after confirming baseline AWQ works, and start with `--speculative-eagle-topk 1 --speculative-num-steps 3 --speculative-num-draft-tokens 4`.
6. **HiCache L2 over NVLink-C2C** is the GH200's killer feature. Use `--enable-hierarchical-cache --hicache-ratio 4 --hicache-write-policy write_through --hicache-mem-layout page_first_direct --hicache-io-backend kernel`. Grace has 480 GB LPDDR5x; with 35 GiB of weights eating into a 96 GB HBM pool, ratio 4 (≈ 200 GB host KV) is conservative and well within budget. C2C practical bandwidth: ~375 GB/s H→D, ~297 GB/s D→H — ~5× cheaper than PCIe H100 offload.

---

## 1. Which 4-bit format SGLang prefers — AWQ vs GPTQ vs Marlin

### 1.1 What SGLang actually does on Hopper

`docs.sglang.io/advanced_features/quantization.html` (verified 2026-05-25) lists every NVIDIA-supported method. **AWQ and GPTQ checkpoints auto-promote to their Marlin variants** when SGLang detects an Ampere/Hopper GPU; this is automatic and not toggled by `--quantization`. The relevant rows:

| Method | --quantization flag | NVIDIA | Notes |
|---|---|---|---|
| AWQ | (auto from config) | yes | Triton dequant on AMD; **CUDA Marlin kernels on NVIDIA** |
| GPTQ | (auto from config) | yes | Same — promoted to gptq_marlin |
| awq_marlin | `awq_marlin` | yes (CUDA-only) | Hopper fast path for AWQ |
| gptq_marlin | `gptq_marlin` | yes (CUDA-only) | Hopper fast path for GPTQ |
| W4A8 MoE Marlin/FlashInfer | (auto) | yes | **New in v0.5.12** — Hopper specific |
| modelopt_fp8 / NVFP4 | flag | yes | broken on this stack (carried constraint) |
| W4A4 MegaMoE | (auto, MoE only) | yes | dense 70B → not applicable |

**Key v0.5.12 release-note line confirming the Hopper fast path**:
> "Marlin/FlashInfer W4A8 MoE kernels on Hopper: [#24816] [#24986]"
> "FlashInfer SM90 cutlass MXFP4 MoE backend (W4A16) for GPT-OSS + DSv4: [#24816]"

Llama-3.3-70B is a dense model, so W4A8/W4A4 MoE paths do not apply. The W4A16 awq_marlin path is the one we ride.

### 1.2 Quality and throughput ranking (third-party 2026 data)

- AWQ retains ~95% of base quality vs GPTQ ~90% (theaiengineer 2026, Spheron 2026).
- Raw kernel throughput: Marlin-AWQ ~741 tok/s vs vanilla AWQ ~68 tok/s (10.9×); Marlin-GPTQ ~712 tok/s vs vanilla GPTQ ~276 tok/s (2.6×). (Spheron AWQ Quantization Guide, 2026.)
- Marlin was originally Ampere-targeted; **Machete** (Hopper-native mixed-precision GEMM) is the H100/H200/GH200 successor. SGLang transparently uses Machete on Hopper via the same `awq_marlin` / `gptq_marlin` path when the underlying vLLM compressed-tensors / sgl-kernel layer is current (Red Hat Machete intro, Oct 2024; vLLM 0.6.2+).
- On H100, Machete on Llama-3.1-70B INT4 hits ~5 user req/s at <250 ms TTFT / <100 ms TPOT (Red Hat). Same kernel runs on GH200 sm90.

### 1.3 Best HF checkpoint recommendation

**Primary: `ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4`**
- Verified config: `w_bit=4`, `q_group_size=128`, `zero_point=True`, `version=GEMM`.
- ~35 GiB weights; HF model card explicitly documents SGLang launch.
- Calibration dataset not disclosed in card — flag if you need provenance.

**Alternates**:
- `shuyuej/Llama-3.3-70B-Instruct-GPTQ` (GPTQ-INT4, runs through gptq_marlin).
- `nvidia/Llama-3.3-70B-Instruct-NVFP4` (NVFP4 — **Blackwell only, not for GH200**; included only as a confusion-blocker).

**Verdict**: Use **`ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4`** as the primary. Have `shuyuej/Llama-3.3-70B-Instruct-GPTQ` as a fallback if you hit any AWQ-specific bug. **Do not specify `--quantization`** — let SGLang parse it from `config.json`; the docs explicitly warn that combining a pre-quantized checkpoint with `--quantization` triggers online re-quant.

---

## 2. Exact launch command

The recipe below is for `lmsysorg/sglang:latest-cu130` (multi-arch; auto-selects linux/arm64 on GH200), single GH200 96GB.

```bash
mkdir -p ~/sglang_cache/{hf,torch,torchinductor,flashinfer,triton}

docker run --gpus all --shm-size 32g --ipc=host \
  -p 30000:30000 \
  -v ~/sglang_cache/hf:/root/.cache/huggingface \
  -v ~/sglang_cache/torch:/root/.cache/torch \
  -v ~/sglang_cache/torchinductor:/root/.cache/torchinductor_root \
  -v ~/sglang_cache/flashinfer:/root/.cache/flashinfer \
  -v ~/sglang_cache/triton:/root/.cache/triton \
  --env HF_TOKEN=$HF_TOKEN \
  --env TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor_root \
  --env FLASHINFER_CUDA_ARCH_LIST="9.0a" \
  lmsysorg/sglang:latest-cu130 \
  python3 -m sglang.launch_server \
    --model-path ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4 \
    --host 0.0.0.0 --port 30000 \
    --dtype float16 \
    --tp-size 1 \
    --mem-fraction-static 0.88 \
    --chunked-prefill-size 8192 \
    --schedule-conservativeness 1.0 \
    --max-running-requests 256 \
    --cuda-graph-max-bs 256 \
    --context-length 32768 \
    --attention-backend fa3 \
    --enable-hierarchical-cache \
    --hicache-ratio 4 \
    --hicache-write-policy write_through \
    --hicache-mem-layout page_first_direct \
    --hicache-io-backend kernel
```

### Flag rationale (each one earned its place)

- **`--dtype float16`**. AWQ checkpoints dequantize to fp16 activations (W4A16). bf16 is *technically* permitted but the AWQ kernels prefer fp16; explicit avoids a silent activation-dtype mismatch.
- **`--tp-size 1`**. Single GH200; tensor parallel is unused at 35 GB weights on a 96 GB part.
- **`--mem-fraction-static 0.88`**. Standard hyperparameter-tuning advice (`docs.sglang.io/.../hyperparameter_tuning.html`) is "raise by 0.01 until OOM." On a 96 GB GH200 with ~35 GiB weights, 0.88 reserves ~12 GB for activations + CUDA-graph buffers and leaves ~50 GB for KV. **Lower to 0.82** if you trip OOM on long-context bursts. Raise to 0.92 if you don't need HiCache headroom.
- **`--chunked-prefill-size 8192`**. Current default. Long-context (>32k) workloads should drop to 4096 to avoid TTFT spikes; short-context workloads can leave at default.
- **`--schedule-conservativeness 1.0`**. Default. **Bump to 1.3** if you see `retracted_reqs` warnings in logs (KV-pool saturation). **Drop to 0.3** if `available_gpu_mem` stays high and queue depth grows.
- **`--max-running-requests 256`** and **`--cuda-graph-max-bs 256`**. CUDA-graph coverage to 256 catches the "batch 256" target the user asked about. Each extra step in `cuda-graph-max-bs` costs ~5–10 MB; 256 is achievable here.
- **`--context-length 32768`**. Llama-3.3-70B-Instruct native context is 128k; capping at 32k contains KV-cache blow-up at high concurrency. Raise if you need it; budget another ~20 GB per doubling of max context at concurrency 64+.
- **`--attention-backend fa3`**. FlashAttention-3 on Hopper sm90; gives the best throughput on H100/H200/GH200. (FlashInfer is the alternative; both work, fa3 is the SGLang default on Hopper and slightly faster for dense models.)
- **HiCache flags**. See §7.

### Optional `--enable-torch-compile`

torch.compile gives ~5–15% throughput uplift on Llama-3.1/3.3 in SGLang's V2 scheduler, **at the cost of 15–25 min of first-run inductor autotune** (SGLang Discussion #16048, 2026). Default `max-autotune-no-cudagraphs` mode is what SGLang uses. **Add `TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor_root --enable-torch-compile` only after you've cached the compiled artifacts to host.** Skip on first deploy.

---

## 3. RadixAttention + 4-bit interaction — caveats

RadixAttention is a software data structure over KV pages and is **fully orthogonal to weight quantization**. Specifically:

- Hit rate is a property of traffic shape, not weight format. Published 50–99% range (chat 75–90%, RAG/code 60–80%) ports 1:1 to 4-bit.
- KV cache itself is **not** 4-bit by default. With AWQ-INT4 weights, KV pages remain fp16/bf16 (W4A16 means weights are 4-bit, activations + KV stay 16-bit). To get 8-bit KV you must add `--kv-cache-dtype fp8_e5m2` or `auto` (Hopper-only), but issue #18550 (silent triton fallback with fa3 + fp8_e5m2) means this is **not recommended on this stack** today. Keep KV at fp16.
- Each cache hit on GH200 is worth more than on H100: GH200 HBM3e ≈ 4.8 TB/s vs H100 SXM5 ≈ 3.35 TB/s ≈ 1.43× the bandwidth — every saved decode step recoups that bandwidth.
- **Composes with HiCache** (HiRadixTree); no extra flag needed.
- **Composes with speculative decoding** without interference: the radix cache stores the verified KV; draft tokens that get rejected do not pollute the cache.

**Net**: no special tuning is needed for RadixAttention with AWQ. Don't disable it.

---

## 4. JIT compile-time on aarch64 first launch — measured cost & pre-warm

### 4.1 What gets JIT'd on arm64

Verified by reading the upstream Dockerfile (`sgl-project/sglang/docker/Dockerfile`, line ~765):

```
if [ "$(uname -m)" = "aarch64" ]; then
  echo "Skipping kernels-community/sgl-flash-attn3 cubin download on aarch64..."
  success=1;
else
  # x86_64 path downloads prebuilt cubins
fi
```

With comment: *"aarch64: kernels-community/sgl-flash-attn3 ships no arm variants; JIT-compile at runtime. Remove this branch once arm cubins are published upstream."*

Three sources of cold-start JIT on GH200:
1. **sgl-flash-attn3** cubins (compiled on first attention launch).
2. **FlashInfer cubins** — recent flashinfer change requires `FLASHINFER_CUDA_ARCH_LIST` to be set or it downloads cubins at runtime per request shape (issue #1352, #10933, #2527). Set `FLASHINFER_CUDA_ARCH_LIST="9.0a"` in env.
3. **Triton kernels** in flashinfer/triton paths — compiled per `(M, N, K)` tile shape on demand.

### 4.2 Measured numbers (caveat: no clean SGLang-on-GH200 publication)

| Source | Workload | Measured cold start |
|---|---|---|
| SGLang Discussion #16048 (Gemma 3 12B, no compile) | x86_64 baseline | ~1:30 (1.5 min) |
| SGLang Issue #9867 (DeepGEMM v2 on multi-node, large) | x86_64 distributed | **~23 min prefill, ~11 min decode** — extreme, MoE+EP-specific |
| Round-1 doc, generic guess | Llama 8B, x86 | ~58 s |
| Inferred for Llama-3.3-70B AWQ on GH200 aarch64 | first cold launch, no torch.compile | **5–10 min** (model weight download + 4-bit weight unpack + sgl-flash-attn3 JIT + flashinfer cubin first-use). Error bar: ±2× given no public publication. |
| Same + `--enable-torch-compile` | inductor autotune kicks in | **+15–25 min**. |

### 4.3 Pre-warm recipe

```bash
# Mount these dirs as named volumes (already shown in §2 docker run).
# 1. HF weights are cached on first download.
# 2. FlashInfer cubin cache (avoid runtime cubin downloads).
# 3. torchinductor / triton compile caches.

# Inside the container, BEFORE the first real launch, run:
python3 -m flashinfer download-cubin   # if available in this flashinfer version
# or simply: do one warmup run of launch_server, then exit, then real launch.

# Set in env to force cubins to local cache rather than re-downloading:
export FLASHINFER_CUDA_ARCH_LIST="9.0a"
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor_root
export TRITON_CACHE_DIR=/root/.cache/triton
```

Once warmed and cached on host volume, **second launches reduce to ~60–90 s** (weight load + CUDA graph capture only).

### 4.4 Honest gap

I could not find a single first-party publication that measures end-to-end first-launch JIT for SGLang on GH200 aarch64. The 5–10 min figure is an extrapolation from (a) the x86 baseline (~1.5 min for similar size), (b) the documented arm64-skips-cubin branch, (c) known flashinfer cubin-download bug behavior, and (d) the 70B 4-bit weight-unpack step. **Confidence: medium.** First in-house measurement on a real GH200 should pin this down.

---

## 5. Expected tok/s at batch 1, 16, 64, 256 — error bars and reasoning

### 5.1 Anchor points (third-party measured)

| Anchor | Hardware | Format | tok/s | Source |
|---|---|---|---|---|
| Llama-3.3-70B FP8 H100 SXM5 80GB | SGLang 0.5.9 | FP8 | 125 / 680 / 1920 / 2460 (conc 1/10/50/100) | Spheron 2026 |
| Llama-3.3-70B INT4 H100 with Machete | vLLM (Red Hat) | W4A16 | ~5 user req/s @ <100 ms TPOT (=~ TPOT-only ~10 tok/s/user) | Red Hat Machete |
| Llama-3.3-70B FP8 H200 single batch=1 | TRT-LLM | FP8 | ~430 tok/s decode | NVIDIA H200 launch |
| Llama-3.3-70B FP8 GH200 vs H100 batch=32 | TRT-LLM | FP8 | GH200 **+32%** vs H100 (Baseten/Lambda) | Baseten 2025-02 |
| Llama-3.1-8B aggregate H100 | SGLang | bf16 | ~16,200 tok/s | Particula 2026 |

### 5.2 Bandwidth-bound extrapolation logic

Llama-3.3-70B at INT4 = ~35 GiB weights. Decode is memory-bound and reads (weights + KV) per token. Roughly:

- Single-stream peak ≈ (HBM BW) / (weights read per token) = 4.8 TB/s / 35 GB ≈ **137 tok/s upper bound at batch=1**, before KV reads. In practice you read ~half the KV touched per layer + overhead → realistic batch=1 is ~50–70% of this ceiling.
- GH200 vs H100 = 4.8 / 3.35 = **1.43× HBM BW**. Memory-bound regions scale at that rate.
- 4-bit vs FP8 = 4-bit weights are half the bytes of FP8. **Decode at batch=1 scales nearly 2×** vs FP8 because the weight-stream dominates the bandwidth. Compute-bound batch=256 sees little to no INT4 advantage over FP8.

### 5.3 Predicted SGLang 0.5.12 / AWQ-INT4 / GH200 96GB / TP=1

| Batch (concurrent users) | Predicted aggregate tok/s | Per-user tok/s | Error bar | Reasoning |
|---|---|---|---|---|
| **1** | **80–105** | 80–105 | ±20% | Memory-bound. Anchor: H100 FP8 batch=1 ≈ 125; INT4 weight stream halves bandwidth ⇒ ~+30%; GH200 ×1.43 over H100; awq_marlin/Machete imperfect at bs=1 (some kernel overhead vs ideal). |
| **16** | **900–1,300** | 56–81 | ±25% | Sweet spot for Marlin/Machete (16–32 batch is its design center). Per-user TPOT drops slightly; aggregate scales near-linearly. |
| **64** | **2,300–3,200** | 36–50 | ±30% | Mixed regime — weight reads amortize, KV reads dominate. RadixAttention boost amplifies on shared-prefix workloads; if unique prompts, stay at low end. |
| **256** | **3,800–5,500** | 15–21 | ±35% | Compute starts to bite. INT4 W4A16 advantage narrows; throughput approaches FP8 ceiling. KV cache pressure may force `--mem-fraction-static` drop or `--chunked-prefill-size` tune. |

**Why the error bars are wide**: no public SGLang-on-GH200 publication exists (verified Round 1 + Round 3); the 70B-INT4-on-Hopper Machete numbers are H100, not GH200; and `awq_marlin` kernel performance on aarch64 GH200 specifically may differ from x86 H100 by a few %.

**Reality check**: if you measure batch=1 < 60 tok/s, suspect (a) `--enable-torch-compile` is *off* and you're paying overhead, or (b) the AWQ path didn't auto-promote to `awq_marlin`. Check the SGLang log line "Using quantization method: awq_marlin" on startup.

---

## 6. Speculative decoding — EAGLE-3 on 4-bit Llama-3.3-70B

### 6.1 The recipe

The first-party draft model exists: **`lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge`** (LMSYS, SpecForge-trained on 1.4M `mlabonne/open-perfectblend` samples, 2 epochs, updated ~Apr 15 2026). The official launch command in the model card is for **bf16 target**:

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --tp 4
```

### 6.2 Adapting to AWQ-INT4 target

The docs explicitly call out the override:

> "Quantization method for the draft model. Use `unquant` to force no quantization even when the target model is quantized."

So on a 4-bit target:

```bash
docker run --gpus all --shm-size 32g --ipc=host -p 30000:30000 \
  -v ~/sglang_cache:/root/.cache \
  --env HF_TOKEN=$HF_TOKEN \
  --env SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  --env FLASHINFER_CUDA_ARCH_LIST="9.0a" \
  lmsysorg/sglang:latest-cu130 \
  python3 -m sglang.launch_server \
    --model-path ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4 \
    --host 0.0.0.0 --port 30000 \
    --dtype float16 \
    --tp-size 1 \
    --mem-fraction-static 0.82 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge \
    --speculative-draft-quantization unquant \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --attention-backend fa3
```

### 6.3 Known sharp edges

- **Issue #4351 (open as of late 2025)**: "Speculative Decoding Fails with AWQ Quantized Model" — shape mismatch `(4×12288) × (8192×4096)` during EAGLE draft CUDA-graph capture against Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4. Workaround: `--disable-cuda-graph`. Cost: ~30–40% of decode throughput. Verify on your target before launch. If you hit it, try the GPTQ checkpoint instead — the bug was reported on AWQ specifically.
- **Lower `--mem-fraction-static` to 0.82**. The draft model adds ~3 GB weights + draft KV; do not run with 0.88.
- **`--speculative-eagle-topk 1`** is the SpecV2 overlap scheduler requirement; matches the model card.
- **Per the EAGLE-3 paper, 4.1–6.5× speedup at temperature 0** on academic benchmarks; SGLang docs claim "up to 4.48× end-to-end" via SpecForge drafters. On 70B + INT4 on GH200, expect **2–3× decode tok/s lift at batch=1**, declining to **1.3–1.6× at batch=64**, near 1× at batch=256 (verification overhead dominates).
- **RadixAttention is unaffected**. Verified KV is cached normally.

### 6.4 Alternative: NGRAM (no draft model)

If EAGLE-3 + AWQ trips bugs, fall back to NGRAM (no separate model, CUDA-only):

```bash
--speculative-algorithm NGRAM
```

Expect smaller speedups (1.2–1.5×) on chat workloads but zero AWQ compatibility risk.

---

## 7. HiCache L2 over NVLink-C2C — Grace LPDDR config

### 7.1 GH200 NVLink-C2C facts

- **Theoretical**: 450 GB/s each direction. **Practical** (Verda Jan-2026 STREAM measurements, corroborated by arXiv 2408.11556v2):
  - H→D (host→device, cache restore): **~375 GB/s** (83% of peak).
  - D→H (device→host, eviction): **~297 GB/s** (66% of peak).
- Latency (cross-C2C ping-pong): 150–400 ns.
- **5–6× cheaper than PCIe-attached H100 host offload (~63 GB/s).**
- Grace LPDDR5x: 480 GB capacity per superchip.

### 7.2 HiCache flags and recommended values

From `docs.sglang.io/advanced_features/hicache_design.html` (verified 2026-05-25):

| Flag | Recommended value (GH200 96GB + 70B-AWQ) | Why |
|---|---|---|
| `--enable-hierarchical-cache` | (set) | Master switch. |
| `--hicache-ratio` | **4** | Host KV pool = 4× device KV pool. With ~50 GB device KV → ~200 GB host pool, well within 480 GB LPDDR. Conservative; can go to 6–8 for very long contexts. |
| `--hicache-size` | (unset; use ratio) | If you want absolute control: `--hicache-size 200` (GB). |
| `--hicache-write-policy` | **`write_through`** | C2C BW is so high that write-through to L2 is cheap and gives strongest hit-rate. Switch to `write_back` only if you observe C2C contention at extreme concurrency. |
| `--hicache-mem-layout` | **`page_first_direct`** | Direct GPU↔host page transfers; best fit for C2C unified memory. `layer_first` is for PCIe legacy. |
| `--hicache-io-backend` | **`kernel`** | GPU-assisted copy kernel; up to 3× transfer speed per Mooncake design. `direct` is the cudaMemcpy fallback. |
| `--hicache-storage-backend` | **omit** (no L3) | L3 only adds value with Mooncake/3FS/NIXL across a cluster. Single-GH200 single-tenant = L1+L2 only. |
| `--hicache-storage-prefetch-policy` | `best_effort` (default) | Only relevant if L3 is enabled. |
| `--page-size` | default | Tune only if you see fragmentation. |

### 7.3 Expected benefit on GH200

LMSYS HiCache blog (2025-09-10) cites "up to 6× throughput, up to 80% TTFT reduction" and Novita "average TTFT dropped 56%, cache hit rate 40%→80%" on H20-3e and H800 clusters. **No GH200 measurement is published.** Inferential argument:

- GH200's C2C is **5–6× the effective BW of H100 PCIe** for the L2 path → eviction and restore cost is far lower than H100 reference.
- Therefore the **HiCache hit-rate vs throughput curve is dramatically flatter on GH200**: hits stay almost free even when KV spills heavily into LPDDR.
- For chat/RAG/multi-turn (the workloads where RadixAttention already pays), HiCache turns "evicted = recompute" into "evicted = restore at 375 GB/s" → expect **+30–60% additional TTFT improvement** on top of RadixAttention's baseline. Throughput delta is workload-dependent but real.

**Honest gap**: this is architectural inference. Run the SGLang HiCache benchmark suite on the actual GH200 to pin numbers — that's the in-house measurement still missing from Round 1 / Round 3 / Round 5.

---

## 8. Friction estimate: 3 / 5

| Phase | Friction | Notes |
|---|---|---|
| Pull Docker image | **1/5** | Single command, multi-arch. |
| Download AWQ checkpoint | **1/5** | HF cache on volume. ~35 GiB. |
| First launch cold start | **3/5** | 5–10 min JIT on aarch64 with no public number to confirm. Add 15–25 min if torch.compile. |
| `awq_marlin` auto-promotion | **1/5** | Just don't set `--quantization`. |
| EAGLE-3 + AWQ | **4/5** | Open issue #4351; works but may need `--disable-cuda-graph` for safety. Document the workaround. |
| HiCache L2 tuning | **2/5** | Flags are documented but GH200-specific values are inferred. |
| Sustained throughput at batch 256 | **3/5** | KV-cache pressure → tune `mem-fraction-static`, `chunked-prefill-size`, `schedule-conservativeness` empirically. |

---

## 9. Sources

Primary docs (verified 2026-05-25):
- [SGLang Quantization docs](https://docs.sglang.io/advanced_features/quantization.html) — auto-detection rule, awq_marlin/gptq_marlin entries, Hopper recommendations
- [SGLang Speculative Decoding docs](https://docs.sglang.io/advanced_features/speculative_decoding.html) — EAGLE-3 hyperparameter defaults, `unquant` draft override
- [SGLang HiCache Design docs](https://docs.sglang.io/advanced_features/hicache_design.html) — all HiCache CLI flags
- [SGLang Hyperparameter Tuning docs](https://docs.sglang.io/backend/hyperparameter_tuning.html) — chunked-prefill-size, schedule-conservativeness, mem-fraction-static
- [SGLang Cookbook — Llama-3.3-70B](https://docs.sglang.io/cookbook/autoregressive/Llama/Llama3.3-70B) — official launch shape (skinny on details)
- [SGLang v0.5.12 release notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.12) — W4A8 MoE Marlin/FlashInfer on Hopper, EAGLE-3 SWA, HiCache+UnifiedRadixTree
- [SGLang Dockerfile (aarch64 branch)](https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile) — line ~765, aarch64 cubin skip + comment

Model cards:
- [ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4](https://huggingface.co/ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4) — AWQ config + SGLang/vLLM commands
- [lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge](https://huggingface.co/lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge) — official 70B EAGLE-3 draft + hyperparameters
- [shuyuej/Llama-3.3-70B-Instruct-GPTQ](https://huggingface.co/shuyuej/Llama-3.3-70B-Instruct-GPTQ) — GPTQ-INT4 fallback

Open bugs / friction:
- [Issue #4351 — Speculative Decoding fails with AWQ Quantized Model](https://github.com/sgl-project/sglang/issues/4351)
- [Issue #9867 — Long DeepGEMM v2 warmup leading to NCCL timeout](https://github.com/sgl-project/sglang/issues/9867)
- [Discussion #16048 — torch.compile startup time](https://github.com/sgl-project/sglang/discussions/16048)
- [Issue #18550 — fa3 + fp8_e5m2 silent triton fallback](https://github.com/sgl-project/sglang/issues/18550)
- [FlashInfer Issue #1352 — flashinfer.aot does not download cubins ahead of time](https://github.com/flashinfer-ai/flashinfer/issues/1352)
- [SGLang Issue #10933 — Flash Infer cubin download AOT regression](https://github.com/sgl-project/sglang/issues/10933)

Kernel / quantization background:
- [Red Hat — Machete mixed-input GEMM for Hopper](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel) — single H100 + INT4 Llama-3.1-70B numbers
- [Spheron — AWQ Quantization Guide 2026](https://www.spheron.network/blog/awq-quantization-guide-llm-deployment/) — Marlin-AWQ 741 vs 68 tok/s
- [theaiengineer — GPTQ vs AWQ vs GGUF 2026](https://theaiengineer.substack.com/p/quantization-in-practice-gptq-vs) — 95% vs 90% quality
- [GPTQModel (ModelCloud)](https://github.com/ModelCloud/GPTQModel) — SGLang backend integration
- [Spheron — SGLang vs TRT-LLM vs vLLM H100 benchmarks 2026](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/) — concurrency 1/10/50/100 FP8 anchor

GH200 / NVLink-C2C / HiCache:
- [Verda — Data Movement on GH200 Grace Hopper (Jan 2026)](https://verda.com/blog/data-movement-in-the-nvidia-gh200-grace-hopper-superchip) — 375/297 GB/s STREAM
- [arXiv 2408.11556v2 — Data Movement in Tightly Coupled Heterogeneous Systems (GH200)](https://arxiv.org/html/2408.11556v2) — C2C utilization %
- [LMSYS HiCache blog 2025-09-10](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/) — 6× throughput, 80% TTFT
- [Mooncake × SGLang HiCache benchmark](https://kvcache-ai.github.io/Mooncake/performance/sglang-hicache-benchmark-results-v1.html) — H20/H800 only, no GH200
- [Baseten/Lambda — Llama-3.3-70B FP8 GH200](https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/) — GH200 +32% vs H100 (TRT-LLM, FP8)

FlashInfer / cubin pre-warm:
- [FlashInfer Installation docs](https://docs.flashinfer.ai/installation.html) — flashinfer-cubin / flashinfer-jit-cache packages, FLASHINFER_CUDA_ARCH_LIST
- [PyTorch torch.compile caching tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html) — TORCHINDUCTOR_CACHE_DIR portable caches
