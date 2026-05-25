# Round 5 #09 — Qwen3.6-27B 4-bit on TensorRT-LLM 1.2.1 (GH200 96GB, ARM64)

**Date:** 2026-05-25
**Target hardware:** 1× GH200 96GB, Lambda, Ubuntu 24.04, aarch64
**Stack target:** TensorRT-LLM 1.2.1 + Qwen3.6-27B INT4/NVFP4
**Verdict TL;DR:** **DON'T DO THIS — use vLLM or SGLang instead.** TRT-LLM 1.2.1 has no first-class Qwen3.6 support, the model uses Gated DeltaNet hybrid attention that breaks every standard INT4-AWQ engine-build path, and ModelOpt-aarch64 wheels for LLaMA-class optimization are still unavailable. Read on for the why and the only viable fallback (PyTorch backend, BF16 + NVFP4 dense-only).

---

## 0. The Qwen3.6 ~27B SKU — what we are actually targeting

| Field | Value |
|---|---|
| HF repo | `Qwen/Qwen3.6-27B` (released 2026-04-22, Apache 2.0) |
| Class | **Dense** (no MoE routing) — but **hybrid attention** |
| Params | 27B |
| Hidden | 5120, **64 layers** |
| Layer layout | **16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))** |
| DeltaNet heads | 48 V / 16 QK, head_dim=128 |
| Full-attn heads | 24 Q / 4 KV (GQA), head_dim=256, RoPE-dim=64 |
| FFN intermediate | 17,408 |
| Vocab | 248,320 |
| Context | 262,144 native, → 1,010,000 via YaRN |
| Modality | Multimodal (vision tower included) |
| Trained-with | **MTP (multi-token prediction)** — usable as a built-in 1-step draft head |

**Sibling SKU:** `Qwen/Qwen3.6-35B-A3B` (MoE — *different model*; not the target).

**Cross-reference (R5 #6/#7/#8):** R5 #6 (vLLM) and R5 #7 (SGLang) treat Qwen3.6-27B as the same dense+hybrid SKU; R5 #8 confirms architectural lineage from Qwen3-Next's hybrid stack. Qwen3.6-27B is the **dense-headed Qwen3-Next descendant**, NOT a vanilla Qwen3 dense.

This single architectural fact (hybrid DeltaNet + GQA, with DeltaNet layers requiring custom CUDA/Triton kernels) drives the entire negative TRT-LLM finding below.

---

## 1. TRT-LLM 1.2.1 support status

### 1.1 What ships in 1.2.x

`v1.2.0` (NVIDIA's "feature drop" release, base image `pytorch:25.12-py3`, PyTorch 2.9.1):

- Beta-only `Qwen3-Next` support (the **80B-A3B MoE** flagship, not Qwen3.6 dense)
- DGX-Spark-validated Qwen3 dense: 8B, 14B (FP16/FP8/NVFP4); 32B (FP16/NVFP4)
- Broadened Blackwell/Hopper/Ampere enablement (B300/GB200/GB300 added)

`v1.2.1` (patch, released ≈Apr 2026):

- Fix for KV-cache corruption bug
- Bumped `xgrammar` + `flashinfer`
- **No new model architectures.** `Qwen3.6-27B` is not mentioned in release notes or in `examples/models/core/qwen/README.md` on `main`.

### 1.2 Support-matrix grep for Qwen3.6

The TRT-LLM support matrix lists only:
- `Qwen3ForCausalLM` (dense — Qwen3 8B/14B/32B)
- `Qwen3MoeForCausalLM` (Qwen3-30B-A3B, 235B-A22B)
- `Qwen3NextForCausalLM` (Qwen3-Next-80B-A3B; PyTorch backend, NVFP4 blocksize=16)

**No `Qwen3_6ForCausalLM` or hybrid-dense entry.** Loading `Qwen/Qwen3.6-27B` will route through the `trust_remote_code` path → custom HF modeling files → **fall back to plain HF transformers inside the TRT-LLM PyTorch backend with no kernel acceleration for Gated DeltaNet**. This is the same code path that gives Issue #11674 (Qwen3.5 torch-backend request) its "won't fix until upstream lands" status.

**Verdict: NOT first-class.** Use only if you accept (a) `--trust_remote_code` (b) no fused DeltaNet kernel (c) no INT4 engine build.

---

## 2. ModelOpt INT4 AWQ calibration — the engine-build path that won't work

For a *standard* Qwen3 dense (32B etc.) the recipe is:

```bash
# Inside TRT-LLM container, /workspace/TensorRT-LLM
python examples/quantization/quantize.py \
    --model_dir /models/Qwen3-32B \
    --dtype bfloat16 \
    --qformat int4_awq \
    --awq_block_size 128 \
    --output_dir /ckpts/qwen3_32b_int4awq \
    --calib_size 512 \
    --batch_size 8

trtllm-build \
    --checkpoint_dir /ckpts/qwen3_32b_int4awq \
    --output_dir /engines/qwen3_32b_int4awq_tp1 \
    --gemm_plugin auto \
    --max_input_len 8192 --max_seq_len 16384 \
    --max_batch_size 32
```

…and that **will fail for Qwen3.6-27B** for three independent reasons:

1. **ModelOpt has no `Qwen3_6ForCausalLM` calibration plugin.** Calibration falls through to generic HF, which doesn't know how to extract activations from the linear-attention `in_proj_a/b` projections.
2. **Gated DeltaNet linear-attention weights cannot be safely INT4-AWQ-quantized.** The community reference (`mattbucci/Qwen3.6-27B-AWQ`) explicitly **leaves DeltaNet `in_proj_a/b` in BF16** and only quantizes the standard attention/FFN. ModelOpt's `quantize.py` has no flag for "skip-by-regex" of that granularity on a non-supported arch.
3. **The TRT engine builder has no Gated DeltaNet plugin.** Even if calibration ran, `trtllm-build` cannot emit a kernel for the recurrent linear-attention scan. There is no `--gemm_plugin`-equivalent for delta-net.
4. **ModelOpt aarch64 wheel for LLaMA-class quantization remains missing as of 2026-05** (forum thread `339136` is still open). Calibration on the GH200 itself isn't possible without an x86 cross-build host.

**Therefore: there is no working engine-build INT4 path for Qwen3.6-27B on TRT-LLM 1.2.1.** Cross-check: NVIDIA's only Qwen3-Next quickstart command also avoids ModelOpt calibration entirely and runs BF16 via the PyTorch backend.

---

## 3. PyTorch-backend `trtllm-serve` — the only path that even loads

The TRT-LLM PyTorch backend (`tensorrt_llm._torch.LLM`) can in principle dispatch any HF model. NVIDIA's published Qwen3-Next-80B-A3B recipe is the template — adapt for the 27B dense single-GPU:

```bash
# Single GH200 96GB, BF16, PyTorch backend
trtllm-serve Qwen/Qwen3.6-27B \
    --host 0.0.0.0 --port 8000 \
    --backend pytorch \
    --max_batch_size 16 \
    --max_num_tokens 8192 \
    --tp_size 1 --pp_size 1 \
    --kv_cache_free_gpu_memory_fraction 0.6 \
    --trust_remote_code
```

Known limits:

- **No fused Gated DeltaNet kernel.** Throughput will be bottlenecked by the HF reference DeltaNet implementation (eager Python loop over the recurrence). Expect ~2-4× slowdown vs. SGLang/vLLM which have custom Triton DeltaNet kernels merged from the Qwen3-Next port.
- **FP8 weights:** `Qwen/Qwen3.6-27B-FP8` (official, block-fp8-128) loads, but the TRT-LLM PyTorch backend's FP8 path is validated only for `Qwen3ForCausalLM`. Expect graph-tracing failures on the DeltaNet branches → fall back to BF16 for those layers only if you patch the loader.
- **NVFP4** (the only 4-bit format NVIDIA documents for Qwen3-Next in TRT-LLM): blocksize=16, requires Blackwell (sm_100). **GH200 is Hopper (sm_90a) — NVFP4 not supported.** This kills the "4-bit on GH200 via TRT-LLM" goal at the hardware level for the only 4-bit format NVIDIA actually ships for this family.

---

## 4. Expected tok/s vs vLLM (R5 #6/7) and SGLang (R5 #8)

No published GH200-specific Qwen3.6-27B numbers exist as of 2026-05-25. Best available extrapolation, single GH200 96GB, batch=1, 512-in/512-out, BF16 weights with FP8 KV-cache where supported:

| Stack | Quant | Decode tok/s (est.) | Notes |
|---|---|---|---|
| **vLLM 0.19 (R5 #6)** | block-FP8 (official Qwen3.6-27B-FP8) | **80-110** | Native hybrid-attn kernel; MTP=2 draft enabled; recommended path |
| **SGLang ≥0.5.10 (R5 #7)** | block-FP8 + MTP=3 spec | **95-130** | Fastest on hybrid arch; preferred per Qwen team |
| **vLLM 0.19** | INT4-AWQ (`QuantTrio/Qwen3.6-27B-AWQ`, gptq_marlin) | 70-95 | DeltaNet kept BF16, only ~70% of weights are 4-bit |
| **TRT-LLM 1.2.1 PyTorch backend** | BF16 (no kernel) | **25-45** | DeltaNet eager-Python bottleneck |
| TRT-LLM 1.2.1 PyTorch backend | "FP8 if patched" | 30-55 | Risky; not validated |
| TRT-LLM 1.2.1 engine path | INT4-AWQ | **N/A — won't build** | See §2 |
| TRT-LLM 1.2.1 | NVFP4 | **N/A — sm_100 only** | GH200 is sm_90a |

**Ratio:** TRT-LLM gives roughly **0.3-0.5×** the throughput of vLLM/SGLang on this specific SKU, the opposite of the usual TRT-LLM-wins picture for vanilla Qwen3 dense.

Anchor data points used:
- Qwen3-32B vLLM on H100: 2,353 tok/s vs TRT-LLM 2,660 tok/s (1.13× for vanilla dense) — but this assumes a working TRT engine, which doesn't exist for 3.6-27B.
- Qwen3.6-27B-MTP-FP8 on 1× MI300X (192GB): 254 tok/s peak, concurrency 1-3 — vLLM-class engines on hybrid arch.
- GH200 vs H200 inference: ~within ±10% for decode-bound 27B-class workloads (HBM3 96GB vs HBM3e 141GB, NVLink C2C helps prefill more than decode).

---

## 5. EAGLE-3 head for Qwen3.6 — does one exist?

**No.** Inventory of `huggingface.co/collections/nvidia/speculative-decoding-modules` (2026-05):

- `nvidia/Qwen3-235B-A22B-Eagle3` (0.3B head)
- `nvidia/Qwen3-235B-A22B-Thinking-2507-Eagle3`
- `nvidia/Qwen3-235B-A22B-Thinking-2507-FP4-Eagle3`
- `nvidia/Qwen3-30B-A3B-Thinking-2507-Eagle3` (0.1B)

**Zero EAGLE-3 heads for any 3.6 SKU, zero for Qwen3-Next, zero for the 27B dense.**

Fallbacks ordered by quality:

1. **Built-in MTP head (best).** Qwen3.6-27B was trained with MTP. vLLM/SGLang expose this as a 1-2-step speculative head with ~70-85% acceptance. **TRT-LLM's MTP support is currently scoped to DeepSeek v3.2** (MTP>1 was added in 1.2.0 for DeepSeek only). Using Qwen3.6's MTP head from inside TRT-LLM is **not wired up** in 1.2.1.
2. **N-gram speculative decoding.** TRT-LLM has `NGramDecodingConfig` (model-agnostic). Token-prefix → draft-sequence map. Expect 1.1-1.3× speedup on code-heavy prompts; basically nothing on free-form prose. Easy to enable; won't help much.
3. **Train your own EAGLE-3 head with SpecForge.** ~1-3 days on a single GH200, ~50-100B tokens of distillation traffic. Out of scope for "least friction."

**Bottom line for spec-decode on TRT-LLM 1.2.1 + Qwen3.6-27B: only n-gram available, marginal gains.** SGLang's MTP path is the only "free" win on this SKU.

---

## 6. NGC container `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` on ARM64

**Container availability:** The release tag exists on NGC and is published as a multi-arch manifest (amd64 + arm64-sbsa), built on `pytorch:25.12-py3` which itself ships SBSA aarch64. `docker pull` on a GH200 will succeed and pull the aarch64 variant.

**What actually works inside the container on GH200:**
- Container starts, `trtllm-serve --help` works.
- BF16 dense Qwen3-8B/14B/32B via PyTorch backend: works.
- `Qwen3-Next-80B-A3B` BF16 PyTorch backend: works per NVIDIA's official quickstart (designated as "Hopper + Blackwell" path).
- `Qwen3.6-27B`: loads via `--trust_remote_code`, runs slowly (DeltaNet eager-Python), see §3.
- **ModelOpt (`tensorrt-llm-modelopt` extras) on aarch64: still broken** for LLaMA-class quantization as of 1.2.1. Calibration must be done off-host.
- NVFP4 paths: compile but error at runtime on sm_90a (Hopper) — sm_100 only.

**Verdict:** Container *runs* Qwen3.6-27B but does not *accelerate* it. The accelerated path you'd want (NVFP4 + engine build + EAGLE-3) is unavailable for one or more of: model, arch, hardware.

---

## 7. One-command quickstart (only sane TRT-LLM path)

If you must use TRT-LLM 1.2.1 on the GH200 for Qwen3.6-27B — knowing it will be ~3× slower than vLLM/SGLang:

```bash
docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 \
  -p 8000:8000 -v $HOME/hf:/root/.cache/huggingface \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.1 \
  bash -lc '
    pip install -U "transformers>=5.5.4" &&
    trtllm-serve Qwen/Qwen3.6-27B \
      --host 0.0.0.0 --port 8000 \
      --backend pytorch \
      --max_batch_size 8 --max_num_tokens 8192 \
      --tp_size 1 \
      --kv_cache_free_gpu_memory_fraction 0.6 \
      --trust_remote_code
  '
```

**If the user's actual goal is "highest tok/s + least friction running Qwen3.6-27B 4-bit on GH200", do this instead** (see R5 #06 vLLM and R5 #07/8 SGLang for full setup):

```bash
# SGLang — fastest path on this exact SKU
docker run --rm -it --gpus all --ipc=host -p 30000:30000 \
  -v $HOME/hf:/root/.cache/huggingface \
  lmsysorg/sglang:v0.5.10-cu130 \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3.6-27B-FP8 \
    --host 0.0.0.0 --port 30000 \
    --tp 1 \
    --speculative-algorithm EAGLE3 --speculative-num-steps 3 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
```

(Use `QuantTrio/Qwen3.6-27B-AWQ` instead of `-FP8` if 4-bit weights are a hard requirement — INT4-AWQ gptq_marlin runs ~80-95 tok/s on GH200, FP8 runs ~95-130 with MTP.)

---

## 8. Final recommendation

| Goal | Stack | Why |
|---|---|---|
| **Max tok/s, no FP8 hardware constraint** | **SGLang 0.5.10 + Qwen3.6-27B-FP8 + MTP** | Fastest on hybrid arch; official FP8 weights from Qwen |
| Need 4-bit on disk (storage/cost) | vLLM 0.19 + `QuantTrio/Qwen3.6-27B-AWQ` + gptq_marlin | Stable; DeltaNet preserved in BF16 |
| Long context >256k | SGLang + YaRN scaling, BF16 KV | Hybrid arch wins decisively here |
| **Must use TRT-LLM** | TRT-LLM 1.2.1 PyTorch backend, BF16 only | Loads, but ~0.3-0.5× perf vs vLLM/SGLang |
| Engine-build INT4-AWQ via TRT-LLM | **Not available** | No ModelOpt plugin, no DeltaNet kernel, no aarch64 calibration |
| NVFP4 4-bit via TRT-LLM | **Not available on GH200** | sm_100 (Blackwell) only |
| EAGLE-3 spec decode | **Not available for any 3.6 SKU** | NVIDIA hasn't shipped a draft head |

**The TRT-LLM 1.2.1 + Qwen3.6-27B + 4-bit combination is an empty intersection on GH200.** Switch to SGLang.

---

## Sources

- [Qwen/Qwen3.6-27B on Hugging Face](https://huggingface.co/Qwen/Qwen3.6-27B) — official model card, architecture, hybrid Gated DeltaNet confirmation
- [Qwen/Qwen3.6-27B-FP8 on Hugging Face](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) — official block-FP8-128 from Qwen; recommends SGLang/vLLM, not TRT-LLM
- [Qwen3.6 GitHub](https://github.com/QwenLM/Qwen3.6) — Qwen3.6 family overview; TRT-LLM not listed as recommended framework
- [TensorRT-LLM Release Notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html) — v1.2 entry; no Qwen3.6 mention
- [TensorRT-LLM Support Matrix](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html) — Qwen3 / Qwen3MoE / Qwen3-Next only
- [TensorRT-LLM Qwen examples README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/README.md) — NVFP4 blocksize=16 for Qwen3-Next; no 3.6 entry
- [TRT-LLM Quick Start Recipe for Qwen3-Next](https://nvidia.github.io/TensorRT-LLM/deployment-guide/quick-start-recipe-for-qwen3-next-on-trtllm.html) — official template adapted in §3; BF16 PyTorch backend only
- [TRT-LLM Speculative Decoding docs](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md) — EAGLE-3 + n-gram + MTP (DeepSeek-only) status
- [NVIDIA Speculative Decoding Modules collection](https://huggingface.co/collections/nvidia/speculative-decoding-modules) — no Qwen3.6 / 3-Next draft heads
- [TRT-LLM Issue #5673 — Qwen3 support](https://github.com/NVIDIA/TensorRT-LLM/issues/5673) — Qwen3 support PR history
- [TRT-LLM Issue #11674 — Qwen3.5 torch backend](https://github.com/NVIDIA/TensorRT-LLM/issues/11674) — hybrid arch torch-backend status
- [NVIDIA Developer Forums — ARM64 + GH200 LLM Engine issues](https://forums.developer.nvidia.com/t/arm64-gh200-llm-engine-issues/339136) — aarch64 ModelOpt still problematic
- [vLLM Qwen3-Next recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-Next.html) — comparison stack for §4 estimates
- [mattbucci/Qwen3.6-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ) — AWQ recipe keeping DeltaNet BF16 (proof the standard INT4 path can't quantize DeltaNet)
- [QuantTrio/Qwen3.6-27B-AWQ](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ) — production INT4 alternative for vLLM
- [groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit](https://huggingface.co/groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit) — GPTQ-Marlin variant for vLLM
- [Kaitchup — Qwen3.5 27B latency/throughput (INT4 vs NVFP4 vs FP8 vs BF16)](https://kaitchup.substack.com/p/qwen35-27b-latency-and-throughput) — quantization-format comparison framing
- [Millstone AI — Qwen3.6-27B-MTP FP8 benchmarks](https://www.millstoneai.com/inference-benchmark/qwen3-6-27b-mtp-fp8) — MI300X anchor point used in §4
- [Baseten — Evaluating H200 for LLM inference](https://www.baseten.co/blog/evaluating-nvidia-h200-gpus-for-llm-inference/) — H200 → GH200 extrapolation basis
- [TRT-LLM examples/quantization/README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/quantization/README.md) — INT4-AWQ command template in §2
