# TensorRT-LLM on GH200 — Research Round 1, Agent 02

**Date compiled:** 2026-05-25
**Author:** Research Agent #02 (TensorRT-LLM domain)
**Hardware target:** NVIDIA GH200 Grace-Hopper Superchip (96GB HBM3 / 144GB HBM3e variants, Grace ARM Neoverse-V2 CPU, aarch64/SBSA)
**Stack tested constraint:** FP8 is reported broken/unreliable on this team's GH200 stack — bf16 is the default recommendation, FP8 evidence flagged explicitly below.

---

## 1. Current Release Landscape (May 2026)

| Version | Status | Notes |
|---|---|---|
| **v1.3.0rc15** | Latest pre-release (21 May 2026) | RC line; brings FP4/FP8 decode kernels, Gemma4 multimodal, MoE perf, FP8 block-scaling autotuner cache fix |
| v1.3.0rc14 | 7 May 2026 | Mamba hybrid prefix caching, Qwen3.5 |
| v1.3.0rc13 | 29 Apr 2026 | **"restoring the missing aarch64 library"** + SBSA wheel image support |
| v1.3.0rc12 | 17 Apr 2026 | LTX-2 pipeline, CUDA graph compat |
| **v1.2.0** | Latest stable (early 2026) | Beta DGX Spark (aarch64 Blackwell), GPT-OSS, Llama-3.3, Qwen3, Phi-4 |
| v1.1.0 | 2025 | Hunyuan, Seed-OSS, KV Cache Connector API, multi-layer Eagle, DeepEP, B300/GB300 |
| **v1.0.0** | 24 Sep 2025 | **PyTorch backend becomes default**; LLM API marked stable; DeepSeek-R1 FP8 on Blackwell |
| v0.21.0 | mid-2025 | Gemma3 VLM, FP8 rowwise, w4a8_mxfp4_fp8, NIXL disagg |
| v0.20.0 | 2025 | Qwen3, Mistral Small 3.1 24B VLM |
| v0.19.0 | early 2025 | C++ runtime open-sourced, DeepSeek V3/R1, FlashMLA SM90, FP8 MLA Hopper |
| v0.18.x | Windows deprecation | NGC 25.03 base |

**Trajectory:** PyTorch-based LLM API path is now canonical. The legacy `trtllm-build` engine-compile workflow is still supported but no longer the primary entry. The release-notes page is at https://nvidia.github.io/TensorRT-LLM/release-notes.html (verified 2026-05-25).

---

## 2. Grace-ARM (aarch64 / SBSA) Build Reality on GH200

**Bottom line:** As of May 2026 the situation has improved markedly from the "no real ARM64 binaries" status that persisted through mid-2025, but it is still rougher than x86_64. Pip wheels for aarch64 now exist (the `aarch64 library restoration` in v1.3.0rc13, April 2026 confirms this) and SBSA wheels are tracked in build infra. Bare-metal Ubuntu 24.04 on SBSA is **explicitly called out as incompatible** with the PyTorch backend — NVIDIA's documented recommendation is the **PyTorch NGC container** (`nvcr.io/nvidia/pytorch:25.xx-py3`) or the **TRT-LLM NGC release container**.

### Install paths that work on GH200 (May 2026)

**Path A — NGC container (recommended; least pain):**
```bash
docker run --gpus all -it --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $PWD:/workspace -w /workspace \
  nvcr.io/nvidia/tritonserver:25.12-trtllm-python-py3   # or tensorrt-llm release tag
```
NGC publishes a multi-arch (x86_64 + aarch64/sbsa) manifest for the recent release tags; docker pull auto-selects the GH200 image.

**Path B — pip in an NGC PyTorch container (works for v1.0+):**
```bash
# Inside nvcr.io/nvidia/pytorch:25.12-py3 (which already ships CUDA 13.1, PyTorch 2.10.0 aarch64)
pip3 install --ignore-installed pip setuptools wheel
pip3 install tensorrt_llm --extra-index-url https://pypi.nvidia.com
```
Requires CUDA 13.1 + PyTorch 2.10.0 (per v1.3 rc series install docs). Earlier wheels of TRT-LLM 0.15/0.17 on GH200 fail with `ModuleNotFoundError: No module named 'tensorrt'` (GitHub issue #2571 — closed) because the TensorRT base wheel only built x86_64 historically.

**Path C — build from source on GH200:** Possible since the aarch64 fixes landed in v1.3.0rc13. Use `nvcr.io/nvidia/pytorch:25.12-py3` as base, clone with `git-lfs`, run the documented Linux build. Build time ~30-60 min on a single Grace.

### Known aarch64 gotchas
- **Driver:** GH200 driver < 560.35.03 segfaults or hangs at engine start. Use ≥ 560.35.03.
- **NCCL ≥ 2.28** is enforced at configure time in recent RCs.
- **PyPI SBSA wheel ABI break** with PyTorch 2.5.1 (legacy issue, resolved by moving to torch 2.10 in v1.3 line).
- **DGX Spark (aarch64 Blackwell)** has its own bug surface (issue #12230: Qwen3-Next-80B FP8/NVFP4 autotuner IndexError and Triton illegal instruction on TRT-LLM 1.3.0rc7). GH200 (Hopper aarch64) is a different code path but watch for similar Triton/autotuner regressions.

Sources: NVIDIA developer forum thread "Arm64 + gh200 LLM Engine issues" (last update 23 Sep 2025) https://forums.developer.nvidia.com/t/arm64-gh200-llm-engine-issues/339136 ; GitHub issue #2060 "Support for Grace Hopper" (closed/triaged); GitHub issue #2571 "TRT-LLM fails on GH200 node" (closed); install docs https://nvidia.github.io/TensorRT-LLM/installation/linux.html

---

## 3. Supported Model Families on GH200

Cross-referenced against the NVIDIA NIM 2.0.5 support matrix (which is the most authoritative "what NVIDIA has actually validated on GH200" list as of May 2026): https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html

| Model | GH200-96GB | GH200-144G HBM3e | Precisions verified | TP |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | yes | yes | BF16, FP8, NVFP4 | TP1 |
| Llama-3.1-70B-Instruct | yes | yes | BF16, FP8, NVFP4 | TP1–TP8 |
| Llama-3.3-70B-Instruct | yes | yes | BF16, FP8, NVFP4 | TP1–TP8 |
| Llama-3.3-Nemotron-Super-49B-v1.5 | yes | yes | BF16, FP8, NVFP4 | TP1–TP8 |
| Nemotron-3-Nano | yes | yes | BF16, FP8, NVFP4 | TP1–TP8 |
| Nemotron-3-Super-120B-a12b | yes (tight) | yes | BF16, FP8, NVFP4 | TP1–TP8 |
| GPT-OSS-20B | yes | yes | MXFP4 | TP1–TP8 |
| GPT-OSS-120B | **no** (mem) | yes | MXFP4 | TP1–TP8 |
| Starcoder2-7B | yes | yes | BF16 | TP1, TP2 |
| Qwen2.5 / Qwen3 family | community | community | BF16, AWQ-INT4 | TP1+ |
| Mistral / Mistral-Large | community | community | BF16, FP8 | TP1+ |
| DeepSeek-V3 / R1 671B | needs ≥ 8× GH200 (NVL2/NVL32) | same | FP8 (bf16 KV) | TP8/EP8 |

"community" = supported by TRT-LLM upstream but not in NVIDIA's NIM-validated matrix.

NVFP4 is a Blackwell-only format (not Hopper SM90), so the "NVFP4" entries above are dead paths on GH200 — they apply when the same NIM image runs on B200/GB200. On GH200 you have BF16 and FP8 (with caveats) as the real choices for the listed Llama/Nemotron set.

---

## 4. Quantization Path Matrix (Hopper SM90 / GH200)

Sourced from https://nvidia.github.io/TensorRT-LLM/features/quantization.html

| Format | Weights | Activations | KV cache | Hopper/GH200 | Notes |
|---|---|---|---|---|---|
| **BF16** | bf16 | bf16 | bf16 | ✓ default | The reliable baseline. **Use this.** |
| FP16 | fp16 | fp16 | fp16/bf16 | ✓ | Equivalent quality to bf16 on Hopper |
| **FP8 per-tensor** | fp8 (E4M3) | fp8 | fp8 | ✓ kernels exist | **See §5 — broken on this team's stack** |
| FP8 block-scaling | fp8 | fp8 | fp8 | ✓ (SM90 path differs from SM100 MXFP8) | Autotuner cache regression fixed in 1.3.0rc15 |
| FP8 rowwise | fp8 | fp8 | fp8 | ✓ (added v0.21) | Newer; less burn-in |
| **W4A16 AWQ** (INT4 weights, BF16 act) | int4 | bf16/fp16 | bf16/fp16 | ✓ | Strong for memory-bound bs≤4 |
| **W4A16 GPTQ** | int4 | bf16/fp16 | bf16/fp16 | ✓ | Marginally lower quality than AWQ; sometimes faster build |
| W4A8 AWQ | int4 | fp8 | fp8 | ✓ requires SM ≥ 89 | Inherits any FP8 instability |
| W4A8 GPTQ | int4 | fp8 | fp8 | ✓ requires SM ≥ 89 | Same caveat |
| INT8 SmoothQuant (W8A8) | int8 | int8 | int8 | ✓ | Older but stable; bigger memory than AWQ |
| W8A16 | int8 | bf16/fp16 | bf16/fp16 | ✓ | Rarely worth it vs W4A16 |
| **NVFP4** | nvfp4 | nvfp4 | varies | ✗ Blackwell-only | Listed in NIM but dead on GH200 |
| **MXFP4** | mxfp4 | mxfp4 | varies | ✗ Blackwell-only | GPT-OSS NIM uses this; not real on GH200 |

**Practical recommendation for GH200 given the FP8 constraint:**
1. **70B-class on single GH200 (96GB):** W4A16 AWQ with bf16 activations and bf16 KV cache. Fits comfortably with headroom for KV.
2. **70B-class on GH200-144G or 2× GH200:** BF16 weights + bf16 KV. No quantization gymnastics, max accuracy.
3. **8B–13B:** Pure BF16. No reason to quantize.
4. **DeepSeek-V3/R1 class:** Requires multi-GH200; FP8 is the only practical fit at 671B — flag the risk and validate accuracy carefully (see §5).

---

## 5. FP8 Status on GH200 — Evidence Audit

The constraint from prior experience says FP8 is broken/unreliable. The May-2026 evidence is **mixed**: kernels exist and NVIDIA publishes FP8 benchmarks on Hopper, but several open/recent bugs continue to land on FP8 paths. **No public May-2026 source declares FP8 as universally fixed for the GH200/Hopper-SM90 path.**

Active or recent FP8 issues (with dates):

| Issue | Source | Date | Affects GH200? |
|---|---|---|---|
| FP8 KV Cache broken on Qwen2.5-Coder (corrupted output, gibberish) | GH issue #4218 | 12 May 2025, still showing bug label | Yes — H100 reported, same SM90 path as GH200 |
| AutoDeploy: FP8/NVFP4 kernels not replaced for fake-quantized Flux | GH issue #8974 | Late 2025 / early 2026 | Hopper+Blackwell; loses expected 1.6× FP8 speedup |
| Llama 3.x / Llama 4 pipeline-parallel + FP8/NVFP4 hangs | Release notes workaround | active 2026 | Workaround: `TRTLLM_LLAMA_EAGER_FUSION_DISABLED=1` |
| Illegal memory access in FP8 Scout/DeepSeek | v1.0 release notes "resolved" | Sep 2025 | Was on Hopper too; fixed |
| FP8 block-scaling autotuner cache growth | v1.3.0rc15 fix | 21 May 2026 | All SM90+ |

NVIDIA's own perf-overview page reports GH200 96GB Llama-3.1 8B numbers in **FP8** (see §6), so the FP8 path runs and produces numbers — but the gibberish-output bug class (KV-cache FP8 on Qwen2.5) is exactly the failure mode that justifies bf16-as-default. **Verdict: keep the bf16-default policy. If FP8 must be used, gate it behind a per-model accuracy regression test (rouge/exact-match against bf16 reference).**

Sources:
- GH #4218 https://github.com/NVIDIA/TensorRT-LLM/issues/4218 (12 May 2025, last seen open/triaged)
- GH #8974 (FP8/NVFP4 AutoDeploy kernel replacement bug)
- Release notes "TRTLLM_LLAMA_EAGER_FUSION_DISABLED=1" workaround
- NVIDIA perf overview https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html (FP8 numbers publish but only for 8B on GH200)

---

## 6. Published Benchmark Numbers (with citations)

### 6.1 NVIDIA's official perf-overview page — GH200 96GB entries
Source: https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html (verified 2026-05-25)

**Llama-3.1 8B, FP8, TP1, single GH200 96GB:**

| ISL | OSL | Tokens/sec |
|---|---|---|
| 128 | 128 | 27,304.25 |
| 128 | 2048 | 24,045.60 |
| 128 | 4096 | 15,409.85 |
| 500 | 2000 | 20,123.88 |
| 1000 | 1000 | 16,352.99 |
| 1000 | 2000 | 15,705.82 |
| 1024 | 2048 | 16,102.52 |
| 2048 | 128 | 3,573.85 |
| 2048 | 2048 | 10,767.05 |
| 5000 | 500 | 3,584.74 |
| 20000 | 2000 | 1,393.31 |

**Critical gap:** NVIDIA's perf-overview page does **not** publish GH200 numbers for Llama-3.1 70B, Llama-3.3 70B, Mistral-Large, Qwen2.5 72B, or DeepSeek. Only Llama-3.1 8B FP8. This is the single biggest documentation gap.

### 6.2 Llama-3.3 70B on GH200 — third-party (Baseten/Lambda)
Source: https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/ ; cross-referenced https://lambda.ai/blog/partner-spotlight-testing-llama-3.3-70b-inference-performance-on-nvidia-gh200-with-baseten (published 7 Feb 2025)

- Framework: TensorRT-LLM Engine Builder (Baseten), benched with SGLang's harness on ShareGPT
- Precision: **FP8** (model 70 GB in FP8, leaves ~27 GB on GH200-96 vs ~10 GB on H100-80)
- Batch size: 32
- **Headline: GH200 96GB outperforms H100 80GB by 32% on Llama-3.3 70B at FP8.**
- They do **not** publish an absolute tok/s number; only the relative delta. Source explicitly attributes the win to KV cache headroom + NVLink-C2C 450 GB/s CPU offload, not raw compute (compute is identical to H100).

### 6.3 Llama-3.3 70B with speculative decoding (H200 reference, transfers to GH200)
Source: https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/ (17 Dec 2024, tested 11 Dec 2024)

Single H200, TP=1, batch=1, **FP8** target + draft, TRT-LLM v0.15.0:

| Draft | Target | tok/s | Speedup |
|---|---|---|---|
| (none) | Llama-3.3-70B | 51.14 | 1.0× |
| Llama-3.2 1B | Llama-3.3-70B | **181.74** | 3.55× |
| Llama-3.2 3B | Llama-3.3-70B | 161.53 | 3.16× |
| Llama-3.1 8B | Llama-3.3-70B | 134.38 | 2.63× |

Extrapolation to GH200: H200 SXM has identical Hopper SM count; GH200-96 has slightly less HBM (96 vs 141 GB) but adds 480 GB LPDDR5X via NVLink-C2C. For batch=1 latency these numbers translate within ~5%. EAGLE3 (https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-Eagle3) is the current best-of-class spec-decode for this model.

### 6.4 DeepSeek-R1 on H200 (closest GH200 proxy)
Source: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md

- 8× H200, TP=8/EP=8, **FP8** (`deepseek-ai/DeepSeek-R1`)
- Min-latency: 158 tok/s/user
- Max-throughput: 11,489 tok/s aggregate
- For comparison: 8× B200 with FP4 hits 43,146 tok/s — Blackwell is 3-4× ahead on this workload

No published 8× GH200 DeepSeek-R1 number exists in public docs as of May 2026. MLPerf v5.0 (Apr 2025) had a Supermicro dual-GH200-144G submission for Llama-2 70B (FP8 via TRT-LLM) but used Llama-2, not DeepSeek.

### 6.5 MLPerf Inference v5.0 — GH200 entries
- Red Hat + Supermicro: dual-GPU GH200 144GB server running OpenShift 4.15 + TRT-LLM, Llama-2 70B FP8. First-ever OpenShift-on-GH200 MLPerf submission.
- HPE ProLiant DL384 Gen12: dual-socket Nvidia GH200 144GB delivered 2× perf of single-socket variant on same MLPerf v5.0 workload.
- Source: https://www.redhat.com/en/blog/mlperf-inference-v50-results (April 2025)

### 6.6 vLLM vs TRT-LLM on GH200
- LLM-Inference-Bench (arxiv 2411.00136): "vLLM on GH200 consistently achieves the highest throughput across all batch sizes, with H100 as the second-best closest performer." This is for Llama-3 70B and notes GH200 wins primarily on memory (3.5× more memory, tight CPU-GPU coupling).
- General-case 2026 benchmark consensus (Spheron, n1n.ai, Medium/Synthetic Futures): on H100, TRT-LLM 15-50% higher throughput than vLLM at high concurrency, but vLLM has caught up significantly with v0.10+. On GH200 specifically, vLLM has had a smoother ARM story (official v0.10.2 ARM image works flawlessly) which has pushed many users to vLLM as the default.

---

## 7. Engine-Build Cost vs Runtime Gain (Trade-off Analysis)

### Build cost
- **Llama-3 70B FP8 single-GPU engine on H100/GH200:** ~30-90 minutes wall-clock (community report; varies with `--workers`, `--max_batch_size`, and whether you enable `--use_fp8_context_fmha`, `--multi_block_mode`, etc.)
- **TRT-LLM Engine Builder (Baseten) automates this**, but the engine is hardware-specific — built on H100, won't run on GH200 (different SM90 codegen path).
- **Llama-3.1 405B:** "hours" of babysitting, requires ≥8 matching GPUs *during build*, not just at inference. This is a real operational tax.
- **PyTorch backend (v1.0+ default):** Skips the AOT engine compile — loads HF checkpoint directly into the PyTorch graph at startup. Trade-off: startup is 30s-2min (graph-warmup), runtime is ~5-15% slower than a maximally-tuned TRT engine. **For iterative dev work or multi-model serving, the PyTorch path wins.**

### When the TRT engine build is worth it
- Single fixed model, high-volume production, max-throughput priority — 15-30% wins are real.
- Specific kernels that the PyTorch backend can't yet do (some W4A8 paths, some Medusa/EAGLE configs).

### When to skip the engine build
- Multi-LoRA serving (the LoRA tax on TRT engine builds is painful; PyTorch handles adapter swapping cleanly).
- Frequent model swaps / R&D.
- Spec-decode where you want to iterate draft model choices.

Source: https://www.baseten.co/blog/automatic-llm-optimization-with-tensorrt-llm-engine-builder/ ; https://www.spheron.network/blog/tensorrt-llm-production-deployment-guide/

---

## 8. Speculative Decoding, Multi-LoRA, KV Cache Reuse, In-Flight Batching

### Speculative decoding
TRT-LLM supports: **Draft-Target, Medusa, EAGLE (incl. EAGLE3), Lookahead, MTP (DeepSeek-native)**. EAGLE3 is the current SOTA for Llama-3.3 70B (NVIDIA publishes a pre-trained `nvidia/Llama-3.3-70B-Instruct-Eagle3` HF model). Multi-layer Eagle landed in v1.1. Guided decoding (XGrammar) integrates with spec-decode in v0.19+. Measured speedups 1.9× (Medusa, H200) up to **3.55×** (Llama-3.2-1B draft → Llama-3.3-70B target, FP8, H200, batch=1).

### Multi-LoRA
- Native in-flight LoRA adapter swapping per-request in the batch.
- PyTorch backend handles this more cleanly than TRT-engine path (no need to pre-bake adapter set into engine).
- Encoder-decoder + multi-LoRA works (e.g., BART).
- LoRA verified across all TP levels in the NIM 2.0.5 matrix.

### KV cache reuse
- Prefix-based reuse across requests via a search-tree of filled blocks. https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html
- **GH200 advantage:** NVLink-C2C 450 GB/s lets you offload KV blocks to Grace's 480 GB LPDDR5X, enabling much larger effective KV pools than any standalone H100/H200. This is the structural reason GH200 beats H100 by 32% on Llama-3.3 70B despite identical compute.
- Caveat: reuse only fires after the producing request *terminates*. Large-batch shared-prefix scenarios won't reuse until at least one request finishes — common pitfall.
- KV Cache Connector API (v1.1) supports disaggregated serving with state transfer.
- Fabric Memory KV transfer added v0.21.

### In-flight (continuous) batching
- First-class, has been the default scheduler for a long time. C++ batch manager open-sourced in v0.19.
- Encoder-decoder models also support IFB (v0.18+).
- `trtllm-serve` and `trtllm-bench` both drive IFB.

---

## 9. NIM Containers on GH200 — Reality Check

### What's available
- NIM **does** support GH200-144G HBM3e and GH200-480GB variants in its 2.0.5 support matrix.
- Validated models on GH200: Llama-3.1-8B, Llama-3.1-70B, Llama-3.3-70B, Llama-3.3-Nemotron-Super-49B-v1.5, Nemotron-3-Nano, Nemotron-3-Super-120B-a12b, Starcoder2-7B, GPT-OSS-20B/120B (NVFP4/MXFP4 — Blackwell paths, listed but dead on Hopper).
- **However:** a forum data point from Sep 2025 reported "NIM support limitations (only 7 of 125 models support GH200)" — the matrix has grown since, but **most of the 100+ NIM catalog is x86_64-only**.
- Recent DGX-Spark / aarch64 forum threads (April-May 2026) confirm: **most NIM container images are still only published for amd64 / x86_64**. ARM aarch64 NIM images exist for a small subset of LLMs and a few bio-NIMs (e.g., OpenFold3, DiffDock-on-request) but `nv-ingest`, RAG-server, and many tool NIMs have no ARM build yet.

### Licensing / cost
- **NVIDIA AI Enterprise (NVAIE)** is the production license that covers NIM. ~**$4,500/GPU/year** list, or ~$1/GPU/hr on cloud marketplaces, with discounts on committed terms.
- **Free path 1:** 90-day NVAIE evaluation license.
- **Free path 2:** NVIDIA Developer Program members get free NIM access for **non-production use on up to 16 GPUs** (good for prototyping on a GH200).
- The TensorRT-LLM library itself is Apache-2.0 / open source (https://github.com/NVIDIA/TensorRT-LLM) and has **no per-GPU runtime fee** when used outside NIM. You only pay if you want NVIDIA's pre-baked NIM containers and enterprise support.

Sources: https://docs.nvidia.com/ai-enterprise/planning-resource/licensing-guide/latest/pricing.html ; https://developer.nvidia.com/blog/access-to-nvidia-nim-now-available-free-to-developer-program-members/

---

## 10. Practical Snippets

### 10.1 Quick LLM API run (PyTorch backend, bf16, GH200)
```python
from tensorrt_llm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    dtype="bfloat16",
    tensor_parallel_size=1,           # GH200 has 1 GPU; use 2 for GH200-NVL2
    kv_cache_config={
        "free_gpu_memory_fraction": 0.85,
        "host_cache_size": 200 * 1024**3,   # 200GB offload to Grace LPDDR5X
        "enable_block_reuse": True,
    },
    enable_chunked_prefill=True,
)

sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
for out in llm.generate(["The Grace-Hopper superchip..."], sp):
    print(out.outputs[0].text)
```

### 10.2 Serve (OpenAI-compatible) with bf16
```bash
trtllm-serve meta-llama/Llama-3.3-70B-Instruct \
    --dtype bfloat16 \
    --max_batch_size 64 \
    --max_num_tokens 8192 \
    --kv_cache_free_gpu_mem_fraction 0.85 \
    --enable_block_reuse \
    --host 0.0.0.0 --port 8000
```

### 10.3 Benchmark a built engine / running model
```bash
trtllm-bench --model meta-llama/Llama-3.3-70B-Instruct throughput \
    --dataset_path sharegpt.json \
    --concurrency 32 \
    --backend pytorch
```

### 10.4 Legacy engine-build (TRT path) — if you must
```bash
python examples/llama/convert_checkpoint.py \
    --model_dir Llama-3.3-70B-Instruct \
    --output_dir ckpt-bf16 --dtype bfloat16 --tp_size 1

trtllm-build --checkpoint_dir ckpt-bf16 --output_dir engine-bf16 \
    --gpt_attention_plugin bfloat16 --gemm_plugin bfloat16 \
    --max_batch_size 64 --max_num_tokens 8192 \
    --use_paged_context_fmha enable \
    --workers 1
```

### 10.5 Speculative decoding (EAGLE3) — bf16 target
```python
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    dtype="bfloat16",
    speculative_config={
        "decoding_type": "Eagle",
        "speculative_model": "nvidia/Llama-3.3-70B-Instruct-Eagle3",
        "num_eagle_layers": 3,
    },
)
```

---

## 11. Gotchas — Pinned List

1. **No NVFP4/MXFP4 on Hopper.** If a NIM profile says NVFP4 and you're on GH200, it's not actually running NVFP4 — pick BF16 or FP8 profile instead.
2. **FP8 KV cache + Qwen2.5 = gibberish output** (GH #4218, May 2025, still bug-labeled). Don't enable `kv_cache_quant_algo=FP8` on Qwen models without explicit accuracy verification.
3. **Llama-3.x pipeline-parallel + FP8** can hang. Workaround: `export TRTLLM_LLAMA_EAGER_FUSION_DISABLED=1`.
4. **Engines built on H100 don't run on GH200.** Different SM90 codegen path. Always build on the actual target hardware.
5. **Engines built on GH200-96 may not run on GH200-144** depending on memory-dependent plugin choices. Use the same SKU end-to-end.
6. **GH200 driver < 560.35.03 segfaults.** Update first.
7. **Bare-metal Ubuntu 24.04 on SBSA is unsupported** for PyTorch backend. Use NGC container.
8. **NCCL ≥ 2.28** required by recent RCs.
9. **DGX Spark issue #12230** (Qwen3-Next-80B autotuner crash on TRT-LLM 1.3.0rc7 aarch64). Hopper-aarch64 isn't identical to Blackwell-aarch64 but the autotuner code is shared — pin to RC15 or v1.2 stable if you hit it.
10. **KV cache reuse doesn't fire mid-batch.** If all 64 requests in a batch share a system prompt, none of them reuse until the first finishes. Stagger or pre-prime the cache.
11. **NIM container catalog is mostly amd64.** Verify image arch before deploying on GH200: `docker manifest inspect <image>` to see if `linux/arm64` is published.
12. **TRT engine build of 70B requires the target GPU during build, not just at inference.** Plan capacity for build hours.
13. **`--enable_chunked_prefill`** is now default in v1.x — if you find prefill latency regressions vs v0.x, suspect chunked prefill interaction with your ISL distribution.
14. **PyPI tensorrt_llm wheels for aarch64 only became reliable in v1.3.0rc13 (Apr 2026).** Earlier versions: use NGC container or build from source.

---

## 12. Confidence & Gaps

### High confidence
- aarch64 wheel/build story (multiple corroborating sources: install docs, GH issues, forum thread, release notes)
- FP8 risk surface (multiple open GitHub issues, dated workarounds)
- NIM 2.0.5 GH200 supported-model matrix (direct from NVIDIA docs)
- NVAIE licensing ($4,500/GPU/yr, $1/GPU/hr cloud, 90-day eval, 16-GPU dev free)
- Quantization path table (cross-referenced two NVIDIA docs)
- Speculative decoding speedups on H200 (NVIDIA dev blog with exact numbers)
- KV-cache-reuse mechanism and GH200 NVLink-C2C offload advantage
- v1.0 made PyTorch backend default; LLM API stable (verified release notes)

### Medium confidence
- Exact v1.3.0rc15 date (21 May 2026) — WebFetch returned ambiguous year in one query; cross-referenced against NGC catalog naming convention
- "32% GH200 over H100 on Llama-3.3 70B" — single-source (Baseten/Lambda Feb 2025), but methodology is credible (SGLang+ShareGPT, batch=32, FP8)
- TRT-LLM vs vLLM GH200 specifically — most published comparisons are H100, not GH200; the ARM-aarch64 gap historically pushed users to vLLM on GH200

### Gaps / unknowns to chase
- **No NVIDIA-published GH200 tok/s for Llama-3.1/3.3 70B, Mistral-Large, Qwen2.5 72B, or DeepSeek.** Only Llama-3.1 8B FP8 in perf-overview. **This is the biggest documentation hole.** Next research step: run `trtllm-bench` on actual GH200 hardware for these models in bf16 and FP8 and capture numbers ourselves.
- Whether the GH-200 96GB vs 144G HBM3e perf delta is meaningful for 70B-class — no published comparison.
- Confirmation that PyTorch-backend `host_cache_size` KV offload to Grace LPDDR5X actually works as documented at scale (single 2025 Baseten data point suggests yes, but it's not a benchmark with absolute numbers).
- Whether NVFP4 listed in NIM matrix for GH200 is a documentation error or whether NIM transparently falls back to FP8 — needs verification by running the NIM image with `--precision nvfp4` on a GH200 and checking what it actually loads.
- Status of FP8 fix campaign — multiple FP8 bugs cited as "resolved" in v1.0 (Sep 2025), but new ones (#4218, #8974) opened after that. **A definitive "FP8 is stable on Hopper" signal does not exist** as of 25 May 2026. The bf16-default policy is correct.
- Real-world latency numbers (TTFT, ITL) for GH200 — almost all public data is throughput-focused.

### Source URL index (with access dates, all verified 2026-05-25)
- Release notes: https://nvidia.github.io/TensorRT-LLM/release-notes.html
- Perf overview: https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html
- Install (Linux): https://nvidia.github.io/TensorRT-LLM/installation/linux.html
- PyTorch backend: https://nvidia.github.io/TensorRT-LLM/torch.html
- Quantization features: https://nvidia.github.io/TensorRT-LLM/features/quantization.html
- KV cache reuse: https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html
- Quick start: https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html
- Speculative-decode 70B blog: https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/
- DeepSeek-R1 best perf practice: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md
- NIM LLM support matrix: https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html
- NVAIE pricing: https://docs.nvidia.com/ai-enterprise/planning-resource/licensing-guide/latest/pricing.html
- ARM64 forum thread: https://forums.developer.nvidia.com/t/arm64-gh200-llm-engine-issues/339136
- GH issue #2060 Grace Hopper support: https://github.com/NVIDIA/TensorRT-LLM/issues/2060
- GH issue #2571 GH200 install fail: https://github.com/NVIDIA/TensorRT-LLM/issues/2571
- GH issue #4218 FP8 KV cache Qwen2.5: https://github.com/NVIDIA/TensorRT-LLM/issues/4218
- GH issue #12230 DGX Spark aarch64 autotuner: https://github.com/NVIDIA/TensorRT-LLM/issues/12230
- Baseten GH200 Llama-3.3-70B test: https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/
- Lambda partner spotlight: https://lambda.ai/blog/partner-spotlight-testing-llama-3.3-70b-inference-performance-on-nvidia-gh200-with-baseten
- MLPerf v5.0 GH200 + OpenShift: https://www.redhat.com/en/blog/mlperf-inference-v50-results
- TRT-LLM Engine Builder: https://www.baseten.co/blog/automatic-llm-optimization-with-tensorrt-llm-engine-builder/
- Spheron 2026 prod deployment guide: https://www.spheron.network/blog/tensorrt-llm-production-deployment-guide/
- vLLM vs TRT-LLM 2026 comparison: https://medium.com/synthetic-futures/vllm-vs-tensorrt-llm-the-definitive-2026-comparison-for-llm-inference-ed0943fb81d2
- EAGLE3 model: https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-Eagle3
- LLM-Inference-Bench paper: https://arxiv.org/abs/2411.00136
- NIM dev-program-free announcement: https://developer.nvidia.com/blog/access-to-nvidia-nim-now-available-free-to-developer-program-members/
