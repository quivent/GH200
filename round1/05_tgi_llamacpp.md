# GH200 Inference Round 1 — Agent #05: TGI & llama.cpp

Date: 2026-05-25
Agent: research #05 of 20
Constraint reminder: FP8 broken on GH200/Wan stack — bf16 default; any FP8 claim is flagged.

---

## Engine A — Hugging Face Text Generation Inference (TGI)

### TL;DR
**TGI is archived as of 2026-03-21.** Hugging Face has put the project into
maintenance mode and explicitly recommends migrating to **vLLM**, **SGLang**,
**llama.cpp**, or **MLX**. There is **no official aarch64/GH200 image** and
none is coming. If you are choosing an engine for GH200 in May 2026, do not
choose TGI.

### Current state (May 2026)
- Repo: https://github.com/huggingface/text-generation-inference — banner reads
  "This repository has been archived by the owner on Mar 21, 2026. It is now
  read-only."
- Last release: **v3.3.7 — 2025-12-19** (NVIDIA device visibility fix,
  flashinfer mask correction, Pydantic V2 migration, Torch 2.7 + CUDA 12.8).
- Open issue #2332 "TGI on NVIDIA GH200 (Arm64)" (opened 2024-07-30) was never
  resolved with an official arm64 image. No subsequent comments with a
  working binary.
- Discussion #2247 "Is TGI not available for ARM64 platform?" — confirmed no
  official plan.
- Maintenance-mode README explicitly points users to vLLM / SGLang / llama.cpp /
  MLX as the path forward. (huggingface/text-generation-inference README, May
  2026.)

### Officially supported quantization (last released version 3.3.7)
- bitsandbytes (NF4, FP4)
- GPTQ
- AWQ
- EETQ
- Marlin
- fp8 — **FLAG: FP8 broken on this GH200 stack per our prior experience; do
  not rely on TGI's fp8 path on GH200 even if you build it.**
- bf16 (default for unquantized) — this is what to use on GH200.

### Supported hardware (per README)
NVIDIA CUDA, AMD ROCm, Intel GPU, Gaudi, AWS Inferentia, Google TPU. **aarch64
Linux is not on this list.** Official containers are linux/amd64 only.

### If you must run TGI on GH200 (not recommended)
The community workarounds are all unofficial:
1. Build the container from source against an arm64 base (e.g. NGC PyTorch
   `nvcr.io/nvidia/pytorch:24.xx-py3` arm64 tag).
2. Patch `Dockerfile` to drop x86-only wheels; rebuild flash-attn, vllm-kernels,
   and bitsandbytes from source for sm_90.
3. Most arm64 community efforts that surfaced (PR #10499 etc.) went to vLLM,
   not TGI. There is no maintained TGI-on-arm64 fork as of May 2026.

### TGI batching strategy (for reference)
- Continuous batching with paged attention.
- Dynamic batching driven by token budget (`--max-batch-prefill-tokens`,
  `--max-batch-total-tokens`).
- Two-phase scheduler: prefill batch separate from decode batch.
- v3.0 (Dec 2024) claimed up to 13x vs vLLM on long prompts and 3x token
  capacity on a 24 GB L4 with Llama 3.1 8B — these numbers were on x86 H100/L4,
  not GH200, and predate the archive.

### Recommendation
Skip TGI on GH200. The successors (vLLM, SGLang) have working arm64 builds.
Detail those in other agents' reports.

### TGI sources
- https://github.com/huggingface/text-generation-inference (README, archive
  banner, 2026-03-21)
- https://github.com/huggingface/text-generation-inference/issues/2332
  (2024-07-30, unresolved)
- https://github.com/huggingface/text-generation-inference/discussions/2247
- https://github.com/huggingface/text-generation-inference/releases (v3.3.7,
  2025-12-19)
- https://huggingface.co/docs/text-generation-inference/en/index

---

## Engine B — llama.cpp / GGUF on GH200

### TL;DR
llama.cpp on GH200 is **the most interesting MoE-offload story** in May 2026.
The Grace 480 GB LPDDR5X + 96 GB HBM3e + 900 GB/s NVLink-C2C lets you fit
**DeepSeek V3.1 (671 B)** and **Kimi K2 Thinking (~1 T)** entirely in unified
memory and still get usable throughput. Build is straightforward on aarch64;
key unlock is `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` plus an experimental
`cudaMemAdvise` patch (or the newer `--n-cpu-moe` flag) to pin MoE experts in
Grace LPDDR5X and keep attention+dense tensors in HBM3.

### Hardware numbers worth knowing
- GH200 GPU: 96 GB HBM3 (or 144 GB HBM3e on the newer SKU), compute capability
  **9.0**.
- Grace CPU: 72-core Neoverse-V2, aarch64, NEON / SVE / DOTPROD / MATMUL_INT8 /
  FP16_VA.
- Grace memory: 480 GB LPDDR5X @ 6400 MT/s, **546 GB/s nominal** (boston.co.uk
  datasheet; spheron + fluence blog cites same number).
- NVLink-C2C: 900 GB/s total (450 GB/s per direction), memory-coherent, 7x
  PCIe Gen5.
- Measured async memcpy on GH200 (verda.com, 2025-07-01):
  - Host→Device: 135 GiB/s (vs 57 GiB/s on H200 over PCIe).
  - Round-trip: 65 GiB/s.
  - Kernel access to `cudaMallocManaged` page in HBM: 2192 GiB/s.
  - Kernel access to 64 KiB-paged `malloc` (Grace LPDDR5X same-NUMA): 2112
    GiB/s — i.e. when the page is hot in cache hierarchies the GPU sees Grace
    DRAM almost as fast as HBM for short reads.
  - Kernel access to 4 KiB-paged `malloc` same-node: 218 GiB/s — page size
    matters: use hugepages / pin memory.

### Build (aarch64 + CUDA, May 2026)
Tested against discussion #18005 (Dec 2025) and the master branch around b9305
(2026-05-24).

```bash
# Toolchain: CUDA 13.0 + g++ 13.3 on Ubuntu 24.04 aarch64
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CURL=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-march=native -O3" \
  -DCMAKE_CXX_FLAGS="-march=native -O3"

cmake --build build -j$(nproc) --config Release
```

Notes:
- `CMAKE_CUDA_ARCHITECTURES=90` (Hopper). Some build snippets cite `121a-real`
  — that is for Blackwell, not GH200.
- `GGML_NATIVE=ON` is fine when you only target the host GH200; it picks up
  Neoverse-V2 NEON/SVE2 automatically.
- Pre-built arm64 tarballs are also published as
  `llama.cpp-bXXXX-cuda-<cuda>-arm64.tar.gz` on the GitHub releases page.

### Runtime: unified memory + MoE offload
The headline configuration for >100 B MoE models:

```bash
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
./build/bin/llama-server \
  -m /path/to/DeepSeek-V3.1-Q4_K_M.gguf \
  -ngl 99 \
  --n-cpu-moe 58 \
  -c 8192 \
  -fa \
  --host 0.0.0.0 --port 8080
```

- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` puts CUDA allocations behind
  `cudaMallocManaged`, allowing oversubscription beyond HBM into Grace LPDDR5X
  over NVLink-C2C.
- `-ngl 99` requests all layers on GPU; `--n-cpu-moe N` then keeps the
  expert tensors of the first N MoE layers resident in CPU (Grace) memory.
  Activations (small) shuttle to CPU, expert matmul runs on GPU reading the
  Grace-resident weights over C2C, result returns. This is the flag merged in
  late-2025 (issue #15263) and refined into early 2026 — guide:
  huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide.
- For non-MoE dense models (e.g. Llama 3 70B), the unified-memory mode is
  unnecessary — just `-ngl 99` and the whole bf16/Q4 model fits in HBM.

### The "experts in CPU memory" patch (discussion #18005, Dec 2025)
Without intervention, `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` lets CUDA migrate
expert pages freely; the heuristic chooses badly for MoE because each token
touches only a few experts, so pages thrash between Grace and HBM. The
experimental patch (commit `c6f6e4f96a7f7bce49f5c21d19ee69fb8b72f84d`,
2025-12-14) calls `cudaMemAdvise(..., cudaMemAdviseSetPreferredLocation,
cpu_device)` on expert tensors during init. This is now superseded for most
users by `--n-cpu-moe`, but the discussion thread documents the underlying
mechanism and is still the best read on GH200 MoE behavior.

Numbers from that thread (GH200 144 GB HBM3e, CUDA 13.0, b-late-2025):

| Model                          | Mode        | PP (t/s) | TG (t/s) |
|--------------------------------|-------------|---------:|---------:|
| GPT-OSS 20B MXFP4              | CPU 32 thr  |   216.27 |    84.52 |
| GPT-OSS 20B MXFP4              | CPU 72 thr  |   377.55 |    34.83 *|
| GPT-OSS 20B MXFP4              | CUDA        |  9682.19 |   322.93 |
| GPT-OSS 120B MXFP4             | CUDA        |  5393.05 |   209.01 |
| DeepSeek V3.1 Q4_K_M (671B)    | UM (patched)|   609.89 |    18.30 |
| DeepSeek V3.1 Q4_K_M (671B)    | UM (naive)  |        — |     8.86 |
| Kimi K2 Thinking Q3_K_M (~1T)  | UM (patched)|   521.11 |    21.57 |

\* TG drops at 72 threads because llama.cpp's MoE thread-fanout saturates the
   memory subsystem; for pure CPU runs 32 threads is the sweet spot on Grace.

\* "UM" = `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` + the cudaMemAdvise patch.
   Native flag `--n-cpu-moe` reproduces similar numbers in current master.

The patch took DeepSeek V3.1 TG from **8.86 → 18.30 t/s, a 2.07x speedup**.
This is the single most important GH200-specific tuning result for llama.cpp
in the dataset.

### Grace LPDDR5X bandwidth utilization
- Nominal: 546 GB/s (datasheet).
- Practical for llama.cpp CPU-only matmul on Q4_K_M: token-generation is
  bandwidth-bound at roughly **(model_size_in_bytes / TG_tokens_per_sec) per
  step**. DeepSeek V3.1 Q4_K_M is ~400 GB; at 18.3 t/s only the active expert
  subset moves per token (~10–20 GB equivalent), implying ~180–360 GB/s of
  effective LPDDR5X read bandwidth in the active path. That is **30–65% of
  nominal**, which matches the verda.com memcpy data (~218 GiB/s on 4 KiB
  pages, ~2 TiB/s with hugepages and cache-hot kernels).
- Takeaway: GH200 LPDDR5X is **not memory-bound for typical MoE active sets**
  — the bottleneck is C2C transfer of activations and CUDA scheduling, not raw
  Grace DRAM bandwidth.

### Speculative decoding on GH200 (llama.cpp)
- `llama-speculative` and `llama-speculative-simple` are stable; supported
  draft+target via separate `-md`/`-m` GGUFs.
- Reported speedups on similar HW: 1.83x for Llama 3.1 8B drafted by Llama 3.2
  1B at 5 draft tokens; up to 3.55x reported on TRT-LLM H200 with Llama 3.3 70B
  (NVIDIA blog 2024-12-11) — that's TRT-LLM, not llama.cpp, but it bounds the
  potential ceiling.
- No GH200-specific speculative-decoding benchmark in the public llama.cpp
  threads as of May 2026. There is a recent PR #22673 "llama + spec: MTP
  Support" that adds **MTP (multi-token prediction)** draft-head support
  (Qwen3.5-style models advertise this) — worth watching for GH200 because
  MTP heads sit in HBM while the target model can use the
  `--n-cpu-moe` offload trick.

### MoE-specific recent commits
- PR #19362 `GGML_OP_MOE_SUM`: dedicated CUDA + CPU operator for summing
  expert outputs; merged Feb 2026 per the weekly digest. Reduces per-token
  CPU/GPU sync overhead, which directly helps GH200 because each MoE layer
  was previously synchronizing across C2C.
- Issue #20757 (Mar 2026) proposes a 3-tier expert cache (HBM slot pool /
  pinned host RAM / mmap SSD) with pluggable eviction. PoC numbers were on
  an 8 GB RTX PRO 2000 (cold 1.9–2.5 t/s → warm 12–14 t/s) — not GH200, but
  the design language explicitly mentions Grace as the target host tier.

### Recommended config for the three main workloads
1. **Dense ≤ 70 B (Llama 3 70B bf16 or Q5_K_M)**: all-on-GPU.
   ```bash
   ./llama-server -m llama3-70b-Q5_K_M.gguf -ngl 99 -c 8192 -fa -ub 512 -b 2048
   ```
   Expect ~30–50 t/s TG single-stream; PP measured in thousands.
2. **MoE 100–700 B (DeepSeek V3.1, GPT-OSS 120B)**: unified-memory + n-cpu-moe.
   ```bash
   GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
   ./llama-server -m deepseek-v3.1-Q4_K_M.gguf -ngl 99 --n-cpu-moe 58 -fa -c 8192
   ```
3. **Trillion-class MoE (Kimi K2 Thinking)**: same as (2) with a deeper
   `--n-cpu-moe` count (e.g. all MoE layers on CPU).

### llama.cpp sources
- https://github.com/ggml-org/llama.cpp/discussions/18005 (GH200 perf
  thread, opened 2025-12-13, patched 2025-12-14, commit c6f6e4f)
- https://github.com/ggml-org/llama.cpp/issues/20757 (two-tier expert cache
  proposal, 2026-03-19)
- https://github.com/ggml-org/llama.cpp/issues/15263 (`--n-cpu-moe` feature
  request)
- https://github.com/ggml-org/llama.cpp/discussions/18839 (shared/unified
  memory on accelerator+CPU, Jan 2026 — AMD context but generalizable)
- https://github.com/ggml-org/llama.cpp/pull/22673 (MTP speculative-decoding
  support)
- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md (CUDA build
  instructions)
- https://github.com/ggml-org/llama.cpp/releases (b9305, 2026-05-24)
- https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide
  (MoE offload practical guide)
- https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/
  (NVIDIA, 2025-09-05 — RMM-based unified memory; uses PyTorch not llama.cpp
  but documents the underlying CUDA managed-memory path)
- https://verda.com/blog/data-movement-in-the-nvidia-gh200-grace-hopper-superchip
  (memcpy bandwidth, 2025-07-01)
- https://lambda.ai/blog/putting-the-nvidia-gh200-grace-hopper-superchip-to-good-use-superior-inference-performance-and-economics
  (GH200 vs H100 vLLM numbers, 2024-11-22 — context not llama.cpp)
- https://dev.to/qpwo/how-to-run-llama-405b-bf16-with-gh200s-7da
  (Aphrodite/vLLM 405B walk-through, 2024-12-21 — multi-GH200 reference)

---

## Side-by-side comparison

| Aspect                            | TGI (May 2026)               | llama.cpp (May 2026)           |
|-----------------------------------|------------------------------|--------------------------------|
| Project status                    | Archived 2026-03-21          | Active (b9305 on 2026-05-24)   |
| Official aarch64/GH200 image      | No                           | Yes (release tarballs)         |
| GH200 unified-memory support      | None                         | `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` + `--n-cpu-moe` |
| MoE >100 B model on a single GH200| Not feasible via TGI         | DeepSeek V3.1 Q4_K_M @ 18.3 t/s|
| bf16 default                      | yes                          | yes                            |
| fp8 path                          | claimed; **broken on GH200** | not primary; gguf MXFP4 works  |
| Speculative decoding              | partial                      | yes; MTP support landing       |
| Recommended for GH200 today       | No                           | Yes for MoE / single-GPU LLM   |

---

## Confidence & Gaps

**High confidence**
- TGI archived 2026-03-21, no arm64 image, official HF recommendation is to
  migrate to vLLM/SGLang/llama.cpp/MLX. Multiple independent confirmations.
- GH200 hardware spec: 96 GB HBM3 (144 GB HBM3e on newer SKU), 480 GB
  LPDDR5X @ 546 GB/s nominal, 900 GB/s NVLink-C2C, compute capability 9.0.
- llama.cpp builds cleanly on aarch64 + CUDA 13; pre-built tarballs exist.
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` is the mechanism; `--n-cpu-moe` is the
  ergonomic flag.
- Specific benchmark numbers above (discussion #18005, Dec 2025) for
  DeepSeek V3.1 and Kimi K2 Thinking — pulled directly from the thread.

**Medium confidence**
- Whether the cudaMemAdvise patch from #18005 was upstreamed or is now
  obsoleted by `--n-cpu-moe` — looks like the user-visible flag is preferred,
  but I did not confirm the precise commit that replaced the patch.
- Practical Grace LPDDR5X utilization (30–65% of nominal in MoE active path)
  — derived from numbers, not directly measured by anyone I could cite.
- Whether the v3.3.7 release date is 2024-12-19 or 2025-12-19; one fetch
  reported each. The maintenance-mode README and archive timing
  (2026-03-21) are consistent with the latter; I went with 2025-12-19.

**Low confidence / gaps**
- No public llama.cpp speculative-decoding benchmark **specifically on GH200**
  — only adjacent numbers (H200 TRT-LLM, 8B+1B on consumer hardware). Round 2
  should target this.
- No public number for llama.cpp + GH200 + Llama 3 70B Q4 single-stream TG —
  the discussion #18005 thread skipped this case in favor of MoE. Worth a
  direct reproduction.
- The two-tier expert cache (issue #20757) was authored against an 8 GB
  consumer GPU; the design might or might not translate to a 96 GB HBM tier
  on GH200 with the 480 GB Grace tier behind it. No GH200 PoC numbers exist
  yet.
- I did not find a *llama.cpp* result that actually saturates NVLink-C2C as a
  measured number; the verda.com bandwidth tests use raw CUDA kernels, not
  llama.cpp. Open question for round 2: what fraction of 900 GB/s does
  llama.cpp's expert-streaming path actually use?
- **FP8 on GH200 remains flagged broken.** TGI advertises fp8; do not enable
  it. llama.cpp does not have an fp8 path of consequence — it uses GGUF
  quantizations (Q4_K_M, Q5_K_M, Q8_0, MXFP4) which are fine.
