# GH200 Inference Round 5 — Llama-3.3-70B Q4 on llama.cpp (Narrow & Deep)

Date: 2026-05-25
Agent: research #01, Round 5
Target: 1× GH200 96 GB HBM3, Lambda Ubuntu 24.04 aarch64, CUDA 13
Goal: highest tok/s, least friction, Llama-3.3-70B GGUF via llama.cpp
Constraint reminder: FP8 broken on this stack — irrelevant here, llama.cpp uses GGUF quants.

---

## Friction score: **2 / 5**

One-time cost: ~5 min build, ~10 min download. After that everything is one
flag. Two reasons it isn't 1/5:
1. No published GH200 + llama.cpp + Llama-3.3-70B Q4_K_M number exists.
   You're the first to measure (round 3 already flagged this).
2. KV cache type matters more than it looks — `q4_0` cache **collapses at
   long context**. You have to know to use `q8_0` for K and `f16` for V (or
   `q8_0` for both) — defaults work, but the "obvious" memory-saving choice
   is a trap.

---

## One-command quickstart

```bash
# Assume llama.cpp is built (see §3) and the GGUF is at $GGUF.
GGUF=$HOME/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf
./build/bin/llama-server -m $GGUF -ngl 99 -fa 1 -c 8192 \
  -b 2048 -ub 512 --host 0.0.0.0 --port 8080
```

That's it. Single GH200, all layers on GPU, flash-attention on, 8K context,
defaults for everything else. Expected ~40 t/s single-stream TG (bandwidth-
derived; see §4).

---

## 1. Quant choice + source + footprint

### Recommendation: **`bartowski/Llama-3.3-70B-Instruct-GGUF`, file `Llama-3.3-70B-Instruct-Q4_K_M.gguf`**

| Field         | Value                                                                                                |
|---------------|------------------------------------------------------------------------------------------------------|
| Repo          | https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF                                         |
| File          | `Llama-3.3-70B-Instruct-Q4_K_M.gguf`                                                                 |
| Size on disk  | **42.52 GB**                                                                                         |
| Bits/weight   | ~4.83 (Q4_K_M is mixed: most blocks Q4_K, some attention blocks Q6_K)                                |
| HBM weights   | ~42.5 GB (weights load 1:1 from disk into HBM)                                                       |
| KV cache @ 8K (f16) | ~2.5 GB (Llama-3.3-70B = 80 layers, 8 KV heads, head_dim=128 → 0.31 MB/token × 8192)            |
| KV cache @ 32K (f16) | ~10 GB                                                                                          |
| KV cache @ 128K (f16) | ~40 GB                                                                                         |
| Activations / overhead | ~2-4 GB                                                                                       |
| **Total HBM @ 8K**  | **~48 GB / 96 GB** — half the GPU free                                                          |
| **Total HBM @ 32K** | **~55 GB**                                                                                       |
| **Total HBM @ 128K**| **~85 GB** — at this point switch K cache to q8_0 to free ~5 GB                                 |

### Quant comparison (same repo)

| Quant   | Size GB | Notes                                                                |
|---------|--------:|----------------------------------------------------------------------|
| Q4_K_M  |   42.52 | **default**, best quality-per-byte for 4-bit on Hopper               |
| Q4_K_S  |   40.35 | -2.2 GB, ~0.5% perplexity hit, marginally faster                     |
| Q4_0    |   40.12 | legacy; designed for ARM CPU online-repacking; **slower on Hopper**  |
| IQ4_XS  |   37.90 | smallest 4-bit, importance-matrix; **CUDA matmul kernels are slower than Q4_K_M** despite less data — use only when memory matters (it doesn't, here) |
| Q5_K_M  |   49.95 | quality bump worth considering — 50 GB still leaves 40 GB for KV     |
| Q6_K    |   57.89 | near-lossless; ~25-30 t/s ceiling                                    |
| Q8_0    |   74.98 | overkill; bandwidth-bound at ~25 t/s ceiling                         |

**Pick Q4_K_M.** Better quality than IQ4_XS, faster than Q4_0 on CUDA, smaller
than Q5_K_M. The 96 GB HBM gives so much headroom that the smaller IQ4_XS
buys nothing.

If quality matters more than speed: jump to Q6_K (57.89 GB) — still fits with
30+ GB free for KV at 64K context.

---

## 2. Build (aarch64 + CUDA 13, Hopper sm_90)

### Prereqs
- CUDA 13.0 toolkit installed under `/usr/local/cuda-13.0`
- `g++` ≥ 13.3 (Ubuntu 24.04 default is fine)
- `cmake` ≥ 3.18

### Exact build commands

```bash
cd $HOME
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Sanity: pin to a known-good tag. Round 1 verified b9305 (2026-05-24).
# Master also fine; MTP support merged 2026-05-16.
# git checkout b9305

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

Notes / gotchas:
- `CMAKE_CUDA_ARCHITECTURES=90` is Hopper (GH200). **Do not** copy a
  `121-real` or `121a-real` from a Blackwell/DGX-Spark recipe — that
  produces a binary that won't run on GH200 SMs.
- `GGML_NATIVE=ON` picks up Neoverse-V2 NEON/SVE2/MATMUL_INT8 automatically.
  This matters only for the CPU fallback path; you won't hit it for a dense
  70B that fits in HBM, but it's free.
- `GGML_CURL=ON` lets `llama-server -hf bartowski/...` download GGUF
  directly. Saves a step.
- Build time on Grace's 72-core Neoverse-V2 with `-j$(nproc)`: ~2-4 min.
- If you want to skip building entirely: pre-built arm64 tarballs are on the
  GitHub releases page as `llama.cpp-bXXXX-cuda-13-arm64.tar.gz`. Friction
  drops to 1/5 if you trust those.

### Verify

```bash
./build/bin/llama-cli --version
./build/bin/llama-cli --list-devices
# Expect: "CUDA0: NVIDIA GH200 480GB, compute capability 9.0, ..."
```

---

## 3. Launch commands (the actual flags that matter)

### 3a. Max single-stream TG tok/s

```bash
GGUF=$HOME/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf

./build/bin/llama-server \
  -m $GGUF \
  -ngl 99 \
  -fa 1 \
  -c 8192 \
  -b 2048 -ub 512 \
  --no-mmap \
  --host 0.0.0.0 --port 8080
```

Why each flag:
- `-ngl 99` — all 80 layers on GPU. With 96 GB HBM and a 42.5 GB model this
  is trivial; do NOT touch `--n-cpu-moe` (Llama-3.3-70B is dense, not MoE;
  the flag does nothing for dense models).
- `-fa 1` — flash-attention. Without it, the cache has to dequantize per
  attention computation; this is a free 1.3–2x speedup on Hopper.
- `-c 8192` — context length. Bump to 32K/128K if needed; each doubling
  costs ~2.5/10/40 GB at f16.
- `-b 2048 -ub 512` — defaults are correct for dense-GPU. Increasing `-ub`
  helps PP but doesn't move TG. Don't go larger than 2048 for single-stream
  — wastes HBM.
- `--no-mmap` — on GH200 specifically, mmap of GGUF from disk can confuse
  CUDA's unified-memory heuristic if `GGML_CUDA_ENABLE_UNIFIED_MEMORY` is
  set elsewhere. Safer to disable mmap here. (Costs ~5 sec extra load time.)

**Do NOT set `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`** for this dense 70B
case. UM is only useful when the model exceeds HBM (MoE > 100B). For a
42 GB Q4_K_M model in 96 GB HBM, plain `cudaMalloc` is faster and more
predictable. Round 3 confirmed UM=1 + dense model has no upside.

### 3b. Max throughput at batch=16/64/256

```bash
./build/bin/llama-server \
  -m $GGUF \
  -ngl 99 \
  -fa 1 \
  -c 32768 \
  -np 16 \
  -b 4096 -ub 2048 \
  --cont-batching \
  --host 0.0.0.0 --port 8080
```

For batch=64 use `-np 64 -c 65536`; for batch=256 use `-np 256 -c 131072`.
Per-slot context = `-c / -np` (so np=256, c=131072 → 512 tok/slot).

Why each flag:
- `-np N` — N independent slots / parallel sequences. Each slot has its own
  KV chunk. Total KV at f16 = `(N × ctx_per_slot × 0.31 MB)`.
- `-c (N × per_slot)` — total KV pool. For 70B at f16:
  - np=16, 2K each → ~10 GB KV. Plenty of room.
  - np=64, 1K each → ~20 GB KV.
  - np=256, 512 each → ~40 GB KV. At this point switch to `-ctk q8_0`
    to halve KV → 20 GB, leaving ~30 GB free.
- `-b 4096 -ub 2048` — larger physical batch makes GPU-saturated parallel
  decode efficient. Per discussion #18308, `-ub 512` is a starting point;
  for parallel inference on Hopper bump to 2048.
- `--cont-batching` — continuous batching (default on in recent server
  builds; explicit for clarity).

**Caveat from discussion #18308**: CPU-side sampling becomes the bottleneck
at high `-np`. GPU util drops below 90% past `-np 32` even with 70B. The
fix is in-flight (backend sampling); for May 2026 expect diminishing
returns past `-np 64`.

### 3c. Max throughput WITH KV cache quantization (for very high parallelism)

```bash
./build/bin/llama-server \
  -m $GGUF \
  -ngl 99 \
  -fa 1 \
  -ctk q8_0 -ctv q8_0 \
  -c 131072 -np 256 \
  -b 4096 -ub 2048 \
  --cont-batching
```

Doubles effective parallel capacity at < 5% TG cost (§6).

---

## 4. Published Llama-3.3-70B Q4_K_M tok/s on Grace-Hopper / H100 / H200

### Direct llama.cpp number on GH200: **still none published as of 2026-05-25**

Confirmed across discussion #18005, #15013, #16578, and a wide web search
this round. The round 3 gap is unfilled. **Action: you will be the first to
publish.**

### Closest comparable: H100 PCIe 80 GB, Llama-3-70B Q4_K_M (XiongjieDai repo)

| Test          | t/s          |
|---------------|-------------:|
| pp512         | 1012.73      |
| pp1024        |   984.06     |
| tg512         |    25.03     |
| **tg1024**    |  **25.01**   |

Source: github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference (May 2024
benches, llama.cpp + CUDA 12.1.1, RunPod).

### GH200 expected vs H100 PCIe

- H100 PCIe HBM3 bandwidth: 2.0 TB/s
- GH200 96 GB HBM3 bandwidth: **4.0 TB/s** (2x)
- llama.cpp dense TG is bandwidth-bound for ≥30B Q4.
- Naive scaling: **~50 t/s tg on GH200** for Llama-3.3-70B Q4_K_M.
- Bandwidth-ceiling math: 42.5 GB/step at 4 TB/s = 94 t/s ceiling;
  llama.cpp typically hits 40-55% of ceiling on dense 70B Q4 → **40-55 t/s**.
  Matches round 3's calculation.

### Other adjacent (non-llama.cpp) GH200 numbers

| Source                   | Engine     | Quant | Throughput claim                                 |
|--------------------------|------------|-------|--------------------------------------------------|
| Baseten + Lambda blog    | TRT-LLM    | FP8   | GH200 32% faster than H100 at batch 32, ShareGPT |
| Lambda blog              | vLLM       | bf16  | GH200 4.33 t/s vs H100 0.57 t/s (H100 paged)     |
| NVIDIA TRT-LLM blog      | TRT-LLM    | FP8   | up to 3.55x with spec-decode on H200             |

All are TRT-LLM/vLLM with FP8 — **not directly comparable** to llama.cpp
Q4_K_M, and FP8 is broken on this stack anyway.

### TG on H100 SXM (slightly faster than PCIe)

H100 SXM bandwidth 3.35 TB/s vs PCIe 2.0 TB/s. Expect H100 SXM Q4_K_M TG
≈ 40 t/s. The GH200 should beat H100 SXM by ~20% on TG due to 4.0 vs 3.35
TB/s HBM (~48 t/s).

---

## 5. Speculative decoding

### Target + draft pair

- **Target**: `bartowski/Llama-3.3-70B-Instruct-GGUF :: Q4_K_M` (42.52 GB)
- **Draft**:  `bartowski/Llama-3.2-1B-Instruct-GGUF :: Q4_K_M` (0.81 GB)

Together: 43.3 GB weights + KV. Trivially fits in 96 GB.

**Tokenizer match: YES.** Llama-3.2-1B-Instruct and Llama-3.3-70B-Instruct
use the **identical** 128K Llama-3 tokenizer — required for spec decode to
work. Confirmed by Meta's model cards and verified in the medium.com
walkthrough (5x speedup with this exact pair on RTX 3090).

### Two paths

#### Path A — Classic draft-model spec decode (recommended for May 2026)

```bash
TGT=$HOME/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf
DFT=$HOME/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

./build/bin/llama-server \
  -m $TGT \
  -md $DFT \
  -ngl 99 -ngld 99 \
  -fa 1 \
  -c 8192 \
  --draft-max 16 --draft-min 5 \
  --host 0.0.0.0 --port 8080
```

Reported speedups in the wild:
- **RTX 3090 (24 GB)**: 30 → 160+ t/s (~5x), Q4_K_M target, Q4_K_M draft
  (medium.com/write-a-catalyst article).
- **General 1B→70B literature**: 2.0-2.5x is the calibrated steady-state
  on quality-aligned models; 5x is best-case on easy distributions.
- **Expected on GH200**: 1.8-2.5x sustained on conversational text;
  ~40 t/s baseline → **70-100 t/s with spec decode**.

Why GH200 should be at the lower end of the spread vs RTX 3090's 5x:
the RTX 3090 baseline is memory-starved (24 GB → 70B Q4 barely fits, KV is
tight); spec decode partly compensates for poor memory headroom. GH200's
baseline is already healthy, so the speculative gains are pure decode-step
amortization, not memory-pressure relief. Expect 1.8-2.2x.

#### Path B — MTP (Multi-Token Prediction), `--spec-type draft-mtp`

Status: PR #22673 **merged 2026-05-16** (9 days before today).

```bash
# HYPOTHETICAL — no MTP GGUF for Llama-3.3-70B exists yet
./build/bin/llama-server \
  -m $TGT_MTP_GGUF \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  -ngl 99 -fa 1 -c 8192 \
  --host 0.0.0.0 --port 8080
```

**Blocker**: only Qwen3.6-{27B, 35B-A3B} have published MTP-converted GGUFs
as of May 2026. There is **no Llama-3.3-70B MTP head** published. Meta did
not train Llama 3.3 with an MTP head; it would need to be trained
independently.

**Limitations from PR**:
- `n_parallel = 1` only (no batched MTP yet).
- Negative PP hit (D2H embedding transfers in draft path).
- Not yet in `llama-bench` (issue #22947).

**Sweet spot from PR data**: `--spec-draft-n-max 2` (81% acceptance) or
3 (72%). Larger → diminishing returns.

**Use Path A**, not Path B, until Meta or a community contributor publishes
a Llama-3.3-70B MTP GGUF.

### Expected GH200 spec-decode performance

| Config                                  | TG t/s (estimate) | Notes                            |
|-----------------------------------------|------------------:|----------------------------------|
| Baseline (Q4_K_M, fa, c=8192)           |             40-50 | Bandwidth-bound                  |
| + spec-decode (1B draft, --draft-max 16)|             70-100 | 1.8-2.2x typical                 |
| + spec-decode (1B draft, --draft-max 8) |             65-90  | Less variance, slightly lower    |
| + MTP (if a 70B MTP head ever lands)    |             80-110 | Pure speculation                 |

### Cautions

- Spec decode **hurts** at high `-np` because verification has to wait for
  draft per slot. Path A speedup ~1.0x at `-np ≥ 16`. Use spec decode for
  **interactive single-stream**, plain decode for **bulk throughput**.
- HackMD April 2026 post (round 3): spec decode can actually **slow down**
  small-active MoE targets. Not relevant here (Llama-3.3-70B is dense), but
  worth knowing.

---

## 6. KV cache type tradeoffs

The DGX Spark Nemotron-3-Nano-30B benchmark (forums.developer.nvidia.com,
2025, CUDA 13) is the cleanest public data on this. Architecturally close
enough to GH200 (Grace + CUDA 13, dense decoder) to read straight across.

### Quality (perplexity delta vs F16)

| K cache | V cache | ΔPPL (Qwen 2.5 Coder 7B) | Verdict                          |
|---------|---------|--------------------------|----------------------------------|
| F16     | F16     | 0.0000 (baseline)        | Reference                        |
| Q8_0    | Q8_0    | +0.0043                  | Essentially lossless             |
| Q8_0    | F16     | ~+0.001 (extrapolated)   | Safest "saving" config           |
| F16     | Q4_0    | small but visible        | V cache tolerates Q4_0           |
| Q8_0    | Q4_0    | small but visible        | Recommended-by-some hybrid       |
| Q4_0    | Q4_0    | large, "weird results"   | Avoid for GQA-heavy models       |

K cache is **much more sensitive** to quantization than V cache (multiple
sources, ggml-org/llama.cpp discussion #5932).

### Speed (DGX Spark, Nemotron-3-Nano-30B, context-stratified, t/s)

| Context | PP F16 | PP Q4_0 | TG F16 | TG Q4_0   |
|---------|-------:|--------:|-------:|----------:|
|  ~8K    |  371   |   363   |  14.7  |  14.2     |
|  ~16K   |  361   |   346   |  13.9  |  12.7     |
|  ~32K   |  328   |   317   |  13.5  |  11.0     |
|  ~64K   |  283   | **21.3**|  13.3  |  **8.6**  |

**The Q4_0 cliff at 64K is real.** Q4_0 KV cache is 92% **slower** at long
context due to per-attention dequantization overhead — and on the unified-
memory DGX Spark it actually consumed MORE RSS than F16 due to ggml's
buffer alignment. (Note: this collapse is partly a DGX Spark / unified-
memory artifact; on dedicated HBM the cliff is shallower but still real.)

Author's verdict: **"q8_0 is the only quantization worth running"** —
2x compression of KV, <5% speed hit at all context lengths.

### Memory (GH200, Llama-3.3-70B, 70B = 80 layers, 8 KV heads, head_dim=128)

| Cache type | Bytes/token (K+V) | KV @ 8K | KV @ 32K | KV @ 128K |
|------------|------------------:|--------:|---------:|----------:|
| F16/F16    | 320 KB            | 2.6 GB  | 10.5 GB  |  42 GB    |
| Q8_0/Q8_0  | 160 KB            | 1.3 GB  |  5.2 GB  |  21 GB    |
| Q4_0/Q4_0  |  80 KB            | 0.7 GB  |  2.6 GB  |  10.5 GB  |

### Recommendation by use case

| Use case                       | -ctk | -ctv  | Why                                       |
|--------------------------------|------|-------|-------------------------------------------|
| Single-stream, c ≤ 32K (default)| f16  | f16   | Plenty of HBM headroom, max quality      |
| Single-stream, c = 128K         | q8_0 | f16   | Save K cache (~20 GB), V quality matters  |
| High `-np` parallel, c ≤ 32K    | q8_0 | q8_0  | Effectively doubles per-slot capacity     |
| Long-context + parallel         | q8_0 | q8_0  | Same                                      |
| Anywhere                        | q4_0 | q4_0  | **Avoid** — 92% slower at long context    |

`-fa 1` is **required** when using quantized KV — otherwise dequant happens
in the inner attention loop and the perf collapses further.

---

## 7. `llama-bench` recipe (validate locally in < 5 minutes)

```bash
GGUF=$HOME/models/Llama-3.3-70B-Instruct-Q4_K_M.gguf

# ~4 minutes wall time on GH200. Sweeps PP and TG at several depths.
./build/bin/llama-bench \
  -m $GGUF \
  -ngl 99 \
  -fa 1 \
  -ub 2048 \
  -p 2048 \
  -n 128 \
  -d 0,4096,16384,65536
```

What this measures:
- `-p 2048` → prompt-processing throughput on a 2048-token prompt
- `-n 128` → token-generation throughput for 128 generated tokens
- `-d 0,4096,16384,65536` → pre-fill that many tokens of dummy KV first,
  so you see PP/TG **at depth** (the attention-cost growth curve).

Expected output shape (single line per `(p, n, d)` cell):

```
| model        | size     | params | backend | ngl | fa | ub  | test          | t/s       |
| llama 70B Q4_K_M | 42.5 GB | 70.6B | CUDA    | 99  | 1  | 2048| pp2048        | ~800-1100 |
| llama 70B Q4_K_M | 42.5 GB | 70.6B | CUDA    | 99  | 1  | 2048| tg128         | ~40-55    |
| llama 70B Q4_K_M | 42.5 GB | 70.6B | CUDA    | 99  | 1  | 2048| pp2048 @ d=65536 | ~250-450 |
| llama 70B Q4_K_M | 42.5 GB | 70.6B | CUDA    | 99  | 1  | 2048| tg128 @ d=65536  | ~25-35   |
```

To **also** sweep KV cache type (10-min variant):

```bash
./build/bin/llama-bench \
  -m $GGUF \
  -ngl 99 -fa 1 -ub 2048 \
  -p 2048 -n 128 \
  -d 0,16384,65536 \
  -ctk f16,q8_0 -ctv f16,q8_0
```

This gives a 3 (depths) × 2 (K type) × 2 (V type) × 2 (PP/TG) = **24 cells**
in about 10-12 minutes. That's the publishable result.

---

## 8. Sources

- https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF — file sizes,
  quant table.
- https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF — draft model.
- https://github.com/ggml-org/llama.cpp/discussions/18005 — GH200 perf
  thread (no 70B but reference for benchmark methodology).
- https://github.com/ggml-org/llama.cpp/discussions/15013 — CUDA scoreboard.
- https://github.com/ggml-org/llama.cpp/discussions/16578 — DGX Spark perf
  thread (closest aarch64+CUDA reference).
- https://github.com/ggml-org/llama.cpp/discussions/18308 — parallel
  inference tuning.
- https://github.com/ggml-org/llama.cpp/discussions/5932 — KV cache
  quantization quality discussion.
- https://github.com/ggml-org/llama.cpp/pull/22673 — MTP spec decoding
  PR (merged 2026-05-16).
- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md — build
  instructions.
- https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md —
  spec decode docs.
- https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md —
  llama-bench flag reference.
- https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference — H100 PCIe
  Llama-3-70B Q4_K_M reference numbers (tg1024 = 25.01 t/s).
- https://forums.developer.nvidia.com/t/kv-cache-quantization-benchmarks-on-dgx-spark-q4-0-vs-q8-0-vs-f16-llama-cpp-nemotron-30b-128k-context/365138 —
  KV cache type cliff data on aarch64+CUDA.
- https://medium.com/write-a-catalyst/how-i-doubled-my-local-llms-speed-without-buying-new-hardware-0201539eb935 —
  Llama-3.2-1B + Llama-3.3-70B spec-decode walkthrough on RTX 3090 (30 →
  160+ t/s).
- https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/ —
  GH200 vs H100 (TRT-LLM FP8, not llama.cpp; 32% throughput edge).
- https://lambda.ai/blog/partner-spotlight-testing-llama-3.3-70b-inference-performance-on-nvidia-gh200-with-baseten — same study, Lambda side.
- https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide —
  MoE offload guide (informational, dense-70B doesn't need it).

---

## 9. Confidence

**High**
- File name, size, source for Q4_K_M (42.52 GB on bartowski's repo).
- Build command (Hopper sm_90, CUDA 13, aarch64) — directly from llama.cpp
  build docs + round 1 validation.
- Launch flags for single-stream and parallel — directly from llama-server
  + llama-bench docs.
- Speculative decoding flag set (`-md`, `-ngld`, `--draft-max`, `--draft-min`)
  for Path A — verified in docs/speculative.md.
- KV cache Q4_0 cliff at long context — verified from DGX Spark data.
- No public GH200 Llama-3.3-70B Q4_K_M llama.cpp number exists (verified
  via 10+ searches this round).

**Medium**
- 40-55 t/s GH200 tg128 estimate — bandwidth math, not measured.
- 1.8-2.2x spec-decode speedup on GH200 — extrapolated from RTX 3090's 5x
  with confidence interval narrowed by GH200's healthier baseline.
- Q4_0 cliff magnitude on dedicated HBM (vs Spark unified memory) —
  expected smaller but still significant; nobody published the GH200
  number.

**Low / unmeasured**
- All concrete GH200 numbers for Llama-3.3-70B Q4_K_M. **Reproduce locally
  with the recipe in §7.**
- Whether `--no-mmap` actually helps on GH200 specifically — heuristic
  recommendation; safe but might be unnecessary.
- MTP path performance — no Llama-3.3-70B MTP GGUF exists; can't test.
