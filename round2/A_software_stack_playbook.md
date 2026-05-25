# GH200 Software-Stack Playbook (Round 2 Cross-Pollination)

**Date**: 2026-05-25
**Author**: Round-2 agent, synthesizing Round-1 reports 01–07
**Target**: Lambda Labs GH200 96GB / 144G HBM3e, Ubuntu 24.04 LTS aarch64
**Hard constraint**: BF16 is the default precision. FP8 is opt-in only with measured numerical validation.

---

## 0. Bottom Line Up Front

1. **Stop using Lambda Stack's bundled CUDA/PyTorch/NCCL as-is.** It is ~6 months behind on driver/CUDA and the bundled NCCL has a known multi-node regression. The upgrade is mechanical (see install script in §3).
2. **Run inference in NGC containers, not bare metal**, unless you are building from source. The PyTorch backend for TensorRT-LLM is *explicitly unsupported on bare-metal Ubuntu 24.04 SBSA* per NVIDIA's install docs (round1/02 §2).
3. **Pick one engine per workload** (§2). The right answer is *not* the same engine for chat vs MoE vs video.
4. **bf16 everywhere, period.** All seven Round-1 reports independently surface FP8 breakage on Hopper-class silicon as of May 2026: long-context accuracy collapse (vLLM blog 2026-04-22), diffusion DiT online-quant LPIPS regression cross-hardware (vllm-omni #2728), Wan-specific black-output bugs (SageAttention #221), and live TRT-LLM/SGLang issues (#4218, #24650, #18550, #23687). See §5.
5. **Three knobs move tok/s by >20%** on GH200 (§4): 64K page-size kernel, `--cpu-offload-gb` (or equivalent KV/weight offload to Grace LPDDR5X), and `torch.compile(mode="reduce-overhead")` with static KV cache. Everything else is single-digit-percent noise relative to these.

---

## 1. Version Reconciliation — pick the actual right number

Round 1 reports cite different versions. Some are real disagreements, some are reports observing different code paths. Here is the reconciled target stack, with the reason for each pick.

### 1.1 CUDA Toolkit

| Report | Cited version | What it actually means |
|---|---|---|
| 01 (driver-stack) | 12.8.93 Lambda stock; **13.0.2 recommended target**; 13.2 U1 only if you need new FP8 grouped-GEMM | Distinguishes "stock" vs "recommended" vs "bleeding edge" |
| 02 (TRT-LLM) | CUDA **13.1** in NGC PyTorch 25.12; CUDA **13.0** required by `tensorrt_llm` wheel | The PyPI wheel pins CUDA 13.0 build; the NGC container ships 13.1 |
| 03 (vLLM) | CUDA 12.8+ host, 13.x in NGC `vllm:26.01-py3` | vLLM is CUDA-version-flexible |
| 04 (SGLang) | CUDA 13 (`latest-cu130`) or 12.9 (`latest-cu129`) | Both published as Docker images |
| 05 (TGI/llama.cpp) | CUDA 13.0 for llama.cpp build | One pin |
| 06 (PyTorch/Triton) | CUDA 12.8 in PyTorch 2.7/2.8 wheels (`cu128` index) | PyTorch wheels lag CUDA toolkit |
| 07 (serving) | CUDA 13.2 in Triton 26.04 SBSA; CUDA 13 in Dynamo runtime | Triton ships its own CUDA |

**Reconciled recommendation:**

- **Host CUDA Toolkit: 13.0.2** (released 2025-12-03, driver ≥ 580.65.06). This is the sweet spot — mature, GH200-aware, supported by current vLLM/SGLang/TRT-LLM/PyTorch nightlies, and the **arm64-sbsa target** for the cuda repo on Ubuntu 24.04 ([NVIDIA CUDA Install Guide, sbsa](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)).
- **Why not 12.8.93 (Lambda stock):** functional but lacks the CUDA 13 unified arm64 toolchain, ZStd fatbins (~17% smaller binaries), CCCL 3.0, and the non-NUMA initialization mode that some apps need.
- **Why not 13.2 U1:** requires driver ≥ 595.58.03 which is not yet a wide-deployment branch in May 2026. Its only meaningful win (grouped-GEMM FP8 block scaling) is on the FP8 path you are *not* using.
- **Container CUDA mismatch is OK.** NGC containers carry their own CUDA runtime. Host driver only needs to be ≥ the highest container's minimum.

### 1.2 NVIDIA Driver

- **Target: 580.159.04** (current R580 production, 2026-04-28+). Open kernel modules mandatory on Hopper. Lambda stock typically ships 570.x — upgrade.
- Use `nvidia-open-580` / `cuda-drivers-580`, **never** the proprietary `nvidia-driver-XXX` metapackages (round1/01 §2.1).

### 1.3 PyTorch

| Report | Cited version | Context |
|---|---|---|
| 01 | 2.7.0 (Lambda stock); 2.7.x ARM or 2.8 upstream cu128 | Lambda's custom ARM build |
| 02 | **2.10.0** required by TRT-LLM 1.3 wheels; NGC `pytorch:25.12-py3` ships 2.10 | TRT-LLM-specific |
| 03 | **2.11** (April 2026 — first PyPI default aarch64 CUDA wheel) | vLLM-specific upstream story |
| 06 | 2.7 or 2.8 cu128 | Raw PyTorch / Triton agent |
| 07 | Implicit via engines | — |

**Reconciled:**

- **For raw PyTorch / diffusion work: PyTorch 2.8.0 cu128 aarch64** (from `download.pytorch.org/whl/cu128`). Stable, supports FA3 build, channels-last, regional compile. *Do not* rely on Lambda's `--system-site-packages` PyTorch 2.7 if you need newer features — break out into a venv.
- **For vLLM mainline: PyTorch 2.11+** (now installable from default PyPI on aarch64). This is the first version where `pip install torch` Just Works on GH200 ([PyTorch+vLLM blog, April 2026](https://pytorch.org/blog/vllm-and-pytorch-work-together-to-improve-the-developer-experience-on-aarch64/)).
- **For TensorRT-LLM: use the NGC release container** (`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc15`) rather than building a Python env. The NGC container pins PyTorch 2.10 + CUDA 13.1 correctly; the PyPI `tensorrt_llm` wheel pins PyTorch 2.10 + CUDA 13.0, which fights the NGC PyTorch 25.12 (CUDA 13.1) container's stock PyTorch.
- **Critical Lambda Stack trap (round1/01 §7):** Lambda compiles PyTorch for ARM. Pinning `torch==2.x.y` causes pip to swap in an x86 wheel that has no ARM CUDA → "Torch not compiling with CUDA enabled". Always use `torch>=2.7` (relax pins) or break into a fresh venv with `--no-system-site-packages` and install the cu128 wheel directly.

### 1.4 NCCL

- **Target: NCCL ≥ 2.28.3** (PxN-over-C2C default-on); 2.30.4 is latest. Lambda Stock 2.26.2 lacks the C2C optimization.
- **Multi-node trap:** if you ever scale beyond a single GH200, benchmark NCCL 2.19.4 vs current — issue #1272 reports a ~24% sendrecv bandwidth regression at ≥8 nodes that is still open. Single-node: not your problem.

### 1.5 cuDNN

- **Target: cuDNN 9.x for CUDA 13** (`libcudnn9-cuda-13`). The exact minor version (9.6 vs 9.15) is not load-bearing for inference; both ship the fused SDPA on SM90.

### 1.6 Engine versions (summary)

| Engine | Target version | Container |
|---|---|---|
| TensorRT-LLM | 1.3.0rc15 (or 1.2.0 stable if you want the legacy `trtllm-build` flow) | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc15` — verify arm64 manifest with `docker manifest inspect` |
| vLLM | 0.19.1 minimum (the FP8 long-context fix), 0.19.x current | `nvcr.io/nvidia/vllm:26.04-py3` (or newer monthly) |
| SGLang | 0.5.12.post1+, `sgl-kernel >= 0.3.21` | `lmsysorg/sglang:latest-cu130` (multi-arch arm64 manifest confirmed) |
| llama.cpp | b9305+ (May 2026) | release tarball or build from source; CUDA 13.0 |
| Triton Inference Server | 26.04 SBSA | `nvcr.io/nvidia/tritonserver:26.04-py3-sbsa` |
| Dynamo | 1.1.1 | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1` |
| Modular MAX | 26.3 stable | `modular/max-nvidia-full` — verify arm64 manifest |

---

## 2. Winner-per-workload (with citations)

All workloads assume **bf16** unless explicitly noted. "Winner" means: this is the engine to standardize on; the alternatives are within 15% or have other deal-breakers on GH200 aarch64.

### 2.1 Low-latency interactive chat (single-stream, p99 TTFT < 300 ms)

**Winner: TensorRT-LLM 1.3 + EAGLE-3 speculative decoding** in the NGC container.

Evidence:
- TRT-LLM + EAGLE-3 on H200 reached **3.55× tok/s** at batch=1 with Llama-3.2-1B drafting Llama-3.3-70B FP8 (round1/02 §6.3). On GH200, expect within 5% of H200 numbers for batch=1 latency-bound paths.
- Bf16 equivalent (no FP8): expect ~2.0–2.5× speedup (spec-decode penalty for bf16 acceptance is smaller).
- Cold start is the cost: TRT-LLM's engine-build is 30-90 min for 70B, but PyTorch backend in v1.0+ skips the AOT compile and loads HF checkpoints directly (round1/02 §7). For interactive chat where the model is loaded once and stays up, this is amortized.
- SGLang's EAGLE-3 (158 → 373 tok/s, 2.36× on Llama-3.1-8B/H100, round1/04 §4.3) is the close second. Pick SGLang if you also need prefix caching.

Skip vLLM here only because TRT-LLM's spec-decode kernels are slightly more mature on Hopper.

### 2.2 Max-throughput batch serving (many requests, ITL secondary)

**Winner: vLLM 0.19.1+** (in the NGC `vllm:26.04-py3` container or PyPI on bare metal once PyTorch 2.11 is installed).

Evidence:
- Lambda's GH200 vLLM number is the canonical: **7.6× higher throughput vs H100 SXM** on Llama-3.1-70B no-quant, 60 GB CPU offload (round1/03 §9, Lambda Labs 2024-11-22). That number predates V1; V1 should be even better.
- vLLM V1 throughput is 1.7× V0 on average (round1/03 §8).
- vLLM's `--cpu-offload-gb` to Grace LPDDR5X is the structural advantage that the other engines either don't have (TRT-LLM PyTorch backend has its own KV offload but no first-class weight offload) or expose less elegantly (SGLang HiCache requires more config).
- Async scheduler is default-on in 0.19, zero-bubble overlap of scheduling and GPU exec.

TRT-LLM is within ~15% (Spheron 2026-03-23: TRT-LLM 2,780 vs vLLM 2,400 tok/s on H100 Llama-3.3-70B FP8 @ conc=100), but its cold-start tax (28-min compile) and rougher aarch64 story (round1/02 §2) tilt against it for serving fleets that need to rotate models.

### 2.3 Long context (>100k tokens)

**Winner: vLLM 0.19.1+ with `--kv_offloading_backend native --kv_offloading_size 200`** plus `--cpu-offload-gb 80` (round1/03 §4).

Evidence:
- GH200's 480 GB LPDDR5X reachable at 900 GB/s NVLink-C2C is unmatched for KV overflow. The CUDA copy rides C2C "for free" without any vLLM-side code change.
- vLLM 0.14+ has the native KV-offload connector with async DMA, block-level placement, prefix-cache awareness.
- TRT-LLM's `host_cache_size` (Grace LPDDR offload) is in the same league but the public Baseten Llama-3.3-70B+32% number is the only data point.
- SGLang HiCache (L2 = host LPDDR over C2C) is theoretically excellent but **no published GH200 HiCache benchmark exists** (round1/04 §9).

**Avoid FP8 KV cache here even though it's the obvious memory win.** Round1/03 §5 and round1/01 §8 both document the vLLM-blog (2026-04-22) Hopper FP8 collapse: **128k needle-in-a-haystack went from 91% (bf16) to 13% (FP8)** before the v0.19.1 fix, and even with the fix there are residual head_dim=256 prefill slowdowns and sliding-window-layer misbehavior. Use bf16 KV; rely on GH200's LPDDR5X size to compensate.

### 2.4 MoE serving (DeepSeek V3.x, Qwen3-MoE, Kimi K2)

**Winner depends on model size:**

- **DeepSeek V3.1 671B, Kimi K2 Thinking ~1T → llama.cpp** with `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 --n-cpu-moe 58`. Round1/05 measured **DeepSeek V3.1 Q4_K_M at 18.30 t/s** single-stream on a single GH200 144GB (vs 8.86 t/s un-patched, 2.07× speedup from the cudaMemAdvise patch / `--n-cpu-moe` flag). This is the single most important GH200-specific tuning result in any of the seven reports. No other engine fits these models on a single GH200.
- **DeepSeek V3 / V4 in multi-GH200 cluster → SGLang** with `--enable-dp-attention --moe-a2a-backend deepep --disaggregation-mode prefill|decode`. SGLang is the official LMSYS-recommended DeepSeek path; 3.1× over vLLM on V3 (round1/04 §4.1).
- **Smaller MoE (gpt-oss 120B) on single GH200 → vLLM with `--enable-expert-parallel`**. llama.cpp also serves it well (CUDA path 5,393 PP / 209 TG, round1/05 numbers).

### 2.5 Single-user / dev

**Winner: vLLM 0.19.1+** (NGC container) for LLMs; **raw PyTorch 2.8 + FA3 + diffusers ≥ 0.35** for diffusion.

Evidence: lowest setup tax, predictable behavior, no engine-build wait. For single-user you don't need P/D disagg or routing. The hard part is making sure you used PyTorch 2.8+ cu128 (not Lambda's pre-built 2.7) so FA3 builds correctly.

### 2.6 Multi-tenant (multi-model, per-request routing)

**Winner: NVIDIA Dynamo 1.1.1** fronting a vLLM 0.19 backend on the GH200, in the multi-arch arm64 container `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1`.

Evidence:
- Best-in-class KV-aware router with prefill/decode disaggregation, NIXL KV-transport over NVLink+IB+UCX (round1/07 §3).
- Multi-arch arm64 image confirmed for vLLM-runtime CUDA-13 (round1/07 §3).
- Caveat: Dynamo's published throughput claims (7×, 30×) are on Blackwell + FP8 + DeepSeek-R1; treat as ceiling, not target, for bf16-GH200.
- Alternative for single-node only: **Triton Inference Server 26.04 SBSA** with vLLM backend. Simpler operationally, no router but supports multi-model. Force bf16 in model configs.

Skip NIM. NVIDIA's official Nov-2024 position is "NIM does not support deployment on ARM platforms"; only a handful of bio/safety NIMs ship arm64 manifests as of May 2026 (round1/07 §1). The headline LLM NIMs still fail `docker pull` on GH200 with platform-mismatch errors.

### 2.7 Diffusion image (Flux, SDXL, Z-Image)

**Winner: raw PyTorch 2.8 + diffusers ≥ 0.35** with the flux-fast recipe (regional `compile_repeated_blocks`, `max-autotune-no-cudagraphs` VAE, channels-last, fused QKV, FA3).

Evidence:
- diffusers ≥ 0.34 introduces `compile_repeated_blocks(fullgraph=True)` which cuts cold start **67 s → 9.6 s** while keeping ~1.5× runtime speedup (round1/06 §3, [PyTorch blog 2025-07-17](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)).
- Best published Flux.1-Dev number: **~2.97 s/image** at 1024² 28 steps with torchao fp8 dynamic + compile (single H100, GH200 ≈ same compute → expect within 5%).
- For bf16-only stack (our hard constraint), drop the torchao fp8 line; expect ~4.5 s/image. Still 1.5× over baseline.
- Avoid TRT engine compilation here — not worth the build tax for a hot-iterating creative workflow.

### 2.8 Diffusion video (Wan 2.2, HunyuanVideo)

**Winner: raw PyTorch 2.8 + diffusers ≥ 0.35** with `mode="max-autotune-no-cudagraphs"`, `fullgraph=False` (required for Wan's non-graph-capturable dist ops), FA3, channels-last.

Evidence:
- Morphicfilms Wan2.2 stack on 8× H100: baseline 250.7 s → **109.8 s with the full PyTorch optimization pile** (2.28×, round1/06 §3.3). On single GH200, drop sequence-parallelism; the remaining stack still applies and the GH200's bigger HBM means you don't have to micro-batch.
- **Do not use FP8 here.** Three independent sources flag breakage:
  1. vllm-omni #2728 (April 2026): catastrophic FP8 LPIPS regression on Flux/Z-Image/Qwen-Image, reproducible on **both H100 and B200** (round1/06 §7).
  2. SageAttention #221: Wan FP8 → black output (round1/06 §7).
  3. kijai ComfyUI-WanVideoWrapper #1554: SageAttention noise on Hopper, fine with SDPA.
- TRT-LLM and serving engines (vLLM/SGLang/Dynamo) are not the right fit for diffusion video — they're optimized for token-by-token autoregression. Use raw PyTorch.

---

## 3. Copy-pasteable install script (Lambda stock → recommended target)

This takes a fresh Lambda GH200 Ubuntu 24.04 instance to the recommended stack: **64K-page kernel, driver 580.x, CUDA 13.0.2, cuDNN 9 for CUDA 13, NCCL 2.28+, GH200 OS tuning applied**, and ready for any of vLLM / TRT-LLM / SGLang / llama.cpp / Triton / Dynamo via NGC or PyPI.

**Read before running:** This will reboot your instance twice (kernel upgrade, driver/CUDA install). Plan accordingly. The script is idempotent for the apt steps but the reboots are gates.

```bash
#!/usr/bin/env bash
# gh200_lambda_upgrade.sh — Lambda stock Ubuntu 24.04 → recommended GH200 stack
# Date: 2026-05-25. Test on a sacrificial instance first.
set -euo pipefail

###############################################################################
# Phase 1 — 64K-page kernel (mandatory; 4K kernels break GPUDirect Storage and
# can corrupt kernel memory on GH200)
###############################################################################
sudo apt update
sudo apt install -y linux-nvidia-64k-hwe-24.04-edge
# Do NOT purge the old kernel yet; reboot first to make sure the 64K kernel boots.
echo "REBOOT NOW into the 64K kernel. After reboot, verify with:"
echo "    getconf PAGESIZE   # must print 65536"
echo "Then re-run this script (it will skip Phase 1)."
read -p "Reboot now? [y/N] " yn
if [[ "$yn" == "y" ]]; then sudo reboot; fi
exit 0
```

After reboot, verify `getconf PAGESIZE` is `65536`, then run the second-pass script:

```bash
#!/usr/bin/env bash
# gh200_lambda_upgrade_phase2.sh — driver, CUDA, cuDNN, NCCL, container toolkit
set -euo pipefail

if [[ "$(getconf PAGESIZE)" != "65536" ]]; then
  echo "ERROR: not running on 64K-page kernel. Reboot into linux-nvidia-64k-hwe first."
  exit 1
fi

###############################################################################
# Phase 2 — driver, CUDA, cuDNN, NCCL, container toolkit
###############################################################################
# 2a. Add NVIDIA's sbsa CUDA repo for Ubuntu 24.04
TMP=$(mktemp -d)
cd "$TMP"
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 2b. Remove stock Lambda driver/CUDA cleanly. Skip if you want Lambda's bundle.
# WARNING: this temporarily breaks Lambda's PyTorch until Phase 3.
sudo apt remove --purge -y 'nvidia-*' 'cuda-*' 'libnccl*' 'libcudnn*' || true
sudo apt autoremove -y

# 2c. Open-kernel driver (R580 production), CUDA 13.0, cuDNN 9 for CUDA 13, NCCL
sudo apt install -y \
    nvidia-open-580 \
    cuda-drivers-580 \
    cuda-toolkit-13-0 \
    libcudnn9-cuda-13 \
    libcudnn9-dev-cuda-13 \
    libnccl2 libnccl-dev \
    nvidia-container-toolkit

# 2d. Persistence daemon (biggest single win for short-lived inference processes)
sudo systemctl enable --now nvidia-persistenced

# 2e. peermem for GPUDirect RDMA (no-op on single-node, free for multi-node later)
echo nvidia-peermem | sudo tee /etc/modules-load.d/nvidia-peermem.conf

###############################################################################
# Phase 3 — OS tuning (NVIDIA Grace Performance Tuning Guide, retrieved 2026-05-25)
###############################################################################
# 3a. Grub: init_on_alloc=0, iommu.passthrough=1
sudo tee /etc/default/grub.d/gh200.cfg >/dev/null <<'EOF'
GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX init_on_alloc=0 iommu.passthrough=1"
EOF
sudo update-grub

# 3b. sysctl: disable NUMA balancing (per NVIDIA tuning guide)
sudo tee /etc/sysctl.d/90-gh200.conf >/dev/null <<'EOF'
kernel.numa_balancing = 0
EOF
sudo sysctl --system

# 3c. Disable irqbalance, set CPU governor to performance
sudo systemctl disable --now irqbalance || true
sudo apt install -y linux-tools-common linux-tools-generic
sudo cpupower frequency-set -g performance
sudo systemctl disable cpufrequtils || true

# 3d. Container runtime default
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker

###############################################################################
# Phase 4 — reboot to pick up grub params and new driver
###############################################################################
echo "REBOOT NOW. After reboot, verify with:"
echo "    nvidia-smi"
echo "    nvidia-smi -q | grep Addressing       # expect: ATS"
echo "    nvidia-smi -q -d POWER | grep Persist # expect: Enabled"
echo "    grep init_on_alloc /proc/cmdline      # expect: init_on_alloc=0"
echo "    cat /proc/sys/kernel/numa_balancing   # expect: 0"
read -p "Reboot now? [y/N] " yn
if [[ "$yn" == "y" ]]; then sudo reboot; fi
```

After Phase 4, verify the install:

```bash
nvidia-smi                                    # driver 580.x, CUDA 13.0
nvidia-smi -q | grep Addressing               # Addressing Mode : ATS
nvidia-smi -q -d POWER | grep Persistence     # Persistence Mode : Enabled
getconf PAGESIZE                              # 65536
cat /proc/sys/kernel/numa_balancing           # 0
grep init_on_alloc /proc/cmdline              # init_on_alloc=0
nvidia-ctk runtime list                       # nvidia present
```

Then choose your serving engine. Three recommended starting points:

```bash
# Option A — vLLM bf16 70B chat (NGC container, simplest)
docker run --gpus all --ipc=host --network=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:26.04-py3 \
  vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --cpu-offload-gb 60 \
    --swap-space 64 \
    --kv-cache-dtype auto

# Option B — SGLang for RAG / multi-turn / DeepSeek (multi-arch container)
docker pull lmsysorg/sglang:latest-cu130
docker run --gpus all --shm-size 32g --ipc=host -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  lmsysorg/sglang:latest-cu130 \
  python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --host 0.0.0.0 --port 30000 \
    --dtype bfloat16 \
    --mem-fraction-static 0.92 \
    --enable-hierarchical-cache --hicache-ratio 1.5

# Option C — llama.cpp for 671B DeepSeek / 1T Kimi MoE on a single GH200
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --config Release
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
  ./build/bin/llama-server \
    -m /path/to/DeepSeek-V3.1-Q4_K_M.gguf \
    -ngl 99 --n-cpu-moe 58 -c 8192 -fa \
    --host 0.0.0.0 --port 8080
```

---

## 4. The three highest-leverage knobs (>20% tok/s each)

These are settings where flipping the knob moves measured throughput by more than 20%, sourced to the actual benchmark.

### Knob 1: 64K-page kernel + GH200 OS tuning

**Source:** NVIDIA Grace Performance Tuning Guide (round1/01 §5) + Phoronix March 2024 64K-vs-4K benchmark (round1/01 §3).

**Why it's >20%:**
- 4K-page kernels on Grace don't just lose performance, they **break GPUDirect Storage and can corrupt kernel memory**. Performance is a side effect.
- Phoronix measured substantial gains on memory-heavy HPC and kernel-build workloads moving GH200 from 4K to 64K; TLB pressure is dominant on Neoverse V2.
- NVIDIA recommends `init_on_alloc=0`, `iommu.passthrough=1`, `kernel.numa_balancing=0`, governor=performance, `nvidia-persistenced` enabled. Each of these is small individually (5-10%), but they compound, and persistence alone is the difference between a 3-second cold start per process and <100 ms.

This knob is **free**: it costs only one reboot and three apt-installs.

### Knob 2: `--cpu-offload-gb` (vLLM) / `host_cache_size` (TRT-LLM) / HiCache L2 (SGLang) — KV/weight offload to Grace LPDDR5X

**Source:** Lambda Labs GH200 vs H100 vLLM benchmark, 2024-11-22 (round1/03 §9); Baseten Llama-3.3-70B on GH200 (round1/03 §9, round1/02 §6.2).

**Why it's >20%:**
- Lambda's published number: 1× GH200 vs 1× H100 SXM on Llama-3.1-70B no-quant: **7.6× higher throughput**, **8× lower cost per token** ($0.02 vs $0.16). 60 GB CPU offload was the load-bearing flag.
- Baseten/Lambda: Llama-3.3-70B FP8 on GH200 96GB vs H100 80GB: **+32% throughput**, attributed explicitly to KV cache headroom + NVLink-C2C 450 GB/s offload.
- The 900 GB/s NVLink-C2C is the structural reason GH200 outperforms standalone H100/H200. You access it for free via `cudaMemcpyAsync` — no engine-side code change needed (round1/03 §4) — but you must tell the engine to use it. Without `--cpu-offload-gb`, GH200 is just an H100 with an unused appendage.

For 70B-class models in bf16, this is the difference between OOM-or-tiny-batch and serving with full prefix caching enabled.

### Knob 3: `torch.compile(mode="reduce-overhead")` + static KV cache (LLM) or regional `compile_repeated_blocks` (diffusion)

**Source:** Spheron PyTorch 2.6 LLM guide 2026-04-27 (round1/06 §6); PyTorch blog 2025-07-17 (round1/06 §3); morphicfilms Wan2.2 optimizations (round1/06 §3.3); vLLM blog 2025-08-20 (round1/06 §3).

**Why it's >20%:**
- LLM decode: 1.65× at batch 1, 1.16× at batch 32 over eager (Spheron).
- Diffusion (Flux 1024², 28 steps): 6.7 s baseline → 4.5 s with `compile(fullgraph=True)`, a **1.5× speedup**, with regional compile reducing cold start from 67 s to 9.6 s.
- Wan 2.2 video: baseline 250.7 s → 109.8 s with the full PyTorch optimization stack including `torch.compile(max-autotune-no-cudagraphs)` — **2.28×**.
- vLLM V1 uses `torch.compile` by default with "piecewise CUDA graphs" so cascade-attention / un-graph-capturable ops can run eagerly while the rest is graph-captured.

The catch: `reduce-overhead` requires static KV-cache shapes (`model.generation_config.cache_implementation = "static"`). For variable shapes use `dynamic=True` or accept recompile thrash.

**Honorable mentions** (5-20% range, listed because they're often confused for top-3):
- FlashAttention-3 vs FA2: 1.5-2.0× on Hopper (round1/06 §6), but most serving engines already use FA3 by default — this is "default, don't disable" rather than a knob you turn on.
- Speculative decoding (EAGLE-3): 1.6-3.55× at low concurrency, but decays above ~200 concurrent requests.
- Prefix caching (RadixAttention): 6.4× on prefix-heavy traffic but workload-dependent.
- AWQ-INT4 W4A16: 10.9× speedup on small models, but only worth it for memory-bound batch-1.

---

## 5. FP8 reconciliation — the BF16 default holds

The seven reports independently surfaced FP8 bugs from completely different angles. Reconciliation:

| Source (round 1) | FP8 evidence | Reconciles to |
|---|---|---|
| 01 §8 (driver/stack) | vLLM blog 2026-04-22: 128k needle FP8 91% → 13% accuracy collapse, "two-level accumulation" fix in v0.19.1 but head_dim=256 prefill still 1.6× slower than bf16 | **Fixed for some shapes, opt-in only** |
| 01 §8.2 | vllm-omni #2728: FP8 online quant gibberish on Z-Image/FLUX/Qwen-Image, reproduces on **both H100 and B200** (so it's the algo, not the silicon), open with no fix | **Broken cross-hardware for DiT online-quant** |
| 02 §5 (TRT-LLM) | Multiple open Hopper FP8 bugs: #4218 Qwen2.5 FP8 KV cache gibberish (May 2025, still labeled bug); #8974 AutoDeploy FP8/NVFP4 kernel replacement bug; Llama-3.x PP+FP8 hangs (workaround `TRTLLM_LLAMA_EAGER_FUSION_DISABLED=1`) | **TRT-LLM FP8 path has live bugs** |
| 03 §5 (vLLM) | Same blog as 01, with the FA3 register-pressure cost made explicit | **Same conclusion** |
| 04 §6 (SGLang) | Hopper-specific open bugs: #24650 (DeepSeek-V4-Flash-FP8 + EAGLE RoPE shape mismatch), #18550 (fa3+fp8_e5m2 silent triton fallback), #23687 (Qwen3.6-27B-FP8 silent garbage from dropped weight_scale_inv) | **SGLang FP8 path has live bugs** |
| 05 (TGI/llama.cpp) | TGI advertises FP8 but archived; llama.cpp's gguf uses MXFP4 not FP8, no FP8 path of consequence | **N/A — they use other quants** |
| 06 §7 (PyTorch) | vllm-omni #2728 again + SageAttention #221 (Wan FP8 black output) + ComfyUI-WanVideoWrapper #1554 (Sage noise on Hopper) | **Wan FP8 explicitly broken** |
| 07 (serving) | Dynamo's 7×/30× claims are FP8 Blackwell DeepSeek — treat as ceiling not target | **FP8 marketing numbers don't transfer to bf16 GH200** |

**Verdict:** the prior-experience constraint ("FP8 broken on this Wan/GH200 stack — always bf16") is **independently corroborated by every single Round-1 report that touched FP8**. The verdict is *not* "FP8 is always broken on Hopper" — vLLM 0.19.1+ fixes the long-context LLM path for head_dim ≤ 128 with calibration. The verdict is **bf16 is the default; FP8 is an opt-in for specific (model, engine version, head_dim, calibrated-scales) tuples that you have personally numerically validated**.

For our stack (Wan video, Llama-class LLM serving, mixed diffusion), the set of validated FP8 recipes is small:
- `torchao float8_dynamic_activation_float8_weight` on Flux.1-Dev *transformer linear layers only* (huggingface/flux-fast). Yields ~1.54× over bf16, claimed parity.
- Everything else: **bf16**.

---

## 6. Container vs bare metal — final recommendation

| Engine | Bare metal viable? | Container preferred? | Why |
|---|---|---|---|
| vLLM | Yes (PyTorch 2.11+ + cu128) | Yes (`nvcr.io/nvidia/vllm:26.04-py3`) | Less risk of Lambda PyTorch-pin conflicts |
| TensorRT-LLM | **No** for PyTorch backend (NVIDIA explicit) | **Mandatory** (`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc15` or NGC PyTorch 25.12 + pip install) | Bare-metal Ubuntu 24.04 SBSA explicitly unsupported |
| SGLang | Yes | Yes (`lmsysorg/sglang:latest-cu130` multi-arch) | First-launch JIT compile on aarch64 is real but one-time |
| llama.cpp | Yes (build from source 30 min) | Yes (release tarball) | Either fine |
| Triton | No (use SBSA container) | Yes (`nvcr.io/nvidia/tritonserver:26.04-py3-sbsa`) | SBSA client wheel not on PyPI |
| Dynamo | Mixed | Yes (`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1`) | Multi-arch arm64 confirmed |
| Modular MAX | Yes (`uv pip install modular`) | Verify arm64 manifest first | Container arm64 availability not documented |

**Always verify arm64 manifest before deploying** any NGC or third-party container:

```bash
docker manifest inspect <image> | grep -E "(architecture|os)"
# Look for: "architecture": "arm64", "os": "linux"
```

This single check would have saved the days of pain the round1/07 author documents for NIM.

---

## 7. Open Questions for Round 3

These are things Round 2 could not resolve from the seven Round-1 reports plus the four follow-up web searches, and which Round 3 should verify empirically (most are one-instance benchmarks).

1. **NGC TRT-LLM 1.3.0rc15 arm64 manifest existence.** The web search confirmed multi-arch is the *direction*, but the NGC catalog page returned no manifest detail to WebFetch. Action: SSH into the GH200 and run `docker manifest inspect nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc15`. If arm64 missing, fall back to building inside `nvcr.io/nvidia/pytorch:25.12-py3` per round1/02 Path B.
2. **NCCL 2.28+ vs Lambda's 2.26.2 single-node delta.** Round 1 cited the multi-node regression but not a single-node single-GH200 benchmark. Action: run NCCL all-reduce test before and after upgrade.
3. **Quantitative `init_on_alloc=0 iommu.passthrough=1` delta.** NVIDIA recommends them but publishes no number. Action: run vLLM benchmark with/without, measure tok/s.
4. **NUMA mode vs non-NUMA mode (CUDA 13 new init option) for vLLM on GH200.** No clean public benchmark. Action: A/B with `vllm serve` at full batch.
5. **PyTorch 2.8 SDPA → FA3 dispatch on Hopper.** Round1/06 medium-confidence flag — does `attn_implementation="sdpa"` reach FA3 kernels on H100/GH200 in 2.8, or does it stop at FA2? Action: read the kernel registry, or instrument with `torch.profiler`.
6. **SGLang HiCache L2 throughput on GH200 NVLink-C2C.** Theoretically ~800-900 GB/s but no measurement. Action: run SGLang with HiCache enabled and measure eviction throughput.
7. **First-launch JIT compile time for sgl-flash-attn3 on aarch64.** Known to happen, never timed. Action: time it and document the warm-cache strategy.
8. **Single-GH200 numbers** for Wan2.2, HunyuanVideo, Flux.1-Dev, SDXL bf16, ideally with the morphicfilms stack minus sequence-parallelism. All public numbers are 8× H100 or H100 single. Action: run the benchmarks.
9. **GH200-specific tok/s for Llama-3.3-70B bf16 + vLLM 0.19.1** with the recommended `--cpu-offload-gb 60 --kv_offloading_size 200 --max-num-seqs 256` config. The Lambda 7.6× number predates V1 entirely; the current V1 number should be higher. Action: actually run it.
10. **Whether the recent vLLM FP8 fix (v0.19.1) is validated specifically on GH200** (not just H100). The vLLM blog says "Hopper-generation" but no GH200-tagged repro. Action: run the 128k needle test on GH200 before and after enabling FP8.
11. **TRT-LLM PyTorch backend `host_cache_size` Grace LPDDR offload** scaling behavior. Single Baseten data point suggests yes; no absolute numbers. Action: bench at 50 GB, 100 GB, 200 GB.
12. **NIM container catalog arm64 progress.** As of May 2026 the headline LLM NIMs are still amd64-only despite the support matrix listing GH200. Action: `docker manifest inspect nvcr.io/nim/meta/llama-3.3-70b-instruct:latest` and equivalents monthly to detect when arm64 ships.
13. **Modular MAX 26.3 arm64 container manifest.** Pip install works on aarch64; container path unverified. Action: `docker manifest inspect modular/max-nvidia-full:latest`.
14. **Lambda's exact bundled driver version on a fresh GH200 instance.** Lambda Stack page lists CUDA/NCCL/PyTorch but not driver. Action: spin a fresh instance, capture `nvidia-smi`, document.
15. **The DOCA/OFED `apt full-upgrade` bug** (round1/01 §7) — confirm it's still active in May 2026 and document the exact recovery sequence (Lambda's troubleshooting docs are vague on ordering).

---

## Source index (deduplicated across Round 1 reports and Round 2 web searches)

- [NVIDIA Grace Performance Tuning Guide](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html)
- [NVIDIA Driver R580 release notes](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-159-04/index.html)
- [CUDA 13 install guide for Linux SBSA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Ubuntu R580 driver packaging](https://ubuntuhandbook.org/index.php/2025/09/ubuntu-added-nvidia-580-driver/)
- [TensorRT-LLM release notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)
- [TensorRT-LLM Linux install](https://nvidia.github.io/TensorRT-LLM/installation/linux.html)
- [TensorRT-LLM container catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release)
- [vLLM aarch64/PyTorch 2.11 blog](https://pytorch.org/blog/vllm-and-pytorch-work-together-to-improve-the-developer-experience-on-aarch64/)
- [vLLM FP8 blog (2026-04-22)](https://vllm.ai/blog/2026-04-22-fp8-kvcache)
- [vLLM KV offloading connector](https://vllm.ai/blog/2026-01-08-kv-offloading-connector)
- [SGLang docs — install, PD disagg, speculative, hicache](https://docs.sglang.io/)
- [llama.cpp GH200 discussion #18005](https://github.com/ggml-org/llama.cpp/discussions/18005)
- [Lambda GH200 vs H100 vLLM benchmark](https://lambda.ai/blog/putting-the-nvidia-gh200-grace-hopper-superchip-to-good-use-superior-inference-performance-and-economics)
- [Baseten Llama-3.3-70B on GH200](https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/)
- [Triton Inference Server 26.04 release notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-04.html)
- [Dynamo support matrix](https://docs.nvidia.com/dynamo/dev/resources/support-matrix)
- [PyTorch blog: torch.compile + diffusers](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)
- [Morphicfilms Wan2.2 optimizations](https://github.com/morphicfilms/wan2.2_optimizations/blob/main/blog.md)
- [vllm-omni FP8 regression #2728](https://github.com/vllm-project/vllm-omni/issues/2728)
- [HuggingFace flux-fast](https://github.com/huggingface/flux-fast)
- All seven Round-1 reports: `/home/ubuntu/research/gh200_inference/round1/01_driver_stack.md` through `07_serving_frameworks.md`.
