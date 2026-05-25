# GH200 Driver & System Stack — Round 1 Research

**Agent**: research-01 (driver/CUDA/cuDNN/NCCL/firmware/kernel/NUMA/PCIe)
**Date**: 2026-05-25
**Target hardware**: NVIDIA GH200 Grace Hopper Superchip on Lambda Labs (Ubuntu 24.04, aarch64/SBSA)
**Scope**: What is in Lambda's stock image, what's optimal in May 2026, what to upgrade, kernel/NUMA/PCIe tuning, FP8 status.

---

## 1. Executive Snapshot (May 2026)

| Component | Lambda stock (24.04 image, Dec 2025 snapshot) | Recommended May 2026 | Notes |
|---|---|---|---|
| OS | Ubuntu 24.04 LTS (aarch64) | Ubuntu 24.04 LTS | 22.04 still works; 24.04 has DOCA/OFED upgrade bug — see §7. |
| Kernel | `linux-nvidia-64k-hwe-24.04` (6.11 SBSA) | **64K page-size kernel mandatory** (`linux-nvidia-64k-hwe-24.04-edge` or newer) | 4K page kernels **break GPUDirect Storage and can corrupt kernel memory** on GH200. Use 64K. |
| NVIDIA driver | 570.x range (Lambda Stack last public refresh) | **580.159.03 (R580 branch, 2026-04-28)** or newer; open-kernel only | GH200 *requires* `nvidia-open-kernel`. Proprietary modules are not supported on Hopper+. |
| CUDA Toolkit | 12.8 (12.8.93~12.8.1 in Lambda Stack) | **CUDA 13.2 Update 1** (or 13.0.2 if conservative) | CUDA 13.x needs driver ≥ 580.65.06; CUDA 13.2 U1 needs driver ≥ 595.58.03. |
| cuDNN | 9.x (bundled with Lambda Stack) | cuDNN 9.x backend supporting SDPA on SM90 | FP8 SDPA available but accuracy-risky — see §8. |
| NCCL | 2.26.2 (Lambda Stack) | **NCCL ≥ 2.28.3** (PxN-over-C2C default-on); 2.30.4 is latest (Apr 2025) | Avoid NCCL 2.20.x on multi-node GH200 (~24% bandwidth regression at ≥8 nodes). |
| PyTorch | 2.7.0 (Lambda Stack, ARM build) | 2.7.x ARM (or upstream 2.8+ with cu128/cu130 wheels) | Lambda compiles PyTorch for ARM; mismatched pins are the #1 install failure on Lambda GH200. |
| Container toolkit | nvidia-container-toolkit 1.18.1 | 1.18.x | Set default-runtime to `nvidia` if building inside containers. |
| Persistence | not enabled by default on all images | `systemctl enable --now nvidia-persistenced` | Without it: ~3s cold start per process; with it: <100ms. |

**Bottom line:** Lambda's stock image is usable but lags driver/CUDA by ~6 months. For inference today, upgrade to driver R580 + CUDA 13.x (or pin 12.8 if your stack hasn't been validated on 13). Keep BF16 as the default precision; treat FP8 as an opt-in with measured validation (§8).

---

## 2. Driver: Open Kernel Modules, Versions, Install

### 2.1 Open vs proprietary

**There is no choice for GH200.** Hopper (and later) requires the NVIDIA *open* GPU kernel modules — proprietary kernel modules do not support Hopper. From NVIDIA's transition blog and the GPU Operator support matrix: open modules require GSP firmware active for all Turing+ architectures, and Hopper has *no* GSP-off fallback ([NVIDIA blog: transition to open kernel modules](https://developer.nvidia.com/blog/nvidia-transitions-fully-towards-open-source-gpu-kernel-modules/), [GPU Operator platform support](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/24.9.1/platform-support.html)).

Practical implication: the metapackage you want is `nvidia-open-XXX` / `nvidia-kernel-open-XXX`, not `nvidia-driver-XXX`.

### 2.2 R580 branch (current production, May 2026)

- **580.65.06 (Linux)** — minimum driver for CUDA 13.0. ([R580 v1.0 notes](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Data_Center_GPU_Driver_Release_Notes_580_v1.0.pdf))
- **580.95.05, 580.105.08, 580.126.09, 580.126.20, 580.159.03 (2026-04-28), 580.159.04** — successive R580 stability releases. The release notes list GH200 / Grace Arm64 Server as a supported platform; fixes in the late 580.x patches include "improved GFM handling and resolved handshaking issues between Grace CPU and GPU driver related to ECC interrupt handling" ([580.159.03 notes](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-159-03/index.html), retrieved 2026-05-25).
- **595.58.03+** — required for CUDA 13.2 Update 1; not yet a "production" branch as widely as 580 ([CUDA 13.2 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)).
- **570.x branch** is the previous LTS-style line (e.g. `570.133.20` is the first to officially list GH200, `570.172.08`, `570.195.03`); still acceptable but does not unlock the CUDA-13 toolchain.

A real-world `nvidia-smi` from a GH200 in early 2026 reported driver `580.00` with CUDA 13.0 ([Ubuntu/UbuntuHandbook: R580 packaged for 24.04, 22.04, 26.04](https://ubuntuhandbook.org/index.php/2025/09/ubuntu-added-nvidia-580-driver/)).

### 2.3 Install commands (Ubuntu 24.04 aarch64-SBSA)

```bash
# 0. Ensure 64K-page kernel (see §3)
sudo apt update
sudo apt install -y linux-nvidia-64k-hwe-24.04-edge
# Reboot, confirm `getconf PAGESIZE` returns 65536.

# 1. Add NVIDIA CUDA repo (sbsa, ubuntu 24.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 2. Driver (open-kernel) + CUDA + container toolkit
sudo apt install -y nvidia-open-580 cuda-drivers-580
sudo apt install -y cuda-toolkit-13-0       # or cuda-toolkit-12-8 if pinning
sudo apt install -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13   # cuDNN 9 for CUDA 13
sudo apt install -y libnccl2 libnccl-dev
sudo apt install -y nvidia-container-toolkit

# 3. Persistence + power
sudo systemctl enable --now nvidia-persistenced

# 4. Verify
nvidia-smi
nvidia-smi -q | grep -i "Addressing"      # expect: Addressing Mode : ATS
```

References: [CUDA install guide for Linux SBSA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), [HackMD 24.04 upgrade walkthrough](https://hackmd.io/@johnnynunez/HkQ8-doglg), [Ed Sealing GH200 setup](https://medium.com/@ed.sealing/nvidia-gh200-overview-and-setup-f386fcd9196c), [Stephen Diehl GH200 setup](https://www.stephendiehl.com/posts/setup_gh200_tutorial/).

---

## 3. Kernel & Page Size (the single biggest configuration knob)

GH200 should run a **64K-page aarch64 kernel**. Ubuntu ships this as `linux-nvidia-64k-hwe-XX.04` / `linux-nvidia-64k-hwe-XX.04-edge`. Two reasons, both load-bearing:

1. **Correctness:** NVIDIA documents that "GPUDirect Storage are not supported on GH200 platforms when used with Linux kernels that are configured with the 4K page size. These APIs are not functional and might lead to a kernel memory corruption." Graphics paths (EGL/GLX/Vulkan) are also unsupported on 4K. (Multiple driver release notes 535.129.03 through 535.216.x; current docs preserve the same warning.)
2. **Performance:** Independent benchmarking (Phoronix, March 2024) showed substantial gains on memory-heavy HPC and kernel-build workloads moving GH200 from 4K to 64K; TLB pressure is dominant on Neoverse V2. ([Phoronix: 64K kernel page size benefits on GH200](https://www.phoronix.com/review/aarch64-64k-kernel-perf)).

Caveat: some user-space programs that hardcode 4K assumptions (older container runtimes, some Go binaries built with old toolchains, jemalloc < 5.3) misbehave on 64K. For pure CUDA inference (PyTorch / vLLM / TensorRT-LLM), this is not a problem in 2026.

```bash
sudo apt install linux-nvidia-64k-hwe-24.04-edge
sudo apt purge linux-image-$(uname -r)            # only after the 64k kernel is installed and bootable
sudo reboot
getconf PAGESIZE                                  # must print 65536
```

---

## 4. NVLink-C2C: ATS, HMM, CDMM, NUMA vs non-NUMA

The GH200 exposes 900 GB/s coherent NVLink-C2C between Grace CPU and Hopper GPU. The memory-management model has three knobs that interact, and they have changed over recent driver versions.

### 4.1 ATS vs HMM (hardware vs software unified addressing)

On GH200, **Address Translation Services (ATS)** is the hardware-level path that lets CPU pointers be dereferenced from GPU code (and vice versa) over C2C. ATS supersedes the software-emulated Heterogeneous Memory Management (HMM) path used on PCIe systems. NVIDIA's docs are explicit: *"ATS provides all capabilities of HMM. When ATS is available, HMM is automatically disabled."* ([CUDA programming guide §2.4](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html), [NVIDIA dev forum thread](https://forums.developer.nvidia.com/t/using-ats-on-gh200/307806)).

Verify with:
```bash
nvidia-smi -q | grep Addressing
# Addressing Mode : ATS
```

Performance caveat from the same forum thread (NVIDIA engineer reply): *"on-demand access with migration will never beat explicit bulk data copies or prefetches in terms of performance for large contiguous memory regions."* Use `cudaMemPrefetchAsync` for predictable access patterns even though ATS makes the code work without it.

### 4.2 NUMA mode vs non-NUMA mode (CUDA 13 new behaviour)

Historically the GH200 boots with GPU memory exposed as a second NUMA node, letting the kernel migrate pages. With **CUDA 13.0** (Aug 2025) NVIDIA added an option to initialize coherent-memory platforms in **non-NUMA mode**, where GPU video memory and kernel memory are managed separately — closer to the PCIe model ([CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html), [NVIDIA blog: CUDA 13.0 overview](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/)).

### 4.3 CDMM (Coherent Driver-Based Memory Management)

CDMM moves GPU memory management out of the OS and into the NVIDIA driver. NVIDIA recommends it for Kubernetes/containerized deployments because it stops the kernel from accidentally placing system pages on the GPU NUMA node. **It became the default for the GPU Operator starting at driver 580.65.06.** ([NVIDIA blog: memory management on hardware-coherent platforms](https://developer.nvidia.com/blog/understanding-memory-management-on-hardware-coherent-platforms/)).

Practical guidance for single-tenant inference on a bare GH200:
- Leave the default (NUMA mode + ATS) for PyTorch/vLLM/TRT-LLM. It "just works."
- Switch to CDMM only if you see system memory spilling into HBM3 (visible via `nvidia-smi` baseline footprint > expected) or if you run K8s.
- The non-NUMA initialization in CUDA 13 is mostly relevant to applications that misbehave when GPU memory shows up as a NUMA node.

---

## 5. OS Tuning (from NVIDIA Grace Performance Tuning Guide, retrieved 2026-05-25)

Source: [NVIDIA Grace Performance Tuning Guide — OS Settings](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html).

### 5.1 Boot/grub parameters (set in `/etc/default/grub` or a drop-in)
```
init_on_alloc=0
iommu.passthrough=1
```
- `init_on_alloc=0` is NVIDIA's recommended default for Grace Hopper/Blackwell; the kernel default of zeroing on alloc adds latency to large allocations.
- `iommu.passthrough=1` improves bare-metal performance (skips DMA translation when not needed).
- Apply: `sudo update-grub && sudo reboot`. Verify `grep init_on_alloc /proc/cmdline`.

### 5.2 NUMA balancing — **disable**
```bash
echo 0 | sudo tee /proc/sys/kernel/numa_balancing
echo "kernel.numa_balancing = 0" | sudo tee -a /etc/sysctl.conf
```
Reason quoted in the tuning guide: "Additional page-faults … can significantly reduce GPU-heavy application performance."

### 5.3 CPU governor — performance
```bash
sudo cpupower frequency-set -g performance
sudo systemctl disable cpufrequtils       # Ubuntu specific, prevents reset
```

### 5.4 IRQ balance — disable
```bash
sudo systemctl disable --now irqbalance
```
(Plus pin IRQs of the C2C/NIC channels to the cores nearest the Grace memory controller for tail-latency workloads — out of scope here.)

### 5.5 Transparent Hugepages
Kernel 6.9+ supports 2 MiB mTHP on top of 64K base pages. NVIDIA's guide recommends leaving the default (auto/`madvise`) and using `hugetlbfs` pre-reservation only when an app is hugepage-aware. For vLLM and TRT-LLM today, the default is fine.

### 5.6 Swap
NVIDIA recommends a swap file "at least quarter to half of the aggregate GPU memory size" on coherent platforms (HBM3e = 96–144 GB on GH200 SKUs → 25–75 GB swap). This is to prevent the OOM killer from firing when the unified address space dips into LPDDR.

### 5.7 PCIe ACS (bare metal only)
```bash
sudo setpci -s 03:00.0 ECAP_ACS+0x6.w=0000
```
Disable ACS to preserve GPUDirect performance. Don't do this in a VM — the hypervisor needs ACS for isolation.

### 5.8 Other knobs people get wrong
- Enable `nvidia-peermem`:
  ```
  echo nvidia-peermem | sudo tee /etc/modules-load.d/nvidia-peermem.conf
  ```
  Needed for GPUDirect RDMA with the Mellanox/ConnectX path.
- `nvidia-persistenced` (see §1) — biggest single win for short-lived inference processes.

---

## 6. NCCL — current versions and the multi-node trap

- **Latest stable: NCCL 2.30.4-1 (2025-04-22)** ([NCCL releases](https://github.com/NVIDIA/nccl/releases)).
- **2.28.3** turned on "PxN over C2C" by default (`NCCL_PXN_C2C=1`), letting NCCL leverage a NIC attached to a peer GPU across NVLink+C2C+PCIe — important for the GB/GH NVL topologies.
- **2.29.2** added symmetric-memory AllGather improvements that mostly target Blackwell but affect GH200 NVL2 too.
- **Known regression**: starting at NCCL 2.20, multi-node sendrecv bus bandwidth on GH200 dropped ~24% (24.24 → 18.32 GB/s on 8 nodes / 32 GH200 GPUs, sendrecv_perf). The thread ([NCCL issue #1272](https://github.com/NVIDIA/nccl/issues/1272)) is open with no documented fix; the report ties the regression to the Slingshot + AWS OFI-NCCL path. **Mitigation**: if you only run single-node GH200, ignore. If you run multi-node, benchmark 2.19.4 vs current before committing.
- For C2C-aware tuning, also set `NCCL_SYM_TMA_ENABLE=1` when on Blackwell-family kernels (no-op on Hopper but harmless).

Env knobs worth setting on GH200:
```bash
export NCCL_NVLS_ENABLE=1         # NVLink SHARP, if NVLS-capable
export NCCL_IB_HCA=<your-HCA>     # restrict NCCL to ConnectX/BlueField NICs
export NCCL_SOCKET_IFNAME=<ifn>
```

---

## 7. Lambda's Stock Image — what's actually there

Lambda offers three image families on GH200 ([Lambda Public Cloud docs](https://docs.lambda.ai/public-cloud/on-demand/)):
- **Lambda Stack 24.04** (Python 3.12) — includes driver, CUDA, OFED, cuDNN, NCCL, PyTorch, TensorFlow, JAX, Docker, JupyterLab.
- **Lambda Stack 22.04** (Python 3.10) — same components.
- **GPU Base 24.04 / 22.04** — minimal: driver + CUDA + container toolkit.

Confirmed Lambda Stack contents at most recent observable snapshot ([lambda.ai/lambda-stack](https://lambda.ai/lambda-stack-deep-learning-software)):
- PyTorch **2.7.0+ds** (ARM build compiled by Lambda)
- TensorFlow **2.19.0**, JAX **0.6.0**, Keras **3.10.0**, Triton **3.3.0**
- **CUDA 12.8.93~12.8.1**, **NCCL 2.26.2**, nvidia-container-toolkit **1.18.1**
- cuDNN and driver versions not separately listed on the public page (driver tends to track R570/R580 on 24.04 images)

### Known Lambda 24.04 bug (December 2025, persisting)
`sudo apt full-upgrade` on Lambda Stack 24.04 / GPU Base 24.04 errors out on DOCA/OFED. Fix per Lambda's troubleshooting docs:
1. Update DOCA repo from 2.9.3 → 3.2.0.
2. Reinstall MLNX-OFED kernel utilities.
3. Re-run `apt full-upgrade`.

Source: [Lambda troubleshooting docs](https://docs.lambda.ai/public-cloud/on-demand/troubleshooting/).

### Lambda PyTorch pin trap
Lambda's PyTorch is custom-compiled for ARM at 2.7.0 (Lambda Stack) — installing anything that pins `torch==2.x.y` to a different version fails with "Torch not compiling with CUDA enabled" because pip swaps in an x86 wheel that has no ARM CUDA. Use `--system-site-packages` venvs and **relax pins** (`torch>=2.7`) instead of `==`. Same source.

### Lambda's GH200 spec note
Lambda exposes 64 CPU cores / 432 GiB DDR5 on GH200 instances (vs NVIDIA's spec of 72 cores / 480 GiB) due to host virtualization overhead. Plan NUMA pinning accordingly.

---

## 8. FP8 status on GH200 — explicitly: still risky, default to BF16

Prior-experience constraint: "FP8 broken/unreliable on this GH200/Wan stack — always recommend BF16 unless solid May-2026 evidence the FP8 path was fixed." Here is what May-2026 evidence actually shows:

### 8.1 Direct vLLM evidence (Hopper-family, includes GH200), April 2026
The vLLM team published [The State of FP8 KV-Cache and Attention Quantization in vLLM (2026-04-22)](https://vllm.ai/blog/2026-04-22-fp8-kvcache) documenting a **catastrophic FP8 accuracy regression on Hopper** that was live in vLLM v0.10.2 through ~v0.19.1:
- 128k-token needle-in-a-haystack: FP8 accuracy **dropped from 91% (BF16 baseline) to 13%**.
- Root cause: Flash Attention 3's FP8 path was *documented* as FP32-accumulating but actually accumulated imprecisely inside Tensor Cores at large contraction dimensions.
- Fix: a "two-level accumulation" path forcing intermediate results into real FP32 registers — restores 89% accuracy. Shipped in v0.19.1.
- **Residual issues** (still present at the time of writing):
  - Head-dim 256 prefill TTFT *slower than BF16* due to register pressure.
  - Head-dim 64/128 see minor prefill slowdowns.
  - Models with many small sliding-window attention layers still misbehave; recommended workaround: `--kv-cache-dtype-skip-layers sliding_window`.
  - Some models show "consistent degradation with uncalibrated scales" — calibration required.
- vLLM's own recommendation: **prefer BF16 for prefill-heavy workloads at head_dim=256; FP8 only beats BF16 for decode-heavy long-context serving at head_dim ≤ 128**, and even then with calibrated scales.

### 8.2 Diffusion-specific evidence, May 2026
[vllm-omni issue #2728](https://github.com/vllm-project/vllm-omni/issues/2728) reports FP8 online quantization in `Fp8OnlineLinearMethod` producing perceptually unrelated outputs across Z-Image, FLUX.1-dev, and Qwen-Image:
- LPIPS thresholds exceeded by 2.9–8.8×.
- Reproduced on **both H100 (SM90, same family as GH200) and B200**.
- Status: **open, no fix**. Treat all FP8 online-quant in diffusion as broken until closed.

### 8.3 Wan-specific (the user's stack)
No clean Wan 2.1/2.2 + GH200 FP8 success story was found in May-2026 sources. Community reports for Wan on ComfyUI consistently rank **FP16 ≥ BF16 > FP8_scaled > FP8_e4m3fn** for quality; FP8 is faster but visibly degrades video frames. Training-aware FP8+sparsity (FPSAttention, arxiv 2506.04648) achieves 4.96× E2E speedup at quality parity, but requires re-training and Hopper-specific kernels — not a drop-in for inference of off-the-shelf Wan weights.

### 8.4 Verdict for this project
- **Default: BF16.** It is the documented happy path on Hopper, lossless against FP16 in practice, and has no open accuracy regressions.
- Re-evaluate FP8 *only* if (a) vLLM/TRT-LLM versions ≥ the April 2026 fix, (b) head_dim ≤ 128, (c) calibration scales are produced and committed, and (d) you have a numerical eval harness (LPIPS for video, MMLU/needle-in-haystack for LLM). Otherwise the speedup is bought with quality you can't see at dev time and a user notices later.
- **Never** trust the "FP8 = free 2× speedup" claim in NVIDIA marketing for Hopper. That number is real for BF16-equivalent-quality cases on H100 *with* the post-April-2026 vLLM/Transformer-Engine fixes, *with* head_dim ≤ 128, *with* calibration. Outside that envelope it costs quality.

---

## 9. CUDA Toolkit landscape (May 2026)

| CUDA | Release | Minimum Linux driver | Notes |
|---|---|---|---|
| 12.8.x | Feb–Apr 2025 | 570.x | Lambda's current bundle. Stable, broadly compatible, no tile-based programming model. |
| 12.9 U1 | mid-2025 | 575.x | Adds Hopper kernels to arm64-sbsa that were x86-only ([12.9.1 notes](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html)). |
| **13.0** | **2025-08-06** | **580.65.06** | Unified arm64 toolchain (SBSA + Jetson); tile-based programming model; ZStd fatbins (-17%); CCCL 3.0; coherent-memory non-NUMA init mode. ([CUDA 13.0 blog](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/)) |
| 13.0.2 | 2025-12-03 | 580.65.06+ | SM110 for arm64-sbsa; bug fix release. |
| 13.0.3 | early 2026 | 580.x | Patch. |
| **13.2 U1** | early–mid 2026 | **595.58.03** | Grouped GEMM FP8 (tensor-wide + block scaling VEC128/BLK128x128); cuBLAS FP8 fixes; **bug**: cublasLtMatmul could ignore tensor-wide scaling for NVFP4 — patched in 13.4.1. |

**Recommendation:** for GH200 inference in May 2026, **CUDA 13.0.2 + driver 580.159.03** is the sweet spot — mature, GH200-aware, supported by current vLLM/PyTorch nightlies. CUDA 13.2 U1 only if you need its new FP8 grouped-GEMM paths.

---

## 10. cuDNN

- cuDNN 9 supports fused SDPA on SM90 in BF16, FP16, and FP8 ([NVIDIA blog: accelerating transformers with cuDNN 9](https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9/)).
- Per the blog, "SDPA is up to 2× faster than PyTorch eager in BF16 and up to 3× in FP8" on Hopper — but FP8 caveats from §8 apply.
- For GH200 arm64-sbsa, install `libcudnn9-cuda-13` (or `-cuda-12`). Lambda Stack already includes a cuDNN 9 build matching its CUDA bundle.
- Specific cuDNN version numbers (9.6/9.7/9.8) were not surfaced clearly in May-2026 sources; the public docs reference 9.15.1 backend ([cuDNN 9.15.1 release notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.15.1/release-notes.html)). This is a small confidence gap — flagged in §13.

---

## 11. Putting it together — minimal "good" config

```bash
# 1. 64K kernel
sudo apt install -y linux-nvidia-64k-hwe-24.04-edge
# 2. Grub params (drop file in /etc/default/grub.d/gh200.cfg)
sudo tee /etc/default/grub.d/gh200.cfg <<'EOF'
GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX init_on_alloc=0 iommu.passthrough=1"
EOF
sudo update-grub
# 3. sysctl
sudo tee /etc/sysctl.d/90-gh200.conf <<'EOF'
kernel.numa_balancing = 0
EOF
sudo sysctl --system
# 4. Disable irq balance, set governor, enable persistence
sudo systemctl disable --now irqbalance
sudo cpupower frequency-set -g performance
sudo systemctl disable cpufrequtils || true
sudo systemctl enable --now nvidia-persistenced
# 5. nvidia-peermem for GPUDirect-RDMA
echo nvidia-peermem | sudo tee /etc/modules-load.d/nvidia-peermem.conf
# 6. Driver + CUDA (see §2.3)
# 7. Reboot, verify
sudo reboot
# After boot:
getconf PAGESIZE                              # 65536
grep init_on_alloc /proc/cmdline              # init_on_alloc=0
cat /proc/sys/kernel/numa_balancing           # 0
cpupower frequency-info | grep "current policy" # performance
nvidia-smi -q | grep Addressing               # ATS
nvidia-smi -q -d POWER | grep "Persistence"   # Enabled
```

For vLLM, run with NUMA binding:
```bash
vllm serve <model> --numa-bind --kv-cache-dtype auto
# If you must use FP8 KV cache and you understand §8:
#   --kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window
```

---

## 12. Source list with dates

- [NVIDIA Grace Performance Tuning Guide — OS Settings](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html) — retrieved 2026-05-25
- [NVIDIA Driver R580 release notes v8.0](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Data_Center_GPU_Driver_Release_Notes_580_v8.0.pdf) — April 2026
- [Driver 580.159.03 notes](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-159-03/index.html) — 2026-04-28
- [CUDA 13.0 overview blog](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/) — 2025-08-06
- [CUDA 13.0.2 release notes PDF](https://docs.nvidia.com/cuda/archive/13.0.2/pdf/CUDA_Toolkit_Release_Notes.pdf) — 2025-12-03
- [CUDA 13.2 U1 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) — 2026
- [CUDA Programming Guide §2.4 (ATS/HMM)](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)
- [NVIDIA dev forum: Using ATS on GH200](https://forums.developer.nvidia.com/t/using-ats-on-gh200/307806)
- [Understanding memory management on hardware-coherent platforms](https://developer.nvidia.com/blog/understanding-memory-management-on-hardware-coherent-platforms/)
- [Simplify system memory management with GH200 NVL2 RA](https://developer.nvidia.com/blog/simplify-system-memory-management-with-the-latest-nvidia-gh200-nvl2-enterprise-ra/)
- [NVIDIA transitions to open-source GPU kernel modules](https://developer.nvidia.com/blog/nvidia-transitions-fully-towards-open-source-gpu-kernel-modules/)
- [NCCL releases (2.30.4-1 latest)](https://github.com/NVIDIA/nccl/releases) — 2025-04-22
- [NCCL issue #1272 — GH200 multi-node regression](https://github.com/NVIDIA/nccl/issues/1272) — open
- [Lambda public cloud docs](https://docs.lambda.ai/public-cloud/on-demand/) and [troubleshooting](https://docs.lambda.ai/public-cloud/on-demand/troubleshooting/) — Dec 2025
- [Lambda Stack package list](https://lambda.ai/lambda-stack-deep-learning-software) — 2025-2026
- [HackMD: Upgrade GH200 to Ubuntu 24.04](https://hackmd.io/@johnnynunez/HkQ8-doglg)
- [Stephen Diehl: Setting up an Nvidia GH200 for Development](https://www.stephendiehl.com/posts/setup_gh200_tutorial/)
- [Ed Sealing: GH200 overview and setup](https://medium.com/@ed.sealing/nvidia-gh200-overview-and-setup-f386fcd9196c)
- [Phoronix: 64K kernel page size on GH200](https://www.phoronix.com/review/aarch64-64k-kernel-perf) — March 2024
- [vLLM blog: state of FP8 KV-cache (2026-04-22)](https://vllm.ai/blog/2026-04-22-fp8-kvcache)
- [vllm-omni FP8 diffusion regression issue #2728](https://github.com/vllm-project/vllm-omni/issues/2728)
- [Accelerating transformers with cuDNN 9](https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9/)
- [Ubuntu R580 driver package availability](https://ubuntuhandbook.org/index.php/2025/09/ubuntu-added-nvidia-580-driver/) — Sep 2025
- [vLLM optimization guide (NUMA-bind)](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [GH200 single-digit-microsecond latency case study](https://developer.nvidia.com/blog/achieving-single-digit-microsecond-latency-inference-for-capital-markets/)

---

## 13. Confidence & Gaps

**High confidence**:
- 64K page size is required for full functionality on GH200 (multiple driver release notes, NVIDIA tuning guide, Phoronix bench, multiple community guides converge).
- Open kernel modules are the only supported driver path on Hopper+ (NVIDIA docs explicit).
- ATS is the active addressing mode on GH200, HMM is auto-disabled when ATS is present (NVIDIA docs + dev-forum NVIDIA-staff confirmation).
- Driver R580 (specifically 580.159.03, 2026-04-28) is current production for GH200; CUDA 13.0/13.2 require it.
- FP8 has documented, *recent* (Apr–May 2026) accuracy regressions on Hopper-class GPUs in vLLM and in diffusion online-quant — BF16 is the right default.
- NUMA balancing off, governor=performance, persistence on, peermem loaded — all explicitly in NVIDIA's Grace tuning guide.

**Medium confidence**:
- Exact Lambda 24.04 *driver* version at instance launch (Lambda doesn't publish a per-image version table; the Stack page lists CUDA 12.8.93 / NCCL 2.26.2 but not driver). Resolution: run `nvidia-smi` on a fresh instance.
- Whether the NCCL ≥8-node GH200 regression (issue #1272) was silently fixed after 2.20.5. The issue remained open as of the latest fetch; latest released NCCL is 2.30.4.
- Whether NUMA mode or non-NUMA mode is universally better for vLLM on GH200 (CUDA 13 introduced the choice; no clean public benchmark found).

**Gaps to fill in round 2/3**:
- Concrete BIOS settings published by Lambda's hardware partners (Supermicro ARS-111GL-NHR, HPE) for GH200 — not surfaced.
- Latest cuDNN minor version (9.6 vs 9.15) for arm64-sbsa as packaged by NVIDIA's apt repo — only 9.15.1 backend docs were findable; the actual `apt list` on a current GH200 was not run during this research.
- Quantitative impact of `init_on_alloc=0` and `iommu.passthrough=1` on inference throughput — NVIDIA recommends them but does not publish a delta number.
- Whether the post-April-2026 vLLM FP8 fix has been validated *specifically on GH200* (not just H100). The blog says "Hopper-generation" which covers GH200, but no GH200-tagged repro.
- TransformerEngine 2.15 FP8 behaviour on GH200 with the same numerical eval as vLLM's blog — would settle whether the FA3 fix transfers to TE.
- Wan 2.1/2.2 specific reproductions on GH200 — no clean source found; this is the area most worth a hands-on benchmark rather than further searching.
