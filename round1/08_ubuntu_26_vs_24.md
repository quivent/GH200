# Round 1 / Agent 08 — Ubuntu 26.04 vs 24.04 (and alternates) for GH200 Inference

**Date:** 2026-05-25
**Scope:** OS substrate selection for NVIDIA GH200 Grace-Hopper inference (bf16 default — FP8 is broken on this stack)
**Domain:** Ubuntu 26.04 LTS "Resolute Raccoon" (released 2026-04-23) vs Ubuntu 24.04 LTS "Noble Numbat" (released 2024-04-25), with RHEL 9.x / 10, Rocky, SLES 15/16, NixOS as alternates.

---

## TL;DR

- **Production-safe today (May 2026): Ubuntu 24.04 LTS "Noble" + kernel 6.8 (or HWE 6.11) + NVIDIA driver 580.x + CUDA 13.2 from NVIDIA's `ubuntu2404/sbsa` repo.** This is the only LTS-class combo that is *both* NVIDIA-prevalidated for Grace and runs on every shipping cloud image (Lambda, AWS, Azure, OCI). Lambda Stack 24.04 is the path of least resistance for bf16 LLM inference.
- **Ubuntu 26.04 "Resolute Raccoon" (kernel 7.0) is 4 weeks old and not yet supported by NVIDIA Data Center driver 580.126.16 (Feb 2026 release).** CUDA 13.2 docs do not list 26.04 as a tested SBSA distro. The native-CUDA-in-archive feature is real but ships *Ubuntu's* CUDA 13.1 build, not NVIDIA's 13.2. Open kernel module issues already reported on 26.04 + GH200.
- **CUDA 13.2 minimum aarch64 kernel for Ubuntu 24.04 = `6.8.0-87`, glibc 2.39.** For SLES 15 = 6.4.0-150700.51, glibc 2.38. For RHEL 9 = 5.14.0-611.16.1.el9_7, glibc 2.34 (RHEL is much older kernel — relevant for `sched_ext` / iouring / page-table features).
- **Open kernel module is mandatory on GH200** (Hopper SM90). Never install `nvidia-driver-*` non-open variants on GH200. Known regression: 565-open and 570-open packages fail to enumerate the GH200 GPU; 560-open and 580.126.16 are the known-good points.
- **Don't migrate production GH200 fleets to 26.04 before ~26.04.1 (Aug 2026) and an NVIDIA 590/595 release that explicitly lists ubuntu2604/sbsa.** The native-CUDA convenience is not worth the early-adopter tax for inference SLAs.

---

## 1. Version Matrix

### 1.1 Distro core components

| Distro | Released | Kernel (default) | glibc | GCC | Python | aarch64 status |
|---|---|---|---|---|---|---|
| **Ubuntu 24.04 LTS "Noble"** | 2024-04-25 | 6.8 (HWE up to 6.11/6.14) | 2.39 | 13.2 | 3.12 | First-class; 4K page default; 64K kernel via mainline PPA |
| **Ubuntu 26.04 LTS "Resolute Raccoon"** | 2026-04-23 | **7.0** | **2.43** | **15.2** (16 opt.) | **3.14** | First-class; Livepatch on Arm64 added; RVA23 baseline |
| **Ubuntu 22.04 LTS "Jammy"** | 2022-04 | 5.15 (HWE 6.8) | 2.35 | 11.4 | 3.10 | Mature; Lambda default image still 22.04 |
| **RHEL 9.6** | ~late 2025 | 5.14.0-611.x (backports to ~6.5 features) | 2.34 | 11.5 | 3.9/3.11 | NVIDIA-prevalidated for Grace |
| **RHEL 10** | 2025 | 6.12 | 2.39 | 14 | 3.12 | New; CUDA 13.2 supports it for sbsa |
| **Rocky Linux 9 / 10** | tracks RHEL | mirrors RHEL | mirrors RHEL | mirrors RHEL | mirrors RHEL | CUDA 13.2 supports both for sbsa |
| **SLES 15 SP6 (+QU1)** | 2024 | 6.4.0-150700.51 | 2.38 | 7.5/13 | 3.6/3.11 | **Only SLES variant NVIDIA pre-validated for Grace** |
| **SLES 16** | 2025 | 6.12 | 2.39 | 14 | 3.13 | CUDA 13.2 supports it for sbsa |
| **NixOS 25.05/25.11** | rolling-ish | configurable (24.11 ships 6.6, unstable up to 6.12+) | configurable | configurable | configurable | Works via `hardware.nvidia.open = true`; community-grade only |

### 1.2 NVIDIA driver / CUDA support matrix for GH200 (Hopper SM90, open module **mandatory**)

| OS | NVIDIA repo path | Driver 580.x supported? | CUDA 13.2 supported? | Native CUDA in distro? | GH200 status |
|---|---|---|---|---|---|
| Ubuntu 24.04 aarch64 | `developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/` | **Yes** (580.126.16, Feb 2026, supports Ubuntu 24.04.z where z≤3) | **Yes** (kernel 6.8.0-87, glibc 2.39) | No (use NVIDIA repo) | **Production-ready** |
| Ubuntu 22.04 aarch64 | `.../ubuntu2204/sbsa/` | Yes (580.126.16 supports z≤5) | Yes | No | Production-ready; still Lambda default |
| **Ubuntu 26.04 aarch64** | `.../ubuntu2604/` listed in repo index, **but no SBSA entry confirmed yet** | **Not in 580.126.16 supported-OS table** | **Not listed in CUDA 13.2 install guide** | **Yes (Canonical archive ships CUDA 13.1)** | **Bleeding-edge — multiple reported issues** |
| RHEL 9.y (y≤6) aarch64 | `.../rhel9/sbsa/` | Yes | Yes | No | Prevalidated by NVIDIA for Grace |
| RHEL 10 aarch64 | `.../rhel10/sbsa/` | Yes | Yes | No | New; CUDA 13.2 supports it |
| Rocky 9 / 10 aarch64 | uses RHEL repos | Yes | Yes | No | Works; not separately prevalidated |
| SLES 15 SP6 (QU1) aarch64 | `.../sles15/sbsa/` | Yes | Yes | No | **The only NVIDIA-prevalidated SLES path for Grace** |
| SLES 16 aarch64 | `.../sles16/sbsa/` | Yes | Yes | No | New; not separately prevalidated |
| NixOS aarch64 | n/a (nixpkgs) | community-managed; `hardware.nvidia.open = true` required | community-managed | n/a | Works on community reports (560.x era) — no SLA |

### 1.3 Cloud image availability (as of 2026-05-25)

| Cloud | GH200 images offered |
|---|---|
| **Lambda Cloud** | Lambda Stack 24.04 (py3.12), Lambda Stack 22.04 (py3.10), GPU Base 24.04, GPU Base 22.04, Ubuntu Server 24.04, Ubuntu Server 22.04. **No 26.04 image yet.** Default still 22.04. |
| **AWS / Azure / GCP / OCI / IBM** | All offer optimized Ubuntu 26.04 cloud images per Canonical's announcement, but none publish a GH200-specific 26.04 AMI yet (GH200 SKUs are Lambda/CoreWeave/Crusoe-heavy). |

---

## 2. Key Technical Deltas (24.04 → 26.04)

### 2.1 What 26.04 buys you for GH200 inference

- **Native CUDA in archive.** First Ubuntu release where `apt install nvidia-cuda-toolkit` pulls a fully-supported CUDA (13.1 series). For *desktop developers* this is huge. For *production inference* it's largely irrelevant — you should still use NVIDIA's own `ubuntu2604/sbsa` repo for CUDA 13.2 and matched cuDNN/NCCL/TensorRT.
- **Kernel 7.0** adds `sched_ext` (eBPF-hot-swappable schedulers), default crash dumps, retires `linux-lowlatency`. Marginal for inference; potentially useful for tuning Grace CPU CPU-side preprocess/tokenization.
- **Livepatch on Arm64 (new!).** First time Canonical Livepatch supports rebootless kernel patching on Arm. Concrete operational win for GH200 fleets — no more 5-minute reboot windows for kernel CVEs.
- **DOCA-OFED 26.01** now ships in the Ubuntu archive (no more PPA dance), per Canonical's distribution agreement with NVIDIA.
- **glibc 2.43 / GCC 15.2 / Python 3.14.** Newer is sometimes worse: prebuilt wheels (vLLM, FlashAttention, TransformerEngine) may not yet have py3.14 aarch64 wheels. **This is a real risk for May 2026 inference deployments.**
- **AMD ROCm in archive** (irrelevant for GH200 but signals Canonical's GPU posture).

### 2.2 What 26.04 costs you (today)

- **NVIDIA Data Center driver 580.126.16 (Feb 2026) does not list Ubuntu 26.04 in its supported-OS table.** You can still install the open module from source, but you are off the prevalidated path.
- **CUDA 13.2 Linux Install Guide (Apr 2026) does not list Ubuntu 26.04 as a tested arm64-sbsa distro.** Confirmed supported: Ubuntu 22.04, Ubuntu 24.04, RHEL 9, RHEL 10, Rocky 9, Rocky 10, SLES 15, SLES 16, KylinOS V11, Azure Linux 3.0.
- **Reported issues on 26.04 + NVIDIA today (May 2026):**
  - `ubuntu-drivers devices` recommends `nvidia-open-595` but `ubuntu-drivers install` installs 580-open instead.
  - `apt install nvidia-open` returns "Unable to locate package nvidia-open" on some 26.04 mirrors.
  - Sleep/resume regressions with NVIDIA driver after 24.04→26.04 upgrade.
  - General-population GH200 regression (independent of 26.04): driver packages `nvidia-driver-565-open` and `nvidia-driver-570-open` fail to enumerate GH200; `560-open` and `580.126.16` are known-good. (GitHub issue NVIDIA/open-gpu-kernel-modules#774.)
- **Python 3.14 default** breaks the standard `pip install vllm` / `pip install torch` recipe — neither ships official aarch64-cp314 wheels at May 2026. Lambda Stack 24.04 ships a known-good PyTorch 2.4.1-ARM compile precisely because of this class of issue.
- **No Lambda Cloud 26.04 image yet** — if your fleet is Lambda-resident you can't get there without manual `do-release-upgrade`, which is documented as problematic for GH200 images even from 22.04→24.04 (see DOCA-OFED 2.9→3.2 fix-up dance).

### 2.3 The 64K page-size question

- NVIDIA recommends 64K pages on Grace CPU for HPC workloads. Ubuntu 24.04 ships **4K pages by default** on aarch64; 64K variant available via the Ubuntu Mainline Kernel PPA (Phoronix benchmarked it).
- Ubuntu 26.04 also defaults to 4K. No change in default policy.
- For **LLM inference specifically** the 4K vs 64K delta is modest (single-digit %) — main savings are in HPC mem-bandwidth-bound codes. Inference is GPU-bound; bf16 throughput on Hopper is unaffected. Don't burn a week on a 64K kernel rebuild unless you've already exhausted GPU-side gains.

---

## 3. Migration Recommendation

### Decision tree for a GH200 inference deployment as of 2026-05-25

```
Q1: Do you have an active production SLA on this fleet?
├─ Yes → Stay on Ubuntu 24.04 LTS + Lambda Stack 24.04 (or RHEL 9.6 if regulated).
│        Driver 580.126.16-open. CUDA 13.2 from NVIDIA ubuntu2404/sbsa.
│        Revisit in Aug 2026 after 26.04.1 + a 590-series driver that lists 26.04.
│
└─ No (greenfield / R&D box) → Two acceptable paths:
   ├─ Path A (safe): Ubuntu 24.04 + Lambda Stack — minimal surprise.
   └─ Path B (forward-looking, accept friction): Ubuntu 26.04 + nvidia-driver-580-open
       from Ubuntu archive + CUDA 13.1 from Ubuntu archive.
       Build PyTorch/vLLM from source against py3.14, or pin a py3.12 venv.
       Validate end-to-end bf16 inference before any user traffic.
```

### Why **not** the alternates for a GH200 inference workload

- **RHEL 9.6:** Slower-moving kernel (5.14-based) means newer io_uring / `cgroup-v2` / sched features take longer to land. Use if you're already a RHEL shop or need FedRAMP / FIPS. Otherwise no advantage over Ubuntu 24.04 for inference.
- **RHEL 10 / Rocky 10 / SLES 16:** Supported by CUDA 13.2 but new and not separately prevalidated for Grace. No clear win.
- **SLES 15 SP6 + QU1:** Only choose if you specifically need NVIDIA's prevalidated SLES Grace install guide (regulated industries, NVIDIA partner SLA). Mandatory SUSE subscription.
- **NixOS:** Excellent reproducibility, but no NVIDIA prevalidation, no enterprise driver SLA, community-maintained `hardware.nvidia.open` toggle. Use for personal GH200 dev boxes, **not** production.
- **Ubuntu 22.04:** Still works (kernel 5.15 default or 6.8 HWE), but the runway shortens — many newer NVIDIA features assume 6.x and CUDA 13 supports it as a transitional target. If you're on 22.04, plan the 24.04 jump now.

### Concrete steps if you must move to 26.04 today

1. Stand up a non-production GH200 box.
2. Use Ubuntu Server 26.04 minimal install (not desktop).
3. Pin `nvidia-driver-580-open` from Canonical's archive **or** install 580.126.16 from NVIDIA's `ubuntu2604/sbsa` repo if/when it goes live (check `developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/sbsa/` — directory exists in parent index; contents not yet confirmed populated for sbsa).
4. **Verify GPU enumeration**: `nvidia-smi` must show GH200 + ECC + MIG-N/A. If not, downgrade to 580.126.09 or 560-open.
5. Create a py3.12 venv (don't use 26.04's py3.14 for inference frameworks yet).
6. Install PyTorch nightly aarch64 wheel from `download.pytorch.org/whl/nightly/cu130` — verify `torch.cuda.is_available()` and that bf16 matmul actually executes on GPU (CPU-fallback failures are a known class of bug in fresh installs).
7. **Run an inference smoke test in bf16 only** (no fp8). vLLM `--dtype bfloat16`; TRT-LLM build with `--use_fp8=False`.
8. Compare token/sec against a known 24.04 baseline before declaring the migration successful.

---

## 4. Benchmark Deltas — 24.04 vs 26.04 on GH200 Inference

**No published 26.04 vs 24.04 inference benchmark deltas exist as of 2026-05-25.** What's known:

- MLPerf Inference v4.1 / v5.0 baseline numbers on GH200 are reported on Ubuntu 22.04 + CUDA 12.3 + driver 545.14, and on Red Hat OpenShift with Supermicro GH200 servers. Neither used 26.04.
- Lambda's published GH200 benchmarks (Llama 3.1 70B: 7.6× throughput vs single H100 SXM; Llama 3.3 70B; DeepSeek-R1/V3 ~400 tok/s aggregate / ~10 tok/s/query) were all on Ubuntu 22.04 / 24.04 Lambda Stack with bf16 — no 26.04 results.
- **Expected delta:** Marginal. The kernel-side wins for inference are scheduler / NUMA-balancing on the Grace side, which are <5% on most LLM workloads (GPU-bound). The CUDA toolkit version (13.1 in 26.04 archive vs 13.2 from NVIDIA repo) matters more than the OS version. Driver version matters most.
- **Possible regression risks on 26.04:** glibc 2.43 has had reported issues with some prebuilt CUDA-linked binaries; HWE-class kernels on aarch64 occasionally have NVMe/Ethernet driver regressions that show up as I/O stalls under sustained inference load.

---

## 5. Lambda Cloud-Specific Notes

- **Default image is still Ubuntu 22.04** on Lambda ODC instances.
- **Lambda Stack supports 20.04 / 22.04 / 24.04 only** — no 26.04 Lambda Stack at this time.
- **Lambda's PyTorch 2.4.1-ARM build** is the de-facto canonical aarch64 PyTorch for GH200; available via `--system-site-packages` venvs.
- **Known issue on Lambda Stack 24.04 / GPU Base 24.04 (documented Dec 2025, still relevant):** `sudo apt full-upgrade` fails due to DOCA-OFED 2.9.3 → 3.2.0 transition. Fix: update DOCA repo, `apt update`, then `apt install -y -o Dpkg::Options::="--force-confnew" mlnx-ofed-kernel-utils`. Then run the upgrade again.
- **Lambda GH200 hardware spec quirk:** 64 vCPU cores and 432 GiB DDR5 visible (less than NVIDIA's bare-metal spec) due to virtualization overhead. Single-tenant compute, multi-tenant net/storage.
- **cuDNN license restriction on Lambda:** bundled cuDNN is licensed only for Lambda Stack's PyTorch + TensorFlow. External frameworks need symlink workarounds. Easy to trip over with vLLM/TRT-LLM source builds.

---

## 6. Confidence & Gaps

### High confidence (multiple sources, dated docs)
- Ubuntu 26.04 release date (2026-04-23), kernel 7.0, glibc 2.43, GCC 15.2, Python 3.14.
- Ubuntu 24.04 ships kernel 6.8, glibc 2.39, supported through 2029 (free) / 2034 (Pro).
- NVIDIA driver 580.126.16 (Feb 2026) explicitly supports Ubuntu 24.04.z (z≤3) and 22.04.z (z≤5) on aarch64; **does not list 26.04**.
- CUDA 13.2 install guide (Apr 2026) supported aarch64-sbsa list: Ubuntu 22.04, Ubuntu 24.04, RHEL 9 (incl 9.6), RHEL 10, Rocky 9/10, SLES 15/16, KylinOS V11, Azure Linux 3.0. **26.04 not listed.**
- Open kernel module is mandatory on GH200 (Hopper / Blackwell).
- 26.04 ships native CUDA + ROCm in archive (first Ubuntu LTS to do so).
- Lambda Cloud GH200 default image still 22.04; no 26.04 Lambda Stack yet.
- 565-open / 570-open GH200 enumeration bug is real (NVIDIA/open-gpu-kernel-modules#774).

### Medium confidence
- NVIDIA's `ubuntu2604/sbsa` repo population. Parent index lists `ubuntu2604/` but I did not confirm a populated `/sbsa/` subdir with current packages. (Worth a direct check via `curl` from a GH200 box before relying on it.)
- Whether Ubuntu 26.04's archive CUDA is 13.1.x specifically or somewhere in the 13.1.x band — multiple secondary sources say 13.1, primary Canonical docs don't pin a version on the linked release-notes page.
- RHEL 9.6 exact kernel sub-version on aarch64. CUDA guide pins 9 generically at 5.14.0-611.16.1.el9_7.

### Gaps / not investigated
- Apples-to-apples inference benchmark (e.g. Llama-3-70B bf16, batch=1, ISL=512/OSL=128) on 26.04 vs 24.04 on the *same* GH200 — does not appear to exist publicly.
- 64K-page kernel availability on 26.04 (likely same PPA pattern as 24.04, not confirmed).
- TransformerEngine / FlashAttention 3.x prebuilt wheel matrix for Python 3.14 aarch64 — appears unsupported but I did not check the latest releases on PyPI today.
- Azure Linux 3.0 / KylinOS V11 maturity for inference (listed by CUDA 13.2 but not common in Western inference deployments).
- Real-world MTBF / kernel-panic data on 26.04 GA + GH200 (release is only 4 weeks old).
- Whether Canonical's "Native NVIDIA CUDA" archive packages on 26.04 include matched cuDNN + NCCL + TensorRT, or only the toolkit core (probably only the core — confirm before basing a stack on it).

### Recommendation summary
**Stay on Ubuntu 24.04 LTS + Lambda Stack 24.04 (or roll your own with driver 580.126.16-open + CUDA 13.2 from NVIDIA's `ubuntu2404/sbsa` repo) for any GH200 inference workload with an SLA today.** Re-evaluate 26.04 in August 2026 after the 26.04.1 point-release and a 590-series NVIDIA driver that explicitly lists 26.04. For greenfield R&D, 26.04 is acceptable but expect to burn 1–3 days on packaging issues (PyTorch on py3.14 aarch64, driver/CUDA repo alignment, vLLM wheel availability).

---

## Sources

- [Ubuntu 26.04 LTS release notes (documentation.ubuntu.com)](https://documentation.ubuntu.com/release-notes/26.04/) — 2026-04-23
- [Canonical releases Ubuntu 26.04 LTS Resolute Raccoon (canonical.com)](https://canonical.com/blog/canonical-releases-ubuntu-26-04-lts-resolute-raccoon) — 2026-04-23
- [Ubuntu 26.04 LTS "Resolute Raccoon" released with Linux 7.0 (cnx-software.com)](https://www.cnx-software.com/2026/04/24/ubuntu-26-04-lts-resolute-raccoon-released-with-linux-7-0/) — 2026-04-24 (kernel/toolchain versions)
- [Ubuntu 26.04 LTS Resolute Raccoon is now available with Linux 7.0 and native CUDA (Neowin)](https://www.neowin.net/news/ubuntu-2604-lts-resolute-raccoon-is-now-available-with-linux-70-and-native-cuda/) — 2026-04
- [Ubuntu 26.04 LTS Arrives with Native CUDA and ROCm Support (BigGo Finance)](https://finance.biggo.com/news/202604300054_Ubuntu_26.04_native_CUDA_ROCm_support) — 2026-04-30
- [Introducing Kernel 6.8 for the 24.04 Noble Numbat Release (discourse.ubuntu.com)](https://discourse.ubuntu.com/t/introducing-kernel-6-8-for-the-24-04-noble-numbat-release/41958)
- [Canonical releases Ubuntu 24.04 LTS Noble Numbat (canonical.com)](https://canonical.com/blog/canonical-releases-ubuntu-24-04-noble-numbat) — 2024-04-25
- [CUDA Installation Guide for Linux 13.2 (docs.nvidia.com)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) — release Apr 2026
- [NVIDIA Data Center Driver 580.126.16 release notes (docs.nvidia.com)](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-126-16/index.html) — 2026-02-09
- [NVIDIA Open GPU Kernel Modules — issue #774 (GH200 565+ enumeration bug)](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/774)
- [NVIDIA Grace Software with Generic Linux install guide](https://docs.nvidia.com/grace-linux-install-guide.pdf)
- [NVIDIA Grace Software with RHEL install guide](https://docs.nvidia.com/grace-rhel-install-guide.pdf)
- [NVIDIA Grace Software with SLES 15 SP6 install guide](https://docs.nvidia.com/dccpu/sles-install-guide/index.html)
- [NVIDIA GPU Operator 25.3 platform support](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/25.3.0/platform-support.html)
- [Feb 2024 Update: NVIDIA Grace MGX + Canonical Support (discourse.ubuntu.com)](https://discourse.ubuntu.com/t/feb-2024-update-nvidia-grace-mgx-canonical-support/42671)
- [NVIDIA CUDA repos index (developer.download.nvidia.com)](https://developer.download.nvidia.com/compute/cuda/repos/) — confirms ubuntu2604 directory exists
- [Lambda Docs — Public Cloud On-Demand Overview](https://docs.lambda.ai/public-cloud/on-demand/) — image catalog (no 26.04)
- [Lambda Docs — Troubleshooting](https://docs.lambda.ai/public-cloud/on-demand/troubleshooting/) — DOCA-OFED 2.9→3.2 fix, GH200 PyTorch ARM notes
- [Lambda — Putting the NVIDIA GH200 Grace Hopper Superchip to good use](https://lambda.ai/blog/putting-the-nvidia-gh200-grace-hopper-superchip-to-good-use-superior-inference-performance-and-economics) — Llama-3 70B bf16 numbers
- [Lambda — How to serve DeepSeek-R1 & V3 on GH200](https://lambda.ai/blog/how-to-serve-deepseek-r1-v3-on-gh200) — 400 tok/s aggregate
- [Lambda Stack Docker images (github.com/lambdal/lambda-stack-dockerfiles)](https://github.com/lambdal/lambda-stack-dockerfiles) — supported Ubuntu versions
- [Phoronix — 64K Kernel Page Size Performance Benefits with NVIDIA GH200 Grace](https://www.phoronix.com/review/aarch64-64k-kernel-perf)
- [MLPerf Inference v5.0 results with Supermicro GH200 + Red Hat OpenShift (redhat.com)](https://www.redhat.com/en/blog/mlperf-inference-v50-results)
- [NixOS Wiki — NVIDIA (wiki.nixos.org)](https://wiki.nixos.org/wiki/NVIDIA) — `hardware.nvidia.open` requirement on Grace Hopper
- [26.04 ubuntu-drivers install won't install the recommended nvidia driver (discourse.ubuntu.com)](https://discourse.ubuntu.com/t/26-04-ubuntu-drivers-install-wont-install-the-recommended-nvidia-driver/79703)
- [Sleeping mode issue after upgrading to Ubuntu 26.04, Nvidia driver (discourse.ubuntu.com)](https://discourse.ubuntu.com/t/sleeping-mode-issue-after-upgrading-to-ubuntu-26-04-nvidia-driver/81490)
- [Unable to locate package nvidia-open on Ubuntu 26.04 (forums.developer.nvidia.com)](https://forums.developer.nvidia.com/t/error-unable-to-locate-package-nvidia-open-on-ubuntu-26-04/368145)
- [Upgrade GH200 to Ubuntu 24.04 — HackMD (johnnynunez)](https://hackmd.io/@johnnynunez/HkQ8-doglg) — manual upgrade procedure
- [Canonical announces it will distribute NVIDIA DOCA-OFED in Ubuntu](https://canonical.com/blog/canonical-announces-it-will-distribute-nvidia-doca-ofed-in-ubuntu)
- [Colfax — Installing Ubuntu 22.04 LTS over the Network on GH200](https://research.colfax-intl.com/ubuntu-22-04-netboot-on-servers-with-the-nvidia-grace-hopper-superchip/)
