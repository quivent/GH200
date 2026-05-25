# GH200 Driver Stack — Round 3 Verification & Deepening

**Agent**: research-round3 (verification + deepening of Round 1 driver/CUDA/NCCL/kernel/Lambda claims)
**Date**: 2026-05-25
**Anchor**: [round1/01_driver_stack.md](../round1/01_driver_stack.md)
**Scope**: Drill on 6 specific load-bearing claims with primary-source verification; correct Round 1 where it was wrong; deepen where it was thin.

---

## 0. What changed since Round 1

Round 1 was directionally right but contained four factual errors and one piece of underspecified guidance. Corrections, in priority order:

| # | Round 1 claim | Round 3 finding | Status |
|---|---|---|---|
| 1 | NCCL 2.30.4 released **2025-04-22** | Actually released **2026-04-22** (GitHub release API timestamp). Off by one year. | **Corrected** |
| 2 | NCCL issue #1272 (GH200 ≥8-node 24% bandwidth drop) is **open with no documented fix** | Issue is **CLOSED since 2024-04-30**. The "fix" is the workaround `NCCL_NCHANNELS_PER_NET_PEER=4`. NVIDIA never landed a code fix; they explained the behaviour and the reporter closed it. | **Corrected** |
| 3 | "580.159.04" listed as if generally available | Confirmed: `cuda-drivers-580_580.159.04-1ubuntu1_arm64.deb` is present in NVIDIA's sbsa repo *today*. There is **no `580.165` or `580.190`**. The "newer" R580 patch is 580.159.04 only. | **Confirmed + narrowed** |
| 4 | open-gpu-kernel-modules has no recent (last 60d) GH200-loading bugs worth flagging | **There is a long-standing GH200 NVL2 device-detection regression** (#774, open since Feb 2025) affecting drivers 565+ on certain VBIOS revisions. NVIDIA's recommended fix as of Feb 2026 is a VBIOS update to `96.00.D8.00.03`, not a driver downgrade. Confirmed working by an independent reporter (mkroening) on 2026-02-19. | **New finding** |
| 5 | Lambda Stack driver pin & upgrade procedure | Lambda's *official* answer is to lower the apt priority of `/etc/apt/preferences.d/lambda-repository` (per Lambda staff @Nicholas_C on DeepTalk). They do **not** publish a current driver version. Lambda Stack 24.04 most-recently-observed contains CUDA 12.8.93 / NCCL 2.26.2; the driver tracks R570 on those images. | **Deepened** |
| 6 | "64K kernel just works in May 2026" | Confirmed via Launchpad: `linux-nvidia-64k-hwe-24.04-edge` is currently at **6.17.0-1018.18, published 2026-05-19** (six days before this writing). No public regression report for that build on GH200. | **Confirmed** |

Bottom line: **the Round 1 install recipe is still safe**, but treat NCCL #1272 as "still requires `NCCL_NCHANNELS_PER_NET_PEER=4` env var on multi-node GH200" rather than "open bug, beware"; and warn explicitly about the GH200 NVL2 + 565+ + old-VBIOS interaction (#774) for anyone with a freshly-racked Supermicro ARS-221GL-NHIR or similar.

---

## 1. Claim verification: Driver 580.159.03 in NVIDIA's sbsa apt repo

**Method**: WebFetch of `https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/` Packages index, 2026-05-25.

**Finding**: `580.159.03` is present *and* there is one newer patch:

```
cuda-drivers-580_580.65.06-0ubuntu1_arm64.deb
cuda-drivers-580_580.82.07-0ubuntu1_arm64.deb
cuda-drivers-580_580.95.05-0ubuntu1_arm64.deb
cuda-drivers-580_580.105.08-0ubuntu1_arm64.deb
cuda-drivers-580_580.126.09-1ubuntu1_arm64.deb
cuda-drivers-580_580.126.16-1ubuntu1_arm64.deb
cuda-drivers-580_580.126.20-1ubuntu1_arm64.deb
cuda-drivers-580_580.159.03-1ubuntu1_arm64.deb   ← Round 1 baseline
cuda-drivers-580_580.159.04-1ubuntu1_arm64.deb   ← current head of R580
# Newer-branch packages also present:
cuda-drivers_595.45.04-1ubuntu1_arm64.deb
cuda-drivers_595.58.03-1ubuntu1_arm64.deb
cuda-drivers_595.71.05-1ubuntu1_arm64.deb
```

**Implication for the install recipe**: bump `cuda-drivers-580` to `580.159.04`. Round 1's `580.159.03` still works but is one patch behind. If you want the CUDA 13.2 U1 toolchain (NVFP4 fixes, grouped-GEMM block scaling) you need R595 — `595.58.03` minimum, `595.71.05` is the current head.

**Source**: [NVIDIA sbsa apt repo index](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/) (fetched 2026-05-25).

---

## 2. Claim verification: Is there a newer R580 patch (580.165 / 580.190)?

**Method**: WebSearch for "580.165 OR 580.190 release notes 2026"; cross-check against the sbsa repo listing above.

**Finding**: **No.** The R580 branch heads at `580.159.04` in NVIDIA's repo today. Search returned no docs.nvidia.com release notes for any 580.165 or 580.190 build. The branches present in 2026 driver docs are 580.65.06, 580.82.07, 580.95.05, 580.105.08, 580.126.09, 580.126.16, 580.126.20, 580.159.03, 580.159.04 — and that's the entire R580 line (RN-08625-580 v8.0, April 2026 PDF).

NVIDIA has moved on to R595 (595.45.04 / 595.58.03 / 595.71.05) for CUDA 13.2-class work. R595 is required for CUDA 13.2 U1; if you stay on CUDA 13.0/13.0.2/13.0.3 you can stay on R580.

**Source**: [Driver R580 v8.0 release notes (April 2026)](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Data_Center_GPU_Driver_Release_Notes_580_v8.0.pdf); sbsa repo listing above.

---

## 3. open-gpu-kernel-modules: recent GH200 / 64K / SBSA issues (last 60 days)

**Method**: `gh issue list --repo NVIDIA/open-gpu-kernel-modules` with search filter `GH200 OR Grace OR Hopper OR SBSA OR 64K created:>2026-03-01`.

### 3.1 Hits in the strict window (post 2026-03-01)
- **#1122 — "Some GPM/DCGM statistics incorrectly use cycle count rather than elapsed time"** (opened 2026-04-26, open). Hopper/Grace-class telemetry bug, not a load failure. Low severity for inference.
- **#1128 — "Kernel Module Fails"** (opened 2026-04-28, open). Generic build failure title; not GH200-tagged in the issue body based on labels. Not a known GH200 blocker.

No new GH200-specific *boot/loading* regressions filed in the last 60 days.

### 3.2 The older issue that still bites (and got new activity in March 2026)
- **#774 — "Can not find device after 565+ on GH200 NVL2"** (opened 2025-02-02, **OPEN**, last activity 2026-03-20).
  - **Symptom**: With `nvidia-open-565` or `nvidia-open-570`, `nvidia-smi` reports "No devices were found" on a freshly-rebooted Supermicro ARS-221GL-NHIR GH200 NVL2. Driver 560.35.05 worked.
  - **Kernel log signature**:
    ```
    NVRM: numa memblock size of zero found during device start
    [drm:nv_drm_load [nvidia_drm]] *ERROR* [nvidia-drm] [GPU ID 0x00090100] Failed to allocate NvKmsKapiDevice
    ```
    NVIDIA engineer (@aritger) notes the same message appears even on the working 560 build, so it's not the actual root cause — just a confounder.
  - **NVIDIA action (2026-02-06)**: opened internal bug 5862796.
  - **NVIDIA fix recommendation (2026-02-11, @abchauhan-nv)**: update VBIOS to **`96.00.D8.00.03`** from the system vendor. Old VBIOS was `96.00.A0.00.01`.
  - **Independent confirmation (2026-02-19, @mkroening)**: "We have had the same issue and could successfully resolve it by updating to the new VBIOS version we got from our distributor."
  - **Status today**: original reporter still waiting on Supermicro for the VBIOS (2026-03-20 follow-up).
  - **Implication for Lambda**: Lambda's GH200 fleet is presumably already on a VBIOS NVIDIA has validated. The risk is only for bare-metal owners of older-VBIOS GH200 NVL2 boards. If you provision a fresh Lambda GH200 and `nvidia-smi` says "No devices were found" after a driver upgrade — VBIOS, not driver, is the variable to investigate.

### 3.3 Counter Collection Unit (CCU) bug — informational
- **#911 — "Counter Collection Unit (CCU) support for GH200"** (opened 2025-08-01, open). Affects DCGM/profiling on driver 570.172.08. Not a correctness issue for inference; matters if you depend on hardware perf counters via DCGM/Nsight.

**Source**: [NVIDIA/open-gpu-kernel-modules issues](https://github.com/NVIDIA/open-gpu-kernel-modules/issues); [issue #774 full thread](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/774); [issue #911](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/911).

---

## 4. NCCL #1272 — does 2.30.4 close it? **No, and it's already closed.**

**Method**: `gh issue view 1272 --repo NVIDIA/nccl` (full thread, all comments).

### 4.1 Actual chronology
- **2024-04-30 09:42 UTC** — Issue opened by @teojgo (CSCS). Reports 24.24 → 18.32 GB/s drop on sendrecv_perf at 8 nodes × 4 GH200 going from NCCL 2.19.4 → 2.20.5 on Slingshot + aws-ofi-nccl.
- **2024-04-30 09:57 UTC** — @sjeaugey (NVIDIA, NCCL maintainer) suggests `NCCL_NCHANNELS_PER_NET_PEER=4`.
- **2024-04-30 10:36 UTC** — Reporter confirms: setting the var **restores 24.2–24.3 GB/s**, matching 2-node and 4-node behaviour.
- **2024-04-30 12:16 UTC** — sjeaugey explains: NCCL send/recv ops are optimised for all-to-all (2 CTAs per peer pair); on sparse send/recv patterns like ring sendrecv_perf, 4 CTAs per peer helps. Speculates "some tuning inside NCCL which changes the number of channels per peer we use as we scale" caused the 8-node regression. **No code fix proposed.**
- **2024-04-30 12:51 UTC** — Reporter: "Now it's clear and I think you can close the issue now."
- **2024-04-30 15:55 UTC** — Issue closed.

### 4.2 What this means for our Round 1 advice
- The regression is **not fixed in code** in any NCCL ≥ 2.20.5. It is fixed *only* by setting `NCCL_NCHANNELS_PER_NET_PEER=4` (or higher) at runtime.
- NCCL 2.30.4-1 (released 2026-04-22, latest) does **not** close #1272 — its changelog is `nccl_param` headers + GIN elastic buffer, full stop.
- The behaviour is consistent with NVIDIA's general posture: "users should not need to tune NCCL env vars on GH200" is aspirational; for *sparse* send/recv patterns at 8+ nodes, you still must.

### 4.3 Updated guidance
For multi-node GH200 inference/training where send/recv patterns matter (pipeline parallelism, expert routing in MoE, certain RL setups):
```bash
export NCCL_NCHANNELS_PER_NET_PEER=4   # restores 8-node sendrecv to 4-node levels
# Keep the Round-1 set:
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=<your-HCA>
export NCCL_SOCKET_IFNAME=<ifn>
```
For pure all-reduce-dominated workloads (most pure DDP), leave the default — setting it to 4 *can* degrade all-to-all throughput, per sjeaugey.

**Sources**: [NCCL issue #1272 (CLOSED)](https://github.com/NVIDIA/nccl/issues/1272); [NCCL v2.30.4-1 release](https://github.com/NVIDIA/nccl/releases/tag/v2.30.4-1) (changelog text verified via GitHub API).

---

## 5. Latest NCCL release timeline (corrected dates)

Pulled directly from the GitHub Releases API, 2026-05-25:

| Version | Published | Highlights |
|---|---|---|
| nccl4py v0.2.0 | 2026-04-24 | Python bindings for NCCL 2.30 RMA/GIN/elastic |
| **v2.30.4-1** | **2026-04-22** | `nccl_param` header fix (#2105); elastic buffer with GIN. **Latest.** |
| v2.30.3-1 | 2026-04-15 | Device API/GIN enhancements; elastic buffers (LSA); symmetric memory improvements; doca-gpunetio 2.0.2-rc1 |
| v2.29.7-1 | 2026-02-27 | Device API/GIN; dynamic memory offload; hybrid symmetric ReduceScatter |
| v2.29.3-1 | 2026-02-03 | ARM hang fix (CAS with pre-gcc-10) |
| v2.29.2-1 | 2025-12-24 | Device API versioning; one-sided host APIs (ncclPutSignal/Wait) |
| v2.28.9-1 | 2025-11-10 | Large-scale hang ordering fix; GIN bug |
| v2.28.7-1 | 2025-10-18 | GPU-Initiated Networking (GIN); ncclCommRevoke fault tolerance |
| v2.28.3-1 | 2025-10-06 | **PxN-over-C2C default on**; CE collectives; symmetric memory; Device API (experimental) |

**Correction to Round 1**: I wrote "2.30.4-1 (2025-04-22)". The correct date is **2026-04-22** — only one month before this Round-3 write-up. NCCL 2.28.3-1's PxN-over-C2C default-on is also 2025-10-06, not "Round 1 era".

**Source**: `gh api repos/NVIDIA/nccl/releases` (verified 2026-05-25).

---

## 6. Lambda Stack — driver pin and force-upgrade path

### 6.1 What Lambda Stack actually ships
Lambda **does not publish a per-image driver-version table**. From the public Lambda Stack package page and the Lambda 24.04 troubleshooting docs:
- Lambda Stack 24.04 includes: PyTorch 2.7.0+ds (ARM), CUDA 12.8.93~12.8.1, NCCL 2.26.2, nvidia-container-toolkit 1.18.1.
- The driver "tracks" R570 on the most recent observable images (community reports of `nvidia-smi` from June 2025 GH200 instances show 570.124.06 / CUDA 12.8; earlier 2024 instances showed 550.127.05 / CUDA 12.4).
- Lambda's troubleshooting docs document **one** known apt issue on GH200: DOCA/OFED breaks `apt full-upgrade` after Lambda's image baseline. Fix:
  ```bash
  sudo sed -i 's/2.9.3/3.2.0/' /etc/apt/sources.list.d/doca.list
  # then reinstall MLNX-OFED kernel utilities and re-run apt full-upgrade
  ```

### 6.2 Lambda's *official* answer to "I want a newer driver/CUDA than Lambda Stack ships"
From Lambda staff (@Nicholas_C) on the DeepTalk forum thread "Possible to upgrade CUDA in the Lambda Stack?":

> "You can move the /etc/apt/preferences.d/lambda-repository file or change the priority level within the file to lower it as needed."

Plus the standard "create a Python venv with `--system-site-packages`" advice for the userland side. There is **no** official documentation page for this; it's forum guidance.

### 6.3 Concrete recipe to force-upgrade to driver 580.159.04 + CUDA 13.0.2 on Lambda GH200
This is synthesized from the Lambda forum guidance + NVIDIA's CUDA install guide. **Untested on a live Lambda instance during this research** — flagged as such.

```bash
# 0. Snapshot the instance (Lambda console) before any of this.

# 1. Lower the lambda-repository apt priority so NVIDIA's repo can take precedence.
sudo mv /etc/apt/preferences.d/lambda-repository /etc/apt/preferences.d/lambda-repository.disabled
# (Or edit the file and reduce Pin-Priority from 1001 to a low value like 100.)

# 2. Hold the Lambda Stack metapackage so apt doesn't drag the old driver back.
sudo apt-mark hold lambda-stack-cuda

# 3. Fix the DOCA repo (Lambda's known bug) before any full-upgrade.
sudo sed -i 's/2.9.3/3.2.0/' /etc/apt/sources.list.d/doca.list

# 4. Add NVIDIA's sbsa repo (Round 1 §2.3).
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 5. Remove the old open-driver and install 580.
sudo apt purge -y 'nvidia-driver-*' 'nvidia-open-*' 'cuda-drivers*'
sudo apt install -y nvidia-open-580=580.159.04-* cuda-drivers-580=580.159.04-1ubuntu1

# 6. CUDA toolkit.
sudo apt install -y cuda-toolkit-13-0

# 7. Reboot; verify.
sudo reboot
# After boot:
nvidia-smi | grep "Driver Version"     # 580.159.04
nvcc --version                          # 13.0.x
nvidia-smi -q | grep Addressing         # ATS
```

**Known gotcha**: Lambda's compiled PyTorch 2.7.0 ARM wheel is built against CUDA 12.8. After upgrading to CUDA 13.0 you may need to either (a) re-link against the 13.0 runtime libs (CUDA 13 keeps backward-compat for 12.8-compiled wheels via Forward Compatibility on the same driver branch) or (b) install a CUDA-13 PyTorch nightly. Test `import torch; torch.cuda.is_available()` before committing.

**Source**: [DeepTalk thread "Possible to upgrade CUDA in the Lambda Stack"](https://deeptalk.lambda.ai/t/possible-to-upgrade-cuda-in-the-lambda-stack/4785); [Lambda troubleshooting docs](https://docs.lambda.ai/public-cloud/on-demand/troubleshooting/).

---

## 7. linux-nvidia-64k-hwe-24.04-edge: current state and GH200 boot evidence

### 7.1 Current package version
Per Launchpad, as of 2026-05-25:
- Package: `linux-nvidia-64k-hwe-24.04-edge`
- Current arm64 build: **6.17.0-1018.18**, published to the security pocket **2026-05-19**.
- Package definition: "always depends on the latest complete nvidia-64k Linux kernel and headers."

### 7.2 GH200 boot reports for this build
**Direct evidence**: not found. No reddit / forum / blog post specifically calls out a clean boot of `6.17.0-1018.18` on GH200 in the May 2026 window.

**Indirect evidence (strong)**:
- The package is in Ubuntu's security pocket (= passed canonical's QA), not just `-proposed`.
- The package metadata explicitly identifies GH200/Grace as its target audience (this is the canonical-supplied 64K-page nvidia-tuned aarch64 kernel; it has no other reason to exist).
- No open issues in NVIDIA/open-gpu-kernel-modules tagged against kernel 6.17 + GH200.
- Stephen Diehl's GH200 setup tutorial, the HackMD 24.04-upgrade walkthrough, and Ed Sealing's setup guide — all of which were last touched in 2025–2026 — all use the same `linux-nvidia-64k-hwe-24.04-edge` metapackage and report it working.

**Indirect evidence (weak / contra)**:
- Lambda Stack still ships its own `linux-nvidia-64k-hwe-XX.04` (without `-edge`), suggesting Lambda hasn't validated `-edge` for its production images yet. The `-edge` variant tracks the HWE bleeding-edge kernel; the non-`-edge` variant is conservative.

### 7.3 Recommendation
- For a fresh Lambda GH200 instance, use `-edge` if you want the May-2026 kernel; use the non-`-edge` metapackage if you want to stay closer to what Lambda QAs against.
- Either way, **purge the stock 4K kernel only after the 64K kernel boots once** — failing to do so is the most common way people brick a GH200 instance.

**Source**: [Launchpad: linux-nvidia-64k-hwe-24.04-edge / noble / arm64](https://launchpad.net/ubuntu/noble/arm64/linux-nvidia-64k-hwe-24.04-edge); [Stephen Diehl GH200 tutorial](https://www.stephendiehl.com/posts/setup_gh200_tutorial/).

---

## 8. Updated bottom-line install matrix (May 2026)

| Layer | Round 1 said | Round 3 says |
|---|---|---|
| 64K kernel | `linux-nvidia-64k-hwe-24.04-edge` | Same. Build is now 6.17.0-1018.18 (2026-05-19). No known regressions. |
| Driver (R580 line head) | 580.159.03 | **580.159.04** (newest R580 patch in NVIDIA sbsa repo) |
| Driver (R595 line) | 595.58.03+ for CUDA 13.2 U1 | **595.71.05** is now head of R595 in the sbsa repo |
| NCCL | 2.30.4 | Same; date is **2026-04-22**, not 2025-04-22 |
| NCCL multi-node GH200 (≥8 nodes) | "open bug, beware" | **Closed bug — set `NCCL_NCHANNELS_PER_NET_PEER=4`** |
| GH200 NVL2 driver-565+ device-not-found | Not flagged | **Real bug (#774). Fix is VBIOS 96.00.D8.00.03, not a driver downgrade.** |
| Lambda upgrade procedure | Generic "edit apt pins" | Explicit recipe in §6.3 above; flagged as untested live |
| FP8 | "broken; default BF16" | Unchanged. No Round-3 evidence shifts this. |

---

## 9. Confidence & remaining gaps

**High confidence**:
- 580.159.04 is in the sbsa repo (direct fetch).
- NCCL #1272 closed 2024-04-30 with env-var workaround (full thread read).
- open-gpu-kernel-modules #774 still open, VBIOS-fix-recommended (full thread read).
- NCCL 2.30.4-1 published 2026-04-22 (GitHub API).
- linux-nvidia-64k-hwe-24.04-edge at 6.17.0-1018.18, 2026-05-19 (Launchpad).

**Medium confidence**:
- Lambda's current driver pin. Lambda doesn't publish a version table; community `nvidia-smi` outputs span 550.x → 570.x. Highest-confidence answer requires running `nvidia-smi` on a *fresh* Lambda GH200 instance today.
- The §6.3 force-upgrade recipe. Synthesised from forum guidance + NVIDIA install guide; not executed on a live Lambda instance during Round 3.
- Whether `NCCL_NCHANNELS_PER_NET_PEER=4` is still the right workaround in NCCL 2.30.x. The maintainer's 2024 explanation hasn't been contradicted in any later release notes, but no public 2026-era 8-node GH200 + NCCL 2.30 benchmark was found.

**Open gaps for Round 4 or hands-on validation**:
- A controlled benchmark of `NCCL_NCHANNELS_PER_NET_PEER ∈ {default, 4, 8, 16}` on multi-node GH200 with NCCL 2.30.4.
- Whether Lambda's image as of 2026-05-25 ships `nvidia-open-580` or still `nvidia-open-570` — requires a live instance probe.
- Whether the VBIOS issue (#774) is silently affecting *any* Lambda hardware. Probably not (Lambda would have caught it), but a single `nvidia-smi` dmesg check on first boot would confirm.

---

## 10. Sources

Primary (fetched 2026-05-25):
- [NVIDIA CUDA sbsa apt repo index — Ubuntu 24.04](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/)
- [NCCL issue #1272 — Performance drop on ≥8 nodes (CLOSED 2024-04-30)](https://github.com/NVIDIA/nccl/issues/1272)
- [open-gpu-kernel-modules issue #774 — GH200 NVL2 device-not-found (OPEN, last activity 2026-03-20)](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/774)
- [open-gpu-kernel-modules issue #911 — CCU support for GH200 (OPEN)](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/911)
- [NCCL v2.30.4-1 release (2026-04-22)](https://github.com/NVIDIA/nccl/releases/tag/v2.30.4-1)
- [NCCL releases timeline via gh API](https://github.com/NVIDIA/nccl/releases)
- [Launchpad: linux-nvidia-64k-hwe-24.04-edge noble arm64 — 6.17.0-1018.18, 2026-05-19](https://launchpad.net/ubuntu/noble/arm64/linux-nvidia-64k-hwe-24.04-edge)
- [Driver R580 v8.0 release notes (April 2026)](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Data_Center_GPU_Driver_Release_Notes_580_v8.0.pdf)

Lambda-specific:
- [Lambda troubleshooting docs](https://docs.lambda.ai/public-cloud/on-demand/troubleshooting/)
- [DeepTalk: Possible to upgrade CUDA in the Lambda Stack? — staff response on lambda-repository apt priority](https://deeptalk.lambda.ai/t/possible-to-upgrade-cuda-in-the-lambda-stack/4785)
- [DeepTalk: Lambda Stack CUDA and Nvidia drivers — historical pin behaviour](https://deeptalk.lambda.ai/t/lambda-stack-cuda-and-nvidia-drivers/3218)
- [Lambda Stack software list](https://lambda.ai/lambda-stack-deep-learning-software)

Round 1 anchor:
- [round1/01_driver_stack.md](../round1/01_driver_stack.md)
