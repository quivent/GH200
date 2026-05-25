# Round 3 / Agent 08 — OS Substrate Verification & Deepening

**Date:** 2026-05-25
**Scope:** Re-verify Round 1 / Agent 08 claims about Ubuntu 26.04 vs 24.04 for GH200 inference (bf16; FP8 broken). Drill on six specific claims with primary-source lookups.
**Prior doc:** `/home/ubuntu/research/gh200_inference/round1/08_ubuntu_26_vs_24.md`

---

## TL;DR — what changed since Round 1

Round 1 was directionally correct about "stay on 24.04 for production" but is **stale on three material facts** that change the calculus for greenfield deployments:

1. **`ubuntu2604/sbsa/` IS populated** with NVIDIA driver **595.71.05** + `cuda-compat-13-2` + `nvidia-driver-open` packages. Round 1 said "contents not yet confirmed populated" — that gap is closed. NVIDIA ships a real 26.04 SBSA channel today.
2. **R595 is the current Production Branch** (released 2026-03-24, latest .71.05 on 2026-04-28). Round 1 anchored on R580.126.16 from Feb 2026 — that's now the **LTSB**, not the front edge. The R595 release notes explicitly list Ubuntu 26.04 LTS as supported on Arm64 Server, and GH200 is in the Grace Hopper family table.
3. **R580 LTSB EOL active support is 2026-08-04 (≈10 weeks out), security support to 2028-08-04.** Round 1 didn't surface the lifecycle clock. If you pin R580 today, plan a R595→R610-class hop within ~12 months.

What stays true from Round 1:
- CUDA 13.2 *Install Guide* table (Apr 2026) still does **not** list Ubuntu 26.04 sbsa — only 22.04 / 24.04. Driver supports it; install-guide table is lagging documentation.
- Lambda Stack still 20.04 / 22.04 / 24.04 only. **No 26.04 Lambda Stack and no 26.04 Lambda Cloud image as of 2026-05-25.**
- Open kernel module mandatory on GH200; 565-open / 570-open enumeration bug (issue #774) still the canonical regression-bait example.
- For production SLAs: stay on 24.04 + R580 LTSB.

---

## Drill-down findings

### 1. `https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/sbsa/` — populated?

**Yes — populated and active.** Direct fetch of `Packages` confirms:

- `nvidia-driver`, `nvidia-driver-open`, `nvidia-open` at **595.58.03-1ubuntu1** and **595.71.05-1ubuntu1**
- `nvidia-driver-pinning-595.58.03`, `nvidia-driver-pinning-595.71.05`
- `nvidia-driver-assistant 0.43.58.03-1` and `0.51.71.05-1`
- `cuda-compat-13-2 595.58.03-1ubuntu1` and `595.71.05-1ubuntu1`
- `cuda-drivers 595.58.03-1ubuntu1` and `595.71.05-1ubuntu1`
- `cuda-keyring 1.1-1`
- DKMS, container-toolkit, MFT, NVLSM, firmware packages all present
- `InRelease`, `Release`, `Release.gpg`, `Packages`, `Packages.gz`, `60DF8A40.pub` — full signed APT repo metadata

**Implication:** Round 1's "directory exists in parent index; contents not yet confirmed populated" is **no longer accurate**. The 26.04 SBSA channel is shipping R595. This is the channel you'd use if you wanted CUDA 13.x + matched compat on bare 26.04.

**What's still missing on that channel:** No `cuda-toolkit-13-2` metapackage was visible in the Packages file scan — only `cuda-compat-13-2` and `cuda-drivers`. The full toolkit may live under `cuda-13-2-*` individual component packages that I didn't enumerate exhaustively; or it may genuinely be pending. **TODO before relying on this channel for vLLM source builds:** `curl -s .../ubuntu2604/sbsa/Packages.gz | gunzip | grep '^Package: cuda-toolkit'` from a GH200 box.

---

### 2. Lambda Stack apt repo for noble / 26.04?

**Not yet.** Three converging signals as of 2026-05-25:

- `github.com/lambdal/lambda-stack-dockerfiles` README still lists only 20.04 / 22.04 / 24.04.
- Lambda Cloud public docs (`docs.lambda.ai/public-cloud/on-demand/`) image catalog: Ubuntu 22.04, 24.04, Lambda Stack 22.04 (legacy + current), Lambda Stack 24.04. **No 26.04 entry.** Default image still 22.04.
- No DeepTalk / Lambda blog post announcing 26.04 support in March–May 2026.

**Implication:** If your fleet is Lambda-resident, 26.04 is not a viable substrate today regardless of what NVIDIA's repos ship. The path forward is Lambda Stack 24.04 → wait for Lambda Stack 26.04 (historically Lambda lags Ubuntu LTS GA by 4–9 months; expect Sep–Dec 2026 for a 26.04 Lambda Stack image).

---

### 3. `nvidia-driver-580-open` in Ubuntu 26.04 archive?

**Yes — and the archive has *both* 580 and 595 open variants on arm64.** Confirmed via `packages.ubuntu.com` for suite "resolute":

| Package | Version | Arch | Notes |
|---|---|---|---|
| `nvidia-driver-580-open` | **580.159.03-0ubuntu0.26.04.1** | amd64, arm64 | Marked "security" — restricted, open kernel metapackage |
| `nvidia-driver-595-open` | **595.71.05-0ubuntu0.26.04.1** | amd64, arm64 | Marked "security" — restricted, open kernel metapackage |

Two things this changes vs Round 1:

- Round 1's working version of 580 was **580.126.16** (Feb 2026 LTSB release). Canonical's archive build is now **580.159.03** — a later 580 LTSB respin with security-tag, dated post-Round-1.
- The forum thread "Unable to locate package nvidia-open on Ubuntu 26.04" (NVIDIA dev forums #368145) was **resolved in April–May 2026**: the fix is "add the 2604 cuda repo from developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/". That repo is now populated (see §1). So the "package not found" tax is fully retired.

**However** — the Ubuntu-archive `nvidia-driver-580-open` is Canonical's repackage of NVIDIA's open kernel module + userspace, **not the NVIDIA-prevalidated DC driver**. For inference SLA work, you should still prefer the NVIDIA `ubuntu2604/sbsa` repo over Canonical's archive, because that's the chain NVIDIA's Grace install guide tests against.

---

### 4. RHEL 9.6 / Rocky 9 / SLES 15 SP6 — CUDA 13.2 SBSA freshness

| Distro | NVIDIA SBSA repo populated? | Latest driver | CUDA 13.2 packages? | Notes |
|---|---|---|---|---|
| **RHEL 9 / Rocky 9** | Yes — `rhel9/sbsa/` actively updated | **595.71.05** (`cuda-drivers-595.71.05-1.el9.aarch64.rpm`, 2026-04-24); 590.44.01 & 590.48.01 also present | Yes — `cuda-13-2-13.2.0-1.aarch64.rpm`, `cuda-13-2-13.2.1-1.aarch64.rpm`, full `cuda-cccl-13-2-13.2.27`/`75` component tree | RHEL 9.6 released 2025-05-20 with kernel **5.14.0-570.12.1**. Round 1's claim "5.14.0-611" is **wrong** — that's RHEL 9.7. NVIDIA's CUDA 13.2 install guide pins "RHEL 9" generically at 5.14.0-611, which means 9.7 is the install-guide reference; 9.6 (570 series) is older and may be on the edge of supported. |
| **Rocky 10** | n/a (use RHEL 10 repo); CUDA 13.2 install guide lists Rocky 10 explicitly for arm64-sbsa | (same as RHEL 10) | Yes | Verified in install guide table. |
| **SLES 15 SP6** | Yes — `sles15/sbsa/` populated | **595.71.05** (`cuda-cloud-opengpu-595.71.05-1.aarch64.rpm`, 2026-04-24) | Yes — `cuda-13-2-13.2.1`, `cuda-compat-13-2-595.71.05` present | NVIDIA SLES Grace install guide still pins SLES 15 SP6 + QU1. |
| **SLES 15 SP7** | (released **2026-02-27**) | SUSE-shipped 535.104.05 does **not** enable GH200; SUSE explicitly notes "check for 545.29.02 or later or contact NVIDIA" | n/a | **SP7 is not yet a viable GH200 substrate via SUSE-native packages** — you'd be back-installing NVIDIA's SBSA RPMs. Stick with SP6 + QU1 if SLES is mandatory. |

**Fresh-data correction to Round 1:**
- RHEL 9.6 kernel is `5.14.0-570.12.1`, not `5.14.0-611`. (Round 1 conflated 9.6 with 9.7.)
- Both `rhel9/sbsa` and `sles15/sbsa` ship the **same** 595.71.05 driver and CUDA 13.2.1 component packages as the `ubuntu2604/sbsa` channel — NVIDIA's repo cadence is essentially synchronized across distros for sbsa.

---

### 5. NixOS GH200 status, May 2026

**Stable but community-grade, no SLA.** Verified directly:

- NixOS Wiki NVIDIA page last updated **2026-05-14** (11 days before Round 3's date). Explicit text: *"Data center GPUs starting from Grace Hopper or Blackwell must use open-source modules — proprietary modules are no longer supported."*
- `nixos/modules/hardware/video/nvidia.nix` exposes a `hardware.nvidia.branch` option with **`legacy_580` documented as the LTSB**, version checks for 430.09, 435.21, 470, 495, 510, 555, 560, **595+**. No Grace/Hopper architecture-specific code paths.
- NVIDIA GPU Operator 25.3 platform support page lists `driver.useOpenKernelModules=true` as required for GH200 (consistent with NixOS guidance).
- **No "Hannes Karppila / nixpkgs-cuda team" status thread surfaced for May 2026.** The CUDA / nixpkgs maintainer activity appears routine; no public regression or stop-ship for GH200 announced.

**Implication:** NixOS on GH200 is a viable **personal dev box** today with `hardware.nvidia.open = true` and `hardware.nvidia.branch = "legacy_580"` (or "production" once that resolves to 595). It is **not** an inference-SLA substrate — no NVIDIA prevalidation, no enterprise support contract, no DOCA-OFED package in nixpkgs by default. Round 1's bottom line ("personal GH200 dev boxes, not production") is unchanged.

---

### 6. NVIDIA driver R580 LTSB successor — has R610 / R615 / R620 been announced?

**Not yet.** As of 2026-05-25:

- **endoflife.date/nvidia** (most current public lifecycle tracker):
  - R580 = current **LTSB** (Windows + Linux). Active support ends **2026-08-04**. Security support ends **2028-08-04**.
  - R590 = Production Branch, released **2025-12-22**.
  - R595 = Production Branch, released **2026-03-24**, latest 595.71.05 on **2026-04-28**.
  - No R600 / R605 / R610 / R615 / R620 / R625 listed.
- NVIDIA's own `datacenter/tesla/drivers/` index lists only R580, R590, R595 (highest visible). No newer branch.
- NVIDIA's GTC 2026 (March) datacenter roadmap coverage (DCD, Tom's Hardware) discusses Rosa CPU, Feynman GPU, NVLink optical, Groq LPUs with NVFP4 — **no driver-branch roadmap announcement**.
- R580 LTSB → next LTSB cadence: historically NVIDIA ships an LTSB every ~12 months. R580 LTSB was Aug 2025; the next LTSB is plausibly an **R610-series in mid-to-late 2026**, but I found **no public announcement**.

**Implication:**
- R595 is the right Production Branch to evaluate **now** for greenfield 26.04 + GH200 work — it explicitly lists Ubuntu 26.04 + Grace Hopper.
- R580 LTSB is the safe pin for SLA fleets, but the **active-support clock is ticking** (10 weeks). Plan the R595 (or R610-LTSB-when-announced) migration window for Q3 2026.
- R595 important Hopper caveat: **fails to initialize on Hopper GPUs with subrevision=3 and VBIOS older than 96.00.68.00.xx**. Verify `nvidia-smi -q | grep -i vbios` before upgrading.
- R595 also introduces **CDMM (Coherent Driver-Based Memory Management)** for GB200 platforms — driver-managed GPU memory instead of OS-managed. Relevant for Blackwell GB200, not GH200, but useful context for fleet planning.

---

## Corrections to Round 1

| Round 1 claim | Verified status | Correction |
|---|---|---|
| `ubuntu2604/sbsa/` populated state "not confirmed" | **Wrong now** | Populated with R595.71.05 + cuda-compat-13-2 + DKMS + container-toolkit |
| "580.126.16 is the known-good point" / "Feb 2026 release" | **Stale** | R580 is now LTSB at .159.03 (Canonical archive) / .159.04 (NVIDIA repo); R595 is current Production Branch at .71.05 (Apr 2026) |
| "no 26.04 Lambda Stack" | **Still true** | Confirmed: 20.04 / 22.04 / 24.04 only |
| "26.04 not listed in CUDA 13.2 install guide" | **Still true** | Install-guide table lags — 26.04 absent though driver supports it |
| RHEL 9.6 kernel = "5.14.0-611.16.1.el9_7" | **Wrong** | RHEL 9.6 = 5.14.0-570.12.1 (May 2025); 5.14.0-611 = RHEL 9.7 |
| "565-open / 570-open GH200 enumeration bug" | **Still true** | Issue #774 still the canonical example; 560-open and 580+ are good |
| `nvidia-open` "package not found" on 26.04 | **Resolved** | Fix: add NVIDIA ubuntu2604 repo; archive also has nvidia-driver-{580,595}-open |
| "Python 3.14 default breaks vLLM/torch wheels" | **Not re-verified** | Still likely true (no PyPI wheel survey in this round); use a py3.12 venv |
| "Open kernel module mandatory on GH200" | **Confirmed** | NixOS wiki, GPU Operator docs, all consistent |

---

## Updated Recommendation (May 2026)

**Production SLA, GH200 inference, bf16:**
- Ubuntu 24.04 LTS + Lambda Stack 24.04 + NVIDIA driver R580 LTSB (latest .159.03+) from `ubuntu2404/sbsa` repo + CUDA 13.2 (whichever 13.x R580 supports via minor-version compatibility — verify against `supported-drivers-and-cuda-toolkit-versions.html`, which says R580 caps at CUDA 12.x; for CUDA 13.x you need R595 Production).
- **Important nuance not in Round 1:** R580 LTSB officially caps at **CUDA 12.x** per NVIDIA's support matrix. If you need CUDA 13.2 you're on R595 Production, **not** R580 LTSB. This breaks Round 1's "R580 + CUDA 13.2" recommendation as a combined stack. Pick one:
  - **Pure LTSB path:** R580 LTSB + CUDA 12.x (12.6 / 12.8). Maximum stability, slightly older toolkit.
  - **Current Production path:** R595 + CUDA 13.2. Mainline. Accepted risk: ~12 month branch lifetime before next Production Branch.

**Greenfield 26.04 + GH200, May 2026:**
- Ubuntu Server 26.04 minimal → install via NVIDIA `ubuntu2604/sbsa` repo (now confirmed live) → `nvidia-driver-open=595.71.05` + `cuda-compat-13-2` → py3.12 venv (avoid 3.14 default) → PyTorch nightly aarch64 + cu130 wheel → vLLM `--dtype bfloat16`.
- Verify VBIOS ≥ 96.00.68.00.xx before installing R595 on Hopper. Verify `nvidia-smi` enumerates GH200 with ECC on.
- **Do not deploy this to Lambda Cloud** — no 26.04 image, manual `do-release-upgrade` from 24.04 known-problematic on Lambda GH200.

---

## Confidence

**High confidence (primary-source fetches dated within last 8 weeks):**
- `ubuntu2604/sbsa` repo populated with R595.71.05 (direct Packages fetch)
- `rhel9/sbsa`, `sles15/sbsa` populated with same R595.71.05 + CUDA 13.2.1 (direct directory fetches, 2026-04-24 timestamps)
- Ubuntu 26.04 archive ships `nvidia-driver-{580,595}-open` arm64 (packages.ubuntu.com resolute suite query)
- R580 = LTSB, R590 + R595 = Production Branches, no R610+ announced (endoflife.date + docs.nvidia.com cross-check)
- Lambda Stack still 20.04 / 22.04 / 24.04 only (lambdal/lambda-stack-dockerfiles README + Lambda docs)
- NixOS wiki updated 2026-05-14, Grace Hopper open-module-mandatory text confirmed
- RHEL 9.6 = kernel 5.14.0-570.12.1, released 2025-05-20
- R580 LTSB active-support EOL 2026-08-04, security EOL 2028-08-04

**Medium confidence:**
- Whether `ubuntu2604/sbsa` has full `cuda-toolkit-13-2` metapackage (saw only `cuda-compat-13-2` and `cuda-drivers` in Packages scan; may need direct ls of repo)
- R595's exact supported-OS table for arm64-SBSA (release-notes page renders the table only as "Arm64 Server (Generic)" + "NVIDIA Grace Arm64 Server" columns; "Ubuntu 26.04 LTS" confirmed present at row level, kernel-specific minimums not enumerated)
- R580 LTSB CUDA cap — NVIDIA's matrix page says CUDA 12.x for R580 but minor-version-compat with CUDA 13 may be possible in practice; **not tested**

**Gaps still open:**
- No 26.04-vs-24.04 inference benchmark on GH200 published anywhere (same as Round 1).
- TransformerEngine / FlashAttention 3.x py3.14 aarch64 wheel availability — not surveyed this round.
- Lambda's roadmap for 26.04 — no public commitment found.
- Whether R610 / R615 / R620 will be next LTSB — no public roadmap.

---

## Sources

- [NVIDIA CUDA repos — ubuntu2604/sbsa/](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/sbsa/) — verified populated, R595.71.05 + cuda-compat-13-2
- [NVIDIA CUDA repos — ubuntu2604/sbsa/Packages](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/sbsa/Packages) — direct Packages file inspection
- [NVIDIA CUDA repos — rhel9/sbsa/](https://developer.download.nvidia.com/compute/cuda/repos/rhel9/sbsa/) — R595.71.05 + CUDA 13.2.x RPMs
- [NVIDIA CUDA repos — sles15/sbsa/](https://developer.download.nvidia.com/compute/cuda/repos/sles15/sbsa/) — R595.71.05 + CUDA 13.2.1 RPMs
- [NVIDIA CUDA repos — index](https://developer.download.nvidia.com/compute/cuda/repos/) — confirms ubuntu2604, rhel9, rhel10, sles15, sles16 present; no rocky*, no rhel96
- [NVIDIA Data Center Driver 595.71.05 release notes](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-595-71-05/index.html) — released 2026-04-28; Ubuntu 26.04 LTS listed on Arm64 Server (Generic); GH200 in Grace Hopper family
- [NVIDIA Datacenter Drivers index](https://docs.nvidia.com/datacenter/tesla/index.html) — R580 / R590 / R595 visible; no R610+
- [NVIDIA Supported Drivers and CUDA Toolkit Versions](https://docs.nvidia.com/datacenter/tesla/drivers/supported-drivers-and-cuda-toolkit-versions.html) — R580 LTSB caps at CUDA 12.x; R595 supports CUDA 13.x
- [NVIDIA Driver Lifecycle](https://docs.nvidia.com/datacenter/tesla/drivers/driver-lifecycle.html) — LTSB = 3-year support definition
- [endoflife.date — NVIDIA Driver](https://endoflife.date/nvidia) — R580 LTSB EOL active 2026-08-04 / security 2028-08-04; R595 Production released 2026-03-24
- [CUDA Installation Guide for Linux 13.2](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) — arm64-sbsa supported distros: 22.04, 24.04, RHEL 9 (5.14.0-611 = 9.7), RHEL 10, Rocky 10, SLES 15/16; 26.04 NOT in table
- [Ubuntu packages — nvidia-driver-580-open resolute](https://packages.ubuntu.com/search?keywords=nvidia-driver-580-open&searchon=names&suite=resolute&section=all) — 580.159.03-0ubuntu0.26.04.1 amd64+arm64
- [Ubuntu packages — nvidia-driver-595-open resolute](https://packages.ubuntu.com/search?keywords=nvidia-driver-595-open&searchon=names&suite=resolute&section=all) — 595.71.05-0ubuntu0.26.04.1 amd64+arm64
- [UbuntuHandbook — Ubuntu 26.04 NVIDIA 595 driver](https://ubuntuhandbook.org/index.php/2026/04/nvidia-595-driver-ubuntu-26-04/) — confirms archive ships 595.58.03 in restricted; published 2026-04-04, updated 2026-04-29
- [Lambda Cloud on-demand docs](https://docs.lambda.ai/public-cloud/on-demand/) — image catalog, no 26.04
- [lambda-stack-dockerfiles README](https://github.com/lambdal/lambda-stack-dockerfiles) — supports 20.04 / 22.04 / 24.04 only
- [NixOS Wiki — NVIDIA](https://wiki.nixos.org/wiki/NVIDIA) — Grace Hopper open-module mandatory; updated 2026-05-14
- [nixpkgs/nixos/modules/hardware/video/nvidia.nix](https://github.com/NixOS/nixpkgs/blob/master/nixos/modules/hardware/video/nvidia.nix) — `legacy_580` LTSB branch + 595+ version checks
- [NVIDIA dev forum issue — "Unable to locate package nvidia-open on Ubuntu 26.04"](https://forums.developer.nvidia.com/t/error-unable-to-locate-package-nvidia-open-on-ubuntu-26-04/368145) — resolved Apr–May 2026 via adding 2604 cuda repo
- [GitHub NVIDIA/open-gpu-kernel-modules issue #774 — GH200 NVL2 GPU not found on 565+](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/774) — 565-open / 570-open enumeration bug, 560-open and 580+ work
- [RHEL 9.6 Release Notes](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9/html-single/9.6_release_notes/index) — kernel 5.14.0-570.12.1, released 2025-05-20
- [SLES 15 SP7 release notes](https://www.suse.com/releasenotes/x86_64/SUSE-SLES/15-SP7/index.html) — released 2026-02-27; ships 535.104.05 which does NOT enable GH200
- [Phoronix — NVIDIA 580 last for Maxwell/Pascal/Volta](https://www.phoronix.com/news/NVIDIA-580-Linux-Driver-Last-HW) — context on R580 LTSB scope
