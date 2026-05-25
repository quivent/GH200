# GH200 on Lambda Cloud — OS / Kernel / Boot Deploy Guide

**Round 2 / Agent C — Cross-pollination of Round-1 agents 01, 08, 09, 16**
**Date:** 2026-05-25
**Target hardware:** 1× NVIDIA GH200 Grace-Hopper Superchip on Lambda Cloud (`gpu_1x_gh200`)
**Goal:** Take the stock Lambda image (Ubuntu 24.04 / Lambda Stack 24.04, default at boot) to the recommended tuned production state for bf16 LLM inference.
**Stack constraint:** FP8 broken on this stack — bf16 throughout.

---

## 0. TL;DR — What this guide does

Stock Lambda Stack 24.04 ships:
- 4K-page generic kernel (need 64K-page kernel)
- NVIDIA driver R570/R580 from Lambda Stack (need 580.159.03-open)
- CUDA 12.8.93 (recommended: stay on 12.8 OR move to 13.0.2 — see §3.3)
- NCCL 2.26.2 (need ≥ 2.28.3 for PxN-over-C2C; 2.30.4 is current)
- No persistence, no GRUB tuning, no NUMA pinning, no hugepages, no IRQ pinning

This guide produces one GRUB drop-in (§1), one `bootstrap_gh200.sh` script (§2-3), a `launch_inference.sh` wrapper (§4), a verification checklist (§5), a rollback runbook (§6), and a 26.04 migration assessment (§7).

**One landmine to flag upfront:** Running `sudo apt full-upgrade` on the stock Lambda Stack 24.04 / GPU Base 24.04 image **will fail** on a DKMS build error for `mlnx-ofed-kernel-utils` because the bundled DOCA-OFED repo pins 2.9.3 and the new kernel header set is from the 3.2.0 series. §2.1 fixes this **first**, before anything else. If you let `apt full-upgrade` run on the unpatched image, it can leave dpkg in a half-configured state where the kernel can't rebuild and the 64K kernel install (§2.2) silently fails. Patch DOCA-OFED first.

---

## 1. GRUB drop-in — boot parameters

Write to `/etc/default/grub.d/99-gh200.cfg`. Do **not** edit `/etc/default/grub` directly — Canonical's `grub.d` drop-in pattern survives `do-release-upgrade` better.

```bash
# /etc/default/grub.d/99-gh200.cfg
# NVIDIA Grace Performance Tuning Guide — boot parameters for GH200
# Sources: docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html
#          docs.nvidia.com/grace-ubuntu-install-guide.pdf

GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX init_on_alloc=0 iommu.passthrough=1 numa_balancing=disable transparent_hugepage=madvise"
```

Per-flag rationale (consolidated from Round-1 agents 01 §5.1 and 09 §3):

| Flag | Why | Verify after reboot |
|------|-----|---------------------|
| `init_on_alloc=0` | Kernel-default page-zeroing slows cudaMalloc on Grace's coherent memory. Mandatory per NVIDIA Grace OS Settings doc. | `grep init_on_alloc /proc/cmdline` |
| `iommu.passthrough=1` | Disables SMMU translation for DMA. Required for full GDS / GPUDirect-RDMA bandwidth on bare metal. (Lambda exposes GH200 as bare-metal-equivalent — single-tenant compute.) | `grep iommu.passthrough /proc/cmdline` |
| `numa_balancing=disable` | AutoNUMA can migrate pages off HBM back to LPDDR5 mid-inference. Catastrophic for KV-cache locality. | `cat /proc/sys/kernel/numa_balancing` → `0` |
| `transparent_hugepage=madvise` | Lets vLLM / TRT-LLM opt into 2 MiB mTHP (kernel 6.9+) without giving every allocation a 512 MiB page on the arm64-64K kernel (kernel 6.8). | `cat /sys/kernel/mm/transparent_hugepage/enabled` → `[madvise]` |

**Note on 64K-page kernel selection:** The kernel package itself (`linux-nvidia-64k-hwe-24.04`) is installed via apt — there is no GRUB flag for it; the page size is baked into the kernel image at compile time. GRUB just boots whichever kernel is currently the default. See §2.2 for the install.

**Apply:**
```bash
sudo update-grub
sudo reboot
```

**Optional — hugepages at boot.** If you know your model size in advance and want 2 MiB hugepages reserved before any allocator starts (vLLM with `--enable-prefix-caching` benefits), add to the same file:
```bash
# Reserve 32 GiB of 2M hugepages at boot (16384 × 2 MiB)
GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX default_hugepagesz=2M hugepagesz=2M hugepages=16384"
```
For 1 GiB hugepages (single-shard 70B weights, ~140 GiB): `default_hugepagesz=1G hugepagesz=1G hugepages=64` (64 GiB). 1 GiB hugepages are **boot-time-only**.

**Optional — CPU isolation for tail-latency workloads only:** add `isolcpus=8-71 nohz_full=8-71 rcu_nocbs=8-71` if you're running latency-pinned serving and the 4-7 housekeeping cores are sufficient. Round-1 agent 09 §3 flags this as inference-non-essential and unmeasured for GH200; skip by default.

---

## 2. `bootstrap_gh200.sh` — stock-Lambda → recommended state

Save to `/root/bootstrap_gh200.sh`, `chmod +x`, run as root from a fresh Lambda Stack 24.04 instance. Designed to be idempotent and stop on any error.

```bash
#!/usr/bin/env bash
# bootstrap_gh200.sh — Stock Lambda Stack 24.04 → recommended GH200 inference state.
# Run as root on a freshly-launched 1× GH200 instance. Idempotent. Reboots twice.
# Date: 2026-05-25
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

log()  { echo "[$(date +%H:%M:%S)] $*"; }
need() { command -v "$1" >/dev/null 2>&1 || { log "missing: $1"; exit 1; }; }

# Use a state file so we can re-enter after each reboot.
STATE=/var/lib/gh200-bootstrap.state
mkdir -p "$(dirname "$STATE")"
STAGE=$(cat "$STATE" 2>/dev/null || echo "stage0")

#----------------------------------------------------------------------#
# STAGE 0 — DOCA-OFED 2.9.3 → 3.2.0 fix (MUST BE FIRST — see §0 landmine)
#----------------------------------------------------------------------#
if [ "$STAGE" = "stage0" ]; then
  log "STAGE 0: DOCA-OFED 2.9.3 -> 3.2.0 transition fix"
  # Lambda's stock /etc/apt/sources.list.d/doca.list pins 2.9.3.
  # Bump to 3.2.0, refresh, reinstall kernel utils with --force-confnew.
  if [ -f /etc/apt/sources.list.d/doca.list ]; then
    sudo sed -i 's/2\.9\.3/3.2.0/g' /etc/apt/sources.list.d/doca.list
  fi
  apt-get update
  apt-get install -y -o Dpkg::Options::="--force-confnew" mlnx-ofed-kernel-utils
  # Now the full-upgrade will succeed (it will gripe about linux-headers but
  # exit 0). Run it twice if you see the gripe — it's a known cosmetic issue.
  apt-get -y full-upgrade || apt-get -y full-upgrade
  echo "stage1" > "$STATE"
fi

#----------------------------------------------------------------------#
# STAGE 1 — Install 64K-page kernel, GRUB drop-in, reboot
#----------------------------------------------------------------------#
if [ "$STAGE" = "stage1" ]; then
  log "STAGE 1: 64K-page kernel install + GRUB tuning"
  apt-get update
  # Pull the metapackage that tracks latest 64K kernel.
  # On 24.04 noble the package family is linux-nvidia-64k-hwe-24.04[-edge].
  # -edge gets you the kernel.org-aligned kernel, non-edge gets you the
  # canonically-stable HWE pin. For inference, non-edge is correct.
  apt-get install -y linux-nvidia-64k-hwe-24.04 linux-headers-nvidia-64k-hwe-24.04

  # GRUB drop-in (see §1)
  cat > /etc/default/grub.d/99-gh200.cfg <<'EOF'
GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX init_on_alloc=0 iommu.passthrough=1 numa_balancing=disable transparent_hugepage=madvise"
EOF
  update-grub
  echo "stage2" > "$STATE"
  log "rebooting into 64K kernel; rerun this script after reboot"
  systemctl reboot
  exit 0
fi

#----------------------------------------------------------------------#
# STAGE 2 — Verify 64K kernel, install driver R580 + CUDA + NCCL
#----------------------------------------------------------------------#
if [ "$STAGE" = "stage2" ]; then
  log "STAGE 2: verify 64K, install driver/CUDA/NCCL"

  PAGESZ=$(getconf PAGESIZE)
  if [ "$PAGESZ" != "65536" ]; then
    log "ERROR: PAGESIZE=$PAGESZ, expected 65536. 64K kernel did not boot."
    log "Investigate /boot/grub/grub.cfg menuentry order and try again."
    exit 2
  fi
  log "OK: PAGESIZE=$PAGESZ (64K)"

  # Verify GRUB params landed
  for flag in init_on_alloc=0 iommu.passthrough=1 numa_balancing=disable; do
    grep -q "$flag" /proc/cmdline || { log "ERROR: $flag missing from /proc/cmdline"; exit 3; }
  done
  log "OK: GRUB flags present in /proc/cmdline"

  # NVIDIA apt repo (cuda-keyring is preferred; works for both 12.x and 13.x)
  wget -q -O /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
  dpkg -i /tmp/cuda-keyring.deb
  apt-get update

  # Remove any stale Lambda-Stack-managed driver before installing R580-open.
  # Lambda Stack pins nvidia-driver-* via meta-packages; we *don't* purge the
  # whole stack (PyTorch et al. depend on it), we just upgrade the driver.
  # If a non-open driver is present, it must go (GH200 = open-only).
  apt-get install -y nvidia-open-580 cuda-drivers-580

  # CUDA toolkit: pin 12.8 by default (Lambda Stack's PyTorch 2.7.0-ARM is
  # compiled against cu128 wheels). Switch to 13-0 only if you control the
  # full Python venv (see §3.3 argument).
  CUDA_CHOICE="${CUDA_CHOICE:-12-8}"
  apt-get install -y "cuda-toolkit-${CUDA_CHOICE}"

  # cuDNN matched to CUDA major
  case "$CUDA_CHOICE" in
    12-*) apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 ;;
    13-*) apt-get install -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13 ;;
  esac

  # NCCL — upgrade from Lambda Stack's 2.26.2 to whatever NVIDIA repo has.
  apt-get install -y libnccl2 libnccl-dev

  # Container toolkit (Lambda already ships this but pin to current)
  apt-get install -y nvidia-container-toolkit

  echo "stage3" > "$STATE"
  log "rebooting into new driver; rerun this script after reboot"
  systemctl reboot
  exit 0
fi

#----------------------------------------------------------------------#
# STAGE 3 — Runtime tuning (sysctl, persistence, clocks, IRQs, peermem)
#----------------------------------------------------------------------#
if [ "$STAGE" = "stage3" ]; then
  log "STAGE 3: runtime tuning"

  # sysctl — NVIDIA Grace OS Settings + inference-specific buffers
  cat > /etc/sysctl.d/90-gh200.conf <<'EOF'
# NVIDIA Grace OS Settings (Round-1 agents 01 §5, 09 §4)
kernel.numa_balancing = 0
vm.compaction_proactiveness = 0
vm.swappiness = 10
vm.zone_reclaim_mode = 0
vm.max_map_count = 1048576
# 32 GiB worth of 2 MiB hugepages (skip if reserved at boot via GRUB)
vm.nr_hugepages = 16384
# RDMA / large-msg TCP tuning
net.core.rmem_max = 268435456
net.core.wmem_max = 268435456
net.ipv4.tcp_rmem = 4096 87380 268435456
net.ipv4.tcp_wmem = 4096 87380 268435456
EOF
  sysctl --system

  # ulimits for memlock (RDMA QPs, cudaHostAlloc pinned)
  cat > /etc/security/limits.d/90-gpu.conf <<'EOF'
*    soft   memlock      unlimited
*    hard   memlock      unlimited
*    soft   stack        65536
*    hard   stack        65536
*    soft   nofile       1048576
*    hard   nofile       1048576
EOF

  # Persistence daemon (mandatory; <100 ms cold start vs 1-3 s)
  systemctl enable --now nvidia-persistenced

  # Persistence mode flag (belt-and-suspenders)
  nvidia-smi -pm 1

  # Clock locks for deterministic latency. Hopper SM max = 1980 MHz,
  # HBM3 = 2619 MHz. Power limit: 700 W (air-cooled default). Lambda
  # GH200 is air-cooled per published specs; raise to 900-1000 W only
  # if Lambda confirms your specific chassis is liquid-cooled.
  nvidia-smi --lock-gpu-clocks=1410,1980    || log "warn: lock-gpu-clocks failed"
  nvidia-smi --lock-memory-clocks=2619      || log "warn: lock-memory-clocks failed"
  nvidia-smi -pl 700                        || log "warn: power-limit set failed"

  # CPU governor — performance
  apt-get install -y linux-tools-common linux-tools-generic cpufrequtils
  cpupower frequency-set -g performance || true
  systemctl disable cpufrequtils || true

  # Kill irqbalance (we'll pin manually below)
  systemctl disable --now irqbalance || true

  # IRQ affinity — pin Mellanox/ConnectX channels to Grace cores 4-15.
  # On Lambda's 1× GH200 instance only 64 vCPUs are exposed (vs 72 bare-metal)
  # so 4-15 stays in-range. Skip the Mellanox-supplied script if the HCA is
  # absent (single-node deployments without RDMA NIC).
  if ls /sys/class/infiniband/mlx5_0 >/dev/null 2>&1; then
    if [ -x /usr/sbin/set_irq_affinity_cpulist.sh ]; then
      /usr/sbin/set_irq_affinity_cpulist.sh 4-15 mlx5_0 || true
    else
      for irq in $(grep mlx5_0 /proc/interrupts | awk -F: '{print $1}'); do
        echo 4-15 > /proc/irq/$irq/smp_affinity_list 2>/dev/null || true
      done
    fi
  fi

  # nvidia-peermem for GPUDirect-RDMA
  echo nvidia-peermem > /etc/modules-load.d/nvidia-peermem.conf
  modprobe nvidia-peermem || true

  # Kill libvirtd if present (bridges hurt RDMA on bare-metal-equivalent)
  systemctl disable --now libvirtd.service 2>/dev/null || true

  # Make clocks + persistence survive reboot via a systemd oneshot.
  cat > /etc/systemd/system/gh200-runtime-tuning.service <<'EOF'
[Unit]
Description=GH200 runtime GPU tuning (persistence, clocks, hugepages)
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi --lock-gpu-clocks=1410,1980
ExecStart=/usr/bin/nvidia-smi --lock-memory-clocks=2619
ExecStart=/usr/bin/nvidia-smi -pl 700
ExecStart=/bin/sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
ExecStart=/bin/sh -c 'echo 16384 > /proc/sys/vm/nr_hugepages'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable --now gh200-runtime-tuning.service

  echo "done" > "$STATE"
  log "bootstrap complete. Run /root/verify_gh200.sh to validate."
fi

if [ "$STAGE" = "done" ]; then
  log "bootstrap already complete (state=done). Delete $STATE to re-run."
fi
```

### 2.1 Why this order matters

1. **DOCA-OFED first.** If you let the 64K kernel install pull in a new kernel package while DOCA-OFED 2.9.3 DKMS still owns the active running kernel, dpkg can end up with a half-built kernel module that blocks every subsequent install. The Lambda troubleshooting page is unambiguous: fix DOCA first, then anything else. (Source: `docs.lambda.ai/public-cloud/on-demand/troubleshooting/`.)
2. **64K kernel before R580.** The driver postinstall builds DKMS modules against the *running* kernel. If you install R580 first and reboot into 64K, DKMS rebuilds — wasted work and one more failure surface. Install 64K kernel, reboot, then install the driver.
3. **Runtime tuning last.** Clocks and persistence only mean anything once the right driver is loaded.

### 2.2 What about the existing Lambda-Stack-managed driver?

Lambda Stack pins `nvidia-driver-XXX-server-open` via meta-packages. `apt install nvidia-open-580` will upgrade the existing meta — it does **not** require purging Lambda Stack. PyTorch 2.7.0-ARM continues to work because cuDNN/NCCL ABI is stable across R570→R580. **Do not** `apt purge lambda-stack-cuda` — that will rip out the ARM PyTorch wheel and you'll spend an afternoon rebuilding it.

### 2.3 Verified package availability (May 25, 2026)

Confirmed via direct fetch of `developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/`:
- `cuda-drivers-580_580.159.03-1ubuntu1_arm64.deb` — published 2026-04-24 ✓
- `cuda-drivers_580.159.04-1ubuntu1_arm64.deb` — published 2026-05-06 ✓ (highest 580.x)
- `cuda-drivers_595.71.05-1ubuntu1_arm64.deb` — published 2026-04-24 ✓ (CUDA 13.2-U1 path)
- `cuda-toolkit-13-0_13.0.3-1_arm64.deb` — 2026-03-25 ✓
- `cuda-toolkit-13-2_13.2.1-1_arm64.deb` — 2026-04-07 ✓

Round-1 agent 01 named **580.159.03** as the recommended driver. **580.159.04** is now available (a stability bump). Either works; `bootstrap_gh200.sh` uses the apt meta `nvidia-open-580` which resolves to whichever is current in the repo at run-time. To pin exactly, replace with `nvidia-open-580=580.159.03-0ubuntu1` (verify the exact package version with `apt-cache policy nvidia-open-580`).

---

## 3. CUDA 12.8 vs 13.0.2 — argue which

Round-1 agent 01 §11 named **CUDA 13.0.2 + driver 580.159.03** as "the sweet spot." Round-1 agent 16 confirms Lambda Stack ships **CUDA 12.8.93** in May 2026. Round-1 agent 08 reports Ubuntu 26.04's *archive* CUDA is 13.1.x. Three different versions in play.

### 3.1 The case for staying on CUDA 12.8

- Lambda's pre-compiled **PyTorch 2.7.0-ARM** wheel is built against **cu128**. Move to 13.0.x and you either build PyTorch from source for arm64 (1–2 hours; not painful but a step) or use the upstream `download.pytorch.org/whl/nightly/cu130` aarch64 wheels (which work but are nightlies — looser SLA than Lambda's release build).
- vLLM 0.10+ ships cu128 wheels; cu130 wheels exist but lag.
- TRT-LLM 0.21+ supports CUDA 13 but the recommended pin is still cu128 for stability as of May 2026.
- No bf16 inference feature added in 13.x is load-bearing for the constraint (FP8 disabled).
- The CUDA 13.0 "non-NUMA mode" coherent-memory init (Round-1 agent 01 §4.2) only matters if you've measured kernel pages spilling into HBM. For single-tenant inference, default-NUMA + ATS just works.

### 3.2 The case for moving to CUDA 13.0.2

- ZStd-compressed fatbins (-17% binary size) shorten cold-start when shipping containers.
- Unified arm64 toolchain (SBSA + Jetson) — one less version to think about if you also have Jetson edge devices.
- CCCL 3.0 + tile-based programming for any kernel work you do yourself.
- Driver 580.65.06+ already required for R580 anyway, so the driver step has no incremental cost.
- Forward-looking: vLLM nightlies and TransformerEngine 2.15 increasingly assume CUDA 13.

### 3.3 Recommendation

**Stay on CUDA 12.8 for the production path** (`CUDA_CHOICE=12-8` in the script). The bf16 ceiling on Hopper is the same on 12.8 and 13.0.2; switching costs you the Lambda PyTorch wheel and gains you nothing measurable for this workload.

**Move to 13.0.2 only when** you're about to enable something that needs it — TransformerEngine 2.15 features, CUDA-graph capture changes in PyTorch 2.8, or you're building TRT-LLM engines for cross-platform deployment with Jetson. **Do not** move to 13.2.U1 yet — it needs driver R595+ (which IS in the repo, but R595 is not yet a widely-deployed "production" branch on GH200; R580 is the LTS-equivalent).

To switch: set `CUDA_CHOICE=13-0` before re-running stage 2, and pin a `cu130` PyTorch in your venv. Don't mix — uninstall the 12.8 toolkit first or use the `update-alternatives` `cuda` symlink so vLLM picks the right one.

### 3.4 NCCL version

Lambda Stack ships 2.26.2. Recommend **≥ 2.28.3** (PxN-over-C2C default-on, important for any multi-node GH200 work). Current is **2.30.4** (April 2025). The script's `apt install libnccl2 libnccl-dev` from NVIDIA's `ubuntu2404/sbsa` repo pulls the current packaged NCCL, which for May 2026 should be 2.30.x.

**Single-node-multi-node case for GH200:** the practical "multi-node" scenarios for someone on Lambda Cloud public on-demand are limited (Lambda's 1-Click Clusters are H100/B200 only — no GH200 cluster SKU). If you spin up two on-demand GH200 instances in the same region and want them to talk over the data-plane network, you're not on InfiniBand — you're on plain TCP. Set `NCCL_IB_DISABLE=1` and let NCCL use the socket transport; bandwidth tops out at the instance NIC speed (10/25 Gbps). For real multi-GH200 work you need a 1-Click Cluster or Private Cloud reservation (GH200 NVL32 / GH200 NVL2). Round-1 agent 16 confirms there's no on-demand multi-node GH200 SKU.

NCCL env-vars to put in `/etc/environment` or your launch wrapper (suppresses bootstrap deadlock on the rare multi-node GH200 + Quantum-2 IB setup that Lambda Private Cloud customers may have):
```bash
# Always-safe defaults for any GH200 NCCL workload on Lambda
export NCCL_SOCKET_IFNAME=eno1               # or your control-plane iface; NEVER ib*
export NCCL_IB_HCA=mlx5_0                    # narrow to actual RDMA HCA(s); on single-node, this is a no-op
export NCCL_NVLS_ENABLE=2                    # default; SHARP on if NVL switch present
# Diagnostics on first bring-up; remove from steady-state
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
# Multi-QP routing entropy (only matters on Quantum-2 fabric)
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_QPS=0
# Service Level / TC — leave default unless your subnet manager demands specific values
# export NCCL_IB_SL=0
# export NCCL_IB_TC=41
export NCCL_IB_TIMEOUT=22                    # ~17 s, longer than default 4.3 s
export NCCL_IB_GID_INDEX=3                   # RoCEv2 default; verify with `show_gids`
```

---

## 4. `launch_inference.sh` — NUMA-bound wrapper

```bash
#!/usr/bin/env bash
# launch_inference.sh — NUMA-pinned bf16 vLLM launch on a single GH200
set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.1-70B-Instruct}"
PORT="${PORT:-8000}"

# Confirm tuning is still applied (gh200-runtime-tuning.service is enabled,
# but if you've manually relaxed it, redo here).
ulimit -l unlimited
ulimit -s 65536
ulimit -n 1048576

# NCCL env (no-op for single-GPU but harmless and ready for scale-out)
export NCCL_SOCKET_IFNAME=eno1
export NCCL_NVLS_ENABLE=2
# Comment in once you've validated the path:
# export NCCL_DEBUG=INFO

# NUMA bind: CPU + alloc on Grace node 0; allow HBM (node 1) for explicit prefer.
# Leave cores 0-3 for housekeeping/IRQs (matches IRQ pin range 4-15 in §2).
exec numactl --physcpubind=4-71 --membind=0,1 -- \
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92 \
    --enable-prefix-caching \
    --port "$PORT"
```

(bf16 — FP8 disabled per stack constraint and per Round-1 agent 01 §8 evidence.)

---

## 5. Verification checklist

Run this whole block after the final reboot. Save as `/root/verify_gh200.sh`:

```bash
#!/usr/bin/env bash
# verify_gh200.sh — gate to declare the box "deploy-ready"
set -u
PASS=0; FAIL=0
check() {
  local name="$1"; local expected="$2"; local actual="$3"
  if [[ "$actual" == *"$expected"* ]]; then
    printf "  [PASS] %-40s : %s\n" "$name" "$actual"; PASS=$((PASS+1))
  else
    printf "  [FAIL] %-40s : got '%s', expected '%s'\n" "$name" "$actual" "$expected"; FAIL=$((FAIL+1))
  fi
}

echo "=== Kernel & boot ==="
check "page size"                "65536"                   "$(getconf PAGESIZE)"
check "init_on_alloc in cmdline" "init_on_alloc=0"         "$(cat /proc/cmdline)"
check "iommu.passthrough in cmd" "iommu.passthrough=1"     "$(cat /proc/cmdline)"
check "numa_balancing"           "0"                       "$(cat /proc/sys/kernel/numa_balancing)"
check "THP setting"              "[madvise]"               "$(cat /sys/kernel/mm/transparent_hugepage/enabled)"

echo ""
echo "=== NUMA topology (expect 2 nodes: Grace 0 = CPUs, Hopper 1 = HBM, no CPUs) ==="
numactl -H | head -20

echo ""
echo "=== GPU & driver ==="
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
check "driver R580 family"       "580."                    "$DRIVER"
check "ATS addressing mode"      "ATS"                     "$(nvidia-smi -q | grep 'Addressing Mode' | head -1)"
check "persistence mode"         "Enabled"                 "$(nvidia-smi -q -d POWER | grep 'Persistence Mode' | head -1)"
check "ECC"                      "Enabled"                 "$(nvidia-smi -q -d ECC | grep 'Current' | head -1)"

echo ""
echo "=== Clocks (informational) ==="
nvidia-smi -q -d CLOCK | grep -E 'SM|Memory' | head -8
nvidia-smi -lgc 2>&1 | head -3       # show current lock state

echo ""
echo "=== NVLink-C2C status ==="
nvidia-smi nvlink --status -i 0 | head -20

echo ""
echo "=== CPU governor ==="
check "governor"                 "performance"             "$(cpupower frequency-info | grep 'current policy' | head -1)"

echo ""
echo "=== Modules ==="
check "nvidia-peermem loaded"    "nvidia_peermem"          "$(lsmod | grep nvidia_peermem || echo MISSING)"
check "nvidia-open module"       "nvidia"                  "$(lsmod | grep '^nvidia ' | head -1)"

echo ""
echo "=== Hugepages ==="
grep -E 'HugePages_Total|Hugepagesize' /proc/meminfo

echo ""
echo "=== ulimits (run as inference user, not root) ==="
runuser -u nobody -- bash -c 'echo "memlock=$(ulimit -l) stack=$(ulimit -s) nofile=$(ulimit -n)"'

echo ""
echo "=== Summary: $PASS pass / $FAIL fail ==="
exit $FAIL
```

Expected on a correctly-bootstrapped box:
- `getconf PAGESIZE` → **65536**
- `nvidia-smi -q | grep Addressing` → **Addressing Mode : ATS** (confirms NVLink-C2C ATS path; HMM auto-disabled)
- `numactl -H` → 2 nodes: node 0 has CPUs (0-63 on Lambda's GH200, vs 0-71 bare-metal), node 1 has memory but no CPUs (HBM)
- `nvidia-smi -lgc` shows the locked clock range
- `nvidia-smi nvlink --status -i 0` shows ~25 GB/s per link active (C2C internal — these are the GPU-to-GPU NVLinks in NVL setups; on a single GH200, this command reports the on-chip NVLink links to the Grace CPU)

---

## 6. Rollback path — if a step bricks the boot

The danger steps are §1 (GRUB) and §2.2 (64K kernel). Driver upgrades almost never brick boot; they fail and leave you on the previous kernel/module pair. Plan rollback per step.

### 6.1 GRUB drop-in broke boot (kernel won't start)

Lambda's instances have console access via the Lambda Cloud dashboard ("Console" tab on the instance card). You can interrupt GRUB:

1. From the dashboard console, reboot the instance.
2. At the GRUB menu, press `e` to edit the entry.
3. Remove the offending flag(s) from the `linux` line (e.g. delete `numa_balancing=disable` if you suspect it).
4. Press `Ctrl-X` to boot once with the modified cmdline.
5. Once back in: `sudo rm /etc/default/grub.d/99-gh200.cfg && sudo update-grub`.

If you can't get to the console: open a Lambda support ticket. Lambda has no published "rescue mode" image; recovery is operator-assisted.

**Prevention:** keep the previous-known-good GRUB config in `/etc/default/grub.d/99-gh200.cfg.bak` before any edit. `bootstrap_gh200.sh` creates the file fresh each run; if you hand-edit, take the backup.

### 6.2 64K kernel won't boot (kernel panic, hangs on init)

GRUB's "Advanced options for Ubuntu" submenu still has the previous (generic 4K) kernel as a boot entry — the 64K install does **not** remove it.

1. Console-attach (as above).
2. At GRUB, pick "Advanced options for Ubuntu" → the previous (generic) kernel entry.
3. Boot.
4. Once back in:
   ```bash
   sudo apt purge linux-nvidia-64k-hwe-24.04 linux-headers-nvidia-64k-hwe-24.04
   # Find and pin the working kernel as default
   awk -F\' '/menuentry / {print $2}' /boot/grub/grub.cfg
   sudo sed -i 's/^GRUB_DEFAULT=.*/GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-XX-generic"/' /etc/default/grub
   sudo update-grub
   ```
5. **Do not** retry the 64K install without first checking that DOCA-OFED is on 3.2.0 (the single most common root cause of a 64K kernel that won't finish its initramfs is leftover MLNX-OFED DKMS state from the unpatched image).

### 6.3 Driver upgrade left the box without a working GPU

`nvidia-smi` returns "No devices were found" or a CUDA-version mismatch.

```bash
# Roll back to whatever was installed before
sudo apt install --reinstall nvidia-open-570    # or the previous version you noted
sudo apt-mark hold nvidia-open-570 cuda-drivers-570
sudo reboot
```

If `apt --reinstall` can't find the old version (NVIDIA removes superseded packages from `sbsa/` sometimes): use the runfile installer for the exact old version from `developer.nvidia.com/datacenter-driver-XYZ-download-archive`. Append `--no-drm --no-systemd --no-nvidia-modprobe` for headless GH200.

**Always check before upgrading:** `apt list --installed 2>/dev/null | grep nvidia-open` and **note the version**. This is your rollback target.

### 6.4 DOCA-OFED fix made `apt full-upgrade` worse

Rare but happens if the instance was caught mid-transition (e.g. the image was generated during the Dec 2025 → Jan 2026 DOCA repo cutover). Manual cleanup:

```bash
# Purge everything DOCA / MLNX-OFED related
for f in $(dpkg --list | grep -E 'doca|flexio|dpa-gdbserver|dpa-stats|dpa-resource-mgmt|dpaeumgmt|mlnx-ofed' | awk '{print $2}'); do
  sudo apt-get remove --purge -y "$f"
done
sudo /usr/sbin/ofed_uninstall.sh --force 2>/dev/null || true
sudo apt-get autoremove -y
sudo reboot
# Then reinstall DOCA 3.2.0 from scratch per Lenovo HPC guide (Sources)
```

This is destructive of any custom DOCA config. Lambda recommends opening a ticket before going this route on a paid instance.

### 6.5 Nuclear option

Lambda instances are stateless except for `/home/ubuntu` and any persistent filesystem you've attached. Cost of a full re-image is one shutdown + one new-instance launch (~5 minutes) plus the bootstrap re-run. If you're on the on-demand $2.29/hr tier, this is cheaper than 30 minutes of debugging.

---

## 7. Ubuntu 26.04 migration assessment

Round-1 agent 08 says "stay on 24.04, revisit 26.04 in August 2026 after 26.04.1 + a 590-series driver that lists 26.04." This Round 2 research confirms the picture sharpened, with one update:

### 7.1 What changed since Round 1 (May 25, 2026 fresh verification)

- **`developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/sbsa/` IS now populated.** Round-1 agent 08 §6 listed this as "not yet confirmed populated." It is. As of 2026-05-22 InRelease, the repo contains:
  - `nvidia-open_595.58.03`, `nvidia-open_595.71.05` (April 2026)
  - `nvidia-driver-open_595.58.03`, `nvidia-driver-open_595.71.05`
  - **No `nvidia-open-580`** — meaning NVIDIA decided 580 LTS would not be back-packaged to ubuntu2604/sbsa.
  - **No CUDA toolkit** packages (no `cuda-toolkit-13-*`). NVIDIA's repo provides only drivers on ubuntu2604; CUDA toolkit comes from Ubuntu's archive (13.1.x).
  - No cuDNN, no NCCL packages either.
- **Lambda Stack still 20.04/22.04/24.04 only.** No 26.04 line in `github.com/lambdal/lambda-stack-dockerfiles`. No 26.04 image in Lambda's instance catalog. No announced ETA.
- **Driver R595 is the only path on 26.04.** This means migrating to 26.04 today means *also* moving off R580-LTS to R595, which is a much newer branch with less production exposure on GH200. Round-1 agent 01 named R580 as the "current production" line for a reason.

### 7.2 Right-moment migration triggers

Stop watching the calendar. Watch these:

1. **Lambda Stack publishes a 26.04 line in `lambda-stack-dockerfiles`.** Mechanical signal — Lambda is doing the integration work for you. Historical lag from Canonical release to Lambda Stack line: ~5 months for 24.04. Project ~Sep 2026 earliest.
2. **NVIDIA publishes `cuda-toolkit-13-*` for ubuntu2604/sbsa.** Tells you NVIDIA decided 26.04 was worth full toolchain support, not just the driver-only minimal repo. Likely tied to CUDA 13.3 or 14.0 release.
3. **Ubuntu 26.04.1 ships (August 2026).** The first point-release catches the obvious early-life regressions (e.g. the `apt install nvidia-open` 26.04 issue from agent 08 §2.2).
4. **R580 or R595 driver release notes explicitly list `Ubuntu 26.04` as a supported OS** in the supported-OS table (not just as a build target in their package repo).
5. **PyTorch's `cu130` aarch64 wheels add a cp314 (Python 3.14) line on `download.pytorch.org`.** Until then, 26.04's default Python breaks `pip install torch`, and you'd need to maintain a py3.12 venv on a py3.14-default OS — extra ops cost.

When **3 of 5** are true, schedule the migration on a non-prod box. When **all 5** are true, migrate prod.

### 7.3 Earliest realistic prod migration date

Per the above, late Q4 2026 at the earliest, Q1 2027 more likely. There is no inference-side feature in 26.04 worth the early-adopter tax — the wins (Livepatch on Arm, native CUDA in archive, kernel 7.0) are operational and developer-experience improvements, not throughput improvements.

### 7.4 Hedge: what to do in the meantime

If you anticipate needing to move within 12 months, install the **`linux-nvidia-64k-hwe-24.04-edge`** kernel today (instead of the non-edge variant). It tracks closer to the upstream 6.11/6.14 line, which is what 26.04 will have backported features from. Your runtime tuning, IRQ pinning, NUMA scripts, and bootstrap will all transfer 1:1. The only thing that changes on 26.04 is the package names (driver from `nvidia-open-580` to `nvidia-open-595`) and the CUDA toolkit source (NVIDIA repo → Ubuntu archive — major behavioral change for CI).

---

## 8. Lambda-specific quirks consolidated

From Round-1 agents 01, 08, 16 — items the bootstrap silently assumes:

- Lambda's 1× GH200 exposes **64 vCPUs / 432 GiB DDR5 / 4 TiB local SSD** (not the 72/480 NVIDIA spec). Adjust `numactl` ranges and IRQ pins accordingly — `4-71` ranges from this guide work because `numactl --physcpubind` silently ignores out-of-range CPU IDs, but for cleanliness use `4-63` if you want to be precise.
- Lambda Stack's bundled **cuDNN is licensed only for Lambda's PyTorch/TF builds.** External frameworks (vLLM, TRT-LLM, your own builds) need symlinks to the NVIDIA-repo cuDNN. The bootstrap installs `libcudnn9-cuda-12` from NVIDIA which is fully licensed — but if you `pip install nvidia-cudnn-cu12` separately you may end up with three cuDNNs on the box. Pick one and `update-alternatives`.
- **Persistent filesystems are region-locked.** If you bootstrap in `us-east-1` you can't move the FS to `us-west-1` later. Keep model weights on a persistent FS in the region you plan to scale in.
- **No spot/preemptible tier.** Don't try to script auto-respawn against price signals — that's a feature for other clouds (Vast, RunPod).
- **`do-release-upgrade` from 22.04 → 24.04 on Lambda is documented as problematic** (the HackMD `@johnnynunez/HkQ8-doglg` walkthrough has been a community fixture for a year). Don't try to upgrade an existing 22.04 instance to 24.04 in place — launch a fresh 24.04 instance.

---

## Sources

Verified by direct fetch on 2026-05-25 or cross-confirmed by Round-1 agents 01 / 08 / 09 / 16:

- [NVIDIA Grace Performance Tuning Guide — OS Settings](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html)
- [NVIDIA Data Center Driver 580.159.03 release notes](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-159-03/index.html)
- [NVIDIA Data Center Driver 580.159.04 release notes](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-159-04/index.html)
- [NVIDIA CUDA repo — ubuntu2404/sbsa](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/) — confirmed 580.159.03, 580.159.04, 595.x, cuda-toolkit-13-0.3, cuda-toolkit-13-2.1
- [NVIDIA CUDA repo — ubuntu2604/sbsa](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/sbsa/) — populated with 595.x driver only; **no 580.x, no CUDA toolkit, no cuDNN, no NCCL** (this updates Round-1 agent 08's confidence band)
- [Lambda — On-demand troubleshooting (DOCA-OFED fix)](https://docs.lambda.ai/public-cloud/on-demand/troubleshooting/)
- [Lenovo HPC — DOCA 3 Install/Upgrade on RHEL 9 / Ubuntu 24.04](https://hpc.lenovo.com/users/documentation/doca3-install.html)
- [Lambda Stack package list](https://lambda.ai/lambda-stack-deep-learning-software) — CUDA 12.8.93, NCCL 2.26.2, PyTorch 2.7.0 confirmed
- [lambdal/lambda-stack-dockerfiles](https://github.com/lambdal/lambda-stack-dockerfiles) — 20.04/22.04/24.04 only, no 26.04
- [linux-nvidia-64k-hwe-24.04-edge on Launchpad](https://launchpad.net/ubuntu/noble/arm64/linux-nvidia-64k-hwe-24.04-edge)
- [NCCL Environment Variables (2.30.3)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NCCL issue #1291 — two-node GH200 over Quantum-2 stuck](https://github.com/NVIDIA/nccl/issues/1291)
- [HackMD: Upgrade GH200 to Ubuntu 24.04 (johnnynunez)](https://hackmd.io/@johnnynunez/HkQ8-doglg)
- [Stephen Diehl: Setting up an Nvidia GH200 for Development](https://www.stephendiehl.com/posts/setup_gh200_tutorial/)
- [NVIDIA — Performance Tuning for Mellanox Adapters](https://enterprise-support.nvidia.com/s/article/performance-tuning-for-mellanox-adapters)
- [NVIDIA blog — CUDA 13.0 overview (NUMA / non-NUMA init mode)](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/)
- [Phoronix — 64K Kernel Page Size Performance Benefits with NVIDIA GH200 Grace](https://www.phoronix.com/review/aarch64-64k-kernel-perf)
- Round-1 source files (this repo): `round1/01_driver_stack.md`, `round1/08_ubuntu_26_vs_24.md`, `round1/09_os_kernel_tuning.md`, `round1/16_lambda_pricing.md`

---

## Open Questions

1. **Is `linux-nvidia-64k-hwe-24.04` (non-edge) actually published on Ubuntu noble's arm64 archive today, or only `-edge`?** Round-1 agent 01 §2.3 names both; Launchpad confirms `-edge` exists. The non-edge variant is what most NVIDIA docs reference, but the Launchpad search surfaced only the edge package in the May-2026 search. Worth a `apt-cache search linux-nvidia-64k` on a live Lambda instance before depending on the non-edge package name.

2. **Does the May-2026 Lambda Stack 24.04 image *already* ship the 64K kernel by default, or does it still default to the 4K generic kernel?** Round-1 agent 01 §1 says "Lambda stock kernel = `linux-nvidia-64k-hwe-24.04` (6.11 SBSA)" — meaning Lambda may have switched. The bootstrap's stage 1 is a no-op in that case. `getconf PAGESIZE` on a fresh `gpu_1x_gh200` instance, before stage 1 runs, would settle this. The bootstrap is safe either way (idempotent apt install), but the reboot is wasted if 64K is already active.

3. **What is the exact `apt list --installed` driver version on a freshly-launched Lambda Stack 24.04 GH200 in May 2026?** Round-1 agent 01 says "R570 range"; agent 16 says "580.xx series (per dasroot.net Jan 2026)." Neither is from a live `nvidia-smi`. The driver-upgrade step in §2.2 only matters if Lambda hasn't already moved to R580. If they have, this is again a stage 2 no-op (apt detects current version satisfies). Confirmable in one minute on a live instance.

4. **Does NVIDIA's `ubuntu2404/sbsa` `libnccl2` track the upstream NCCL 2.30.4, or lag at 2.28.x / 2.29.x?** Round-1 agent 01 §6 says 2.30.4 is the upstream release; the apt repo version was not verified. If the apt-package NCCL lags, you may need to install NCCL from NVIDIA's separate NCCL apt repo or as a tarball, which adds steps not in this guide.

5. **Has the `nvidia-peermem` → `nvidia-dmabuf` transition happened on R580?** Round-1 agent 09 §10a says "modern path is dmabuf, but nvidia-peermem is still loaded for compatibility." The bootstrap loads `nvidia-peermem` defensively; whether it's still strictly needed on 580.159.03 with dmabuf-aware NICs (ConnectX-7 / BlueField-3) is unclear. Harmless if loaded redundantly; worth a follow-up to drop the modprobe if obsolete.

6. **Is the `gh200-runtime-tuning.service` `--lock-gpu-clocks=1410,1980` safe on a Lambda air-cooled GH200, or will it cause thermal throttling that the lock prevents from clearing?** The lock is a *limit* not an *override*, so thermals will still pull the clock down — but a locked range can lead to oscillation. Worth a 24-hour soak with `nvidia-smi dmon -s pucvmet -d 5` to confirm steady-state.

7. **CDMM mode (Coherent Driver-Based Memory Management) on R580.65.06+ — is it actually the default on bare-metal Lambda GH200, or only when GPU Operator is installed?** Round-1 agent 01 §4.3 says it became the GPU Operator default. Lambda's image doesn't use GPU Operator (it's a bare Linux box, not K8s). So CDMM is probably NOT active by default — meaning the GH200 still exposes HBM as a NUMA node. This is what the bootstrap and `numactl` wrappers assume, but worth confirming with `nvidia-smi -q | grep -i CDMM` or equivalent.

8. **For the rare Lambda Private Cloud customer with multi-node GH200 (NVL2 or NVL32):** the NCCL env-vars in §3.4 are starting points cribbed from OCI's published GH200 tuning. Lambda's network/subnet-manager/SL-PKey configuration is not publicly documented. These should be confirmed with Lambda support before relying on them in production multi-node training.
