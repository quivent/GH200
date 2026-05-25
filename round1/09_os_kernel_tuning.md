# GH200 OS / Kernel / Networking Tuning for Inference

**Research agent #09 of 20 — round 1**
**Date:** 2026-05-25
**Constraint:** FP8 broken on this stack — assume bf16 throughout.

---

## 1. Platform topology (what you see on a stock GH200)

A single GH200 Grace-Hopper superchip exposes **two primary NUMA nodes** (when MIG is off) — *or* **nine NUMA nodes** (when MIG is on):

| Node | Owner             | Notes |
|------|-------------------|-------|
| 0    | Grace CPU (LPDDR5X, 480 GB or 120 GB) | 72 Neoverse-V2 cores, CPUs 0–71 |
| 1    | Hopper HBM3/HBM3e (~96 GB or 144 GB) | 0 CPUs, GPU memory exposed as a CPU-less NUMA node |
| 2–8  | MIG instances     | Only present when MIG is enabled |

Distances: node0→node0 = 10, node0→node1 ≈ 80 (NVLink-C2C 900 GB/s coherent path).

```bash
numactl -H              # show NUMA topology
nvidia-smi topo -m      # show GPU-NIC affinity matrix
lscpu                   # confirm aarch64, 72 cores, 1 socket
```

Source: [NVIDIA Grace Performance Tuning Guide — Understanding Your Machine](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/system.html) (NVIDIA, current 2025/2026).

---

## 2. Kernel page size — 64K is the recommended default

NVIDIA explicitly recommends a **64 KB kernel page size** (`CONFIG_ARM64_64K_PAGES=y`) for Grace systems. On Ubuntu 22.04/24.04 the package is:

```bash
sudo apt purge linux-image-$(uname -r) linux-headers-$(uname -r) linux-modules-$(uname -r)
sudo apt install linux-nvidia-64k-hwe-22.04   # or -hwe-24.04
sudo reboot
```

Why it matters for inference:
- GPUDirect Storage with ext4 + local NVMe is qualified **only** with 64K host OS page size on Grace-Hopper.
- Fewer TLB misses for the multi-GB KV-cache pages.
- 4K kernels also work (`CONFIG_ARM64_4K_PAGES=y`), but lose ~5–15% on memory-bound LLM workloads in NVIDIA's reported numbers.

Sources: [NVIDIA Grace OS Settings](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html), [Stephen Diehl — Setting up a Nvidia GH200 for Development](https://www.stephendiehl.com/posts/setup_gh200_tutorial/).

---

## 3. Kernel boot parameters (GRUB)

Edit `/etc/default/grub`, append to `GRUB_CMDLINE_LINUX`:

```text
init_on_alloc=0 iommu.passthrough=1 numa_balancing=disable transparent_hugepage=madvise default_hugepagesz=2M hugepagesz=2M hugepages=0
```

Then:
```bash
sudo update-grub && sudo reboot
```

### Per-flag rationale

| Flag | Reason | Source |
|------|--------|--------|
| `init_on_alloc=0` | **Required on Grace-Hopper.** With cudaMalloc on coherent memory, page zeroing causes massive perf loss. | [Grace OS Settings](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html) |
| `iommu.passthrough=1` | Disables SMMU translation for DMA — needed for GDS / GPUDirect RDMA full bandwidth | [Grace OS Settings](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html) |
| `numa_balancing=disable` | AutoNUMA can migrate pages **off HBM back to LPDDR5**, killing inference. Disable it. | [Grace OS Settings](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html) |
| `transparent_hugepage=madvise` | Kernel 6.8 and earlier only support 512 MB THP on arm64-64K — `madvise` lets the runtime opt-in. Kernel 6.9+ enables 2 MB mTHP. Kernel 6.11+ has further arm64 improvements. | [Grace OS Settings](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html) |
| `isolcpus=…` / `nohz_full=…` | **Optional for inference; only useful for latency-critical pinned workers**. Keep CPU 0 for housekeeping. Example: `isolcpus=8-71 nohz_full=8-71 rcu_nocbs=8-71`. | [Red Hat — isolcpus/nohz_full/rcu_nocbs](https://access.redhat.com/articles/3720611), [Ubuntu Real-time kernel boot params](https://documentation.ubuntu.com/real-time/latest/reference/kernel-boot-parameters/) |

---

## 4. sysctl / runtime settings

Put in `/etc/sysctl.d/90-gh200.conf` and `sysctl -p`:

```ini
kernel.numa_balancing = 0
vm.compaction_proactiveness = 20      # default; tune down to 0 for steady-state inference
vm.swappiness = 10
vm.zone_reclaim_mode = 0
vm.max_map_count = 1048576            # large maps for many-shard models / vLLM
net.core.rmem_max = 268435456
net.core.wmem_max = 268435456
net.ipv4.tcp_rmem = 4096 87380 268435456
net.ipv4.tcp_wmem = 4096 87380 268435456
```

`vm.compaction_proactiveness` controls background memory compaction (range 0–100, default 20 per Grace tuning guide). Aggressive compaction wastes CPU under steady inference; if you have stable working sets, drop it to 0 or 5. To force a one-shot compaction: `echo 1 > /proc/sys/vm/compact_memory`.

Source: [NVIDIA Grace OS Settings](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html).

### ulimits (`/etc/security/limits.d/90-gpu.conf`)

```text
*    soft   memlock      unlimited
*    hard   memlock      unlimited
*    soft   stack        65536
*    hard   stack        65536
*    soft   nofile       1048576
*    hard   nofile       1048576
```

`memlock=unlimited` is **mandatory** for RDMA HCA queue-pair pinning and large cudaHostAlloc pinned buffers. Container equivalent: `docker run --ulimit memlock=-1 --ulimit stack=67108864 …` or `--shm-size=…`. Sources: [NVIDIA container-toolkit issue #226](https://github.com/NVIDIA/nvidia-container-toolkit/issues/226), HPE memlock docs.

---

## 5. Hugepages for KV-cache and model weights

`bf16` Llama-3 70B weights = ~140 GB. A KV-cache for 4096-tok × 256-seq fp16 batch can be tens of GB. Backing these with 2 MB hugepages reduces TLB pressure substantially.

```bash
# Reserve 32 GiB of 2M hugepages at runtime (16384 × 2 MiB)
echo 16384 | sudo tee /proc/sys/vm/nr_hugepages

# Persistent (sysctl):
echo "vm.nr_hugepages = 16384" | sudo tee -a /etc/sysctl.d/90-gh200.conf
sysctl -p

# Verify
grep Huge /proc/meminfo
```

1 GiB hugepages (boot-time only) — useful when serving very large single-shard models:
```text
GRUB_CMDLINE_LINUX="… default_hugepagesz=1G hugepagesz=1G hugepages=64 …"   # 64 GiB
```

CUDA pinned memory automatically benefits from THP set to `madvise`; vLLM/TensorRT-LLM will pick up explicit hugetlbfs mounts if configured.

Note for arm64-64K kernels: the "2 MB" THP is a kernel 6.9+ feature (mTHP); on older 6.8 kernels you only get 512 MB pages, which are too coarse and create fragmentation. **Run kernel 6.9 or newer**. Source: [Grace OS Settings](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html).

---

## 6. CPU governor, IRQ affinity, housekeeping

```bash
# Performance governor — required for stable latency
sudo apt install linux-tools-common cpufrequtils
sudo cpupower frequency-set -g performance
sudo systemctl disable cpufrequtils          # Ubuntu's autoreset

# Disable IRQ balancing (we'll pin manually)
sudo systemctl disable --now irqbalance

# Disable libvirtd on bare metal — it spawns bridges that hurt RDMA perf
sudo systemctl disable --now libvirtd.service
```

### Manual IRQ pinning for the Mellanox/ConnectX-7 HCA

`mlx5_core` exposes one IRQ per channel/queue. Pin them to CPUs **on the Grace NUMA node** (node 0 is the only CPU-bearing node on GH200, so this is implicit but still worth pinning to specific cores away from CPU 0):

```bash
# Use Mellanox-provided helper if present
sudo /usr/sbin/set_irq_affinity_cpulist.sh 4-15 mlx5_0
# Or manually
for irq in $(grep mlx5_0 /proc/interrupts | awk -F: '{print $1}'); do
  echo 4-15 | sudo tee /proc/irq/$irq/smp_affinity_list
done
```

Sources: [NVIDIA — Performance Tuning for Mellanox Adapters](https://enterprise-support.nvidia.com/s/article/performance-tuning-for-mellanox-adapters), [What is IRQ Affinity](https://enterprise-support.nvidia.com/s/article/what-is-irq-affinity-x).

---

## 7. nvidia-smi — persistence, clocks, power

```bash
# 1. Persistence (driver always loaded — avoids 1-2s cold-start per process)
sudo nvidia-smi -pm 1

# Make it survive reboots: persistence daemon
sudo systemctl enable --now nvidia-persistenced

# 2. Lock clocks for deterministic latency (H100 SM max boost = 1980 MHz)
sudo nvidia-smi --lock-gpu-clocks=1410,1980        # set range or single
sudo nvidia-smi --lock-memory-clocks=2619          # HBM3 max
# To release: sudo nvidia-smi --reset-gpu-clocks && sudo nvidia-smi --reset-memory-clocks

# 3. Power limit — GH200 configurable 450W..1000W; default 700W
sudo nvidia-smi -q -d POWER                        # check min/max for your part
sudo nvidia-smi -pl 1000                           # max performance (liquid-cooled SKUs)
sudo nvidia-smi -pl 700                            # default; safe air-cooled

# 4. ECC (already on by default for HBM3; do not disable)
nvidia-smi -q -d ECC
```

Caveats:
- `--lock-gpu-clocks` *limits* — it does not override thermal throttle.
- Persistence mode "does not persist across reboots" by default; the `nvidia-persistenced` systemd unit does.
- The Stephen Diehl GH200 setup overrides the unit with `--persistence-mode --verbose`.

Sources: [nvidia-smi docs](https://docs.nvidia.com/deploy/nvidia-smi/index.html), [Microway — nvidia-smi control](https://www.microway.com/hpc-tech-tips/nvidia-smi_control-your-gpus/), [Driver Persistence](https://docs.nvidia.com/deploy/driver-persistence/index.html), [Stephen Diehl GH200 setup](https://www.stephendiehl.com/posts/setup_gh200_tutorial/).

---

## 8. MIG on GH200 — **supported**

GH200's Hopper die is the H100 96 GB SKU; MIG is fully supported. Available profiles:

| Profile     | Memory  | SM fraction | Max instances |
|-------------|---------|-------------|---------------|
| 1g.12gb     | 1/8     | 1/7         | 7             |
| 1g.12gb+me  | 1/8     | 1/7         | 1 (with media engines) |
| 1g.24gb     | 1/4     | 1/7         | 4             |
| 2g.24gb     | 2/8     | 2/7         | 3             |
| 3g.48gb     | 4/8     | 3/7         | 2             |
| 4g.48gb     | 4/8     | 4/7         | 1             |
| 7g.96gb     | full    | 7/7         | 1             |

```bash
# Enable MIG (driver reload needed; no running processes)
sudo nvidia-smi -i 0 -mig 1

# List profiles, instances
nvidia-smi mig -lgip
nvidia-smi mig -lcip

# Create e.g. 2x 3g.48gb (model parallel pair)
sudo nvidia-smi mig -cgi 9,9 -C
# Or 7x 1g.12gb (max tenants for tiny models)
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -C
```

After enabling MIG, the system exposes the 7 MIG NUMA nodes (nodes 2–8 in `numactl -H`). NCCL inside a MIG instance is restricted to that instance's compute slice — *do not* use MIG for tensor-parallel; use full GPU.

Source: [NVIDIA MIG User Guide — Supported Profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html).

---

## 9. CUDA MPS for multi-tenant inference

When serving several small models on a single full GPU (no MIG), MPS lets multiple processes share the SM scheduler instead of context-switching:

```bash
# Start MPS daemon (system-wide)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
sudo -E nvidia-cuda-mps-control -d

# Per-client throttle (optional)
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25   # cap each client to 25% SMs
```

When to prefer MPS over MIG:
- **MPS**: variable-size workloads, want to oversubscribe, no hard memory isolation needed.
- **MIG**: hard QoS isolation, predictable latency tail, multi-tenant SLAs.

Sources: [NVIDIA MPS overview](https://docs.nvidia.com/deploy/mps/latest/index.html), [When to Use MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html), [Databricks — Scaling Small LLMs with MPS](https://www.databricks.com/blog/scaling-small-llms-nvidia-mps).

---

## 10. NCCL tuning for multi-GH200 nodes

### 10a. Required modules / system setup

```bash
# Peer memory bridge: NIC ↔ GPU HBM DMA. Modern path is dmabuf, but nvidia-peermem
# is still loaded for compatibility.
echo nvidia-peermem | sudo tee /etc/modules-load.d/nvidia-peermem.conf
sudo modprobe nvidia-peermem

# RDMA shared-namespace mode (Kubernetes / multi-tenant)
sudo rdma system set netns shared
```

### 10b. NCCL environment variables — recommended baseline

```bash
# Mandatory for two-node GH200 over IB to avoid the bootstrap deadlock
export NCCL_SOCKET_IFNAME=bond0          # or your control-plane iface; NOT ib*
export NCCL_IB_HCA=mlx5_2                # filter to actual RDMA HCA(s)

# Default OK; leave NCCL auto unless tuning
# export NCCL_NVLS_ENABLE=2              # default; 1 = require, 0 = disable
# export NCCL_P2P_LEVEL=NVL              # NVLink-only intra-node; SYS to allow cross-NUMA P2P
# export NCCL_NET_GDR_LEVEL=SYS          # always use GPUDirect RDMA (Grace-Hopper has only 1 NUMA path)
# export NCCL_CROSS_NIC=0                # for rail-optimized fabrics; 1 only if your switch design tolerates cross-rail

# Diagnostics first time you bring up a cluster
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV

# Multi-QP for routing entropy on adaptive-routing fabrics (Quantum-2)
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_QPS=0       # round-robin (default since 2.20)

# Service Level / Traffic Class — match what your subnet manager / switch QoS expects
export NCCL_IB_SL=0                      # OCI uses 0 or 5
export NCCL_IB_TC=41                     # OCI standard for control-msg priority
export NCCL_IB_TIMEOUT=22                # ~17s; longer than default 4.3s for large fabrics
export NCCL_IB_GID_INDEX=3               # RoCEv2 typical; check `show_gids`

# Buffer size (default 4 MiB) — increase to 8M only if memory permits and tests show benefit
# export NCCL_BUFFSIZE=4194304
```

### 10c. Key behaviors specific to GH200 / NVL32

- **GH200 NVL32**: 32 GH200 GPUs, 9 NVSwitch trays, one NVLink domain. Inside the domain NVLS (NVLink-SHARP) for AllReduce is supported and **on by default** (`NCCL_NVLS_ENABLE=2`). NCCL's internal tuning is generally optimal; per NVIDIA's Multi-Node Tuning Guide, "users should not need to tune or set NCCL environment variables to achieve peak performance".
- **GH200 multi-node over Quantum-2 IB**: known issue where bootstrap selects mismatched interfaces across nodes (ib0 vs bond0). Workaround: explicit `NCCL_SOCKET_IFNAME` + `NCCL_IB_HCA`.
- **Bandwidth ceiling**: with a single Bluefield-3 (2×200Gb RoCE) port pair, NCCL allreduce/busbw tops out around 48 GB/s (~384 Gbps). NDR400 ConnectX-7 with 16× Gen5 lanes delivers up to 100 GB/s aggregate across superchips.
- **NCCL_MNNVL_ENABLE=1** (default on supported NCCL ≥ 2.25.2) enables multi-node NVLink. Set to `0` to force IB/RoCE path for testing.

Sources:
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [GB200 NVL Multi-Node Tuning Guide — NCCL](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/nccl.html)
- [Multi-Node GH200 NCCL Testing — Ed Sealing](https://medium.com/@ed.sealing/multi-node-gh200-nccl-testing-dc2fc64d97a0)
- [nccl-tests stuck on two-node GH200 / Quantum-2 — NCCL issue #1291](https://github.com/NVIDIA/nccl/issues/1291)
- [Low Latency Inference — GH200 NVL32 with NVLink Switch](https://developer.nvidia.com/blog/low-latency-inference-chapter-2-blackwell-is-coming-nvidia-gh200-nvl32-with-nvlink-switch-gives-signs-of-big-leap-in-time-to-first-token-performance/)
- [One Giant Superchip — GH200 NVL32](https://developer.nvidia.com/blog/one-giant-superchip-for-llms-recommenders-and-gnns-introducing-nvidia-gh200-nvl32/)

---

## 11. NUMA pinning for inference workers (vLLM / TRT-LLM / sglang)

For a single-GPU GH200 process: bind to Grace CPU node, prefer HBM for allocations that you want on-GPU:

```bash
# CPU + memory bound to Grace node 0 (CPU), but allow access to node 1 (HBM)
numactl --cpunodebind=0 --membind=0,1 -- python -m vllm.entrypoints.openai.api_server …

# Force all CPU-side allocations to HBM (advanced; e.g. for HBM-resident KV cache scratch)
numactl --cpunodebind=0 --preferred=1 -- python …

# Pinned to specific Grace cores (leave 0-3 for housekeeping)
numactl --physcpubind=4-71 --membind=0,1 -- python …
```

vLLM has open issues asking for first-class NUMA pinning ([vllm-project/vllm#13855](https://github.com/vllm-project/vllm/issues/13855), [#11720](https://github.com/vllm-project/vllm/issues/11720)). Wrapping with `numactl` is the current best practice.

Sources: [Grace Performance Tuning Guide](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/index.html), [vllm CPU binding doc](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/cpu_binding.html).

---

## 12. Storage — checkpoint cache & GDS

### 12a. Filesystem choice

| FS    | GDS native | Notes for checkpoint/model cache |
|-------|------------|----------------------------------|
| ext4  | Yes (compat + native with 64K kernel + local NVMe) | Recommended default for local checkpoint cache. |
| xfs   | Yes (compat mode) | Good large-file throughput; use for >TB checkpoint volumes. |
| btrfs | Not qualified | Avoid for hot paths. |
| Lustre / WekaFS / GPFS / BeeGFS / NFSoRDMA | Full GDS native | Use for cluster-wide model registries. |

Source: [GDS Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html), [GDS Release Notes](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html).

### 12b. Mount options

```bash
# /etc/fstab — local NVMe model cache
/dev/nvme0n1p1  /var/lib/models  ext4  noatime,nodiratime,discard,errors=remount-ro  0  2

# xfs alternative
/dev/nvme0n1p1  /var/lib/models  xfs   noatime,nodiratime,inode64,allocsize=64m       0  2
```

### 12c. Enable GDS

```bash
sudo apt install nvidia-gds
sudo /usr/local/cuda/gds/tools/gdscheck.py -p     # verify support matrix
# Per-app override:
export CUFILE_ENV_PATH_JSON=/etc/cufile.json
```

Open files with `O_DIRECT` from the loader to bypass the page cache (required for GDS native path). PyTorch via `torch.cuda.gds` or `safetensors` newer GDS-aware loaders.

---

## 13. End-to-end inference launch template (single GH200)

```bash
#!/usr/bin/env bash
# Run as root once per boot, OR put into systemd
nvidia-smi -pm 1
nvidia-smi -pl 1000
nvidia-smi --lock-gpu-clocks=1410,1980
nvidia-smi --lock-memory-clocks=2619
echo 0 > /proc/sys/kernel/numa_balancing
echo 16384 > /proc/sys/vm/nr_hugepages

# As inference user
ulimit -l unlimited
ulimit -s 65536
ulimit -n 1048576

exec numactl --cpunodebind=0 --membind=0,1 -- \
    python -m vllm.entrypoints.openai.api_server \
      --model meta-llama/Llama-3.1-70B-Instruct \
      --dtype bfloat16 \
      --gpu-memory-utilization 0.92 \
      --enable-prefix-caching
```

(bf16 — FP8 disabled per stack constraint.)

---

## Confidence & Gaps

### High confidence
- NUMA layout (2 nodes baseline, 9 with MIG) — confirmed by NVIDIA Grace Performance Tuning Guide.
- `init_on_alloc=0`, `iommu.passthrough=1`, `numa_balancing=0` — explicitly required per NVIDIA OS Settings doc.
- 64K page-size kernel recommendation — explicitly stated by NVIDIA.
- MIG support on GH200 and exact profile table — from MIG User Guide r580.
- NCCL bootstrap deadlock on multi-node GH200 + Quantum-2 + the `NCCL_SOCKET_IFNAME` workaround — confirmed in NCCL GitHub issue #1291.
- nvidia-smi commands for persistence/clocks/power-limit — well-documented across NVIDIA pages.

### Medium confidence
- Exact optimal `vm.nr_hugepages` value — depends on model size; the 16384 (32 GiB) figure is a sensible starting point but workload-dependent. NVIDIA Grace docs do not publish a recommended number.
- `NCCL_IB_TIMEOUT=22`, `NCCL_IB_QPS_PER_CONNECTION=4`, `NCCL_IB_SL=0`, `NCCL_IB_TC=41` — these are Oracle Cloud / community best-practice values, not NVIDIA-blessed for GH200 specifically. Treat as starting point and tune.
- `NCCL_CROSS_NIC=0` recommendation assumes rail-optimized fabric; may need to be 1 on flatter topologies.
- 1 GiB hugepage benefit on 64K arm64 kernels — plausible but not measured in any source I located; NVIDIA's THP guidance focuses on 2 MB mTHP (kernel 6.9+).

### Low confidence / open questions
- Whether `isolcpus`/`nohz_full` measurably help inference latency on GH200 specifically. The Grace docs don't recommend them; they're real-time-Linux idioms. For p99 latency-sensitive online serving they're worth A/B testing, but I found no published GH200 numbers.
- GDS performance numbers with ext4 + local NVMe on 64K Grace-Hopper kernel — NVIDIA states it's qualified, but I found no public benchmarks (release notes only).
- Exact behavior of `NCCL_MNNVL_ENABLE` for GH200 NVL32 (vs the better-documented GB200 NVL72). NVIDIA's Multi-Node Tuning Guide is GB200-centric; GH200 NVL32 inherits most of it but specifics like NVLS thresholds may differ. The 2024 NVIDIA blog post on GH200 NVL32 (TTFT for Llama 3.1 405B) is the best public reference.
- vLLM/sglang NUMA-pinning auto-detection on GH200 — still being landed upstream as of vllm#13855/#11720.

### Cross-team handoffs needed
- **Networking team**: confirm subnet manager SL/PKey/TC mapping before fixing `NCCL_IB_SL`/`NCCL_IB_TC`.
- **Driver team**: GH200 needs CUDA 12.4+ and driver r555+ at minimum; multi-node NVLink (MNNVL) needs NCCL ≥ 2.25.2 (GB200 baseline) — GH200 NVL32 may have an earlier compatible version, worth checking.
- **Filesystem team**: confirm /var/lib/models is on a 64K-page-compatible local NVMe before claiming GDS native path.

---

## Sources (all reachable, dates verified at fetch time on 2026-05-25)

1. [NVIDIA Grace Performance Tuning Guide — Operating System Settings](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html)
2. [NVIDIA Grace Performance Tuning Guide — Understanding Your Machine](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/system.html)
3. [NVIDIA Grace Performance Tuning Guide — Index](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/index.html)
4. [NVIDIA Multi-Instance GPU User Guide — Supported MIG Profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html)
5. [NCCL Environment Variables (2.30.3)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
6. [GB200 NVL Multi-Node Tuning Guide — NCCL](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/nccl.html)
7. [GB200 NVL Multi-Node Tuning Guide — System](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/system.html)
8. [NVIDIA Multi-Process Service](https://docs.nvidia.com/deploy/mps/latest/index.html) / [When to Use MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)
9. [nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
10. [Driver Persistence](https://docs.nvidia.com/deploy/driver-persistence/index.html)
11. [GPUDirect Storage — Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html), [Release Notes](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html), [Overview](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)
12. [NVIDIA — Performance Tuning for Mellanox Adapters](https://enterprise-support.nvidia.com/s/article/performance-tuning-for-mellanox-adapters)
13. [NVIDIA — IRQ Affinity for Mellanox](https://enterprise-support.nvidia.com/s/article/what-is-irq-affinity-x)
14. [NCCL issue #1291 — two-node GH200 over Quantum-2 stuck](https://github.com/NVIDIA/nccl/issues/1291)
15. [Multi-Node GH200 NCCL Testing — Ed Sealing, Medium](https://medium.com/@ed.sealing/multi-node-gh200-nccl-testing-dc2fc64d97a0)
16. [Dual-Network Kubernetes Pods with RDMA on NVIDIA GH200 — BaaZ](https://baaz.dev/blog/dual-network-rdma-kubernetes-gh200)
17. [Setting up a Nvidia GH200 for Development — Stephen Diehl](https://www.stephendiehl.com/posts/setup_gh200_tutorial/)
18. [Low Latency Inference, Ch. 2: GH200 NVL32 with NVLink Switch — NVIDIA Blog](https://developer.nvidia.com/blog/low-latency-inference-chapter-2-blackwell-is-coming-nvidia-gh200-nvl32-with-nvlink-switch-gives-signs-of-big-leap-in-time-to-first-token-performance/)
19. [One Giant Superchip — Introducing GH200 NVL32 — NVIDIA Blog](https://developer.nvidia.com/blog/one-giant-superchip-for-llms-recommenders-and-gnns-introducing-nvidia-gh200-nvl32/)
20. [GH200 Superchip Accelerates Inference 2x on Llama — NVIDIA Blog](https://developer.nvidia.com/blog/nvidia-gh200-superchip-accelerates-inference-by-2x-in-multiturn-interactions-with-llama-models/)
21. [Red Hat — isolcpus / nohz_full / rcu_nocbs](https://access.redhat.com/articles/3720611)
22. [Red Hat — isolcpus overview](https://access.redhat.com/solutions/480473)
23. [Ubuntu — Real-time kernel boot parameters](https://documentation.ubuntu.com/real-time/latest/reference/kernel-boot-parameters/)
24. [vLLM issue #13855 — NUMA pinning](https://github.com/vllm-project/vllm/issues/13855)
25. [vLLM issue #11720 — membind NUMA nodes](https://github.com/vllm-project/vllm/issues/11720)
26. [NVIDIA — Microway nvidia-smi guide](https://www.microway.com/hpc-tech-tips/nvidia-smi_control-your-gpus/)
27. [nvidia-container-toolkit issue #226 — pinned memory](https://github.com/NVIDIA/nvidia-container-toolkit/issues/226)
28. [Databricks — Scaling Small LLMs with NVIDIA MPS](https://www.databricks.com/blog/scaling-small-llms-nvidia-mps)
