# GH200 OS / Kernel / Networking Tuning — Round 3 Verification

**Research agent #09 of 20 — round 3 (verification + deepening)**
**Date:** 2026-05-25
**Constraint:** FP8 broken on this stack — bf16 throughout.
**Reads-from:** `/home/ubuntu/research/gh200_inference/round1/09_os_kernel_tuning.md`

---

## TL;DR — what changed between round 1 (claims) and round 3 (measured)

| Setting                              | Round 1 claim                    | Round 3 measured                          | Verdict |
|--------------------------------------|----------------------------------|-------------------------------------------|---------|
| 64K vs 4K kernel page                 | "5–15% on memory-bound LLM"      | **+15% geomean across HPC suite (Phoronix); 15.9x faster free(); 5x faster Qiskit init; 58% faster page migration on 64KB** | Confirmed and *understated* on memory-heavy paths |
| `init_on_alloc=0`                    | "Required, massive perf loss otherwise" | NVIDIA + Werner et al. recommend it; **no public numeric delta** found       | Plausible, unmeasured |
| nvidia persistence mode              | "Avoids 1–2s cold start"         | **3.16–3.25 s cold → 0.09 s with -pm 1 ≈ 35x; variance ±0.3 s → ±0.01 s (30x tighter)** | Confirmed and *understated* |
| GPU clock locking                     | "Deterministic latency"          | Tighter variance, but **does not override power-limit throttle**; H100 still drops to 1395 MHz when 350 W cap hits | Caveat needed |
| NUMA pinning (numactl)                | "Best practice"                  | **vLLM on GH200 is insensitive to OMP/CPU bind in published tests** (8B Llama, ShareGPT). Auto-NUMA-balancing disable is documented, no measured delta | Weak evidence for inference; strong for HPC |
| MPS multi-tenant tail-latency        | "Use for small models"           | **+50% throughput on H100 with small models, <2k ctx**; Orion paper measures **2.2x worse p99 vs ideal** for MPS without interference scheduling | Mixed — throughput up, p99 down |
| NCCL 2.30.4 GH200 fixes              | Not mentioned                    | **No GH200/MNNVL/aarch64 entries**; only `nccl_param` headers fix + elastic buffer w/ GIN | No GH200-specific fix in latest |
| nccl-tests single-superchip C2C       | implicit "high"                  | **150 GB/s intra-node measured at CSCS Alps**; cudaMemcpy hits **375 GB/s H2D, 297 GB/s D2H (~78% of 450 theoretical)** | Concrete numbers found |
| BlueField-3 vs ConnectX-7 on Lambda  | Round 1 NCCL block assumed BF-3  | **Lambda 1-Click GH200 clusters: Quantum-2 NDR 400 Gb/s** (ConnectX-7 family hardware on the GH200 side); BF-3 referenced on Ed Sealing on-prem testbed, not Lambda | Round 1 conflated two different deployments |

---

## 1. Per-setting measured deltas

### 1a. 64K kernel vs 4K kernel — *measured*

**Phoronix (Linux 6.8, GH200 Grace CPU, 72-core)**:
- **Geomean ≈ +15%** across "several dozen CPU benchmarks" (Phoronix Feb 2024 review).

**Werner / Andersson et al. (arXiv 2407.07850 — "Harnessing Integrated CPU-GPU System Memory", a Grace-Hopper first-look paper)**:
- `free()` / de-allocation: **15.9x faster** average (range 4.6x to 38x).
- Qiskit 33-qubit GPU-side init: **5x faster**.
- Qiskit 34-qubit oversubscription with page migration: **+58%**.

Why it matters for inference:
- Model load (mmap of multi-GB safetensors) hits the de-allocation path on unload/reload.
- KV-cache page migration (when HMM/UVM is in play for paged attention or weight streaming) is the 58% case.
- LLM serve steady-state is less sensitive to page size than HPC — but cold-start, hot-reload, and any HMM-backed allocator (TensorRT-LLM weight streaming) benefits.

Source: [Phoronix — 64K Kernel Page Size Benefits HPC on GH200](https://www.phoronix.com/review/aarch64-64k-kernel-perf), [arXiv 2407.07850 — Harnessing CPU-GPU System Memory on Grace Hopper](https://arxiv.org/html/2407.07850v1).

### 1b. `init_on_alloc=0` — *no published number*

- NVIDIA Grace OS Settings doc states `CONFIG_INIT_ON_ALLOC_DEFAULT_ON` "can cause heavy performance impacts to cudaMalloc()" on coherent memory.
- Werner et al. note the setting is mandatory but **publish no isolated delta**.
- All Grace benchmarking guides use it as a baseline assumption — no A/B numbers in public literature.

Status: **plausible but uninstrumented in public**. If you need a number, run cudaMalloc/free in a tight loop with and without the boot arg.

Source: [NVIDIA Grace OS Settings](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html), [arXiv 2407.07850](https://arxiv.org/html/2407.07850v1).

### 1c. Persistence mode — *concrete numbers*

From Abhik Sarkar's measurement post (single-host GPU; same character-device init path on GH200):

| Metric                      | -pm 0 (default)       | -pm 1 (persistence)   |
|-----------------------------|-----------------------|-----------------------|
| First `nvidia-smi`          | 3.247 s               | 0.090 s               |
| Repeat (driver still loaded)| 0.089 s               | 0.090 s               |
| After driver unload         | 3.156 s               | 0.090 s               |
| Variance across runs        | ±0.3 s                | ±0.01 s               |
| 100-job batch overhead      | 320 s                 | 10 s                  |
| Speedup factor              | —                     | **~35x**              |
| Resource cost               | —                     | ~2–4 MB/GPU           |

Why round 1's "1–2 s" was conservative: it conflated steady-state cost with cold-boot cost. Real penalty on each cold start is closer to **3.2 s**.

For inference servers this means:
- vLLM/TRT-LLM cold start saves ~3 s per process spawn.
- More important: **scheduler / autoscaler with frequent pod recycle** sees massive aggregate.
- Variance reduction (30x tighter) is the *real* benchmarking win — locks out a noise source bigger than most algorithm tweaks.

Source: [Abhik Sarkar — NVIDIA Persistence Daemon (measured)](https://www.abhik.ai/concepts/gpu/nvidia-persistence-daemon).

### 1d. Clock-locking — *necessary but not sufficient*

`--lock-gpu-clocks` reduces benchmark variance, **but does not override the power-limit throttle**.
- StandardKernel.com observed H100 dropping from boost (1980 MHz) to **1395 MHz** under sustained kernel benchmarks because the 350 W (PCIe SKU) limit pulls clocks down.
- For GH200 (700 W default, configurable up to 1000 W), thermal headroom is much better — but lock + raise power limit are *both* needed for full reproducibility.

Practical recipe (extends round 1):
```bash
sudo nvidia-smi -pl 1000                          # max for liquid-cooled
sudo nvidia-smi --lock-gpu-clocks=1410,1980       # set range
sudo nvidia-smi --lock-memory-clocks=2619         # HBM3
sudo nvidia-smi -q -d POWER                       # verify cap
```

Source: [StandardKernel — "This Kernel Was Faster Yesterday" (GPU benchmark fidelity)](https://standardkernel.com/blog/in-pursuit-of-high-fidelity-gpu-kernel-benchmarking/), [NVIDIA SetStablePowerState blog](https://developer.nvidia.com/blog/advanced-api-performance-setstablepowerstate/).

### 1e. NUMA pinning — *measured insensitivity on GH200*

vLLM forum (user-reported measurement, 8B Llama-3.1, ShareGPT 4096 prompts, 55 req/s, GH200):
- `OMP_NUM_THREADS=2` vs `=72`: **no measurable throughput delta**.
- `VLLM_CPU_OMP_THREADS_BIND=0-1` vs `=0-71`: **no measurable throughput delta**.

This contradicts the "best practice" framing in round 1. The likely explanation:
1. GH200 has only **one CPU NUMA node** (Grace, node 0) — there's nothing to mis-bind across at the CPU side.
2. The HBM-side node (node 1) is CPU-less, so AutoNUMA can only thrash if you allocate large CPU-side buffers that overflow LPDDR5X — uncommon in vLLM's hot path.
3. Decoding is GPU-bound; CPU thread placement is in the noise.

When NUMA pinning *does* matter on GH200:
- **HPC workloads** that touch LPDDR5X and HBM mixed (NVIDIA Grace Tuning Guide gives 5–15% on STREAM/HPL).
- **Multi-GH200 nodes via NVLink-C2C** (NVL2 / NVL32) where the wrong superchip's CPU is bound.
- **Long-context prefill** with prompt-side embeddings on CPU.

For single-GH200 inference, round 1's `numactl --cpunodebind=0 --membind=0,1` is **harmless but not measurably beneficial** based on public data.

Source: [vLLM forum — performance insensitive to CPU binding on GH200](https://discuss.vllm.ai/t/vllm-performance-insensitive-to-cpu-binding/1662), [vLLM issue #13855](https://github.com/vllm-project/vllm/issues/13855), [NVIDIA Grace OS Settings](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html).

---

## 2. MPS multi-tenant — measured tail-latency picture

**Databricks (Jan 2026, H100, Qwen2.5 family 0.5B–7B, vLLM)**:
- **+50% throughput** for small models with short context (≤128 tok).
- **+100% throughput** for prefill-heavy Qwen2.5 <3B.
- **+35% throughput** for Gemma-4B in CPU-bound regime.
- Drops below 10% benefit at 2K context.
- **Slowdown** above ~7B model with long context.
- **No p99 numbers published** by Databricks — only mean throughput.

**Orion paper (EuroSys 2024, "Interference-aware GPU sharing")**:
- Bare MPS gives **2.2x worse p99 latency** versus ideal (no-interference baseline).
- Orion's scheduler reclaims most of that gap by ordering kernels by compute/memory profile.

**Predictable LLM Serving (arXiv 2508.20274)**:
- Compared to naive MIG/MPS placement, a smarter scheduler gives **−32% SLO miss-rate**, **−15% p99**, at <5% throughput cost.
- TTFT p99 improves **10–15%** from PCIe-aware placement alone.

**Implication for GH200**:
- MPS gains for small-model multi-tenant exist but **p99 is the casualty**. Use MIG (1g.12gb, 2g.24gb profiles) for QoS-sensitive serving; reserve MPS for throughput-maximization of stateless small workloads.
- No GH200-specific MPS tail-latency paper exists yet (May 2026).

Source: [Databricks — Scaling Small LLMs with NVIDIA MPS](https://www.databricks.com/blog/scaling-small-llms-nvidia-mps), [Orion (EuroSys 2024 PDF)](https://anakli.inf.ethz.ch/papers/orion_eurosys24.pdf), [Predictable LLM Serving (arXiv 2508.20274)](https://arxiv.org/html/2508.20274v1).

---

## 3. NCCL 2.30.4 — *no GH200-specific changes*

From GitHub release notes (Apr 22, 2026) plus archive scan back through 2.28.3:

| Version  | Date          | GH200 / MNNVL / aarch64 entries |
|----------|---------------|---------------------------------|
| 2.30.4-1 | 2026-04-22    | None. Only `nccl_param` headers fix + elastic buffer w/ GIN. |
| 2.30.3-1 | 2026-04-15    | None GH200-specific; "LL128 protocol in heterogeneous scenarios for Hopper and later" (generic Hopper). |
| 2.29.7-1 | 2026-02-27    | None; GIN decoupled from NET plugin. |
| 2.29.2-1 | 2025-12-24    | Fix: "hang on GB200/300 + CX8 when user disables GDR" (GB200, not GH200). |
| 2.28.7-1 | 2025-10-18    | None. Device API framework. |
| **2.28.3-1** | **2025-10-06** | **"Enable PxN over C2C by default — improves Grace-Blackwell performance by leveraging the NIC attached to a peer GPU."** Grace-Blackwell, not GH200, but architecturally similar C2C topology. |

**Open GH200 issue still unresolved** as of May 2026:
- [NCCL #1272](https://github.com/NVIDIA/nccl/issues/1272) — *Performance drop on ≥ 8 nodes (32 GH200) for NCCL ≥ 2.20*.
  - 4 nodes: fine.
  - 8 nodes: **24.24 GB/s (2.19.4)** → **18.32 GB/s (2.20.5)** = ~**−25%** sendrecv_perf bus bandwidth.
  - No fix shipped through 2.30.4 per release notes.

**Implication**: if you're running multi-node GH200 at scale, **NCCL 2.19.4 may still beat 2.30.4** on 8+ nodes. Test before upgrading. The hot fix candidate would be the PxN-over-C2C work from 2.28.3 if your topology has NIC-per-GPU, but the regression on the bootstrap/aws-ofi-nccl path is still open.

Source: [NCCL releases](https://github.com/NVIDIA/nccl/releases), [NCCL 2.30.3 release notes](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-30-3.html), [NCCL issue #1272](https://github.com/NVIDIA/nccl/issues/1272), [NCCL issue #1291](https://github.com/NVIDIA/nccl/issues/1291).

---

## 4. nccl-tests on a single GH200 — *intra-superchip C2C*

The "single GH200" case is **not allreduce** in the usual sense (1 GPU → ring of size 1). The relevant intra-superchip number is CPU↔GPU bandwidth through NVLink-C2C.

Published measurements:

| Direction              | Theoretical | Measured              | Source |
|------------------------|-------------|-----------------------|--------|
| H2D (CPU→GPU)          | 450 GB/s    | **375 GB/s** (~83%)   | Werner et al., arXiv 2407.07850 |
| D2H (GPU→CPU)          | 450 GB/s    | **297 GB/s** (~66%)   | same |
| Bidirectional aggregate| 900 GB/s    | **350 GB/s** uni-best | Schieffer/Faj (arXiv 2408.11556) |
| GPU-side STREAM (HBM3) | 4 TB/s      | **3.4 TB/s** (~85%)   | Werner et al. |
| CPU-side STREAM (LPDDR5X) | 500 GB/s | **486 GB/s** (~97%)   | Werner et al. |

**Multi-node intra-node bandwidth** (CSCS Alps, GH200 per node):
- **150 GB/s** sustained intra-node for collectives — the C2C bottlenecks AllReduce because the CPU↔GPU coherent path is shared with the network egress.

H2D-D2H asymmetry (375 vs 297) is real and reproducible across multiple papers — favor designs that minimize device→host transfers (pinned host write buffers, async D2H aggregation).

Source: [arXiv 2407.07850 — Harnessing CPU-GPU System Memory](https://arxiv.org/html/2407.07850v1), [arXiv 2408.11556 — Understanding Data Movement on Grace Hopper](https://arxiv.org/html/2408.11556v2), [verda.com — Data Movement in NVIDIA's Superchip Era](https://verda.com/blog/data-movement-in-the-nvidia-gh200-grace-hopper-superchip), [HPI Werner GH200 paper](https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/rabl/publications/2025/hcds25-Werner-GraceHopper.pdf).

---

## 5. BlueField-3 vs ConnectX-7 on Lambda — what ships?

This was the most-ambiguous claim in round 1. Reconciled:

| Source                                     | Hardware claim                  | Multi-node fabric           |
|--------------------------------------------|---------------------------------|-----------------------------|
| Lambda 1-Click Clusters (GH200)            | "NVIDIA Quantum-2 400 Gb/s IB"  | **NDR400 InfiniBand**       |
| Lambda on-demand 1x GH200 (single-node)    | Spec page lists no NIC          | n/a (single node)           |
| Lambda announcement blog (May 2024)        | "demo clusters up to 720 Grace CPUs, 960 GB H100 mem" | Doesn't disambiguate NIC vendor |
| Ed Sealing's on-prem GH200 testbed         | **BlueField-3 (2×200 Gb/s RoCE)** — gave 48 GB/s NCCL bus bw | RoCE, not Lambda |
| NVIDIA MGX GH200 reference design          | "BlueField-3 *or* ConnectX-7"   | Both supported by reference |

**Conclusion**:
- **Lambda 1-Click GH200 multi-node ships Quantum-2 NDR-400 InfiniBand** — that is the ConnectX-7 family on the host side (CX-7 OSFP NDR400). Lambda public collateral never names a BlueField-3 on GH200 boxes.
- **The 48 GB/s number in round 1 (and Ed Sealing's blog) is BlueField-3 over RoCE, not InfiniBand** — that was on-prem testing, not Lambda. Round 1 implicitly conflated the two.
- Expected Lambda multi-node ceiling: NDR400 line rate (≈47 GB/s per port × multiple ports per node). The exact NIC count per GH200 node on Lambda is not published.
- Single-node Lambda GH200 has **no relevant NIC** — there's no fabric to use.

Action: if your code paths assume "BlueField-3 RoCE, NCCL_IB_TC=41 OCI tunables", **none of that applies on Lambda**. Use Quantum-2 / IB defaults; `NCCL_IB_HCA=mlx5_0` (or whatever `ibstat` reports); drop the RoCE-specific GID-index tuning.

Source: [Lambda 1-Click Clusters](https://lambda.ai/1-click-clusters), [Lambda Cloud Clusters GH200 announcement](https://lambda.ai/blog/lambda-cloud-clusters-now-available-with-nvidia-gh200-grace-hopper-superchip), [Lambda on-demand docs](https://docs.lambda.ai/public-cloud/on-demand/), [Ed Sealing — multi-node GH200 NCCL testing (on-prem BlueField-3 testbed)](https://medium.com/@ed.sealing/multi-node-gh200-nccl-testing-dc2fc64d97a0), [NVIDIA MGX GH200 reference](https://aiserver.eu/product/nvidia-mgx-gh200/).

---

## 6. Net corrections to round 1

1. **§7 persistence mode "1-2s cold-start"** → replace with **3.2 s, 35x speedup, 30x variance reduction** (Sarkar).
2. **§10c "BlueField-3 single port pair tops out at ~48 GB/s"** → keep as a *general* GH200-BF3-RoCE data point, but **strike the implicit Lambda association**. Add a note that **Lambda ships Quantum-2 NDR400 InfiniBand**, not RoCE.
3. **§11 NUMA pinning "best practice"** → soften to "**harmless for inference; only measurably wins on HPC/HPL/STREAM-mixed workloads.** GH200 has one CPU NUMA node, so the binding cost-of-omission is low".
4. **§9 MPS "when to prefer"** → add **tail-latency caveat (Orion 2.2x p99 cost)** and explicit recommendation to use **MIG for QoS, MPS for throughput-only**.
5. **§7 clock-lock** → add caveat **lock-clocks does not override power-cap throttle**; raise `-pl` first.
6. **§2 page-size "5–15% on memory-bound LLM"** → strengthen with **Werner numbers (15.9x free, 5x init, 58% migration)** for cold-path operations.

---

## 7. Confidence after round 3

### High confidence (now backed by numbers)
- 64K page kernel: +15% geomean (Phoronix), 15.9x free, 5x Qiskit init, 58% page migration.
- Persistence mode: 3.2 s → 0.09 s, 30x variance reduction.
- C2C asymmetry: H2D 375 GB/s, D2H 297 GB/s.
- HBM3 STREAM 3.4 TB/s, LPDDR5X STREAM 486 GB/s.
- Multi-node GH200 NCCL ≥2.20 regression: −25% on 8 nodes vs 2.19.4; **unresolved in 2.30.4**.
- Lambda 1-Click GH200 multi-node fabric: NDR400 InfiniBand (Quantum-2).

### Medium confidence
- MPS p99 on GH200 specifically — extrapolated from H100 (Databricks) + general GPU (Orion).
- vLLM NUMA insensitivity on GH200 — single user-reported measurement, not a controlled benchmark.

### Low confidence / still open
- `init_on_alloc=0` isolated numeric delta — no public A/B benchmark.
- `--lock-gpu-clocks` GH200 variance reduction in *milliseconds* of latency, not just consistency.
- NCCL 2.30.4 single-node GH200 actual numbers vs 2.19.4 — no published comparison; only the multi-node regression is documented.
- Lambda GH200 NICs *per node* (1 or 2 NDR400 ports?) — Lambda doesn't publish.

### Round 1 errors corrected
- Persistence cold-start cost understated by ~2x.
- Lambda fabric mis-attributed to BlueField-3/RoCE; it's actually Quantum-2 NDR400 IB.
- NUMA pinning oversold for inference workloads.
- MPS tail-latency cost omitted entirely.

---

## 8. Sources

1. [Phoronix — 64K Kernel Page Size Benefits HPC on GH200 (Feb 2024)](https://www.phoronix.com/review/aarch64-64k-kernel-perf)
2. [arXiv 2407.07850 — Harnessing Integrated CPU-GPU System Memory on Grace Hopper](https://arxiv.org/html/2407.07850v1)
3. [arXiv 2408.11556 — Understanding Data Movement in Tightly Coupled Heterogeneous Systems](https://arxiv.org/html/2408.11556v2)
4. [HPI Werner et al. — Memory Disaggregation via NVLink C2C](https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/rabl/publications/2025/hcds25-Werner-GraceHopper.pdf)
5. [Verda — Data Movement in NVIDIA's Superchip Era](https://verda.com/blog/data-movement-in-the-nvidia-gh200-grace-hopper-superchip)
6. [Chips and Cheese — Grace Hopper, Nvidia's Halfway APU](https://chipsandcheese.com/p/grace-hopper-nvidias-halfway-apu)
7. [Abhik Sarkar — NVIDIA Persistence Daemon (measured)](https://www.abhik.ai/concepts/gpu/nvidia-persistence-daemon)
8. [StandardKernel — High-Fidelity GPU Kernel Benchmarking](https://standardkernel.com/blog/in-pursuit-of-high-fidelity-gpu-kernel-benchmarking/)
9. [NVIDIA — SetStablePowerState API blog](https://developer.nvidia.com/blog/advanced-api-performance-setstablepowerstate/)
10. [Databricks — Scaling Small LLMs with NVIDIA MPS (Jan 2026)](https://www.databricks.com/blog/scaling-small-llms-nvidia-mps)
11. [Orion — Interference-aware GPU sharing (EuroSys 2024)](https://anakli.inf.ethz.ch/papers/orion_eurosys24.pdf)
12. [Predictable LLM Serving on GPU Clusters (arXiv 2508.20274)](https://arxiv.org/html/2508.20274v1)
13. [NCCL GitHub releases](https://github.com/NVIDIA/nccl/releases)
14. [NCCL 2.30.3 release notes (NVIDIA docs)](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-30-3.html)
15. [NCCL issue #1272 — GH200 8-node regression ≥ 2.20](https://github.com/NVIDIA/nccl/issues/1272)
16. [NCCL issue #1291 — 2-node GH200 over Quantum-2 stuck](https://github.com/NVIDIA/nccl/issues/1291)
17. [Ed Sealing — Multi-Node GH200 NCCL Testing (on-prem BlueField-3 testbed)](https://medium.com/@ed.sealing/multi-node-gh200-nccl-testing-dc2fc64d97a0)
18. [Lambda 1-Click Clusters](https://lambda.ai/1-click-clusters)
19. [Lambda Cloud Clusters now available with GH200 (announcement)](https://lambda.ai/blog/lambda-cloud-clusters-now-available-with-nvidia-gh200-grace-hopper-superchip)
20. [Lambda on-demand docs](https://docs.lambda.ai/public-cloud/on-demand/)
21. [vLLM forum — performance insensitive to CPU binding on GH200](https://discuss.vllm.ai/t/vllm-performance-insensitive-to-cpu-binding/1662)
22. [Baseten — Llama 3.3 70B on GH200 Lambda Cloud](https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/)
23. [NVIDIA Grace OS Settings](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/os-settings.html)
24. [NVIDIA MGX GH200 reference (BlueField-3 *or* ConnectX-7)](https://aiserver.eu/product/nvidia-mgx-gh200/)
