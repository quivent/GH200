# GH200 Inference — Executive Decision Brief

**Compiled:** 2026-05-25 by the main orchestrator from Rounds 1 + 2 (Round 3 verification still running).
**Audience:** You, deciding what to do this week with your Lambda GH200.
**Companion docs:** `round2/H_recommendations_draft.md` (full recipe), `round2/A_software_stack_playbook.md` (install script), `round2/D_hardware_decision_matrix.md` (alternatives), `round2/E_pricing_leaderboard.md` (rentals), `round2/B_fp8_forensic_audit.md` (FP8 truth).

---

## The headline

**Keep the Lambda GH200. Upgrade three things. Run one cheap experiment.**

You are on the right platform for at least one workload class no alternative serves better today (Wan2.2-A14B 720p + Flux.2-dev 32B at bf16, single-GPU, no offload). For everything else, GH200 is *fine* — not best — but the switching cost dominates the per-token cost savings until utilization is high and committed.

---

## What to do this week

### 1. Upgrade the stack (4 hours, no rental cost — runs on your existing instance)

From Lambda Stack default → recommended:

```text
DRIVER + CUDA — two valid combinations, pick by horizon:
  CONSERVATIVE (recommended for production today, June-Aug 2026):
    nvidia-open-580 (580.159.04 head, ubuntu2404/sbsa)  +  CUDA 12.8
    R580 LTSB is capped at CUDA 12.x per NVIDIA matrix; active-support EOL = 2026-08-04 (10 weeks)
    cu128 matches Lambda's PyTorch 2.7-ARM wheel; PyTorch up to 2.11 supports cu128.
  FORWARD (required if you need CUDA 13.x grouped-GEMM / FA4 paths):
    nvidia-open-595 (595.71.05)  +  CUDA 13.0.3 or 13.2.1  (both live in ubuntu2404/sbsa)
    R595 is current Production Branch (released 2026-03-24). PyTorch 2.12+ requires cu130.
    R595 Hopper VBIOS gotcha: GH200 subrevision=3 fails to init with VBIOS < 96.00.68.00.xx — verify first.
NCCL  2.26.2 → 2.30.4-1  (released 2026-04-22)
       Multi-node ≥8 GH200s: also set NCCL_NCHANNELS_PER_NET_PEER=4 (closes issue #1272).
KERNEL  4K → 64K  (linux-nvidia-64k-hwe-24.04-edge, 6.17.0-1018.18 from 2026-05-19)
       MANDATORY — 4K can corrupt kernel memory + breaks GDS on GH200.
DOCA  2.9.3 → 3.2.0  ← MUST happen BEFORE 64K kernel install or apt deadlocks.
GH200 NVL2 only: VBIOS 96.00.D8.00.03 (fix for open-gpu-kernel-modules #774; Lambda Cloud unaffected).
```

**Ubuntu 26.04 trap (verified R3 #8):** `ubuntu2604/sbsa/` is populated but **only with R595 (not R580 LTSB)**. Lambda Stack still 20/22/24 only. Don't migrate until ≥Q4 2026.

**Install simplification (R3 #6, since March 2026):**
- PyTorch aarch64 cu130 + cu128 wheels on PyPI — `pip install torch` Just Works on GH200.
- FA3 prebuilt aarch64 wheels at `download.pytorch.org/whl/flash-attn-3/` (no more build-from-source).
- FlashInfer SBSA wheels in nightly since 2026-05-18 (cu128 aarch64 only; cu130 still missing).
- `pip install torch flash-attn-3 flashinfer-python` with the right index URL = full stack.

Plus the 5 GRUB boot params: `init_on_alloc=0 iommu.passthrough=1 numa_balancing=disable transparent_hugepage=madvise default_hugepagesz=2M hugepagesz=2M`. Full 4-stage stateful script (handles 2 reboots) in `round2/C_os_deploy_guide.md` §3 (`bootstrap_gh200.sh`).

**Ubuntu 26.04 trap (new):** NVIDIA's `ubuntu2604/sbsa` apt repo *is now live* (May 22) but **only carries driver 595.x — no R580 path**. If you migrate to 26.04, you leave the LTS R580 branch behind. Don't.

Then pick the engine matching your workload (in `round2/H_recommendations_draft.md` §2.4 table). For the workloads I expect you actually run: vLLM V1 ≥0.19.1 for dense LLMs, llama.cpp + `--n-cpu-moe` for MoE ≥100B, diffusers + FA3 + `torch.compile` for Wan/Flux. **TGI is archived. NIM is still amd64-only for headline LLMs — skip both.**

### 2. Update the FP8 mental model (revised statement below)

Your existing memory `feedback_no_fp8.md` ("FP8 broken on GH200/Wan; always bf16") is *under-specified*, not wrong. Three Round-1 agents and the dedicated forensic audit (Round 2 B) converged on this distinction:

| Path | Status May 2026 | Source |
|---|---|---|
| FP8 on DiT video (Wan, HunyuanVideo) | **Still broken everywhere** (H100, GH200, B200). Color shift, motion artifacts, LPIPS regression. **bf16 forever** for now. | vllm-omni #2728 (not covered by PR #2795); DiffSynth #466; SageAttention #221; ComfyUI-WanVideoWrapper #1554. |
| FP8 on DiT image (Flux, Z-Image, Qwen-Image) | **Fixed 2026-04-15** (vllm-omni PR #2795). LPIPS Flux 0.93 → 0.12. Worth retesting on your data with a regression gate. | vllm-omni PR #2795 merged 2026-04-15. |
| FP8 on dense Hopper LLMs (Llama-3.3-70B, Qwen2.5-72B) | **Production-grade post-vLLM 0.19.1** (FA3 FP32-accumulation fix). 128k needle: 13% → 89% (bf16 baseline 91%). Caveats: head_dim=256 1.6× prefill penalty; sliding-window needs `--kv-cache-dtype-skip-layers sliding_window`. | vLLM blog 2026-04-22; MLPerf v5.0/v5.1 winners. |
| FP8 on MoE LLMs (DeepSeek, Mixtral, Qwen3-Next) | **Fragile** — multiple open bugs across vLLM/SGLang/TRT-LLM. Use AWQ or bf16. | sglang #24650/#18550/#23687; TRT-LLM #12230. |
| FP8 on AMD MI300X | **Slower than bf16 on MoE.** Not the escape hatch from your Hopper FP8 pain. | vLLM #31475 open; pytorch #143465. |

→ **Net rule:** bf16 default everywhere; FP8 *only* with a per-model needle/LPIPS regression eval and only on the paths marked production-grade above. **Never FP8 on Wan.** (See `round2/B_fp8_forensic_audit.md` for the memory-update wording.)

### 3. Spend ~$74 on measurement to close the 6 highest-impact gaps

Agent G's Round-2 roadmap identified that **zero public GH200-specific benchmarks exist for any headline workload.** All defensible numbers in Rounds 1-2 are extrapolated from H100/H200 + one Baseten/Lambda 2025 TRT-LLM-FP8 datapoint. A small measurement budget gets you actual numbers:

| Gap | Experiment | Hardware | Cost | Closes |
|---|---|---|---|---|
| Wan2.2-A14B 720p 5-s timing on GH200 bf16 | One 1-hour benchmark run, 5 seeds | Lambda GH200 1h | ~$2.29 | The "stay on GH200" load-bearing claim |
| Llama-3.3-70B BF16 vLLM V1 tok/s on GH200 | `benchmark_serving.py` at conc=1/16/64/256 | Lambda GH200 2h | ~$5 | Multiple Round-1 reports flagged this as missing |
| FP8 head_dim=128 vs bf16 on dense Llama-70B | Run the vLLM 0.19.1 fix in regression eval | Lambda GH200 2h | ~$5 | Validates the memory-update text above |
| H200 baseline for Wan2.2 — does it fit? VRAM check at 96/141GB | Live test on H200 single-GPU | Sesterce H200 1h | ~$3 | Whether GH200's Wan moat is real |
| Vultr GH200 vs Lambda GH200 — same kernel + driver behavior | Provision Vultr; install upgraded stack; rerun Llama-70B | Vultr GH200 4h | ~$8 | Whether to migrate cheap |
| llama.cpp `--n-cpu-moe` on DeepSeek-V3.1 671B Q4 — verify 18 t/s | Single benchmark | Lambda GH200 2h | ~$5 | Confirms the MoE/Grace LPDDR play |
| Total (rough) | | | **~$28-74** | Top-6 gaps closed |

(Agent G's roadmap has the exact recipe per row in `round2/G_gap_verification_roadmap.md`.)

### 4. (Optional, parallel) Five-vendor sales-quote queue

These can't be answered by the web; one email each:

1. **Lambda** — quote for 1× GH200 reserved 12 / 24 / 36 months. Aggregator suggests ~$1.49/hr but unverified.
2. **CoreWeave** — GH200 reserved monthly tiers (only Platinum-tier cloud with public GH200 SKU).
3. **Nebius** — confirm H200 preemptible $1.45/hr + committed $2.30/hr, ask Llama-3.3 throughput on their HGX H200.
4. **Oracle OCI** — UCC tier discount on H200 bare-metal (list $10/hr; 25-60% off with commit).
5. **Lambda support** — Ubuntu 26.04 image roadmap + when driver 580 will land in Lambda Stack apt repo.

---

## When to actually switch off GH200

| Trigger | Switch to | Why |
|---|---|---|
| Your workload is dense Llama-70B-class at >50% sustained utilization | **DeepInfra Llama-3.3-70B-Turbo at $0.10/$0.32** | Beats every single-GH200 self-host at full util. (Agent E pricing leaderboard.) |
| Your workload is DeepSeek-V3 / Kimi-K2 671B+ MoE serving | **GB200 NVL72 (CoreWeave/Oracle/Azure) — $0.10/Mtok** vs your ~$1.56 on GH200 | The 15× $/Mtok gap is real per InferenceMAX. |
| Your workload is interactive low-batch chat with sub-200ms TTFT priority | **Groq API** (~180ms TTFT P50, $0.05/Mtok 8B class) | Verified May 2026. Skip Cerebras unless throughput >> TTFT (1.5-3s TTFT). |
| You need transparent monthly pricing and clean self-serve H200 | **Nebius $2.30/hr committed (~$1,656/mo)** | Cleanest published tier; ClusterMAX 2.0 Gold. |
| You want the cheapest hyperscaler reserved Hopper | **AWS p5 3-yr RI at $2,170/GPU/mo** (R3 #19 verified) | GCP a3-high 3-yr CUD is actually $3,153/mo — the $1,292 floor in earlier R1/R2 was a misread of the OD/CUD column. |
| You only need bigger HBM than 96GB on a single chip | **H200 SXM 141GB** | Standard HGX, no ARM coupling, often cheaper $/Mtok. |
| Memory-bound + memory-capacity workload with reserved commit OK | **AMD MI300X / MI325X (Vultr 24-36mo)** | But: don't expect FP8 to fix your pain; ROCm has its own FP8 bugs. |

**Do not switch to**: TGI (archived), NIM (no aarch64 for headline LLMs), Apple Mac Studio (512GB SKU pulled from sale March 2026), Etched (vapor), Intel Gaudi 3 (sunsetting product), TPU v7 (reservation-gated, no public 70B numbers).

---

## Round 3 status

20 verification + deepening agents launched against Round 1 + 2 outputs. Their results will:
- Live-verify all pricing claims via WebFetch of provider pages today.
- Validate or refute the "Wan FP8 fails on H100 too" hypothesis.
- Check whether driver 580.159.03 is actually in NVIDIA's `ubuntu2404/sbsa` apt repo today.
- Resolve the Cerebras/Groq/d-Matrix latency contradictions.

I will fold their findings into a final `synthesis/RECOMMENDATIONS.md` once they complete.
