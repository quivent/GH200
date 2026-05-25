# GH200 Inference Research Dossier

**Started:** 2026-05-25
**Goal:** Identify optimal inference pathways for GH200 on Lambda's Ubuntu 24 stack, evaluate Ubuntu 26 improvements, survey alternative hardware platforms, and price monthly GPU rentals.

**Constraint carried from prior experience:** FP8 is broken/unreliable on the tested GH200 + Wan stack. Default recommendation is bf16 unless an agent finds solid May-2026 evidence of a fix.

## Iteration plan

- **Round 1 — initial sweep** (in progress): 20 agents in parallel, each domain-focused. Each writes a markdown report to `round1/NN_<topic>.md` with versions, configs, benchmark numbers, source URLs with dates, and a "Confidence & Gaps" section.
- **Round 2 — cross-pollination**: Each Round-2 agent reads the relevant Round-1 reports (peers + adjacent domains), reconciles contradictions, fills gaps that peers couldn't, and rewrites a sharper report to `round2/`.
- **Round 3 — adversarial validation**: A smaller set of agents stress-test the strongest claims, verify pricing is current, and check that recommended commands/versions actually exist.
- **Synthesis**: Single dossier in `synthesis/RECOMMENDATIONS.md` with: top inference pathway for GH200 today, top Ubuntu-26 changes worth waiting for, ranked alternative platforms, and a monthly-rental price table.

## Round 1 agent assignments

| # | Domain | Output file |
|---|--------|-------------|
| 01 | NVIDIA driver / CUDA / cuDNN / NCCL stack on Ubuntu 24 + GH200 | `round1/01_driver_stack.md` |
| 02 | TensorRT-LLM on GH200 (aarch64, engine builds, quant paths) | `round1/02_tensorrt_llm.md` |
| 03 | vLLM on GH200 (V1 engine, FA3, paged attn, prefix cache) | `round1/03_vllm.md` |
| 04 | SGLang on GH200 (RadixAttention, P/D split) | `round1/04_sglang.md` |
| 05 | HF TGI + llama.cpp on GH200 (Grace ARM hybrid CPU/GPU) | `round1/05_tgi_llamacpp.md` |
| 06 | PyTorch 2.x + torch.compile + Triton + FA3 (LLM & diffusion) | `round1/06_pytorch_triton.md` |
| 07 | NIM / Triton Server / Dynamo / MAX / Ray Serve | `round1/07_serving_frameworks.md` |
| 08 | Ubuntu 26.04 vs 24.04 for GH200 + RHEL/Rocky/SUSE/NixOS | `round1/08_ubuntu_26_vs_24.md` |
| 09 | Kernel/OS tuning (NUMA, hugepages, NCCL, GDS, MPS, IRQ) | `round1/09_os_kernel_tuning.md` |
| 10 | H100 / H200 as GH200 alternatives | `round1/10_h100_h200_alternatives.md` |
| 11 | Blackwell B100/B200/B300 + GB200/GB300 NVL availability | `round1/11_blackwell.md` |
| 12 | AMD MI300X / MI325X / MI355X (ROCm) | `round1/12_amd_mi300_etc.md` |
| 13 | Apple Silicon (M3/M4 Ultra, MLX) | `round1/13_apple_silicon.md` |
| 14 | Specialty inference chips (Groq, Cerebras, SambaNova, etc.) | `round1/14_specialty_chips.md` |
| 15 | Intel Gaudi 3 + Google TPU v5/v6/v7 | `round1/15_gaudi_tpu.md` |
| 16 | Lambda Labs monthly/reserved pricing + image versions | `round1/16_lambda_pricing.md` |
| 17 | RunPod / Vast.ai / TensorDock / Hyperbolic / marketplace | `round1/17_marketplace_gpu_clouds.md` |
| 18 | CoreWeave / Crusoe / Together / Nebius / OCI tier-1 | `round1/18_tier1_specialty_clouds.md` |
| 19 | AWS / GCP / Azure hyperscaler reserved + managed endpoints | `round1/19_hyperscaler_pricing.md` |
| 20 | Diffusion image/video on GH200 (Flux, Wan, Hunyuan, etc.) | `round1/20_diffusion_video.md` |

## Status

- 2026-05-25 02:05 — Round 1 dispatched, 20 background agents.
- 2026-05-25 ~02:15 — All 20 Round-1 reports landed in `round1/`.
- 2026-05-25 ~02:20 — Round 2 dispatched (8 cross-pollination agents).
- 2026-05-25 ~02:35 — Round 2: A, B, C, D, E, G, H landed (7/8); F (diffusion deep-dive) still running.
- 2026-05-25 ~02:25 — Round 3 dispatched (20 verification + deepening agents reading R1 + R2). All running.
- 2026-05-25 ~02:40 — `synthesis/EXEC_BRIEF.md` first draft written; updated with R2-C corrections.
- 2026-05-25 ~03:00 — Round 3 complete (20/20). Big corrections to driver/CUDA strategy (R580 EOL Aug 4 2026; cap at CUDA 12.x), pricing (GCP H100 floor was a misread — real AWS p5 3-yr at $2,170/mo is the hyperscaler-Hopper floor), Flux.2-dev VRAM (doesn't fit on GH200 96GB no-offload after all), Lambda fabric (NDR400 IB, not BlueField-3), llama.cpp config (validated combos are A or B, not both).
- 2026-05-25 ~03:05 — Round 4 dispatched: 20 LLM-only agents iterating on R1-R3.
- 2026-05-25 ~03:20 — Round 4 complete (20/20).
- 2026-05-25 ~03:25 — Round 5 dispatched: 15 narrow-deep agents on Llama-3.3-70B Q4 + Qwen3.6-27B Q4 + peers, per user request.
- 2026-05-25 ~04:15 — Round 5 complete (15/15). All 83 reports landed.
- 2026-05-25 ~04:20 — `synthesis/MODEL_PICK_BRIEF.md` written — final substitution proposal: drop Llama-3.3-70B → Qwen3.6-27B + GPT-OSS-120B.
- Repository: `git log --oneline` shows 6 commits to date.

## Headline findings from Round 1 (pre-Round-2)

1. **Target stack**: Driver R580 (`580.159.03`) + CUDA 13.0.2 + NCCL 2.30.4 + 64K-page kernel (`linux-nvidia-64k-hwe-24.04-edge`) + open kernel modules (mandatory on Hopper+). Lambda Stack lags at CUDA 12.8.93 / NCCL 2.26.2 — manual upgrade needed.
2. **Stay on Ubuntu 24.04**. Ubuntu 26.04 has no NVIDIA support tables, no Lambda image, Python 3.14 wheel desert.
3. **FP8 reframe** (high-impact update to user memory): the "FP8 broken on Wan" issue is a **DiT/diffusion-transformer FP8 problem reproducible on H100 and B200**, not a GH200 hardware problem. Multiple sources (vllm-omni #2728, DiffSynth #466, agent 01/02/06/20 all corroborate). The vLLM dense-LLM FA3 Hopper FP8 bug *was* GH200-relevant and *was* fixed in v0.19.1 (with 1.6× prefill penalty on head_dim=256).
4. **Best engines on GH200 aarch64 today**: vLLM V1 (mainline 0.19, install-clean since PyTorch 2.11 aarch64 wheels), SGLang ≥0.5.12 (official aarch64 wheels, real arm64 Docker), TRT-LLM 1.2.0 (NGC container path, bare metal SBSA still unsupported), llama.cpp (CUDA-13 arm64 tarballs, Grace unified-memory unlock for >100B MoE).
5. **NIM unfit for GH200** (still `linux/amd64`-only for headline LLMs). NVIDIA Dynamo and Triton SBSA are the cleanest serving frameworks.
6. **Hardware alternative ranking**: H200 SXM 141GB is the cleanest GH200 swap (more HBM, no ARM, standard HGX). B200 = ~3× $/tok dense, GB200 NVL72 = ~10-15× $/tok on DeepSeek MoE — but FP4/FP8 maturity caveats. AMD MI300X/325X wins memory-bound + reserved-pricing only; FP8 path has its own bugs. Apple M3 Ultra 512GB: pulled from sale. Specialty chips: Cerebras sold-out, Groq best for low-batch interactive, Etched still vapor. Gaudi sunset, TPU v6e = real GH200 alt if JAX/GCP OK.
7. **Pricing floor for GH200 monthly**: Vultr **$1.99/hr ≈ $1,433/mo** (cheapest), Lambda $2.29/hr ≈ $1,672/mo, CoreWeave $6.50/hr. Tier-1 has deprioritized GH200; CoreWeave is the only Platinum-tier with public GH200 SKU. Hyperscalers don't sell GH200 self-serve.
8. **Cheapest H200 monthly self-serve**: Nebius preemptible $1.45/hr (~$1,044/mo), committed $2.30/hr (~$1,656/mo). Wins on transparency.
9. **Cheapest H100 reserved**: GCP a3-high 3-yr CUD ~$1,292/mo (hyperscaler floor).
10. **Diffusion stack**: bf16 + FA3 + `torch.compile(max-autotune-no-cudagraphs)` + step-distill LoRA + MagCache/TeaCache. Wan2.2-A14B 720p 5-s estimated 25-45s on GH200; only fits without offload on GH200 (96GB), *not* H100 (80GB) — a real GH200 win.
