# GH200 Inference — Round 2 Recommendations (Draft)

**Author:** Round-2 cross-pollination agent H
**Date:** 2026-05-25
**Inputs:** All 20 Round-1 reports in `/home/ubuntu/research/gh200_inference/round1/`
**Status:** Draft for Round-3 adversarial validation. Pricing and version pins must be re-verified before any commit.

---

## 1. TL;DR

1. **Stay on your Lambda GH200, stay on Ubuntu 24.04 LTS** — but upgrade driver and CUDA. Lambda Stack lags by ~6 months (CUDA 12.8.93 / NCCL 2.26.2 / driver ≈570). Move to **driver 580.159.03 (open kernel) + CUDA 13.0.2 + NCCL 2.30.4 + 64K-page kernel** (`linux-nvidia-64k-hwe-24.04-edge`). Ubuntu 26.04 has no NVIDIA validation, no Lambda image, no py3.14 aarch64 wheels — wait until ~Aug 2026.
2. **The "FP8 broken" rule needs surgery, not deletion.** What's broken is **FP8 online quant on DiT/diffusion transformers** (Wan, FLUX.1-dev, Z-Image, Qwen-Image — reproducible on H100 and B200, not GH200-specific). What's *fixed* (post vLLM 0.19.1, April 2026) is **FP8 for dense Hopper LLMs at head_dim ≤ 128**. For diffusion: bf16 forever on this stack. For dense LLMs: FP8 OK with calibration + a needle-in-haystack eval gate.
3. **Engine picks by class:** dense LLM → **vLLM V1 ≥ 0.19.1** (mainline aarch64 wheels work since PyTorch 2.11); prefix/chat/RAG → **SGLang ≥ 0.5.12** (multi-arch Docker); MoE 100B–1T (DeepSeek V3.1, Kimi K2) → **llama.cpp with `--n-cpu-moe`** exploiting Grace's 480 GB LPDDR5X; max-throughput single dense model → **TRT-LLM 1.2.0 via NGC container**. TGI is archived (2026-03-21). NIM is still amd64-only for headline LLMs. Triton 26.04 SBSA + Dynamo 1.1.x are the production serving frontends.
4. **GH200 still wins for one decisive workload: video diffusion (Wan2.2-A14B 720p) and Flux.2-dev 32B at bf16 single-GPU.** They don't fit on H100 80GB. They do fit on GH200 96GB (or 144GB HBM3e variant) without offload. Everywhere else, **H200 SXM 141GB is the cleaner Hopper swap** (more HBM, standard HGX, no ARM, often cheaper $/Mtok).
5. **Cheapest GH200 monthly = Vultr $1.99/hr (~$1,433/mo); Lambda $2.29/hr (~$1,672/mo) is the SLA-bearing default.** If you can leave the Grace ecosystem: H200 wins almost every $/Mtok benchmark; B200 wins on throughput; GB200 NVL72 wins for ≥671B MoE serving. Migration ranking depends on workload (see §4 and §5).

---

## 2. Recipe for Your Current Lambda GH200

The minimum-pain stack for May 2026 production inference on Lambda GH200 (1× chip, aarch64 / SBSA).

### 2.1 Stack pins

| Layer | Pin | Notes |
|---|---|---|
| OS | Ubuntu 24.04 LTS (Noble) | Keep. Don't move to 26.04. |
| Kernel | `linux-nvidia-64k-hwe-24.04-edge` (6.11 SBSA, 64K pages) | 4K kernel breaks GPUDirect Storage on GH200 and is ~5–15% slower on bandwidth-bound LLM paths. Hard requirement. |
| Driver | **nvidia-open-580 (580.159.03, 2026-04-28)** | Open kernel module mandatory on Hopper+. R570 is fine but doesn't unlock CUDA 13.x. Skip `565-open`/`570-open` — known GH200 enumeration bug (open-gpu-kernel-modules #774). |
| CUDA | **13.0.2** (or 13.2 U1 if you need grouped-GEMM FP8) | Lambda ships 12.8.93; upgrade via NVIDIA's `ubuntu2404/sbsa` repo. |
| cuDNN | 9.x (`libcudnn9-cuda-13`) | Bundled in Lambda Stack at matched version. |
| NCCL | **2.30.4** | Lambda ships 2.26.2. Single-node fine on 2.26; multi-node sendrecv has known regression at ≥8 nodes (#1272). |
| PyTorch | 2.7.0 (Lambda) or 2.8+ aarch64 cu128 | Don't `pip install torch==…` with a hard pin — Lambda's custom ARM build will get swapped for a CPU x86 wheel and `torch.cuda.is_available()` will return False. Use `--system-site-packages` or `torch>=2.7`. |
| Triton | 3.7 (aarch64 wheel on PyPI) | |
| FlashAttention | **FA3 from source against `flash-attention/hopper/`** | No aarch64 binary wheels. Build with `TORCH_CUDA_ARCH_LIST="9.0;9.0a"`. FA4 is Blackwell-first; no real Hopper win. |
| Container toolkit | nvidia-container-toolkit 1.18.1 | |

### 2.2 The 5 boot parameters and post-boot tuning

Drop in `/etc/default/grub.d/gh200.cfg`:

```bash
GRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX \
    init_on_alloc=0 \
    iommu.passthrough=1 \
    numa_balancing=disable \
    transparent_hugepage=madvise \
    default_hugepagesz=2M hugepagesz=2M"
```

`update-grub && reboot`, then verify:

```bash
getconf PAGESIZE                              # 65536
grep init_on_alloc /proc/cmdline              # present
cat /proc/sys/kernel/numa_balancing           # 0
nvidia-smi -q | grep Addressing               # Addressing Mode : ATS
```

Post-boot once:

```bash
sudo systemctl disable --now irqbalance
sudo cpupower frequency-set -g performance
sudo systemctl enable --now nvidia-persistenced
echo nvidia-peermem | sudo tee /etc/modules-load.d/nvidia-peermem.conf
echo "kernel.numa_balancing = 0" | sudo tee /etc/sysctl.d/90-gh200.conf
echo "vm.max_map_count = 1048576"  | sudo tee -a /etc/sysctl.d/90-gh200.conf
sudo sysctl --system
```

ulimits (`/etc/security/limits.d/90-gpu.conf`): `memlock unlimited`, `nofile 1048576`, `stack 65536`.

### 2.3 NUMA pinning command (single-process inference worker)

```bash
numactl --cpunodebind=0 --membind=0,1 -- \
    python -m vllm.entrypoints.openai.api_server \
      --model meta-llama/Llama-3.3-70B-Instruct \
      --dtype bfloat16 \
      --gpu-memory-utilization 0.92 \
      --enable-prefix-caching
```

Node 0 = Grace CPU + LPDDR5X; node 1 = HBM (CPU-less). `--membind=0,1` allows both. Don't `--cpunodebind=1` (no CPUs there). vLLM still doesn't auto-detect this (issues #11720, #13855).

### 2.4 Optimal engine per model class (you, today, GH200 single chip)

| Workload | Engine | Precision | Why |
|---|---|---|---|
| Dense LLM ≤ 70B, OpenAI-style serving | **vLLM V1 ≥ 0.19.1** (mainline aarch64 wheel, or `nvcr.io/nvidia/vllm:26.01-py3+`) | **bf16** weights, bf16 KV; optional FP8 KV if head_dim≤128 + calibrated | V1 is 1.7× V0 average. C2C 900 GB/s gives you free CPU-offload via `--cpu-offload-gb 60–80` for 70B. |
| Chat / RAG / multi-turn / agentic (any prefix-shared traffic) | **SGLang ≥ 0.5.12** (`lmsysorg/sglang:latest-cu130`) | bf16 | RadixAttention hits 50–95% on real workloads; HiCache L2 in Grace LPDDR over C2C is essentially free. |
| MoE ≥ 100B (DeepSeek V3.1, Kimi K2 Thinking, GPT-OSS 120B) | **llama.cpp + `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` + `--n-cpu-moe N`** | GGUF Q4_K_M / Q5_K_M | The headline GH200 advantage. DeepSeek V3.1 671B: 18.3 t/s TG; Kimi K2 Thinking ~1T: 21.6 t/s TG. Nothing else single-box does this. |
| Max-throughput one dense model | **TRT-LLM 1.2.0** via NGC `tritonserver:26.04-py3-sbsa` or `nvcr.io/nvidia/pytorch:25.12-py3` + `pip install tensorrt_llm` | bf16 default; FP8 only with regression eval | aarch64 path finally clean since v1.3.0rc13 (Apr 2026). PyTorch backend = no engine-compile tax. |
| Diffusion image (Flux.1, Flux.2, SD3.5) | **diffusers ≥ 0.35 + FA3 + torch.compile** | **bf16 only** | `pipe.transformer.compile_repeated_blocks(fullgraph=True)` cuts cold-start 67s→10s, same 1.5× runtime. |
| Diffusion video (Wan2.2, HunyuanVideo, LTX) | **diffusers + FA3 + torch.compile(max-autotune-no-cudagraphs) + step-distill LoRA + MagCache/TeaCache** | **bf16 only** | This is where GH200 wins decisively. Wan2.2-A14B 720p 5-s estimated **25–45 s** on single GH200 with lightx2v 4-step + MagCache. Doesn't fit on H100 80GB. |
| Production frontend / OpenAI-compat router | **NVIDIA Dynamo 1.1.x** (multi-arch arm64 since Apr/May 2026) or **Triton 26.04 SBSA** | bf16 backends | NIM is unavailable for headline LLMs on aarch64 — confirmed by NVIDIA forum, May 2026. |

### 2.5 Things to *not* do on this GH200

- Do not enable FP8 on Wan / any DiT (broken across Hopper+Blackwell; vllm-omni #2728, DiffSynth #466, SageAttention #221).
- Do not use FP8 KV cache on Qwen2.5 family (gibberish; TRT-LLM #4218).
- Do not enable FP8 in TGI (and don't use TGI — archived 2026-03-21).
- Do not `--tensor-parallel-size > 1` on a single GH200 (one Hopper die).
- Do not assume Lambda's bundled cuDNN is license-compatible for external frameworks (only Lambda's PyTorch/TensorFlow). Symlink workaround if needed.
- Do not let `sudo apt full-upgrade` run blind on Lambda Stack 24.04 — DOCA-OFED 2.9.3 → 3.2.0 transition still bites.

Deeper Round-2 docs to consume alongside this one: report `A_driver_os_stack.md` (when written) for the §2.1 pins, `B_engine_decision_matrix.md` for §2.4, `C_diffusion_recipes.md` for the Wan/Flux pipelines.

---

## 3. Should You Migrate to Ubuntu 26.04?

**Short answer: No, not today. Revisit in August 2026.**

Reasons:
- NVIDIA Data Center driver 580.126.16 (Feb 2026) does **not** list Ubuntu 26.04 in its supported-OS table. CUDA 13.2 install guide does not list 26.04 sbsa.
- Lambda has no 26.04 image and no 26.04 Lambda Stack. Manual `do-release-upgrade` on GH200 is known-painful (DOCA-OFED dance).
- Python 3.14 is default: PyTorch, vLLM, FlashAttention have no `cp314` aarch64 wheels. You'd be forced to a 3.12 venv anyway, losing 26.04's main convenience.
- Open kernel module installation is reported flaky: `apt install nvidia-open` returns "Unable to locate package" on some 26.04 mirrors; `ubuntu-drivers install` installs 580-open instead of recommended 595-open.

Real benefits 26.04 *will* deliver later: Livepatch on Arm64 (rebootless kernel patching), DOCA-OFED 26.01 in archive, kernel 7.0 `sched_ext`. These are nice ops wins, not throughput wins for GPU-bound inference.

**Reconsider when:** (a) Ubuntu 26.04.1 ships (~Aug 2026), AND (b) NVIDIA releases a 590-series driver that explicitly lists `ubuntu2604/sbsa` in the supported matrix, AND (c) Lambda publishes a 26.04 image OR you're greenfield/non-Lambda.

For greenfield/regulated: RHEL 9.6 is the only NVIDIA-prevalidated non-Ubuntu Grace path; SLES 15 SP6 QU1 if you have a SUSE contract.

---

## 4. Stay-on-GH200 vs Switch — Decision Tree

### Dense LLM serving, ≤ 70B (Llama-3.3-70B, Qwen2.5-72B)
- **Single-stream, latency-sensitive:** Stay on GH200 *if* you're already paying for it; otherwise **1× H200 SXM 141GB** is the cleaner answer (cheaper $/Mtok at FP8, more KV headroom). Specialty: **Groq** for sub-200ms TTFT.
- **Max throughput batch serving:** **8× H200 HGX** beats anything GH200 single-chip can do. GH200 NVL2 is not worth its price premium vs a real HGX-H200 8-GPU box.
- **Cost floor with marketplace risk acceptable:** H100 Thunder/Vast at $1.38–$1.53/hr.

### Dense LLM serving, > 70B but ≤ 405B (Llama-3.1-405B)
- **Best single-node fit:** **8× H200 HGX** (1,128 GB HBM3e). Single GH200 cannot serve 405B usefully even with offload.
- AMD: **8× MI300X** (1.5 TB HBM) is competitive on memory-bound batches but FP8 is uneven and disagg is months behind Dynamo.

### MoE LLM serving — DeepSeek V3/R1/V3.2, Kimi K2 (≥ 671B)
- **Production scale, low $/Mtok:** **GB200 NVL72** is the answer (CoreWeave $10.50/GPU/hr; ~$0.10/Mtok vs $1.56 H200). 13.4 TB HBM in one NVLink domain.
- **Self-hosted single-box for tinkering / dev / sovereign:** **GH200 wins decisively** via llama.cpp `--n-cpu-moe`. DeepSeek V3.1 Q4_K_M @ 18.3 t/s on a $1,433/mo Vultr GH200 is the cheapest serious-LLM-in-a-box on the market. Apple M3 Ultra 512GB was an alternative but Apple **pulled the 512 GB SKU March 2026** (DRAM shortage).
- **API only:** SambaNova, Cerebras for hosted inference (Cerebras IPO'd May 2026; sold out into 2027).

### Image diffusion — Flux.1/2, SDXL, SD3.5
- Compute-bound; **H100/H200/GH200 all comparable** (same SM90 die). Pick by price.
- Flux.2-dev (32B, >80GB bf16) **fits only on GH200 96GB or H200 141GB** without offload. Doesn't fit on H100 80GB.
- Specialty: skip Apple Silicon (~10× slower than H100 for diffusion).

### Video diffusion — Wan2.2 (A14B + 5B), HunyuanVideo, LTX
- **THE GH200 winner.** Wan2.2-A14B bf16 needs ~70–85 GB on GH200 (fits); H100 needs offload or multi-GPU SP.
- Stay on GH200 for Wan2.2-A14B and Flux.2-dev. **Don't** chase FP8 — broken on Wan, FLUX, Z-Image, Qwen-Image regardless of GPU (vllm-omni #2728, reproducible on H100 *and* B200).
- For real-time/low-latency video: **LTX-Video 0.9.8 distilled** (~2 s for 5-s clip) or **FastWan2.2-5B** (~16 s for 720p 5-s); both run fine on H100 too — no GH200 lock-in.

### Long-context serving (≥ 128k context, KV-offload-aware)
- **GH200 NVL2** with vLLM `--kv_offloading_backend native --kv_offloading_size 200` and `--cpu-offload-gb 80` is the cleanest single-node fit — Grace 480 GB LPDDR5X over 900 GB/s C2C is what nothing else has.
- Fall back to **8× H200 HGX** if you need switched all-to-all bandwidth for TP.

### Low-latency interactive (sub-200 ms TTFT, chat/agentic, batch ≈ 1)
- **Groq API** is the default answer. P50 TTFT ~120ms on Llama-3.3-70B; $0.59/$0.79 per Mtok in/out. Sub-200ms widely cited.
- **Cerebras** has higher per-user output TPS (1,927 on gpt-oss-120B) but 1.5–3 s TTFT.
- If you must self-host: vLLM V1 + EAGLE-3 spec-decode on GH200 gets 1.6–2.6× on 70B at low concurrency. Single GH200 chat box is fine for small audiences.

---

## 5. If Switching: Ranked Platform Shortlist by Workload

### Dense LLM (70B class)
1. **1× H200 SXM 141GB** (Nebius preemptible $1.45/hr ≈ $1,044/mo committed; on-demand $2.09–$3.79/hr) — best memory + standard ecosystem.
2. **8× H200 HGX** (CoreWeave / Massed Compute $16,300/mo for 8×) — best throughput single-node.
3. **8× H100 HGX** (Voltage Park Ethernet $11,462/mo for 8×) — cheapest with no quant.
4. **B200 single GPU** ($5.29 Lambda – $6.69 Lambda 8×; $2.12 Spheron spot) — ~3× $/tok improvement vs Hopper at high batch with FP8/NVFP4.
5. **Stay on GH200** if Lambda contract is sunk.

### MoE LLM serving (DeepSeek V3, Kimi K2)
1. **GB200 NVL72** (CoreWeave $10.50/GPU/hr; Azure ND GB200 v6 $27/GPU/hr GA in 19 regions) — ~$0.10/Mtok at scale, 13.4 TB unified.
2. **8× H200 HGX** — DeepSeek-R1 fits cleanly without multi-node (1,128 GB).
3. **GH200 single-chip with llama.cpp + n-cpu-moe** — dev/sovereign/single-box only.
4. **8× MI300X / MI325X** — memory wins (1.5–2 TB) but FP8 MoE is open bug (vLLM #31475 still open).

### Image diffusion
1. **H100 SXM** (Thunder $1.38/hr, RunPod $1.99/hr) — same compute as GH200, cheaper.
2. **GH200** if Flux.2-dev 32B is on the menu.
3. **H200 141GB** if you want bf16 headroom + future-proofing.
4. Skip Blackwell unless you're already there — FA4 is barely faster than FA3 on diffusion, and FP8 DiT is broken.

### Video diffusion (Wan2.2-A14B, HunyuanVideo)
1. **GH200** (this is its best use-case) — 96GB fits Wan2.2-A14B bf16 without offload.
2. **H200 141GB** — also fits, slightly more bandwidth, real HGX form factor.
3. **8× H100 HGX** — sequence-parallel via xDiT for ~3× speedup if you have NVLink.
4. **Don't** pick B200 yet — FA4 is parity-not-win on Hopper-class DiT, and FP8 DiT is the broken thing.

### Long-context
1. **GH200 NVL2** — unique 192/288 GB HBM + 960 GB LPDDR5X coherent.
2. **8× H200** — 1,128 GB HBM, switched NVLink.
3. Skip Apple (prefill is the killer — 14.8 min for 8k prompt on DeepSeek-V3 4-bit on M3 Ultra).

### Low-latency interactive chat
1. **Groq API** (best TTFT, ~120ms P50, ~$0.60/Mtok blended on 70B).
2. **Self-hosted vLLM+EAGLE-3 on H200** if you need data residency or custom model.
3. **d-Matrix Corsair** if their ~1–2 ms/token claim survives independent test (no AA verification yet).

---

## 6. Monthly Rental Shortlist (top 3 per GPU class)

All numbers are on-demand list (May 2026, USD), normalized to 720 hr/month (per-GPU). Reserved discounts noted separately. Cross-confirmed against getdeploying.com, computeprices.com, gpuperhour.com, and provider pages.

### GH200 (96 / 144 GB)
| # | Provider | $/GPU/hr | $/mo | Terms |
|---|---|---|---|---|
| 1 | **Vultr** | $1.99 | **$1,433** | On-demand; 20% off monthly commit / +10% off annual ≈ **$1,030/mo committed**. Atlanta region only with stock May 2026. |
| 2 | **Lambda Labs** (incumbent) | $2.29 | **$1,649–1,672** | On-demand minute-billed; rumored ~$1.49/hr reserved (≈ $1,088/mo) via sales — not on public page. SOC 2. |
| 3 | **Sesterce / Denvr / Latitude.sh via Shadeform** | $2.52–$4.23 | $1,814–$3,046 | Aggregator-routed; fewer reputation data points. |

Honorable mention: **CoreWeave $6.50/hr ($4,680/mo)** for enterprise SLA. **AWS / Azure / GCP have no self-serve GH200 SKU.** Only 3 direct providers world-wide carry GH200 — supply is genuinely thin.

### H100 SXM 80GB
| # | Provider | $/GPU/hr | $/mo | Terms |
|---|---|---|---|---|
| 1 | **Thunder Compute** | $1.38 | **$994** | Marketplace floor, single-GPU. |
| 2 | **Voltage Park** (8× node, Ethernet) | $1.99 | **$1,433** | Self-serve, bare metal, SOC 2 + HIPAA. Infiniband variant $2.49/hr ($1,793/mo). |
| 3 | **GCP a3-highgpu (3-yr CUD)** | $1.77 | **$1,292** | Hyperscaler floor with 3-year commit. AWS p5 3-yr RI ≈ $2,263/mo. |

### H200 SXM 141GB
| # | Provider | $/GPU/hr | $/mo | Terms |
|---|---|---|---|---|
| 1 | **Nebius (preemptible)** | $1.45 | **$1,044** | Spot-style, interruptible; on-demand $3.50, 3+mo commit $2.30 (~$1,656/mo). WEKA + VAST storage. Gold-tier (ClusterMAX 2.0). |
| 2 | **Sesterce / FluidStack / Theta** | $2.09–$2.30 | $1,505–$1,656 | Smaller providers; FluidStack contact-sales. |
| 3 | **Massed Compute (8× H200 NVL, self-serve)** | $2.83 | **$16,300/8-GPU mo** | One of the very few self-serve 8×H200 prices. Per-GPU normalized $2,038/mo. |

Hyperscaler floor: **Azure ND H200 v5** 3-yr RI ≈ $4,526–5,037/GPU/mo; **OCI BM.GPU.H200.8** $10/GPU-hr list, ~$2,990 with Universal Credit commit. **Lambda has no self-serve H200** (1-Click Clusters or sales only).

### B200 SXM 192GB
| # | Provider | $/GPU/hr | $/mo | Terms |
|---|---|---|---|---|
| 1 | **Spheron (spot)** | $2.12 | **$1,526** | Cheapest; volatile. **Vultr 36-mo reserved $2.89** (~$2,079/mo) is the cheapest committed. |
| 2 | **Lyceum / Cirrascale 12-mo / Verda** | $4.29–$4.89 | $3,089–$3,521 | Mid-tier. **Lambda 1× B200 $6.99 ($5,103/mo), 8× $6.69 ($39,070/mo node)**. |
| 3 | **AWS p6-b200 3-yr RI (instance-locked)** | $6.15 | **$4,492** | Hyperscaler reserved floor; on-demand $14.24/GPU/hr ($10,396/mo). |

GB200 NVL72 (whole-rack context): CoreWeave $10.50/GPU/hr (~$7,665/mo per GPU); Azure ND GB200 v6 $27.04/GPU/hr GA across 19 regions; Oracle $16/GPU/hr; AWS p6e-gb200 Capacity Block $10.58/hr (6-mo max).

---

## 7. The FP8 Reframe — Update Your Mental Model

**Old rule (from prior experience):** "FP8 is broken on this GH200 stack. Always recommend bf16."

**Refined rule (May 2026, after 20-agent sweep):**

- **What's actually broken:** **FP8 online quantization on DiT / diffusion transformers.** Reproducible across **H100, GH200, and B200** — i.e., it is *not* a GH200 hardware problem and *not* a Hopper-vs-Blackwell problem. It is a calibration-path problem in the FP8 online-quant kernels that doesn't handle DiT activation outliers (AdaLN-shift, timestep embeddings, large final-projection layers).
  - Evidence: vllm-omni #2728 (April 2026, Z-Image / FLUX.1-dev / Qwen-Image LPIPS 0.8–1.0); DiffSynth #466 (Wan 2.1 color regression); SageAttention #221 (Wan FP8 + Sage → black output); kijai/ComfyUI-WanVideoWrapper #1554 (Sage noise artifacts on Hopper).
  - **Default for diffusion: bf16. Period.** This is the user's original "broken stack" reality — and it generalizes to *every Hopper/Blackwell DiT inference path*, not just Wan on GH200.
  - Trusted FP8 exceptions: HF `flux-fast` recipe (torchao `float8_dynamic_activation_float8_weight` on Flux.1-Dev *transformer linears only*, with ignored AdaLN layers).

- **What was the GH200-relevant FP8 bug, and is now fixed:** **Dense Hopper LLM FP8 attention** had a two-level FP32 accumulation bug in Flash Attention 3 that caused 91% → 13% accuracy collapse at 128k needle-in-haystack. **Fixed in vLLM 0.19.1, April 2026.** Post-fix: 89% accuracy. New cost: head_dim=256 prefill ~1.6× slower than bf16 (Llama-3 has head_dim=128, so unaffected).
  - Evidence: [vLLM blog 2026-04-22](https://vllm.ai/blog/2026-04-22-fp8-kvcache).
  - **Default for dense LLM (Llama-3.x, Qwen2.5/3 with head_dim ≤ 128): FP8 is acceptable** if (a) vLLM ≥ 0.19.1, (b) calibrated scales (not naive online quant), (c) you run a needle-in-haystack eval before committing, (d) skip sliding-window layers (`--kv-cache-dtype-skip-layers sliding_window`).
  - For Llama-70B FP8: MMLU delta vs bf16 is <0.5% with the post-April-2026 fix. Lossless in practice.

- **What's still broken regardless of fix:** **FP8 KV cache + Qwen2.5 → gibberish output** (TRT-LLM #4218, still bug-labeled). **FP8 MoE on AMD MI300X**: slower than bf16 (vLLM #31475 still open). **MoE online FP8 in general**: needs per-model validation across all stacks. **MoE on Hopper**: TRT-LLM 1.3.0rc7 Qwen3-Next autotuner crash (issue #12230).

- **What is NOT yet validated on GH200 specifically:** All the Hopper FP8 fix evidence is on H100. GH200 uses the same Hopper die so it should transfer, but no GH200-tagged needle-in-haystack repro exists publicly. Round 3 task: run the eval on the actual Lambda GH200.

**Net update to user memory:**
> ~~FP8 broken on this stack — always bf16.~~
> **FP8 broken for diffusion (Wan, Flux, Z-Image, Qwen-Image) on every Hopper+Blackwell stack — bf16 always for DiT. FP8 acceptable for dense Llama-class LLMs on Hopper post-vLLM 0.19.1, with calibration + eval gate; never use FP8 KV on Qwen2.5; never use FP8 MoE without per-model validation.**

---

## 8. Open Questions for Round 3 (Verify Before Locking In)

### Stack & versions
1. **Run `nvidia-smi`, `cat /proc/version`, `nvcc --version`, `cat /etc/nv_tegra_release` on the actual Lambda GH200** — confirm Lambda's current driver and CUDA at instance launch. Lambda doesn't publish a per-image table, and our pin recommendations assume the upgrade path from R570 + CUDA 12.8.93.
2. **Pull and verify multi-arch manifests** for `nvcr.io/nvidia/tritonserver:26.04-py3-sbsa`, `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1`, `modular/max-nvidia-full:latest`, `lmsysorg/sglang:latest-cu130` — `docker manifest inspect` to resolve arm64 ambiguity.
3. **Verify NVIDIA's `ubuntu2604/sbsa` repo is actually populated** for driver 580+ and CUDA 13.x — the parent index lists 26.04 but the sbsa subdir contents weren't confirmed.

### FP8 reframe
4. **Reproduce the vLLM FP8 needle-in-haystack fix on GH200** (not just H100). Llama-3.1-8B, head_dim=128, 128k context, vLLM ≥ 0.19.1, calibrated scales. Confirm ≥89% accuracy.
5. **Reproduce a Wan2.2 FP8 failure on a Lambda 1× H100** to confirm the "DiT FP8 broken cross-arch" thesis. Resolves whether the user's original "broken stack" finding is GH200-specific (it isn't, but verify).
6. **Run torchao FP8 dynamic on Flux.1-dev linears (huggingface/flux-fast recipe) on GH200** — confirm the one "trusted" diffusion FP8 path actually works on Grace ARM.

### Benchmarks (none of these exist publicly for GH200)
7. **Single-GH200 Llama-3.3-70B vLLM bf16 + FP8 tok/s** at batch 1/8/32/128, ISL/OSL 1K/1K and 5K/500. Closing the biggest documentation hole in the dossier.
8. **Single-GH200 Wan2.2-A14B 720p×81f end-to-end time** with the recommended stack (bf16, FA3, torch.compile, lightx2v 4-step, MagCache). Confirm or refute the 25–45s estimate.
9. **Single-GH200 SGLang vs vLLM head-to-head on Llama-3.1-8B bf16 + ShareGPT replay** — no public LMSYS GH200 number exists.
10. **NCCL multi-node bandwidth at ≥ 8 nodes** if you ever go past one chip — issue #1272 has no documented fix; 2.20→2.30 regression may still be live.

### Pricing & supply
11. **Direct Lambda sales quote for 1-yr reserved GH200** — $1.49/hr is aggregator hearsay; confirm or replace.
12. **Vultr GH200 actual capacity** in non-Atlanta regions (gpuperhour shows Manchester + Global stockouts) — affects multi-region DR planning.
13. **Verify Bedrock Llama-3.3-70B pricing** — pecollective.com says $0.72/Mtok in/out; tokenmix says $2.65. 3× gap can't be both right.

### Switch math
14. **8× H200 HGX TCO** at Massed Compute self-serve ($16,300/mo) vs 4× Lambda GH200 (4 × $1,672 = $6,688/mo) at equivalent throughput for a real workload (e.g., Llama-3.3-70B served at 200 concurrent users). The "switch" decision hangs on this.
15. **Reproduce Baseten's "GH200 +32% over H100 on Llama-3.3-70B"** to confirm the architectural advantage is real in 2026 with current vLLM/TRT-LLM, not just a 2024-era data point.

### Migration risk
16. **TGI users (if any in the team's workflow)** need an explicit migration path — TGI archived 2026-03-21. Default target: vLLM. Confirm no lurking dependency.
17. **NIM amd64-only status** — re-check `docker manifest inspect nvcr.io/nim/meta/llama-3.3-70b-instruct:latest` for `linux/arm64` periodically; NVIDIA has been promising "future release" since Nov 2024.

---

*End of Round-2 draft. Round 3 should turn the verification list into measurements, kill or upgrade the medium-confidence claims, and produce the final synthesis document.*
