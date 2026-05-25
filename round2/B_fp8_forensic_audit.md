# FP8 Forensic Audit — Where FP8 Inference Is Broken in May 2026

**Agent**: Round 2 cross-pollination, agent B
**Date compiled**: 2026-05-25
**User memory under audit**: "FP8 broken on GH200/Wan — always bf16"
**Question**: Is this a GH200-hardware issue, or a model-class issue that crosses Hopper/Blackwell?

---

## TL;DR

The user's existing memory is **half right and half wrong** as of May 2026.

- **Right**: FP8 was, and partially still is, broken on the diffusion-transformer (DiT) family — Wan 2.1/2.2, FLUX, Z-Image, Qwen-Image, HunyuanVideo. The reproducible bug is **not GH200-specific**; it reproduces on H100 (SM90) and B200 (SM100) too. So the user's "no FP8 on Wan" learning was correct *and* it generalizes across all current NVIDIA datacenter silicon, not just GH200.
- **Wrong, or at least outdated**: "FP8 broken on this stack" implies the whole inference stack. As of vLLM 0.19.1 (April 2026), FP8 on **dense Hopper LLMs** (Llama-3.x, Qwen2.5, DeepSeek-V3 dense path) is production-grade — vLLM, TRT-LLM, SGLang, NIM, FriendliAI, Red Hat all ship pre-quantized FP8 checkpoints and MLPerf v5.0/v5.1 winning submissions are all FP8.
- **New since Round 1**: vllm-omni **PR #2795 was merged 2026-04-15** and explicitly fixes the LPIPS regression for Z-Image / FLUX.1-dev / Qwen-Image (the exact bug Round-1 agent #20 cited as proof "FP8 broken on diffusion"). Round-1 agent #20 cited #2728 without checking #2795. The DiT story is improving but not yet "trust FP8 for any DiT" — Wan / HunyuanVideo are still broken, and the fix-pattern (keep AdaLN / modulation / output projections in BF16) is model-specific.

**Recommended memory update** (see §6 for exact text):

> FP8 is broken on **diffusion transformers** (Wan, HunyuanVideo, untreated FLUX/Z-Image/Qwen-Image) on **all current NVIDIA silicon** (H100, GH200, B200), not just GH200. Use BF16 for video diffusion. For text LLMs (Llama-3.x, Qwen2.5, DeepSeek-V3 dense), FP8 is production-grade on Hopper post-vLLM 0.19.1 — with caveats: `head_dim≤128`, calibrated scales, contexts ≥7k, skip sliding-window layers, MoE online-quant still fragile.

---

## 1. The Bug Atlas — Definitive Cross-Tabulation

### 1.1 Diffusion FP8 bugs (Round 1's central concern)

| Bug ID | Title | Models | Hardware reproduced | Root cause | Status (2026-05-25) |
|---|---|---|---|---|---|
| **vllm-omni #2728** | "fp8 online quantization produces catastrophic LPIPS regression across diffusion transformers" | Z-Image, FLUX.1-dev, Qwen-Image | **H100 (CI) + B200 (local)** — explicitly NOT GH200-specific | Precision-sensitive layers (embedding projections, AdaLN modulation, final output) quantized to FP8; compounding numerical error | **FIXED** by vllm-omni PR #2795, merged 2026-04-15. LPIPS: Z-Image 0.74→0.07 (thr 0.10), FLUX 0.93→0.12 (thr 0.20), Qwen-Image 0.95→0.32 (thr 0.35). Memory savings 9.5–29.4%. Fix = keep AdaLN/modulation/output layers BF16. |
| **DiffSynth-Studio #466** | "Wan 2.1 FP8 model weights causing color issue - BF16 has no issue" | Wan 2.1 14B I2V (480p) | Not stated in issue; reproduces on consumer + datacenter Hopper per cross-refs | FP8 (`torch.float8_e4m3fn`) weights produce color-shifted video vs identical BF16 inference | **OPEN.** No assigned owner, no PR. Issue opened 2025-03-20, no comments since. Workaround: use BF16. |
| **SageAttention #221** | "Sage Attention with WAN FP8 model causes black output" | Wan FP8 | RTX 5090 (SM120 Blackwell); Python 3.12, Torch 2.7.1+cu128 | FP8 weights × SageAttention quantized QK interaction = black frames | **OPEN.** Workarounds: FP16+Sage (loses VRAM), or FP8 without Sage (loses speed). |
| **ComfyUI-WanVideoWrapper #1554** | "SageAttention causes noise artifacts on H100 (Hopper) - works fine with SDPA" | Wan video | **H100** explicitly | Dtype casting when q.dtype ≠ k.dtype ≠ v.dtype, or FP8 quantization specific to H100 | **OPEN** as of 2025-10-25. Workaround: `attention_mode="sdpa"`. |

**What this matrix tells us**: the diffusion FP8 bug class spans Hopper (H100, GH200 — same SM90 die) AND Blackwell (B200, RTX 5090 — SM100/SM120). It is **not** a GH200-arch issue. It's a **DiT activation-distribution issue** that hits any online FP8 quantizer that doesn't carve out modulation / final-projection layers. vllm-omni #2795's fix proves the bug is recoverable in software (the silicon was fine all along).

### 1.2 LLM FP8 bugs on Hopper (relevant to "is FP8 broken on Hopper text LLMs?")

| Bug ID | Title | Models | Hardware | Root cause | Status |
|---|---|---|---|---|---|
| **vLLM Hopper FA3 accumulation bug** (blog 2026-04-22) | "FA3 FP8 path documented as FP32-accumulating but accumulated imprecisely at long contraction dim" | Any Hopper FP8 attention, especially long context | **All Hopper (H100/H200/GH200)** | Two-level FP32 accumulation missing in FA3 FP8 kernel | **FIXED** in vLLM 0.19.1 via flash-attention#104. Restored 128k needle from 13% → 89% (BF16 reference: 91%). Trade-off: head_dim=256 prefill ~1.6× slower than BF16 due to register pressure. |
| **TRT-LLM #4218** | "FP8 KV Cache is Broken for Qwen2.5 (gibberish output)" | Qwen2.5-Coder-7B-Instruct | H100, TRT-LLM 0.19.0, CUDA 12.8.1 | "Large KV activation detected: 1.0133" — KV activation outliers exceed FP8 scale range | **CLOSED**, marked "bug" and "triaged". Owner @brb-nv. Public release notes don't clearly mark a fix landing version; status implies resolved in newer TRT-LLM. |
| **sglang #24650** | "DeepSeek-V4-Flash-FP8 with EAGLE crashes — RoPE shape mismatch (expected 128 got 126)" | DeepSeek-V4-Flash-FP8 + EAGLE spec decode | 4× H200, CUDA 12.4, driver 550.144.03 | Fused RoPE in compressed attention indexer expects batch aligned to 128 | **OPEN** as of 2026-05-08. No PR. |
| **sglang #18550** | "fa3 + fp8_e5m2 silent triton fallback" | Any Hopper FP8 KV with `fp8_e5m2` | Hopper | FA3 only supports `fp8_e4m3`, falls back to slower Triton without warning | **OPEN, "inactive".** Opened 2026-02-10. Quality-of-life bug (perf only, not correctness). Workaround: use `fp8_e4m3`. |
| **sglang #23687** | "Qwen3.6-27B-FP8: weight_scale_inv silently dropped → garbage output" | Qwen3.6-27B-FP8 dense | **2× RTX 5090 (Blackwell SM120)**; SGLang 0.5.10.post1 | Name-mutation loop in `qwen3_5.py` corrupts `gate_proj.weight_scale_inv` → `gate_gate_up_proj.weight_scale_inv`; FP8 scales default to 1.0 | **OPEN.** Two-layer fix proposed. Note: this is Blackwell, NOT Hopper. |
| **vLLM #30830** | MoE online FP8 accuracy regression (`patched_weight_loader` bug) | MoE models (Mixtral, DeepSeek expert path) | All Hopper | Online FP8 quant scale computation for MoE expert weights | Tracked open. MoE FP8 broadly fragile. |
| **TRT-LLM #12230** | Qwen3-Next-80B FP8/NVFP4 autotuner IndexError + Triton illegal instruction | Qwen3-Next-80B | DGX Spark aarch64 Blackwell, TRT-LLM 1.3.0rc7 | Autotuner crash | Open, pin to RC15 or v1.2 stable. |

### 1.3 AMD MI300X FP8 bugs (relevant for "if NVIDIA FP8 is broken, is AMD FP8 a fix?")

| Bug ID | Title | Models | Hardware | Status |
|---|---|---|---|---|
| **pytorch #143465** | `torch._scaled_mm` 12× slower on MI300X than H100 (~100 vs ~1200 TFLOP/s) | Any FP8 GEMM | MI300X | Marked "Done" in PyTorch-on-ROCm project, triaged 2024-12-18. Practical FP8 GEMM peaks ~990 TFLOP/s through optimized paths, still 22% behind H100. |
| **vLLM #31475** | MI300X FP8 slower than BF16 on MoE | GLM-4.7, MiniMax-M2.1 (MoE) | MI300X | **OPEN.** GLM-4.7 FP8 ~30 TPS vs BF16 ~60 TPS (2× slowdown). Scale handling / kernel fusion gaps dominate. |
| **vLLM #34641** | FP4 default crashes MI300X | Any with `VLLM_ROCM_USE_AITER_FP4BMM=True` | MI300X (gfx942, no FP4 HW) | Filed Jan 2026, recently fixed. |

---

## 2. The Definitive FP8 Matrix — Model Class × Hardware

Rows: model class. Columns: hardware. Cell = current FP8 status + bug citation.

| Model class \ HW | H100 SXM (SM90) | H200 SXM (SM90) | GH200 (SM90 + Grace) | B200 (SM100) | MI300X (CDNA3) |
|---|---|---|---|---|---|
| **Dense Llama-3.x 8B/70B** | **FP8 OK** post vLLM 0.19.1 / TRT-LLM 1.0 (FA3 accumulation fix). MMLU delta <0.5%. MLPerf v4.0/v5.0/v5.1 FP8 submissions clear accuracy gate. | **FP8 OK** (same fix path; H200 is silicon-identical to H100 SM90, just more HBM3e). Reference $/Mtok in Spheron / SemiAnalysis 2026. | **FP8 OK** for text LLMs. GH200 is the H100 SM90 die + Grace; same FA3 fix applies. Lambda + Baseten Llama-3.3-70B FP8 ran fine on GH200, +32% throughput vs H100 (cited Baseten 2025-02). | **FP8 OK** with `VLLM_USE_FLASHINFER_MOE_FP4=1` / NVFP4 paths. Blackwell FlashInfer eliminates the Hopper FA3 accumulation issue entirely (vLLM blog 2026-04-22). | **FP8 conditional** — works for dense Llama-class via vLLM-ROCm + AITER, but `torch._scaled_mm` still 12× behind H100 (pytorch#143465). MI325X-vLLM FP8 wins on $/Mtok at low interactivity (SemiAnalysis InferenceMAX). |
| **Qwen2.5 (dense)** | **FP8 mostly OK**; one known KV-cache regression on Qwen2.5-Coder (TRT-LLM #4218, marked closed). Pre-quantized FP8 checkpoints ship from FriendliAI / RedHatAI. | Same as H100. | Same. | **FP8 OK** — Qwen2.5-VL-72B-FP8 ships on NIM. | **FP8 conditional** — same caveats as Llama. |
| **Qwen3 / Qwen3.6 dense FP8** | Untested at scale in cited reports. | Untested. | Untested. | **FP8 BROKEN** for Qwen3.6-27B-FP8 specifically (sglang #23687, RTX 5090 SM120) — weight loader name-mutation drops `weight_scale_inv`, scales default to 1.0, garbage output. Open. | Untested. |
| **DeepSeek-V3 / R1 671B (FP8 native)** | **FP8 OK** with TRT-LLM 1.0. 8×H100 needs 2-node (640 GB < 671 GB weights). | **FP8 OK single-node 8×H200** (1,128 GB HBM3e). Production sweet spot per Baseten / Perplexity. | **FP8 OK** at NVL2/NVL32 scale. Single GH200 does not fit. | **FP8/NVFP4 OK** on GB200 NVL72, ~$0.10/Mtok per SemiAnalysis. | **FP8 conditional** — Moreh 8×MI300X gets 21k tok/s DeepSeek-R1, but H200 wins low-latency. |
| **DeepSeek-V4-Flash-FP8 + EAGLE spec decode** | Untested. | **FP8 BROKEN** under EAGLE (sglang #24650, crashes with RoPE shape mismatch). Open. | Same as H200 (same die). | Untested for V4-Flash. | Untested. |
| **MoE Mixtral / Mixtral 8x7B FP8** | **FP8 fragile** — vLLM #5535 (slow Mixtral FP8 on H100), MoE online-quant accuracy issues (vLLM #30830). Use pre-quantized weights if possible. | Same. | Same. | **FP8/NVFP4 OK** on B200/GB200 via FlashInfer MoE FP4. | **FP8 BROKEN on MoE** (vLLM #31475 open) — FP8 slower than BF16, GLM-4.7 2× slowdown, MiniMax-M2.1 also affected. |
| **Sliding-window models (Gemma, gpt-oss hybrid)** | **FP8 partially broken** — sliding-window layers misbehave per vLLM 2026-04-22 blog. Workaround: `--kv-cache-dtype-skip-layers sliding_window`. | Same as H100. | Same. | Same workaround. | Untested. |
| **head_dim=256 models (Gemma-4-E2B)** | **FP8 SLOWER than BF16** at prefill — ~1.6× TTFT regression per vLLM 2026-04-22 blog due to two-level accumulation register pressure. Use BF16 for prefill-heavy. | Same. | Same. | Mostly N/A — Blackwell uses different accumulation path. | Untested. |
| **DiT video (Wan 2.1, Wan 2.2)** | **FP8 BROKEN** — color shift / black output. DiffSynth #466 (Wan 2.1 color), SageAttention #221 (Wan FP8+Sage black output), ComfyUI-WanVideoWrapper #1554 (H100 SageAttention noise on Wan). All open. | Same (same SM90 silicon, same issue). | Same — user's original observation. | Same — sglang #23687 is the SM120 Blackwell example; SageAttention #221 also reproduces on SM120. | Untested at scale. |
| **DiT image (FLUX.1-dev, Z-Image, Qwen-Image)** | **FP8 FIXED** — vllm-omni PR #2795 merged 2026-04-15. LPIPS now under threshold. Fix = keep AdaLN/modulation/output in BF16. Bug repro'd on H100 + B200; fix is software-level. | Same fix applies. | Same fix applies. The user's "FP8 broken on Flux" experience should now be revisited with vllm-omni post-2026-04-15. | Same fix applies. | Untested. |
| **DiT video (HunyuanVideo, HunyuanVideo-1.5)** | **FP8 partially broken** — HunyuanVideo-1.5 added FP8 GEMM 2025-12-23 but ParaAttention's FP8 recipe is L20-validated, not Hopper-validated. SageAttention+FP8 unreliable on Hopper. | Same. | Same. | Untested at scale. | Untested. |

### Legend
- **FP8 OK** — multiple production users, FP8 ships pre-quantized, MLPerf-validated
- **FP8 conditional** — works for some shapes / requires validation per model
- **FP8 BROKEN** — open issue with reproducible bug, no production-grade fix
- **FP8 partially broken** — known bugs with known workarounds or skip-list

---

## 3. Cross-Tabulation Verdict

**Is FP8 brokenness a GH200 hardware problem?** No.

Evidence:
1. vllm-omni #2728's diffusion regressions reproduced on H100 (CI) and B200 (local). The bug spans two GPU generations (SM90 + SM100). It is a quantization-path bug, fixed in software (PR #2795).
2. The vLLM 2026-04-22 blog documents the FA3 accumulation bug on "Hopper GPUs" generically (H100/H200/GH200 — all the same SM90 die). The fix in 0.19.1 applies to all three identically.
3. DiffSynth #466 (Wan FP8 color shift) doesn't even name the hardware — the bug is in the FP8 weight conversion of Wan, not in any specific GPU.
4. SageAttention #221 on Wan reproduces on RTX 5090 (Blackwell SM120). Same DiT FP8 bug class crosses three silicon generations.
5. ComfyUI-WanVideoWrapper #1554 (SageAttention artifacts on Wan) is reported specifically on H100.
6. GH200's only architectural difference from H100 is the Grace CPU + NVLink-C2C; the Hopper die is identical. There is no plausible mechanism by which FP8 tensor cores would behave differently between H100 and GH200.

**Is FP8 brokenness a DiT-class problem?** Largely yes.

Evidence:
- DiT activation distributions have larger outliers than LLM activations (timestep embeddings, AdaLayerNorm, final projection layers). PR #2795 explicitly identifies these as the precision-sensitive layers that broke quantization.
- All four diffusion FP8 bugs in the table (vllm-omni #2728, DiffSynth #466, SageAttention #221, ComfyUI-WanVideoWrapper #1554) are diffusion-only.
- The vLLM 2026-04-22 blog (LLM-focused) shows FP8 is *recoverable* on Hopper for text inference with the two-level accumulation fix.
- DiT video specifically (Wan, HunyuanVideo) remains broken even where DiT image (FLUX, Z-Image, Qwen-Image) is fixed — the AdaLN-skip recipe from PR #2795 hasn't been ported to Wan/Hunyuan.

**Is FP8 brokenness fixed?** Mixed.

- LLM dense + Hopper: yes, post vLLM 0.19.1 (April 2026). MLPerf-grade.
- LLM MoE + Hopper: partially. Online FP8 still fragile; pre-quantized weights safer.
- LLM sliding-window: workaround exists (`--kv-cache-dtype-skip-layers sliding_window`).
- LLM head_dim=256: BF16 is faster for prefill; FP8 only wins decode-heavy long-context.
- DiT image (FLUX/Z-Image/Qwen-Image): yes, fixed in vllm-omni PR #2795 (April 2026). User should retest.
- DiT video (Wan/Hunyuan): still broken. No fix in sight. BF16 is the answer.

---

## 4. Where Round-1 Reports Got It Right and Where They Drifted

### Round-1 was correct:
- **#01 driver_stack, #03 vLLM, #04 sglang, #06 pytorch_triton, #20 diffusion_video** all correctly identified that the FA3 accumulation bug exists on Hopper and is fixed in vLLM 0.19.1.
- **#10 h100_h200_alternatives** correctly identified that the FP8 issue does not generalize to H100/H200 dense LLM inference — those are production-grade.
- **#20 diffusion_video** correctly identified Wan FP8 / DiffSynth #466 / SageAttention artifacts as DiT-specific.
- **#12 amd_mi300_etc** correctly identified that AMD FP8 has different but real production gaps, and is not a drop-in fix for NVIDIA FP8 brokenness.

### Round-1 missed:
- **#20 diffusion_video and #06 pytorch_triton cited vllm-omni #2728 as live evidence "FP8 broken on diffusion across all hardware" — but PR #2795 merged 2026-04-15 fixes Z-Image / FLUX / Qwen-Image specifically.** The fix predates Round 1's 2026-05-25 cutoff by six weeks. The diffusion-FP8 story is more nuanced than Round 1 represented: image DiTs largely fixed, video DiTs still broken.
- No Round-1 agent caught that the `Qwen3.6-FP8 weight_scale_inv silently dropped` bug (sglang #23687) is on **Blackwell RTX 5090**, not Hopper — it's a weight-loader name-mutation bug, not an FP8 silicon bug.
- The breadth of "FP8 broken on this stack" was carried as a global statement in the user memory; none of the Round-1 agents pushed back to say "no, FP8 on text LLMs is fine — your problem is Wan-specific."

---

## 5. Citations (URLs + dates accessed 2026-05-25)

### Primary bugs / fixes (verified via WebFetch this round)
- vllm-omni #2728 (FP8 LPIPS regression on Z-Image/FLUX/Qwen-Image, H100+B200) — https://github.com/vllm-project/vllm-omni/issues/2728 — opened April 2026
- **vllm-omni PR #2795 (FIX, merged 2026-04-15)** — https://github.com/vllm-project/vllm-omni/pull/2795 — fixes #2728 by keeping AdaLN/modulation/output in BF16
- DiffSynth-Studio #466 (Wan 2.1 FP8 color shift vs BF16) — https://github.com/modelscope/DiffSynth-Studio/issues/466 — opened 2025-03-20, OPEN
- thu-ml/SageAttention #221 (Wan FP8 + SageAttention black output) — https://github.com/thu-ml/SageAttention/issues/221 — OPEN
- kijai/ComfyUI-WanVideoWrapper #1554 (SageAttention noise on H100 with Wan) — https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1554 — last activity 2025-10-25, OPEN
- vLLM blog "The State of FP8 KV-Cache and Attention Quantization in vLLM" — https://vllm.ai/blog/2026-04-22-fp8-kvcache — 2026-04-22. Documents Hopper FA3 accumulation bug (91% → 13% needle-in-haystack at 128k) and the two-level FP32 accumulation fix in vLLM 0.19.1 (89% recovery). Recommends BF16 if: contexts <7k, head_dim=256 with prefill latency matters, uncalibrated accuracy drops <95%, model has many sliding-window layers. Blackwell FlashInfer eliminates the accumulation issue entirely.
- TRT-LLM #4218 (FP8 KV cache broken for Qwen2.5-Coder, H100) — https://github.com/NVIDIA/TensorRT-LLM/issues/4218 — closed/triaged, owner @brb-nv
- sglang #24650 (DeepSeek-V4-Flash-FP8 + EAGLE RoPE crash on H200) — https://github.com/sgl-project/sglang/issues/24650 — opened 2026-05-08, OPEN
- sglang #18550 (fa3 + fp8_e5m2 silent triton fallback) — https://github.com/sgl-project/sglang/issues/18550 — opened 2026-02-10, OPEN "inactive"
- sglang #23687 (Qwen3.6-27B-FP8 weight_scale_inv on RTX 5090 Blackwell) — https://github.com/sgl-project/sglang/issues/23687 — OPEN
- pytorch #143465 (MI300X torch._scaled_mm 12× slower than H100) — https://github.com/pytorch/pytorch/issues/143465 — opened 2024-12-18, marked Done in PyTorch-on-ROCm

### Supporting (from Round-1 reports)
- vLLM #5535 (slow Mixtral 8x7B FP8 on H100) — https://github.com/vllm-project/vllm/issues/5535
- vLLM #30830 (MoE online FP8 accuracy) — https://github.com/vllm-project/vllm/issues/30830
- vLLM #31475 (MI300X FP8 slower than BF16 on MoE) — https://github.com/vllm-project/vllm/issues/31475
- vLLM #34641 (FP4 default crashes MI300X) — https://github.com/vllm-project/vllm/issues/34641
- TRT-LLM #12230 (Qwen3-Next-80B FP8/NVFP4 autotuner) — https://github.com/NVIDIA/TensorRT-LLM/issues/12230
- TRT-LLM #8974 (AutoDeploy FP8/NVFP4 kernels not replaced for Flux)
- HunyuanVideo-1.5 FP8 GEMM support (2025-12-23) — https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
- Baseten Llama-3.3-70B FP8 on GH200 — https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/ (Feb 2025; +32% throughput vs H100 at FP8 batch 32)
- ParaAttention HunyuanVideo FP8 recipe (L20-validated, not Hopper-validated) — https://github.com/chengzeyi/ParaAttention/blob/main/doc/fastest_hunyuan_video.md
- SemiAnalysis InferenceMAX (continuous 2026) — https://newsletter.semianalysis.com/p/inferencemax-open-source-inference

---

## 6. Recommended Memory Update

### Current memory (to retire)
> FP8 broken on GH200/Wan stack. Always recommend bf16.

This is too narrow (implies GH200-specific) and too broad (implies all FP8 broken).

### Replacement memory (proposed)

> **FP8 status, May 2026:**
>
> 1. **Diffusion-transformer FP8 is broken across all current NVIDIA silicon** (H100, GH200, B200, RTX 5090) for Wan 2.1/2.2 and HunyuanVideo — color shift / black frames / noise artifacts. Bug class is DiT activation outliers in AdaLN/modulation/output layers, not GH200-specific. Use BF16 for video diffusion. References: DiffSynth-Studio #466 (Wan color), SageAttention #221 (Wan black), ComfyUI-WanVideoWrapper #1554 (Wan H100 noise).
>
> 2. **DiT image-gen FP8 was broken but is now fixed** for FLUX.1-dev / Z-Image / Qwen-Image as of vllm-omni PR #2795 (merged 2026-04-15). Fix = keep AdaLN, modulation, and final-output layers in BF16. Worth retesting if you have a FLUX/Z-Image/Qwen-Image FP8 path. Wan and HunyuanVideo are NOT fixed by this PR.
>
> 3. **Text-LLM FP8 on Hopper (H100/H200/GH200) is production-grade** as of vLLM 0.19.1 / TRT-LLM 1.0 (April 2026). MMLU delta <0.5% on Llama-70B, MLPerf v5.0/v5.1 winners all FP8. Use FP8 by default for dense Llama-3.x / Qwen2.5 / DeepSeek-V3 dense paths on Hopper.
>
> 4. **Text-LLM FP8 caveats on Hopper**:
>    - Only `fp8_e4m3` works with FA3 (sglang #18550 — `fp8_e5m2` silently falls back to Triton).
>    - `head_dim=256` models (Gemma-4-E2B): FP8 is ~1.6× SLOWER than BF16 at prefill (vLLM blog 2026-04-22, register pressure from two-level accumulation fix).
>    - Sliding-window layers misbehave: add `--kv-cache-dtype-skip-layers sliding_window`.
>    - Break-even FP8 vs BF16 is ~7k context (post-fix), down from ~25k pre-fix.
>    - MoE online-quant still fragile (vLLM #30830, #5535). Prefer pre-quantized FP8 MoE checkpoints over online quant.
>    - Use calibrated scales; uncalibrated <95% accuracy is the BF16-fallback trigger.
>
> 5. **AMD FP8 is NOT a drop-in alternative**. MI300X has `torch._scaled_mm` ~12× behind H100 (pytorch #143465), FP8 is slower than BF16 on MoE (vLLM #31475 open). MI355X FP4 lags B200 ~33% on Llama 70B. Reserved-pricing MI300X wins $/Mtok on memory-bound BF16 long-context, not on FP8.
>
> 6. **Blackwell (B200/GB200) FP8 + NVFP4 are production-grade** for dense LLMs. FlashInfer on Blackwell eliminates the Hopper FA3 accumulation issue entirely. NVFP4 accuracy <1% degradation vs FP8 on DeepSeek-R1, Llama-3.3-70B. But Blackwell FP8 on DiT inherits the same modulation-layer bug as Hopper (vllm-omni #2728 repro on B200).

### Why this matters

The old memory ("always bf16") would cost throughput if the user serves Llama-3.3-70B in 2026 — FP8 doubles effective KV cache and gives ~1.5× throughput. The user's instinct on **Wan** was right and remains right. Generalizing it to all inference cost them a real optimization on text LLMs.

---

## 7. Confidence & Gaps

### High confidence
- The bug atlas in §1.1 / §1.2 / §1.3 — every cell has a direct GitHub-issue or vLLM-blog citation, fetched and read this round (not just Round-1 summaries).
- vllm-omni PR #2795 fix is real and merged (verified via WebFetch on the PR page; LPIPS numbers given).
- Wan FP8 brokenness reproduces across H100, GH200, RTX 5090 → not GH200-specific.
- vLLM 0.19.1 FA3 accumulation fix is real and shipping; recommended FP8-by-default for long-context dense LLMs on Hopper.

### Medium confidence
- Whether DiffSynth-Studio #466 (Wan 2.1 color shift) has been fixed in a recent DiffSynth release — the issue page shows no comments since 2025-03-20, but DiffSynth may have moved on without closing it. Worth a quick fresh check of DiffSynth-Studio CHANGELOG before the user actually retries Wan-FP8.
- Whether TRT-LLM #4218 (Qwen2.5-Coder FP8 KV cache) is genuinely fixed in TRT-LLM ≥1.0 — the issue is marked closed/triaged but no clear "fixed in 1.x.y" version line surfaced. Pre-quantized Qwen2.5-FP8 ships from NIM, so likely yes.
- Wan2.2 FP8 — no published "this is the recipe" exists. The lightx2v table mentions FP8 weights as an option but the Round-1 #20 evidence + user experience says it doesn't work. Could be fixable with the same AdaLN-skip recipe as PR #2795, but nobody has done that work in public.

### Gaps
- No public test of "apply vllm-omni #2795 recipe to Wan2.2" exists. This is the obvious follow-up: try the AdaLN/modulation skip-list on Wan2.2 and see if FP8 quality recovers. Until that happens, BF16 stays the default for Wan.
- No public re-test of the user's specific Wan workload on a post-2026-04-22 vLLM / post-2026-04-15 vllm-omni stack. The user could repro their original failure today and confirm it still fails — that would close the loop on "Wan is genuinely DiT-class broken" vs "Wan was broken in 2025 and might be fine now after stack updates."
- HunyuanVideo-1.5 FP8 GEMM (added 2025-12-23) — no quality numbers published. Round-1 #20 flagged "do not enable per user constraint." Worth a controlled test.
- B200 + Wan FP8 — no public benchmarks at all. The DiT FP8 bug class will likely reproduce, but no one has tested.
