# Round 3 — Verification & Deepening of round1/06_pytorch_triton.md

**Agent**: research #06 of 20 (round 3)
**Date**: 2026-05-25
**Constraint**: FP8 broken on this GH200/Wan DiT stack — bf16 default. Carried forward unchanged from round 1.
**Method**: 14 WebSearch/WebFetch + GitHub API drills against the five flagged claims in round1/06.

---

## TL;DR — what changed since round 1

| Round-1 statement | Round-3 verdict | Update |
|---|---|---|
| "Default `pip install torch` on aarch64 gives a CPU-only wheel until PyTorch 2.11" | **CONFIRMED.** PyTorch 2.11.0 released **2026-03-23**; first version to ship CUDA-on-PyPI for both x86_64 and aarch64 by default. Verified working on GB200 by youkaichao 2026-04-14. | Default is now **CUDA 13.0**, not cu128. cu128 aarch64 wheel still available via `download.pytorch.org/whl/cu128`. |
| "No official prebuilt aarch64 wheels for FA3" | **CHANGED. NOW FALSE.** Official PyTorch-hosted FA3 wheels published **2026-03-10** at `https://download.pytorch.org/whl/flash-attn-3/`, including `manylinux_2_34_aarch64` for cu126/cu128/cu129/cu130. Stable-ABI (`cp39-abi3`). | This is the headline change for the GH200 install recipe. The Dao-AILab repo *itself* still ships no aarch64 wheels in its GitHub releases (only `fa4-*-py3-none-any` source-bundle wheels, last asset 2026-05-20), but PyTorch's index has them. |
| "FlashInfer SBSA support — unclear" | **CONFIRMED present in Q2 2026.** FlashInfer nightly releases since at least 2026-05-18 ship `flashinfer_jit_cache-*+cu128-cp39-abi3-manylinux_2_28_aarch64.whl`. v0.6.12rc1 (2026-05-22) is the most recent. | The `flashinfer_python` wheel itself is `py3-none-any` (pure Python loader); the heavy kernels are in `flashinfer_jit_cache` and `flashinfer_cubin`. aarch64 covered for cu128. |
| "SDPA dispatches to FA3 in 2.7/2.8 — needs verification" | **STILL FALSE through 2.11.** As of May 2026, `F.scaled_dot_product_attention` does NOT auto-dispatch to FA3 on Hopper. FA3 is reachable only via the explicit `flash_attn_func` API or `transformers` `attn_implementation="flash_attention_3"`. PyTorch issue #137901 still open; per drisspg / jbschlosser the path is "manually enable CuDNN [attention] for now." | 2.11 added FlexAttention + FA4 backend (Hopper/Blackwell), but core SDPA still routes FLASH_ATTENTION → FA2. |
| "torch.compile + GH200 + diffusion — any May-2026 benchmark blogs?" | **NO new GH200-specific diffusion benchmarks found.** Public benchmarks remain H100/H200 SXM. PyTorch 2.11 blog (2026-03-23) reiterates the FlexAttention/FA4 numbers but does not add GH200 measurements. Round-1's "GH200 ≈ H100 SXM ±5%" extrapolation still stands. | Gap still open. Round 2 measurement remains the only way to close it. |

---

## 1. PyTorch 2.11 aarch64 CUDA wheels — DRILL #1

### Claim verified
PyTorch 2.11.0 released **2026-03-23**. From the [official release blog](https://pytorch.org/blog/pytorch-2-11-release-blog/):

> "Starting with this release, CUDA 13 is now the default version installed for both x86_64 and ARM platforms. Users who need an alternative build can still access the CPU-only version as well as CUDA 12.8 builds from the respective https://download.pytorch.org/whl subfolders."

End-to-end verification by vLLM's [youkaichao on GB200 (2026-04-14)](https://github.com/pytorch/pytorch/issues/160162#issuecomment-4244969014):
```
$ uv run --no-project --python 3.12 --with 'torch==2.11.0' -- python -c "import torch; print(torch.cuda.is_available())"
True
$ uv run --no-project --python 3.12 --with 'torch==2.10.0' -- python -c "import torch; print(torch.cuda.is_available())"
False
```
2.10 → False, 2.11 → True. Same Python, same uv install. That's the clean cutover.

### Status of cu128 vs cu131 specifically
- **cu128**: For PyTorch 2.8.0 on aarch64, **no cu128 wheel was built** — confirmed by malfet (PyTorch release infra) in [issue #160162](https://github.com/pytorch/pytorch/issues/160162#issuecomment-3189161009): *"For aarch64 we build only one architecture per release, so only cuda-12.9 binaries were published there."* So 2.8.0 aarch64 ⇒ cu129 only. cu128 aarch64 exists for 2.7.1 and for 2.11/2.12 via the `whl/cu128` subfolder.
- **cu131**: I found **no evidence of cu131** wheels for PyTorch on aarch64 in May 2026. 2.11 defaults to **cu130**. cu131 does not appear in any wheel index I could verify. PyTorch 2.12.0 (released **2026-05-13** per PyPI) keeps cu130 default and **removed cu128 binaries** from the matrix.
- **CUDA 12.6 still available** for older drivers.

### Practical install for GH200 (May 2026)
```bash
# Easiest: 2.11 / 2.12 from PyPI directly — but CUDA 13 driver required
pip install torch                     # default cu130, both x86_64 and aarch64

# CUDA 12.x driver only:
pip install torch --index-url https://download.pytorch.org/whl/cu128   # 2.11 aarch64 cu128 OK
# but cu128 dropped in 2.12; pin to torch==2.11.x if you need cu128
```

Sources: [PyTorch 2.11 release blog](https://pytorch.org/blog/pytorch-2-11-release-blog/), [Issue #160162](https://github.com/pytorch/pytorch/issues/160162), [PyPI torch 2.12.0](https://pypi.org/project/torch/), [Issue #157548 nightly cu128 aarch64](https://github.com/pytorch/pytorch/issues/157548).

---

## 2. FA3 prebuilt aarch64 wheels — DRILL #2

### Major change vs round 1
**PyTorch now hosts official FA3 wheels** at `https://download.pytorch.org/whl/flash-attn-3/`, published **2026-03-10** per [dev-discuss "Flash Attention 3 Wheels" thread #3322](https://dev-discuss.pytorch.org/t/flash-attention-3-wheels/3322).

Verified asset filenames at the index include:
- `flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl`
- `flash_attn_3-3.0.0-cp39-abi3-manylinux_2_34_aarch64.whl`  ← **GH200 wheel**
- `flash_attn_3-3.0.0-cp39-abi3-win_amd64.whl`

Properties:
- **Stable ABI** (`cp39-abi3`) — one wheel covers Python 3.9–3.14.
- **LibTorch-stable ABI** — compatible with **any** `torch >= 2.9`.
- CUDA variants: **cu126, cu128, cu129, cu130** — pick the one matching your driver.
- Install:
  ```bash
  pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu130/flash-attn-3/
  # or cu128 if you're staying on CUDA 12.x driver
  ```

This is a meaningful upgrade: the round-1 from-source recipe (gist by Dominoer, ~10 min build, `FLASH_ATTENTION_DISABLE_SM80=TRUE`, etc.) is **no longer required** on GH200 for FA3. Build-from-source remains the fallback only for unusual configs.

### Dao-AILab's own releases still ship no aarch64 binaries
For completeness — the [Dao-AILab/flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases) shows:
- Most recent: **fa4-v4.0.0.beta14** (2026-05-20), assets are `flash_attn_4-4.0.0b14-py3-none-any.whl` + `.tar.gz`. The `py3-none-any` wheel is a thin Python package; the CUDA kernels are built JIT or via CuTe-DSL at runtime — that's the FA4 install pattern.
- Last classic FA2 binary wheel release: **v2.8.3** (2025-08-14) — 50 wheel assets, all `linux_x86_64`, **no aarch64**.
- Open issues confirming the gap: [#2045](https://github.com/Dao-AILab/flash-attention/issues/2045) (2.9 aarch64), [#1875](https://github.com/Dao-AILab/flash-attention/issues/1875) (2.8 aarch64). Per NVIDIA's nWEIdia (2025-09-23 in #1875): *"PyTorch 2.8 aarch64 pytorch wheel was a planning miss and won't likely be fixed."* Per johnnynunez: *"when cutlass v4.2.0 comes out"* — cutlass 4.2.0 shipped 2025-09-15, but FA2 prebuilt aarch64 wheels never followed.

### Community fallbacks (still relevant for FA2-only consumers)
- [windreamer/flash-attention3-wheels](https://github.com/windreamer/flash-attention3-wheels) — prebuilt FA3 wheels, "supports GH200 now" per 2026-01-14 comment in #2045. Predates the official PyTorch wheels by ~2 months.
- [sskalnik/flash_attn_wheels](https://github.com/sskalnik/flash_attn_wheels) — community FA2 wheels for various aarch64 combos (2025-09).
- [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels) — GH Actions CI builds.
- [varunneal/flash-attention-3 (HF Hub)](https://huggingface.co/varunneal/flash-attention-3) — round-1 already mentioned.

### Recommendation (May 2026)
Use the official PyTorch index. Build-from-source is now plan B, not plan A.

---

## 3. FlashInfer SBSA / aarch64 — DRILL #3

### Status: aarch64 wheels are shipping on every nightly since at least 2026-05-18

Direct evidence from the GitHub Releases API ([flashinfer-ai/flashinfer releases](https://github.com/flashinfer-ai/flashinfer/releases)):

| Release | Date | aarch64 asset |
|---|---|---|
| `v0.6.12rc1` | 2026-05-22 | `flashinfer_jit_cache-0.6.12rc1+cu128-cp39-abi3-manylinux_2_28_aarch64.whl` |
| nightly-v0.6.12-20260524 | 2026-05-24 | `flashinfer_jit_cache-0.6.12.dev20260524+cu128-cp39-abi3-manylinux_2_28_aarch64.whl` |
| nightly-v0.6.11-20260522 | 2026-05-22 | `flashinfer_jit_cache-0.6.11.dev20260522+cu128-cp39-abi3-manylinux_2_28_aarch64.whl` |
| nightly-v0.6.11-20260518 | 2026-05-18 | `flashinfer_jit_cache-0.6.11.dev20260518+cu128-cp39-abi3-manylinux_2_28_aarch64.whl` |

Properties:
- aarch64 build is for **cu128** (the GH200-friendly CUDA 12.x line) — cu130 aarch64 not yet seen.
- `flashinfer_python` package itself is `py3-none-any` (pure-Python dispatcher); kernels come via `flashinfer_jit_cache` + `flashinfer_cubin` (the latter is also `py3-none-any`).
- Stable ABI (`cp39-abi3`) — same pattern as the PyTorch FA3 wheels.

### Older context (issue #1236)
The [flashinfer #1236](https://github.com/flashinfer-ai/flashinfer/issues/1236) issue from earlier (v0.2.6.post1, mid-2025) noted that aarch64 wheels existed for v0.2.5 then disappeared. As of May 2026, they're back and shipping continuously via the JIT-cache split. Good — round 1 listed FlashInfer as a gap; that gap is now closed for cu128/Hopper.

### Install (GH200, May 2026)
```bash
pip install flashinfer-python
# Pulls flashinfer_python (any) + auto-resolves flashinfer_jit_cache (cu128 aarch64)
# Or pin the rc:
pip install flashinfer-python==0.6.12rc1
```

Caveat: the [DGX Spark SM121 audit #3170](https://github.com/flashinfer-ai/flashinfer/issues/3170) and the [sgl-kernel sglang #13304](https://github.com/sgl-project/sglang/issues/13304) issue both show downstream consumers (sglang's sgl-kernel) still lack aarch64 prebuilt wheels — so SGLang deployment on GH200 may need extra work even though FlashInfer itself is fixed.

---

## 4. SDPA → FA3 dispatch in PyTorch — DRILL #4

### Verdict: still not automatic in any released PyTorch as of 2026-05.

Direct evidence from [pytorch#137901](https://github.com/pytorch/pytorch/issues/137901):
- 2025-05-20, jbschlosser: *"@KevinZeng08 no not yet, FA3 is only available right now as a separate package outside of PyTorch."*
- 2025-05-20, drisspg: *"we are planning to not make this an opt in feature (so that you can get the speed boost transparently) in a future release but for now you can manually specify it."*
- Subsequent comments through October 2025 ("Any news on this?") — no landed PR.

The follow-up issue [pytorch#148891](https://github.com/pytorch/pytorch/issues/148891) (opened 2025-03-10) outlines two integration paths — *"Replace entirely FAv2 w/ FAv3"* or *"Add FAv3 along w/ FAv2."* Blocker noted: *"FAv3 doesn't support Dropout."* No completion as of May 2026.

### What 2.11 did add
From the 2.11 release blog (2026-03-23):
> "FlexAttention now has a FlashAttention-4 backend on Hopper and Blackwell GPUs. This backend adds support for automatically generating CuTeDSL score/mask modification functions and JIT-instantiating FlashAttention-4 kernels from PyTorch, enabling 1.2× to 3.2× speedups over the existing Triton implementation on compute-bound workloads."

Note carefully: this is **FlexAttention** auto-dispatching to FA4, not core SDPA auto-dispatching to FA3. They are different code paths. `F.scaled_dot_product_attention` with `SDPBackend.FLASH_ATTENTION` still goes to FA2 on Hopper in stock PyTorch 2.11.

A separate 2.11 line item — *"Added torch.compile compatibility to FP8 SDPA using FlashAttention3, including meta registration and inductor lowering fallback"* — wires FP8 SDPA through FA3 specifically (under `torch.compile`). This is **not** what most users mean by "SDPA dispatches to FA3"; it's a narrower path for FP8 training. And per our hard constraint, FP8 is broken on this GH200/Wan stack — so this 2.11 addition is irrelevant to bf16 inference on GH200.

### How to actually get FA3 on Hopper in May 2026
1. `pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu130/flash-attn-3/`
2. In transformers: `attn_implementation="flash_attention_3"` (`transformers >= 4.50`).
3. In diffusers: `pipe.transformer.set_attention_backend("_flash_3_hub")`.
4. Direct: `from flash_attn_3 import flash_attn_func`.

There is **no** `with sdpa_kernel([SDPBackend.FLASH_ATTENTION]): ... ` path that picks FA3 on Hopper in 2.11. Still FA2. Use the explicit APIs above.

Alternative for users who want a "transparent" speedup: `with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]): ...` — cuDNN 9.4+ attention is competitive with FA3 on bf16 per fno2010's benchmarks in #137901, and lands transparently through SDPA. drisspg's recommendation in May 2025 still stands as the bridge.

---

## 5. torch.compile + GH200 + diffusion benchmarks (May 2026) — DRILL #5

### Search outcome: nothing new for GH200 specifically.

Searches for: `"torch.compile" GH200 diffusion benchmark "2026" Flux Wan Hopper`; `huggingface diffusers torch.compile Flux GH200 benchmark 2026`; `flux-fast diffusers benchmark March April May 2026 Hopper H200 GH200`.

What I found:
- [Lambda's GH200 Hugging Face tutorial](https://docs.lambda.ai/education/running-huggingface-diffusers-transformers-gh200/) — still SD-v1.5 hello-world, no compile benchmark. Same as round 1.
- The [PyTorch 2.11 release blog](https://pytorch.org/blog/pytorch-2-11-release-blog/) (2026-03-23) advertises FlexAttention+FA4 speedups but cites the older [FlexAttention + FA4 blog](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/) (2026-03-05) for numbers; both report on H200 / B200, no GH200.
- [PyTorch torch.compile + diffusers hands-on](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/) is still the canonical reference (2025-07-17). Numbers are on H100/H200.
- [Pruna AI's FLUX-Juiced 2.6× endpoint announcement](https://huggingface.co/blog/PrunaAI/flux-fastest-image-generation-endpoint) — vendor blog, doesn't specify GH200.

### Read-through
Public GH200 diffusion benchmarks remain a gap. Round-1's recipe and "H100 SXM ±5%" extrapolation are unchanged. The only path to closing this is to measure ourselves on the actual GH200 instance — which round 1 and round 2 both flagged. No new public data should be expected before round-4; the public benchmark publishing pattern is overwhelmingly H100 SXM / H200 SXM, not GH200.

---

## 6. Updated install recipe (replaces round1/06 §1.1–1.3)

```bash
# 1. PyTorch — pick ONE
#    Default (CUDA 13 driver required):
pip install torch                                  # 2.11 or 2.12 from PyPI, works on aarch64 GH200

#    CUDA 12.x driver only (pin to 2.11.x; 2.12 dropped cu128):
pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu128

# 2. Triton — bundled with torch via pip; if you want explicit:
pip install --upgrade pytorch-triton --index-url https://download.pytorch.org/whl/cu130

# 3. FlashAttention-3 — NOW PREBUILT for aarch64
pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu130/flash-attn-3/
# CUDA 12.x driver:
pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu128/flash-attn-3/

# 4. FlashInfer — aarch64 wheels via nightly/RC index
pip install flashinfer-python                      # pulls flashinfer_jit_cache cu128 aarch64

# 5. torchao, bitsandbytes, diffusers, transformers — unchanged from round 1
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu130
pip install -U "diffusers>=0.35" "transformers>=4.50" accelerate "huggingface_hub[hf_xet]"
```

What's gone vs round 1: the `git clone Dao-AILab/flash-attention && cd hopper && python setup.py install` ritual. You may still want it for FA4 (CuTe-DSL) experiments; FA3 itself no longer needs it on GH200.

---

## 7. Confidence after round 3

### Now high confidence (was medium in round 1)
- PyTorch 2.11+ ships aarch64 CUDA wheels on PyPI by default. **Cutover date: 2026-03-23.** Verified end-to-end on real Grace-Hopper class hardware (GB200) by youkaichao.
- Official prebuilt FA3 aarch64 wheels exist at `download.pytorch.org/whl/flash-attn-3/` since 2026-03-10. Stable-ABI, cu126/cu128/cu129/cu130.
- FlashInfer ships aarch64 cu128 wheels in nightly + RC builds since at least 2026-05-18.
- SDPA does **not** auto-dispatch to FA3 on Hopper through 2.11; bridge via cuDNN-attention or explicit FA3 APIs.

### Still medium / low confidence
- **GH200-specific diffusion benchmarks** — still no public May-2026 data. Round-1 extrapolation unchanged.
- **PyTorch 2.12.0** (released 2026-05-13 per PyPI) details — searched, found the release listed on PyPI; full release blog not yet harvested. Action: round-4 fetch `https://pytorch.org/blog/pytorch-2-12-release-blog/` once it exists.
- **FP8 SDPA via FA3 + torch.compile** path that 2.11 added — could in principle help a future "FP8 fixed" world, but with bf16 hard-constraint it's irrelevant.

### Unchanged from round 1
- FA4 is Blackwell-first; FA3 remains the GH200 workhorse.
- FP8 catastrophic regression on diffusion DiTs (vllm-omni #2728) — no resolution found.
- Inductor flag set (conv_1x1_as_mm, coordinate_descent_tuning, epilogue_fusion=False, shape_padding) — unchanged.
- Diffusers regional compile (`compile_repeated_blocks`) — unchanged.

---

## 8. Sources added in round 3 (chronological)

- 2025-03-10 — [pytorch#148891 FA3 SDPA integration proposal](https://github.com/pytorch/pytorch/issues/148891)
- 2025-05-20 — [pytorch#137901 comments: "no not yet" on FA3-in-SDPA](https://github.com/pytorch/pytorch/issues/137901#issuecomment-2891-ish)
- 2025-08-08 — [pytorch#160162 aarch64 PyPI wheel discussion](https://github.com/pytorch/pytorch/issues/160162)
- 2025-09 — [flash-attention#1875 PyTorch 2.8 aarch64 won't-fix](https://github.com/Dao-AILab/flash-attention/issues/1875)
- 2025-12 — [flash-attention#2045 PyTorch 2.9 aarch64 + cu130](https://github.com/Dao-AILab/flash-attention/issues/2045)
- 2026-01-14 — [windreamer/flash-attention3-wheels — community GH200 wheels](https://github.com/windreamer/flash-attention3-wheels)
- 2026-03-05 — [PyTorch blog: FlexAttention + FlashAttention-4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
- 2026-03-10 — [dev-discuss: Flash Attention 3 Wheels (official PyTorch index)](https://dev-discuss.pytorch.org/t/flash-attention-3-wheels/3322)
- 2026-03-23 — [PyTorch 2.11.0 release blog](https://pytorch.org/blog/pytorch-2-11-release-blog/) — CUDA 13 default on x86_64 + ARM
- 2026-04-14 — [pytorch#160162 youkaichao confirms torch==2.11 works on GB200, 2.10 does not](https://github.com/pytorch/pytorch/issues/160162#issuecomment-4244969014)
- 2026-05-13 — [PyPI: torch 2.12.0 release](https://pypi.org/project/torch/)
- 2026-05-18..2026-05-24 — [flashinfer nightly releases — aarch64 cu128 jit_cache wheels](https://github.com/flashinfer-ai/flashinfer/releases)
- 2026-05-20 — [Dao-AILab/flash-attention fa4-v4.0.0.beta14 release](https://github.com/Dao-AILab/flash-attention/releases) (no aarch64 binaries; py3-none-any only)
- [PyTorch official FA3 wheel index](https://download.pytorch.org/whl/flash-attn-3/) — manylinux_2_34_aarch64 visible
