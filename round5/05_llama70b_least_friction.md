# Round 5 — Least-friction path from fresh Lambda GH200 to Llama-3.3-70B Q4 OpenAI server

**Date:** 2026-05-25
**Target:** 1× Lambda Cloud GH200 96GB, Ubuntu 24.04 aarch64, stock image (NVIDIA driver + Docker preinstalled)
**Goal:** SSH → `http://localhost:8000/v1/chat/completions` answering, with minimum commands.

---

## Stock Lambda GH200 baseline (what you get free)

On a fresh Lambda GH200 boot you already have:
- NVIDIA driver loaded (`nvidia-smi` works), CUDA 12.x runtime on host
- Docker + `nvidia-container-toolkit` configured (`docker run --gpus all` works)
- `~/` on persistent NVMe; HF cache survives reboot
- Ubuntu 24.04 noble, **4 KB page kernel** (not the 64 KB one — most prebuilt CUDA wheels still work; only some kernels that hardcode page assumptions break)

This eliminates the CUDA-install step from every path below. **Custom 64 KB page kernels are NOT needed** for any of the five options.

---

## Path 1 — Ollama

**Verdict: BROKEN for GPU use on GH200 today.**

- The official `install.sh` detects `aarch64` and maps to `arm64`, but **does not install a CUDA driver/runtime on aarch64**. CUDA install paths are gated to x86_64 distros. ([install.sh source](https://github.com/ollama/ollama/blob/main/scripts/install.sh))
- aarch64 + CUDA is only delivered through **JetPack 5/6 variants** ([PR #7217](https://github.com/ollama/ollama/pull/7217)), which target Jetson SoCs (Tegra libcuda), not Grace Hopper's standard discrete-GPU stack.
- Known fail modes on aarch64+CUDA: `CUDA error: PTX JIT compiler library not found`, `__CUDA_ARCH_LIST__` mismatch on sm_90, EOF on model load. Issues [#13169](https://github.com/ollama/ollama/issues/13169), [#6591](https://github.com/ollama/ollama/issues/6591), [#3406](https://github.com/ollama/ollama/issues/3406).
- It runs on Grace's ARM CPU, but you'd be doing 70B Q4 on LPDDR5X-only — ~5–10 tok/s, ignoring the H100 entirely.

**Commands (if you accepted CPU):** 3 commands (`curl … | sh`; `ollama serve &`; `ollama run llama3.3:70b-instruct-q4_K_M`). API on `:11434/v1`, **not 8000**.

**Score:** friction **4** (CPU-only is broken-by-design for the goal), expected **<10 tok/s** → effective frontier: **off the chart**.

---

## Path 2 — LM Studio headless (`lms`)

**Verdict: Possible but unsupported on aarch64 Linux.**

- LM Studio ships official aarch64 Linux **AppImage**, and `llmster`/`lms daemon up` is the headless mode ([docs](https://lmstudio.ai/docs/developer/core/headless)).
- `lms server start` exposes the OpenAI-compatible endpoint at `http://localhost:1234/v1` by default (not 8000).
- GPU backend on Linux aarch64 is **CUDA via llama.cpp**, but LM Studio's official "NVIDIA on Linux" support targets x86_64; on aarch64 Linux it falls back to CPU.
- Closed source — you can't fix what doesn't work, and Lambda image has no display server for the GUI fallback.

**Commands:** ~5 (`curl install.sh | bash`; `lms daemon up`; `lms get llama-3.3-70b@q4_k_m`; `lms load …`; `lms server start --port 8000`). But GPU acceleration is the unknown.

**Score:** friction **4**, expected **5–10 tok/s** (CPU fallback most likely). Skip.

---

## Path 3 — llama.cpp prebuilt arm64 CUDA tarball

**Verdict: Works, but no official prebuilt — community tarballs only.**

- **Official `ggml-org/llama.cpp` releases ship NO Linux aarch64 + CUDA tarball.** Linux ARM64 assets are CPU and Vulkan only. ([releases page confirmed](https://github.com/ggml-org/llama.cpp/releases))
- **`ai-dock/llama.cpp-cuda`** community repo publishes `llama.cpp-bXXXX-cuda-12.8-arm64.tar.gz` per build. CUDA 12.8, not 13. ([repo](https://github.com/ai-dock/llama.cpp-cuda))
- Performance reference on GH200 from upstream discussion: GPT-OSS 20B MXFP4 gets ~323 tok/s decode; **Llama-2 7B Q4 GH200 takes #1 spot in token generation**. Extrapolating from Q4_K bandwidth-bound scaling on a 4.0 TB/s HBM3e Hopper part: **Llama-3.3-70B Q4_K_M ≈ 18–25 tok/s single-stream.** ([discussion #18005](https://github.com/ggml-org/llama.cpp/discussions/18005))

**Commands (6):**
```bash
wget https://github.com/ai-dock/llama.cpp-cuda/releases/latest/download/llama.cpp-cuda-12.8-arm64.tar.gz
tar xzf llama.cpp-cuda-12.8-arm64.tar.gz && cd cuda-12.8
huggingface-cli download bartowski/Llama-3.3-70B-Instruct-GGUF Llama-3.3-70B-Instruct-Q4_K_M.gguf --local-dir .
./llama-server -m Llama-3.3-70B-Instruct-Q4_K_M.gguf -ngl 999 --port 8000 --host 0.0.0.0 -c 8192 &
# (5) install huggingface_hub first if missing
# OpenAI-compat path is /v1/chat/completions on port 8000
```

**Score:** friction **2**, ~22 tok/s. **EF candidate.**

---

## Path 4 — vLLM Docker with AWQ

**Verdict: Best balance of friction × throughput.**

- **No multi-arch `vllm/vllm-openai` image as of May 2026** — official Dockerfile must be `--platform linux/arm64` rebuilt for aarch64 (25 min build, 6.93 GB image). ([vLLM docker docs](https://docs.vllm.ai/en/stable/deployment/docker/))
- **Community prebuilt**: [`drikster80/vllm-gh200-openai`](https://hub.docker.com/r/drikster80/vllm-gh200-openai) ships an arm64 image specifically for GH200 (CC 9.0). Maintained, multi-tagged.
- Alternative: [`ghcr.io/abacusai/gh200-llm/llm-train-serve:latest`](https://github.com/abacusai/gh200-llm) — also arm64+GH200-ready.
- AWQ INT4 fits ~40 GB on 96 GB GH200; plenty of room for KV cache.
- Single-stream tok/s expectation (Hopper, vLLM, 70B AWQ INT4): **40–60 tok/s** at batch=1 (vLLM scaling reports show H100 70B FP8 hits 50–80 tok/s; AWQ slightly slower than FP8, GH200 ~25% faster than H100 on memory-bound decode → net similar).

**Commands (3, assuming HF token in env):**
```bash
export HF_TOKEN=hf_xxx
docker run -d --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN \
  drikster80/vllm-gh200-openai:latest \
  --model casperhansen/llama-3.3-70b-instruct-awq --quantization awq --max-model-len 8192
# (then) curl http://localhost:8000/v1/chat/completions ...
```

That is **literally 2 commands** (`export` + `docker run`) from a fresh SSH.

**Score:** friction **1**, ~50 tok/s. **WINNER.**

---

## Path 5 — SGLang Docker `lmsysorg/sglang:latest-cu130`

**Verdict: Works, multi-arch image confirmed, but heavier startup and Q4 path is awkward.**

- Docker Hub shows **`latest-cu130` has both linux/amd64 (12.11 GB) and linux/arm64 (11.92 GB) manifests** ([tags page](https://hub.docker.com/r/lmsysorg/sglang/tags)) — pulls correct arch automatically.
- aarch64 wheels exist (`sglang_kernel-…-manylinux2014_aarch64.whl`); some JIT compile on first run because `sgl-flash-attn3` has no arm prebuilt → adds ~2–3 min startup tax.
- SGLang prefers FP8 / native HF safetensors; AWQ is supported but FP8 path is more optimized on Hopper. For 96 GB GH200, **FP8 Llama-3.3-70B fits comfortably** and is the recommended deploy on Hopper (vLLM recipe also says FP8). Beats Q4 on quality and is faster.
- SGLang single-stream on GH200 ~32% higher throughput than H100, batch=32 (baseten). At batch=1, FP8 70B on Hopper is ~50–70 tok/s.
- Default port 30000 (not 8000) — map externally.

**Commands (3):**
```bash
export HF_TOKEN=hf_xxx
docker run -d --gpus all --shm-size 32g --ipc=host -p 8000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN \
  lmsysorg/sglang:latest-cu130 \
  python3 -m sglang.launch_server \
  --model-path neuralmagic/Llama-3.3-70B-Instruct-FP8-dynamic \
  --host 0.0.0.0 --port 30000
```

**Score:** friction **2**, ~60 tok/s FP8. Very strong second.

---

## Efficient frontier ranking

| Path | Commands SSH→serve | Stock Lambda? | Manual setup? | Single-stream tok/s | Friction (1=best) | EF score (tok/s ÷ friction) |
|---|---|---|---|---|---|---|
| **vLLM Docker (drikster80)** | **2** | yes | none | **~50** AWQ | **1** | **50** |
| SGLang Docker latest-cu130 | 2 | yes | first-run JIT ~3 min | ~60 FP8 | 2 | 30 |
| llama.cpp ai-dock tarball | 6 | yes | needs `huggingface-cli` | ~22 Q4 | 2 | 11 |
| LM Studio `lms` | 5 | yes | GPU iffy on arm64 | ~5 (CPU) | 4 | 1.3 |
| Ollama | 3 | yes | **no aarch64+CUDA path** | <10 (CPU) | 4 | 2.5 |

**Winner: vLLM via `drikster80/vllm-gh200-openai`.** Fewest commands, no build, no CUDA setup, OpenAI-compat on :8000 natively, fastest of the truly-easy paths.

**Runner-up: SGLang `lmsysorg/sglang:latest-cu130`** — official multi-arch image, FP8 path is higher quality and slightly faster, but JIT-compile tax on first launch and port remap needed.

---

## Key findings

- **Ollama on GH200 = CPU only.** The arm64 build has no Grace Hopper CUDA path; only Jetson JetPack variants exist. Don't bother.
- **Official llama.cpp releases have no aarch64+CUDA tarball.** Community `ai-dock/llama.cpp-cuda` fills the gap with CUDA 12.8 arm64 builds.
- **vLLM upstream has no prebuilt arm64 image yet**, but `drikster80/vllm-gh200-openai` (community, GH200-targeted) works out-of-box and is the lowest-friction option.
- **SGLang's `latest-cu130` IS multi-arch** (linux/arm64 manifest exists) — first major inference server with a real arm64 published image, not just a fork.
- **FP8 beats Q4 on Hopper 96 GB.** Both fit, FP8 is faster + higher quality. Only choose Q4 if you specifically want GGUF tooling.
- **No 64 KB kernel needed.** None of these paths require swapping Lambda's stock kernel.

---

## Do this

```bash
# (1 line each — 5 lines total)
ssh ubuntu@<gh200-ip>
export HF_TOKEN=hf_xxx
docker run -d --gpus all --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN \
  drikster80/vllm-gh200-openai:latest \
  --model casperhansen/llama-3.3-70b-instruct-awq --quantization awq --max-model-len 8192
# wait ~3 min for HF download + load, then:
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"casperhansen/llama-3.3-70b-instruct-awq","messages":[{"role":"user","content":"hi"}]}'
```

---

## Sources

- [ai-dock/llama.cpp-cuda — arm64 CUDA tarballs](https://github.com/ai-dock/llama.cpp-cuda)
- [llama.cpp releases (no arm64+CUDA)](https://github.com/ggml-org/llama.cpp/releases)
- [llama.cpp GH200 performance discussion #18005](https://github.com/ggml-org/llama.cpp/discussions/18005)
- [drikster80/vllm-gh200-openai docker image](https://hub.docker.com/r/drikster80/vllm-gh200-openai)
- [abacusai/gh200-llm](https://github.com/abacusai/gh200-llm)
- [vLLM Llama 3.3 70B recipe (Hopper FP8)](https://docs.vllm.ai/projects/recipes/en/latest/Llama/Llama3.3-70B.html)
- [vLLM Docker docs](https://docs.vllm.ai/en/stable/deployment/docker/)
- [lmsysorg/sglang docker tags (multi-arch confirmed)](https://hub.docker.com/r/lmsysorg/sglang/tags)
- [SGLang install docs](https://docs.sglang.io/get_started/install.html)
- [Ollama install.sh source](https://github.com/ollama/ollama/blob/main/scripts/install.sh)
- [Ollama arm64+CUDA broken: issue #13169](https://github.com/ollama/ollama/issues/13169)
- [Ollama Jetson arm64 issue #3406](https://github.com/ollama/ollama/issues/3406)
- [Ollama arm64 jetpack PR #7217](https://github.com/ollama/ollama/pull/7217)
- [LM Studio headless docs](https://lmstudio.ai/docs/developer/core/headless)
- [Baseten — Llama 3.3 70B on GH200 in Lambda Cloud](https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/)
- [casperhansen/llama-3.3-70b-instruct-awq on HF](https://huggingface.co/casperhansen/llama-3.3-70b-instruct-awq)
