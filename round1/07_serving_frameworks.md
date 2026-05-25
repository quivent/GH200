# GH200 Inference — Round 1, Agent #07: Production Serving Frameworks

**Author**: Research Agent #07 of 20
**Date**: 2026-05-25
**Scope**: NVIDIA NIM, NVIDIA Triton, NVIDIA Dynamo, Modular MAX, Ray Serve, LitServe — on GH200 aarch64 (Grace Hopper), May 2026
**Stack constraint**: FP8 broken on this GH200 stack; BF16 default. FP8 claims flagged below.

---

## TL;DR

| Framework | Runs on GH200 aarch64? | Effort | Disagg P/D | License/Cost | Verdict (May 2026) |
|---|---|---|---|---|---|
| **NVIDIA NIM (LLM)** | Partially — most LLM containers are still `linux/amd64` only on NGC; NVIDIA's official position (Nov 2024) is "no ARM"; isolated SBSA NIMs exist (OpenFold3, some retriever/safety) but Llama 3.1 8B/70B NIMs still failed on GH200 as of latest user reports | High (workaround = run TRT-LLM / vLLM raw) | Not in NIM itself | NVIDIA AI Enterprise required for prod: ~$4,500/GPU/yr (or ~$1/GPU/hr cloud); free for ≤16 GPUs dev | Wait or skip. Use raw engines |
| **NVIDIA Triton 26.04** | **Yes via SBSA arm64 container** (Ubuntu 24.04, CUDA 13.2, sm_90 supported). Backends: TensorRT-LLM 1.2.1, vLLM 0.19.0, Python, PyTorch, ONNX, TRT, FIL | Medium | No native P/D — use Dynamo on top, or `tensorrtllm_backend` orchestrator features | Open source (Apache 2.0); AI Enterprise optional for support | **Solid choice** for single-node multi-model serving |
| **NVIDIA Dynamo 1.1.x / 1.2-dev** | **Yes — multi-arch arm64 containers exist** (vLLM-runtime CUDA 13 is multi-arch; SGLang CUDA-13 arm64 image shipped April–May 2026). Support matrix lists Blackwell/Hopper/Ada/Ampere (GH200 is Hopper-class GPU; works in practice). Caveats: vLLM-Omni and some multimodal paths AMD64-only on arm64 | Medium-High | **Yes — flagship feature.** xPyD prefill/decode disaggregation with NIXL KV transport (NVLink/IB/UCX), KV-cache-aware router, runtime-reconfigurable | Open source (Apache 2.0); no license fee | **Best-in-class for routing & P/D** at multi-GPU scale |
| **Modular MAX 25.7 / 26.3** | **Yes — BF16 on ARM CPU hosts (GH200/GB200) since 25.7 (2025-11-20).** v26.3 stable 2026-05-07. Containers `modular/max-nvidia-full` / `max-nvidia-base` — arm64 image availability for these tags not explicitly documented; install via `uv pip install modular` works on aarch64 | Low (pip) to Medium (container) | No first-party P/D split | Apache-style/open (MAX & Mojo open-sourced 2025) | **Strong contender** for BF16-only stack; check container arch before relying on it |
| **Ray Serve LLM** (Ray 2.55) | Yes via pip + vLLM/SGLang on the box. No NVIDIA-published arm64 container, but Ray itself runs fine on aarch64; pair with vLLM aarch64 build (drikster80/abacusai images) | Medium | **Yes — `PDPrefillServer` / `PDDecodeServer` deployments**, NIXL/UCX/EFA KV transfer connectors, prefix+session-aware routing | Open source (Apache 2.0) | **Best Python-native multi-node orchestration**; needs you to BYO arm64 vLLM image |
| **LitServe 0.2.17** | Yes — pure Python, runs anywhere PyTorch arm64 wheels run | Low | No | Open source | Good for **custom inference handlers / non-LLM** workloads; not the right tool for KV-router-heavy LLM serving |

---

## 1. NVIDIA NIM on GH200 — still painful

- **Official NVIDIA statement (forum, 2024-11-07)** on Llama-3.1-8B/70B NIMs on GH200: *"Unfortunately NIM does not support deployment on ARM platforms at the moment, including Grace CPUs. We're working to close this gap in future releases."* Docker pull fails with `linux/amd64 != linux/arm64/v8`. ([forum](https://forums.developer.nvidia.com/t/nim-does-not-support-llama-3-1-8b-instruct-and-llama-3-1-70b-instruct-on-gh200-on-prem-deployment/312542))
- **Status May 2026**: still incomplete. The DGX Spark / GB10 forum thread (active 2025-2026) lists Llama 2 70B, several embedding/translation NIMs as missing native arm64 images; users are directed to TRT-LLM / vLLM / SGLang instead. ([forum](https://forums.developer.nvidia.com/t/missing-official-native-arm64-nim-images-for-essential-ai-models/350681))
- **Where NIM-on-arm64 does work**: a handful of bionemo NIMs (OpenFold3 has explicit `prerequisites-arm64-with-dgx-stack`), some retriever / safety-guard NIMs.
- **Support matrix lists GH200** (144 GB HBM3e and 480 GB variants) as supported hardware on paper — but that lists the *GPU*, not whether the container image is published for `linux/arm64`. Multiple 2026 third-party guides repeat that NIM is "tested on GH200" while the actual NGC tags remain amd64. ([lucaberton 2026 support matrix](https://lucaberton.com/blog/nvidia-nim-support-matrix-models-gpus-profiles-2026/), [NIM matrix docs](https://docs.nvidia.com/nim/large-language-models/latest/support-matrix.html))

### Licensing/cost (NIM)
- **NVIDIA AI Enterprise** required for production NIM: **~$4,500/GPU/year** subscription, or **~$1/GPU/hour** on cloud marketplaces. Perpetual + 5yr support also available. ([NVAIE pricing guide](https://resources.nvidia.com/en-us-sustainable-computing-microsite/en-us-nvidia-ai-enterprise/nvaie-pricing-guide), [costbench 2026](https://costbench.com/software/llm-api-providers/nvidia-nim/))
- **Free for dev**: NVIDIA Developer Program members get NIM containers free for evaluation on **≤16 GPUs**; post-GTC 2026 evaluation/dev no longer requires AI Enterprise.
- FP8 NIM profiles exist for H100/H200/GH200 in the support matrix — **DO NOT trust on this stack** (FP8 broken). Force BF16 profile.

### Deployment (when amd64-compatible model is needed — not GH200 path)
```bash
# Will FAIL on GH200 for Llama 3.1 today:
docker run --gpus all -p 8000:8000 \
  nvcr.io/nim/meta/llama-3.3-70b-instruct:latest
# -> exec format error / platform mismatch on arm64
```

---

## 2. NVIDIA Triton Inference Server 26.04 — solid on GH200

- **ARM SBSA container is officially published** for aarch64 (Ubuntu 24.04, CUDA 13.2.1, Python 3.12, TRT 10.16.1.11, vLLM 0.19.0, TRT-LLM 1.2.1). The SBSA image targets Grace-class servers including GH200. ([release notes 26.04](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-04.html))
- **Compute capability 7.5+** supported → covers Hopper sm_90 / GH200.
- **Backends**: TensorRT-LLM, vLLM, Python, PyTorch, ONNX Runtime, TensorRT, FIL, DALI.
- **LLM features**: constrained decoding, function calling, speculative decoding, OpenAI-compatible API, KServe API, in-process C/Python/Java.
- **Known issues that bite on GH200**: vLLM backend in Triton does **not** support `tensor-parallel-size > 1` with default `distributed_executor_backend` under explicit model control. TRT-LLM backend may core-dump at shutdown (cosmetic). ARM SBSA client wheel **not on PyPI** — must extract from SBSA SDK image. ([release notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-04.html))
- **Newer arm tags lag**: e.g. for Jetson, 26.03/26.04 didn't publish new artifacts; use 26.02 / v2.66.0 — for SBSA GH200, 26.04 SBSA is current per release notes.

### Deployment
```bash
# Pull SBSA (arm64) container — Triton 26.04
docker pull nvcr.io/nvidia/tritonserver:26.04-py3-sbsa
# Or for vLLM backend specifically:
docker pull nvcr.io/nvidia/tritonserver:26.04-vllm-python-py3-sbsa

docker run --gpus all --rm -p8000-8002:8000-8002 \
  -v $(pwd)/model_repo:/models \
  nvcr.io/nvidia/tritonserver:26.04-py3-sbsa \
  tritonserver --model-repository=/models
```
*Force BF16 in your model config — FP8 broken on this stack.*

---

## 3. NVIDIA Dynamo 1.1.x / 1.2-dev — best routing + P/D, multi-arch arm64

- **GA**: Dynamo 1.0 went GA **March 16, 2026** at GTC. ([Spheron Dynamo guide](https://www.spheron.network/blog/nvidia-dynamo-disaggregated-inference-guide/), [GitHub releases](https://github.com/ai-dynamo/dynamo/releases))
- **Latest**: 1.1.1 (2026-05-09 patch — TRT-LLM scheduler/KV-reuse deadlock fix); 1.1.0 (2026-05-04 — resilient KV routing, Anthropic Messages API); 1.2.0-deepseek-v4-dev.3 (2026-05-09 — DSv4 stack + GB200 recipes).
- **Backends**: vLLM, SGLang, TensorRT-LLM — all three feature-complete (disagg, KV-aware routing, SLA planning, multimodal on TRT-LLM).
- **GH200 / aarch64 status**:
  - **Multi-arch ARM64 containers exist on NGC**. vLLM-runtime CUDA-13 image is **multi-arch (amd64 + arm64)**; SGLang containers split — CUDA-12.9 amd64, **CUDA-13 arm64** (arm64-only on CUDA-13).
  - **Caveat**: official support matrix lists Blackwell/Hopper/Ada/Ampere *architectures* — GH200 is Hopper GPU class and works, but is not enumerated by SKU name. ([Dynamo support matrix](https://docs.nvidia.com/dynamo/dev/resources/support-matrix))
  - **Known limitation**: vLLM-Omni and parts of the multimodal stack are amd64-only — multimodal on arm64 needs TensorRT-LLM backend.
  - A vLLM-runtime 1.0.1 ARM64 + A100 binary-incompat bug (CuPy + missing kernel images for sm_80) was opened 2026-03-24 and **closed as not planned** ([issue #7594](https://github.com/ai-dynamo/dynamo/issues/7594)). sm_90 / GH200 specifically not implicated, but it's a signal that early arm64 images had rough edges.

### Disaggregated Prefill/Decode (the headline feature)
- **NIXL KV-cache transport**: direct GPU↔GPU over NVLink, InfiniBand/UCX, or TCP fallback. Non-blocking.
- **PrefillRouter**: dynamic xPyD (x prefill workers, y decode workers), runtime-reconfigurable.
- **KV-cache-aware router** decisions = `prefill_cost(new_blocks) + decode_cost(active_blocks)` − `overlap_score × overlap_blocks`, plus per-worker queue load (FCFS/WSPT/LCFS).
- **Reported gains** (NVIDIA, Blackwell DeepSeek-R1): up to **30×** requests served vs non-disagg baseline; up to **7×** throughput vs monolithic on a Llama-3.1-70B FP8 reference (**FP8 — flag: do NOT replicate on this stack**). Numerical scale will be lower on Hopper GH200 with BF16; treat NVIDIA marketing numbers as upper-bound ceilings.

### Deployment (GH200 aarch64, BF16, two-node disagg)
```bash
# Pull multi-arch vLLM runtime (CUDA 13)
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1

# Prefill worker(s) (e.g. node A)
python3 -m dynamo.vllm \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --dtype bfloat16 \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

# Decode worker(s) (e.g. node B)
python3 -m dynamo.vllm \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --dtype bfloat16 \
  --disaggregation-mode decode \
  --kv-transfer-config '{"kv_connector":"NixlConnector"}'

# Frontend / router
python3 -m dynamo.frontend --port 8000 \
  --router-mode kv \
  --router-kv-overlap-score-weight 1.0 \
  --router-prefill-load-model aic
```

### State-of-the-art P/D landscape (May 2026)
- All four major engines support disagg at varying maturity: **vLLM (experimental in vLLM core, prod-tested at Meta/HuggingFace), SGLang (Mooncake + NIXL backends), TensorRT-LLM (full), and via Dynamo on all three**. ([Groundy](https://groundy.com/articles/prefill-decode-disaggregation-the-architecture-shift-redefining-llm-serving-at-scale/), [BentoML handbook](https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation))
- **llm-d** (CNCF Sandbox 2026-03-24, backed by NVIDIA/CoreWeave/AMD/HF/Intel) — Kubernetes-native alternative, validated 3.1k tok/s/B200 decode and 50k out-tok/s on 16×16 B200 P/D topology. Multi-accelerator (NVIDIA/AMD/Gaudi). ARM/GH200 support still requested (issue #281), not flagship. ([llm-d.ai](https://llm-d.ai/), [Spheron guide](https://www.spheron.network/blog/llm-d-kubernetes-disaggregated-inference-guide/))
- **DuetServe (arxiv:2511.04791)** — research direction harmonizing P/D rather than splitting. Not production.
- **Important caveat from the literature**: disagg can be **20–30% slower** on small/poorly-tuned workloads, or when decode worker has high prefix-cache hit rate. Don't disaggregate by default; benchmark monolithic vs disagg per workload.

---

## 4. Modular MAX 25.7 / 26.3 — BF16 GH200 first-class since Nov 2025

- **25.7 release (2025-11-20)** explicitly added: *"Added support for bfloat16 models running on GPUs with ARM-based CPU hosts, such as Grace Hopper (GH200) and Grace Blackwell (GB200)."* This is **directly aligned with the FP8-broken / BF16-only constraint of this stack** — Modular's GH200 path is BF16 by design.
- **v26.3 stable (2026-05-07)**, nightly v26.4 in progress.
- **Reported perf**: +30–80% throughput on Qwen2.5-VL vs 25.6; ">2× vLLM" on vision models (Modular's own benchmark).
- **GPU recommendation in docs**: B200 / H200 / H100 / MI355X / MI325X / MI300X. GH200 not on the marquee list but explicitly called out as supported via the 25.7 ARM-host-bf16 work.
- **Install** (aarch64-friendly):
  ```bash
  uv pip install modular --index https://whl.modular.com/nightly/simple/
  # or
  pixi add modular
  ```
- **Containers**: `modular/max-nvidia-full`, `max-nvidia-base`, `max-amd`, `max-full`. **Public arm64 image availability is not explicitly documented** in MAX container docs — verify with `docker manifest inspect modular/max-nvidia-full:latest` before relying on it. If no arm64 image, fall back to pip install.
- **MAX serve** (OpenAI-compatible):
  ```bash
  max serve --model-path meta-llama/Llama-3.1-70B-Instruct --dtype bfloat16
  ```
- **MAX is fully open-source** (Apache-style; Mojo + MAX Python API open-sourced 2025).
- **No first-party P/D disaggregation** as of 26.3; for routing/disagg you'd front MAX instances with Dynamo or an external router.

---

## 5. Ray Serve LLM (Ray 2.55) — Python-native multi-node

- **`PDPrefillServer` / `PDDecodeServer` deployments** are documented and stable as a serving pattern. KV transfer via vLLM's NIXLConnector (UCX/libfabric/EFA backends). Prefix-aware + session-aware routing built in. ([Ray docs](https://docs.ray.io/en/latest/serve/llm/architecture/serving-patterns/prefill-decode.html))
- **Pipeline/tensor/expert/data parallelism**, multi-LoRA, autoscaling, Grafana dashboards.
- **Engines**: vLLM, SGLang, others via adapter.
- **aarch64/GH200**: Ray's own packages install fine on aarch64 (pure Python + small Rust crates). NVIDIA does **not publish a Ray-Serve-on-GH200** container; standard path is your own image starting from a vLLM aarch64 build:
  - Community arm64/GH200 vLLM images: `drikster80/vllm-gh200-openai`, `drikster80/vllm-aarch64-openai`, `ghcr.io/abacusai/gh200-llm/llm-train-serve:latest` (multi-arch).
  - vLLM upstream PR #10499 added ARM64/GH200 dockerfile support.
- **Anyscale managed offering** (commercial) bundles Ray Serve LLM with disagg + wide-EP recipes.

### Deployment sketch
```bash
pip install "ray[serve,llm]"   # aarch64 wheels exist
# Then deploy a PD-disagg config
ray start --head
python serve_pd.py   # builds PDPrefillServer + PDDecodeServer with vLLM engines
```

---

## 6. LitServe 0.2.17 — minimal Python inference server

- Pure-Python, FastAPI-style "write your own inference server" framework. Runs anywhere PyTorch arm64 wheels run → **fine on GH200**.
- Latest **v0.2.17** added Automatic Worker Restart. No GH200- or ARM-specific work in 2026 releases, but also nothing arch-specific to break.
- **No KV-aware routing, no P/D disagg, no OpenAI-style LLM-batch features out of the box** — it's a generic model server. Good for non-LLM models (CV, audio, RAG glue, custom multimodal); bad fit for high-throughput LLM serving where Dynamo/Triton/Ray-Serve dominate.
- Free (Apache 2.0). Managed Lightning Cloud is the upsell.

---

## Comparison & recommendation table for GH200 BF16 LLM serving

| Need | First choice | Notes |
|---|---|---|
| Single-node, multi-model, mixed (LLM + ONNX + CV) | **Triton 26.04 SBSA** | Official arm64 container, broad backend matrix |
| Multi-GPU LLM with P/D + KV-aware routing | **Dynamo 1.1.x** (vLLM or TRT-LLM backend) | Multi-arch arm64; force BF16 |
| Python-orchestrated multi-node Ray cluster | **Ray Serve LLM** + community arm64 vLLM image | Best Python ergonomics |
| BF16-first, want highest perf on Hopper without FP8 | **Modular MAX 25.7+** | Verify arm64 container; pip install fallback |
| Drop-in OpenAI endpoint, low ops | **NIM (when amd64)** / **Triton SBSA + OpenAI frontend** | Skip NIM on GH200 until images ship arm64 |
| Custom non-LLM inference handlers | **LitServe** | Minimal, flexible |
| Kubernetes-native fleet, multi-vendor accelerators | **llm-d** (not GH200 today; on roadmap) | Watch CNCF Sandbox progress |

---

## FP8 flags (per stack constraint — FP8 broken on tested GH200 stack)

- **NIM**: FP8 profiles enumerated in support matrix for GH200 — *do not use*; force BF16/FP16 profile if image ever becomes available.
- **Triton**: backends (TRT-LLM, vLLM) accept FP8 weights — *do not deploy FP8 on this stack*.
- **Dynamo published P/D throughput claims (7×, 30×)** are largely on **FP8 Blackwell** — treat as ceiling, not target. Expect smaller gains on BF16 GH200.
- **Modular 25.7 GH200 announcement is BF16-specific** — *aligned* with this stack.
- **Spheron FP8 quantization study** claims 1.4–1.8× tok/s for BF16→FP8 on Llama-3.1-70B at large batch — *do not pursue on this stack*.

---

## Sources (with dates)

- NIM forum, GH200 Llama 3.1 (NVIDIA official 2024-11-07): https://forums.developer.nvidia.com/t/nim-does-not-support-llama-3-1-8b-instruct-and-llama-3-1-70b-instruct-on-gh200-on-prem-deployment/312542
- Missing ARM64 NIM images (active 2025–2026): https://forums.developer.nvidia.com/t/missing-official-native-arm64-nim-images-for-essential-ai-models/350681
- NIM support matrix (latest 2026): https://docs.nvidia.com/nim/large-language-models/latest/support-matrix.html
- NIM 2026 matrix third-party summary: https://lucaberton.com/blog/nvidia-nim-support-matrix-models-gpus-profiles-2026/
- NVIDIA AI Enterprise pricing 2026: https://resources.nvidia.com/en-us-sustainable-computing-microsite/en-us-nvidia-ai-enterprise/nvaie-pricing-guide
- NIM pricing 2026 (third-party): https://costbench.com/software/llm-api-providers/nvidia-nim/
- Triton 26.04 release notes (2026): https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-04.html
- Triton releases (GitHub): https://github.com/triton-inference-server/server/releases
- Dynamo overview/blog (NVIDIA): https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/
- Dynamo 1.0 GA + deployment guide (Spheron, Mar 2026): https://www.spheron.network/blog/nvidia-dynamo-disaggregated-inference-guide/
- Dynamo disaggregated serving design doc: https://docs.dynamo.nvidia.com/dynamo/design-docs/disaggregated-serving
- Dynamo KV-aware routing doc: https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing
- Dynamo support matrix: https://docs.nvidia.com/dynamo/dev/resources/support-matrix
- Dynamo releases (May 2026): https://github.com/ai-dynamo/dynamo/releases
- Dynamo arm64 binary incompat issue (closed 2026): https://github.com/ai-dynamo/dynamo/issues/7594
- NGC Dynamo vLLM runtime: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime
- Modular 25.7 announcement (2025-11-20): https://www.modular.com/blog/modular-25-7-faster-inference-safer-gpu-programming-and-a-more-unified-developer-experience
- Modular changelog: https://docs.modular.com/max/changelog/
- Modular MAX container docs: https://docs.modular.com/max/container/
- Ray Serve LLM index (2.55): https://docs.ray.io/en/latest/serve/llm/index.html
- Ray Serve P/D disaggregation: https://docs.ray.io/en/latest/serve/llm/architecture/serving-patterns/prefill-decode.html
- Anyscale wide-EP + disagg blog: https://www.anyscale.com/blog/ray-serve-llm-anyscale-apis-wide-ep-disaggregated-serving-vllm
- LitServe releases: https://github.com/Lightning-AI/litserve/releases
- llm-d CNCF Sandbox acceptance (2026-03-24): https://www.cncf.io/blog/2026/03/24/welcome-llm-d-to-the-cncf-evolving-kubernetes-into-sota-ai-infrastructure/
- llm-d ARM support request (issue #281): https://github.com/llm-d/llm-d/issues/281
- llm-d deployment guide (Spheron 2026): https://www.spheron.network/blog/llm-d-kubernetes-disaggregated-inference-guide/
- P/D disagg state of art (Groundy 2026): https://groundy.com/articles/prefill-decode-disaggregation-the-architecture-shift-redefining-llm-serving-at-scale/
- BentoML LLM inference handbook P/D: https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation
- Disaggregated inference 18 months later (Hao AI Lab): https://haoailab.com/blogs/distserve-retro/
- vLLM PR #10499 GH200 dockerfile: https://github.com/vllm-project/vllm/pull/10499
- Community GH200 vLLM image: https://hub.docker.com/r/drikster80/vllm-gh200-openai
- abacusai GH200 multi-arch image: https://github.com/abacusai/gh200-llm
- NVIDIA deploying disaggregated workloads on K8s (NVIDIA blog): https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/
- DuetServe paper (Nov 2025): https://arxiv.org/pdf/2511.04791

---

## Confidence & Gaps

**High confidence**:
- NIM on GH200 is still effectively unavailable for the headline LLM models as of May 2026 — confirmed by NVIDIA's own forum reply and current arm64 image gap.
- Triton 26.04 has an official SBSA arm64 container that supports GH200 (sm_90, CUDA 13.2) with vLLM/TRT-LLM/Python backends.
- Dynamo has multi-arch arm64 images for vLLM-runtime (CUDA 13) and SGLang-runtime (CUDA 13 arm64), shipped April–May 2026.
- Modular MAX 25.7 (Nov 2025) explicitly added BF16-on-Grace-Hopper as a release-note feature; 26.3 is current stable (May 2026).
- NIM production licensing is $4,500/GPU/yr via NVIDIA AI Enterprise; dev is free up to 16 GPUs.
- All four engines (vLLM/SGLang/TRT-LLM/Dynamo-on-top) ship P/D disagg of varying maturity. Dynamo is the most full-featured router in this stack.

**Medium confidence**:
- The Modular `max-nvidia-full` container's arm64 manifest availability — pip install path is the safe assumption; container needs verification with `docker manifest inspect` on the actual host.
- The exact set of Dynamo vLLM-runtime tags that are multi-arch vs amd64-only — there's evidence of both (vLLM-Omni features are amd64-only on arm64; CUDA-13 base images are multi-arch). The matrix changes per-release; check the NGC tag manifest at deploy time.
- llm-d on GH200 — the upstream issue #281 (ARM expansion) is still open; assume "not today" without further confirmation.

**Low confidence / gaps**:
- Quantitative GH200 BF16 disagg throughput numbers from NVIDIA (Dynamo benchmarks were primarily FP8 Blackwell DeepSeek-R1; equivalent BF16 Hopper numbers were not surfaced).
- Whether Triton's vLLM backend tensor-parallel limitation has been patched in 26.05/26.06 (latest confirmed release info was 26.04).
- LitServe-on-GH200 has zero published benchmarks; functional but unmeasured.
- NIM "tested on GH200" claim in support matrix vs reality of amd64-only container tags — the matrix says yes, the docker pull says no for Llama 3.1; treat the matrix as forward-looking.
- Modular MAX container arm64 tag — could not confirm a published arm64 manifest from docs alone.

**Recommended Round 2 follow-ups**:
1. Pull and inspect actual NGC container manifests (`docker manifest inspect`) for: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1`, `nvcr.io/nvidia/tritonserver:26.04-py3-sbsa`, `modular/max-nvidia-full:latest`. This resolves the arm64-availability ambiguity for sure.
2. Find NVIDIA-published BF16 GH200 disagg benchmarks (likely sparse — most marketing is FP8 Blackwell).
3. Check Dynamo 1.1.x KV-aware router behavior under GH200's coherent HBM3e + LPDDR5X unified memory — is NIXL exploiting C2C bandwidth (900 GB/s) for KV staging, or is it forced through inter-node IB only?
4. Hands-on test: does Triton 26.04 SBSA + vLLM backend + Llama-3.1-70B-BF16 actually serve on a single GH200 144 GB? Resolves the matrix-vs-reality question.
