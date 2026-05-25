# Round 4 — Niche & Replacement LLM Serving Engines on GH200 aarch64

**Date:** 2026-05-25
**Scope:** Survey of LLM serving engines outside the "big four" (vLLM 0.21 / SGLang 0.5.12 / TensorRT-LLM 1.2.1 / llama.cpp) — specifically with respect to fitness on NVIDIA GH200 Grace Hopper aarch64 nodes.
**Methodology:** ~18 WebSearch / WebFetch calls against GitHub release pages, official docs, vendor blogs, and 2026 community write-ups.

---

## TL;DR Verdict Table

| Engine | aarch64 GH200 ready? | Last release (2026) | Recommend over vLLM/SGLang/TRT-LLM/llama.cpp? |
|---|---|---|---|
| **LightLLM** (ModelTC) | No native build; aarch64 untested in docs | v1.1.0 Sep 2025; main branch active through May 2026 | **No** — research playground; H200 only |
| **MLC-LLM** (mlc.ai) | Edge-targeted; CUDA path exists but no GH200 optimization | Continuous main, no tagged release surfaced | **No** — wrong tool for server LLM serving |
| **DeepSpeed-MII** | No aarch64 wheels | v0.3.3 Mar 2025 — effectively frozen | **No** — superseded by vLLM/SGLang |
| **PowerInfer / PowerInfer-2** | Not applicable — consumer GPU / smartphone | Original repo last meaningful work 2024 | **No** — wrong target |
| **ipex-llm** (Intel) | Intel-only; no NVIDIA path | Active May 2026 | **No** — doesn't run on GH200 |
| **Lorax** (Predibase) | TGI-fork; no aarch64 release | v0.12.1 Nov 2024 (no 2026 releases visible) | **Only if you need 100s of LoRA adapters** — and even then, vLLM LoRA + AIBrix is the modern path |
| **Ollama** | Yes — `ollama-linux-arm64.tar.zst` ships; CUDA on aarch64 supported | v0.24.0 stable / v0.30.0 pre-release May 2026 | **No for production** — convenience wrapper, llama.cpp backend |
| **Friendli / PerfX / TensorWave** | Friendli=closed SaaS; PerfX not found; TensorWave=AMD-focused | Friendli InferenceSense Mar 2026 (commercial) | **No** — closed-source or off-target |
| **AIBrix** (ByteDance, now vllm-project) | Multi-arch Docker (AMD+ARM); is a control plane *over* vLLM | v0.6.0 Mar 2026 | **Yes — as a control plane**, not as a replacement engine |
| **vllm-omni** | aarch64 install path explicitly skipped on container build | v0.20.0 May 2026 (paper v0.14 stable Feb 2026) | **No for LLM-only** — designed for any-to-any multimodal pipelines |

**Bottom line for your stack:** none of these displace vLLM 0.21, SGLang 0.5.12, TRT-LLM 1.2.1, or llama.cpp on a GH200 LLM-only workload. The *only* one worth considering as a complement is **AIBrix as a Kubernetes control plane in front of vLLM**, if you have orchestration / autoscaling / KV-aware routing needs.

---

## 1. LightLLM (ModelTC)

**Why it exists.** Pure-Python serving framework — easy to hack on, used as the basis for ASPLOS'25 scheduler paper. Token-level KV cache management, integrates Triton kernels, FlashAttention, FlashInfer.

**Current state (May 2026).** Last tagged release **v1.1.0 on 2025-09-03**. Main branch shows ongoing commits ("Prefix KV Cache Transfer between DP rankers" Nov 2025) — actively developed but not formally re-cut. v1.0.0 (Feb 2025) advertised "fastest DeepSeek-R1 serving on single H200." v1.1.0 added "CPU-GPU Unified Folding Architecture," autotuner for Triton, Qwen3-MoE support.

**aarch64 / GH200 status.** No mention of aarch64 or GH200 in release notes, READMEs, or DeepSeek deployment guide (the only deployment doc is H100/H200). Building on GH200 would require self-compiling its Triton/FlashInfer/Flash-Attention deps — feasible (same as vLLM aarch64 build pattern from drikster80 Docker images) but unsupported.

**Performance claims.** On 8x80GB H100, 200 CUDA graphs concurrent without OOM. No published GH200 numbers.

**Verdict.** **Skip.** vLLM and SGLang have caught up on the optimizations LightLLM pioneered (continuous batching, paged KV). If your team isn't writing scheduling research papers, there's no reason to fight an aarch64 build.

---

## 2. MLC-LLM (mlc.ai)

**Why it exists.** TVM-compiler-based universal deployment — same model description gets compiled to CUDA, Vulkan, Metal, ROCm, WebGPU, iOS, Android. Designed to maximize platform coverage, not throughput.

**Current state.** 1,799 commits on main; no clear release tag highlighted. Continues to expand model coverage. Their wheelhouse is mobile/web and Apple Silicon — not server LLM serving.

**aarch64 / GH200 status.** No first-class GH200 support. CUDA backend works in principle on sm_90 aarch64 but the project's optimization focus is edge devices. Their published GPU support matrix lists NVIDIA generically — no Grace Hopper benchmarks.

**Performance.** No 2026 server-class numbers. Independent 2026 surveys characterize MLC-LLM as "not positioned as the primary contender for raw throughput on large models like vLLM or TensorRT-LLM."

**Verdict.** **Skip for GH200 server use.** MLC-LLM is the right answer for "run Qwen on a Snapdragon phone" or "Llama in the browser" — wrong tool for a $40k Grace Hopper node.

---

## 3. DeepSpeed-MII

**Why it exists.** Microsoft DeepSpeed team's productionization of DeepSpeed-Inference + DeepSpeed-FastGen with claimed 2.3-2.5x throughput vs vLLM (circa 2024, before vLLM v0.6).

**Current state (May 2026).** Last release **v0.3.3 on 2025-03-25**. Snyk lists it "Healthy" but the cadence has slowed dramatically. Latest activity is build-infrastructure (pyproject.toml legacy backend) — no new perf features. PyPI shows v0.3.3 as current.

**aarch64 / GH200 status.** No aarch64 wheels, no GH200-specific kernel paths, no mention in release notes.

**Performance reality.** The "2.5x vs vLLM" claim is stale. Independent 2026 comparisons place it behind both SGLang and current vLLM (v0.6+). Still relevant only for teams already on the Azure/DeepSpeed stack.

**Verdict.** **Skip.** Effectively in maintenance mode. vLLM 0.21 surpasses it across the board.

---

## 4. PowerInfer / PowerInfer-2

**Why it exists.** Sparse-activation (hot/cold neuron) inference. PowerInfer (SOSP'24) targets single consumer GPU + CPU offload using power-law neuron locality. PowerInfer-2 (arxiv 2406.06282) targets **smartphones**, claims 29.2x speedup vs SOTA, runs TurboSparse-Mixtral-47B at 11.68 tok/s on a phone.

**Current state.** Original SJTU-IPADS repo last meaningful work in late 2024. No 2026 datacenter port. Several forks (Tiiny-AI, amir2pl, YuMJie) — all consumer-GPU oriented. PowerInfer-2 v3 paper updated Dec 2024, no public code release for datacenter scale.

**aarch64 / GH200 status.** **Architecturally wrong target.** PowerInfer's premise is CPU-GPU hybrid execution to compensate for tiny GPU VRAM. GH200 has 96GB HBM3 + 480GB LPDDR5X behind a 900 GB/s NVLink-C2C — exactly the system PowerInfer is trying to *emulate* on consumer hardware. The sparse-activation trick only matters when models don't fit; on GH200, Llama 3.1 70B fits with room to spare.

**Verdict.** **Skip.** Wrong problem domain. The hot/cold neuron idea is interesting but you'd need to re-architect for GH200's unified memory model.

---

## 5. ipex-llm (Intel)

**Why it exists.** Intel's LLM acceleration library for Intel CPUs (Xeon w/ AMX), Intel GPUs (Arc, Flex, Max), and Intel Gaudi. Wraps llama.cpp, Ollama, vLLM with Intel-specific kernels.

**Current state (May 2026).** Active development; Apr 2026 Medium article documents current capabilities.

**aarch64 / GH200 status.** **Does not support NVIDIA GPUs.** ipex-llm is Intel-hardware-only. Its competitive pitch is explicitly *against* NVIDIA ("repurpose existing infrastructure without requiring expensive NVIDIA GPUs").

**Verdict.** **Confirmed: irrelevant for GH200.** Listed in your survey for completeness — there's no path to running ipex-llm on a Hopper GPU.

---

## 6. Lorax (Predibase)

**Why it exists.** Multi-LoRA serving — load 100s/1000s of LoRA adapters against one base model, with dynamic adapter swapping, heterogeneous continuous batching, SGMV kernels. Originally a TGI fork.

**Current state.** Last visible tagged release **v0.12.1 on 2024-11-25**. Some 2025/2026 commit activity reported (one source claims update May 2026) but no fresh tagged releases — and TGI itself is archived, which leaves Lorax on a dead upstream. Recent v0.12 series adds prefix caching, FP8 KV, mllama, function calling.

**aarch64 / GH200 status.** No aarch64 wheels, no Docker arm64 build, no GH200 documentation. Inherits TGI-style CUDA-only build assumptions.

**vLLM has caught up.** vLLM now has native LoRA Adapters support (`docs.vllm.ai/en/stable/features/lora/`) plus AIBrix's "High-Density LoRA Management." 2026 LoRA serving benchmarks place vLLM and TGI ahead of Lorax on throughput; Lorax's edge is purely adapter density / cost model.

**Verdict.** **Skip unless** your business *requires* 100+ active adapters per replica AND you can tolerate the unsupported aarch64 build. Otherwise: vLLM LoRA + AIBrix gives you 90% of the value with first-class GH200 support.

---

## 7. Ollama

**Why it exists.** UX layer over llama.cpp. `ollama pull`, model registry, one-process server.

**Current state (May 2026).** **v0.24.0 stable, v0.30.0 pre-release, both 2026-05-13/14.** Very active.

**aarch64 / GH200 status.** **Yes — ships native arm64 binaries:**
- `ollama-linux-arm64.tar.zst` (1.23 GB, generic arm64)
- `ollama-linux-arm64-jetpack5.tar.zst` (251 MB)
- `ollama-linux-arm64-jetpack6.tar.zst` (247 MB)

CUDA on arm64 works (llama.cpp backend compiles for sm_90 aarch64). Known issue (#4705): **arm64 llama runner startup ~200s vs 3-10s on amd64**, then steady-state is fast. Inference-time perf matches llama.cpp directly.

**Performance.** Inherits llama.cpp performance. Per the llama.cpp GH200 discussion #18005, llama.cpp wins TG (token generation) on GH200; Blackwell wins PP (prompt processing). Same applies to Ollama.

**Verdict.** **Use directly only for dev/prototyping.** Wrapping llama.cpp adds nothing for production throughput — you'd be paying the wrapper overhead and slow startup. If you want llama.cpp, run llama.cpp.

---

## 8. Friendli / PerfX / TensorWave (vendor stacks)

### Friendli (FriendliAI)
- **Closed-source SaaS.** Continuous-batching pioneers (the team that invented it). Custom kernel library, FP8/INT8/AWQ.
- Mar 2026: launched **Friendli InferenceSense** — a *monetization* platform for idle GPU capacity.
- "Up to 3x faster than vLLM, 50-90% cost savings vs closed APIs" — vendor claims.
- **Aligned with NVIDIA, optimized for Blackwell** (per Cerebral Valley interview).
- **No open-source release.** Available as API / cloud only.

### PerfX
- **Not found.** No project, blog, or release surfaced under that name in May 2026 search results.

### TensorWave
- AMD MI300X cloud provider, not a serving engine vendor. Their open-source contribution is **ScalarLM** — but ScalarLM's inference path "inherits vLLM's hardware support," so it *is* vLLM underneath.

**Verdict.** **No open release worth evaluating.** Friendli is the only credible one; you can rent it but can't deploy it on your own GH200.

---

## 9. AIBrix (ByteDance → vllm-project)

**Why it exists.** Kubernetes-native control plane *for* vLLM. ByteDance contributed it to the vLLM org in Feb 2025. Provides LLM-aware autoscaling, distributed KV cache, prefix-aware routing, high-density LoRA management, P/D disaggregation orchestration, intelligent gateway routing (Envoy Gateway based).

**Current state (May 2026).** Latest release **v0.6.0 on 2026-03-05**. Active. Previous: v0.5.0 Nov 2025, v0.4.0 Aug 2025, v0.3.0 May 2025.

**aarch64 / GH200 status.** **Multi-arch Docker builds (AMD + ARM)** — v0.3.0 release notes explicitly "Supports multi-arch (AMD, ARM) Docker builds" and v0.3.0-rc.2 added "Upload arm build images." Since AIBrix is a control plane, the inference engine underneath (vLLM) provides the actual aarch64 GH200 path. **Compatibility with vanilla Kubernetes confirmed.**

**Production track record.** ByteDance reports 79% P99 latency reduction, 4.7x cost reduction in low-traffic periods.

**Verdict.** **The one engine on this list worth evaluating for your stack.** It's not a replacement for vLLM — it's a force multiplier when you need:
- Multi-replica autoscaling on Kubernetes
- KV-aware request routing across replicas
- 100s of LoRA adapters managed dynamically
- P/D disaggregation orchestration
- Cost-efficient mixed-priority workloads

If you're running a single vLLM process on one GH200, you don't need it. If you're running a fleet, this is the production answer in 2026.

---

## 10. vllm-omni

**Why it exists.** vLLM extension for any-to-any multimodal serving (text + image + video + audio + diffusion + TTS). Released Nov 2025 (vLLM blog announcement); paper arxiv 2602.02204. "Fully disaggregated serving" — stages of the omni pipeline run on independent engines connected via inter-stage routers. Claims up to 91.4% JCT reduction vs naive serving for multimodal pipelines.

**Current state (May 2026).** v0.14.0 first stable Feb 2026, v0.16.0 expanded diffusion/image-video, **v0.20.0 May 2026** refreshes runtime stack.

**aarch64 / GH200 status.** **Explicitly broken on aarch64.** Per FAQ: "vLLM-Omni is currently only installed on amd64 builds — on arm64, the container build skips the install and vLLM-Omni features are unavailable." Same constraint reported across community sources.

**Relevance to LLM-only.** None. The entire value proposition is orchestrating heterogeneous stages — autoregressive LLMs + diffusion transformers + audio codecs. For pure text LLM serving you'd use vanilla vLLM.

**Verdict.** **Not relevant to LLM-only on GH200.** And even for multimodal: aarch64 is unsupported.

---

## Bonus context: llm-d (Red Hat, related to your survey)

Red Hat's distributed inference project (vLLM-based, Kubernetes-native, NVIDIA Dynamo integrated). Notable because **the ARM64/GH200/GB200 support request (issue #281, Sep 2025) is still in "Todo" status as of May 2026** — no assignee, no PR, no progress. This is a leading indicator: even Red Hat's tier-1 distributed inference project hasn't formalized GH200 aarch64 support yet, which explains why most engines in this survey haven't either.

---

## Summary Recommendation

For a GH200 aarch64 LLM-only deployment in May 2026, **the big four remain the right answer**:
- **Throughput:** vLLM 0.21 or SGLang 0.5.12
- **NVIDIA-optimized:** TensorRT-LLM 1.2.1 (with the caveat that TRT-LLM wheels are still x86_64-only — must build from source on aarch64)
- **CPU offload / quantized:** llama.cpp

**The one engine worth adding to that list: AIBrix**, as a Kubernetes control plane in front of vLLM, when fleet-scale orchestration becomes the bottleneck.

Everything else in this survey is either:
- Frozen / superseded (DeepSpeed-MII, Lorax)
- Wrong hardware target (PowerInfer, ipex-llm)
- Wrong problem domain (vllm-omni, MLC-LLM)
- Closed-source (Friendli)
- A llama.cpp wrapper that adds startup overhead (Ollama)
- A research playground without aarch64 support (LightLLM)

---

## Sources

- [LightLLM GitHub releases](https://github.com/ModelTC/lightllm/releases)
- [LightLLM repo](https://github.com/ModelTC/LightLLM)
- [LightLLM DeepSeek deployment guide](https://lightllm-en.readthedocs.io/en/latest/tutorial/deepseek_deployment.html)
- [MLC-LLM repo](https://github.com/mlc-ai/mlc-llm)
- [MLC-LLM installation docs](https://llm.mlc.ai/docs/install/gpu.html)
- [DeepSpeed-MII releases](https://github.com/deepspeedai/DeepSpeed-MII/releases)
- [DeepSpeed-MII repo](https://github.com/deepspeedai/DeepSpeed-MII)
- [Snyk DeepSpeed-MII health](https://snyk.io/advisor/python/deepspeed-mii)
- [PowerInfer arXiv 2312.12456](https://arxiv.org/abs/2312.12456)
- [PowerInfer-2 arXiv 2406.06282](https://arxiv.org/abs/2406.06282)
- [SJTU-IPADS PowerInfer repo](https://github.com/SJTU-IPADS/PowerInfer)
- [Intel ipex-llm](https://github.com/intel/ipex-llm)
- [IPEX-LLM Medium Apr 2026](https://dheerajbhumanapalli.medium.com/ipex-llm-accelerating-large-language-models-on-intel-hardware-83382a7fe122)
- [Lorax releases](https://github.com/predibase/lorax/releases)
- [LoRAX vs vLLM comparison](https://aicoolies.com/comparisons/lorax-vs-vllm)
- [Ollama releases](https://github.com/ollama/ollama/releases)
- [Ollama arm64 startup issue #4705](https://github.com/ollama/ollama/issues/4705)
- [llama.cpp GH200 perf discussion #18005](https://github.com/ggml-org/llama.cpp/discussions/18005)
- [FriendliAI homepage](https://friendli.ai/)
- [FriendliAI InferenceSense Mar 2026](https://www.hpcwire.com/aiwire/2026/03/13/friendliai-launches-inferencesense-to-monetize-idle-gpu-capacity/)
- [FriendliAI Cerebral Valley interview](https://cerebralvalley.beehiiv.com/p/friendliai-the-inference-engine-behind-vllm)
- [TensorWave ScalarLM blog](https://tensorwave.com/blog/scalarlm-open-source-llm-training-inference-on-amd-rocm)
- [AIBrix GitHub](https://github.com/vllm-project/aibrix)
- [AIBrix releases](https://github.com/vllm-project/aibrix/releases)
- [AIBrix blog release intro](https://blog.vllm.ai/2025/02/21/aibrix-release.html)
- [AIBrix arXiv 2504.03648](https://arxiv.org/html/2504.03648v1)
- [AIBrix Read the Docs](https://aibrix.readthedocs.io/latest/getting_started/installation/installation.html)
- [vllm-omni repo](https://github.com/vllm-project/vllm-omni)
- [vllm-omni arXiv 2602.02204](https://arxiv.org/abs/2602.02204)
- [vLLM-Omni announcement blog](https://blog.vllm.ai/2025/11/30/vllm-omni.html)
- [vllm-omni FAQ — aarch64 limitation](https://docs.vllm.ai/projects/vllm-omni/en/stable/usage/faq/)
- [llm-d ARM64 support issue #281](https://github.com/llm-d/llm-d/issues/281)
- [vLLM aarch64 GH200 install issue #10459](https://github.com/vllm-project/vllm/issues/10459)
- [drikster80/vllm-gh200-openai Docker image](https://hub.docker.com/r/drikster80/vllm-gh200-openai)
- [How to serve DeepSeek-R1 on GH200 (Lambda)](https://lambda.ai/blog/how-to-serve-deepseek-r1-v3-on-gh200)
- [NVIDIA NIM arm64 DGX prerequisites](https://docs.nvidia.com/nim/bionemo/openfold3/1.2.0/prerequisites-arm64-with-dgx-stack.html)
