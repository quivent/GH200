# TensorRT-LLM on GH200 — Round 3 Verification & Deepening (Agent 02)

**Date compiled:** 2026-05-25
**Prior doc verified:** `/home/ubuntu/research/gh200_inference/round1/02_tensorrt_llm.md`
**Constraint reaffirmed:** FP8 broken on this team's DiT stack — bf16 is the default for GH200 inference. FP8 evidence below shows the *LLM* FP8 situation is also fragile, not just DiT.

---

## What changed since Round 1

Round 1 was directionally correct but had a few concrete inaccuracies and missed two important nuances. Net delta:

1. **Latest stable is v1.2.1, not v1.2.0.** Round 1 listed v1.2.0 as the latest stable. v1.2.1 shipped **2026-04-20** with two fixes: KV cache corruption (#12770) and an xgrammar/flashinfer upgrade (#12811). Source: `gh release list --repo NVIDIA/TensorRT-LLM` confirms v1.2.1 carries the "Latest" tag while v1.3.0rc15 (2026-05-21) is still pre-release.
2. **v1.3.0 GA has NOT shipped.** Confirmed via GitHub release list as of 2026-05-25 — every 1.3.0 entry is `rc1` through `rc15`, all marked pre-release. Round 1 implied 1.3.0 was the active release line; it's still the RC line. GA target remains unannounced.
3. **Base container in v1.2.0 is `nvcr.io/nvidia/pytorch:25.12-py3`**, not `25.10` (which is what 1.1.x used). Confirmed from v1.2.0 release notes verbatim: "The base Docker image for TensorRT-LLM is updated to nvcr.io/nvidia/pytorch:25.12-py3."
4. **GH issue #4218 (FP8 KV cache Qwen2.5 gibberish) was closed 2025-05-16 — but not fixed.** The maintainer (`brb-nv`) punted the bug to the NVIDIA ModelOpt repo and the reporter closed it with "Will repost there." **The bug class still exists**; ownership just moved. Round 1 listed it as "still showing bug label" — that's stale. Current state: closed-without-fix, root cause is ModelOpt's auto-enabling `lm_head` quantization when KV cache FP8 is requested.
5. **GH issue #8974 (AutoDeploy FP8/NVFP4 Flux kernel replacement) was closed 2026-04-28 — "not actively worked on."** That's an abandonment closure, not a resolution closure. Quote from `lucaslie` (NVIDIA): "closing this as it's not actively worked on". Round 1 listed it as "Late 2025 / early 2026" open — correctly identified the bug existed, missed the recent abandonment closure.
6. **NGC container naming convention clarified.** TRT-LLM ships its own image at `nvcr.io/nvidia/tensorrt-llm/release:<version>` (e.g. `:1.2.1`, `:1.3.0rc15`). The Triton-bundled image at `nvcr.io/nvidia/tritonserver:<yy.mm>-trtllm-python-py3` exists but is a separate Triton-backend artifact, not the primary TRT-LLM container. Round 1 conflated the two.
7. **vLLM GH200 ARM support is officially flawless since v0.10.2 (2025-09).** Confirmed by the original GH200 ARM complainant on the NVIDIA forum thread (last update 23 Sep 2025): "vLLM just released their official arm image at v0.10.2, it works flawlessly." vLLM v0.12.0 ships an official aarch64 manifest. This widens the gap with TRT-LLM whose aarch64 wheel restoration only landed in v1.3.0rc13 (Apr 2026).

---

## 1. NGC TRT-LLM Container — SBSA/arm64 Manifest Reality

### Container name(s) — the actual current paths

| Path | Purpose | Latest tag (2026-05-25) |
|---|---|---|
| `nvcr.io/nvidia/tensorrt-llm/release:<ver>` | **Primary TRT-LLM release container** | `1.2.1` (stable), `1.3.0rc15` (RC) |
| `nvcr.io/nvidia/tensorrt-llm/devel:<ver>` | Dev/build container | Same tags |
| `nvcr.io/nvidia/tritonserver:<yy.mm>-trtllm-python-py3` | Triton + TRT-LLM backend bundle | `26.04-trtllm-python-py3` (newest), `25.12-trtllm-python-py3` |
| `nvcr.io/nvidia/pytorch:<yy.mm>-py3` | Base PyTorch (per v1.2.0 notes) | `25.12-py3` (used by 1.2.0/1.2.1) |

### SBSA/arm64 manifest status

NGC's public catalog page renders client-side and `WebFetch` returns only the title, so the live manifest list is not directly verifiable from this environment. **Indirect evidence is strong but not 100% conclusive:**

- v1.3.0rc13 (29 Apr 2026) release notes explicitly **add** "LLM_SBSA_WHEEL_DOCKER_IMAGE" and the "restoring the missing aarch64 library" fix. This wording implies the prior images either lacked the arm64 binary or had it broken — and that 1.3.0rc13+ is the first RC line with a working SBSA path.
- Multiple search hits reference `nvcr.io/nvidia/tensorrt-llm/release:1.2.0` and `:1.3.0rc10` being used on aarch64 (DGX Spark + GH200), so multi-arch manifests do exist for at least those tags.
- The Triton backend's build docs note: building for aarch64 requires `TORCH_INSTALL_TYPE="src_non_cxx11_abi"` — a sign that the Triton-trtllm-python-py3 image is *built* for aarch64 but may need a special path.
- **Caveat:** the NVIDIA forum thread last updated 23 Sep 2025 still flags "no real arm64 binaries" as a complaint, and the response from NVIDIA staff confirmed the situation was being actively fixed but had not yet shipped at that date. The 1.3.0rc13 (Apr 2026) entry is the receipts for that fix landing.

### Recommended operational stance

For GH200 in May 2026, **use one of**:
- `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` (stable, aarch64 confirmed via community use)
- `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc15` (latest RC; required if you want the post-rc13 SBSA fixes)
- `nvcr.io/nvidia/tritonserver:25.12-trtllm-python-py3` (if you want Triton's HTTP/gRPC front-end bundled)

Verify before deploying: `docker manifest inspect nvcr.io/nvidia/tensorrt-llm/release:1.2.1 | jq '.manifests[].platform'` — should show both `linux/amd64` and `linux/arm64`.

Sources:
- Release list: GitHub API via `gh release list --repo NVIDIA/TensorRT-LLM` (verified 2026-05-25)
- NGC container path doc: https://nvidia.github.io/TensorRT-LLM/installation/containers.html
- v1.2.0 release notes (base image): GitHub releases v1.2.0 (verified 2026-05-25)
- v1.3.0rc13 SBSA notes: GitHub releases v1.3.0rc13 + Round 1 doc cross-reference
- Forum thread (last update 23 Sep 2025): https://forums.developer.nvidia.com/t/arm64-gh200-llm-engine-issues/339136

---

## 2. v1.3.0 GA Status — Not Yet

| Version | Date | State |
|---|---|---|
| v1.3.0rc15 | 2026-05-21 | Pre-release (latest) |
| v1.3.0rc14 | 2026-05-07 | Pre-release |
| v1.3.0rc13 | 2026-04-29 | Pre-release (SBSA fix) |
| v1.3.0rc12 | 2026-04-17 | Pre-release |
| **v1.2.1** | **2026-04-20** | **Latest stable / GA** |
| v1.3.0rc11 | 2026-04-09 | Pre-release |
| v1.3.0rc10 | 2026-03-31 | Pre-release |
| v1.3.0rc9 | 2026-03-24 | Pre-release |
| v1.3.0rc8 | 2026-03-17 | Pre-release |
| v1.2.0 | 2026-03-12 | Stable |
| v1.3.0rc7 | 2026-03-10 | Pre-release |
| v1.3.0rc6 | 2026-03-03 | Pre-release |
| v1.3.0rc5.post1 | 2026-03-06 | Pre-release |
| v1.3.0rc5 | 2026-02-24 | Pre-release |
| v1.3.0rc1 | 2026-01-27 | Pre-release |

The RC cycle for 1.3.0 has been running **~4 months** (Jan → late May 2026) with 15 RCs and no GA. That's an unusually long stabilization window — consistent with the FP8/MoE/spec-decode surface area expanding faster than test coverage. v1.2.1 picking up a critical KV-cache corruption fix on 2026-04-20 (mid-RC cycle) suggests NVIDIA is still backporting fixes to the 1.2 LTS line, which they wouldn't bother doing if 1.3.0 GA were imminent.

**Practical implication:** v1.2.1 is the right production target as of 2026-05-25. v1.3.0rc15 is acceptable for non-prod if you need a 1.3-only feature (better SBSA wheels, FP4 paged MQA, Gemma4 multimodal).

Source: `gh release list --repo NVIDIA/TensorRT-LLM --limit 20`

---

## 3. FP8 Issue Status — More Closed-Without-Fix Than Round 1 Reported

### #4218 — FP8 KV Cache Qwen2.5 gibberish

- **State:** CLOSED 2025-05-16
- **Resolution:** Punted to ModelOpt repo, no TRT-LLM-side fix
- **Root cause from issue thread:** Setting `kv_cache_quant_algo=FP8` causes ModelOpt to auto-enable `lm_head` quantization, which Qwen2.5 doesn't tolerate (no asymmetric quant support). Maintainer `brb-nv` advised the user to reproduce against ModelOpt directly; reporter closed with "Will repost there."
- **Verdict:** Bug class is real and unresolved. Quote from maintainer: *"To verify if ModelOpt quantization is broken, you can install ModelOpt directly and try running the quantization script... If you see incoherent responses, then issue is likely in ModelOpt and we can create an issue there."* Translation: "this isn't our bug." Round 1 marked it as still open with bug label — actual status is closed but the failure mode persists.

### #8974 — AutoDeploy FP8/NVFP4 kernel replacement broken for Flux

- **State:** CLOSED 2026-04-28
- **Resolution:** Abandoned. Quote from `lucaslie` (NVIDIA contributor): *"closing this as it's not actively worked on"*
- **What was lost:** Expected 1.6× FP8 / 2.8× NVFP4 speedup on fake-quantized Flux never materialized because the AutoDeploy kernel replacement pass doesn't fire on quantized graphs.
- **Verdict:** This is an AutoDeploy-pipeline bug, not a low-level FP8 kernel bug — so it doesn't directly affect manual TRT-LLM FP8 builds. But it tells you the AutoDeploy path is not getting maintenance attention.

### Llama-3.x / Llama-4 pipeline parallel + FP8 hangs

- **State:** **STILL OPEN** as a documented known issue. The v1.0 release notes (Sep 2025) and every subsequent release-notes refresh through v1.3.0rc15 carry the workaround verbatim: *"For the Llama 3.x and Llama 4 models, there is an issue with pipeline parallelism when using FP8 and NVFP4 weights. As a workaround, you can set the environment variable `export TRTLLM_LLAMA_EAGER_FUSION_DISABLED=1`."*
- **PR #3422** ("Fix pipeline parallelism for Llama-style models") was closed unmerged — superseded by PR #3449 — but the underlying RMS-norm/AR-fusion + PP + FP8 interaction is still patched only via the env-var disable, not a proper fix in the fusion pass.
- **Verdict:** If you're doing single-GH200 or single-GH200-NVL2 deployment (TP, not PP), this doesn't bite you. If you're doing multi-node PP across GH200 systems (DeepSeek-V3 671B on multiple GH200 chassis), set the env var or stick to bf16.

### v1.3.0rc15 FP8-specific changes (per release notes)

- "Fix FP8 block scaling GEMM autotuner cache growth"
- "Fix workspace size calculation for fmha_bmm1_scale_size with FP8ContextMLA"
- "Add cute dsl FP4 paged MQA logits decode kernel" (FP4, Blackwell-only, irrelevant for GH200)

Nothing in rc15 fixes the Qwen FP8 KV gibberish class or the Llama PP+FP8 hang. **The bf16-default policy stands.**

Sources:
- gh issue 4218 thread (closed comments verbatim)
- gh issue 8974 thread (closed comments verbatim)
- TRT-LLM release-notes.md.txt v1.0 known-issues section
- v1.3.0rc15 GitHub release notes

---

## 4. Concrete Llama-3.3-70B BF16 GH200 Throughput — The Hunt

**Bottom line: a clean, publicly published GH200 Llama-3.3-70B BF16 tok/s number does NOT exist in May 2026.** This is the single most frustrating gap in the documentation. Here's what *does* exist:

### Closest available numbers

| Source | Hardware | Model | Precision | Number | Notes |
|---|---|---|---|---|---|
| NVIDIA perf-overview | GH200 96GB | Llama-3.1 **8B** | **FP8** | 27,304 tok/s @ ISL/OSL 128/128 | Only published GH200 entry; not 70B, not BF16 |
| NVIDIA perf-overview | H100 (not H200, not GH200) | Llama-3.1 **70B** | FP8 | listed | No BF16, no GH200 |
| NVIDIA dgxc-benchmarking repo | H100 | Llama-3.3 70B | **FP8 only** | scripts published, no numbers in README | "Uses FP8 on H100, NVFP4 on GB200/B200" — explicitly no BF16 path |
| Baseten/Lambda (Feb 2025) | GH200 96GB | Llama-3.3 70B | **FP8** | **+32% vs H100 80GB** (relative only) | Did NOT publish absolute tok/s — only the delta. Confirmed by re-reading the post. |
| NVIDIA dev blog 17 Dec 2024 | H200 (single) | Llama-3.3 70B | **FP8** target | 51.14 tok/s (no spec-decode), 181.74 tok/s (Llama-3.2-1B draft) | H200, not GH200, batch=1 latency-focused |
| dev.to qpwo (Aphrodite/vLLM fork) | 10× GH200 | Llama-3.1 **405B** | **BF16** | **5.2 tok/s** | Latency, not throughput; tiny batch; 405B not 70B |
| LLM-Inference-Bench (arxiv 2411.00136) | GH200 | Llama-3 70B | mixed | "vLLM on GH200 consistently achieves the highest throughput" — table numbers not extracted by WebFetch | Cited qualitatively; PDF binary extraction failed |
| Artificial Analysis API providers | various clouds | Llama-3.3 70B | mixed | Groq 322, SambaNova 291, GVertex 149, Bedrock 135 tok/s | No GH200 attribution, API-provider single-user latency |

### Closest defensible numbers we can use as anchors

For sizing a GH200-96GB Llama-3.3-70B BF16 deployment **without** running our own benchmark, the least-bad triangulation is:

1. Take NVIDIA's H200 Llama-3.3-70B FP8 batch=1 baseline of **51 tok/s** (single-user, no spec-decode).
2. Adjust for BF16 vs FP8: BF16 is roughly **1.4–1.8× slower** at large batch (vLLM published number for Llama-3.1 70B); at batch=1 the delta is smaller (memory-bandwidth bound), maybe 1.2–1.4×. So single-user BF16 ≈ **35–42 tok/s** on H200/GH200.
3. Apply Baseten's +32% GH200-vs-H100 delta as an upper bound. H200 ≈ GH200 in raw compute; GH200's win comes from KV cache + NVLink-C2C, which matters more at high concurrency than at batch=1.
4. **Result: GH200-96 Llama-3.3-70B BF16, single-user, no spec-decode ≈ 35–45 tok/s.** With EAGLE3 spec-decode in bf16 (1.5–2.5× speedup typical at batch=1 for 70B), **single-user ≈ 60–110 tok/s.** Both ranges have ~30% error bars.

For throughput-mode (batch=32, ShareGPT-like), the only direct data point is Baseten's "+32% over H100" claim. Public H100 batch-32 Llama-3.3-70B FP8 numbers cluster around **1,500–2,500 tok/s aggregate** depending on ISL/OSL. So **GH200-96 batch-32 FP8 ≈ 2,000–3,300 tok/s aggregate**, and **GH200-96 batch-32 BF16 ≈ 1,200–2,200 tok/s aggregate** (assuming 1.5× FP8 advantage).

**Recommendation: don't trust the triangulation. Run `trtllm-bench throughput` and `trtllm-bench latency` on the actual GH200 hardware before publishing any number that drives capacity planning.**

Sources:
- NVIDIA perf-overview https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html
- NVIDIA dgxc-benchmarking https://github.com/NVIDIA/dgxc-benchmarking/blob/main/llama3.3/inference/README.md
- NVIDIA NIM benchmarks https://docs.nvidia.com/nim/benchmarking/llm/latest/performance.html (no GH200 entries confirmed)
- NVIDIA speculative-decode 70B blog https://developer.nvidia.com/blog/boost-llama-3-3-70b-inference-throughput-3x-with-nvidia-tensorrt-llm-speculative-decoding/
- Baseten GH200 test https://www.baseten.co/blog/testing-llama-inference-performance-nvidia-gh200-lambda-cloud/ (re-verified: relative only, no abs tok/s)
- dev.to 405B-on-GH200 https://dev.to/qpwo/how-to-run-llama-405b-bf16-with-gh200s-7da
- Artificial Analysis https://artificialanalysis.ai/models/llama-3-3-instruct-70b/providers

---

## 5. TRT-LLM PyTorch Backend vs vLLM V1 on GH200 — 2026 Comparisons

### Direct GH200 comparisons (May 2026)

**None of the public 2026 vLLM-vs-TRT-LLM articles publish a head-to-head GH200 benchmark with absolute numbers.** Synthetic Futures (Medium), Spheron, n1n.ai, and Northflank all default to H100 comparisons. The GH200-specific qualitative claims are:

- **vLLM ARM image works officially since v0.10.2** (Sep 2025) and v0.12.0 ships a proper `linux/aarch64` manifest. TRT-LLM only got the SBSA wheel fix in v1.3.0rc13 (Apr 2026) — **vLLM had a ~7-month head start on the ARM-native experience.**
- **Spheron/Synthetic Futures consensus:** on H100, TRT-LLM is 8–30% higher throughput than vLLM at matched concurrency, but vLLM closes the gap with v0.10+. On GH200, "vLLM consistently achieves the highest throughput across all batch sizes" per LLM-Inference-Bench (arxiv 2411.00136), but this paper predates TRT-LLM's PyTorch-backend default in v1.0 (Sep 2025) and the SBSA fix in v1.3.0rc13.
- **vLLM Llama-3.3 70B recipe** explicitly recommends FP8 + KV cache FP8 for Hopper (`docs.vllm.ai/projects/recipes`), with TP1 for throughput and TP2/4/8 for latency — but publishes no concrete tok/s in the recipe.
- **Third-party docker images** like `drikster80/vllm-gh200-openai` exist and are referenced in production deployment guides. The need for community-maintained images suggests vLLM's official aarch64 path still has gaps that a GH200-specific build addresses.
- **InferenceMAX / SemiAnalysis InferenceX** (formerly InferenceMAX v1) publishes vLLM data on Blackwell vs Hopper for Llama-3.3-70B at 50 TPS/user interactive serving — but does not break out GH200 separately from H200, and the public dashboard lives behind JS rendering that WebFetch can't access. Quoted finding: "up to 3.7× more throughput using vLLM on Blackwell vs. Hopper for Llama 3.3 70B."

### TRT-LLM PyTorch backend specifics

- v1.0 (Sep 2025) made PyTorch backend default. v1.2.x is the production-grade PyTorch-backend release.
- **No published apples-to-apples PyTorch-backend-vs-TRT-engine number on GH200.** General guidance from Synthetic Futures: PyTorch backend trades ~5–15% peak throughput for startup time (no AOT compile) and operational simplicity (multi-LoRA, hot model swap).
- The TRT-LLM PyTorch backend gained GH200-specific functional support per the v0.21 / v1.0 release notes, including the `host_cache_size` KV offload to Grace LPDDR5X — but no benchmark publication validates how much that actually wins versus vLLM's CPU offload story.

### Practical decision matrix

| Scenario | Pick |
|---|---|
| New GH200 deployment, want least friction | **vLLM v0.12+ with official aarch64 image** |
| Already on TRT-LLM, max throughput at high concurrency, willing to babysit | **TRT-LLM v1.2.1 PyTorch backend** |
| Need EAGLE3 / Medusa spec-decode SOTA | **TRT-LLM v1.2.1** (vLLM spec-decode lags) |
| Multi-LoRA hot swap | TRT-LLM PyTorch backend OR vLLM — both OK |
| Multi-node PP with FP8 | **Neither, until pipeline-parallel + FP8 fusion lands;** bf16 only |

Sources:
- Medium / Synthetic Futures comparison https://medium.com/synthetic-futures/vllm-vs-tensorrt-llm-the-definitive-2026-comparison-for-llm-inference-ed0943fb81d2
- Spheron H100 benchmarks https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/
- LLM-Inference-Bench paper https://arxiv.org/abs/2411.00136
- vLLM Llama-3.3 recipe https://docs.vllm.ai/projects/recipes/en/latest/Llama/Llama3.3-70B.html
- SemiAnalysis InferenceX https://inferencex.semianalysis.com/ (dashboard behind JS — qualitative claims only)
- vLLM Blackwell blog https://blog.vllm.ai/2025/10/09/blackwell-inferencemax.html
- vLLM aarch64 docker drikster80/vllm-aarch64-openai
- NVIDIA forum 23 Sep 2025 update https://forums.developer.nvidia.com/t/arm64-gh200-llm-engine-issues/339136

---

## 6. Net Recommendations Update (delta from Round 1)

1. **Container target:** `nvcr.io/nvidia/tensorrt-llm/release:1.2.1` (not 1.2.0, not 1.3.0rc-anything-unless-you-need-SBSA-fix). For SBSA-only fixes: `:1.3.0rc13` or later RC.
2. **The bf16-default policy is reinforced**, not weakened. Two of the three FP8 issues Round 1 flagged are now CLOSED but **not fixed** — they were closed because nobody is working them. The Llama PP+FP8 hang is still officially documented across every release-notes revision with the same env-var workaround.
3. **There is still no public NVIDIA GH200 70B benchmark.** Round 1 called this "the biggest documentation hole." Round 3 confirms: no new GH200 70B numbers were published in the ~3 months since Round 1's compile. The hole has not been filled.
4. **vLLM has a clean lead on the GH200 aarch64 operational story** (7-month head start on an official ARM image). If the team's priority is "stand up Llama-3.3-70B on GH200 by next week," vLLM is the lower-risk choice. If the priority is "extract every last token/sec with EAGLE3 spec-decode," TRT-LLM v1.2.1 is still the right pick — but plan to run `trtllm-bench` yourself for capacity numbers.

---

## 7. Confidence Updates

### High confidence
- v1.3.0 GA has not shipped (`gh release list` is authoritative)
- v1.2.1 is the latest stable (2026-04-20)
- #4218 closed-without-fix 2025-05-16 (issue thread direct evidence)
- #8974 closed as abandoned 2026-04-28 (NVIDIA contributor quote)
- Llama-3.x PP+FP8 hang workaround is still the official guidance (release notes verbatim)
- No published NVIDIA GH200 70B BF16 or FP8 tok/s exists as of 2026-05-25

### Medium confidence
- NGC `release:1.2.1` carries a multi-arch (amd64+arm64) manifest — strong indirect evidence (community use on GH200, NGC docs reference SBSA wheel, v1.3.0rc13 explicit SBSA fix) but no direct `docker manifest inspect` verification from this environment.
- The +32% Baseten GH200-vs-H100 delta is from Feb 2025 and used FP8 + SGLang harness — assuming it transfers to BF16 and to 2026 stacks is an extrapolation, not a measurement.
- The 35–45 tok/s single-user BF16 triangulation for Llama-3.3-70B on GH200 is a derived estimate, not a measured number.

### Still unresolved gaps
- No NGC multi-arch manifest directly inspected (would need shell access to a Docker host).
- No `trtllm-bench` numbers run on our own GH200 — every absolute number in this document is from a public source, none from us.
- vLLM-vs-TRT-LLM-PyTorch-backend head-to-head on GH200 with absolute numbers does not exist in public benchmarks; need to run it ourselves.

---

## 8. Updated Source URL Index (2026-05-25 verified)

New / updated since Round 1:
- gh issue 4218 thread (closure receipts): https://github.com/NVIDIA/TensorRT-LLM/issues/4218
- gh issue 8974 thread (abandonment receipts): https://github.com/NVIDIA/TensorRT-LLM/issues/8974
- gh PR 3422 (closed unmerged): https://github.com/NVIDIA/TensorRT-LLM/pull/3422
- v1.2.1 release: https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.2.1
- v1.3.0rc15 release: https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0rc15
- v1.2.0 release (base image quote): https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.2.0
- NGC containers doc: https://nvidia.github.io/TensorRT-LLM/installation/containers.html
- vLLM aarch64 docker: https://hub.docker.com/r/drikster80/vllm-aarch64-openai
- vLLM Llama-3.3 recipe: https://docs.vllm.ai/projects/recipes/en/latest/Llama/Llama3.3-70B.html
- SemiAnalysis InferenceX (Blackwell vs Hopper for Llama-3.3-70B with vLLM): https://blog.vllm.ai/2025/10/09/blackwell-inferencemax.html
- llama.cpp on GH200 discussion (GPT-OSS 20B 322 tok/s, no Llama 70B): https://github.com/ggml-org/llama.cpp/discussions/18005
- dev.to 405B BF16 on 10× GH200 (5.2 tok/s): https://dev.to/qpwo/how-to-run-llama-405b-bf16-with-gh200s-7da

All Round 1 URLs remain valid as of 2026-05-25.
