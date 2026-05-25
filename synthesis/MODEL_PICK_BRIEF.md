# Model-pick brief: Llama-3.3-70B vs Qwen3.6-27B (+ peers) on GH200

**Compiled 2026-05-25 from Round 5 (15 narrow-deep agents).**
**Question**: highest tok/s, least friction running Llama-3.3-70B Q4 + Qwen3.6-27B Q4 on a single Lambda GH200 96GB. Then: comparable models worth considering.

---

## The 30-second answer

1. **For least friction**: `drikster80/vllm-gh200-openai` Docker, 2 commands from fresh SSH, ~50 tok/s with AWQ INT4 on Llama-3.3-70B. Same image runs Qwen3.6-27B.
2. **FP8 vs Q4 picks differently by workload — not a one-size answer**:
   - **Single-user / batch=1 / decode-bound**: Q4 wins on tok/s by ~1.4-1.6× because decode is memory-bandwidth-bound and Q4 weights are ~17 GB vs FP8's ~27 GB for Qwen3.6-27B (~40 GB vs ~70 GB for Llama-3.3-70B). R4 #8 measured AWQ-Marlin at ~2× FP8 at batch 1-8.
   - **Batch ≥16 / prefill / TTFT-bound**: FP8 wins because compute becomes the bottleneck and FP8 tensor cores deliver ~2× FLOPs with no dequant overhead.
   - **Quality**: FP8 W8A8 loses ≤0.5 pt MMLU vs BF16; AWQ Q4 loses ~1-2 pt (FOEM-calibrated GPTQ closes to ~+1.95% PPL).
   - **FP8 is safe for text LLMs on Hopper** post vLLM 0.19.1. The user-memory "FP8 broken" rule is **DiT-class diffusion only** (Round-2 forensic audit + 3× cross-arch reproduction).
3. **Qwen3.6-27B dominates Llama-3.3-70B on the workloads that matter**: MMLU-Pro +17, GPQA-Diamond +37, LiveCodeBench v6 83.9 (Llama doesn't compete), AIME26 94.1 (Llama n/a). Llama wins only IFEval. Qwen3.6-27B is multimodal native (text+image+video), 201 languages, 262K ctx.
4. **Both fit on GH200 simultaneously (~87 GB combined Q4)** — run both, A/B on your actual workload.

---

## Recipes (highest-tps, least-friction)

### Llama-3.3-70B — single-stream interactive, max tok/s

```bash
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/models:/root/.cache/huggingface \
  drikster80/vllm-gh200-openai:latest \
  --model casperhansen/llama-3.3-70b-instruct-awq \
  --quantization awq_marlin \
  --tensor-parallel-size 1 \
  --enable-prefix-caching --enable-chunked-prefill \
  --max-num-batched-tokens 8192 --max-num-seqs 256 \
  --gpu-memory-utilization 0.92 \
  --speculative-config '{"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.3-Instruct-70B","num_speculative_tokens":3}'
```

**Critical gotcha**: `--quantization awq_marlin` must be passed *explicitly* — auto-detect chooses plain AWQ kernel which is **10.9× slower** (67 vs 741 tok/s on Qwen-32B reference). Expected: 55-75 tok/s @ b=1 → 3,800-5,500 @ b=256 (±35% — no public GH200 number to anchor).

Friction: 2/5.

### Llama-3.3-70B — pure-GGUF llama.cpp path

`bartowski/Llama-3.3-70B-Instruct-GGUF :: Q4_K_M.gguf` (42.52 GB). Build with `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90` (not `121-real` from Blackwell recipes, won't run on GH200). Expected 40-55 t/s. **KV cache cliff**: `-ctk q4_0 -ctv q4_0` is 92% slower at 64K context — use `-ctk q8_0 -ctv q8_0 -fa 1`. **No MTP head exists for Llama-3.3** — use classic spec with `Llama-3.2-1B-Instruct-Q4_K_M.gguf` draft.

### Qwen3.6-27B — recommended (highest-quality)

`unsloth/Qwen3.6-27B-MTP-GGUF :: UD-Q6_K_XL` (25.6 GB) — **Q6 over Q4** because GH200's headroom is wasted otherwise (Q6 is only 22% slower, near-reference quality). Or **`Qwen/Qwen3.6-27B-FP8`** via vLLM ≥0.19 — fits HBM with full 256K KV.

```bash
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  drikster80/vllm-gh200-openai:latest \
  --model Qwen/Qwen3.6-27B-FP8 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 131072 \
  --enable-prefix-caching \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

**Do not use SGLang for this checkpoint** — bug #23687 (Qwen3.6 FP8 silent garbage in `qwen3_5.py` loader) still open. AWQ/GPTQ paths in SGLang use a different loader and *are* safe.

**EAGLE-3 head does NOT exist for any Qwen3.6 SKU** — use the built-in MTP head (`qwen3_next_mtp` method) which is trained into the model. ~92% acceptance at draft pos 1-3, 1.4-2.2× speedup. Expected: 140-180 tok/s @ b=1 with MTP, 6,500-8,500 @ b=256.

**Smoke test required**: vLLM #40252 (NVFP4 silent garbage on hybrid-attn fused projections like `in_proj_qkvz`) is the structural twin of SGLang #23687; PRs #34697/#35413 fix it but verify on first boot.

Friction: 2/5.

### Qwen3.6-27B — pure-GGUF

`unsloth/Qwen3.6-27B-MTP-GGUF :: UD-Q4_K_XL` (17.9 GB). Requires llama.cpp built post-mid-May 2026 (PR #19408 DeltaNet ops + PR #22673 MTP merged 2026-05-16). Use **MTP path** with `--spec-type draft-mtp --spec-draft-p-min 0.75` — no external draft, ~92% acceptance, 1.4-2.2× speedup. Expected: 75-110 single-stream, 95-140 with MTP at Q4. SGLang #23687 does not affect GGUF (FP8-only bug). Friction: 2/5.

---

## Comparable models worth considering

### Top substitution shortlist (all Apache-2.0 unless flagged)

| Model | Footprint on GH200 | tok/s anchor | Notable quality | Why |
|---|---|---|---|---|
| **GPT-OSS-120B (MXFP4)** | 60.8 GiB fits no-offload | **209 t/s measured (llama.cpp on GH200)** | MMLU 90 (> Llama 86), AIME-2025 97.9 | Best single-GH200 frontier reasoning. Native MXFP4 — quantization-clean. |
| **GPT-OSS-20B (MXFP4)** | 12 GiB | **323 t/s measured (GH200)** | MMLU 85.3 (> Llama 81.9), AIME-2025 89.3 | Dark horse. Replaces Qwen3.6-27B for raw throughput when 5pt MMLU acceptable. |
| **Qwen3.6-27B** (FP8) | 27 GiB | est. 80-110 single, 200 w/MTP | MMLU-Pro 86.2, GPQA-D 87.8, LCB 83.9, AIME26 94.1 | New baseline. Multimodal, 201 langs, 262K ctx, hybrid attn (small KV growth). |
| **Nemotron-Super-49B-v1.5** | ~50 GB Q4 | est. ~1.4× Llama 70B | MATH-500 **97.4**, AIME-24 87.5, GPQA 72 | Quality king on single GH200 — TRT-LLM only path officially. |
| **DeepSeek-R1-Distill-Llama-70B** | ~40 GB Q4 | drop-in replacement | +54 AIME, +17 MATH-500, +15 GPQA vs Llama-3.3-70B | Zero-friction reasoning upgrade — same serving stack as Llama-3.3-70B. |
| **Granite-4.1-30B** | ~16 GB Q4 | ~Qwen-27B class | MMLU 80.2, 512K ctx | Sleeper if you need very-long-context with small KV. ~2pt behind Llama-3.1-70B. |
| **Phi-4-reasoning-plus (14B)** | ~7 GB Q4 | ~2× Qwen-27B decode | Reasoning intact, knowledge narrower | Pair as side-car on the same GH200 with Qwen3.6-27B. |

### License blockers (avoid for commercial)
- **Command A**: CC-BY-NC (need Cohere contract).
- **Gemma-3-27B**: Gemma TOS (not Apache). Gemma-4 fixed this; Gemma-3 did not.

### Phantom models (do not exist as released open weights, May 2026)
- Qwen3-72B / Qwen3.6-72B / Qwen3.6-32B (Qwen3 dense caps at 32B; Qwen3.6 dense caps at 27B).
- Hermes-4-Llama-3.3-70B (Hermes-4-70B is on Llama-3.1).
- Phi-4-70B.
- Yi-2-34B.

### Doesn't fit single GH200 96GB (even at Q4)
- Mistral-Large-3 (675B MoE, ~360 GB Q4).
- DeepSeek-V3.1 (671B MoE, ~380 GB Q4) — but llama.cpp + `--n-cpu-moe 58` runs it at **18.3 t/s single / 499 t/s @ b=32** via Grace LPDDR offload.
- Llama-4-Maverick (400B MoE, ~220 GB).
- Kimi K2 Thinking (~1T, ~560 GB Q4) — runs at 21.6 t/s via llama.cpp + Grace.

---

## The headline economic point (from R4 #18, #20)

**A single GH200 on Vultr or Lambda cannot beat DeepInfra-Turbo's $0.21/Mtok blended floor on Llama-3.3-70B at any realistic utilization** (need 132-151% util — impossible). If you're below ~100B Mtok/mo and not constrained by compliance / multi-LoRA / latency-floor, **DeepInfra wins on cost. Your GH200 justifies on**:
- MoE 671B+ via llama.cpp (Grace LPDDR is the only single-box path — genuine moat)
- Multi-LoRA serving across 5-10 fine-tunes (break-even drops 3-10×)
- Compliance / data-residency forcing self-host (comparison shifts to Bedrock $0.72 → 25% util wins)
- Need >50 tok/s/user streaming (DI Turbo is 17.3 tok/s — wrong vendor)

For pure dense-LLM token economics on Llama-3.3-70B, switch to DeepInfra. For Qwen3.6-27B serving, you're own-host because the managed market hasn't shipped it yet.

---

## The substitution proposal

**Drop Llama-3.3-70B as your daily driver. Run Qwen3.6-27B (FP8 via vLLM with built-in MTP) as the baseline + GPT-OSS-120B (MXFP4) as the high-effort reasoning route.** Both fit no-offload on GH200, both Apache-2.0. You gain:
- +17 to +37 pts on hard reasoning benchmarks (GPQA-Diamond +37 is the largest delta)
- ~2-3× single-stream tok/s
- Native multimodal + 201 languages + 262K context

You lose only IFEval-style strict short-prompt instruction-following (~7 pts) and ecosystem inertia.

If your workload is IFEval-dominant or you need DeepSeek-R1 reasoning specifically, swap to **DeepSeek-R1-Distill-Llama-70B** — drop-in same-serving-stack with +54 AIME / +17 MATH-500 / +15 GPQA vs Llama-3.3-70B.
