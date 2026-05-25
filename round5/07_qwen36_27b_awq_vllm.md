# Qwen3.6-27B @ 4-bit (AWQ / GPTQ) on vLLM 0.21.0 / GH200 96GB

**Date:** 2026-05-25
**Target:** 1× GH200 96GB, Lambda Ubuntu 24.04 aarch64
**Stack:** vLLM 0.21.0, CUDA 12.8/13.0, bf16 base + 4-bit weight-only quant

---

## 1. SKU verification — Qwen3.6 ~27B exists

The exact target exists: **`Qwen/Qwen3.6-27B`** — released **2026-04-22**, Apache-2.0, dense 27B, multimodal (vision+text).

Independent verification:
- HF card: `Qwen/Qwen3.6-27B` — hidden_size 5120, **64 layers**, vocab 248,320, native ctx **262,144** (YaRN to 1.01M), MTP head trained in.
- Architecture is **hybrid Qwen3-Next-style**: layer block = `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`. DeltaNet heads: 48 V / 16 QK, head_dim 128. Full-attn: 24 Q / 4 KV, head_dim 256.
- Sibling SKU is **Qwen3.6-35B-A3B** (MoE 35B/3B-active). Only those two SKUs ship in the 3.6 family today.

Conclusion: use Qwen3.6-27B dense. No need to fall back to 32B/30B-A3B.

**Caveat — this is a hybrid model, not a vanilla transformer.** Every Qwen3-Next/Qwen3.5/Qwen3.6 quantization story is gated by whether the inference engine handles the `linear_attn.in_proj_qkvz` / `linear_attn.in_proj_ba` fused projections correctly. See section 8.

---

## 2. Best AWQ/GPTQ checkpoint

Two viable options. Both INT4, both work with vLLM 0.21 on Hopper via Marlin.

| Checkpoint | Method | Group | Size | Calibration | Verdict |
|---|---|---|---|---|---|
| **`btbtyler09/Qwen3.6-27B-GPTQ-4bit`** | GPTQ + FOEM | 32 | 21 GB | evol-codealpaca + C4, 256 samples | **Recommended.** Reported PPL +1.95% vs BF16 (vs +8.45% vanilla GPTQ). Quant config explicitly handles `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` as INT4 and leaves `in_proj_a`/`in_proj_b` in FP16. MTP head BF16. Loads via `--quantization gptq_marlin`. |
| `cyankiwi/Qwen3.6-27B-AWQ-INT4` | AWQ | (unspec, likely 128) | 19.0 GB | STEM + Agentic | Smaller. 1.5M downloads, broadly used. AWQ group/ignore-list not published — risk of silent linear_attn zero-init unless vLLM ≥ 0.20 packed-modules mapping covers it (PR #34697 / #35413). |
| `QuantTrio/Qwen3.6-27B-AWQ` | "Data-free" AWQ | — | 21 GiB | None (RTN-style) | Cheapest to load; no calibration → expect worse downstream than GPTQ-FOEM. |

**Pick GPTQ-FOEM** as primary. Its quant config is transparent about which hybrid-attention tensors stay FP16 — that is exactly the thing that has bitten every NVFP4/FP8 community quant of this architecture (vLLM #40252, HF disc on RedHatAI and Sehyo). AWQ as fallback if you want 2 GB more KV cache headroom.

---

## 3. Exact `vllm serve` command (TP=1, GH200, single-stream + small-batch)

```bash
# Environment — GH200 quirks
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_FLASHINFER_MOE_FP16=0     # dense model, MoE flag irrelevant
export OMP_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST="9.0+PTX"     # Hopper / GH200

vllm serve btbtyler09/Qwen3.6-27B-GPTQ-4bit \
  --served-model-name qwen3.6-27b \
  --tensor-parallel-size 1 \
  --quantization gptq_marlin \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --max-model-len 131072 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --swap-space 32 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
```

For the AWQ checkpoint instead, swap the two lines:
```bash
  cyankiwi/Qwen3.6-27B-AWQ-INT4 \
  ...
  --quantization awq_marlin \
```

### Why each flag

- `--quantization gptq_marlin` — Hopper-optimal kernel; fuses dequant into the GEMM. Marlin path on Hopper is ~3× over BF16 W-only dequant. AWQ uses `awq_marlin` (the auto-converted Marlin variant), **not** plain `awq`.
- `--dtype bfloat16` — activations/KV in bf16. (Per your memory anchor: never FP8 on this stack.)
- `--max-model-len 131072` — 128K is comfortable on 96 GB with one bf16 KV stream of this hybrid model. Drop to 32k if you want headroom for `--max-num-seqs > 64`. Native model max is 262144 (YaRN extensible to ~1M).
- `--enable-prefix-caching` + `--enable-chunked-prefill` — both default-on in 0.21, set explicit for clarity. Mandatory for any agentic/coding workload (repeated prefixes, long prompts).
- `--max-num-batched-tokens 8192` — chunked prefill window. Raise to 16k for higher prefill TPS if you don't care about TTFT.
- `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` — uses the **built-in MTP layer that ships with Qwen3.6-27B weights** (no external draft model required). `n=2` is the public sweet spot for vLLM; `n=3` is borderline (position-4 acceptance collapses to ~21% per the rtx-3090 stack report). MTP helps low-concurrency only; turn off for batch ≥ 64 throughput tests.
- `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder` — required for proper agentic output handling.
- `--gpu-memory-utilization 0.90` — leaves ~9 GB for activations, FlashInfer workspace, and graph capture on 96 GB. Push to 0.93 only after you've measured.

### What NOT to use

- `--enforce-eager` — only needed on Blackwell-consumer / AMD where graphs break. **Do not set on GH200.**
- `--quantization awq` (without `_marlin`) — non-fused, kills throughput.
- `--kv-cache-dtype fp8` — works on Hopper but you said no FP8. `auto` → bf16 KV.
- `--language-model-only` — only if you don't need vision; saves a few hundred MB.

---

## 4. EAGLE-3 head for Qwen3.6-27B — availability

**No EAGLE-3 head exists for Qwen3.6-27B as of 2026-05-25.** Verified:

- SpecForge issue #486 (opened 2026-03-03) is the **request** for EAGLE-3 / DFlash training support for Qwen3.5-class hybrid models. Status: open, no merge. No Qwen3.6 mention.
- lmsys SpecBundle phase-1 has shipped EAGLE-3 drafts for **Qwen3-30B-A3B**, **Qwen3-Coder-30B-A3B**, and **Qwen3-Coder-480B-A35B** — none for the Qwen3.5/3.6 hybrid line (DeltaNet+gated-attn complicates draft training).
- SafeAILab EAGLE repo has no Qwen3-Next-class config either.

**Practical recommendation:** use the model's **built-in MTP head** (`qwen3_next_mtp`, n=2) — that's what the official Qwen and vLLM recipes assume. If/when an EAGLE-3 draft drops on `lmsys/SGLang-EAGLE3-Qwen3.6-27B-*`, swap to `--speculative-config '{"method":"eagle3","model":"lmsys/...","num_speculative_tokens":3}'`. Until then, MTP is the only game in town for this SKU.

---

## 5. Expected tok/s on GH200 96GB, bf16+GPTQ-4bit, TP=1

Extrapolated from: (a) published Qwen3.6-27B-AWQ on RTX 3090 (~85 TPS sustained / 106 peak with MTP-3, single-stream) and (b) Lambda's published "GH200 = 7.6× H100" Llama-70B BF16 number, then renormalized for 27B + AWQ-Marlin (≈3× over BF16). These are **modeled estimates**, not measurements — confirm with `vllm bench` once running.

| Batch / concurrency | Decode tok/s **(total)** | Per-stream tok/s | Notes |
|---:|---:|---:|---|
| 1   | **140–180** with MTP-2 (acceptance ~80%); 95–120 without | same | Single-stream coding/agent workload. MTP wins ~1.6×. |
| 16  | **1,400–1,800** | ~90–115 | Marlin saturates SMs; KV still small (~12 GB at 8k ctx). |
| 64  | **3,800–5,200** | ~60–80 | Hybrid KV manager keeps memory uniform; turn MTP **off** here. |
| 256 | **6,500–8,500** | ~25–35 | Prefill-bound; raise `--max-num-batched-tokens` to 16384. KV @ ~50 GB at 4k avg ctx. |

Sanity check: published "2× consumer-GPU 16GB" NVFP4 setup hits 345 tok/s at c=64. GH200 has ~10× the memory bandwidth of one of those, Marlin INT4 ≈ NVFP4 throughput, so 3.5–5k tok/s at c=64 on GH200 is consistent.

**Floor:** at c=1 without MTP and without prefix cache, you will see ~80 tok/s. If your measurement comes in under that, suspect the linear_attn ignore-list bug (section 8).

---

## 6. KV-cache offload settings

Qwen3.6-27B has hybrid KV: full-attn layers (16/64) hold the real KV cache; DeltaNet layers (48/64) hold a small recurrent state. vLLM's hybrid KV manager handles the block-size matching automatically (the Qwen3-Next blog details this). You don't set anything manually for that.

For CPU/Grace-memory offload on GH200's 480 GB unified LPDDR5X:

```bash
  --swap-space 32                  # 32 GB CPU-side swap for evicted blocks
  --cpu-offload-gb 0               # leave at 0 — 4-bit weights fit on-GPU
  --kv-cache-dtype auto            # bf16; fp8 KV is the only real win and you've ruled it out
```

Notes:
- **Do not turn on `--cpu-offload-gb`** — at 21 GB GPTQ weights you have 75 GB free on the H100 die. CPU-offloading weights to Grace would tank decode by ~5×.
- For very long contexts (>200k), enable `--swap-space 64` and let prefix-cache hit/evict from Grace-side memory; transfer is ~1 GB/s NVLink-C2C → cheap.
- `VLLM_ATTENTION_BACKEND=FLASHINFER` for prefill (default on Hopper in 0.21).

---

## 7. One-command Docker quickstart (GH200 aarch64)

There is **no official vLLM ARM64 image** from vllm-project; the maintained community image is `drikster80/vllm-gh200-openai` (built from NGC pytorch:24.07 + flash-attn + flashinfer compiled for sm_90). For vLLM 0.21 you'll likely need to pin the tag (`drikster80/vllm-gh200-openai:0.21.0` if published, else build from source via the PR #10499 Dockerfile path).

```bash
docker run --runtime nvidia --gpus all --ipc=host --shm-size 32g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  drikster80/vllm-gh200-openai:latest \
  --model btbtyler09/Qwen3.6-27B-GPTQ-4bit \
  --quantization gptq_marlin \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --trust-remote-code
```

If you'd rather build native (no Docker), the friction-minimal path is:
```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install "vllm==0.21.0" --torch-backend=auto
uv pip install "transformers>=5.5.4" flashinfer-python
```

---

## 8. SGLang #23687 FP8 bug — does it affect AWQ/GPTQ Marlin?

**No.** That bug is FP8-block-quant-specific: the FP8 path in SGLang's `qwen3_5.py` failed to register `weight_scale_inv` on the fused `gate_up_proj` for the dense MLP, so dequant ran with scale=1.0 → token salad. The mechanism is unique to FP8's per-tensor inverse-scale handling.

**But there is a structurally identical risk on AWQ/GPTQ for this architecture**, just in a different tensor:
- vLLM **#40252** (NVFP4) and a chain of HF discussions (RedHatAI/Sehyo NVFP4 quants) all describe the same root cause: vLLM fuses linear-attention `in_proj_qkv` + `in_proj_z` into stacked params (`in_proj_qkvz`, `in_proj_ba`) **before** the `quantization_config.ignore` check fires. If the fused names aren't in the ignore list, vLLM tries to quantize them, finds them packed under `weight_packed`, fails silently, and zero-inits → garbage out.
- vLLM PRs **#34697** and **#35413** added the packed-modules mapping for these aliases; ≥ 0.20 is supposed to autohandle it. **0.21.0 should be safe**, but verify on first boot.

### Smoke test (mandatory first run)

```bash
# After server is up:
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"Write a Python function that returns the 10th Fibonacci number."}],"max_tokens":128,"temperature":0.0}' \
  | jq -r .choices[0].message.content
```

If output is `!!!!!!!` / single-token repeats / random Chinese chars, **you've hit the linear_attn silent-load bug.** Fix:

```bash
# Patch the model's quantization_config.ignore
python - <<'PY'
import json, pathlib
p = pathlib.Path.home() / ".cache/huggingface/hub" / \
    "models--btbtyler09--Qwen3.6-27B-GPTQ-4bit" / "snapshots"
cfg = next(p.rglob("config.json"))
d = json.loads(cfg.read_text())
ig = d.setdefault("quantization_config", {}).setdefault("ignore", [])
for pat in [r"re:.*linear_attn\.in_proj_qkvz$",
            r"re:.*linear_attn\.in_proj_ba$"]:
    if pat not in ig: ig.append(pat)
cfg.write_text(json.dumps(d, indent=2))
print("patched:", cfg)
PY
```

Then restart the server. The GPTQ-FOEM checkpoint already lists the unfused names in its ignore list, so this is mostly a defense-in-depth step.

---

## 9. Open risks / things to verify on first run

1. **vLLM 0.21.0 ARM64 wheel availability.** PyPI may not have prebuilt aarch64; falling back to source build is ~25 min on GH200 (use `MAX_JOBS=66 NVCC_THREADS=2`).
2. **Marlin INT4 + hybrid KV manager interaction.** Newish combo. If you see CUDA-graph capture warnings, drop `--max-num-seqs` to 32 and retry. Do **not** set `--enforce-eager` unless graphs hard-fail.
3. **MTP head packaging.** The HF Qwen3.6-27B repo bundles the MTP weights. Some community quant repos strip them — verify the `mtp` tensors are in the safetensors index of whichever quant you pick (GPTQ-FOEM keeps them BF16).
4. **Vision encoder.** Stays BF16 in all 4-bit quants. Adds ~1.5 GB. If text-only, set `--language-model-only`.
5. **Context-length vs concurrency tradeoff.** 128k × 64 seqs ≈ 50 GB KV. Going to 256k native context forces `--max-num-seqs ≤ 16`.

---

## Sources

- [Qwen/Qwen3.6-27B (HF model card)](https://huggingface.co/Qwen/Qwen3.6-27B) — architecture, MTP, inference engine recipes
- [Qwen3.6-27B vLLM Recipes](https://recipes.vllm.ai/Qwen/Qwen3.6-27B) — official serve commands
- [vLLM Qwen3.5/3.6 Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [vLLM Qwen3-Next Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-Next.html)
- [vLLM blog: Qwen3-Next hybrid KV cache manager](https://vllm.ai/blog/2025-09-11-qwen3-next)
- [btbtyler09/Qwen3.6-27B-GPTQ-4bit](https://huggingface.co/btbtyler09/Qwen3.6-27B-GPTQ-4bit) — recommended GPTQ-FOEM checkpoint
- [cyankiwi/Qwen3.6-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.6-27B-AWQ-INT4) — alt AWQ checkpoint
- [QuantTrio/Qwen3.6-27B-AWQ](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ) — data-free AWQ alternative
- [SGLang #23687 FP8 silent garbage bug](https://github.com/sgl-project/sglang/issues/23687) — FP8-only, confirms AWQ/GPTQ unaffected by *this* bug
- [vLLM #40252 NVFP4 linear_attn silent garbage](https://github.com/vllm-project/vllm/issues/40252) — structurally similar risk for hybrid quants; fixed by PR #34697/#35413
- [vLLM #38643 Qwen3.5 FLA gibberish](https://github.com/vllm-project/vllm/issues/38643) — related linear-attn tensor-format bug
- [SpecForge #486 EAGLE-3 Qwen3.5 RFC](https://github.com/sgl-project/SpecForge/issues/486) — confirms no Qwen3.6 EAGLE-3 head shipped
- [drikster80/vllm-gh200-openai (Docker)](https://hub.docker.com/r/drikster80/vllm-gh200-openai) — community ARM64 image
- [vLLM PR #10499 GH200 Dockerfile](https://github.com/vllm-project/vllm/pull/10499)
- [Lambda: GH200 vs H100 inference economics](https://lambda.ai/blog/putting-the-nvidia-gh200-grace-hopper-superchip-to-good-use-superior-inference-performance-and-economics)
- [Medium: Qwen3.6-27B 85 TPS on RTX 3090](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914) — MTP n=2 vs n=3 acceptance data
- [LLMKube: Qwen3.6-27B consumer-GPU bakeoff](https://llmkube.com/blog/qwen3-6-27b-bakeoff) — c=64 throughput reference
- [hec-ovi/vllm-awq4-qwen](https://github.com/hec-ovi/vllm-awq4-qwen) — AWQ-INT4 + DFlash spec config on ROCm (config flags transferable)
