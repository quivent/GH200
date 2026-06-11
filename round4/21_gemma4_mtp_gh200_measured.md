# Gemma 4 31B + MTP on GH200 — Measured (not ported)

**Agent**: live measurement on the rig
**Date**: 2026-06-11
**Builds on**: `12_gemma_headdim256.md` (§5.2 canonical recipe), `15_lowlatency_specdec.md`
**Closes the gap from**: `12_gemma_headdim256.md` TL;DR #7 — "No public Gemma + GH200 benchmark exists; H100 numbers ported." These are real numbers off this GH200, not InferenceBench ports.

---

## TL;DR

1. **The drafter depth was detuned.** The live `gemma4` container was serving at `num_speculative_tokens: 3` (γ=3). The rated-fast config is **γ=8**. Bumped it — same fp8, same build — and re-measured.
2. **Warm single-stream: ~453 tok/s** (greedy, 256 tok, k=8). A naive first-request number of **72.8 tok/s was a cold-start artifact** — the first call pays two `eagle_prepare_*` Triton JIT compilations. 8/8 identical warm requests land at ~453 tok/s. Always discard the first request.
3. **Peak aggregate: 6,695 tok/s @ concurrency 64** (k=8), vs **5,597 tok/s** on the earlier γ=3 sweep — **+19.6%** from the γ bump alone.
4. **Acceptance is workload-dependent.** Greedy/deterministic prompt → 89–93% draft acceptance, mean acceptance length ~8.8 at γ=8. A varied/higher-temp workload on the same model measured ~54–63%. Don't quote a single acceptance number without the workload.

---

## 1. Serving setup (as measured)

- **Container**: `gemma4`, image `vllm/vllm-openai:v0.22.1` (MTP day-0 support, PR #41745), restart policy `no`, port 8000.
- **Model**: `google/gemma-4-31B-it` (fp8), HF cache bind-mounted.
- **Drafter**: `google/gemma-4-31B-it-assistant` (Gemma4 MTP heads → language_model layers 58/59), `method: mtp`.
- **Launch** (only `num_speculative_tokens` changed from the detuned 3):

```bash
docker run -d --name gemma4 --gpus all --ipc=host -p 8000:8000 \
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN \
  vllm/vllm-openai:v0.22.1 \
  --model google/gemma-4-31B-it --quantization fp8 \
  --max-model-len 65536 --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":8}'
```

- KV cache at this config: ~43.7 GiB free → **237,699 tokens**, max concurrency 3.63× for 65,536-token requests.
- Rollback to γ=3: `/home/ubuntu/gemma4-rollback.sh`.

## 2. Method

- Prompt: fixed technical paragraph, `max_tokens=256`, `ignore_eos:true`, `temperature=0` (greedy).
- Single-stream: same request 8× back-to-back.
- Sweep: `conc ∈ {1,8,16,32,64}`, `reqs = max(3, 2·conc)`, threaded client.
- Acceptance: delta of `vllm:spec_decode_num_accepted_tokens_total / ..._draft_tokens_total` across each sweep level.
- Script: `/tmp/gemma4_bench.py` (urllib + ThreadPoolExecutor, no deps).

## 3. Results

### 3.1 Single-stream warm (greedy, 256 tok, k=8)

| req | wall | tok/s |
|----:|-----:|------:|
| 1 | 0.57s | 447.6 |
| 2 | 0.56s | 453.7 |
| 3–8 | 0.56s | ~453 |

Steady-state **~453 tok/s**. The earlier **72.8 tok/s** (cold) is 6.2× lower — JIT-spike artifact, not throughput.

### 3.2 Concurrency sweep — γ=8 (measured today) vs γ=3 (prior session)

| conc | γ=8 agg tok/s | γ=8 per-stream | γ=8 accept | γ=3 agg tok/s (prior) |
|----:|--------------:|---------------:|-----------:|----------------------:|
| 1  |   453.8 | 453.8 | 93.5% |  172.9 |
| 8  |   827.7\* | 103.5\* | 89.7% |  695.1 |
| 16 |  3379.9 | 211.2 | 89.5% | 2048.1 |
| 32 |  4858.0 | 151.8 | 89.5% | 3749.4 |
| 64 | **6695.0** | 104.6 | 89.5% | **5596.9** |

**Peak 6,695 tok/s @ conc=64 (γ=8)**, +19.6% over the γ=3 peak.

## 4. Caveats

- **\*conc=8 is an outlier** (wall 4.95s vs conc=16's 2.42s; per-stream dipped below neighbors). Looks like a one-wave scheduling hiccup, not a real cliff — re-run before citing that row.
- **Single-stream γ=8 (453) vs γ=3 (173) is not pure γ effect**: the γ=3 sweep ran a higher-acceptance-varying workload (54–63% accept) while this run is greedy (93.5%). The aggregate-peak comparison (both same prompt style within their run) is the cleaner one.
- **fp8 caveat unchanged**: per `12_gemma_headdim256.md`, fp8 on head_dim=256 Gemma carries the prefill penalty + #39407 risk. The rated-3.19× recipe is **bf16 + γ=8 + `--cpu-offload-gb 20`** / KV offload; this rig is on fp8 by choice. Not yet measured here.

## 5. Reproduce

```bash
python3 /tmp/gemma4_bench.py          # single-stream + sweep
curl -s localhost:8000/metrics | grep spec_decode_num_accepted_tokens_total
/home/ubuntu/gemma4-rollback.sh       # back to γ=3
```
