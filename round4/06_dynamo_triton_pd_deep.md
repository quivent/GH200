# GH200 Inference — Round 4: Dynamo / Triton / P/D Deep Dive (LLM-only)

**Author**: Research Agent (Round 4, LLM-only deep dive)
**Date**: 2026-05-25
**Scope**: Production-ready LLM-only serving on GH200 aarch64. Concrete configs and recipes for Dynamo 1.1.1, vLLM NixlConnector P/D, Triton 26.04 SBSA + TRT-LLM, llm-d v0.7, KServe.
**Stack constraint reminder**: FP8 broken on DiT (diffusion). LLMs on Hopper Hopper-class GPUs (sm_90) **do** support FP8 in hardware, and most published NVIDIA P/D recipes use FP8 for Llama-3.3-70B. We surface FP8 recipes verbatim (they are the only ones NVIDIA publishes) but flag BF16 fallbacks where the user needs them.
**Builds on**: round1/07_serving_frameworks.md and round3/07_serving_verify.md.

---

## TL;DR

1. **Single-node Dynamo 1.1.1 on one GH200 (OpenAI-compat)** — works today. Pull `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1` (multi-arch arm64+amd64), bring up etcd+NATS via Docker Compose, run `python -m dynamo.frontend` + `python -m dynamo.vllm`. Force BF16 via `--dtype bfloat16` if you don't want the FP8 path that the published recipes ship.
2. **Dynamo P/D on 2× GH200 NVL2** — no NVIDIA reference architecture exists; only viable port is from the H100/H200 `disagg-single-node` recipe (`tensor-parallel-size 2` for prefill, `tensor-parallel-size 4` for decode is the H100 layout — for 2× GH200, drop to TP=1 prefill / TP=1 decode; or co-locate same-process for NVLink KV transfer). Confirmed YAML structure: `DynamoGraphDeployment` CR with `Frontend` + `VllmPrefillWorker(subComponentType: prefill)` + `VllmDecodeWorker(subComponentType: decode)`, both using `NixlConnector` with `kv_role: kv_both`. **NIXL transport between two GH200 nodes uses InfiniBand/RoCE**, not NVL fabric; same-process intra-chassis can use UCX `cuda_ipc` over GPU-GPU NVLink (900 GB/s) but requires explicit `UCX_TLS` override.
3. **KV-cache-aware router formula** is exact and small: `logit = kv_overlap_score_weight × potential_prefill_blocks + potential_active_blocks` (selected worker = `argmin(logit)`). 8 documented CLI flags with defaults (full table below). `router-prefill-load-model aic` enables the active-inference-cache predictor — *off by default* (`none`). Queue policy options: `fcfs` (default), `wspt`, `lcfs`.
4. **vLLM NixlConnector** ships in upstream vLLM. Minimal P/D = prefiller (`kv_role: kv_producer`) + decoder (`kv_role: kv_consumer`) + toy proxy. `kv_role` is **functionally a placeholder** — actual roles are determined by the upper-level proxy; the connector itself doesn't distinguish producer/consumer. Default UCX transport with optional LIBFABRIC backend (used by Dynamo's EFA recipes).
5. **Triton 26.04 SBSA + TRT-LLM** on GH200: official SBSA image (Ubuntu 24.04, CUDA 13.2, TRT-LLM 1.2.1, vLLM 0.19.0) is the cleanest single-node multi-backend path. Build TRT-LLM engine with `--dtype bfloat16 --tp_size 1 --pp_size 1` for one GH200, then serve via `launch_triton_server.py --world_size=1 --model_repo=/triton_model_repo`. OpenAI-compat frontend now in 26.04. **Caveat**: Triton's vLLM backend can't do TP>1 with default explicit-model-control; if you need TP>1, use TRT-LLM backend or Dynamo's vLLM backend instead.
6. **llm-d v0.7.0 (2026-05-12)** added a `llm-d-cuda-gb200` image variant — but **no ARM64/GH200 image**. Issue #281 (filed 2025-09-30) remains **open, no assignees, no PRs, "Todo" status**. v0.7 also requires NVIDIA driver 580+ (CUDA 13.0.2 runtime). **Not GH200-deployable today**; reasonable bet for ARM64 is post-v0.8 or later (no public roadmap).
7. **KServe + vLLM on aarch64** is a viable production frontend. The KServe `huggingface` ServingRuntime auto-selects vLLM backend; on aarch64 you must replace the default container image with a community GH200 vLLM build (`drikster80/vllm-aarch64-openai` or `ghcr.io/abacusai/gh200-llm/...`) and pin `kubernetes.io/arch: arm64` in nodeSelector. KServe-on-GH200 is community-validated, not NVIDIA-blessed.

---

## 1. Dynamo 1.1.1 single-GH200 OpenAI-compat serving — minimal config

### Container & infra

- **Image**: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1` (multi-arch: amd64 + **arm64**)
- **Required infra**: etcd (worker registry / service discovery), NATS (KV-cache event propagation). Both **optional in 1.1.x** via `--discovery-backend file` flag — that's the *simplest* single-node path and avoids the docker-compose dance.

### Option A: minimal "no etcd" path (1 GPU, OpenAI-compat)

```bash
# Pull
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1

# Run container (interactive)
docker run --gpus all -it --rm \
  --network=host \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1 bash

# Inside container — terminal 1: frontend (OpenAI-compatible on :8000)
python3 -m dynamo.frontend --discovery-backend file --port 8000

# Inside container — terminal 2: vLLM worker
python3 -m dynamo.vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --discovery-backend file \
  --kv-events-config '{"enable_kv_cache_events": false}'
```

Then from the host:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role":"user","content":"Hello."}],
    "max_tokens": 64
  }'
```

The frontend exposes `/openapi.json` (OpenAPI 3 spec) and accepts `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, plus the Anthropic Messages API (added in 1.1.0).

### Option B: production-style with etcd+NATS

```yaml
# deploy/docker-compose.yml
services:
  etcd-server:
    image: bitnamilegacy/etcd:3.6.1
    network_mode: host
    environment:
      ALLOW_NONE_AUTHENTICATION: "yes"
      ETCD_ADVERTISE_CLIENT_URLS: "http://0.0.0.0:2379"
  nats-server:
    image: nats:2.11.4
    network_mode: host
    command: "-js"   # JetStream enabled if you want durable KV events
```

```bash
docker compose up -d
# Then frontend + worker without --discovery-backend file:
python3 -m dynamo.frontend --port 8000
python3 -m dynamo.vllm --model meta-llama/Llama-3.1-70B-Instruct --dtype bfloat16
```

### BF16 vs FP8 on a single GH200

- Dynamo's **published recipes** for Llama-3-70B (`recipes/llama-3-70b/vllm/agg`) use `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic`. Hopper sm_90 has hardware FP8 (E4M3/E5M2) and this is a separate kernel path from the broken-on-this-stack FP8 *DiT* attention path. **For LLMs, FP8 on GH200 is generally working**; this stack's constraint primarily bit diffusion models.
- If a user needs BF16 anyway: drop the `-FP8-dynamic` suffix from the model id and pass `--dtype bfloat16`. Verified flag-equivalent to the recipe.

### Single-GH200 memory budget

- Llama-3.3-70B BF16 weights ≈ **140 GB** → does **not** fit in a single GH200 96 GB HBM3e card.
  - Path: use a 480 GB GH200 SKU (rare), OR FP8 (RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic ≈ 70 GB), OR offload to LPDDR5X (480 GB unified) and pay perf, OR use 2× GH200 with TP=2.
- Llama-3.1-8B BF16 ≈ 16 GB → fits easily on one GH200.
- For "one GH200" demos, **8B BF16 or 70B FP8** are the practical model sizes.

---

## 2. Dynamo P/D across 2× GH200 — derived recipe (no NVIDIA reference exists)

### What NVIDIA actually ships (the published H100/H200 recipes)

Verified by fetching the raw YAMLs from `github.com/ai-dynamo/dynamo/main/recipes/llama-3-70b/vllm/`:

| Recipe | Layout | Prefill | Decode | Notes |
|---|---|---|---|---|
| `agg/deploy.yaml` | 4× H100/H200 | (single worker, TP=4) | n/a | Aggregated baseline |
| `disagg-single-node/deploy.yaml` | 8× H100/H200, 1 node | 2 workers × TP=2 (= 4 GPUs) | 1 worker × TP=4 (= 4 GPUs) | Co-located in 1 K8s pod set |
| `disagg-multi-node/deploy.yaml` | 16× H100/H200, 2 nodes | 1 worker × TP=8 = 1 node | 1 worker × TP=8 = 1 node | Cross-node KV via NixlConnector |

The disagg-multi-node config is the closest analog for 2× GH200. Key flags from the actual YAML (`recipes/llama-3-70b/vllm/disagg-multi-node/deploy.yaml`):

```bash
# Prefill worker (8 GPUs in H100 recipe; for GH200 → 1 GPU TP=1)
python3 -m dynamo.vllm \
  --model RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic \
  --served-model-name RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --block-size 128

# Decode worker (8 GPUs in H100 recipe; for GH200 → 1 GPU TP=1)
python3 -m dynamo.vllm \
  --model RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --block-size 128
```

Notes from the YAML:
- Both workers use **`kv_role: "kv_both"`** (Dynamo's chosen pattern; the role separation comes from `--disaggregation-mode prefill` on the prefill side, not the NixlConnector field). Confirmed by upstream vLLM docs: *"NixlConnector currently does not distinguish kv_role; the actual prefiller/decoder roles are determined by the upper-level proxy"*.
- **`--no-enable-prefix-caching`** is mandatory for disagg in the recipe (prefix cache would compete with KV transfer).
- **`--block-size 128`** (vs default 16) — larger blocks reduce overlap granularity but reduce KV transfer round-trips.
- `sharedMemory: 80Gi` per pod (for IPC + NIXL staging).
- Prefill worker has `subComponentType: prefill`; decode worker has `subComponentType: decode`. The Dynamo operator uses this for routing.

### Adaptation for 2× GH200 NVL2 (1 prefill + 1 decode)

```yaml
# Adapted DynamoGraphDeployment for GH200 NVL2
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: llama3-70b-disagg-gh200nvl2
spec:
  backendFramework: vllm
  services:
    Frontend:
      componentType: frontend
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1
          workingDir: /workspace/examples/backends/vllm
      replicas: 1
    VllmPrefillWorker:
      componentType: worker
      subComponentType: prefill
      envFromSecret: hf-token-secret
      sharedMemory:
        size: 80Gi
      extraPodSpec:
        nodeSelector:
          kubernetes.io/arch: arm64
        affinity:
          podAntiAffinity:
            requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchExpressions:
                - key: nvidia.com/dynamo-sub-component-type
                  operator: In
                  values: [decode]
              topologyKey: kubernetes.io/hostname
        mainContainer:
          env:
            - name: MODEL_PATH
              value: "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"   # or meta-llama/Llama-3.3-70B-Instruct + --dtype bfloat16
            - name: UCX_TLS
              value: "rc,ud,tcp,cuda_copy,cuda_ipc"
            - name: UCX_NET_DEVICES
              value: "all"
          args:
          - >-
            python3 -m dynamo.vllm --model $MODEL_PATH
            --tensor-parallel-size 1 --data-parallel-size 1
            --disaggregation-mode prefill
            --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
            --gpu-memory-utilization 0.90 --no-enable-prefix-caching --block-size 128
          command: [/bin/sh, -c]
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1
      replicas: 1
      resources:
        limits:    { gpu: "1" }
        requests:  { gpu: "1" }
    VllmDecodeWorker:
      componentType: worker
      subComponentType: decode
      # ... (mirror of prefill but without --disaggregation-mode prefill)
      replicas: 1
      resources:
        limits:    { gpu: "1" }
        requests:  { gpu: "1" }
```

### Key adaptation decisions vs the H100 reference

| Decision | H100 multi-node recipe | GH200 NVL2 adaptation | Rationale |
|---|---|---|---|
| GPUs/worker | 8 (TP=8) | 1 (TP=1) | 2× GH200 = 2 GPUs total |
| Cross-node fabric | IB/Spectrum-X | IB/Spectrum-X (or same-process NVLink-via-cuda_ipc if you skip K8s) | Dynamo K8s guide: "NVLink cannot be used between Kubernetes pods" |
| Architecture | amd64 | **arm64 nodeSelector required** | GH200 is aarch64 |
| Container image | `vllm-runtime:1.1.1` | same (multi-arch confirmed) | unchanged |
| Quantization | FP8 dynamic | FP8 (default) or BF16 (`--dtype bfloat16`) | 70B BF16 = 140 GB → fits across 2× GH200 (~70 GB each) with TP=2; FP8 fits on either layout |

### To actually exploit the 900 GB/s GPU-GPU NVLink on NVL2

The pod-isolated K8s deployment above will fall back to IB/RoCE (the per-NIC ~50 GB/s of the 2-2-3-400 RA). To get NVLink, you must either:
1. **Co-locate** prefill + decode in the *same pod* with `--shareProcessNamespace: true`, OR
2. **Skip K8s** and run both workers as siblings under a `torchrun`-style launcher in one process tree, with `UCX_TLS=cuda_copy,cuda_ipc,...`.

Path (2) is what the H100 `disagg-single-node` recipe effectively does (8 GPUs, 1 node, same pod set). For 2× GH200 it's the recommended same-chassis layout.

### Status of MNNVL (multi-node NVLink) for GH200

- UCX has `UCX_CUDA_IPC_ENABLE_MNNVL=y` for cross-server NVL fabric.
- **Currently broken on GB200**: NIXL issue [#1614](https://github.com/ai-dynamo/nixl/issues/1614) — `NIXL_ERR_BACKEND` reproducibly. GH200 NVL2 is not GB200 NVL72 so the bug may not apply, but no NVIDIA-published recipe exercises it on GH200.
- **Default container ships with NVLINK disabled in UCX** (Dynamo issue #493, closed not-planned). Must explicitly override `UCX_TLS`.

---

## 3. KV-cache-aware router — formula and tunables

### Exact formula

From NVIDIA Dynamo router docs (verified):

```
logit_worker_w = kv_overlap_score_weight × potential_prefill_blocks_w
              + potential_active_blocks_w

selected_worker = argmin_w(logit_w)
```

(With `--router-temperature > 0`, softmax sampling is applied over `-logit`.)

Where:
- `potential_prefill_blocks_w` = newly computed KV blocks if request lands on worker *w* (= `total_input_blocks − overlap_with_w`).
- `potential_active_blocks_w` = total active KV blocks already on worker *w* (decode load).
- `kv_overlap_score_weight` (default **1.0**) scales the prefill-cost contribution. Higher → cache-locality first (better TTFT). Lower → load distribution first (better ITL).

Cost intuition: at weight 0 → pure load balancing; at weight ∞ → pure prefix matching.

### Tunable parameters (Dynamo 1.1.1)

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--router-mode` | `DYN_ROUTER_MODE` | `round-robin` | One of: `round-robin`, `kv`, `random`, `least-loaded`, `device-aware-weighted`, `direct`. **Set to `kv` to enable the formula above.** |
| `--router-temperature` | `DYN_ROUTER_TEMPERATURE` | `0.0` | Routing randomness (softmax over logits). `0.0` = deterministic argmin. |
| `--kv-cache-block-size` | `DYN_KV_CACHE_BLOCK_SIZE` | backend-default | Block size used for overlap calculation. Must match worker's `--block-size`. |
| `--router-kv-events` | `DYN_ROUTER_USE_KV_EVENTS` | `true` | Subscribe to worker KV cache events (NATS) for real-time prefix-tree updates. |
| `--router-kv-overlap-score-weight` | `DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT` | `1.0` | The weight in the formula above. |
| `--router-track-prefill-tokens` | n/a | enabled | Include prefill-side load in worker accounting. |
| `--router-prefill-load-model` | n/a | `none` | `none` or **`aic`** (Active Inference Cache predictor — uses a learned cost model for prefill duration). |
| `--router-queue-threshold` | n/a | `4.0` | Queue fraction at which priority scheduling kicks in. |
| `--router-queue-policy` | `DYN_ROUTER_QUEUE_POLICY` | `fcfs` | One of: `fcfs`, `wspt` (weighted shortest processing time), `lcfs`. |
| `--serve-indexer` | n/a | `false` | Frontend serves remote indexer (advanced). |
| `--use-remote-indexer` | n/a | `false` | Query worker's remote indexer (advanced). |

### Recommended values by workload

| Workload | Settings | Rationale |
|---|---|---|
| **High cache-hit (chat with long shared system prompts, RAG with repeated context)** | `--router-mode kv --router-kv-overlap-score-weight 1.5 --router-prefill-load-model aic` | Prioritize prefix reuse; AIC compensates for varying prefill cost when prompts have different shapes. |
| **Low cache-hit / sparse prefixes (one-shot completions, varied prompts)** | `--router-mode kv --router-kv-overlap-score-weight 0.5 --router-temperature 0.1` | Prefer load balance; small temperature jitter prevents thundering-herd onto one worker. |
| **Latency-critical TTFT** | `--router-kv-overlap-score-weight 1.5 ... 2.0`, `--router-queue-policy wspt` | High weight → minimize prefill; WSPT shortens average wait. |
| **Throughput-critical (ITL/tok/s)** | `--router-kv-overlap-score-weight 0.5 ... 0.8`, `--router-queue-policy fcfs` | Spread decode load evenly. |
| **Heterogeneous worker GPUs** | `--router-mode device-aware-weighted` | Per-GPU weighting; skip KV formula entirely. |
| **Baseline / debug** | `--router-mode round-robin` | No KV overhead. |

### How to set on frontend

```bash
python3 -m dynamo.frontend \
  --port 8000 \
  --router-mode kv \
  --router-kv-overlap-score-weight 1.0 \
  --router-prefill-load-model aic \
  --router-queue-policy fcfs \
  --router-temperature 0.0
```

Equivalents as env: `DYN_ROUTER_MODE=kv DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT=1.0`.

### Tuning advice from NVIDIA docs

- If your **prefix-tree depth is short relative to input sequence length** (i.e., most of each request is novel), *reduce* `overlap_score_weight`. Alternatively normalize the overlap score by input length.
- For workloads with **dynamic per-worker cache patterns**, increase `--router-update-interval` (if present in your build — flag is documented in some 0.7-era docs but not in the latest CLI; the recommended mechanism in 1.1.x is the event-driven `--router-kv-events true` default).

---

## 4. NIXL + vLLM NixlConnector + GH200 — deployable config

### Connector config reference

Full `kv-transfer-config` fields:

| Field | Values | Default | Notes |
|---|---|---|---|
| `kv_connector` | `"NixlConnector"` | — | Required to enable NIXL path |
| `kv_role` | `kv_producer` / `kv_consumer` / `kv_both` | — | **Note: NixlConnector currently ignores this field; actual roles come from upper-level proxy** |
| `kv_load_failure_policy` | `"fail"` / `"recompute"` | `fail` | What to do if KV transfer fails |
| `kv_buffer_device` | `"cuda"` / `"cpu"` | `cuda` | CPU transfer added 2026 (PR #18293) — useful for GH200 where LPDDR5X is C2C-attached |
| `kv_connector_extra_config.backends` | `["UCX"]`, `["LIBFABRIC"]` | `["UCX"]` | UCX default; LIBFABRIC for EFA / Slingshot |
| `kv_connector_extra_config.enable_cross_layers_blocks` | `"True"` | off | Experimental — cross-layer transfer |
| `enable_permute_local_kv` | `"True"` | off | Experimental — heterogeneous KV layout |

### Environment variables

| Var | Default | Notes |
|---|---|---|
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | `5600` | Handshake / metadata channel |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | `localhost` | For cross-node, set to the prefiller's reachable hostname |
| `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` | `480` (s) | KV-cache release timeout |
| `UCX_TLS` | `all` | **For GH200 same-node NVLink:** `rc,ud,tcp,cuda_copy,cuda_ipc` |
| `UCX_NET_DEVICES` | `all` | Restrict to IB devices if needed: e.g. `mlx5_0:1` |
| `UCX_CUDA_IPC_ENABLE_MNNVL` | `n` | Set to `y` for cross-server NVL fabric (buggy on GB200; untested GH200) |

### Two-process P/D on a single GH200 NVL2 chassis (1 prefiller + 1 decoder)

```bash
# Prefiller — GPU 0, side-channel 5600
CUDA_VISIBLE_DEVICES=0 \
UCX_TLS=rc,ud,tcp,cuda_copy,cuda_ipc \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --port 8100 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --block-size 128 \
  --gpu-memory-utilization 0.90 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'

# Decoder — GPU 1, side-channel 5601
CUDA_VISIBLE_DEVICES=1 \
UCX_TLS=rc,ud,tcp,cuda_copy,cuda_ipc \
UCX_NET_DEVICES=all \
VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --port 8200 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --block-size 128 \
  --gpu-memory-utilization 0.90 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'

# Toy proxy (from vLLM tests)
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
  --port 8192 \
  --prefiller-hosts localhost --prefiller-ports 8100 \
  --decoder-hosts  localhost --decoder-ports  8200
```

GH200-specific notes:
- 70B BF16 (140 GB weights) **will not fit** in either 96 GB GH200 alone — you'd need TP=2 *within each role*, which means 2× GH200 per role (i.e., 4 GH200s total). Or use FP8 dynamic (70 GB) or a 480 GB GH200 SKU.
- For TP=2 P/D on a NVL2 chassis (1 prefiller using both GH200s for TP=2 weight-sharding, 1 decoder also using both GH200s) — that's mathematically impossible with only 2 GPUs unless prefill and decode time-share the GPUs (not P/D anymore).
- Realistic 70B BF16 P/D on GH200 needs **4 GH200s** (2 prefill + 2 decode, each TP=2) — i.e., 2 NVL2 chassis.

### NIXL transport summary (carry-over from round3)

- Same-process intra-chassis: UCX `cuda_ipc` over GPU-GPU NVLink (900 GB/s).
- K8s pod-isolated: IB/RoCE (~50 GB/s on 400 Gbps Spectrum-X / IB) per the GH200 NVL2 RA's `-3-400` NIC config.
- TCP fallback: ~200-500× degradation in TTFT — never run in TCP for production.
- ROCm equivalent of NIXL = RIXL (mentioned in vLLM docs).

---

## 5. Triton 26.04 SBSA + TRT-LLM backend on GH200 — Llama-3.3-70B BF16

### Container

- **NGC tag**: `nvcr.io/nvidia/tritonserver:26.04-py3-sbsa` (base) or `nvcr.io/nvidia/tritonserver:26.04-trtllm-python-py3-sbsa` (TRT-LLM bundle).
- Triton 26.04 = `v2.68.0`, released 2026-04-28. CUDA 13.2, Ubuntu 24.04, TRT-LLM 1.2.1, vLLM 0.19.0, Python 3.12, sm_75+ (covers Hopper sm_90).
- **Known limit**: Triton client wheels are not on PyPI for ARM SBSA — extract from the SBSA SDK image manually.

### Step 1 — Build TRT-LLM engine for Llama-3.3-70B BF16 on a single GH200

```bash
docker run --gpus all -it --rm \
  -v $PWD:/workspace \
  nvcr.io/nvidia/tritonserver:26.04-trtllm-python-py3-sbsa bash

# Inside container
cd /workspace
# Get model weights
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct \
  --local-dir /workspace/llama-3.3-70b-hf

# Convert HF -> TRT-LLM checkpoint (BF16, TP=1 - single GH200)
# Note: 70B BF16 weights are 140 GB - DOES NOT FIT on one 96 GB GH200.
# Either use TP=2 across two GH200s, or use a 480 GB SKU, or quantize to FP8/INT4.
# Below: TP=2 example (build on whatever node; deploy on 2× GH200 with NVLink).
python3 /opt/tensorrt_llm/examples/llama/convert_checkpoint.py \
  --model_dir /workspace/llama-3.3-70b-hf \
  --output_dir /workspace/tllm_ckpt_2gpu_bf16 \
  --dtype bfloat16 \
  --tp_size 2 \
  --pp_size 1

# Build TRT-LLM engine
trtllm-build \
  --checkpoint_dir /workspace/tllm_ckpt_2gpu_bf16 \
  --output_dir /workspace/engines/llama-3.3-70b/bf16/tp2 \
  --gemm_plugin auto \
  --max_input_len 8192 \
  --max_seq_len 16384 \
  --max_batch_size 64 \
  --use_paged_context_fmha enable
```

For a **single GH200 96 GB**, switch to FP8: change the convert step to `--use_fp8` (or use a pre-quantized `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` checkpoint) with `--tp_size 1`. FP8 on Hopper sm_90 LLM kernels is broadly stable in TRT-LLM 1.2.1.

### Step 2 — Build Triton model repository

Standard four-model ensemble (`preprocessing`, `tensorrt_llm`, `postprocessing`, `ensemble`). The `tensorrt_llm/config.pbtxt`:

```pbtxt
name: "tensorrt_llm"
backend: "tensorrtllm"
max_batch_size: 64

model_transaction_policy { decoupled: true }
input  [{ name: "input_ids" data_type: TYPE_INT32 dims: [-1] }, ...]
output [{ name: "output_ids" data_type: TYPE_INT32 dims: [-1, -1] }, ...]

instance_group [{ count: 1 kind: KIND_CPU }]

parameters {
  key: "engine_dir"
  value: { string_value: "/workspace/engines/llama-3.3-70b/bf16/tp2" }
}
parameters {
  key: "batching_strategy"
  value: { string_value: "inflight_fused_batching" }
}
parameters {
  key: "batch_scheduler_policy"
  value: { string_value: "MAX_UTILIZATION" }
}
parameters {
  key: "kv_cache_free_gpu_mem_fraction"
  value: { string_value: "0.90" }
}
parameters {
  key: "enable_kv_cache_reuse"
  value: { string_value: "true" }
}
```

### Step 3 — Launch Triton with OpenAI-compatible frontend

```bash
# TP=2 needs MPI-based launch
python3 /app/scripts/launch_triton_server.py \
  --world_size=2 \
  --model_repo=/workspace/triton_model_repo \
  --http_port=8000 \
  --grpc_port=8001 \
  --metrics_port=8002

# Then start OpenAI-compat frontend (new in 26.04)
tritonfrontend \
  --openai-compatible \
  --http-port=8080 \
  --triton-grpc-url=localhost:8001
```

### Step 4 — Test

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ensemble",
    "messages": [{"role":"user","content":"What is GH200?"}],
    "max_tokens": 128
  }'
```

### Caveats on GH200

- **vLLM backend in Triton has the TP>1 issue** (default `distributed_executor_backend` under explicit model control). For TP>1 LLMs, use TRT-LLM backend (above) or Dynamo's vLLM backend instead.
- **TRT-LLM backend may core-dump at shutdown** (cosmetic) — known issue carried over since 25.x.
- For TP across two GH200s on NVL2, ensure `NCCL_TOPO_FILE` and the GH200-aware NCCL topology are correctly loaded (NCCL 2.20+ handles GH200 NVL2 natively).
- **FP8 on the GH200 LLM path**: known working in TRT-LLM 1.2.1 on sm_90; this stack's FP8 problems were on DiT/diffusion attention, not LLM GEMMs. Still: validate with `trtllm-bench` before committing FP8 to prod.

---

## 6. llm-d v0.7.0 — GH200/ARM64 readiness

### Release timeline (verified via GitHub API)

| Version | Date | Major change |
|---|---|---|
| v0.7.0 | **2026-05-12** | CUDA 13.0.2 runtime (driver 580+); **new `llm-d-cuda-gb200` image variant**; default deployment shifted to standalone mode (generic proxy instead of full gateway) |
| v0.6.0 | 2026-04-03 | vLLM 0.17.1; Gateway API v1.4.0; XPU/HPU re-enabled |
| v0.5.1 | 2026-03-05 | RoCM/HPU images added back |
| v0.5.0 | 2026-02-04 | Workload-Variant-Autoscaler core; Istio 1.28.1 |
| v0.4.0 | 2025-11-26 | XPU/CPU/AWS-EFA variants |
| v0.3.1 | 2025-11-06 | Mentions "ARM support" in release notes (the only time ARM is mentioned in any llm-d release body — but this is *not* a published ARM image) |

### v0.7.0 image variants (verified from release body)

| Image | Hardware target | Architecture |
|---|---|---|
| `llm-d-cuda:v0.7.0` | NVIDIA CUDA generic | amd64 |
| `llm-d-cuda (debug):v0.7.0` | NVIDIA CUDA generic | amd64 |
| **`llm-d-cuda-gb200:v0.7.0`** | NVIDIA Grace-Blackwell GB200 | **amd64 only (despite GB200 being aarch64)** — likely Grace runtime via amd64 cross-platform; or aarch64-tagged but not enumerated; needs `docker manifest inspect` to confirm |
| `llm-d-aws (EFA):v0.7.0` | AWS Trainium/EFA | amd64 |
| `llm-d-xpu:v0.7.0` | Intel Data Center GPU | amd64 |
| `llm-d-hpu:v0.7.0` | Intel Gaudi (HPU) | amd64 |
| `llm-d-rocm:v0.7.0` | AMD ROCm | amd64 |
| `llm-d-cpu:v0.7.0` | CPU only | amd64 |

**Notable**: no `llm-d-cuda-arm64` or `llm-d-cuda-gh200` variant.

### Issue #281 status (ARM64/GH200 expansion)

- **Opened**: 2025-09-30 (8 months old)
- **State**: **Open**
- **Assignees**: none
- **Labels**: none
- **Linked PRs**: none
- **Milestone**: none
- **Project status**: "Todo"
- **Recent activity**: no 2026 comments

### When will it work?

There is **no public roadmap commitment** for ARM64 on llm-d. The `llm-d-cuda-gb200` image is the closest signal — but it targets the GB200 NVL72 platform (Blackwell + Grace), not GH200. The fact that even GB200 (which is the higher-priority new platform) shipped without explicit aarch64 tagging suggests llm-d's CUDA image is built for the amd64 control plane while the GPU side runs whatever container is shipped — i.e., the image arch and the host arch can differ for x86 hosts running aarch64 emulation, but a true aarch64 image is still pending.

**Practical guidance**: Don't plan llm-d on GH200 for May 2026. If you must use llm-d-style routing on GH200, **Dynamo 1.1.1 is the equivalent (and more mature) choice today** — it ships multi-arch arm64 across all three of vLLM/SGLang/TRT-LLM standard runtimes. The earliest llm-d ARM64 could realistically ship would be a v0.8 (if added to a Q3 2026 cycle), but with no PR, no assignee, and no roadmap entry, that's a guess, not a forecast.

---

## 7. KServe + vLLM on aarch64 — alternative production frontend

### Why KServe

If you have a Kubernetes-native deployment with Knative autoscaling and want **OpenAI-compatible serving without writing operator CRs** (the Dynamo path requires the `DynamoGraphDeployment` operator), KServe's `huggingface` ServingRuntime is the cleanest fit. It auto-selects vLLM backend when available.

### Path A — Use upstream KServe huggingface runtime with custom aarch64 vLLM image

The default KServe huggingface runtime image targets amd64. For GH200 you must override the container image and pin to aarch64 nodes.

**ServingRuntime YAML (custom for GH200)**:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ClusterServingRuntime
metadata:
  name: kserve-huggingfaceserver-gh200
spec:
  annotations:
    prometheus.kserve.io/path: /metrics
    prometheus.kserve.io/port: "8080"
  supportedModelFormats:
    - autoSelect: true
      name: huggingface
      priority: 1
      version: "1"
  protocolVersions:
    - v2
  containers:
    - name: kserve-container
      image: ghcr.io/abacusai/gh200-llm/llm-train-serve:latest   # multi-arch GH200 vLLM
      command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
      args:
        - --model
        - {{.Name}}
        - --port
        - "8080"
        - --dtype
        - bfloat16
      resources:
        requests:
          cpu: "4"
          memory: 64Gi
          nvidia.com/gpu: "1"
        limits:
          cpu: "8"
          memory: 128Gi
          nvidia.com/gpu: "1"
  nodeSelector:
    kubernetes.io/arch: arm64
    nvidia.com/gpu.product: NVIDIA-GH200-480GB   # or 96GB SKU
```

**InferenceService**:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama31-8b-gh200
spec:
  predictor:
    model:
      modelFormat:
        name: huggingface
      runtime: kserve-huggingfaceserver-gh200
      storageUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
      resources:
        requests:
          nvidia.com/gpu: "1"
```

### Path B — Use the new `LLMInferenceService` (KServe + llm-d integration)

KServe added an `LLMInferenceService` resource that delegates to llm-d under the covers. **Not usable on GH200 today** (llm-d v0.7 has no aarch64 image — see §6). When llm-d adds ARM64, this becomes the most ergonomic path.

### Community aarch64 vLLM images for KServe

| Image | Source | Notes |
|---|---|---|
| `drikster80/vllm-aarch64-openai:latest` | drikster80 | Built from upstream vLLM with `torch_cuda_arch_list=9.0+PTX`. PyTorch nightly. Experimental. |
| `drikster80/vllm-gh200-openai:latest` | drikster80 | GH200-tuned variant of the above. |
| `ghcr.io/abacusai/gh200-llm/llm-train-serve:latest` | Abacus.AI | Multi-arch (amd64+arm64), public. Used in their internal training+serving pipeline. |

### Caveats

- These community images lag upstream vLLM by 1-2 releases.
- They typically don't include NixlConnector configured for IB/RoCE — fine for aggregated serving, **not** for P/D disagg without rebuilding.
- KServe **does not** ship Dynamo's KV-aware router; routing inside the runtime is single-vLLM-instance only. For multi-replica KV-aware routing on KServe, you'd front it with the Gateway API Inference Extension (GAIE) — same one used by llm-d.

---

## Consolidated decision matrix

| Goal | Stack | Confidence |
|---|---|---|
| Single GH200, OpenAI-compat, simplest path | Dynamo 1.1.1 vLLM-runtime + `--discovery-backend file` + `--dtype bfloat16` | **High** |
| Single GH200, multi-backend (LLM + ONNX + custom) | Triton 26.04 SBSA + TRT-LLM backend | **High** |
| 2× GH200, P/D disagg, K8s-native | Dynamo 1.1.1 `DynamoGraphDeployment`, derived from `disagg-multi-node` recipe (drop TP=8 → TP=1) | **Medium** (no published recipe) |
| 2× GH200, P/D disagg, max NVLink bandwidth | Same-process two-process vLLM + NixlConnector, `UCX_TLS=...,cuda_ipc` | **Medium** (works but requires explicit UCX override) |
| KV-aware routing across N GH200s | Dynamo 1.1.1 frontend `--router-mode kv` | **High** (formula and flags well-documented) |
| KServe deployment | KServe ClusterServingRuntime with abacusai/drikster80 aarch64 vLLM image + arm64 nodeSelector | **Medium** (community images, lag upstream) |
| llm-d deployment | **Skip**. Not GH200-ready before v0.8+ at earliest. | **High** (clearly skip) |

---

## Sources (Round 4, with dates)

1. Dynamo recipes README (raw): https://raw.githubusercontent.com/ai-dynamo/dynamo/main/recipes/README.md (2026-05)
2. Dynamo Llama-3-70B vLLM agg deploy.yaml: https://raw.githubusercontent.com/ai-dynamo/dynamo/main/recipes/llama-3-70b/vllm/agg/deploy.yaml
3. Dynamo Llama-3-70B vLLM disagg-single-node deploy.yaml: https://raw.githubusercontent.com/ai-dynamo/dynamo/main/recipes/llama-3-70b/vllm/disagg-single-node/deploy.yaml
4. Dynamo Llama-3-70B vLLM disagg-multi-node deploy.yaml: https://raw.githubusercontent.com/ai-dynamo/dynamo/main/recipes/llama-3-70b/vllm/disagg-multi-node/deploy.yaml
5. Dynamo KV-aware routing user guide (CLI flags, defaults): https://docs.nvidia.com/dynamo/user-guides/kv-cache-aware-routing
6. Dynamo KV router (v0.7.1 archive — cost formula): https://docs.nvidia.com/dynamo/v-0-7-1/components/router
7. Dynamo router design doc: https://docs.nvidia.com/dynamo/design-docs/component-design/router-design
8. Dynamo Llama-3-70B README: https://raw.githubusercontent.com/ai-dynamo/dynamo/main/recipes/llama-3-70b/README.md
9. vLLM NixlConnector usage guide: https://docs.vllm.ai/en/stable/features/nixl_connector_usage/
10. vLLM Disaggregated Prefilling (experimental): https://docs.vllm.ai/en/stable/features/disagg_prefill/
11. vLLM Dynamo integration: https://docs.vllm.ai/en/latest/deployment/integrations/dynamo/
12. vLLM PR #18293 (CPU transfer in NixlConnector, 2026): https://github.com/vllm-project/vllm/pull/18293
13. Triton 26.04 release notes: https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-04.html
14. Triton server v2.68.0 release: https://github.com/triton-inference-server/server/releases/tag/v2.68.0 (2026-04-28)
15. Triton TRT-LLM backend README: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html
16. Triton NGC container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver
17. Dynamo vLLM runtime NGC container: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime
18. Dynamo Spheron deployment guide (2026): https://www.spheron.network/blog/nvidia-dynamo-disaggregated-inference-guide/
19. Vultr Dynamo + vLLM deployment guide: https://docs.vultr.com/how-to-deploy-inference-using-nvidia-dynamo-and-vllm
20. Dynamo support matrix: https://docs.nvidia.com/dynamo/dev/resources/support-matrix
21. Dynamo release-artifacts (per-container arch): https://docs.nvidia.com/dynamo/dev/resources/release-artifacts
22. NIXL repo + backends list: https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md
23. NIXL issue #1614 (GB200 MNNVL bug, open): https://github.com/ai-dynamo/nixl/issues/1614
24. Dynamo issue #493 (UCX NVLink disabled in container, closed not-planned): https://github.com/ai-dynamo/dynamo/issues/493
25. llm-d releases (GitHub API, verified dates): https://api.github.com/repos/llm-d/llm-d/releases
26. llm-d v0.7.0 release notes: https://github.com/llm-d/llm-d/releases/tag/v0.7.0 (2026-05-12)
27. llm-d issue #281 (ARM64 expansion): https://github.com/llm-d/llm-d/issues/281
28. KServe huggingface runtime overview: https://kserve.github.io/website/docs/model-serving/generative-inference/overview
29. KServe ServingRuntime concepts: https://kserve.github.io/website/docs/concepts/resources/servingruntime
30. vLLM KServe integration: https://docs.vllm.ai/en/stable/deployment/integrations/kserve/
31. abacusai GH200 multi-arch image: https://github.com/abacusai/gh200-llm
32. drikster80 GH200 vLLM image: https://hub.docker.com/r/drikster80/vllm-gh200-openai
33. GH200 NVL2 Enterprise RA (2-2-3-400): https://developer.nvidia.com/blog/simplify-system-memory-management-with-the-latest-nvidia-gh200-nvl2-enterprise-ra/
34. AKS Dynamo on AKS Part 3 (2026-03-16): https://blog.aks.azure.com/2026/03/16/dynamo-on-aks-part-3

---

## Confidence

**High**:
- Dynamo 1.1.1 single-GH200 OpenAI-compat path works as written (multi-arch container confirmed; `--discovery-backend file` documented).
- KV-aware router cost formula is exact (`logit = w × prefill_blocks + active_blocks`) and verified across two NVIDIA docs.
- KV router has 8 documented CLI flags with defaults (table above).
- llm-d v0.7.0 (2026-05-12) confirmed via GitHub API; no ARM64 image; issue #281 stalled.
- Triton 26.04 SBSA arm64 container with TRT-LLM 1.2.1 + vLLM 0.19.0 exists and supports sm_90.
- vLLM NixlConnector's `kv_role` is functionally a placeholder; actual roles set by upper-level proxy (confirmed in vLLM docs).
- Llama-3-70B BF16 = 140 GB and does NOT fit on a single 96 GB GH200; FP8 (~70 GB) does.

**Medium**:
- The 2× GH200 NVL2 P/D recipe I derived is structurally sound (mirrors `disagg-single-node` minus 6 GPUs) but **not validated by NVIDIA**. The H100 recipe's TP=8 → my GH200 adaptation's TP=1 swap is mechanical; performance unverified.
- `--router-prefill-load-model aic` is documented as a flag but the AIC predictor's training data, accuracy, and tradeoffs aren't published in detail.
- KServe + aarch64 vLLM community image path works but lags upstream vLLM and isn't NVIDIA-supported.

**Low / gaps**:
- Whether the `kv-overlap-score-weight ∈ {0.5, 1.0, 1.5}` recommendations actually optimize GH200 P/D — no GH200 ablation published; values are from NVIDIA's generic tuning guide.
- Same-process NVLink-NIXL path on GH200 NVL2: untested by NVIDIA in a published recipe; bug surface from GB200 (#1614) suggests there may be GH200 issues too.
- Triton 26.04 OpenAI-compat frontend exact CLI (`tritonfrontend` binary) — release notes mention "explicit model control mode and model management in the OpenAI-compatible frontend" but the exact invocation isn't surfaced in 26.04 docs that I could fetch.
- llm-d's `cuda-gb200` image arch — release notes don't enumerate it as aarch64 explicitly; needs `docker manifest inspect` to confirm.
