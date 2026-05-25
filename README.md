# gh200_inference research dossier

Multi-round subagent research dossier for optimal inference on the NVIDIA GH200 Grace-Hopper superchip (Lambda Cloud, Ubuntu 24.04 aarch64) and alternatives. Started 2026-05-25.

See [INDEX.md](INDEX.md) for the master plan, agent assignments, and headline findings.
See [synthesis/EXEC_BRIEF.md](synthesis/EXEC_BRIEF.md) for the executive decision brief.

## Layout

| Path | Contents |
|---|---|
| `round1/` | 20 initial-sweep reports — one per research domain. Versions, configs, benchmark numbers, sources. |
| `round2/` | 8 cross-pollination reports (A-H) — each reads multiple Round-1 reports, reconciles contradictions. |
| `round3/` | 20 verification + deepening reports — live-verified pricing, GitHub issue threads, vendor catalog manifests. |
| `round4/` | 20 LLM-only deep dives — engines, model classes, workloads, $/Mtok, break-even. |
| `round5/` | 15 narrow-deep reports on Llama-3.3-70B Q4 + Qwen3.6-27B Q4 + comparable peer models. |
| `synthesis/` | Decision briefs distilling all rounds into actionable recommendations. |

## Constraint

FP8 is broken on the user's GH200/Wan stack. Round-2 forensic audit (`round2/B_fp8_forensic_audit.md`) and Round-3 corroboration narrowed this to a **DiT × online-FP8 quantization-path bug** reproducible on Ampere, Hopper, and Blackwell — *not* a GH200-hardware issue. Dense Hopper LLM FP8 is production-grade post-vLLM 0.19.1 (with head_dim and sliding-window caveats). All recommendations default to **bf16** for diffusion and bf16-or-4bit-AWQ for LLMs unless an explicit per-model validation gate is passed.
