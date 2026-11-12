# Continuous-batching rollout ablation — optimization-merged branch

Date: 2026-11-12. Node: A100-SXM4-40GB (gpua021). torch 2.15.0+cu132, transformers 5.22.2, deepspeed from `/u/msajid/projects/DeepSpeed` (branch `fix/paged-layer-lazy-init` = `fork/optimization-merged` + lazy-init fix), flashinfer 0.7.4, dskernels (blocked-flash JIT). bf16, seed 0, warmup 1, one measured iteration per cell, chunked prefill off everywhere, `--top-k -1`, `--save-response-ids`. Task shapes: 6k = 6144 prompt / 16 new (C=4, T=0.7, PP=0.9); 20k = 20480 prompt / 8 new (C=2, T=0.0, PP=1.0); 3k = 3072 prompt / 3072 new (C=4, T=0.7, PP=0.9).

## Levels

| Level | Definition | Flags |
|---|---|---|
| A | continuous batching + eager | `--no-graph-capture` |
| B | continuous batching + single graph | `--graph-capture-batch-sizes <C>` (one graph at the full continuous width) |
| C | continuous batching + multiple graphs | default bucket ladder |
| D | C + paged attention, prefill + decode together | `--use-paged-cache --enable-paged-prefill` |
| E | D + fused kernels | `--torch-compile` |
| F | D + FlashInfer kernels + compiled sampler | `--flashinfer-prefill --use-flashinfer-decode --compiled-sampler` |

F does not stack `--torch-compile` on E: the FlashInfer backends and inductor are alternative kernel strategies (FlashInfer custom ops are graph breaks), consistent with their separate isolation in `runs/optimization_ablation.md`.

## TL;DR

- **qwen3_4b 3k**; D (paged, multi-graph) vs A (eager): 2.74x total, 2.82x decode; C (ladder) vs B (single graph): 1.00x total; E (torch.compile) vs D: 1.08x total; F (FlashInfer + compiled sampler) vs D: 1.49x total
- **qwen3_8b 3k**; D (paged, multi-graph) vs A (eager): 2.46x total, 2.52x decode; C (ladder) vs B (single graph): 1.00x total; E (torch.compile) vs D: 1.07x total; F (FlashInfer + compiled sampler) vs D: 1.37x total

## qwen3_4b — 3k (batch 4, C=4, prompt 3072, 3072 new, T=0.7, top_p=0.9)

| Level | Prefill ms | Decode ms | Total ms | Tok/s | Peak MiB | Response SHA | Tokens vs A |
|---|---:|---:|---:|---:|---:|---|---|
| A_eager | 631.1 | 203426.5 | 207988.5 | 59.1 | 12857 | `34e10d98` | — |
| B_single_graph | 695.7 | 168298.9 | 172646.7 | 71.2 | 13788 | `6427f744` | 0/4 (min prefix 44) |
| C_multi_graph | 656.8 | 168295.0 | 172588.8 | 71.2 | 13788 | `6427f744` | 0/4 (min prefix 44) |
| D_paged | 667.5 | 72118.0 | 75969.8 | 161.8 | 11364 | `22a7a81e` | 0/4 (min prefix 1) |
| E_fused | 592.9 | 66504.9 | 70264.5 | 174.9 | 12940 | `2eab5ebc` | 0/4 (min prefix 1) |
| F_flashinfer_sampler | 734.8 | 46971.8 | 50925.2 | 241.4 | 11641 | `4c435487` | 0/4 (min prefix 1) |

Increments (total, decode):

| Step | Total speedup | Decode speedup | Peak delta |
|---|---:|---:|---|
| A (eager baseline) | 1.00x | 1.00x | — |
| B (single graph) vs A | 1.20x | 1.21x | +7.2% |
| C (bucket ladder) vs B | 1.00x | 1.00x | +0.0% |
| D (paged) vs C | 2.27x | 2.33x | -17.6% |
| E (torch.compile) vs D | 1.08x | 1.08x | +13.9% |
| F (FlashInfer + compiled sampler) vs D | 1.49x | 1.54x | +2.4% |

## qwen3_8b — 3k (batch 4, C=4, prompt 3072, 3072 new, T=0.7, top_p=0.9)

| Level | Prefill ms | Decode ms | Total ms | Tok/s | Peak MiB | Response SHA | Tokens vs A |
|---|---:|---:|---:|---:|---:|---|---|
| A_eager | 993.8 | 218022.1 | 222962.8 | 55.1 | 20807 | `5f4bd8aa` | — |
| B_single_graph | 1299.5 | 183272.0 | 188247.1 | 65.3 | 22063 | `4caab9ce` | 0/4 (min prefix 2) |
| C_multi_graph | 1000.4 | 183251.4 | 187897.6 | 65.4 | 22063 | `4caab9ce` | 0/4 (min prefix 2) |
| D_paged | 1024.0 | 86472.4 | 90715.1 | 135.5 | 19395 | `3073eac2` | 0/4 (min prefix 1) |
| E_fused | 942.7 | 80963.3 | 85140.2 | 144.3 | 20910 | `f0ac7b2e` | 0/4 (min prefix 1) |
| F_flashinfer_sampler | 1077.4 | 61876.0 | 66220.1 | 185.6 | 19673 | `21b15dc2` | 0/4 (min prefix 1) |

Increments (total, decode):

| Step | Total speedup | Decode speedup | Peak delta |
|---|---:|---:|---|
| A (eager baseline) | 1.00x | 1.00x | — |
| B (single graph) vs A | 1.18x | 1.19x | +6.0% |
| C (bucket ladder) vs B | 1.00x | 1.00x | +0.0% |
| D (paged) vs C | 2.07x | 2.12x | -12.1% |
| E (torch.compile) vs D | 1.07x | 1.07x | +7.8% |
| F (FlashInfer + compiled sampler) vs D | 1.37x | 1.40x | +1.4% |

## Path proof

| model | task | level | graph capture | ladder | captures | decode bucket widths |
|---|---|---|---|---|---|---|
| qwen3_4b | 3k | A_eager | no | None | [None] | [None] |
| qwen3_4b | 3k | B_single_graph | yes | [4] | [0] | [{'4': 3071}] |
| qwen3_4b | 3k | C_multi_graph | yes | [1, 2, 4] | [0] | [{'4': 3071}] |
| qwen3_4b | 3k | D_paged | yes | [1, 2, 4] | [0] | [{'4': 3071}] |
| qwen3_4b | 3k | E_fused | yes | [1, 2, 4] | [0] | [{'4': 3071}] |
| qwen3_4b | 3k | F_flashinfer_sampler | yes | [1, 2, 4] | [0] | [{'4': 3071}] |
| qwen3_8b | 3k | A_eager | no | None | [None] | [None] |
| qwen3_8b | 3k | B_single_graph | yes | [4] | [0] | [{'4': 3071}] |
| qwen3_8b | 3k | C_multi_graph | yes | [1, 2, 4] | [0] | [{'4': 3071}] |
| qwen3_8b | 3k | D_paged | yes | [1, 2, 4] | [0] | [{'4': 3071}] |
| qwen3_8b | 3k | E_fused | yes | [1, 2, 4] | [0] | [{'4': 3071}] |
| qwen3_8b | 3k | F_flashinfer_sampler | yes | [1, 2, 4] | [0] | [{'4': 3071}] |

## Caveats

- **Tokens vs A:** the 6k rows use Gumbel sampling (T=0.7), which amplifies bf16 ULP differences into different sampled tokens: graph capture (B/C), paged attention (D/E/F), and even the batched dense prefill (batch 4 vs the old batch-1 shape) can flip boundary tokens. The 20k greedy rows are the stability check: B/C stay exact vs A; D/E/F can differ on one row (known paged/compile ULP behavior, see `runs/optimization_ablation.md` §Paged decode). At 4B the D/E/F rows share one digest; at 8B even paged-family paths can flip boundary tokens against each other.
- **C vs B = 1.00x is expected here:** every run drains the full continuous width, so the bucket ladder ([1,2,4] / [1,2]) replays at its top bucket — identical work to a single graph captured at the full width. The ladder only pays off when the active batch width varies over the run (see the old bucket-ladder results in `runs/optimization_ablation.md` §CUDA graphs and the bucket ladder).
- **Peaks:** graph rows (B/C) peak above eager because the captured static cache/KV buffers stay allocated; paged rows (D/E/F) run on the block pool and peak lower. E adds the inductor workspace; treat torch.compile as a latency option, not a memory option.
- **Single measured iteration** after one warmup (autotune for E is inside the warmup). Directional, per the source ablation's convention.


## vLLM oracle (same concurrency, ignore-EOS)

| Model | vLLM wall | vLLM tok/s | F (FlashInfer) wall | F vs vLLM | D (Paged) wall | D vs vLLM |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 31.97s | 384 | 50.93s | 1.59x | 75.97s | 2.38x |
| Qwen3-8B | 47.13s | 261 | 66.22s | 1.40x | 90.72s | 1.92x |

Note: vLLM reports no prefill/decode split and no peak memory in this environment (engine resets torch peak stats), so phase-level comparisons are DS-internal only.

## Rerun provenance (ignore-EOS)

The first 3k attempt decoded EOS-aware, so several 8B cells stopped early and did less work than the vLLM oracles (which generate all 4 x 3072 tokens). All 12 cells were therefore rerun with `--ignore-eos` (every row generates exactly 3072 tokens, uniform across all cells and oracles); every cell above ran 3071 decode steps with all rows at exactly 3072 tokens. The superseded EOS-aware attempt is archived locally in `pre_ignore_eos/` (not committed).
_Status: 12/36 cells complete. Raw JSONs in `raw/`, logs in `logs/`. Regenerate with `python scripts/run_cb_ablation.py` (skips finished cells, rewrites this file)._

