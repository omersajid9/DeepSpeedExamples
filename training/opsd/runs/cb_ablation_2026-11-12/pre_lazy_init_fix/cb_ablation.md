# Continuous-batching rollout ablation — optimization-merged branch

Date: 2026-11-12. Node: A100-SXM4-40GB (gpua021). torch 2.15.0+cu132, transformers 5.22.2, deepspeed from `/u/msajid/projects/DeepSpeed` (branch `fork/optimization-merged`), flashinfer 0.7.4, dskernels (blocked-flash JIT). bf16, seed 0, warmup 1, one measured iteration per cell, chunked prefill off everywhere, `--top-k -1`, `--save-response-ids`.

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

- **qwen3_4b 6k**; D (paged, multi-graph) vs A (eager): 1.14x total, 1.43x decode; C (ladder) vs B (single graph): 1.00x total; E (torch.compile) vs D: 1.11x total; F (FlashInfer + compiled sampler) vs D: 0.94x total
- **qwen3_4b 20k**; D (paged, multi-graph) vs A (eager): 2.69x total, 2.22x decode; C (ladder) vs B (single graph): 1.00x total; E (torch.compile) vs D: 1.06x total; F (FlashInfer + compiled sampler) vs D: 0.96x total
- **qwen3_8b 6k**; D (paged, multi-graph) vs A (eager): 1.13x total, 1.58x decode; C (ladder) vs B (single graph): 1.00x total; E (torch.compile) vs D: 1.09x total; F (FlashInfer + compiled sampler) vs D: 0.95x total
- **qwen3_8b 20k**; D (paged, multi-graph) vs A (eager): 2.25x total, 2.19x decode; C (ladder) vs B (single graph): 1.00x total; E (torch.compile) vs D: 1.02x total; F (FlashInfer + compiled sampler) vs D: 0.97x total

## qwen3_4b — 6k (batch 4, C=4, prompt 6144, 16 new, T=0.7, top_p=0.9)

| Level | Prefill ms | Decode ms | Total ms | Tok/s | Peak MiB | Response SHA | Tokens vs A |
|---|---:|---:|---:|---:|---:|---|---|
| A_eager | 1371.8 | 1002.0 | 2410.4 | 26.6 | 14648 | `d685ee30` | — |
| B_single_graph | 1374.2 | 829.4 | 2240.6 | 28.6 | 16510 | `a4c6c840` | 2/4 (min prefix 8) |
| C_multi_graph | 1371.2 | 829.7 | 2237.8 | 28.6 | 16510 | `a4c6c840` | 2/4 (min prefix 8) |
| D_paged | 1382.5 | 700.0 | 2116.2 | 30.2 | 11686 | `291233a0` | 1/4 (min prefix 0) |
| E_fused | 1234.6 | 646.1 | 1915.0 | 33.4 | 14849 | `291233a0` | 1/4 (min prefix 0) |
| F_flashinfer_sampler | 1457.5 | 758.3 | 2250.4 | 28.4 | 11961 | `291233a0` | 1/4 (min prefix 0) |

Increments (total, decode):

| Step | Total speedup | Decode speedup | Peak delta |
|---|---:|---:|---|
| A (eager baseline) | 1.00x | 1.00x | — |
| B (single graph) vs A | 1.08x | 1.21x | +12.7% |
| C (bucket ladder) vs B | 1.00x | 1.00x | +0.0% |
| D (paged) vs C | 1.06x | 1.19x | -29.2% |
| E (torch.compile) vs D | 1.11x | 1.08x | +27.1% |
| F (FlashInfer + compiled sampler) vs D | 0.94x | 0.92x | +2.4% |

## qwen3_4b — 20k (batch 2, C=2, prompt 20480, 8 new, T=0.0, top_p=1.0)

| Level | Prefill ms | Decode ms | Total ms | Tok/s | Peak MiB | Response SHA | Tokens vs A |
|---|---:|---:|---:|---:|---:|---|---|
| A_eager | 8063.3 | 733.7 | 8821.7 | 1.8 | 19250 | `86afbeb7` | — |
| B_single_graph | 8073.3 | 651.4 | 8757.1 | 1.8 | 23551 | `86afbeb7` | 2/2 |
| C_multi_graph | 8069.0 | 652.2 | 8746.2 | 1.8 | 23551 | `86afbeb7` | 2/2 |
| D_paged | 2931.2 | 330.3 | 3282.0 | 4.9 | 15057 | `89c5280e` | 1/2 (min prefix 1) |
| E_fused | 2783.9 | 301.8 | 3106.7 | 5.2 | 26082 | `89c5280e` | 1/2 (min prefix 1) |
| F_flashinfer_sampler | 3051.8 | 346.0 | 3418.0 | 4.7 | 15329 | `89c5280e` | 1/2 (min prefix 1) |

Increments (total, decode):

| Step | Total speedup | Decode speedup | Peak delta |
|---|---:|---:|---|
| A (eager baseline) | 1.00x | 1.00x | — |
| B (single graph) vs A | 1.01x | 1.13x | +22.3% |
| C (bucket ladder) vs B | 1.00x | 1.00x | +0.0% |
| D (paged) vs C | 2.66x | 1.97x | -36.1% |
| E (torch.compile) vs D | 1.06x | 1.09x | +73.2% |
| F (FlashInfer + compiled sampler) vs D | 0.96x | 0.95x | +1.8% |

## qwen3_8b — 6k (batch 4, C=4, prompt 6144, 16 new, T=0.7, top_p=0.9)

| Level | Prefill ms | Decode ms | Total ms | Tok/s | Peak MiB | Response SHA | Tokens vs A |
|---|---:|---:|---:|---:|---:|---|---|
| A_eager | 2090.4 | 1076.6 | 3204.2 | 20.0 | 22547 | `a4f8cfbf` | — |
| B_single_graph | 2091.3 | 902.5 | 3031.2 | 21.1 | 25056 | `bf90e48f` | 2/4 (min prefix 1) |
| C_multi_graph | 2092.2 | 903.2 | 3032.6 | 21.1 | 25056 | `bf90e48f` | 2/4 (min prefix 1) |
| D_paged | 2124.4 | 681.7 | 2840.2 | 22.5 | 19747 | `a2aaa448` | 1/4 (min prefix 0) |
| E_fused | 1934.8 | 644.3 | 2613.5 | 24.5 | 22784 | `a62af59e` | 1/4 (min prefix 0) |
| F_flashinfer_sampler | 2202.1 | 752.4 | 2989.2 | 21.4 | 20023 | `ff131825` | 1/4 (min prefix 0) |

Increments (total, decode):

| Step | Total speedup | Decode speedup | Peak delta |
|---|---:|---:|---|
| A (eager baseline) | 1.00x | 1.00x | — |
| B (single graph) vs A | 1.06x | 1.19x | +11.1% |
| C (bucket ladder) vs B | 1.00x | 1.00x | +0.0% |
| D (paged) vs C | 1.07x | 1.32x | -21.2% |
| E (torch.compile) vs D | 1.09x | 1.06x | +15.4% |
| F (FlashInfer + compiled sampler) vs D | 0.95x | 0.91x | +1.4% |

## qwen3_8b — 20k (batch 2, C=2, prompt 20480, 8 new, T=0.0, top_p=1.0)

| Level | Prefill ms | Decode ms | Total ms | Tok/s | Peak MiB | Response SHA | Tokens vs A |
|---|---:|---:|---:|---:|---:|---|---|
| A_eager | 9272.9 | 769.8 | 10067.8 | 1.6 | 27148 | `9c437ea5` | — |
| B_single_graph | 9280.9 | 687.3 | 9993.6 | 1.6 | 32130 | `9c437ea5` | 2/2 |
| C_multi_graph | 9278.4 | 688.1 | 9992.5 | 1.6 | 32130 | `9c437ea5` | 2/2 |
| D_paged | 4098.6 | 350.8 | 4469.8 | 3.6 | 23495 | `91f6745e` | 1/2 (min prefix 2) |
| E_fused | 3881.4 | 458.8 | 4363.2 | 3.7 | 34869 | `cbd56072` | 0/2 (min prefix 0) |
| F_flashinfer_sampler | 4222.4 | 346.5 | 4589.1 | 3.5 | 23767 | `91f6745e` | 1/2 (min prefix 2) |

Increments (total, decode):

| Step | Total speedup | Decode speedup | Peak delta |
|---|---:|---:|---|
| A (eager baseline) | 1.00x | 1.00x | — |
| B (single graph) vs A | 1.01x | 1.12x | +18.3% |
| C (bucket ladder) vs B | 1.00x | 1.00x | +0.0% |
| D (paged) vs C | 2.24x | 1.96x | -26.9% |
| E (torch.compile) vs D | 1.02x | 0.76x | +48.4% |
| F (FlashInfer + compiled sampler) vs D | 0.97x | 1.01x | +1.2% |

## Path proof

| model | task | level | graph capture | ladder | captures | decode bucket widths |
|---|---|---|---|---|---|---|
| qwen3_4b | 20k | A_eager | no | None | [None] | [None] |
| qwen3_4b | 20k | B_single_graph | yes | [2] | [0] | [{'2': 7}] |
| qwen3_4b | 20k | C_multi_graph | yes | [1, 2] | [0] | [{'2': 7}] |
| qwen3_4b | 20k | D_paged | yes | [1, 2] | [0] | [{'2': 7}] |
| qwen3_4b | 20k | E_fused | yes | [1, 2] | [0] | [{'2': 7}] |
| qwen3_4b | 20k | F_flashinfer_sampler | yes | [1, 2] | [0] | [{'2': 7}] |
| qwen3_4b | 6k | A_eager | no | None | [None] | [None] |
| qwen3_4b | 6k | B_single_graph | yes | [4] | [0] | [{'4': 15}] |
| qwen3_4b | 6k | C_multi_graph | yes | [1, 2, 4] | [0] | [{'4': 15}] |
| qwen3_4b | 6k | D_paged | yes | [1, 2, 4] | [0] | [{'4': 15}] |
| qwen3_4b | 6k | E_fused | yes | [1, 2, 4] | [0] | [{'4': 15}] |
| qwen3_4b | 6k | F_flashinfer_sampler | yes | [1, 2, 4] | [0] | [{'4': 15}] |
| qwen3_8b | 20k | A_eager | no | None | [None] | [None] |
| qwen3_8b | 20k | B_single_graph | yes | [2] | [0] | [{'2': 7}] |
| qwen3_8b | 20k | C_multi_graph | yes | [1, 2] | [0] | [{'2': 7}] |
| qwen3_8b | 20k | D_paged | yes | [1, 2] | [0] | [{'2': 7}] |
| qwen3_8b | 20k | E_fused | yes | [1, 2] | [0] | [{'2': 7}] |
| qwen3_8b | 20k | F_flashinfer_sampler | yes | [1, 2] | [0] | [{'2': 7}] |
| qwen3_8b | 6k | A_eager | no | None | [None] | [None] |
| qwen3_8b | 6k | B_single_graph | yes | [4] | [0] | [{'4': 15}] |
| qwen3_8b | 6k | C_multi_graph | yes | [1, 2, 4] | [0] | [{'4': 15}] |
| qwen3_8b | 6k | D_paged | yes | [1, 2, 4] | [0] | [{'4': 15}] |
| qwen3_8b | 6k | E_fused | yes | [1, 2, 4] | [0] | [{'4': 15}] |
| qwen3_8b | 6k | F_flashinfer_sampler | yes | [1, 2, 4] | [0] | [{'4': 15}] |

## Caveats

- **Tokens vs A:** the 6k rows use Gumbel sampling (T=0.7), which amplifies bf16 ULP differences into different sampled tokens: graph capture (B/C), paged attention (D/E/F), and even the batched dense prefill (batch 4 vs the old batch-1 shape) can flip boundary tokens. The 20k greedy rows are the stability check: B/C stay exact vs A; D/E/F can differ on one row (known paged/compile ULP behavior, see `runs/optimization_ablation.md` §Paged decode). At 4B the D/E/F rows share one digest; at 8B even paged-family paths can flip boundary tokens against each other.
- **C vs B = 1.00x is expected here:** every run drains the full continuous width, so the bucket ladder ([1,2,4] / [1,2]) replays at its top bucket — identical work to a single graph captured at the full width. The ladder only pays off when the active batch width varies over the run (see the old bucket-ladder results in `runs/optimization_ablation.md` §CUDA graphs and the bucket ladder).
- **Peaks:** graph rows (B/C) peak above eager because the captured static cache/KV buffers stay allocated; paged rows (D/E/F) run on the block pool and peak lower. E adds the inductor workspace; treat torch.compile as a latency option, not a memory option.
- **Single measured iteration** after one warmup (autotune for E is inside the warmup). Directional, per the source ablation's convention.

_Status: 24/24 cells complete. Raw JSONs in `raw/`, logs in `logs/`. Regenerate with `python scripts/run_cb_ablation.py` (skips finished cells, rewrites this file)._

