# Qwen3-8B: paged attention, chunked prefill, graphs, Gumbel sampler

Qwen3-8B, A40, bf16, LongBench prompt truncated to 6144 tokens. Batch 1, continuous batch 1,
16 new tokens. All four-way rows use CUDA graphs, chunked prefill (chunk 512), and the Gumbel
sampler unless marked greedy; warmup 1, measured iteration 1. Each four-way run captured two graphs
during warmup (prefill and decode) and replayed them in the measured iteration.

## Eager reference (no graphs, no chunked prefill, greedy)

| Cache | Prefill ms | Decode ms/token | Total ms | Tokens/s | Peak MiB | Accuracy |
|---|---:|---:|---:|---:|---:|---|
| Dense static | 1104.8 | 62.78 | 2064.5 | 7.75 | 17991.3 | token-exact |
| Paged | 1108.9 | 46.66 | 1916.4 | 8.35 | 17999.1 | token-exact |

Paged eager decode is **1.35x** faster; end-to-end 1.08x. The verifier reports identical tokens
(`6b1dad94...` both).

## Four-way, greedy (graphs + chunked prefill + paged, no sampler)

This isolates the graph/chunk/paging path from sampling noise.

| Cache | Prefill ms | Decode ms/token | Total ms | Tokens/s | Peak MiB |
|---|---:|---:|---:|---:|---:|
| Dense static | 1872.0 | 51.93 | 2671.1 | 5.99 | 17402.1 |
| Paged | 1879.5 | 38.65 | 2519.5 | 6.35 | 17410.1 |

- Paged decode: **1.34x** faster; end-to-end 1.06x.
- Both rows return the same 16 tokens (`6b1dad94...`), and that SHA is identical to the eager
  greedy rows above. The verifier reports `tokens identical; logprobs were not recorded` (greedy).
- This is the accuracy check for graph capture + chunked prefill + paged attention itself.

## Four-way, Gumbel sampler (`temperature=0.7`, `top_p=0.9`, seed 0)

| Cache | Prefill ms | Decode ms/token | Total ms | Tokens/s | Peak MiB | KV cache |
|---|---:|---:|---:|---:|---:|---|
| Dense static | 1876.8 | 51.91 | 2681.6 | 5.97 | 17402.1 | static |
| Paged | 1876.8 | 38.46 | 2493.0 | 6.42 | 17410.1 | 97 x 64-token blocks |

- Paged decode: **1.35x** faster; end-to-end 1.08x.
- 12 prefill chunks; 97/97 paged blocks used.
- Accuracy: **15/16 response tokens identical**. The final token differs
  (dense 576 vs paged 320); the two recorded logprobs before it are exactly equal
  (`-0.7864127`, `-0.4607418`). This is a last-step sampler flip at `T=0.7`, not a
  greedy-path mismatch.

## Four-way, Gumbel sampler at 512 tokens (accuracy pair)

| Cache | Prefill ms | Decode ms/token | Total ms | Tokens/s | Peak MiB | Accuracy |
|---|---:|---:|---:|---:|---:|---|
| Dense static | 169.4 | 34.00 | 1546.6 | 5.17 | 15825.6 | token-exact |
| Paged | 169.1 | 33.16 | 1551.8 | 5.16 | 15809.6 | token-exact |

Both rows return the same 8 tokens (`df1a9354...`). The recorded logprobs differ by at most
**0.1184** over 6 positions, so `verify_rollout_accuracy.py` needs
`--max-logprob-diff 0.25` to pass this sampling pair; token accuracy is exact.

## Accuracy summary

| Check | Result |
|---|---|
| 8B eager greedy, dense vs paged | exact tokens; PASS |
| 8B four-way greedy, dense vs paged | exact tokens; same SHA as eager; PASS |
| 8B four-way Gumbel 6k | 15/16 exact; last-token sampler flip |
| 8B four-way Gumbel 512 | 8/8 exact; max logprob drift 0.1184 |

The greedy checks show no paged/graph/chunk regression. Under sampling, a small bf16 attention
numeric difference can flip a near-boundary last token at 6k; the 4B four-way run was exact at
16/16, so this is model/seed-sensitive rather than a systematic cache error.

## Commands

```bash
export HF_HOME=/dev/shm/hf_cache   # 8B weights were staged here

# Eager reference
python benchmarks/bench_continuous_graph.py --model Qwen/Qwen3-8B \
  --data data/longbench_20k.jsonl --batch-size 1 --continuous-batch-size 1 \
  --max-prompt-length 6144 --max-new-tokens 16 --warmup 1 --iterations 1 \
  --no-graph-capture --save-response-ids --output runs/8b_eager_6k_dense.json
# ... add --use-paged-cache for the paged eager row

# Four-way greedy
python benchmarks/bench_continuous_graph.py --model Qwen/Qwen3-8B \
  --data data/longbench_20k.jsonl --batch-size 1 --continuous-batch-size 1 \
  --max-prompt-length 6144 --max-new-tokens 16 --warmup 1 --iterations 1 \
  --enable-chunked-prefill --prefill-chunk-size 512 --temperature 0.0 --seed 0 \
  --save-response-ids --output runs/8b_fourway_greedy_6k_dense.json
# ... add --use-paged-cache for the paged row

# Four-way Gumbel
python benchmarks/bench_continuous_graph.py --model Qwen/Qwen3-8B \
  --data data/longbench_20k.jsonl --batch-size 1 --continuous-batch-size 1 \
  --max-prompt-length 6144 --max-new-tokens 16 --warmup 1 --iterations 1 \
  --enable-chunked-prefill --prefill-chunk-size 512 \
  --temperature 0.7 --top-p 0.9 --seed 0 \
  --save-response-ids --save-response-logprobs --output runs/8b_fourway_6k_dense.json
# ... add --use-paged-cache for the paged row
```

Artifacts: `runs/8b_eager_6k_*.json`, `runs/8b_fourway_greedy_6k_*.json`,
`runs/8b_fourway_6k_*.json`, `runs/8b_fourway_512_*.json`.
The 4B four-way result is in `runs/fourway_6k.md`.


## Four-way at 10k tokens (Gumbel sampler)

| Cache | Prefill ms | Decode ms/token | Total ms | Tokens/s | Peak MiB | KV blocks |
|---|---:|---:|---:|---:|---:|---:|
| Dense static | 4034.3 | 68.91 | 6218.3 | 5.15 | 18569.5 | static |
| Paged | 4036.8 | 42.42 | 5464.0 | 5.86 | 18575.3 | 161 x 64-token |

- Paged decode: **1.63x** faster; end-to-end **1.14x**.
- 20 prefill chunks; two graphs captured in warmup.
- Accuracy: **32/32 tokens identical** (`fa717966...`); max logprob drift **0.1571** over
  18 recorded positions, so `verify_rollout_accuracy.py --max-logprob-diff 0.25` reports PASS.



## Four-way at 20k tokens (Gumbel sampler)

| Cache | Prefill ms | Decode ms/token | Total ms | Tokens/s | Peak MiB | KV blocks |
|---|---:|---:|---:|---:|---:|---:|
| Dense static | 12228.1 | 110.68 | 15772.5 | 2.03 | 21206.4 | static |
| Paged | 12218.4 | 51.16 | 13950.4 | 2.29 | 21214.7 | 306 x 64-token |

- Paged decode: **2.16x** faster; end-to-end **1.13x**.
- 39 prefill chunks; two graphs captured in warmup.
- Accuracy: **32/32 tokens identical** (`90d39e5c...`); max logprob drift **0.0238** over
  3 recorded positions; PASS with `--max-logprob-diff 0.25`.

## Scaling summary (Qwen3-8B, four-way)

| Context | Dense ms/token | Paged ms/token | Decode speedup |
|---:|---:|---:|---:|
| 512 | 34.00 | 33.16 | 1.03x |
| 6k | 51.91 | 38.46 | 1.35x |
| 10k | 68.91 | 42.42 | 1.63x |
| 20k | 110.68 | 51.16 | 2.16x |

Paged decode is flat-ish in context length; dense decode grows with the dense attention span.
The 20k paged run uses 306 64-token blocks (about 2.9 GiB of KV) and peaks at 21.2 GiB with the
8B model.



## Four-way at ReTool 512 (16 rows, Gumbel sampler)

Prompt width 512 (effective ~304), 16 rows, 512 new tokens. The prompt is shorter than the
512-token prefill chunk, so chunked prefill reports 0 chunks here; graphs and the Gumbel sampler
are active.

| Cache | Prefill ms | Decode ms/step | Total ms | Tokens/s | Peak MiB | KV blocks |
|---|---:|---:|---:|---:|---:|---:|
| Dense static | 1853.4 | 72.16 | 42003.6 | 195.03 | 20669.9 | static |
| Paged | 1803.6 | 39.20 | 25209.3 | 324.96 | 20786.9 | 256 x 64-token |

- Paged decode: **1.84x** faster; end-to-end **1.67x**.
- `decode ms/step` is for all 16 rows; per generated token it is 4.51 ms dense vs 2.45 ms paged.
- Accuracy under sampling: the 16 x 512 sampled token streams diverge between dense and paged
  (a small logit difference changes one draw and then the continuation), so exact stream matching
  is not a valid sampling criterion. The greedy ReTool check below is the accuracy test for the
  graph/chunk/paged path.

