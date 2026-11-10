# Four-way rollout: graphs + chunked prefill + Gumbel sampler + paged attention

Qwen3-4B, A40, bf16, LongBench prompt truncated to 6144 tokens. Batch 1, continuous batch 1,
16 new tokens, chunked prefill with chunk 512, CUDA graph capture, Gumbel sampler
(`temperature=0.7`, `top_p=0.9`, seed 0), warmup 1, iteration 1.

The dense row is the same four-way stack with the static dense KV cache; only the paged
attention/cache differs. Both rows captured two graphs (prefill and decode) during warmup and
replayed them during the measured iteration.

| Cache | Prefill ms | Decode ms/token | Total ms | Tokens/s | Peak MiB | KV cache | Prefill chunks |
|---|---:|---:|---:|---:|---:|---|---:|
| Dense static | 1379.3 | 39.73 | 2011.7 | 7.95 | 9503.6 | static | 12 |
| Paged | 1381.5 | 25.86 | 1820.4 | 8.79 | 9511.6 | 97 x 64-token blocks | 12 |

- Decode speedup: **1.54x** (39.73 -> 25.86 ms/token).
- End-to-end speedup: **1.10x** (2011.7 -> 1820.4 ms); chunked prefill dominates the 6k run.
- The paged path uses 97/97 blocks for the single row (`block_utilization=1.0`), block size 64.
- The 25.86 ms/token decode is essentially the plan's 25 ms phase-4 target (within 3.4%).

## Accuracy verification

`scripts/verify_rollout_accuracy.py` compares the two result files:

```text
tokens identical; logprob max abs diff=0.00152457 over 2 recorded positions
PASS
```

Both paths produce the same 16 response tokens and the same response SHA-256
(`d4a324e903ed2fe8e2d5b676b43ec94a22efbbdd5663c30a310d93d0144cf02c`). The current graph path
records logprobs only for the final two response steps (a pre-existing logging gap, identical
in both rows), so the token-exact match is the primary accuracy check; the recorded logprob
drift is 1.5e-3.

## Commands

```bash
# Dense four-way reference
python benchmarks/bench_continuous_graph.py --model Qwen/Qwen3-4B \
  --data data/longbench_20k.jsonl --batch-size 1 --continuous-batch-size 1 \
  --max-prompt-length 6144 --max-new-tokens 16 --warmup 1 --iterations 1 \
  --enable-chunked-prefill --prefill-chunk-size 512 \
  --temperature 0.7 --top-p 0.9 --seed 0 \
  --save-response-ids --save-response-logprobs --output runs/fourway_dense_6k.json

# Paged four-way
python benchmarks/bench_continuous_graph.py --model Qwen/Qwen3-4B \
  --data data/longbench_20k.jsonl --batch-size 1 --continuous-batch-size 1 \
  --max-prompt-length 6144 --max-new-tokens 16 --warmup 1 --iterations 1 \
  --enable-chunked-prefill --prefill-chunk-size 512 \
  --temperature 0.7 --top-p 0.9 --seed 0 --use-paged-cache \
  --save-response-ids --save-response-logprobs --output runs/fourway_paged_6k.json

python scripts/verify_rollout_accuracy.py \
  --reference runs/fourway_dense_6k.json --candidate runs/fourway_paged_6k.json
```

Artifacts: `runs/fourway_dense_6k.json`, `runs/fourway_paged_6k.json`.
