# Paged attention decode - Qwen3-4B, 6k context, A40

Qwen3-4B, bf16, SDPA dense reference, LongBench prompt truncated to 6144 tokens.
Batch 1, continuous batch 1, 16 new tokens, greedy, no graph capture, warmup 1.
Paged blocks are 64 tokens; 97/97 blocks used for the row (`block_utilization=1.0`).

| Path | Prefill ms | Decode ms/token | Total ms | Wall ms | Tokens/s | Peak MiB | KV cache MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense static cache | 714.5 | 51.59 | 1513.8 | 1514.1 | 10.57 | 9931.1 | 908.4 |
| Paged decode | 713.5 | 45.70 | 1507.2 | 1507.5 | 10.45 | 9938.7 | 915.5 |

Both paths return the same response SHA-256:
`d4a324e903ed2fe8e2d5b676b43ec94a22efbbdd5663c30a310d93d0144cf02c`.

## What changed

- Decode attention reads through a per-row block table with `BlockedFlashAttn`; no dense decode mask.
- Atoms and block-table checks are built once per decode step and reused by all 36 layers.
- Prefill still uses the dense `DeepSpeedStaticCache` and copies the valid prompt tokens into blocks once.
  That copy is why the profile's `generation_overhead_ms` is 108.1 ms versus 25.2 ms for the dense path.
- The paged prefill/mixed path (phase 3), graph capture (phase 4), and prefix reuse (phase 5) are not
  implemented.

## Kernel-level check

One layer, 6k KV, one decode query, head_dim 128, 8 KV heads, 32 query heads:

| Attention | ms/layer |
|---|---:|
| `BlockedFlashAttn` | 0.167 |
| Dense SDPA with decode mask | 0.346 |

Over 36 layers that is ~6.0 ms versus ~12.5 ms; the observed decode saving is 5.89 ms/token
(51.59 → 45.70). The model forward floor is larger than the attention term, so the plan's 25 ms target is
not reached without phase 4 graph/CPU-overhead work.

## Commands

```bash
# Dense reference
python benchmarks/bench_continuous_graph.py --model Qwen/Qwen3-4B \
  --data data/longbench_20k.jsonl --batch-size 1 --continuous-batch-size 1 \
  --max-prompt-length 6144 --max-new-tokens 16 --warmup 1 --iterations 1 \
  --no-graph-capture --save-response-ids --output runs/paged_6k_dense.json

# Paged decode
python benchmarks/bench_continuous_graph.py --model Qwen/Qwen3-4B \
  --data data/longbench_20k.jsonl --batch-size 1 --continuous-batch-size 1 \
  --max-prompt-length 6144 --max-new-tokens 16 --warmup 1 --iterations 1 \
  --no-graph-capture --use-paged-cache --save-response-ids \
  --output runs/paged_6k_paged_opt.json
```

Artifacts: `runs/paged_6k_dense.json`, `runs/paged_6k_paged_opt.json`.
Unit check: `python -m pytest tests/unit/runtime/rollout/test_paged_cache.py`.
The blocked kernel requires `deepspeed-kernels` (`dskernels`); `RaggedOpsBuilder` builds against it.
