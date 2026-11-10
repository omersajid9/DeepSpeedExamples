# 20k Prompt Rollout: Chunked Prefill vs Naive, Old vs New Graph Capture — Qwen3-4B on A40

Data: `data/longbench_20k.jsonl` (prompt length 20480, max-new-tokens 32).
Engine: continuous batching + CUDA graphs (`bench_continuous_graph.py`).
"Old" = commit `5145b7cf` (single captured graph). "New" = commit `c9c92c3d`
(bucket graph capture: a ladder of batch widths for decode and chunked prefill).

## Results

| Config | Active slots | Code | Wall/iter | Prefill | Decode | Prefill fwd | Decode fwd | Peak mem | Tok/s |
|---|---|---|---|---|---|---|---|---|---|
| bs16 naive | 2 | old | 134.2 s | 96.4 s | 36.5 s | 8 | 248 | 23.6 GB | 3.81 |
| bs16 naive | 2 | new | 134.2 s | 96.7 s | 36.5 s | 8 | 248 | 23.6 GB | 3.81 |
| bs16 chunked | 4 | old | 316.0 s | 173.4 s | 141.1 s | 638 | 496 | 22.6 GB | 1.62 |
| bs16 chunked | 4 | new | 197.0 s | 156.8 s | 35.3 s | 160 | 124 | 32.2 GB | 2.60 |
| bs16 chunked | 8 | old | 452.5 s | 173.5 s | 276.1 s | 638 | 496 | 34.1 GB | 1.13 |
| bs16 chunked | 8 | new | OOM | — | — | — | — | — | — |

## With vs without the bucket-graph optimization (equal cont=4)

| Metric | Old (single graph) | New (bucket ladder [1,2,4]) | Delta |
|---|---|---|---|
| Wall/iter | 316.0 s | 197.0 s | **−37.7%** |
| Decode total | 141.1 s | 35.3 s | **−75.0%** |
| Decode forwards | 496 | 124 | **−75.0%** |
| Prefill forwards | 638 | 160 | **−74.9%** |
| Prefill total | 173.4 s | 156.8 s | −9.6% |
| Peak memory | 22.6 GB | 32.2 GB | **+42.5%** |
| Tok/s | 1.62 | 2.60 | **+60.5%** |

The win comes from decode batching. The old single-graph path ran 496 decode
forwards for 16 rows — one row per forward — so each 20k-token decode step paid
the full cache cost for a single row. The bucket ladder dispatches to width 4,
so 124 forwards carry all 16 rows. Decode time drops 4×, from 141 s to 35 s.
The cost is memory: capturing graphs at widths 1, 2, and 4 plus the wider staging
cache raises peak memory from 22.6 GB to 32.2 GB. At cont=8 the extra graphs push
the card past 48 GB and the run OOMs.

## Why "naive is 3.2× faster than optimized" is a configuration artifact

The 3.2× figure compares **naive at 2 active slots** (3.81 tok/s) with **chunked
at 8 active slots** (1.13 tok/s). Those are not the same configuration. Naive
holds 2 concurrent 20k sequences; chunked holds 8. Eight concurrent 20k KV caches
saturate memory bandwidth, so per-forward decode is far more expensive. The
apples-to-apples comparison at equal active slots is cont=4:

- new chunked (bucket graphs): 2.60 tok/s
- old chunked (single graph): 1.62 tok/s

So the optimization is **+60%**, not a regression. Naive still wins in raw tok/s
(3.81) but only because it runs 2 slots; it cannot exceed 2 concurrent 20k
sequences without OOM. Chunked is the only path that reaches 4 or 8 concurrent
20k sequences on 48 GB, and bucket capture makes 4 slots ~1.6× faster than the
old single-graph code.

## Commands

```bash
# new bucket-graph code (c9c92c3d)
python benchmarks/bench_continuous_graph.py --model Qwen/Qwen3-4B \
  --data data/longbench_20k.jsonl --batch-size 16 --continuous-batch-size 4 \
  --max-prompt-length 20480 --max-new-tokens 32 --warmup 0 --iterations 2 \
  --enable-chunked-prefill --prefill-chunk-size 512 \
  --output runs/new_bs16_chunked_cont4.json

# old single-graph code (5145b7cf)
git checkout 5145b7cf   # in /workspace/DeepSpeed
python benchmarks/bench_continuous_graph.py ... --output runs/old_bs16_chunked_cont4.json
```

Artifacts: `runs/new_bs16_naive_cont2.json`, `runs/new_bs16_chunked_cont4.json`,
`runs/old_bs16_chunked_cont4.json`, `runs/qwen3_4b_20k_bs16_naive_cont2.json`,
`runs/qwen3_4b_20k_bs16_chunked_prefill.json`.
