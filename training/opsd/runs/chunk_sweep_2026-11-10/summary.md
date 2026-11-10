# Chunked-prefill sweep — Qwen3-4B, 5k/10k prompts

Model: Qwen3-4B. batch 16, continuous 3, max_new_tokens 32. Baseline = eager, no graph, no chunk. graph_only = decode graphs, no chunk. chunk_X = decode graphs + chunked prefill at X tokens.

| Prompt | Arm | Chunk | Wall (s) | Prefill (s) | Decode (s) | Chunks | Staging (GB) | Peak (MiB) | Wall Δ vs baseline | Prefill Δ vs graph_only |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5120 | baseline | 0 | 22.52 | 9.27 | 13.02 | 0 | 0.00 | 13243 | 0.0 | 0.1 |
| 5120 | graph_only | 0 | 20.28 | 9.26 | 10.83 | 0 | 0.00 | 13269 | -10.0 | 0.0 |
| 5120 | chunk | 256 | 28.83 | 17.72 | 10.82 | 320 | 2.26 | 12349 | 28.0 | 91.4 |
| 5120 | chunk | 512 | 27.33 | 16.21 | 10.82 | 160 | 2.26 | 12383 | 21.3 | 75.0 |
| 5120 | chunk | 1024 | 25.62 | 14.54 | 10.82 | 80 | 2.26 | 12443 | 13.7 | 56.9 |
| 5120 | chunk | 2048 | 27.92 | 16.85 | 10.83 | 48 | 2.26 | 12603 | 24.0 | 81.9 |
| 5120 | chunk | 4096 | 33.32 | 22.22 | 10.83 | 32 | 2.26 | 13128 | 47.9 | 139.9 |
| 5120 | chunk | 8192 | 20.33 | 9.29 | 10.83 | 0 | 2.26 | 15429 | -9.7 | 0.3 |
| 10240 | baseline | 0 | 43.49 | 21.30 | 21.89 | 0 | 0.00 | 18722 | 0.0 | -0.0 |
| 10240 | graph_only | 0 | 41.48 | 21.30 | 19.89 | 0 | 0.00 | 18748 | -4.6 | 0.0 |
| 10240 | chunk | 256 | 75.04 | 54.72 | 19.86 | 640 | 4.53 | 16918 | 72.6 | 156.9 |
| 10240 | chunk | 512 | 68.51 | 48.22 | 19.87 | 320 | 4.53 | 16958 | 57.5 | 126.3 |
| 10240 | chunk | 1024 | 62.51 | 42.27 | 19.87 | 160 | 4.53 | 17035 | 43.7 | 98.4 |
| 10240 | chunk | 2048 | 60.28 | 40.04 | 19.87 | 80 | 4.53 | 17188 | 38.6 | 88.0 |
| 10240 | chunk | 4096 | 67.46 | 47.19 | 19.88 | 48 | 4.53 | 17569 | 55.1 | 121.5 |
| 10240 | chunk | 8192 | 82.09 | 61.84 | 19.88 | 32 | 4.53 | 18740 | 88.8 | 190.3 |


## Findings

1. **Graph-only is the only clear win.** 5k: 20.28 s vs 22.52 s baseline (-10.0%); 10k: 41.48 s vs 43.49 s (-4.6%). Prefill is unchanged (9.26/21.30 s); decode drops 13.02->10.83 s and 21.89->19.89 s. The graph is doing the work, not chunking.

2. **Chunked prefill is slower than the eager baseline at every active chunk size tested.** Best chunked: 5k 1024 at 25.62 s (+13.7% vs baseline, +26.3% vs graph_only); 10k 2048 at 60.28 s (+38.6% vs baseline, +45.3% vs graph_only). The chunked path keeps the graph decode (10.82/19.87 s), so the entire loss is prefill.

3. **The best chunk is the largest exact divisor, not the largest chunk.** 5120/1024 = 5 exactly; 10240/2048 = 5 exactly. 4096 does not divide 5120 (2 chunks, the second is a full 4096-token forward for 1024 real tokens); 8192 does not divide 10240 (2 chunks, the second is a full 8192-token forward for 2048 real tokens). That padding is why 5k 4096 (33.32 s) is worse than 5k 2048 (27.92 s) and why 10k 8192 (82.09 s) is worse than 10k 4096 (67.46 s). The curve is non-monotonic in chunk size.

4. **The 5k chunk 8192 row is not chunked.** 5120 <= 8192, so no prompt is staged; `chunked_prefill_chunks=0`. Its 20.33 s matches graph_only 20.28 s; its 15.43 GiB peak vs 13.27 GiB graph_only shows the staging cache and prefill-bucket cost of enabling chunked prefill even when no chunks execute.

5. **Staging is constant; peak grows with chunk size.** Staging: 2.26 GB at 5k, 4.53 GB at 10k, independent of chunk. Peak: 5k 12.35->15.43 GiB and 10k 16.92->18.74 GiB as chunk grows; small chunks are the memory-friendlier choice (10k 256 saves ~1.8 GiB vs graph_only).

6. **Mechanism.** `_run_chunked_prefill` spends a scalar `budget = chunk_size` per scheduler step in admission order, so it advances one staged prompt at a time instead of prefilling the active batch in parallel. Each chunk is a separate forward plus a staging-cache copy, and the last window slides back to end at the prompt, so a partial remainder is computed at the full chunk width.

## When chunked prefill actually helps

- **Not for prefill-bound 5k/10k with 32 new tokens.** Disable chunked prefill and keep decode graphs; that is the fastest arm here.
- **Memory-constrained prompts.** Chunking lowers the peak (0.9 GiB at 5k, 1.8 GiB at 10k with 256) but costs 28-73% wall time. Use it only when the unchunked path does not fit.
- **Long decode.** Chunking can overlap staged prefill with decode tokens, but this sweep does not demonstrate that win: the ReTool speedup was decode graphs with chunking inactive (prompt 512 <= chunk 512). A decode-heavy long-prompt test is the missing experiment.
- **Chunk choice.** Pick the largest chunk that divides the prompt length exactly (or leaves a small remainder), not the largest available chunk. The tested best points are 1024 for 5k and 2048 for 10k; `[INFERENCE]` 2560/5k and 5120/10k would be the minimal two-chunk points, but graph-only still wins because it is one prefill forward.

## Scope
- One model (Qwen3-4B), A40, batch 16, continuous 3, greedy, 32 new tokens, 3 measured iterations (medians).
- Performance comparison only; graph/batch-width numerics can change token IDs after ~10-30 tokens.
- 32 new tokens makes these workloads prefill-bound; do not generalize the chunk-size ranking to decode-heavy or memory-bound regimes.
