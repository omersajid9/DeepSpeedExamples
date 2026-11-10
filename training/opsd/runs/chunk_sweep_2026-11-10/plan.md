# Chunked-prefill sweep — Qwen3-4B, 5k/10k prompts

Purpose: isolate what chunked-prefill actually buys, and where it stops paying.
Previous runs showed the bundle (decode graphs + chunked prefill) loses at 20k with
32 new tokens, and won at ReTool because chunking was a no-op. This sweep fixes the
model and the prompt length, then sweeps only the chunk size.

## Matrix
- Model: `Qwen/Qwen3-4B` (`/dev/shm/hf_cache/models/Qwen3-4B`), bf16, SDPA.
- Prompt lengths: 5120, 10240 (`data/longbench_20k.jsonl` truncated by `--max-prompt-length`).
- Batch 16, continuous batch size 3, max new tokens 32, greedy.
- Warmup 1, measured iterations 3, medians reported.
- Chunk sizes: 256, 512, 1024, 2048, 4096, 8192.
- Arms per prompt length:
  - `single_graph`: continuous batching + single graph from dc046b15, no chunked prefill (the requested baseline).
  - `eager`: continuous eager, no CUDA graph, no chunked prefill (reference).
  - `graph_only`: continuous batching + the new bucket graph, no chunked prefill.
  - `chunk_X`: continuous batching + the new bucket graph + chunked prefill at X tokens.

`graph_only` vs `single_graph` is the bucket-graph effect; `chunk_X` vs
`single_graph` is the full optimized bundle; `chunk_X` vs `graph_only` is the
chunking cost/benefit at fixed graph state.

## Metrics
Per run: wall, prefill forward, decode forward, chunks executed, staging bytes,
peak memory, graph captures, response hash. Summary in `summary.md`/`summary.csv`
with deltas vs baseline and vs graph_only.

## What to look for
- Whether a larger chunk (2048/4096/8192) recovers the baseline at 10k.
- Whether any chunk size beats `graph_only` at 5k/10k with 32 new tokens.
- The staging cost (`chunk_count * 2.8 GB` at 20k; proportionally less at 5k/10k).
- The peak-memory curve; no run should exceed ~43 GiB.

## Code-level mechanism (why the curve should slope up)
`_run_chunked_prefill` spends a scalar token budget per scheduler step:
`budget = prefill_chunk_size`, consumed in admission order. A step advances the
first staged prompt by up to `chunk_size` tokens; only if that prompt has less left
than the budget does the next staged prompt get the remainder. The chunked path
therefore serializes staged prompts instead of running one parallel prefill over
the active batch. The graph bucket width is the *number of staged prompts*, not the
chunk size; the chunk width is fixed at `prefill_chunk_size`.

Consequences the sweep should expose:
- Chunking cannot beat a single full prefill on standalone prefill throughput; it
  can only recover time by overlapping with decode rows that are already running.
  With 32 new tokens the overlap window is tiny, so the curve should approach
  `graph_only` only when the chunk divides the prompt; otherwise the last-window
  padding can make a larger chunk worse (4096 at 5k, 8192 at 10k), and small
  chunks lose badly.
- Each chunk is a forward + staging-cache copy. Fewer forwards help, but the last
  window slides back to end at the prompt, so a chunk that does not divide the
  prompt length computes a full chunk width for a partial remainder. The curve is
  not monotonic: 4096 is worse than 2048 at 5k; 8192 is worse than 4096 at 10k.
  The best chunk is the largest exact divisor of the prompt length, not the largest
  chunk. The 5k 8192 arm is not chunked (5120 <= 8192); it is graph-only with the
  staging cache allocated.
- The staging cache is sized to the prompt, not the chunk; `staging_bytes` stays
  roughly constant across chunk sizes, while chunk count scales as
  `ceil(prompt_len / chunk_size)`.

## Caveats
- Graph and batch-width numerics can change token IDs after ~10-30 tokens; this sweep
  compares performance, not bitwise outputs.
- 32 new tokens makes these prefill-bound workloads. A decode-heavy workload can show
  the opposite ranking; do not generalize the chunk-size winner across workloads.
