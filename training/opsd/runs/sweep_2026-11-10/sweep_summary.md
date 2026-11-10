# OPSD rollout optimization sweep — results

Environment: A40 46,068 MiB; torch 2.14.1+cu126; transformers 5.22.2; DeepSpeed local checkout. Baseline is continuous eager (no graph, no chunk). Optimized is decode CUDA graphs + chunked prefill. Qwen3.5 prefill graphs are run eagerly because hybrid recurrent-state capture is not token-stable; its decode graphs and chunked prefill remain active.

## Findings (partial sweep)

The optimized bundle is **decode-graph + chunked-prefill**, but the two parts
help opposite workload shapes. It wins when decode dominates and loses when
prefill dominates, because chunked prefill is a concurrency/memory feature, not
a prefill-latency feature.

### Why ReTool improves

- ReTool prompts are 197-462 tokens, below `prefill_chunk_size=512`, so chunked
  prefill is a no-op (`chunked_prefill_chunks=0`). The optimized arm is effectively
  **decode graphs only**.
- 512 output tokens make decode ~98% of baseline wall time (156.4 s of 159.7 s).
  The bucket graphs replace the eager per-token launch path and replay the smallest
  captured width; Qwen3-4B ReTool goes 159.7 s -> 86.9 s (decode 156.4 s -> 83.8 s).
- This is the intended decode win: fewer launches and a fixed-shape replay for a
  long, narrow decode loop.

### Why 10k/20k regress

- With `max_new_tokens=32`, decode is only ~20-27% of baseline wall time. The graph
  saves a few seconds, while chunked prefill adds tens of seconds.
- Qwen3-4B 20k: prefill 109.5 s -> 162.7 s (+49%), decode 42.4 s -> 40.4 s (-5%),
  wall 152.3 s -> 204.1 s (+34%). Qwen3-4B 10k: prefill 21.4 s -> 48.2 s (+125%),
  wall 49.3 s -> 68.6 s (+39%). Qwen3-8B 10k: prefill 31.5 s -> 58.4 s (+85%),
  wall 56.1 s -> 81.0 s (+44%).
- **Chunking is not free.** A 20k prompt becomes 40 x 512-token forwards instead of
  one large forward. The smaller GEMMs are less efficient, each chunk pays launch/
  mask/graph-dispatch overhead, and the prefill bucket may run a wider parked row
  than the active wave (see `prefill_padded_rows`).
- **The staging cache doubles KV traffic.** Each chunk writes into a second KV cache,
  and the completed prompt is copied into the decode cache. For 20k this is roughly
  2.8 GB of KV per slot (Qwen3-4B/8B geometry) copied once per prompt, on top of the
  chunked writes. That is pure overhead for a prefill-only workload.
- **There is almost no decode to overlap.** Chunked prefill is useful when a long
  prompt can fill while other rows keep decoding. With 32 new tokens and 16 prompts
  the loop is prefill-bound; the overlap window is too small to amortize the chunks.
- The same mechanism explains 10k: 20 chunks instead of 40, but the per-chunk
  overhead and staging copy are still proportional to prompt length.

### Practical takeaway

- Use **decode graphs** for decode-heavy workloads (ReTool-like: short prompts, long
  responses).
- Do **not** enable chunked prefill for 20k/10k with short responses unless the goal
  is concurrency/memory, not latency. If chunking is required, try a larger chunk
  (2048-4096) to reduce per-chunk overhead and staging traffic.
- Report the bundle per workload; a single aggregate would hide the sign flip.

### Caveats

- Partial sweep: Qwen3-4B LongBench baseline, Qwen3-8B 20k C=2 pair, and all
  Qwen3.5 cells are still pending.
- Baseline vs optimized response IDs diverge after 10-30 tokens for several rows.
  ReTool (no chunking) shows the same divergence, so it is decode-graph/batch-width
  numerical divergence rather than chunked-prefill corruption. Performance numbers
  remain valid; the strict bitwise-token-match gate is not met.
- Qwen3.5 hybrid prefill graphs are run eagerly (recurrent-state capture is not
  replay-stable); its decode graphs and chunked prefill remain active.

## Per-cell results

| Model | Workload | Arm | C | Wall (s) | Prefill (s) | Decode (s) | Wall tok/s | Prefill tok/s | Decode tok/s | Peak (MiB) | Match | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| qwen3-4b | 10k | baseline | 3 | 49.31 | 21.42 | 27.30 | 10.38 | 7649.68 | 18.76 | 18722 | False | token_mismatch |
| qwen3-4b | 10k | optimized | 3 | 68.64 | 48.24 | 19.88 | 7.46 | 3396.70 | 25.76 | 16958 | False | token_mismatch |
| qwen3-4b | 20k | baseline | 3 | 152.31 | 109.48 | 42.38 | 3.36 | 2992.98 | 12.08 | 31491 | False | token_mismatch |
| qwen3-4b | 20k | optimized | 3 | 204.10 | 162.70 | 40.36 | 2.51 | 2014.03 | 12.68 | 26111 | False | token_mismatch |
| qwen3-4b | longbench | optimized | 3 | 162.30 | 34.31 | 126.81 | 22.80 | 3820.57 | 29.18 | 15258 | None | missing |
| qwen3-4b | retool | baseline | 3 | 159.67 | 0.91 | 156.38 | 51.31 | 35977.50 | 52.39 | 8454 | False | token_mismatch |
| qwen3-4b | retool | optimized | 3 | 86.87 | 0.92 | 83.80 | 94.30 | 35613.20 | 97.75 | 8675 | False | token_mismatch |
| qwen3-8b | 10k | baseline | 3 | 56.10 | 31.54 | 24.26 | 9.13 | 5194.86 | 21.10 | 27430 | False | token_mismatch |
| qwen3-8b | 10k | optimized | 3 | 81.02 | 58.39 | 22.25 | 6.32 | 2806.02 | 23.01 | 24868 | False | token_mismatch |

## Optimization deltas

| Model | Workload | Wall Δ | Prefill Δ | Decode Δ | Wall tok/s Δ | Peak Δ | Match |
|---|---|---:|---:|---:|---:|---:|---|
| qwen3-4b | 10k | -28.2 | -55.6 | 37.3 | -28.2 | -9.4 | False |
| qwen3-4b | 20k | -25.4 | -32.7 | 5.0 | -25.4 | -17.1 | False |
| qwen3-4b | retool | 83.8 | -1.0 | 86.6 | 83.8 | 2.6 | False |
| qwen3-8b | 10k | -30.8 | -46.0 | 9.0 | -30.8 | -9.3 | False |

