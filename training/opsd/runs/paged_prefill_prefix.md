# Paged prefill and prefix reuse - Qwen3-4B on A40

Implementation: `PagedAttentionCache` now accepts `q_len > 1`; `HybridEngineRollout`
prefills directly into the paged block table in `prefill_chunk_size` windows. The dense
staging cache and the prefill-to-paged copy are not used on the paged path.

Prefix reuse: complete prompt blocks are hashed with a chained blake2b digest. A new row
points at a matching block chain with refcounts incremented, then only the uncached suffix
is prefilled. This cache shares blocks between live rows; it does not retain blocks after
the source row retires, and it is disabled for the graph-safe cache.

Setup: Qwen3-4B, bf16, A40 48 GB, greedy, no graph capture, chunk 512, 8 new tokens.
Commands are at the bottom.

## Results

| Workload | Prefill | Prefill ms | Forwards | Decode ms | Total ms | Peak MiB | Response SHA |
|---|---|---:|---:|---:|---:|---:|---|
| 6k, batch 2 | dense | 3253.7 | 24 | 934.9 | 4509.2 | 10522.8 | `5530534d…` |
| 6k, batch 2 | paged | **2081.3** | 24 | 420.1 | **2704.6** | **9521.3** | `5530534d…` |
| 20k, batch 1 | dense | 10913.2 | 39 | 792.5 | 11954.9 | 13585.9 | `c4291b80…` |
| 20k, batch 1 | paged | **3972.8** | 39 | 418.3 | **4591.5** | **10521.6** | `c4291b80…` |
| 6k, batch 2, duplicate prompts | paged, no reuse | 2081.3 | 24 | 420.1 | 2704.6 | 9521.3 | `5530534d…` |
| 6k, batch 2, duplicate prompts | paged, prefix reuse | **1269.7** | **13** | 382.2 | **1875.8** | 9521.3 | `5530534d…` |

The duplicate-prefix run reused 95 blocks / 6080 tokens and halved the prefill forwards
(24 -> 13), with identical response tokens. The 6k paged prefill is 1.56x faster than the
dense prefill and the 20k paged prefill is 2.75x faster; end-to-end total drops 1.67x and
2.60x respectively. Prefix reuse lowers the 6k prefill 1.64x and total latency 1.44x.
Peak memory falls for paged prefill because the dense staging buffers are gone.

Peak memory for prefix reuse is unchanged: `PagedAttentionCache` sizes the
`BlockedKVCache` pool for the full per-row block table, so shared blocks reduce KV writes
and prefill compute, not pool capacity. `blocks_in_use` counts a shared block once per row.

## Accuracy

- 6k batch 2: dense and paged response IDs and SHA are identical.
- 20k batch 1: dense and paged response IDs and SHA are identical.
- Prefix reuse: reuse and no-reuse response IDs and SHA are identical; both rows match.
- 512-token standalone dense one-shot run (`runs/dense_prefill_smoke2.json`) starts at
  token 323, while the paged path and HF `generate` both start at token 11. The 512 dense
  run is not used as an accuracy reference in this table; the 6k/20k staged runs agree.
- Paged prefill with graph capture was also checked on the 6k prompt: response SHA
  `7fe6d968…`, identical tokens to the eager paged run; graph capture only covers decode.

## Implementation notes

- `_build_atoms` splits a multi-token query row into one atom per `q_block_size`; the
  kernel requires that split even when the row is prefilling.
- The paged attention output is `[rows, q, heads, head_dim]`; the shim transposes it back
  to `[rows, q, hidden]` before `o_proj` (the dense forward returns that layout).
- `_paged_prefill` calls `set_eager_mode()` first: a new request can arrive after a decode
  graph has captured, and the graph atoms must not be reused for a multi-token forward.
- `set_graph_mode()` rebuilds the decode atom structure before capture/replay because the
  eager prefill writes into the shared atom buffer.
- Prefix hashes are complete-block hashes only; the suffix starts at a block boundary and
  at least one query token is always prefilled to produce the first response logits.

## Commands

```bash
cd /workspace/DeepSpeedExamples/training/opsd
export HF_HOME=/dev/shm/hf_cache

# dense prefill, 6k batch 2
python benchmarks/bench_continuous_graph.py --model Qwen/Qwen3-4B \
  --data data/prefix_reuse_6k.jsonl --batch-size 2 --continuous-batch-size 2 \
  --max-prompt-length 6144 --max-new-tokens 8 --warmup 0 --iterations 1 \
  --enable-chunked-prefill --prefill-chunk-size 512 --temperature 0.0 \
  --top-p 0.9 --seed 0 --no-graph-capture --save-response-ids \
  --output runs/dense_prefill_6k_b2.json

# paged prefill, 6k batch 2
python benchmarks/bench_continuous_graph.py --model Qwen/Qwen3-4B \
  --data data/prefix_reuse_6k.jsonl --batch-size 2 --continuous-batch-size 2 \
  --max-prompt-length 6144 --max-new-tokens 8 --warmup 0 --iterations 1 \
  --enable-chunked-prefill --prefill-chunk-size 512 --temperature 0.0 \
  --top-p 0.9 --seed 0 --no-graph-capture --use-paged-cache --save-response-ids \
  --output runs/prefix_no_reuse.json

# paged prefill + prefix reuse
python benchmarks/bench_continuous_graph.py --model Qwen/Qwen3-4B \
  --data data/prefix_reuse_6k.jsonl --batch-size 2 --continuous-batch-size 2 \
  --max-prompt-length 6144 --max-new-tokens 8 --warmup 0 --iterations 1 \
  --enable-chunked-prefill --prefill-chunk-size 512 --temperature 0.0 \
  --top-p 0.9 --seed 0 --no-graph-capture --use-paged-cache --enable-prefix-reuse \
  --save-response-ids --output runs/prefix_reuse.json

# 20k paged prefill
python benchmarks/bench_continuous_graph.py --model Qwen/Qwen3-4B \
  --data data/longbench_20k.jsonl --batch-size 1 --continuous-batch-size 1 \
  --max-prompt-length 20480 --max-new-tokens 8 --warmup 0 --iterations 1 \
  --enable-chunked-prefill --prefill-chunk-size 512 --temperature 0.0 \
  --top-p 0.9 --seed 0 --no-graph-capture --use-paged-cache --save-response-ids \
  --output runs/paged_prefill_20k_b1.json

# 20k dense prefill
python benchmarks/bench_continuous_graph.py --model Qwen/Qwen3-4B \
  --data data/longbench_20k.jsonl --batch-size 1 --continuous-batch-size 1 \
  --max-prompt-length 20480 --max-new-tokens 8 --warmup 0 --iterations 1 \
  --enable-chunked-prefill --prefill-chunk-size 512 --temperature 0.0 \
  --top-p 0.9 --seed 0 --no-graph-capture --save-response-ids \
  --output runs/dense_prefill_20k_b1.json
```

Artifacts: `runs/dense_prefill_6k_b2.json`, `runs/prefix_no_reuse.json`,
`runs/prefix_reuse.json`, `runs/paged_prefill_20k_b1.json`,
`runs/dense_prefill_20k_b1.json`, `runs/graph_paged_prefill_6k.json`.
