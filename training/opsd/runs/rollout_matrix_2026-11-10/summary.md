# Rollout optimization ablation - HEAD b7ed2d79

## ReTool A/B/C

| Group | Arm | Rows | ms/dec-token | decode ms | gen ms | cache ms | sched ms | prefill ms | prefill tok | peak MiB | fwd | out | stop |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A | noncb_greedy | 16 | 55.36 | 28.29 | 0.33 |  |  | 0.86 | 7392 | 12683 | 511 | 512 | budget |
| A | noncb_greedy_graph | 16 |  | 0.00 | 30.93 |  |  |  | 7392 | 12413 | 0 | 512 | budget |
| A | noncb_sampling | 16 | 61.08 | 31.21 | 1.01 |  |  | 0.87 | 7392 | 10495 | 511 | 512 | budget |
| A | noncb_sampling_graph | 16 |  | 0.00 | 31.05 |  |  | 0.87 | 7392 | 12413 | 0 | 512 | budget |
| B | cb_greedy | 16 | 69.26 | 35.39 | 0.92 | 0.05 | 0.02 | 0.80 | 7392 | 11534 | 511 | 512 | budget |
| B | cb_greedy_graph | 16 | 58.29 | 29.79 | 1.47 | 0.84 | 0.02 | 0.80 | 7392 | 11558 | 511 | 512 | budget |
| B | cb_greedy_graph_buckets | 16 | 58.28 | 29.78 | 1.44 | 0.84 | 0.02 | 0.80 | 7392 | 11558 | 511 | 512 | budget |
| B | cb_sampling | 16 | 69.32 | 35.42 | 1.90 | 0.05 | 0.02 | 0.80 | 7392 | 11534 | 511 | 512 | budget |
| B | cb_sampling_graph | 16 | 58.28 | 29.78 | 2.39 | 0.84 | 0.02 | 0.80 | 7392 | 11556 | 511 | 512 | budget |
| C | cb3_greedy_graph_buckets | 3 | 27.33 | 83.80 | 2.13 | 0.98 | 0.08 | 0.92 | 7392 | 8479 | 3066 | 512 | budget |
| C | cb6_greedy_graph_buckets | 6 | 33.17 | 50.85 | 1.75 | 0.90 | 0.04 | 0.85 | 7392 | 9234 | 1533 | 512 | budget |
| C | cb12_greedy_graph_buckets | 12 | 38.89 | 39.74 | 1.53 | 0.88 | 0.03 | 0.83 | 7392 | 10611 | 1022 | 512 | budget |
| C | cb16_greedy_graph_buckets | 16 | 58.29 | 29.78 | 1.47 | 0.84 | 0.02 | 0.80 | 7392 | 11558 | 511 | 512 | budget |

## Eager feature ablation

| Arm | Rows | ms/dec-token | decode ms | peak MiB | out | stop | logprobs |
|---|---:|---:|---:|---:|---:|---|---|
| eager_static_cache | 16 | 60.97 | 31.16 | 10495 | 512 | budget | True |
| eager_device_sampler | 16 | 61.03 | 31.18 | 10495 | 512 | budget | True |
| continuous_eager_3 | 3 | 53.21 | 163.13 | 8453 | 512 | budget | False |
| continuous_eager_8 | 8 | 50.87 | 51.99 | 9693 | 512 | budget | False |

## Path proof

| Arm | keys | graph_captures | bucket_widths | prefill_padded | logprobs | attn |
|---|---|---|---|---|---|---|
| noncb_greedy | [] | None | None | None | False | sdpa |
| noncb_greedy_graph | [] | None | None | None | False | sdpa |
| noncb_sampling | [] | None | None | None | True | sdpa |
| noncb_sampling_graph | [] | None | None | None | True | sdpa |
| cb_greedy | ['cache_capacity', 'cache_memory_bytes', 'chunked_prefill_chunks', 'chunked_prefill_staging_bytes', 'decode_steps', 'end_to_end_ms', 'peak_cache_length', 'tokens_per_second', 'trim_bytes_moved', 'trim_count', 'trim_frequency', 'trim_latency_ms', 'trimmed_columns'] | None | None | None | False | sdpa |
| cb_greedy_graph | ['cache_capacity', 'cache_memory_bytes', 'chunked_prefill_chunks', 'chunked_prefill_staging_bytes', 'decode_bucket_steps', 'decode_bucket_widths', 'decode_padded_rows', 'decode_steps', 'end_to_end_ms', 'graph_capture_ladder', 'graph_captures', 'peak_cache_length', 'prefill_bucket_steps', 'prefill_bucket_widths', 'prefill_padded_rows', 'tokens_per_second', 'trim_bytes_moved', 'trim_count', 'trim_frequency', 'trim_latency_ms', 'trimmed_columns'] | 0 | {'16': 511} | 0 | False | sdpa |
| cb_greedy_graph_buckets | ['cache_capacity', 'cache_memory_bytes', 'chunked_prefill_chunks', 'chunked_prefill_staging_bytes', 'decode_bucket_steps', 'decode_bucket_widths', 'decode_padded_rows', 'decode_steps', 'end_to_end_ms', 'graph_capture_ladder', 'graph_captures', 'peak_cache_length', 'prefill_bucket_steps', 'prefill_bucket_widths', 'prefill_padded_rows', 'tokens_per_second', 'trim_bytes_moved', 'trim_count', 'trim_frequency', 'trim_latency_ms', 'trimmed_columns'] | 0 | {'16': 511} | 0 | False | sdpa |
| cb_sampling | ['cache_capacity', 'cache_memory_bytes', 'chunked_prefill_chunks', 'chunked_prefill_staging_bytes', 'decode_steps', 'end_to_end_ms', 'peak_cache_length', 'tokens_per_second', 'trim_bytes_moved', 'trim_count', 'trim_frequency', 'trim_latency_ms', 'trimmed_columns'] | None | None | None | True | sdpa |
| cb_sampling_graph | ['cache_capacity', 'cache_memory_bytes', 'chunked_prefill_chunks', 'chunked_prefill_staging_bytes', 'decode_bucket_steps', 'decode_bucket_widths', 'decode_padded_rows', 'decode_steps', 'end_to_end_ms', 'graph_capture_ladder', 'graph_captures', 'peak_cache_length', 'prefill_bucket_steps', 'prefill_bucket_widths', 'prefill_padded_rows', 'tokens_per_second', 'trim_bytes_moved', 'trim_count', 'trim_frequency', 'trim_latency_ms', 'trimmed_columns'] | 0 | {'16': 511} | 0 | True | sdpa |
| cb3_greedy_graph_buckets | ['cache_capacity', 'cache_memory_bytes', 'chunked_prefill_chunks', 'chunked_prefill_staging_bytes', 'decode_bucket_steps', 'decode_bucket_widths', 'decode_padded_rows', 'decode_steps', 'end_to_end_ms', 'graph_capture_ladder', 'graph_captures', 'peak_cache_length', 'prefill_bucket_steps', 'prefill_bucket_widths', 'prefill_padded_rows', 'tokens_per_second', 'trim_bytes_moved', 'trim_count', 'trim_frequency', 'trim_latency_ms', 'trimmed_columns'] | 0 | {'3': 2555, '1': 511} | 0 | False | sdpa |
| cb6_greedy_graph_buckets | ['cache_capacity', 'cache_memory_bytes', 'chunked_prefill_chunks', 'chunked_prefill_staging_bytes', 'decode_bucket_steps', 'decode_bucket_widths', 'decode_padded_rows', 'decode_steps', 'end_to_end_ms', 'graph_capture_ladder', 'graph_captures', 'peak_cache_length', 'prefill_bucket_steps', 'prefill_bucket_widths', 'prefill_padded_rows', 'tokens_per_second', 'trim_bytes_moved', 'trim_count', 'trim_frequency', 'trim_latency_ms', 'trimmed_columns'] | 0 | {'6': 1022, '4': 511} | 0 | False | sdpa |
| cb12_greedy_graph_buckets | ['cache_capacity', 'cache_memory_bytes', 'chunked_prefill_chunks', 'chunked_prefill_staging_bytes', 'decode_bucket_steps', 'decode_bucket_widths', 'decode_padded_rows', 'decode_steps', 'end_to_end_ms', 'graph_capture_ladder', 'graph_captures', 'peak_cache_length', 'prefill_bucket_steps', 'prefill_bucket_widths', 'prefill_padded_rows', 'tokens_per_second', 'trim_bytes_moved', 'trim_count', 'trim_frequency', 'trim_latency_ms', 'trimmed_columns'] | 0 | {'12': 511, '4': 511} | 0 | False | sdpa |
| cb16_greedy_graph_buckets | ['cache_capacity', 'cache_memory_bytes', 'chunked_prefill_chunks', 'chunked_prefill_staging_bytes', 'decode_bucket_steps', 'decode_bucket_widths', 'decode_padded_rows', 'decode_steps', 'end_to_end_ms', 'graph_capture_ladder', 'graph_captures', 'peak_cache_length', 'prefill_bucket_steps', 'prefill_bucket_widths', 'prefill_padded_rows', 'tokens_per_second', 'trim_bytes_moved', 'trim_count', 'trim_frequency', 'trim_latency_ms', 'trimmed_columns'] | 0 | {'16': 511} | 0 | False | sdpa |

## Noise

cb_greedy_graph_buckets reps ms/dec-token: [58.29, 58.29, 58.29]; spread=0.01

## Retool C=3
cb3_greedy_graph_buckets: ms/dec-token=27.33, rows=3, peak=8479 MiB

## 6k6k

