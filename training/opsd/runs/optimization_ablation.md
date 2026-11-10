# Rollout optimization ablation — consolidated

This consolidates the optimization results that were previously spread across
`runs/fourway_6k.md`, `runs/fourway_8b.md`, `runs/paged_attention_6k.md`,
`runs/paged_prefill_prefix.md`, `runs/20k_chunked_vs_naive.md`,
`runs/chunk_sweep_2026-11-10/summary.md`, `runs/rollout_matrix_2026-11-10/`, and
`runs/vllm_comparison.md`.

The old ablation data was **not uniformly isolated**: several “before” rows included
chunked prefill and/or graph capture, so a few numbers in the earlier reports attribute
too much to one optimization. This report separates the clean no-chunk 4B 6k/20k
ablations from the confounded four-way and chunked-prefill runs.

## TL;DR

- **Paged decode is the consistent win.** It improves decode 1.5x at 4B/6k, 2.5x at
  4B/20k, 1.35x at 8B/6k, and 2.16x at 8B/20k.
- **CUDA graphs are the second win**, mostly on decode: 1.14x total / 1.31x decode at
  4B/6k and 1.02x total / 1.12x decode at 4B/20k. The bucket graph ladder adds 1.60x
  at 20k with 4 active slots, but costs 42.5% peak memory.
- **Paged prefill is mostly a memory/staging win, not a speed win when chunking is off.**
  At 4B/6k it is time-neutral (1.02x total) and cuts peak memory 8.7%; at 4B/20k it is
  slightly slower (0.99x) but cuts peak memory 18.6%. The large 1.56-2.75x prefill gains
  in the earlier report compare against the dense **chunked staging + copy** path, not
  against a no-chunk dense prefill.
- **Prefix reuse works on duplicate prefixes:** 1.64x prefill and 1.44x total at 6k,
  with 6080 tokens / 95 blocks reused and identical tokens. It does not reduce peak
  memory with the current fixed-size block pool.
- **Chunked prefill is not a universal throughput optimization.** It helps LongBench
  prefill latency (-11.5%) but only +1.3% throughput; it hurts prefill-bound 5k/10k
  (17.8-45.3% slower) and adds 9.3% at 6k and 13.3% at 20k versus the no-chunk path.
- **Best current vs vLLM:** 1.37x slower at 4B/6k (1129 vs 824 ms) and 1.22x slower at
  4B/20k (3398 vs 2794 ms), both no-chunk. Before the clean optimization stack, dense
  eager was 1.84x / 1.38x slower.

## Clean 4B 6k ablation (no chunked prefill)

Qwen3-4B, A40, bf16, LongBench prompt truncated to 6144 tokens, batch 1, 16 new tokens,
Gumbel sampler (`temperature=0.7`, `top_p=0.9`, seed 0), warmup 1, one measured iteration.

| Step | Prefill ms | Decode ms | Total ms | Tok/s | Peak MiB | Response SHA |
|---|---:|---:|---:|---:|---:|---|
| Dense eager (no graph) | 716.6 | 775.0 | 1515.6 | 10.56 | 9931.1 | `d4a324e9…` |
| + CUDA graph capture | 715.6 | 590.1 | 1325.8 | 12.07 | 9947.6 | `d4a324e9…` |
| + paged decode | 718.5 | 389.8 | 1147.8 | 13.94 | 9955.6 | `d4a324e9…` |
| + paged prefill | 719.4 | 389.9 | **1128.6** | **14.18** | **9091.5** | `d4a324e9…` |
| vLLM 0.33 no-chunk reference | — | — | **823.6** | **19.43** | — | `d4a324e9…` |

| Increment | Total speedup | Decode speedup | Other effect |
|---|---:|---:|---|
| CUDA graph capture | 1.14x | 1.31x | +16.5 MiB peak |
| Paged decode (on graph) | 1.15x | 1.51x | +8.0 MiB peak |
| Paged prefill (on paged decode) | 1.02x | ~1.00x | **-8.7% peak** |
| All three vs dense eager | **1.34x** | **1.99x** | peak 9931 -> 9092 MiB |
| Best vs vLLM (no chunk) | **1.37x slower** | — | vLLM 1.37x faster |
| Dense eager vs vLLM (no chunk) | **1.84x slower** | — | — |

The chunked four-way row from the earlier report is **2011.7 ms**, but that is not a
clean “before” row: chunked prefill with dense staging adds ~686 ms versus the no-chunk
graph path. Chunking is measured separately below.

## Clean 4B 20k ablation (no chunked prefill)

Qwen3-4B, A40, bf16, LongBench 20k prompt, batch 1, 8 new tokens, greedy, warmup 1,
one measured iteration.

| Step | Prefill ms | Decode ms | Total ms | Tok/s | Peak MiB | Response SHA |
|---|---:|---:|---:|---:|---:|---|
| Dense eager (no graph) | 3058.6 | 767.6 | 3852.9 | 2.08 | 14697.0 | `c4291b80…` |
| + CUDA graph capture | 3067.5 | 688.4 | 3777.5 | 2.12 | 14713.6 | `c4291b80…` |
| + paged decode | 3071.1 | 271.5 | **3397.9** | **2.35** | 14717.3 | `c4291b80…` |
| + paged prefill | 3129.2 | 273.9 | 3421.5 | 2.34 | **11975.2** | `c4291b80…` |
| vLLM 0.33 no-chunk reference | — | — | **2794.4** | **2.86** | — | `c4291b80…` |

| Increment | Total speedup | Decode speedup | Other effect |
|---|---:|---:|---|
| CUDA graph capture | 1.02x | 1.12x | +16.6 MiB peak |
| Paged decode (on graph) | 1.11x | **2.53x** | +3.7 MiB peak |
| Paged prefill (on paged decode) | 0.99x | 0.99x | **-18.6% peak** |
| Best vs vLLM (no chunk) | **1.22x slower** | — | paged decode is 3398 vs 2794 ms |
| Dense eager vs vLLM (no chunk) | **1.38x slower** | — | — |

At 20k, **paged decode is the fastest path** (3398 ms); paged prefill trades 24 ms
(0.7%) for 2.7 GiB less peak memory. Both are about 22% behind vLLM's no-chunk path
(2794 ms).

## Chunked prefill

### LongBench / ReTool / training-step measurements

| Workload | Baseline | Chunked | Delta |
|---|---:|---:|---|
| LongBench 8k, 256 new | 22.4 tok/s | 22.7 tok/s | **+1.3%** |
| LongBench prefill | 1854.5 ms | 1640.1 ms | **-11.5%** |
| LongBench peak | 9996 MiB | 9621 MiB | **-3.8%** |
| ReTool 304 (no chunks) | 82.4 tok/s | 82.5 tok/s | +0.1% |
| Training rollout 2048 | 25.89 s | 26.00 s | +0.4% |
| Training step | 38.06 s | 38.25 s | +0.5% |

### Prefill-bound batch sweep (5k/10k, batch 16, continuous 3, 32 new)

| Prompt | Graph-only wall | Best chunked wall | Chunk | Chunked vs graph-only |
|---:|---:|---:|---:|---:|
| 5120 | 20.28 s | 25.62 s | 1024 | **+26.3% slower** |
| 10240 | 41.48 s | 60.28 s | 2048 | **+45.3% slower** |

Chunking did lower peak memory (about 0.9 GiB at 5k and 1.8 GiB at 10k) and the best
chunk is the largest exact divisor, not the largest chunk. It is a memory/interleaving
feature for this workload, not a throughput win.

### Chunk penalty: ours vs vLLM

| Engine / path | 6k no-chunk | 6k chunk 512 | Chunk penalty | 20k no-chunk | 20k chunk 512 | Chunk penalty |
|---|---:|---:|---:|---:|---:|---:|
| vLLM 0.33 | 823.6 ms | 954.8 ms | **+15.9%** | 2794.4 ms | 3363.6 ms | **+20.4%** |
| DeepSpeed paged prefill/decode | 1128.6 ms | 1233.8 ms | **+9.3%** | 3421.5 ms | 3871.2 ms | **+13.1%** |
| DeepSpeed dense staging + paged decode | 1325.8 ms | 1809.2 ms | **+36.5%** | — | — | — |

vLLM's chunked path is also slower, with a **larger relative penalty** than our paged
chunk path. The uniquely bad case is our dense-staging chunk path; paged prefill replaces
it. Absolute chunked latency still favors vLLM: 1234 vs 955 ms at 6k (1.29x) and
3871 vs 3364 ms at 20k (1.15x).

### 6k/20k four-way comparison

| Workload | Best no-chunk | Chunked four-way | Chunk penalty |
|---|---:|---:|---:|
| 4B 6k dense prefill + paged decode | 1147.8 ms | 1809.2 ms | **+57.6%** |
| 4B 6k paged prefill + paged decode | 1128.6 ms | 1233.8 ms | **+9.3%** |
| 4B 20k paged prefill + paged decode | 3421.5 ms | 3878.5 ms | **+13.3%** |

### 20k chunked vs naive with bucket graphs

| Config | Active slots | Wall | Tok/s | Peak | Note |
|---|---:|---:|---:|---:|---|
| naive, bucket graph | 2 | 134.2 s | 3.81 | 23.6 GB | only 2 concurrent 20k rows fit |
| chunked, single graph | 4 | 316.0 s | 1.62 | 22.6 GB | old code |
| chunked, bucket graph | 4 | 197.0 s | 2.60 | 32.2 GB | **1.60x vs old chunked** |
| chunked, bucket graph | 8 | OOM | — | — | graphs + staging exceed 48 GB |

Naive still has higher raw tok/s because it runs 2 slots; chunked is the only path that
reaches 4 concurrent 20k rows. Bucket capture makes that 4-slot path 1.60x faster than
the old single-graph path, but it costs 42.5% peak memory and OOMs at 8 slots.

## CUDA graphs and the bucket ladder

- **Graph capture, clean 4B 6k:** 1515.6 -> 1325.8 ms total (1.14x), decode 775.0 ->
  590.1 ms (1.31x).
- **Graph capture, clean 4B 20k:** 3852.9 -> 3777.5 ms total (1.02x), decode 767.6 ->
  688.4 ms (1.12x). The prefill dominates, so total gain is small.
- **ReTool continuous batching matrix:** eager 69.26 -> graph 58.29 ms/dec-token
  (1.19x); sampling 69.32 -> 58.28 ms/dec-token (1.19x).
- **Bucket ladder, 20k continuous 4:** wall 316.0 -> 197.0 s (1.60x), decode 141.1 ->
  35.3 s (4.0x), prefill 173.4 -> 156.8 s (1.11x), tok/s 1.62 -> 2.60, peak 22.6 ->
  32.2 GB (+42.5%). At continuous 8 it OOMs.
- **Bucket ladder, 5k/10k:** graph-only 20.28/41.48 s vs single-graph 21.74/43.43 s
  (1.07x / 1.05x); eager was 22.52/43.49 s.
- **Bucket ladder, ReTool width 16:** 58.29 vs 58.28 ms/dec-token — no gain at full
  width; it helps when the active decode batch is below the captured width.

## Paged decode

| Model / workload | Dense | Paged | Decode speedup | Total speedup | Accuracy |
|---|---:|---:|---:|---:|---|
| 4B 6k, eager | 51.59 ms/token | 45.70 | 1.13x | 1.004x | exact |
| 4B 6k, graph | 590.1 ms | 389.8 | **1.51x** | 1.15x | exact |
| 4B 20k, graph | 688.4 ms | 271.5 | **2.53x** | 1.11x | exact |
| 8B 6k, eager | 62.78 ms/token | 46.66 | 1.35x | 1.08x | exact |
| 8B 6k, four-way greedy | 51.93 | 38.65 | 1.34x | 1.06x | exact |
| 8B 6k, four-way Gumbel | 51.91 | 38.46 | 1.35x | 1.08x | 15/16 |
| 8B 10k, four-way Gumbel | 68.91 | 42.42 | 1.63x | 1.14x | 32/32 exact |
| 8B 20k, four-way Gumbel | 110.68 | 51.16 | **2.16x** | 1.13x | 32/32 exact |
| 8B 512, four-way Gumbel | 34.00 | 33.16 | 1.03x | 0.997x | exact |
| 8B ReTool 16x512, sampling | 72.16 ms/step | 39.20 | 1.84x | **1.67x** | divergent streams |
| 8B ReTool 16x512, greedy | 72.15 ms/step | 39.25 | 1.84x | **1.67x** | **1/16 rows exact** |

The 8B ReTool greedy result is the main accuracy caveat: 1 of 16 rows is exact; the first
mismatch is at response token 71, median common prefix 104, 2371/8192 tokens match.
The 6k batch-1, 10k, and 20k checks are exact. The ReTool divergence is consistent with
a bf16 paged-attention numeric flip after a long decode, not a graph-capture failure.

## Paged prefill

| Workload | Dense prefill | Paged prefill | Prefill speedup | Total speedup | Peak change |
|---|---:|---:|---:|---:|---:|
| 4B 6k, no chunk | 718.5 ms | 719.4 | 0.999x | 1.02x | **-8.7%** |
| 4B 20k, no chunk | 3071.1 ms | 3129.2 | 0.981x | 0.99x | **-18.6%** |
| 4B 6k batch 2, dense chunked staging | 3253.7 ms | 2081.3 | **1.56x** | 1.67x | -9.5% |
| 4B 20k batch 1, dense chunked staging | 10913.2 ms | 3972.8 | **2.75x** | 2.60x | -22.5% |
| 4B 6k graph + chunk | 1381.6 ms | 822.7 | **1.68x** | 1.47x | +0.2% |

The 1.56-2.75x rows are **not** paged-prefill-alone gains: they compare against the dense
chunked staging path, which creates a staging cache and copies KV into the paged table.
With chunking disabled, paged prefill is time-neutral and wins on memory. The best 6k/20k
paths use paged prefill at 6k and paged decode at 20k (or paged prefill for its memory
saving).

## Prefix reuse

6k, batch 2, two identical prompts, chunked prefill, no graph:

| Path | Prefill ms | Prefill fwd | Total ms | Tok/s | Peak MiB | Prefix reuse |
|---|---:|---:|---:|---:|---:|---|
| Paged, no reuse | 2081.3 | 24 | 2704.6 | 5.92 | 9521.3 | 0 |
| Paged, prefix reuse | **1269.7** | **13** | **1875.8** | **8.15** | 9521.3 | 95 blocks / 6080 tokens |

- Prefill speedup **1.64x**; total **1.44x**; response tokens identical.
- Peak memory is unchanged: the block pool is sized for the full per-row block table;
  sharing blocks reduces KV writes and prefill compute, not pool capacity.
- Reuse is limited to live rows in the same cache. There is no graph-safe prefix path,
  no cross-generate retention, no `prefix_id` session, and no weight-version salt.

## Other measured effects

| Item | Measurement | Result |
|---|---|---|
| Device-side Gumbel sampler | eager static 60.97 vs device sampler 61.03 ms/dec-token | +0.1% cost; enables logprobs |
| Continuous batching, ReTool 16 rows | non-CB greedy 55.36 vs CB greedy 69.26 ms/dec-token | **CB alone is 1.25x slower**; it is a scheduling feature |
| Continuous batching, sampling | 61.08 vs 69.32 ms/dec-token | 1.13x slower |
| Graph-safe paged cache | enabler for graph + paged decode | no standalone speed claim |
| Shared prefill | not benchmarked in this set | not attributed |

## vLLM reference

vLLM 0.33.0, torch 2.14.0+cu130, A40, same Qwen3-4B, tokenizer, prompt, chunk budget,
sampler, and generated-token count. The KV pool was capped to approximate the
DeepSpeed pool at 6k; results are in `runs/vllm_comparison.md`.

| Workload | Before clean optimization (dense eager) | Best DeepSpeed | vLLM | Best vs vLLM |
|---|---:|---:|---:|---:|
| 4B 6k, 16 new, no chunk | 1515.9 ms | **1128.6 ms** | **823.6 ms** | **1.37x slower** |
| 4B 20k, 8 new greedy | 3852.9 ms | **3397.9 ms** | **2794.4 ms** | **1.22x slower** |

Intermediate 6k ratios to no-chunk vLLM: dense graph 1.61x slower, paged decode 1.39x
slower, paged prefill 1.37x slower. Intermediate 20k: dense graph 1.35x slower, paged
decode 1.22x slower, paged prefill 1.22x slower. The earlier 6k 1.18x and 20k 1.01x
figures compared our no-chunk path to vLLM with chunked prefill, which is not the same
setting and was slower for vLLM.

The 6k 16-token and 20k greedy response tokens are identical between DeepSpeed and vLLM.
At 20k with `T=0.7`, the first four sampled tokens agree and the fifth diverges; vLLM's
sample matched its own greedy run. That is a sampler/numeric boundary, not a cache
correctness mismatch.

## What this ablation does not isolate

- **Graph vs no-graph at 8B** and at 8B 10k/20k: the 8B four-way rows include chunked
  prefill; paged decode is isolated, graph capture is not.
- **Paged prefill at 8B:** not measured; only 4B.
- **Prefix reuse without chunking / at 8B / with graph:** not measured.
- **Continuous batching with variable arrival times:** the matrix is full-batch ReTool;
  it shows no raw throughput win, not the scheduling/interleaving benefit.
- **Multi-model sweep:** Qwen3.5-hybrid remains unsupported by the paged shim; the full
  4-model x 4-setting sweep is not complete.
- **Mixed prefill+decode in one forward:** not implemented; paged prefill is one request
  at a time.
- Most measurements are one measured iteration after one warmup on one A40. The
  LongBench chunked run and the chunk sweep are more stable (3 iterations); treat the
  rest as directional.

## Source map

- `runs/ablation_6k_dense_eager_nochunk.json`
- `runs/ablation_6k_dense_graph_nochunk.json`
- `runs/ablation_6k_paged_decode_graph_nochunk.json`
- `runs/ablation_6k_paged_prefill_graph_nochunk.json`
- `runs/ablation_20k_dense_eager_nochunk.json`
- `runs/ablation_20k_dense_graph_nochunk.json`
- `runs/ablation_20k_paged_decode_eager_nochunk.json`
- `runs/ablation_20k_paged_decode_graph_nochunk.json`
- `runs/ablation_20k_paged_prefill_graph_nochunk.json`
- `runs/fourway_6k.md`, `runs/fourway_8b.md`, `runs/paged_attention_6k.md`
- `runs/paged_prefill_prefix.md`, `runs/20k_chunked_vs_naive.md`
- `runs/chunk_sweep_2026-11-10/summary.md`
- `runs/rollout_matrix_2026-11-10/summary.md`
- `runs/vllm_comparison.md`
