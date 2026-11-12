# Rollout optimization ablation — A100 re-run

Qwen3-4B, A100-SXM4-40GB, bf16. Same data, batch 1, warmup 1, one measured iteration, seed 0 as the A40 run (`runs/optimization_ablation.md`). 6k = LongBench truncated to 6144, 16 new tokens, Gumbel T=0.7/top_p=0.9; 20k = LongBench 20480, 8 new tokens, greedy. A40 column: official md values. JSON outputs: `runs/a100_2026-11-11/`.

Note: three on-disk A40 JSONs (6k dense-eager, dense-graph, paged-prefill) were re-run on 2026-11-11 and show a ~355 ms prefill versus the md's ~716 ms; cause not established (md values remain the official A40 reference). All A100 6k prefills measured ~355 ms on every path, so prefill-vs-A40 ratios in the tables use the md's 716 ms basis and should be read as upper bounds on the prefill speedup.

## Clean 4B 6k ablation (no chunked prefill)

| Step | A40 Prefill ms | A40 Decode ms | A40 Total ms | A40 Tok/s | A40 Peak MiB | A100 Prefill ms | A100 Decode ms | A100 Total ms | A100 Tok/s | A100 Peak MiB | SHA A40 | SHA A100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense eager (no graph) | 716.6 | 775.0 | 1515.6 | 10.6 | 9931 | 355.9 | 717.5 | 1094.2 | 14.6 | 9923 | `d4a324e9` | `d4a324e9` |
| + CUDA graph capture | 715.6 | 590.1 | 1325.8 | 12.1 | 9948 | 358.7 | 448.9 | 824.6 | 19.4 | 9923 | `d4a324e9` | `d4a324e9` |
| + paged decode | 718.5 | 389.8 | 1147.8 | 13.9 | 9956 | 357.1 | 305.9 | 697.2 | 23.0 | 9931 | `d4a324e9` | `d4a324e9` |
| + paged prefill | 719.4 | 389.9 | 1128.6 | 14.2 | 9092 | 347.9 | 306.2 | 670.5 | 23.9 | 9066 | `d4a324e9` | `d4a324e9` |
| + torch.compile | 649.9 | 348.1 | 1017.8 | 15.7 | 10436 | 310.5 | 279.4 | 611.1 | 26.3 | 11232 | `d4a324e9` | `d4a324e9` |
| vLLM 0.33 no-chunk reference | - | - | 823.6 | 19.4 | - | - | - | 401.1 | - | - | `d4a324e9` | `d4a324e9` |

| Increment | Total A40 | Total A100 | Decode A40 | Decode A100 |
|---|---:|---:|---:|---:|
| CUDA graph capture | 1.1 | 1.3 | 1.3 | 1.6 |
| Paged decode (on graph) | 1.2 | 1.2 | 1.5 | 1.5 |
| Paged prefill (on paged decode) | 1.0 | 1.0 | 1.0 | 1.0 |
| torch.compile on top | 1.1 | 1.1 | 1.1 | 1.1 |
| Best vs vLLM (no chunk) | 1.2 | 1.5 | — | — |

## Clean 4B 20k ablation (no chunked prefill)

| Step | A40 Prefill ms | A40 Decode ms | A40 Total ms | A40 Tok/s | A40 Peak MiB | A100 Prefill ms | A100 Decode ms | A100 Total ms | A100 Tok/s | A100 Peak MiB | SHA A40 | SHA A100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense eager (no graph) | 3058.6 | 767.6 | 3852.9 | 2.1 | 14697 | 1481.8 | 568.5 | 2066.7 | 3.9 | 14690 | `c4291b80` | `c4291b80` |
| + CUDA graph capture | 3067.5 | 688.4 | 3777.5 | 2.1 | 14714 | 1516.0 | 485.6 | 2015.4 | 4.0 | 14690 | `c4291b80` | `c4291b80` |
| + paged decode | 3071.1 | 271.5 | 3397.9 | 2.4 | 14717 | 1481.7 | 262.6 | 1791.3 | 4.5 | 14694 | `c4291b80` | `c4291b80` |
| + paged prefill | 3129.2 | 273.9 | 3421.5 | 2.3 | 11975 | 1429.8 | 263.7 | 1706.3 | 4.7 | 11952 | `c4291b80` | `c4291b80` |
| + torch.compile | 2933.0 | 254.9 | 3208.0 | 2.5 | 16244 | 1329.3 | 251.9 | 1593.9 | 5.0 | 17202 | `c4291b80` | `c4291b80` |
| vLLM 0.33 no-chunk reference | - | - | 2794.4 | 2.9 | - | - | - | 1307.4 | - | - | `c4291b80` | `c4291b80` |

| Increment | Total A40 | Total A100 | Decode A40 | Decode A100 |
|---|---:|---:|---:|---:|
| CUDA graph capture | 1.0 | 1.0 | 1.1 | 1.2 |
| Paged decode (on graph) | 1.1 | 1.1 | 2.5 | 1.8 |
| Paged prefill (on paged decode) | 1.0 | 1.0 | 1.0 | 1.0 |
| torch.compile on top | 1.1 | 1.1 | 1.1 | 1.0 |
| Best vs vLLM (no chunk) | 1.1 | 1.2 | — | — |

## vLLM reference and distance

vLLM 0.33.0, same prompt/sampler/counts; KV capped to 1.1 GiB (6k) / 3.3 GiB (20k). vLLM cells re-run on A100 2026-11-12.

| Workload | Before clean (dense eager) | Best DeepSpeed A100 | vLLM | Best vs vLLM (A100) |
|---|---:|---:|---:|---:|
| 4B 6k, 16 new, no chunk | 1094.2 ms | 611.1 ms (compiled) | 401.1 ms | 1.5x slower |
| 4B 20k, 8 new greedy | 2066.7 ms | 1593.9 ms (compiled) | 1307.4 ms | 1.2x slower |

## Chunk penalty: DeepSpeed (A100 re-run vs A40)

| Engine / path | 6k no-chunk A100 | 6k chunk 512 A100 | Penalty A100 | 20k no-chunk A100 | 20k chunk 512 A100 | Penalty A100 |
|---|---:|---:|---:|---:|---:|---:|
| DeepSpeed paged prefill/decode | 670.5 | 879.8 | 31.2% | 1706.3 | 2601.0 | 52.4% |
| DeepSpeed dense staging + paged decode | 824.6 | 1190.1 | 44.3% | — | — | — |
A40 reference: vLLM +15.9%/+20.4%, DeepSpeed paged +9.3%/+13.1%, dense staging +36.5% (see `runs/optimization_ablation.md`).

## Prefix reuse (6k, batch 2, identical prompts, chunk 512, no graph)

| Path | Prefill ms A40 | Prefill ms A100 | Total ms A100 | Tok/s A100 | Peak MiB A100 | Reuse (A100) |
|---|---:|---:|---:|---:|---:|---|
| Paged, no reuse | 2081.3 | 1141.1 | 1482.5 | 10.8 | 9513 | — |
| Paged, prefix reuse | 1269.7 | 621.6 | 967.8 | 16.5 | 9513 | 95 blocks / 6080 tokens |

- Prefill speedup on A100: **1.8x**; total: **1.5x**.
- A100 prefix cells re-run 2026-11-12 with `--enable-paged-prefill`: the original cells ran on code where paged prefill was still implicit under `--use-paged-cache` (the A40 effective config), so they took the dense-staging chunk path instead (0 blocks reused, 24 prefill forwards) — the "prefix anomaly" in the profile notes, a flag/code drift, not hardware.

## FlashInfer prefill backend

| Workload | Blocked prefill A100 | FlashInfer prefill A100 | Total A100 | Peak MiB A100 |
|---|---:|---:|---:|---:|
| 4B 6k | 670.3 ms | 692.9 ms | 1.0 | 9202 vs 9066 |
| 4B 20k | 1706.3 ms | 1758.5 ms | 1.0 | 12088 vs 11952 |

## Response-token check

| Cell | A40 SHA | A100 SHA | Match |
|---|---|---|---|
| ablation_6k_dense_eager_nochunk | `d4a324e9` | `d4a324e9` | yes |
| ablation_6k_dense_graph_nochunk | `d4a324e9` | `d4a324e9` | yes |
| ablation_6k_paged_decode_graph_nochunk | `d4a324e9` | `d4a324e9` | yes |
| ablation_6k_paged_prefill_graph_nochunk | `d4a324e9` | `d4a324e9` | yes |
| compile_6k_graph | `d4a324e9` | `d4a324e9` | yes |
| compiled_sampler_6k | `d4a324e9` | `d4a324e9` | yes |
| flashinfer_6k_baseline | `d4a324e9` | `d4a324e9` | yes |
| flashinfer_6k | `d4a324e9` | `d4a324e9` | yes |
| chunk_paged_prefill_6k | `d4a324e9` | `d4a324e9` | yes |
| chunk_dense_staging_6k | `d4a324e9` | `d4a324e9` | yes |
| vllm_6k_nochunk | `d4a324e9` | `d4a324e9` | yes |
| ablation_20k_dense_eager_nochunk | `c4291b80` | `c4291b80` | yes |
| ablation_20k_dense_graph_nochunk | `c4291b80` | `c4291b80` | yes |
| ablation_20k_paged_decode_eager_nochunk | `c4291b80` | `c4291b80` | yes |
| ablation_20k_paged_decode_graph_nochunk | `c4291b80` | `c4291b80` | yes |
| ablation_20k_paged_prefill_graph_nochunk | `c4291b80` | `c4291b80` | yes |
| compile_20k_graph | `c4291b80` | `c4291b80` | yes |
| flashinfer_20k | `c4291b80` | `c4291b80` | yes |
| chunk_paged_prefill_20k_greedy | `c4291b80` | `c4291b80` | yes |
| vllm_20k_nochunk_greedy | `c4291b80` | `c4291b80` | yes |
| prefix_no_reuse | `5530534d` | `5530534d` | yes |
| prefix_reuse | `5530534d` | `5530534d` | yes |

