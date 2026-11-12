# Rollout optimization ablation — A100 re-run

Qwen3-4B, A100-SXM4-40GB, bf16. Same data, batch 1, warmup 1, one measured iteration, seed 0 as the A40 run (`runs/optimization_ablation.md`). 6k = LongBench truncated to 6144, 16 new tokens, Gumbel T=0.7/top_p=0.9; 20k = LongBench 20480, 8 new tokens, greedy. A40 column: official md values. JSON outputs: `runs/a100_2026-11-11/`.

Note: three on-disk A40 JSONs (6k dense-eager, dense-graph, paged-prefill) were re-run on 2026-11-11 and show a ~355 ms prefill versus the md's ~716 ms; cause not established (md values remain the official A40 reference). All A100 6k prefills measured ~355 ms on every path, so prefill-vs-A40 ratios in the tables use the md's 716 ms basis and should be read as upper bounds on the prefill speedup.

Note: all A100 cells in this study were re-run on 2026-11-12 on the post-optimization code base, so the decode elementwise/GQA cleanup (no-cat RoPE, head-major attention output, causal-mask skip, scalar position) is included in every paged cell; the original 2026-11-11 JSONs are backed up in `logs/ablation_11-11_original/`. The new-best rows add `--use-flashinfer-decode` and `--compiled-sampler` on top of the previous best (paged + paged-prefill + torch.compile). A40 has no matching cell for the new best (no FlashInfer decode), so its columns show "—" and its SHA is checked against the expected token reference instead. The chunk-size sweep (15 cells) and the 6k greedy isolation cell were added the same day; all sweep cells reproduce the 6k/20k token SHAs.

## Clean 4B 6k ablation (no chunked prefill)

| Step | A40 Prefill ms | A40 Decode ms | A40 Total ms | A40 Tok/s | A40 Peak MiB | A100 Prefill ms | A100 Decode ms | A100 Total ms | A100 Tok/s | A100 Peak MiB | SHA A40 | SHA A100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense eager (no graph) | 716.6 | 775.0 | 1515.6 | 10.6 | 9931 | 355.9 | 704.5 | 1080.7 | 14.8 | 9923 | `d4a324e9` | `d4a324e9` |
| + CUDA graph capture | 715.6 | 590.1 | 1325.8 | 12.1 | 9948 | 361.4 | 447.2 | 825.4 | 19.4 | 9923 | `d4a324e9` | `d4a324e9` |
| + paged decode | 718.5 | 389.8 | 1147.8 | 13.9 | 9956 | 357.8 | 311.3 | 702.0 | 22.8 | 9931 | `d4a324e9` | `d4a324e9` |
| + paged prefill | 719.4 | 389.9 | 1128.6 | 14.2 | 9092 | 352.6 | 312.0 | 679.9 | 23.5 | 9066 | `d4a324e9` | `d4a324e9` |
| + torch.compile | 649.9 | 348.1 | 1017.8 | 15.7 | 10436 | 318.8 | 284.0 | 619.0 | 25.9 | 10442 | `d4a324e9` | `d4a324e9` |
| + FlashInfer decode + compiled sampler (new best) | — | — | — | — | — | 305.1 | 175.6 | 497.6 | 32.2 | 12359 | `…` | `d4a324e9` |
| vLLM 0.33 no-chunk reference | - | - | 823.6 | 19.4 | - | - | - | 401.1 | - | - | `d4a324e9` | `d4a324e9` |

| Increment | Total A40 | Total A100 | Decode A40 | Decode A100 |
|---|---:|---:|---:|---:|
| CUDA graph capture | 1.1 | 1.3 | 1.3 | 1.6 |
| Paged decode (on graph) | 1.2 | 1.2 | 1.5 | 1.4 |
| Paged prefill (on paged decode) | 1.0 | 1.0 | 1.0 | 1.0 |
| torch.compile on top | 1.1 | 1.1 | 1.1 | 1.1 |
| FlashInfer decode + compiled sampler (new best) | - | 1.2 | - | 1.6 |
| Best vs vLLM (no chunk) | 1.2 | 1.2 | — | — |

## Clean 4B 20k ablation (no chunked prefill)

| Step | A40 Prefill ms | A40 Decode ms | A40 Total ms | A40 Tok/s | A40 Peak MiB | A100 Prefill ms | A100 Decode ms | A100 Total ms | A100 Tok/s | A100 Peak MiB | SHA A40 | SHA A100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense eager (no graph) | 3058.6 | 767.6 | 3852.9 | 2.1 | 14697 | 1490.8 | 567.2 | 2074.5 | 3.9 | 14690 | `c4291b80` | `c4291b80` |
| + CUDA graph capture | 3067.5 | 688.4 | 3777.5 | 2.1 | 14714 | 1493.3 | 483.4 | 1990.5 | 4.0 | 14690 | `c4291b80` | `c4291b80` |
| + paged decode | 3071.1 | 271.5 | 3397.9 | 2.4 | 14717 | 1494.2 | 264.0 | 1804.8 | 4.4 | 14694 | `c4291b80` | `c4291b80` |
| + paged prefill | 3129.2 | 273.9 | 3421.5 | 2.3 | 11975 | 1429.7 | 266.3 | 1707.9 | 4.7 | 11952 | `c4291b80` | `c4291b80` |
| + torch.compile | 2933.0 | 254.9 | 3208.0 | 2.5 | 16244 | 1322.2 | 254.8 | 1589.5 | 5.0 | 16324 | `c4291b80` | `c4291b80` |
| + FlashInfer decode + compiled sampler (new best) | — | — | — | — | — | 1325.1 | 92.1 | 1429.3 | 5.6 | 16460 | `…` | `c4291b80` |
| vLLM 0.33 no-chunk reference | - | - | 2794.4 | 2.9 | - | - | - | 1307.4 | - | - | `c4291b80` | `c4291b80` |

| Increment | Total A40 | Total A100 | Decode A40 | Decode A100 |
|---|---:|---:|---:|---:|
| CUDA graph capture | 1.0 | 1.0 | 1.1 | 1.2 |
| Paged decode (on graph) | 1.1 | 1.1 | 2.5 | 1.8 |
| Paged prefill (on paged decode) | 1.0 | 1.1 | 1.0 | 1.0 |
| torch.compile on top | 1.1 | 1.1 | 1.1 | 1.0 |
| FlashInfer decode + compiled sampler (new best) | - | 1.1 | - | 2.8 |
| Best vs vLLM (no chunk) | 1.1 | 1.1 | — | — |

## vLLM reference and distance

vLLM 0.33.0, same prompt/sampler/counts; KV capped to 1.1 GiB (6k) / 3.3 GiB (20k). vLLM cells re-run on A100 2026-11-12.

| Workload | Before clean (dense eager) | Best DeepSpeed A100 | vLLM | Best vs vLLM (A100) |
|---|---:|---:|---:|---:|
| 4B 6k, 16 new, no chunk | 1080.7 ms | 497.6 ms (new best) | 401.1 ms | 1.2x slower |
| 4B 20k, 8 new greedy | 2074.5 ms | 1429.3 ms (new best) | 1307.4 ms | 1.1x slower |

## Chunk penalty: DeepSpeed (A100 re-run vs A40)

| Engine / path | 6k no-chunk A100 | 6k chunk 512 A100 | Penalty A100 | 20k no-chunk A100 | 20k chunk 512 A100 | Penalty A100 |
|---|---:|---:|---:|---:|---:|---:|
| DeepSpeed paged prefill/decode | 679.9 | 890.1 | 30.9% | 1707.9 | 2614.7 | 53.1% |
| DeepSpeed dense staging + paged decode | 825.4 | 1189.2 | 44.1% | — | — | — |
A40 reference: vLLM +15.9%/+20.4%, DeepSpeed paged +9.3%/+13.1%, dense staging +36.5% (see `runs/optimization_ablation.md`).

## Prefill chunk-size sweep (new-best config)

New-best config (`--use-paged-cache --enable-paged-prefill --torch-compile --compiled-sampler --use-flashinfer-decode`) with `--enable-chunked-prefill` at powers-of-two chunk sizes; 6k = Gumbel T=0.7/top_p=0.9, 16 new; 20k = greedy, 8 new. 6k chunk 8192 >= 6144 prompt, so that row is the chunk machinery without actual chunking. Penalty vs the no-chunk new-best row (positive = chunked slower).

| Chunk | 6k Wall ms | 6k Prefill ms | 6k Peak MiB | 6k Penalty | 20k Wall ms | 20k Prefill ms | 20k Peak MiB | 20k Penalty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (no chunk) | 497.6 | 305.1 | 12359 | — | 1429.3 | 1325.1 | 16460 | — |
| 128 | 2121.5 | 1927.3 | 10429 | 326.4% | 6160.5 | 6041.1 | 14190 | 331.0% |
| 256 | 1178.8 | 987.1 | 10468 | 136.9% | 3319.1 | 3208.4 | 13525 | 132.2% |
| 512 | 682.6 | 492.7 | 10545 | 37.2% | 2203.5 | 2097.1 | 13665 | 54.2% |
| 1024 | 556.4 | 367.3 | 10698 | 11.8% | 1765.7 | 1660.9 | 13969 | 23.5% |
| 2048 | 524.0 | 334.3 | 11404 | 5.3% | 1549.5 | 1445.0 | 15059 | 8.4% |
| 4096 | 510.9 | 320.1 | 12028 | 2.7% | 1483.5 | 1379.9 | 17029 | 3.8% |
| 8192 | 492.7 | 303.2 | 13223 | -1.0% | 1454.9 | 1351.2 | 19122 | 1.8% |

- Gumbel vs greedy (6k, new-best, same 16 steps): Gumbel 497.6 ms vs greedy 502.7 ms → no measurable Gumbel-sampling overhead (-5.1 ms, within noise; the compiled sampler absorbs it, and both cells sampled identical tokens `d4a324e9`).

## Prefix reuse (6k, batch 2, identical prompts, chunk 512, no graph)

| Path | Prefill ms A40 | Prefill ms A100 | Total ms A100 | Tok/s A100 | Peak MiB A100 | Reuse (A100) |
|---|---:|---:|---:|---:|---:|---|
| Paged, no reuse | 2081.3 | 1138.8 | 1471.9 | 10.9 | 9513 | — |
| Paged, prefix reuse | 1269.7 | 611.1 | 948.3 | 16.9 | 9513 | 95 blocks / 6080 tokens |

- Prefill speedup on A100: **1.9x**; total: **1.6x**.
- A100 prefix cells re-run 2026-11-12 with `--enable-paged-prefill`: the original cells ran on code where paged prefill was still implicit under `--use-paged-cache` (the A40 effective config), so they took the dense-staging chunk path instead (0 blocks reused, 24 prefill forwards) — the "prefix anomaly" in the profile notes, a flag/code drift, not hardware.

## FlashInfer prefill backend

| Workload | Blocked prefill A100 | FlashInfer prefill A100 | Total A100 | Peak MiB A100 |
|---|---:|---:|---:|---:|
| 4B 6k | 673.9 ms | 690.8 ms | 1.0 | 9202 vs 9066 |
| 4B 20k | 1707.9 ms | 1745.9 ms | 1.0 | 12088 vs 11952 |

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
| newbest_6k | — (no A40 cell) | `d4a324e9` | yes |
| chunk_sweep_6k_c128 | — (no A40 cell) | `d4a324e9` | yes |
| chunk_sweep_6k_c256 | — (no A40 cell) | `d4a324e9` | yes |
| chunk_sweep_6k_c512 | — (no A40 cell) | `d4a324e9` | yes |
| chunk_sweep_6k_c1024 | — (no A40 cell) | `d4a324e9` | yes |
| chunk_sweep_6k_c2048 | — (no A40 cell) | `d4a324e9` | yes |
| chunk_sweep_6k_c4096 | — (no A40 cell) | `d4a324e9` | yes |
| chunk_sweep_6k_c8192 | — (no A40 cell) | `d4a324e9` | yes |
| newbest_6k_greedy | — (no A40 cell) | `d4a324e9` | yes |
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
| newbest_20k | — (no A40 cell) | `c4291b80` | yes |
| chunk_sweep_20k_c128 | — (no A40 cell) | `c4291b80` | yes |
| chunk_sweep_20k_c256 | — (no A40 cell) | `c4291b80` | yes |
| chunk_sweep_20k_c512 | — (no A40 cell) | `c4291b80` | yes |
| chunk_sweep_20k_c1024 | — (no A40 cell) | `c4291b80` | yes |
| chunk_sweep_20k_c2048 | — (no A40 cell) | `c4291b80` | yes |
| chunk_sweep_20k_c4096 | — (no A40 cell) | `c4291b80` | yes |
| chunk_sweep_20k_c8192 | — (no A40 cell) | `c4291b80` | yes |
| chunk_paged_prefill_20k_greedy | `c4291b80` | `c4291b80` | yes |
| vllm_20k_nochunk_greedy | `c4291b80` | `c4291b80` | yes |
| prefix_no_reuse | `5530534d` | `5530534d` | yes |
| prefix_reuse | `5530534d` | `5530534d` | yes |

