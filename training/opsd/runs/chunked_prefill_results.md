# Chunked Prefill Rollout Results — Qwen3-4B on A40

Measured on branch `chunked-prefill-rollout` (DeepSpeed `0.19.9+5145b7cf`),
editable install from `/workspace/DeepSpeed`, run from
`/workspace/DeepSpeedExamples/training/opsd`.

- GPU: NVIDIA A40 (48 GB)
- Model: `Qwen/Qwen3-4B`, bf16, SDPA attention
- Greedy sampling (`temperature=0.0`), graph capture on, profiling on
- Chunked prefill: `enable_chunked_prefill=true`, `prefill_chunk_size=512`

## Summary

| Workload | Prompt length | Chunking active | Throughput delta | Prefill delta | Peak mem delta |
|---|---|---|---|---|---|
| LongBench | 8192 padded (6076 real) | yes, 16 chunks | 22.4 → 22.7 tok/s (+1.3%) | −11.5% | −3.8% |
| ReTool | 304 | no, 0 chunks | 82.4 → 82.5 tok/s (+0.1%) | −0.7% | 0 |
| Training step | 2048 | yes, 4 chunks | rollout +0.4% | — | — |

## LongBench — chunking active

Batch 2, continuous batch size 1, max prompt 8192, 256 new tokens.

| Metric | Baseline | Chunked (512) | Delta |
|---|---|---|---|
| Wall time | 18,197 ms | 17,993 ms | −1.1% |
| Prefill forwards | 2 | 16 | 8× more |
| Prefill total | 1,854.5 ms | 1,640.1 ms | **−11.5%** |
| Decode total | 16,150.5 ms | 16,153.8 ms | +0.02% |
| Decode forwards | 406 | 406 | same |
| Scheduler overhead | 9.2 ms | 9.7 ms | +0.5 ms |
| Cache overhead | 1,871.7 ms | 1,652.4 ms | −11.7% |
| Peak memory | 9,996 MB | 9,621 MB | **−3.8%** |
| Tokens/s | 22.4 | 22.7 | **+1.3%** |
| Staging cache | 0 | 854.44 MB | — |

Stable across 3 iterations (< 0.1% variance).

## ReTool — chunking does not trigger

Batch 4, continuous batch size 2, max prompt 2048, 512 new tokens.
Actual prompt length 304 < `prefill_chunk_size=512`, so `chunked_prefill_chunks=0`.

| Metric | Baseline | Chunked (512) | Delta |
|---|---|---|---|
| Wall time | 24,846 ms | 24,836 ms | −0.04% |
| Prefill total | 168.5 ms | 167.4 ms | −0.7% |
| Decode total | 24,074.9 ms | 24,072.7 ms | −0.01% |
| Decode forwards | 1,022 | 1,022 | same |
| Scheduler overhead | 23.4 ms | 24.2 ms | +0.8 ms |
| Peak memory | 8,145 MB | 8,145 MB | 0 |
| Tokens/s | 82.4 | 82.5 | +0.1% |

## Training step — 2048-token prompt

`deepspeed --num_gpus 1 main.py --config <config>`, one optimizer step.

| Metric | Baseline | Chunked (512) | Delta |
|---|---|---|---|
| Rollout | 25.89 s | 26.00 s | +0.4% |
| Teacher | 1.41 s | 1.43 s | +1.4% |
| Student | 10.76 s | 10.81 s | +0.5% |
| Step | 38.06 s | 38.25 s | +0.5% |
| Loss | 0.0000 | 0.0000 | same |
| Response tokens | 2,048 | 2,048 | same |
| Peak memory | 38.9 GB | 38.9 GB | same |

A transient allocator OOM warning appeared in both runs; the process recovered
and exited 0.

## Standalone `bench_chunked_prefill.py` (CPU sanity run)

64-token prompts, 4 requests, 4 slots, 8 new tokens, fp32, graph off.

| Config | Prefill fwd | Prefill ms | ms/fwd | Decode fwd | Decode ms | Total ms | Tokens/s |
|---|---|---|---|---|---|---|---|
| naive | 1 | 443.80 | 443.80 | 7 | 47.29 | 557.43 | 57.41 |
| chunked@16 | 16 | 73.58 | 4.60 | 19 | 87.02 | 204.87 | 156.20 |
| chunked@32 | 8 | 34.04 | 4.25 | 13 | 90.08 | 138.97 | 230.26 |

All greedy token streams matched naive prefill.

## Interpretation

- **Long prompts win.** Chunked prefill cuts prefill latency 11.5% and peak
  memory 3.8% on the 8192-token LongBench path. End-to-end gain is only 1.1%
  because decode dominates (89% of wall time).
- **Short prompts no-op.** ReTool prompts are 304 tokens; nothing to split, so
  results are identical by design.
- **Training slightly slower.** The 2048-token training prompt splits into 4
  chunks, adding ~0.2 s to the step. Decode (1,022 forwards) dominates, so the
  penalty is 0.5%.
- **Correctness held.** 86/86 DeepSpeed rollout unit tests pass, including
  `test_chunked_prefill_cache_matches_single_forward_prefill_on_a_real_model`
  and `test_static_eager_chunked_prefill_matches_naive_generate`.

## Commands

```bash
source /workspace/opsd_env.sh
cd /workspace/DeepSpeedExamples/training/opsd

# LongBench, chunked
python benchmarks/bench_continuous_graph.py \
  --model Qwen/Qwen3-4B --data data/longbench_mfqa.jsonl \
  --batch-size 2 --continuous-batch-size 1 \
  --max-prompt-length 8192 --max-new-tokens 256 \
  --warmup 1 --iterations 3 \
  --enable-chunked-prefill --prefill-chunk-size 512 \
  --output runs/qwen3_4b_longbench_chunked_prefill.json

# ReTool, chunked
python benchmarks/bench_continuous_graph.py \
  --model Qwen/Qwen3-4B --data data/retool_prompts.jsonl \
  --batch-size 4 --continuous-batch-size 2 \
  --max-prompt-length 2048 --max-new-tokens 512 \
  --warmup 1 --iterations 3 \
  --enable-chunked-prefill --prefill-chunk-size 512 \
  --output runs/qwen3_4b_retool_chunked_prefill.json

# Training step, chunked
bash scripts/run_baseline_local.sh configs/chunked_prefill_qwen3_4b_continuous_graph.json
```

## Artifacts

| File | Contents |
|---|---|
| `runs/qwen3_4b_longbench_chunked_prefill.json` | LongBench chunked raw metrics |
| `runs/qwen3_4b_retool_chunked_prefill.json` | ReTool chunked raw metrics |
| `runs/chunked_prefill_qwen3_4b_continuous_graph/step_metrics.jsonl` | Training-step metrics |
| `configs/chunked_prefill_qwen3_4b_continuous_graph.json` | Chunked training config |
| `configs/chunked_prefill_qwen3_4b_longbench_continuous_graph.json` | Chunked LongBench config |

Baseline artifacts: `runs/qwen3_4b_longbench_continuous_graph.json`,
`runs/qwen3_4b_retool_continuous_graph.json`,
`runs/qwen3_4b_continuous_graph/step_metrics.jsonl`.

## Code changes enabling this

- `config.py`: added `enable_chunked_prefill: bool = False` and
  `prefill_chunk_size: int = 512` to `RolloutConfig`.
- `benchmarks/bench_continuous_graph.py`: added `--enable-chunked-prefill` and
  `--prefill-chunk-size`, passed both to `HybridEngineRolloutConfig`, and
  recorded them in the result JSON.
- Engine support (`enable_chunked_prefill`, `prefill_chunk_size`) already
  existed in `HybridEngineRolloutConfig` on the branch; it is read via
  `getattr` in `HybridEngineRollout.__init__`.
