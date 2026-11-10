# vLLM comparison — Qwen3-4B on A40

vLLM 0.33.0, torch 2.14.0+cu130, A40 48 GB. vLLM is isolated in
`/workspace/vllm_env`; the DeepSpeed environment is untouched.

Same input contract as `bench_continuous_graph.py`: first LongBench prompt, chat
template, truncation to 6144 or 20480 tokens, batch 1, no chunked prefill, graph
capture, sampler settings, and generated-token count. vLLM's KV pool was capped to
1.1 GiB at 6k and 3.3 GiB at 20k to approximate the DeepSpeed cache budget.

The no-chunk configuration is the important comparison: both engines are faster with
chunking disabled. Earlier vLLM runs with `max_num_batched_tokens=512` were 954.8 ms at
6k and 3363.6 ms at 20k; those are retained in `runs/vllm_6k_kv1g.json` and
`runs/vllm_20k_greedy.json`, but they are not the same setting as the best path.

## Results

| Workload | Engine / path | Wall ms | Tokens/s | Response SHA |
|---|---|---:|---:|---|
| 6k, 16 new | DeepSpeed dense eager | 1515.9 | 10.56 | `d4a324e9…` |
| 6k, 16 new | DeepSpeed + CUDA graph | 1325.8 | 12.07 | `d4a324e9…` |
| 6k, 16 new | DeepSpeed + paged decode | 1147.8 | 13.94 | `d4a324e9…` |
| 6k, 16 new | DeepSpeed + paged prefill/decode | **1128.6** | **14.18** | `d4a324e9…` |
| 6k, 16 new | vLLM no-chunk | **823.6** | **19.43** | `d4a324e9…` |
| 20k, 8 new greedy | DeepSpeed dense eager | 3852.9 | 2.08 | `c4291b80…` |
| 20k, 8 new greedy | DeepSpeed + CUDA graph | 3777.5 | 2.12 | `c4291b80…` |
| 20k, 8 new greedy | DeepSpeed + paged decode | **3397.9** | **2.35** | `c4291b80…` |
| 20k, 8 new greedy | DeepSpeed + paged prefill/decode | 3421.5 | 2.34 | `c4291b80…` |
| 20k, 8 new greedy | vLLM no-chunk | **2794.4** | **2.86** | `c4291b80…` |

## Distance to vLLM

| Workload | Before clean optimization | Best DeepSpeed | vLLM | Best vs vLLM |
|---|---:|---:|---:|---:|
| 4B 6k, 16 new | 1515.9 ms | 1128.6 ms | 823.6 ms | **1.37x slower** |
| 4B 20k, 8 new greedy | 3852.9 ms | 3397.9 ms | 2794.4 ms | **1.22x slower** |

- At 6k, vLLM is **1.37x faster** than the best DeepSpeed path; before the clean
  optimization stack it was **1.84x faster**.
- At 20k, vLLM is **1.22x faster**; before the clean stack it was **1.38x faster**.
- Both engines return identical 6k tokens and identical 20k greedy tokens.
- The gap is now mostly the fused prefill/attention and sampler path, not the paged
  cache layout. Paged decode is the main DeepSpeed win; paged prefill is a memory win.

### Gap decomposition

| Workload | vLLM prefill+1 | vLLM decode | DeepSpeed prefill | DeepSpeed decode | Prefill gap | Decode gap |
|---|---:|---:|---:|---:|---:|---:|
| 4B 6k | 551.2 ms | 18.2 ms/token | 719.4 ms | 24.4 ms/token | **168 ms (59%)** | 117 ms (41%) |
| 4B 20k | 2647.9 ms | 20.9 ms/token | 3071.1 ms | 33.9 ms/token | **423 ms (77%)** | 125 ms (23%) |

Prefill is the larger absolute gap, especially at 20k; decode is the larger per-token gap.
The next work should target fused prefill/compile first, then decode fusion; a decode-only
change cannot close 20k.

## Chunked vs no-chunk (same KV budgets)

The chunked runs use `max_num_batched_tokens=512` for vLLM and `prefill_chunk_size=512`
for DeepSpeed. The no-chunk runs use the same engines with one prefill forward.

| Engine / path | 6k no-chunk | 6k chunk 512 | Chunk penalty | 20k no-chunk | 20k chunk 512 | Chunk penalty |
|---|---:|---:|---:|---:|---:|---:|
| vLLM | 823.6 ms | 954.8 ms | **+15.9%** | 2794.4 ms | 3363.6 ms | **+20.4%** |
| DeepSpeed paged prefill/decode | 1128.6 ms | 1233.8 ms | **+9.3%** | 3421.5 ms | 3871.2 ms | **+13.1%** |
| DeepSpeed dense staging + paged decode | 1325.8 ms | 1809.2 ms | **+36.5%** | — | — | — |

So **vLLM's chunked path is also slower**, and its relative penalty is larger than our
paged-prefill chunk penalty (+15.9%/+20.4% vs +9.3%/+13.1%). The uniquely bad case is our
**dense-staging** chunk path (+36.5% at 6k), which paged prefill replaces. In absolute terms,
our chunked paged path is still 1.29x slower than vLLM chunked at 6k and 1.15x slower at 20k.

## Commands

```bash
cd /workspace/DeepSpeedExamples/training/opsd
export HF_HOME=/dev/shm/hf_cache
export VLLM_CACHE_ROOT=/dev/shm/vllm_cache
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/torchinductor
export TMPDIR=/dev/shm

# 6k no-chunk vLLM
/workspace/vllm_env/bin/python benchmarks/bench_vllm_offline.py \
  --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 1 \
  --max-prompt-length 6144 --max-new-tokens 16 --prefill-chunk-size 8192 \
  --temperature 0.7 --top-p 0.9 --seed 0 --warmup 1 --iterations 1 \
  --gpu-memory-utilization 0.75 --kv-cache-memory-gib 1.1 \
  --output runs/vllm_6k_nochunk.json

# 20k no-chunk vLLM
/workspace/vllm_env/bin/python benchmarks/bench_vllm_offline.py \
  --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 1 \
  --max-prompt-length 20480 --max-new-tokens 8 --prefill-chunk-size 20488 \
  --temperature 0.0 --top-p 1.0 --seed 0 --warmup 1 --iterations 1 \
  --gpu-memory-utilization 0.75 --kv-cache-memory-gib 3.3 \
  --output runs/vllm_20k_nochunk_greedy.json
```

DeepSpeed rows: `runs/ablation_6k_*_nochunk.json`,
`runs/ablation_20k_*_nochunk.json`; full attribution in
`runs/optimization_ablation.md`. Split runs: `runs/vllm_6k_nochunk_new1.json`,
`runs/vllm_20k_nochunk_new1.json`.
