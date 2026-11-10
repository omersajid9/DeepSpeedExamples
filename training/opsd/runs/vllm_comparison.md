# vLLM comparison - Qwen3-4B on A40

vLLM 0.33.0, torch 2.14.0+cu130, A40 48 GB. The vLLM process is isolated in
`/workspace/vllm_env`; the DeepSpeed environment is untouched.

Same input contract as `bench_continuous_graph.py`: first LongBench prompt, chat template,
truncation to 6144 tokens, batch 1, chunk budget 512, graph capture, greedy-off sampling
(`temperature=0.7`, `top_p=0.9`, seed 0), 16 generated tokens, `ignore_eos=True`.

vLLM's KV pool was capped at **1.1 GiB** to approximate the DeepSpeed paged pool for this
single 6k request. The default vLLM pool (24.97 GiB at `gpu_memory_utilization=0.75`) gave
965.1 ms, so pool size did not materially change this single-request result.

| Workload | Engine | Wall ms | Tokens/s | Response SHA |
|---|---|---:|---:|---|
| 6k, 16 new | DeepSpeed graph + chunk 512 + paged prefill/decode + Gumbel | 1234.2 | 12.97 | `d4a324e9…` |
| 6k, 16 new | vLLM torch.compile + CUDA graphs + chunk 512 + paged | **954.8** | **16.76** | `d4a324e9…` |
| 20k, 8 new, greedy | DeepSpeed graph + chunk 512 + paged prefill/decode | 3871.2 | 2.07 | `c4291b80…` |
| 20k, 8 new, greedy | vLLM torch.compile + CUDA graphs + chunk 512 + paged | **3363.6** | **2.38** | `c4291b80…` |
| 20k, 8 new, T=0.7 | DeepSpeed | 3878.5 | 2.06 | `37a86029…` |
| 20k, 8 new, T=0.7 | vLLM | **3363.1** | **2.38** | `c4291b80…` |

- **vLLM is 1.29x faster end-to-end at 6k and 1.15x faster at 20k**; DeepSpeed is 29% / 15%
  slower respectively.
- Both engines produce the **same 16 response tokens at 6k** (`d4a324e9…`) and the **same
  8 greedy tokens at 20k** (`c4291b80…`), despite different attention/sampling implementations.
  At 20k with `T=0.7`, the first four sampled tokens agree and the fifth diverges
  (DeepSpeed 773, vLLM 1077); vLLM's sample matched its own greedy run, so this is a
  sampler/numeric boundary difference, not a cache-correctness mismatch.
- Component split: vLLM prefill+first token is 695.1 ms and 16 tokens take 954.8 ms, so its
  decode is about 17.3 ms/token. DeepSpeed's graph/paged profile is 822.7 ms prefill and
  389.3 ms for 16 decode tokens (24.3 ms/token). vLLM is ~1.2x faster in prefill and ~1.4x
  faster in decode.
- The remaining gap is consistent with vLLM's compiled/fused attention and sampler path; it is
  much smaller than the gap measured before paged prefill (dense-staging four-way was 1809.2 ms,
  where vLLM is 1.90x faster).
- Memory is not directly comparable: vLLM's engine runs in a subprocess (main-process peak reads
  0); its logs report 7.58 GiB weights plus the configured KV pool. DeepSpeed's in-process peak is
  9.53 GiB.

## What this means

Our optimization stack works: the paged four-way path is 1.47x faster end-to-end than the
dense-staging four-way path, and it is now within **1.29x at 6k** and **1.15x at 20k** of vLLM.
The remaining gap is mostly the fused prefill/attention and sampler path, not the paged cache
layout; the next meaningful work is closing those kernels, not another cache rewrite.

## Commands

```bash
cd /workspace/DeepSpeedExamples/training/opsd
export HF_HOME=/dev/shm/hf_cache
export VLLM_CACHE_ROOT=/dev/shm/vllm_cache
export TORCHINDUCTOR_CACHE_DIR=/dev/shm/torchinductor
export TMPDIR=/dev/shm

/workspace/vllm_env/bin/python benchmarks/bench_vllm_offline.py \
  --model Qwen/Qwen3-4B --data data/longbench_20k.jsonl --batch-size 1 \
  --max-prompt-length 6144 --max-new-tokens 16 --prefill-chunk-size 512 \
  --temperature 0.7 --top-p 0.9 --seed 0 --warmup 1 --iterations 1 \
  --gpu-memory-utilization 0.75 --kv-cache-memory-gib 1.1 \
  --output runs/vllm_6k_kv1g.json
```

Artifacts: `runs/vllm_6k_kv1g.json`, `runs/vllm_6k.json`, `runs/vllm_6k_new1.json`,
`runs/vllm_20k.json`, `runs/vllm_20k_greedy.json`,
`runs/fourway_paged_prefill_6k.json`, `runs/fourway_current_best_6k.json`,
`runs/fourway_paged_prefill_20k.json`, `runs/fourway_paged_prefill_20k_greedy.json`.
