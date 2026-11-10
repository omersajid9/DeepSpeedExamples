# vLLM parity optimization plan

This is the forward-looking optimization plan. The measured attribution of what has
already been implemented is in `runs/optimization_ablation.md`.

Implemented stack: CUDA graph capture, paged decode, paged prefill, prefix reuse, and
chunked prefill. The best no-chunk path is **1.37x behind vLLM at 6k** and **1.22x behind
at 20k**; paged decode is the strongest implemented win and paged prefill is a memory win.
Chunked prefill is slower on both engines for these single prompts, so it is not the parity
lever.

## Current gap

Current best is 1.37x behind vLLM at 6k and 1.22x behind at 20k. The paged cache layout is
not the main remaining gap: paged decode is already the strongest win, and paged prefill is
time-neutral with a memory advantage. The remaining gap is in the model forward kernels and
the surrounding execution path.

### Gap decomposition

vLLM no-chunk prefill+first-token and decode timings (same KV budgets):

| Workload | vLLM prefill+1 | vLLM decode | DeepSpeed prefill | DeepSpeed decode | Prefill gap | Decode gap |
|---|---:|---:|---:|---:|---:|---:|
| 4B 6k | 551.2 ms | 18.2 ms/token | 719.4 ms | 24.4 ms/token | **168 ms (59%)** | 117 ms (41%) |
| 4B 20k | 2647.9 ms | 20.9 ms/token | 3071.1 ms | 33.9 ms/token | **423 ms (77%)** | 125 ms (23%) |

The prefill is the larger absolute gap, especially at 20k; decode is the larger per-token
gap. A decode-only optimization cannot close 20k.

### 1. Fuse/compile the model forward (highest leverage)

vLLM uses torch.compile, fused RMSNorm/RoPE/residual/MLP kernels, and an optimized attention
backend. The DeepSpeed path still executes the HuggingFace Qwen3 layer stack and a Python
attention shim per layer. The decode roofline for 4B weights on an A40 is roughly 11.5 ms/token;
our 6k decode is 24.4 ms/token and 20k is 34.2 ms/token, while vLLM is materially closer to the
roofline. Work items:

- `torch.compile` the Qwen3 decoder with static shapes, then capture/replay the compiled graph.
- Fuse RMSNorm + residual, rotary embedding, and the gate/up/SiLU/down MLP path.
- Use the same fused kernels for prefill and decode, not only decode.
- Keep the paged cache/attention as the memory interface; compile the surrounding layers.

This is the most likely route to close most of the 1.2-1.4x gap.

### 2. Replace the attention path with a vLLM-grade backend

Our `BlockedFlashAttn` is good on decode (1.5-2.5x over dense SDPA) but prefill still runs a
blocked kernel and a Python atom/shim path. For parity:

- Use FlashAttention/FlashInfer for dense prefill and, if possible, for paged decode.
- Batch atoms once per step and remove per-layer Python work from the hot path.
- Capture the prefill forward in a graph, not only decode; our current paged prefill is eager.
- Compare `BlockedFlashAttn` against FlashInfer at 6k/20k before replacing it; the decode
  kernel may already be competitive, while the prefill kernel is the weaker side.

### 3. Make no-chunk the default for single-request rollouts

Chunked prefill is slower on both engines for these prefill-bound single prompts. Our paged
chunk penalty is +9.3% at 6k and +13.1% at 20k; vLLM's is +15.9%/+20.4%. Keep chunking for
memory/interleaving, but do not use it as the default parity configuration. The default should
be one paged prefill forward plus graph-captured decode.

### 4. Persistent prefix caching (agentic workloads)

Our prefix reuse is live-row-only: no cross-generate retention, no `prefix_id` session, no
weight-version salt, and no graph-safe path. vLLM has automatic prefix caching. To match vLLM
on repeated system prompts and multi-turn sessions:

- Retain prefix blocks after the source row retires, with eviction/refcounts.
- Add a session/prefix id and a weight-version salt.
- Support prefix reuse with the graph-safe cache.
- Measure prefill tokens served / total prefill; the current duplicate-prefix result is
  1.64x prefill and 1.44x total, but only while both rows are live.

### 5. Dynamic block pool and scheduler

The fixed per-row block table and full-batch ReTool matrix show no raw continuous-batching
win. vLLM's scheduler/block pool is dynamic and handles arrival/departure and mixed prefill+
decode. For serving-style and multi-request parity:

- Dynamic block allocation with eviction instead of a fixed per-row capacity.
- Mixed prefill+decode in one forward (our paged prefill is one request at a time).
- Remove the continuous-batching overhead visible in the ReTool matrix; use vLLM-style
  scheduling or keep the single-request fast path.

### 6. Sampler and small overheads

The device Gumbel sampler costs only ~0.1% in the eager ablation, so it is not the gap. vLLM's
fused top-p/top-k sampler and its compiled graph are still faster; fold sampling into the
compiled decode graph rather than treating it as a separate step.

### 7. Pragmatic parity option

If the requirement is vLLM-level performance rather than a DeepSpeed-native kernel stack, the
shortest path is to use vLLM as the rollout generation backend behind the existing rollout API
and keep DeepSpeed for training/ZeRO. Porting the kernels is the more maintainable long-term
option, but it is a larger project than the remaining cache work.

### What not to spend time on

- More paged-cache layout work: decode is already faster than dense and the paged path is
  within 1.2-1.4x of vLLM.
- Chunked prefill tuning as a parity lever: both engines are slower when chunking this workload.
- Bucket graph widths for batch 1: the ladder does not help at width 16 and costs memory; it is
  a multi-slot/long-context feature.
- Prefix reuse memory reduction with the current fixed pool: it saves prefill compute, not pool
  capacity.

## References

- Ablation study: `runs/optimization_ablation.md`
- vLLM measurements: `runs/vllm_comparison.md`
- Source artifacts: `runs/vllm_6k_nochunk_new1.json`,
  `runs/vllm_20k_nochunk_new1.json`, `runs/ablation_6k_*_nochunk.json`,
  `runs/ablation_20k_*_nochunk.json`
