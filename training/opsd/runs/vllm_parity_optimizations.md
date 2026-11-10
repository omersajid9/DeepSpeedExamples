# vLLM parity optimization plan

This is the forward-looking optimization plan. The measured attribution of what has
already been implemented is in `runs/optimization_ablation.md`.

Implemented stack: CUDA graph capture, paged decode, paged prefill, prefix reuse, chunked
prefill, and an opt-in `torch.compile` fusion. The best no-chunk path is **1.235x behind vLLM
at 6k** (compiled: 1018 vs 824 ms) and **1.148x behind at 20k** (compiled: 3208 vs 2794 ms). Paged decode
is the strongest implemented win; paged prefill is a memory win; chunked prefill is slower on
both engines for these single prompts.

## Current gap

Current best is 1.235x behind vLLM at 6k (compiled) and 1.148x behind at 20k (compiled). The paged cache
layout is not the main remaining gap: paged decode is already the strongest win, and paged
prefill is time-neutral with a memory advantage. The remaining gap is in the model forward
kernels and the surrounding execution path.

The optional `torch.compile` fusion delivers **1.11x total / 1.11x prefill / 1.12x decode**
at 6k (1129 -> 1018 ms), closing the vLLM gap from 1.37x to 1.235x at +14.8% peak memory.
At 20k it delivers **1.067x total** (3421 -> 3208 ms), closing 1.224x to 1.148x at +35.6%
peak memory. It is only useful with graph capture; 20k compiled peak memory is materially
higher.

### Gap decomposition

vLLM no-chunk prefill+first-token and decode timings (same KV budgets):

| Workload | vLLM prefill+1 | vLLM decode | DeepSpeed prefill | DeepSpeed decode | Prefill gap | Decode gap |
|---|---:|---:|---:|---:|---:|---:|
| 4B 6k, compiled | 551.2 ms | 18.2 ms/token | 649.9 ms | 21.8 ms/token | **99 ms (63%)** | 58 ms (37%) |
| 4B 20k, compiled | 2647.9 ms | 20.9 ms/token | 2933.0 ms | 31.9 ms/token | **285 ms (76%)** | 88 ms (24%) |

The prefill is the larger absolute gap, especially at 20k; decode is the larger per-token
gap. A decode-only optimization cannot close 20k.

### 1. Fuse/compile the model forward (partially implemented; highest leverage)

**Implemented:** optional `torch.compile(mode="max-autotune-no-cudagraphs")` with graph
capture, `--torch-compile`. It gives 1.11x total at 6k (1129 -> 1018 ms) and 1.067x at
20k (3421 -> 3208 ms); prefill/decode both improve. It closes the vLLM gap to 1.235x at
6k and 1.148x at 20k, at +14.8% / +35.6% peak memory.

**Remaining:** the compile still hits per-layer recompile limits in the paged-attention shim
and the blocked-flash custom op. Eliminate those graph breaks and fuse the custom attention
into the compiled graph; the current compile is a partial win, not full vLLM-grade fusion.

### Why the fusion is partial

`torch.compile` is applied **after** `install_paged_attention` and **before** the rollout's
external CUDA-graph capture, so the measured 1017.8/3208.0 ms rows already include paged
prefill, paged decode, the Gumbel sampler, and graph capture. The warning is not a conflict
with those optimizations.

- `_paged_qwen3_forward` uses `self.layer_idx` (a Python int) to index
  `past_key_values._key_views[layer_idx]`. Dynamo treats that int as static and specializes
  the graph per layer. With 36 layers and the default recompile limit of 8, it recompiles
  layers 0-7 and then falls back for the remaining layers.
- The blocked-flash kernel is a pybind11 extension (`flash_attn_by_atoms`) that Dynamo cannot
  trace; it is an opaque graph break regardless of the limit.
- The Gumbel sampler is outside the model forward and is unchanged; CUDA graph capture wraps
  the compiled model and remains active.

Tried fixes:

- `@torch._dynamo.disable` on the shim: no recompile warnings, but slower
  (1038.9 vs 1017.8 ms at 6k) because the attention path is then fully opaque.
- Raising the recompile limit: the environment variable did not take effect; a code-level
  limit increase was not committed because compiling 36 per-layer specializations adds
  compile time/memory and still leaves the blocked-flash op opaque.

To get full fusion, register the blocked-flash op as a PyTorch custom op
(`torch.library.custom_op` / `allow_in_graph`) and remove per-layer Python indexing, or port
the fused Qwen3 layer path from vLLM. The current opt-in compile is already a measured
partial win, not a replacement for the paged cache or graph path.

vLLM uses torch.compile, fused RMSNorm/RoPE/residual/MLP kernels, and an optimized attention
backend. The DeepSpeed path still executes the HuggingFace Qwen3 layer stack and a Python
attention shim per layer. The decode roofline for 4B weights on an A40 is roughly 11.5 ms/token;
our compiled 6k decode is 21.8 ms/token and compiled 20k is 31.9 ms/token, while vLLM is
materially closer to the roofline. Work items:

- `torch.compile` the Qwen3 decoder with static shapes, then capture/replay the compiled graph.
- Fuse RMSNorm + residual, rotary embedding, and the gate/up/SiLU/down MLP path.
- Use the same fused kernels for prefill and decode, not only decode.
- Keep the paged cache/attention as the memory interface; compile the surrounding layers.

This is the most likely route to close most of the 1.2-1.4x gap.

### 2. Replace the attention path with a vLLM-grade backend

**Implemented (prefill only):** optional `--flashinfer-prefill` uses FlashInfer's
`BatchPrefillWithPagedKVCacheWrapper` with the existing block table. It is neutral at 6k
(-0.4%) and gives **1.063x total at 20k** (3421.5 -> 3217.9 ms), token-exact, with similar
peak memory. Decode still uses `BlockedFlashAttn`, and the 6k/20k gap remains 1.378x/1.152x
to vLLM, so this is not yet a full replacement.


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
