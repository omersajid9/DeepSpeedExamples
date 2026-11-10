# OPSD rollout optimization sweep — plan

Status: **draft for approval.** No model-scale sweep jobs have been run. Only a
tiny random-weight GPU probe was run to check cache compatibility (section 3).

Scope: measure the rollout cost of the bucket CUDA-graph capture and chunked
prefill on four models and four workloads, with one config that fits on the
single A40 for every cell.

- Models: `Qwen/Qwen3-4B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3.5-9B`
- Workloads: LongBench, ReTool, 10k prompt, 20k prompt
- Arms: baseline (optimizations off) and optimized (graph ladder + chunked prefill on)
- Matrix: 4 models x 4 workloads x 2 arms = **32 runs**
- Runner: the existing rollout-only `benchmarks/bench_continuous_graph.py`
  (not the training step; teacher/optimizer noise would obscure the comparison)

Environment: A40 46,068 MiB, driver 595.99.02, torch 2.14.1+cu126,
transformers 5.22.2, DeepSpeed `c9c92c3d`, DeepSpeedExamples `2db591c`.

---

## 0. Bottom line

Two issues must be resolved before the 32-cell matrix is valid:

1. **Qwen3.5 cannot run the continuous or graph paths today.** `Qwen3.5-4B/9B`
   are hybrid Gated-DeltaNet + full-attention models. Their linear-attention
   layers call `get_conv_state` / `update_recurrent_state` on the cache.
   `DeepSpeedStaticCache` implements only KV cache and raises
   `AttributeError: 'DeepSpeedStaticCache' object has no attribute 'get_conv_state'`.
   The baseline continuous arm fails too, so this is not just an optimized-arm
   problem. Section 3 has the probe. The sweep either needs hybrid-cache support
   added first, or Qwen3.5 has to be dropped from this sweep.
2. **Disk is short.** `/workspace` has 14 GB free; the four bf16 checkpoints are
   ~53 GB. Use `/dev/shm/hf_cache` (250 GB tmpfs, 2 TB RAM) for this session, or
   free space another way.

Recommended frozen sweep config, pending the probes in section 4:

```
batch-size            16
continuous-batch-size 3
prefill-chunk-size    512
graph-capture-ladder  1,2,3
max-new-tokens        256 (LongBench) / 512 (ReTool) / 32 (10k, 20k)
max-prompt-length     8192 (LongBench) / 2048 (ReTool) / 10240 (10k) / 20480 (20k)
```

Estimated optimizer-arm peak at 20k on A40: Qwen3-4B ~26 GB, Qwen3-8B ~34.5 GB,
Qwen3.5-4B ~13 GB, Qwen3.5-9B ~23 GB. The binding cell is Qwen3-8B; `cont=3`
leaves roughly 11 GB headroom. `cont=4` is also plausible (~40.6 GB for Qwen3-8B,
~5.5 GB headroom), but it is not necessary for a valid sweep and is the cell most
likely to OOM. Section 4 keeps `cont=3` unless the probe shows `cont=4` is safe.

---

## 1. What the existing runs tell us

Anchors from `DeepSpeedExamples/training/opsd/runs/` (all Qwen3-4B, A40):

| Artifact | Workload | batch | cont | graph | chunk | Wall | Prefill | Decode | Peak |
|---|---|---:|---:|---|---|---:|---:|---:|---:|
| `new_bs16_naive_cont2.json` | 20k | 16 | 2 | on | off | 134.2 s | 96.4 s | 36.5 s | 23.6 GB |
| `new_bs16_chunked_cont4.json` | 20k | 16 | 4 | on | on | 197.0 s | 156.8 s | 35.3 s | 32.2 GB |
| `qwen3_4b_longbench_continuous_graph.json` | LongBench | 2 | 1 | on | off | 18.2 s | 1.85 s | 16.15 s | 10.0 GB |
| `qwen3_4b_longbench_chunked_prefill.json` | LongBench | 2 | 1 | on | on | 18.0 s | 1.64 s | 16.15 s | 9.6 GB |
| `qwen3_4b_retool_continuous_graph.json` | ReTool | 4 | 2 | on | off | 24.8 s | 0.17 s | 24.1 s | 8.1 GB |
| `qwen3_4b_retool_chunked_prefill.json` | ReTool | 4 | 2 | on | on | 24.8 s | 0.17 s | 24.1 s | 8.1 GB |

These are **anchors, not the new baseline**: both arms in the old artifacts had
graph capture on. The new baseline turns both graph capture and chunking off.

Key memory fact from `new_bs16_chunked_cont4.json`:

- `cache_memory_bytes` = 24.18 GB, `chunked_prefill_staging_bytes` = 12.08 GB.
- At cont=4, chunked prefill holds a main KV cache **and** a staging KV cache,
  so KV memory doubles. The 32.2 GB peak is essentially model weights (~8.0 GB)
  plus those 24.18 GB. Graph capture itself adds little on top.
- Therefore **peak memory scales with `continuous_batch_size` (slots), not with
  `batch_size` (queue depth)**. `batch-size 16` is safe; it only stores request
  metadata and prompt IDs.

At cont=8, the old-code JSON (`qwen3_4b_20k_bs16_chunked_prefill.json`) shows
34.1 GB, but its stats have no graph ladder and only one staging slot
(`chunked_prefill_staging_bytes` = 3.02 GB), so it is not the new bucket-graph
path. The old notes report the new-code cont=8 run as OOM. Treat cont=8 as
unsafe and out of scope.

### Why cont=3 is the recommendation

At 20,480 tokens, per-slot KV is:

- Qwen3 full attention: `2 * 36 layers * 8 kv_heads * 128 head_dim * 2 B`
  = 147,456 B/token = 144 KiB/token, i.e. **3.02 GB per slot**.
- Qwen3.5 hybrid: only 8 of 32 layers are full attention, 4 kv heads,
  head_dim 256: **0.671 GB per slot**. The linear-attention conv + recurrent
  state adds ~27 MB/slot, negligible here.

Chunked prefill doubles that (main + staging). With `C` slots, 20k KV cost is
`2 * C * 3.02 GB` for Qwen3 and `2 * C * 0.671 GB` for Qwen3.5.

| Model | Weights (bf16) | 20k KV, C=2 | C=3 | C=4 | Est. opt peak C=3 | Est. C=4 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | ~8.0 GB | 12.1 GB | 18.1 GB | 24.2 GB | ~26 GB | 32.2 GB (measured) |
| Qwen3-8B | ~16.4 GB | 12.1 GB | 18.1 GB | 24.2 GB | **~34.5 GB** | ~40.6 GB |
| Qwen3.5-4B | ~9.3 GB | 2.7 GB | 4.0 GB | 5.4 GB | ~13.5 GB | ~15 GB |
| Qwen3.5-9B | ~19.3 GB | 2.7 GB | 4.0 GB | 5.4 GB | ~23.5 GB | ~25 GB |

Add ~1-2 GB for CUDA context, activations, and allocator fragmentation. At
`C=3`, every cell has at least ~10 GB headroom. At `C=4`, Qwen3-8B has ~4-5 GB
and is the only risky cell. Since the sweep must complete for all four models
with one config, `C=3` is the safe default; `C=4` stays optional and only if
section 4's probe confirms it.

---

## 2. Sweep matrix and workload definitions

All runs use the same first 16 prompts per workload, greedy decoding, and
`n_samples_per_prompt=1`. The prompt text is model-independent; each model's own
chat template and tokenizer are applied.

| Workload | Data | batch | max prompt | max new | Rationale |
|---|---|---:|---:|---:|---|
| LongBench | `data/longbench_mfqa.jsonl` | 16 | 8192 | 256 | long-context prefill + decode; first 16 real widths 1,708-9,239, truncated to 8,192 |
| ReTool | `data/retool_prompts.jsonl` | 16 | 2048 | 512 | decode-heavy; first 16 real widths 197-462, no truncation of the batch |
| 10k prompt | `data/longbench_20k.jsonl` | 16 | 10240 | 32 | prefill stress; first 16 real widths 17,315-29,548, truncated to 10,240 |
| 20k prompt | `data/longbench_20k.jsonl` | 16 | 20480 | 32 | prefill stress; same prompts, padded/truncated to 20,480 |

Notes:

- The 10k and 20k cells are the same prompts at two truncation lengths. That is
  intentional: it isolates prompt length from prompt content.
- `longbench_20k.jsonl` is a repeated-context stress set, not a natural
  benchmark. Report it as such.
- 20k `max_new_tokens=32` matches the existing anchors. LongBench/ReTool cover
  the decode-heavy regime.

---

## 3. Model/topology findings that change the plan

### 3.1 Qwen3.5 is a hybrid architecture, not Qwen3 with different weights

`Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B` are `Qwen3_5ForConditionalGeneration`
(image-text-to-text), text backbone `Qwen3_5TextConfig`:

- 32 layers, layer pattern `[linear, linear, linear, full] x 8`.
- Only 8 full-attention layers; 24 Gated-DeltaNet linear-attention layers.
- Full attention: 4 KV heads, head_dim 256, max positions 262,144.
- 4B: hidden 2,560; 9B: hidden 4,096; both bf16.

Consequences:

1. **`DeepSpeedStaticCache` fails immediately.** Its linear layers call
   `cache_params.get_conv_state`, `get_recurrent_state`, `update_conv_state`,
   `update_recurrent_state`, `is_recording_past`; the current class has none of
   those methods. Probe with a tiny random `Qwen3_5ForCausalLM` on GPU:
   `StaticCache` works, `DeepSpeedStaticCache` raises
   `AttributeError ... has no attribute 'get_conv_state'`. This blocks both the
   baseline continuous arm and the optimized arm.
2. **`AutoModelForCausalLM` is the wrong loader for the released checkpoint.**
   The checkpoint keys are `model.language_model.*` and it contains a vision
   tower; `Qwen3_5ForCausalLM` expects `model.layers.*`. Loading the multimodal
   repo through `AutoModelForCausalLM` would leave the language model randomly
   initialized while reporting only ignored/missing keys. The plan uses
   `Qwen3_5ForConditionalGeneration`, extracts `model.model.language_model` plus
   `lm_head`, deletes the vision tower, and wraps it as a text-only rollout
   module (`config = text_config`, `_supports_cache_class = True`,
   `generation_config`, `forward(..., logits_to_keep=1)`).
3. **`causal-conv1d` and `flash-linear-attention` are not installed.** Both
   probes fall back to reference PyTorch kernels with a warning. The A/B is still
   fair because both arms use the same kernels, but absolute Qwen3.5 throughput
   will be poor and 10k/20k runtimes may be unacceptable. Install both before the
   sweep if possible; record the kernel mode in every artifact either way.
4. **Graph capture of recurrent state needs extra care.** Unlike KV, the linear
   state is advanced by the forward. The 3 warmup forwards plus capture would
   leave the state advanced before the first replay. The implementation must
   snapshot linear state before capture and restore it after, and must reset
   per-row linear state when a request is admitted into a freed slot. Chunked
   prefill also needs to copy the linear state from the staging row to the
   decode slot, and the graph prefill bucket needs a 2D linear-attention mask
   (the existing 4D causal mask is ignored by `create_recurrent_attention_mask`).

### 3.2 Disk

`/workspace` is 32 GB with 14 GB free. Download sizes: Qwen3-8B 16.4 GB,
Qwen3.5-4B 9.3 GB, Qwen3.5-9B 19.3 GB; Qwen3-4B is already cached. Use
`/dev/shm/hf_cache` (250 GB) for this session. Keep all run artifacts in
`/workspace/DeepSpeedExamples/training/opsd/runs/`; `/dev/shm` is ephemeral.

### 3.3 Existing profiling field is double-counted

In both `_generate_continuous` and `_generate_continuous_graph`,
`cache_management_overhead_ms` starts before the prefill forward and ends after
it, so it includes `prefill_forward_ms`. Example from
`new_bs16_chunked_cont4.json`: `cache_management_overhead_ms=160.8 s` while
`prefill_forward_ms=156.8 s`. The same overlap appears in the LongBench table
(1,652 ms vs 1,640 ms). The wall/prefill/decode numbers are unaffected, but
"cache overhead" from old reports is not an overhead metric. Fix the timer scope
before this sweep, or use `generation_overhead_ms` instead.

---

## 4. Preflight gates and OOM probe

Do these in order. Do not start the 32-run matrix until every gate passes.

| Gate | Check | Pass condition | If it fails |
|---|---|---|---|
| G0 | GPU idle, disk/cache ready, model revisions pinned | `nvidia-smi` shows no other process; `HF_HOME=/dev/shm/hf_cache`; all four configs/tokenizers load | Fix environment; do not run |
| G1 | Qwen3 loader + graph path | Qwen3-4B short prompt: baseline and optimized both exit 0; optimized `graph_captures > 0`; response IDs match | Fix loader/engine |
| G2 | Qwen3.5 hybrid cache | Tiny hybrid config passes `DeepSpeedStaticCache`; real Qwen3.5-4B short prompt: baseline and optimized exit 0; response IDs match; optimized captures >0 | **Stop.** Implement hybrid support or drop Qwen3.5 from the sweep |
| G3 | Qwen3.5 chunked prefill | Chunked vs naive same prompt produces identical response IDs | Fix state transfer/masking |
| G4 | 20k memory probe | Qwen3-8B and Qwen3.5-9B, 20k, optimized arm, `C` in `{4,3,2}`; peak < 43,000 MiB (>=3 GB headroom) | Drop to the next lower C; never shrink mid-sweep |

### G4 command shape

Run each probe in a fresh process; do not reuse an allocator after an OOM.

```bash
python benchmarks/bench_continuous_graph.py \
  --model Qwen/Qwen3-8B \
  --data data/longbench_20k.jsonl \
  --batch-size 16 --continuous-batch-size 4 \
  --max-prompt-length 20480 --max-new-tokens 32 \
  --warmup 0 --iterations 1 \
  --graph-capture --graph-capture-batch-sizes 1,2,3,4 \
  --enable-chunked-prefill --prefill-chunk-size 512 \
  --output runs/sweep_probe/qwen3-8b_20k_c4.json
```

Repeat for Qwen3.5-9B and for C=3, C=2. Qwen3-4B C=4 is already measured at
32.2 GB, so it is not the binding probe. Freeze the largest C that passes both
models with >=3 GB headroom; record it in `sweep_manifest.json`. The expected
answer is C=3; C=2 is the fallback.

---

## 5. Arms and pair identity

The only differences between the two arms are the two optimizations. Everything
else is byte-identical: same data, same batch topology, same lengths, same
sampling, same warmup/iteration count, same process environment.

| Knob | Baseline | Optimized |
|---|---|---|
| `use_graph_capture` | `False` | `True` |
| `graph_capture_batch_sizes` | n/a | `(1,2,3)` from frozen C |
| `enable_chunked_prefill` | `False` | `True` |
| `prefill_chunk_size` | 512 (inactive) | 512 |
| `continuous_batch_size` | 3 | 3 |
| `batch_size` | 16 | 16 |
| `max_prompt_length` | workload value | same |
| `max_new_tokens` | workload value | same |
| `temperature` / `top_p` / `top_k` | 0.0 / 1.0 / -1 | same |
| `n_samples_per_prompt` | 1 | same |
| data / seed / warmup / iterations | workload value | same |

Naming:

- `baseline`: continuous batching, eager forward, no chunked prefill.
- `optimized`: continuous batching, bucket graph capture, chunked prefill.

If the intent was "baseline = single graph, no chunk" rather than "no graph at
all", swap the baseline to `use_graph_capture=True` with
`graph_capture_batch_sizes=(C,)`; the runner supports both. The plan's default
is the unambiguous "both optimizations off" baseline. An optional third
`single_graph` arm (graph on, one bucket, chunk off) can be run on Qwen3-4B only
for attribution without touching the 32-cell matrix.

---

## 6. Code changes required before the sweep

These are additive; existing benchmark fields keep their names.

### A. `benchmarks/bench_continuous_graph.py`

- `--graph-capture / --no-graph-capture` (default on), instead of the hardcoded
  `use_graph_capture=True`.
- `--graph-capture-batch-sizes 1,2,3`, passed to `HybridEngineRolloutConfig`.
- `--model-backend auto|qwen3_5_text`; auto-detect `config.model_type == "qwen3_5"`
  and use the conditional-model wrapper.
- `--save-response-ids`: store each request's generated token IDs; also store a
  SHA-256 digest so baseline/optimized can be compared without loading both
  files.
- Record `graph_capture`, `graph_capture_batch_sizes`, `enable_chunked_prefill`,
  `prefill_chunk_size` in the result JSON (the current `graph_capture` field is
  hardcoded `True`).
- Record `warmup_peak_memory_mb` and `warmup_stats` (graph captures, bucket
  widths, chunk count) before the post-warmup reset, plus `peak_memory_mb` and
  `peak_reserved_mb` for measured iterations. The true memory budget is the max
  of warmup and measured peaks.
- Fail the run if an optimized measured iteration reports `graph_captures > 0`
  or any bucket was disabled; a silent eager fallback is not an optimized run.

### B. `deepspeed/utils/static_cache.py`

- Build per-layer cache objects from `text_config.layer_types`: full-attention
  layers use the existing `DeepSpeedStaticLayer`; linear-attention layers use a
  static linear layer with preallocated conv and recurrent state (the HF
  `LinearAttentionLayer` already marks static addresses and updates in place, or
  an equivalent local class).
- Implement the cache-level dispatch: `update`, `get_seq_length`,
  `get_mask_sizes`, `is_sliding`, `get/update_conv_state`,
  `get/update_recurrent_state`, `is_recording_past`, `reset`, `compact`,
  `trim_left`, and cache-byte accounting for both state kinds.
- Add `reset_rows(slots)` to zero conv/recurrent state when a request is admitted
  into a freed slot. KV rows are overwritten by prefill; linear state is not.
- Keep static tensor addresses and in-place state copies; no reassignment during
  replay.

### C. `deepspeed/runtime/rollout/hybrid_engine_rollout.py`

- Call `cache.reset_rows(...)` on admission for both eager and graph paths.
- Extend `_publish_staged_prompt`, `_publish_prefill_chunk`, and
  `_continuous_prefill_cache` copies to move linear state as well as KV.
- Add a hybrid mask path for graph chunked prefill: pass a dict
  `{"full_attention": 4D_causal, "linear_attention": 2D_chunk_validity}` to the
  model, because `create_recurrent_attention_mask` ignores a 4D mask.
- Snapshot/restore linear state around bucket capture warmup; otherwise capture
  itself advances the recurrent state before the first replay.
- Expose `graph_capture_failures`, disabled bucket widths, and the capture ladder
  in `continuous_stats`.
- Fix the cache-management timer scope (section 3.3), or add non-overlapping
  `staging_copy_ms` / `cache_compaction_ms` timers.

### D. Tests

- Tiny `Qwen3_5TextConfig` parity: prefill, decode, and chunked prefill logits
  match HF `StaticCache`/`DynamicCache` within bf16 tolerance.
- Real `Qwen3.5-4B` short prompt: baseline vs optimized response IDs match;
  chunked vs naive match.
- Graph-capture gate: Qwen3.5 optimized run reports `graph_captures > 0` and no
  disabled bucket.
- Existing Qwen3 rollout unit tests still pass.

If Qwen3.5 is dropped from this sweep, B/C/D's hybrid-specific parts are
deferred and the 16-cell (2 models x 4 workloads x 2 arms) matrix can start
immediately.

---

## 7. Profiling schema and outputs

Keep the existing `profile` and `continuous_stats` keys; add derived fields at
aggregation time. Per iteration, record:

| Field | Source | Meaning |
|---|---|---|
| `wall_ms` | benchmark timer | end-to-end generate + device sync |
| `profile.prefill_forward_ms` | engine | sum of prefill forward time |
| `profile.decode_forward_ms` | engine | sum of decode forward time |
| `profile.generation_overhead_ms` | engine | generation wall minus prefill/decode; scheduler/post-processing |
| `profile.num_prefill_forwards`, `num_decode_forwards` | engine | forward counts |
| `profile.active_batch_size`, `continuous_batch_size` | engine | occupancy |
| `profile.tokens_per_second` | engine | generated tokens / total (not prompt throughput) |
| `continuous_stats.cache_memory_bytes` | engine | main cache bytes |
| `continuous_stats.chunked_prefill_staging_bytes` | engine | staging cache bytes |
| `continuous_stats.graph_captures` | engine | captures in this iteration; 0 after warmup |
| `continuous_stats.decode_bucket_widths`, `prefill_bucket_widths` | engine | which graph buckets ran |
| `continuous_stats.decode_padded_rows`, `prefill_padded_rows` | engine | bucket padding cost |
| `continuous_stats.chunked_prefill_chunks` | engine | number of chunk forwards |
| `peak_memory_mb`, `peak_reserved_mb` | benchmark | allocated/reserved peaks |
| `response_sha256` | benchmark | token-match check |

Derived per run (median over measured iterations):

```
prefill_tps = batch_size * prompt_length / (prefill_forward_ms / 1000)
decode_tps  = num_generated_tokens / (decode_forward_ms / 1000)
wall_tps    = num_generated_tokens / (wall_ms / 1000)
```

`prefill_tps` uses padded prompt tokens, which is the work the model actually
does. `decode_tps` isolates decode and is the right metric for ReTool.
`wall_tps` is the user-visible number.

Also record per run: model revision, tokenizer class, torch/transformers/
DeepSpeed versions and git commits, GPU name/driver, `nvidia-smi` clocks if
available, `HF_HOME`, kernel mode for Qwen3.5 (`causal-conv1d` /
`flash-linear-attention` vs reference fallback), and the exact command line.

---

## 8. Execution order

1. **G0** environment: stage models in `/dev/shm/hf_cache`, install Qwen3.5
   kernels, check GPU idle.
2. **G1** Qwen3-4B short-prompt smoke: baseline, optimized, token match.
3. **G2/G3** Qwen3.5-4B short-prompt smoke: loader, hybrid cache, graph capture,
   chunked token match. Stop here if the hybrid work is not done.
4. **G4** memory probe on Qwen3-8B and Qwen3.5-9B at 20k, optimized, C=4 -> 3 -> 2.
   Freeze C in the manifest.
5. **Full matrix**, model-major, 20k first (most likely OOM), then 10k, LongBench,
   ReTool. For each workload, run baseline then optimized back-to-back so thermal
   and clock state are as close as possible.
6. **Token match** every pair; **aggregate** the tables.

Runner behavior:

- One fresh subprocess per run; no allocator reuse across runs.
- Resume from existing valid JSON; verify the manifest's config hash matches.
- Stop on OOM, token mismatch, `graph_captures == 0` in an optimized run, or a
  process hang. Do not silently lower C or change lengths mid-sweep.
- Write `sweep_manifest.json` after every run with status and peak memory.

Warmup/iterations:

- LongBench/ReTool: `warmup=1`, `iterations=3`.
- 10k/20k: `warmup=1`, `iterations=2` (existing 20k used 2).
- If any measured iteration reports `graph_captures > 0`, rerun with
  `warmup=2` before accepting the result.

---

## 9. Acceptance criteria

A sweep cell is valid only if all of these hold:

- Process exit 0, no CUDA OOM, no NaN, no hang.
- `peak_memory_mb` (max of warmup/measured) < 43,000 MiB.
- Optimized arm: `graph_captures > 0` during warmup and `0` during measured
  iterations; no disabled buckets; expected bucket widths exercised.
- Baseline arm: `graph_captures` missing or `0`; no chunked prefill stats.
- Baseline and optimized response token IDs match exactly for every request.
- Same frozen C, batch, lengths, data, and kernels across the pair.
- At least two measured iterations; report median and min/max. If coefficient of
  variation exceeds 5%, rerun the cell.

The overall sweep is valid only if all 32 cells pass. If Qwen3.5 is dropped by
explicit decision, the matrix becomes 16 cells (2 models x 4 workloads x 2 arms)
and the same criteria apply.

---

## 10. Deliverables

Directory: `DeepSpeedExamples/training/opsd/runs/sweep_2026-11-10/`

```
raw/<model>__<workload>__<arm>.json     # full per-iteration profile + stats
logs/<model>__<workload>__<arm>.log     # stdout/stderr
sweep_manifest.json                     # config hash, status, peak, versions
sweep_summary.csv                       # one row per cell
sweep_summary.md                         # tables + interpretation
token_match.json                        # per-pair response digests
```

Summary tables:

1. **Latency/throughput**: model x workload x arm with wall, prefill, decode,
   prefill/decode/wall tok/s, forward counts.
2. **Memory**: model x workload x arm with peak allocated, peak reserved, cache
   bytes, staging bytes, headroom.
3. **Optimization delta**: model x workload with wall/prefill/decode/tok-s
   percent change and peak-memory percent change.
4. **Activity**: graph captures, bucket widths, padded rows, chunk count, mean
   active batch size.
5. **Validation**: token match, OOM status, kernel mode, notes.

Report the Qwen3.5 hybrid architecture caveat and the C choice at the top of the
summary. Do not compare absolute tok/s across models; compare baseline vs
optimized within a model/workload, and use the same C everywhere.

---

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Qwen3.5 hybrid cache not implemented | Cannot run 2 of 4 models | Gate G2; implement B/C/D or drop Qwen3.5 by explicit decision |
| Qwen3.5 graph capture of recurrent kernels fails | Optimized arm silently eager | Fail on `graph_captures == 0` / disabled buckets; do not report as optimized |
| Qwen3.5 reference kernels | Very slow 10k/20k, possibly hours | Install `causal-conv1d` + `flash-linear-attention`; record kernel mode |
| Qwen3-8B OOM at C=4 | Missing cell | Use C=3; probe before freezing |
| Disk exhaustion on `/workspace` | Model load failure | Use `/dev/shm/hf_cache`; keep artifacts on `/workspace` |
| Profiling timer overlap | Misleading "cache overhead" | Fix timer scope; use `generation_overhead_ms` |
| Thermal/clock drift | Noisy deltas | Baseline/optimized back-to-back; median of 2-3; rerun if CV > 5% |
| Silent eager fallback in graph buckets | Invalid speedup | Expose capture failures; hard-fail the cell |
| Different tokenizer templates across models | Absolute numbers not comparable | Compare within model only; report tokenizer/class |

---

## 12. Decisions requested

Please reply with these choices; I will not start the sweep until they are
settled.

1. **Qwen3.5 scope.** (a) Implement hybrid static-cache/graph support first, so
   the sweep covers all four models (recommended if the 4-model comparison is
   required). (b) Drop Qwen3.5-4B/9B from this sweep and run Qwen3-4B/8B now.
   (c) Run Qwen3.5 eager-only as a non-comparable reference. Default: (a).
2. **Baseline definition.** (a) Both optimizations off: `use_graph_capture=False`
   + `enable_chunked_prefill=False` (plan default). (b) Single graph baseline:
   `use_graph_capture=True`, `graph_capture_batch_sizes=(C,)`, chunk off. Default:
   (a).
3. **Concurrency.** (a) Freeze C=3 now (recommended; ~11 GB headroom on Qwen3-8B).
   (b) Run the C=4 probe first and use C=4 if peak < 43,000 MiB. Default: (a).
4. **Qwen3.5 kernels.** (a) Install `causal-conv1d` + `flash-linear-attention`
   before the sweep (recommended for runtime). (b) Accept reference-kernel
   fallback and record it. Default: (a), with (b) as fallback if installation
   fails.
