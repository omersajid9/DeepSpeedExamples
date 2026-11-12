# A100 Profiling Study: vLLM vs Best-Optimized DeepSpeed Rollout

**Date:** 2026-11-11/12 · **Host:** gpua075.delta.ncsa.illinois.edu (Slurm 23821699, GPU 0)
**Model:** Qwen3-4B (36 layers, hidden 2560, 32 q-heads, 8 KV-heads, head_dim 128, intermediate 9728, bf16; 3.63B GEMM params)
**Workloads:** 6k = 6144-token prompt + 16 new tokens (T0.7/PP0.9); 20k = 20480 + 8 (greedy). Batch 1, continuous-batch 1, seed 0, 1 warmup + 1 measured iteration.
**Companion report:** `optimization_ablation_a100.md` (24-cell ablation incl. new-best cells, A40 parity).

**Engines profiled**

| | DeepSpeed "best" config | vLLM reference |
|---|---|---|
| What | paged cache + paged prefill + bucket CUDA graphs + torch.compile (`max-autotune-no-cudagraphs`) | no-chunk offline bench config; `enable_prefix_caching=False` (default OFF in profile driver) |
| Attention | FlashAttention-2 `flash_fwd_kernel` (same kernel prefill & decode) | FlashAttention-2 `flash_fwd_splitkv` + combine (prefill **and** decode) |
| Engine core | in-process | in-process for profiling (`VLLM_ENABLE_V1_MULTIPROCESSING=0`; see control below) |
| Token check | response SHA `d4a324e9` (6k) / `c4291b80` (20k) — identical to the ablation runs | same SHAs |

**Hardware rooflines (A100-SXM4-40GB, full 400 W, P0, no throttles):** bf16 dense 312 TFLOPS · HBM 1555 GB/s · 108 SMs.

---

## 1. Headline results

| Run | Wall (ms) | GPU kernel time (ms) | Prefill (ms) | Decode (ms) | Decode steps |
|---|---|---|---|---|---|
| **DS best 6k** | 658.6 (625.0 under nsys) | 586.3 | 302.6 | 283.7 + 54.2 gaps | 15 → **22.5 ms/step** |
| **vLLM 6k** | 400.8 mp / 398.7 sp (403.4 under nsys) | 399.1 | 264.4 | 134.7 + 5.9 gaps | 15 → **9.4 ms/step** |
| **DS best 20k** | 1615.6 (1619.1 under nsys) | 1561.3 | 1309.0 | 252.3 + 22.8 gaps | 7 → **39.3 ms/step** |
| **vLLM 20k** | 1296.1 mp / 1295.7 sp (1322.5 under nsys) | 1317.2 | 1244.2 | 73.0 + 2.7 gaps | 7 → **10.8 ms/step** |

**Where the time goes, in one line each:**

1. **DS decode attention is SM-starved, not bandwidth-bound.** `flash_fwd_kernel` runs with grid **(32,1,1) = 32 CTAs on 108 SMs** (one CTA per q-head, no sequence split) → 237 µs/layer (6k), 106 GB/s effective KV read = **~7% of the 1555 GB/s roofline**. vLLM's `flash_fwd_splitkv` runs grid **(1,25,8) = 200 CTAs**, 26 µs/layer ≈ **62% of roofline**. Same kernel family, ~9× per-layer cost.
2. **DS re-launches per-step GQA bookkeeping on GPU** (reshape/cat/index + bf16↔fp32 casts): 48.0 ms (6k) / 22.6 ms (20k) of decode elementwise vs ~0.6 ms in vLLM (precomputed C++ paged block tables, in-kernel access).
3. **DS pays 3.6 ms/step of GPU idle between kernels** (CPU-side step loop: sampler outside the bucket graph, paged-cache management) vs 0.4 ms/step in vLLM (full-step graphs).
4. **DS prefill pays 51 ms (6k) / 158 ms (20k) of elementwise/cast/gather overhead** vs ~0 in vLLM, and its GEMMs run at 76–79% of peak vs vLLM's 82–86%.
5. **DS prefill attention is already faster than vLLM's** (55.7 vs 66.1 ms @6k; 524 vs 620 ms @20k) — vLLM uses split-KV for prefill too, which is less efficient for long causal sequences. This is the one place DS wins.

---

## 2. Prefill analysis

Per-layer GEMM FLOPs: 2·M·3.63e9 → **6k: 44.7 TFLOP (143 ms roofline)**, **20k: 148.8 TFLOP (477 ms roofline)**. Causal attention: **6k: 11.1 TFLOP (35.6 ms)**, **20k: 124 TFLOP (397 ms)**.

| Prefill component | DS 6k | vLLM 6k | DS 20k | vLLM 20k |
|---|---|---|---|---|
| GEMM (triton fused + cuBLAS/cutlass) | 187.9 ms → **238 TFLOPS (76%)** | 175.5 ms → **254 (82%)** | 601.9 ms → **247 (79%)** | 552.6 ms → **269 (86%)** |
| Flash attention | 55.7 ms → **199 TFLOPS (64%)** | 66.1 ms → 168 (54%) | 524.3 ms → **236 (76%)** | 619.7 ms → 200 (64%) |
| Elementwise/cast/gather | **51.0 ms** | 0.0 ms | **158.2 ms** | 0.0 ms |
| Norm/act (unfused part) + reduce | 7.9 ms | 15.6 + 7.2 ms | 24.7 ms | 50.0 + 21.8 ms |
| **Total** | **302.6** | **264.4** | **1309.0** | **1244.2** |

- DS prefill GEMM gap to vLLM: −12 ms (6k), −49 ms (20k). Mostly tile/heuristic effects (vLLM issues 4 clean M×N GEMMs per layer; DS inductor fusions split some layers into extra transposed GEMMs, e.g. `triton_tem_fused_mm_t_view_0` x10 @ 2.2 ms each on 6k).
- DS prefill elementwise (both workloads): `CatArrayBatchedCopy` (paged gather), `index_elementwise`, `neg_`, bf16↔fp32 `converting_copy` around the compiled regions. vLLM does paged access in-kernel and keeps everything bf16.
- DS attention grid: prefill (32,48,1) @6k = 1536 CTAs (32 heads × 48 blocks of 128 tokens) — well parallelized; decode collapses to (32,1,1).
- **Net prefill:** vLLM wins 264.4 vs 302.6 ms (6k) and 1244.2 vs 1309.0 ms (20k) — its GEMM efficiency + zero elementwise overhead outweigh DS's ~10 ms (6k) / ~95 ms (20k) flash win.

## 3. Decode analysis (where the big gap is)

### 3.1 Attention roofline + grid evidence (the key finding)

Per-step KV reads (bf16, 2×8 heads×128 dim = 4096 B/token/layer):

| | 6k | 20k |
|---|---|---|
| KV per step (36 layers) | 907 MB | 3.02 GB |
| Roofline @1555 GB/s | 583 µs/step (16.2 µs/layer) | 1.94 ms/step (54 µs/layer) |
| **DS** `flash_fwd`, grid (32,1,1)×128 thr | 8.55 ms/step, **237.4 µs/layer** → 106 GB/s = **7% of roofline** | 25.8 ms/step, **717 µs/layer** → 117 GB/s = **7%** |
| **vLLM** `flash_fwd_splitkv`, grid (1,25,8) / (1,23,8) | 1.33 ms/step (26.2 + 10.8 µs/layer) → main kernel 961 GB/s = **62%** | 2.73 ms/step (64.7 + 11.2 µs/layer) → main 1297 GB/s = **83%** |

32 CTAs of 128 threads can engage at most 30% of the 108 SMs at 20% occupancy each → the kernel is latency/parallelism-bound, and the 9× slowdown vs split-KV shows the sequence-split parallelism is the missing factor (classic FlashDecoding; longer sequences make it worse, 717 µs/layer @20k).

### 3.2 Per-decode-step breakdown (6k; 15 steps)

| Component (ms/step) | DS best | vLLM |
|---|---|---|
| Attention | **8.55** (7% BW roofline) | **1.33** (62% BW roofline) |
| M=1 GEMM/GEMV (fused triton incl. lm_head + cuBLAS/cutlass) | 6.75 | 6.80 |
| GQA reshape / cat / index / casts (elementwise) | **3.20** | **0.04** |
| Norms/act (unfused) + reduce + misc | 0.42 | 0.81 |
| Inter-kernel CPU gap (GPU idle) | **3.61** | **0.39** |
| **Total** | **22.5** (measured step period 22.25) | **9.4** (kernel 8.98 + gap 0.39) |

20k per step: DS 39.3 ms (flash 25.8 + GEMV ~6.8 + elementwise 3.2 + gap 3.3) vs vLLM 10.8 ms (flash 2.7 + GEMM 6.9 + gap 0.4).

- **M=1 GEMM is NOT a differentiator:** both engines sit at ~69% of the weight-bandwidth roofline (7.27 GB/step → 4.68 ms; DS 6.75 / vLLM 6.80 ms/step). DS's compiled fused GEMVs are on par with vLLM's cuBLAS/cutlass. No action needed.
- DS decode elementwise top kernels (6k): `elementwise_kernel` x3240 9.3 ms, `index_elementwise` x1080 4.3 ms, `CatArrayBatchedCopy` x1080 4.1 ms, `neg_`/casts ~6 ms — i.e. ~89 µs/layer of GQA regrouping + paged indexing + dtype casts re-launched **every step**.
- DS CPU gap: 26.5k inter-kernel gaps totalling 54.2 ms (median 0.1 µs, max 2.1 ms) — the sampler (top-p/top-k, radix sort) and paged-cache management run between graph replays, not inside them.

## 4. nsys corroboration

`nsys profile -t cuda,cublas,cudnn` ×4 (full runs incl. warmup + graph capture), Nsight 2026.1.3:

- **Wall times match the bench within 2–5%** (ds 625.0/1619.1; vllm 403.4/1322.5 ms) → profiler overhead is small; the torch-trace phase split (boundary = end of last kernel ≥1 ms) is the primary per-phase source.
- **Grids confirmed in CUPTI records:** DS prefill flash (32,48,1) @6k, (32,153,1) @20k; decode flash (32,1,1) in both. vLLM splitkv prefill (96,1,32) @6k, (305,1,32) @20k; decode (1,25,8)/(1,23,8).
- **API overhead:** DS 6k whole-run `cudaLaunchKernel` 255 ms / 10 550 calls (median 5.8 µs) + `cudaGraphLaunch` 30 calls; vLLM 6k 329 ms / 2 850 calls (median 7.9 µs). Consistent with the gap analysis: vLLM issues far fewer, heavier launches (full-step graphs).
- Caveat: nsys **undercounts CUDA-graph-replayed kernels** in this trace mode (ds_6k: 180 flash instances vs 1152 executed) → per-phase kernel *time* comes from the torch profiler, which records every replay instance; nsys is used for wall, grids, and API stats only.

## 5. ncu attempt (blocked on this node)

`ncu --set full` (Nsight Compute 2026.1.1) was attempted for SOL/occupancy of the two decode-attention kernels (`scripts/ncu_runs.sh`: DS with `--no-graph-capture`, vLLM with `--enforce-eager`, `--launch-skip 36 --launch-count 4`). Both failed at process attach:

```
Failed to create counter availability image (error = 20)
Profiling failed because a driver resource was unavailable.
Ensure that no other tool (like DCGM) is concurrently collecting profiling data.
```

Root cause: the node's root-owned **DCGM hostengine** (`/usr/bin/nv-hostengine`, pid 77222) holds the performance-counter interface; `RmProfilingAdminOnly=0` allows non-root PM but cannot displace DCGM's reservation. Only GPU 0 is allocated to this job (GPUs 1–3 not usable). nsys (CUPTI activity tracing) is unaffected, which is why sections 1–4 still have hard data.

**Workaround for a future run:** ask NCSA (`help+delta@ncsa.illinois.edu`) to restart DCGM with `DCGM_PROF_DISABLE=1` (or run ncu on a node without the DCGM hostengine). Expected confirmation value: DS decode flash ~7% DRAM SOL + ~30% SM engagement at low occupancy vs vLLM splitkv ~60–80% DRAM SOL; the grid/roofline math above already establishes the same ranking.

## 6. Ranked optimization opportunities (expected gains)

Per-item deltas vs the vLLM reference (which achieves the target behavior today). Gains are additive toward the floor.

| # | Change | 6k gain | 20k gain | Evidence / how |
|---|---|---|---|---|
| 1 | **Split-KV / FlashDecoding for decode attention** (sequence-split parallelism; FlashInfer decode or FA split) | **−108 ms** (flash 8.55→1.33 ms/step) | **−162 ms** (25.8→2.7 ms/step) | §3.1: 7% vs 62–83% of KV-BW roofline; 32 vs 200 CTAs. FlashInfer is already a dependency (used by the FlashInfer-prefill ablation cell) |
| 2 | **Kill per-step GQA/cast elementwise** — precompute block tables (C++/GPU), in-kernel paged access, stay bf16 | **−47 ms** | **−22 ms** | §3.2: 3.20 vs 0.04 ms/step; top kernels listed in §3.2 |
| 3 | **Close the decode CPU gap** — full-step CUDA graph (sampler inside) or C++ step loop | **−48 ms** (3.61→0.39 ms/step) | **−20 ms** | §3.2 gap row; nsys API stats §4 |
| 4 | **Prefill: remove elementwise/cast/gather tax** (in-kernel paged gather, no bf16↔fp32 round-trips) | **−50 ms** | **−155 ms** | §2: 51.0/158.2 ms vs ~0 |
| 5 | **Prefill GEMM efficiency** (tile heuristics; fewer transposed GEMM variants) | −12 ms | −49 ms | §2: 76–79% vs 82–86% of peak |
| | **Floor after 1–5** | **≈392 ms** (vs vLLM 400.8) | **≈1207 ms** (vs vLLM 1296.1) | DS's faster prefill flash (§2) is what lets it beat vLLM at 20k |

**Keep:** DS prefill attention (beats vLLM splitkv-prefill by 10–19%), DS compiled fused M=1 GEMVs (on par with cuBLAS), bucket graphs + paged prefill (both already in the best config).

**Out of scope but high-value:** vLLM's default prefix caching gave **2.6× on repeated prompts** (154 vs 398 ms @6k; gen-0 437.8 ms then cache hits).

**Prefix-anomaly resolution (2026-11-12):** the DS ablation's `prefix_reuse_blocks=0` on A100 was flag/code drift, not hardware. The A40 cells ran on pre-`7ed57f2e0` code, where `--use-paged-cache` alone took the paged prefill path; after that commit paged prefill became opt-in (`use_paged_cache and enable_paged_prefill`), so the original A100 prefix cells (same flags, no `--enable-paged-prefill`) fell back to the dense-staging chunk path — 0 blocks reused, 24 prefill forwards, sequential per-row prefill (14 vs 7 decode forwards). Re-running both cells with `--enable-paged-prefill` restores the A40 effective config (95 blocks / 6080 tokens reused, 13 prefill forwards; see `optimization_ablation_a100.md`).

## 7. Validity controls

- **vLLM process-mode A/B** (same harness, 3 iters each): 6k mp 400.4/397.5/398.7 vs sp 397.5/397.4/396.9; 20k mp 1291.6/1293.7/1295.7 vs sp 1293.9/1295.3/1295.0 → in-process engine costs only ~3 ms, so the official multiprocess numbers are valid; in-process was used only so Kineto can see the kernels.
- **Prefix caching:** ON (vLLM default) would corrupt the profile (cache-hit runs finish in ~154 ms @6k vs 398 ms fresh); driver defaults it OFF to match the DS no-reuse baseline.
- **Token identity:** all four profiled runs reproduce the ablation SHAs (`d4a324e9`, `c4291b80`) → same tokens/timing regime.
- **Phasing:** boundary = end of last kernel ≥1 ms (graph-launch position is unreliable: vLLM v2 replays prefill GEMMs from piecewise CUDA graphs; DS prefill is eager).
- **Caveats:** single measured iteration (decode is stable: step period 21.6–23.4 ms @6k); profiler adds ~0.7–2% wall; nsys graph-replay undercount (see §4); ncu SOL unmeasured (see §5) — SOL figures are computed from bytes/time and grids, not PM counters.

## 8. Post-optimization results (2026-11-12)

Implemented from §6: **(1) FlashInfer split-KV decode** — `BatchDecodeWithPagedKVCacheWrapper` (fa2 tensor-core, split-KV) routed in via `use_flashinfer_decode`; the paged cache re-plans every step from host write positions with a ragged page table (each row claims exactly the blocks holding its valid tokens; idle rows claim one block for the fixed-batch kernel), so no D2H sync lands inside the replay; **(2) decode-side elementwise cleanup** — no-cat RoPE in the paged shim, head-major attention output (no flatten/copy before `o_proj`), dense causal-mask update skipped for paged rows, scalar position for compact paged rows; **(3) compiled sampler** (`enable_compiled_sampler`). New best = old best + these three.


**Headline (bench, median of 3 iterations, batch 1 / C1 / seed 0; tokens verified against fresh vLLM oracles):**

| | 6k old best | 6k new best | 6k vLLM | 20k old best | 20k new best | 20k vLLM |
|---|---:|---:|---:|---:|---:|---:|
| Wall (ms) | 602.1 | **492.1** | 401.1 | 1579.0 | **1413.3** | 1307.4 |
| Decode forward (ms) | 284.6 | **174.3** | — | 254.3 | **92.2** | — |
| Tok/s | 26.3 | 32.4 | — | 5.0 | 5.6 | — |
| vs vLLM | 1.50× | **1.23×** | — | 1.22× | **1.08×** | — |

Gain: **−110 ms (−18%) @6k, −166 ms (−11%) @20k** — entirely from decode; prefill is unchanged by design (304–307 ms both). Correctness: all three configs (old best, new best, fresh vLLM) produce identical tokens — 6k `d4a324e9`, 20k `c4291b80` — so the kernel swap moved zero sampled tokens.

Same configs under the study's 1-iteration convention: new best **497.6 / 1429.3 ms** vs old best 619.0 / 1589.5 ms and the vLLM reference 401.1 / 1307.4 ms; the full rerun on post-optimization code is in `optimization_ablation_a100.md` (new `newbest_6k`/`newbest_20k` rows).

**Where the 6k decode step went (15 steps; trace `profile_raw/ds_6k_postopt.*`):**

| Component (ms/step) | DS old best | DS new best | vLLM |
|---|---:|---:|---:|
| Attention | 8.55 (7% of KV-BW roofline, grid 32 CTAs) | **1.01** (flashinfer split-KV, 28 µs/layer, **58% of roofline** vs vLLM's 62%) | 1.33 |
| M=1 GEMM/GEMV | 6.75 | 6.65 | 6.80 |
| Elementwise / paged bookkeeping | 3.20 | 3.65 | 0.04 |
| Norm/act + reduce + misc | 0.42 | ~0.5 | 0.81 |
| Inter-kernel CPU gap | 3.61 | 3.77 | 0.39 |
| **Total** | **22.5** | **~15.6** | **9.4** |

Decode attention now sits at vLLM parity on the KV-bandwidth roofline (§3.1); M=1 GEMMs unchanged as predicted (weight-bandwidth-bound). The elementwise line is flat, not down: the RoPE `rotate_half` cat and head-flatten copies are gone, but paged write/position/index bookkeeping and the per-step plan (one small block-table read + pinned H2D copies) now land in it.

**Residual gap to the §6 floor (≈392 ms @6k):** the unimplemented prefill items — elementwise/cast/gather 46.0 ms of the 51.0 ms prefill tax (item 4) and GEMM tile heuristics (item 5, −12 ms) — plus the CPU gap, which the compiled sampler did not shrink (the per-step host plan + replay launch + cache management still run between graph replays; item 3 needs the plan inside the graph). 492.1 − 46 − 12 − ~40 (gap) ≈ 394 ms, matching the §6 floor.


## Appendix: raw data & commands

- Raw traces/stats: `runs/a100_2026-11-11/profile_raw/` — `ds_6k|ds_20k|vllm_6k_noreuse|vllm_20k_noreuse.{trace.json,events.json,summary.txt,result.json,phases.txt}`, post-optimization `ds_6k_postopt.{trace.json,events.json,summary.txt,result.json,phases.txt}` (§8), `nsys_*.nsys-rep/.sqlite/.kernsum.txt/.log`, A/B `vllm_ab_{6k,20k}_{mp,sp}.json/.log`, ncu logs `ncu_*.log`.
- Drivers: `benchmarks/profile_torch_rollout.py` (DS; `--workload 6k|20k --no-graph-capture` for ncu; §8 adds `--use-flashinfer-decode --compiled-sampler`), `benchmarks/profile_torch_vllm.py` (`--enforce-eager` for ncu).
- Parsers: `scripts/parse_trace_phases.py` (chrome-trace phase split), `scripts/parse_nsys_phases.py` (naive; superseded), `scripts/parse_nsys_phases2.py` (flash-anchored; unreliable with graph replay — kept for reference).
- Capture scripts: `scripts/nsys_runs.sh` (done), `scripts/ncu_runs.sh` (blocked, §5).
- Env for all runs: `CUDA_VISIBLE_DEVICES=0`, `CUDA_HOME=/u/msajid/cuda-12.9-shim`, gcc-toolset-14, `HF_HOME=/u/msajid/hf_cache`, `TORCHINDUCTOR_CACHE_DIR=/u/msajid/torchinductor_cache`, `VLLM_CACHE_ROOT=/u/msajid/vllm_cache`, `FLASHINFER_EXTRA_CUDAFLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`, opsd `bin` on PATH (ninja for vLLM JIT).
