# Running the paged/fused-kernel ablation on the B300 box

The B300 box is a **4th hardware column** for the ablation — it has its own
output dir (`runs/paged_fused_ablation_2026-11-15_b300`), never mixed into
the Delta a100/h200/a40 dirs. The driver records the probed GPU name
("NVIDIA B300 SXM6 AC") and sm_103 arch in the summary, so the numbers stay
labeled correctly.

## What's here

| File | Where it runs | What it does |
|---|---|---|
| `transfer_to_b300.sh` | Delta login node | rsyncs repos + 9B model (+4B cache) to `user@box:~/pfuse/` |
| `setup_b300.sh` | B300 box | Miniforge + `opsd` & `vllm33` conda envs (pinned to the Delta versions), PyPI nvcc 13.2 + CUDA shim, DeepSpeed editable from the perf branch, end-to-end verification |
| `b300.env` | B300 box | the `PFUSE_*` overrides the driver reads (paths, caches, CUDA shim, gcc) |
| `run_smoke.sh` | B300 box | 6 cells (4B b02, A–E + vllm): JIT cold start for sm_103 + every engine path |
| `run_matrix.sh` | B300 box | full 60-cell matrix; resumable |

## Steps

```bash
# 0) from the Delta login node (one-time, ~20 GB):
export PFUSE_B300_DEST=ubuntu@compute1-st-queue-1
bash b300/transfer_to_b300.sh          # add PFUSE_B300_4B=1 if the box is offline

# 1) on the B300 box (one-time, ~15 min + downloads):
cd ~/pfuse/projects/DeepSpeedExamples/training/opsd
bash b300/setup_b300.sh

# 2) smoke (~30-60 min incl. JIT cold start + 4B download if needed):
bash b300/run_smoke.sh
#    check: all 6 of runs/paged_fused_ablation_2026-11-15_b300/raw/*.json exist
#           and the summary shows the B300 as the node; compare the ladder
#           shape (monotonic A>B>C>D>E, vllm reference nearby) with A100.

# 3) full matrix (~2-4 h; resumable — re-run after any interruption):
nohup bash b300/run_matrix.sh > b300/matrix.out 2>&1 &
tail -f b300/matrix.out                # per-cell progress + logs

# 4) fetch results back to Delta:
rsync -avz ubuntu@compute1-st-queue-1:pfuse/projects/DeepSpeedExamples/training/opsd/runs/paged_fused_ablation_2026-11-15_b300/ \
  /u/msajid/projects/DeepSpeedExamples/training/opsd/runs/paged_fused_ablation_2026-11-15_b300/
```

## Notes / gotchas

- **No SLURM on the box** — the driver is SLURM-agnostic; it just probes
  whatever GPU is visible and writes to `--out`. `CUDA_VISIBLE_DEVICES` is
  honored (default 0).
- **First-run cost**: sm_103 JIT cold start (fused_glu/segki, flashinfer,
  dskernels, inductor) — the smoke absorbs it; `TORCH_EXTENSIONS_DIR` /
  inductor cache live under `~/pfuse/` so later runs reuse them.
- **4B model**: downloaded from HF on first run (`PFUSE_HF_HOME`); if the
  box has no internet, push the cache with `PFUSE_B300_4B=1` in the
  transfer script instead.
- **9B model**: transferred locally (~19 GB) — no download needed.
- **Parallelize over the 8 B300s (optional)**: each driver git-checkouts the
  *shared* `~/pfuse/projects/DeepSpeed` between cells, so don't run 8
  drivers against one repo. Instead, per GPU i: clone the repo
  (`git clone ~/pfuse/projects/DeepSpeed ~/pfuse/ds_gpu$i` + `git submodule
  update` not needed), then
  `PFUSE_DS_REPO=~/pfuse/ds_gpu$i CUDA_VISIBLE_DEVICES=i \
   python3 scripts/run_paged_fused_ablation.py --cuda-arch auto \
   --only "qwen35_4b,4|qwen35_4b,8|qwen35_4b,16|qwen35_4b,32" ...`
  splitting the 60 cells by batch, each with its own `--out`
  (`..._b300_gpu<i>`), and merge the raw dirs afterwards. Sequential on one
  B300 is simpler and ~fine (2-4 h).
- **If setup's sm_103 test fails**: `pip install -U nvidia-cuda-nvcc` inside
  the opsd env (newer compiler) and re-run `setup_b300.sh`.
- **Driver portability**: the `PFUSE_*` env vars (see `b300.env`) override
  every hardcoded Delta path in `scripts/run_paged_fused_ablation.py`;
  unset, they default to the Delta values, so the SLURM conveyor is
  unaffected.
