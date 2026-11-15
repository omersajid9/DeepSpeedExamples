# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Ablation driver: perf/paged_fused_kernels ladder (A-E) + vLLM reference.

Workload: 3072-token context -> 1024-token decode (LongBench prompts truncated
to 3072; every row decodes exactly 1024 tokens, greedy, seed 0, warmup 1, one
measured iteration). Loop order, slowest to fastest: model -> batch -> level:

  models : qwen35_4b, qwen35_9b
  batches: 2, 4, 8, 16, 32
  levels : A, B, C, D, E, vllm

Levels (DeepSpeed checkout at the level's commit; flags enable the new
capability on top of the checked-out state; chunked prefill off everywhere):

  A  commit 8b05554f8 - no optimizations enabled. At this commit continuous
     batching rejects hybrid GDN models, so Qwen3.5 runs the non-continuous
     generate path (continuous_batch_size=None), graph capture on.
  B  commit 2aaea670c - + paged attention for hybrid (GDN) models
     (--use-paged-cache --enable-paged-prefill; paged prefill is required for
     paged hybrid runs). CB with continuous width = batch size.
  C  commit 7cf94a64b - + FlashInfer split-KV paged decode, cache-independent
     fusions kept (--use-flashinfer-decode --use-segki: the fused add+norm and
     lm-head fusions stay active while FlashInfer owns the attention core).
  D  commit 86375e7b2 - + fused attention-prep kernel (--use-fused-attn-prep:
     one launch for q|gate split, q/k per-head norm, partial RoPE, paged K/V
     write, gate sigmoid; it writes into the pool the FlashInfer core reads).
  E  commit bee5f1b22 - + stacked GDN input-projection GEMM
     (--gdn-proj-concat) with every remaining omersajid9 lever on
     (--use-fused-residual-tail). E = all omersajid9 optimizations.
     Note: use_segki_paged_decode is intentionally OFF - the fused segki
     paged-decode attention core is the alternative C replaced (373 us/layer
     serial-KV vs 54 us/layer + 4 us merge split-KV), and it is bypassed
     whenever use_flashinfer_decode is set.
  vllm  reference: vLLM 0.33.0 (separate conda env), same workload, ignore-EOS.

The driver checks out each level's commit in the DeepSpeed repo before running
its cells and restores the original branch when it finishes (also on
Ctrl-C/failure). Cells are resumable: raw/<model>_b<batch>_<level>.json that
parses with a non-empty "iterations" is skipped.

Usage:
  python scripts/run_paged_fused_ablation.py                 # full 60-cell matrix
  python scripts/run_paged_fused_ablation.py --list          # show matrix + done/missing
  python scripts/run_paged_fused_ablation.py --only qwen35_4b,4,C   # one cell
  python scripts/run_paged_fused_ablation.py --force qwen35_4b,4,C  # redo a cell
  python scripts/run_paged_fused_ablation.py --dry-run       # print commands only
"""

import argparse
import csv
import fcntl
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

def _env(name, default):
    """Infra path override: PFUSE_* env vars let the same driver run on a
    non-Delta box (e.g. the B300 setup in b300/b300.env); defaults are the
    Delta paths, so SLURM/interactive runs are unaffected."""
    return os.environ.get(name, default)


OPSD = Path(__file__).resolve().parent.parent
DS = Path(_env("PFUSE_DS_REPO", "/u/msajid/projects/DeepSpeed"))
OUT = OPSD / "runs" / "paged_fused_ablation_2026-11-15"
RAW = OUT / "raw"
LOGS = OUT / "logs"

OPS_PY = _env("PFUSE_OPS_PY", "/work/nvme/bcjw/msajid/conda/opsd/bin/python")
VLLM_PY = _env("PFUSE_VLLM_PY", "/work/nvme/bcjw/msajid/conda/vllm33/bin/python")
SP = _env("PFUSE_SP", "/work/nvme/bcjw/msajid/conda/opsd/lib/python3.12/site-packages")
DS_BENCH = OPSD / "benchmarks" / "bench_paged_fused_ablation.py"
VLLM_BENCH = OPSD / "benchmarks" / "bench_vllm_offline.py"

MODELS = {
    "qwen35_4b": _env("PFUSE_MODEL_4B", "Qwen/Qwen3.5-4B"),  # or a local dir
    "qwen35_9b": _env("PFUSE_MODEL_9B",
                      "/work/nvme/bcjw/runguol2/models/Qwen3.5-9B"),
}

HF_HOME = _env("PFUSE_HF_HOME", "/work/nvme/bcjw/msajid/hf_cache")
TORCHINDUCTOR_CACHE_DIR = _env("PFUSE_TORCHINDUCTOR_CACHE",
                               "/u/msajid/torchinductor_cache")
TORCH_EXTENSIONS_DIR = _env("PFUSE_TORCH_EXTENSIONS", "/u/msajid/torch_extensions")
BENCH_TMPDIR = _env("PFUSE_TMPDIR", "/u/msajid/tmp_bench")
VLLM_CACHE_ROOT = _env("PFUSE_VLLM_CACHE", "/u/msajid/vllm_cache")
CUDA_HOME = _env("PFUSE_CUDA_HOME", "/u/msajid/cuda-12.9-shim-wrap")
NVCC_PATH = _env("PFUSE_NVCC_PATH",
                 "/u/msajid/cuda-12.9-shim-wrap/bin:/u/msajid/cuda-12.9-shim/bin")
CC = _env("PFUSE_CC", "/opt/rh/gcc-toolset-14/root/usr/bin/gcc")
CXX = _env("PFUSE_CXX", "/opt/rh/gcc-toolset-14/root/usr/bin/g++")
CPATH = _env("PFUSE_CPATH",
             f"{SP}/nvidia/cu13/include:{SP}/nvidia/cu13/include/cccl")
FLASHINFER_FLAGS = _env("PFUSE_FLASHINFER_FLAGS",
                        "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK")

BATCHES = [2, 4, 8, 16, 32]

TASK = dict(data="data/longbench_20k.jsonl", prompt=3072, new=1024,
            temperature=0.0, top_p=1.0)

WARMUP, ITERATIONS = 1, 1

# level -> (commit or None for vllm, extra bench flags, continuous-batch-size)
# A: non-continuous path (CB at that commit does not support hybrid GDN models)
# B-E: continuous width = batch size
LEVELS = {
    "A": ("8b05554f847bd7ca21573cbf0ea49d77d2f2cb9d",
          [], 0),
    "B": ("2aaea670cda107456d7819246bdd7f74d90a6a55",
          ["--use-paged-cache", "--enable-paged-prefill"], None),
    "C": ("7cf94a64b0f2f98fb1c03010a18638b66e898701",
          ["--use-paged-cache", "--enable-paged-prefill",
           "--use-flashinfer-decode", "--use-segki"], None),
    "D": ("86375e7b239394449278793baa22c2a1a8651060",
          ["--use-paged-cache", "--enable-paged-prefill",
           "--use-flashinfer-decode", "--use-segki",
           "--use-fused-attn-prep"], None),
    "E": ("bee5f1b22ebc67e5e7d2254e7041d3a40d0d379e",
          ["--use-paged-cache", "--enable-paged-prefill",
           "--use-flashinfer-decode", "--use-segki",
           "--use-fused-attn-prep", "--gdn-proj-concat",
           "--use-fused-residual-tail"], None),
    "vllm": (None, None, None),
}
LEVEL_ORDER = ["A", "B", "C", "D", "E", "vllm"]


def git(*args):
    return subprocess.run(["git", "-C", str(DS), *args],
                          capture_output=True, text=True, check=True)


def ds_commit_now():
    return git("rev-parse", "HEAD").stdout.strip()


def ds_ref_now():
    out = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    return "HEAD" if out == "detached HEAD" else out


CUDA_ARCH = "8.0"
HARDWARE = ""


def probe_cuda_arch():
    """(arch, gpu_name) of the allocated GPU; the JIT builds target that arch."""
    try:
        proc = subprocess.run(
            [OPS_PY, "-c",
             "import torch\n"
             "p = torch.cuda.get_device_properties(0)\n"
             "print(f'{p.major}.{p.minor}\\t{p.name}')"],
            capture_output=True, text=True, timeout=180, env=cell_env("ds"))
        line = proc.stdout.strip().splitlines()[-1]
        parts = line.split("\t")
        arch = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else ""
        if arch and arch.replace(".", "").isdigit():
            return arch, name
    except Exception as exc:
        print(f"[arch] probe failed ({exc}); falling back to 8.0", flush=True)
    return "8.0", ""


def cell_env(kind):
    env = dict(os.environ)
    # SLURM sets CUDA_VISIBLE_DEVICES to the allocated GPU(s); honour it and
    # only default to GPU 0 for interactive runs.
    gpu = env.get("CUDA_VISIBLE_DEVICES", "0")
    if kind == "ds":
        env.update({
            "PATH": f"{Path(OPS_PY).parent}:{NVCC_PATH}:{env.get('PATH', '')}",
            "PYTHONPATH": f"{DS}:{OPSD}:{env.get('PYTHONPATH', '')}",
            "HF_HOME": HF_HOME,
            "TORCHINDUCTOR_CACHE_DIR": TORCHINDUCTOR_CACHE_DIR,
            "TORCH_EXTENSIONS_DIR": TORCH_EXTENSIONS_DIR,
            "TMPDIR": BENCH_TMPDIR,
            "TORCH_CUDA_ARCH_LIST": CUDA_ARCH,
            "TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS": "0",
            "CC": CC,
            "CXX": CXX,
            # The op-builder JIT (fused_glu: segki kernels + fused attn-prep)
            # compiles the 12.9 node nvcc against the pip cu13 CCCL headers
            # (CPATH); the wrap shim adds -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
            # (the same macro flashinfer's own JIT passes).
            "CUDA_HOME": CUDA_HOME,
            "CPATH": CPATH,
            "LD_LIBRARY_PATH": f"{Path(OPS_PY).parent.parent / 'lib'}:"
                               f"{env.get('LD_LIBRARY_PATH', '')}",
            "FLASHINFER_EXTRA_CUDAFLAGS": FLASHINFER_FLAGS,
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTHONUNBUFFERED": "1",
        })
    else:  # vllm
        env.update({
            "PATH": f"{Path(VLLM_PY).parent}:{env.get('PATH', '')}",
            "HF_HOME": HF_HOME,
            "VLLM_CACHE_ROOT": VLLM_CACHE_ROOT,
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTHONUNBUFFERED": "1",
        })
    return env


def out_path(model, batch, level):
    return RAW / f"{model}_b{batch:02d}_{level}.json"


def is_done(model, batch, level):
    p = out_path(model, batch, level)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
        return bool(data.get("iterations"))
    except Exception:
        return False


def consecutive_failures(model, batch, level, limit=3):
    """Trailing run of failures for this cell in status.jsonl (capped).

    Lets a genuinely broken cell (e.g. OOM on the smallest GPU) be skipped
    instead of re-failing forever inside a longer-running SLURM job; fix it
    and re-run with --force later."""
    n = 0
    try:
        with open(OUT / "status.jsonl") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return 0
    for line in reversed(lines):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if (d.get("model"), d.get("batch"), d.get("level")) != (model, batch, level):
            continue
        if d.get("ok"):
            break
        n += 1
        if n >= limit:
            break
    return n


def cell_cmd(model, batch, level, t):
    commit, extra, cont = LEVELS[level]
    if level == "vllm":
        return [
            VLLM_PY, str(VLLM_BENCH),
            "--model", MODELS[model],
            "--data", t["data"],
            "--batch-size", str(batch),
            "--max-prompt-length", str(t["prompt"]),
            "--max-new-tokens", str(t["new"]),
            "--prefill-chunk-size", "4096",
            "--temperature", f"{t['temperature']:.1f}",
            "--top-p", f"{t['top_p']:.1f}",
            "--top-k", "-1",
            "--seed", "0",
            "--warmup", str(WARMUP),
            "--iterations", str(ITERATIONS),
            "--output", str(out_path(model, batch, level)),
        ]
    cont_size = str(batch if cont is None else cont)
    return [
        OPS_PY, str(DS_BENCH),
        "--model", MODELS[model],
        "--level", level,
        "--ds-commit", commit,
        "--data", t["data"],
        "--batch-size", str(batch),
        "--continuous-batch-size", cont_size,
        "--max-prompt-length", str(t["prompt"]),
        "--max-new-tokens", str(t["new"]),
        "--temperature", f"{t['temperature']:.1f}",
        "--top-p", f"{t['top_p']:.1f}",
        "--top-k", "-1",
        "--seed", "0",
        "--warmup", str(WARMUP),
        "--iterations", str(ITERATIONS),
        "--save-response-ids",
        "--ignore-eos",
        *extra,
        "--output", str(out_path(model, batch, level)),
    ]


class Checkout:
    """Holds the DeepSpeed repo at the requested commit; restores on exit."""

    def __init__(self):
        self.original_ref = ds_ref_now()
        self.original_head = ds_commit_now()
        # rev-parse --abbrev-ref yields "HEAD" when detached; restore() must
        # be able to get back to a concrete commit in that case (e.g. a
        # previously killed run left the tree detached).
        if self.original_ref == "HEAD":
            self.original_ref = self.original_head
        self.current = None
        dirty = [l for l in git("status", "--porcelain").stdout.splitlines()
                 if not l.startswith("??")]
        if dirty:
            raise SystemExit(
                f"DeepSpeed working tree has tracked modifications; clean it first:\n"
                + "\n".join(dirty))

    def set(self, commit):
        if commit is None:
            return
        if self.current == commit:
            return
        git("checkout", "-q", commit)
        self.current = commit
        print(f"[git] DeepSpeed -> {commit[:12]}", flush=True)

    def restore(self):
        if self.current is not None or self.original_head != ds_commit_now():
            print(f"[git] restoring {self.original_ref} "
                  f"({self.original_head[:12]})", flush=True)
        git("checkout", "-q", self.original_ref)


def run_cell(model, batch, level, ck):
    commit, extra, _ = LEVELS[level]
    ck.set(commit)
    out = out_path(model, batch, level)
    log = LOGS / f"{model}_b{batch:02d}_{level}.log"
    cmd = cell_cmd(model, batch, level, TASK)
    t0 = time.time()
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, cwd=OPSD,
                              env=cell_env("ds" if level != "vllm" else "vllm"),
                              stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    ok = proc.returncode == 0 and is_done(model, batch, level)
    with open(OUT / "status.jsonl", "a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model,
                             "batch": batch, "level": level, "rc": proc.returncode,
                             "elapsed_s": round(elapsed, 1), "ok": ok,
                             "log": str(log)}) + "\n")
    print(f"[{'done' if ok else 'FAIL'}] {model} / b{batch:02d} / {level} "
          f"rc={proc.returncode} {elapsed/60:.1f}min log={log.name}", flush=True)
    return ok


def load_rows():
    rows = []
    for p in sorted(RAW.glob("*.json")):
        m = re.match(r"^(qwen35_\d+b)_b(\d{2})_(\w+)\.json$", p.name)
        if not m or m.group(1) not in MODELS or m.group(3) not in LEVELS:
            continue
        model, batch, level = m.group(1), int(m.group(2)), m.group(3)
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        iters = d.get("iterations") or []
        if not iters:
            continue

        def median(key):
            vals = [i["profile"].get(key) for i in iters]
            vals = [v for v in vals if v is not None]
            return statistics.median(vals) if vals else None

        wall = statistics.median([i["wall_ms"] for i in iters])
        prefill = median("prefill_forward_ms")
        decode = median("decode_forward_ms")
        tok_s = median("tokens_per_second")
        rows.append({
            "model": model, "batch": batch, "level": level,
            "wall_ms": wall, "prefill_ms": prefill, "decode_ms": decode,
            "tok_s": tok_s,
            "peak_mb": max(d.get("peak_memory_mb", 0.0), d.get("warmup_peak_memory_mb", 0.0)),
            "engine": d.get("engine", "deepspeed-rollout"),
            "ds_commit": (d.get("ds_commit") or LEVELS[level][0] or "")[:12],
            "config": d.get("config", {}),
            "sha": (d.get("response_sha256") or iters[0].get("response_sha256", ""))[:8],
            "response_ids": d.get("response_ids"),
        })
    return rows


def fmt(v, digits=1):
    return "" if v is None else f"{v:.{digits}f}"


def token_agreement(base_rows, other_rows):
    if not base_rows or not other_rows:
        return "n/a"
    same = 0
    prefix = None
    for b, o in zip(base_rows, other_rows):
        if b == o:
            same += 1
        else:
            p = next((i for i, (x, y) in enumerate(zip(b, o)) if x != y),
                     min(len(b), len(o)))
            prefix = p if prefix is None else min(prefix, p)
    if same == len(base_rows):
        return f"{same}/{len(base_rows)}"
    return f"{same}/{len(base_rows)} (min prefix {prefix})"


def summarize(rows, hardware="A100-SXM4-40GB"):
    md = ["# Paged/fused-kernel ladder ablation — perf/paged_fused_kernels", ""]
    md.append(f"Date: {time.strftime('%Y-%m-%d')}. Node: {hardware}. Workload: "
              f"{TASK['prompt']} prompt -> {TASK['new']} decode, greedy seed 0, warmup 1, "
              f"one measured iteration, ignore-EOS, chunked prefill off. "
              f"DeepSpeed env: {OPS_PY} (torch 2.15.0+cu132, transformers 5.22.2, flashinfer 0.7.4); "
              f"vLLM env: {VLLM_PY} (vLLM 0.33.0). Levels: A=8b05554f8 (no optimizations; non-CB "
              f"path — CB at that commit rejects hybrid GDN models), B=2aaea670c +paged attention, "
              f"C=7cf94a64b +FlashInfer split-KV decode w/ cache-independent fusions, "
              f"D=86375e7b2 +fused attention-prep kernel, E=bee5f1b22 +stacked GDN projections "
              f"(all omersajid9 levers on); vllm = reference implementation.")
    md.append("")
    by = {}
    for r in rows:
        by.setdefault((r["model"], r["batch"]), {})[r["level"]] = r
    for model in MODELS:
        for batch in BATCHES:
            rows_b = by.get((model, batch))
            if not rows_b:
                continue
            base = rows_b.get("A")
            vllm = rows_b.get("vllm")
            md.append(f"## {model} — batch {batch} ({TASK['prompt']} -> {TASK['new']})")
            md.append("")
            md.append("| Level | Commit | Prefill ms | Decode ms | Total ms | Tok/s | Peak MiB | "
                      "vs A | vs prev | vs vLLM | Response SHA | Tokens vs A |")
            md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
            prev = None
            for lv in LEVEL_ORDER:
                r = rows_b.get(lv)
                if r is None:
                    md.append(f"| {lv} | — | — | — | — | — | — | — | — | — | — | — |")
                    continue
                vs_a = (f"{base['wall_ms']/r['wall_ms']:.2f}x"
                        if lv != "A" and base and r["wall_ms"] and base["wall_ms"] else "—")
                vs_prev = (f"{prev['wall_ms']/r['wall_ms']:.2f}x"
                           if prev and r["wall_ms"] and prev["wall_ms"] else "—")
                vs_v = (f"{vllm['wall_ms']/r['wall_ms']:.2f}x"
                        if vllm and r["wall_ms"] and vllm["wall_ms"] else "—")
                agree = ("—" if (lv == "A" or base is None or not r.get("response_ids")
                                 or not base.get("response_ids"))
                         else token_agreement(base["response_ids"], r["response_ids"]))
                md.append(f"| {lv} | `{r['ds_commit'][:8] if r['ds_commit'] else '—'}` | "
                          f"{fmt(r['prefill_ms'])} | {fmt(r['decode_ms'])} | {fmt(r['wall_ms'])} | "
                          f"{fmt(r['tok_s'], 2)} | {fmt(r['peak_mb'], 0)} | {vs_a} | {vs_prev} | "
                          f"{vs_v} | `{r['sha']}` | {agree} |")
                prev = r
            md.append("")
            md.append("vs vLLM > 1x means the DeepSpeed level is faster than vLLM on the same "
                      "workload. Decode-only increments use the profile's decode_forward_ms when present.")
            md.append("")
    done = sum(1 for m in MODELS for b in BATCHES for lv in LEVEL_ORDER if is_done(m, b, lv))
    total = len(MODELS) * len(BATCHES) * len(LEVEL_ORDER)
    md.append(f"_Status: {done}/{total} cells complete. Raw JSONs in `raw/`, logs in `logs/`. "
              f"Regenerate with `python scripts/run_paged_fused_ablation.py` "
              f"(skips finished cells, rewrites this file)._")
    md.append("")
    (OUT / "paged_fused_ablation.md").write_text("\n".join(md) + "\n")
    fields = ["model", "batch", "level", "ds_commit", "wall_ms", "prefill_ms", "decode_ms",
              "tok_s", "peak_mb", "sha"]
    with open(OUT / "paged_fused_ablation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"wrote {OUT/'paged_fused_ablation.md'} ({done}/{total} cells)", flush=True)


def main():
    global CUDA_ARCH, HARDWARE
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", default="",
                        help='cell spec "model,batch,level"; multiple separated by | or spaces')
    parser.add_argument("--force", default="",
                        help="cell specs to re-run even if done")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--summary-only", action="store_true",
                        help="regenerate md/csv from existing raw JSONs; run nothing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="",
                        help="alternate output dir (default "
                             "runs/paged_fused_ablation_2026-11-15)")
    parser.add_argument("--cuda-arch", default="auto",
                        help="TORCH_CUDA_ARCH_LIST for the JIT builds; 'auto' "
                             "probes the allocated GPU (A100=8.0, H200=9.0, A40=8.6)")
    parser.add_argument("--hardware", default="",
                        help="label recorded in the summary (default: probed GPU name)")
    args = parser.parse_args()

    global OUT, RAW, LOGS
    if args.out:
        OUT = Path(args.out).resolve()
        RAW = OUT / "raw"
        LOGS = OUT / "logs"

    if args.summary_only:
        RAW.mkdir(parents=True, exist_ok=True)
        summarize(load_rows(), hardware=HARDWARE or "unknown")
        return

    def specs(text):
        return [s for s in re.split(r"[\s|]+", text) if s]

    only = specs(args.only)
    force = specs(args.force)

    cells = []
    for model in MODELS:
        for batch in BATCHES:
            for level in LEVEL_ORDER:
                key = f"{model},{batch},{level}"
                if only and not any(key.startswith(o) for o in only):
                    continue
                cells.append((model, batch, level, key in force))

    run_cells = not (args.list or args.dry_run)

    # Keep two drivers from racing on one output dir (SLURM resubmits can
    # overlap with a manual re-run); the lock is released on exit.
    lock_fh = None
    if run_cells:
        RAW.mkdir(parents=True, exist_ok=True)
        lock_fh = open(OUT / ".driver.lock", "w")
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another driver is already running on {OUT} "
                             f"(pid in {OUT/'.driver.lock'})")
        lock_fh.write(f"{os.getpid()}\n")
        lock_fh.flush()

    ck = None
    try:
        if run_cells:
            if args.cuda_arch != "auto":
                CUDA_ARCH = args.cuda_arch
                if args.hardware:
                    HARDWARE = args.hardware
            else:
                CUDA_ARCH, name = probe_cuda_arch()
                HARDWARE = args.hardware or name
            print(f"[arch] TORCH_CUDA_ARCH_LIST={CUDA_ARCH} hardware={HARDWARE or '(unknown)'}",
                  flush=True)
            ck = Checkout()
        for model, batch, level, f in cells:
            state = "redo" if f else ("done" if is_done(model, batch, level) else "run")
            commit, _, _ = LEVELS[level]
            extra = "" if level == "vllm" else f" @ {commit[:8]}"
            print(f"[{state:4}] {model} / b{batch:02d} / {level}{extra}", flush=True)
        if args.list:
            return
        if run_cells:
            LOGS.mkdir(parents=True, exist_ok=True)
        for model, batch, level, f in cells:
            if not f and is_done(model, batch, level):
                continue
            if not f and consecutive_failures(model, batch, level) >= 3:
                print(f"[skip] {model} / b{batch:02d} / {level}: 3 consecutive failures; "
                      f"poisoned until --force (see {OUT/'status.jsonl'})", flush=True)
                continue
            if args.dry_run:
                print(shlex.join(cell_cmd(model, batch, level, TASK)))
                continue
            run_cell(model, batch, level, ck)
        if not args.dry_run:
            # Chain marker: the SLURM epilogue and the watchdog read this to
            # decide whether to queue another hour of this partition (cells
            # remain) or to advance to the next partition. "poison" = cells
            # skipped after 3 consecutive failures; they count as handled.
            # Always reported against the FULL matrix (not the --only
            # filtered list), so a partial --only run can never make the
            # epilogue think the whole partition is done.
            all_cells = [(m, b, l) for m in MODELS for b in BATCHES
                         for l in LEVEL_ORDER]
            done = sum(1 for m, b, l in all_cells if is_done(m, b, l))
            poison = [f"{m},{b},{l}" for m, b, l in all_cells
                      if not is_done(m, b, l)
                      and consecutive_failures(m, b, l) >= 3]
            (OUT / "driver_result.json").write_text(
                json.dumps({"done": done, "total": len(all_cells), "poison": poison},
                           indent=2) + "\n")
            summarize(load_rows(), hardware=HARDWARE or "unknown")
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
        raise
    finally:
        if ck is not None:
            ck.restore()
        if lock_fh is not None:
            lock_fh.close()


if __name__ == "__main__":
    main()
