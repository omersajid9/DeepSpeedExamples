# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Continuous-batching ablation on the optimization-merged branch.

Matrix: models (Qwen3-4B, Qwen3-8B) -> tasks (6k, 20k) -> levels A..F.

Levels (chunked prefill is off everywhere; seed 0; warmup 1; 1 measured iteration):
  A  continuous batching + eager            (--no-graph-capture)
  B  continuous batching + single graph     (--graph-capture-batch-sizes <C>)
  C  continuous batching + multiple graphs  (default bucket ladder)
  D  C + paged attention, prefill+decode    (--use-paged-cache --enable-paged-prefill)
  E  D + fused kernels                      (+ --torch-compile)
  F  D + FlashInfer kernels + compiled sampler
                                   (+ --flashinfer-prefill --use-flashinfer-decode --compiled-sampler)
F intentionally does NOT add --torch-compile on top of E: torch.compile and the
FlashInfer backends are alternative kernel strategies (FlashInfer's custom ops are
graph breaks for inductor), matching how they were isolated in
runs/optimization_ablation.md.

Resumable: every cell writes raw/<model>_<task>_<level>.json; a cell whose JSON
already exists and parses is skipped. Re-running this script therefore continues
wherever a previous instance stopped. The summary md/csv are regenerated from all
raw JSONs after every pass.

Usage:
  python scripts/run_cb_ablation.py                 # full matrix, resume-safe
  python scripts/run_cb_ablation.py --list          # show the matrix + done/missing
  python scripts/run_cb_ablation.py --only qwen3_8b,20k,E  # run a single cell
  python scripts/run_cb_ablation.py --force qwen3_4b,6k,A  # redo a finished cell
"""

import argparse
import csv
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

OPSD = Path(__file__).resolve().parent.parent
DS = Path("/u/msajid/projects/DeepSpeed")
OUT = OPSD / "runs" / "cb_ablation_2026-11-12"
RAW = OUT / "raw"
LOGS = OUT / "logs"

PY = "/work/nvme/bcjw/msajid/conda/opsd/bin/python"
BENCH = OPSD / "benchmarks" / "bench_continuous_graph.py"

MODELS = {
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
}

# Task shapes: 6k = Gumbel sampling like the old clean 6k rows; 20k = greedy like
# the old clean 20k rows. continuous_batch_size > 1 so that the single-graph vs
# bucket-ladder distinction (B vs C) is real: at C=1 the ladder degenerates to [1].
TASKS = {
    "6k": dict(data="data/longbench_20k.jsonl", batch=4, c=4,
               prompt=6144, new=16, temperature=0.7, top_p=0.9),
    "20k": dict(data="data/longbench_20k.jsonl", batch=2, c=2,
                prompt=20480, new=8, temperature=0.0, top_p=1.0),
}

WARMUP, ITERATIONS = 1, 1

# extra flags per level, in loop order
LEVELS = {
    "A_eager": ["--no-graph-capture"],
    "B_single_graph": None,  # single graph: --graph-capture-batch-sizes <C>
    "C_multi_graph": [],     # default bucket ladder
    "D_paged": ["--use-paged-cache", "--enable-paged-prefill"],
    "E_fused": ["--use-paged-cache", "--enable-paged-prefill", "--torch-compile"],
    "F_flashinfer_sampler": ["--use-paged-cache", "--enable-paged-prefill",
                             "--flashinfer-prefill", "--use-flashinfer-decode",
                             "--compiled-sampler"],
}


def cell_env():
    env = dict(os.environ)
    env.update({
        "PATH": f"/work/nvme/bcjw/msajid/conda/opsd/bin:{env.get('PATH', '')}",
        "PYTHONPATH": f"{DS}:{OPSD}:{env.get('PYTHONPATH', '')}",
        "HF_HOME": "/work/nvme/bcjw/msajid/hf_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/u/msajid/torchinductor",
        "TORCH_EXTENSIONS_DIR": "/u/msajid/torch_extensions",
        "TMPDIR": "/u/msajid/tmp",
        "TORCH_CUDA_ARCH_LIST": "8.0",
        "TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
    })
    return env


def cell_cmd(model, task, level):
    t = TASKS[task]
    c = t["c"]
    extra = LEVELS[level]
    if extra is None:  # B: single graph at the full continuous width
        extra = ["--graph-capture-batch-sizes", str(c)]
    return [
        PY, str(BENCH),
        "--model", MODELS[model],
        "--output", str(out_path(model, task, level)),
        "--data", t["data"],
        "--batch-size", str(t["batch"]),
        "--continuous-batch-size", str(c),
        "--max-prompt-length", str(t["prompt"]),
        "--max-new-tokens", str(t["new"]),
        "--temperature", f"{t['temperature']:.1f}",
        "--top-p", f"{t['top_p']:.1f}",
        "--top-k", "-1",
        "--seed", "0",
        "--warmup", str(WARMUP),
        "--iterations", str(ITERATIONS),
        "--save-response-ids",
        *extra,
    ]


def out_path(model, task, level):
    return RAW / f"{model}_{task}_{level}.json"


def is_done(model, task, level):
    p = out_path(model, task, level)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
        return bool(data.get("iterations"))
    except Exception:
        return False


def run_cell(model, task, level):
    out = out_path(model, task, level)
    log = LOGS / f"{model}_{task}_{level}.log"
    cmd = cell_cmd(model, task, level)
    t0 = time.time()
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, cwd=OPSD, env=cell_env(),
                              stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    ok = proc.returncode == 0 and is_done(model, task, level)
    with open(OUT / "status.jsonl", "a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model,
                             "task": task, "level": level, "rc": proc.returncode,
                             "elapsed_s": round(elapsed, 1), "ok": ok,
                             "log": str(log)}) + "\n")
    print(f"[{'done' if ok else 'FAIL'}] {model}/{task}/{level} rc={proc.returncode} "
          f"{elapsed/60:.1f}min log={log.name}", flush=True)
    return ok


def load_rows():
    rows = []
    for p in sorted(RAW.glob("*.json")):
        model = next((m for m in MODELS if p.stem.startswith(m + "_")), None)
        if model is None:
            continue
        rest = p.stem[len(model) + 1:]
        task, level = rest.split("_", 1)
        if task not in TASKS or level not in LEVELS:
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        iters = d.get("iterations") or []
        if not iters:
            continue
        prof = iters[0].get("profile", {})
        wall = statistics.median([i["wall_ms"] for i in iters])
        prefill = statistics.median([i["profile"].get("prefill_forward_ms") for i in iters if i["profile"].get("prefill_forward_ms") is not None])
        decode = statistics.median([i["profile"].get("decode_forward_ms") for i in iters if i["profile"].get("decode_forward_ms") is not None])
        tok_s = statistics.median([i["profile"].get("tokens_per_second") for i in iters if i["profile"].get("tokens_per_second") is not None])
        cs = iters[0].get("continuous_stats", {})
        ladder = d.get("graph_capture_batch_sizes") or (d.get("warmup_stats") or {}).get("graph_capture_ladder")
        rows.append({
            "model": model, "task": task, "level": level,
            "wall_ms": wall, "prefill_ms": prefill, "decode_ms": decode,
            "tok_s": tok_s,
            "peak_mb": max(d.get("peak_memory_mb", 0.0), d.get("warmup_peak_memory_mb", 0.0)),
            "graph_capture": bool(d.get("graph_capture")),
            "ladder": ladder,
            "response_ids": d.get("response_ids"),
            "paged": bool(d.get("use_paged_cache")),
            "paged_prefill": bool(d.get("enable_paged_prefill")),
            "torch_compile": bool(d.get("torch_compile")),
            "fi_prefill": bool(d.get("flashinfer_prefill")),
            "fi_decode": bool(d.get("use_flashinfer_decode")),
            "compiled_sampler": bool(d.get("compiled_sampler")),
            "sha": (d.get("response_sha256") or iters[0].get("response_sha256", ""))[:8],
            "graph_captures": [i.get("continuous_stats", {}).get("graph_captures") for i in iters],
            "bucket_widths": [i.get("continuous_stats", {}).get("decode_bucket_widths") for i in iters],
        })
    return rows


def fmt(v, digits=1):
    return "" if v is None else f"{v:.{digits}f}"


def token_agreement(base_rows, other_rows):
    """Fraction of rows identical to base, plus the shortest common-prefix length."""
    if not base_rows or not other_rows:
        return "n/a"
    same = 0
    prefix = None
    for b, o in zip(base_rows, other_rows):
        if b == o:
            same += 1
        else:
            p = next((i for i, (x, y) in enumerate(zip(b, o)) if x != y), min(len(b), len(o)))
            prefix = p if prefix is None else min(prefix, p)
    if same == len(base_rows):
        return f"{same}/{len(base_rows)}"
    return f"{same}/{len(base_rows)} (min prefix {prefix})"


def summarize(rows):
    levels = list(LEVELS)
    md = ["# Continuous-batching rollout ablation — optimization-merged branch", ""]
    md.append(f"Date: 2026-11-12. Node: A100-SXM4-40GB (gpua021). torch 2.15.0+cu132, "
              f"transformers 5.22.2, deepspeed from `{DS}` (branch `fix/paged-layer-lazy-init` = "
              f"`fork/optimization-merged` + lazy-init fix), "
              f"flashinfer 0.7.4, dskernels (blocked-flash JIT). bf16, seed 0, warmup 1, "
              f"one measured iteration per cell, chunked prefill off everywhere, "
              f"`--top-k -1`, `--save-response-ids`.")
    md.append("")
    md.append("**Rerun note (2026-11-12):** the original D/E/F cells ran on "
              "`fork/optimization-merged`, where every paged graph capture raised "
              "`AttributeError: '_PagedLayer' object has no attribute 'is_initialized'` "
              "(the merged engine's lazy-init checks had no counterpart in the up-front-allocated "
              "`paged_cache.py`) and silently fell back to eager. All 12 D/E/F cells were "
              "re-run on 2026-11-12 on `fix/paged-layer-lazy-init` (`00d4ff3b7`); A/B/C are "
              "unaffected (dense paths never touch the paged lazy-init code) and are kept as "
              "originally measured. Pre-fix D/E/F numbers are preserved in "
              "`pre_lazy_init_fix/`.")
    md.append("")
    md.append("## Levels")
    md.append("")
    md.append("| Level | Definition | Flags |")
    md.append("|---|---|---|")
    md.append("| A | continuous batching + eager | `--no-graph-capture` |")
    md.append("| B | continuous batching + single graph | `--graph-capture-batch-sizes <C>` (one graph at the full continuous width) |")
    md.append("| C | continuous batching + multiple graphs | default bucket ladder |")
    md.append("| D | C + paged attention, prefill + decode together | `--use-paged-cache --enable-paged-prefill` |")
    md.append("| E | D + fused kernels | `--torch-compile` |")
    md.append("| F | D + FlashInfer kernels + compiled sampler | `--flashinfer-prefill --use-flashinfer-decode --compiled-sampler` |")
    md.append("")
    md.append("F does not stack `--torch-compile` on E: the FlashInfer backends and inductor "
              "are alternative kernel strategies (FlashInfer custom ops are graph breaks), "
              "consistent with their separate isolation in `runs/optimization_ablation.md`.")
    md.append("")
    md.append("## TL;DR")
    md.append("")
    by2 = {}
    for r in rows:
        by2.setdefault((r["model"], r["task"]), {})[r["level"]] = r
    for model in MODELS:
        for task in TASKS:
            rows_t = by2.get((model, task))
            if not rows_t:
                continue
            a, b, c, d, e, f = (rows_t.get(l) for l in LEVELS)
            def ratio(r_, base):
                """base/new: >1 means r_ is faster than base."""
                if r_ is None or base is None or not r_["wall_ms"] or not base["wall_ms"]:
                    return None
                return base["wall_ms"] / r_["wall_ms"]
            def dratio(r_, base):
                if r_ is None or base is None or not r_["decode_ms"] or not base["decode_ms"]:
                    return None
                return base["decode_ms"] / r_["decode_ms"]
            parts = [f"**{model} {task}**"]
            if a and d:
                parts.append(f"D (paged, multi-graph) vs A (eager): {fmt(ratio(d, a), 2)}x total, "
                             f"{fmt(dratio(d, a), 2)}x decode")
            if b and c:
                parts.append(f"C (ladder) vs B (single graph): {fmt(ratio(c, b), 2)}x total")
            if d and e:
                parts.append(f"E (torch.compile) vs D: {fmt(ratio(e, d), 2)}x total")
            if d and f:
                parts.append(f"F (FlashInfer + compiled sampler) vs D: {fmt(ratio(f, d), 2)}x total")
            md.append("- " + "; ".join(parts))
    md.append("")
    for model in MODELS:
        for task in TASKS:
            t = TASKS[task]
            rows_t = {r["level"]: r for r in rows if r["model"] == model and r["task"] == task}
            if not rows_t:
                continue
            md.append(f"## {model} — {task} (batch {t['batch']}, C={t['c']}, prompt {t['prompt']}, "
                      f"{t['new']} new, T={t['temperature']}, top_p={t['top_p']})")
            md.append("")
            base = rows_t.get(levels[0])
            md.append("| Level | Prefill ms | Decode ms | Total ms | Tok/s | Peak MiB | Response SHA | Tokens vs A |")
            md.append("|---|---:|---:|---:|---:|---:|---|---|")
            for lv in levels:
                r = rows_t.get(lv)
                if r is None:
                    md.append(f"| {lv} | — | — | — | — | — | — | — |")
                    continue
                agree = ("—" if (lv == levels[0] or base is None or not r.get("response_ids"))
                         else token_agreement(base["response_ids"], r["response_ids"]))
                md.append(f"| {lv} | {fmt(r['prefill_ms'])} | {fmt(r['decode_ms'])} | "
                          f"{fmt(r['wall_ms'])} | {fmt(r['tok_s'])} | {fmt(r['peak_mb'], 0)} | "
                          f"`{r['sha']}` | {agree} |")
            md.append("")
            md.append("Increments (total, decode):")
            md.append("")
            md.append("| Step | Total speedup | Decode speedup | Peak delta |")
            md.append("|---|---:|---:|---|")
            chain = [(levels[0], "A (eager baseline)"), (levels[1], "B (single graph) vs A"),
                     (levels[2], "C (bucket ladder) vs B"), (levels[3], "D (paged) vs C"),
                     (levels[4], "E (torch.compile) vs D"), (levels[5], "F (FlashInfer + compiled sampler) vs D")]
            for lv, label in chain:
                r = rows_t.get(lv)
                if lv == levels[0]:
                    md.append(f"| {label} | 1.00x | 1.00x | — |")
                    continue
                if r is None:
                    md.append(f"| {label} | — | — | — |")
                    continue
                prev_lv = {levels[1]: levels[0], levels[2]: levels[1], levels[3]: levels[2],
                           levels[4]: levels[3], levels[5]: levels[3]}[lv]
                prev = rows_t.get(prev_lv)
                if not prev:
                    md.append(f"| {label} | — | — | — |")
                    continue
                tot = prev["wall_ms"] / r["wall_ms"] if r["wall_ms"] and prev["wall_ms"] else None
                dec = prev["decode_ms"] / r["decode_ms"] if r["decode_ms"] and prev["decode_ms"] else None
                dpeak = (r["peak_mb"] - prev["peak_mb"]) / prev["peak_mb"] * 100.0 if prev["peak_mb"] else None
                md.append(f"| {label} | {fmt(tot, 2)}x | {fmt(dec, 2)}x | "
                          f"{'+' if dpeak >= 0 else ''}{fmt(dpeak, 1)}% |")
            md.append("")
    md.append("## Path proof")
    md.append("")
    md.append("| model | task | level | graph capture | ladder | captures | decode bucket widths |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['model']} | {r['task']} | {r['level']} | "
                  f"{'yes' if r['graph_capture'] else 'no'} | {r['ladder']} | "
                  f"{r['graph_captures']} | {r['bucket_widths']} |")
    md.append("")
    md.append("## Caveats")
    md.append("")
    md.append("- **Tokens vs A:** the 6k rows use Gumbel sampling (T=0.7), which amplifies bf16 "
              "ULP differences into different sampled tokens: graph capture (B/C), paged attention "
              "(D/E/F), and even the batched dense prefill (batch 4 vs the old batch-1 shape) can "
              "flip boundary tokens. The 20k greedy rows are the stability check: B/C stay exact vs A; "
              "D/E/F can differ on one row (known paged/compile ULP behavior, see "
              "`runs/optimization_ablation.md` §Paged decode). At 4B the D/E/F rows share one digest; "
              "at 8B even paged-family paths can flip boundary tokens against each other.")
    md.append("- **C vs B = 1.00x is expected here:** every run drains the full continuous width, so "
              "the bucket ladder ([1,2,4] / [1,2]) replays at its top bucket — identical work to a "
              "single graph captured at the full width. The ladder only pays off when the active batch "
              "width varies over the run (see the old bucket-ladder results in "
              "`runs/optimization_ablation.md` §CUDA graphs and the bucket ladder).")
    md.append("- **Peaks:** graph rows (B/C) peak above eager because the captured static cache/KV "
              "buffers stay allocated; paged rows (D/E/F) run on the block pool and peak lower. E adds "
              "the inductor workspace; treat torch.compile as a latency option, not a memory option.")
    md.append("- **Single measured iteration** after one warmup (autotune for E is inside the warmup). "
              "Directional, per the source ablation's convention.")
    md.append("")
    done = sum(1 for m in MODELS for t in TASKS for lv in LEVELS if is_done(m, t, lv))
    total = len(MODELS) * len(TASKS) * len(LEVELS)
    md.append(f"_Status: {done}/{total} cells complete. Raw JSONs in `raw/`, logs in `logs/`. "
              f"Regenerate with `python scripts/run_cb_ablation.py` (skips finished cells, rewrites this file)._")
    md.append("")
    (OUT / "cb_ablation.md").write_text("\n".join(md) + "\n")
    fields = ["model", "task", "level", "wall_ms", "prefill_ms", "decode_ms", "tok_s",
              "peak_mb", "sha"]
    with open(OUT / "cb_ablation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"wrote {OUT/'cb_ablation.md'} ({done}/{total} cells)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", default="",
                        help='cell spec "model,task,level"; multiple specs separated by | or spaces (prefixes ok)')
    parser.add_argument("--force", default="",
                        help='cell specs to re-run even if done; separated by | or spaces')
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--summary-only", action="store_true",
                        help="regenerate md/csv from existing raw JSONs; do not run any cell")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.summary_only:
        RAW.mkdir(parents=True, exist_ok=True)
        summarize(load_rows())
        return

    def specs(text):
        return [s for s in re.split(r"[\s|]+", text) if s]

    only = specs(args.only)
    force = specs(args.force)

    cells = []
    for model in MODELS:
        for task in TASKS:
            for level in LEVELS:
                key = f"{model},{task},{level}"
                if only and not any(key.startswith(o) for o in only):
                    continue
                cells.append((model, task, level, key in force))

    for model, task, level, f in cells:
        state = "redo" if f else ("done" if is_done(model, task, level) else "run")
        print(f"[{state:4}] {model} / {task} / {level}", flush=True)
    if args.list:
        return

    RAW.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    for model, task, level, f in cells:
        if not f and is_done(model, task, level):
            continue
        if args.dry_run:
            print(shlex.join(cell_cmd(model, task, level)))
            continue
        run_cell(model, task, level)
    summarize(load_rows())


if __name__ == "__main__":
    main()
