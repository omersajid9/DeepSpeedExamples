# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Checkout each optimization level and run the rollout ablation matrix."""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

OPSD = Path(__file__).resolve().parent.parent
DEEPSPEED = Path("/workspace/DeepSpeed")
OUT = OPSD / "runs" / "optimization_ablation_2026-11-10"

ARMS = [
    ("single_graph", "dc046b15d7a58f6c8dacba435508b52b1ebc620c"),
    ("eager_sampling", "dc046b15d7a58f6c8dacba435508b52b1ebc620c"),
    ("multiple_graphs", "c9c92c3d"),
    ("device_sampler", "ea21bf01"),
    ("static_cache", "1ac2c883"),
    ("all_greedy", "72124505"),
    ("all_sampling", "72124505"),
]
TASKS = ["6k6k", "retool", "longbench"]


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value, digits=2):
    return "" if value is None else f"{value:.{digits}f}"


def git_checkout(ref):
    subprocess.run(["git", "checkout", ref], cwd=DEEPSPEED, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def run_cell(task, arm, out_path):
    log_path = OUT / "logs" / f"{task}__{arm}.log"
    cmd = [sys.executable, str(OPSD / "scripts" / "run_rollout_cell.py"),
           "--task", task, "--arm", arm, "--out", str(out_path)]
    env = dict(os.environ)
    env["HF_HOME"] = "/dev/shm/hf_cache"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{OPSD}:{env.get('PYTHONPATH', '')}"
    print(f"[cell] {task}/{arm}", flush=True)
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, cwd=OPSD, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[FAIL] {task}/{arm} exit={proc.returncode}", flush=True)
        return None
    try:
        data = json.loads(out_path.read_text())
    except Exception as exc:
        print(f"[FAIL] {task}/{arm} unreadable: {exc}", flush=True)
        return None
    print(f"[done] {task}/{arm}", flush=True)
    return data


def summarize(rows):
    md = ["# Rollout optimization ablation - Qwen3-4B", ""]
    md.append("Tasks: 6k prompt + 6k decode (batch 1, C=1, 1 measured iteration), "
              "ReTool (batch 16, C=3, 512 new, 3 iterations), LongBench MFQA "
              "(batch 16, C=3, 8192 prompt, 256 new, 3 iterations). "
              "Sampling arms use temperature=0.7, top_p=0.9, top_k=-1. "
              "Chunked prefill is disabled in every arm.")
    md.append("")
    md.append("| Task | Arm | Wall (s) | Prefill (s) | Decode (s) | Tok/s | Peak (MiB) | Graph | Sampling |")
    md.append("|---|---|---:|---:|---:|---:|---:|---|---|")
    for task in TASKS:
        for arm, _ in ARMS:
            row = next((r for r in rows if r["task"] == task and r["arm"] == arm), None)
            if row is None:
                continue
            md.append("| {t} | {a} | {w} | {p} | {d} | {tok} | {peak} | {g} | {s} |".format(
                t=task, a=arm, w=fmt(row["wall_s"]), p=fmt(row["prefill_s"]),
                d=fmt(row["decode_s"]), tok=fmt(row["tok_s"]), peak=fmt(row["peak_mb"], 0),
                g="yes" if row["graph"] else "no", s=fmt(row["temperature"], 1)))
    md.append("")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    fields = ["task", "arm", "wall_s", "prefill_s", "decode_s", "tok_s", "peak_mb", "graph", "temperature"]
    with open(OUT / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    print(f"wrote {OUT/'summary.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--arms", default=",".join(name for name, _ in ARMS))
    args = parser.parse_args()
    tasks = [task for task in args.tasks.split(",") if task]
    arms = [pair for pair in ARMS if pair[0] in set(args.arms.split(","))]
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)

    rows = []
    try:
        for task in tasks:
            for arm, commit in arms:
                out_path = OUT / "raw" / f"{task}__{arm}.json"
                data = json.loads(out_path.read_text()) if args.resume and out_path.exists() else None
                if data is None:
                    git_checkout(commit)
                    data = run_cell(task, arm, out_path)
                if data is None:
                    continue
                wall = median([it["wall_ms"] for it in data["iterations"]]) / 1000.0
                prefill = median([it["profile"].get("prefill_forward_ms") for it in data["iterations"]]) / 1000.0
                decode = median([it["profile"].get("decode_forward_ms") for it in data["iterations"]]) / 1000.0
                tok_s = median([it["profile"].get("tokens_per_second") for it in data["iterations"]])
                peak = max(data.get("peak_memory_mb", 0.0), data.get("warmup_peak_memory_mb", 0.0))
                rows.append({"task": task, "arm": arm, "wall_s": wall, "prefill_s": prefill,
                             "decode_s": decode, "tok_s": tok_s, "peak_mb": peak,
                             "graph": bool(data.get("graph_capture")),
                             "temperature": data.get("temperature")})
    finally:
        git_checkout("chunked-prefill-rollout")
    summarize(rows)


if __name__ == "__main__":
    main()
