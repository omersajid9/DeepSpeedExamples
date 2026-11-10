# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Run the retool A/B/C ablation at HEAD with strict per-cell metrics."""

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
OUT = OPSD / "runs" / "rollout_matrix_2026-11-10"
HEAD = "684a184c"
LEGACY = "dc046b15d7a58f6c8dacba435508b52b1ebc620c"

A = ["noncb_greedy", "noncb_greedy_graph", "noncb_sampling", "noncb_sampling_graph"]
B = ["cb_greedy", "cb_greedy_graph", "cb_greedy_graph_buckets", "cb_sampling", "cb_sampling_graph"]
C = ["cb3_greedy_graph_buckets", "cb6_greedy_graph_buckets", "cb12_greedy_graph_buckets", "cb16_greedy_graph_buckets"]
GATES = ["gate_topk1_sampling", "gate_t1_sampling"]
NOISE = ["cb_greedy_graph_buckets", "cb_greedy_graph_buckets", "cb_greedy_graph_buckets"]
LEGACY_CELLS = ["legacy_single_graph"]
FEATURES = [
    ("eager_static_cache", "1ac2c883"),
    ("eager_device_sampler", "ea21bf01"),
    ("continuous_eager_3", HEAD),
    ("continuous_eager_8", HEAD),
]


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value, digits=2):
    return "" if value is None else f"{value:.{digits}f}"


def git_checkout(ref):
    subprocess.run(["git", "checkout", ref], cwd=DEEPSPEED, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def run_cell(task, arm, out_path, tag=""):
    log = OUT / "logs" / f"{task}__{arm}{tag}.log"
    cmd = [sys.executable, str(OPSD / "scripts" / "run_rollout_cell.py"),
           "--task", task, "--arm", arm, "--out", str(out_path)]
    env = dict(os.environ)
    env["HF_HOME"] = "/dev/shm/hf_cache"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = f"{OPSD}:{env.get('PYTHONPATH', '')}"
    print(f"[cell] {task}/{arm}{tag}", flush=True)
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, cwd=OPSD, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[FAIL] {task}/{arm}{tag} exit={proc.returncode}", flush=True)
        return None
    try:
        return json.loads(out_path.read_text())
    except Exception as exc:
        print(f"[FAIL] {task}/{arm}{tag} {exc}", flush=True)
        return None


def rows_from(data, task, arm, tag=""):
    if data is None or not data.get("iterations"):
        return None
    its = data["iterations"]
    dec = median([it["profile"].get("decode_forward_ms") for it in its]) or 0.0
    fwd = median([it["profile"].get("num_decode_forwards") for it in its]) or 0
    prof0 = its[0]["profile"]
    cs0 = its[0]["continuous_stats"]
    return {
        "task": task, "arm": arm, "tag": tag,
        "ms_decode_token": dec / fwd if fwd else None,
        "decode_ms": dec,
        "gen_ms": median([it["profile"].get("generation_overhead_ms") for it in its]),
        "cache_ms": median([it["profile"].get("cache_management_overhead_ms") for it in its]),
        "sched_ms": median([it["profile"].get("scheduler_overhead_ms") for it in its]),
        "prefill_ms": median([it["profile"].get("prefill_forward_ms") for it in its]),
        "prefill_tokens": (prof0.get("prompt_length") or 0) * (data.get("batch_size") or 0),
        "peak_mb": data.get("peak_memory_mb"),
        "forwards": fwd,
        "output_len": statistics.median(data.get("output_lengths") or [0]),
        "stop": ",".join(sorted(set(data.get("stop_reasons") or []))),
        "rows": data.get("continuous_batch_size") or data.get("batch_size"),
        "graph": data.get("graph_capture"), "buckets": data.get("buckets"),
        "temp": data.get("temperature"), "top_k": data.get("top_k"),
        "logprobs_present": data.get("logprobs_present"),
        "logprobs_finite": data.get("logprobs_finite"),
        "logprobs_max": data.get("logprobs_max"),
        "graph_captures": cs0.get("graph_captures"),
        "bucket_widths": cs0.get("decode_bucket_widths"),
        "prefill_padded": cs0.get("prefill_padded_rows"),
        "keys": sorted(cs0.keys()),
        "attn": data.get("attn_implementation"),
    }


def summarize(collected):
    md = [f"# Rollout optimization ablation - HEAD {HEAD}", ""]
    md.append("## ReTool A/B/C")
    md.append("")
    md.append("| Group | Arm | Rows | ms/dec-token | decode ms | gen ms | cache ms | sched ms | prefill ms | prefill tok | peak MiB | fwd | out | stop |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for group, arms in (("A", A), ("B", B), ("C", C)):
        for arm in arms:
            row = collected.get(("retool", arm))
            if row is None:
                continue
            md.append("| {g} | {a} | {r} | {ms} | {d} | {gen} | {c} | {s} | {p} | {pt} | {peak} | {f} | {o} | {st} |".format(
                g=group, a=arm, r=row["rows"], ms=fmt(row["ms_decode_token"]), d=fmt(row["decode_ms"] / 1000),
                gen=fmt(row["gen_ms"] / 1000), c=fmt(row["cache_ms"] / 1000), s=fmt(row["sched_ms"] / 1000),
                p=fmt(row["prefill_ms"] / 1000), pt=row["prefill_tokens"], peak=fmt(row["peak_mb"], 0),
                f=row["forwards"], o=fmt(row["output_len"], 0), st=row["stop"]))
    md.append("")
    md.append("## Eager feature ablation")
    md.append("")
    md.append("| Arm | Rows | ms/dec-token | decode ms | peak MiB | out | stop | logprobs |")
    md.append("|---|---:|---:|---:|---:|---:|---|---|")
    for arm, _ in FEATURES:
        row = collected.get(("retool", arm))
        if row is None:
            continue
        md.append("| {a} | {r} | {ms} | {d} | {peak} | {o} | {st} | {lp} |".format(
            a=arm, r=row["rows"], ms=fmt(row["ms_decode_token"]), d=fmt(row["decode_ms"] / 1000),
            peak=fmt(row["peak_mb"], 0), o=fmt(row["output_len"], 0), st=row["stop"],
            lp=row["logprobs_present"]))
    md.append("")
    md.append("## Path proof")
    md.append("")
    md.append("| Arm | keys | graph_captures | bucket_widths | prefill_padded | logprobs | attn |")
    md.append("|---|---|---|---|---|---|---|")
    for arm in A + B + C:
        row = collected.get(("retool", arm))
        if row is None:
            continue
        md.append(f"| {arm} | {row['keys']} | {row['graph_captures']} | {row['bucket_widths']} | {row['prefill_padded']} | {row['logprobs_present']} | {row['attn']} |")
    md.append("")
    md.append("## Noise")
    md.append("")
    reps = [collected.get(("retool", "cb_greedy_graph_buckets", f"rep{i}")) for i in (1, 2, 3)]
    vals = [r["ms_decode_token"] for r in reps if r is not None]
    if vals:
        md.append(f"cb_greedy_graph_buckets reps ms/dec-token: {[round(v, 2) for v in vals]}; spread={(max(vals)-min(vals)):.2f}")
    md.append("")
    md.append("## Retool C=3")
    for arm in ("cb3_greedy_graph_buckets",):
        row = collected.get(("retool", arm))
        if row:
            md.append(f"{arm}: ms/dec-token={fmt(row['ms_decode_token'])}, rows={row['rows']}, peak={fmt(row['peak_mb'], 0)} MiB")
    md.append("")
    md.append("## 6k6k")
    for arm in ("noncb_greedy", "noncb_greedy_graph", "noncb_sampling", "noncb_sampling_graph",
                "cb_greedy", "cb_greedy_graph", "cb_greedy_graph_buckets", "cb_sampling", "cb_sampling_graph"):
        row = collected.get(("6k6k", arm))
        if row:
            md.append(f"{arm}: ms/dec-token={fmt(row['ms_decode_token'])}, out={fmt(row['output_len'], 0)}, stop={row['stop']}, peak={fmt(row['peak_mb'], 0)}")
    md.append("")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    fields = ["task", "arm", "tag", "ms_decode_token", "decode_ms", "gen_ms", "cache_ms", "sched_ms",
              "prefill_ms", "prefill_tokens", "peak_mb", "forwards", "output_len", "stop", "rows",
              "graph", "buckets", "temp", "logprobs_present", "graph_captures", "bucket_widths", "prefill_padded"]
    with open(OUT / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in collected.values():
            writer.writerow({key: row.get(key) for key in fields})
    print(f"wrote {OUT/'summary.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tasks", default="retool")
    args = parser.parse_args()
    tasks = [task for task in args.tasks.split(",") if task]
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    collected = {}
    try:
        git_checkout(HEAD)
        for task in tasks:
            order = A + B + C + GATES
            for arm in order:
                out_path = OUT / "raw" / f"{task}__{arm}.json"
                data = json.loads(out_path.read_text()) if args.resume and out_path.exists() else None
                if data is None:
                    data = run_cell(task, arm, out_path)
                row = rows_from(data, task, arm)
                if row:
                    collected[(task, arm)] = row
            for i, arm in enumerate(NOISE, start=1):
                out_path = OUT / "raw" / f"{task}__{arm}__rep{i}.json"
                data = json.loads(out_path.read_text()) if args.resume and out_path.exists() else None
                if data is None:
                    data = run_cell(task, arm, out_path, tag=f"__rep{i}")
                row = rows_from(data, task, arm, f"rep{i}")
                if row:
                    collected[(task, arm, f"rep{i}")] = row
            for arm in LEGACY_CELLS:
                out_path = OUT / "raw" / f"{task}__{arm}.json"
                data = json.loads(out_path.read_text()) if args.resume and out_path.exists() else None
                if data is None:
                    git_checkout(LEGACY)
                    data = run_cell(task, arm, out_path)
                    git_checkout(HEAD)
                row = rows_from(data, task, arm)
                if row:
                    collected[(task, arm)] = row
            for arm, commit in FEATURES:
                out_path = OUT / "raw" / f"{task}__{arm}.json"
                data = json.loads(out_path.read_text()) if args.resume and out_path.exists() else None
                if data is None:
                    git_checkout(commit)
                    data = run_cell(task, arm, out_path)
                    git_checkout(HEAD)
                row = rows_from(data, task, arm)
                if row:
                    collected[(task, arm)] = row
    finally:
        git_checkout(HEAD)
    summarize(collected)


if __name__ == "__main__":
    main()
