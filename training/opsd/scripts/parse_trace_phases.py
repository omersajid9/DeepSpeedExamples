#!/usr/bin/env python3
"""Phase-split analysis of a torch chrome trace (Kineto).

Splits GPU kernel time into prefill vs decode using cudaGraphLaunch boundaries:
  --ds   : every forward is a graph replay -> boundary = start of 2nd graph launch
  --vllm : prefill is eager, decode is graph replay -> boundary = start of 1st launch
Also reports inter-kernel gaps (CPU/launch overhead) per phase.
"""
import json
import re
import sys

CATEGORIES = [
    ("flash-attn", re.compile(r"flash_fwd|flashinfer", re.I)),
    ("paged/blocked attn", re.compile(r"paged|blocked|block_table|kv_cache|decode_attn", re.I)),
    ("GEMM cublas/cutlass", re.compile(r"gemm|gemvx|cublas|cutlass|nvjet", re.I)),
    ("triton GEMM/MLP", re.compile(r"triton_tem_fused.*mm", re.I)),
    ("triton norm/act", re.compile(r"triton_(red|poi)_fused", re.I)),
    ("elementwise/copy", re.compile(r"elementwise|copy|CatArray|index_element|vectorized", re.I)),
    ("reduce", re.compile(r"reduce_kernel|ReduceOp", re.I)),
    ("sampling", re.compile(r"gumbel|topk|top_k|sample|multinomial|argmax", re.I)),
]


def classify(name):
    for label, rx in CATEGORIES:
        if rx.search(name):
            return label
    return "other"


def main():
    trace_file, mode = sys.argv[1], sys.argv[2]
    trace = json.load(open(trace_file))
    events = trace["traceEvents"] if isinstance(trace, dict) else trace

    kernels = [e for e in events if e.get("cat") == "kernel" and "ts" in e and "dur" in e]
    launches = [e for e in events if e.get("ph") == "X" and e.get("name") == "cudaGraphLaunch"
                and "ts" in e]
    launches.sort(key=lambda e: e["ts"])
    kernels.sort(key=lambda e: e["ts"])

    total_us = sum(e["dur"] for e in kernels)
    print(f"{trace_file}: {len(kernels)} kernel instances, {total_us / 1000:.1f} ms total, "
          f"{len(launches)} cudaGraphLaunch events")

    # Phase boundary: end of the last prefill-scale kernel (dur >= threshold).
    # Works for both engines: vLLM v2 replays prefill GEMMs from CUDA graphs, so
    # graph-launch position is not a reliable prefill/decode boundary.
    threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    big = [e for e in kernels if e["dur"] >= threshold]
    if not big:
        print("no prefill-scale kernels; aborting")
        return
    boundary = big[-1]["ts"] + big[-1]["dur"]
    phases = {}
    for e in kernels:
        ph = "prefill" if e["ts"] < boundary else "decode"
        phases.setdefault(ph, []).append(e)

    gap_stats = {}
    for ph, ks in phases.items():
        gaps = []
        for a, b in zip(ks, ks[1:]):
            g = b["ts"] - (a["ts"] + a["dur"])
            if g > 0:
                gaps.append(g)
        if gaps:
            gaps.sort()
            gap_stats[ph] = (sum(gaps), max(gaps), gaps[len(gaps) // 2], len(gaps))

    for ph in ("prefill", "decode"):
        ks = phases.get(ph, [])
        tot = sum(e["dur"] for e in ks)
        print(f"\n== {ph}: {tot / 1000:.1f} ms kernel time ({100 * tot / total_us:.0f}% of total), "
              f"{len(ks)} kernels")
        by_cat = {}
        by_name = {}
        for e in ks:
            c = classify(e["name"])
            by_cat[c] = by_cat.get(c, 0) + e["dur"]
            by_name[e["name"]] = by_name.get(e["name"], [0, 0])
            by_name[e["name"]][0] += e["dur"]
            by_name[e["name"]][1] += 1
        for c, us in sorted(by_cat.items(), key=lambda kv: -kv[1]):
            print(f"   {100 * us / tot:5.1f}%  {us / 1000:8.1f} ms  {c}")
        print("   top kernels:")
        for name, (us, n) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[:8]:
            print(f"     {us / 1000:8.2f} ms  x{n:<6d} avg {us / n:8.1f} us  {name[:75]}")
        if ph in gap_stats:
            s, mx, md, n = gap_stats[ph]
            print(f"   inter-kernel gaps: total {s / 1000:.1f} ms over {n} gaps, median {md:.1f} us, max {mx / 1000:.1f} ms")


if __name__ == "__main__":
    main()
