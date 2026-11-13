# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Gap analysis between two torch-profiler chrome traces (DS vs vLLM).

Usage: analyze_prof.py TRACE_A TRACE_B [LABEL_A] [LABEL_B]

For each trace prints: measured wall time (from the trace), total CUDA kernel
time, GPU busy %, and the top-20 CUDA kernels by total time (self time),
bucketed into attention / GEMM / sampler / elementwise / other.
"""

import json
import sys
import gzip
from pathlib import Path


def resolve(path):
    p = Path(path)
    if p.is_dir():
        cands = list(p.glob("*.json")) + list(p.glob("*.json.gz"))
        if not cands:
            raise SystemExit(f"no trace in {p}")
        return max(cands, key=lambda x: x.stat().st_size)
    return p


def load_cuda_events(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path) as f:
        tr = json.load(f)
    evs = []
    for e in tr.get("traceEvents", []):
        if e.get("cat") == "kernel":
            ts, dur = e.get("ts"), e.get("dur", 0)
            if ts and dur:
                evs.append((ts, dur, e.get("name", "")))
    return evs


def classify(name):
    n = name.lower()
    if any(k in n for k in ("fmha", "flash", "flashinfer", "attention", "attn",
                            "paged", "mha", "mla")):
        return "attention"
    if any(k in n for k in ("gemm", "cutlass", "wgmma", "nvjet", "sgemmv",
                            "hgemm", "matmul", "cublas", "gemv")):
        return "gemm"
    if any(k in n for k in ("sampl", "gumbel", "topk", "top_k", "softmax_sampl",
                            "argmax", "multinomial", "randint", "philox")):
        return "sampler"
    if any(k in n for k in ("elementwise", "unary", "binary", "copy", "cat",
                            "split", "slice", "contiguous", "transpose",
                            "reduce", "sum", "norm", "rsqrt", "silu",
                            "rotary", "rope", "embed", "index")):
        return "elementwise/move"
    return "other"


def report(title, path, wall_ms=None):
    evs = load_cuda_events(path)
    if not evs:
        print(f"{title}: no CUDA kernel events in {path}")
        return
    total_us = sum(d for _, d, _ in evs)
    # GPU busy %: union of kernel intervals / span
    iv = sorted((t, t + d) for t, d, _ in evs)
    busy, s, e = 0, iv[0][0], iv[0][1]
    for a, b in iv[1:]:
        if a <= e:
            e = max(e, b)
        else:
            busy += e - s
            s, e = a, b
    busy += e - s
    span = iv[-1][1] - iv[0][0]
    print(f"== {title} ({path})")
    print(f"   kernel events: {len(evs)}  total kernel time: {total_us/1e3:8.1f} ms"
          f"  span: {span/1e3:8.1f} ms"
          f"  GPU busy: {100*busy/span:5.1f}%")
    if wall_ms and wall_ms * 1000 > span:
        idle = 100.0 * (1.0 - busy / 1e3 / wall_ms)
        print(f"   measured wall_ms (bench): {wall_ms:8.1f} ms"
              f"  -> GPU idle within wall: {idle:4.1f}%")
    agg, agg_count = {}, {}
    for _, d, name in evs:
        agg[name] = agg.get(name, 0) + d
        agg_count[name] = agg_count.get(name, 0) + 1
    top = sorted(agg.items(), key=lambda kv: -kv[1])[:20]
    print("   top-20 kernels (total us, count):")
    for name, tot in top:
        print(f"   {tot/1e3:9.2f} ms x{agg_count[name]:6d}  "
              f"[{classify(name):14s}] {name[:96]}")
    buckets = {}
    for _, d, name in evs:
        buckets[classify(name)] = buckets.get(classify(name), 0) + d
    print("   buckets:")
    for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"   {100*v/total_us:5.1f}%  {v/1e3:9.2f} ms  {k}")
    print()


if __name__ == "__main__":
    a, b = resolve(sys.argv[1]), resolve(sys.argv[2])
    la = sys.argv[3] if len(sys.argv) > 3 else "A"
    lb = sys.argv[4] if len(sys.argv) > 4 else "B"
    wa = float(sys.argv[5]) if len(sys.argv) > 5 else None
    wb = float(sys.argv[6]) if len(sys.argv) > 6 else None
    report(la, a, wa)
    report(lb, b, wb)
