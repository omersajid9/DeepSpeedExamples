#!/usr/bin/env python3
"""Parse profile_torch_*.events.json into kernel categories + prefill/decode split.

Usage: python3 parse_profile.py <events.json> [wall_ms]
"""
import json
import re
import sys

KERNEL_NAME_RE = re.compile(
    r"^(void |triton_|ampere_|cutlass|nvjet|sm80|sm90|gemvx|flash_|"
    r"std::|at::native::(elementwise|vectorized|index|reduce|CatArray|gpu_|unrolled)|"
    r"kernel|batch_|persistent)", re.X)

CATEGORIES = [
    ("flash-attn (prefill/decode)", re.compile(r"flash_fwd|flashinfer", re.I)),
    ("paged/blocked attn", re.compile(r"paged|blocked|block_table|kv_cache|decode_attn", re.I)),
    ("GEMM (cublas/cutlass)", re.compile(r"gemm|gemvx|cublas|cutlass|nvjet|sgemm|hgemm", re.I)),
    ("triton GEMM/fused-mlp", re.compile(r"triton_tem_fused.*mm|triton_poi_fused.*mm", re.I)),
    ("triton fused norm/act", re.compile(r"triton_(red|poi)_fused", re.I)),
    ("elementwise/copy", re.compile(r"elementwise|copy|CatArray|index_element|vectorized", re.I)),
    ("reduce", re.compile(r"reduce_kernel|ReduceOp", re.I)),
    ("sampling", re.compile(r"gumbel|topk|top_k|topk_gpu|sample|multinomial|argmax", re.I)),
    ("graph", re.compile(r"Graph|graph", re.I)),
]


def classify(name):
    for label, rx in CATEGORIES:
        if rx.search(name):
            return label
    return "other"


def main():
    events = json.load(open(sys.argv[1]))
    wall_ms = float(sys.argv[2]) if len(sys.argv) > 2 else None

    device = [e for e in events if e["cpu_total_us"] == 0 and e["cuda_self_us"] > 0]
    cpu_ops = [e for e in events if e["cpu_total_us"] > 0]

    total_kernel_us = sum(e["cuda_self_us"] for e in device)
    by_name = {}
    for e in device:
        by_name.setdefault(e["name"], {"us": 0.0, "count": 0})
        by_name[e["name"]]["us"] += e["cuda_self_us"]
        by_name[e["name"]]["count"] += e["count"]

    by_cat = {}
    for name, v in by_name.items():
        c = classify(name)
        by_cat.setdefault(c, {"us": 0.0, "kernels": {}})
        by_cat[c]["us"] += v["us"]
        by_cat[c]["kernels"][name] = (v["us"], v["count"])

    print(f"== {sys.argv[1]} ==")
    print(f"total leaf-kernel self time: {total_kernel_us / 1000:.1f} ms"
          + (f" (wall {wall_ms:.1f} ms -> GPU busy {100 * total_kernel_us / 1e3 / wall_ms:.0f}%)" if wall_ms else ""))
    print("\n-- by category --")
    for c, v in sorted(by_cat.items(), key=lambda kv: -kv[1]["us"]):
        print(f"{100 * v['us'] / total_kernel_us:5.1f}%  {v['us'] / 1000:9.1f} ms  {c}")
    print("\n-- top 20 kernels --")
    for name, v in sorted(by_name.items(), key=lambda kv: -kv[1]["us"])[:20]:
        print(f"{v['us'] / 1000:9.2f} ms  x{v['count']:<6d} {name[:80]}")

    # prefill vs decode split via CPU-op input shapes (M dim)
    shape_us = {"prefill": 0.0, "decode": 0.0, "noshape": 0.0}
    shape_examples = {}
    for e in cpu_ops:
        m = 0
        for s in e["input_shapes"]:
            if s and isinstance(s[0], (tuple, list)) and s[0] and isinstance(s[0][0], int):
                m = max(m, s[0][0])
        if m == 0:
            shape_us["noshape"] += e["cuda_self_us"]
        elif m >= 512:
            shape_us["prefill"] += e["cuda_self_us"]
            shape_examples.setdefault("prefill", []).append((e["cuda_self_us"], e["name"], m))
        else:
            shape_us["decode"] += e["cuda_self_us"]
            shape_examples.setdefault("decode", []).append((e["cuda_self_us"], e["name"], m))
    print("\n-- CPU-op-attributed device time by M dim --")
    for k in ("prefill", "decode", "noshape"):
        print(f"{k:8s}: {shape_us[k] / 1000:9.1f} ms")
    for k in ("prefill", "decode"):
        top = sorted(shape_examples.get(k, []), reverse=True)[:6]
        print(f"   top {k}:")
        for us, name, m in top:
            print(f"     {us / 1000:8.2f} ms  M={m:<6d} {name[:60]}")


if __name__ == "__main__":
    main()
