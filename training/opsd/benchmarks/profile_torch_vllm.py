# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Torch-profiler driver for the vLLM no-chunk reference (A100).

Same workload/params as ``bench_vllm_offline.py``. Run with
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so the v1 engine core stays in-process and
Kineto can capture its kernels.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from bench_vllm_offline import _load_prompts, _digest

WORKLOADS = {
    "6k": dict(prompt=6144, new=16, temp=0.7, top_p=0.9, chunk=8192, kv_gib=1.1),
    "20k": dict(prompt=20480, new=8, temp=0.0, top_p=1.0, chunk=20488, kv_gib=3.3),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="6k")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-torch-profile", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="disable cudagraphs/torch.compile (for ncu)")
    parser.add_argument("--enable-prefix-caching", action="store_true",
                        help="default OFF to match the official bench reference")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    wl = WORKLOADS[args.workload]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = _load_prompts(args.data, 1, tokenizer, wl["prompt"])
    prompts = [{"prompt_token_ids": row} for row in ids]

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=wl["prompt"] + wl["new"],
        max_num_seqs=1,
        max_num_batched_tokens=wl["chunk"],
        gpu_memory_utilization=0.75,
        kv_cache_memory_bytes=int(wl["kv_gib"] * (1024 ** 3)),
        disable_log_stats=True,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    sampling = SamplingParams(
        max_tokens=wl["new"],
        temperature=wl["temp"],
        top_p=wl["top_p"],
        top_k=-1,
        seed=args.seed,
        ignore_eos=True,
    )

    with torch.no_grad():
        llm.generate(prompts, sampling)  # warmup (AOT compile + caches)
        torch.cuda.synchronize()

        if not args.no_torch_profile:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True) as prof:
                outputs = llm.generate(prompts, sampling)
                torch.cuda.synchronize()
                note = "torch-profile"
        else:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sampling)
            torch.cuda.synchronize()
            note = f"wall {(time.perf_counter() - t0) * 1000.0:.1f} ms"

    rows = [list(o.outputs[0].token_ids) for o in outputs]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"workload": args.workload, "response_sha256": _digest(rows), "note": note}
    out.with_suffix(".result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    if not args.no_torch_profile:
        prof.export_chrome_trace(str(out.with_suffix(".trace.json")))
        events = []
        for ev in prof.key_averages():
            if ev.self_device_time_total <= 0 and ev.self_cpu_time_total <= 0:
                continue
            events.append({
                "name": ev.key,
                "count": ev.count,
                "cpu_total_us": ev.cpu_time_total,
                "cpu_self_us": ev.self_cpu_time_total,
                "cuda_self_us": ev.self_device_time_total,
                "cuda_total_us": ev.device_time_total,
                "input_shapes": ev.input_shapes[:8] if ev.input_shapes else [],
            })
        events.sort(key=lambda e: -e["cuda_self_us"])
        out.with_suffix(".events.json").write_text(json.dumps(events, indent=1) + "\n")

        def fmt(e):
            shapes = ";".join(",".join(map(str, s)) for s in e["input_shapes"][:3])
            return (f"{e['cuda_self_us'] / 1000:10.2f} ms  x{e['count']:<5d} "
                    f"{e['name'][:70]:70s} {shapes[:60]}")

        lines = [f"# top ops by self CUDA time ({args.workload}, vLLM)"]
        lines += [fmt(e) for e in events[:30]]
        out.with_suffix(".summary.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines[:35]))


if __name__ == "__main__":
    main()
