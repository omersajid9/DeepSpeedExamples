# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Torch-profiler driver for the best-optimized DeepSpeed rollout (A100).

Reuses the exact setup of ``bench_continuous_graph.py`` for the "best" config
(paged cache + paged prefill + bucket CUDA graphs + torch.compile) and wraps a
single measured iteration in ``torch.profiler``.

Modes:
  default            : warmup + one measured iteration under torch.profiler
  --no-torch-profile : warmup + measured iteration only (for running under nsys/ncu)
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.profiler import ProfilerActivity, profile

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.rollout.base import SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import (
    HybridEngineRollout,
    HybridEngineRolloutConfig,
)

from bench_continuous_graph import _load_model, _load_request, _response_ids, _response_digest

WORKLOADS = {
    "6k": dict(prompt=6144, new=16, temp=0.7, top_p=0.9),
    "20k": dict(prompt=20480, new=8, temp=0.0, top_p=1.0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="6k")
    parser.add_argument("--out", required=True, help="base name; .trace.json/.events.json/.summary.txt are written")
    parser.add_argument("--no-graph-capture", dest="graph_capture", action="store_false")
    parser.add_argument("--no-torch-profile", action="store_true")
    parser.add_argument("--use-flashinfer-decode", action="store_true")
    parser.add_argument("--compiled-sampler", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    wl = WORKLOADS[args.workload]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(get_accelerator().current_device_name())

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, _ = _load_model(args.model, "auto")
    model.to(device)
    request = _load_request(args.data, 1, tokenizer, wl["prompt"], device)

    rollout = HybridEngineRollout(
        SimpleNamespace(module=model),
        tokenizer,
        cfg=HybridEngineRolloutConfig(
            use_graph_capture=args.graph_capture,
            enable_profiling=True,
            use_paged_cache=True,
            enable_paged_prefill=True,
            enable_torch_compile=True,
            use_flashinfer_decode=args.use_flashinfer_decode,
            enable_compiled_sampler=args.compiled_sampler,
        ),
    )
    sampling = SamplingConfig(
        max_new_tokens=wl["new"],
        temperature=wl["temp"],
        top_p=wl["top_p"],
        top_k=-1,
        n_samples_per_prompt=1,
        continuous_batch_size=1,
    )

    with torch.no_grad():
        rollout.generate(request, sampling)  # warmup (compile + capture + caches)
        get_accelerator().synchronize()

        if not args.no_torch_profile:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True) as prof:
                batch = rollout.generate(request, sampling)
                get_accelerator().synchronize()
                start_note = "torch-profile"
        else:
            t0 = time.perf_counter()
            batch = rollout.generate(request, sampling)
            get_accelerator().synchronize()
            start_note = f"wall {(time.perf_counter() - t0) * 1000.0:.1f} ms"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    profile_dict = dict(rollout.get_last_profile() or {})
    rows = _response_ids(batch)
    result = {
        "workload": args.workload,
        "response_sha256": _response_digest(rows),
        "engine_profile": profile_dict,
        "note": start_note,
    }
    (out.with_suffix(".result.json")).write_text(json.dumps(result, indent=2) + "\n")
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

        lines = [f"# top ops by self CUDA time ({args.workload}, best config)"]
        lines += [fmt(e) for e in events[:30]]
        out.with_suffix(".summary.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines[:35]))


if __name__ == "__main__":
    main()
