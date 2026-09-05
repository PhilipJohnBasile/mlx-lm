#!/usr/bin/env python3
"""Paired Metal benchmarks: routing, inverse-scatter ablation, and complete MoE block.

No throughput is predicted. Every timing is conditional on bitwise correctness.
Run on a quiet, plugged-in M5 machine. Results include A/A noise and A/B samples.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_case(text):
    try:
        t, k, d, e = map(int, text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use tokens,top_k,hidden_dim,experts") from exc
    if min(t, k, d, e) < 1 or k > e:
        raise argparse.ArgumentTypeError("All dimensions must be positive and top_k <= experts.")
    return t, k, d, e


def bootstrap_ratio(pairs, seed=399):
    logs = [math.log(a / b) for a, b in pairs]
    rng = random.Random(seed)
    estimates = sorted(
        math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(2000)
    )
    return {
        "geomean_baseline_over_candidate": math.exp(statistics.mean(logs)),
        "paired_bootstrap_95pct": [estimates[50], estimates[1949]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["routing", "block"], default="routing")
    parser.add_argument("--case", type=parse_case, action="append")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threadgroup-width", type=int, choices=[32, 64, 128, 256], default=256)
    parser.add_argument("--ffn", type=int, default=768)
    parser.add_argument("--bits", type=int, choices=[0, 4, 8], default=4, help="0 = dense experts")
    parser.add_argument("--allow-other-metal", action="store_true")
    parser.add_argument("--capture-dir", type=Path, help="Optional .gputrace captures; set MTL_CAPTURE_ENABLED=1 before launch")
    parser.add_argument("--output", type=Path, default=Path("single-sort-results.json"))
    args = parser.parse_args()
    if args.rounds < 5 or args.reps < 1 or args.warmup < 1:
        parser.error("Use rounds >= 5, reps >= 1, warmup >= 1.")
    if os.environ.get("MLX_MOE_SINGLE_SORT", "0") != "0":
        parser.error("Run benchmarks with MLX_MOE_SINGLE_SORT=0 so the baseline cannot use the candidate.")
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as exc:
        raise SystemExit("Native MLX is required; no simulated speed results are produced.") from exc
    if not mx.metal.is_available():
        raise SystemExit("A Metal GPU is required; CPU timings are not a substitute.")
    mx.set_default_device(mx.gpu)
    sys.path.insert(0, str(ROOT / "mlx_lm/models"))
    import _single_sort_moe as candidate

    if not candidate._m5_or_later() and not args.allow_other_metal:
        raise SystemExit("This run targets M5 and later; --allow-other-metal enables a hardware control.")
    import importlib.metadata

    metadata = {
        "device": mx.device_info(mx.gpu), "os": platform.platform(),
        "python": platform.python_version(), "mlx": mx.__version__,
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "scope": args.scope, "dtype": args.dtype,
        "threadgroup_width": args.threadgroup_width,
        "source_sha256": hashlib.sha256((ROOT / "mlx_lm/models/_single_sort_moe.py").read_bytes()).hexdigest(),
        "parameters": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        "baseline": "mlx-lm switch_layers.py blob 1fe5d917e6b194b1681bbb1c69589ad3dc759d65",
        "timing": "host wall time, new lazy graph + eval + synchronize per call; milliseconds",
        "cold_note": "first measured calls follow reference construction; these are NOT isolated cold-compile comparisons",
        "input_reuse": "resident, same-input warm microbenchmark; no cold-cache or whole-model claim",
        "block_scope": "shared real SwitchGLU projections and activation plus routing and weighted reduction; forward algebra checked against the installed class, not full-model serving or production-gate overhead",
        "status": "in_progress",
        "environment": {k: os.environ.get(k) for k in ["MLX_ENABLE_TF32", "MLX_METAL_FAST_SYNCH", "MLX_MOE_SINGLE_SORT"]},
    }
    runtime_root = Path(mx.__file__).resolve().parent
    binaries = [Path(mx.__file__).resolve()]
    for name in ("libmlx.dylib", "mlx.metallib"):
        binaries.extend(runtime_root.rglob(name))
    metadata["binary_sha256"] = {}
    for binary in sorted(set(binaries)):
        hasher = hashlib.sha256()
        with binary.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b""):
                hasher.update(chunk)
        metadata["binary_sha256"][str(binary)] = hasher.hexdigest()
    try:
        version = subprocess.run(["xcodebuild", "-version"], capture_output=True, text=True, timeout=10)
        metadata["xcode"] = version.stdout.strip() or version.stderr.strip()
    except (OSError, subprocess.TimeoutExpired):
        metadata["xcode"] = "unavailable"
    dtype = getattr(mx, args.dtype)
    cases = args.case or (
        [(8,8,2048,128), (64,8,2048,128), (512,8,2048,128),
         (2048,8,2048,128), (4097,8,2048,128)] if args.scope=="routing" else
        [(1,8,2048,64), (8,8,2048,64), (512,8,2048,64), (2048,8,2048,64)]
    )
    metadata["cases"] = []

    def execute(fn):
        outputs = fn()
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        mx.eval(*outputs)
        mx.synchronize()
        return outputs

    def identical(lhs, rhs):
        if len(lhs) != len(rhs):
            raise RuntimeError("Output arity mismatch")
        for a, b in zip(lhs, rhs):
            if a.shape != b.shape or a.dtype != b.dtype:
                raise RuntimeError("Output shape/dtype mismatch")
            if a.dtype in (mx.float16, mx.bfloat16, mx.float32):
                word = mx.uint32 if a.dtype == mx.float32 else mx.uint16
                a, b = a.view(word), b.view(word)
            if not mx.array_equal(a, b).item():
                raise RuntimeError("Bitwise mismatch: refusing performance result")

    def timed(fn, reps):
        mx.synchronize()
        start = time.perf_counter_ns()
        for _ in range(reps):
            outputs = execute(fn)
            del outputs
        return (time.perf_counter_ns()-start)/1e6/reps

    for cell_id, (tokens, top_k, dim, experts) in enumerate(cases):
        mx.random.seed(399 + cell_id)
        x = mx.random.normal((1,tokens,dim)).astype(dtype)
        # Unique selections per token; ties across tokens are deliberately common.
        idx = mx.argsort(mx.random.uniform(shape=(1,tokens,experts)))[...,:top_k]
        expanded = mx.expand_dims(x,(-2,-3))
        scores = mx.softmax(mx.random.normal((1,tokens,top_k)),axis=-1)
        mx.eval(x,idx,scores)
        routes = {
            "baseline": candidate.gather_sort_baseline,
            "scatter_inverse": candidate.gather_sort_scatter,
            "fused": lambda a,i: candidate.gather_sort_fused(a,i,threadgroup_width=args.threadgroup_width),
            "compiled_baseline": mx.compile(candidate.gather_sort_baseline),
        }
        if args.scope == "routing":
            functions = {name:(lambda route=route: route(expanded,idx)) for name,route in routes.items()}
            golden = execute(functions["baseline"])
        else:
            from mlx_lm.models.switch_layers import SwitchGLU
            if args.bits and (dim%64 or args.ffn%64):
                parser.error("Quantized block dim and ffn must be multiples of 64")
            block = SwitchGLU(dim,args.ffn,experts)
            block.set_dtype(dtype)
            if args.bits:
                nn.quantize(block,group_size=64,bits=args.bits)
            block.eval()
            mx.eval(block.parameters())

            def make_forward(route):
                def forward():
                    a = expanded
                    do_sort = idx.size >= 64
                    ix,inv = idx,None
                    if do_sort:
                        a,ix,inv = route(a,idx)
                    up = block.up_proj(a,ix,sorted_indices=do_sort)
                    gate = block.gate_proj(a,ix,sorted_indices=do_sort)
                    result = block.down_proj(block.activation(up,gate),ix,sorted_indices=do_sort)
                    if do_sort:
                        result = mx.unflatten(result[inv],0,idx.shape)
                    result = result.squeeze(-2)
                    return (result*scores[...,None]).sum(axis=-2).astype(result.dtype)
                return forward

            functions = {name:make_forward(route) for name,route in routes.items() if name!="compiled_baseline"}
            def original_block():
                result=block(x,idx)
                return (result*scores[...,None]).sum(axis=-2).astype(result.dtype)
            golden = execute(original_block)
            # A fair graph-compiled full block, not just a compiled routing helper.
            compiled = mx.compile(lambda a,i,w: (block(a,i)*w[...,None]).sum(axis=-2).astype(a.dtype))
            functions["compiled_baseline"] = lambda: compiled(x,idx,scores)

        cell = {"tokens":tokens,"top_k":top_k,"dim":dim,"experts":experts,
                "sort_route_engaged":args.scope=="routing" or tokens*top_k>=64,
                "first_measured_call_ms":{},"comparisons":{},"correctness":"bitwise pass"}
        # Baseline checked against actual installed SwitchGLU before any timing claim.
        for name,fn in functions.items():
            start=time.perf_counter_ns()
            output=execute(fn)
            cell["first_measured_call_ms"][name]=(time.perf_counter_ns()-start)/1e6
            identical(output,golden)
            del output
            for _ in range(args.warmup):
                execute(fn)
        del golden
        comparisons = ["aa", "scatter_inverse", "fused", "compiled_baseline"]
        samples = {name:[] for name in comparisons}
        for iteration in range(args.rounds):
            # Rotate the comparisons; reverse within-pair order on alternating rounds.
            order = comparisons[iteration%len(comparisons):] + comparisons[:iteration%len(comparisons)]
            for name in order:
                base_fn=functions["baseline"]
                other_fn=base_fn if name=="aa" else functions[name]
                if iteration%2:
                    b=timed(other_fn,args.reps); a=timed(base_fn,args.reps)
                else:
                    a=timed(base_fn,args.reps); b=timed(other_fn,args.reps)
                samples[name].append([a,b])
        aa = bootstrap_ratio(samples["aa"])
        # Reject noisy cells rather than silently present their ratio as a win.
        aa_range=aa["paired_bootstrap_95pct"]
        aa_ok = aa_range[0]>=0.95 and aa_range[1]<=1.05
        for name,pairs in samples.items():
            stats=bootstrap_ratio(pairs)
            cell["comparisons"][name]={**stats,"paired_ms":pairs,"aa_calibration_ok":aa_ok,
                "candidate_median_ms":statistics.median(b for a,b in pairs),
                "baseline_median_ms":statistics.median(a for a,b in pairs)}
        # Confirm outputs after the timed section before persisting this cell.
        reference=execute(functions["baseline"])
        for fn in functions.values():
            identical(execute(fn),reference)
        del reference
        if args.capture_dir is not None:
            args.capture_dir.mkdir(parents=True, exist_ok=True)
            cell["captures"] = {}
            for name in ("baseline", "scatter_inverse", "fused"):
                target = args.capture_dir / f"{args.scope}-{cell_id}-{name}.gputrace"
                mx.metal.start_capture(str(target))
                try:
                    execute(functions[name])
                finally:
                    mx.metal.stop_capture()
                cell["captures"][name] = str(target)
        mx.synchronize()
        cell["active_memory_bytes_after_cell"]=mx.get_active_memory()
        metadata["cases"].append(cell)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(metadata,indent=2,default=str)+"\n")
        f=cell["comparisons"]["fused"]
        print(f"{args.scope} T={tokens} K={top_k} D={dim}: fused ratio "
              f"{f['geomean_baseline_over_candidate']:.4f}, CI={f['paired_bootstrap_95pct']}, "
              f"A/A={'PASS' if aa_ok else 'INVALID'}",flush=True)
        del functions,routes
        if args.scope=="block":
            del block,compiled
    metadata["status"] = "completed"
    args.output.write_text(json.dumps(metadata,indent=2,default=str)+"\n")
    print(f"Raw results: {args.output}")
    if any(not c["comparisons"]["aa"]["aa_calibration_ok"] for c in metadata["cases"]):
        raise SystemExit("At least one A/A calibration was noisy; do not publish those speedup cells.")


if __name__=="__main__":
    main()
