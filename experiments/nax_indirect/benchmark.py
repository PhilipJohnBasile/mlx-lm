"""Paired M5 qualification of built-in, matched-JIT, and indirect projections.

All reported times include Python dispatch and synchronization. No downloads,
model weights, environment mutations, or automatic enablement are performed.
"""

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_nax_indirect import project_pair, require_m5_nax, validate_routing
from qualification_utils import (
    calibration_passes,
    check_outputs,
    route_statistics,
    sample_routes,
    valid_timing_summary,
)


def measure(fn, iterations):
    start = time.perf_counter_ns()
    for _ in range(iterations):
        output = fn()
        mx.eval(output)
    return (time.perf_counter_ns() - start) / iterations / 1e6


def paired(a, b, rounds, iterations):
    rows = []
    for i in range(rounds):
        order = ("a", "b", "b", "a") if i % 2 == 0 else ("b", "a", "a", "b")
        samples = {"a": [], "b": []}
        for key in order:
            samples[key].append(measure(a if key == "a" else b, iterations))
        av = statistics.mean(samples["a"])
        bv = statistics.mean(samples["b"])
        rows.append(
            {"a_ms": av, "b_ms": bv, "speedup": av / bv, "order": "".join(order)}
        )
    logs = np.log([row["speedup"] for row in rows])
    rng = np.random.default_rng(4707)
    boot = np.exp(rng.choice(logs, size=(3000, len(logs)), replace=True).mean(axis=1))
    return {
        "geomean_speedup": float(np.exp(logs.mean())),
        "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "pairs": rows,
    }


def peak(fn):
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    output = fn()
    mx.eval(output)
    result = {
        "active_before_bytes": before,
        "peak_bytes": mx.get_peak_memory(),
        "incremental_peak_bytes": mx.get_peak_memory() - before,
        "active_with_outputs_bytes": mx.get_active_memory(),
    }
    del output
    mx.synchronize()
    return result


def fingerprints():
    result = {}
    path = Path(mx.__file__).resolve()
    files = [path, *path.parent.rglob("*.dylib"), *path.parent.rglob("*.metallib")]
    for item in dict.fromkeys(files):
        if item.is_file():
            digest = hashlib.sha256()
            with item.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            result[str(item)] = digest.hexdigest()
    return result


def make_workload(
    tokens, k, n, experts, top_k, bits, group, dtype, distribution, scope
):
    mx.random.seed(4707 + tokens)
    x = (mx.random.normal((tokens, k)) / math.sqrt(k)).astype(dtype)
    # Workload generation is outside timed execution. Top-k IDs are unique
    # within a token, as they are for an ordinary router's top-k selection.
    host_ids = sample_routes(tokens, experts, top_k, distribution, seed=4707 + tokens)
    route_info = route_statistics(host_ids, experts)
    ids = mx.array(host_ids, dtype=mx.uint32)
    if scope == "switch":
        import mlx.nn as nn
        from mlx_nax_indirect import switch_glu

        from mlx_lm.models.switch_layers import SwitchGLU

        module = SwitchGLU(k, n, experts)
        module.set_dtype(dtype)
        nn.quantize(module, group_size=group, bits=bits)
        module.eval()
        mx.eval(module.parameters(), x, ids)
        funcs = {
            mode: (
                lambda mode=mode: switch_glu(module, x, ids, mode=mode, validate=False)
            )
            for mode in ("upstream", "jit_contiguous", "jit_indirect")
        }
        return funcs, (x, ids, module), route_info
    first = mx.quantize(
        mx.random.normal((experts, n, k)).astype(dtype), group_size=group, bits=bits
    )
    second = mx.quantize(
        mx.random.normal((experts, n, k)).astype(dtype), group_size=group, bits=bits
    )
    flat = ids.flatten()
    order = mx.argsort(flat)
    rows = (order // top_k).astype(mx.uint32)
    sorted_ids = flat[order]
    mx.eval(x, *first, *second, rows, sorted_ids)
    validate_routing(rows, sorted_ids, tokens, experts)
    funcs = {
        mode: (
            lambda mode=mode: project_pair(
                x,
                first,
                second,
                rows,
                sorted_ids,
                mode=mode,
                group_size=group,
                bits=bits,
                validate=False,
            )
        )
        for mode in ("upstream", "jit_contiguous", "jit_indirect")
    }
    return funcs, (x, ids, first, second, rows, sorted_ids), route_info


def compare_outputs(funcs, dtype):
    outputs = {key: fn() for key, fn in funcs.items()}
    for value in outputs.values():
        mx.eval(value)
    tolerance = 3e-3 if dtype == mx.float16 else 2e-2
    return check_outputs(outputs, mx, tolerance)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", default="512,2048,8192")
    p.add_argument("--k", type=int, default=2048)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--bits", type=int, choices=(4, 8), default=4)
    p.add_argument("--group-size", type=int, choices=(64, 128), default=64)
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument("--distribution", choices=("uniform", "skewed"), default="uniform")
    p.add_argument("--scope", choices=("pair", "switch"), default="pair")
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--output", type=Path, default=Path("results/m5-benchmark.json"))
    p.add_argument(
        "--capture", type=str, help="Optional .gputrace of the indirect path"
    )
    args = p.parse_args()
    if (
        min(args.k, args.n, args.experts, args.top_k, args.rounds, args.iterations) <= 0
        or args.rounds < 3
    ):
        p.error("Positive dimensions and at least three rounds are required")
    if args.top_k > args.experts:
        p.error("--top-k cannot exceed --experts")
    token_counts = [int(t) for t in args.tokens.split(",")]
    if any(t <= 0 for t in token_counts):
        p.error("Token counts must be positive")
    dtype = mx.float16 if args.dtype == "float16" else mx.bfloat16
    device = require_m5_nax()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "macos": platform.mac_ver()[0],
        "python": sys.version,
        "mlx": mx.__version__,
        "core_upstream_reference": "b6368984b8e02a3fb3ee7986846c0fb85e1fccf7",
        "runtime_fingerprints": fingerprints(),
        "pilot_fingerprints": {
            str(f.relative_to(Path(__file__).resolve().parent)): hashlib.sha256(
                f.read_bytes()
            ).hexdigest()
            for f in (Path(__file__).resolve().parent / "mlx_nax_indirect").rglob("*")
            if f.is_file() and f.suffix in (".py", ".h", ".metal")
        },
        "parameters": {**vars(args), "output": str(args.output)},
        "method": "synchronized wall-clock, alternating ABBA/BAAB; bootstrap is per-round only",
        "qualification_schema": 2,
        "native_performance_claim": "Per-cell candidate only; no whole-model promotion",
        "sampling": "Unique top-k per token, synthetic uniform/weighted without replacement",
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tokens in token_counts:
        funcs, keepalive, route_info = make_workload(
            tokens,
            args.k,
            args.n,
            args.experts,
            args.top_k,
            args.bits,
            args.group_size,
            dtype,
            args.distribution,
            args.scope,
        )
        correctness = compare_outputs(funcs, dtype)
        for fn in funcs.values():
            for _ in range(4):
                mx.eval(fn())
        aa_before = paired(
            funcs["upstream"], funcs["upstream"], args.rounds, args.iterations
        )
        jit_control = paired(
            funcs["upstream"], funcs["jit_contiguous"], args.rounds, args.iterations
        )
        direct = paired(
            funcs["upstream"], funcs["jit_indirect"], args.rounds, args.iterations
        )
        isolated = paired(
            funcs["jit_contiguous"], funcs["jit_indirect"], args.rounds, args.iterations
        )
        aa_after = paired(
            funcs["upstream"], funcs["upstream"], args.rounds, args.iterations
        )
        aa_ok = all(calibration_passes(x) for x in (aa_before, aa_after))
        # Catch gross drift across the entire run, not just adjacent A/A pairs.
        initial_ms = statistics.median(r["a_ms"] for r in aa_before["pairs"])
        final_ms = statistics.median(r["a_ms"] for r in aa_after["pairs"])
        drift_ok = abs(final_ms / initial_ms - 1) <= 0.05
        correctness_after = compare_outputs(funcs, dtype)
        timing_valid = all(
            valid_timing_summary(x) for x in (jit_control, direct, isolated)
        )
        credible = (
            aa_ok
            and drift_ok
            and timing_valid
            and direct["ci95"][0] > 1.03
            and isolated["ci95"][0] > 1.0
        )
        record = {
            "tokens": tokens,
            "routes": tokens * args.top_k,
            "eliminated_gather_allocation_bytes": tokens * args.top_k * args.k * 2,
            "correctness_before": correctness,
            "correctness_after": correctness_after,
            "routing_statistics": route_info,
            "aa_before": aa_before,
            "aa_after": aa_after,
            "upstream_vs_jit_contiguous": jit_control,
            "upstream_vs_indirect": direct,
            "matched_jit_vs_indirect": isolated,
            "aa_ok": aa_ok,
            "drift_ok": drift_ok,
            "timing_valid": timing_valid,
            "candidate_speed_gate_passed": credible,
            "memory": {key: peak(fn) for key, fn in funcs.items()},
        }
        report["results"].append(record)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"T={tokens}: upstream/indirect={direct['geomean_speedup']:.4f}x "
            f"CI={direct['ci95']} candidate_speed_gate_passed={credible}",
            flush=True,
        )
        if args.capture:
            mx.metal.start_capture(args.capture)
            try:
                mx.eval(funcs["jit_indirect"]())
            finally:
                mx.metal.stop_capture()
            args.capture = None
        del funcs, keepalive
        gc.collect()
        mx.clear_cache()
    print(f"Saved {args.output}; no automatic runtime policy was changed.")


if __name__ == "__main__":
    main()
