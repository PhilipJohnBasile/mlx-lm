"""Actual-checkpoint packed gate/up investigation; no model installation or defaults."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import struct
import sys
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "experiments/gdn_preprocessing")
)
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "experiments/m5_inference")
)
import mlx.core as mx
from benchmark_requests import memory, sha256
from mlx_gdn_prep.timing import calibration_ok, summarize

from mlx_lm.models.activations import swiglu
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

p = argparse.ArgumentParser()
p.add_argument("--model", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--tokens", default="1,8,512,2048")
p.add_argument("--rounds", type=int, default=9)
p.add_argument("--iterations", type=int, default=10)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
c = json.loads((a.model / "config.json").read_text())
q = c["quantization"]
tc = c.get("text_config", c)
bits = q["bits"]
group = q["group_size"]
moe = bool(tc.get("num_experts", 0))
prefix = "language_model.model.layers.0.mlp." + ("switch_mlp." if moe else "")
source = sorted(a.model.glob("model-00001-of-*.safetensors"))[0]
tensor_hashes = {}
with source.open("rb") as stream:
    header_bytes = struct.unpack("<Q", stream.read(8))[0]
    header = json.loads(stream.read(header_bytes))
    for name in ("gate_proj", "up_proj", "down_proj"):
        for suffix in ("weight", "scales", "biases"):
            key = prefix + name + "." + suffix
            lo, hi = header[key]["data_offsets"]
            stream.seek(8 + header_bytes + lo)
            digest = hashlib.sha256()
            left = hi - lo
            while left:
                data = stream.read(min(left, 8 * 1024 * 1024))
                assert data
                digest.update(data)
                left -= len(data)
            tensor_hashes[key] = {
                "sha256": digest.hexdigest(),
                "bytes": hi - lo,
                "shape": header[key]["shape"],
                "dtype": header[key]["dtype"],
            }
loaded = mx.load(str(source))
weights = {
    name: tuple(
        loaded[prefix + name + "." + suffix]
        for suffix in ("weight", "scales", "biases")
    )
    for name in ("gate_proj", "up_proj", "down_proj")
}
del loaded
mx.eval(weights)
packed = tuple(
    mx.concatenate([g, u], axis=-2)
    for g, u in zip(weights["gate_proj"], weights["up_proj"])
)
mx.eval(packed)
for original, combined in zip(zip(weights["gate_proj"], weights["up_proj"]), packed):
    halves = mx.split(combined, 2, axis=-2)
    for expected, actual in zip(original, halves):
        dtype = mx.uint32 if expected.dtype == mx.uint32 else mx.uint16
        assert mx.array_equal(
            expected.view(dtype), actual.view(dtype)
        ).item(), "Packed bits changed"
base_bytes = sum(v.nbytes for k in ("gate_proj", "up_proj") for v in weights[k])
assert sum(v.nbytes for v in packed) == base_bytes
report = {
    "scope": "Warm actual-checkpoint MLP block; separate original and packed buffers are comparison fixtures only. No whole-model or nonduplicating runtime integration claim. Existing compiled SwiGLU is reused.",
    "complete": False,
    "model": str(a.model),
    "tensor_hashes": tensor_hashes,
    "config_sha256": sha256(a.model / "config.json"),
    "runtime": {
        "mlx": importlib.metadata.version("mlx"),
        "mlx_lm": importlib.metadata.version("mlx-lm"),
        "binary_sha256": sha256(mx.__file__),
        "device": mx.device_info(mx.gpu),
        "os": platform.platform(),
    },
    "source_file": str(source),
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "prefix": prefix,
    "bits": bits,
    "group_size": group,
    "dtype": str(weights["gate_proj"][1].dtype),
    "packed_bits_equal": True,
    "original_gate_up_bytes": base_bytes,
    "packed_gate_up_bytes": sum(v.nbytes for v in packed),
    "arithmetic_tolerance": {"atol": 0.005, "rtol": 0.02},
    "performance_qualified": False,
    "cells": [],
}
(a.output / "source.py").write_bytes(Path(__file__).read_bytes())
K = tc["hidden_size"]
dtype = weights["gate_proj"][1].dtype
mx.set_cache_limit(1_000_000_000)
for S in map(int, a.tokens.split(",")):
    mx.random.seed(7209 + S)
    x = mx.random.normal((1, S, K)).astype(dtype)
    ids = (
        mx.argsort(mx.random.uniform(shape=(1, S, tc["num_experts"])), axis=-1)[
            ..., : tc["num_experts_per_tok"]
        ].astype(mx.uint32)
        if moe
        else None
    )
    mx.eval(x)
    if ids is not None:
        mx.eval(ids)

    def dense(v, w):
        return mx.quantized_matmul(
            v, *w, transpose=True, bits=bits, group_size=group, mode="affine"
        )

    def run(fused):
        v = x
        if moe:
            v = mx.expand_dims(v, (-2, -3))
            idx = ids
            sorted_ids = idx.size >= 64
            if sorted_ids:
                v, idx, inverse = _gather_sort(v, idx)

            def proj(w):
                return mx.gather_qmm(
                    v,
                    *w,
                    rhs_indices=idx,
                    transpose=True,
                    bits=bits,
                    group_size=group,
                    mode="affine",
                    sorted_indices=sorted_ids,
                )

        else:

            def proj(w):
                return dense(v, w)

        if fused:
            g, u = mx.split(proj(packed), 2, axis=-1)
        else:
            g, u = proj(weights["gate_proj"]), proj(weights["up_proj"])
        out = swiglu(g, u)
        if moe:
            out = mx.gather_qmm(
                out,
                *weights["down_proj"],
                rhs_indices=idx,
                transpose=True,
                bits=bits,
                group_size=group,
                mode="affine",
                sorted_indices=sorted_ids,
            )
            if sorted_ids:
                out = _scatter_unsort(out, inverse, ids.shape)
            return out.squeeze(-2)
        return dense(out, weights["down_proj"])

    def correctness():
        one, two = run(False), run(True)
        mx.eval(one, two)
        f, g = one.astype(mx.float32), two.astype(mx.float32)
        return {
            "bitwise_equal": bool(
                mx.array_equal(one.view(mx.uint16), two.view(mx.uint16)).item()
            ),
            "max_abs": float(mx.max(mx.abs(f - g)).item()),
            "passed": bool(
                (
                    mx.all(mx.isfinite(f))
                    & mx.all(mx.isfinite(g))
                    & mx.allclose(f, g, **report["arithmetic_tolerance"])
                ).item()
            ),
        }

    cell = {"tokens": S, "correctness_before": correctness(), "raw": []}
    report["cells"].append(cell)
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    if not cell["correctness_before"]["passed"]:
        raise RuntimeError("Block parity failed")
    for _ in range(10):
        mx.eval(run(False), run(True))
    cell["memory_before"] = memory()

    def measure(fused):
        mx.synchronize()
        t = time.perf_counter_ns()
        for _ in range(a.iterations):
            mx.eval(run(fused))
        mx.synchronize()
        return (time.perf_counter_ns() - t) / 1e6 / a.iterations

    def collect(label, order):
        pairs = []
        for i in range(a.rounds):
            modes = order if i % 2 == 0 or label != "ab" else [True, False, False, True]
            values = [measure(mode) for mode in modes]
            cell["raw"].append(
                {"label": label, "round": i, "order": modes, "ms": values}
            )
            if label == "ab":
                pairs.append(
                    tuple(
                        statistics.mean(
                            t for mode, t in zip(modes, values) if mode == wanted
                        )
                        for wanted in (False, True)
                    )
                )
            else:
                pairs.append((values[0], values[1]))
        return summarize(pairs)

    cell["aa_before"] = collect("aa_before", [False, False])
    cell["ab"] = collect("ab", [False, True, True, False])
    cell["aa_after"] = collect("aa_after", [False, False])
    cell["correctness_after"] = correctness()
    cell["drift"] = (
        cell["aa_after"]["reference_median_ms"]
        / cell["aa_before"]["reference_median_ms"]
    )
    cell["timing_controls_valid"] = (
        calibration_ok(cell["aa_before"])
        and calibration_ok(cell["aa_after"])
        and 0.95 <= cell["drift"] <= 1.05
    )
    cell["memory_after"] = memory()
    cell["swap_delta"] = {
        k: cell["memory_after"][k] - cell["memory_before"][k]
        for k in ("swapins", "swapouts", "pageouts")
    }
    cell["valid"] = (
        cell["timing_controls_valid"]
        and all(v == 0 for v in cell["swap_delta"].values())
        and cell["memory_after"]["free_percent"] >= 20
        and cell["correctness_after"]["passed"]
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        S,
        cell["ab"],
        "valid",
        cell["timing_controls_valid"],
        "correct",
        cell["correctness_after"],
        flush=True,
    )

report["complete"] = True
(a.output / "report.json").write_text(json.dumps(report, indent=2))
