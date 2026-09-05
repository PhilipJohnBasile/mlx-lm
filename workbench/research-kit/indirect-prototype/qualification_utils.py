"""Portable qualification helpers; no MLX import or device execution."""
from __future__ import annotations

from typing import Any
import math
import numpy as np


def sample_routes(tokens: int, experts: int, top_k: int, distribution: str,
                  seed: int = 4707) -> np.ndarray:
    """Generate unique experts per token; repetition across tokens is allowed.

    Skewed inputs allocate 80% of categorical sampling mass to a hot set.
    Sampling is without replacement, so realized route frequency is not 80%.
    This is synthetic workload generation, not a learned-router simulator.
    """
    if min(tokens, experts, top_k) <= 0 or top_k > experts:
        raise ValueError("Require tokens > 0 and 0 < top_k <= experts")
    if experts > np.iinfo(np.uint32).max:
        raise ValueError("Expert IDs must fit uint32")
    if distribution not in ("uniform", "skewed"):
        raise ValueError("distribution must be uniform or skewed")
    rng = np.random.default_rng(seed)
    probability = None
    if distribution == "skewed":
        hot = min(experts, max(top_k, 8))
        if hot < experts:
            probability = np.full(experts, .2 / (experts - hot))
            probability[:hot] = .8 / hot
    result = np.empty((tokens, top_k), dtype=np.uint32)
    for token in range(tokens):
        result[token] = rng.choice(experts, top_k, replace=False, p=probability)
    return result


def route_statistics(ids: np.ndarray, experts: int) -> dict[str, Any]:
    ids = np.asarray(ids)
    if ids.ndim != 2 or not ids.size or experts <= 0:
        raise ValueError("Expected a nonempty [tokens, top_k] integer array")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Expert IDs must be integers")
    if np.any(ids < 0) or np.any(ids >= experts):
        raise ValueError("Expert ID out of range")
    ordered = np.sort(ids, axis=1)
    duplicates = int(np.any(ordered[:, 1:] == ordered[:, :-1], axis=1).sum())
    counts = np.bincount(ids.reshape(-1).astype(np.int64), minlength=experts)
    active = counts[counts > 0]
    return {
        "tokens_with_duplicate_experts": duplicates,
        "active_experts": int(active.size),
        "empty_experts": int(experts - active.size),
        "routes_per_active_expert_quantiles": {
            str(q): float(np.quantile(active, q)) for q in (0, .25, .5, .75, 1)
        },
        "routes_per_expert": counts.tolist(),
        "mean_routes_all_experts": float(counts.mean()),
        "mean_routes_active_experts": float(active.mean()),
        "count_coefficient_of_variation": float(counts.std() / counts.mean()),
    }


def scalar_bool(value: Any) -> bool:
    return bool(value.item() if hasattr(value, "item") else value)


def bitwise_equal(a: Any, b: Any, xp: Any) -> bool:
    """Compare floating payload bits, including signed zero and NaN payloads."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    for name, integer in (("float16", "uint16"), ("bfloat16", "uint16"),
                          ("float32", "uint32"), ("float64", "uint64")):
        dtype = getattr(xp, name, None)
        if dtype is not None and a.dtype == dtype:
            return scalar_bool(xp.array_equal(a.view(getattr(xp, integer)),
                                             b.view(getattr(xp, integer))))
    raise ValueError(f"Unsupported payload dtype: {a.dtype}")


def check_outputs(outputs: dict[str, Any], xp: Any, tolerance: float) -> list[dict]:
    """Reject missing outputs, broadcast comparisons and nonfinite controls."""
    modes = ("upstream", "jit_contiguous", "jit_indirect")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    if set(outputs) != set(modes):
        raise ValueError("All three comparison paths are required")
    parts = {mode: tuple(outputs[mode]) if isinstance(outputs[mode], (tuple, list))
             else (outputs[mode],) for mode in modes}
    sizes = {len(value) for value in parts.values()}
    if len(sizes) != 1 or 0 in sizes:
        raise ValueError("Comparison paths must have the same nonzero output count")
    checks = []
    for index in range(len(parts["upstream"])):
        a, b, c = [parts[mode][index] for mode in modes]
        if a.shape != b.shape or a.shape != c.shape or a.dtype != b.dtype or a.dtype != c.dtype:
            raise ValueError("Output shape/dtype mismatch; broadcasting is not parity")
        finite = all(scalar_bool(xp.all(xp.isfinite(x))) for x in (a, b, c))
        exact = bitwise_equal(b, c, xp)
        close = all(scalar_bool(xp.allclose(a, other, atol=tolerance, rtol=tolerance))
                    for other in (b, c))
        error = float(xp.max(xp.abs(a.astype(xp.float32) - c.astype(xp.float32))).item())
        checks.append({"finite_all_paths": finite, "matched_jit_bitwise": exact,
                       "upstream_allclose_both_paths": close,
                       "max_abs_upstream_error": error})
    if not all(row["finite_all_paths"] and row["matched_jit_bitwise"] and
               row["upstream_allclose_both_paths"] for row in checks):
        raise RuntimeError(f"Correctness gate failed: {checks}")
    return checks


def valid_timing_summary(summary: dict) -> bool:
    try:
        lo, hi = summary["ci95"]
        mean = summary["geomean_speedup"]
        pairs = summary["pairs"]
        if len(pairs) < 3 or not all(math.isfinite(x) and x > 0 for x in (lo, hi, mean)):
            return False
        if not lo <= hi:
            return False
        return all(all(math.isfinite(row[k]) and row[k] > 0
                       for k in ("a_ms", "b_ms", "speedup")) for row in pairs)
    except (KeyError, TypeError, ValueError):
        return False


def calibration_passes(summary: dict, fraction: float = .05) -> bool:
    """Require the entire A/A interval, not only its mean, inside tolerance."""
    if not valid_timing_summary(summary):
        return False
    lo, hi = summary["ci95"]
    return (1 - fraction <= lo <= hi <= 1 + fraction and
            1 - fraction <= summary["geomean_speedup"] <= 1 + fraction)
