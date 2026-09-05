"""Experimental, explicitly selected NAX row-indirect MoE projections.

This module does not modify MLX or install a global model hook.
Native Metal execution and performance must be qualified on the target machine.
"""

import platform
from functools import lru_cache
from pathlib import Path

from .policy import Geometry, compatible_device


def _mlx():
    import mlx.core as mx

    return mx


@lru_cache(maxsize=1)
def require_m5_nax() -> dict:
    mx = _mlx()
    if platform.system() != "Darwin" or not mx.metal.is_available():
        raise RuntimeError("This pilot requires a real M5-or-newer Metal device")
    info = mx.device_info(mx.gpu)
    if not compatible_device(info, platform.mac_ver()[0]):
        raise RuntimeError(f"Device is outside the pilot's M5+ NAX gate: {info}")
    return info


def kernel_sources() -> tuple[str, str]:
    root = Path(__file__).resolve().parent / "kernels"
    header = "\n".join(
        (root / name).read_text().replace("#pragma once", "")
        for name in ("upstream_nax.h", "route_address.h", "indirect_load.h")
    )
    return header, (root / "gather_body.metal").read_text()


@lru_cache(maxsize=1)
def _kernel():
    mx = _mlx()
    header, source = kernel_sources()
    return mx.fast.metal_kernel(
        name="mlx_nax_route_gather_pilot",
        input_names=["x", "w", "scales", "biases", "indices", "rows"],
        output_names=["y"],
        source=source,
        header=header,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def _geometry(x, w, scales, biases, rows, experts, group_size, bits):
    mx = _mlx()
    if x.ndim != 2 or w.ndim != 3 or scales.ndim != 3 or biases.ndim != 3:
        raise ValueError("Expected X[T,K], W[E,N,Kpacked], scales/biases[E,N,K/group]")
    if rows.ndim != 1 or experts.ndim != 1 or rows.size != experts.size:
        raise ValueError("rows and experts must be equal-length 1-D routing arrays")
    if rows.dtype != mx.uint32 or experts.dtype != mx.uint32 or w.dtype != mx.uint32:
        raise ValueError("Packed weights and routing arrays must be uint32")
    if scales.dtype != x.dtype or biases.dtype != x.dtype:
        raise ValueError("Activations, scales, and affine biases must have one dtype")
    dtype = {mx.float16: "float16", mx.bfloat16: "bfloat16"}.get(x.dtype, "unsupported")
    geom = Geometry(
        x.shape[0],
        rows.size,
        w.shape[0],
        x.shape[1],
        w.shape[1],
        group_size,
        bits,
        dtype,
    )
    geom.validate()
    if w.shape[2] * (32 // bits) != geom.k:
        raise ValueError("Packed weight width does not match K/bits")
    expected = (geom.experts, geom.n, geom.k // group_size)
    if scales.shape != expected or biases.shape != expected:
        raise ValueError(f"Expected scales and affine biases of shape {expected}")
    return geom


def validate_routing(rows, experts, source_rows: int, expert_count: int) -> None:
    """Synchronizing debug check. Do not include it in a steady-state benchmark."""
    mx = _mlx()
    valid = mx.all(rows < source_rows) & mx.all(experts < expert_count)
    ordered = mx.all(experts[1:] >= experts[:-1])
    if not (valid & ordered).item():
        raise ValueError("Routing must be in range, with nondecreasing expert IDs")


def gather_qmm(
    x,
    w,
    scales,
    biases,
    rows,
    experts,
    *,
    group_size=64,
    bits=4,
    indirect=True,
    validate=True,
    stream=None,
):
    """Return [routes,1,N] in expert-sorted route order.

    indirect=True reads X[T,K] using rows. indirect=False is a matched JIT
    control and expects the already-gathered X[routes,K]. With validate=False,
    routing correctness is the caller's contract. Invalid row/expert indices
    are memory-bounded in the pilot kernel and poison results with NaNs.
    This custom kernel is forward-only: do not differentiate through it.
    """
    mx = _mlx()
    require_m5_nax()
    geom = _geometry(x, w, scales, biases, rows, experts, group_size, bits)
    if not indirect and x.shape[0] != rows.size:
        raise ValueError("The contiguous control requires one input row per route")
    if validate:
        if indirect:
            validate_routing(rows, experts, geom.source_rows, geom.experts)
        elif not (
            mx.all(experts < geom.experts) & mx.all(experts[1:] >= experts[:-1])
        ).item():
            raise ValueError("Expert IDs must be sorted and in range")
    return _kernel()(
        inputs=[x, w, scales, biases, experts, rows],
        output_shapes=[(geom.routes, 1, geom.n)],
        output_dtypes=[x.dtype],
        grid=(32 * (geom.n // 64), 2 * ((geom.routes + geom.bm - 1) // geom.bm), 2),
        threadgroup=(32, 2, 2),
        template=[
            ("T", x.dtype),
            ("GROUP_SIZE", group_size),
            ("BITS", bits),
            ("BM", geom.bm),
            ("ALIGNED_M", geom.routes % geom.bm == 0),
            ("INDIRECT", indirect),
        ],
        stream=mx.gpu if stream is None else stream,
    )[0]


def project_pair(
    x,
    first,
    second,
    rows,
    experts,
    *,
    mode="upstream",
    group_size=64,
    bits=4,
    validate=True,
):
    """Two projections; first/second are (packed_weight, scale, affine_bias).

    upstream: one materialized gather shared by both built-in projections.
    jit_contiguous: the same gather with the matched custom contiguous kernel.
    jit_indirect: no materialized activation gather.
    """
    mx = _mlx()
    if mode not in ("upstream", "jit_contiguous", "jit_indirect"):
        raise ValueError("Unknown projection mode")
    if validate:
        for weight in (first, second):
            _geometry(x, *weight, rows, experts, group_size, bits)
            validate_routing(rows, experts, x.shape[0], weight[0].shape[0])
    if mode == "jit_indirect":
        return tuple(
            gather_qmm(
                x,
                *weight,
                rows,
                experts,
                group_size=group_size,
                bits=bits,
                validate=False,
            )
            for weight in (first, second)
        )
    gathered = x[rows]
    if mode == "jit_contiguous":
        return tuple(
            gather_qmm(
                gathered,
                *weight,
                rows,
                experts,
                group_size=group_size,
                bits=bits,
                indirect=False,
                validate=False,
            )
            for weight in (first, second)
        )
    return tuple(
        mx.gather_qmm(
            gathered[:, None, :],
            *weight,
            rhs_indices=experts,
            transpose=True,
            group_size=group_size,
            bits=bits,
            sorted_indices=True,
        )
        for weight in (first, second)
    )


def switch_glu(module, x, indices, *, mode="upstream", validate=True):
    """Opt-in drop-in call for an evaluated, affine-quantized MLX-LM SwitchGLU.

    Example: y = switch_glu(layer.switch_mlp, x, expert_indices,
                            mode="jit_indirect", validate=False)
    It returns per-route outputs, preserving the caller's routing-weight sum.
    The down projection, activation, and output unsort remain upstream code.
    """
    if mode == "upstream":
        return module(x, indices)
    if getattr(module, "training", True):
        raise ValueError("The pilot is inference-only; call model.eval() first")
    mx = _mlx()
    if indices.dtype != mx.uint32:
        raise ValueError("The pilot requires uint32 expert indices from the router")
    if indices.shape[:-1] != x.shape[:-1] or indices.size < 64:
        raise ValueError("Expected indices[...,top_k] with at least 64 routes")
    layers = (module.up_proj, module.gate_proj)
    for layer in layers:
        if getattr(layer, "mode", None) != "affine" or "bias" in layer:
            raise ValueError(
                "The pilot requires affine quantized projections without linear bias"
            )
    group_size, bits = layers[0].group_size, layers[0].bits
    if any((p.group_size, p.bits) != (group_size, bits) for p in layers):
        raise ValueError(
            "Both projections must use the same quantization configuration"
        )
    flat = indices.flatten()
    order = mx.argsort(flat)
    # Keep the current inverse sort; this is not the earlier single-sort proposal.
    inverse = mx.argsort(order)
    rows = (order // indices.shape[-1]).astype(mx.uint32)
    experts = flat[order].astype(mx.uint32)
    weights = [(p.weight, p.scales, p.biases) for p in layers]
    up, gate = project_pair(
        x.reshape(-1, x.shape[-1]),
        *weights,
        rows,
        experts,
        mode=mode,
        group_size=group_size,
        bits=bits,
        validate=validate,
    )
    result = module.down_proj(module.activation(up, gate), experts, sorted_indices=True)
    return mx.unflatten(result[inverse], 0, indices.shape).squeeze(-2)
