# SPDX-License-Identifier: MIT

"""Experimental single-sort routing for inference; disabled by default.

The Metal path is not an autodiff primitive. Callers doing training, grad,
jvp or vmap must use the existing MLX routing path. No speed claim is made
until the included native benchmark passes on the target machine.
"""

import math
import os
import re
from functools import lru_cache

import mlx.core as mx


# Set before importing mlx_lm. Do not change this inside a compiled graph.
_ENABLED = os.environ.get("MLX_MOE_SINGLE_SORT", "0") == "1"

# x contains uint16/uint32 bit patterns, not floating-point values.
PACK_SOURCE = r"""
const uint column = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
const uint width = uint(x_shape[2]);
const uint routes = uint(order_shape[0]);
if (column >= width || row >= routes) {
  return;
}

const uint original = order[long(row) * long(order_strides[0])];
const uint token = original / TopK;
const long offset = long(token) * long(x_strides[0]) +
    long(column) * long(x_strides[2]);
packed[size_t(row) * size_t(width) + size_t(column)] = x[offset];

if (column == 0) {
  sorted_experts[row] = experts[long(original) * long(experts_strides[0])];
  inverse[original] = row;
}
"""


@lru_cache(maxsize=1)
def _pack_kernel():
    return mx.fast.metal_kernel(
        name="moe_pack_and_invert_v1",
        input_names=["x", "experts", "order"],
        output_names=["packed", "sorted_experts", "inverse"],
        source=PACK_SOURCE,
        ensure_row_contiguous=False,
        atomic_outputs=False,
    )


@lru_cache(maxsize=1)
def _m5_or_later():
    if not mx.metal.is_available():
        return False
    info = mx.device_info(mx.gpu)
    name = str(info.get("device_name", ""))
    match = re.search(r"\bApple M(\d+)\b", name)
    return match is not None and int(match.group(1)) >= 5


def _layout(x, indices):
    if indices.ndim < 1 or x.ndim != indices.ndim + 2:
        raise ValueError("Expected expanded activations [..., 1, 1, D].")
    if x.shape[-3:-1] != (1, 1) or x.shape[:-3] != indices.shape[:-1]:
        raise ValueError("Activations and expert indices have different token shapes.")
    top_k = indices.shape[-1]
    if top_k < 1:
        raise ValueError("The number of routes per token must be positive.")
    routes = indices.size
    if routes > 2**31 - 1:
        raise ValueError("The flattened route dimension must fit in int32.")
    return math.prod(indices.shape[:-1]), top_k, routes, x.shape[-1]


def enabled_for(x, indices):
    """Opt-in gate used only by non-training SwitchGLU/SwitchMLP calls."""
    if not _ENABLED or mx.default_device() != mx.gpu or not _m5_or_later():
        return False
    if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        return False
    if indices.dtype not in (mx.int32, mx.uint32, mx.int64, mx.uint64):
        return False
    try:
        _, _, routes, width = _layout(x, indices)
    except ValueError:
        return False
    return routes >= 64 and width > 0


def gather_sort_baseline(x, indices):
    """The unmodified mlx-lm routing algorithm (benchmark control)."""
    *_, top_k = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inverse = mx.argsort(order)
    return x.flatten(0, -3)[order // top_k], indices[order], inverse


def gather_sort_scatter(x, indices):
    """Single-sort ablation using ordinary MLX ops, not the fused kernel."""
    _, top_k, _, _ = _layout(x, indices)
    indices = indices.flatten()
    order = mx.argsort(indices)
    inverse = mx.zeros_like(order)
    inverse[order] = mx.arange(order.size, dtype=order.dtype)
    return x.flatten(0, -3)[order // top_k], indices[order], inverse


def _pack_from_order(x, experts, order, top_k, threadgroup_width=256):
    """Private: order must be the uint32 permutation from argsort(experts)."""
    if order.dtype != mx.uint32:
        raise ValueError("Expected a uint32 argsort permutation.")
    if not 1 <= threadgroup_width <= 256:
        raise ValueError("threadgroup_width must be in [1, 256].")
    width, routes = x.shape[-1], experts.size
    if width == 0 or routes == 0:
        raise ValueError("Empty tensors are handled before Metal dispatch.")
    word_type = mx.uint32 if x.dtype == mx.float32 else mx.uint16
    bits = x.view(word_type)
    outputs = _pack_kernel()(
        inputs=[bits, experts, order],
        template=[("TopK", top_k)],
        grid=(
            (width + threadgroup_width - 1) // threadgroup_width * threadgroup_width,
            routes,
            1,
        ),
        threadgroup=(threadgroup_width, 1, 1),
        output_shapes=[(routes, 1, width), (routes,), (routes,)],
        output_dtypes=[word_type, experts.dtype, mx.uint32],
    )
    return outputs[0].view(x.dtype), outputs[1], outputs[2]


def gather_sort_fused(x, indices, *, threadgroup_width=256):
    """One expert sort followed by a fused pack/metadata/inverse kernel.

    Inference only. This direct entry point runs on any Metal GPU for A/B
    controls; automatic integration is opt-in and gated to M5-family or later
    named devices. Future-chip performance is not assumed.
    """
    _, top_k, routes, width = _layout(x, indices)
    if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("Supported activation dtypes: float16, bfloat16, float32.")
    if indices.dtype not in (mx.int32, mx.uint32, mx.int64, mx.uint64):
        raise ValueError("Expert indices must use a 32- or 64-bit integer dtype.")
    if routes == 0 or width == 0:
        return gather_sort_baseline(x, indices)
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("The fused routing experiment requires a Metal GPU stream.")
    experts = indices.flatten()
    order = mx.argsort(experts)
    return _pack_from_order(
        x.flatten(0, -3), experts, order, top_k, threadgroup_width
    )
