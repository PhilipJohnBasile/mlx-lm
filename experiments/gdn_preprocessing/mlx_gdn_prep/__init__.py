"""Explicitly selected GDN preparation; default mode remains the reference.

No model is patched on import. Custom paths are forward-only. The two custom
paths are experimental until qualified on the caller's GPU and installed MLX.
"""
from functools import lru_cache
from typing import NamedTuple

from .geometry import Geometry
from .kernels import DIRECT, FUSED, HEADER


class Prepared(NamedTuple):
    q: object
    k: object
    v: object
    conv_state: object


def _mlx():
    import mlx.core as mx
    return mx


def _validate(qkv, weight, state, hk, hv, mask, lengths):
    mx = _mlx()
    if qkv.ndim != 3 or weight.ndim != 3 or weight.shape[-1] != 1:
        raise ValueError("Expected qkv[B,S,C] and depthwise weight[C,taps,1]")
    geom = Geometry(*qkv.shape[:2], hk, hv, weight.shape[1])
    if qkv.shape[2] != geom.channels or weight.shape[0] != geom.channels:
        raise ValueError("Channel count must equal (2*key_heads + value_heads)*128")
    if qkv.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("Only float16, bfloat16 and float32 are supported")
    if weight.dtype != qkv.dtype:
        raise ValueError("Convolution weights must have the activation dtype")
    if state is not None and (state.shape != geom.state_shape or state.dtype != qkv.dtype):
        raise ValueError("History shape/dtype mismatch")
    if mask is not None and (mask.shape != qkv.shape[:2] or mask.dtype != mx.bool_):
        raise ValueError("mask must be bool[B,S]")
    if lengths is not None and (lengths.shape != (geom.batch,) or
                                lengths.dtype not in (mx.int32, mx.int64)):
        raise ValueError("lengths must be int32/int64[B]")
    return geom


def _normalize(conv, geom):
    mx = _mlx()
    import mlx.nn as nn
    activated = nn.silu(conv)
    key_dim = geom.key_heads * 128
    q, k, v = mx.split(activated, [key_dim, key_dim * 2], axis=-1)
    q = q.reshape(geom.batch, geom.tokens, geom.key_heads, 128)
    k = k.reshape(geom.batch, geom.tokens, geom.key_heads, 128)
    v = v.reshape(geom.batch, geom.tokens, geom.value_heads, 128)
    inv_scale = 128**-0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    return q, k, v


@lru_cache(maxsize=2)
def _kernel(mode):
    mx = _mlx()
    outputs = ["out_q", "out_k", "out_v", "next_history"] if mode == "fused" else ["conv_out", "next_history"]
    return mx.fast.metal_kernel(
        name="gdn_prepare_" + mode,
        input_names=["qkv", "weight", "history", "mask", "lengths", "scales"],
        output_names=outputs, source=FUSED if mode == "fused" else DIRECT,
        header=HEADER, ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def prepare(qkv, weight, conv_state=None, *, key_heads, value_heads,
            mask=None, lengths=None, mode="reference", stream=None):
    """Prepare normalized Q/K, activated V and pre-convolution history.

    `lengths` selects the next history per request, independently of `mask`.
    A mask zeroes only new QKV inputs, exactly as in the pinned reference;
    it does not mask old history or forcibly zero a convolution's output.
    Negative/overlong lengths are clipped without CPU synchronization.
    mode: reference (default), direct (custom convolution only), fused (one
    custom dispatch for convolution, SiLU, normalization/scaling and history).
    No projections, recurrent updates, or weight quantization are performed.
    """
    if mode not in ("reference", "direct", "fused"):
        raise ValueError("mode must be reference, direct or fused")
    mx = _mlx()
    geom = _validate(qkv, weight, conv_state, key_heads, value_heads, mask, lengths)
    if mode != "reference" and not mx.metal.is_available():
        raise RuntimeError("The requested custom mode requires a Metal GPU")
    selected = mx.default_stream(mx.default_device() if mode == "reference" else mx.gpu) if stream is None else stream
    with mx.stream(selected):
        state = conv_state
        if state is None:
            state = mx.zeros(geom.state_shape, dtype=qkv.dtype)
        if mode == "reference":
            x = qkv if mask is None else mx.where(mask[..., None], qkv, 0)
            joined = mx.concatenate([state, x], axis=1)
            if lengths is None:
                new_state = mx.contiguous(joined[:, -(geom.taps - 1):, :])
            else:
                ends = mx.clip(lengths, 0, geom.tokens)
                positions = (ends[:, None] + mx.arange(geom.taps - 1))[..., None]
                new_state = mx.take_along_axis(joined, positions, axis=1)
            conv = mx.conv1d(joined, weight, groups=geom.channels)
            return Prepared(*_normalize(conv, geom), new_state)
        # Templates carry optional-input presence. Small arrays remain valid
        # constant-space pointers because the kernel accesses them directly.
        mask_arg = mx.array([True]) if mask is None else mask
        length_arg = mx.array([geom.tokens], dtype=mx.int32) if lengths is None else mx.clip(lengths, 0, geom.tokens).astype(mx.int32)
        scale = 128**-0.5
        scales = mx.array([scale**2, scale], dtype=mx.float32)
        if mode == "fused":
            shapes = [(geom.batch, geom.tokens, key_heads, 128)] * 2
            shapes += [(geom.batch, geom.tokens, value_heads, 128), geom.state_shape]
            grid, group = (32, geom.batch * geom.tokens * geom.heads, 1), (32, 1, 1)
        else:
            shapes = [qkv.shape, geom.state_shape]
            grid, group = (geom.channels, geom.tokens, geom.batch), (128, 1, 1)
        values = _kernel(mode)(
            inputs=[qkv, weight, state, mask_arg, length_arg, scales],
            output_shapes=shapes, output_dtypes=[qkv.dtype] * len(shapes),
            grid=grid, threadgroup=group,
            template=[("T", qkv.dtype), ("HK", key_heads), ("HV", value_heads),
                      ("TAPS", geom.taps), ("HAS_MASK", mask is not None),
                      ("HAS_LENGTHS", lengths is not None)],
            stream=selected,
        )
        if mode == "fused":
            return Prepared(*values)
        return Prepared(*_normalize(values[0], geom), values[1])
