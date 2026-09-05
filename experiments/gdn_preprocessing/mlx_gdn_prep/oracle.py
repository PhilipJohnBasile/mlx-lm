"""NumPy semantic oracle, not a substitute for Metal correctness testing."""
import numpy as np
from .geometry import Geometry, window_location


def round_dtype(a, dtype):
    a = np.asarray(a, dtype=np.float32)
    if dtype == "float32":
        return a
    if dtype == "float16":
        return a.astype(np.float16).astype(np.float32)
    if dtype != "bfloat16":
        raise ValueError("Unknown dtype")
    bits = a.view(np.uint32)
    rounded = bits + np.uint32(0x7fff) + ((bits >> np.uint32(16)) & np.uint32(1))
    nan = np.isnan(a)
    rounded = np.where(nan, bits | np.uint32(0x400000), rounded).astype(np.uint32)
    return (rounded & np.uint32(0xffff0000)).view(np.float32)


def _postprocess(conv, geom, dtype):
    x = round_dtype(conv, dtype)
    # Float32 exp is a semantic control, not Metal's half/bfloat overload.
    with np.errstate(over="ignore", invalid="ignore"):
        z = 1 / (1 + np.exp(np.abs(x)))
        sigmoid = round_dtype(np.where(x < 0, z, 1 - z), dtype)
        a = round_dtype(x * sigmoid, dtype)
    keydim = geom.key_heads * 128
    q, k, v = np.split(a, (keydim, 2 * keydim), axis=-1)
    def norm_scale(y, scale):
        y = y.reshape(geom.batch, geom.tokens, geom.key_heads, 128)
        inv = 1 / np.sqrt(np.mean(y * y, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-6))
        return round_dtype(round_dtype(y * inv, dtype) * round_dtype(scale, dtype), dtype)
    scale = 128**-0.5
    return norm_scale(q, scale**2), norm_scale(k, scale), v.reshape(geom.batch, geom.tokens, geom.value_heads, 128)


def reference(qkv, weight, state, geom, mask=None, lengths=None, dtype="float32"):
    qkv, weight, state = (round_dtype(a, dtype) for a in (qkv, weight, state))
    x = qkv if mask is None else np.where(mask[..., None], qkv, np.float32(0))
    joined = np.concatenate((state, x), axis=1)
    conv = np.zeros_like(qkv)
    for tap in range(geom.taps):
        conv = conv + joined[:, tap:tap + geom.tokens] * weight[:, tap, 0]
    ends = np.full(geom.batch, geom.tokens) if lengths is None else np.clip(lengths, 0, geom.tokens)
    new_state = np.stack([joined[b, int(end):int(end) + geom.taps - 1] for b, end in enumerate(ends)])
    return (*_postprocess(conv, geom, dtype), new_state)


def direct(qkv, weight, state, geom, mask=None, lengths=None, dtype="float32"):
    qkv, weight, state = (round_dtype(a, dtype) for a in (qkv, weight, state))
    conv = np.zeros_like(qkv)
    def read(b, pos, c):
        old, off = window_location(b, pos, c, geom.tokens, geom.channels, geom.taps - 1)
        if old:
            return state.flat[off]
        t = pos - (geom.taps - 1)
        return qkv.flat[off] if mask is None or mask[b, t] else np.float32(0)
    for b in range(geom.batch):
        for t in range(geom.tokens):
            for c in range(geom.channels):
                acc = np.float32(0)
                for tap in range(geom.taps):
                    acc = acc + read(b, t + tap, c) * weight[c, tap, 0]
                conv[b, t, c] = acc
    new_state = np.empty_like(state)
    for b in range(geom.batch):
        end = geom.tokens if lengths is None else int(np.clip(lengths[b], 0, geom.tokens))
        for j in range(geom.taps - 1):
            for c in range(geom.channels):
                new_state[b, j, c] = read(b, end + j, c)
    return (*_postprocess(conv, geom, dtype), new_state)
