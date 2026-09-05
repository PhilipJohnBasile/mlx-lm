"""Explicit adapter for the pinned Qwen3.5-family GatedDeltaNet interface."""
from . import prepare


def forward(module, inputs, mask=None, cache=None, *, mode="reference"):
    """Call a GDN layer with only its preprocessing changed; no global hooks.

    Shared by dense and MoE Qwen3.5-family GDN layers. Projection quantization
    is untouched. Training, tensor parallelism, and non-128 heads are excluded
    from the experimental adapter, not claimed to be qualified.
    """
    if mode == "reference":
        return module(inputs, mask=mask, cache=cache)
    if mode not in ("direct", "fused"):
        raise ValueError("Unknown preprocessing mode")
    if module.training or getattr(module, "sharding_group", None) is not None:
        raise ValueError("Custom adapter supports single-device inference only")
    if (module.head_k_dim, module.head_v_dim) != (128,128):
        raise ValueError("Custom adapter requires 128-wide key/value heads")
    from mlx_lm.models.qwen3_5 import gated_delta_update
    B, S, _ = inputs.shape
    qkv = module.in_proj_qkv(inputs)
    z = module.in_proj_z(inputs).reshape(B, S, module.num_v_heads, module.head_v_dim)
    b, a = module.in_proj_b(inputs), module.in_proj_a(inputs)
    conv_state = cache[0] if cache is not None else None
    state = cache[1] if cache is not None else None
    lengths = cache.lengths if cache is not None else None
    prepared = prepare(qkv, module.conv1d.weight, conv_state,
                       key_heads=module.num_k_heads, value_heads=module.num_v_heads,
                       mask=mask, lengths=lengths, mode=mode)
    out, new_state = gated_delta_update(
        prepared.q, prepared.k, prepared.v, a, b,
        module.A_log, module.dt_bias, state, mask, use_kernel=True)
    if cache is not None:
        cache[0], cache[1] = prepared.conv_state, new_state
        cache.advance(S)
    out = module.norm(out, z)
    return module.out_proj(out.reshape(B,S,-1))
