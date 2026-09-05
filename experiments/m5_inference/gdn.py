"""Explicit, model-local GDN selection using the preserved handoff kernels."""

import sys
from contextlib import contextmanager
from pathlib import Path

import mlx.nn as nn

from mlx_lm.models.qwen3_5 import GatedDeltaNet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gdn_preprocessing"))
from mlx_gdn_prep.integration import forward


class SelectedGDN(nn.Module):
    def __init__(self, original, mode):
        super().__init__()
        self.original = original
        self.mode = mode

    def __call__(self, inputs, mask=None, cache=None):
        return forward(self.original, inputs, mask=mask, cache=cache, mode=self.mode)


@contextmanager
def select_gdn(model, mode):
    """Temporarily select a mode on one model; retain the same weight arrays."""
    if mode not in ("reference", "direct", "fused"):
        raise ValueError("Expected reference, direct or fused")
    if mode == "reference":
        yield
        return
    originals = []
    for layer in model.layers:
        if not getattr(layer, "is_linear", False):
            continue
        module = layer.linear_attn
        if type(module) is not GatedDeltaNet:
            raise ValueError("Only the inspected Qwen3.5 GatedDeltaNet is supported")
        if module.training or module.sharding_group is not None:
            raise ValueError("Single-device inference is required")
        if (module.head_k_dim, module.head_v_dim) != (128, 128):
            raise ValueError("128-wide GDN heads are required")
        originals.append((layer, module))
    if not originals:
        raise ValueError("No compatible GDN layers found")
    try:
        for layer, module in originals:
            layer.linear_attn = SelectedGDN(module, mode)
        yield
    finally:
        for layer, module in originals:
            layer.linear_attn = module
