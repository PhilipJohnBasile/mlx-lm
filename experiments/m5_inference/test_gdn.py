"""Native integration and selection-lifecycle regressions."""

import unittest
from types import SimpleNamespace

import mlx.core as mx
from gdn import select_gdn

from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet


def fixture(dtype=mx.float32):
    args = SimpleNamespace(
        hidden_size=64,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    module = GatedDeltaNet(args)
    module.set_dtype(dtype)
    module.eval()
    layer = SimpleNamespace(is_linear=True, linear_attn=module)
    return SimpleNamespace(layers=[layer]), layer, module


class SelectionTests(unittest.TestCase):
    def test_restores_on_exception_and_keeps_weight_identity(self):
        model, layer, original = fixture()
        weight = original.in_proj_qkv.weight
        with self.assertRaisesRegex(RuntimeError, "request failed"):
            with select_gdn(model, "fused"):
                self.assertIs(layer.linear_attn.original.in_proj_qkv.weight, weight)
                raise RuntimeError("request failed")
        self.assertIs(layer.linear_attn, original)
        self.assertIs(original.in_proj_qkv.weight, weight)

    def test_rejects_partial_admission_before_mutating_any_layer(self):
        model, layer, original = fixture()
        model.layers.append(SimpleNamespace(is_linear=True, linear_attn=object()))
        with self.assertRaisesRegex(ValueError, "inspected"):
            with select_gdn(model, "fused"):
                self.fail("Unsupported model was admitted")
        self.assertIs(layer.linear_attn, original)

    def test_reference_fallback_and_training_rejection(self):
        model, layer, original = fixture()
        original.train()
        with select_gdn(model, "reference"):
            self.assertIs(layer.linear_attn, original)
        with self.assertRaisesRegex(ValueError, "inference"):
            with select_gdn(model, "direct"):
                self.fail("Training was admitted")
        self.assertIs(layer.linear_attn, original)

    def test_ragged_batch_cache_then_following_token(self):
        mx.random.seed(20260905)
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            model, layer, original = fixture(dtype)
            x = (mx.random.normal((4, 12, 64)) * 0.1).astype(dtype)
            for mode in ("direct", "fused"):
                with self.subTest(dtype=str(dtype), mode=mode):
                    a, b = ArraysCache(2), ArraysCache(2)
                    a.prepare(lengths=[10, 8, 6, 4])
                    b.prepare(lengths=[10, 8, 6, 4])
                    for lo, hi in ((0, 1), (1, 4), (4, 10), (10, 11), (11, 12)):
                        if lo == 10:
                            a.finalize()
                            b.finalize()
                        ref = original(x[:, lo:hi], mask=a.make_mask(hi - lo), cache=a)
                        with select_gdn(model, mode):
                            got = layer.linear_attn(
                                x[:, lo:hi], mask=b.make_mask(hi - lo), cache=b
                            )
                            mx.eval(ref, got, a.state, b.state)
                        atol, rtol = {
                            mx.float32: (2e-5, 1e-4),
                            mx.float16: (1e-4, 2e-3),
                            mx.bfloat16: (8e-4, 1.5e-2),
                        }[dtype]
                        self.assertTrue(
                            mx.allclose(ref, got, atol=atol, rtol=rtol).item()
                        )
                        word = mx.uint32 if dtype == mx.float32 else mx.uint16
                        self.assertTrue(
                            mx.array_equal(a[0].view(word), b[0].view(word)).item()
                        )
                        self.assertTrue(
                            mx.allclose(a[1], b[1], atol=atol, rtol=rtol).item()
                        )
                        if a.lengths is not None:
                            self.assertTrue(mx.array_equal(a.lengths, b.lengths).item())
                        self.assertIs(layer.linear_attn, original)


if __name__ == "__main__":
    if not mx.metal.is_available():
        raise SystemExit("Native Metal is required")
    unittest.main(verbosity=2)
