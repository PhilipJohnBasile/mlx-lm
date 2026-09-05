"""Native Metal tests. Run this file directly; unavailable Metal exits with code 2."""

import importlib.util
import itertools
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    mx = None
    nn = None

AVAILABLE = mx is not None and mx.metal.is_available()
if AVAILABLE:
    sys.path.insert(0, str(ROOT / "mlx_lm/models"))
    import _single_sort_moe as candidate


def load_switch_module(path, suffix):
    import mlx_lm.models

    sys.modules["mlx_lm.models._single_sort_moe"] = candidate
    name = f"mlx_lm.models._single_sort_test_{suffix}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(AVAILABLE, "Native MLX with Metal is required")
class TestSingleSortMetal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.set_default_device(mx.gpu)
        mx.random.seed(399)

    def assert_identical(self, a, b):
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.dtype, b.dtype)
        if a.dtype in (mx.float16, mx.bfloat16, mx.float32):
            word = mx.uint32 if a.dtype == mx.float32 else mx.uint16
            a, b = a.view(word), b.view(word)
        self.assertTrue(mx.array_equal(a, b).item())

    def check(self, x, indices, threadgroup_width=256):
        expanded = mx.expand_dims(x, (-2, -3))
        expected = candidate.gather_sort_baseline(expanded, indices)
        got = candidate.gather_sort_fused(
            expanded, indices, threadgroup_width=threadgroup_width
        )
        control = candidate.gather_sort_scatter(expanded, indices)
        mx.eval(*expected, *got, *control)
        for ref, actual, ablation in zip(expected, got, control):
            self.assert_identical(actual, ref)
            self.assert_identical(ablation, ref)
        restored = got[0][got[2]].reshape((*indices.shape, 1, x.shape[-1]))
        repeated = mx.broadcast_to(expanded, (*indices.shape, 1, x.shape[-1]))
        self.assert_identical(restored, repeated)

    def test_shape_dtype_and_topk_matrix(self):
        for leading, k, d, dtype in itertools.product(
            [(7,), (2, 5)], [1, 3, 8], [1, 33, 257],
            [mx.float16, mx.bfloat16, mx.float32]
        ):
            with self.subTest(shape=leading, k=k, d=d, dtype=str(dtype)):
                x = mx.random.normal((*leading, d)).astype(dtype)
                idx = mx.random.randint(0, 17, (*leading, k)).astype(mx.uint32)
                self.check(x, idx)

    def test_route_boundaries(self):
        for routes in [1, 63, 64, 65, 32767, 32768, 32769, 65537]:
            with self.subTest(routes=routes):
                x = mx.arange(routes * 3, dtype=mx.float32).reshape(routes, 3)
                idx = (mx.arange(routes, dtype=mx.uint32) % 17).reshape(routes, 1)
                self.check(x, idx)

    def test_index_dtypes_and_duplicate_experts(self):
        for dtype in [mx.int32, mx.uint32, mx.int64, mx.uint64]:
            with self.subTest(dtype=str(dtype)):
                x = mx.random.normal((17, 65))
                idx = mx.full((17, 8), 2, dtype=dtype)
                self.check(x, idx)

    def test_noncontiguous_activations(self):
        original = mx.random.normal((17, 130)).astype(mx.float16)
        layouts = [original[:, ::2], original[::-1, :65],
                   original[:, 64::-1], original[::-1, 64::-1],
                   mx.random.normal((65, 17)).T.astype(mx.float16),
                   mx.broadcast_to(original[:1, :65], (17, 65))]
        idx = mx.random.randint(0, 17, (17, 8)).astype(mx.uint32)
        for number, x in enumerate(layouts):
            with self.subTest(layout=number):
                self.check(x, idx)

    def test_noncontiguous_indices(self):
        x = mx.random.normal((17, 65)).astype(mx.bfloat16)
        storage = mx.random.randint(0, 17, (17, 16)).astype(mx.int32)
        for idx in [storage[:, ::2], storage[::-1, ::2], storage[:, 14::-2]]:
            self.check(x, idx)

    def test_float_bit_patterns(self):
        # Exhaust all 16-bit patterns, including NaNs, subnormals and signed zero.
        words = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16)
        idx = (mx.arange(1024, dtype=mx.uint32) % 17).reshape(1024, 1)
        for dtype in [mx.float16, mx.bfloat16]:
            with self.subTest(dtype=str(dtype)):
                self.check(words.view(dtype).reshape(1024, 64), idx)
        f32 = mx.array([0, 0x80000000, 1, 0x007FFFFF, 0x00800000,
                        0x7F800000, 0xFF800000, 0x7FC12345, 0x7F812345,
                        0x3F800000, 0xFFFFFFFF], dtype=mx.uint32).view(mx.float32)
        self.check(mx.broadcast_to(f32, (17, f32.size)), mx.zeros((17, 3), dtype=mx.uint32))

    def test_empty_inputs(self):
        for tokens, width in [(0, 65), (3, 0)]:
            x = mx.zeros((tokens, width), dtype=mx.float16)
            idx = mx.zeros((tokens, 8), dtype=mx.uint32)
            expanded = mx.expand_dims(x, (-2, -3))
            ref = candidate.gather_sort_baseline(expanded, idx)
            got = candidate.gather_sort_fused(expanded, idx)
            for a, b in zip(ref, got):
                self.assert_identical(a, b)

    def test_threadgroup_variants(self):
        x = mx.random.normal((17, 257))
        idx = mx.random.randint(0, 17, (17, 3)).astype(mx.uint32)
        for width in [32, 64, 128, 256]:
            with self.subTest(threadgroup_width=width):
                self.check(x, idx, width)

    def test_one_sort_and_kernel_engagement(self):
        x = mx.zeros((17, 1, 1, 65), dtype=mx.float16)
        idx = mx.zeros((17, 8), dtype=mx.uint32)
        original_argsort = mx.argsort
        for fun, count in [(candidate.gather_sort_baseline, 2),
                           (candidate.gather_sort_scatter, 1),
                           (candidate.gather_sort_fused, 1)]:
            with patch.object(mx, "argsort", wraps=original_argsort) as sort:
                mx.eval(*fun(x, idx))
                self.assertEqual(sort.call_count, count)
        with patch.object(candidate, "_pack_from_order", wraps=candidate._pack_from_order) as pack:
            mx.eval(*candidate.gather_sort_fused(x, idx))
            self.assertEqual(pack.call_count, 1)

    def test_separate_streams(self):
        streams = [mx.new_stream(mx.gpu), mx.new_stream(mx.gpu)]
        for stream in streams:
            with mx.stream(stream):
                self.check(mx.random.normal((17, 65)), mx.zeros((17, 8), dtype=mx.uint32))
            mx.synchronize(stream)

    def test_switch_blocks_and_training_fallback(self):
        module = load_switch_module(ROOT / "mlx_lm/models/switch_layers.py", "patched")
        for block_type, quantized in itertools.product([module.SwitchGLU, module.SwitchMLP], [False, True]):
            block = block_type(128, 256, 8)
            block.set_dtype(mx.float16)
            if quantized:
                nn.quantize(block, group_size=64, bits=4)
            block.eval()
            mx.eval(block.parameters())
            x = mx.random.normal((1, 64, 128)).astype(mx.float16)
            idx = mx.random.randint(0, 8, (1, 64, 4)).astype(mx.uint32)
            with patch.object(module, "enabled_for", return_value=False):
                ref = block(x, idx)
                mx.eval(ref)
            with patch.object(module, "enabled_for", return_value=True):
                got = block(x, idx)
                mx.eval(got)
            self.assert_identical(got, ref)
            block.train()
            with patch.object(module, "enabled_for", return_value=True), patch.object(
                module, "gather_sort_fused", side_effect=AssertionError("training used custom kernel")
            ):
                mx.eval(block(x, idx))
                if not quantized:
                    grad = mx.grad(lambda a: block(a, idx).astype(mx.float32).sum())(x)
                    mx.eval(grad)
                    self.assertTrue(mx.all(mx.isfinite(grad)).item())


if __name__ == "__main__":
    if not AVAILABLE:
        print("NOT RUN: native MLX with a Metal GPU is required.", file=sys.stderr)
        raise SystemExit(2)
    unittest.main(verbosity=2)
