"""Host-only contract and permutation tests. These do not import native MLX."""

import ast
import hashlib
import itertools
import math
import unittest
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "mlx_lm/models/_single_sort_moe.py"


def load_function(name, **namespace):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    scope = {"math": math, "lru_cache": lru_cache, **namespace}
    exec(compile(module, str(SOURCE), "exec"), scope)
    return scope[name]


def shape_object(shape):
    return SimpleNamespace(shape=tuple(shape), ndim=len(shape), size=math.prod(shape))


class TestHostContract(unittest.TestCase):
    def test_baseline_blob(self):
        data = (ROOT / "baseline/mlx_lm/models/switch_layers.py").read_bytes()
        digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        self.assertEqual(digest, "1fe5d917e6b194b1681bbb1c69589ad3dc759d65")

    def test_layout_shapes(self):
        layout = load_function("_layout")
        for leading in [(), (1,), (2, 7), (2, 3, 5)]:
            for top_k in [1, 3, 8]:
                x = shape_object((*leading, 1, 1, 65))
                idx = shape_object((*leading, top_k))
                self.assertEqual(
                    layout(x, idx),
                    (math.prod(leading), top_k, math.prod(leading) * top_k, 65),
                )

    def test_empty_shapes(self):
        layout = load_function("_layout")
        self.assertEqual(layout(shape_object((0, 1, 1, 65)), shape_object((0, 8))), (0, 8, 0, 65))
        self.assertEqual(layout(shape_object((3, 1, 1, 0)), shape_object((3, 8))), (3, 8, 24, 0))

    def test_invalid_layouts(self):
        layout = load_function("_layout")
        for xs, ids in [((2, 64), (2, 8)), ((2, 1, 2, 64), (2, 8)), ((3, 1, 1, 64), (2, 8)), ((2, 1, 1, 64), (2, 0))]:
            with self.subTest(x=xs, indices=ids), self.assertRaises(ValueError):
                layout(shape_object(xs), shape_object(ids))

    def test_permutation_width_guard(self):
        layout = load_function("_layout")
        with self.assertRaises(ValueError):
            layout(shape_object((2**29, 1, 1, 64)), shape_object((2**29, 8)))

    def test_every_small_permutation(self):
        tested = 0
        for n in range(1, 9):
            for values in itertools.permutations(range(n)):
                order = np.array(values, dtype=np.uint32)
                inverse = np.empty(n, dtype=np.uint32)
                inverse[order] = np.arange(n, dtype=np.uint32)
                np.testing.assert_array_equal(inverse, np.argsort(order))
                tested += 1
        self.assertEqual(tested, 46233)

    def test_duplicate_experts_and_roundtrip(self):
        rng = np.random.default_rng(399)
        for tokens, k, width in itertools.product([1, 7, 65], [1, 3, 8], [1, 33, 65]):
            for experts in [1, 17]:
                x = rng.integers(0, 2**32, size=(tokens, width), dtype=np.uint32)
                ids = rng.integers(0, experts, size=tokens*k, dtype=np.int32)
                order = np.argsort(ids, kind="stable")
                packed = np.empty((tokens*k, width), dtype=x.dtype)
                sorted_ids = np.empty_like(ids)
                inverse = np.empty(tokens*k, dtype=np.uint32)
                for r, original in enumerate(order):
                    packed[r] = x[original // k]
                    sorted_ids[r] = ids[original]
                    inverse[original] = r
                np.testing.assert_array_equal(packed[inverse], np.repeat(x, k, axis=0))
                np.testing.assert_array_equal(sorted_ids, ids[order])
                np.testing.assert_array_equal(inverse, np.argsort(order))

    def test_disabled_gate_does_not_query_gpu(self):
        mx = Mock()
        gate = load_function("enabled_for", _ENABLED=False, mx=mx)
        self.assertFalse(gate(None, None))
        self.assertEqual(mx.mock_calls, [])

    def test_m5_device_selector(self):
        import re
        for name, expected in [("Apple M4 Max", False), ("Apple M5", True), ("Apple M5 Max", True), ("Apple M6", True), ("Apple A19 Pro", False), ("unknown", False)]:
            mx = SimpleNamespace(gpu="gpu", metal=SimpleNamespace(is_available=lambda: True), device_info=lambda device: {"device_name": name})
            test = load_function("_m5_or_later", mx=mx, re=re)
            self.assertEqual(test(), expected, name)

    def test_training_calls_do_not_enable_custom_kernel(self):
        tree = ast.parse((ROOT / "mlx_lm/models/switch_layers.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_gather_sort"]
        self.assertEqual(len(calls), 2)
        for call in calls:
            kw = next(k for k in call.keywords if k.arg == "allow_fused")
            self.assertEqual(ast.unparse(kw.value), "not self.training")

    def test_route_count_is_not_a_jit_template(self):
        tree = ast.parse(SOURCE.read_text())
        templates = [k.value for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords if k.arg == "template"]
        self.assertEqual(len(templates), 1)
        self.assertEqual(ast.unparse(templates[0]), "[('TopK', top_k)]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
