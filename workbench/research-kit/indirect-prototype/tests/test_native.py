"""Real Metal tests. --require-metal fails rather than silently skipping."""
import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification_utils import bitwise_equal
from mlx_nax_indirect import gather_qmm, project_pair, require_m5_nax, validate_routing


def make_case(tokens, routes, experts, k, n, dtype, group=64, bits=4, skew=False):
    import mlx.core as mx
    mx.random.seed(121)
    x = (mx.random.normal((tokens, k)) * k**-0.5).astype(dtype)
    dense = (mx.random.normal((experts, n, k)) * 0.3).astype(dtype)
    w = mx.quantize(dense, group_size=group, bits=bits)
    rows = mx.random.randint(0, tokens, (routes,)).astype(mx.uint32)
    idx = mx.zeros((routes,), dtype=mx.uint32) if skew else mx.sort(mx.random.randint(0, experts, (routes,)).astype(mx.uint32))
    mx.eval(x, *w, rows, idx)
    return x, w, rows, idx


class TestNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            require_m5_nax()
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(str(exc))
        import mlx.core as mx
        cls.mx = mx

    def test_direct_vs_contiguous_bitwise(self):
        mx = self.mx
        for dtype in (mx.float16, mx.bfloat16):
            for routes in (8, 31, 32, 33, 64, 65, 127, 128, 129):
                for skew in (False, True):
                    with self.subTest(dtype=dtype, routes=routes, skew=skew):
                        x, w, rows, ids = make_case(17, routes, 4, 128, 128, dtype, skew=skew)
                        indirect = gather_qmm(x, *w, rows, ids)
                        control = gather_qmm(x[rows], *w, rows, ids, indirect=False)
                        mx.eval(indirect, control)
                        self.assertTrue(bitwise_equal(indirect, control, mx))

    def test_quantization_configurations(self):
        mx = self.mx
        for dtype in (mx.float16, mx.bfloat16):
            for group in (64, 128):
                for bits in (4, 8):
                    with self.subTest(dtype=dtype, group=group, bits=bits):
                        x, w, rows, ids = make_case(63, 256, 4, 256, 128, dtype, group, bits)
                        got = gather_qmm(x, *w, rows, ids, group_size=group, bits=bits)
                        expected = mx.gather_qmm(x[rows, None, :], *w, rhs_indices=ids,
                                                 group_size=group, bits=bits, sorted_indices=True)
                        mx.eval(got, expected)
                        tolerance = 2e-3 if dtype == mx.float16 else 2e-2
                        self.assertTrue(mx.allclose(got, expected, atol=tolerance, rtol=tolerance).item())

    def test_independent_dense_reference(self):
        import numpy as np
        mx = self.mx
        x, w, rows, ids = make_case(17, 65, 4, 128, 64, mx.float16)
        got = gather_qmm(x, *w, rows, ids)
        weight = mx.dequantize(*w, group_size=64, bits=4).astype(mx.float32)
        xn = np.array(x.astype(mx.float32)).astype(np.float64)
        wn = np.array(weight).astype(np.float64)
        rn, en = np.array(rows), np.array(ids)
        expected = np.einsum("rk,rnk->rn", xn[rn], wn[en])
        np.testing.assert_allclose(np.array(got[:, 0, :].astype(mx.float32)), expected, atol=3e-3, rtol=3e-3)

    def test_repeated_source_rows(self):
        mx = self.mx
        x, w, rows, ids = make_case(1, 65, 4, 128, 128, mx.float16)
        a = gather_qmm(x, *w, rows, ids)
        b = gather_qmm(x[rows], *w, rows, ids, indirect=False)
        self.assertTrue(bitwise_equal(a, b, mx))

    def test_noncontiguous_source_copy(self):
        mx = self.mx
        x, w, rows, ids = make_case(17, 65, 4, 128, 64, mx.float16)
        x = mx.stack((x, x), axis=-1)[..., 0]
        a = gather_qmm(x, *w, rows, ids)
        b = gather_qmm(x[rows], *w, rows, ids, indirect=False)
        self.assertTrue(bitwise_equal(a, b, mx))

    def test_bad_indices_rejected_by_checked_api(self):
        mx = self.mx
        x, w, rows, ids = make_case(17, 64, 4, 128, 64, mx.float16)
        with self.assertRaises(ValueError):
            gather_qmm(x, *w, mx.full(rows.shape, 17, mx.uint32), ids)
        with self.assertRaises(ValueError):
            validate_routing(rows, mx.full(ids.shape, 4, mx.uint32), 17, 4)

    def test_unchecked_invalid_indices_poison_without_overread(self):
        mx = self.mx
        x, w, rows, ids = make_case(17, 65, 4, 128, 64, mx.float16)
        a = gather_qmm(x, *w, mx.full(rows.shape, 0xffffffff, mx.uint32), ids, validate=False)
        b = gather_qmm(x, *w, rows, mx.full(ids.shape, 0xffffffff, mx.uint32), validate=False)
        self.assertTrue(mx.all(mx.isnan(a)).item())
        self.assertTrue(mx.all(mx.isnan(b)).item())

    def test_switch_glu_integration(self):
        try:
            import mlx.nn as nn
            from mlx_lm.models.switch_layers import SwitchGLU
        except ImportError:
            self.skipTest("mlx-lm is required for the full SwitchGLU integration test")
        from mlx_nax_indirect import switch_glu
        mx = self.mx
        mx.random.seed(271)
        module = SwitchGLU(128, 128, 4)
        module.set_dtype(mx.float16)
        nn.quantize(module, group_size=64, bits=4)
        module.eval()
        x = mx.random.normal((1, 17, 128)).astype(mx.float16)
        ids = mx.random.randint(0, 4, (1, 17, 4)).astype(mx.uint32)
        expected = module(x, ids)
        got = switch_glu(module, x, ids, mode="jit_indirect")
        control = switch_glu(module, x, ids, mode="jit_contiguous")
        self.assertTrue(bitwise_equal(got, control, mx))
        self.assertTrue(mx.allclose(got, expected, atol=2e-3, rtol=2e-3).item())

    def test_pair(self):
        mx = self.mx
        x, w, rows, ids = make_case(17, 65, 4, 128, 128, mx.float16)
        a = project_pair(x, w, w, rows, ids, mode="jit_indirect")
        b = project_pair(x, w, w, rows, ids, mode="jit_contiguous")
        for first, second in zip(a, b):
            self.assertTrue(bitwise_equal(first, second, mx))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--require-metal", action="store_true")
    args, rest = parser.parse_known_args()
    if args.require_metal:
        require_m5_nax()
    unittest.main(argv=[sys.argv[0], *rest])
