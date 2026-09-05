"""Host-only tests. They do not validate a Metal kernel or GPU speed."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification_utils import (sample_routes, route_statistics, bitwise_equal,
                                 check_outputs, calibration_passes,
                                 valid_timing_summary)


def summary(lo=.99, hi=1.01, mean=1):
    return {"ci95": [lo, hi], "geomean_speedup": mean,
            "pairs": [{"a_ms": 1., "b_ms": 1., "speedup": 1.} for _ in range(5)]}


class TestRouting(unittest.TestCase):
    def test_each_token_has_distinct_experts(self):
        for distribution in ("uniform", "skewed"):
            ids = sample_routes(257, 256, 8, distribution)
            self.assertEqual(ids.dtype, np.uint32)
            self.assertTrue(all(len(set(row)) == 8 for row in ids))
            self.assertEqual(route_statistics(ids, 256)["tokens_with_duplicate_experts"], 0)

    def test_skew_is_not_two_expert_with_replacement(self):
        ids = sample_routes(1024, 256, 8, "skewed")
        stats = route_statistics(ids, 256)
        self.assertGreater(stats["active_experts"], 8)
        self.assertGreater(sum(stats["routes_per_expert"][:8]), 1024)
        self.assertEqual(sum(stats["routes_per_expert"]), 8192)

    def test_deterministic(self):
        a = sample_routes(200, 32, 8, "skewed", 123)
        np.testing.assert_array_equal(a, sample_routes(200, 32, 8, "skewed", 123))
        self.assertFalse(np.array_equal(a, sample_routes(200, 32, 8, "skewed", 124)))

    def test_all_experts(self):
        for dist in ("uniform", "skewed"):
            ids = sample_routes(3, 8, 8, dist)
            np.testing.assert_array_equal(np.sort(ids, axis=1), np.tile(np.arange(8), (3, 1)))

    def test_invalid_sampling(self):
        for args in ((1, 2, 8, "uniform"), (0, 8, 2, "uniform"),
                     (1, 8, 0, "skewed"), (1, 8, 2, "unknown")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                sample_routes(*args)

    def test_statistics_detect_duplicates(self):
        stats = route_statistics(np.array([[0, 0], [1, 2]], dtype=np.uint32), 4)
        self.assertEqual(stats["tokens_with_duplicate_experts"], 1)
        self.assertEqual(stats["empty_experts"], 1)

    def test_statistics_reject_bad_ids(self):
        for ids in (np.array([[0., 1.]]), np.array([[-1, 0]]), np.array([[0, 4]])):
            with self.assertRaises(ValueError):
                route_statistics(ids, 4)


class TestParity(unittest.TestCase):
    def test_signed_zero_is_not_bitwise_equal(self):
        a, b = np.array([0.], np.float16), np.array([-0.], np.float16)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(bitwise_equal(a, b, np))

    def test_nan_payloads(self):
        a = np.array([0x7fc00001], np.uint32).view(np.float32)
        b = np.array([0x7fc00002], np.uint32).view(np.float32)
        self.assertTrue(bitwise_equal(a, a.copy(), np))
        self.assertFalse(bitwise_equal(a, b, np))

    def test_supported_widths(self):
        for dtype in (np.float16, np.float32, np.float64):
            a = np.array([1., 2.], dtype)
            self.assertTrue(bitwise_equal(a, a.copy(), np))

    def test_triplet_passes(self):
        a = np.array([1., 2.], np.float16)
        report = check_outputs({key: a.copy() for key in ("upstream", "jit_contiguous", "jit_indirect")}, np, .001)
        self.assertTrue(report[0]["matched_jit_bitwise"])

    def test_no_broadcast_parity(self):
        a = np.ones((2, 2), np.float16)
        with self.assertRaises(ValueError):
            check_outputs({"upstream": a, "jit_contiguous": a, "jit_indirect": a[:1]}, np, .01)

    def test_no_zip_truncation(self):
        a = np.ones(2, np.float16)
        with self.assertRaises(ValueError):
            check_outputs({"upstream": (a, a), "jit_contiguous": (a,), "jit_indirect": (a,)}, np, .01)

    def test_all_paths_must_be_finite(self):
        for bad_mode in ("upstream", "jit_contiguous", "jit_indirect"):
            outputs = {key: np.ones(2, np.float16) for key in ("upstream", "jit_contiguous", "jit_indirect")}
            outputs[bad_mode][0] = np.inf
            with np.errstate(invalid="ignore"), self.assertRaises(RuntimeError):
                check_outputs(outputs, np, .01)

    def test_signed_zero_rejects_candidate(self):
        with self.assertRaises(RuntimeError):
            check_outputs({"upstream": np.array([0.], np.float16),
                           "jit_contiguous": np.array([0.], np.float16),
                           "jit_indirect": np.array([-0.], np.float16)}, np, .01)


class TestTiming(unittest.TestCase):
    def test_tight_calibration(self):
        self.assertTrue(calibration_passes(summary()))

    def test_mean_hides_wide_interval(self):
        self.assertFalse(calibration_passes(summary(.8, 1.2)))

    def test_nonfinite_timing(self):
        s = summary()
        s["pairs"][0]["a_ms"] = float("nan")
        self.assertFalse(valid_timing_summary(s))

    def test_missing_or_empty_timing(self):
        self.assertFalse(calibration_passes({}))
        s = summary()
        s["pairs"] = []
        self.assertFalse(calibration_passes(s))

    def test_reversed_interval(self):
        self.assertFalse(calibration_passes(summary(1.02, .98)))


if __name__ == "__main__":
    unittest.main()
