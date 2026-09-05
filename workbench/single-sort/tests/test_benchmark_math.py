"""Host tests for benchmark reporting; no synthetic GPU timings are emitted."""

import argparse
import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "tools/bench_single_sort.py"
spec = importlib.util.spec_from_file_location("benchmark_math", path)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class TestBenchmarkMath(unittest.TestCase):
    def test_aa_identity(self):
        result = bench.bootstrap_ratio([[1.0, 1.0]] * 15)
        self.assertEqual(result["geomean_baseline_over_candidate"], 1.0)
        self.assertEqual(result["paired_bootstrap_95pct"], [1.0, 1.0])

    def test_ratio_direction(self):
        result = bench.bootstrap_ratio([[2.0, 1.0]] * 15)
        self.assertAlmostEqual(result["geomean_baseline_over_candidate"], 2.0)
        self.assertEqual(result["paired_bootstrap_95pct"], [2.0, 2.0])

    def test_deterministic_pair_resampling(self):
        pairs = [[2.0, 1.1], [1.8, 1.1], [2.1, 1.0], [2.0, 1.2], [2.2, 1.0]]
        self.assertEqual(bench.bootstrap_ratio(pairs), bench.bootstrap_ratio(pairs))

    def test_case_validation(self):
        self.assertEqual(bench.parse_case("512,8,2048,64"), (512, 8, 2048, 64))
        for value in ["1,2,3", "0,8,2048,64", "1,8,2048,4", "x,8,2048,64"]:
            with self.assertRaises(argparse.ArgumentTypeError):
                bench.parse_case(value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
