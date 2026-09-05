import dataclasses
import unittest

from mlx_nax_indirect import kernel_sources
from mlx_nax_indirect.policy import Geometry, compatible_device


class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.geometry = Geometry(4096, 32768, 64, 4096, 2048, 64, 4, "float16")

    def test_admitted_shapes(self):
        for dtype in ("float16", "bfloat16"):
            for bits in (4, 8):
                for group in (64, 128):
                    dataclasses.replace(
                        self.geometry, dtype=dtype, bits=bits, group_size=group
                    ).validate()

    def test_invalid_geometry(self):
        for change in (
            {"k": 65},
            {"n": 65},
            {"bits": 3},
            {"group_size": 32},
            {"source_rows": 0},
            {"routes": 7},
            {"routes": 32769},
            {"dtype": "float32"},
            {"k": 65536, "n": 65536},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                dataclasses.replace(self.geometry, **change).validate()

    def test_row_tile_boundary(self):
        self.assertEqual(dataclasses.replace(self.geometry, routes=4095).bm, 32)
        self.assertEqual(dataclasses.replace(self.geometry, routes=4096).bm, 64)

    def test_saved_allocation(self):
        self.assertEqual(self.geometry.eliminated_gather_bytes, 256 * 1024**2)

    def test_supported_device_policy(self):
        self.assertTrue(
            compatible_device(
                {"device_name": "Apple M5 Max", "architecture": "applegpu_g17s"}, "26.2"
            )
        )
        self.assertTrue(
            compatible_device(
                {"device_name": "Apple M5", "architecture": "applegpu_g18p"}, "26.5.2"
            )
        )
        # Synthetic future-device policy checks, not evidence of hardware support.
        self.assertTrue(
            compatible_device(
                {"device_name": "Apple M9", "architecture": "applegpu_g21x"}, "30.0"
            )
        )

    def test_rejected_device_policy(self):
        for name, arch, os in (
            ("Apple M4 Max", "applegpu_g16s", "26.5"),
            ("Apple M5 Max", "applegpu_g17s", "26.1"),
            ("Apple M5", "applegpu_g17p", "26.2"),
            ("Apple Paravirtual device", "unknown", "26.2"),
            ("Apple M5 Max", "", "26.2"),
        ):
            self.assertFalse(
                compatible_device({"device_name": name, "architecture": arch}, os)
            )
        self.assertFalse(compatible_device({}, "not-macos"))

    def test_kernel_is_self_contained(self):
        header, body = kernel_sources()
        self.assertNotIn('#include "mlx/', header)
        self.assertNotIn("[[function_constant(", header)
        self.assertIn("mpp::tensor_ops::matmul2d", header)
        self.assertIn("row_map.initialize", body)
        self.assertIn("row_map.load(Atile, x_base", body)
        self.assertNotIn("xn += BK;\n          loader_w", body)

    def test_row_map_is_outside_inner_loops(self):
        _, body = kernel_sources()
        self.assertLess(
            body.index("row_map.initialize"), body.index("while (n < tgp_bm)")
        )
        self.assertEqual(body.count("row_map.initialize("), 1)
        self.assertNotIn("if (!align_K)", body)


if __name__ == "__main__":
    unittest.main()
