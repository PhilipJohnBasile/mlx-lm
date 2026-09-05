"""Prevent incomplete checkpoint files from being reported as complete."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

from inventory import inspect_model


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5_text"})
        )

    def shard(self, filename, name="weight", payload=b"\x00" * 4):
        header = json.dumps(
            {name: {"dtype": "U32", "shape": [1], "data_offsets": [0, 4]}}
        ).encode()
        (self.root / filename).write_bytes(
            struct.pack("<Q", len(header)) + header + payload
        )

    def index(self, mapping):
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": mapping})
        )

    def test_complete_indexed_file(self):
        self.shard("model.safetensors")
        self.index({"weight": "model.safetensors"})
        self.assertTrue(inspect_model(self.root)["complete_weight_inventory"])

    def test_numbered_shards_detect_missing_download_without_index(self):
        self.shard("model-00003-of-00003.safetensors")
        result = inspect_model(self.root)
        self.assertFalse(result["complete_weight_inventory"])
        self.assertEqual(
            result["missing_shards"],
            ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors"],
        )

    def test_truncated_payload_is_rejected(self):
        self.shard("model.safetensors", payload=b"\x00")
        self.index({"weight": "model.safetensors"})
        result = inspect_model(self.root)
        self.assertFalse(result["complete_weight_inventory"])
        self.assertEqual(
            result["invalid_payloads"],
            [{"file": "model.safetensors", "tensor": "weight"}],
        )

    def test_tensor_in_wrong_file_does_not_satisfy_index(self):
        self.shard("model-a.safetensors", name="other")
        self.shard("model-b.safetensors", name="weight")
        self.index({"weight": "model-a.safetensors"})
        result = inspect_model(self.root)
        self.assertFalse(result["complete_weight_inventory"])
        self.assertEqual(result["wrong_index_files"], {"weight": "model-a.safetensors"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
