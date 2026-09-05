"""Pinned source checks do not require MLX or network access."""
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("emitter", ROOT / "tools/emit_metal.py")
emitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(emitter)


class HeaderTests(unittest.TestCase):
    def test_bundled_headers_are_used_offline(self):
        with patch.object(emitter.urllib.request, "urlopen", side_effect=AssertionError("network used")):
            text, count = emitter.emit()
        self.assertEqual(count, 576)
        self.assertEqual(
            hashlib.sha256(text.encode()).hexdigest(),
            "d9ebdb5dd7cd2737694186da028950e515ad1b5cb1d470356343833e98101a87",
        )

    def test_tampered_header_is_rejected_without_refetch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bf16.h").write_text("wrong source")
            with patch.object(emitter.urllib.request, "urlopen", side_effect=AssertionError("network used")):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    emitter.read_header(root, "bf16.h")


if __name__ == "__main__":
    unittest.main(verbosity=2)
