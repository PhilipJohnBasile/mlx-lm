from pathlib import Path

path = Path("vllm_metal/v1/proposer.py")
text = path.read_text()
old = "from typing import TYPE_CHECKING, Protocol\n"
new = "from typing import TYPE_CHECKING, Any, Protocol\n"
if text.count(old) != 1:
    raise RuntimeError("Qwen proposer typing import marker mismatch")
text = text.replace(old, new, 1)
if "class QwenNativeMTPProposer" in text:
    raise RuntimeError("QwenNativeMTPProposer already exists")
fragment = Path(
    "../lab/downstream/vllm-metal-610/qwen_native_mtp_proposer.pyfrag"
).read_text()
path.write_text(text + fragment)
print("Appended Qwen native-MTP proposer.")
