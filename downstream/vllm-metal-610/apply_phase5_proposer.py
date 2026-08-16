from pathlib import Path

path = Path("vllm_metal/v1/proposer.py")
text = path.read_text()
marker = "\n\n@dataclass(slots=True)\nclass _QwenMTPRequestState:"
start = text.find(marker)
if start < 0:
    raise RuntimeError("phase-5 Qwen proposer replacement marker was not found")
new = Path(
    "../lab/downstream/vllm-metal-610/qwen_native_mtp_proposer_phase5.pyfrag"
).read_text()
# Phase 4 appends the Qwen proposer at the end of proposer.py. Replacing from
# the stable class marker to EOF survives ruff formatting of the phase-4 commit.
path.write_text(text[:start] + new)
print("Replaced request-local Qwen MTP proposer with scheduler-owned phase 5.")
