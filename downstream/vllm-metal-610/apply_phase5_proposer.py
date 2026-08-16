from pathlib import Path

path = Path("vllm_metal/v1/proposer.py")
text = path.read_text()
old = Path(
    "../lab/downstream/vllm-metal-610/qwen_native_mtp_proposer.pyfrag"
).read_text()
new = Path(
    "../lab/downstream/vllm-metal-610/qwen_native_mtp_proposer_phase5.pyfrag"
).read_text()
if text.count(old) != 1:
    raise RuntimeError(
        "phase-5 Qwen proposer replacement expected the exact phase-4 fragment"
    )
path.write_text(text.replace(old, new, 1))
print("Replaced request-local Qwen MTP proposer with scheduler-owned phase 5.")
