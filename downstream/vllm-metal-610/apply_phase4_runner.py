from pathlib import Path

path = Path("vllm_metal/v1/model_runner.py")
text = path.read_text()
old = '''from vllm_metal.v1.proposer import (
    Gemma4MTPProposer,
    MetalProposer,
    ProposeContext,
)
'''
new = '''from vllm_metal.v1.proposer import (
    Gemma4MTPProposer,
    MetalProposer,
    ProposeContext,
    QwenNativeMTPProposer,
)
'''
if text.count(old) != 1:
    raise RuntimeError("Qwen proposer import marker mismatch")
text = text.replace(old, new, 1)
old = '''        if Gemma4MTPAssistantSource.is_gemma4_mtp(spec):
            self._drafter = Gemma4MTPProposer(self)
        elif spec.uses_draft_model():
'''
new = '''        if Gemma4MTPAssistantSource.is_gemma4_mtp(spec):
            self._drafter = Gemma4MTPProposer(self)
        elif spec.method == "mtp":
            self._drafter = QwenNativeMTPProposer(self)
        elif spec.uses_draft_model():
'''
if text.count(old) != 1:
    raise RuntimeError("Qwen proposer install marker mismatch")
text = text.replace(old, new, 1)
old = '"(supported: Gemma4 MTP, draft_model, ngram)."\n'
new = '"(supported: Gemma4/Qwen native MTP, draft_model, ngram)."\n'
if text.count(old) != 1:
    raise RuntimeError("proposer support message marker mismatch")
path.write_text(text.replace(old, new, 1))
print("Wired Qwen native-MTP proposer into MetalModelRunner.")
