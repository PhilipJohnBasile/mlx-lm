from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(rel: str) -> str:
    return (ROOT / rel).read_text()


def write(rel: str, text: str) -> None:
    (ROOT / rel).write_text(text)


def replace_once(rel: str, old: str, new: str, label: str) -> None:
    text = read(rel)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    write(rel, text.replace(old, new, 1))


runner_rel = "vllm_metal/v1/model_runner.py"
runner_text = read(runner_rel)

# Preserve the old env-only behavior for hybrid models, while forcing merged
# verification windows when hybrid speculation is actually configured.
property_start = runner_text.index("    @property\n    def merge_verify_windows")
property_end = runner_text.index("\n    @property", property_start + 10)
property_text = runner_text[property_start:property_end]
return_start = property_text.index("        return (")
return_close = property_text.index("\n        )", return_start) + len("\n        )")
new_return = """        return (
            not self.is_mla
            and max_head_dim <= PA_WINDOW_MAX_HEAD_SIZE
            and (
                force_hybrid_transactions
                or (envs.VLLM_METAL_SPEC_VERIFY_WINDOW and not self.is_hybrid)
            )
        )"""
property_text = (
    property_text[:return_start]
    + new_return
    + property_text[return_close:]
)
runner_text = (
    runner_text[:property_start]
    + property_text
    + runner_text[property_end:]
)

# Treat legacy runtime stubs without the new capability method as unsupported,
# rather than crashing before the fail-closed policy can run.
method_start = runner_text.index("    def _validate_spec_decode_supported(")
method_end = runner_text.index("\n    def ", method_start + 5)
method_text = runner_text[method_start:method_end]
runtime_start = method_text.index("        runtime = self._paged_attention_runtime")
validate_call = method_text.index(
    "        self._spec_decode_controller.validate_supported(", runtime_start
)
new_runtime = """        runtime = self._paged_attention_runtime
        supports_transactions = getattr(
            runtime, "supports_hybrid_speculative_transactions", None
        )
        hybrid_gdn_transactions_enabled = bool(
            supports_transactions is not None and supports_transactions()
        )
"""
method_text = method_text[:runtime_start] + new_runtime + method_text[validate_call:]
runner_text = runner_text[:method_start] + method_text + runner_text[method_end:]
write(runner_rel, runner_text)

replace_once(
    "tests/test_v1_model_runner_generate.py",
    """    def test_false_for_hybrid_even_when_opted_in(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_SPEC_VERIFY_WINDOW", "1")
        runner = make_stub_runner(model_args={"full_attention_interval": 4})
        assert runner.merge_verify_windows is False
""",
    """    def test_false_for_hybrid_even_when_opted_in(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_SPEC_VERIFY_WINDOW", "1")
        runner = make_stub_runner(model_args={"full_attention_interval": 4})
        assert runner.merge_verify_windows is False

    def test_true_for_hybrid_speculation_without_env_opt_in(self) -> None:
        runner = make_stub_runner(model_args={"full_attention_interval": 4})
        runner.vllm_config.speculative_config = SimpleNamespace(
            num_speculative_tokens=3
        )
        assert runner.merge_verify_windows is True
""",
    "cover automatic merged verify windows for hybrid speculation",
)

print("Applied broad-suite compatibility fixes for hybrid GDN transactions.")
