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


replace_once(
    "vllm_metal/v1/model_runner.py",
    """        return (
            not self.is_mla
            and max_head_dim <= PA_WINDOW_MAX_HEAD_SIZE
            and (envs.VLLM_METAL_SPEC_VERIFY_WINDOW or force_hybrid_transactions)
        )
""",
    """        return (
            not self.is_mla
            and max_head_dim <= PA_WINDOW_MAX_HEAD_SIZE
            and (
                force_hybrid_transactions
                or (envs.VLLM_METAL_SPEC_VERIFY_WINDOW and not self.is_hybrid)
            )
        )
""",
    "keep env-only merged verify disabled for hybrid models",
)

replace_once(
    "vllm_metal/v1/model_runner.py",
    """        runtime = self._paged_attention_runtime
        hybrid_gdn_transactions_enabled = (
            runtime is not None and runtime.supports_hybrid_speculative_transactions()
        )
""",
    """        runtime = self._paged_attention_runtime
        supports_transactions = getattr(
            runtime, "supports_hybrid_speculative_transactions", None
        )
        hybrid_gdn_transactions_enabled = bool(
            supports_transactions is not None and supports_transactions()
        )
""",
    "support legacy runtime stubs without transaction capability method",
)

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
