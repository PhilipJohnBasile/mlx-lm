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


platform_tests = "tests/test_platform.py"

replace_once(
    platform_tests,
    """            (
                "1",
                SimpleNamespace(
                    block_size=None,
                    kv_cache_dtype_skip_layers=[],
                    enable_prefix_caching=True,
                    mamba_cache_mode="align",
                    mamba_ssm_cache_dtype="float32",
                ),
                SimpleNamespace(
                    use_heterogeneous_vocab=False,
                    num_speculative_tokens=2,
                ),
                "speculative decoding",
            ),
""",
    "",
    "remove obsolete prefix-plus-spec config rejection",
)
replace_once(
    platform_tests,
    '        ids=["prefix_caching_non_paged", "mamba_cache_mode_all", "prefix_and_sd"],\n',
    '        ids=["prefix_caching_non_paged", "mamba_cache_mode_all"],\n',
    "update unsupported hybrid cache-mode case ids",
)

replace_once(
    platform_tests,
    """    def test_check_and_update_config_accepts_hybrid_align_prefix_caching(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        \"\"\"Paged hybrid + prefix caching (align mode) passes config checks.\"\"\"
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        reset_config()
        try:
            vllm_config = self._hybrid_vllm_config(
                SimpleNamespace(
                    block_size=None,
                    kv_cache_dtype_skip_layers=[],
                    enable_prefix_caching=True,
                    mamba_cache_mode="align",
                    mamba_ssm_cache_dtype="float32",
                ),
            )
            MetalPlatform.check_and_update_config(vllm_config)
        finally:
            reset_config()
""",
    """    def test_check_and_update_config_accepts_hybrid_align_prefix_caching(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        \"\"\"Paged hybrid + prefix caching (align mode) passes config checks.\"\"\"
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        reset_config()
        try:
            vllm_config = self._hybrid_vllm_config(
                SimpleNamespace(
                    block_size=None,
                    kv_cache_dtype_skip_layers=[],
                    enable_prefix_caching=True,
                    mamba_cache_mode="align",
                    mamba_ssm_cache_dtype="float32",
                ),
            )
            MetalPlatform.check_and_update_config(vllm_config)
        finally:
            reset_config()

    def test_check_and_update_config_accepts_hybrid_align_with_speculation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        \"\"\"Config accepts prefix caching plus speculation; runtime owns rollback.\"\"\"
        self._patch_stt_resolution(monkeypatch, is_stt=False)
        monkeypatch.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        reset_config()
        try:
            vllm_config = self._hybrid_vllm_config(
                SimpleNamespace(
                    block_size=None,
                    kv_cache_dtype_skip_layers=[],
                    enable_prefix_caching=True,
                    mamba_cache_mode="align",
                    mamba_ssm_cache_dtype="float32",
                ),
                speculative_config=SimpleNamespace(
                    use_heterogeneous_vocab=False,
                    num_speculative_tokens=2,
                ),
            )
            MetalPlatform.check_and_update_config(vllm_config)
            assert vllm_config.cache_config.mamba_cache_mode == "align"
        finally:
            reset_config()
""",
    "add accepted hybrid prefix-plus-spec config coverage",
)

print("Updated platform tests for runtime-owned hybrid speculative transactions.")
