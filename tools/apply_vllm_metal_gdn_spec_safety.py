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


# Reserve scheduler-owned shadow blocks only for align-mode state caching.
replace_once(
    "vllm_metal/v1/cache_policy.py",
    """    def _num_speculative_blocks(self) -> int:
        spec = self._runner.vllm_config.speculative_config
        return 0 if spec is None else int(spec.num_speculative_tokens)
""",
    """    def _num_speculative_blocks(self) -> int:
        if self._runner.cache_config.mamba_cache_mode != "align":
            return 0
        spec = self._runner.vllm_config.speculative_config
        return 0 if spec is None else int(spec.num_speculative_tokens)
""",
    "reserve speculative blocks only for align mode",
)

# Expose an explicit runtime capability instead of treating all hybrid runtimes as safe.
replace_once(
    "vllm_metal/attention/runtime/protocol.py",
    """    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None: ...
    def commit_speculative_states(
""",
    """    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None: ...
    def supports_hybrid_speculative_transactions(self) -> bool: ...
    def commit_speculative_states(
""",
    "runtime protocol hybrid transaction capability",
)
replace_once(
    "vllm_metal/attention/runtime/base.py",
    """    def commit_speculative_states(
        self, req_ids: list[str], accepted_token_counts: list[int]
    ) -> None:
""",
    """    def supports_hybrid_speculative_transactions(self) -> bool:
        return False

    def commit_speculative_states(
        self, req_ids: list[str], accepted_token_counts: list[int]
    ) -> None:
""",
    "runtime base hybrid transaction capability",
)
replace_once(
    "vllm_metal/attention/runtime/hybrid.py",
    """    def commit_speculative_states(
        self, req_ids: list[str], accepted_token_counts: list[int]
    ) -> None:
""",
    """    def supports_hybrid_speculative_transactions(self) -> bool:
        return self._mamba_cache_mode == "align" and self._num_speculative_blocks > 0

    def commit_speculative_states(
        self, req_ids: list[str], accepted_token_counts: list[int]
    ) -> None:
""",
    "hybrid runtime transaction capability",
)

# Keep hybrid speculation fail-closed unless the installed runtime owns exact
# align-mode shadow checkpoints.
replace_once(
    "vllm_metal/v1/spec_decode.py",
    """        is_hybrid: bool,
        use_async_scheduling: bool = False,
""",
    """        is_hybrid: bool,
        hybrid_gdn_transactions_enabled: bool = False,
        use_async_scheduling: bool = False,
""",
    "spec policy hybrid transaction argument",
)
replace_once(
    "vllm_metal/v1/spec_decode.py",
    """        del is_hybrid
        spec_tokens = self.active_spec_decode_tokens(scheduler_output)
""",
    """        spec_tokens = self.active_spec_decode_tokens(scheduler_output)
""",
    "restore hybrid policy parameter",
)
replace_once(
    "vllm_metal/v1/spec_decode.py",
    """        if (active_spec_tokens or has_invalid_spec_tokens) and not paged_attention_enabled:
            raise NotImplementedError(
                "Speculative decode verification on Metal requires paged "
                "attention so draft-token rows can share scheduler-assigned "
                "KV slots."
            )

        decode_req_ids = {req_id for req_id, _ in decode_reqs}
""",
    """        if (active_spec_tokens or has_invalid_spec_tokens) and not paged_attention_enabled:
            raise NotImplementedError(
                "Speculative decode verification on Metal requires paged "
                "attention so draft-token rows can share scheduler-assigned "
                "KV slots."
            )
        if (
            (active_spec_tokens or has_invalid_spec_tokens)
            and is_hybrid
            and not hybrid_gdn_transactions_enabled
        ):
            raise NotImplementedError(
                "Speculative decode verification for hybrid GDN models on Metal "
                "requires align-mode transactional state blocks. Enable prefix "
                "caching with mamba_cache_mode='align' or disable speculation."
            )

        decode_req_ids = {req_id for req_id, _ in decode_reqs}
""",
    "fail closed without align-mode transactions",
)

# Feed the installed runtime capability into the preflight policy.
replace_once(
    "vllm_metal/v1/model_runner.py",
    """    def _validate_spec_decode_supported(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        self._spec_decode_controller.validate_supported(
            scheduler_output,
            self._spec_decode_preflight_reqs(scheduler_output),
            paged_attention_enabled=self._paged_attention_runtime is not None,
            is_hybrid=self.is_hybrid,
            use_async_scheduling=self.use_async_scheduling,
            speculative_config=self.vllm_config.speculative_config,
        )
""",
    """    def _validate_spec_decode_supported(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        runtime = self._paged_attention_runtime
        hybrid_gdn_transactions_enabled = (
            runtime is not None
            and runtime.supports_hybrid_speculative_transactions()
        )
        self._spec_decode_controller.validate_supported(
            scheduler_output,
            self._spec_decode_preflight_reqs(scheduler_output),
            paged_attention_enabled=runtime is not None,
            is_hybrid=self.is_hybrid,
            hybrid_gdn_transactions_enabled=hybrid_gdn_transactions_enabled,
            use_async_scheduling=self.use_async_scheduling,
            speculative_config=self.vllm_config.speculative_config,
        )
""",
    "runner hybrid transaction preflight",
)

# Policy tests cover both the enabled and fail-closed arms.
replace_once(
    "tests/test_spec_decode_metadata.py",
    """    def test_hybrid_scheduled_tokens_are_supported(self) -> None:
        SpeculativeDecodeController().validate_supported(
            _scheduler_output(scheduled_spec_decode_tokens={"r0": [1]}),
            [("r0", _request_state())],
            paged_attention_enabled=True,
            is_hybrid=True,
        )
""",
    """    def test_hybrid_scheduled_tokens_are_supported(self) -> None:
        SpeculativeDecodeController().validate_supported(
            _scheduler_output(scheduled_spec_decode_tokens={"r0": [1]}),
            [("r0", _request_state())],
            paged_attention_enabled=True,
            is_hybrid=True,
            hybrid_gdn_transactions_enabled=True,
        )

    def test_hybrid_scheduled_tokens_require_transactional_state(self) -> None:
        with pytest.raises(NotImplementedError, match="transactional state blocks"):
            SpeculativeDecodeController().validate_supported(
                _scheduler_output(scheduled_spec_decode_tokens={"r0": [1]}),
                [("r0", _request_state())],
                paged_attention_enabled=True,
                is_hybrid=True,
            )
""",
    "hybrid transaction policy tests",
)

print("Applied fail-closed hybrid GDN speculative transaction gating.")
