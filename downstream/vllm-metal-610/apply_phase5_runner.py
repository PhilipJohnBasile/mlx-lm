from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str, label: str) -> None:
    file = Path(path)
    text = file.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    file.write_text(text.replace(old, new, 1))


runner_path = "vllm_metal/v1/model_runner.py"
replace_once(
    runner_path,
    """        return (
            envs.VLLM_METAL_SPEC_VERIFY_WINDOW
            and not self.is_mla
            and not self.is_hybrid
            and max_head_dim <= PA_WINDOW_MAX_HEAD_SIZE
        )
""",
    """        runtime = self._paged_attention_runtime
        if (
            self.is_hybrid
            and runtime is not None
            and callable(getattr(runtime, \"supports_hybrid_speculative_decode\", None))
            and runtime.supports_hybrid_speculative_decode()
        ):
            # The phase-2 GDN wrapper requires one segment per request so it
            # can emit a recurrent/conv checkpoint after every verify token.
            return True
        return (
            envs.VLLM_METAL_SPEC_VERIFY_WINDOW
            and not self.is_mla
            and not self.is_hybrid
            and max_head_dim <= PA_WINDOW_MAX_HEAD_SIZE
        )
""",
    "enable merged hybrid verification windows",
)
replace_once(
    runner_path,
    """            speculative_config=self.vllm_config.speculative_config,
        )
""",
    """            speculative_config=self.vllm_config.speculative_config,
            hybrid_speculative_ready=(
                self._paged_attention_runtime is not None
                and callable(
                    getattr(
                        self._paged_attention_runtime,
                        \"supports_hybrid_speculative_decode\",
                        None,
                    )
                )
                and self._paged_attention_runtime.supports_hybrid_speculative_decode()
            ),
        )
""",
    "pass hybrid transactional capability to verifier gate",
)
replace_once(
    runner_path,
    """                    logits = target_output.logits
                    target_hidden_states = target_output.hidden_states
                    del target_output
        finally:
            clear_context()
""",
    """                    logits = target_output.logits
                    target_hidden_states = target_output.hidden_states
                    del target_output

            if (
                ctx is not None
                and runtime is not None
                and target_hidden_states is not None
                and bool(getattr(runtime, \"qwen_mtp_ready\", False))
            ):
                runtime.store_qwen_mtp_target_hidden(ctx, target_hidden_states)
        finally:
            clear_context()
""",
    "store target boundary hidden states",
)

spec_path = "vllm_metal/v1/spec_decode.py"
replace_once(
    spec_path,
    """        speculative_config: SpeculativeConfig | None = None,
    ) -> None:
""",
    """        speculative_config: SpeculativeConfig | None = None,
        hybrid_speculative_ready: bool = False,
    ) -> None:
""",
    "extend speculative validation capability",
)
replace_once(
    spec_path,
    """        if (active_spec_tokens or has_invalid_spec_tokens) and is_hybrid:
            raise NotImplementedError(
                \"Speculative decode verification is not supported for hybrid \"
                \"GDN models on Metal yet.\"
            )
""",
    """        if (
            (active_spec_tokens or has_invalid_spec_tokens)
            and is_hybrid
            and not hybrid_speculative_ready
        ):
            raise NotImplementedError(
                \"Speculative decode verification for hybrid GDN models requires \"
                \"the transactional paged Qwen MTP runtime.\"
            )
""",
    "open hybrid verification only for the transactional runtime",
)

platform_path = "vllm_metal/platform.py"
replace_once(
    platform_path,
    """            if (
                cache_config.enable_prefix_caching
                and vllm_config.speculative_config is not None
            ):
                raise NotImplementedError(
                    \"Prefix caching for hybrid GDN models on Metal does not \"
                    \"support speculative decoding yet: draft-state rollback \"
                    \"across mamba state blocks (num_speculative_blocks) is \"
                    \"not implemented. Disable one of the two.\"
                )
""",
    """            if (
                cache_config.enable_prefix_caching
                and vllm_config.speculative_config is not None
            ):
                text_config = model_config.hf_config.get_text_config()
                native_qwen_mtp = (
                    vllm_config.speculative_config.method == \"mtp\"
                    and int(
                        vllm_config.speculative_config.num_speculative_tokens or 0
                    )
                    == 1
                    and int(getattr(text_config, \"mtp_num_hidden_layers\", 0)) > 0
                    and cache_config.mamba_cache_mode == \"align\"
                    and config.use_paged_attention
                )
                if not native_qwen_mtp:
                    raise NotImplementedError(
                        \"Hybrid prefix caching plus speculative decoding on \"
                        \"Metal is supported only for one-token native Qwen MTP \"
                        \"with paged attention and mamba_cache_mode='align'.\"
                    )
""",
    "open platform gate for native Qwen MTP transaction",
)

print("Applied phase-5 runner, verifier, and platform gates.")
