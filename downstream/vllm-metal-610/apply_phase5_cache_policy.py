from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str, label: str) -> None:
    file = Path(path)
    text = file.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    file.write_text(text.replace(old, new, 1))


path = "vllm_metal/v1/cache_policy.py"
replace_once(
    path,
    """from vllm_metal.v1.model_adapter import ModelAdapter
""",
    """from vllm_metal.v1.model_adapter import ModelAdapter
from vllm_metal.v1.qwen_mtp_paged import (
    QwenMTPAttentionSpec,
    QwenMTPPagedState,
)
""",
    "import Qwen MTP cache types",
)
replace_once(
    path,
    """    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
""",
    """    def _qwen_mtp_metadata(
        self,
    ) -> tuple[object, int, int, int, int] | None:
        spec = self._runner.vllm_config.speculative_config
        if spec is None or spec.method != \"mtp\":
            return None
        model = self._runner._forward_model
        if not bool(getattr(model, \"supports_mtp\", False)):
            raise ValueError(
                \"method='mtp' was configured but the loaded Qwen checkpoint \"
                \"does not contain native MTP weights\"
            )
        mtp = getattr(model, \"mtp\", None)
        args = getattr(model, \"args\", None)
        layers = list(getattr(mtp, \"layers\", ()))
        if mtp is None or args is None or not layers:
            raise ValueError(
                \"native Qwen MTP requires model.mtp.layers and model.args\"
            )
        try:
            num_kv_heads = int(args.num_key_value_heads)
            head_dim = int(args.head_dim)
            hidden_size = int(args.hidden_size)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                \"native Qwen MTP model metadata is missing KV/head dimensions\"
            ) from exc
        return model, len(layers), num_kv_heads, head_dim, hidden_size

    def qwen_mtp_aux_bytes_per_block(self) -> int:
        metadata = self._qwen_mtp_metadata()
        if metadata is None:
            return 0
        _, num_layers, num_kv_heads, head_dim, hidden_size = metadata
        block_size = self._runner.cache_config.block_size
        dtype_size = self._require_kv_cache_dtype().size
        mtp_kv = (
            num_layers
            * 2
            * block_size
            * num_kv_heads
            * head_dim
            * dtype_size
        )
        boundary_hidden = block_size * hidden_size * dtype_size
        return mtp_kv + boundary_hidden

    def _build_qwen_mtp_state(self) -> QwenMTPPagedState | None:
        metadata = self._qwen_mtp_metadata()
        if metadata is None:
            return None
        if self._use_turboquant(get_config()):
            raise NotImplementedError(
                \"native Qwen MTP paged KV currently requires dense KV cache; \"
                \"disable TurboQuant KV while using method='mtp'\"
            )
        model, num_layers, num_kv_heads, head_dim, hidden_size = metadata
        return QwenMTPPagedState(
            model=model,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            dtype=self._require_kv_cache_dtype(),
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
""",
    "add Qwen MTP metadata and memory contract",
)
replace_once(
    path,
    """        return specs

    def _build_mha_attention_spec(
""",
    """        metadata = self._qwen_mtp_metadata()
        if metadata is not None:
            if use_turboquant:
                raise NotImplementedError(
                    \"native Qwen MTP paged KV does not support TurboQuant KV\"
                )
            _, num_mtp_layers, mtp_kv_heads, mtp_head_dim, _ = metadata
            mtp_spec = QwenMTPAttentionSpec(
                block_size=block_size,
                # HiddenStateCacheSpec reports one latent tensor. Doubling the
                # logical head slots makes its page bytes equal the physical
                # dense K+V arrays owned by MetalPagedKVCache.
                num_kv_heads=2 * mtp_kv_heads,
                head_size=mtp_head_dim,
                dtype=torch_dtype,
            )
            target_page = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self._runner.num_kv_heads,
                head_size=self._runner.head_dim,
                dtype=torch_dtype,
            ).page_size_bytes
            if mtp_spec.page_size_bytes != target_page:
                raise NotImplementedError(
                    \"native Qwen MTP requires its KV page size to match the \"
                    \"target full-attention page size\"
                )
            # HiddenStateCacheSpec is vLLM's cache-only grouping path. It keeps
            # the one-layer MTP cache distinct without reducing every hybrid
            # target/GDN group to size one.
            for layer_idx in range(num_mtp_layers):
                specs[f\"mtp.layers.{layer_idx}.self_attn\"] = mtp_spec

        return specs

    def _build_mha_attention_spec(
""",
    "append distinct MTP cache specs",
)
replace_once(
    path,
    """        block_size = kv_cache_config.kv_cache_groups[
            group_index
        ].kv_cache_spec.block_size
        # Align mode keys GDN state slabs by scheduler block id.  The engine
""",
    """        block_size = kv_cache_config.kv_cache_groups[
            group_index
        ].kv_cache_spec.block_size

        mtp_group_index: int | None = None
        mtp_block_size: int | None = None
        metadata = self._qwen_mtp_metadata()
        if metadata is not None:
            _, num_mtp_layers, _, _, _ = metadata
            mtp_names = tuple(
                f\"mtp.layers.{layer_idx}.self_attn\"
                for layer_idx in range(num_mtp_layers)
            )
            mtp_group_indices = self._scheduler_group_indices_for_layers(
                kv_cache_config,
                mtp_names,
            )
            if len(mtp_group_indices) != 1:
                raise RuntimeError(
                    \"native Qwen MTP layers must share one scheduler cache group\"
                )
            mtp_group_index = mtp_group_indices[0]
            if mtp_group_index == group_index:
                raise RuntimeError(
                    \"native Qwen MTP cache was incorrectly merged into the \"
                    \"target SDPA scheduler group\"
                )
            # vLLM treats method='mtp' as EAGLE. When no group is explicitly
            # annotated, KVCacheCoordinator conservatively applies the lookahead
            # drop to every group, preserving one shared prefix lineage across
            # target SDPA, GDN, and this cache-only MTP group.
            mtp_group = kv_cache_config.kv_cache_groups[mtp_group_index]
            mtp_block_size = mtp_group.kv_cache_spec.block_size

        # Align mode keys GDN state slabs by scheduler block id.  The engine
""",
    "resolve MTP scheduler group",
)
replace_once(
    path,
    """            layer_group_ordinals=layer_group_ordinals,
            layer_pool_ordinals=layer_pool_ordinals,
        )
""",
    """            layer_group_ordinals=layer_group_ordinals,
            layer_pool_ordinals=layer_pool_ordinals,
            mtp_group_index=mtp_group_index,
            mtp_block_size=mtp_block_size,
        )
""",
    "pass MTP group to hybrid runtime",
)
replace_once(
    path,
    """            mamba_cache_mode=self._runner.cache_config.mamba_cache_mode,
            num_speculative_blocks=self._num_speculative_blocks(),
            turboquant=config.turboquant,
""",
    """            mamba_cache_mode=self._runner.cache_config.mamba_cache_mode,
            num_speculative_blocks=self._num_speculative_blocks(),
            qwen_mtp_state=self._build_qwen_mtp_state(),
            turboquant=config.turboquant,
""",
    "construct Qwen MTP paged state",
)
replace_once(
    path,
    """        per_block_bytes += self._hybrid_align_growth_bytes_per_block()
        usable_metal = int(metal_limit * fraction)
""",
    """        per_block_bytes += self._hybrid_align_growth_bytes_per_block()
        # The MTP cache group shares vLLM's logical block pool, but Metal owns
        # separate physical arrays for its KV and the target-boundary shadow.
        # Reserve those bytes in the worker plan so the scheduler round-trips
        # the same safe block count.
        per_block_bytes += (
            self._worker.model_runner._cache_policy.qwen_mtp_aux_bytes_per_block()
        )
        usable_metal = int(metal_limit * fraction)
""",
    "reserve MTP KV and boundary-hidden memory",
)

print("Applied phase-5 Qwen MTP cache policy.")
