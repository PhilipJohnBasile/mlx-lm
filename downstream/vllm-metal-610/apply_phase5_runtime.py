from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str, label: str) -> None:
    file = Path(path)
    text = file.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    file.write_text(text.replace(old, new, 1))


fragment = Path(
    "../lab/downstream/vllm-metal-610/qwen_mtp_paged.pyfrag"
).read_text()
old_imports = """from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
"""
new_imports = """from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import HiddenStateCacheSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
"""
if fragment.count(old_imports) != 1:
    raise RuntimeError("Qwen MTP cache-spec import marker mismatch")
fragment = fragment.replace(old_imports, new_imports, 1)
fragment = fragment.replace(
    "class QwenMTPAttentionSpec(MLAAttentionSpec):",
    "class QwenMTPAttentionSpec(HiddenStateCacheSpec):",
    1,
)
old_spec_body = """    Subclassing ``MLAAttentionSpec`` prevents vLLM's generic
    ``FullAttentionSpec.merge`` from folding this layer into the target SDPA
    group.  The physical format remains ordinary K+V and is reported exactly
    by the override below.
    \"\"\"

    @property
    def real_page_size_bytes(self) -> int:
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * get_dtype_size(self.dtype)
        )


"""
new_spec_body = """    The cache-only marker makes vLLM pull this layer out before hybrid
    grouping, so a single MTP layer does not collapse every target/GDN group to
    size one. ``num_kv_heads`` is reported as combined K+V head slots; the
    inherited latent-page byte formula therefore equals the physical dense
    K+V page owned by :class:`MetalPagedKVCache`.
    \"\"\"


"""
if fragment.count(old_spec_body) != 1:
    raise RuntimeError("Qwen MTP cache-spec body marker mismatch")
fragment = fragment.replace(old_spec_body, new_spec_body, 1)
registration_start = fragment.index(
    "# Register as its own uniform type while reusing the standard"
)
boundary_start = fragment.index("class QwenMTPBoundaryHiddenCache:")
registration = """# Initialize the built-in registry first, then register this cache-only
# subtype as its own uniform base. Cache policy inserts it first in the spec
# map, so vLLM's early uniformity probe cannot absorb it into target attention.
KVCacheSpecRegistry._ensure_registered()
KVCacheSpecRegistry.register(
    QwenMTPAttentionSpec,
    FullAttentionManager,
    uniform_type_base_spec=QwenMTPAttentionSpec,
)


"""
fragment = fragment[:registration_start] + registration + fragment[boundary_start:]
Path("vllm_metal/v1/qwen_mtp_paged.py").write_text(fragment)

path = "vllm_metal/attention/runtime/hybrid.py"
replace_once(
    path,
    """from vllm_metal.attention.state import AlignGDNStateManager, HybridGDNStateManager
""",
    """from vllm_metal.attention.state import AlignGDNStateManager, HybridGDNStateManager
from vllm_metal.v1.qwen_mtp_paged import QwenMTPPagedState
""",
    "import Qwen MTP paged state",
)
replace_once(
    path,
    """        # One scheduler-owned GDN snapshot block per possible draft token.
        num_speculative_blocks: int = 0,
        # TurboQuant (SDPA layers only)
""",
    """        # One scheduler-owned GDN snapshot block per possible draft token.
        num_speculative_blocks: int = 0,
        # Optional native-Qwen MTP cache transaction.
        qwen_mtp_state: QwenMTPPagedState | None = None,
        # TurboQuant (SDPA layers only)
""",
    "extend hybrid constructor with Qwen MTP state",
)
replace_once(
    path,
    """        self._num_speculative_blocks = num_speculative_blocks

        # SDPA params
""",
    """        self._num_speculative_blocks = num_speculative_blocks
        self._qwen_mtp_state = qwen_mtp_state

        # SDPA params
""",
    "store Qwen MTP state",
)
replace_once(
    path,
    """        self._gdn_state_manager = (
            AlignGDNStateManager(
                self._state_cache,
                self._block_size,
                self._num_speculative_blocks,
            )
            if align
            else HybridGDNStateManager(self._state_cache)
        )

        logger.info(
""",
    """        self._gdn_state_manager = (
            AlignGDNStateManager(
                self._state_cache,
                self._block_size,
                self._num_speculative_blocks,
            )
            if align
            else HybridGDNStateManager(self._state_cache)
        )
        if self._qwen_mtp_state is not None:
            self._qwen_mtp_state.initialize(
                num_blocks=num_blocks,
                block_size=self._block_size,
            )

        logger.info(
""",
    "initialize Qwen MTP caches",
)
replace_once(
    path,
    """        state_group_indices: tuple[int, ...] = (),
        layer_group_ordinals: list[int] | None = None,
        layer_pool_ordinals: list[int] | None = None,
    ) -> None:
""",
    """        state_group_indices: tuple[int, ...] = (),
        layer_group_ordinals: list[int] | None = None,
        layer_pool_ordinals: list[int] | None = None,
        mtp_group_index: int | None = None,
        mtp_block_size: int | None = None,
    ) -> None:
""",
    "extend scheduler-group adoption",
)
replace_once(
    path,
    """        self._scheduler_group_indices = (group_index,)
        self._group_block_sizes = (block_size,)
        self._state_group_indices = tuple(state_group_indices)
""",
    """        if self._qwen_mtp_state is not None:
            if mtp_group_index is None or mtp_block_size is None:
                raise RuntimeError(
                    "native Qwen MTP runtime is missing its scheduler cache group"
                )
            self._qwen_mtp_state.configure_groups(
                target_group_index=group_index,
                mtp_group_index=mtp_group_index,
                target_block_size=block_size,
                mtp_block_size=mtp_block_size,
            )
            self._scheduler_group_indices = (group_index, mtp_group_index)
            self._group_block_sizes = (block_size, mtp_block_size)
        else:
            self._scheduler_group_indices = (group_index,)
            self._group_block_sizes = (block_size,)
        self._state_group_indices = tuple(state_group_indices)
""",
    "publish target and MTP scheduler groups",
)
replace_once(
    path,
    """        self.kv_cache.copy_blocks(block_copies)
        if self._mamba_cache_mode == \"align\":
            self.state_cache.copy_blocks(block_copies)
""",
    """        self.kv_cache.copy_blocks(block_copies)
        if self._mamba_cache_mode == \"align\":
            self.state_cache.copy_blocks(block_copies)
        if self._qwen_mtp_state is not None:
            self._qwen_mtp_state.copy_blocks(block_copies)
""",
    "copy MTP and boundary blocks",
)
replace_once(
    path,
    """    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        self.gdn_state_manager.extend_forward_eval_outputs(outputs)

    def release_requests(self, req_ids: set[str]) -> None:
""",
    """    @property
    def qwen_mtp_ready(self) -> bool:
        return self._qwen_mtp_state is not None and self._qwen_mtp_state.ready

    def supports_hybrid_speculative_decode(self) -> bool:
        return self._mamba_cache_mode == \"align\" and self.qwen_mtp_ready

    def store_qwen_mtp_target_hidden(
        self,
        ctx: PagedAttentionContext,
        hidden_states: mx.array,
    ) -> None:
        if self._qwen_mtp_state is None:
            return
        self._qwen_mtp_state.store_target_hidden(ctx, hidden_states)

    def qwen_mtp_boundary_hidden(
        self,
        block_ids_by_group: Sequence[Sequence[int]],
        token_position: int,
    ) -> mx.array:
        if self._qwen_mtp_state is None:
            raise RuntimeError(\"Qwen MTP boundary state is not installed\")
        return self._qwen_mtp_state.read_boundary_hidden(
            block_ids_by_group,
            token_position,
        )

    def qwen_mtp_run_pairs(
        self,
        *,
        hidden_rows: mx.array,
        next_token_ids: Sequence[int],
        block_ids_by_group: Sequence[Sequence[int]],
        start_pos: int,
    ) -> int:
        if self._qwen_mtp_state is None:
            raise RuntimeError(\"Qwen MTP paged state is not installed\")
        return self._qwen_mtp_state.run_pairs(
            hidden_rows=hidden_rows,
            next_token_ids=next_token_ids,
            block_ids_by_group=block_ids_by_group,
            start_pos=start_pos,
        )

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        self.gdn_state_manager.extend_forward_eval_outputs(outputs)
        if self._qwen_mtp_state is not None:
            self._qwen_mtp_state.extend_forward_eval_outputs(outputs)

    def release_requests(self, req_ids: set[str]) -> None:
""",
    "expose Qwen MTP transaction API",
)

print("Applied phase-5 hybrid paged-MTP runtime.")
