from __future__ import annotations

from pathlib import Path


def read(path: str) -> str:
    return Path(path).read_text()


def write(path: str, text: str) -> None:
    Path(path).write_text(text)


def replace_once(path: str, old: str, new: str, label: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    write(path, text.replace(old, new, 1))


linear_path = "vllm_metal/attention/impls/linear.py"
replace_once(
    linear_path,
    "from mlx_lm.models.gated_delta import compute_g\n",
    "from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel\n",
    "import state-producing gated delta kernel",
)
replace_once(
    linear_path,
    """    slot_ids: list[int]
    num_decode_requests: int
""",
    """    slot_ids: list[int]
    num_decode_requests: int
    # Per request: [initial, after_token_0, ..., after_token_K]. Empty
    # entries retain the ordinary one-destination path.
    state_chains: list[list[int]] | None = None
""",
    "extend GDN forward state",
)

old_prepare_return = """        return _GDNForwardState(
            x=x,
            cu_seqlens=cu_seqlens,
            num_requests=num_requests,
            total_tokens=x.shape[1],
            slot_ids=slot_ids,
            num_decode_requests=ctx.num_decode_requests,
        )
"""
new_prepare_return = """        state_chains = None
        if ctx.gdn_group_state_chains is not None:
            ordinal = self._gdn_state_cache.layer_group_ordinal(self._gdn_cache_idx)
            state_chains = ctx.gdn_group_state_chains[ordinal]
            if len(state_chains) != num_requests:
                raise RuntimeError(
                    "GDN wrapper requires one speculative state chain per request"
                )
            for req_idx, chain in enumerate(state_chains):
                if not chain:
                    continue
                request_tokens = cu_seqlens[req_idx + 1] - cu_seqlens[req_idx]
                if len(chain) != request_tokens + 1:
                    raise RuntimeError(
                        f"GDN request {req_idx} has {request_tokens} input tokens "
                        f"but a {len(chain)}-entry state chain"
                    )
                self._gdn_state_cache.require_allocated_slots(chain)

        return _GDNForwardState(
            x=x,
            cu_seqlens=cu_seqlens,
            num_requests=num_requests,
            total_tokens=x.shape[1],
            slot_ids=slot_ids,
            num_decode_requests=ctx.num_decode_requests,
            state_chains=state_chains,
        )
"""
replace_once(
    linear_path,
    old_prepare_return,
    new_prepare_return,
    "extract speculative GDN state chains",
)

replace_once(
    linear_path,
    """    def _run_conv(self, mixed_qkv: mx.array, state: _GDNForwardState) -> mx.array:
        # === Step 2: Conv1d (per-request, needs conv_state) ===
        inner = self._inner
""",
    """    def _run_conv(self, mixed_qkv: mx.array, state: _GDNForwardState) -> mx.array:
        # === Step 2: Conv1d (per-request, needs conv_state) ===
        if state.state_chains is not None and any(state.state_chains):
            return self._run_conv_state_chains(mixed_qkv, state)

        inner = self._inner
""",
    "dispatch conv state-chain path",
)

conv_insert_marker = """    def _split_and_normalize(
"""
conv_method = """    def _run_conv_state_chains(
        self, mixed_qkv: mx.array, state: _GDNForwardState
    ) -> mx.array:
        \"\"\"Produce an observable conv-state checkpoint after every token.

        This is the correctness-first path for speculative verification. It is
        intentionally request/token sequential; ordinary decode and prefill
        remain on their existing fused/lazy kernels.
        \"\"\"
        inner = self._inner
        state_cache = self._gdn_state_cache
        cache_idx = self._gdn_cache_idx
        state_cache.apply_pending_conv_state(cache_idx)
        pool = state_cache.conv_states[cache_idx]
        outputs: list[mx.array] = []

        assert state.state_chains is not None
        for req_idx in range(state.num_requests):
            start = state.cu_seqlens[req_idx]
            end = state.cu_seqlens[req_idx + 1]
            request_qkv = mixed_qkv[:, start:end, :]
            chain = state.state_chains[req_idx]

            if not chain:
                slot = state.slot_ids[req_idx]
                conv_state = pool[slot : slot + 1]
                conv_input = mx.concatenate([conv_state, request_qkv], axis=1)
                new_conv = conv_input[:, -(inner.conv_kernel_size - 1) :]
                pool[slot : slot + 1] = new_conv
                conv_out = nn.silu(inner.conv1d(conv_input))
                outputs.append(conv_out[:, -(end - start) :, :])
                continue

            conv_state = pool[chain[0] : chain[0] + 1]
            request_outputs: list[mx.array] = []
            for token_offset, dst in enumerate(chain[1:]):
                token_qkv = request_qkv[:, token_offset : token_offset + 1, :]
                conv_input = mx.concatenate([conv_state, token_qkv], axis=1)
                new_conv = conv_input[:, -(inner.conv_kernel_size - 1) :]
                pool[dst : dst + 1] = new_conv
                conv_state = new_conv
                conv_out = nn.silu(inner.conv1d(conv_input))
                request_outputs.append(conv_out[:, -1:, :])
            outputs.append(mx.concatenate(request_outputs, axis=1))

        state_cache.store_conv_state(cache_idx, pool)
        return mx.concatenate(outputs, axis=1)

""" + conv_insert_marker
replace_once(
    linear_path,
    conv_insert_marker,
    conv_method,
    "add conv state-chain producer",
)

replace_once(
    linear_path,
    """    def _run_recurrent(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        state: _GDNForwardState,
    ) -> mx.array:
        # === Step 5: Batched recurrent update ===
        if state.num_decode_requests == state.num_requests:
""",
    """    def _run_recurrent(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        state: _GDNForwardState,
    ) -> mx.array:
        # === Step 5: Batched recurrent update ===
        if state.state_chains is not None and any(state.state_chains):
            return self._run_recurrent_state_chains(q, k, v, g, beta, state)

        if state.num_decode_requests == state.num_requests:
""",
    "dispatch recurrent state-chain path",
)

recurrent_insert_marker = """    def _should_try_recurrent_prefill_containing_lazy(
"""
recurrent_method = """    def _run_recurrent_state_chains(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        state: _GDNForwardState,
    ) -> mx.array:
        \"\"\"Produce one recurrent-state snapshot per verification token.\"\"\"
        state_cache = self._gdn_state_cache
        cache_idx = self._gdn_cache_idx
        state_cache.apply_pending_recurrent_state(cache_idx)
        pool = state_cache.recurrent_states[cache_idx]
        outputs: list[mx.array] = []

        assert state.state_chains is not None
        for req_idx in range(state.num_requests):
            start = state.cu_seqlens[req_idx]
            end = state.cu_seqlens[req_idx + 1]
            chain = state.state_chains[req_idx]

            if not chain:
                slot = state.slot_ids[req_idx]
                request_output, new_state = gated_delta_kernel(
                    q[:, start:end],
                    k[:, start:end],
                    v[:, start:end],
                    g[:, start:end],
                    beta[:, start:end],
                    pool[slot : slot + 1],
                )
                pool[slot : slot + 1] = new_state
                outputs.append(request_output.reshape(end - start, *v.shape[2:]))
                continue

            recurrent_state = pool[chain[0] : chain[0] + 1]
            request_outputs: list[mx.array] = []
            for token_offset, dst in enumerate(chain[1:]):
                token = start + token_offset
                token_output, new_state = gated_delta_kernel(
                    q[:, token : token + 1],
                    k[:, token : token + 1],
                    v[:, token : token + 1],
                    g[:, token : token + 1],
                    beta[:, token : token + 1],
                    recurrent_state,
                )
                pool[dst : dst + 1] = new_state
                recurrent_state = new_state
                request_outputs.append(token_output.reshape(1, *v.shape[2:]))
            outputs.append(mx.concatenate(request_outputs, axis=0))

        state_cache.store_recurrent_state(cache_idx, pool)
        return mx.concatenate(outputs, axis=0).astype(state.x.dtype)

""" + recurrent_insert_marker
replace_once(
    linear_path,
    recurrent_insert_marker,
    recurrent_method,
    "add recurrent state-chain producer",
)


test_path = "tests/test_gdn_lazy_wrapper.py"
insert_marker = """class TestGDNPagedAttentionWrapperLazyKernels:
"""
new_test = """class TestGDNSpeculativeStateChains:
    def test_full_wrapper_writes_conv_and_recurrent_snapshot_per_token(self) -> None:
        inner = _TinyGDNInner()
        cache = _make_state_cache(
            max_seqs=4,
            conv_kernel_dim=inner.conv_kernel_size,
            conv_dim=inner.conv_dim,
            num_v_heads=inner.num_v_heads,
            value_head_dim=inner.head_v_dim,
            key_head_dim=inner.head_k_dim,
        )
        wrapper = GDNPagedAttentionWrapper(
            inner, layer_idx=0, cache_idx=0, state_cache=cache
        )
        context = PagedAttentionContext(
            slot_mapping=[],
            cu_seqlens=[0, 3],
            num_decode_requests=1,
            gdn_group_slot_mappings=([2],),
            gdn_group_state_chains=([[0, 0, 1, 2]],),
        )
        tokens = mx.stack(
            [
                mx.full((inner.conv_dim,), 0.01, dtype=mx.float32),
                mx.full((inner.conv_dim,), 0.02, dtype=mx.float32),
                mx.full((inner.conv_dim,), 0.03, dtype=mx.float32),
            ],
            axis=0,
        )[None]

        set_context(context)
        try:
            output = wrapper(tokens)
        finally:
            clear_context()
        mx.eval(output, *cache.updated_state_arrays())

        assert output.shape == (1, 3, inner.head_v_dim)
        conv = np.array(cache.conv_states[0])
        np.testing.assert_allclose(conv[0], 0.01, atol=1e-6)
        np.testing.assert_allclose(conv[1], 0.02, atol=1e-6)
        np.testing.assert_allclose(conv[2], 0.03, atol=1e-6)

        recurrent = np.array(cache.recurrent_states[0])
        assert np.any(recurrent[0] != 0)
        assert np.any(recurrent[1] != recurrent[0])
        assert np.any(recurrent[2] != recurrent[1])


""" + insert_marker
replace_once(
    test_path,
    insert_marker,
    new_test,
    "add GDN state-production regression",
)

print("Applied vllm-metal #610 phase-2 per-token GDN state production patch.")
