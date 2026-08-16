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


# Keep existing direct _GDNForwardState construction source-compatible.
replace_once(
    "vllm_metal/attention/impls/linear.py",
    """    slot_ids: list[int]
    token_slot_ids: tuple[tuple[int, ...] | None, ...] | None
    num_decode_requests: int
""",
    """    slot_ids: list[int]
    num_decode_requests: int
    token_slot_ids: tuple[tuple[int, ...] | None, ...] | None = None
""",
    "make speculative checkpoint mapping optional",
)

append = r'''

class TestGDNSpeculativeCheckpointing:
    @staticmethod
    def _wrapper(*, max_seqs: int = 4) -> tuple[
        _TinyGDNInner, GDNPagedStateCache, GDNPagedAttentionWrapper
    ]:
        inner = _TinyGDNInner()
        cache = _make_state_cache(
            max_seqs=max_seqs,
            conv_kernel_dim=inner.conv_kernel_size,
            conv_dim=inner.conv_dim,
            num_v_heads=inner.num_v_heads,
            value_head_dim=inner.head_v_dim,
            key_head_dim=inner.head_k_dim,
        )
        wrapper = GDNPagedAttentionWrapper(
            inner, layer_idx=0, cache_idx=0, state_cache=cache
        )
        return inner, cache, wrapper

    @staticmethod
    def _state(inner: _TinyGDNInner) -> attention_linear._GDNForwardState:
        return attention_linear._GDNForwardState(
            x=mx.zeros((1, 3, inner.conv_dim), dtype=mx.float32),
            cu_seqlens=[0, 3],
            num_requests=1,
            total_tokens=3,
            slot_ids=[0],
            num_decode_requests=1,
            token_slot_ids=((0, 1, 2),),
        )

    def test_conv_retains_one_checkpoint_per_verify_token(self) -> None:
        inner, cache, wrapper = self._wrapper()
        state = self._state(inner)
        mixed_qkv = mx.stack(
            [
                mx.full((inner.conv_dim,), float(token + 1), dtype=mx.float32)
                for token in range(3)
            ],
            axis=0,
        )[None]

        result = wrapper._run_conv(mixed_qkv, state)
        cache.apply_pending_conv_state(0)
        mx.eval(result, cache.conv_states[0])

        assert result.shape == (1, 3, inner.conv_dim)
        for slot, expected in enumerate((1.0, 2.0, 3.0)):
            np.testing.assert_allclose(
                np.asarray(cache.conv_states[0][slot]),
                expected,
                rtol=0,
                atol=0,
            )

    def test_recurrent_retains_exact_state_after_each_verify_token(self) -> None:
        inner, cache, wrapper = self._wrapper()
        state = self._state(inner)
        q = (
            mx.arange(3 * inner.head_k_dim, dtype=mx.float32)
            .reshape(1, 3, 1, inner.head_k_dim)
            / 97.0
        )
        k = mx.flip(q, axis=-1) * 0.5
        v = (
            mx.arange(3 * inner.head_v_dim, dtype=mx.float32)
            .reshape(1, 3, 1, inner.head_v_dim)
            / 11.0
        )
        g = mx.full((1, 3, 1), -0.25, dtype=mx.float32)
        beta = mx.full((1, 3, 1), 0.5, dtype=mx.float32)

        reference_state = mx.zeros_like(cache.recurrent_states[0][0:1])
        expected_states: list[mx.array] = []
        expected_outputs: list[mx.array] = []
        for token in range(3):
            y, reference_state = wrapper._gated_delta_checkpoint_step(
                q[:, token : token + 1],
                k[:, token : token + 1],
                v[:, token : token + 1],
                g[:, token : token + 1],
                beta[:, token : token + 1],
                reference_state,
            )
            expected_outputs.append(y)
            expected_states.append(reference_state)

        result = wrapper._run_recurrent(q, k, v, g, beta, state)
        cache.apply_pending_recurrent_state(0)
        expected_result = mx.concatenate(expected_outputs, axis=1).reshape(
            3, inner.num_v_heads, inner.head_v_dim
        )
        mx.eval(result, expected_result, cache.recurrent_states[0], *expected_states)

        np.testing.assert_allclose(
            np.asarray(result),
            np.asarray(expected_result),
            rtol=1e-5,
            atol=1e-6,
        )
        for slot, expected_state in enumerate(expected_states):
            np.testing.assert_allclose(
                np.asarray(cache.recurrent_states[0][slot]),
                np.asarray(expected_state[0]),
                rtol=1e-5,
                atol=1e-6,
            )
'''

path = ROOT / "tests/test_gdn_lazy_wrapper.py"
path.write_text(path.read_text() + append)

print("Added direct speculative conv and recurrent checkpoint coverage.")
