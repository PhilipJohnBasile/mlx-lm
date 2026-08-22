"""Tests-first handoff for selected native-MTP correctness hardening.

This file is intentionally expected to be red at the `e8ceecc` branch point.
It captures the semantics from closed mlx-lm #1740 that AirRunner offered to
carry into the live #990 review path. The implementation may be refactored;
these externally observable behaviors should remain.
"""

from __future__ import annotations

import importlib
import tempfile
import unittest

import mlx.core as mx

from mlx_lm.generate import generate_step, mtp_generate_step
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache


def _make_qwen3_5_mtp_model():
    """Create a tiny hybrid Qwen model with one trained-MTP-shaped layer."""
    module = importlib.import_module("mlx_lm.models.qwen3_5")
    args = module.ModelArgs.from_dict(
        {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 256,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 3,
                "full_attention_interval": 2,
                "tie_word_embeddings": True,
                "rms_norm_eps": 1e-5,
                "head_dim": 32,
                "rope_theta": 1000.0,
                "partial_rotary_factor": 0.5,
                "max_position_embeddings": 128,
                "mtp_num_hidden_layers": 1,
            },
        }
    )
    model = module.Model(args)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


def _make_qwen3_5_moe_mtp_model():
    module = importlib.import_module("mlx_lm.models.qwen3_5_moe")
    args = module.ModelArgs.from_dict(
        {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 64,
                "full_attention_interval": 1,
                "head_dim": 8,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 16,
                "shared_expert_intermediate_size": 16,
                "mtp_num_hidden_layers": 1,
            },
        }
    )
    model = module.Model(args)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model


class TestMTPHardeningHandoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(17)
        cls.model = _make_qwen3_5_mtp_model()

    def _mtp_prompt_state_type(self):
        cache_module = importlib.import_module("mlx_lm.models.cache")
        state_type = getattr(cache_module, "MTPPromptCacheState", None)
        self.assertIsNotNone(
            state_type,
            "native MTP prompt-cache reuse requires serializable boundary state",
        )
        return state_type

    def _assert_transactional_prompt_cache(self, prompt_cache, expected_tokens):
        state_type = self._mtp_prompt_state_type()
        n_main = len(self.model.layers)
        n_mtp = len(self.model.make_mtp_cache())

        self.assertEqual(len(prompt_cache), n_main + n_mtp + 1)
        state = prompt_cache[-1]
        self.assertIsInstance(state, state_type)
        self.assertFalse(state.empty())
        self.assertEqual(state.num_tokens, expected_tokens)
        self.assertEqual(state.last_hidden.shape, (1, 1, 64))

        target_attention = next(
            cache for cache in prompt_cache[:n_main] if cache.is_trimmable()
        )
        mtp_attention = prompt_cache[n_main]
        self.assertEqual(target_attention.offset, expected_tokens)
        self.assertEqual(mtp_attention.offset, expected_tokens - 1)

    def test_missing_dense_mtp_weights_disable_the_head(self):
        model = _make_qwen3_5_mtp_model()
        self.assertTrue(bool(getattr(model, "supports_mtp", False)))

        model.sanitize({})

        self.assertFalse(bool(getattr(model, "supports_mtp", True)))
        self.assertEqual(model.make_mtp_cache(), [])

    def test_missing_moe_mtp_weights_disable_the_head_without_crashing(self):
        model = _make_qwen3_5_moe_mtp_model()
        self.assertTrue(bool(getattr(model, "supports_mtp", False)))

        model.sanitize({})

        self.assertFalse(bool(getattr(model, "supports_mtp", True)))
        self.assertEqual(model.make_mtp_cache(), [])

    def test_stateful_logits_processor_fails_closed(self):
        class StatefulProcessor:
            def __call__(self, tokens, logits):
                return logits

        generator = mtp_generate_step(
            mx.array([0, 1, 2, 3], dtype=mx.uint32),
            self.model,
            max_tokens=1,
            logits_processors=[StatefulProcessor()],
        )
        with self.assertRaisesRegex(ValueError, "stateless logits processors"):
            next(generator)

    def test_accepted_draft_reports_target_distribution(self):
        """Accepted tokens expose target, not draft-head, log probabilities.

        Ordinary greedy generation supplies an independent serial target
        distribution for each emitted position. Native MTP must emit the same
        token sequence, and every accepted-draft row must match that target
        distribution within the small floating-point difference allowed by the
        batched verifier path.
        """
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        n_tokens = 64

        standard_cache = make_prompt_cache(self.model)
        standard = []
        for token, logprobs in generate_step(
            prompt,
            self.model,
            prompt_cache=standard_cache,
        ):
            standard.append((int(token), logprobs))
            if len(standard) == n_tokens:
                break

        mtp = list(
            mtp_generate_step(
                prompt,
                self.model,
                max_tokens=n_tokens,
            )
        )

        self.assertEqual(
            [token for token, _ in standard],
            [int(token) for token, _, _ in mtp],
        )
        accepted_positions = [
            index for index, (_, _, from_draft) in enumerate(mtp) if from_draft
        ]
        self.assertTrue(accepted_positions, "test observed no accepted MTP draft")

        for index in accepted_positions:
            target_logprobs = standard[index][1]
            yielded_logprobs = mtp[index][1]
            self.assertTrue(
                mx.allclose(
                    yielded_logprobs,
                    target_logprobs,
                    rtol=1e-4,
                    atol=1e-4,
                ).item(),
                f"accepted draft at output position {index} did not expose "
                "the target distribution",
            )

    def test_populated_target_cache_without_mtp_boundary_fails_closed(self):
        prompt_cache = make_prompt_cache(self.model)
        self.model(mx.array([[0, 1]], dtype=mx.uint32), cache=prompt_cache)

        generator = mtp_generate_step(
            mx.array([2, 3], dtype=mx.uint32),
            self.model,
            prompt_cache=prompt_cache,
        )
        with self.assertRaisesRegex(
            ValueError,
            "MTP cache entries and boundary state",
        ):
            next(generator)

    def test_prompt_cache_finalizes_at_output_limit(self):
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        prompt_cache = make_prompt_cache(self.model)

        output = list(
            mtp_generate_step(
                prompt,
                self.model,
                max_tokens=1,
                prompt_cache=prompt_cache,
            )
        )

        self.assertEqual(len(output), 1)
        self._assert_transactional_prompt_cache(prompt_cache, len(prompt) + 1)

    def test_prompt_cache_finalizes_when_generator_is_closed(self):
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        prompt_cache = make_prompt_cache(self.model)
        generator = mtp_generate_step(
            prompt,
            self.model,
            max_tokens=10,
            prompt_cache=prompt_cache,
        )

        next(generator)
        generator.close()

        self._assert_transactional_prompt_cache(prompt_cache, len(prompt) + 1)

    def test_multi_turn_cached_generation_matches_uncached_generation(self):
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        suffix = [10, 11]
        prompt_cache = make_prompt_cache(self.model)

        first_turn = [
            int(token)
            for token, _, _ in mtp_generate_step(
                prompt,
                self.model,
                max_tokens=3,
                prompt_cache=prompt_cache,
            )
        ]
        transcript = prompt.tolist() + first_turn

        cached_tokens = [
            int(token)
            for token, _, _ in mtp_generate_step(
                mx.array(suffix, dtype=mx.uint32),
                self.model,
                max_tokens=6,
                prompt_cache=prompt_cache,
            )
        ]
        uncached_tokens = [
            int(token)
            for token, _, _ in mtp_generate_step(
                mx.array(transcript + suffix, dtype=mx.uint32),
                self.model,
                max_tokens=6,
            )
        ]

        self.assertEqual(cached_tokens, uncached_tokens)
        self._assert_transactional_prompt_cache(
            prompt_cache,
            len(transcript) + len(suffix) + len(cached_tokens),
        )

    def test_prompt_cache_boundary_state_round_trips(self):
        state_type = self._mtp_prompt_state_type()
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        prompt_cache = make_prompt_cache(self.model)
        list(
            mtp_generate_step(
                prompt,
                self.model,
                max_tokens=2,
                prompt_cache=prompt_cache,
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/mtp-cache.safetensors"
            save_prompt_cache(path, prompt_cache)
            loaded = load_prompt_cache(path)

        self.assertIsInstance(loaded[-1], state_type)
        self.assertEqual(loaded[-1].num_tokens, prompt_cache[-1].num_tokens)
        self.assertTrue(
            mx.allclose(
                loaded[-1].last_hidden,
                prompt_cache[-1].last_hidden,
            ).item()
        )


if __name__ == "__main__":
    unittest.main()
