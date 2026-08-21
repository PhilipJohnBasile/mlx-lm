import importlib
import itertools
import tempfile
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.generate import generate, generate_step, mtp_generate_step, stream_generate
from mlx_lm.models.cache import (
    MTPPromptCacheState,
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
)
from mlx_lm.sample_utils import make_logits_processors
from mlx_lm.tokenizer_utils import TokenizerWrapper

generate_module = importlib.import_module("mlx_lm.generate")


def _make_qwen3_5_mtp_model():
    """Create a tiny Qwen3.5 model with an MTP head for testing."""
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
    return module.Model(args)


class _TestTokenizer:
    """Small tokenizer surface needed to exercise stream_generate dispatch."""

    eos_token_id = -1
    vocab_size = 256
    chat_template = None
    bos_token = None

    def get_vocab(self):
        return {}

    def encode(self, _text, add_special_tokens=False):
        return [1, 2, 3]

    def decode(self, tokens):
        return "".join(str(token) for token in tokens)


class _NoopMTPTestCache:
    @property
    def state(self):
        return mx.array(0)

    def is_trimmable(self):
        return False


class _ForcedRejectionMTPModel:
    """Minimal native-MTP model with a deterministic draft rejection.

    The target distribution is p=[0,.45,.35,.20] and the draft distribution
    is q=[.35,.45,0,.20].  A controlled sampler chooses target token 1 and
    draft token 0.  Token 0 is rejected and p-q has support only at token 2.
    """

    supports_mtp = True
    layers = [None]

    def __init__(self):
        self.verify_calls = 0

    def make_cache(self):
        return [_NoopMTPTestCache()]

    def make_mtp_cache(self):
        return [_NoopMTPTestCache()]

    def __call__(self, tokens, cache, return_hidden=False, n_confirmed=0):
        del cache, n_confirmed
        steps = tokens.shape[1]
        if steps == 2:
            self.verify_calls += 1
        logits = mx.log(mx.array([0.0, 0.45, 0.35, 0.20]))
        logits = mx.broadcast_to(logits, (1, steps, 4))
        hidden = mx.zeros((1, steps, 1))
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(self, hidden, next_ids, cache):
        del hidden, cache
        logits = mx.log(mx.array([0.35, 0.45, 0.0, 0.20]))
        return mx.broadcast_to(logits, (1, next_ids.shape[1], 4))


class TestMTP(unittest.TestCase):
    """Tests for native MTP (Multi-Token Prediction) speculative decoding.

    Uses a tiny synthetic Qwen3.5 model (4 layers, hidden=64, vocab=256)
    with mtp_num_hidden_layers=1 and full_attention_interval=2, giving a
    mix of GatedDeltaNet (SSM) and full-attention layers.

    Not tested here:
    - Server integration (--mtp flag, is_batchable)
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _make_qwen3_5_mtp_model()
        cls.tokenizer = TokenizerWrapper(_TestTokenizer())

    def test_mtp_module_exists(self):
        """Model with mtp_num_hidden_layers=1 should have MTP head."""
        self.assertTrue(hasattr(self.model, "mtp_forward"))
        self.assertTrue(hasattr(self.model, "make_mtp_cache"))
        lm = self.model.language_model
        self.assertTrue(hasattr(lm, "mtp"))
        self.assertEqual(len(lm.mtp.layers), 1)

    def test_make_mtp_cache(self):
        """make_mtp_cache should return one KVCache per MTP layer."""
        mtp_cache = self.model.make_mtp_cache()
        self.assertEqual(len(mtp_cache), 1)
        self.assertTrue(mtp_cache[0].is_trimmable())

    def test_missing_mtp_weights_disable_head(self):
        model = _make_qwen3_5_mtp_model()
        self.assertTrue(model.supports_mtp)
        model.sanitize({})
        self.assertFalse(model.supports_mtp)
        self.assertEqual(model.make_mtp_cache(), [])

    def test_missing_mtp_moe_weights_do_not_crash(self):
        model = _make_qwen3_5_moe_mtp_model()
        self.assertTrue(model.supports_mtp)
        model.sanitize({})
        self.assertFalse(model.supports_mtp)

    def test_mtp_rejects_input_embeddings(self):
        prompt = mx.array([0, 1], dtype=mx.uint32)
        embeddings = mx.zeros((2, 64))
        with self.assertRaisesRegex(ValueError, "input_embeddings"):
            next(mtp_generate_step(prompt, self.model, input_embeddings=embeddings))

    def test_return_hidden(self):
        """return_hidden=True should return (logits, hidden) with correct shapes."""
        inputs = mx.array([[0, 1, 2]])
        cache = make_prompt_cache(self.model)
        out, hidden = self.model(inputs, cache=cache, return_hidden=True)
        self.assertEqual(out.shape, (1, 3, 256))
        self.assertEqual(hidden.shape, (1, 3, 64))

    def test_mtp_forward_shape(self):
        """mtp_forward should produce logits of shape (B, 1, vocab)."""
        hidden = mx.random.normal((1, 1, 64))
        next_ids = mx.array([[5]])
        mtp_cache = self.model.make_mtp_cache()
        logits = self.model.mtp_forward(hidden, next_ids, mtp_cache)
        self.assertEqual(logits.shape, (1, 1, 256))

    def test_hidden_is_pre_norm(self):
        """Hidden states returned with return_hidden should be pre-norm.

        This verifies the fix for double normalization: the backbone returns
        pre-norm hidden states, and the final norm is applied only before
        lm_head (not before the MTP head).
        """
        lm = self.model.language_model
        inputs = mx.array([[0, 1, 2]])
        cache = make_prompt_cache(self.model)

        _, hidden = lm(inputs, cache=cache, return_hidden=True)

        # Apply the final norm manually and check it changes the values.
        normed = lm.model.norm(hidden)
        self.assertFalse(mx.allclose(hidden, normed, atol=1e-5).item())

    def test_mtp_generate_identity(self):
        """mtp_generate_step should produce the same greedy tokens as generate_step.

        This is the most important correctness test: it proves that the
        draft/verify loop, SSM state rollback on rejection, and MTP cache
        management are all correct.  Any bug in these would cause the MTP
        path to diverge from standard generation.
        """
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        n_tokens = 10

        # Standard generation, greedy (default sampler is argmax).
        std_cache = make_prompt_cache(self.model)
        std_tokens = []
        for i, (tok, _) in enumerate(
            generate_step(prompt, self.model, prompt_cache=std_cache)
        ):
            std_tokens.append(int(tok))
            if i + 1 >= n_tokens:
                break

        # MTP generation, greedy (sampler=None uses exact-match acceptance).
        mtp_tokens = []
        for tok, _, _ in mtp_generate_step(prompt, self.model, max_tokens=n_tokens):
            mtp_tokens.append(int(tok))
            if len(mtp_tokens) >= n_tokens:
                break

        self.assertEqual(
            std_tokens,
            mtp_tokens,
            f"Token mismatch: std={std_tokens}, mtp={mtp_tokens}",
        )

    def test_mtp_probabilistic_acceptance_completes(self):
        """mtp_generate_step should complete without errors with a stochastic sampler.

        Exercises the probabilistic acceptance path: min(1, p_target / p_draft),
        both with bare temp (no filters) and with top_k applied.
        """
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        n_tokens = 10

        for kwargs in [
            {"temp": 0.7},
            {"temp": 0.7, "top_k": 10},
        ]:
            tokens = []
            for tok, _, _ in mtp_generate_step(
                prompt, self.model, max_tokens=n_tokens, **kwargs
            ):
                tokens.append(int(tok))
                if len(tokens) >= n_tokens:
                    break
            self.assertEqual(len(tokens), n_tokens, f"kwargs={kwargs}")

    def test_mtp_infinite_max_tokens(self):
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        tokens = list(
            itertools.islice(
                mtp_generate_step(prompt, self.model, max_tokens=-1),
                3,
            )
        )
        self.assertEqual(len(tokens), 3)

    def test_mtp_extends_fresh_prompt_cache(self):
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        prompt_cache = make_prompt_cache(self.model)
        n_main = len(prompt_cache)
        generator = mtp_generate_step(
            prompt,
            self.model,
            max_tokens=1,
            prompt_cache=prompt_cache,
        )
        next(generator)
        generator.close()
        self.assertEqual(len(prompt_cache), n_main + 2)

    def test_mtp_rejects_unaligned_populated_prompt_cache(self):
        prompt_cache = make_prompt_cache(self.model)
        self.model(mx.array([[0, 1]], dtype=mx.uint32), cache=prompt_cache)
        with self.assertRaisesRegex(ValueError, "MTP cache entries and boundary state"):
            next(
                mtp_generate_step(
                    mx.array([2, 3], dtype=mx.uint32),
                    self.model,
                    prompt_cache=prompt_cache,
                )
            )

    def test_mtp_rejects_populated_cache_with_empty_boundary(self):
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        prompt_cache = make_prompt_cache(self.model)
        list(
            mtp_generate_step(
                prompt,
                self.model,
                max_tokens=1,
                prompt_cache=prompt_cache,
            )
        )
        boundary = prompt_cache[-1]
        self.assertIsInstance(boundary, MTPPromptCacheState)
        boundary.last_hidden = None
        boundary.num_tokens = 0

        with self.assertRaisesRegex(ValueError, "non-empty boundary metadata"):
            next(
                mtp_generate_step(
                    mx.array([4], dtype=mx.uint32),
                    self.model,
                    prompt_cache=prompt_cache,
                )
            )

    def test_mtp_accepts_all_empty_cache_with_empty_boundary(self):
        prompt_cache = make_prompt_cache(self.model)
        prompt_cache.extend(self.model.make_mtp_cache())
        prompt_cache.append(MTPPromptCacheState())

        result = list(
            mtp_generate_step(
                mx.array([0, 1, 2, 3], dtype=mx.uint32),
                self.model,
                max_tokens=1,
                prompt_cache=prompt_cache,
            )
        )
        self.assertEqual(len(result), 1)

    def test_mtp_invalidates_cache_after_processor_failure(self):
        prompt_cache = make_prompt_cache(self.model)
        list(
            mtp_generate_step(
                mx.array([0, 1, 2, 3], dtype=mx.uint32),
                self.model,
                max_tokens=1,
                prompt_cache=prompt_cache,
            )
        )

        def failing_processor(_tokens, _logits):
            raise RuntimeError("injected processor failure")

        failing_processor.is_stateless = True
        with self.assertRaisesRegex(RuntimeError, "injected processor failure"):
            next(
                mtp_generate_step(
                    mx.array([4], dtype=mx.uint32),
                    self.model,
                    prompt_cache=prompt_cache,
                    logits_processors=[failing_processor],
                )
            )

        boundary = prompt_cache[-1]
        self.assertTrue(boundary.empty())
        with self.assertRaisesRegex(ValueError, "non-empty boundary metadata"):
            next(
                mtp_generate_step(
                    mx.array([4], dtype=mx.uint32),
                    self.model,
                    prompt_cache=prompt_cache,
                )
            )

    def test_mtp_rejects_stateful_logits_processor(self):
        class StatefulProcessor:
            def __call__(self, tokens, logits):
                return logits

        with self.assertRaisesRegex(ValueError, "stateless logits processors"):
            next(
                mtp_generate_step(
                    mx.array([0, 1, 2, 3], dtype=mx.uint32),
                    self.model,
                    logits_processors=[StatefulProcessor()],
                )
            )

    def test_mtp_rejects_unmarked_closure_and_partial(self):
        calls = []

        def stateful_processor(_tokens, logits):
            calls.append(1)
            return logits

        for processor in (stateful_processor, partial(stateful_processor)):
            with self.assertRaisesRegex(ValueError, "stateless logits processors"):
                next(
                    mtp_generate_step(
                        mx.array([0, 1, 2, 3], dtype=mx.uint32),
                        self.model,
                        logits_processors=[processor],
                    )
                )
        self.assertEqual(calls, [])

    def test_mtp_accepts_library_logits_processors(self):
        processors = make_logits_processors(
            logit_bias={0: 1.0},
            repetition_penalty=1.1,
            presence_penalty=0.1,
            frequency_penalty=0.1,
        )
        self.assertTrue(all(processor.is_stateless for processor in processors))
        result = list(
            mtp_generate_step(
                mx.array([0, 1, 2, 3], dtype=mx.uint32),
                self.model,
                max_tokens=1,
                logits_processors=processors,
            )
        )
        self.assertEqual(len(result), 1)

    def test_mtp_generate_identity_with_logits_processor(self):
        """mtp_generate_step must produce the same greedy tokens as generate_step
        when a context-sensitive stateless processor is applied.

        A processor that boosts (tokens[-1] + 1) % vocab biases sampling based on
        the last token.  Incorrect prev_tokens management in the verify pass would
        cause the bonus token or the token after a rejection to be sampled with
        the wrong bias, producing a sequence that diverges from serial generation.
        """
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        n_tokens = 10

        def context_processor(tokens, logits):
            if tokens is None or tokens.size == 0:
                return logits
            target = (int(tokens[-1].item()) + 1) % logits.shape[-1]
            # 1D boost broadcasts correctly for both (vocab,) and (1, vocab) logits.
            boost = mx.zeros(logits.shape[-1])
            return logits + boost.at[target].add(10.0)

        context_processor.is_stateless = True

        std_cache = make_prompt_cache(self.model)
        std_tokens = []
        for i, (tok, _) in enumerate(
            generate_step(
                prompt,
                self.model,
                prompt_cache=std_cache,
                logits_processors=[context_processor],
            )
        ):
            std_tokens.append(int(tok))
            if i + 1 >= n_tokens:
                break

        mtp_tokens = []
        for tok, _, _ in mtp_generate_step(
            prompt,
            self.model,
            max_tokens=n_tokens,
            logits_processors=[context_processor],
        ):
            mtp_tokens.append(int(tok))
            if len(mtp_tokens) >= n_tokens:
                break

        self.assertEqual(std_tokens, mtp_tokens)

    def test_mtp_processor_prev_tokens_correct_at_draft_step(self):
        """The processor must see the just-sampled backbone token as tokens[-1]
        when the MTP head runs, not the preceding input token.

        A forcing processor logs tokens[-1] on every call.  When tokens[-1] equals
        the last prompt token (3) it applies a large boost to token 4, guaranteeing
        the backbone samples token 4 regardless of model weights.  The second
        processor call comes from the MTP head: if the token context is correct it
        sees 4; if stale it sees 3 again.
        """
        # Last prompt token is 3; the forcing processor boosts token 4 when it
        # sees 3, so the backbone deterministically samples T0 = 4 regardless of weights.
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)

        logged: list[int] = []

        def forcing_processor(tokens, logits):
            if tokens is not None and tokens.size > 0:
                last = int(tokens[-1].item())
                logged.append(last)
                if last == 3:
                    boost = mx.zeros(logits.shape[-1])
                    return logits + boost.at[4].add(1000.0)
            return logits

        forcing_processor.is_stateless = True

        for _tok, _, _ in mtp_generate_step(
            prompt,
            self.model,
            max_tokens=2,
            logits_processors=[forcing_processor],
        ):
            pass

        # First call (backbone): context is the last prompt token.
        self.assertGreaterEqual(len(logged), 2)
        self.assertEqual(logged[0], 3)
        # Second call (MTP head): context must be T0 = 4, not the prompt token.
        self.assertEqual(logged[1], 4)

    def _assert_transactional_prompt_cache(self, prompt_cache, expected_tokens):
        n_main = len(self.model.layers)
        n_mtp = len(self.model.make_mtp_cache())
        self.assertEqual(len(prompt_cache), n_main + n_mtp + 1)

        state = prompt_cache[-1]
        self.assertIsInstance(state, MTPPromptCacheState)
        self.assertFalse(state.empty())
        self.assertEqual(state.num_tokens, expected_tokens)
        self.assertEqual(state.last_hidden.shape, (1, 1, 64))

        target_attention = next(c for c in prompt_cache[:n_main] if c.is_trimmable())
        mtp_attention = prompt_cache[n_main]
        self.assertEqual(target_attention.offset, expected_tokens)
        self.assertEqual(mtp_attention.offset, expected_tokens - 1)

    def test_mtp_prompt_cache_finalizes_at_length_boundary(self):
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

    def test_mtp_prompt_cache_finalizes_on_generator_close(self):
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

    def test_mtp_prompt_cache_invalidates_on_generator_throw(self):
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        prompt_cache = make_prompt_cache(self.model)
        generator = mtp_generate_step(
            prompt,
            self.model,
            max_tokens=10,
            prompt_cache=prompt_cache,
        )

        next(generator)
        with self.assertRaisesRegex(RuntimeError, "injected consumer failure"):
            generator.throw(RuntimeError("injected consumer failure"))

        boundary = prompt_cache[-1]
        self.assertTrue(boundary.empty())
        with self.assertRaisesRegex(ValueError, "non-empty boundary metadata"):
            next(
                mtp_generate_step(
                    mx.array([4], dtype=mx.uint32),
                    self.model,
                    prompt_cache=prompt_cache,
                )
            )

    def test_stream_mtp_finalizes_before_terminal_response(self):
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        prompt_cache = make_prompt_cache(self.model)
        generator = stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=1,
            mtp=True,
            prompt_cache=prompt_cache,
        )

        response = next(generator)
        self.assertEqual(response.finish_reason, "length")
        self._assert_transactional_prompt_cache(prompt_cache, len(prompt) + 1)
        generator.close()

    def test_stream_mtp_rejects_custom_sampler(self):
        def forcing_sampler(logprobs):
            return mx.zeros(logprobs.shape[:-1], dtype=mx.uint32)

        with self.assertRaisesRegex(ValueError, "does not support custom samplers"):
            next(
                stream_generate(
                    self.model,
                    self.tokenizer,
                    mx.array([0, 1, 2, 3], dtype=mx.uint32),
                    max_tokens=1,
                    mtp=True,
                    sampler=forcing_sampler,
                )
            )

    def test_stream_external_draft_with_mtp_honors_custom_sampler(self):
        def forcing_sampler(logprobs):
            return mx.full(logprobs.shape[:-1], 7, dtype=mx.uint32)

        captured = {}

        def fake_speculative_step(_prompt, _model, _draft_model, **kwargs):
            captured.update(kwargs)
            logprobs = mx.zeros(256)
            token = kwargs["sampler"](logprobs[None]).item()
            yield token, logprobs, True

        with patch.object(
            generate_module,
            "speculative_generate_step",
            side_effect=fake_speculative_step,
        ):
            response = next(
                stream_generate(
                    self.model,
                    self.tokenizer,
                    mx.array([0, 1, 2, 3], dtype=mx.uint32),
                    max_tokens=1,
                    draft_model=self.model,
                    mtp=True,
                    sampler=forcing_sampler,
                )
            )

        self.assertIs(captured["sampler"], forcing_sampler)
        self.assertEqual(response.finish_reason, "length")
        self.assertEqual(response.token, 7)

    def test_cli_mtp_sampler_routing(self):
        args = SimpleNamespace(
            seed=None,
            prompt_cache_file=None,
            trust_remote_code=False,
            model="target",
            adapter_path=None,
            quantize_activations=False,
            extra_eos_token=(),
            chat_template_config=None,
            prompt="hello",
            ignore_chat_template=True,
            system_prompt=None,
            prefill_response=None,
            draft_model="draft",
            temp=0.7,
            top_p=0.9,
            min_p=0.0,
            min_tokens_to_keep=1,
            top_k=0,
            xtc_probability=0.0,
            xtc_threshold=0.0,
            max_tokens=1,
            verbose=True,
            max_kv_size=None,
            kv_bits=None,
            kv_group_size=64,
            quantized_kv_start=0,
            num_draft_tokens=3,
            mtp=True,
        )
        parser = SimpleNamespace(parse_args=lambda: args)
        draft_model = object()
        sampler = object()

        with (
            patch.object(generate_module, "setup_arg_parser", return_value=parser),
            patch.object(
                generate_module,
                "load",
                side_effect=[
                    (self.model, self.tokenizer),
                    (draft_model, self.tokenizer),
                ],
            ),
            patch.object(generate_module, "make_sampler", return_value=sampler),
            patch.object(generate_module, "generate", return_value="") as generate,
        ):
            generate_module.main()

        kwargs = generate.call_args.kwargs
        self.assertIs(kwargs["sampler"], sampler)
        self.assertIs(kwargs["draft_model"], draft_model)
        self.assertTrue(kwargs["mtp"])

        args.draft_model = None
        with (
            patch.object(generate_module, "setup_arg_parser", return_value=parser),
            patch.object(
                generate_module, "load", return_value=(self.model, self.tokenizer)
            ),
            patch.object(generate_module, "make_sampler", return_value=sampler),
            patch.object(generate_module, "generate", return_value="") as generate,
        ):
            generate_module.main()

        kwargs = generate.call_args.kwargs
        self.assertIsNone(kwargs["sampler"])
        self.assertIsNone(kwargs["draft_model"])
        self.assertTrue(kwargs["mtp"])

    def test_mtp_prefill_reports_progress(self):
        callbacks = []
        prompt = mx.array([0, 1, 2, 3, 4, 5], dtype=mx.uint32)

        list(
            mtp_generate_step(
                prompt,
                self.model,
                max_tokens=1,
                prefill_step_size=2,
                prompt_progress_callback=lambda processed, total: callbacks.append(
                    (processed, total)
                ),
            )
        )

        self.assertEqual(callbacks, [(0, 6), (2, 6), (4, 6), (5, 6), (6, 6)])

    def test_mtp_max_tokens_zero_matches_prefill_cache_and_progress(self):
        prompt = mx.array([0, 1, 2, 3, 4, 5], dtype=mx.uint32)
        standard_cache = make_prompt_cache(self.model)
        mtp_cache = make_prompt_cache(self.model)
        standard_progress = []
        mtp_progress = []

        standard_output = list(
            generate_step(
                prompt,
                self.model,
                max_tokens=0,
                prompt_cache=standard_cache,
                prefill_step_size=2,
                prompt_progress_callback=lambda processed, total: standard_progress.append(
                    (processed, total)
                ),
            )
        )
        mtp_output = list(
            mtp_generate_step(
                prompt,
                self.model,
                max_tokens=0,
                prompt_cache=mtp_cache,
                prefill_step_size=2,
                prompt_progress_callback=lambda processed, total: mtp_progress.append(
                    (processed, total)
                ),
            )
        )

        self.assertEqual(standard_output, [])
        self.assertEqual(mtp_output, [])
        self.assertEqual(mtp_progress, standard_progress)
        self.assertEqual(mtp_progress[-1], (len(prompt), len(prompt)))
        self._assert_transactional_prompt_cache(mtp_cache, len(prompt))

        standard_attention = next(c for c in standard_cache if c.is_trimmable())
        mtp_attention = next(
            c for c in mtp_cache[: len(self.model.layers)] if c.is_trimmable()
        )
        self.assertEqual(mtp_attention.offset, standard_attention.offset)
        self.assertTrue(
            mx.allclose(mtp_attention.state[0], standard_attention.state[0]).item()
        )
        self.assertTrue(
            mx.allclose(mtp_attention.state[1], standard_attention.state[1]).item()
        )

        suffix = mx.array([6, 7], dtype=mx.uint32)
        resumed_tokens = [
            int(token)
            for token, _, _ in mtp_generate_step(
                suffix,
                self.model,
                max_tokens=4,
                prompt_cache=mtp_cache,
            )
        ]
        uncached_tokens = [
            int(token)
            for token, _, _ in mtp_generate_step(
                mx.concatenate([prompt, suffix]),
                self.model,
                max_tokens=4,
            )
        ]

        self.assertEqual(resumed_tokens, uncached_tokens)
        self._assert_transactional_prompt_cache(
            mtp_cache, len(prompt) + len(suffix) + len(resumed_tokens)
        )

    def test_public_generation_max_tokens_zero_has_explicit_terminal_response(self):
        prompt = mx.array([0, 1, 2, 3], dtype=mx.uint32)
        prompt_cache = make_prompt_cache(self.model)

        responses = list(
            stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=0,
                mtp=True,
                prompt_cache=prompt_cache,
            )
        )

        self.assertEqual(len(responses), 1)
        response = responses[0]
        self.assertEqual(response.text, "")
        self.assertIsNone(response.token)
        self.assertIsNone(response.logprobs)
        self.assertFalse(response.from_draft)
        self.assertEqual(response.generation_tokens, 0)
        self.assertEqual(response.generation_tps, 0.0)
        self.assertEqual(response.finish_reason, "length")
        self._assert_transactional_prompt_cache(prompt_cache, len(prompt))

        self.assertEqual(
            generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=0,
                mtp=True,
                prompt_cache=make_prompt_cache(self.model),
            ),
            "",
        )

    def test_mtp_prompt_cache_reuse_matches_uncached_generation(self):
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

    def test_mtp_prompt_cache_state_round_trip(self):
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

        self.assertIsInstance(loaded[-1], MTPPromptCacheState)
        self.assertEqual(loaded[-1].num_tokens, prompt_cache[-1].num_tokens)
        self.assertTrue(
            mx.allclose(
                loaded[-1].last_hidden,
                prompt_cache[-1].last_hidden,
            ).item()
        )

    def test_mtp_rejection_residual_sampling(self):
        """A controlled rejection emits residual token 2, not target token 1."""
        model = _ForcedRejectionMTPModel()

        def controlled_sampling(logprobs, _temp):
            # q has zero mass at token 2; p has zero mass at token 0.
            is_draft = not mx.isfinite(logprobs[2]).item()
            return mx.array([0 if is_draft else 1], dtype=mx.uint32)

        with patch.object(
            generate_module, "categorical_sampling", side_effect=controlled_sampling
        ):
            output = list(
                mtp_generate_step(
                    mx.array([0, 1, 2, 3], dtype=mx.uint32),
                    model,
                    max_tokens=2,
                    temp=1.0,
                )
            )

        self.assertEqual(model.verify_calls, 1)
        self.assertEqual(len(output), 2)
        self.assertEqual(int(output[0][0]), 1)
        self.assertFalse(output[1][2])
        self.assertEqual(int(output[1][0]), 2)


if __name__ == "__main__":
    unittest.main()
