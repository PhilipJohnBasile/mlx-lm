# Native MTP hardening carry-forward — August 22, 2026

This branch starts at AirRunner's `feat/mtp-native` head
`e8ceeccf118d4a089e22431f28f89d5ffe7d9d2b` and intentionally opens no pull
request. It records the selected correctness work from closed
[ml-explore/mlx-lm#1740](https://github.com/ml-explore/mlx-lm/pull/1740)
that AirRunner offered to carry into the live review path.

Source commit:

- `PhilipJohnBasile/mlx-lm@bc1d11414c12372cdf76538e09ce1fbc54b7fc7b`
- https://github.com/PhilipJohnBasile/mlx-lm/commit/bc1d11414c12372cdf76538e09ce1fbc54b7fc7b

The semantics below should survive even if #990 is refactored around
`make_draft_model` and `speculative_generate_step`.

## 1. Transactional prompt-cache reuse

Preserve `MTPPromptCacheState` or an architectural equivalent that records the
native-MTP continuation boundary. The implementation must:

- keep target cache, MTP-head cache, and boundary metadata mutually aligned;
- fail closed when a populated target cache lacks compatible MTP entries or
  boundary metadata;
- finalize the cache transaction at an output-length boundary;
- finalize the cache transaction when the generator is closed early;
- preserve cached-versus-uncached output parity on a later conversation turn;
- survive `save_prompt_cache` / `load_prompt_cache` round trips; and
- reject exact-hit or stale-boundary reuse when no safe suffix can realign the
  lagging MTP cache.

Relevant source areas in `bc1d114`:

- `mlx_lm/generate.py`: `_MTPPromptFinalizePlan`, prompt-cache splitting,
  restoration, and `finally`-based finalization;
- `mlx_lm/models/cache.py`: `MTPPromptCacheState`;
- `mlx_lm/server.py`: fail-closed reusable-cache validation; and
- `tests/test_mtp.py`: transactional finalization, reuse parity, and round-trip
  coverage.

## 2. Accepted-token log probabilities

When a draft token is accepted, the yielded token is part of the target model's
output distribution. Its public log-probability vector must therefore be the
verifier/target distribution.

The inspected `e8ceecc` head still yields `draft_lp` on acceptance. The #1740
fix yields `verify_lp` instead. Preserve this semantic behavior regardless of
how the final sampling API is structured.

Required regression:

1. force or observe an accepted draft;
2. retain the target/verifier log-probability vector for that position;
3. assert that the yielded vector equals the target/verifier vector; and
4. assert that the test would fail when the draft-head vector is returned.

## 3. Fail-closed capability and processor guards

Preserve a loaded-head capability check equivalent to `_model_supports_mtp`.
Method presence alone is insufficient: converted checkpoints may expose model
methods while carrying no usable trained MTP weights.

Preserve the `logits_processors` guard:

- plain functions and `functools.partial` may be accepted as stateless;
- explicitly marked safe callables may be accepted;
- unknown stateful callable objects must fail closed; and
- supported context-sensitive stateless processors must match ordinary serial
  generation.

## 4. Regression checklist

The implementation that lands should retain tests equivalent to:

- `test_missing_mtp_weights_disable_head`
- `test_missing_mtp_moe_weights_do_not_crash`
- `test_mtp_rejects_unaligned_populated_prompt_cache`
- `test_mtp_rejects_stateful_logits_processor`
- `test_mtp_generate_identity_with_logits_processor`
- `test_mtp_processor_prev_tokens_correct_at_draft_step`
- `test_mtp_prompt_cache_finalizes_at_length_boundary`
- `test_mtp_prompt_cache_finalizes_on_generator_close`
- `test_mtp_prompt_cache_reuse_matches_uncached_generation`
- `test_mtp_prompt_cache_state_round_trip`
- an explicit accepted-draft target-log-probability regression

The exact names may change. The behaviors should not.

## 5. Attribution

AirRunner's #990 established the reference implementation and native Qwen MTP
approach. The transactional cache, accepted-token log-probability correction,
and fail-closed guards above came from #1740 and `bc1d114`.

Referencing #1740 and `bc1d114` in the integrating commit or PR history is
sufficient. Preserve commit authorship when code is copied directly.

## 6. Vacation authorization

Philip John Basile may be off-grid for approximately two weeks beginning
August 26, 2026. AirRunner may integrate, refactor, and merge the selected work
without waiting for a response. The goal is to preserve the correctness
semantics, not the closed PR's exact function structure.
