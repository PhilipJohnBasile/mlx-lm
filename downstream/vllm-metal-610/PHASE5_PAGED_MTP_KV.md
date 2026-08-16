# Phase 5 — scheduler-owned Qwen MTP-head KV

Phase 4 intentionally refuses scheduler prefix hits because target KV/GDN state
without the MTP head's own attention KV is not a complete speculative state.
This document defines the final cache strand needed before the prefix-cache +
MTP gate can be removed.

## State ownership

A cached Qwen hybrid prefix is valid only when all five components share the
same committed token boundary:

1. target paged SDPA KV;
2. target GDN convolution state;
3. target GDN recurrent state;
4. Qwen MTP-head paged KV, intentionally committed through `L - 1`;
5. target boundary hidden state `h_L`.

The first suffix token `t_(L+1)` closes the one-token MTP lag by evaluating
`mtp_forward(h_L, t_(L+1))` before normal drafting resumes.

## Scheduler representation

The MTP head should be represented as an additional full-attention cache group,
not as a request-local `mlx_lm.models.cache.KVCache`.

```text
Hybrid request block table
├── group 0..N: target SDPA attention groups
├── group N+1..M: target GDN/Mamba state groups
└── group M+1: Qwen MTP-head attention KV
```

The MTP group uses the same content-addressed prefix hash and block lifecycle as
the target prefix. Cache admission is atomic: a hit is adopted only when target
SDPA, GDN checkpoints, MTP KV, and boundary metadata are all present.

## Runtime

- create a one-layer `MHAPagedAttentionRuntime` for `model.mtp`;
- patch only the MTP decoder layer's attention module;
- adopt a dedicated scheduler `FullAttentionSpec`/layout group;
- use the request's MTP scheduler group when constructing proposer context;
- maintain the one-token lag by shifting MTP positions relative to target
  positions, never by trimming or synthesizing hidden history;
- materialize target GDN promotion and MTP KV writes before scheduler block
  reuse, copy-on-write, preemption, or eviction.

## Prefix-hit admission

A scheduler prefix hit at target length `L` is admissible only if:

```text
mtp_cached_tokens == L - 1
boundary_hidden_token == L
boundary_hidden is present
all target/GDN/MTP block groups share the same prefix hash lineage
```

An exact hit with no suffix token cannot close the lag and should either:

- wait for the next non-empty suffix, or
- fall back to non-MTP generation for that step.

It must never pair a fresh MTP cache with a restored target prefix.

## Required tests

- fresh prompt versus warm exact-prefix hit;
- shared prefix with divergent suffix;
- hit at `block_size - 1`, `block_size`, and `block_size + 1`;
- accepted and rejected draft immediately after a hit;
- target/GDN/MTP group copy-on-write;
- missing MTP group or boundary hidden fails closed;
- eviction and reallocation cannot alias an old MTP KV block;
- preemption/resume preserves all groups or disables drafting;
- cached continuation equals uncached full-transcript greedy output;
- structured tool output and multi-turn agent continuation;
- all-cold and warm-prefix performance reported separately.

## Gate-removal criteria

The platform configuration and runtime hybrid-speculation gates remain until:

1. the dedicated MTP cache group is included in scheduler memory planning;
2. prefix-hit admission is atomic across every state group;
3. the full correctness matrix passes with a real Qwen MTP checkpoint;
4. warm-prefix TTFT and decode throughput both improve without an all-cold
   regression being hidden in aggregate numbers.
