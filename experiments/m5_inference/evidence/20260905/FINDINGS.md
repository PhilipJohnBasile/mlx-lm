# Initial physical M5 findings

These are warm complete-request measurements through the unchanged MLX-LM scheduler, with 64 greedy output tokens per request, ragged concurrency-four prompts, and prefill chunks of 512. Ratios are paired reference time divided by candidate time. Validity includes A/A, drift, correctness and no observed swap activity. No cell reaches the 5% end-to-end target.

| Model | Prompt target | Concurrency | Reference / candidate ms | Speed ratio (95% CI) | Valid timing |
|---|---:|---:|---|---|---|
| Qwen3.8 dense 27B, affine4/g64 | 32 | 1 | 2228.4 / 2224.5 | 1.0026 (0.9979-1.0097) | yes |
| Qwen3.8 dense 27B, affine4/g64 | 32 | 4 | 3268.0 / 3205.8 | 1.0159 (1.0083-1.0235) | yes |
| Qwen3.8 dense 27B, affine4/g64 | 2048 | 1 | 4503.6 / 4492.7 | 1.0055 (1.0017-1.0105) | yes |
| Qwen3.8 dense 27B, affine4/g64 | 2048 | 4 | 13171.4 / 12956.9 | 1.0182 (1.0150-1.0205) | NO |
| Ornith Qwen3.5 MoE35B, affine4/g64 | 32 | 1 | 639.8 / 632.7 | 1.0214 (1.0107-1.0407) | NO |
| Ornith Qwen3.5 MoE35B, affine4/g64 | 32 | 4 | 1231.7 / 1217.2 | 1.0189 (1.0086-1.0313) | NO |
| Ornith Qwen3.5 MoE35B, affine4/g64 | 2048 | 1 | 1226.9 / 1213.8 | 1.0125 (1.0007-1.0244) | yes |
| Ornith Qwen3.5 MoE35B, affine4/g64 | 2048 | 4 | 2848.6 / 2803.7 | 1.0149 (1.0104-1.0195) | yes |

All eight completed cells passed both pre/post correctness checks: every compared generated log-probability and final cache array was bitwise identical, and greedy tokens/text matched.

The dense 2048/concurrency-four timing is invalid because one sample recorded four swap-in pages. Both short-prompt MoE timings are invalid because a TTFT A/A confidence interval extends beyond the declared bounds. Failed observations are retained. Repeats are separate receipts; none of these invalid estimates is an accepted gain.

The original physical qualifier compiled 576 shader instantiations (24 constexpr extension warnings) and passed all six native test methods. Fused preprocessing at 2048 tokens was 1.8569x (dense geometry) / 1.8090x (MoE geometry), which is a component result. All timed BF16 component outputs matched payload bits. Four new native adapter tests and four inventory tests passed.

The independent dense6 pilot also passed bitwise logits/cache parity and matching generated tokens for both direct and fused modes. Its four single-request diagnostic observations are not a repeated full performance qualification.

Archived handoff files remain unchanged. Compressed raw receipts and executed active sources are listed by SHA-256 in MANIFEST.json. Decompress a file with `python3 -m gzip -d filename.gz` in a scratch directory; preserve this evidence directory. Offline AIR/metallib binaries are reproducible and their hashes are retained in the manifest.

The MoE model shards and tokenizer match remote revision `bdadcddf7e8abf7234e7710c58c45c1b45765a47`. The dense4 local files are fingerprinted, but the recorded publication revision `e37c5433c552be35c1db4563e3add4726db9d55b` could not be verified: the repository returns 401 without authentication and 404 with the existing login. Do not promote that metadata claim into a verified remote revision.

Flash, speculative recovery, paired projections, single-sort and indirect-NAX execution are not qualified by this increment. Full project objective remains open.

## MoE short-prompt repeat

Fifteen rounds per cell, unchanged gates, fresh output. Both repetitions pass A/A, drift, no-swap and pre/post bitwise correctness. The original invalid observations above remain retained.

| Concurrency | Reference / candidate ms | Paired speed ratio (95% CI) |
|---:|---|---|
| 1 | 630.1 / 627.2 | 1.0007 (0.9917-1.0069) |
| 4 | 1229.1 / 1212.4 | 1.0157 (1.0093-1.0225) |

Concurrency one is neutral; concurrency four improves about 1.57%. Neither reaches the 5% target. The dense long/concurrency-four memory-invalid cell still requires a separate repeat.

## Dense long/concurrency-four repeat

A separate five-round repeat completed with bitwise log-probability and final-cache
parity before and after timing. It remains **invalid**: four swap-in pages occurred
in one timed request (no swap-out or page-out). The A/A and drift gates passed.
The observed 1.0205x throughput ratio (95% interval 1.0195–1.0216) is retained as
adverse/inconclusive evidence, not an accepted performance result. No timing or
correctness threshold was relaxed and no automatic mode was enabled. The raw
receipt is `dense4-long-c4-repeat-02__report.json.gz`.
