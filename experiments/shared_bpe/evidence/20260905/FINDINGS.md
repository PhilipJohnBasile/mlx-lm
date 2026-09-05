# Shared-map physical M5 findings

Implementation: `7e4ffb0403ab0941c55cd1f5fddf7719c2a19201`. The metadata-sharing
selector is independent of GDN preprocessing; both arms use unchanged original
model arithmetic, quantization, cache, sampling and scheduler behavior.

## MoE: completed, all controls pass

Installed Ornith-1.5-35B-A3B-V2 MTPLX, Qwen3.5 MoE architecture, BF16 compute,
affine 4-bit/group-64 main projections. All checkpoint/config/tokenizer and
runtime binary fingerprints are in the raw request receipt. The loaded main
shards were independently SHA-256 matched to the pinned remote model metadata
in the GDN qualification. Preparation of the shared vocabulary took 102.9 ms
once per loaded tokenizer and is excluded from warm service latency.

Each row generates 64 tokens per stream through complete requests with ragged
prompts at concurrency four. Warmup, fifteen alternating ABBA/BAAB rounds,
before/after A/A, drift, pre/post exact log-probability and cache comparisons,
and zero-swap checks passed. All 480 timed requests retained the same greedy
tokens. No timing threshold or numerical tolerance was weakened.

| Prompt tokens | Concurrency | Reference / candidate median ms | Throughput ratio (95% interval) | Reference / candidate TTFT ms |
| --- | ---: | --- | --- | --- |
| 32 | 1 | 633.4 / 537.6 | 1.1771x (1.1744–1.1796) | 155.7 / 59.4 |
| 32 | 4 | 1250.1 / 861.5 | 1.4517x (1.4433–1.4606) | 522.0 / 130.3 |
| 2048 | 1 | 1225.2 / 1129.7 | 1.0842x (1.0806–1.0883) | 726.8 / 634.0 |
| 2048 | 4 | 2971.2 / 2641.5 | 1.1287x (1.1228–1.1342) | 2191.0 / 1858.6 |

The paired sustained-decode rate ratios after the first token are 0.9978,
0.9973, 1.0019 and 0.9951 respectively. This removes repeated request setup;
it does not establish faster GPU prefill arithmetic or sustained GPU decoding.
No required complete-request control regressed. Cold loading, one-time map
preparation, thread-safe concurrent selector mutation and mutable vocabularies
are outside this warm-service qualification. Concurrency means independent
streams in the existing single scheduler batch.

## Dense coverage in progress

The dense matrix is still running. Its first short/concurrency-one cell passed
bitwise state/text checks but failed A/A TTFT noise controls, so its observed
latency improvement is not an accepted result. That receipt will be retained
separately and repeated without changing the gates. No dense improvement is
claimed until its required timing controls pass.

The map only admits the original BPE detokenizer on a frozen tokenizer. It has
a tested reference path, unsupported/nested-selection rejection and restoration
after exceptions. No automatic selection or installed inference service changed.
