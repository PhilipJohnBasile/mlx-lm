# Shared immutable BPE token map (explicit experiment)

Real-request cProfile diagnostics on the physical M5 Max found about 89–93 ms
spent constructing one BPE streaming detokenizer. Its constructor reads the
full vocabulary twice. This experiment builds the same token-ID map once per
loaded tokenizer and stores it as an immutable tuple. Every stream keeps its
own tokens, text, pending UTF-8 bytes and offset. Decode methods are inherited
unchanged from the original BPE implementation.

The target is warm complete-request latency and time to first delivered text.
No projection, attention, recurrence, cache, quantization or sampling operation
changes. This does not aim to increase sustained GPU decode throughput. The
one-time preparation cost and token-map hash are recorded separately. Initial
model/tokenizer loading and preparation are excluded from warm service timings.

## Admission and fallback

Selection is explicit and local to one `TokenizerWrapper`. Only the inspected
original `BPEStreamingDetokenizer` is admitted. The tokenizer must remain frozen
for the prepared object's lifetime. Replacing its backend, changing vocabulary
size, or nesting selections is rejected. Same-size vocabulary edits are outside
the contract and require explicit preparation again. Other tokenizer types use
the unchanged reference path. No automatic mode is enabled.

The selector restores the original factory on success and exceptions. The
shared map contains no per-stream text or tokens. Model weights are unchanged.

## Validation contract

Before pretrained testing, five regressions passed: exact streaming prefixes
across interleaved streams and reset/finalize cycles (UTF-8, whitespace, partial
and invalid bytes), independent stream state, separate tokenizer maps, exception
restoration, and admission after vocabulary growth. A pure metadata-sharing
change requires bitwise generated log-probability/cache equality plus identical
greedy tokens and final text before and after timings. Arithmetic tolerances
from the reused harness do not override this stronger gate.

Use the existing checkout's `.venv`, a fresh output directory, and one GPU
workload at a time:

```sh
.venv/bin/python experiments/shared_bpe/test_shared_bpe.py -v
.venv/bin/python experiments/shared_bpe/benchmark.py --model /path/to/model \
  --tokens 32,2048 --concurrency 1,4 --max-tokens 64 --rounds 5 \
  --output work/m5-qualification/shared-bpe-NEW
```

The request harness is from `experiments/m5_inference`; its defaults still select
GDN preprocessing. This runner supplies a different explicit selector, enforces
bitwise parity, and snapshots its additional sources. Warmup, alternating A/B,
before/after A/A, drift and zero-swap gates are unchanged. All raw samples and
source/binary/checkpoint hashes are recorded. Results remain experimental until
that matrix finishes. No speedup is inferred from the profile alone.

For repeat qualification, `--warmup-rounds 4` warms each arm four times.
The default remains one round; all correctness, A/A, drift and swap gates remain
unchanged. Short-prompt repeats use fifteen measured rounds to resolve noisy
TTFT calibration, with all earlier invalid receipts preserved.
