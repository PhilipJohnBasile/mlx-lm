# Independent physical M5 single-sort evaluation

The original `workbench/single-sort` sources remain unchanged at handoff commit
`5db58608fa5d21d2613985cefe94a243420ccd63`. This directory records independent
native execution; it does not integrate or combine the candidate with NAX.

Apple M5 Max, 40 GPU cores, 128 GiB; macOS 27.0 build 26A5425a; Python 3.13.14,
MLX/MLX-Metal 0.32.2 and checkout MLX-LM 0.32.0. Eleven unchanged native test
methods passed, including payload-bit comparisons, routing boundaries, dtype and
layout matrices, separate streams, SwitchGLU integration and training fallback.

## Timing result

BF16, hidden size 2048, 256 experts and top-8 routing. Ratios are baseline time
/ candidate time, so greater than one is faster. These are warm components,
not complete requests or a qualified automatic dispatch policy.

| Scope | Tokens | Fused single-sort | Scatter inverse control | Compiled baseline control | A/A |
| --- | ---: | ---: | ---: | ---: | --- |
| Routing | 8 | 1.0409x | 1.0174x | 0.9909x | pass |
| Routing | 512 | 0.9629x | 1.0213x | 1.0074x | pass |
| Routing | 2048 | 0.9526x | 1.1100x | 0.9993x | pass |
| SwitchGLU | 1 | inconclusive | inconclusive | inconclusive | fail |
| SwitchGLU | 8 | 1.0050x | 1.0033x | 1.0239x | pass |
| SwitchGLU | 512 | 0.9970x | 1.0039x | 1.0272x | pass |
| SwitchGLU | 2048 | 0.9836x | 1.0038x | 1.0986x | pass |

All reported component correctness checks passed bitwise. The routing-only
scatter gain largely disappears in the complete SwitchGLU block. The fused
single-sort candidate is neutral or slower in the block and is rejected for
promotion. The compiled original block is a distinct positive control worth
investigating in pretrained requests; its 9.86% component gain is not yet a
whole-model result or a new compiler algorithm.

The initial routing run used five repetitions and failed A/A at all three
shapes. It is preserved alongside the repeat (100 repetitions, 20 warmups,
15 paired rounds). The block run used affine 4-bit/group-64 weights, intermediate
size 512, 20 repetitions, 10 warmups and 15 paired rounds. Its token-one cell is
invalid and remains so. These archived component runners do not implement the
full request runner's per-request zero-swap gate; A/A passing here must not be
presented as equivalent to full-model qualification. Six-bit block weights were
not exercised by this archived runner.

## Reproduce

From the existing checkout, use fresh output paths and run sequentially:

```sh
.venv/bin/python workbench/single-sort/tests/test_single_sort_native.py -v
.venv/bin/python workbench/single-sort/tools/bench_single_sort.py \
  --scope routing --case 8,8,2048,256 --case 512,8,2048,256 \
  --case 2048,8,2048,256 --dtype bfloat16 --rounds 15 --reps 100 \
  --warmup 20 --output work/m5-qualification/single-sort-routing-NEW.json
.venv/bin/python workbench/single-sort/tools/bench_single_sort.py \
  --scope block --case 1,8,2048,256 --case 8,8,2048,256 \
  --case 512,8,2048,256 --case 2048,8,2048,256 --dtype bfloat16 \
  --ffn 512 --bits 4 --rounds 15 --reps 20 --warmup 10 \
  --output work/m5-qualification/single-sort-block-NEW.json
python3 workbench/verify_inventory.py
```

Compressed raw timings, all noise failures, logs and their SHA-256 hashes are in
`evidence/20260905`. Source/runtime fingerprints are embedded in those receipts.
