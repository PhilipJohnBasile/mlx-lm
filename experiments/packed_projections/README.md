# Actual-weight packed gate/up investigation

This probe reads only the first MLP block's packed tensors from installed
checkpoints. Activations and unique top-8 expert routes are seeded synthetic
inputs. It compares the ordinary separate gate/up projections with one projection
whose packed output rows are concatenated, followed by the **existing upstream
compiled SwiGLU** and the unchanged down projection. For MoE this is SwitchGLU
with supplied routes; the learned router, weighted expert sum and shared expert
are outside the timed scope. It does not implement a new
activation kernel or install a model fast path. The related prior proposal is
[MLX-LM #1676](https://github.com/ml-explore/mlx-lm/pull/1676), closed unmerged at
the inspected revision; no result from that proposal is transferred here.

Both original and packed buffers are retained only as single-block comparison
fixtures. The packed representation has exactly the original combined byte
count and involves no dequantization or requantization. No model integration,
permanent duplicate-weight runtime, or default selection is added. In particular,
the reference uses its original contiguous matrices. Taking gate/up views from a
packed MoE tensor can cause extra upstream weight copies for sorted workloads;
that would unfairly slow the reference and is deliberately avoided.

## Physical M5 result

Apple M5 Max, 40 GPU cores and 128 GiB; macOS 27.0 build 26A5425a; MLX 0.32.2,
MLX-LM 0.32.0 and Python 3.13.14. All inputs use BF16 compute and affine/group-64
packed weights. Dense geometry is K=5120, intermediate=17408. MoE geometry is
K=2048, intermediate=512, 256 experts and top-8 routing.

Every tested packed tensor retained identical payload bits. Every output before
and after timing was bitwise identical. The declared arithmetic tolerances
(absolute 0.005, relative 0.02) were not needed or changed. Source snapshots,
exact tensor sizes/hashes, configuration and runtime fingerprints accompany
all raw receipts.

Each cell uses ten warmups, nine alternating ABBA/BAAB rounds, ten evaluations
per timed sample, before/after A/A, drift and zero-swap checks. Ratios below are
reference time / candidate time; a value above one is faster.

| Installed checkpoint | Tokens | Throughput ratio | Controls |
| --- | ---: | ---: | --- |
| Dense27B 4-bit MTPLX | 1 | inconclusive | A/A failed |
| Dense27B 4-bit MTPLX | 8 | 1.0033x | pass |
| Dense27B 4-bit MTPLX | 512 | 0.9813x | pass |
| Dense27B 4-bit MTPLX | 2048 | inconclusive | A/A and 19% drift failed |
| Dense27B 6-bit native MTP package | 1 | 0.9904x | pass |
| Dense27B 6-bit native MTP package | 8 | 1.0087x | pass |
| Dense27B 6-bit native MTP package | 512 | 0.9869x | pass |
| Dense27B 6-bit native MTP package | 2048 | 1.0119x | pass |
| Ornith35B MoE 4-bit MTPLX | 1 | inconclusive | A/A failed |
| Ornith35B MoE 4-bit MTPLX | 8 | inconclusive | A/A failed |
| Ornith35B MoE 4-bit MTPLX | 512 | 1.0365x | pass |
| Ornith35B MoE 4-bit MTPLX | 2048 | 1.0062x | pass |

All twelve cells had zero swap-in, swap-out and page-out deltas; noise failures
remain invalid. None of these valid block results supports a 5% full-request
claim, so no model-wide fusion is promoted. The actual six-bit tensors were
exercised directly: their packed gate/up shape is `[17408, 960]` each, with
BF16 scale/bias shape `[17408, 80]`. MTP sidecars were not used and speculative
verification/recovery remains outside this probe.

## Reproduce

Use the existing checkout's isolated environment and a fresh output directory.
The probe depends on the request harness's memory/fingerprint helpers in
`experiments/m5_inference` and the preserved timing helpers.

```sh
mkdir -p work/m5-qualification
.venv/bin/python experiments/packed_projections/probe.py \
  --model /path/to/installed/checkpoint --tokens 1,8,512,2048 \
  --rounds 9 --iterations 10 --output work/m5-qualification/packed-NEW
```

Run GPU workloads sequentially. The probe is a research comparison with generated
inputs, not pretrained whole-model quality, cache or request qualification. No
conclusion is drawn about untested layers, shapes, dtypes or quantization modes.
The compressed executed sources preserve the exact pre-format implementation.
Formatting preserved the full non-import AST. No new fusion of quantized
projection with activation was implemented; that would require separate profiling
and numerical qualification.
