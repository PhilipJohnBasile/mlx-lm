# Opt-in Gated DeltaNet preprocessing

Implemented research component. Default inference is unchanged; M5 execution and
speed are not yet qualified. No production model code is patched.

`reference` (default) uses the original concatenation, native depthwise convolution,
SiLU and normalization. `direct` replaces just convolution/history preparation.
`fused` combines convolution, SiLU, Q/K normalization/scaling, V and raw next-history
writes in one Metal dispatch. The latter two read old history and new projected QKV
directly rather than materializing their full concatenation. Possible contiguous
input copies and small metadata operations remain.

The adapter leaves projections, quantization, decay gates, recurrent update, output
normalization and output projection unchanged. It rejects training, tensor parallelism
and head dimensions other than 128. This is for compatible Qwen-family GDN layers,
not every model with 27B parameters, and uses ordinary Metal SIMD, not NAX matmul.

## Checks completed

- 18 local Python semantic, timing and header-integrity tests passed.
- Actual kernel bodies translated to C++: 110 FP32 cases / 1,990,656 output elements
  passed under AddressSanitizer and UndefinedBehaviorSanitizer.
- 576 standalone Metal instantiations compiled and linked on macOS 15.7.9 arm64 /
  Xcode 16.4, run 33943739873. The kernel bodies remain unchanged since that run.
  constexpr-if extension warnings remain. This is not GPU execution.
- Local and compiled kernel sources match. No M5 timing or full-model parity result.

## Native qualification

Requires macOS M5+, an existing MLX/MLX-LM/NumPy environment and Xcode Metal tools.
No package installation, model download or live-server modification is performed.
The emitter fetches only two pinned public MLX BF16 source headers if not already
bundled, verifies their SHA-256, and reuses them thereafter. A mismatched local
header is rejected, not silently overwritten.

```sh
cd experiments/gdn_preprocessing
python tools/qualify_m5.py --tokens 1,8,128,2048 --dtype bfloat16 --batch 1
```

This fails on absent hardware/build/test failure. It separates component validation
from performance qualification; the latter remains false until whole-model work is
completed. The benchmark is warm preprocessing only, not tokens/sec. It includes
ABBA/BAAB paired rounds, A/A calibration, confidence intervals, drift checks and
correctness gates before and after timing. It never enables a fast path automatically.

The native tests cover masking, clipped lengths, noncontiguous inputs, all supported
tap counts, dtype/head geometries, chunked history and subsequent recurrent-layer
steps. They use random small projections, not pretrained dense-27B or MoE models.
History payload bits must match; floating Q/K/V use declared tolerances with
bitwise checks reported separately. Approximate parity does not imply exact parity.

```python
from mlx_gdn_prep.integration import forward
candidate = forward(gdn_layer, x, mask=mask, cache=candidate_cache, mode="fused")
baseline = gdn_layer(x, mask=mask, cache=baseline_cache)
```

Use independent caches, eval mode and offline tests. Compare current outputs, both
cache states, offsets and the next token. No global hook is installed.

Host tests (the exporter first populates verified headers):

```sh
python tools/emit_metal.py results/prepare.metal
python -m unittest discover -s tests -p test_host.py -v
python -m unittest discover -s tests -p test_timing.py -v
python -m unittest discover -s tests -p test_headers.py -v
python tools/test_kernel_host.py --output-dir results/host
```

Sources: MLX b6368984b8e02a3fb3ee7986846c0fb85e1fccf7 and MLX-LM
32bb4e68791c941db382d6fc8fa5b35ba9f3d98b. Headers retain LICENSE.upstream.
Code and tests are AI-assisted. Paired quantized projections and speculative-state
policy are separate follow-ups, not implemented here. No upstream PR was opened.
