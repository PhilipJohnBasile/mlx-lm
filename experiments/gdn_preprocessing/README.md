# Opt-in Gated DeltaNet preprocessing

**Status: implemented, pushed, Apple-Metal compiled and native-component tested on a hosted Metal device. Physical M5 execution and speed remain unmeasured.**

This is the first stage of the shared dense-27B / MoE optimization plan. It
replaces only the preparation of Q, K, V and convolution history in compatible
Qwen3.5-family GatedDeltaNet layers. It does not implement paired gate/up
projections, change quantization, or change the recurrent update.

## Implemented paths

| Mode | Behavior |
|---|---|
| `reference` (default) | Original-style concatenation, native depthwise convolution, SiLU and RMS normalization/scaling. |
| `direct` | Direct-window convolution and next-history writes in one custom Metal dispatch; native SiLU and normalization remain separate. |
| `fused` | Convolution, SiLU, Q/K normalization/scaling, V and next-history writes in one custom Metal dispatch. |

The custom modes read old history and new QKV through a conceptual concatenation,
without allocating the full concatenated array. The fused mode also avoids the
full convolution-output and activated-QKV intermediates. Necessary outputs, small
metadata arrays and any MLX contiguous-input copies remain. Allocation changes
are not measured process peak-memory savings or performance results.

The fused kernel assigns one 32-lane SIMD group to a 128-wide head and four
adjacent channels to each lane. It uses ordinary Metal SIMD operations, not NAX
matrix instructions or the Apple Neural Engine. Nothing is enabled globally.

## Numerical and cache contract

The convolution retains tap order and float32 accumulation, followed by an
activation-dtype cast. The fused path retains the explicit convolution, sigmoid,
activation, normalized-output and scaling casts. RMS epsilon is 1e-6. Q and K
scales are the same as the pinned reference. This is **not a proof of bitwise
Q/K/V parity**: compiler lowering, transcendental precision, SIMD reduction and
contraction must still be tested natively.

A mask zeroes only new projected QKV, never old history. A masked position may
therefore have nonzero convolution output; the unchanged recurrent kernel handles
its own mask. Next history contains raw masked pre-convolution inputs, not SiLU
outputs. Per-request lengths select the history window and are clipped before
conversion to int32. History writes have one writer per output element.

Both the current output and next-token recurrent state must be tested. Output
parity at one step cannot establish correct cache recovery. This experiment does
not implement speculative accept/reject bookkeeping.

## Supported experimental contract

- Metal execution; the supplied qualification runner specifically requires M5+.
- FP32, FP16 and BF16 projected QKV, convolution weights and history, same dtype.
- Q/K and value head dimensions exactly 128; value heads divisible by key heads.
- Depthwise convolution weight layout `[channels, taps, 1]`, taps 2 through 8.
- Positive sequence length, 64-bit buffer addressing and bounded grid metadata.
- Optional bool mask `[batch, tokens]`; optional signed int32/int64 lengths `[batch]`.
- The explicit GDN-layer adapter is single-device inference only. No training,
  autodiff, tensor-parallel or arbitrary-model support is claimed.

Dense profile uses Hk=16/Hv=48; MoE profile uses Hk=16/Hv=32. These are head
geometries, **not loaded 27B or 35B checkpoints**. The adapter reuses existing
projections, so it does not repack, concatenate or permanently dequantize weights.
It does not add a new 6-bit quantized kernel.

## Execute on the M5 Max

Use an existing macOS Python environment containing MLX, MLX-LM and NumPy, with
Xcode's Metal compiler installed. These scripts do not install packages, patch a
running server, download model weights or modify model files. The ZIP bundles the
verified MLX BF16 headers. A Git checkout without those headers downloads only the
two pinned public source headers and verifies their SHA-256 before using them. The Python APIs
were reviewed against the pinned revisions below; a different runtime must pass
the same checks.

From this directory:

```bash
python tools/qualify_m5.py --tokens 1,8,128,2048 --dtype bfloat16 --batch 1
```

The runner performs offline compilation/linking, actual MLX/Metal component and
GDN-layer tests, then paired preprocessing timings. It stops on errors and does
not count missing hardware as a successful skip. Results are written under
`results/<timestamp>/`. `component_validation_passed` is separate from
`performance_qualified`, which remains false: this runner is not a whole-model
qualification or automatic fast-path selector.

For concurrency controls, run a separate batch-four sweep:

```bash
python tools/qualify_m5.py --tokens 1,8,128,2048 --dtype float16 --batch 4
```

The six native test methods cover dtype and head geometries, masks, clipped and
large int64 lengths, noncontiguous layouts, all supported tap counts, chunked
history and the following token. The GDN-layer integration test uses small random
projection weights with real head geometries, not pretrained full-model weights.
Finite Q/K/V values must meet predefined tolerances; history must match payload
bits. Bitwise equality is reported separately from approximate equality.

The benchmark measures a warm **preprocessing-only** workload. It uses ABBA/BAAB
paired rounds, before/after A/A controls, full confidence-interval gates, raw times,
source hashes and correctness checks before and after timing. It does not report
model tokens/sec. A per-cell latency-candidate flag requires a stable run and a
lower speedup interval bound above 1.03; it never changes runtime policy.

## Explicit integration

After importing this directory on Python's path and putting the GDN layer in eval
mode, the component call is:

```python
from mlx_gdn_prep.integration import forward

# Keep separate caches for the candidate and baseline.
candidate = forward(gdn_layer, x, mask=mask, cache=candidate_cache, mode="fused")
baseline = gdn_layer(x, mask=mask, cache=baseline_cache)
```

Compare outputs, both cache states, cache offsets and a subsequent token. Do not
share mutable caches between the two arms. This is an explicit offline adapter;
there is no global monkeypatch or production serving integration.

## Validation actually executed

| Check | Result |
|---|---|
| Python host semantic, timing and source-integrity tests | 18 passed on Linux and independently on macOS. |
| Actual kernel bodies translated to C++, FP32 only | 110 cases / 1,990,656 output elements checked under ASan and UBSan. |
| Apple standalone shader compilation and metallib link | 576 instantiations passed, macOS 15.7.9 arm64 / Xcode 16.4. |
| Compiled source vs local kernel/exported source | Byte-for-byte/hash match. |
| Missing-M5 preflight | Exits nonzero on this Linux host; not a native pass. |
| Native MLX/Metal component tests | All 6 methods passed on Apple Paravirtual device, MLX/MLX-Metal 0.32.2. |
| Physical M5 performance and pretrained full-model parity | Not run. |

Host semantic FP16/BF16 tests use simulated rounding and float32 transcendental
math, not Apple's half/BF16 implementation. The sanitizer harness uses C++ threads
and a host SIMD reduction shim, not a GPU scheduler. No ThreadSanitizer run is
claimed. Shader compilation produced constexpr-if extension warnings; it was not
warning-free.

The first standalone build failed because it omitted the MLX BF16 overload
header. Adding the exact pinned headers fixed the exporter without changing the
kernel body. The compiler evidence is included under `validation/macos-compile` and
`validation/macos-final`. Actual native execution logs and the package inventory
are under `validation/native-hosted`. The hosted device identifies as Apple
Paravirtual device (`air64_v27`); it is not an identified physical M5 and is not
used for any M5 speed claim. The six native methods include real MLX GDN-layer
execution and subsequent recurrent/cache-state checks, with random small
projection weights rather than pretrained full models.
AIR/metallib binaries are omitted from this source bundle; their original hashes
and compiler logs are retained, and the tools regenerate them.

Reproduce host checks:

```bash
python -m unittest discover -s tests -p test_host.py -v
python -m unittest discover -s tests -p test_timing.py -v
python -m unittest discover -s tests -p test_headers.py -v
python tools/test_kernel_host.py --output-dir results/host-sanitizers
```

The sanitizer harness requires Clang with C++20 and the sanitizer runtimes.

## Source references and provenance

- MLX-LM base: `32bb4e68791c941db382d6fc8fa5b35ba9f3d98b`.
  https://github.com/ml-explore/mlx-lm/blob/32bb4e68791c941db382d6fc8fa5b35ba9f3d98b/mlx_lm/models/qwen3_5.py
- MLX base: `b6368984b8e02a3fb3ee7986846c0fb85e1fccf7`.
  Depthwise convolution, RMS normalization, Sigmoid, BF16 overloads and custom
  kernel ABI were reviewed at this revision.
  https://github.com/ml-explore/mlx/tree/b6368984b8e02a3fb3ee7986846c0fb85e1fccf7
- Final standalone compiler and host-test run:
  https://github.com/PhilipJohnBasile/mlx-lm/actions/runs/33944434070
- Successful native MLX/Metal component run:
  https://github.com/PhilipJohnBasile/mlx-lm/actions/runs/33944537249
- Published HEAD: `4b1f5191d81427efe8410ea6673b4d67cdc60e0b`.
- Experimental branch:
  https://github.com/PhilipJohnBasile/mlx-lm/tree/experimental/gdn-preprocessing/experiments/gdn_preprocessing

MLX-derived headers retain their MIT license in `LICENSE.upstream`. Source and
tests were produced with AI assistance. No world-first discovery, whole-model
parity or M5 speedup is claimed; no upstream PR was opened for this feature.
