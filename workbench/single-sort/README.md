# Single-sort MoE routing: M5-targeted experiment

**Status: implemented prototype, host-validated; no Metal compilation, GPU
execution, or measured speedup has been obtained in the authoring environment.**
The implementation is opt-in, inference-only, and is not submitted upstream.

## Opportunity found in current MLX-LM

`mlx_lm/models/switch_layers.py::_gather_sort` currently does:

```python
order = mx.argsort(indices.flatten())
inverse = mx.argsort(order)
packed = x.flatten(0, -3)[order // top_k]
sorted_experts = indices.flatten()[order]
```

The second argsort sorts a permutation, not expert IDs. Its output is exactly
`inverse[order[r]] = r`. This experiment keeps the first expert argsort unchanged
and builds that inverse inside the activation-gather kernel. The same kernel
also gathers the sorted expert IDs and computes the token index in registers.

The proposed change is **two argsort calls to one**, plus one fused gather/metadata
kernel. It does not halve total model work. The activation output, sorted IDs,
and inverse output still exist; the redundant sort and separately materialized
`order // top_k` are removed. Actual Metal dispatch counts require the optional
GPU capture, since a single argsort can launch several kernels.

## Why target M5

Apple describes M5 Neural Accelerators as improving the compute-intensive prefill
phase of MLX inference. The hypothesis here is that reducing routing and data
movement around those faster expert matmuls can improve the complete MoE block.
This kernel itself uses ordinary Metal integer loads/stores, **not** new NAX
instructions and not the Apple Neural Engine. It may help older Metal GPUs too.
That is a control to measure, not a claim of M5 exclusivity.

The production integration is disabled by default. With `MLX_MOE_SINGLE_SORT=1`
it is admitted only on named Apple M5-family or later devices, on the default
GPU device, for supported activation/index dtypes, and outside `Module.training`.
Future-chip admission is an explicit opt-in policy, not evidence of performance
on unreleased or untested hardware.

The existing `indices.size >= 64` sort threshold is unchanged. Thus this targets
prefill, batching, and sufficiently large speculative verification steps. For
one token with top-k=8, there are only 8 routes and **nothing changes**. Dense
models without these MoE layers are unaffected.

## What is and is not novel

The inverse-permutation identity is standard; this is not a new algorithmic
invention. Fused MoE dispatch/combine also has prior art, including MLX PR #3158.
The narrower candidate is eliminating MLX-LM's *second* sort by fusing inverse
construction with the activation gather used by its shared switch layers.

No matching proposal for this exact replacement was found in the checked public
MLX/MLX-LM PR and web searches (`inverse permutation`, `inv_order`, `gather_sort`,
`unsort`, and related fused-routing terms). One exact-phrase organization PR
search returned zero results. Search terms do not prove worldwide novelty,
absence from every fork, or that nobody else has considered it.

## Files and scope

- `mlx_lm/models/_single_sort_moe.py`: kernel, opt-in admission, and benchmark controls.
- `mlx_lm/models/switch_layers.py`: two inference call-site integrations.
- `tests/test_single_sort_native.py`: actual Metal acceptance tests, not run here.
- `tools/bench_single_sort.py`: A/A + alternating A/B and ablation measurements.
- `tools/check_kernel_host.py`: extracts the exact Metal body and compiles it as C++
  with checked input/output buffers, AddressSanitizer, and UndefinedBehaviorSanitizer.
- `validation/`: executed host logs and native-not-run status.
- `single-sort-moe.patch`: applies to the referenced MLX-LM source, including native
  tests and the benchmark under the repository's `tests/` and `benchmarks/` paths.

### Correctness contract

Given the exact expert permutation `p` from the retained argsort and `k=top_k`:

```text
packed[r, 0, c] = original_x[p[r] // k, 0, c]
sorted_experts[r] = original_experts[p[r]]
inverse[p[r]] = r
```

Because `p` is a permutation of `[0, routes)`, every inverse slot has exactly one
writer, even when many expert IDs are equal. Only column zero writes metadata;
other columns only copy their own activation element. No atomics, barriers,
floating-point arithmetic, altered expert selection, or changed reduction order
are introduced. The first sort's tie ordering is preserved, whatever it is.

Activation payloads are viewed as equal-width uint16/uint32 words for copying,
then viewed back. This is intended to preserve signed zeros, NaN payloads and
subnormals as well as ordinary fp16/bf16/fp32 values. Native tests must confirm
that the entire MLX view/dispatch pipeline honors this contract.

Indexing uses 64-bit offsets, including signed input strides. Existing flatten
operations can still require a copy for some layouts; that cost is not claimed
eliminated. The custom kernel does not force an additional row-contiguity copy.
The kernel factory is cached. Token count and feature width are not JIT template
parameters; only top-k is specialized, in addition to generated dtype signatures.

### Deliberate restrictions

This prototype is not an autodiff primitive. `train()` mode stays on the original
path. **Do not enable it for `grad`, `vjp`, `jvp`, or `vmap`, including gradient
calculations on a module in `eval()` mode.** A proper derivative/batching rule is
separate work, not silently approximated here. Whole-step `mx.compile` integration
and shapeless replay of this custom kernel are also unvalidated. Keep the feature
disabled for those configurations; the benchmark only compiles its baseline control.

No new GPU output-combine fusion is included. No NAX matmul, quantization,
attention, stream synchronization, or expert weight format is changed. No
existing core gather correctness bugs are fixed by this experiment.

## Validation performed here

Authoring environment: Linux x86_64, Python 3.13.5, no native MLX/Metal device.

| Validation | Result |
|---|---|
| Host Python contract/reporting tests | 15 tests passed |
| All permutations of sizes 1 through 8 | 46,233 inverse identities checked |
| Exact kernel body compiled as C++ with ASan/UBSan | 1,773 cases passed |
| Instrumented kernel output elements checked | 13,065,859 |
| Layouts | contiguous, padded, reversed rows/columns, broadcast, transposed |
| Index boundaries | 32,767 / 32,768 / 32,769 and 65,535 / 65,536 / 65,537 routes |
| Wide-address arithmetic | synthetic offset above 2^32, no giant allocation |
| Source provenance | original Git blob matched byte-for-byte |
| Python syntax | passed |
| Native runner without Metal | exits 2 and explicitly reports NOT RUN |
| Native Metal compile / GPU correctness / speed | **not run** |
| Full MLX / MLX-LM suites, pre-commit | **not run** |

The C++ test models each Metal thread's body sequentially in reverse traversal.
It detects invalid addresses, unwritten outputs and duplicate writers. It is
**not** a Metal compiler test, GPU race detector, proof of MLX graph integration,
or performance result.

## Run on the M5 Max

Use a Python environment with native MLX and MLX-LM installed. The standalone
bundle can be tested without changing the installed package. Its block tests use
MLX-LM's activation definitions, so the referenced MLX-LM revision below is the
reproducible baseline. A different installed version is a compatibility test.

From this bundle directory:

```bash
python tests/test_single_sort_native.py
MLX_MOE_SINGLE_SORT=0 python tools/bench_single_sort.py \
  --scope routing --output routing-fp16.json
MLX_MOE_SINGLE_SORT=0 python tools/bench_single_sort.py \
  --scope block --output block-fp16.json
```

Repeat with `--dtype bfloat16`. To inspect shader dispatches in a small cell:

```bash
MTL_CAPTURE_ENABLED=1 MLX_MOE_SINGLE_SORT=0 \
python tools/bench_single_sort.py --scope routing \
  --case 512,8,2048,128 --capture-dir captures --output captured.json
```

Compare threadgroup widths 64, 128 and 256 with `--threadgroup-width`; the default
256 is an experimental starting point, **not a measured M5 tuning result**.
Do not select a width on a single noisy case. `--allow-other-metal` allows an
older-GPU control through the direct API but does not change production gating.

### Benchmark interpretation

The benchmark compares the original path with (1) inverse construction via
ordinary MLX scatter, and (2) the fused kernel. A graph-compiled baseline is also
included. All timed outputs must match the original bitwise, before and after
timing; a mismatch aborts rather than printing an apparent performance win.
The compiled control may expose a pre-existing numerical difference in an
installed runtime; such a failure needs investigation, not a relaxed candidate
tolerance.

`--scope block` uses the *same* native SwitchGLU projections, weights and activation
for all arms, plus unsorting, routing-weight multiplication and reduction. The
forward algebra is checked against the installed SwitchGLU implementation. It is
not a full model, tokenizer, sampling loop, or serving benchmark, and does not
include the tiny production admission-hook overhead. Default weights are synthetic,
affine-4-bit experts. No model download is required.

Inputs are resident and reused, not cold-cache. Timings include graph construction,
evaluation and synchronization on every call; compilation is warmed before paired
measurements. `first_measured_call_ms` is recorded *after* reference construction
and is not an isolated cold-compilation comparison. Report cold-start cost from
separate fresh processes before making a production rollout decision.

Each cell includes alternating A/B order, rotating comparison order, A/A calibration,
raw paired milliseconds and a paired-bootstrap interval. Ratios above 1 favor the
candidate. An A/A interval outside [0.95, 1.05] marks the cell invalid and makes the
program fail at the end. A valid A/A check does not, by itself, establish a win.
Compare with the best correct control, not only with the slowest baseline.

The output also records device/OS/runtime information, available runtime binary
hashes, source hash and Xcode version. No results are uploaded automatically.

## Optional local integration

Only after the native correctness and performance gates pass, review the patch in
an isolated MLX-LM checkout:

```bash
git checkout 32bb4e68791c941db382d6fc8fa5b35ba9f3d98b
git apply --check /absolute/path/to/single-sort-moe.patch
git apply /absolute/path/to/single-sort-moe.patch
python -m pip install -e .
# Set this before importing mlx_lm in an inference-only process:
export MLX_MOE_SINGLE_SORT=1
```

Unset `MLX_MOE_SINGLE_SORT` or set it to `0` before restarting to disable the path.
The patch does not enable itself for ordinary users and no automatic performance
thresholds have been inferred from host tests.

## Source provenance

Inspected MLX core: `b6368984b8e02a3fb3ee7986846c0fb85e1fccf7`.
Inspected MLX-LM: `32bb4e68791c941db382d6fc8fa5b35ba9f3d98b`.
Original switch-layer Git blob: `1fe5d917e6b194b1681bbb1c69589ad3dc759d65`.

Primary references:

- https://github.com/ml-explore/mlx-lm/blob/32bb4e68791c941db382d6fc8fa5b35ba9f3d98b/mlx_lm/models/switch_layers.py
- https://github.com/ml-explore/mlx/blob/b6368984b8e02a3fb3ee7986846c0fb85e1fccf7/mlx/backend/metal/quantized.cpp
- https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- https://github.com/ml-explore/mlx/pull/3158 (related fused-dispatch/combine prior art)
- https://api.github.com/search/issues?q=org%3Aml-explore%20is%3Apr%20%22inverse%20permutation%22

AI involvement: ChatGPT performed the source/PR investigation, wrote this prototype,
tests and benchmark, and ran the host checks recorded here. No human review, native
M5 measurement, or upstream approval is represented as having occurred.
