# Row-indirect NAX MoE projections

**Status: implemented research prototype; not native-Metal validated and not a measured speedup.**

This opt-in MLX extension targets the activation gather before expert-sorted
quantized MoE gate/up projections on M5-class GPUs. It keeps activations in
original token order and gathers directly into NAX cooperative-tensor fragments.
A small source-row address plan is computed once per SIMD lane, outside both the
K loop and the expert-run loop. There is no expanded activation tensor between
routing and the two projections.

The extension does not patch a running server, alter model files, change MLX
settings, push a branch, or open a pull request. It is forward-only and is not
ready to enable by default. The default Python helper mode is `upstream`.

## The specific upstream opportunity

Source references:

- MLX: `b6368984b8e02a3fb3ee7986846c0fb85e1fccf7`.
- MLX-LM: `32bb4e68791c941db382d6fc8fa5b35ba9f3d98b`.
- In `mlx_lm/models/switch_layers.py`, `_gather_sort` produces
  `x.flatten(0, -3)[order // top_k]`, an expanded expert-sorted activation array.
  Both the up and gate projections consume that same array.
- In `mlx/ops.cpp`, the public `gather_qmm` creates `GatherQMM` with
  `right_sorted = sorted_indices && !lhs_indices_`. Supplying an explicit LHS
  row map therefore does not preserve the sorted-RHS fast-path flag.
- The sorted NAX kernel consumes a contiguous activation tile. Its individual
  fragment lanes already have a deterministic mapping to two rows per 16x16
  fragment. Substituting row addresses at that boundary avoids a new activation
  staging buffer without changing the matrix multiplication.

This is not the earlier single-sort/inverse-permutation proposal. The reference
adapter deliberately keeps both existing sorts and the output unsort. It also
leaves the activation function, down projection, and outer routing-weight sum
unchanged.

The general idea of fusing gather with GEMM is prior art. The claim here is only
a concrete extension of this pinned MLX sorted-RHS NAX path with hoisted indirect
activation rows. Targeted public PR searches did not identify this exact change;
that is not proof of global novelty. Related existing work includes MLX #4390
(grouped CUDA matmul), #3158 (expert-parallel dispatch/combine), and #3888
(small-M weight reuse). No claims from those projects' benchmarks are transferred
to this prototype.

## What is implemented

`mlx_nax_indirect/kernels/route_address.h` contains the shared Metal/C++ row
address calculation and fragment coordinate mapping. It uses 64-bit element
offsets, validates before reading a row-map entry, and distinguishes a padded
row from an invalid source index.

`indirect_load.h` stores two row addresses per lane for BM=32 or four for BM=64.
The addresses are reused for all K tiles and expert runs. Loads preserve the
pinned NAX fragment ordering. Padding produces zero. An invalid active source
index produces NaN without dereferencing outside X.

`gather_body.metal` is derived from the pinned
`affine_gather_qmm_rhs_nax` kernel. The original NAX matmul calls, accumulation
order, weight dequantizer, and normal-path barriers are retained. `INDIRECT=false`
is a matched JIT control that consumes an already-gathered input. An invalid
expert ID is handled uniformly by the threadgroup and poisons its output slice
rather than indexing the weight tensor. The unreachable K-tail code is removed:
K and N are required to be multiples of 64 in this pilot.

The `switch_glu` helper provides an opt-in call for MLX-LM's evaluated,
affine-quantized SwitchGLU. It changes the gate/up input loading only. It refuses
training and unsupported projection configurations rather than silently changing
semantics. No custom gradient is implemented.

The vendor headers are MIT-licensed MLX code. `UPSTREAM.json` records their
Git blob SHA-1 and SHA-256 hashes. `tools/make_vendor.py` and
`tools/make_kernel.py` reproduce the derived source from the pinned MLX tree.
`reference/kernel-body.diff` is a review diff, not a patch to apply to MLX.

## Expected benefit and costs

Let T be tokens, A selected experts per token, and K the input width. The
materialized activation gather has T*A*K elements. With fp16/bf16 it allocates:

```
2 * T * A * K bytes
```

For T=4096, A=8, K=4096, that is **256 MiB**. The original X is 32 MiB, and a
uint32 row map for 32768 routes is 128 KiB. The prototype eliminates the 256 MiB
activation allocation shared by the two projections. It does **not** eliminate
256 MiB twice, and this is not a measured decrease in process peak memory.
Peak behavior also depends on other live arrays and allocator lifetimes.

The prospective speed gain comes from avoiding this gather allocation/write and
accessing a smaller source working set. The kernel still reads every activation
needed by each projection. Logical traffic is not the same as DRAM traffic:
cache residency affects the realized benefit.

The cost is additional address indirection and live pointer/flag registers.
Irregular source rows can hurt cache behavior, and the added registers can reduce
occupancy. A smaller allocation alone is not proof of a faster operation. These
tradeoffs are why the benchmark includes a matched contiguous JIT control.

This primarily targets multi-token MoE prefill and sufficiently large route
batches. It is not a demonstrated single-token decode optimization, does not
help dense models through this API, and does not change speculation acceptance.
No percentage gain or tokens/second result is claimed.

## Supported pilot contract

- A real M5-or-newer Apple GPU with the pinned MLX architecture gate satisfied,
  macOS 26.2 or newer, and Metal Performance Primitives available to the compiler.
  Future chips still require separate compilation, correctness, and performance
  qualification; passing a name/capability gate does not establish a speedup.
- fp16 or bf16 activation/scale/bias tensors.
- Affine 4-bit or 8-bit weights with group size 64 or 128.
- Transposed expert weights, K and N multiples of 64, K a multiple of group size.
- At least eight routes for the custom-kernel input pointer ABI.
- Large non-64-aligned route counts above 32768 are excluded from this pilot.
  This avoids mixing the experiment with known upstream ragged-route defects.
- The SwitchGLU adapter requires at least 64 routes, no linear projection bias,
  matching gate/up quantization configurations, and `module.training == False`.
- The checked API validates routing values and ordering with a synchronization.
  `validate=False` is for trusted internally generated maps, as in the supplied
  benchmark. Do not present checked-API timing as the unchecked fast-path timing.
- Noncontiguous inputs can be copied to contiguous storage by MLX. “No expanded
  activation gather” does not mean every possible input is entirely copy-free.

## Local verification actually performed

Environment: Linux x86_64, Python 3.13.5, Clang C++ compiler. No MLX native module
or Metal device is present in this runtime.

1. The shared row-address functions were compiled and executed under AddressSanitizer
   and UndefinedBehaviorSanitizer, with warnings treated as errors. The sweep
   compares all 32 lanes' fragment coordinates against the pinned upstream mapping,
   checks coverage, reassembles gathered tiles for repeated/random source rows,
   exercises partial route tiles, invalid indices, and offsets above 4 GiB.
   **Passed.** These are CPU address/layout tests, not a NAX arithmetic test.
2. Eight Python policy/source-structure tests passed. They cover safety admission,
   architecture gating, tile selection, arithmetic accounting, standalone header
   generation, and placement of row-map initialization outside the loops.
3. Python byte compilation passed for the package, tools, native tests, and benchmark.
4. Source-generation reproducibility and C++ optimized-build checks are recorded
   in `results/validation.json`.

**Not performed here:** Metal compilation, M5 kernel execution, GPU memory/shader
validation, native MLX-LM integration tests, native timing, whole-model benchmarks,
full upstream test suites, or pre-commit formatting hooks. `native_not_run.log`
records a skipped suite, not a native pass. Native qualification must not be
reported as complete until the commands below succeed on the target hardware.

## Run on the M5 Max

Use an existing Python environment with MLX and NumPy. For the optional full
SwitchGLU scope, it also needs MLX-LM. For the most controlled comparison, use
MLX built from the pinned commit above. The script records the installed version
and binary/source hashes; it does not install or replace packages.

From this extracted directory:

```bash
./verify_m5.sh --tokens 512,2048,8192 --output results/m5-pair.json
```

This emits a standalone translation unit, compiles 64 dtype/quantization/tile/
alignment/path instantiations with Xcode, links the metallib, and runs native
correctness tests before benchmarking. The runtime tests execute the same pilot
through MLX's JIT, so the offline build alone is not the execution check.
The script stops on a build or test failure. It requires actual M5+ hardware;
it will not turn an unavailable GPU into a successful skipped qualification.

Then measure the full tiny SwitchGLU layer, including routing and unsorting:

```bash
python3 benchmark.py --scope switch --tokens 512,2048,8192 \
  --output results/m5-switch.json
```

Test both distributions and dtypes before choosing any default:

```bash
python3 benchmark.py --dtype bfloat16 --distribution skewed \
  --tokens 512,2048,8192 --output results/m5-bf16-skewed.json
```

The benchmark repeats fixed inputs after warm-up. It measures a warm layer
workload, not an uncached whole-model run or tokens per second.

The benchmark uses three paths:

- `upstream`: one gather shared by both built-in projections, not one per projection.
- `jit_contiguous`: the same gather plus the matched contiguous JIT kernel.
- `jit_indirect`: the original activations plus hoisted row maps, with no gather.

It gates measurement on finite results, bitwise equality of matched JIT paths,
and tolerance-based agreement with upstream. It uses alternating ABBA/BAAB
rounds, A/A checks before and after, and rejects gross run-level timing drift.
The reported confidence interval is a within-run bootstrap over paired rounds,
not a claim about every M5 chip or real application. Separate peak-memory probes
include live outputs and record the active-memory baseline.

A candidate speed flag requires an upstream-relative lower interval bound above
1.03 and a positive isolated indirect-versus-contiguous result, with acceptable
A/A and drift checks. This is an engineering qualification threshold, not a
promised gain. Nothing automatically changes a model's runtime policy.

## Small integration call

After `model.eval()`, a compatible MLX-LM call can be evaluated explicitly:

```python
from mlx_nax_indirect import switch_glu

# Equivalent return shape to switch_mlp(x, expert_indices).
y = switch_glu(
    switch_mlp,
    x,
    expert_indices,
    mode="jit_indirect",
    validate=True,
)
```

Keep the current upstream call as the control. Only disable the synchronizing
routing validation when the routing arrays are generated and trusted internally.
Do not run this custom-kernel path through autodiff.

## Sources and provenance

- https://github.com/ml-explore/mlx/tree/b6368984b8e02a3fb3ee7986846c0fb85e1fccf7
- https://github.com/ml-explore/mlx-lm/blob/32bb4e68791c941db382d6fc8fa5b35ba9f3d98b/mlx_lm/models/switch_layers.py
- https://github.com/ml-explore/mlx/blob/b6368984b8e02a3fb3ee7986846c0fb85e1fccf7/mlx/ops.cpp
- https://github.com/ml-explore/mlx/blob/b6368984b8e02a3fb3ee7986846c0fb85e1fccf7/mlx/backend/metal/kernels/quantized_nax.h
- https://machinelearning.apple.com/research/exploring-llms-mlx-m5

The implementation and verification tooling were produced with AI assistance.
It is an experiment for human review, not a submitted upstream contribution.
