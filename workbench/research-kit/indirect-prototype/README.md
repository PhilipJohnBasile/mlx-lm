# Row-indirect NAX prototype: qualification revision 2

**Experimental and opt-in. No native M5 validation or speedup has been obtained in
this research environment.** This revision changes benchmark methodology and
correctness checks, not GPU computation. See the kit's root README and research
report for findings, risks, and next implementation units.

The prototype uses the original token-order activation matrix and a sorted
source-row map for gate/up projections. It avoids an expanded activation gather.
The matched contiguous JIT path still consumes one gathered matrix shared by both
projections. The upstream comparator also performs only one gather.

## Unchanged contract

MLX reference: `b6368984b8e02a3fb3ee7986846c0fb85e1fccf7`.
MLX-LM reference: `32bb4e68791c941db382d6fc8fa5b35ba9f3d98b`.

The pilot requires real M5-class Metal support, FP16/BF16, affine 4/8-bit weights,
group sizes 64/128, transposed weights, aligned K/N, and supported route counts.
Training and unsupported configurations are rejected. Six-bit and NVFP4 remain
unsupported. Device-name/capability gating is not proof of future-chip speed.

The checked API synchronizes to validate routing. The benchmark uses internally
generated, validated maps and excludes that validation from timed calls. Do not
compare checked and unchecked API latency as though they were the same contract.
All original kernel `.h` and `.metal` files are unchanged; their SHA checks are in
`../validation/validation.json`. The inherited inactive-SIMD-group behavior and
ragged-route limits still require native investigation.

## What changed

`qualification_utils.py` supplies synthetic top-k sampling without replacement,
realized occupancy summaries, payload-bit comparisons, structural parity checks,
and stricter A/A validity checks. The benchmark verifies outputs before and after
timing. Native matched-kernel tests also use the payload-bit comparator.

Pair scope times gather plus both projections but uses a precomputed route plan.
Switch scope includes route sorting and output unsorting. Both are warm layer
workloads. Neither establishes cold-weight behavior or whole-model throughput.

Run the root `run_host_checks.sh` for portable checks. On a reviewed M5 environment,
run `./verify_m5.sh` with explicit workload dimensions. The script stops if native
compilation or required tests fail; a missing GPU is not treated as a pass.

This directory is already patched. The separate `benchmark-hardening.patch` is
provided only for users retaining the original prototype directory.

## Provenance

Vendored MLX headers retain `LICENSE.upstream` and `UPSTREAM.json`.
`reference/kernel-body.diff` describes the original derived kernel, not this
benchmark-only revision. `tools/make_vendor.py` and `tools/make_kernel.py` retain
the original generation workflow. No claims from other projects' benchmark
results are transferred to this prototype.
