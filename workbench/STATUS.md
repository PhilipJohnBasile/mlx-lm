# Status at handoff: 2026-09-05

## Implemented

- Single-sort MoE routing and benchmark/patch bundle.
- Direct/indirect NAX source-row prototype, with a matched contiguous control.
- Correctness and timing hardening for the indirect benchmark.
- Model-specific workload and memory planner.
- GDN preprocessing reference/direct/fused paths, layer adapter, tests and qualification tools.
- Launcher exit-status patch with original and patched baselines and regression evidence.

## Not implemented by this work

- Packed-quantized paired gate/up projections or their fused SwiGLU extension.
- New 6-bit projection kernels.
- New speculative accepted-prefix recovery or draft-tree implementation.
- A production scheduler or automatic fast-path selector.
- A Qwen4/new-Flash model adapter validated against a current pretrained checkpoint.

Those items were research recommendations, not hidden completed code. The report
and prior-art discussion are included so work can continue without re-deriving
the plan. No implementation should be inferred from a roadmap heading.

## What remains to validate on the Mac

1. Pin the installed MLX, MLX-LM, compiler, and exact model revision/quantization.
2. Verify model-specific eligibility, especially for newer Flash variants. Names,
   capability claims, and `qwen4_exp` identifiers are not interchangeable proof of
   layer, cache, or numerical compatibility.
3. Compile and run native correctness tests for each experiment independently.
4. Check next-token state, rejection positions, batching and masking where applicable.
5. Measure A/A noise and paired A/B results on the physical M5, then pretrained
   full-model parity and complete request throughput/latency.
6. Only combine independently validated changes and remeasure interactions.

Reference/default paths remain unchanged. All implemented experimental paths can
be explicitly selected; they are not locked or inaccessible. The workbench is
packaged for development, not silently enabled in a production server.
