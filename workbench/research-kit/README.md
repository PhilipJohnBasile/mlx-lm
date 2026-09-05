# M5+ dense 27B and MoE research kit

This kit improves the **qualification tooling**, not the numerical GPU kernel.
No M5 speedup is claimed. The accompanying research PDF contains the revised
architecture, prior-art review, risk analysis, and native promotion criteria.

## Contents

- `indirect-prototype/`: the original row-indirect NAX prototype with a hardened
  benchmark. Its GPU kernels and format/admission limits are unchanged.
- `benchmark-hardening.patch`: changes for the **original prototype root**, not
  for an upstream MLX checkout. Do not apply it to the already-hardened copy.
- `planner/plan_workloads.py`: a CPU-only manifest generator for the Qwen GDN
  geometry studied here, including dense, routed/shared-expert and GDN shapes.
- `planner/examples/`: labeled extracted config fields and derived manifests.
  These are not byte-for-byte copies of the upstream config files.
- `validation/`: logs from this research pass. No GPU timings are included.
- `SHA256SUMS.json`: integrity hashes for the delivered kit.

## Actual changes

Representative performance routes now use distinct top-k expert IDs per token,
including the skewed case. Repeated experts across different tokens remain valid.
Sampling is synthetic, deterministic and outside timed execution; it does not
attempt to reproduce a learned router. Realized occupancy is recorded.

Matched-path parity now compares payload bits and validates output count, shape,
dtype and finiteness on all three paths. It runs before and after timing. A/A
requires its complete confidence interval within the chosen bounds, and missing
or nonfinite timings cannot pass the candidate gate.

The existing benchmark still measures warm layer workloads. It does not implement
whole-model serving, stochastic-distribution tests, a physical-memory governor,
new dense kernels, GDN fusion, state replay, or a production selector.

## Host verification

Requires Python 3.10+ and NumPy. The C++ check additionally uses Clang with ASan
and UBSan. No package installation is performed by these commands.

```bash
./run_host_checks.sh
```

Executed here: 20 new benchmark-helper tests, 12 new planner tests, 8 existing
policy/source tests, and 11,010,821 existing C++ address/layout checks. These are
Linux host checks, **not** Metal execution or evidence of speed.

## Plan the actual checkpoint first

```bash
python planner/plan_workloads.py /path/to/model/config.json \
  --output results/model-plan.json
```

The default 90,000,000,000-byte budget is a conservative **planning choice**.
Without explicit resident-weight and workspace totals, the limited budget
comparison is unknown. `safe_to_run` is always unknown: the planner neither
observes physical footprint nor authorizes allocations. It models unquantized KV
at the selected element width, not every KV-quantization scheme.

The official dense geometries produce 144 MiB FP32 GDN state and 8 GiB BF16 KV at
128K per request. The selected MoE produces 60 MiB and 2.5 GiB respectively.
The manifests show the formulas' assumptions and omitted memory.

## Native M5 pilot

After review of source, configuration and memory headroom:

```bash
cd indirect-prototype
./verify_m5.sh --tokens 512 --k 2048 --n 512 \
  --experts 256 --top-k 8 --scope pair \
  --output results/m5-moe-pair.json
```

This command requires actual macOS/M5 hardware and the required MLX/Metal build
stack. It compiles and tests before measuring. It intentionally refuses Linux.
The current pilot admits affine 4/8-bit only; planner requests for 6-bit or NVFP4
are future qualification coverage, not newly implemented support.

For independent full-SwitchGLU coverage, use `--scope switch` with MLX-LM
installed, then both dtypes/distributions. That is still a layer test, not a
complete checkpoint benchmark. The known ragged-route restrictions remain.

## Publication and provenance

No remote repository was modified. No PR or new performance claim was published.
Upstream source licenses and the prototype's original provenance are retained.
`UPSTREAM.json` describes the vendored source. AI assistance was used for research,
code changes, host tests and report preparation. Human and native-device review
remain required before upstream submission or runtime enablement.
