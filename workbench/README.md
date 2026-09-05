# MLX M5 optimization workbench

Development handoff, 2026-09-05. This tree collects the source code, patches,
regression tests, benchmark tools, source provenance, validation logs, and research
report produced in the MLX investigation. It is a development PR in Phil's fork,
not a production release or an upstream performance claim.

## Get this exact working branch

```sh
git clone --branch research/m5-optimization-workbench --single-branch \
  https://github.com/PhilipJohnBasile/mlx-lm.git mlx-m5-workbench
cd mlx-m5-workbench
python3 workbench/verify_inventory.py
```

Continue normal development with `git pull --ff-only`, commits, and pushes on
`research/m5-optimization-workbench`. The PR targets the separate pinned
`research/m5-workbench-base` branch so it does not modify your fork's main or pull
unrelated fork-main changes into this research diff.

## Where everything is

| Work | Location | Handoff status |
|---|---|---|
| Shared GDN preprocessing | `../experiments/gdn_preprocessing/` | Reference, direct, and fused source; Apple shader compilation and hosted Metal component evidence. |
| Single-sort MoE routing | `single-sort/` | Original implementation, MLX-LM patch, tests, and benchmarks. |
| Row-indirect NAX, original | `nax-indirect-original/` | Original prototype, original benchmark, source generators, vendor headers, tests, and evidence preserved. |
| Row-indirect NAX, hardened benchmark | `research-kit/indirect-prototype/` | Preferred starting point for new NAX measurements: benchmark correctness and timing fixes. |
| Workload/memory planner | `research-kit/planner/` | Planner, historical config snapshots, manifests, and tests. |
| Research report | `research/MLX_M5_Dense27B_MoE_Research.pdf` | Original PDF, preserved byte-for-byte. |
| Benchmark-hardening delta | `research-kit/benchmark-hardening.patch` | Exact patch linking the original and revised measurement work. |
| Launcher exit-status fix | `bugfixes/launcher/` | Final patch, original/patched files, before/after tests, verifier, and logs. |
| Earlier launcher iteration | `archive/launcher-v1/` | Earlier delivered version, archived separately rather than silently overwritten. |
| Earlier MLX/JACCL branches | `RELATED_WORK.md` | Pinned references and checkout instructions for the separate core-MLX repository. |

The GDN source is at repository-level `experiments/gdn_preprocessing`, not inside
this directory. The historical READMEs are preserved as evidence; their phrases
such as "not published" describe the original experiment, before this handoff.
This README and `STATUS.md` describe the consolidated checkout.

## Start on the Mac

Use a macOS Python environment with MLX, MLX-LM, and NumPy, plus Xcode's Metal
compiler. The commands below do not install packages, download models, modify
model files, or enable anything globally.

GDN preparation and physical-M5 qualification:

```sh
cd experiments/gdn_preprocessing
python3 tools/qualify_m5.py --tokens 1,8,128,2048 --dtype bfloat16 --batch 1
```

For an explicit layer call, `reference` is the control and `direct`/`fused` select
the candidates. Use separate candidate and control cache objects.

Hardened row-indirect NAX qualification, from the repository root:

```sh
cd workbench/research-kit/indirect-prototype
bash verify_m5.sh --tokens 512,2048,8192 --output results/m5-pair.json
```

Single-sort tests, from the repository root:

```sh
cd workbench/single-sort
python3 tests/test_single_sort_native.py
MLX_MOE_SINGLE_SORT=0 python3 tools/bench_single_sort.py --scope routing --output routing.json
```

The benchmark selects its own control and candidate paths; the environment value
above keeps unrelated production integration off. See the single-sort README
before applying its patch to a separate MLX-LM checkout. Do not blindly stack
patches from independent experiments or change the archived baselines.

## Evidence and limits

GDN's hosted native tests ran on an Apple Paravirtual device, not an identified
physical M5. Single-sort and indirect-NAX work have their own host-only validation
limits. No physical M5 speedup, full pretrained-model parity, production-serving
integration, universal Qwen4/Flash support, or global novelty is asserted.

The requested target includes dense 27B and the newer Flash/Qwen4-capability
models. The inherited configs and adapter are pinned historical inputs, not proof
that every newer model shares their architecture. Verify the actual downloaded
checkpoint's model type, GDN geometry, gating, normalization, convolution state,
and cache semantics before admitting it. See `STATUS.md`.

`MANIFEST.json` records per-file SHA-256 hashes and source provenance. The checker
reads files only. Any recorded result remains associated with the source/version
and hardware that generated it. Running a benchmark can overwrite that experiment's
local output files; retain historical evidence or write new results elsewhere.

AI assistance was used in the research, code, tests, and handoff. Upstream-derived
code retains its accompanying licenses. This is source for human continuation.
