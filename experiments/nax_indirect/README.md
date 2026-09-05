# Physical M5 indirect-NAX compiler repair

Explicit research implementation copied from
`workbench/research-kit/indirect-prototype` at handoff commit
`5db58608fa5d21d2613985cefe94a243420ccd63`. The archived directory is unchanged.
No automatic selection or pretrained-model integration is enabled.

## Result

The archived shader failed to compile with Metal 32023.921 (Xcode 27.0,
27A5237l). Its `fragment` and `half` parameter names collide with Metal reserved
names, and a local fragment reference needs the `thread` address space. This
active copy fixes those two headers. Indexing, arithmetic, dimensions,
quantization and correctness tolerances are unchanged. Python files were
formatted for this repository; their non-import ASTs and generated Metal match
the tested versions. `ACTIVE_PROVENANCE.json` identifies the original files.

On the physical Apple M5 Max, 40 GPU cores and 128 GiB, macOS 27.0 build
26A5425a, MLX/MLX-Metal 0.32.2 and Python 3.13.14:

- Metal compilation and linking passed (8 compiler warnings).
- All 9 unchanged native test methods passed, including 4/8-bit quantization,
  bitwise matched-kernel outputs, dense reference tolerances and invalid routes.
- 8 policy tests and 20 benchmark-contract tests passed.
- AddressSanitizer/UndefinedBehaviorSanitizer route checks passed: 11,010,821.
- Both BF16 paired-projection timing cells passed A/A and drift controls.

| Tokens | Upstream / indirect speed | 95% interval | Decision |
| --- | ---: | --- | --- |
| 512 | 0.8780x | 0.8698–0.8865 | Reject: about 14% more latency |
| 2048 | 0.7360x | 0.7329–0.7395 | Reject: about 36% more latency |

These are warm gate/up projection pairs with K=2048, N=512, 256 experts,
top-8 routing, affine 4-bit/group-64 weights. Both paths share one route plan;
the upstream comparator gathers activations once. These results do not measure
complete requests or justify combining this candidate with single-sort routing.
Six-bit quantization is unsupported by this prototype and remains separate work.

## Reproduce

Use the repository's isolated `.venv` and a fresh output directory. Run GPU
workloads sequentially. From the checkout root:

```sh
mkdir work/m5-qualification/nax-NEW
.venv/bin/python experiments/nax_indirect/tools/emit_metal.py \
  work/m5-qualification/nax-NEW/pilot.metal
xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math \
  -c work/m5-qualification/nax-NEW/pilot.metal \
  -o work/m5-qualification/nax-NEW/pilot.air
xcrun -sdk macosx metallib work/m5-qualification/nax-NEW/pilot.air \
  -o work/m5-qualification/nax-NEW/pilot.metallib
.venv/bin/python experiments/nax_indirect/tests/test_native.py --require-metal -v
PYTHONPATH=experiments/nax_indirect .venv/bin/python -m unittest discover \
  -s experiments/nax_indirect/tests -p test_policy.py -v
PYTHONPATH=experiments/nax_indirect .venv/bin/python -m unittest discover \
  -s experiments/nax_indirect/tests -p test_qualification_utils.py -v
.venv/bin/python experiments/nax_indirect/benchmark.py \
  --tokens 512,2048 --k 2048 --n 512 --experts 256 --top-k 8 \
  --bits 4 --group-size 64 --dtype bfloat16 --scope pair \
  --rounds 9 --iterations 8 --output work/m5-qualification/nax-NEW/pair.json
python3 workbench/verify_inventory.py
```

The inherited `verify_m5.sh` writes to a fixed `results` directory. Use the
explicit fresh-output commands above to preserve earlier receipts.

The checked API validates routing synchronously. Timings use internally
constructed, validated maps and exclude validation. Supported native interfaces
remain FP16/BF16, affine 4/8-bit, groups 64/128 and aligned K/N on admitted M5
hardware. Training and unsupported cases raise errors. This device gate is not
a performance qualification. No model weight or installed inference service is
modified. Upstream headers retain their license and provenance metadata.

Compressed raw evidence and hashes accompany this repair in `evidence/20260905`.
The exact pre-format Python and header sources are retained with the receipts.
