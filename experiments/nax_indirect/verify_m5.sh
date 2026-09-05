#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p results
if [[ "$(uname -s)" != Darwin ]]; then
  printf '%s\n' 'Native qualification requires macOS on M5-or-newer hardware.' >&2
  exit 2
fi
python3 tools/emit_metal.py results/pilot.metal
xcrun -sdk macosx metal -std=metal4.0 -fno-fast-math -c results/pilot.metal -o results/pilot.air \
  > results/metal-compile.log 2>&1
xcrun -sdk macosx metallib results/pilot.air -o results/pilot.metallib
python3 tests/test_native.py --require-metal -v 2>&1 | tee results/native-tests.log
python3 benchmark.py "$@"
