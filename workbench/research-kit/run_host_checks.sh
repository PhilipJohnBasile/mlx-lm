#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p validation
python3 -m unittest discover -s indirect-prototype/tests -p test_qualification_utils.py -v \
  2>&1 | tee validation/qualification-host-tests.log
python3 -m unittest discover -s planner/tests -v \
  2>&1 | tee validation/planner-host-tests.log
(cd indirect-prototype && python3 -m unittest discover -s tests -p test_policy.py -v) \
  2>&1 | tee validation/existing-policy-tests.log
python3 -m compileall -q indirect-prototype planner
build_dir="$(mktemp -d)"
trap 'rm -rf -- "$build_dir"' EXIT
clang++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -Wall -Wextra -Werror -Iindirect-prototype \
  indirect-prototype/tests/test_route_address.cpp -o "$build_dir/route-test"
"$build_dir/route-test" 2>&1 | tee validation/route-address-sanitizers.log
printf '%s\n' 'Host checks passed; no native Metal test or performance measurement was run.'
