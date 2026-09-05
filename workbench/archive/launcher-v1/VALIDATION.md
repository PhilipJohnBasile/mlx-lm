# MLX distributed launcher: exit-status fix

This is a local code patch and validation record, not a submitted pull request.

## Scope

Repository: https://github.com/ml-explore/mlx
Base commit: `b6368984b8e02a3fb3ee7986846c0fb85e1fccf7`.

The selected defect is `mlx.launch` reporting success after a worker fails.
It is separately reported in https://github.com/ml-explore/mlx/issues/4319:
a rank exited with 255 but the launcher exited with 0. That issue is principally
about a JACCL / Thunderbolt RDMA crash. **This patch fixes the launcher exit
status, not that underlying crash, and must not be used to close the whole issue.**

The issue and comments were read and PR searches checked the issue number,
`_launch_with_io`, launch/exit-code terms, and exit-related titles. No matching
working PR was found for this specific status-propagation defect. That does
not prove that nobody has implemented a fix elsewhere. This is a high-impact
automation/reliability issue; no upstream "critical" severity label was asserted.

## Changes

Only one production file changes: `python/mlx/_distributed_utils/launch.py`.

- Return the observed worker failure through the shared launcher, each backend
  launcher, and `main()` to the existing console-script entry point.
- Keep the first failure observed by the coordinator so peer cleanup cannot
  replace the triggering rank's error with a termination status.
- Take one worker-liveness snapshot per poll. A rank finishing between two
  liveness checks must not turn the final failed job into a success.
- Convert negative subprocess signal statuses to `128 + signal`.
- Treat a stopped worker with no recorded process status as failure (1).
- Forward MPI's subprocess return code and return 130 for its caught interrupt.

Successful jobs remain successful. The patch does not change tensor operations,
communication kernels, model execution, native exception handling, or SSH cleanup.
The triggering failure is the first one *observed*, not a guarantee of temporal
ordering between simultaneous rank failures.

## Validation performed

Environment: Linux x86_64, Python 3.13.5.

The same regression file ran against both source versions:

| Source | Result |
| --- | --- |
| Exact pinned upstream | 13 test methods, 38 regression failures |
| Patched | 13 test methods, zero failures or errors |

There are 46 scenarios when parameterized subtests are counted separately.
The tests cover local launches through ring, NCCL, JACCL and JACCL-ring routing;
normal and nonzero exits; SIGTERM and SIGKILL; failure before the first poll;
the last-rank polling race; worker-start exceptions; peer cancellation;
early successful peers; successful multi-rank launches; MPI return codes and
its interrupt handler; `--print-python`; and invalid-backend errors.

A further shell-gating reproduction used a worker exiting with 42 followed by
`&& touch success-marker`. On upstream the marker was incorrectly created and
the shell returned 0. With the patch the marker was absent and the shell
returned 42. See `validation/shell-gate-repro.json`.

Python byte compilation and `git diff --check` passed. The patch was applied
with `git apply --check` and `git apply` to the exact original source, and the
resulting files were byte-compared with the files that passed the tests.

The two upstream source files were retrieved from GitHub and their Git blob
hashes verified before testing. This was a focused source snapshot, not a full
repository build. See `provenance.json`.

## Important test limits

MLX's native extension is not installed in this container. For these launcher
regressions only, an isolated import shim supplied `mlx.core.cuda.is_available()`
as false. No tensor operation was provided or emulated. Real Python workers,
bash processes, pipes, operating-system signals, and process exit statuses were
used. The shim is included under `validation/` for transparency and is **not**
part of the patch or production code.

The MPI tests use a small local executable standing in for `mpirun`, plus an
injected `KeyboardInterrupt` test. Backend routing tests do not initialize or
validate distributed collectives. No actual remote SSH connection, MPI cluster,
Metal/CUDA execution, or JACCL/Thunderbolt hardware was tested. The full MLX test
suite and pre-commit/Black/isort checks were not run; those formatters were not
installed in the container.

## Apply and run in an MLX development checkout

Use the Python environment in which that checkout's MLX development build is
available. First confirm the target file still matches the intended base.

```sh
git apply --check /path/to/mlx-launch-exit-status.patch
git apply /path/to/mlx-launch-exit-status.patch
PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" \
  python -m unittest discover -s python/tests -p test_distributed_launch.py -v
```

Do not install the validation import shim in a normal MLX environment. The
upstream-style tests import the actual MLX package when run in that checkout.

Before submission, complete the repository's formatting and relevant native
checks and review the code. The repository explicitly prohibits automated
pushes, PR creation, and AI-authored public posts. No branch was pushed, no PR
was opened, and no comment or commit message was posted on the user's behalf.

Policies inspected at the pinned commit:
- https://github.com/ml-explore/mlx/blob/b6368984b8e02a3fb3ee7986846c0fb85e1fccf7/AGENTS.md
- https://github.com/ml-explore/mlx/blob/b6368984b8e02a3fb3ee7986846c0fb85e1fccf7/CONTRIBUTING.md

## Survey coverage

The public repository inventory and open issue/PR reports were surveyed for
all 11 repositories returned under `ml-explore`: mlx, mlx-lm, mlx-examples,
mlx-swift, mlx-swift-lm, mlx-swift-examples, mlx-data, mlx-c, mlx-onnx,
mlx-framework.org and .github. This was an issue/PR survey, not an exhaustive
source-code audit of every repository or every issue. Existing fix candidates
for the server-dead-thread and multi-output-array leak reports were excluded.

AI assistance was used for the survey, source analysis, patch, tests and these
local validation notes. This document is not a PR description or a public reply.
