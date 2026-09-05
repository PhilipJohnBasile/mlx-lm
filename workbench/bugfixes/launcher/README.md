# MLX distributed launcher: preserve failure exit status

## Result and scope

This is a tested local patch, not an upstream submission. It fixes the false-success
exit-status bug explicitly reported alongside `ml-explore/mlx#4319`. It does **not**
fix or claim to fix that issue's Thunderbolt RDMA `tbt_post_recv` segmentation fault.
No remote branch was changed, no commit was pushed, and no pull request was opened.

Source baseline: `ml-explore/mlx` main at
`b6368984b8e02a3fb3ee7986846c0fb85e1fccf7`.
The two launcher source files were retrieved through the GitHub connector and their
Git blob SHA-1 hashes were verified byte-for-byte. This bundle contains those two
modules, not a complete checkout or build of MLX.

## Why this bug matters

A worker's failure can be printed as a warning while `mlx.launch` itself exits 0.
A caller that checks only the shell exit status can therefore accept a failed
training, inference, test or benchmark invocation as successful. That is a
high-impact orchestration defect; the upstream issue is labelled `bug` and
`distributed`, not formally labelled `critical`.

The public report describes a rank exiting 255 while the launcher exits 0.
The same status loss was reproduced here with ordinary local child processes,
without requiring the reporter's four-Mac Thunderbolt setup.

## Cause and changes

`_launch_with_io` collected rank statuses but returned no result. The ring, NCCL
and JACCL adapters and `main()` also discarded return values. The MPI launcher
independently discarded `subprocess.run(...).returncode` and swallowed interrupts.

The patch:

- Records the first failure the supervisor observes before it stops other ranks.
- Inspects completed workers even when all workers finish before the first poll.
- Treats a worker that ends without publishing a status as a failure, not success.
- Converts negative subprocess signal statuses to the shell convention, 128 + signal.
- Returns the status through all native backend adapters and the console entry point.
- Returns MPI's status and 130 for an interrupted MPI invocation.

Success remains 0. Output forwarding and the existing peer-termination path are
preserved. The first observed failure is not necessarily the chronologically first
failure if several ranks terminate between polls; it is selected in rank order
within that observation. Peer terminations after that selection cannot replace it.

No numerical kernel, collective implementation or model code is changed. Existing
unrelated launcher issues, such as general remote cleanup and terminal I/O handling,
are not claimed to be fixed.

## Validation completed

Host: Linux x86_64, Python 3.13.5.

| Check | Result |
|---|---|
| Identical new tests on unmodified launcher | 16 test methods; 33 failed assertions/subtests |
| New tests on patched launcher | 16 passed |
| Repeated race-sensitive tests | 15 passed, five repetitions of three cases |
| Python byte compilation | Passed |
| `git diff --check` | Passed |
| Apply patch to verified baseline and compare resulting files | See `evidence/patch-check.log` |

The tests cover exit codes 1, 7 and 255; SIGTERM; a missing executable (127);
thread startup errors; immediate completion; late failure after another rank has
succeeded; stdout/stderr forwarding; peer termination without replacing the original
error; backend return plumbing; MPI return/interrupt handling; and CLI controls.
The peer-termination test also requires both peer processes to acknowledge SIGTERM.

**Validation boundary:** Native `mlx.core` is not installed in this environment.
An empty test-only import placeholder was used because these explicit-backend
launcher tests do not call the numerical engine. The real launcher source,
argument parsing, hostfile code, subprocesses, pipes, supervisor threads and local
signals execute. NCCL/JACCL/MPI adapter tests mock their downstream launch results.
No actual MPI collective, CUDA, Metal, SSH cluster, Thunderbolt fabric, full MLX
suite, or repository-wide pre-commit run was exercised. The format/lint tools are
not installed here. No native/GPU or full-CI claim is made.

## Apply in an MLX checkout

Review the patch first, then from the repository root:

```sh
git apply --check /path/to/mlx-launch-exit-status.patch
git apply /path/to/mlx-launch-exit-status.patch
python -m unittest discover -s python/tests -p test_launch_exit_status.py -v
```

The last command uses the installed MLX import normally. It does not install the
placeholder used for this bundle's isolated validation.

For a small launcher-only reproduction after installing the checkout:

```sh
mlx.launch --backend ring python -c 'raise SystemExit(255)'
echo "$?"
# Before: 0. After: 255.
```

Use the Python executable appropriate for the environment. Backend selection is
explicit so the launched payload does not need a GPU or distributed initialization.

## Re-run this bundle without the native MLX engine

```sh
python verify_launcher_only.py
```

The verifier creates a temporary isolated package, checks baseline source hashes,
runs the same tests against both versions, and writes logs to `verification/`.
It treats the known baseline failures as expected and exits 0 only when the patched
suite passes. Its empty `mlx.core` placeholder exists only in the temporary package;
it is not part of the production patch. This verifies launcher behavior, not MLX
numerical correctness.

## Repository survey and duplicate check

The survey inventoried the 11 public repositories under `ml-explore` and searched
open issue trackers across the organization, with additional core and smaller-repo
queries. This was issue/PR triage, not an exhaustive audit of all source code:

`mlx`, `mlx-lm`, `mlx-examples`, `mlx-swift`, `mlx-swift-lm`,
`mlx-swift-examples`, `mlx-c`, `mlx-data`, `mlx-onnx`, `mlx-framework.org`, `.github`.

Several plausible candidates were excluded rather than duplicated:

| Candidate | Reason excluded |
|---|---|
| MLX #3940, cross-thread stale compile-cache result | Issue already closed |
| MLX #3932, compiled multi-output memory leak | Existing fix PRs #4183 and #4453 |
| MLX #4444, strided singleton-slice gradient corruption | Existing fix PR #4446 |
| mlx-c #104 / MLX #3201, shapeless reductions | Existing follow-up fix PR #3672 |
| MLX #4072, distributed NaN reduction | Existing PR #4073 |
| MLX #4379, CPU variance precision | Existing PR #4387 |

For the selected launcher bug, the issue, its comments, launcher-titled PRs,
`launch exit` PR results, exact issue-number PR search, and `launch "exit status"`
PR results were checked. No matching fix was found. This does not establish that
no private or unindexed work exists, nor that other engineers have tried and failed.

## Primary sources

- Incident and explicit false-success report:
  https://github.com/ml-explore/mlx/issues/4319
- Current source used for the patch:
  https://github.com/ml-explore/mlx/blob/b6368984b8e02a3fb3ee7986846c0fb85e1fccf7/python/mlx/_distributed_utils/launch.py
- Hostfile and logging code used by the tests:
  https://github.com/ml-explore/mlx/blob/b6368984b8e02a3fb3ee7986846c0fb85e1fccf7/python/mlx/_distributed_utils/common.py
- Console entry-point declaration:
  https://github.com/ml-explore/mlx/blob/b6368984b8e02a3fb3ee7986846c0fb85e1fccf7/setup.py
- Contribution policy:
  https://github.com/ml-explore/mlx/blob/b6368984b8e02a3fb3ee7986846c0fb85e1fccf7/CONTRIBUTING.md
- Agent-submission restrictions:
  https://github.com/ml-explore/mlx/blob/b6368984b8e02a3fb3ee7986846c0fb85e1fccf7/AGENTS.md

MLX allows disclosed AI-assisted code but requires contributors to understand their
changes. It prohibits AI-written public posts and agent-created pushes/PRs. This
bundle is code and private verification evidence for Phil's review, not a public
PR description or a commit message.
