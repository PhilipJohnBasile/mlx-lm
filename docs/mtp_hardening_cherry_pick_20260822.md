# Cherry-pick procedure for the focused native-MTP hardening

This procedure applies only to the two architecture-independent fixes prepared
against AirRunner's inspected `feat/mtp-native` head:

```text
e8ceeccf118d4a089e22431f28f89d5ffe7d9d2b
```

It does not implement transactional prompt-cache reuse and does not choose the
final `make_draft_model` architecture.

## Preferred: code-only branch

A GitHub Actions materialization job publishes this branch from the exact base:

```text
PhilipJohnBasile/mlx-lm:mtp-native-hardening-applied-20260822
```

From AirRunner's checkout:

```bash
git switch feat/mtp-native
git fetch https://github.com/PhilipJohnBasile/mlx-lm.git \
  mtp-native-hardening-applied-20260822

# Confirm both sides share the inspected base before applying anything.
test "$(git merge-base HEAD FETCH_HEAD)" = \
  e8ceeccf118d4a089e22431f28f89d5ffe7d9d2b

# Apply the two commits in authored order.
git cherry-pick $(
  git rev-list --reverse \
    e8ceeccf118d4a089e22431f28f89d5ffe7d9d2b..FETCH_HEAD
)
```

Expected commit subjects:

```text
fix(mtp): report target logprobs for accepted drafts
fix(mtp): fail closed on stateful logits processors
```

## Alternative: patch files

```bash
git switch feat/mtp-native
git am \
  /path/to/0001-mtp-accepted-token-target-logprobs.patch \
  /path/to/0002-mtp-reject-stateful-logits-processors.patch
```

The patches are stored under:

```text
PhilipJohnBasile/mlx-lm:archive-mtp-native-hardening-handoff-20260822-r2/patches
```

## Alternative: guarded applicator

The applicator makes both source edits in one working-tree change and refuses
partial or changed source anchors:

```bash
python tools/apply_mtp_hardening_patches.py check
python tools/apply_mtp_hardening_patches.py apply
python tools/apply_mtp_hardening_patches.py verify-applied
```

The patch or cherry-pick path is preferable because it retains two independent
commit records and attribution.

## Verification

At minimum:

```bash
python -m py_compile mlx_lm/generate.py
git diff --check e8ceeccf118d4a089e22431f28f89d5ffe7d9d2b..HEAD
grep -F 'yield draft_tok_id, verify_lp, True' mlx_lm/generate.py
grep -F 'supports only stateless logits processors' mlx_lm/generate.py
```

Then run the focused tests appropriate to the final branch architecture. The
handoff acceptance suite includes a direct target-distribution assertion and a
stateful-processor rejection assertion, but also includes intentionally red
transactional-cache tests that should not be treated as failures of these two
focused commits.

## Abort / recovery

For a cherry-pick conflict:

```bash
git cherry-pick --abort
```

For a patch-series conflict:

```bash
git am --abort
```

A conflict means the reviewed source moved beyond the exact anchors. Reconcile
the semantics in the new architecture rather than forcing the old textual
patch.
