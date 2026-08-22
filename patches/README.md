# Native MTP hardening patches

These patches are deliberately small and apply to AirRunner's inspected
`feat/mtp-native` head:

```text
e8ceeccf118d4a089e22431f28f89d5ffe7d9d2b
```

They do **not** choose the disputed `ArraysCache` / `make_draft_model`
architecture and do not implement transactional prompt-cache reuse. They only
package the two semantics that are independent of that decision:

1. accepted drafts expose target/verifier log probabilities;
2. unknown stateful `logits_processors` fail closed.

## Apply

From a clean checkout of the target branch:

```bash
git am \
  patches/0001-mtp-accepted-token-target-logprobs.patch \
  patches/0002-mtp-reject-stateful-logits-processors.patch
```

When applying from another clone, fetch or copy the patch files first. To apply
without creating commits:

```bash
git apply patches/0001-mtp-accepted-token-target-logprobs.patch
git apply patches/0002-mtp-reject-stateful-logits-processors.patch
```

## Validate

The tests-first handoff file intentionally contains additional red tests for
all promised semantics:

```bash
python -m pytest -q tests/test_mtp_hardening_handoff.py
```

After applying these two patches, the accepted-logprob and stateful-processor
tests should move toward green. The transactional-cache and usable-head tests
remain expected failures until their architecture is integrated.

Run the existing focused tests as well:

```bash
python -m pytest -q tests/test_mtp.py tests/test_mtp_hardening_handoff.py
```

The static marker check is a supplemental omission detector, not a substitute
for runtime tests:

```bash
python tools/check_mtp_hardening_markers.py
```

## Attribution

The reference implementation is ml-explore/mlx-lm#990 by AirRunner. These two
fixes and the broader transactional-cache hardening were preserved from closed
ml-explore/mlx-lm#1740 at:

```text
PhilipJohnBasile/mlx-lm@bc1d11414c12372cdf76538e09ce1fbc54b7fc7b
```

Preserve authorship when copying code directly, or reference #1740 and the
source commit in the integrating commit history.
