from __future__ import annotations

import subprocess
from pathlib import Path


BASE = "083f581f048b1b4460f166b31ec18dc6ec000e17"
OUT = Path("../lab/downstream/vllm-metal-610/stack")
PHASES = (
    {
        "slug": "hybrid-gdn-speculative-state-chain",
        "title": "Hybrid GDN speculative state-chain planning",
        "summary": (
            "Reserve scheduler-owned Mamba/GDN speculative blocks, expose a "
            "per-request state-chain contract, and keep multi-token prefill on "
            "the ordinary path."
        ),
        "tests": (
            "tests/attention/test_align_gdn_state_manager.py",
            "tests/test_turboquant_hybrid_sizing.py",
            "tests/test_platform.py",
        ),
        "review": (
            "vllm_metal/attention/context.py",
            "vllm_metal/attention/state/align.py",
            "vllm_metal/attention/runtime/hybrid.py",
            "vllm_metal/v1/cache_policy.py",
        ),
    },
    {
        "slug": "gdn-per-token-state-snapshots",
        "title": "Per-token GDN convolution and recurrent snapshots",
        "summary": (
            "Produce an observable convolution and recurrent-state checkpoint "
            "after every token in a speculative verification window."
        ),
        "tests": (
            "tests/test_gdn_lazy_wrapper.py",
            "tests/attention/test_align_gdn_state_manager.py",
        ),
        "review": (
            "vllm_metal/attention/impls/linear.py",
            "tests/test_gdn_lazy_wrapper.py",
        ),
    },
    {
        "slug": "verifier-promotes-gdn-state",
        "title": "Verifier-selected GDN state promotion",
        "summary": (
            "Promote the recurrent/conv checkpoint selected by the verifier "
            "before scheduler length advancement, block reuse, or preemption."
        ),
        "tests": (
            "tests/test_v1_model_runner_generate.py",
            "tests/attention/test_align_gdn_state_manager.py",
        ),
        "review": (
            "vllm_metal/v1/model_runner.py",
            "vllm_metal/attention/runtime/hybrid.py",
            "vllm_metal/attention/state/align.py",
        ),
    },
    {
        "slug": "native-qwen-mtp-proposer",
        "title": "Native Qwen MTP proposer and hidden-state contract",
        "summary": (
            "Use the trained Qwen MTP head, preserve pre-final-norm target "
            "hidden states, handle chunked prefill, and keep unsupported prefix "
            "hits fail-closed."
        ),
        "tests": (
            "tests/test_qwen_native_mtp_proposer.py",
            "tests/test_v1_model_runner_generate.py",
        ),
        "review": (
            "vllm_metal/v1/model_adapter.py",
            "vllm_metal/v1/proposer.py",
            "vllm_metal/v1/model_runner.py",
        ),
    },
    {
        "slug": "paged-qwen-mtp-prefix-transaction",
        "title": "Scheduler-owned paged Qwen MTP prefix transaction",
        "summary": (
            "Add a distinct EAGLE/MTP cache group, target-boundary hidden-state "
            "shadow, atomic warm-prefix adoption, and the guarded hybrid "
            "prefix-cache plus MTP serving path."
        ),
        "tests": (
            "tests/test_qwen_native_mtp_proposer.py",
            "tests/test_qwen_mtp_paged_cache.py",
            "tests/test_spec_decode_metadata.py",
            "tests/test_paged_prefix_caching.py",
        ),
        "review": (
            "vllm_metal/v1/qwen_mtp_paged.py",
            "vllm_metal/v1/cache_policy.py",
            "vllm_metal/attention/runtime/hybrid.py",
            "vllm_metal/v1/proposer.py",
            "vllm_metal/v1/model_runner.py",
            "vllm_metal/v1/spec_decode.py",
            "vllm_metal/platform.py",
        ),
    },
)


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


commits = run("git", "rev-list", "--reverse", f"{BASE}..HEAD").splitlines()
if len(commits) != len(PHASES):
    raise RuntimeError(
        f"expected {len(PHASES)} phase commits after {BASE}, found {len(commits)}"
    )

OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.glob("*.patch"):
    old.unlink()
for old in OUT.glob("PR-*.md"):
    old.unlink()

series: list[str] = []
rows: list[str] = []
for index, (commit, phase) in enumerate(zip(commits, PHASES, strict=True), start=1):
    filename = f"{index:04d}-{phase['slug']}.patch"
    patch = subprocess.check_output(
        ["git", "format-patch", "-1", "--stdout", "--no-signature", commit],
        text=True,
    )
    (OUT / filename).write_text(patch)
    series.append(filename)

    subject = run("git", "show", "-s", "--format=%s", commit)
    parent = run("git", "rev-parse", f"{commit}^")
    tests = "\n".join(f"- `{test}`" for test in phase["tests"])
    review = "\n".join(f"- `{path}`" for path in phase["review"])
    dependency = "Pinned upstream" if index == 1 else f"Phase {index - 1}"
    body = f"""# PR {index}: {phase['title']}

## Dependency

- Base: **{dependency}**
- Parent commit: `{parent}`
- Phase commit: `{commit}`
- Commit subject: `{subject}`

## Scope

{phase['summary']}

## Validation

This phase was committed only after formatting, linting, `git diff --check`, and
its focused macOS/Apple-Silicon test gate passed.

{tests}

## Suggested review order

{review}

## Apply

```bash
git am {filename}
```

This patch is intentionally dependency-scoped. It does not include later phases.
"""
    (OUT / f"PR-{index:02d}.md").write_text(body)
    rows.append(
        f"| {index} | [{phase['title']}](PR-{index:02d}.md) | "
        f"`{commit[:12]}` | `{filename}` |"
    )

(OUT / "series").write_text("\n".join(series) + "\n")
stack_doc = f"""# vLLM Metal #610 merge-ready stack

This directory is generated from five tested commits based on:

`vllm-project/vllm-metal@{BASE}`

Each patch contains exactly one phase and can be reviewed independently. Apply
in the order listed below; phases 1-4 are complete prerequisites and phase 5 is
the final scheduler-owned MTP prefix-cache transaction.

| Phase | Review document | Commit | Patch |
|---:|---|---|---|
{chr(10).join(rows)}

## Apply the complete series

```bash
git checkout {BASE}
while read patch; do
  git am "/path/to/stack/$patch"
done < /path/to/stack/series
```

## Merge order

1. Phase 1 establishes scheduler memory and state-chain metadata.
2. Phase 2 makes every speculative GDN state observable.
3. Phase 3 commits only the verifier-selected state.
4. Phase 4 connects the trained native Qwen MTP head.
5. Phase 5 adds scheduler-owned MTP KV and safe warm-prefix adoption.

The stack deliberately keeps unsupported configurations fail-closed. Real-model
benchmark claims require the separate Qwen checkpoint integration gate; the
unit and scheduler tests here establish the cache/state invariants.
"""
(OUT / "STACK.md").write_text(stack_doc)

print(f"Exported {len(series)} merge-ready patches to {OUT}")
