#!/usr/bin/env python3
"""Apply the two architecture-independent native-MTP hardening fixes.

This script is scoped to the inspected ``feat/mtp-native`` source shape at
``e8ceeccf118d4a089e22431f28f89d5ffe7d9d2b``. It intentionally does not
implement transactional prompt-cache reuse or select the final
``make_draft_model`` architecture.

The changes are:

1. accepted MTP drafts expose the verifier/target log-probability vector;
2. opaque stateful logits-processor objects fail closed.

Every edit is guarded by exact source anchors. A changed or partially applied
source tree fails without writing anything.
"""

from __future__ import annotations

import argparse
from pathlib import Path


IMPORT_OLD = "import copy\nimport json\n"
IMPORT_NEW = "import copy\nimport inspect\nimport json\n"

MTP_ENTRY_OLD = """    validate_prompt_and_embeddings(model, prompt, input_embeddings)

    y = prompt.astype(mx.uint32)
"""

MTP_ENTRY_NEW = """    validate_prompt_and_embeddings(model, prompt, input_embeddings)

    if logits_processors:
        unsafe_processors = [
            type(processor).__name__
            for processor in logits_processors
            if not inspect.isfunction(processor)
            and not isinstance(processor, partial)
            and not getattr(processor, "is_stateless", False)
        ]
        if unsafe_processors:
            names = ", ".join(unsafe_processors)
            raise ValueError(
                "Native MTP currently supports only stateless logits processors. "
                "Use plain functions, functools.partial, or set is_stateless=True "
                f"on a safe callable. Unsupported processor(s): {names}."
            )

    y = prompt.astype(mx.uint32)
"""

ACCEPT_OLD = "                yield draft_tok_id, draft_lp, True\n"
ACCEPT_NEW = "                yield draft_tok_id, verify_lp, True\n"


class PatchStateError(RuntimeError):
    """Raised when the target source is neither clean nor fully patched."""


def _count(text: str, marker: str) -> int:
    return text.count(marker)


def _state(text: str) -> str:
    old_counts = (
        _count(text, IMPORT_OLD),
        _count(text, MTP_ENTRY_OLD),
        _count(text, ACCEPT_OLD),
    )
    new_counts = (
        _count(text, IMPORT_NEW),
        _count(text, MTP_ENTRY_NEW),
        _count(text, ACCEPT_NEW),
    )

    if old_counts == (1, 1, 1) and new_counts == (0, 0, 0):
        return "clean"
    if old_counts == (0, 0, 0) and new_counts == (1, 1, 1):
        return "applied"

    raise PatchStateError(
        "generate.py is partially applied or no longer matches the inspected "
        "source anchors; refusing to edit. "
        f"old_counts={old_counts}, new_counts={new_counts}"
    )


def _apply(text: str) -> str:
    state = _state(text)
    if state == "applied":
        return text
    return (
        text.replace(IMPORT_OLD, IMPORT_NEW, 1)
        .replace(MTP_ENTRY_OLD, MTP_ENTRY_NEW, 1)
        .replace(ACCEPT_OLD, ACCEPT_NEW, 1)
    )


def _reverse(text: str) -> str:
    state = _state(text)
    if state == "clean":
        return text
    return (
        text.replace(IMPORT_NEW, IMPORT_OLD, 1)
        .replace(MTP_ENTRY_NEW, MTP_ENTRY_OLD, 1)
        .replace(ACCEPT_NEW, ACCEPT_OLD, 1)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("check", "apply", "reverse", "verify-applied"),
        nargs="?",
        default="check",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("mlx_lm/generate.py"),
        help="Path to generate.py (default: mlx_lm/generate.py)",
    )
    args = parser.parse_args()

    path = args.path
    text = path.read_text(encoding="utf-8")
    state = _state(text)

    if args.action == "check":
        print(f"{path}: {state}; anchors are internally consistent")
        return 0

    if args.action == "verify-applied":
        if state != "applied":
            raise PatchStateError(f"{path}: expected applied state, found {state}")
        print(f"{path}: hardening is fully applied")
        return 0

    updated = _apply(text) if args.action == "apply" else _reverse(text)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        print(f"{path}: {args.action} completed")
    else:
        print(f"{path}: already in requested state")

    expected = "applied" if args.action == "apply" else "clean"
    actual = _state(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise PatchStateError(f"post-write state is {actual}; expected {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
