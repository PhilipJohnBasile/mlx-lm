#!/usr/bin/env python3
"""Static handoff check for selected native-MTP hardening.

This does not prove runtime correctness. It catches accidental omission of the
specific #1740 behaviors that AirRunner offered to carry into the live #990
path. Run it after integrating or refactoring the hardening.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Marker:
    path: str
    text: str
    purpose: str


MARKERS = (
    Marker(
        "mlx_lm/generate.py",
        "_model_supports_mtp",
        "fail-closed usable-head detection",
    ),
    Marker(
        "mlx_lm/generate.py",
        "stateless logits processors",
        "stateful logits-processor rejection",
    ),
    Marker(
        "mlx_lm/generate.py",
        "yield draft_tok_id, verify_lp",
        "accepted drafts expose target/verifier log probabilities",
    ),
    Marker(
        "mlx_lm/models/cache.py",
        "class MTPPromptCacheState",
        "serializable native-MTP boundary state",
    ),
    Marker(
        "tests/test_mtp.py",
        "test_missing_mtp_weights_disable_head",
        "missing trained-head regression",
    ),
    Marker(
        "tests/test_mtp.py",
        "test_mtp_rejects_unaligned_populated_prompt_cache",
        "fail-closed populated-cache regression",
    ),
    Marker(
        "tests/test_mtp.py",
        "test_mtp_rejects_stateful_logits_processor",
        "stateful processor regression",
    ),
    Marker(
        "tests/test_mtp.py",
        "test_mtp_generate_identity_with_logits_processor",
        "supported processor parity regression",
    ),
    Marker(
        "tests/test_mtp.py",
        "test_mtp_prompt_cache_finalizes_at_length_boundary",
        "length-boundary transaction finalization",
    ),
    Marker(
        "tests/test_mtp.py",
        "test_mtp_prompt_cache_finalizes_on_generator_close",
        "early-close transaction finalization",
    ),
    Marker(
        "tests/test_mtp.py",
        "test_mtp_prompt_cache_reuse_matches_uncached_generation",
        "multi-turn cache reuse parity",
    ),
    Marker(
        "tests/test_mtp.py",
        "test_mtp_prompt_cache_state_round_trip",
        "prompt-cache serialization regression",
    ),
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    missing: list[Marker] = []

    for marker in MARKERS:
        path = root / marker.path
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            missing.append(marker)
            continue
        if marker.text not in content:
            missing.append(marker)

    if not missing:
        print("All native-MTP hardening handoff markers are present.")
        print("Run the focused and full test suites; this static check is not proof.")
        return 0

    print("Missing native-MTP hardening handoff markers:")
    for marker in missing:
        print(f"- {marker.path}: {marker.text!r} ({marker.purpose})")
    print(
        "\nA renamed or refactored implementation may be correct. In that case, "
        "replace this marker with an equivalent focused regression rather than "
        "silently deleting the behavior."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
