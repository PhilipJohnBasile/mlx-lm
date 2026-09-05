"""Compare unchanged model inference with shared immutable BPE request metadata."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m5_inference"))
from benchmark_requests import main
from shared_bpe import make_selector

if __name__ == "__main__":
    main(
        modes=("shared-bpe",),
        require_bitwise=True,
        make_selector=make_selector,
        extra_sources=tuple(Path(__file__).resolve().parent.glob("*.py")),
    )
