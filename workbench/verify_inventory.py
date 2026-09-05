#!/usr/bin/env python3
"""Read-only verification of the imported research files; no MLX import required."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "workbench" / "MANIFEST.json").read_text())
errors = []
for item in manifest["files"]:
    path = ROOT / item["path"]
    try:
        data = path.read_bytes()
    except OSError as exc:
        errors.append(f"{item['path']}: {exc}")
        continue
    if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
        errors.append(f"{item['path']}: content differs from the handoff snapshot")
if errors:
    print("\n".join(errors), file=sys.stderr)
    sys.exit(1)
print(f"Verified {len(manifest['files'])} handoff files. No GPU or performance test was run.")
