#!/usr/bin/env python3
"""Run before/after launcher tests without loading the MLX numerical engine.

The tests execute the real launcher source and real local child processes.
A test-only, empty mlx.core module satisfies an unused import. Every launcher
invocation selects its backend explicitly; no MLX array, GPU or collective
operation is implemented or exercised here. This is not full MLX validation.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAUNCH = Path("python/mlx/_distributed_utils/launch.py")
COMMON = Path("python/mlx/_distributed_utils/common.py")
EXPECTED_BLOBS = {
    LAUNCH: "0e661e358ffb13fd8b110145cc73c7e46554eac8",
    COMMON: "6289da0c468d75ab3245e15f59f34f1d7793aab2",
}


def blob_sha(path):
    content = path.read_bytes()
    header = b"blob " + str(len(content)).encode() + b"\0"
    return hashlib.sha1(header + content).hexdigest()


def run_variant(variant, log_dir):
    with tempfile.TemporaryDirectory(prefix="mlx-launch-test-") as tmp:
        work = Path(tmp)
        shutil.copytree(ROOT / variant / "python", work / "python")
        tests = work / "python/tests"
        tests.mkdir(exist_ok=True)
        shutil.copyfile(
            ROOT / "patched/python/tests/test_launch_exit_status.py",
            tests / "test_launch_exit_status.py",
        )
        package = work / "python/mlx"
        (package / "__init__.py").write_text("")
        (package / "_distributed_utils/__init__.py").write_text("")
        (package / "core.py").write_text(
            '"""Test-only placeholder; no native MLX operations are provided."""\n'
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(work / "python")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(tests),
                "-p",
                "test_launch_exit_status.py",
                "-v",
            ],
            cwd=work,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
        )
        (log_dir / f"{variant}.log").write_text(result.stdout)
        print(f"{variant}: process exit {result.returncode}")
        print("\n".join(result.stdout.strip().splitlines()[-4:]))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "verification")
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("These launcher tests require POSIX pipes and signals")
    for path, expected in EXPECTED_BLOBS.items():
        actual = blob_sha(ROOT / "baseline" / path)
        if actual != expected:
            parser.error(f"Baseline source hash mismatch for {path}: {actual}")
    log_dir = args.output_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    print("Launcher-only validation; no native MLX, CUDA, Metal or RDMA execution.")
    before = run_variant("baseline", log_dir)
    after = run_variant("patched", log_dir)
    expected_failure = (
        before.returncode == 1
        and "Ran 16 tests" in before.stdout
        and "FAILED (failures=33)" in before.stdout
        and "ERROR:" not in before.stdout
    )
    success = (
        expected_failure
        and after.returncode == 0
        and "Ran 16 tests" in after.stdout
        and after.stdout.rstrip().endswith("OK")
    )
    print("Before/after verification:", "PASS" if success else "FAIL")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
