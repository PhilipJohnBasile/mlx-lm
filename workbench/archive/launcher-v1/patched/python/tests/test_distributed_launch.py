import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx._distributed_utils import launch
from mlx._distributed_utils.common import Host


@unittest.skipUnless(os.name == "posix", "The launcher uses POSIX pipes and bash")
class TestDistributedLaunch(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mlx-launch-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = os.environ | {"TMPDIR": str(self.root)}

    def run_launch(self, code, *, backend="ring", ranks=1, setup="", options=None):
        script = self.root / "rank.py"
        script.write_text(textwrap.dedent(code))
        hostfile = self.root / "hosts.json"
        hostfile.write_text(
            json.dumps(
                {
                    "hosts": [
                        {
                            "ssh": "127.0.0.1",
                            "ips": ["127.0.0.1"],
                            "rdma": [
                                None if i == j else "rdma_test" for j in range(ranks)
                            ],
                        }
                        for i in range(ranks)
                    ]
                }
            )
        )
        wrapper = (
            "import sys\n"
            "from mlx._distributed_utils import launch\n"
            + textwrap.dedent(setup)
            + "\nraise SystemExit(launch.main())\n"
        )
        args = [sys.executable, "-c", wrapper, "--hostfile", str(hostfile)]
        if backend is not None:
            args += ["--backend", backend]
        args += options or []
        args += ["--", sys.executable, str(script)]
        p = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
            start_new_session=True,
        )
        try:
            stdout, stderr = p.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            stdout, stderr = p.communicate(timeout=5)
            self.fail(f"Launcher did not exit\n{stdout}\n{stderr}")
        return subprocess.CompletedProcess(args, p.returncode, stdout, stderr)

    def assert_exit(self, result, expected):
        self.assertEqual(
            result.returncode, expected, msg=result.stdout + result.stderr
        )

    def test_local_rank_exit_codes(self):
        # These test launch routing, not GPU or distributed collectives.
        for backend in ("ring", "nccl", "jaccl", "jaccl-ring"):
            for code in (0, 1, 7, 42, 255):
                with self.subTest(backend=backend, code=code):
                    result = self.run_launch(
                        f"import sys; sys.exit({code})", backend=backend
                    )
                    self.assert_exit(result, code)

    def test_default_backend_exit_code(self):
        self.assert_exit(self.run_launch("raise SystemExit(42)", backend=None), 42)

    def test_signal_exit_codes(self):
        for backend in ("ring", "nccl", "jaccl", "jaccl-ring"):
            for sig in (signal.SIGTERM, signal.SIGKILL):
                with self.subTest(backend=backend, signal=sig):
                    result = self.run_launch(
                        f"import os; os.kill(os.getpid(), {int(sig)})",
                        backend=backend,
                    )
                    self.assert_exit(result, 128 + sig)

    def test_failure_before_first_poll(self):
        self.assert_exit(
            self.run_launch(
                "raise SystemExit(255)",
                setup="""
                    class JoinedThread(launch.threading.Thread):
                        def start(self):
                            super().start()
                            self.join()

                    launch.threading.Thread = JoinedThread
                """,
            ),
            255,
        )

    def test_last_rank_finishes_during_poll(self):
        self.assert_exit(
            self.run_launch(
                "raise SystemExit(42)",
                setup="""
                    class FinishingThread(launch.threading.Thread):
                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            self.polls = 0

                        def start(self):
                            super().start()
                            self.join()

                        def is_alive(self):
                            self.polls += 1
                            return self.polls == 1

                    launch.threading.Thread = FinishingThread
                """,
            ),
            42,
        )

    def test_worker_start_failure(self):
        result = self.run_launch(
            "raise SystemExit(0)",
            setup="""
                def fail_to_start(*args, **kwargs):
                    raise OSError("injected worker start failure")

                launch.Popen = fail_to_start
            """,
        )
        self.assertIn("injected worker start failure", result.stderr)
        self.assert_exit(result, 1)

    def test_failure_survives_peer_cancellation(self):
        for failing_rank in (0, 1):
            with self.subTest(failing_rank=failing_rank):
                for pidfile in self.root.glob("ready-*"):
                    pidfile.unlink()
                result = self.run_launch(
                    f"""
                    import os
                    import time
                    from pathlib import Path

                    root = Path(os.environ["TMPDIR"])
                    rank = int(os.environ["MLX_RANK"])
                    (root / f"ready-{{rank}}").write_text(str(os.getpid()))
                    deadline = time.monotonic() + 5
                    while not all((root / f"ready-{{i}}").exists() for i in range(2)):
                        if time.monotonic() > deadline:
                            raise SystemExit(99)
                        time.sleep(0.01)
                    if rank == {failing_rank}:
                        raise SystemExit(23)
                    time.sleep(60)
                    """,
                    ranks=2,
                )
                self.assert_exit(result, 23)
                for rank in range(2):
                    pidfile = self.root / f"ready-{rank}"
                    pid = int(pidfile.read_text())
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
                    pidfile.unlink()

    def test_failure_after_peer_success(self):
        self.assert_exit(
            self.run_launch(
                """
                import os
                import time

                if int(os.environ["MLX_RANK"]) == 1:
                    time.sleep(0.2)
                    raise SystemExit(42)
                """,
                ranks=2,
            ),
            42,
        )

    def test_successful_ranks(self):
        self.assert_exit(
            self.run_launch(
                """
                import os
                import time

                time.sleep(0.1 * int(os.environ["MLX_RANK"]))
                """,
                ranks=3,
            ),
            0,
        )

    def test_mpi_exit_codes(self):
        # The fake executable tests subprocess status handling without MPI.
        mpirun = self.root / "mpirun"
        mpirun.write_text(
            '#!/bin/sh\nwhile [ "$#" -gt 0 ] && [ "$1" != "--" ]; '
            'do shift; done\nshift\nexec "$@"\n'
        )
        mpirun.chmod(0o755)
        self.env["PATH"] = str(self.root) + os.pathsep + self.env.get("PATH", "")
        for code in (0, 1, 7, 42, 255):
            with self.subTest(code=code):
                self.assert_exit(
                    self.run_launch(f"raise SystemExit({code})", backend="mpi"), code
                )
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with self.subTest(signal=sig):
                self.assert_exit(
                    self.run_launch(
                        f"import os; os.kill(os.getpid(), {int(sig)})", backend="mpi"
                    ),
                    128 + sig,
                )

    def test_mpi_interrupt(self):
        args = argparse.Namespace(cwd=None, env=[], mpi_arg=[], verbose=False)
        hosts = [Host(0, "127.0.0.1", ["127.0.0.1"], [])]
        which = subprocess.CompletedProcess([], 0, stdout=b"/test/mpirun\n")
        with patch.object(launch, "get_mpi_libname", return_value=None):
            with patch.object(launch, "run", side_effect=[which, KeyboardInterrupt]):
                self.assertEqual(launch.launch_mpi(None, hosts, args, ["test.py"]), 130)

    def test_print_python(self):
        result = self.run_launch("", options=["--print-python"])
        self.assert_exit(result, 0)
        self.assertEqual(result.stdout.strip(), sys.executable)

    def test_invalid_backend(self):
        self.assert_exit(self.run_launch("", backend="invalid"), 2)


if __name__ == "__main__":
    unittest.main()
