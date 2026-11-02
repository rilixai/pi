"""Prepare the unchanged JobBench calculator before hosted rollouts start."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from openpyxl import Workbook

from .runtime import log_tail, setup_command, stop_process

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/jobbench/eval"))
from jobbench_eval.calc import RUNTIME, image_name, recalculate


async def docker_ready(arguments: list[str], env: dict[str, str]) -> bool:
    """Check the daemon or image without retaining potentially large inspection output."""
    process = await asyncio.create_subprocess_exec(
        "docker", *arguments, env=env, stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
    )
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except TimeoutError:
        await stop_process(process)
        return False
    except BaseException:
        await stop_process(process)
        raise
    return process.returncode == 0


@asynccontextmanager
async def calculator_runtime() -> AsyncIterator[None]:
    """Own hosted Docker startup, image preparation, and teardown for one attempt."""
    # Local evaluation retains the user's existing Docker setup. Structural
    # smoke outside a hosted run also remains free of container startup/builds.
    if not os.environ.get("BEAKER_RUN_ID"):
        yield
        return
    if shutil.which("docker") is None:
        raise RuntimeError("JobBench scoring requires docker.io in the integration's apt_install list")
    allowed = {
        "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy", "SSL_CERT_FILE",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    # Debian's Docker package includes the classic builder, but not necessarily
    # the buildx plugin. The upstream Dockerfile needs no BuildKit features.
    env["DOCKER_BUILDKIT"] = "0"
    daemon = None
    with tempfile.TemporaryDirectory(prefix="pi-jobbench-calculator-") as temporary:
        directory = Path(temporary)
        with (directory / "dockerd.log").open("wb") as log:
            try:
                if not await docker_ready(["info"], env):
                    if os.geteuid() != 0 or shutil.which("dockerd") is None:
                        raise RuntimeError("JobBench scoring needs a running Docker daemon in a Modal VM sandbox")
                    if env.get("DOCKER_HOST") or env.get("DOCKER_CONTEXT"):
                        raise RuntimeError("The configured JobBench Docker daemon is unavailable")
                    # A private data root and no bridge/iptables changes keep
                    # calculator startup independent of Beaker's provider proxy.
                    daemon = await asyncio.create_subprocess_exec(
                        "dockerd", "--host=unix:///var/run/docker.sock",
                        f"--data-root={directory / 'data'}", f"--exec-root={directory / 'exec'}",
                        f"--pidfile={directory / 'dockerd.pid'}", "--storage-driver=vfs",
                        "--iptables=false", "--bridge=none", "--ip-forward=false", "--ip-masq=false",
                        env=env, stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
                    )
                    deadline = asyncio.get_running_loop().time() + 60
                    while not await docker_ready(["info"], env):
                        if daemon.returncode is not None or asyncio.get_running_loop().time() >= deadline:
                            raise RuntimeError(f"JobBench Docker daemon failed to start: {log_tail(directory / 'dockerd.log')}")
                        await asyncio.sleep(0.25)
                if not await docker_ready(["image", "inspect", image_name()], env):
                    await setup_command(
                        # The isolated daemon has no bridge. Only the trusted
                        # image build uses host networking; calculations remain offline.
                        ["docker", "build", "--network=host", "-t", image_name(), str(RUNTIME.parent)],
                        root=RUNTIME.parent, env=env, log=directory / "image-build.log",
                    )
                # Verify real recalculation through the official bounded worker,
                # before any paid candidate or judge calls can be made.
                probe = directory / "runtime-check.xlsx"
                workbook = Workbook()
                workbook.active["A1"] = "=1+1"
                workbook.save(probe)
                workbook.close()
                calculated = await asyncio.to_thread(recalculate, probe, {"Sheet": ["A1"]})
                cell = calculated["cells"]["Sheet"]["A1"]
                if cell["value"] != 2 or cell["error_code"] != 0:
                    raise RuntimeError("The official JobBench calculator failed its formula preflight")
                yield
            finally:
                # Never shut down a daemon supplied by the environment.
                if daemon is not None:
                    await stop_process(daemon)
