"""Build the candidate runtime and own one offline Pi container per task."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
import uuid
from pathlib import Path

from .calculator import docker_ready
from .relay import model_relay
from .runtime import setup_command, stop_process


CONTAINER_FILES = Path(__file__).with_name("container")


def docker_environment() -> dict[str, str]:
    """Pass Docker configuration and build connectivity, never provider credentials."""
    allowed = {
        "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy", "SSL_CERT_FILE",
    }
    return {**{key: value for key, value in os.environ.items() if key in allowed}, "DOCKER_BUILDKIT": "0"}


def stage_runtime(root: Path, context: Path) -> str:
    """Hash and copy runtime inputs only; exclude evaluator, datasets, and secrets."""
    digest = hashlib.sha256()

    def copy(source: Path, target: Path) -> None:
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Pi runtime input must be a regular file: {source}")
        content = source.read_bytes()
        name = target.relative_to(context).as_posix().encode()
        executable = bool(source.stat().st_mode & 0o111)
        digest.update(len(name).to_bytes(8, "big") + name + bytes([executable]) + len(content).to_bytes(8, "big") + content)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        target.chmod(0o755 if executable else 0o644)

    for name in ("Dockerfile", "entrypoint.mjs"):
        copy(CONTAINER_FILES / name, context / name)
    for name in ("pyproject.toml", "uv.lock"):
        copy(root / name, context / "python" / name)
    manifests = {root / "package.json", root / "package-lock.json"}
    for pattern in json.loads((root / "package.json").read_text())["workspaces"]:
        manifests.update(root.glob(pattern + "/package.json"))
    for source in sorted(manifests):
        copy(source, context / "manifests" / source.relative_to(root))
    for source in sorted(root.glob("tsconfig*.json")):
        copy(source, context / "source" / source.name)
    copy(root / "nix/model-catalog.json", context / "source/nix/model-catalog.json")
    for package in sorted((root / "packages").iterdir()):
        if not package.is_dir() or package.is_symlink():
            continue
        for child in sorted(package.iterdir()):
            if child.name in ("src", "scripts", "docs", "assets", "native", "schemas"):
                for directory, directories, files in os.walk(child, followlinks=False):
                    if any((Path(directory) / name).is_symlink() for name in directories):
                        raise ValueError("Symlink directories are not allowed in the Pi runtime")
                    directories[:] = sorted(name for name in directories if name not in (
                        "node_modules", "test", "tests", "__pycache__", ".git",
                    ))
                    for name in sorted(files):
                        source = Path(directory) / name
                        # Hidden local state is not part of a source runtime.
                        if name.startswith(".") or name.endswith((".test.ts", ".test.mjs", ".pyc")):
                            continue
                        copy(source, context / "source" / source.relative_to(root))
            elif child.name in ("package.json", "README.md", "CHANGELOG.md"):
                copy(child, context / "source" / child.relative_to(root))
    return "pi-jobbench-runtime:" + digest.hexdigest()


def download_catalog(root: Path) -> bytes:
    """Verify immutable model metadata before putting it in the runtime image."""
    revision = json.loads((root / "nix/model-catalog.json").read_text())["revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"sha256-[0-9a-f]{64}", revision):
        raise ValueError("Invalid pinned Pi model catalog revision")
    with urllib.request.urlopen(f"https://pi.dev/api/models/revisions/{revision}?types=chat,image,classifier", timeout=120) as response:
        content = response.read()
    if "sha256-" + hashlib.sha256(content).hexdigest() != revision:
        raise ValueError("Pi model catalog content does not match its pinned revision")
    return content


async def prepare_image(root: Path, *, run_dir: Path) -> str:
    """Cache a source-addressed image; never fall back to running Pi on the host."""
    env = docker_environment()
    if shutil.which("docker") is None or not await docker_ready(["info"], env):
        raise RuntimeError("Isolated Pi execution requires a local running Docker daemon with Unix socket bind-mount support")
    with tempfile.TemporaryDirectory(prefix="pi-jobbench-image-") as temporary:
        context = Path(temporary)
        image = await asyncio.to_thread(stage_runtime, root, context)
        if not await docker_ready(["image", "inspect", image], env):
            (context / "models.all.json").write_bytes(await asyncio.to_thread(download_catalog, root))
            await setup_command(
                ["docker", "build", "--network=host", "-t", image, str(context)],
                root=root, env=env, log=run_dir / "pi-image-build.log",
            )
        return image


async def run_container(
    *, image: str, workspace: Path, run_dir: Path, request: dict[str, str], thinking_level: str, timeout: float,
) -> tuple[int, bytes]:
    """Attach to one bounded container and remove it even on cancellation or timeout."""
    env = docker_environment()
    name = "pi-jobbench-" + uuid.uuid4().hex
    # The enclosing run directory is private. The unprivileged container UID
    # needs write access to this single bind mount.
    (workspace / "output").chmod(0o777)
    for directory in [workspace / "task_folder", *(workspace / "task_folder").rglob("*")]:
        directory.chmod(0o755 if directory.is_dir() else 0o644)
    # Hosted attempt paths can exceed AF_UNIX's ~108-byte path limit. Keep the
    # socket in a short, private directory rather than under the rollout path.
    with (
        tempfile.TemporaryDirectory(prefix="pi-bridge-", dir="/tmp") as temporary,
        model_relay(Path(temporary) / "socket", api_key=os.environ["OPENAI_API_KEY"], model=request["model"]) as bridge,
    ):
        arguments = [
            "docker", "create", "--name", name, "--interactive", "--network=none", "--read-only",
            "--user=65534:65534", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--pids-limit=256", "--memory=4g", "--cpus=2", "--workdir=/workspace",
            "--tmpfs=/workspace:rw,nosuid,nodev,size=64m,mode=1777",
            "--tmpfs=/workspace/scratch:rw,nosuid,nodev,size=512m,mode=1777",
            "--tmpfs=/tmp:rw,nosuid,nodev,size=512m,mode=1777",
            "--tmpfs=/home/pi:rw,nosuid,nodev,size=128m,mode=1777",
            "--mount", f"type=bind,src={workspace / 'task_folder'},dst=/workspace/task_folder,readonly",
            "--mount", f"type=bind,src={workspace / 'output'},dst=/workspace/output",
            "--mount", f"type=bind,src={bridge},dst=/run/pi-bridge,readonly",
            "--env", "PI_THINKING_LEVEL=" + thinking_level, image,
        ]
        process = None
        communication = None
        created = False
        try:
            await setup_command(arguments, root=run_dir, env=env, log=run_dir / "pi-container-create.log")
            created = True
            with (run_dir / "pi.stderr.log").open("wb") as stderr:
                process = await asyncio.create_subprocess_exec(
                    "docker", "start", "--attach", "--interactive", name, cwd=run_dir, env=env,
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=stderr,
                    start_new_session=True,
                )
                communication = asyncio.create_task(process.communicate(json.dumps(request).encode()))
                stdout, _ = await asyncio.wait_for(asyncio.shield(communication), timeout=timeout)
                await setup_command(
                    ["docker", "inspect", "--format", "{{json .State}}", name],
                    root=run_dir, env=env, log=run_dir / "pi-container-state.json",
                )
                state = json.loads((run_dir / "pi-container-state.json").read_text())
                if state["Running"]:
                    raise RuntimeError("Pi's Docker attachment ended before the container exited")
                return state["ExitCode"], stdout
        finally:
            # Remove the container before reading deliverables: detached shell
            # descendants must not outlive the task or keep writing output.
            cleanup = asyncio.create_task(asyncio.wait_for(setup_command(
                ["docker", "rm", "--force", name], root=run_dir, env=env, log=run_dir / "pi-container-remove.log",
            ), timeout=15))
            try:
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise
                except (RuntimeError, TimeoutError) as error:
                    if created:
                        raise RuntimeError(f"Failed to remove Pi container {name}") from error
            finally:
                if process is not None and process.returncode is None:
                    await stop_process(process)
                if communication is not None:
                    await communication
