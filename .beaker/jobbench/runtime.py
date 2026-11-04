"""Prepare each candidate checkout's Node dependencies and pinned model data."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import signal
import urllib.request
from pathlib import Path

import nodejs_wheel


NODE_ROOT = Path(nodejs_wheel.__file__).parent
NODE = NODE_ROOT / "bin/node"
NPM = NODE_ROOT / "lib/node_modules/npm/bin/npm-cli.js"


def log_tail(path: Path) -> str:
    """Read bounded diagnostics before temporary rollout files are removed."""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 4000))
        return stream.read().decode("utf-8", errors="replace").strip()


async def stop_process(process: asyncio.subprocess.Process) -> None:
    """Stop Node and let Pi's signal handler kill its detached shell groups."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    await process.wait()


async def setup_command(
    command: list[str], *, root: Path, env: dict[str, str], log: Path, timeout_seconds: int = 600
) -> None:
    """Run bounded setup without lifecycle hooks or model calls."""
    with log.open("wb") as output:
        process = await asyncio.create_subprocess_exec(
            *command, cwd=root, env=env, stdout=output, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
        except TimeoutError as error:
            await stop_process(process)
            raise RuntimeError(f"Pi setup exceeded {timeout_seconds} seconds ({log.name}): {log_tail(log)}") from error
        except BaseException:
            await stop_process(process)
            raise
    if process.returncode != 0:
        raise RuntimeError(f"Pi setup failed ({log.name}, exit {process.returncode}): {log_tail(log)}")


async def prepare_checkout(root: Path, *, run_dir: Path, env: dict[str, str]) -> None:
    """Install dependencies in the candidate tree; Beaker omits node_modules when copying it."""
    # Setup needs registry/catalog access, but never receives the candidate provider key.
    setup_env = {key: value for key, value in env.items() if key not in ("OPENAI_API_KEY", "OPENROUTER_API_KEY")}
    # Multiple rollout processes can share a candidate tree. npm ci does not
    # modify this lockfile, so its inode remains stable while setup is serialized.
    with (root / "package-lock.json").open("rb") as lock:
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.1)
        if not (root / "node_modules/.package-lock.json").is_file():
            await setup_command(
                [str(NODE), str(NPM), "ci", "--ignore-scripts", "--no-audit", "--no-fund", "--include=dev"],
                root=root, env=setup_env, log=run_dir / "npm-install.log",
            )
        if not (root / "packages/ai/src/providers/data/.manifest.json").is_file():
            revision = json.loads((root / "nix/model-catalog.json").read_text(encoding="utf-8"))["revision"]
            if not isinstance(revision, str) or not re.fullmatch(r"sha256-[0-9a-f]{64}", revision):
                raise ValueError("Invalid pinned Pi model catalog revision")

            def download_catalog() -> bytes:
                url = f"https://pi.dev/api/models/revisions/{revision}?types=chat,image,classifier"
                with urllib.request.urlopen(url, timeout=120) as response:
                    content = response.read()
                if "sha256-" + hashlib.sha256(content).hexdigest() != revision:
                    raise ValueError("Pi model catalog content does not match its pinned revision")
                return content

            catalog = run_dir / "models.all.json"
            catalog.write_bytes(await asyncio.to_thread(download_catalog))
            await setup_command(
                [str(NODE), str(root / "packages/ai/scripts/hydrate-model-catalog.ts"), str(catalog)],
                root=root, env=setup_env, log=run_dir / "model-catalog.log",
            )
