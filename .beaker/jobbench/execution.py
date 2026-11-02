"""Invoke the existing Pi source runner with evaluation-scoped process state."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Any

from .data import PILOT, TaskInput


# Keep the binary payload below Beaker's 8 MiB execution envelope after base64.
MAX_DELIVERABLE_BYTES = 4 * 1024 * 1024


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


async def run_task(task: TaskInput, *, inputs: Path, root: Path, run_dir: Path) -> dict[str, Any]:
    """Run the fixed gpt-6-sol configuration and retain files outside the source tree."""
    if os.name != "posix":
        raise RuntimeError("The local JobBench evaluator currently requires Linux or macOS")
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js 22.19+ is required; prepare the Pi checkout first")
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is required for local candidate execution")
    entry_dir = root / "packages/coding-agent/src/experimental"
    if not (entry_dir / "local-runner-cli.ts").is_file():
        raise RuntimeError(f"No candidate Pi source checkout at {root}")
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    shutil.copytree(inputs / "task_folder", workspace / "task_folder")
    (workspace / "output").mkdir()
    agent_dir = run_dir / "home/.pi/agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "settings.json").write_text(json.dumps({
        "defaultThinkingLevel": PILOT["thinking_level"],
    }), encoding="utf-8")
    # Each child gets its own HOME/config. Judge and Beaker credentials never enter Pi's environment.
    allowed = {
        "PATH", "LANG", "LC_ALL", "TMPDIR", "OPENAI_API_KEY", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "no_proxy", "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS",
        "SYSTEMROOT", "SystemRoot",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    # Absolute invocation of the evaluator's Python does not activate its venv in PATH.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", os.defpath)
    env.update(HOME=str(run_dir / "home"), PI_CODING_AGENT_DIR=str(agent_dir))
    prompt = (
        "Read task_folder/TASK_INSTRUCTIONS.txt and the reference files under task_folder/. "
        "Complete that task. Save only final deliverables under output/. "
        "Do not access files outside this workspace. For Office files, PDFs, images, or databases, "
        "use appropriate tools or write code to inspect them. Explain conflicting sources.\n\n"
        "Task instructions:\n" + task.instructions
    )
    request = {"cwd": str(workspace), "prompt": prompt, "model": "gpt-6-sol"}
    with (run_dir / "pi.stderr.log").open("wb") as stderr:
        process = await asyncio.create_subprocess_exec(
            str(Path(node).resolve()), "--import", (entry_dir / "source-resolver.ts").as_uri(),
            str(entry_dir / "local-runner-cli.ts"), cwd=workspace, env=env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=stderr,
            start_new_session=True,
        )
        communication = asyncio.create_task(process.communicate(json.dumps(request).encode()))
        try:
            stdout, _ = await asyncio.wait_for(asyncio.shield(communication), timeout=PILOT["timeout_seconds"])
        except BaseException:
            await stop_process(process)
            await communication
            raise
    (run_dir / "pi.stdout.json").write_bytes(stdout)
    if process.returncode != 0:
        raise RuntimeError(f"Pi exited with code {process.returncode}; inspect {run_dir / 'pi.stderr.log'}")
    envelope = json.loads(stdout)
    if envelope.get("ok") is not True:
        raise RuntimeError(f"Pi runner failed: {envelope.get('error')}")
    result = envelope["result"]
    if result["model"] != "gpt-6-sol" or result["provider"] != "openai":
        raise RuntimeError("Candidate execution changed the fixed provider/model")
    packed = []
    total = 0
    for path in sorted((workspace / "output").rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Submission contains a symlink: {path.name}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Submission contains a non-regular file: {path.name}")
        total += path.stat().st_size
        if total > MAX_DELIVERABLE_BYTES:
            raise RuntimeError("Submission exceeds the pilot's 4 MiB binary transport limit; not scored as zero")
        content = path.read_bytes()
        packed.append({
            "name": path.relative_to(workspace / "output").as_posix(),
            "sha256": hashlib.sha256(content).hexdigest(), "size": len(content),
            "content_base64": base64.b64encode(content).decode("ascii"),
        })
    observed = {
        "task": task.task, "response": result["response"], "files": packed, "stats": result["stats"],
        "protocol": PILOT,
    }
    (run_dir / "result.json").write_text(json.dumps(observed), encoding="utf-8")
    return observed
