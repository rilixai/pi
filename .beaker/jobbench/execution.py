"""Invoke the existing Pi source runner with evaluation-scoped process state."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .data import PROTOCOL, TaskInput
from .runtime import NODE, log_tail, prepare_checkout, stop_process


# Keep the binary payload below Beaker's 8 MiB execution envelope after base64.
MAX_DELIVERABLE_BYTES = 4 * 1024 * 1024


def rollout_model(model: str | None = None) -> str:
    """Resolve an OpenAI API model ID from the launch override or protocol default."""
    selected = PROTOCOL["agent_model"] if model is None else model
    provider, separator, model_id = selected.strip().partition(":")
    if not separator:
        provider, model_id = "openai", provider
    if provider != "openai" or not model_id.strip():
        raise ValueError("Pi JobBench rollouts require an OpenAI model ID, optionally prefixed with openai:")
    return model_id.strip()


async def run_task(
    task: TaskInput, *, inputs: Path, root: Path, run_dir: Path, model: str | None = None,
) -> dict[str, Any]:
    """Run the selected OpenAI model and retain files outside the source tree."""
    model_id = rollout_model(model)
    if os.name != "posix":
        raise RuntimeError("The local JobBench evaluator currently requires Linux or macOS")
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is required for local candidate execution")
    entry_dir = root / "packages/coding-agent/src/experimental"
    if not (entry_dir / "local-runner-cli.ts").is_file():
        raise RuntimeError(f"No candidate Pi source checkout at {root}")
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    shutil.copytree(inputs / "task_folder", workspace / "task_folder")
    (workspace / "output").mkdir()
    (workspace / "scratch").mkdir()
    agent_dir = run_dir / "home/.pi/agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "settings.json").write_text(json.dumps({
        "defaultThinkingLevel": PROTOCOL["thinking_level"],
    }), encoding="utf-8")
    # Separate configuration and sessions, without imposing a filesystem sandbox.
    allowed = {
        "PATH", "LANG", "LC_ALL", "TMPDIR", "OPENAI_API_KEY", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "no_proxy", "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS",
        "SYSTEMROOT", "SystemRoot",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["PATH"] = os.pathsep.join((str(NODE.parent), str(Path(sys.executable).parent), env.get("PATH", os.defpath)))
    env.update(HOME=str(run_dir / "home"), PI_CODING_AGENT_DIR=str(agent_dir))
    await prepare_checkout(root, run_dir=run_dir, env=env)
    prompt = (
        "Read task_folder/TASK_INSTRUCTIONS.txt and the reference files under task_folder/. "
        "Complete that task. Save only final deliverables under output/. "
        "Do not access files outside this workspace. For Office files, PDFs, images, or databases, "
        "use appropriate tools or write code to inspect them. Explain conflicting sources.\n\n"
        "Task instructions:\n" + task.instructions
    )
    request = {"cwd": str(workspace), "prompt": prompt, "model": model_id}
    with (run_dir / "pi.stderr.log").open("wb") as stderr:
        process = await asyncio.create_subprocess_exec(
            str(NODE), "--import", (entry_dir / "source-resolver.ts").as_uri(),
            str(entry_dir / "local-runner-cli.ts"), cwd=workspace, env=env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=stderr,
            start_new_session=True,
        )
        communication = asyncio.create_task(process.communicate(json.dumps(request).encode()))
        try:
            stdout, _ = await asyncio.wait_for(asyncio.shield(communication), timeout=PROTOCOL["timeout_seconds"])
        except BaseException:
            await stop_process(process)
            await communication
            raise
    returncode = process.returncode
    (run_dir / "pi.stdout.json").write_bytes(stdout)
    if returncode != 0:
        # The temporary rollout directory is deleted on failure. Preserve the
        # runner's structured error, or Node's startup error, in Beaker's case report.
        detail = log_tail(run_dir / "pi.stderr.log")
        try:
            failed = json.loads(stdout)
            if isinstance(failed, dict) and isinstance(failed.get("error"), dict):
                detail = str(failed["error"].get("message") or detail)
        except (ValueError, UnicodeDecodeError):
            pass
        detail = detail.replace(env["OPENAI_API_KEY"], "[REDACTED]")
        raise RuntimeError(f"Pi exited with code {returncode}: {detail[-4000:] or 'No diagnostics emitted'}")
    envelope = json.loads(stdout)
    if envelope.get("ok") is not True:
        raise RuntimeError(f"Pi runner failed: {envelope.get('error')}")
    result = envelope["result"]
    if result["model"] != model_id or result["provider"] != "openai":
        raise RuntimeError("Pi returned a different provider or model than requested")
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
            raise RuntimeError("Submission exceeds the 4 MiB binary transport limit; not scored as zero")
        content = path.read_bytes()
        packed.append({
            "name": path.relative_to(workspace / "output").as_posix(),
            "sha256": hashlib.sha256(content).hexdigest(), "size": len(content),
            "content_base64": base64.b64encode(content).decode("ascii"),
        })
    observed = {
        "task": task.task, "response": result["response"], "messages": result["messages"], "files": packed,
        "stats": result["stats"], "protocol": {**PROTOCOL, "agent_model": f"openai:{model_id}"},
    }
    (run_dir / "result.json").write_text(json.dumps(observed), encoding="utf-8")
    return observed
