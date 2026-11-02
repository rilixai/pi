"""Invoke the existing Pi source runner with evaluation-scoped process state."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .data import PILOT, TaskInput
from .runtime import log_tail
from .sandbox import prepare_image, run_container


# Keep the binary payload below Beaker's 8 MiB execution envelope after base64.
MAX_DELIVERABLE_BYTES = 4 * 1024 * 1024


async def run_task(task: TaskInput, *, inputs: Path, root: Path, run_dir: Path) -> dict[str, Any]:
    """Run the fixed gpt-6-sol configuration and retain files outside the source tree."""
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
    image = await prepare_image(root, run_dir=run_dir)
    prompt = (
        "Read task_folder/TASK_INSTRUCTIONS.txt and the reference files under task_folder/. "
        "Complete that task. Save only final deliverables under output/. "
        "Do not access files outside this workspace. For Office files, PDFs, images, or databases, "
        "use appropriate tools or write code to inspect them. Explain conflicting sources.\n\n"
        "Task instructions:\n" + task.instructions
    )
    request = {"cwd": "/workspace", "prompt": prompt, "model": "gpt-6-sol"}
    returncode, stdout = await run_container(
        image=image, workspace=workspace, run_dir=run_dir, request=request,
        thinking_level=PILOT["thinking_level"], timeout=PILOT["timeout_seconds"],
    )
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
        detail = detail.replace(os.environ["OPENAI_API_KEY"], "[REDACTED]")
        raise RuntimeError(f"Pi exited with code {returncode}: {detail[-4000:] or 'No diagnostics emitted'}")
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
