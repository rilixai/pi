"""Call this checkout's Pi coding agent from Python, without a terminal UI."""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Sequence, TypedDict, cast


class PiRunResult(TypedDict):
    response: str
    messages: list[dict[str, object]]
    stats: dict[str, object]
    provider: str
    model: str
    cwd: str


class PiRunError(RuntimeError):
    """Pi failed to start, failed its task, or returned an invalid result."""

    def __init__(
        self, message: str, *, stderr: str = "", returncode: int | None = None
    ) -> None:
        super().__init__(message)
        self.stderr = stderr
        self.returncode = returncode


class PiRunTimeout(PiRunError):
    """The caller's time budget expired; Pi and its child processes were stopped."""


def run_pi(
    *,
    prompt: str,
    cwd: str | os.PathLike[str],
    model: str,
    skill_paths: Sequence[str | os.PathLike[str]] | None = None,
    append_system_prompt: str | None = None,
    timeout: float | None = None,
) -> PiRunResult:
    """Run one fresh Pi session in an existing workspace and keep its files.

    Uses OpenAI with OPENAI_API_KEY from the inherited environment. Resources
    and settings load normally from the workspace and Pi's agent directory.
    Relative skill paths resolve from cwd. timeout is in seconds and includes
    startup; None waits without imposing a deadline. No Python packages needed.
    """
    for name, value in (("prompt", prompt), ("model", model)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    if append_system_prompt is not None and not isinstance(append_system_prompt, str):
        raise TypeError("append_system_prompt must be a string")
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be finite and greater than zero")
    workspace = Path(cwd).expanduser().resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError(f"cwd must be a directory: {workspace}")
    request: dict[str, object] = {
        "prompt": prompt,
        "cwd": str(workspace),
        "model": model,
    }
    if skill_paths is not None:
        if isinstance(skill_paths, (str, bytes, os.PathLike)):
            raise TypeError("skill_paths must be a sequence of paths")
        paths = [os.fspath(path) for path in skill_paths]
        if any(not isinstance(path, str) or not path.strip() for path in paths):
            raise ValueError("skill_paths must contain nonempty paths")
        request["skillPaths"] = paths
    if append_system_prompt is not None:
        request["appendSystemPrompt"] = append_system_prompt

    node = shutil.which("node")
    if node is None:
        raise PiRunError("Node.js is not on PATH. Install Node.js 22.19 or newer.")
    source_dir = Path(__file__).resolve().parent.parent / "packages/coding-agent/src/experimental"
    command = [
        str(Path(node).resolve()),
        "--import",
        (source_dir / "source-resolver.ts").as_uri(),
        str(source_dir / "local-runner-cli.ts"),
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=os.name == "posix",
        )
    except OSError as error:
        raise PiRunError(f"Cannot start Pi: {error}") from error
    try:
        stdout, stderr = process.communicate(
            json.dumps(request, ensure_ascii=False), timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        _, stderr = _stop_pi(process)
        raise PiRunTimeout(
            f"Pi exceeded the {timeout}s timeout",
            stderr=stderr,
            returncode=process.returncode,
        ) from error
    except BaseException:
        _stop_pi(process)
        raise

    try:
        envelope = json.loads(stdout)
    except (ValueError, TypeError) as error:
        raise PiRunError(
            "Pi did not return a JSON result. Check stderr, install the checkout's "
            "Node dependencies and hydrate its model catalog (see python-harness.md).",
            stderr=stderr,
            returncode=process.returncode,
        ) from error
    if not isinstance(envelope, dict):
        raise PiRunError("Invalid Pi response", stderr=stderr, returncode=process.returncode)
    if process.returncode != 0 or envelope.get("ok") is not True:
        detail = envelope.get("error", {})
        message = (
            detail.get("message", "Pi run failed")
            if isinstance(detail, dict)
            else "Pi run failed"
        )
        raise PiRunError(str(message), stderr=stderr, returncode=process.returncode)
    result = envelope.get("result")
    if not isinstance(result, dict) or not (
        isinstance(result.get("response"), str)
        and isinstance(result.get("messages"), list)
        and isinstance(result.get("stats"), dict)
    ):
        raise PiRunError("Invalid Pi result", stderr=stderr, returncode=process.returncode)
    return cast(PiRunResult, result)


def _stop_pi(process: subprocess.Popen[str]) -> tuple[str, str]:
    if os.name == "posix":
        try:
            # The runner's SIGTERM handler also kills Pi's detached bash groups.
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            return process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # Clear group members even if Node exited during the grace period.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    else:
        taskkill = Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32/taskkill.exe"
        subprocess.run(
            [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    return process.communicate()
