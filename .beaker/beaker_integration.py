"""Optimize Pi's source harness against a pinned 3-train/2-test JobBench pilot."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from beaker import Case, CaseResult, CaseScore, Integration, JsonValue, RolloutRuntime, repository
from beaker.tracing import Trace

from jobbench.data import Expected, JobBenchSetup, TaskInput
from jobbench.execution import rollout_model, run_task
from jobbench.scoring import score_async


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime) -> CaseResult:
    model_id = rollout_model(runtime.model)
    task = TaskInput.model_validate(case_input)
    # Repository evaluators set cwd to the candidate tree. Never fall back to a seed install.
    root = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="pi-jobbench-rollout-") as temporary:
        with runtime.trace.stage("pi.jobbench", inputs={"task": task.task, "model": f"openai:{model_id}"}) as stage:
            observed = await run_task(
                task, inputs=runtime.case_files_dir, root=root, run_dir=Path(temporary), model=model_id,
            )
            trace_session(runtime.trace, observed["messages"])
            manifest = [{key: value for key, value in item.items() if key != "content_base64"} for item in observed["files"]]
            stage.output({"files": manifest, "stats": observed["stats"]})
            return CaseResult(
                output={"response": observed["response"][:16000], "deliverables": manifest, "stats": observed["stats"]},
                output_kind="record",
                context={"files": observed["files"]},
            )


def trace_session(trace: Trace, messages: list[dict[str, Any]]) -> None:
    """Replay Pi's session as one model call per assistant turn, each followed by the tools it ran."""
    results = {message["toolCallId"]: message for message in messages if message.get("role") == "toolResult"}
    inputs: list[Any] = []
    for message in messages:
        role = message.get("role")
        if role == "toolResult":
            # The result itself is the output of its tool span; the next model call only references it.
            inputs.append({"role": "toolResult", "toolCallId": message["toolCallId"], "toolName": message["toolName"]})
            continue
        if role != "assistant":
            inputs.append(_content_message(message))
            continue
        with trace.model_call(
            operation="chat", provider=message["provider"], model=message["model"], input_messages=inputs,
        ) as call:
            call.output({
                **_content_message(message),
                **{key: message[key] for key in ("stopReason", "errorMessage") if message.get(key) is not None},
            })
            usage = message["usage"]
            # Pi's input excludes cached tokens; Beaker counts cache reads and writes as slices of the input.
            call.usage(
                input_tokens=usage["input"] + usage["cacheRead"] + usage["cacheWrite"],
                output_tokens=usage["output"],
                cached_input_tokens=usage["cacheRead"],
                cache_write_input_tokens=usage["cacheWrite"],
            )
            if message.get("durationMs") is not None:
                call.attribute("pi.duration_ms", message["durationMs"])
        inputs = []
        for block in message["content"]:
            if block.get("type") != "toolCall":
                continue
            with trace.tool_call(block["name"], arguments=block["arguments"], call_id=block["id"]) as tool:
                result = results.get(block["id"])
                if result is None:
                    tool.output({"isError": True, "content": "Pi recorded no result for this tool call"})
                    continue
                tool.output({"isError": result["isError"], "content": [_block(part) for part in result["content"]]})
                if result.get("durationMs") is not None:
                    tool.attribute("pi.duration_ms", result["durationMs"])


def _content_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if content is None:
        # Pi's custom session messages (bash executions, summaries) carry their own fields.
        return message
    return {"role": message["role"], "content": content if isinstance(content, str) else [_block(part) for part in content]}


def _block(part: dict[str, Any]) -> dict[str, Any]:
    """Keep readable content; drop opaque provider signatures and image bytes."""
    if part.get("type") == "image":
        return {"type": "image", "mimeType": part.get("mimeType"), "base64_chars": len(part.get("data", ""))}
    return {key: value for key, value in part.items() if not key.endswith("Signature")}


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case_files_dir
    if "files" not in result.context or not isinstance(result.context["files"], list):
        raise RuntimeError("The binary submission is missing; the case cannot be scored")
    return await score_async(Expected.model_validate(case.expected), result.context["files"])


integration = Integration(
    targets=repository(paths=(
        "packages/agent/src/agent.ts",
        "packages/agent/src/agent-loop.ts",
        "packages/coding-agent/src/core/agent-session.ts",
        "packages/coding-agent/src/core/messages.ts",
        "packages/coding-agent/src/core/system-prompt.ts",
        "packages/coding-agent/src/core/compaction",
        "packages/coding-agent/src/core/tools",
    )),
    run_setup=JobBenchSetup,
    run_case=run_case,
    score_case=score_case,
)
