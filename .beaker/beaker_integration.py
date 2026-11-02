"""Optimize Pi's source harness against a pinned 3-train/2-test JobBench pilot."""

from __future__ import annotations

import tempfile
from pathlib import Path

from beaker import Case, CaseResult, CaseScore, Integration, JsonValue, RolloutRuntime, repository

from jobbench.data import Expected, JobBenchSetup, PILOT, TaskInput
from jobbench.execution import run_task
from jobbench.scoring import score_async


async def run_case(*, case_input: JsonValue, runtime: RolloutRuntime) -> CaseResult:
    if runtime.model not in (None, PILOT["agent_model"]):
        raise ValueError("This harness experiment fixes the model to openai:gpt-6-sol")
    task = TaskInput.model_validate(case_input)
    # Repository evaluators set cwd to the candidate tree. Never fall back to a seed install.
    root = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="pi-jobbench-rollout-") as temporary:
        with runtime.trace.stage("pi.jobbench", inputs={"task": task.task, "model": PILOT["agent_model"]}) as stage:
            observed = await run_task(task, inputs=runtime.case_files_dir, root=root, run_dir=Path(temporary))
            manifest = [{key: value for key, value in item.items() if key != "content_base64"} for item in observed["files"]]
            stage.output({"files": manifest, "stats": observed["stats"]})
            return CaseResult(
                output={"response": observed["response"][:16000], "deliverables": manifest, "stats": observed["stats"]},
                output_kind="record",
                context={"files": observed["files"]},
            )


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
