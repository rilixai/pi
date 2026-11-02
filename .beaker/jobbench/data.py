"""Load explicitly selected JobBench tasks and stage their agent-visible files."""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import urllib.request
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path, PurePosixPath
from typing import Literal

from beaker import Case, CaseFile, RepositoryRunSetupResult, RunSetup, RunSetupResults, SetupRuntime
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .calculator import calculator_runtime


PROTOCOL = json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf-8"))
HF_ROOT = "https://huggingface.co/datasets/JobBench/job-bench/resolve/" + PROTOCOL["dataset_revision"]
METADATA_URL = "https://huggingface.co/api/datasets/JobBench/job-bench/revision/" + PROTOCOL["dataset_revision"] + "?blobs=true"


class InputFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    size: int = Field(ge=0)
    digest: str
    algorithm: Literal["sha256", "git-sha1"]

    @field_validator("name")
    @classmethod
    def safe_name(cls, name: str) -> str:
        parts = name.split("/")
        if "\\" in name or any(part in ("", ".", "..") for part in parts) or parts[0] != "task_folder":
            raise ValueError("Input files must be relative paths under task_folder/")
        return name


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str
    instructions: str = Field(min_length=1)
    files: list[InputFile] = Field(min_length=1)

    @field_validator("task")
    @classmethod
    def selected_task(cls, task: str) -> str:
        parts = task.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1].startswith("task"):
            raise ValueError(f"Task {task!r} is not a valid JobBench task path")
        return task


class Rubric(BaseModel):
    rubric: str = Field(min_length=1)
    criterion: list[str] = Field(min_length=1)
    weight: float = Field(ge=0, allow_inf_nan=False)


class Expected(BaseModel):
    rubrics: list[Rubric] = Field(min_length=1)

    @field_validator("rubrics")
    @classmethod
    def positive_total(cls, rubrics: list[Rubric]) -> list[Rubric]:
        if sum(item.weight for item in rubrics) <= 0:
            raise ValueError("Rubrics must have a positive total weight")
        return rubrics


class TaskRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    input: TaskInput
    expected: Expected


def read_url(url: str) -> bytes:
    """Read a pinned public benchmark resource, without provider credentials."""
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def rows(tasks: Sequence[str]) -> list[TaskRow]:
    """Load the requested tasks from the fixed upstream dataset revision."""
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Provide a nonempty list of distinct JobBench tasks")
    metadata = json.loads(read_url(METADATA_URL))
    if metadata["sha"] != PROTOCOL["dataset_revision"]:
        raise ValueError("Hugging Face returned a different dataset revision")
    result = []
    for task in tasks:
        prefix = f"dataset/{task}/"
        files = []
        for item in metadata["siblings"]:
            path = item["rfilename"]
            if path.startswith(prefix + "task_folder/"):
                lfs = item.get("lfs")
                files.append(InputFile(
                    name=path.removeprefix(prefix), size=item["size"],
                    digest=lfs["sha256"] if lfs else item["blobId"],
                    algorithm="sha256" if lfs else "git-sha1",
                ))
        if not files:
            raise ValueError(f"Task {task!r} has no task_folder files at the pinned dataset revision")
        instructions = read_url(f"{HF_ROOT}/{prefix}task_folder/TASK_INSTRUCTIONS.txt").decode("utf-8")
        expected = Expected.model_validate_json(read_url(f"{HF_ROOT}/{prefix}RUBRICS.json"))
        result.append(TaskRow(
            id="main--" + task.replace("/", "--"),
            input=TaskInput(task=task, instructions=instructions, files=sorted(files, key=lambda item: item.name)),
            expected=expected,
        ))
    return result


def stage_inputs(task: TaskInput, destination: Path) -> tuple[CaseFile, ...]:
    """Verify upstream hashes and expose only task_folder files to the candidate."""
    attached = []
    for source in task.files:
        content = read_url(f"{HF_ROOT}/dataset/{task.task}/{source.name}")
        if len(content) != source.size:
            raise ValueError(f"Input size mismatch: {source.name}")
        digest = hashlib.sha256(content).hexdigest() if source.algorithm == "sha256" else hashlib.sha1(
            f"blob {len(content)}\0".encode() + content
        ).hexdigest()
        if digest != source.digest:
            raise ValueError(f"Input hash mismatch: {source.name}")
        target = destination.joinpath(*PurePosixPath(source.name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        attached.append(CaseFile(
            name=source.name, path=target,
            expected_sha256=hashlib.sha256(content).hexdigest(), expected_size_bytes=len(content),
        ))
    return tuple(attached)


class JobBenchSetup(RunSetup[TaskRow]):
    row_model = TaskRow

    @asynccontextmanager
    async def prepare_run(self, *, runtime: SetupRuntime) -> AsyncIterator[RunSetupResults]:
        del runtime
        async with calculator_runtime():
            yield {"repository": RepositoryRunSetupResult()}

    async def load_cases(self, row: TaskRow, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        destination = runtime.case_files_dir(row.id)
        files = await asyncio.to_thread(stage_inputs, row.input, destination)
        yield Case(
            id=row.id, input=row.input.model_dump(mode="json"),
            expected=row.expected.model_dump(mode="json"), files=files,
            metadata={"benchmark": "JobBench", "dataset_revision": PROTOCOL["dataset_revision"]},
        )


@contextmanager
def jobbench_dataset(*, train_tasks: Sequence[str], test_tasks: Sequence[str]) -> Iterator[Path]:
    """Materialize explicit, disjoint task splits in a temporary directory."""
    if not train_tasks or not test_tasks:
        raise ValueError("Provide nonempty train and test task lists")
    tasks = rows([*train_tasks, *test_tasks])
    by_task = {row.input.task: row for row in tasks}
    with tempfile.TemporaryDirectory(prefix="pi-jobbench-dataset-") as temporary:
        directory = Path(temporary)
        for split, selected in (("train", train_tasks), ("test", test_tasks)):
            (directory / f"{split}.jsonl").write_text(
                "".join(by_task[task].model_dump_json() + "\n" for task in selected), encoding="utf-8"
            )
        yield directory
