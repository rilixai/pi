# Pi harness optimization on JobBench: local pilot

This integration evaluates Pi's TypeScript source harness on five real tasks from
the JobBench `main` split. It is prepared for **local use only**: no Beaker agent,
hosted dataset, paid baseline, or optimization run is created by setup.

The editable surface includes Pi's agent loop, agent session, system prompt,
message conversion, compaction, and tool implementations. The Python bridge,
source resolver, provider adapters, benchmark, judge, and evaluation policy remain
outside that surface. See `beaker_integration.py` for the exact source paths.

## Fixed protocol

`jobbench/pilot.json` records the protocol:

- Dataset: `JobBench/job-bench@d43a66575635da0d3f6cb9a20976dc1a60f3ffc9`.
- Evaluator: `Job-Bench/job-bench-eval@660b33a86ed032559df0b2fdc1ad38cf7ceb83f3`.
- Pi model: `openai:gpt-6-sol`, thinking level `medium`, one-hour wall-clock budget.
- Judge: `xai:grok-4.3`, independent of the Pi model.
- Objective: official per-task rubric-weighted score, normalized to `[0, 1]`.
  The unweighted rubric pass rate is diagnostic only. Average task scores with
  equal task weights when reporting a subset aggregate.

| Beaker split | Original JobBench task |
| --- | --- |
| train | `data_entry_keyers/task1` |
| train | `human_resources_specialists/task1` |
| train | `supply_chain_managers/task1` |
| test | `social_science_research_assistants/task1` |
| test | `sociology_teachers_postsecondary/task1` |

All five are from the original `main` split. The selection exercises CSV, Excel,
images, PDF, DOCX, SQLite, and notebooks, without supplying hidden search-reference
folders. This is a pilot cohort, **not the 65-task JobBench leaderboard result**.
Do not use the two held-out cases to select candidate changes.

## Prepare the local checkout

Use Linux or macOS with Python 3.12+, Node.js 22.19+, and `uv`:

```bash
cd /path/to/pi
UV_PROJECT_ENVIRONMENT=.beaker/.venv uv sync --locked --group dev
npm ci --ignore-scripts
npm run hydrate:model-data
```

Python dependencies are evaluation tooling; Pi's normal Node runtime does not
import Beaker. The Node subprocess uses this checkout's `source-resolver.ts` and
`local-runner-cli.ts`, so edited candidate TypeScript is loaded without a build.
No npm registry version of Pi is used as the evaluated harness.

The official judge recalculates spreadsheet formulas with its pinned LibreOffice
runtime. Install Docker and prepare that runtime once before scoring:

```bash
.beaker/.venv/bin/python -c 'import sys; sys.path.insert(0, ".beaker/vendor/jobbench/eval"); from jobbench_eval.calc import setup_runtime; setup_runtime()'
```

The unchanged Dockerfile and extractor sources are included under
`vendor/jobbench/`, alongside their Apache-2.0 license and `PROVENANCE.json` with
the upstream Git blob hashes. Missing Docker or failed extraction is an
evaluation error, not a score of zero. `JOBBENCH_CALC_NATIVE=1` is only suitable
inside the exact upstream calculation image; it is not a generic LibreOffice
fallback.

## Structural smoke: no model calls

From the checkout root, run this Python snippet with `.beaker/.venv/bin/python`:

```python
import json
import subprocess
import sys

sys.path.insert(0, ".beaker")
from jobbench.data import pilot_dataset

with pilot_dataset() as dataset:
    subprocess.run(
        [
            ".beaker/.venv/bin/beaker", "run", "smoke", "--strict",
            "--integration-id", "pi-jobbench",
            "--config", json.dumps({"local_dataset_path": str(dataset)}),
        ],
        check=True,
    )
```

This fetches real instructions, rubrics, and task files at the pinned revision,
checks input sizes and content hashes, and validates the Beaker integration.
Generated `train.jsonl` and `test.jsonl` exist only in an OS-managed temporary
directory that is removed when the context exits. Do not commit generated rows,
raw datasets, or runtime workspaces.

Smoke does **not** execute Pi, run the judge, validate paid API authentication,
or prove Docker is available. It is a structural check, not an evaluated score.

## Run a real task locally

These commands make paid API calls. Set `OPENAI_API_KEY` for generation and
`JUDGE_API_KEY` for the official xAI judge in the calling environment, or keep
local secrets in `.beaker/.env` and explicitly load that file in your shell.
The Python scripts do not automatically source `.env` files.

Generate and score one train task, using a new output directory outside the
checkout:

```bash
.beaker/.venv/bin/python .beaker/evaluate.py \
  --task human_resources_specialists/task1 \
  --run-dir /tmp/pi-jobbench-hr-run-001
```

To generate without calling the judge, append `--phase generate`. To score the
saved deliverables later, reuse the task and run directory with `--phase score`.
The scorer verifies the saved task and protocol before reusing an output.

Every generation creates a fresh `workspace/task_folder/`, `workspace/output/`,
and Pi HOME/config. Personal extensions and instructions do not influence this
benchmark path. The child process receives the candidate provider key, not the
judge or Beaker credentials. Ordinary `scripts/pi_harness.py` behavior is unchanged.

The run directory retains the deliverables, raw Pi transcript/result JSON,
stderr diagnostics, and a rubric-level `score.json`. An empty submission is a
normal zero. Provider, timeout, transport, malformed-judge, and infrastructure
failures propagate as errors instead of being turned into incorrect answers.

## Beaker contract and limitations

The loader stages only `task_folder/` files using `CaseFile`; task rubrics stay in
trusted `Case.expected`. Each `run_case` starts Node in the current candidate
checkout, then transports the binary deliverables in `CaseResult.context` before
cleaning its workspace. The scorer does not depend on an evaluator-local path.
The sample output contains the final response, file manifest, and native usage
statistics. Passing and failing rubric criteria become `CaseScore.checks`.

- Binary submissions currently have a 4 MiB total limit, leaving room for base64
  encoding inside Beaker's 8 MiB result envelope. Exceeding this limit is an
  explicit transport failure; submissions are never silently truncated.
- Evaluation configuration fixes the model, thinking level, and timeout. Local
  execution is not an adversarial sandbox: `cwd` and the prompt do not enforce
  filesystem or network isolation. Do not treat a raw-code optimization as
  hardened against deliberate evaluator access. Hosted isolation and model-call
  enforcement must be validated before an optimization launch.
- The trace includes an actual candidate-workflow stage. Individual Node model
  and tool calls are available in the saved native transcript, but are not yet
  exported as individual Beaker trace spans.
- Hosted image provisioning (modern Node, npm dependencies, catalog data, and
  the exact calculator runtime), provider routing, agent registration, and
  dataset upload are deliberately not completed by this local pilot.

No benchmark, judge, or optimization API calls are made by importing the integration.
