# Pi harness optimization on JobBench: five-task pilot

This integration evaluates Pi's TypeScript source harness on five real tasks from
the JobBench `main` split. The configuration selects the `pi-jobbench` Beaker
integration. Importing the integration does not create a dataset or launch a run.

The editable surface includes Pi's agent loop, agent session, system prompt,
message conversion, compaction, and tool implementations. The Python bridge,
source resolver, provider adapters, benchmark, judge, and evaluation policy remain
outside that surface. See `beaker_integration.py` for the exact source paths.

## Fixed protocol

`jobbench/pilot.json` records the protocol:

- Dataset: `JobBench/job-bench@d43a66575635da0d3f6cb9a20976dc1a60f3ffc9`.
- Evaluator: `Job-Bench/job-bench-eval@660b33a86ed032559df0b2fdc1ad38cf7ceb83f3`.
- Pi model: `openai:gpt-6-sol`, thinking level `medium`, one-hour wall-clock budget.
- Judge: `openrouter:x-ai/grok-4.7`, independent of the Pi model.
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

Use Linux with Python 3.12+, a local Docker daemon, and `uv`. On macOS, run
the evaluator in a Linux VM: the relay needs Unix socket bind-mount support.

```bash
cd /path/to/pi
UV_PROJECT_ENVIRONMENT=.beaker/.venv uv sync --locked --group dev
npm ci --ignore-scripts
npm run hydrate:model-data
```

Python dependencies are evaluation tooling; Pi's normal Node runtime does not
import Beaker. The Node subprocess uses this checkout's `source-resolver.ts` and
`local-runner-cli.ts`, so edited candidate TypeScript is loaded without a build.
No npm registry version of Pi is used as the evaluated harness. JobBench generation
requires a running Docker daemon; it never falls back to an unsandboxed process.

The JobBench adapter uses Node.js 24.19.0 supplied by the pinned
[`nodejs-wheel-binaries`](https://pypi.org/project/nodejs-wheel-binaries/24.19.0/)
evaluation dependency. It does not depend on the system's Node version.
The adapter builds a runtime image from the candidate's current TypeScript
sources, pinned Python lockfile, and npm lockfile. Dependencies are installed in
the image with `uv sync --frozen --no-dev` and
`npm ci --ignore-scripts --include=dev`. Images are cached by source content;
each task still gets a fresh container. Setup commands have a ten-minute timeout
and receive no provider API key. Only runtime source and dependency manifests
enter the build context; `.beaker`, Git history, datasets, and local secrets do not.

When building an uncached image, setup downloads the immutable revision in
`nix/model-catalog.json`, verifies its SHA-256, and uses Pi's existing offline
hydrator. It never replaces candidate source with an installed Pi package or
fetches a moving model catalog. npm and hydration failures include bounded logs
in the case error. Pi failures include its structured error or stderr before
the temporary workspace is removed, with the candidate API key redacted.

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

Hosted setup installs `docker.io` through `apt_install`. Before loading cases,
`JobBenchSetup.prepare_run` connects to Docker or starts an attempt-owned daemon
inside Beaker's Modal VM, builds the unchanged upstream calculator Dockerfile,
and verifies formula recalculation with the official worker. It also prepares
the seed Pi image before rollouts; modified candidates get their own image. Startup/build
failures therefore stop setup before paid rollouts. The daemon receives no
provider or Beaker API credentials and is stopped when the attempt closes.
The trusted image build uses host networking; workbook calculation retains the
official read-only, offline container restrictions. Local structural smoke
does not start Docker, and local scoring still uses the Docker setup above.

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

Every generation creates a dedicated container with this filesystem:

| Container path | Access and lifetime |
| --- | --- |
| `/workspace/task_folder/` | Current task's references only, read-only bind mount |
| `/workspace/output/` | Writable bind mount, collected as final deliverables |
| `/workspace/scratch/`, `/tmp/` | Writable temporary files, discarded with the container |
| `/home/pi/` | Fresh temporary Pi configuration and session state |
| `/opt/pi/` | Candidate harness sources and dependencies, read-only |

The container runs as UID 65534 with no capabilities, no privilege escalation,
a read-only root, and separate PID and network namespaces. It has no Docker
socket, evaluator files, rubric files, other tasks, or host credentials. Limits
are 2 CPUs, 4 GiB RAM, 256 processes, and bounded temporary filesystems.
Personal extensions and instructions do not influence this benchmark path.

Network access is disabled. A per-task Unix socket relays only OpenAI Responses
and Chat Completions requests for `gpt-6-sol`. The relay rejects hosted tools,
remote image URLs, persistent provider resources, and other endpoints/models.
Inline images and native local Pi tools remain supported. The real OpenAI key
stays in the evaluator; the container receives a nonsecret placeholder. Hosted
requests still pass through Beaker's existing provider proxy and accounting.
Tasks requiring external browsing or downloading packages during execution are
therefore unsupported. Install additional tools in the runtime Dockerfile instead.

On success, error, timeout, or cancellation, the adapter force-removes its own
container, stops the relay, and retains only the parent-side output/log files.
Ordinary `scripts/pi_harness.py` behavior is unchanged.

The run directory retains the deliverables, raw Pi transcript/result JSON,
stderr diagnostics, and a rubric-level `score.json`. An empty submission is a
normal zero. Provider, timeout, transport, malformed-judge, and infrastructure
failures propagate as errors instead of being turned into incorrect answers.

## Beaker contract and limitations

The loader stages only `task_folder/` files using `CaseFile`; task rubrics stay in
trusted `Case.expected`. Each `run_case` loads the current candidate's native Pi
sources inside its container, then transports binary deliverables in `CaseResult.context` before
cleaning its workspace. The scorer does not depend on an evaluator-local path.
The sample output contains the final response, file manifest, and native usage
statistics. Passing and failing rubric criteria become `CaseScore.checks`.

- Binary submissions currently have a 4 MiB total limit, leaving room for base64
  encoding inside Beaker's 8 MiB result envelope. Exceeding this limit is an
  explicit transport failure; submissions are never silently truncated.
- Evaluation configuration fixes the model, thinking level, and timeout. The
  container enforces filesystem and network isolation rather than relying on
  `cwd` or prompt instructions. This uses the host's Linux container boundary;
  the Docker daemon and kernel remain trusted infrastructure.
- The trace includes an actual candidate-workflow stage. Individual Node model
  and tool calls are available in the saved native transcript, but are not yet
  exported as individual Beaker trace spans.
- Hosted Node startup, npm dependencies, and pinned catalog hydration are
  prepared by the adapter. End-to-end model execution and scoring still need
  hosted validation. Hosted setup prepares the exact upstream calculator
  container described above; it requires a VM sandbox that supports Docker.

No benchmark, judge, or optimization API calls are made by importing the integration.
