# Call the local Pi harness from Python

`scripts/pi_harness.py` runs the coding agent from this source checkout in a
workspace you supply. It uses the existing SDK, tools, agent loop, context
management, and resource discovery. There is no terminal UI or HTTP server.
The wrapper needs Python 3.10+ and uses only its standard library.

## Prepare the checkout

With Node.js 22.19+ installed, run once from the Pi repository root:

```bash
npm ci --ignore-scripts
npm run hydrate:model-data
```

Hydration downloads the ignored model catalog data required by source imports.
There is no build step. Every invocation uses the current TypeScript sources,
including internal workspace packages, through the existing source resolver.
Changes to the harness take effect on the next invocation.

Set `OPENAI_API_KEY` in the Python caller's environment. The runner selects the
`openai` provider and uses this key even if Pi has a saved subscription login.
Pass the exact API model ID available to your account and Pi's catalog; for
example, `gpt-6-sol`. No model or provider fallback is performed.

## Call from any working directory

Make the checkout's `scripts` directory importable, for example with
`PYTHONPATH=/absolute/path/to/pi/scripts`, then:

```python
from pi_harness import PiRunError, PiRunTimeout, run_pi

result = run_pi(
    prompt=(
        "Read task_folder/TASK_INSTRUCTIONS.txt and its reference files. "
        "Complete the task. Save final deliverables under output/."
    ),
    cwd="/absolute/path/to/workspace",
    model="gpt-6-sol",
    skill_paths=["/absolute/path/to/extra-skill"],  # optional
    append_system_prompt="Explain conflicting sources in the final report.",  # optional
    timeout=3600,  # optional, seconds; default: no deadline
)

print(result["response"])
print(result["stats"]["tokens"])
print(result["stats"]["toolCalls"])
```

`cwd` must already exist. The Node process and the SDK both use it as their
working directory. The runner's absolute path is resolved from the Python module,
so the checkout and the workspace may be in completely different directories.
Neither needs to be the Python caller's current directory. `cwd` selects the
working directory; it does not create a filesystem sandbox.

Each call creates a fresh session and waits for the full task, including tool
calls, automatic recovery, and queued work. The wrapper never copies or removes
the workspace or its deliverables. Native Pi session persistence is retained.

## Workspace resources

The normal SDK configuration applies, including personal configuration from
`~/.pi/agent` (or `PI_CODING_AGENT_DIR`), workspace settings, context files, skills,
and discovered extensions. Project resources use the SDK's normal trusted-project
default; the wrapper does not open an interactive trust dialog. See
[Configuration](configuration.md) for the native discovery and precedence rules.

For example:

```text
workspace/
├── AGENTS.md
├── .pi/
│   ├── APPEND_SYSTEM.md
│   └── skills/review/SKILL.md
├── task_folder/
│   ├── TASK_INSTRUCTIONS.txt
│   └── reference.csv
└── output/
```

`skill_paths` adds skill directories or files to the discovered set. Relative
paths resolve from `cwd`; they do not replace workspace or personal skills. Pi
keeps its usual lazy skill loading: it advertises skills and reads their full
instructions when needed. Use `/skill:name ...` in `prompt` to invoke one
explicitly.

`append_system_prompt` adds literal text after the discovered system-prompt
complements, including `APPEND_SYSTEM.md`. To supply another file, pass
`Path("/path/to/instructions.md").read_text(encoding="utf-8")`. It does not replace
the native prompt or the discovered complements.

## Results and errors

The result contains:

- `response`: final assistant text, which may be empty for a file-producing task.
- `messages`: native session messages, including tool calls and their results.
- `stats`: native `getSessionStats()` output, including token totals, cost,
  tool-call count, session ID, and session file path.
- `provider`, `model`, `cwd`: the selected OpenAI model and workspace.

The subprocess protocol is one JSON request on stdin and one JSON envelope on
stdout: `{ "ok": true, "result": ... }` or `{ "ok": false, "error": { "name":
"...", "message": "..." } }`. Console and extension output is redirected to
stderr to keep stdout parseable.

Startup errors, provider failures, aborted responses, and token-limit truncation
raise `PiRunError`. Its `stderr` and `returncode` attributes contain subprocess
diagnostics. Invalid Python arguments raise standard Python exceptions.

`timeout` includes Node startup and the task. On expiration, the wrapper raises
`PiRunTimeout`, signals the Node process group, and allows five seconds for
cleanup before forcing termination. The runner also kills detached shell groups
tracked by Pi and disposes its session and extensions. Windows uses `taskkill /T
/F`. Python interruption also triggers cleanup. Existing workspace files remain
available after failures and timeouts.

## Small file-producing example

Run this from any directory, using a workspace path that does not yet exist:

```bash
python3 /absolute/path/to/pi/scripts/run_pi_workspace.py \
  --workspace /tmp/pi-revenue-task \
  --model gpt-6-sol \
  --timeout 300
```

The example creates a task directory, CSV reference data, local instructions,
a skill, and an output directory, then calls `run_pi`. It retains everything at
the supplied path. This resembles a JobBench task layout; it does not download
JobBench, provision its dependencies, or run its judge.
