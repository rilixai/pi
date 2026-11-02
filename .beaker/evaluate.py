"""Run one pinned JobBench task locally with Pi, then the official rubric judge.

This command makes paid API calls. Use structural Beaker smoke for a no-LLM check.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from jobbench.data import PILOT, rows, stage_inputs
from jobbench.execution import rollout_model, run_task
from jobbench.scoring import score_submission


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=PILOT["train"] + PILOT["test"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="A new directory to retain all outputs")
    parser.add_argument("--phase", choices=("generate", "score", "all"), default="all")
    parser.add_argument("--model", help="OpenAI model ID; defaults to pilot.json agent_model for generation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    selected = next(row for split in rows().values() for row in split if row.input.task == args.task)
    directory = args.run_dir.expanduser().resolve()
    if directory.is_relative_to(root):
        parser.error("Use a run directory outside the Pi checkout")
    if args.phase in ("generate", "all"):
        directory.mkdir(parents=True, exist_ok=False)
        inputs = directory / "inputs"
        stage_inputs(selected.input, inputs)
        result = await run_task(selected.input, inputs=inputs, root=root, run_dir=directory, model=args.model)
    else:
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        saved_model = result["protocol"].get("agent_model")
        expected_protocol = {
            **PILOT, "agent_model": f"openai:{rollout_model(args.model if args.model is not None else saved_model)}",
        }
        # Re-score the same deliverables with the current judge, while retaining
        # their original rollout model and checking the benchmark configuration.
        saved_protocol = {**result["protocol"], "scorer_model": PILOT["scorer_model"]}
        if saved_protocol != expected_protocol or result["task"] != args.task:
            raise ValueError("Saved run uses a different task or protocol")
    print(result["response"])
    if args.phase in ("score", "all"):
        score = await asyncio.to_thread(
            score_submission, selected.expected, result["files"], report_dir=directory,
            protocol={**result["protocol"], "scorer_model": PILOT["scorer_model"]},
        )
        print(json.dumps({"objective": score.objective, "field_scores": dict(score.field_scores)}, indent=2))
    print(f"Run retained at {directory}")


if __name__ == "__main__":
    asyncio.run(main())
