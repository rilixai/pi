"""Create a small JobBench-style workspace and ask Pi to produce a deliverable.

Run: python3 /path/to/pi/scripts/run_pi_workspace.py --workspace /tmp/pi-task --model MODEL_ID
Requires OPENAI_API_KEY. The new workspace and its outputs are kept on disk.
"""

import argparse
import json
from pathlib import Path

from pi_harness import PiRunError, run_pi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="A new directory to create")
    parser.add_argument("--model", required=True, help="Exact OpenAI API model ID")
    parser.add_argument("--timeout", type=float, default=300, help="Time budget in seconds")
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    # Refuse to overwrite an existing task or a previous run's deliverables.
    workspace.mkdir(parents=True, exist_ok=False)
    task_folder = workspace / "task_folder"
    task_folder.mkdir()
    (workspace / "output").mkdir()
    (workspace / "AGENTS.md").write_text(
        "Keep the reference files unchanged. Write final deliverables under output/.\n",
        encoding="utf-8",
    )
    skill_dir = workspace / ".pi/skills/revenue-analysis"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: revenue-analysis\n"
        "description: Calculate revenue from sales CSV files and write an analysis.\n"
        "---\nRead the input CSV with Python's csv module. Revenue is units * unit_price.\n"
        "Show per-product revenue and the grand total in the final Markdown report.\n",
        encoding="utf-8",
    )
    (workspace / ".pi/APPEND_SYSTEM.md").write_text(
        "Use the revenue-analysis skill for sales analysis tasks.\n", encoding="utf-8"
    )
    (task_folder / "TASK_INSTRUCTIONS.txt").write_text(
        "Analyze sales.csv and save a Markdown revenue report to output/report.md.\n"
        "Reference Files: task_folder/sales.csv\n",
        encoding="utf-8",
    )
    (task_folder / "sales.csv").write_text(
        "product,units,unit_price\nA,4,12.50\nB,3,20.00\n", encoding="utf-8"
    )
    try:
        result = run_pi(
            prompt=(
                "Read task_folder/TASK_INSTRUCTIONS.txt and the reference files. "
                "Complete the task and save the final deliverables under output/."
            ),
            cwd=workspace,
            model=args.model,
            append_system_prompt="Write the report in English.",
            timeout=args.timeout,
        )
    except PiRunError as error:
        parser.exit(1, f"{error}\n{error.stderr}\nWorkspace kept at {workspace}\n")
    print(result["response"])
    print(json.dumps(result["stats"], ensure_ascii=False, indent=2))
    print(f"Workspace and deliverables kept at {workspace}")


if __name__ == "__main__":
    main()
