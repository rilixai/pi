"""Score transported deliverables with JobBench's unchanged official judge."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from beaker import CaseScore, Check, scoring_inference_target

from .data import Expected, PILOT
from .execution import MAX_DELIVERABLE_BYTES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/jobbench/eval"))
import judge as official_judge


MAX_JUDGE_WORKERS = 8


def restore_submission(files: list[dict[str, Any]], directory: Path) -> None:
    """Validate the candidate's binary transport before writing scorer-owned files."""
    total = 0
    seen = set()
    for item in files:
        name = item["name"]
        if not isinstance(name, str) or "\\" in name or any(part in ("", ".", "..") for part in name.split("/")):
            raise ValueError("Unsafe submission path")
        if name in seen:
            raise ValueError("Duplicate submission path")
        seen.add(name)
        if len(item["content_base64"]) > 4 * ((MAX_DELIVERABLE_BYTES + 2) // 3):
            raise ValueError("Submission payload exceeds the transport limit")
        content = base64.b64decode(item["content_base64"], validate=True)
        total += len(content)
        if total > MAX_DELIVERABLE_BYTES or len(content) != item["size"]:
            raise ValueError("Submission size is invalid")
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError("Submission hash is invalid")
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def score_submission(
    expected: Expected, files: list[dict[str, Any]], *, report_dir: Path | None = None,
) -> CaseScore:
    """Keep official rubric weighting; treat judge failures as failures, never zeros."""
    with tempfile.TemporaryDirectory(prefix="pi-jobbench-score-") as temporary:
        directory = Path(temporary)
        restore_submission(files, directory)
        rubric_results = []
        if files:
            extraction: dict[str, Any] = {}
            text = official_judge.extract_all_file_contents(directory, extraction)
            images = official_judge.collect_image_attachments(directory)
            target = scoring_inference_target()
            if target is None:
                api_base = "https://api.x.ai/v1"
                api_key = os.environ.get("JUDGE_API_KEY", "")
                model = "grok-4.7"
                if not api_key.strip():
                    raise RuntimeError("JUDGE_API_KEY is required for local Grok scoring")
            else:
                if target.model != PILOT["scorer_model"]:
                    raise ValueError(f"The scorer must remain {PILOT['scorer_model']}")
                api_base, api_key, model = target.base_url, target.api_key, target.model
            executor = ThreadPoolExecutor(max_workers=MAX_JUDGE_WORKERS)
            try:
                futures = [
                    executor.submit(
                        official_judge.judge_rubric,
                        index, rubric.model_dump(), text, model, api_base, api_key,
                        image_attachments=images,
                    )
                    for index, rubric in enumerate(expected.rubrics)
                ]
                # Consume in rubric order so outputs and the first reported failure stay stable.
                for index, rubric in enumerate(expected.rubrics):
                    result, debug = futures[index].result()
                    if debug.get("api_exit_code") != 0 or debug.get("error"):
                        raise RuntimeError(f"JobBench judge failed on rubric {index}: {debug.get('error')}")
                    parsed, _ = official_judge.parse_judge_json(debug["raw_response"])
                    criteria = parsed.get("criteria_results")
                    if (
                        not isinstance(criteria, list) or len(criteria) != len(rubric.criterion)
                        or any(not isinstance(item, dict) for item in criteria)
                        or any(type(item.get("index")) is not int for item in criteria)
                        or [item.get("index") for item in criteria] != list(range(len(rubric.criterion)))
                        or any(type(item.get("passed")) is not bool for item in criteria)
                        or type(parsed.get("rubric_passed")) is not bool
                        or parsed["rubric_passed"] != all(item["passed"] for item in criteria)
                    ):
                        raise RuntimeError(f"JobBench judge returned an invalid criterion schema for rubric {index}")
                    rubric_results.append(result)
            finally:
                # In-flight calls cannot be interrupted; cancel queued work and ignore later results on failure.
                executor.shutdown(wait=False, cancel_futures=True)
        else:
            rubric_results = [
                official_judge.build_failed_rubric_result(index, rubric.model_dump(), "No output files produced.")
                for index, rubric in enumerate(expected.rubrics)
            ]
        scorecard = official_judge.build_scorecard(rubric_results)
        if report_dir is not None:
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "score.json").write_text(json.dumps({
                "protocol": PILOT, "scorecard": scorecard, "rubrics": rubric_results,
            }, indent=2), encoding="utf-8")
        checks = tuple(
            Check(
                name=f"Rubric {rubric['index'] + 1}, criterion {criterion['index'] + 1}",
                description=criterion["criterion"],
                verdict="pass" if criterion["passed"] else "fail",
                message=criterion.get("reasoning") or rubric["result"]["overall_reasoning"],
                group=rubric["rubric"],
                informational=rubric["weight"] == 0,
            )
            for rubric in rubric_results for criterion in rubric["result"]["criteria_results"]
        )
        return CaseScore(
            objective=float(scorecard["normalized_score"]),
            field_scores={
                "jobbench_weighted_score": float(scorecard["normalized_score"]),
                "rubric_pass_rate": float(scorecard["pass_rate"]),
            },
            checks=checks,
        )


async def score_async(expected: Expected, files: list[dict[str, Any]]) -> CaseScore:
    return await asyncio.to_thread(score_submission, expected, files)
