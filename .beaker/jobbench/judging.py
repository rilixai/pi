"""Structured rubric calls using JobBench's prompt and result layout (Apache-2.0).

The prompt and result layout follow vendor/jobbench/eval/judge.py. Extraction,
vision selection, JSON recovery, and score aggregation still use that module.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import httpx
from openai import APIError, APIStatusError, OpenAI, Stream, omit
from openai.types.chat import ChatCompletionChunk, ChatCompletionContentPartParam, ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import BaseModel, ConfigDict, ValidationError

from .data import Rubric

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/jobbench/eval"))
import judge as official_judge


MAX_JUDGE_ATTEMPTS = 3
# GLM 5.3 Flash spent the old 32k cap on reasoning alone for some rubrics; this still stops a looping answer.
MAX_JUDGE_COMPLETION_TOKENS = 64000
# A 32k-token GLM 5.3 Flash judgment took up to 275 s; leave room for a full 64k one.
JUDGE_REQUEST_TIMEOUT_SECONDS = 900
JUDGE_RETRY_DELAY_SECONDS = 2


class CriterionJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    index: int
    passed: bool
    reasoning: str
    evidence: str


class RubricJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    criteria_results: list[CriterionJudgment]
    rubric_passed: bool
    overall_reasoning: str


class InvalidJudgeResponse(ValueError):
    """A judge response that can be retried without changing the task or rubric."""


class InvalidCriterionSchema(InvalidJudgeResponse):
    """A parsed judgment with invalid criterion fields or inconsistent verdicts."""


def judge_rubric(
    index: int, rubric: Rubric, text: str, model: str, api_base: str, api_key: str,
    *, image_attachments: list[tuple[str, str]],
) -> dict[str, Any]:
    """Retry only this rubric, returning a validated result or raising after bounded attempts."""
    messages = rubric_messages(rubric, text, image_attachments)
    response_format = rubric_response_format(len(rubric.criterion))
    # Keep a single retry budget instead of multiplying adapter and SDK retries.
    last_error: Exception | None = None
    with OpenAI(base_url=api_base, api_key=api_key, max_retries=0, timeout=JUDGE_REQUEST_TIMEOUT_SECONDS) as client:
        for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
            try:
                stream = client.chat.completions.create(
                    model=model, messages=messages, response_format=response_format,
                    max_completion_tokens=MAX_JUDGE_COMPLETION_TOKENS,
                    # Gemini 3 recommends default sampling and GPT-6 rejects temperature; retain upstream sampling for other models.
                    temperature=omit if model.rsplit("/", 1)[-1].split(":", 1)[-1].startswith(("gemini-3", "gpt-6")) else 0.0,
                    # Beaker's gateway cuts a buffered reply at 280 s; a stream gets keep-alives and 870 s.
                    stream=True,
                )
                finish_reason, content, refusal = read_stream(stream)
                if finish_reason is None:
                    raise InvalidJudgeResponse("The judge stream ended without a finish reason")
                if finish_reason != "stop":
                    raise InvalidJudgeResponse(f"Incomplete judge response: finish_reason={finish_reason}")
                if refusal or not content:
                    raise InvalidJudgeResponse("The judge refused or returned an empty response")
                judgment = validate_judgment(content, len(rubric.criterion))
                return {
                    "index": index, "rubric": rubric.rubric, "weight": rubric.weight,
                    "result": {
                        "passed": judgment.rubric_passed,
                        "score": rubric.weight if judgment.rubric_passed else 0,
                        "criteria_count": len(rubric.criterion),
                        "criteria_passed": sum(item.passed for item in judgment.criteria_results),
                        "criteria_results": [
                            {**item.model_dump(), "criterion": rubric.criterion[item.index]}
                            for item in judgment.criteria_results
                        ],
                        "overall_reasoning": judgment.overall_reasoning,
                    },
                }
            except APIStatusError as error:
                # Invalid schema requests, credentials, and permissions cannot be fixed by retrying.
                if error.status_code not in (408, 409, 429) and error.status_code < 500:
                    raise RuntimeError(f"JobBench judge failed on rubric {index}: HTTP {error.status_code}: {error}") from error
                last_error = error
            # APIError also covers an error event sent after the stream started.
            except (APIError, httpx.TransportError, InvalidJudgeResponse) as error:
                last_error = error
            if attempt < MAX_JUDGE_ATTEMPTS:
                time.sleep(JUDGE_RETRY_DELAY_SECONDS)
    if isinstance(last_error, InvalidCriterionSchema):
        message = f"JobBench judge returned an invalid criterion schema for rubric {index}"
    else:
        message = f"JobBench judge failed on rubric {index}"
    raise RuntimeError(f"{message} after {MAX_JUDGE_ATTEMPTS} attempts: {last_error}") from last_error


def read_stream(stream: Stream[ChatCompletionChunk]) -> tuple[str | None, str, str]:
    """Join the first choice's streamed content and refusal, keeping its finish reason."""
    finish_reason: str | None = None
    content: list[str] = []
    refusal: list[str] = []
    with stream:
        for chunk in stream:
            for choice in chunk.choices:
                if choice.index != 0:
                    continue
                content.append(choice.delta.content or "")
                refusal.append(choice.delta.refusal or "")
                finish_reason = choice.finish_reason or finish_reason
    return finish_reason, "".join(content), "".join(refusal)


def rubric_response_format(criterion_count: int) -> ResponseFormatJSONSchema:
    """Constrain the wire response to typed fields and exactly one result per criterion."""
    schema = RubricJudgment.model_json_schema()
    schema["properties"]["criteria_results"].update(minItems=criterion_count, maxItems=criterion_count)
    schema["$defs"]["CriterionJudgment"]["properties"]["index"]["enum"] = list(range(criterion_count))
    return {"type": "json_schema", "json_schema": {"name": "jobbench_rubric", "strict": True, "schema": schema}}


def validate_judgment(content: str, criterion_count: int) -> RubricJudgment:
    """Keep upstream JSON recovery, then validate types and criterion order."""
    try:
        parsed, _ = official_judge.parse_judge_json(content)
    except ValueError as error:
        raise InvalidJudgeResponse(str(error)) from error
    try:
        judgment = RubricJudgment.model_validate(parsed)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors(include_input=False)[:4]
        )
        raise InvalidCriterionSchema(details) from error
    indices = [item.index for item in judgment.criteria_results]
    if indices != list(range(criterion_count)):
        raise InvalidCriterionSchema(f"Expected criterion indices 0..{criterion_count - 1} in order; received {indices}")
    return judgment


def rubric_messages(
    rubric: Rubric, text: str, image_attachments: list[tuple[str, str]],
) -> list[ChatCompletionMessageParam]:
    """Preserve the pinned upstream judge's system prompt, rubric prompt, and vision selection."""
    rubric_text = rubric.rubric
    criteria_list_text = "\n".join(f"Criterion {index}: {criterion}" for index, criterion in enumerate(rubric.criterion))
    criterion_count = len(rubric.criterion)
    attached_images = image_attachments if official_judge.rubric_needs_vision(rubric.model_dump()) else []
    vision_used = bool(attached_images)
    prompt = f"""You are an evaluation judge. Your task is to evaluate ALL criteria for a single rubric.

## Rubric Description
{rubric_text}

## Criteria to Evaluate (Judge ALL of them)
{criteria_list_text}

## Output Files Content
The following are the contents of all output files to evaluate:

{text}

## Evaluation Rules
- Evaluate EACH criterion listed above independently
- For each criterion: determine if it PASSES or FAILS
- Semantic matching is acceptable (you don't need exact wording match)
- Binary judgment for each criterion: PASS or FAIL only
- The rubric passes ONLY if ALL criteria pass

## Output Format
Return your judgment as a JSON object with EXACTLY this structure (no markdown, no extra text):
{{
  "criteria_results": [
    {{"index": 0, "passed": true/false, "reasoning": "...", "evidence": "..."}},
    {{"index": 1, "passed": true/false, "reasoning": "...", "evidence": "..."}}
  ],
  "rubric_passed": true/false,
  "overall_reasoning": "Summary of why the rubric passed or failed"
}}

IMPORTANT:
- criteria_results array must have exactly {criterion_count} items (one for each criterion)
- rubric_passed should be true ONLY if ALL criteria passed
- Include specific evidence from the output files{' and the attached images' if vision_used else ''}
"""
    if vision_used:
        user_content: str | list[ChatCompletionContentPartParam] = [{"type": "text", "text": prompt}]
        user_content.append({
            "type": "text",
            "text": f"\n## Attached Images ({len(attached_images)} file{'s' if len(attached_images) != 1 else ''})",
        })
        for image_index, (name, url) in enumerate(attached_images, start=1):
            user_content.append({"type": "text", "text": f"Image {image_index}: {name}"})
            user_content.append({"type": "image_url", "image_url": {"url": url}})
    else:
        user_content = prompt
    return [
        {"role": "system", "content": "You are an evaluation judge. You must return valid JSON only, with no markdown formatting or extra text."},
        {"role": "user", "content": user_content},
    ]
