"""Helpers for SkillsBench vals-format final-score metadata."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

SCORE_TYPES = {"score": {"unit": "percent", "description": "Mean SkillsBench task reward.", "extra": {}}}
USAGE_COMPONENTS = [{"component": "generation.model"}]


def build_final_score_metadata(evaluation_results: dict[str, Any], dataset: str | None) -> tuple[float, dict[str, Any]]:
    task_rows = [_vals_format_task(task_id, result, dataset) for task_id, result in evaluation_results.items()]
    scores = [task["extra"]["skillsbench"]["score"] for task in task_rows]
    by_status = Counter(task["status"] for task in task_rows)
    total = len(evaluation_results)
    mean_reward = sum(scores) / total if total else 0.0
    score = mean_reward * 100.0

    return score, {
        "dataset": dataset or "default",
        "total_tasks": total,
        "resolved_tasks": sum(1 for task_score in scores if task_score > 0.0),
        "errored_tasks": by_status.get("error", 0),
        "mean_reward": mean_reward,
        "score_scale": "0-100",
        "score_types": SCORE_TYPES,
        "results": {
            "full": {
                "scores": {"score": {"value": score, "stderr": _standard_error(scores) * 100.0, "extra": {}}},
                "counts": {"total": total, "by_status": dict(by_status), "extra": {}},
                "aggregated_metrics": _rollup_metrics(task_rows),
                "selection": None,
                "extra": {},
            }
        },
        "primary_population": "full",
        "usage_components": USAGE_COMPONENTS,
        "tasks": task_rows,
        "extra": {"skillsbench": {"dataset": dataset or "default"}},
    }


def _vals_format_task(task_id: str, result: Any, dataset: str | None) -> dict[str, Any]:
    result_mapping, payload, wrapper_error = _unwrap_result(result)
    if payload is None:
        score = 0.0
        status = "error"
        metadata: dict[str, Any] = {}
        error = wrapper_error or "Unexpected result shape"
    else:
        score = bounded_score(payload.get("score", payload.get("reward", 0.0)))
        error = wrapper_error or _string_or_none(payload.get("verifier_error"))
        status = "error" if error else "evaluated"
        raw_metadata = payload.get("metadata", {})
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}

    task = {
        "task_id": task_id,
        "category": metadata.get("category"),
        "tags": [],
        "status": status,
        "scores": {"score": {"value": score * 100.0, "stderr": None, "extra": {}}},
        "aggregated_metrics": _metric_group_from_result(result_mapping, payload),
        "evaluations": [],
        "turns": [],
        "extra": {
            "skillsbench": {
                "dataset": metadata.get("dataset", dataset or "default"),
                "task_set": metadata.get("task_set"),
                "category": metadata.get("category"),
                "has_skills": metadata.get("has_skills"),
                "skills_injected": metadata.get("skills_injected"),
                "resolved": score > 0.0,
                "score": score,
            }
        },
    }
    if error:
        task["error"] = {"message": str(error)}
    return task


def _unwrap_result(result: Any) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, str | None]:
    if not isinstance(result, Mapping):
        return None, None, "Unexpected result shape"

    nested = result.get("result")
    payload = nested if isinstance(nested, Mapping) else result
    error = _string_or_none(result.get("error")) or _string_or_none(payload.get("verifier_error"))
    return result, payload, error


def _metric_group_from_result(result: Mapping[str, Any] | None, payload: Mapping[str, Any] | None) -> dict[str, Any]:
    metric_group = _empty_metric_group()
    if payload is None:
        return metric_group

    metadata = _mapping(payload.get("metadata"))
    task_breakdown = _mapping(payload.get("task_breakdown")) or _mapping(
        result.get("task_breakdown") if result else None
    )

    duration = _first_number(
        metadata.get("duration_seconds"),
        metadata.get("latency_sec"),
        metadata.get("latency"),
        task_breakdown.get("agent_run_duration"),
        task_breakdown.get("sandbox_run_duration"),
    )
    if duration is not None:
        metric_group["duration_seconds"] = duration
        metric_group["effective_duration_seconds"] = duration

    cost = _cost_metadata(metadata.get("cost")) or _cost_metadata(payload.get("total_cost"))
    if cost is not None:
        metric_group["metadata"]["cost"] = cost

    return metric_group


def _rollup_metrics(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    total = _empty_metric_group()
    average = _empty_metric_group()
    durations: list[float] = []
    costs: list[float] = []

    for task in tasks:
        metrics = task["aggregated_metrics"]
        duration = _number_or_none(metrics.get("effective_duration_seconds")) or _number_or_none(
            metrics.get("duration_seconds")
        )
        if duration is not None:
            durations.append(duration)

        metadata = _mapping(metrics.get("metadata"))
        cost_total = _cost_total(metadata.get("cost"))
        if cost_total and cost_total > 0:
            costs.append(cost_total)

    if durations:
        total_duration = sum(durations)
        average_duration = total_duration / len(durations)
        total["duration_seconds"] = total_duration
        total["effective_duration_seconds"] = total_duration
        average["duration_seconds"] = average_duration
        average["effective_duration_seconds"] = average_duration

    if costs:
        total["metadata"]["cost"] = {"total": sum(costs)}
        average["metadata"]["cost"] = {"total": sum(costs) / len(costs)}

    return {"total": total, "average_per_task": average}


def _empty_metric_group() -> dict[str, Any]:
    return {
        "duration_seconds": None,
        "retry_overhead_seconds": 0.0,
        "effective_duration_seconds": None,
        "tool_duration_seconds": 0.0,
        "metadata": {},
        "tool_usage": {},
        "extra": {},
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _number_or_none(value: Any) -> float | None:
    return float(value) if _is_number(value) else None


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _number_or_none(value)
        if number is not None:
            return number
    return None


def _cost_metadata(value: Any) -> float | dict[str, float] | None:
    if isinstance(value, Mapping):
        total = _cost_total(value)
        return {"total": total} if total is not None else None
    number = _number_or_none(value)
    if number is not None:
        return number
    return None


def _cost_total(value: Any) -> float | None:
    if isinstance(value, Mapping):
        return _number_or_none(value.get("total"))
    return _number_or_none(value)


def _standard_error(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    return math.sqrt(variance / len(scores))


def bounded_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return min(1.0, max(0.0, score))
