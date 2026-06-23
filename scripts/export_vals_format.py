"""Export BenchFlow SkillsBench jobs into vals-format JSON."""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

SCORE_TYPES = {"score": {"unit": "percent", "description": "Mean SkillsBench task reward."}}
USAGE_COMPONENTS = [{"component": "generation.model"}, {"component": "generation.tools"}]
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".txt", ""}
SKILL_PATTERNS = [
    re.compile(
        r"(?:^|[^A-Za-z0-9_.-])/"
        r"(?:skills|root/\.claude/skills|root/\.codex/skills|workspace/\.agents/skills|app/\.agents/skills)/"
        r"([A-Za-z0-9][A-Za-z0-9_.-]*)"
    ),
    re.compile(
        r"(?:^|[^A-Za-z0-9_.-])"
        r"(?:environment/skills|\.agents/skills|\.claude/skills|\.codex/skills)/"
        r"([A-Za-z0-9][A-Za-z0-9_.-]*)"
    ),
]


def build_vals_format(jobs_dir: Path, *, run_id: str, dataset: str, model: str | None, harness: str | None) -> dict[str, Any]:
    tasks = [_task_result(path, dataset=dataset) for path in _rollout_dirs(jobs_dir)]
    scores = [task["extra"]["skillsbench"]["score"] for task in tasks]
    score_percent = (sum(scores) / len(scores) * 100.0) if scores else 0.0
    by_status = Counter(task["status"] for task in tasks)
    duration_total = sum(float(task["aggregated_metrics"].get("duration_seconds") or 0.0) for task in tasks)
    tool_total = sum(int(task["aggregated_metrics"]["tool_usage"].get("agent_tool_calls", 0)) for task in tasks)
    first_result = _first_result(jobs_dir)
    model_id = model or _string(first_result.get("model")) or "unknown"
    harness_id = harness or _string(first_result.get("agent_name")) or _string(first_result.get("agent")) or "unknown"

    return {
        "schema_version": "vals_format.v1.1",
        "run_id": run_id,
        "benchmark": "skillsbench",
        "status": "FINISHED",
        "score_types": SCORE_TYPES,
        "producer": {
            "dataset": {"name": "skillsbench", "split": dataset, "version": None, "source": None},
            "formatter": {"name": "skillsbench-export-vals-format", "version": None},
        },
        "subject": {"model_id": model_id, "harness_id": harness_id, "extra": {}},
        "results": {
            "full": {
                "scores": {"score": {"value": score_percent, "stderr": _standard_error(scores) * 100.0, "extra": {}}},
                "counts": {"total": len(tasks), "by_status": dict(by_status), "extra": {}},
                "aggregated_metrics": {
                    "total": {
                        "duration_seconds": duration_total,
                        "metadata": {},
                        "tool_usage": {"agent_tool_calls": tool_total},
                        "extra": {},
                    },
                    "average_per_task": {
                        "duration_seconds": duration_total / len(tasks) if tasks else 0.0,
                        "metadata": {},
                        "tool_usage": {"agent_tool_calls": round(tool_total / len(tasks)) if tasks else 0},
                        "extra": {},
                    },
                },
                "selection": None,
                "extra": {
                    "skillsbench": {
                        "dataset": dataset,
                        "resolved_tasks": sum(1 for score in scores if score > 0.0),
                        "errored_tasks": by_status.get("error", 0),
                        "mean_reward": score_percent / 100.0,
                        **_skill_summary(tasks),
                    }
                },
            }
        },
        "primary_population": "full",
        "usage_components": USAGE_COMPONENTS,
        "llms": [],
        "agents": [{"id": harness_id, "role": "agent", "llm_ids": [], "config": {}, "extra": {}}],
        "tasks": tasks,
        "artifacts": [],
        "extra": {"skillsbench": {"dataset": dataset}},
    }


def _rollout_dirs(jobs_dir: Path) -> list[Path]:
    return sorted({path.parent for path in jobs_dir.glob("**/result.json")})


def _first_result(jobs_dir: Path) -> dict[str, Any]:
    for rollout_dir in _rollout_dirs(jobs_dir):
        return _read_json(rollout_dir / "result.json")
    return {}


def _task_result(rollout_dir: Path, *, dataset: str) -> dict[str, Any]:
    result = _read_json(rollout_dir / "result.json")
    reward = _reward(result)
    score = reward if reward is not None else 0.0
    error = _string(result.get("error")) or _string(result.get("verifier_error"))
    status = "error" if error else "evaluated"
    duration = _number(result.get("timing", {}).get("total") if isinstance(result.get("timing"), dict) else None)
    tool_calls = _non_negative_int(result.get("n_tool_calls") or result.get("tool_calls"))

    task = {
        "task_id": _string(result.get("task_id")) or _string(result.get("task_name")) or rollout_dir.name,
        "category": None,
        "status": status,
        "scores": {"score": {"value": score * 100.0, "stderr": None, "extra": {}}},
        "aggregated_metrics": {
            "duration_seconds": duration,
            "metadata": {},
            "tool_usage": {"agent_tool_calls": tool_calls},
            "extra": {},
        },
        "evaluations": [],
        "extra": {
            "skillsbench": {"dataset": dataset, "resolved": score > 0.0, "score": score, **_skill_references(rollout_dir)}
        },
    }
    if error:
        task["error"] = {"message": error}
    return task


def _skill_references(rollout_dir: Path) -> dict[str, Any]:
    for path in rollout_dir.glob("**/skill_references.json"):
        return _coerce_skill_references(_read_json(path))

    skills = Counter()
    for path in rollout_dir.rglob("*"):
        if not path.is_file() or "vals_format" in path.parts or path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            line_skills = set()
            for pattern in SKILL_PATTERNS:
                line_skills.update(match.group(1).strip("`'\"),.:;") for match in pattern.finditer(line))
            for skill in line_skills:
                if skill:
                    skills[skill] += 1
    return {"skill_calls": sum(skills.values()), "skills_used": dict(sorted(skills.items()))}


def _coerce_skill_references(payload: dict[str, Any]) -> dict[str, Any]:
    skills_used = {
        str(skill): value
        for skill, value in payload.get("skills_used", {}).items()
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    }
    skill_calls = payload.get("skill_calls")
    if not isinstance(skill_calls, int) or isinstance(skill_calls, bool) or skill_calls < 0:
        skill_calls = sum(skills_used.values())
    return {"skill_calls": skill_calls, "skills_used": skills_used}


def _skill_summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    skill_calls = 0
    skill_breakdown: Counter[str] = Counter()
    for task in tasks:
        skillsbench = task["extra"]["skillsbench"]
        skill_calls += _non_negative_int(skillsbench.get("skill_calls"))
        skill_breakdown.update(
            {
                str(skill): count
                for skill, count in skillsbench.get("skills_used", {}).items()
                if isinstance(count, int) and not isinstance(count, bool) and count > 0
            }
        )
    return {
        "skill_calls": skill_calls,
        "skill_task_count": len(tasks),
        "skill_calls_per_task": skill_calls / len(tasks) if tasks else 0.0,
        "skill_breakdown": dict(sorted(skill_breakdown.items())),
    }


def _reward(result: dict[str, Any]) -> float | None:
    for key in ("reward", "score"):
        value = _number(result.get(key))
        if value is not None:
            return value
    rewards = result.get("rewards")
    if isinstance(rewards, dict):
        return _number(rewards.get("reward", next(iter(rewards.values()), None)))
    return _number(rewards)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _standard_error(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    return math.sqrt(variance / len(scores))


def _self_test() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        rollout_dir = Path(temp_dir) / "jobs" / "run-1"
        trajectory_dir = rollout_dir / "trajectory"
        trajectory_dir.mkdir(parents=True)
        (rollout_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "pdf-task",
                    "rewards": {"reward": 0.5},
                    "agent_name": "OpenHands",
                    "model": "openai/gpt-5.5",
                    "n_tool_calls": 7,
                    "timing": {"total": 12.0},
                }
            ),
            encoding="utf-8",
        )
        (trajectory_dir / "acp_trajectory.jsonl").write_text(
            "read /skills/pdf/SKILL.md and environment/skills/xlsx/SKILL.md\nlater used /root/.codex/skills/pdf/reference.md\n",
            encoding="utf-8",
        )

        payload = build_vals_format(Path(temp_dir) / "jobs", run_id="run-1", dataset="with-skills", model=None, harness=None)
        full = payload["results"]["full"]
        assert payload["subject"] == {"model_id": "openai/gpt-5.5", "harness_id": "OpenHands", "extra": {}}
        assert full["scores"]["score"]["value"] == 50.0
        assert full["aggregated_metrics"]["average_per_task"]["duration_seconds"] == 12.0
        assert full["aggregated_metrics"]["total"]["tool_usage"] == {"agent_tool_calls": 7}
        assert full["extra"]["skillsbench"]["skill_calls"] == 3
        assert full["extra"]["skillsbench"]["skill_calls_per_task"] == 3
        assert full["extra"]["skillsbench"]["skill_breakdown"] == {"pdf": 2, "xlsx": 1}
        assert payload["tasks"][0]["extra"]["skillsbench"]["skills_used"] == {"pdf": 2, "xlsx": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BenchFlow SkillsBench jobs to vals-format JSON.")
    parser.add_argument("jobs_dir", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, default=Path("vals_format.json"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset", default="default")
    parser.add_argument("--model", default=None)
    parser.add_argument("--harness", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.jobs_dir is None:
        parser.error("jobs_dir is required unless --self-test is set")

    payload = build_vals_format(
        args.jobs_dir,
        run_id=args.run_id or args.jobs_dir.resolve().name,
        dataset=args.dataset,
        model=args.model,
        harness=args.harness,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
