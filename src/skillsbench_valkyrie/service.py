"""Expose SkillsBench tasks through Valkyrie's BenchmarkService contract."""

from __future__ import annotations

import io
import json
import math
import os
import posixpath
import re
import shlex
import tarfile
import tomllib
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from benchmark_service import BenchmarkService, ImageSource, Resources, Sandbox, SnapshotSource
from benchmark_service.schemas import (
    EvaluateResponseRequest,
    FinalScoreResult,
    RetrieveTaskResponse,
    StreamChunk,
    StreamMessageChunk,
    StreamResultChunk,
)
from benchmark_service.v1_schemas import V1Task

REPO_ROOT_ENV = "SKILLSBENCH_REPO_ROOT"
IMAGE_MANIFEST_ENV = "SKILLSBENCH_VALKYRIE_IMAGE_MANIFEST"
DEFAULT_IMAGE_ENV = "SKILLSBENCH_VALKYRIE_DEFAULT_IMAGE"
UPLOAD_ENV_ASSETS_ENV = "SKILLSBENCH_VALKYRIE_UPLOAD_ENV_ASSETS"

DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_CWD = "/root"
PROBLEM_FILENAME = "instruction.md"
SKILLS_DIR = "/skills"
TESTS_DIR = "/tests"
LOGS_DIR = "/logs"
VERIFIER_DIR = "/logs/verifier"

_WORKDIR_RE = re.compile(r"^\s*WORKDIR\s+(.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_dir: Path
    instruction: str
    config: dict[str, Any]
    task_set: Literal["default", "extra"]

    @property
    def environment_dir(self) -> Path:
        return self.task_dir / "environment"

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / "tests"

    @property
    def skills_dir(self) -> Path:
        return self.environment_dir / "skills"

    @property
    def has_skills(self) -> bool:
        return self.skills_dir.is_dir() and any(self.skills_dir.iterdir())

    @property
    def metadata(self) -> dict[str, Any]:
        raw = self.config.get("metadata", {})
        return raw if isinstance(raw, dict) else {}

    @property
    def environment(self) -> dict[str, Any]:
        raw = self.config.get("environment", {})
        return raw if isinstance(raw, dict) else {}

    @property
    def agent(self) -> dict[str, Any]:
        raw = self.config.get("agent", {})
        return raw if isinstance(raw, dict) else {}

    @property
    def verifier(self) -> dict[str, Any]:
        raw = self.config.get("verifier", {})
        return raw if isinstance(raw, dict) else {}

    @property
    def agent_timeout(self) -> float | None:
        value = self.agent.get("timeout_sec")
        return float(value) if isinstance(value, int | float) else None

    @property
    def verifier_timeout(self) -> float | None:
        value = self.verifier.get("timeout_sec")
        return float(value) if isinstance(value, int | float) else None


class SkillsBenchBenchmarkService(BenchmarkService):
    """Valkyrie adapter for local SkillsBench task directories."""

    async def load_datasets(self) -> dict[str, dict[str, TaskSpec]]:
        root = _repo_root()
        default_tasks = _discover_tasks(root / "tasks", task_set="default")
        extra_tasks = _discover_tasks(root / "tasks-extra", task_set="extra")

        return {
            "default": default_tasks,
            "with-skills": default_tasks,
            "extra": extra_tasks,
            "extra-with-skills": extra_tasks,
        }

    async def list_tasks(self, dataset: str | None = None) -> list[V1Task]:
        return [
            V1Task(
                id=task.task_id,
                question=task.instruction,
                timeout=task.agent_timeout,
                category=task.metadata.get("category"),
                subcategory=task.metadata.get("subcategory"),
                difficulty=task.metadata.get("difficulty"),
                task_type=task.metadata.get("task_type"),
                modality=task.metadata.get("modality"),
                interface=task.metadata.get("interface"),
                skill_type=task.metadata.get("skill_type"),
                has_skills=task.has_skills,
            )
            for task in self.get_dataset(dataset).values()
        ]

    async def retrieve_task(
        self, task_id: str, skip_validation: bool = False, dataset: str | None = None
    ) -> RetrieveTaskResponse:
        if not skip_validation:
            await self.validate_task_ids([task_id], dataset=dataset)

        task = _get_task(self.get_dataset(dataset), task_id)
        manifest = _load_image_manifest()
        entry = _manifest_task_entry(manifest, task_id)
        cwd = _task_cwd(task, entry)

        return RetrieveTaskResponse(
            source=_sandbox_source(manifest, entry),
            problem_path=_problem_path(cwd),
            cwd=cwd,
            agent_timeout=task.agent_timeout,
            resources=_resources(task, entry),
        )

    async def setup_task(
        self, task_id: str, sandbox: Sandbox, dataset: str | None = None
    ) -> AsyncGenerator[StreamChunk, None]:
        task = _get_task(self.get_dataset(dataset), task_id)
        entry = _manifest_task_entry(_load_image_manifest(), task_id)
        cwd = _task_cwd(task, entry)
        problem_path = _problem_path(cwd)

        yield StreamMessageChunk(type="message", data=f"Setting up SkillsBench task {task_id}")
        await sandbox.exec(
            f"mkdir -p {shlex.quote(cwd)} {shlex.quote(LOGS_DIR + '/agent')} "
            f"{shlex.quote(VERIFIER_DIR)} {shlex.quote(LOGS_DIR + '/artifacts')}"
        )
        await sandbox.upload_file(problem_path, task.instruction.encode("utf-8"))

        if _upload_environment_assets_enabled():
            uploaded = await _upload_tree(
                sandbox=sandbox,
                source_dir=task.environment_dir,
                remote_tar="/tmp/skillsbench-env-assets.tar.gz",
                target_dir=cwd,
                excluded_prefixes={"skills"},
                excluded_names={"Dockerfile"},
            )
            if uploaded:
                yield StreamMessageChunk(type="message", data="Uploaded task environment assets")

        if _dataset_injects_skills(dataset) and task.has_skills:
            await _upload_tree(
                sandbox=sandbox,
                source_dir=task.skills_dir,
                remote_tar="/tmp/skillsbench-skills.tar.gz",
                target_dir=SKILLS_DIR,
                excluded_prefixes=set(),
                excluded_names=set(),
                replace_target=True,
            )
            yield StreamMessageChunk(type="message", data=f"Injected SkillsBench skills at {SKILLS_DIR}")

        yield StreamResultChunk(
            type="result",
            data={
                "status": "ok",
                "problem_path": problem_path,
                "cwd": cwd,
                "skills_dir": SKILLS_DIR if _dataset_injects_skills(dataset) and task.has_skills else None,
            },
        )

    async def evaluate_response(self, request: EvaluateResponseRequest, dataset: str | None = None) -> Any:
        raise ValueError("SkillsBench tasks require sandbox evaluation through evaluate_instance().")

    async def evaluate_instance(
        self, task_id: str, sandbox: Sandbox, dataset: str | None = None
    ) -> AsyncGenerator[StreamChunk, None]:
        task = _get_task(self.get_dataset(dataset), task_id)
        entry = _manifest_task_entry(_load_image_manifest(), task_id)
        cwd = _task_cwd(task, entry)

        yield StreamMessageChunk(type="message", data=f"Evaluating SkillsBench task {task_id}")
        await sandbox.exec(f"mkdir -p {shlex.quote(VERIFIER_DIR)}")
        await _upload_tree(
            sandbox=sandbox,
            source_dir=task.tests_dir,
            remote_tar="/tmp/skillsbench-tests.tar.gz",
            target_dir=TESTS_DIR,
            excluded_prefixes=set(),
            excluded_names=set(),
            replace_target=True,
        )

        test_cmd = _with_timeout(
            f"chmod +x {shlex.quote(TESTS_DIR + '/test.sh')} && {shlex.quote(TESTS_DIR + '/test.sh')}",
            task.verifier_timeout,
        )
        verifier_error: str | None = None
        try:
            async for text in sandbox.command(test_cmd, cwd=cwd, timeout=task.verifier_timeout):
                if text.strip():
                    yield StreamMessageChunk(type="message", data=text)
        except Exception as exc:  # A verifier can fail before writing reward files.
            verifier_error = f"{type(exc).__name__}: {exc}"
            yield StreamMessageChunk(type="message", data=f"Verifier command failed: {verifier_error}")

        reward_payload, reward_error = await _read_reward(sandbox)
        if reward_error is not None:
            verifier_error = verifier_error or reward_error
            reward_payload = {"reward": 0.0}

        reward = _coerce_reward(reward_payload)
        yield StreamResultChunk(
            type="result",
            data={
                "task_id": task_id,
                "score": reward,
                "reward": reward,
                "resolved": reward > 0.0,
                "verifier_error": verifier_error,
                "reward_payload": reward_payload,
                "metadata": {
                    "dataset": dataset or "default",
                    "task_set": task.task_set,
                    "category": task.metadata.get("category"),
                    "has_skills": task.has_skills,
                    "skills_injected": _dataset_injects_skills(dataset) and task.has_skills,
                },
            },
        )

    def project_trial_result(self, result: Any) -> Any:
        if not isinstance(result, dict):
            return {"score": 0.0, "resolved": False, "verifier_error": "Unexpected result shape"}
        return {
            "score": float(result.get("score") or 0.0),
            "resolved": bool(result.get("resolved")),
            "verifier_error": result.get("verifier_error"),
            "metadata": result.get("metadata", {}),
        }

    async def calculate_final_score(
        self, evaluation_results: dict[str, Any], dataset: str | None = None
    ) -> FinalScoreResult:
        scores: list[float] = []
        errored = 0
        resolved = 0

        for result in evaluation_results.values():
            if not isinstance(result, dict):
                errored += 1
                scores.append(0.0)
                continue
            score = _bounded_score(result.get("score", result.get("reward", 0.0)))
            scores.append(score)
            if score > 0.0:
                resolved += 1
            if result.get("verifier_error"):
                errored += 1

        total = len(evaluation_results)
        mean_reward = sum(scores) / total if total else 0.0
        return FinalScoreResult(
            score=mean_reward * 100.0,
            metadata={
                "dataset": dataset or "default",
                "total_tasks": total,
                "resolved_tasks": resolved,
                "errored_tasks": errored,
                "mean_reward": mean_reward,
                "score_scale": "0-100",
            },
        )


def _repo_root() -> Path:
    configured = os.getenv(REPO_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "tasks").is_dir() and (parent / "CONTRIBUTING.md").is_file():
            return parent
        submodule = parent / "skillsbench"
        if (submodule / "tasks").is_dir() and (submodule / "CONTRIBUTING.md").is_file():
            return submodule

    return Path.cwd().resolve()


def _discover_tasks(tasks_dir: Path, task_set: Literal["default", "extra"]) -> dict[str, TaskSpec]:
    if not tasks_dir.is_dir():
        return {}

    tasks: dict[str, TaskSpec] = {}
    for config_path in sorted(tasks_dir.glob("*/task.toml")):
        task_dir = config_path.parent
        instruction_path = task_dir / "instruction.md"
        if not instruction_path.is_file():
            continue
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        tasks[task_dir.name] = TaskSpec(
            task_id=task_dir.name,
            task_dir=task_dir,
            instruction=instruction_path.read_text(encoding="utf-8"),
            config=config,
            task_set=task_set,
        )
    return tasks


def _get_task(dataset: dict[str, Any], task_id: str) -> TaskSpec:
    task = dataset[task_id]
    if not isinstance(task, TaskSpec):
        raise TypeError(f"Expected TaskSpec for {task_id}, got {type(task).__name__}")
    return task


def _load_image_manifest() -> dict[str, Any]:
    manifest_path = os.getenv(IMAGE_MANIFEST_ENV)
    if not manifest_path:
        return {}
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{IMAGE_MANIFEST_ENV} points to missing file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{IMAGE_MANIFEST_ENV} must contain a JSON object")
    return data


def _manifest_task_entry(manifest: dict[str, Any], task_id: str) -> dict[str, Any]:
    tasks = manifest.get("tasks", {})
    if not isinstance(tasks, dict):
        return {}
    entry = tasks.get(task_id, {})
    return entry if isinstance(entry, dict) else {}


def _sandbox_source(manifest: dict[str, Any], entry: dict[str, Any]) -> ImageSource | SnapshotSource:
    snapshot = entry.get("snapshot")
    if isinstance(snapshot, str) and snapshot:
        return SnapshotSource(snapshot=snapshot)

    image = entry.get("image")
    if not isinstance(image, str) or not image:
        image = manifest.get("default_image")
    if not isinstance(image, str) or not image:
        image = os.getenv(DEFAULT_IMAGE_ENV, DEFAULT_IMAGE)
    return ImageSource(image=image)


def _resources(task: TaskSpec, entry: dict[str, Any]) -> Resources:
    override = entry.get("resources", {})
    if not isinstance(override, dict):
        override = {}

    vcpu = _positive_int(override.get("vcpu")) or _positive_int(task.environment.get("cpus")) or 1
    memory = _positive_int(override.get("memory")) or _mb_to_gb(_memory_mb(task.environment)) or 2
    disk = _positive_int(override.get("disk")) or _mb_to_gb(_storage_mb(task.environment)) or 10
    return Resources(vcpu=vcpu, memory=memory, disk=disk)


def _memory_mb(environment: dict[str, Any]) -> int | None:
    return _size_to_mb(environment.get("memory_mb")) or _size_to_mb(environment.get("memory"))


def _storage_mb(environment: dict[str, Any]) -> int | None:
    return _size_to_mb(environment.get("storage_mb")) or _size_to_mb(environment.get("storage"))


def _size_to_mb(value: Any) -> int | None:
    if isinstance(value, int | float):
        return int(value)
    if not isinstance(value, str):
        return None

    text = value.strip().upper()
    if text.endswith("G"):
        return int(float(text[:-1]) * 1024)
    if text.endswith("M"):
        return int(float(text[:-1]))
    if text.endswith("K"):
        return max(1, int(float(text[:-1]) / 1024))
    return _positive_int(text)


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _mb_to_gb(value: int | None) -> int | None:
    if value is None:
        return None
    return max(1, math.ceil(value / 1024))


def _task_cwd(task: TaskSpec, entry: dict[str, Any]) -> str:
    cwd = entry.get("cwd")
    if isinstance(cwd, str) and cwd.startswith("/"):
        return cwd.rstrip("/") or "/"
    return _infer_workdir(task.environment_dir / "Dockerfile") or DEFAULT_CWD


def _infer_workdir(dockerfile_path: Path) -> str | None:
    if not dockerfile_path.is_file():
        return None

    workdir: str | None = None
    for line in dockerfile_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _WORKDIR_RE.match(line)
        if match:
            candidate = match.group(1).strip().strip('"').strip("'")
            if candidate.startswith("/"):
                workdir = candidate
    return workdir


def _problem_path(cwd: str) -> str:
    return posixpath.join(cwd, PROBLEM_FILENAME)


def _dataset_injects_skills(dataset: str | None) -> bool:
    return dataset in {"with-skills", "extra-with-skills"}


def _with_timeout(command: str, timeout_seconds: float | None) -> str:
    """Wrap a verifier command in an in-container timeout when configured."""
    if timeout_seconds is None:
        return command
    return f"timeout --kill-after=30s {math.ceil(timeout_seconds)}s bash -lc {shlex.quote(command)}"


def _upload_environment_assets_enabled() -> bool:
    value = os.getenv(UPLOAD_ENV_ASSETS_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


async def _upload_tree(
    *,
    sandbox: Sandbox,
    source_dir: Path,
    remote_tar: str,
    target_dir: str,
    excluded_prefixes: set[str],
    excluded_names: set[str],
    replace_target: bool = False,
) -> bool:
    payload = _tar_tree_bytes(source_dir, excluded_prefixes=excluded_prefixes, excluded_names=excluded_names)
    if payload is None:
        return False

    await sandbox.upload_file(remote_tar, payload)
    target = shlex.quote(target_dir)
    command = f"mkdir -p {target}"
    if replace_target:
        command = f"rm -rf {target} && mkdir -p {target}"
    await sandbox.exec(f"{command} && tar -xzf {shlex.quote(remote_tar)} -C {target} && chmod -R a+rX {target}")
    return True


def _tar_tree_bytes(source_dir: Path, *, excluded_prefixes: set[str], excluded_names: set[str]) -> bytes | None:
    if not source_dir.is_dir():
        return None

    files = list(_iter_files(source_dir, excluded_prefixes=excluded_prefixes, excluded_names=excluded_names))
    if not files:
        return None

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(source_dir).as_posix(), recursive=False)
    return buffer.getvalue()


def _iter_files(source_dir: Path, *, excluded_prefixes: set[str], excluded_names: set[str]) -> Iterable[Path]:
    for path in sorted(source_dir.rglob("*")):
        rel = path.relative_to(source_dir)
        parts = set(rel.parts)
        if excluded_prefixes.intersection(rel.parts):
            continue
        if excluded_names.intersection(parts):
            continue
        if path.is_file():
            yield path


async def _read_reward(sandbox: Sandbox) -> tuple[dict[str, Any], str | None]:
    text_reward = await _download_optional(sandbox, f"{VERIFIER_DIR}/reward.txt")
    if text_reward is not None:
        try:
            return {"reward": float(text_reward.decode("utf-8").strip())}, None
        except ValueError as exc:
            return {"reward": 0.0}, f"Could not parse reward.txt: {exc}"

    json_reward = await _download_optional(sandbox, f"{VERIFIER_DIR}/reward.json")
    if json_reward is not None:
        try:
            payload = json.loads(json_reward.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return {"reward": 0.0}, f"Could not parse reward.json: {exc}"
        if isinstance(payload, dict):
            return payload, None
        return {"reward": 0.0}, "reward.json did not contain an object"

    return {"reward": 0.0}, "No reward.txt or reward.json was produced by the verifier"


async def _download_optional(sandbox: Sandbox, remote_path: str) -> bytes | None:
    try:
        return await sandbox.download_file(remote_path)
    except Exception:
        return None


def _coerce_reward(payload: dict[str, Any]) -> float:
    return _bounded_score(payload.get("reward", 0.0))


def _bounded_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return min(1.0, max(0.0, score))
