"""Expose SkillsBench tasks through Valkyrie's BenchmarkService contract."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import math
import os
import posixpath
import re
import shlex
import tarfile
import tomllib
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import yaml  # type: ignore[import-untyped]
from benchmark_service import (
    BenchmarkService,
    DaytonaProviderConfig,
    ImageSource,
    Resources,
    Sandbox,
    SandboxCreateRequest,
    SandboxError,
    SnapshotSource,
)
from benchmark_service.sandbox.daytona import DaytonaSandbox
from benchmark_service.schemas import (
    EvaluateResponseRequest,
    FinalScoreResult,
    RetrieveTaskResponse,
    StreamChunk,
    StreamEvalResumeStateChunk,
    StreamMessageChunk,
    StreamResultChunk,
)
from benchmark_service.v1_schemas import V1Task
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .vals_helper import bounded_score, build_final_score_metadata

logger = logging.getLogger(__name__)

REPO_ROOT_ENV = "SKILLSBENCH_REPO_ROOT"
IMAGE_MANIFEST_ENV = "SKILLSBENCH_VALKYRIE_IMAGE_MANIFEST"
DEFAULT_IMAGE_ENV = "SKILLSBENCH_VALKYRIE_DEFAULT_IMAGE"
UPLOAD_ENV_ASSETS_ENV = "SKILLSBENCH_VALKYRIE_UPLOAD_ENV_ASSETS"

DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_CWD = "/root"
PROBLEM_FILENAME = "instruction.md"
SKILLS_DIR = "/skills"
TESTS_DIR = "/tests"
NATIVE_VERIFIER_DIR = "/verifier"
LOGS_DIR = "/logs"
VERIFIER_DIR = "/logs/verifier"
TEST_LOGS_DIR = "/logs/tests"
VERIFIER_OUTPUT_LOG = f"{VERIFIER_DIR}/test_output.log"
VERIFIER_LOG_TAIL_BYTES = 12000
EVAL_SNAPSHOT_PREFIX = "sb-eval-resume-v1"
EVAL_SNAPSHOT_TIMEOUT_SECONDS = 600
EVAL_SANDBOX_AUTO_STOP_MINUTES = 15
EVAL_SANDBOX_CREATE_TIMEOUT_SECONDS = 600

_WORKDIR_RE = re.compile(r"^\s*WORKDIR\s+(.+?)\s*$", re.IGNORECASE)


def _task_binding(task_id: str, dataset: str) -> str:
    identity = json.dumps([dataset, task_id], separators=(",", ":"))
    return hashlib.sha256(identity.encode()).hexdigest()[:12]


def _snapshot_name(task_id: str, dataset: str, nonce: str) -> str:
    return f"{EVAL_SNAPSHOT_PREFIX}-{_task_binding(task_id, dataset)}-{nonce}"


class EvalResumeState(BaseModel):
    """Task-bound pointer to a durable post-agent filesystem snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    task_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    snapshot: str = Field(pattern=rf"^{EVAL_SNAPSHOT_PREFIX}-[0-9a-f]{{12}}-[0-9a-f]{{32}}$")

    @field_validator("version", mode="before")
    @classmethod
    def validate_exact_version(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("eval_resume_state version must be exact integer 1")
        return value

    @classmethod
    def create(cls, task_id: str, dataset: str) -> EvalResumeState:
        return cls(task_id=task_id, dataset=dataset, snapshot=_snapshot_name(task_id, dataset, uuid4().hex))

    @model_validator(mode="after")
    def require_task_bound_snapshot(self) -> EvalResumeState:
        nonce = self.snapshot.rsplit("-", 1)[-1]
        if self.snapshot != _snapshot_name(self.task_id, self.dataset, nonce):
            raise ValueError("eval_resume_state snapshot is not canonical for its task and dataset")
        return self


def _resume_sandbox_name(state: EvalResumeState) -> str:
    return f"sb-eval-run-v1-{_task_binding(state.task_id, state.dataset)}-{uuid4().hex}"


async def create_daytona_snapshot(sandbox: Sandbox, snapshot_name: str) -> None:
    """Use Daytona's filesystem snapshot hook until CBS exposes one."""
    if not isinstance(sandbox, DaytonaSandbox):
        raise ValueError("SkillsBench eval resume requires a Daytona sandbox")

    inner = getattr(sandbox, "_sandbox", None)
    create_snapshot = getattr(inner, "_experimental_create_snapshot", None)
    if not callable(create_snapshot):
        raise SandboxError("The installed Daytona SDK does not support filesystem snapshots")

    snapshot_creator = cast(Callable[..., Awaitable[None]], create_snapshot)
    await snapshot_creator(snapshot_name, timeout=EVAL_SNAPSHOT_TIMEOUT_SECONDS)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_dir: Path
    instruction: str
    config: dict[str, Any]
    task_set: Literal["default", "extra"]
    verifier_dir_name: str = "tests"
    remote_verifier_dir: str = TESTS_DIR

    @property
    def environment_dir(self) -> Path:
        return self.task_dir / "environment"

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / self.verifier_dir_name

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
    def verifier_timeout(self) -> float:
        value = self.verifier.get("timeout_sec")
        return float(value) if isinstance(value, int | float) else 600.0


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
            V1Task.model_validate(
                {
                    "id": task.task_id,
                    "question": task.instruction,
                    "timeout": task.agent_timeout,
                    "category": task.metadata.get("category"),
                    "subcategory": task.metadata.get("subcategory"),
                    "difficulty": task.metadata.get("difficulty"),
                    "task_type": task.metadata.get("task_type"),
                    "modality": task.metadata.get("modality"),
                    "interface": task.metadata.get("interface"),
                    "skill_type": task.metadata.get("skill_type"),
                    "has_skills": task.has_skills,
                }
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

    async def stream_evaluate_response(
        self, request: EvaluateResponseRequest, dataset: str | None = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Resume verification in a fresh sandbox from a durable Daytona snapshot."""
        if request.response is not None or request.eval_resume_state is None:
            raise ValueError("SkillsBench eval resume requires eval_resume_state")

        state = EvalResumeState.model_validate(request.eval_resume_state)
        requested_dataset = dataset or request.dataset or "default"
        if dataset is not None and request.dataset is not None and dataset != request.dataset:
            raise ValueError(f"request dataset mismatch: {request.dataset} != {dataset}")
        if state.task_id != request.task_id:
            raise ValueError(f"eval_resume_state task_id mismatch: {state.task_id} != {request.task_id}")
        if state.dataset != requested_dataset:
            raise ValueError(f"eval_resume_state dataset mismatch: {state.dataset} != {requested_dataset}")
        if request.sandbox_provider is None:
            raise ValueError("SkillsBench eval resume requires sandbox_provider")
        if not isinstance(request.sandbox_provider, DaytonaProviderConfig):
            raise ValueError("SkillsBench eval resume requires a Daytona sandbox_provider")

        await self.validate_task_ids([request.task_id], dataset=requested_dataset)
        task = _get_task(self.get_dataset(requested_dataset), request.task_id)
        entry = _manifest_task_entry(_load_image_manifest(), request.task_id)
        cwd = _task_cwd(task, entry)
        yield StreamEvalResumeStateChunk(type="eval_resume_state", data=state.model_dump(mode="json"))

        async with request.sandbox_provider.create_provider() as provider:
            sandbox = await provider.create_sandbox(
                SandboxCreateRequest(
                    source=SnapshotSource(snapshot=state.snapshot),
                    resources=_resources(task, entry),
                    name=_resume_sandbox_name(state),
                    labels={
                        "Benchmark": "skillsbench",
                        "Task": state.task_id,
                        "Dataset": state.dataset,
                        "EvalResume": "true",
                    },
                    env_vars={},
                    auto_stop_interval=EVAL_SANDBOX_AUTO_STOP_MINUTES,
                    create_timeout=EVAL_SANDBOX_CREATE_TIMEOUT_SECONDS,
                )
            )
            try:
                async for chunk in self._run_verifier(request.task_id, task, cwd, sandbox, requested_dataset):
                    yield chunk
            finally:
                try:
                    await provider.delete_sandbox(sandbox.id)
                except Exception:
                    logger.exception("Failed to delete SkillsBench eval-resume sandbox %s", sandbox.id)

    async def evaluate_instance(
        self, task_id: str, sandbox: Sandbox, dataset: str | None = None
    ) -> AsyncGenerator[StreamChunk, None]:
        task = _get_task(self.get_dataset(dataset), task_id)
        entry = _manifest_task_entry(_load_image_manifest(), task_id)
        cwd = _task_cwd(task, entry)

        state = EvalResumeState.create(task_id, dataset or "default")
        await create_daytona_snapshot(sandbox, state.snapshot)
        yield StreamEvalResumeStateChunk(type="eval_resume_state", data=state.model_dump(mode="json"))

        async for chunk in self._run_verifier(task_id, task, cwd, sandbox, dataset):
            yield chunk

    async def _run_verifier(
        self,
        task_id: str,
        task: TaskSpec,
        cwd: str,
        sandbox: Sandbox,
        dataset: str | None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Run the existing verifier body without creating another checkpoint."""

        yield StreamMessageChunk(type="message", data=f"Evaluating SkillsBench task {task_id}")
        await sandbox.exec(f"mkdir -p {shlex.quote(VERIFIER_DIR)} {shlex.quote(TEST_LOGS_DIR)}")

        test_cmd = _verifier_command(task.remote_verifier_dir)
        verifier_error: str | None = None
        reward_error: str | None = None
        try:
            async with asyncio.timeout(task.verifier_timeout):
                await _upload_tree(
                    sandbox=sandbox,
                    source_dir=task.tests_dir,
                    remote_tar="/tmp/skillsbench-tests.tar.gz",
                    target_dir=task.remote_verifier_dir,
                    excluded_prefixes=set(),
                    excluded_names=set(),
                    replace_target=True,
                )
                async for text in sandbox.command(test_cmd, cwd=cwd):
                    if text.strip():
                        yield StreamMessageChunk(type="message", data=text)
        except TimeoutError:
            verifier_error = f"verifier timed out after {task.verifier_timeout}s"
            yield StreamMessageChunk(type="message", data=f"Verifier command failed: {verifier_error}")
        except Exception as exc:  # A verifier can fail before writing reward files.
            verifier_error = f"{type(exc).__name__}: {exc}"
            yield StreamMessageChunk(type="message", data=f"Verifier command failed: {verifier_error}")

        reward_payload, reward_error = await _read_reward(sandbox)
        if reward_error is not None:
            verifier_error = verifier_error or reward_error
            reward_payload = {"reward": 0.0}

        verifier_log_tail = None
        if verifier_error:
            verifier_log_tail = await _read_remote_text_tail(sandbox, VERIFIER_OUTPUT_LOG, VERIFIER_LOG_TAIL_BYTES)
        reward = _coerce_reward(reward_payload)
        yield StreamResultChunk(
            type="result",
            data={
                "task_id": task_id,
                "score": reward,
                "reward": reward,
                "resolved": reward > 0.0,
                "verifier_error": verifier_error,
                "verifier_log_tail": verifier_log_tail if verifier_error else None,
                "reward_payload": reward_payload,
                "metadata": {
                    "dataset": dataset or "default",
                    "task_set": task.task_set,
                    "category": task.metadata.get("category"),
                    "difficulty": task.metadata.get("difficulty"),
                    "tags": _string_list(task.metadata.get("tags")),
                    "has_skills": task.has_skills,
                    "skills_injected": _dataset_injects_skills(dataset) and task.has_skills,
                    "verifier_log_path": VERIFIER_OUTPUT_LOG,
                    "verifier_reward_dirs": [VERIFIER_DIR, TEST_LOGS_DIR],
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
            "verifier_log_tail": result.get("verifier_log_tail"),
            "metadata": result.get("metadata", {}),
        }

    async def calculate_final_score(
        self, evaluation_results: dict[str, Any], dataset: str | None = None
    ) -> FinalScoreResult:
        score, metadata = build_final_score_metadata(evaluation_results, dataset)
        return FinalScoreResult(score=score, metadata=metadata)


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
    for task_dir in sorted(path for path in tasks_dir.iterdir() if path.is_dir()):
        task = _read_task_spec(task_dir, task_set)
        if task is None:
            continue
        tasks[task.task_id] = task
    return tasks


def _read_task_spec(task_dir: Path, task_set: Literal["default", "extra"]) -> TaskSpec | None:
    config_path = task_dir / "task.toml"
    instruction_path = task_dir / "instruction.md"
    if config_path.is_file() and instruction_path.is_file():
        return TaskSpec(
            task_id=task_dir.name,
            task_dir=task_dir,
            instruction=instruction_path.read_text(encoding="utf-8"),
            config=tomllib.loads(config_path.read_text(encoding="utf-8")),
            task_set=task_set,
        )

    task_markdown_path = task_dir / "task.md"
    native_verifier_dir = task_dir / "verifier"
    if not task_markdown_path.is_file() or not native_verifier_dir.is_dir():
        return None

    config, instruction = _read_task_markdown(task_markdown_path)
    return TaskSpec(
        task_id=task_dir.name,
        task_dir=task_dir,
        instruction=instruction,
        config=config,
        task_set=task_set,
        verifier_dir_name="verifier",
        remote_verifier_dir=NATIVE_VERIFIER_DIR,
    )


def _read_task_markdown(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path} must start with YAML frontmatter")

    end_index = next((index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if end_index is None:
        raise ValueError(f"{path} has no closing YAML frontmatter marker")

    frontmatter = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :]).lstrip("\n")
    config = yaml.safe_load(frontmatter) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} frontmatter must be a YAML object")

    return config, body


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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _verifier_command(remote_verifier_dir: str) -> str:
    script_path = posixpath.join(remote_verifier_dir, "test.sh")
    command = (
        "set -o pipefail; "
        f"mkdir -p {shlex.quote(VERIFIER_DIR)} {shlex.quote(TEST_LOGS_DIR)}; "
        f"{{ chmod +x {shlex.quote(script_path)} && {shlex.quote(script_path)}; }} "
        f"2>&1 | tee {shlex.quote(VERIFIER_OUTPUT_LOG)}"
    )
    return f"bash -lc {shlex.quote(command)}"


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
    for reward_dir in (VERIFIER_DIR, TEST_LOGS_DIR):
        text_reward = await _download_optional(sandbox, f"{reward_dir}/reward.txt")
        if text_reward is not None:
            try:
                return {"reward": float(text_reward.decode("utf-8").strip())}, None
            except ValueError as exc:
                return {"reward": 0.0}, f"Could not parse {reward_dir}/reward.txt: {exc}"

        json_reward = await _download_optional(sandbox, f"{reward_dir}/reward.json")
        if json_reward is not None:
            try:
                payload = json.loads(json_reward.decode("utf-8"))
            except json.JSONDecodeError as exc:
                return {"reward": 0.0}, f"Could not parse {reward_dir}/reward.json: {exc}"
            if isinstance(payload, dict):
                return payload, None
            return {"reward": 0.0}, f"{reward_dir}/reward.json did not contain an object"

    return {"reward": 0.0}, "No reward.txt or reward.json was produced under /logs/verifier or /logs/tests"


async def _download_optional(sandbox: Sandbox, remote_path: str) -> bytes | None:
    try:
        return await sandbox.download_file(remote_path)
    except Exception:
        return None


async def _read_remote_text_tail(sandbox: Sandbox, remote_path: str, max_bytes: int) -> str | None:
    path = shlex.quote(remote_path)
    try:
        result = await sandbox.exec(f"test -f {path} && tail -c {max_bytes} {path} || true")
    except Exception:
        data = await _download_optional(sandbox, remote_path)
        if data is None:
            return None
        return data[-max_bytes:].decode("utf-8", errors="replace")

    output = getattr(result, "output", None)
    return output if isinstance(output, str) and output else None


def _coerce_reward(payload: dict[str, Any]) -> float:
    return bounded_score(payload.get("reward", 0.0))
