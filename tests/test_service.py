from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from benchmark_service import (
    DaytonaProviderConfig,
    ImageSource,
    ModalProviderConfig,
    Sandbox,
    SandboxCreateRequest,
    SandboxError,
    SandboxProvider,
    SandboxQuery,
    SnapshotSource,
)
from benchmark_service.sandbox.daytona import DaytonaSandbox
from benchmark_service.schemas import EvaluateResponseRequest, StreamResultChunk
from pydantic import ValidationError
from skillsbench_valkyrie import service as service_module
from skillsbench_valkyrie.service import (
    EVAL_SNAPSHOT_TIMEOUT_SECONDS,
    EvalResumeState,
    SkillsBenchBenchmarkService,
    create_daytona_snapshot,
)


class FakeSandbox(Sandbox):
    def __init__(
        self,
        *,
        sandbox_id: str = "fake-sandbox",
        exec_error: Exception | None = None,
        command_error: Exception | None = None,
        reward: bytes | None = b"1\n",
    ) -> None:
        self.sandbox_id = sandbox_id
        self.files: dict[str, bytes] = {}
        if reward is not None:
            self.files["/logs/verifier/reward.txt"] = reward
        self.exec_commands: list[str] = []
        self.command_calls: list[tuple[str, str | None, float | None]] = []
        self.exec_error = exec_error
        self.command_error: Exception | None = command_error

    @property
    def id(self) -> str:
        return self.sandbox_id

    @property
    def name(self) -> str:
        return self.sandbox_id

    @property
    def state(self) -> str:
        return "started"

    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> Any:
        self.exec_commands.append(command)
        if self.exec_error is not None:
            raise self.exec_error
        output = ""
        if "tail -c" in command and "/logs/verifier/test_output.log" in command:
            output = self.files.get("/logs/verifier/test_output.log", b"").decode("utf-8")
        return type("ExecResult", (), {"exit_code": 0, "output": output})()

    async def command(
        self, command: str, *, cwd: str | None = None, timeout: float | None = None
    ) -> AsyncGenerator[str, None]:
        self.command_calls.append((command, cwd, timeout))
        if self.command_error is not None:
            raise self.command_error
        yield "tests passed\n"

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        self.files[remote_path] = content

    async def download_file(self, remote_path: str) -> bytes:
        if remote_path not in self.files:
            raise FileNotFoundError(remote_path)
        return self.files[remote_path]


class FakeProvider(SandboxProvider):
    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox
        self.create_request: SandboxCreateRequest | None = None
        self.deleted: list[str] = []
        self.closed = False

    async def create_sandbox(self, request: SandboxCreateRequest) -> Sandbox:
        self.create_request = request
        return self.sandbox

    async def get_sandbox(self, instance_id: str) -> Sandbox:
        raise AssertionError("resume must create a fresh sandbox")

    async def delete_sandbox(self, instance_id: str) -> None:
        self.deleted.append(instance_id)

    async def list_sandboxes(self, query: SandboxQuery) -> AsyncGenerator[Sandbox, None]:
        if False:
            yield self.sandbox

    async def close(self) -> None:
        self.closed = True


def _daytona_config() -> DaytonaProviderConfig:
    return DaytonaProviderConfig(DAYTONA_API_KEY="key", DAYTONA_API_URL="url", DAYTONA_TARGET="target")


@pytest.fixture
def skillsbench_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path
    task = root / "tasks" / "hello-world"
    (task / "environment" / "skills" / "hello-skill").mkdir(parents=True)
    (task / "tests").mkdir(parents=True)

    (task / "instruction.md").write_text('Create hello.txt with "Hello, world!" as content.\n', encoding="utf-8")
    (task / "task.toml").write_text(
        """
version = "1.0"

[metadata]
difficulty = "easy"
category = "programming"
tags = ["hello", "fixture"]

[agent]
timeout_sec = 120.0

[verifier]
timeout_sec = 60.0

[environment]
build_timeout_sec = 600.0
cpus = 2
memory_mb = 4096
storage_mb = 10240
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n", encoding="utf-8")
    (task / "environment" / "input.txt").write_text("fixture\n", encoding="utf-8")
    (task / "environment" / "skills" / "hello-skill" / "SKILL.md").write_text("# hello\n", encoding="utf-8")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n", encoding="utf-8")

    (root / "tasks-extra").mkdir()
    monkeypatch.setenv("SKILLSBENCH_REPO_ROOT", str(root))
    monkeypatch.delenv("SKILLSBENCH_VALKYRIE_IMAGE_MANIFEST", raising=False)
    monkeypatch.delenv("SKILLSBENCH_VALKYRIE_DEFAULT_IMAGE", raising=False)

    async def create_snapshot(_sandbox: Sandbox, _snapshot_name: str) -> None:
        pass

    monkeypatch.setattr(service_module, "create_daytona_snapshot", create_snapshot)
    return root


async def test_lists_and_retrieves_tasks(skillsbench_root: Path) -> None:
    service = await SkillsBenchBenchmarkService.create()

    tasks = await service.list_tasks("default")
    assert [task.id for task in tasks] == ["hello-world"]
    assert tasks[0].timeout == 120.0
    assert tasks[0].model_dump()["has_skills"] is True

    response = await service.retrieve_task("hello-world", dataset="default")
    assert isinstance(response.source, ImageSource)
    assert response.source.image == "python:3.12-slim"
    assert response.problem_path == "/app/instruction.md"
    assert response.cwd == "/app"
    assert response.resources.vcpu == 2
    assert response.resources.memory == 4
    assert response.resources.disk == 10
    assert response.agent_timeout == 120.0


async def test_setup_default_does_not_inject_skills(skillsbench_root: Path) -> None:
    service = await SkillsBenchBenchmarkService.create()
    sandbox = FakeSandbox()

    chunks = [chunk async for chunk in service.setup_task("hello-world", sandbox, dataset="default")]

    assert isinstance(chunks[-1], StreamResultChunk)
    assert chunks[-1].data["skills_dir"] is None
    assert sandbox.files["/app/instruction.md"].startswith(b"Create hello.txt")
    assert "/tmp/skillsbench-skills.tar.gz" not in sandbox.files
    assert "/tmp/skillsbench-env-assets.tar.gz" in sandbox.files


async def test_setup_with_skills_injects_skills(skillsbench_root: Path) -> None:
    service = await SkillsBenchBenchmarkService.create()
    sandbox = FakeSandbox()

    chunks = [chunk async for chunk in service.setup_task("hello-world", sandbox, dataset="with-skills")]

    assert isinstance(chunks[-1], StreamResultChunk)
    assert chunks[-1].data["skills_dir"] == "/skills"
    assert "/tmp/skillsbench-skills.tar.gz" in sandbox.files


async def test_checkpoint_is_emitted_before_verifier_setup_failure(
    skillsbench_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = await SkillsBenchBenchmarkService.create()
    sandbox = FakeSandbox(exec_error=RuntimeError("verifier setup failed"))
    snapshot_calls: list[str] = []

    async def create_snapshot(_sandbox: Sandbox, snapshot_name: str) -> None:
        assert sandbox.exec_commands == []
        snapshot_calls.append(snapshot_name)

    monkeypatch.setattr(service_module, "create_daytona_snapshot", create_snapshot)
    chunks = []
    with pytest.raises(RuntimeError, match="verifier setup failed"):
        async for chunk in service.evaluate_instance("hello-world", sandbox, dataset="default"):
            chunks.append(chunk)

    assert [chunk.type for chunk in chunks] == ["eval_resume_state", "message"]
    state = EvalResumeState.model_validate(chunks[0].data)
    assert state.version == 1
    assert state.task_id == "hello-world"
    assert state.dataset == "default"
    assert snapshot_calls == [state.snapshot]
    assert sandbox.exec_commands == ["mkdir -p /logs/verifier /logs/tests"]


async def test_evaluate_instance_reads_reward(skillsbench_root: Path) -> None:
    service = await SkillsBenchBenchmarkService.create()
    sandbox = FakeSandbox()

    chunks = [chunk async for chunk in service.evaluate_instance("hello-world", sandbox, dataset="default")]

    assert isinstance(chunks[-1], StreamResultChunk)
    result = chunks[-1].data
    assert result["score"] == 1.0
    assert result["resolved"] is True
    assert result["verifier_error"] is None
    assert result["metadata"]["difficulty"] == "easy"
    assert result["metadata"]["tags"] == ["hello", "fixture"]
    assert "/tmp/skillsbench-tests.tar.gz" in sandbox.files
    assert not sandbox.command_calls[0][0].startswith("timeout ")
    assert sandbox.command_calls[0][0].startswith("bash -lc ")
    assert "set -o pipefail" in sandbox.command_calls[0][0]
    assert "/tests/test.sh" in sandbox.command_calls[0][0]
    assert "tee /logs/verifier/test_output.log" in sandbox.command_calls[0][0]
    assert "mkdir -p /logs/verifier /logs/tests" in sandbox.command_calls[0][0]
    assert sandbox.command_calls[0][1] == "/app"
    assert sandbox.command_calls[0][2] is None
    assert result["metadata"]["verifier_reward_dirs"] == ["/logs/verifier", "/logs/tests"]


async def test_resume_uses_fresh_snapshot_sandbox_without_setup(
    skillsbench_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = await SkillsBenchBenchmarkService.create()
    fresh = FakeSandbox(sandbox_id="resume-sandbox")
    provider = FakeProvider(fresh)
    monkeypatch.setattr(DaytonaProviderConfig, "create_provider", lambda _config: provider)

    async def forbidden_setup(*_args: object, **_kwargs: object) -> AsyncGenerator[Any, None]:
        raise AssertionError("setup_task must not run when resuming a filesystem snapshot")
        yield

    monkeypatch.setattr(service, "setup_task", forbidden_setup)
    state = EvalResumeState.create("hello-world", "default")
    request = EvaluateResponseRequest(
        task_id="hello-world",
        eval_resume_state=state.model_dump(mode="json"),
        sandbox_provider=_daytona_config(),
    )

    chunks = [chunk async for chunk in service.stream_evaluate_response(request)]

    assert chunks[0].type == "eval_resume_state"
    assert chunks[0].data == state.model_dump(mode="json")
    assert isinstance(chunks[-1], StreamResultChunk)
    assert chunks[-1].data["score"] == 1.0
    assert sum(chunk.type == "eval_resume_state" for chunk in chunks) == 1
    assert provider.create_request is not None
    assert provider.create_request.source == SnapshotSource(snapshot=state.snapshot)
    assert provider.create_request.name.startswith("sb-eval-run-v1-")
    assert provider.create_request.labels["EvalResume"] == "true"
    assert fresh.command_calls
    assert "/app/instruction.md" not in fresh.files
    assert "/tmp/skillsbench-env-assets.tar.gz" not in fresh.files
    assert provider.deleted == [fresh.id]
    assert provider.closed is True


@pytest.mark.parametrize(
    ("request_task_id", "request_dataset", "error"),
    [
        ("different-task", None, "task_id mismatch"),
        ("hello-world", "with-skills", "dataset mismatch"),
    ],
)
async def test_resume_rejects_task_or_dataset_mismatch(
    skillsbench_root: Path,
    request_task_id: str,
    request_dataset: str | None,
    error: str,
) -> None:
    service = await SkillsBenchBenchmarkService.create()
    state = EvalResumeState.create("hello-world", "default")
    request = EvaluateResponseRequest(
        task_id=request_task_id,
        dataset=request_dataset,
        eval_resume_state=state.model_dump(mode="json"),
    )

    with pytest.raises(ValueError, match=error):
        _ = [chunk async for chunk in service.stream_evaluate_response(request, dataset=request_dataset)]


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"version": 2},
        {"snapshot": "unrelated-snapshot"},
        {"snapshot": f"sb-eval-resume-v1-{'0' * 12}-{'1' * 32}"},
        {"unexpected": True},
    ],
)
async def test_resume_rejects_malformed_state(skillsbench_root: Path, invalid_fields: dict[str, Any]) -> None:
    service = await SkillsBenchBenchmarkService.create()
    state = EvalResumeState.create("hello-world", "default").model_dump(mode="json")
    state.update(invalid_fields)
    request = EvaluateResponseRequest(task_id="hello-world", eval_resume_state=state)

    with pytest.raises(ValidationError):
        _ = [chunk async for chunk in service.stream_evaluate_response(request)]


async def test_resume_requires_daytona_provider(skillsbench_root: Path) -> None:
    service = await SkillsBenchBenchmarkService.create()
    state = EvalResumeState.create("hello-world", "default").model_dump(mode="json")

    without_provider = EvaluateResponseRequest(task_id="hello-world", eval_resume_state=state)
    with pytest.raises(ValueError, match="requires sandbox_provider"):
        _ = [chunk async for chunk in service.stream_evaluate_response(without_provider)]

    wrong_provider = EvaluateResponseRequest(
        task_id="hello-world", eval_resume_state=state, sandbox_provider=ModalProviderConfig()
    )
    with pytest.raises(ValueError, match="Daytona sandbox_provider"):
        _ = [chunk async for chunk in service.stream_evaluate_response(wrong_provider)]


async def test_resume_deletes_temporary_sandbox_after_verifier_failure(
    skillsbench_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = await SkillsBenchBenchmarkService.create()
    fresh = FakeSandbox(sandbox_id="resume-sandbox", exec_error=RuntimeError("resumed verifier failed"))
    provider = FakeProvider(fresh)
    monkeypatch.setattr(DaytonaProviderConfig, "create_provider", lambda _config: provider)
    state = EvalResumeState.create("hello-world", "default")
    request = EvaluateResponseRequest(
        task_id="hello-world",
        eval_resume_state=state.model_dump(mode="json"),
        sandbox_provider=_daytona_config(),
    )

    with pytest.raises(RuntimeError, match="resumed verifier failed"):
        _ = [chunk async for chunk in service.stream_evaluate_response(request)]

    assert provider.deleted == [fresh.id]
    assert provider.closed is True


async def test_snapshot_adapter_is_guarded_and_calls_daytona_hook() -> None:
    with pytest.raises(ValueError, match="Daytona sandbox"):
        await create_daytona_snapshot(FakeSandbox(), "snapshot")

    calls: list[tuple[str, int]] = []

    async def create_snapshot(name: str, timeout: int) -> None:
        calls.append((name, timeout))

    sandbox = DaytonaSandbox(cast(Any, SimpleNamespace(_experimental_create_snapshot=create_snapshot)))
    await create_daytona_snapshot(sandbox, "snapshot")

    assert calls == [("snapshot", EVAL_SNAPSHOT_TIMEOUT_SECONDS)]

    unsupported = DaytonaSandbox(cast(Any, SimpleNamespace()))
    with pytest.raises(SandboxError, match="does not support filesystem snapshots"):
        await create_daytona_snapshot(unsupported, "snapshot")


async def test_native_task_markdown_uses_verifier_directory(skillsbench_root: Path) -> None:
    task = skillsbench_root / "tasks" / "native-task"
    (task / "environment").mkdir(parents=True)
    (task / "verifier").mkdir()
    (task / "task.md").write_text(
        """---
metadata:
  category: software-engineering
  difficulty: medium
  tags:
    - native
    - verifier
agent:
  timeout_sec: 90.0
verifier:
  timeout_sec: 45.0
environment:
  cpus: 1
  memory_mb: 1024
---

Create the native task output.
""",
        encoding="utf-8",
    )
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /work\n", encoding="utf-8")
    (task / "verifier" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n", encoding="utf-8")

    service = await SkillsBenchBenchmarkService.create()
    sandbox = FakeSandbox(reward=None)
    sandbox.files["/logs/tests/reward.txt"] = b"1\n"

    tasks = await service.list_tasks("default")
    assert {task.id for task in tasks} == {"hello-world", "native-task"}

    response = await service.retrieve_task("native-task", dataset="default")
    assert response.problem_path == "/work/instruction.md"
    assert response.agent_timeout == 90.0

    chunks = [chunk async for chunk in service.evaluate_instance("native-task", sandbox, dataset="default")]

    assert isinstance(chunks[-1], StreamResultChunk)
    assert chunks[-1].data["score"] == 1.0
    assert chunks[-1].data["metadata"]["difficulty"] == "medium"
    assert chunks[-1].data["metadata"]["tags"] == ["native", "verifier"]
    assert any("tar -xzf /tmp/skillsbench-tests.tar.gz -C /verifier" in command for command in sandbox.exec_commands)
    assert "/verifier/test.sh" in sandbox.command_calls[0][0]
    assert sandbox.command_calls[0][2] is None


async def test_evaluate_instance_reports_verifier_log_tail(skillsbench_root: Path) -> None:
    service = await SkillsBenchBenchmarkService.create()
    sandbox = FakeSandbox(command_error=RuntimeError("boom"), reward=b"1\n")
    sandbox.files["/logs/verifier/test_output.log"] = b"setup started\nlast verifier line\n"

    chunks = [chunk async for chunk in service.evaluate_instance("hello-world", sandbox, dataset="default")]

    assert isinstance(chunks[-1], StreamResultChunk)
    result = chunks[-1].data
    assert result["score"] == 1.0
    assert result["resolved"] is True
    assert result["verifier_error"] == "RuntimeError: boom"
    assert result["verifier_log_tail"] == "setup started\nlast verifier line\n"
    assert result["metadata"]["verifier_log_path"] == "/logs/verifier/test_output.log"


async def test_final_score_uses_mean_reward(skillsbench_root: Path) -> None:
    service = await SkillsBenchBenchmarkService.create()

    result = await service.calculate_final_score(
        {
            "a": {
                "score": 1.0,
                "metadata": {"category": "programming", "cost": 1.5},
                "task_breakdown": {"agent_run_duration": 10.0},
            },
            "b": {
                "status": "evaluated",
                "result": {
                    "score": 0.5,
                    "metadata": {
                        "category": "math",
                        "difficulty": "hard",
                        "tags": ["algebra"],
                        "verifier_reward_dirs": ["/logs/verifier", "/logs/tests"],
                        "duration_seconds": 20.0,
                        "cost": 2.5,
                    },
                },
            },
            "c": None,
        },
        dataset="with-skills",
    )

    assert result.score == 50.0
    assert result.metadata["total_tasks"] == 3
    assert result.metadata["resolved_tasks"] == 2
    assert result.metadata["errored_tasks"] == 1
    assert result.metadata["score_types"]["score"]["unit"] == "percent"
    assert result.metadata["primary_population"] == "full"
    assert result.metadata["usage_components"] == [{"component": "generation.model"}]
    assert result.metadata["results"]["full"]["counts"] == {
        "total": 3,
        "by_status": {"evaluated": 2, "error": 1},
        "extra": {},
    }
    assert result.metadata["results"]["full"]["aggregated_metrics"]["total"]["duration_seconds"] == 30.0
    assert result.metadata["results"]["full"]["aggregated_metrics"]["total"]["metadata"]["cost"] == {"total": 4.0}
    assert result.metadata["results"]["full"]["aggregated_metrics"]["average_per_task"]["duration_seconds"] == 15.0
    assert result.metadata["results"]["full"]["aggregated_metrics"]["average_per_task"]["metadata"]["cost"] == {
        "total": 2.0
    }
    assert result.metadata["tasks"][0]["task_id"] == "a"
    assert result.metadata["tasks"][0]["status"] == "evaluated"
    assert result.metadata["tasks"][0]["scores"]["score"]["value"] == 100.0
    assert result.metadata["tasks"][1]["extra"]["skillsbench"]["category"] == "math"
    assert result.metadata["tasks"][1]["extra"]["skillsbench"]["difficulty"] == "hard"
    assert result.metadata["tasks"][1]["extra"]["skillsbench"]["tags"] == ["algebra"]
    assert result.metadata["tasks"][1]["extra"]["skillsbench"]["verifier_reward_dirs"] == [
        "/logs/verifier",
        "/logs/tests",
    ]
    assert result.metadata["tasks"][1]["tags"] == ["algebra"]
    assert result.metadata["tasks"][2]["status"] == "error"
