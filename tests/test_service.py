from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

from benchmark_service import ImageSource, Sandbox
from benchmark_service.schemas import StreamResultChunk
from skillsbench_valkyrie.service import SkillsBenchBenchmarkService


class FakeSandbox(Sandbox):
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {"/logs/verifier/reward.txt": b"1\n"}
        self.exec_commands: list[str] = []
        self.command_calls: list[tuple[str, str | None, float | None]] = []

    @property
    def id(self) -> str:
        return "fake-sandbox"

    @property
    def name(self) -> str:
        return "fake-sandbox"

    @property
    def state(self) -> str:
        return "started"

    async def exec(self, command: str, *, cwd: str | None = None, timeout: float | None = None) -> Any:
        self.exec_commands.append(command)
        return type("ExecResult", (), {"exit_code": 0, "output": ""})()

    async def command(
        self, command: str, *, cwd: str | None = None, timeout: float | None = None
    ) -> AsyncGenerator[str, None]:
        self.command_calls.append((command, cwd, timeout))
        yield "tests passed\n"

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        self.files[remote_path] = content

    async def download_file(self, remote_path: str) -> bytes:
        if remote_path not in self.files:
            raise FileNotFoundError(remote_path)
        return self.files[remote_path]


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

[agent]
timeout_sec = 120.0

[verifier]
timeout_sec = 60.0

[environment]
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


async def test_evaluate_instance_reads_reward(skillsbench_root: Path) -> None:
    service = await SkillsBenchBenchmarkService.create()
    sandbox = FakeSandbox()

    chunks = [chunk async for chunk in service.evaluate_instance("hello-world", sandbox, dataset="default")]

    assert isinstance(chunks[-1], StreamResultChunk)
    result = chunks[-1].data
    assert result["score"] == 1.0
    assert result["resolved"] is True
    assert result["verifier_error"] is None
    assert "/tmp/skillsbench-tests.tar.gz" in sandbox.files
    assert sandbox.command_calls[0][0].startswith("timeout --kill-after=30s 60s bash -lc")
    assert sandbox.command_calls[0][1] == "/app"
    assert sandbox.command_calls[0][2] == 60.0


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
                "result": {"score": 0.5, "metadata": {"category": "math", "duration_seconds": 20.0, "cost": 2.5}},
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
    assert result.metadata["tasks"][2]["status"] == "error"
