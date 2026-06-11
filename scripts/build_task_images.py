#!/usr/bin/env python3
"""Build SkillsBench task environment images and write a Valkyrie manifest."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any

WORKDIR_RE = re.compile(r"^\s*WORKDIR\s+(.+?)\s*$", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--registry", required=True, help="Image registry/repository prefix, without tag")
    parser.add_argument("--output", type=Path, required=True, help="Manifest JSON path to write")
    parser.add_argument("--tag-suffix", default=_git_sha(), help="Image tag suffix. Defaults to current git SHA")
    parser.add_argument("--platform", help="Optional Docker platform, e.g. linux/amd64 for hosted Daytona")
    parser.add_argument("--push", action="store_true", help="Push images after building")
    parser.add_argument("--no-build", action="store_true", help="Only write manifest for already-built tags")
    parser.add_argument("task_ids", nargs="+", help="Task IDs under tasks/")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest = _load_manifest(args.output)

    for task_id in args.task_ids:
        task_dir = repo_root / "tasks" / task_id
        dockerfile = task_dir / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            raise SystemExit(f"Missing Dockerfile for task {task_id}: {dockerfile}")

        image = f"{args.registry}:{_tag(task_id, args.tag_suffix)}"
        if not args.no_build:
            if args.platform:
                build_command = [
                    "docker",
                    "buildx",
                    "build",
                    "--platform",
                    args.platform,
                    "-t",
                    image,
                ]
                build_command.append("--push" if args.push else "--load")
                build_command.append(str(task_dir / "environment"))
                _run(build_command, cwd=repo_root)
            else:
                _run(["docker", "build", "-t", image, str(task_dir / "environment")], cwd=repo_root)
                if args.push:
                    _run(["docker", "push", image], cwd=repo_root)
        elif args.push:
            _run(["docker", "push", image], cwd=repo_root)

        manifest["tasks"][task_id] = {
            "image": image,
            "cwd": _infer_workdir(dockerfile) or "/root",
            "resources": _resources(task_dir / "task.toml"),
        }
        _write_manifest(args.output, manifest)

    print(f"Wrote {args.output}")


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"tasks": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    tasks = data.setdefault("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError(f"Manifest 'tasks' must be a JSON object: {path}")
    return data


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tag(task_id: str, suffix: str) -> str:
    safe_task = re.sub(r"[^a-zA-Z0-9_.-]+", "-", task_id).strip(".-")
    safe_suffix = re.sub(r"[^a-zA-Z0-9_.-]+", "-", suffix).strip(".-")
    return f"{safe_task}-{safe_suffix}"[:128]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], text=True).strip()
    except Exception:
        return "local"


def _infer_workdir(dockerfile: Path) -> str | None:
    workdir: str | None = None
    for line in dockerfile.read_text(encoding="utf-8", errors="replace").splitlines():
        match = WORKDIR_RE.match(line)
        if match:
            candidate = match.group(1).strip().strip("\"").strip("'")
            if candidate.startswith("/"):
                workdir = candidate.rstrip("/") or "/"
    return workdir


def _resources(task_toml: Path) -> dict[str, int]:
    config = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    env = config.get("environment", {})
    memory_mb = _size_to_mb(env.get("memory_mb") or env.get("memory")) or 2048
    storage_mb = _size_to_mb(env.get("storage_mb") or env.get("storage")) or 10240
    return {
        "vcpu": _positive_int(env.get("cpus")) or 1,
        "memory": max(1, (memory_mb + 1023) // 1024),
        "disk": max(1, (storage_mb + 1023) // 1024),
    }


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


def _run(command: list[str], cwd: Path) -> None:
    attempts = 3
    for attempt in range(1, attempts + 1):
        print("+", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=cwd, check=False)
        if completed.returncode == 0:
            return
        if attempt == attempts:
            completed.check_returncode()
        wait_seconds = 10 * attempt
        print(f"Command failed with exit code {completed.returncode}; retrying in {wait_seconds}s", flush=True)
        time.sleep(wait_seconds)


if __name__ == "__main__":
    main()
