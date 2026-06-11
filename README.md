# SkillsBench Benchmark Service

This repository contains the Valkyrie benchmark-service adapter for
SkillsBench. It is intended to be consumed by
`vals-ai/public-benchmark-services-registry` as the `skillsbench` service.

For the full porting guide, see [PORTING.md](PORTING.md).

The adapter exposes SkillsBench task directories through the public
`benchmark_service.BenchmarkService` contract used by Valkyrie:

- `retrieve_task()` maps `task.toml` to a sandbox image/snapshot, resources,
  working directory, and agent timeout.
- `setup_task()` writes `instruction.md` to the sandbox and can optionally
  inject `environment/skills/` for paired with-skills runs.
- `evaluate_instance()` uploads `tests/` after the agent run, executes
  `/tests/test.sh`, and parses `/logs/verifier/reward.txt` or
  `/logs/verifier/reward.json`.
- `calculate_final_score()` aggregates task rewards into a Valkyrie final
  score.

## Datasets

The service currently exposes these datasets:

- `default`: tasks from `tasks/`, no skills injected.
- `with-skills`: tasks from `tasks/`, injects `environment/skills/` when a task
  has skills.
- `extra`: tasks from `tasks-extra/`, no skills injected.
- `extra-with-skills`: tasks from `tasks-extra/`, injects skills when present.

`tasks-extra/` remains opt-in because many of those tasks require credentials,
GPU resources, or external integrations.

The SkillsBench task source is included as the `skillsbench/` git submodule.
The service automatically discovers that submodule. To point at another checkout
or mounted dataset, set:

```bash
export SKILLSBENCH_REPO_ROOT=/path/to/skillsbench
```

## Required Images

Valkyrie creates sandboxes from an image or snapshot. SkillsBench stores
per-task Dockerfiles, so production use should prebuild each
`tasks/<task>/environment/Dockerfile` and provide an image manifest.

Set:

```bash
export SKILLSBENCH_REPO_ROOT=/path/to/skillsbench
export SKILLSBENCH_VALKYRIE_IMAGE_MANIFEST=/path/to/image-manifest.json
```

Manifest shape:

```json
{
  "default_image": "python:3.12-slim",
  "tasks": {
    "latex-formula-extraction": {
      "image": "ghcr.io/benchflow-ai/skillsbench-latex-formula-extraction:sha",
      "cwd": "/root",
      "resources": { "vcpu": 4, "memory": 10, "disk": 20 }
    },
    "some-snapshot-task": {
      "snapshot": "skillsbench-some-snapshot-task-sha"
    }
  }
}
```

If no manifest entry exists, the service falls back to
`SKILLSBENCH_VALKYRIE_DEFAULT_IMAGE` and then `python:3.12-slim`. That fallback
is only useful for smoke tests or very simple tasks; real SkillsBench tasks need
their Dockerfiles prebuilt.

Build a small verification slice and write the manifest:

```bash
python scripts/build_task_images.py \
  --registry <account>.dkr.ecr.us-east-1.amazonaws.com/skillsbench-task-images \
  --output image-manifest.json \
  --push \
  dialogue-parser llm-prefix-cache-replay
```

For hosted Valkyrie/Daytona, publish images to a registry Daytona can pull
without local Docker credentials and build for the hosted runner platform:

```bash
python scripts/build_task_images.py \
  --registry public.ecr.aws/<alias>/skillsbench-task-images \
  --output image-manifest.json \
  --platform linux/amd64 \
  --push \
  dialogue-parser llm-prefix-cache-replay
```

## Local Smoke Run

```bash
uv sync --locked
uv run uvicorn skillsbench_valkyrie.main:app --host 127.0.0.1 --port 8001
```

Then register the service with Valkyrie:

```bash
valkyrie config service set skillsbench http://localhost:8001
```

For public Valkyrie runs, expose the service through the usual tunnel or
deployment path accepted by your Valkyrie environment.

## Registry Entry

The corresponding `public-benchmark-services-registry` entry should point to
this repository as an HTTPS submodule:

```yaml
services:
  skillsbench:
    path: skillsbench-benchmark-service
    repository: https://github.com/benchflow-ai/skillsbench-benchmark-service.git
    branch: main
    public_dependencies:
      - create-benchmark-service
      - skillsbench
    dataset_submodules:
      - https://github.com/benchflow-ai/skillsbench.git
```

## Agent Porting Feedback

This port was completed primarily by an agent using the Valkyrie onboarding doc,
`create-benchmark-service`, and examples from the public benchmark registry.
The process was mostly hands-off once the benchmark-service contract was clear.

Useful improvements for future agent-assisted ports:

- Put the "registry PR" end state directly in the onboarding checklist: create a
  standalone benchmark-service repo, then add it as a submodule and
  `services.yaml` entry in `public-benchmark-services-registry`.
- Make the required production image strategy explicit. SkillsBench needs one
  pullable image or snapshot per task because every task has its own Dockerfile.
- Call out that hosted Daytona runners need public or otherwise pullable
  `linux/amd64` images. Private ECR images fail late and can look like sandbox
  naming/retry issues.
- Include a minimal working `Dockerfile` and `Makefile` in the scaffold so the
  service can be deployed without reverse-engineering the ASGI command.
- Include a checklist for leakage safety: upload `tests/` only in
  `evaluate_instance`, keep `solution/` out of the sandbox, and inject skills
  only for with-skills datasets.
