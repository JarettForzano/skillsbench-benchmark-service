# Porting SkillsBench to Valkyrie

This guide describes the intended path for running SkillsBench as a Valkyrie
benchmark. The recommended port is a Valkyrie benchmark service, not a wrapper
around `bench eval create`.

## Goal

Expose SkillsBench tasks to Valkyrie while preserving the benchmark invariants:

- Agents see `instruction.md` and task input files.
- Agents do not see `tests/` or `solution/`.
- Skills are injected only for with-skills datasets.
- Verifiers remain outcome-based and read reward from `reward.txt` or
  `reward.json`.
- `tasks-extra/` stays opt-in because some tasks need credentials, GPUs, or
  external integrations.

## Architecture

Valkyrie runs a benchmark through a benchmark-service API:

1. `retrieve_task()` tells Valkyrie which sandbox image or snapshot to create,
   what resources to allocate, where the prompt will be written, and which
   directory the agent should run from.
2. `setup_task()` prepares the live sandbox before the agent starts.
3. Valkyrie runs the selected agent contract inside the sandbox.
4. `evaluate_instance()` runs the verifier after the agent finishes.
5. `calculate_final_score()` aggregates per-task results.

The adapter in `src/skillsbench_valkyrie/service.py` implements those methods
against local SkillsBench task directories.

## Format Mapping

| SkillsBench file or field | Valkyrie service behavior |
| --- | --- |
| `instruction.md` | Uploaded to `problem_path`, usually `<cwd>/instruction.md` |
| `task.toml [agent].timeout_sec` | Returned as `agent_timeout` |
| `task.toml [verifier].timeout_sec` | Used as verifier command timeout |
| `task.toml [environment].cpus` | Returned as `Resources.vcpu` |
| `task.toml [environment].memory_mb` | Converted to Valkyrie memory GB |
| `task.toml [environment].storage_mb` | Converted to Valkyrie disk GB |
| `environment/Dockerfile` | Prebuilt into an image or snapshot, referenced by manifest |
| `environment/*` | Uploaded during setup, excluding `Dockerfile` and `skills/` |
| `environment/skills/` | Uploaded to `/skills` only in with-skills datasets |
| `tests/` | Uploaded during `evaluate_instance()`, after the agent has run |
| `tests/test.sh` | Executed from `cwd` as `/tests/test.sh` |
| `/logs/verifier/reward.txt` | Parsed as `{"reward": float}` |
| `/logs/verifier/reward.json` | Parsed as structured reward payload |

## Datasets

The adapter exposes four datasets:

- `default`: `tasks/`, no skills injected.
- `with-skills`: `tasks/`, inject `environment/skills/` when present.
- `extra`: `tasks-extra/`, no skills injected.
- `extra-with-skills`: `tasks-extra/`, inject skills when present.

Use `default` and `with-skills` first. Add `extra` datasets only when the
required credentials and hardware are available.

## Image Manifest

Valkyrie creates sandboxes from an image or a snapshot. SkillsBench currently
stores a Dockerfile per task, so real runs need a manifest that maps task IDs to
prebuilt images or snapshots.

Set these environment variables on the service:

```bash
export SKILLSBENCH_REPO_ROOT=/path/to/skillsbench
export SKILLSBENCH_VALKYRIE_IMAGE_MANIFEST=/path/to/image-manifest.json
```

Manifest shape:

```json
{
  "default_image": "python:3.12-slim",
  "tasks": {
    "dialogue-parser": {
      "image": "ghcr.io/benchflow-ai/skillsbench-dialogue-parser:<sha>",
      "cwd": "/app",
      "resources": { "vcpu": 1, "memory": 4, "disk": 10 }
    },
    "latex-formula-extraction": {
      "image": "ghcr.io/benchflow-ai/skillsbench-latex-formula-extraction:<sha>",
      "cwd": "/root",
      "resources": { "vcpu": 4, "memory": 10, "disk": 20 }
    }
  }
}
```

If a task entry contains `snapshot`, the service returns a Valkyrie
`SnapshotSource` instead of an image:

```json
{
  "tasks": {
    "some-task": {
      "snapshot": "skillsbench-some-task-<sha>",
      "cwd": "/root"
    }
  }
}
```

The current fallback image is only for smoke tests. Most real SkillsBench tasks
will not pass on `python:3.12-slim` because their Dockerfiles install task
dependencies and copy input data.

## Image Build Plan

Build one image per task from `tasks/<task>/environment/Dockerfile`.

Recommended image tag:

```text
<registry>/skillsbench-<task-id>:<git-sha>
```

Recommended builder behavior:

1. Discover task directories from `tasks/*/task.toml`.
2. Build with context `tasks/<task>/environment`.
3. Push the image.
4. Infer `cwd` from the last absolute `WORKDIR` in the Dockerfile, unless
   overridden.
5. Read resource defaults from `task.toml`.
6. Write the manifest consumed by the service.

Do not bake `environment/skills/`, `tests/`, or `solution/` into images. Skills
must remain mode-dependent, and tests/solutions must not be visible before
evaluation.

For hosted Valkyrie/Daytona, use images that the hosted runner can pull without
local Docker credentials. A private ECR image creates the Daytona sandbox object
and then fails during image resolution with `401 Unauthorized`, which can later
surface in Valkyrie as a misleading `Sandbox with name ... already exists`
retry error. The verified path is to publish `linux/amd64` images to a public
registry, for example ECR Public:

```bash
python scripts/build_task_images.py \
  --registry public.ecr.aws/<alias>/skillsbench-task-images \
  --output /tmp/skillsbench-valkyrie-image-manifest.json \
  --platform linux/amd64 \
  --push \
  dialogue-parser llm-prefix-cache-replay
```

## Agent Contracts

Valkyrie agents are configured through agent contracts. The agent command gets
`{problem_statement_path}` and `{task_id}`.

A minimal contract should:

- read the instruction from `{problem_statement_path}`;
- run from the `cwd` returned by the benchmark service;
- use normal filesystem tools to create the requested outputs;
- optionally look for injected skills under `/skills`;
- write agent logs/artifacts under `/logs/agent` if useful.

For Gemini, Claude, Codex, or other existing agent CLIs, create wrapper
contracts that pass the prompt file and model settings through the agent's
native command.

## Local Service Smoke

Run the service locally:

```bash
uv sync --locked
uv run uvicorn skillsbench_valkyrie.main:app --host 127.0.0.1 --port 8001
```

Useful checks:

```bash
curl http://127.0.0.1:8001/health
curl "http://127.0.0.1:8001/verify-task-ids?task_ids=dialogue-parser&dataset=default"
curl "http://127.0.0.1:8001/retrieve-task/?task_id=dialogue-parser&dataset=default"
```

Live sandbox setup and evaluation use WebSockets, so use
`BenchmarkServiceClient` or Valkyrie's tracker rather than plain `curl` for the
full flow.

## Valkyrie CLI Setup

A full `valkyrie run start` requires Valkyrie config:

```bash
valkyrie config init
valkyrie config service set skillsbench https://<service-url>
```

The config must include the AWS/S3/tracker fields Valkyrie uses to store agents,
outputs, logs, and Daytona credentials. Without this, the CLI stops with:

```text
Config not found. Run `valkyrie config init` first.
```

For hosted benchmark-service auth, pass the service header expected by the
deployment, for example:

```bash
valkyrie run start \
  --benchmark skillsbench \
  --dataset with-skills \
  --agent <agent-id-or-agent-path> \
  --task-ids dialogue-parser \
  -H Authorization <credential>
```

The lab-facing `/v1/datasets/.../tasks` endpoint requires Descope auth. The
tracker-style internal routes work with legacy/local service auth.

## Test Ladder

Use this sequence when porting:

1. Unit tests for the adapter with a fake sandbox.
2. Local HTTP/WebSocket service test with `BenchmarkServiceClient`.
3. Real Daytona sandbox using a tiny synthetic task.
4. BenchFlow oracle on a real task, to validate the task itself.
5. Valkyrie service flow on a real task image.
6. Full `valkyrie run start` with a real agent contract.
7. Batch runs across `default` and `with-skills`.

Current verified checks:

- `dialogue-parser` oracle through BenchFlow + Daytona: reward `1.0`.
- `dialogue-parser` Gemini through BenchFlow + Daytona with skills: reward
  `1.0`.
- `xlsx-recover-data` oracle through BenchFlow + Daytona: reward `1.0`.
- `xlsx-recover-data` Gemini through BenchFlow + Daytona with skills: reward
  `0.0`; verifier ran cleanly, but the model wrote formulas where numeric cell
  values were expected.
- Valkyrie benchmark-service HTTP/WebSocket smoke with a real Daytona sandbox:
  reward `1.0`, final score `100.0`.

## Hosted Valkyrie Status

The hosted custom benchmark-service path is wired up and verified:

- Local SkillsBench service is reachable through a public tunnel.
- `valkyrie benchmark tasks skillsbench --dataset with-skills` returns 88 task
  IDs through the hosted tracker.
- `retrieve-task` compatibility returns both the current `source` field and the
  legacy `docker_image` field expected by the hosted worker version.
- `with-skills` runs over current `tasks/` completed under hosted
  Valkyrie/Daytona with OpenCode + DeepSeek v4 Flash.
- The audited trial=1 result was `44/88 = 50.0%` strict pass@1. A subagent
  trajectory audit found no benchmark oracle leakage or reward hacking.

Earlier blocker: hosted runs failed before task setup while the worker downloaded
the frozen agent zip into the Daytona sandbox:

```text
Sandbox error: Failed to upload contract <agent> to sandbox <task_alias>:
Command failed with exit code 22
```

This reproduced with both `gemini-cli-v1-0-0` and `noop-v1-0-0`, and also with
`terminal-bench`, so it was not SkillsBench-specific.

Root cause: Valkyrie's tracker helper generates presigned S3 URLs against the
global endpoint, for example:

```text
<bucket>.s3.amazonaws.com
```

For the original `us-east-2` bucket, S3 returned `TemporaryRedirect` with the
regional endpoint:

```text
<bucket>.s3.us-east-2.amazonaws.com
```

The hosted worker invokes `curl -sfL`; the S3 redirect response produced curl
exit code `22`, so the contract zip never reached the sandbox.

Workaround verified: use a us-east-1 harness bucket and mirror the required
Secrets Manager secrets there. The same Valkyrie helper URL downloads cleanly
for the us-east-1 bucket. With config switched to us-east-1:

- `terminal-bench` + `noop-v1-0-0` reached normal benchmark execution.
- SkillsBench `dialogue-parser` + `noop-v1-0-0` finished with no task errors,
  reward `0.0` as expected.
- SkillsBench `dialogue-parser` + `gemini-cli-v1-0-0` finished with no task
  errors, skills injected, reward `0.667`, final score `66.7`.

Second blocker: task images initially published to private ECR were not
pullable by hosted Daytona. Daytona sandboxes entered `BUILD_FAILED` with:

```text
unexpected status from HEAD request to
https://<account>.dkr.ecr.<region>.amazonaws.com/v2/.../manifests/<tag>:
401 Unauthorized
```

Those failed sandboxes remained in Daytona, so Valkyrie's retry path reported
`Sandbox with name <task_alias> already exists`. Deleting the failed Daytona
sandboxes and switching the manifest to public `linux/amd64` images fixed the
issue.

With public ECR images:

- `dialogue-parser` + `noop-v1-0-0` finished with no task errors, skills
  injected, reward `0.0`.
- `dialogue-parser` + `gemini-cli-v1-0-0` finished with no task errors, skills
  injected, reward `0.833`, final score `83.3`.
- A five-task no-op sweep over `dialogue-parser`,
  `llm-prefix-cache-replay`, `civ6-adjacency-optimizer`,
  `azure-bgp-oscillation-route-leak`, and
  `tictoc-unnecessary-abort-detection` finished with no task errors and
  skills injected for all five tasks.

## Remaining Work

- Decide the permanent public image registry or configure hosted Daytona with
  private registry credentials.
- Add the service repository to `vals-ai/public-benchmark-services-registry` as
  the `skillsbench` benchmark service.
- Add Claude/Codex/OpenCode public agent contracts if those are needed for the
  comparison matrix.
- Ask Vals to update tracker presigned URL generation to use regional S3
  endpoints, for example `endpoint_url=f"https://s3.{region}.amazonaws.com"` or
  an equivalent botocore S3 regional endpoint configuration.
- Keep the public task-image manifest current with each SkillsBench release.
- Add CI smoke tests for adapter routes and manifest validation.
- Document credential-dependent `tasks-extra/` requirements separately.
