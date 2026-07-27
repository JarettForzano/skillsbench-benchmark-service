# Task 8 — SkillsBench PR #4 attention pass

## Scope and topology

- PR base after the required refresh: `07ff6ae` (`origin/dev`)
- Refreshed local merge: `2fe574e`
- Corrective commit: `8f4096917076ca3bb938c30781bf8c64f61d5dfb`
- Files changed by the correction: `src/skillsbench_valkyrie/service.py`, `tests/test_service.py`

The full PR diff was reviewed against the refreshed `origin/dev`. The pre-existing snapshot lifecycle, fresh-sandbox restore, strict state parsing, and cancellation-safe temporary-sandbox cleanup remain in place because each is directly required by the retry behavior. No compatibility path, central retry abstraction, or duplicate validation pass was added. No existing production code was removed: the demonstrated gaps required adding checkpoint bindings and the originating label; the test updates replace old unbound state construction with valid contract-bound state.

## Red-green evidence

Red command:

```text
uv run pytest tests/test_service.py -k 'contract_drift or preserve_originating' -q
```

Before the fix, it failed four cases: task, image, and verifier drift all invoked `create_provider`, and the restored sandbox request lacked `labels["Id"]`.

Green command:

```text
uv run pytest tests/test_service.py -k 'contract_drift or preserve_originating' -q
```

Result: `4 passed, 27 deselected`.

After fetching and merging `origin/dev`, current-head verification was:

```text
uv run --frozen pytest -q
```

Result: `32 passed`. `git diff --check` also passed.

The frozen environment has Ruff configuration but does not pin/install either static executable:

```text
uv run --frozen ruff check src/skillsbench_valkyrie/service.py tests/test_service.py
# Failed to spawn: ruff
uv run --frozen basedpyright src/skillsbench_valkyrie/service.py
# Failed to spawn: basedpyright
```

This is existing unpinned static-tool debt; no unpinned download was introduced.

## Review-thread dispositions

No GitHub review-thread reads, replies, resolutions, pushes, or PR edits were performed in this task. The local audit found two actionable correctness gaps and addressed them with focused tests: contract drift is rejected before provider access, and restored sandbox labels retain the originating `Id`.

## Proposed PR body

### What changed

SkillsBench eval-resume state now carries the originating run ID and a digest of the task metadata, snapshot/image configuration, and verifier inputs. Resume restores the originating `Id` label on its fresh sandbox.

### Why

A snapshot is only safe to reuse for the exact task, image, and verifier contract that created it. Drift is rejected before a provider is opened, preventing a stale workspace from being graded under different inputs.

### Verification

Focused task/image/verifier-drift and label-preservation tests pass, along with the frozen 32-test suite after merging current `dev`. Ruff and BasedPyright are not pinned in this repository’s frozen environment.

### Rollout

This remains an evaluator-only retry path: resume creates a fresh sandbox from the durable snapshot and does not regenerate work. No live smoke is claimed for the current CBS-merge head.
