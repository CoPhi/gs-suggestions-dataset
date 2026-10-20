---
name: conventional-commits
description: >-
  Creates structured, atomic Git commits following the Conventional Commits specification
  and Keep a Changelog rules for the gs-suggestions-dataset repository. Use this skill
  whenever the user asks to create commits after implementing a new feature, refactoring
  code, fixing a bug, or updating project documentation and pipelines.
---

# Conventional Commits & Changelog Workflow

Follow this runbook to group workspace changes into clean, atomic Git commits that match the repository's history and [`AGENTS.md`](../../../AGENTS.md) rules.

## 1. Inspect Changes & Verify `CHANGELOG.md`

1. Run the inspection helper script [commit_inspect.sh](./scripts/commit_inspect.sh) to review modified, staged, and untracked files alongside recent commit history:
   ```bash
   bash .agents/skills/conventional-commits/scripts/commit_inspect.sh
   ```
2. **Mandatory Rule Check**: Verify that [`CHANGELOG.md`](../../../CHANGELOG.md) has been updated under `## [Unreleased]` with the appropriate section (`### Added`, `### Changed`, `### Fixed`, `### Optimized`) referencing the modified files/modules. If not updated yet, update `CHANGELOG.md` before creating the final documentation/chore commit.
3. Ensure no temporary artifacts, scratch files, unignored binary checkpoints, or `.env` secrets are staged.

## 2. Group Files into Atomic Commits

When a task spans multiple architectural layers, split the changes into logical, self-contained commits ordered by dependency (bottom-up):

1. **Core Model / Paradigm Layer** (`feat(<model_scope>)`, `refactor(<model_scope>)`, `perf(<model_scope>)`):
   - Files under `models/<paradigm>/` (e.g., `models/rag/`, `models/bert/`, `models/ithaca/`, `models/ngrams/`) or `packages/`.
2. **Backend API, Config & Core Preprocessing** (`feat(api,<scope>)`, `fix(preprocess)`):
   - Files under `backend/config/`, `backend/core/`, `backend/api/models.py`, `backend/api/services/`, `backend/api/routes/`.
3. **Frontend UI Layer** (`feat(frontend)`, `fix(frontend)`):
   - Files under `frontend/`.
4. **CLI Pipelines, Tests, Skills & Documentation** (`test(<scope>)`, `chore(release)`, `docs`, or combined `feat(<scope>,cli)`):
   - `tests/`, `Makefile`, `scripts/`, `.agents/skills/`, `README.md`, `CHANGELOG.md`.

*(Note: For small, single-purpose fixes, a single atomic commit containing code, tests, and `CHANGELOG.md` is preferred).*

## 3. Format Commit Messages

Format every commit message in **English** (matching the repository's `git log` history) using the following structure:

```text
<type>(<scope>): <concise imperative summary under 72 chars>

- <Bullet 1: key module/class added or modified and its responsibility>
- <Bullet 2: architectural detail, algorithm, or API contract change>
- <Bullet 3: configuration, CLI target, test suite, or documentation update>
```

### Allowed `<type>` Values
- `feat`: New feature, model family, pipeline, endpoint, or CLI target (maps to `### Added` in `CHANGELOG.md`).
- `refactor`: Structural code change that neither fixes a bug nor adds a feature (maps to `### Changed`).
- `fix`: Bug fix, dependency alignment, or edge-case patch (maps to `### Fixed`).
- `perf`: Memory, latency, Docker build, or inference optimization (maps to `### Optimized`).
- `test`: Adding or updating unit/integration tests in `tests/`.
- `docs`: Documentation-only changes (`README.md`, `CHANGELOG.md`, docstrings).
- `chore`: Build scripts, `Makefile`, version bumps, or agent customizations (`.agents/`).

### Common Project `<scope>` Identifiers
`rag`, `bert`, `ithaca`, `ngrams`, `api`, `core`, `preprocess`, `dataset`, `sweep`, `evaluation`, `frontend`, `docker`, `release`, `agents`.

## 4. Execute & Validate Commits

For each planned commit:
1. Stage only the target files:
   ```bash
   git add <file1> <file2> ...
   ```
2. Verify the staged diff before committing:
   ```bash
   git diff --cached --stat
   ```
3. Create the commit using multiple `-m` flags or a heredoc to preserve bullet formatting.
4. **Final Verification**: After all commits are created, run:
   ```bash
   git status --short && git log -n 5 --stat
   ```
   Confirm that the working tree is clean and all commit messages follow the format above.
