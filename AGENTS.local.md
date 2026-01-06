# Local Agent Instructions (Not for Git)

This file is local-only and should never be committed. It defines the workflow
we want to follow so changes stay clean and easy to PR or cherry-pick upstream.

## Workflow goals
- Keep `main` fast-forwardable and in sync with `upstream/main`.
- Do all work on short-lived feature branches.
- Produce focused commits that are easy to review, squash, or cherry-pick.
- Follow repo contribution norms (typing, async, builder pattern).

## Sync with upstream
1) `git fetch upstream`
2) `git switch main`
3) `git merge upstream/main`
4) `git push origin main`

## Create a feature branch
`git switch -c feature/<short-desc>`

## During development
- Keep changes scoped to one topic per branch.
- Avoid unrelated formatting or refactors.
- Prefer small, descriptive commits when it helps review.

## Before PR or merge
- Rebase on latest upstream:
  - `git fetch upstream`
  - `git rebase upstream/main`
- Ensure tests or checks relevant to the change are run (note in PR if skipped).

## Merge strategy
- Prefer PRs from feature branches and squash-merge them.
- If merging locally, do a squash merge:
  - `git switch main`
  - `git merge --squash feature/<short-desc>`
  - `git commit -m "<summary>"`
  - `git push origin main`

## Cherry-picks
- Keep commits self-contained and avoid merge commits so cherry-picks are clean.

## Contribution norms (from CONTRIBUTING.md)
- Typing: avoid `Any` and `type: ignore`; prefer simple, explicit types over convoluted generics.
- Async: use async for operations that take nontrivial time; keep sync loops beginner-friendly where intended.
- Builder pattern: config objects are `chz` configs with `.build()`/`__call__()` that return heavyweight objects.
- Training scripts: keep a main training loop with a rich config, and separate CLI recipe configs.

## Local review
- Before finalizing, ask another agent for a quick review.
- Example:
- `claude --print "Please review this feature branch after checking AGENTS.md, CONTRIBUTING.md, AGENTS.local.md. <brief description of the feature and anything useful to point out about the code or the state of the branch>"`
- If you disagree and want a follow-up:
- `claude -c "Thanks for the feedback. I understand what you're saying, but <response>. Please reevaluate and confirm."`
- Reviews can take time; when running the command, allow at least 1-2 minutes for the agent to respond.
- Fix any substantive issues identified in review. If changes were made, continue the existing review session with `claude -c` rather than starting a new one. Use judgment on minor nits.

## Local tooling
- Use uv for env + test runs.
- If pytest isn't available yet, install dev extras: `uv sync --extra dev`.
- Run unit tests via: `uv run pytest tinker_cookbook/tests/test_renderers.py`
- Run unit tests via: `uv run pytest tinker_cookbook/tests/test_utils.py`
- If renderer tests fail on Kimi, install: `uv pip install tiktoken`.
- Install git hooks: `uv run pre-commit install`.
- Run hooks manually: `uv run pre-commit run --all-files`.
