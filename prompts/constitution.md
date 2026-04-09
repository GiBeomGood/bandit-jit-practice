# Project Constitution

## Overview

This document defines the project-wide rules and constraints that all agents must follow.

## Project Description

A research project implementing bandit algorithms in JAX. Multiple algorithms are implemented and compared in terms of cumulative regret over a single episode.

## Workflow

This project uses a **3-agent multi-agent workflow**: `sprint-planner`, `developer`, and `code-reviewer`.

For orchestration details, see [workflow.md](./workflow.md).

## Python Execution

- **Required**: Always use `uv run python` — never invoke `python` directly.
- Reason: All project dependencies are managed through `uv`.
- Examples:
  ```zsh
  uv run python script.py
  uv run python -m pytest tests/
  ```

## Dependency Management

- All packages are managed via `pyproject.toml`.
- To add a new package: `uv add <package>`
- Do not modify project configuration unless explicitly required by a task.

## File Management Philosophy

### Principles

- **One Concern Per File**: Each file has exactly one clearly scoped responsibility.
- **File Size Target**: Keep prompt files under 200 lines. If a file exceeds this, split it and link between files.
- **Linking**: Express relationships between documents using explicit Markdown links.
- **Language**: All documents (specs, plans, prompts) must be written in English.

### Folder Structure Reference

```
.claude/
└── agents/               # Agent role definitions (sprint-planner, developer, code-reviewer)

prompts/
├── constitution.md       # This file — project-wide rules and constraints
├── workflow.md           # Multi-agent orchestration guide
└── sprints/              # Sprint work management
    ├── spec/             # Sprint specifications (input to developer)
    │   ├── sprint1.md
    │   └── ...
    ├── details/          # Supplementary detail documents for complex tasks
    │   ├── task_1_1_topic.md
    │   └── ...
    ├── todo/             # Deferred tasks (status: open|completed in frontmatter)
    │   │                 # Created by: planner (scope overflow) or orchestrator (BLOCKED ×5)
    │   │                 # Scanned by: planner (startup) and orchestrator (sprint completion)
    │   ├── todo_sprint1_task_title.md
    │   └── ...
    └── complete/         # Completed sprint summaries (written by orchestrator, in Korean)
        ├── sprint1.md
        └── ...

src/                      # Core implementation code
tests/                    # Test code
configs/                  # YAML configuration files
```

## Todo File Template

Every file under `todo/` must use this format:

```markdown
---
status: open          # open | completed
origin_sprint: N
title: Short task title
---

## Background
Why this was deferred and what the original context was.

## What Needs to Be Done
Clear description of the task.

## Acceptance Criteria
- Criterion 1
- Criterion 2
```

File naming: `todo_sprintN_task_title.md` (N = origin sprint number).

**Last Updated**: 2026-04-09
