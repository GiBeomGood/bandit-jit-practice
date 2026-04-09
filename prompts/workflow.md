# Bandit Algorithm Project Workflow

## Workflow Overview

This project follows a **3-stage multi-agent workflow**:

```
[sprint-planner] → ([developer] ⇄ [code-reviewer]  (max 5 cycles)) → [Completion]
```

Each role is strictly separated. The developer-reviewer cycle enforces quality before any task is marked complete.

Agent role definitions live in `.claude/agents/`. Each agent reads `prompts/constitution.md` as its first step.

## Invoking Agents Directly

```
@sprint-planner  Plan Sprint N — <brief goal description>
@developer       Implement Task N.M — <brief description>
@code-reviewer   Review Task N.M implementation
```

> Subagents run in isolated contexts. Always specify which Sprint/Task is being worked on in the prompt, and include all necessary context explicitly — do not assume the agent has memory of prior steps.

## Orchestration Procedure

When the main Claude orchestrates a full sprint run (e.g., user says "Run Sprint N"), follow these steps.

### Step 1: Read the Sprint Spec

Read `prompts/sprints/spec/sprintN.md` to extract:
- Sprint Goal
- Full task list
- Execution order and dependency graph (parallel vs. sequential tasks)

### Step 2: Execute Tasks in Order

For each task, run a developer → code-reviewer cycle.

**When calling the `developer` agent, include:**
- Sprint number and Task number/title
- Full task content copied from the spec (description, inputs/outputs, acceptance criteria)
- File paths of any prerequisite task outputs (if applicable)
- Path to any associated detail document under `prompts/sprints/details/` (if one exists)

**When calling the `code-reviewer` agent, include:**
- Sprint number and Task number/title
- List of files changed or created by the developer
- Path to the sprint spec file (`prompts/sprints/spec/sprintN.md`)
- Developer's implementation summary (if provided)

### Step 3: Handle Results

| Result | Action |
|--------|--------|
| APPROVED | Proceed to the next task |
| BLOCKED | Re-invoke `developer` with the full reviewer feedback; retry up to 5 cycles |
| BLOCKED after 5 cycles | Stop and report to the user with a summary of unresolved issues |

### Step 4: Sprint Completion

After all tasks are APPROVED:
1. Summarize completed tasks and key design decisions to the user.
2. Optionally write a completion summary to `prompts/sprints/complete/sprintN.md`.

## Orchestration Notes

- **Context isolation**: Each subagent starts fresh. Do not rely on implicit shared state — always pass relevant context explicitly in the prompt.
- **Parallelism**: If the sprint spec marks tasks as parallel-safe, you may invoke multiple `developer` agents concurrently, but each must be reviewed independently before the sprint is marked complete.
- **Planner-first for new sprints**: If the user requests a new sprint without an existing spec, invoke `sprint-planner` first before calling `developer`.
- **Design issues**: If the `code-reviewer` routes an issue to the Planner (structural/design flaw), pause the cycle and surface it to the user before proceeding.

**Last Updated**: 2026-04-09
