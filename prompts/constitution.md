# Projection Constitution

## Overview

이 문서는 프로젝트 전반에 적용될 헌법을 규정합니다.

## Project Description

밴딧 알고리즘을 JAX를 사용하여 구현하는 간단한 프로젝트입니다. 여러 개의 알고리즘을 구현하고, 한 에피소드에서 성능(regret)을 비교합니다.

## Workflow

이 프로젝트에서 에이전트는 **Planner, Developer, Reviewer**의 3가지 역할을 수행합니다.

워크플로우에 대한 자세한 내용은 [workflow.md](./workflow.md)를 참고하세요.

## Python Execution

- **필수**: `uv run python` 사용 (절대 `python` 직접 실행 금지)
- 이유: 프로젝트의 모든 의존성이 uv를 통해 관리됨
- 예시:
  ```zsh
  uv run python script.py
  uv run python -m pytest tests/
  ```

## Dependency Management

- 모든 패키지는 `pyproject.toml`을 통해 관리
- 새 패키지 추가 시: `uv add <package>`
- 프로젝트 설정 변경 금지 (기존 구조 유지)

## File Management Philosophy

### Principle

- **One Concern Per File**: 각 파일은 하나의 명확한 책임만 가짐
- **File Size Target**: 프롬프트 파일은 200줄 이하 유지, 불가능할 경우 파일을 쪼갠 뒤 해당 파일을 링크
- **Linking**: 문서 간 명확한 Markdown 링크로 관계 표현

### Folder Structure Reference

```
.claude/
└── agents/               # 에이전트 역할 정의 (Planner, Developer, Reviewer)

prompts/
├── constitution.md       # 이 파일 (원칙 & 가이드라인)
├── workflow.md           # 멀티에이전트 워크플로우 가이드
└── sprints/              # Sprint 작업 관리
    ├── spec/             # 진행할 sprint 상세 정의
    │   ├── sprint1.md
    │   └── ...
    └── complete/         # 완료된 sprint 요약
        ├── sprint1.md
        └── ...

src/                      # 실제 구현 코드
tests/                    # 테스트 코드
```

**Last Updated**: 2026-04-02
