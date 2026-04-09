# Bandit Algorithm Project Workflow

## 워크플로우 개요

이 프로젝트는 **3단계 멀티에이전트 워크플로우**를 따릅니다:

```
[Planner] → ([Developer] ⇄ [Reviewer]  (최대 5 사이클)) → [Completion]
```

각 역할은 명확하게 분리되어 있으며, Developer-Reviewer 사이클을 통해 품질을 보장합니다.

각 에이전트의 역할 정의는 `.claude/agents/`를 참고하세요.

## 에이전트 호출 방법

**직접 지정**:
```
@planner Sprint N 기획해줘
@developer Task N.M 구현해줘
@reviewer 이 코드 검토해줘
```

**오케스트레이션 (메인 Claude에게 요청)**:
```
Sprint N 수행해
```

> 서브에이전트는 독립된 컨텍스트로 실행됩니다. 각 에이전트 호출 시 어떤 Sprint/Task인지 명시하세요.

## 오케스트레이션 절차

메인 Claude가 Sprint를 자동 실행할 때 따르는 절차입니다.

### 1단계: Sprint Spec 읽기

`prompts/sprints/spec/sprintN.md`를 읽어 Task 목록과 실행 순서를 확인합니다.

### 2단계: Task 순서대로 실행

각 Task에 대해 Developer → Reviewer 사이클을 돌립니다.

**Developer 호출 시 프롬프트에 포함할 것**:
- Sprint 번호 및 Task 번호/제목
- Task 전체 내용 (spec에서 복사)
- 선행 Task 결과물 파일 경로 (있는 경우)

**Reviewer 호출 시 프롬프트에 포함할 것**:
- Sprint 번호 및 Task 번호/제목
- Developer가 변경/생성한 파일 목록
- Sprint spec 파일 경로 (`prompts/sprints/spec/sprintN.md`)

### 3단계: 결과 처리

| 결과 | 처리 |
|------|------|
| ✅ APPROVED | 다음 Task로 진행 |
| ❌ BLOCKED | Developer 재호출 — Reviewer 피드백 전달, 최대 5 사이클 |
| 5 사이클 후 BLOCKED | 사용자에게 보고 후 중단 |

**Last Updated**: 2026-04-09
