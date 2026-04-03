# Bandit Algorithm Project Workflow

## 워크플로우 개요

이 프로젝트는 **3단계 멀티에이전트 워크플로우**를 따릅니다:

```
[Planner] → ([Developer] ⇄ [Reviewer]  (최대 5 사이클)) → [Completion]
```

각 역할은 명확하게 분리되어 있으며, Developer-Reviewer 사이클을 통해 품질을 보장합니다. `prompts/roles/` 폴더를 참고하세요.

### 예시: Task 실행 흐름

```
사용자 요청 발생
  ↓
Planner: Task 1.1 - "Hoeffding's Inequality 구현"
  ↓
Developer (Cycle 1): 코드 작성
  ↓
Reviewer: 테스트 → ❌ BLOCKED (테스트 불완전)
  ↓
Developer (Cycle 2): 피드백 반영 &수정
  ↓
Reviewer: 테스트 → ✅ APPROVED
  ↓
다음 Task로 진행
```

**Last Updated**: 2026-04-02
