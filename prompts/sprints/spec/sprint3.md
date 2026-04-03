# Sprint 3: 결과 시각화 및 통합 테스트

**상태**: ✅ **완료** (2026-04-02)

**목표**: Experiment 결과를 논문 스타일의 그래프로 시각화하고 PDF로 저장. End-to-end 파이프라인 검증.

**완료 조건**: 시각화 PDF 생성 확인, 통합 테스트 통과. ✅ **달성**

---

## Task 3.1: 결과 시각화 및 통합 테스트

**설명**:
- `Visualizer` 클래스: 실험 결과를 논문 스타일 그래프로 변환
  - X축: time step t
  - Y축: cumulative regret
  - 플롯: mean regret (실선), 5%-95% quantile range (음영)
  - 스타일: 학술 논문 figure 수준
  
- End-to-end 통합 테스트:
  - Sprint 1, 2의 모든 컴포넌트가 함께 동작하는지 검증
  - 작은 파라미터에서 실행: `num_episodes=2, d=3, K=5, T=10`
  - 결과 파일 생성 확인
  - 시각화 PDF 생성 확인

**입력/의존성**:
- Sprint 1 완료 (Environment, OFUL)
- Sprint 2 완료 (ExperimentRunner)

**출력/결과물**:
- `src/visualization.py`: `Visualizer` 클래스
  - `plot_regret(results, save_path, title=None)`
    - Input: ExperimentRunner의 반환값 (dict with regrets, configs)
    - Output: PDF 파일 저장
  - Matplotlib 사용 (폰트: Times New Roman 또는 serif)
  - 그래프 요소: 타이틀, 축 레이블, 범례, 그리드
  
- `tests/test_integration.py`: 통합 테스트
  - Environment 생성 → OFUL 알고리즘 생성 → ExperimentRunner 실행 → 결과 저장 및 시각화
  - 검증 항목:
    - 데이터 형태 확인
    - Regret 계산 정확성 (각 episode마다 positive 누적)
    - PDF 파일 생성 확인

**JAX/Numpy 사용**:
- Regret 계산: numpy operations
- 시각화: matplotlib

**평가 기준**:
- ✅ Mean regret 선 그리기
- ✅ 5%-95% quantile 음영 처리
- ✅ 축 레이블, 타이틀, 범례 포함
- ✅ PDF 저장 동작 확인
- ✅ 통합 테스트 통과 (모든 컴포넌트 연동)
- ✅ 결과 논문 수준의 가독성

**추가 노트**:
- Quantile 계산: `np.percentile(regrets, [5, 95], axis=0)`
- 음영 영역: `matplotlib.pyplot.fill_between()` 사용
- 결과 저장 경로: `results/oful_regret.pdf`

---

## 프로젝트 최종 구조 (Sprint 3 완료 후)

```
src/
├── __init__.py
├── base.py                      # Environment (ABC), Algorithm (ABC)
├── environment.py               # ContextualLinearBandit(Environment)
├── experiment.py                # ExperimentRunner
├── visualization.py             # Visualizer
└── algorithms/
    ├── __init__.py
    └── oful.py                 # OFUL(Algorithm)

tests/
├── __init__.py
├── test_environment.py
├── test_oful.py
├── test_experiment.py
└── test_integration.py

results/                         # 실험 결과 (runtime에 생성)
├── oful_regret.pdf
├── experiment_results.npz
└── ...
```

---

**Last Updated**: 2026-04-02
