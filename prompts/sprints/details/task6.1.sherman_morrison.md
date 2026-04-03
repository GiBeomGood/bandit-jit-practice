# Task 6.1: Sherman-Morrison 공식 적용 (상세 가이드)

## 수학적 배경

### 현재 방식 (비효율)

OFUL 알고리즘에서 설계 행렬(Design Matrix)을 매 step마다 업데이트:
$$B_{t+1} = B_t + x_t x_t^T$$

현재 구현은 매 step마다 전체 역행렬을 계산:
$$\theta_t = B_t^{-1} \sum_{s=1}^{t} r_s x_s$$

이는 `np.linalg.solve()` 또는 `np.linalg.inv()` 호출로 **O(d³) 복잡도**를 가집니다.

### Sherman-Morrison 공식 (최적화)

Rank-1 업데이트에 대해:
$$(A + uv^T)^{-1} = A^{-1} - \frac{A^{-1} u v^T A^{-1}}{1 + v^T A^{-1} u}$$

우리의 경우 $u = x_t$, $v = x_t$이므로:
$$B_{t+1}^{-1} = B_t^{-1} - \frac{B_t^{-1} x_t x_t^T B_t^{-1}}{1 + x_t^T B_t^{-1} x_t}$$

**복잡도**: O(d²) - **20-30% 성능 향상** 예상

---

**Last Updated**: 2026-04-03
