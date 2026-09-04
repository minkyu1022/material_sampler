# Cu–Ni N=108 Fig. 2e path-weight debugging instructions

Fig. 2b–d raw terminal reproduction은 충분히 확인됐으므로 그대로 보존하고, 이제 **Fig. 2e path-weight/free-energy failure만 별도로 디버깅**해주세요.  
**N=256은 scope에서 제외합니다.**

현재 ESS≈1이므로 trajectory 수를 brute-force로 늘리거나 현재 Fig. 2e를 더 다듬는 것은 하지 마세요.

## 1. 먼저 path-weight implementation 자체를 analytic toy에서 검증

Cu–Ni 코드와 동일한:
- linear interpolant
- Euler–Maruyama
- forward/backward Gaussian path ratio
- Eq. 19–21 estimator

를 사용하되, **target / prior / exact velocity / exact score / partition function을 모두 아는 Gaussian toy problem**을 만드세요.

확인:
- estimated `log Z`가 analytic value와 일치하는가
- weighted observables가 exact value와 일치하는가
- exact fields를 넣었을 때 ESS가 정상적인가
- integration step 수 `M`을 늘릴수록 estimator가 수렴하는가

이 test가 실패하면 Cu–Ni model 문제가 아니라 **path-weight 코드 자체 문제**입니다.

---

## 2. Cu–Ni Eq. 19–20 implementation line-by-line audit

특히 아래를 명시적으로 확인해주세요.

- backward Gaussian drift는 **arrival state `(x_{n+1}, t_{n+1})`**에서 평가
- forward/backward covariance는 정확히 `2 g^2 h`
- author-confirmed
  \[
  g_u^2(T)=g_v^2(T)=0.02^2(T/750)
  \]
- total energy vs per-atom energy 혼용 없음
- `beta`, `Delta mu * N_Cu` sign/convention 정확
- `N v` Jacobian term이 정확히 한 번만 포함
- prior / terminal target density convention 일관
- 모든 log-weight 계산과 누적은 float64

---

## 3. Representative state에서 M-sweep

기존 checkpoint 그대로:

- `T = 750 K`
- `Delta mu = 0.85 eV`

에서

- `M = 50`
- `M = 100`
- `M = 200`
- `M = 400`

을 비교해주세요.

각각 보고:
- `std(logW)`
- `std(logW_u)`
- ESS
- raw terminal observables
- weighted observables

### 해석
- `M`에 따라 weight collapse가 크게 달라지면 → discretization / path-ratio 문제가 큼
- 거의 변하지 않으면 → terminal/proposal mismatch 가능성이 큼

---

## 4. toy + audit가 통과하면 현재 checkpoint mismatch로 판정

이 경우에는 arbitrary fine-tuning을 더 하지 말고, 저자에게 아래만 물어볼 준비를 해주세요.

- Fig. 2e에서 실제 `ntraj` / state
- typical path-weight ESS range
- Fig. 2e checkpoint/training recipe가 Fig. 2b–d와 동일했는지
- free-energy evaluation에 추가 bridge / resampling / proposal modification이 있었는지

---

## 5. 최종 판정은 둘 중 하나로

### A. `path-weight implementation bug found`
→ 수정 후 Fig. 2e 재평가

### B. `implementation validated; current checkpoint has catastrophic importance-weight variance`
→ raw Fig. 2b–d reproduction은 성공으로 유지  
→ Fig. 2e는 author detail 또는 retraining 없이는 unresolved로 기록

**현재 ESS≈1인 Fig. 2e curve는 최종 결과로 사용하지 마세요.**
