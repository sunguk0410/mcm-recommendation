# RecRec 추천 모델 개선 내역

## 1. 기존에 어떤 문제가 있었는지

기존 추천 모델은 세션의 상품 ID와 행동 유형을 임베딩한 후 모든 상호작용을 단순 평균하여 RecRec의 정적 증거 벡터 `x`를 생성했다.

### 최근 관심이 충분히 반영되지 않음

기존 masked mean pooling에서는 세션 초기에 본 상품과 가장 최근에 본 상품이 같은 비중을 가졌다.

```text
A 선택 → B 피팅 → C 찜
x = (eA + eB + eC) / 3
```

매장 세션에서는 최근 행동일수록 현재 관심사를 더 잘 나타낼 가능성이 높지만 기존 모델은 이를 구분하지 못했다.

### 행동 강도의 차이가 pooling에 직접 반영되지 않음

상품 선택, 피팅, 찜은 관심 강도가 서로 다르지만 기존 pooling에서는 동일한 비중으로 집계됐다. 행동 임베딩은 행동 종류를 구분했지만, 찜을 선택보다 강한 증거로 취급한다는 서비스 규칙은 명시적으로 반영하지 않았다.

### 피팅 상태를 구분하지 못함

기존 `FITTING` 이벤트 하나로는 상품을 피팅 목록에 추가한 것인지 제거한 것인지 구분할 수 없었다. 피팅 관심이 시작된 시점과 종료된 시점을 별도의 증거로 활용할 수 없었다.

### 구형 구현과 현재 구현이 함께 존재함

`v2`가 포함된 파일과 이전 파일이 함께 남아 어떤 파일이 현재 기준인지 불분명했다. 실행 경로와 체크포인트 이름에도 버전 문자열이 분산되어 있었다.

## 2. 기존 기능에서 무엇을 개선·변경했는지

### 행동 체계 변경

기존 행동 체계:

```text
PAD, PRODUCT_SELECT, FITTING, WISHLIST_ADD, WISHLIST_REMOVE
```

개선된 행동 체계:

```text
PAD, PRODUCT_SELECT, FITTING_ADD, FITTING_REMOVE,
WISHLIST_ADD, WISHLIST_REMOVE
```

행동 임베딩 개수도 5개에서 6개로 변경했다. 구형 `FITTING` 이벤트는 호환 대상으로 남기지 않았으며, 기존 의미가 피팅 추가였던 데이터는 `FITTING_ADD`로 전환했다.

### 합성 데이터 생성 규칙 변경

합성 데이터 생성기가 `FITTING_ADD`와 `FITTING_REMOVE`를 별도로 생성하도록 변경했다. `FITTING_REMOVE`는 해당 상품의 `FITTING_ADD` 이후에만 발생하도록 해 잘못된 상태 전이를 방지했다.

- `FITTING_ADD`: 상품에 대한 긍정 관심 강화
- `FITTING_REMOVE`: 기존 피팅 관심 일부 감쇠
- `WISHLIST_ADD`: 강한 긍정 관심 강화
- `WISHLIST_REMOVE`: 찜 관심 철회

### 파일 및 실행 경로 정리

현재 구현을 유일한 기준으로 사용하기 위해 파일과 산출물 이름에서 `v2`를 제거했다.

```text
src/train.py
src/evaluate.py
src/synthetic_generator.py
synthetic_interactions.jsonl
checkpoints/recrec_best.pt
```

더 이상 사용하지 않는 이전 학습·평가·생성 구현은 제거했다. Docker 이미지도 새 체크포인트 경로를 사용한다.

## 3. 어떤 기능을 새롭게 개발했는지

### Action-Recency Weighted Pooling

기존 masked mean pooling을 행동 강도와 최근성을 함께 반영하는 weighted pooling으로 교체했다.

$$
h_t = \operatorname{LayerNorm}(e_t^{product} + e_t^{action})
$$

$$
w_t = w_t^{action} \times w_t^{recency}
$$

최근성은 가장 최근 행동으로부터의 상대 거리에 지수 감쇠를 적용한다.

$$
w_t^{recency} = \exp(-\lambda d_t)
$$

- $d_t$: 가장 최근 행동으로부터 떨어진 이벤트 수
- $\lambda$: 최근성 감쇠율, 현재 기본값 `0.25`

최종 정적 증거 벡터는 다음 가중 평균으로 계산한다.

$$
x = \frac{\sum_t w_t h_t}{\sum_t w_t + \epsilon}
$$

현재 행동별 초기 가중치는 다음과 같다.

| 행동 | 가중치 |
|---|---:|
| `PAD` | 0.0 |
| `PRODUCT_SELECT` | 1.0 |
| `FITTING_ADD` | 1.5 |
| `FITTING_REMOVE` | 0.5 |
| `WISHLIST_ADD` | 2.0 |
| `WISHLIST_REMOVE` | 0.5 |

`PAD`는 항상 제외하며 분모가 0인 경우에도 NaN이 발생하지 않도록 안전한 최소 분모를 적용했다.

### 모델 설정 저장 및 복원

`num_behavior_types`, `recency_decay`, `action_weights`를 체크포인트 설정에 저장하고 학습·평가·추론에서 동일하게 복원하도록 개발했다. 이를 통해 학습과 운영 추론이 서로 다른 행동 매핑이나 가중치를 사용하는 문제를 방지했다.

### Weighted Pooling 단위 테스트

다음 동작을 자동 검증하는 테스트를 추가했다.

- 왼쪽 PAD가 결과에 포함되지 않음
- 동일 행동에서는 최근 이벤트의 가중치가 더 큼
- 같은 시점에서는 찜이 선택보다 강하게 반영됨
- 상호작용이 하나이면 원래 벡터가 유지됨
- 전부 PAD인 입력에서도 0 벡터를 반환하고 NaN이 발생하지 않음

## 4. 왜 해당 방식으로 개발했는지

서비스의 핵심 요구는 `A 다음에 B가 발생한다`와 같은 세밀한 전이 패턴보다 최근에 강한 행동을 보인 상품이 현재 취향을 더 잘 나타낸다는 점을 반영하는 것이다.

따라서 SASRec이나 Transformer 기반 순서 인코더를 추가하지 않고 RecRec의 정적 증거 구성 단계만 개선했다.

- RecRec의 recursive refinement와 evidence-anchored correction gate를 유지할 수 있다.
- 모델 구조와 파라미터 증가가 작다.
- 행동별 가중치를 서비스 의미에 맞게 설명할 수 있다.
- 합성 데이터 중심인 현재 상황에서 복잡한 attention 모델의 과적합 위험을 줄인다.
- 기존 masked mean 모델과 직접 비교할 수 있다.
- 실제 timestamp가 수집되면 이벤트 거리 기반 최근성을 시간 기반으로 확장할 수 있다.

## 5. 개발하면서 어떤 문제를 해결했는지

### 왼쪽 패딩과 최근 거리 계산

입력은 최근 64개 행동을 왼쪽 패딩한다. 단순 배열 인덱스로 최근성을 계산하면 PAD가 실제 위치로 잘못 취급될 수 있다. `attention_mask`의 누적합으로 유효 위치를 계산하고 각 샘플의 실제 길이를 기준으로 최근 행동과의 거리를 구해 해결했다.

### 기존 체크포인트 비호환

행동 임베딩이 5개에서 6개로 증가했고 pooling 설정이 추가되어 기존 체크포인트를 사용할 수 없었다. 새 행동 데이터로 다시 학습하고 현재 체크포인트 경로에 교체했다.

### 학습 스크립트 import 불일치

직접 실행 방식의 import와 패키지 상대 import가 혼용되어 있던 문제를 패키지 상대 import로 통일했다.

```powershell
python -m src.synthetic_generator
python -m src.train
python -m src.evaluate
```

### 학습과 추론 설정 불일치 위험

행동 가중치와 감쇠율을 체크포인트에 포함하고 추론 시 복원하도록 변경해 학습 이후 코드 기본값이 바뀌더라도 동일 설정을 사용하게 했다.

## 6. 개선 후 어떤 점이 달라졌는지

기존에는 모든 행동이 동일한 비중으로 평균됐다.

```text
A 선택 → B 피팅 → C 찜
x = (hA + hB + hC) / 3
```

개선 후에는 행동 강도와 최근성을 함께 반영한다.

```text
A PRODUCT_SELECT → 낮은 행동 강도 × 낮은 최근성
B FITTING_ADD    → 중간 행동 강도 × 중간 최근성
C WISHLIST_ADD   → 높은 행동 강도 × 높은 최근성
```

- 최근 상호작용이 현재 preference에 더 크게 반영된다.
- 선택, 피팅, 찜의 관심 강도가 구분된다.
- 피팅 추가와 제거를 별도 행동으로 학습한다.
- RecRec의 재귀 보정 구조는 그대로 유지된다.
- 행동 매핑과 pooling 설정이 체크포인트에 포함되어 학습과 추론의 일관성이 높아졌다.
- 구형 파일과 버전 이름을 제거해 현재 실행 기준이 명확해졌다.

새 합성 데이터는 9,000개 세션과 231,730개 상호작용으로 구성되며 `FITTING_ADD` 52,672건과 `FITTING_REMOVE` 15,329건을 포함한다. 새 체크포인트는 6개 행동 유형과 action-recency weighted pooling 설정으로 정상 로드된다.

## 7. 상품 Metadata ProductEncoder

상품명, 성별, 카테고리, 서브카테고리, Zone, 색상을 고정 폭 metadata feature로 변환하는 콘텐츠 경로를 추가했다. 외부 모델 다운로드 없이 학습과 추론에서 항상 동일한 결과를 만들기 위해 결정론적 feature hashing을 사용한다.

콘텐츠 feature는 projection layer로 RecRec embedding 차원에 맞춘 뒤 ID embedding과 결합한다.

$$
e_i^{product} = \operatorname{LayerNorm}(e_i^{id} + W_c f_i^{metadata})
$$

이력 상품과 전체 추천 후보가 모두 같은 `encode_products()` 함수를 사용하므로 입력 공간과 후보 점수 공간이 어긋나지 않는다. 학습 시 `--content` 옵션으로 활성화하며 metadata feature와 projection parameter는 체크포인트에 함께 저장된다.

```powershell
python -m src.train --content
```

## 8. 평가 결과와 현재 채택 모델

10 epoch 동일 예산의 pooling ablation 결과는 다음과 같다.

| Pooling | HR@1 | HR@5 | HR@10 | NDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Mean | 0.0261 | 0.1495 | 0.2947 | 0.1321 | 0.0839 |
| Action | 0.0269 | 0.1489 | 0.2961 | 0.1334 | 0.0851 |
| Recency | 0.0255 | 0.1432 | 0.2823 | 0.1275 | 0.0816 |
| Action × Recency | 0.0247 | 0.1467 | 0.2867 | 0.1292 | 0.0824 |

이 결과에서는 action weighting만 mean보다 소폭 향상됐고 recency weighting은 개선을 보이지 않았다. 따라서 최근성 가정은 실제 사용자 로그로 다시 검증해야 하며, 합성 데이터 결과만으로 효과가 입증됐다고 주장하지 않는다.

Metadata + Action × Recency 모델은 `HR@10 0.2912`, `NDCG@10 0.1289`, `MRR@10 0.0807`을 기록했다. 콘텐츠 경로는 정상 작동하지만 현재 합성 데이터에서는 ID-only 모델보다 우수하지 않았다.

운영 체크포인트는 더 나은 성능을 보인 ID-only Action × Recency 모델을 유지한다. ProductEncoder 구현과 콘텐츠 실험 체크포인트는 실제 로그 기반 재평가를 위해 보존한다.

## 9. 현재 한계와 다음 단계

- 현재 데이터는 실제 고객 로그가 아니라 합성 데이터이므로 최근성 및 콘텐츠 효과를 확정할 수 없다.
- feature hashing은 상품 속성을 반영하지만 SBERT처럼 단어의 의미적 유사성을 이해하지는 못한다.
- 실제 timestamp가 없어 이벤트 순번을 기준으로 최근성을 계산한다.
- 신규 상품 콜드 스타트를 제대로 검증하려면 상품 단위 holdout 평가가 필요하다.
- 다음 단계는 실제 로그 수집, timestamp 기반 decay, 신규 상품 holdout 평가, SBERT 또는 이미지 임베딩 비교다.
