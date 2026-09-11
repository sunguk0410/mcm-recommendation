# 상품 이미지 유사도 검토

`images/{productCode}.png`를 카탈로그의 상품 코드에 연결해 CLIP 이미지 벡터를 만들고, 같은 카테고리 내 유사 상품을 확인한다. 이 단계는 합성 데이터 생성 규칙을 설계하기 위한 탐색이며 기존 RecRec 학습·추론에는 연결하지 않는다.

## 실행

기존 torch 환경을 재사용하는 별도 가상환경 예시:

```powershell
python -m venv --system-site-packages .venv-image
.\.venv-image\Scripts\python.exe -m pip install -r requirements-image.txt
.\.venv-image\Scripts\python.exe -m src.image_similarity
```

첫 실행 시 `openai/clip-vit-base-patch32` 모델 파일을 다운로드한다. 상품 이미지는 로컬에서 처리한다. 모델 캐시는 출력 폴더에 저장된다. `--revision`에 이전 결과의 commit hash를 지정하면 같은 모델 revision을 사용할 수 있다.

## 처리 및 산출물

- 투명 배경을 흰색으로 합성하고 정사각형 여백을 추가해 중심 crop으로 상품이 잘리는 것을 방지한다.
- 고정된 CLIP vision encoder 및 projection으로 512차원 벡터를 계산하고 L2 정규화한다.
- 자기 자신을 제외하고 같은 카테고리 내 코사인 유사도 상위 5개를 선택한다. 성별은 이 보고서에서 제한하지 않는다.
- `artifacts/image_similarity/embeddings.pt`: 상품 ID·코드 순서, 이미지 hash, 모델 revision, 벡터.
- `artifacts/image_similarity/neighbors.json`: 상품별 이웃과 코사인 유사도.
- `artifacts/image_similarity/report.html`: 원본 이미지와 이웃 이미지를 나란히 표시. 이미지 상대 경로를 사용하므로 프로젝트 내에서 연다.

## 해석

유사도는 고객의 선택·피팅·찜 확률이 아니다. 색상만 비슷한지, 형태와 패턴도 맞는지, 촬영 구도 때문에 묶이지는 않는지 검토한다. 같은 유사도 규칙으로 생성한 데이터에서의 추천 성능만으로 실제 고객 추천 품질을 주장하지 않는다.

검토 후 세션 카테고리, 지속적인 고객 취향, 직전 상품 유사도, 탐색 확률을 조합해 합성 데이터 생성에 반영할 수 있다. 이 경우 기존 ID 기반 RecRec은 생성된 행동 관계를 간접적으로 학습한다.

모델 API 참고: https://huggingface.co/docs/transformers/v4.52.2/en/model_doc/clip
