# Python 코드

제공 코드는 `00.get_images.py`입니다. 이후 코드는 실습에서 에이전트와 작성합니다.

| 권장 파일 | 역할 | 산출물 |
| --- | --- | --- |
| `01.preprocess.py` | 이미지 확인, 공통 분할, 전처리 | `dataset/splits.csv`, `output/samples.png` |
| `02.train.py` | CNN 모델 생성·학습 | `models/cnn.keras`, 학습 이력·곡선 |
| `03.evaluate.py` | 고정 시험 세트 평가 | 지표 JSON, 예측 CSV, 혼동행렬 |
| `04.export.py` | 모델·입력 규격 저장 및 재로드 확인 | `models/export/` |
| `05.predict.py` | 선택: 내보낸 모델로 새 이미지 판정 | `output/predict/` 판정 결과 CSV |
| `06.train_transfer.py` | 선택: 전이학습 비교 | 별도 모델과 비교 표 |

TensorFlow, Pillow, matplotlib, scikit-learn 등 이후 필요한 의존성은 작성한 코드에 맞춰 준비한 환경에 설치합니다.
