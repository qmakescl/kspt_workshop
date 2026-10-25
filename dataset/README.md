# 데이터

`python src/00.get_images.py` 실행 후 `raw/`에 Kaggle 원본, `teachable_machine/train/`과 `teachable_machine/test/`에 클래스별 예제가 생성됩니다. 경로는 실행 위치와 관계없이 프로젝트 기준입니다.

- `teachable_machine/manifest.csv`: 분할, 라벨, 원본 상대경로, 파일명, SHA-256, 시드
- `teachable_machine/preparation.json`: 예제 생성 설정
- `download.json`: 요청한 Kaggle handle, 다운로드 위치, kagglehub 버전

이미지와 생성 기록은 Git에서 제외합니다. 원본은 보존하고 전처리 결과는 별도 폴더에 저장합니다. TM 시험 이미지를 로컬 모델과 비교하려면 해당 이미지와 동일 내용의 파일을 로컬 학습·검증에서 제외하세요. 내용 중복 제거는 환자 단위 분리를 보장하지 않습니다.
