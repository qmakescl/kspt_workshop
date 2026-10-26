# 대한기생충학·열대의학회 워크숍 실습 스켈레톤

말라리아 감염 적혈구 이미지로 Teachable Machine과 로컬 코딩 에이전트의 딥러닝 작업 과정을 경험합니다. 이 저장소는 데이터 준비 코드와 단계별 안내를 제공하며, 학습·평가 코드는 참가자가 에이전트와 작성합니다.

## 예제 데이터 출처

- [Kaggle — Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria): `iarunava/cell-images-for-detecting-malaria`
- 원 제공 기관: NIH/NLM Lister Hill National Center for Biomedical Communications. [공식 자료 안내](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html)
- 관련 논문: Rajaraman S, et al. (2018). *Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images*. PeerJ 6:e4568. [doi:10.7717/peerj.4568](https://doi.org/10.7717/peerj.4568)
- 클래스: `Parasitized`(감염), `Uninfected`(비감염). 실제 이미지 수는 코드가 출력합니다.

원본 이미지는 저장소에 포함하지 않습니다. 참가자가 Kaggle에서 다운로드하며, 데이터 이용·재배포 조건은 해당 데이터셋의 표시를 따릅니다. 버전을 생략하면 최신 버전을 받습니다. 강사가 확인한 버전이 있으면 `--handle iarunava/cell-images-for-detecting-malaria/versions/N`으로 지정하세요. 추출한 파일 목록과 SHA-256은 manifest에 기록합니다.

## 실습 순서

| 단계 | 활동 | 완료 확인 |
| --- | --- | --- |
| 1 | Notion “공통 작업환경 구축”: 상위 작업 폴더 생성 → Git 설치 → uv 설치 → Antigravity IDE 설치·실행 | `git`, `uv` 실행 가능 |
| 2 | 저장소 clone 후 IDE에서 폴더 열기 → `uv init`·`uv venv`로 Python 3.12 가상환경 생성 → 활성화 | `python --version`이 3.12 |
| 3 | 예제 데이터 다운로드 및 생성 | 학습·시험 이미지와 manifest 확인 |
| 4 | Teachable Machine 학습·판독 → 결과 해석 | 실제·예측 라벨과 지표 기록 |
| 5 | 로컬 에이전트로 전처리 → 모델 생성 → 학습 → 평가 → export | 코드·결과·모델 저장 |

가상환경은 저장소를 복제한 뒤 프로젝트 루트에서 직접 만듭니다. 패키지는 `uv add`로 설치해 `pyproject.toml`과 `uv.lock`에 기록합니다. 이 파일들은 참가자가 각자 만들며 저장소에는 포함하지 않습니다.

## 시작하기

미리 만든 상위 작업 폴더에서 실행합니다.

```bash
git clone https://github.com/qmakescl/kspt_workshop.git
cd kspt_workshop
```

Antigravity에서 복제한 폴더를 열고, 터미널에서 프로젝트 루트(`kspt_workshop/`)에 uv 프로젝트와 Python 3.12 가상환경을 만듭니다.

```bash
uv init --python 3.12
uv venv
```

가상환경을 활성화합니다. 이후 명령은 모두 활성화한 터미널에서 실행하며, 새 터미널을 열면 다시 활성화합니다.

| 운영체제·터미널 | 명령 |
| --- | --- |
| macOS (zsh) | `source .venv/bin/activate` |
| Windows (PowerShell) | `.\.venv\Scripts\Activate.ps1` |
| Windows (명령 프롬프트) | `.venv\Scripts\activate.bat` |

Python 버전을 확인하고 kagglehub를 추가한 뒤 데이터를 준비합니다.

```bash
python --version
uv add kagglehub
python src/00.get_images.py
```

인터넷 연결이 필요합니다. 인증을 요구하면 [kagglehub 공식 인증 안내](https://github.com/Kaggle/kagglehub#authentication)를 따르세요. 인증 정보는 저장소에 넣지 않습니다. 다운로드 코드는 `output_dir` 인자를 지원하는 최신 kagglehub를 사용합니다.

- 원본: `dataset/raw/` (내부에 `cell_images`가 중첩될 수 있음)
- TM 학습용: `dataset/teachable_machine/train/`, 클래스당 200장
- TM 시험용: `dataset/teachable_machine/test/`, 클래스당 50장
- 추출 시드: 2025. 내용 해시로 중복 제거 후 학습·시험 이미지를 분리합니다.
- `manifest.csv`: 분할, 클래스, 파일명, 원본 상대경로, SHA-256, 시드
- 기존 예제 폴더는 덮어쓰지 않습니다. 다시 만들려면 `--subset-name teachable_machine_v2`처럼 새 이름을 사용합니다.

이미 원본을 받았다면 네트워크 없이 예제만 만들 수 있습니다. 이 경우 원본은 기존 위치에 유지됩니다.

```bash
python src/00.get_images.py --source /path/to/cell_images
```

옵션은 `python src/00.get_images.py --help`로 확인합니다. 작은 예제는 `--train-per-class 20 --test-per-class 5`로 생성합니다.

## 폴더 구조

```text
kspt_workshop/
├── README.md
├── .gitignore
├── dataset/                 # 다운로드 원본·생성한 예제 (Git 제외)
│   └── README.md
├── src/                     # Python 코드
│   ├── 00.get_images.py     # 다운로드 + TM 학습/시험 예제 생성
│   └── README.md            # 실습에서 작성할 코드 안내
├── output/                  # 도표·지표·작업 결과 (Git 제외)
│   └── README.md
├── models/                  # 학습·export 모델 (Git 제외)
│   └── README.md
└── docs/
    ├── workflow.md
    ├── agent-prompts.md
    └── results.md
```

결과 폴더 이름은 `output/`으로 통일합니다. [진행 안내](docs/workflow.md)와 [에이전트 프롬프트](docs/agent-prompts.md)를 순서대로 사용하세요.

감염(`Parasitized`)을 양성으로 정합니다. 이 과제는 세포 이미지 분류이며 시야 전체의 감염 세포 탐지나 환자 진단 성능을 의미하지 않습니다. 이미지 단위 분할이므로 같은 환자 유래 이미지가 학습·시험에 섞일 수 있습니다. 교육용 결과이며 임상 진단에 사용하지 않습니다.
