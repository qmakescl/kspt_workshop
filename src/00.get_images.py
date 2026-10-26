"""Kaggle 말라리아 세포 이미지 다운로드 및 Teachable Machine 예제 생성."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "iarunava/cell-images-for-detecting-malaria"
CLASSES = ("Parasitized", "Uninfected")
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def find_images(root: Path, label: str) -> list[Path]:
    candidates = sorted(p for p in root.rglob(label) if p.is_dir())
    if root.name == label:
        candidates.insert(0, root)
    groups = [sorted(p for p in folder.iterdir()
                     if p.is_file() and p.suffix.lower() in EXTENSIONS)
              for folder in candidates]
    if not groups or not max(groups, key=len):
        raise ValueError(f"{root} 아래에 {label} 이미지가 없습니다.")
    return max(groups, key=len)


def prepare(source: Path, destination: Path, train: int, test: int, seed: int) -> None:
    """내용 해시로 중복 이미지를 제거한 뒤 겹치지 않는 두 세트를 만든다."""
    if destination.exists():
        raise FileExistsError(f"이미 존재하는 예제 폴더입니다: {destination}\n"
                              "기존 결과를 보존합니다. --subset-name으로 새 이름을 지정하세요.")
    rng = random.Random(seed)
    selected = []
    seen: dict[str, str] = {}
    for label in CLASSES:
        unique = []
        files = find_images(source, label)
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                if seen[digest] != label:
                    raise ValueError(f"같은 이미지에 다른 라벨이 있습니다: {path}")
                continue
            seen[digest] = label
            unique.append((path, digest))
        if len(unique) < train + test:
            raise ValueError(f"{label}: 고유 이미지 {len(unique)}장; {train + test}장 필요")
        picked = rng.sample(unique, train + test)
        for index, (path, digest) in enumerate(picked):
            selected.append((label, "train" if index < train else "test", path, digest))
        print(f"{label}: 원본 {len(files)}장, 고유 {len(unique)}장 → 학습 {train} / 시험 {test}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 완성된 예제만 공개하고, 실패하면 임시 디렉터리를 정리한다.
    with tempfile.TemporaryDirectory(prefix=".tm-", dir=destination.parent) as tmp:
        staging = Path(tmp) / "teachable_machine"
        staging.mkdir()
        with (staging / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "class", "filename", "source_relative", "sha256", "seed"])
            for label, split, path, digest in selected:
                target = staging / split / label / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                writer.writerow([split, label, path.name, path.relative_to(source), digest, seed])
        (staging / "preparation.json").write_text(json.dumps({
            "dataset": DATASET, "seed": seed, "train_per_class": train,
            "test_per_class": test, "split_unit": "image", "positive_class": "Parasitized",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staging.rename(destination)
    print(f"예제 생성 완료: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="이미 받은 원본 폴더 (네트워크 없이 예제만 생성)")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--handle", default=DATASET, help="필요시 /versions/N을 붙여 버전 고정")
    parser.add_argument("--train-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--subset-name", default="teachable_machine")
    args = parser.parse_args()
    if args.train_per_class < 1 or args.test_per_class < 1:
        parser.error("학습·시험 이미지 수는 각각 1 이상이어야 합니다.")
    if Path(args.subset_name).name != args.subset_name or args.subset_name in {".", "..", ""}:
        parser.error("--subset-name은 경로가 아닌 단일 폴더 이름이어야 합니다.")
    data = args.dataset_dir.expanduser().resolve()
    destination = data / args.subset_name
    if destination.exists():
        parser.error(f"{destination}이 이미 있습니다. 다시 만들려면 --subset-name에 새 이름을 지정하세요.")
    if args.source:
        source = args.source.expanduser().resolve()
        if not source.is_dir():
            parser.error(f"원본 폴더가 없습니다: {source}")
    else:
        try:
            import kagglehub
        except ImportError:
            parser.error("kagglehub가 필요합니다. 가상환경을 활성화한 뒤 프로젝트 루트에서 uv add kagglehub를 실행하세요.")
        data.mkdir(parents=True, exist_ok=True)
        # Kaggle 캐시 경로가 아닌 프로젝트 내부에 원본을 직접 저장한다.
        source = Path(kagglehub.dataset_download(args.handle, output_dir=str(data / "raw"))).resolve()
        (data / "download.json").write_text(json.dumps({
            "requested_handle": args.handle, "source": str(source),
            "kagglehub_version": getattr(kagglehub, "__version__", "unknown"),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prepare(source, destination, args.train_per_class, args.test_per_class, args.seed)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError) as error:
        raise SystemExit(str(error)) from error
