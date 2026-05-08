from __future__ import annotations

import argparse
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


def make_splits(data_root: Path, *, ratios: tuple[float, float, float], seed: int) -> dict[str, list[str]]:
    samples = [
        path.relative_to(data_root).as_posix()
        for path in sorted((data_root / "raw").rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    rng = random.Random(seed)
    rng.shuffle(samples)
    train_ratio, val_ratio, test_ratio = ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("split ratios must sum to a positive value")
    train_count = int(len(samples) * train_ratio / total_ratio)
    val_count = int(len(samples) * val_ratio / total_ratio)
    return {
        "train": samples[:train_count],
        "val": samples[train_count : train_count + val_count],
        "test": samples[train_count + val_count :],
    }


def write_splits(data_root: Path, splits: dict[str, list[str]]) -> None:
    splits_root = data_root / "splits"
    splits_root.mkdir(parents=True, exist_ok=True)
    for split_name, samples in splits.items():
        path = splits_root / f"{split_name}.txt"
        path.write_text("\n".join(samples) + ("\n" if samples else ""), encoding="utf-8")


def _parse_ratios(value: str) -> tuple[float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("ratios must use train,val,test format, e.g. 0.7,0.2,0.1")
    return parts[0], parts[1], parts[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create train/val/test split files from data/raw images.")
    parser.add_argument("--data-root", default=Path("data"), type=Path)
    parser.add_argument("--ratios", default=(0.7, 0.2, 0.1), type=_parse_ratios, help="train,val,test ratios.")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    splits = make_splits(args.data_root, ratios=args.ratios, seed=args.seed)
    write_splits(args.data_root, splits)
    print(f"train={len(splits['train'])}")
    print(f"val={len(splits['val'])}")
    print(f"test={len(splits['test'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
