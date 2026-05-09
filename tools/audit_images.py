from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


def audit_images(root: Path) -> dict[str, Any]:
    images = []
    unreadable = []
    hashes: dict[str, list[str]] = defaultdict(list)
    sizes: Counter[str] = Counter()

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = _read_image(path)
        if image is None or image.size == 0:
            unreadable.append(str(path))
            continue
        height, width = image.shape[:2]
        digest = _sha256(path)
        hashes[digest].append(str(path))
        sizes[f"{width}x{height}"] += 1
        images.append({"path": str(path), "width": width, "height": height, "sha256": digest})

    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    return {
        "root": str(root),
        "image_count": len(images),
        "unreadable_count": len(unreadable),
        "duplicate_groups": duplicates,
        "size_distribution": dict(sorted(sizes.items())),
        "unreadable": unreadable,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit dataset images for readability, duplicates, and size distribution.")
    parser.add_argument("--root", default=Path("data/raw"), type=Path, help="Directory to scan.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    args = parser.parse_args()
    report = audit_images(args.root)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"image_count={report['image_count']}")
        print(f"unreadable_count={report['unreadable_count']}")
        print(f"duplicate_group_count={len(report['duplicate_groups'])}")
        print("size_distribution=" + json.dumps(report["size_distribution"], ensure_ascii=False))
    return 1 if report["unreadable_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
