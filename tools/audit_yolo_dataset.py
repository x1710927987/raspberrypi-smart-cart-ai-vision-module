from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from yolo_dataset_utils import dataset_split_dirs, iter_images, parse_class_map, parse_yolo_rows, read_image, read_yolo_config


@dataclass
class SplitAudit:
    split: str
    images: int = 0
    labels: int = 0
    label_rows: int = 0
    mapped_rows: int = 0
    missing_split: bool = False
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    unreadable_images: list[str] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)
    class_counts: Counter[str] = field(default_factory=Counter)
    row_type_counts: Counter[str] = field(default_factory=Counter)
    mapped_class_counts: Counter[str] = field(default_factory=Counter)
    unmapped_class_counts: Counter[str] = field(default_factory=Counter)


def audit_dataset(root: Path, *, class_map: dict[str, str] | None = None) -> tuple[dict[str, Any], list[SplitAudit]]:
    dataset_root = root.resolve()
    config = read_yolo_config(dataset_root / "data.yaml")
    split_audits = [audit_split(dataset_root, split, config.names, class_map=class_map or {}) for split in ("train", "valid", "test")]
    return {"names": config.names, "nc": config.nc}, split_audits


def audit_split(dataset_root: Path, split: str, names: list[str], *, class_map: dict[str, str]) -> SplitAudit:
    audit = SplitAudit(split)
    images_dir, labels_dir = dataset_split_dirs(dataset_root, split)
    if not images_dir.exists() and not labels_dir.exists():
        audit.missing_split = True
        return audit

    image_paths = {path.stem: path for path in iter_images(images_dir)} if images_dir.exists() else {}
    label_paths = {path.stem: path for path in labels_dir.rglob("*.txt") if path.is_file()} if labels_dir.exists() else {}
    audit.images = len(image_paths)
    audit.labels = len(label_paths)

    for stem, image_path in sorted(image_paths.items()):
        image = read_image(image_path)
        if image is None or image.size == 0:
            audit.unreadable_images.append(str(image_path))
        if stem not in label_paths:
            audit.missing_labels.append(str(image_path))

    for stem, label_path in sorted(label_paths.items()):
        if stem not in image_paths:
            audit.orphan_labels.append(str(label_path))
            continue
        rows, invalids = parse_yolo_rows(label_path, names)
        audit.invalid_labels.extend(invalids)
        audit.label_rows += len(rows)
        for row in rows:
            audit.class_counts[row.source_cls] += 1
            audit.row_type_counts[row.row_type] += 1
            mapped_cls = class_map.get(row.source_cls, class_map.get(str(row.class_id)))
            if mapped_cls:
                audit.mapped_rows += 1
                audit.mapped_class_counts[mapped_cls] += 1
            elif class_map:
                audit.unmapped_class_counts[row.source_cls] += 1
    return audit


def print_report(config: dict[str, Any], audits: list[SplitAudit], *, as_json: bool = False) -> None:
    if as_json:
        payload = {
            "classes": config["names"],
            "splits": [_audit_to_dict(audit) for audit in audits],
            "status": "failed" if _has_errors(audits) else "ok",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"classes={config['names']}")
    for audit in audits:
        print(f"[{audit.split}] images={audit.images} labels={audit.labels} label_rows={audit.label_rows}")
        print(f"[{audit.split}] row_type_counts={dict(sorted(audit.row_type_counts.items()))}")
        print(f"[{audit.split}] class_counts={dict(sorted(audit.class_counts.items()))}")
        if audit.mapped_rows:
            print(f"[{audit.split}] mapped_rows={audit.mapped_rows}")
            print(f"[{audit.split}] mapped_class_counts={dict(sorted(audit.mapped_class_counts.items()))}")
        if audit.unmapped_class_counts:
            print(f"[{audit.split}] unmapped_class_counts={dict(sorted(audit.unmapped_class_counts.items()))}")
        if audit.missing_split:
            print(f"[{audit.split}] missing_split=1")
        if audit.missing_labels:
            print(f"[{audit.split}] missing_labels={len(audit.missing_labels)}")
        if audit.orphan_labels:
            print(f"[{audit.split}] orphan_labels={len(audit.orphan_labels)}")
        if audit.unreadable_images:
            print(f"[{audit.split}] unreadable_images={len(audit.unreadable_images)}")
        if audit.invalid_labels:
            print(f"[{audit.split}] invalid_labels={len(audit.invalid_labels)}")
            for item in audit.invalid_labels[:10]:
                print(f"  {item}")
    print("status=failed" if _has_errors(audits) else "status=ok")


def _audit_to_dict(audit: SplitAudit) -> dict[str, Any]:
    payload = asdict(audit)
    for key in ("class_counts", "row_type_counts", "mapped_class_counts", "unmapped_class_counts"):
        payload[key] = dict(sorted(payload[key].items()))
    return payload


def _has_errors(audits: list[SplitAudit]) -> bool:
    return any(audit.unreadable_images or audit.invalid_labels for audit in audits)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a generic Roboflow/YOLO dataset.")
    parser.add_argument("--root", required=True, type=Path, help="Dataset root containing data.yaml and split folders.")
    parser.add_argument(
        "--class-map",
        action="append",
        help="Optional mapping in source=target format. Repeat for multiple mappings.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    args = parser.parse_args()
    config, audits = audit_dataset(args.root, class_map=parse_class_map(args.class_map))
    print_report(config, audits, as_json=args.json)
    return 1 if _has_errors(audits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
