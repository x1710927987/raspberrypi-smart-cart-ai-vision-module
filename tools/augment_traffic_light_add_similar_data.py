from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from audit_traffic_light_yolo_dataset import read_image, read_roboflow_yaml, yolo_row_to_bbox


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = REPO_ROOT / "cache" / "evaluation" / "traffic_light_add_similar_data_plan.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "external" / "traffic_light_add_similar_v1"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "data" / "external" / "roboflow_traffic_light_v1_split"
IMAGE_SUFFIX = ".jpg"
CLASS_NAMES = ["green", "red", "yellow"]


@dataclass(frozen=True)
class LabelBox:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float


@dataclass(frozen=True)
class PlanCase:
    case_id: str
    primary_gt: str
    predicted_state: str
    reason: str
    suggested_new_images: int
    recommended_focus: str
    source_image: Path
    notes: str


@dataclass(frozen=True)
class GeneratedSample:
    case_id: str
    output_image: str
    output_label: str
    source_image: str
    transform: str
    primary_gt: str
    predicted_state: str
    reason: str
    label_rows: int


def augment_add_similar_data(
    *,
    plan_csv: str | Path = DEFAULT_PLAN,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    seed: int = 42,
    max_per_case: int | None = None,
) -> list[GeneratedSample]:
    plan_path = _resolve_path(Path(plan_csv))
    out_root = _resolve_path(Path(output_root))
    src_root = _resolve_path(Path(source_root))
    config = read_roboflow_yaml(src_root / "data.yaml")
    if list(config["names"]) != CLASS_NAMES:
        raise ValueError(f"expected traffic-light classes {CLASS_NAMES}, got {config['names']}")

    cases = read_plan_cases(plan_path)
    rng = random.Random(seed)
    samples: list[GeneratedSample] = []
    for case in cases:
        count = case.suggested_new_images if max_per_case is None else min(case.suggested_new_images, max_per_case)
        source_image = _resolve_path(case.source_image)
        label_path = _label_path_for_image(source_image)
        image = read_image(source_image)
        if image is None or image.size == 0:
            raise ValueError(f"unreadable source image for {case.case_id}: {source_image}")
        labels = read_yolo_labels(label_path)
        if not labels:
            raise ValueError(f"no usable labels for {case.case_id}: {label_path}")
        for index in range(count):
            transform_name = _choose_transform(case, rng)
            augmented, augmented_labels = apply_transform(image, labels, transform_name, rng)
            if not augmented_labels:
                augmented, augmented_labels = apply_transform(image, labels, "color_jitter", rng)
            if not augmented_labels:
                continue
            stem = f"{_safe_name(case.case_id)}_{index:04d}_{transform_name}"
            image_path = out_root / "train" / "images" / f"{stem}{IMAGE_SUFFIX}"
            label_out_path = out_root / "train" / "labels" / f"{stem}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_out_path.parent.mkdir(parents=True, exist_ok=True)
            _write_image(image_path, augmented)
            label_out_path.write_text(format_yolo_labels(augmented_labels), encoding="utf-8")
            samples.append(
                GeneratedSample(
                    case_id=case.case_id,
                    output_image=_repo_relative(image_path),
                    output_label=_repo_relative(label_out_path),
                    source_image=_repo_relative(source_image),
                    transform=transform_name,
                    primary_gt=case.primary_gt,
                    predicted_state=case.predicted_state,
                    reason=case.reason,
                    label_rows=len(augmented_labels),
                )
            )
    _write_data_yaml(out_root / "data.yaml")
    _write_manifest(out_root / "augmentation_manifest.csv", samples)
    _write_summary(out_root / "augmentation_summary.json", plan_path, samples)
    return samples


def read_plan_cases(path: Path) -> list[PlanCase]:
    if not path.exists():
        raise FileNotFoundError(f"plan CSV does not exist: {path}")
    cases: list[PlanCase] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                suggested = int(float(row.get("suggested_new_images", "0")))
            except ValueError:
                suggested = 0
            if suggested <= 0:
                continue
            cases.append(
                PlanCase(
                    case_id=row.get("case_id", "").strip(),
                    primary_gt=row.get("primary_gt", "").strip(),
                    predicted_state=row.get("predicted_state", "").strip(),
                    reason=row.get("reason", "").strip(),
                    suggested_new_images=suggested,
                    recommended_focus=row.get("recommended_focus", "").strip(),
                    source_image=Path(row.get("source_image", "").strip()),
                    notes=row.get("notes", "").strip(),
                )
            )
    if not cases:
        raise ValueError(f"no add_similar_data cases found in {path}")
    return cases


def read_yolo_labels(path: Path) -> list[LabelBox]:
    if not path.exists():
        raise FileNotFoundError(f"missing label file: {path}")
    labels: list[LabelBox] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            continue
        bbox = yolo_row_to_bbox(coords)
        if bbox is None:
            continue
        labels.append(LabelBox(class_id, *bbox))
    return labels


def apply_transform(
    image: np.ndarray,
    labels: list[LabelBox],
    transform_name: str,
    rng: random.Random,
) -> tuple[np.ndarray, list[LabelBox]]:
    if transform_name == "small_canvas":
        return _small_canvas(image, labels, rng)
    if transform_name == "affine_perspective":
        return _affine_perspective(image, labels, rng)
    if transform_name == "blur_noise":
        return _blur_noise(image, labels, rng)
    if transform_name == "shadow_contrast":
        return _shadow_contrast(image, labels, rng)
    return _color_jitter(image, labels, rng)


def format_yolo_labels(labels: Iterable[LabelBox]) -> str:
    rows = [
        f"{label.class_id} {label.cx:.8f} {label.cy:.8f} {label.width:.8f} {label.height:.8f}"
        for label in labels
    ]
    return "\n".join(rows) + ("\n" if rows else "")


def _choose_transform(case: PlanCase, rng: random.Random) -> str:
    focus = f"{case.reason} {case.recommended_focus} {case.notes}".lower()
    if "too_small" in focus or "small or distant" in focus:
        choices = ["small_canvas", "small_canvas", "affine_perspective", "blur_noise", "shadow_contrast"]
    elif "perspective" in focus or "angled" in focus or "side-view" in focus:
        choices = ["affine_perspective", "affine_perspective", "small_canvas", "blur_noise", "color_jitter"]
    elif "wrong_color" in focus or "confused" in focus:
        choices = ["shadow_contrast", "color_jitter", "blur_noise", "affine_perspective"]
    elif "snow" in focus or "shell" in focus or "non-black" in focus:
        choices = ["shadow_contrast", "color_jitter", "blur_noise", "small_canvas"]
    else:
        choices = ["color_jitter", "blur_noise", "affine_perspective", "small_canvas", "shadow_contrast"]
    return rng.choice(choices)


def _color_jitter(
    image: np.ndarray,
    labels: list[LabelBox],
    rng: random.Random,
) -> tuple[np.ndarray, list[LabelBox]]:
    alpha = rng.uniform(0.72, 1.28)
    beta = rng.uniform(-34, 34)
    output = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if rng.random() < 0.45:
        hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= rng.uniform(0.78, 1.22)
        hsv[..., 2] *= rng.uniform(0.84, 1.18)
        hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
        output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return output, list(labels)


def _blur_noise(
    image: np.ndarray,
    labels: list[LabelBox],
    rng: random.Random,
) -> tuple[np.ndarray, list[LabelBox]]:
    output, out_labels = _color_jitter(image, labels, rng)
    if rng.random() < 0.65:
        kernel = rng.choice([3, 5])
        output = cv2.GaussianBlur(output, (kernel, kernel), 0)
    noise_sigma = rng.uniform(2.0, 9.0)
    noise = np.random.default_rng(rng.randrange(1_000_000_000)).normal(0, noise_sigma, output.shape)
    output = np.clip(output.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return output, out_labels


def _shadow_contrast(
    image: np.ndarray,
    labels: list[LabelBox],
    rng: random.Random,
) -> tuple[np.ndarray, list[LabelBox]]:
    output, out_labels = _color_jitter(image, labels, rng)
    height, width = output.shape[:2]
    x1 = rng.randint(0, max(1, width - 1))
    x2 = rng.randint(0, max(1, width - 1))
    mask = np.zeros((height, width), dtype=np.float32)
    polygon = np.array([[x1, 0], [width, 0], [x2, height], [0, height]], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], rng.uniform(0.18, 0.38))
    output = np.clip(output.astype(np.float32) * (1.0 - mask[..., None]), 0, 255).astype(np.uint8)
    return output, out_labels


def _small_canvas(
    image: np.ndarray,
    labels: list[LabelBox],
    rng: random.Random,
) -> tuple[np.ndarray, list[LabelBox]]:
    height, width = image.shape[:2]
    scale = rng.uniform(0.52, 0.82)
    new_width = max(8, int(round(width * scale)))
    new_height = max(8, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    background = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    background = cv2.GaussianBlur(background, (21, 21), 0)
    background = cv2.convertScaleAbs(background, alpha=rng.uniform(0.65, 0.92), beta=rng.uniform(-18, 18))
    max_x = max(0, width - new_width)
    max_y = max(0, height - new_height)
    offset_x = rng.randint(0, max_x) if max_x else 0
    offset_y = rng.randint(0, max_y) if max_y else 0
    background[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = resized

    transformed: list[LabelBox] = []
    for label in labels:
        x1, y1, x2, y2 = _label_to_pixels(label, width, height)
        transformed.extend(_pixels_to_label(label.class_id, offset_x + x1 * scale, offset_y + y1 * scale, offset_x + x2 * scale, offset_y + y2 * scale, width, height))
    return background, transformed


def _affine_perspective(
    image: np.ndarray,
    labels: list[LabelBox],
    rng: random.Random,
) -> tuple[np.ndarray, list[LabelBox]]:
    height, width = image.shape[:2]
    angle = math.radians(rng.uniform(-7.0, 7.0))
    scale = rng.uniform(0.88, 1.06)
    shear = rng.uniform(-0.10, 0.10)
    cos_a = math.cos(angle) * scale
    sin_a = math.sin(angle) * scale
    center_x = width / 2.0
    center_y = height / 2.0
    affine = np.array(
        [
            [cos_a + shear * sin_a, -sin_a + shear * cos_a, 0.0],
            [sin_a, cos_a, 0.0],
        ],
        dtype=np.float32,
    )
    affine[:, 2] = [
        center_x - affine[0, 0] * center_x - affine[0, 1] * center_y + rng.uniform(-0.04, 0.04) * width,
        center_y - affine[1, 0] * center_x - affine[1, 1] * center_y + rng.uniform(-0.04, 0.04) * height,
    ]
    border = tuple(int(value) for value in image.reshape(-1, 3).mean(axis=0))
    output = cv2.warpAffine(image, affine, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)

    transformed: list[LabelBox] = []
    for label in labels:
        x1, y1, x2, y2 = _label_to_pixels(label, width, height)
        corners = np.array([[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]], dtype=np.float32)
        warped = corners @ affine.T
        tx1, ty1 = warped[:, 0].min(), warped[:, 1].min()
        tx2, ty2 = warped[:, 0].max(), warped[:, 1].max()
        transformed.extend(_pixels_to_label(label.class_id, tx1, ty1, tx2, ty2, width, height))
    return output, transformed


def _label_to_pixels(label: LabelBox, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = (label.cx - label.width / 2.0) * width
    y1 = (label.cy - label.height / 2.0) * height
    x2 = (label.cx + label.width / 2.0) * width
    y2 = (label.cy + label.height / 2.0) * height
    return x1, y1, x2, y2


def _pixels_to_label(
    class_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> list[LabelBox]:
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return []
    return [LabelBox(class_id, (x1 + x2) / (2.0 * width), (y1 + y2) / (2.0 * height), (x2 - x1) / width, (y2 - y1) / height)]


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"source image path must include an images directory: {image_path}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"failed to encode image: {path}")
    encoded.tofile(str(path))


def _write_data_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        (path.parent / split / "images").mkdir(parents=True, exist_ok=True)
        (path.parent / split / "labels").mkdir(parents=True, exist_ok=True)
    names_text = "[" + ", ".join(f"'{name}'" for name in CLASS_NAMES) + "]"
    path.write_text(
        "\n".join(
            [
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "",
                f"nc: {len(CLASS_NAMES)}",
                f"names: {names_text}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_manifest(path: Path, samples: list[GeneratedSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(GeneratedSample.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.__dict__)


def _write_summary(path: Path, plan_path: Path, samples: list[GeneratedSample]) -> None:
    by_class: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    by_case: dict[str, int] = {}
    by_transform: dict[str, int] = {}
    for sample in samples:
        by_class[sample.primary_gt] = by_class.get(sample.primary_gt, 0) + 1
        by_reason[sample.reason] = by_reason.get(sample.reason, 0) + 1
        by_case[sample.case_id] = by_case.get(sample.case_id, 0) + 1
        by_transform[sample.transform] = by_transform.get(sample.transform, 0) + 1
    payload = {
        "plan_csv": _repo_relative(plan_path),
        "generated_images": len(samples),
        "generated_labels": len(samples),
        "class_counts": dict(sorted(by_class.items())),
        "reason_counts": dict(sorted(by_reason.items())),
        "case_counts": dict(sorted(by_case.items())),
        "transform_counts": dict(sorted(by_transform.items())),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    normalized = normalized.strip("._-")
    return normalized or "sample"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate YOLO traffic-light add-similar-data samples from reviewed error cases.")
    parser.add_argument("--plan", default=DEFAULT_PLAN, type=Path)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=Path)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-per-case", type=int, help="Optional cap for smoke tests or quick previews.")
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        samples = augment_add_similar_data(
            plan_csv=args.plan,
            output_root=args.output_root,
            source_root=args.source_root,
            seed=args.seed,
            max_per_case=args.max_per_case,
        )
        print(f"output_root={_resolve_path(args.output_root)}")
        print(f"generated_images={len(samples)}")
        print(f"manifest={_resolve_path(args.output_root) / 'augmentation_manifest.csv'}")
        print(f"summary={_resolve_path(args.output_root) / 'augmentation_summary.json'}")
        print("status=ok")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
