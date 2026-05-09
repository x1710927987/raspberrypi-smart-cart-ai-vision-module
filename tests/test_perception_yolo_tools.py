import importlib.util
import json
import sys
from uuid import uuid4
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_yolo_config_parser_keeps_commas_inside_class_names():
    utils = _load_tool("yolo_dataset_utils")
    workspace = REPO_ROOT / "cache" / "pytest" / "test_perception_yolo_tools" / "config_parser"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = workspace / "data.yaml"
    config_path.write_text(
        "\n".join(
            [
                "train: ../train/images",
                "val: ../valid/images",
                "test: ../test/images",
                "nc: 3",
                "names: ['* annotate, and create datasets', 'bike', 'car']",
            ]
        ),
        encoding="utf-8",
    )

    config = utils.read_yolo_config(config_path)

    assert config.names == ["* annotate, and create datasets", "bike", "car"]
    assert config.nc == 3


def test_split_yolo_dataset_preserves_segmentation_rows():
    script = _load_tool("split_yolo_dataset")
    source = _write_yolo_dataset(
        "split_preserves_polygon",
        names=["sidewalk"],
        labels={
            "sample_0": ["0 0.100000 0.700000 0.900000 0.700000 0.900000 0.950000 0.100000 0.950000"],
            "sample_1": ["0 0.200000 0.600000 0.800000 0.600000 0.800000 0.900000 0.200000 0.900000"],
            "sample_2": ["0 0.300000 0.500000 0.700000 0.500000 0.700000 0.850000 0.300000 0.850000"],
        },
    )
    output = source.parent / f"split_preserves_polygon_out_{uuid4().hex}"

    counts = script.split_dataset(source, output, ratios=(1, 1, 1), seed=7)

    assert counts == {"train": 1, "valid": 1, "test": 1}
    label_text = "\n".join(path.read_text(encoding="utf-8") for path in output.rglob("*.txt"))
    assert any(len(line.split()) > 5 for line in label_text.splitlines())


def test_import_objects_maps_supported_classes_and_skips_unmapped():
    script = _load_tool("import_perception_yolo_dataset")
    source = _write_yolo_dataset(
        "import_objects",
        names=["person", "bike", "marketing"],
        labels={"sample_0": ["0 0.50 0.50 0.20 0.40", "1 0.25 0.50 0.10 0.20", "2 0.80 0.50 0.10 0.20"]},
    )
    data_root = source.parent / "data_root"

    summary = script.import_dataset(
        dataset_root=source,
        data_root=data_root,
        task="objects",
        class_map={"person": "pedestrian", "bike": "bicycle"},
        prefix="objects_demo",
        splits=["train"],
    )

    annotation_path = next((data_root / "annotations" / "objects").glob("*.json"))
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    assert summary.imported_images == 1
    assert summary.imported_instances == 2
    assert summary.skipped_unmapped_rows == 1
    assert [obj["cls"] for obj in payload["objects"]] == ["pedestrian", "bicycle"]
    assert (data_root / payload["image"]).exists()


def test_import_laneseg_writes_binary_mask_and_annotation():
    script = _load_tool("import_perception_yolo_dataset")
    source = _write_yolo_dataset(
        "import_laneseg",
        names=["sidewalk"],
        labels={"sample_0": ["0 0.100000 0.600000 0.900000 0.600000 0.900000 0.950000 0.100000 0.950000"]},
    )
    data_root = source.parent / "data_root_laneseg"

    summary = script.import_dataset(
        dataset_root=source,
        data_root=data_root,
        task="laneseg",
        class_map={"sidewalk": "sidewalk"},
        prefix="sidewalk_demo",
        splits=["train"],
    )

    annotation_path = next((data_root / "annotations" / "laneseg").glob("*.json"))
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    mask = cv2.imread(str(data_root / payload["mask"]), cv2.IMREAD_GRAYSCALE)
    assert summary.imported_images == 1
    assert summary.imported_instances == 1
    assert mask is not None
    assert int(np.count_nonzero(mask)) > 0
    assert payload["laneseg"]["classes"] == ["sidewalk"]


def _write_yolo_dataset(name: str, *, names: list[str], labels: dict[str, list[str]]) -> Path:
    root = REPO_ROOT / "cache" / "pytest" / "test_perception_yolo_tools" / name
    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "\n".join(
            [
                "train: ../train/images",
                "val: ../valid/images",
                "test: ../test/images",
                f"nc: {len(names)}",
                f"names: {names!r}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for stem, rows in labels.items():
        _write_image(images_dir / f"{stem}.jpg")
        (labels_dir / f"{stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    frame[:, :] = (20, 40, 60)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    encoded.tofile(str(path))
