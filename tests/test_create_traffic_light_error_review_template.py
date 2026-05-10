import csv
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "create_traffic_light_error_review_template.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("create_traffic_light_error_review_template", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_create_review_template_from_gallery_index():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_create_traffic_light_error_review_template"
    index_path = workspace / "traffic_light_valid_v1_mistakes" / "index.json"
    output_path = workspace / "review.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "status": "rendered",
                    "image": "data/example.jpg",
                    "output": "cache/gallery/0001.jpg",
                    "gt_states": ["green"],
                    "primary_gt": "green",
                    "predicted_state": "red",
                    "confidence": 0.8166,
                }
            ]
        ),
        encoding="utf-8",
    )

    rows = script.create_review_template([index_path], output_path)

    assert len(rows) == 1
    assert rows[0]["case_id"] == "valid_0001"
    assert rows[0]["reason"] == ""
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["gallery_image"] == "cache/gallery/0001.jpg"
    assert csv_rows[0]["action"] == ""


def test_create_review_template_carries_previous_review_by_source_image():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_create_review_template_carries_previous_review"
    index_path = workspace / "traffic_light_valid_v1_mistakes_after_label_fix" / "index.json"
    previous_path = workspace / "previous.csv"
    output_path = workspace / "review.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    source_image = "data/external/traffic_light/valid/example.jpg"
    index_path.write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "status": "rendered",
                    "image": source_image,
                    "output": "cache/gallery/0001.jpg",
                    "gt_states": ["green"],
                    "primary_gt": "green",
                    "predicted_state": "unknown",
                    "confidence": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    with previous_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=script.FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "valid_0015",
                "split": "valid",
                "gallery_image": "cache/old.jpg",
                "source_image": source_image,
                "gt_states": "green",
                "primary_gt": "green",
                "predicted_state": "unknown",
                "confidence": "0.0",
                "status": "rendered",
                "reason": "missed_detection",
                "action": "add_similar_data",
                "priority": "high",
                "reviewed_by": "A",
                "notes": "angled traffic light",
            }
        )

    rows = script.create_review_template([index_path], output_path, previous_path)

    assert rows[0]["case_id"] == "valid_0001"
    assert rows[0]["reason"] == "missed_detection"
    assert rows[0]["action"] == "add_similar_data"
    assert rows[0]["priority"] == "high"
    assert rows[0]["notes"] == "angled traffic light"


def test_create_review_template_uses_unique_case_ids_for_repeated_split_inputs():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_create_review_template_repeated_split"
    first_index = workspace / "traffic_light_first_test_mistakes" / "index.json"
    second_index = workspace / "traffic_light_second_test_mistakes" / "index.json"
    output_path = workspace / "review.csv"
    first_index.parent.mkdir(parents=True, exist_ok=True)
    second_index.parent.mkdir(parents=True, exist_ok=True)
    _write_index(first_index, "data/example1.jpg")
    _write_index(second_index, "data/example2.jpg")

    rows = script.create_review_template([first_index, second_index], output_path)

    assert [row["case_id"] for row in rows] == ["test_0001", "test_0002"]


def test_create_review_template_cli(capsys, monkeypatch):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_create_traffic_light_error_review_template_cli"
    index_path = workspace / "traffic_light_test_v1_mistakes" / "index.json"
    output_path = workspace / "review.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            [
                {
                    "index": 2,
                    "status": "rendered",
                    "image": "data/example2.jpg",
                    "output": "cache/gallery/0002.jpg",
                    "gt_states": ["yellow"],
                    "primary_gt": "yellow",
                    "predicted_state": "unknown",
                    "confidence": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_traffic_light_error_review_template.py",
            str(index_path),
            "--output",
            str(output_path),
        ],
    )

    code = script.main()
    output = capsys.readouterr().out

    assert code == 0
    assert "rows=1" in output
    assert output_path.exists()


def _write_index(path: Path, image: str) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "status": "rendered",
                    "image": image,
                    "output": "cache/gallery/0001.jpg",
                    "gt_states": ["green"],
                    "primary_gt": "green",
                    "predicted_state": "unknown",
                    "confidence": 0.0,
                }
            ]
        ),
        encoding="utf-8",
    )
