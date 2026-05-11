import csv
import importlib.util
import json
import sys
from pathlib import Path
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "analyze_object_detection_errors.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("analyze_object_detection_errors", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_analyze_object_detection_errors_counts_classes_and_writes_scooter_csv():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_analyze_object_detection_errors" / uuid4().hex
    mistakes_path = workspace / "mistakes.json"
    review_csv = workspace / "scooter_review.csv"
    _write_mistakes(mistakes_path)

    analysis = script.analyze_mistake_file(mistakes_path, target_class="scooter", gallery_dir=None)
    report = script.render_markdown(analysis)
    script.write_review_csv(review_csv, analysis.target_review_cases)
    rows = list(csv.DictReader(review_csv.open("r", encoding="utf-8-sig", newline="")))

    assert analysis.total_images == 2
    assert analysis.total_error_events == 5
    assert analysis.error_type_counts["false_negative"] == 2
    assert analysis.error_type_counts["false_positive"] == 2
    assert analysis.error_type_counts["wrong_class"] == 1
    assert analysis.class_errors["scooter"].false_negative == 1
    assert analysis.class_errors["scooter"].false_positive == 1
    assert analysis.class_errors["scooter"].wrong_class_as_pred == 1
    assert analysis.confusion_counts["roadblock->scooter"] == 1
    assert len(analysis.target_review_cases) == 3
    assert {row["error_type"] for row in rows} == {"false_negative", "false_positive", "wrong_class"}
    assert all(row["reason"] == "" and row["action"] == "" for row in rows)
    assert "scooter Review Queue" in report


def test_analyze_object_detection_errors_cli_writes_report_and_csv(monkeypatch, capsys):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_analyze_object_detection_errors_cli" / uuid4().hex
    mistakes_path = workspace / "mistakes.json"
    review_csv = workspace / "review.csv"
    report_path = workspace / "report.md"
    _write_mistakes(mistakes_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_object_detection_errors.py",
            "--input",
            str(mistakes_path),
            "--target-class",
            "scooter",
            "--gallery-dir",
            str(workspace / "missing_gallery"),
            "--review-csv",
            str(review_csv),
            "--report",
            str(report_path),
            "--json",
        ],
    )

    code = script.main()
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["target_class"] == "scooter"
    assert len(payload["target_review_cases"]) == 3
    assert review_csv.exists()
    assert report_path.exists()


def _write_mistakes(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "image": "data/source/a.jpg",
                    "false_negatives": [{"cls": "scooter", "bbox": [1, 2, 3, 4], "conf": None}],
                    "false_positives": [{"cls": "car", "bbox": [5, 6, 7, 8], "conf": 0.61}],
                    "misclassified": [
                        {
                            "gt": {"cls": "roadblock", "bbox": [10, 11, 12, 13], "conf": None},
                            "pred": {"cls": "scooter", "bbox": [10, 11, 12, 13], "conf": 0.7},
                            "iou": 0.82,
                        }
                    ],
                },
                {
                    "image": "data/source/b.jpg",
                    "false_negatives": [{"cls": "bicycle", "bbox": [1, 2, 3, 4], "conf": None}],
                    "false_positives": [{"cls": "scooter", "bbox": [5, 6, 7, 8], "conf": 0.55}],
                    "misclassified": [],
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
