import csv
import importlib.util
import json
import sys
from pathlib import Path
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "analyze_object_detection_error_review.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("analyze_object_detection_error_review", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_analyze_object_detection_review_groups_actions_into_buckets():
    script = _load_script()
    csv_path = _write_review_csv(
        [
            _row("scooter_0001", "false_negative", "scooter", "", "small_object", "add_similar_data"),
            _row("scooter_0002", "false_positive", "", "scooter", "label_missing", "fix_label"),
            _row("scooter_0003", "false_positive", "", "scooter", "background_confusion", "raise_threshold"),
            _row("scooter_0004", "wrong_class", "scooter", "bicycle", "ambiguous_object", "ignore_sample"),
            _row("scooter_0005", "false_negative", "scooter", "", "", ""),
        ]
    )

    analysis = script.analyze_review_csv(csv_path)
    report = script.render_markdown(analysis)

    assert analysis.total_cases == 5
    assert analysis.reviewed_cases == 4
    assert analysis.missing_reason == 1
    assert analysis.missing_action == 1
    assert len(analysis.buckets["补数据"]) == 1
    assert len(analysis.buckets["修标注"]) == 1
    assert len(analysis.buckets["调阈值/后处理"]) == 1
    assert len(analysis.buckets["忽略/剔除"]) == 1
    assert len(analysis.buckets["待确认"]) == 1
    assert analysis.reason_counts["small_object"] == 1
    assert analysis.class_pair_counts["scooter->unknown"] == 2
    assert "### 补数据 (1)" in report
    assert any("missing action" in warning for warning in analysis.warnings)


def test_analyze_object_detection_review_cli_writes_report_and_json(monkeypatch, capsys):
    script = _load_script()
    csv_path = _write_review_csv([_row("scooter_0001", "false_negative", "scooter", "", "small_object", "add_similar_data")])
    report_path = csv_path.with_suffix(".md")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_object_detection_error_review.py",
            "--input",
            str(csv_path),
            "--output",
            str(report_path),
            "--json",
        ],
    )

    code = script.main()
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["total_cases"] == 1
    assert payload["buckets"]["补数据"][0]["case_id"] == "scooter_0001"
    assert report_path.exists()
    assert "Object Detection Error Review Decision Report" in report_path.read_text(encoding="utf-8")


def _write_review_csv(rows: list[dict[str, str]]) -> Path:
    workspace = REPO_ROOT / "cache" / "pytest" / "test_analyze_object_detection_error_review" / uuid4().hex
    workspace.mkdir(parents=True, exist_ok=True)
    csv_path = workspace / "review.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _row(case_id: str, error_type: str, gt_cls: str, pred_cls: str, reason: str, action: str) -> dict[str, str]:
    return {
        "case_id": case_id,
        "target_class": "scooter",
        "error_type": error_type,
        "source_image": f"data/source/{case_id}.jpg",
        "gallery_image": f"cache/gallery/{case_id}.jpg",
        "gt_cls": gt_cls,
        "pred_cls": pred_cls,
        "confidence": "0.72" if pred_cls else "",
        "iou": "0.81" if error_type == "wrong_class" else "",
        "gt_bbox": "[1,2,3,4]" if gt_cls else "",
        "pred_bbox": "[5,6,7,8]" if pred_cls else "",
        "reason": reason,
        "action": action,
        "priority": "",
        "reviewed_by": "",
        "notes": "",
    }
