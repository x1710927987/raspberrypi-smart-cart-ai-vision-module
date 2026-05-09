import csv
import importlib.util
import json
import sys
from pathlib import Path
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "analyze_traffic_light_error_review.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("analyze_traffic_light_error_review", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_analyze_review_csv_groups_actions_into_decision_buckets():
    script = _load_script()
    csv_path = _write_review_csv(
        [
            _row("valid_0001", "green", "unknown", "0.0", "missed_detection", "add_similar_data"),
            _row("valid_0002", "red", "yellow", "0.92", "label_error", "fix_label"),
            _row("valid_0003", "yellow", "red", "0.55", "wrong_color", "adjust_postprocess"),
            _row("valid_0004", "red", "unknown", "0.0", "ambiguous_color", "ignore_sample"),
            _row("valid_0005", "green", "unknown", "0.0", "missed_detection", ""),
        ]
    )

    analysis = script.analyze_review_csv(csv_path)
    report = script.render_markdown(analysis)

    assert analysis.total_cases == 5
    assert analysis.reviewed_cases == 4
    assert analysis.missing_action == 1
    assert len(analysis.buckets["补数据"]) == 1
    assert len(analysis.buckets["修标注"]) == 1
    assert len(analysis.buckets["调阈值/后处理"]) == 1
    assert len(analysis.buckets["忽略/剔除"]) == 1
    assert len(analysis.buckets["待确认"]) == 1
    assert analysis.reason_counts["missed_detection"] == 2
    assert analysis.confusion_counts["green->unknown"] == 2
    assert "### 补数据 (1)" in report
    assert "valid_0001" in report
    assert any("missing action" in warning for warning in analysis.warnings)


def test_analyze_review_cli_writes_report_and_json(capsys, monkeypatch):
    script = _load_script()
    csv_path = _write_review_csv([_row("test_0001", "green", "red", "0.48", "wrong_color", "add_similar_data")])
    report_path = csv_path.with_suffix(".md")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_traffic_light_error_review.py",
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
    assert payload["buckets"]["补数据"][0]["case_id"] == "test_0001"
    assert report_path.exists()
    assert "Traffic Light Error Review Decision Report" in report_path.read_text(encoding="utf-8")


def _write_review_csv(rows: list[dict[str, str]]) -> Path:
    workspace = REPO_ROOT / "cache" / "pytest" / "test_analyze_traffic_light_error_review" / uuid4().hex
    workspace.mkdir(parents=True, exist_ok=True)
    csv_path = workspace / "review.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _row(case_id: str, gt: str, pred: str, conf: str, reason: str, action: str) -> dict[str, str]:
    split = case_id.split("_", 1)[0]
    return {
        "case_id": case_id,
        "split": split,
        "gallery_image": f"cache/gallery/{case_id}.jpg",
        "source_image": f"data/source/{case_id}.jpg",
        "gt_states": gt,
        "primary_gt": gt,
        "predicted_state": pred,
        "confidence": conf,
        "status": "rendered",
        "reason": reason,
        "action": action,
        "priority": "",
        "reviewed_by": "",
        "notes": "",
    }
