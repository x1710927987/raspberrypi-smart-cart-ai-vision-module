import csv
import importlib.util
import sys
from pathlib import Path
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "create_traffic_light_data_addition_plan.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("create_traffic_light_data_addition_plan", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_create_addition_plan_filters_add_similar_data_cases():
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_create_traffic_light_data_addition_plan" / uuid4().hex
    review_path = workspace / "review.csv"
    output_csv = workspace / "plan.csv"
    output_md = workspace / "plan.md"
    _write_review_csv(
        review_path,
        [
            _row("valid_0001", "green", "unknown", "missed_detection", "add_similar_data"),
            _row("valid_0002", "red", "green", "multiple_light", "fix_label"),
            _row("test_0001", "yellow", "red", "wrong_color", "add_similar_data"),
        ],
    )

    plan = script.create_addition_plan(review_path, output_csv=output_csv, output_markdown=output_md)

    assert len(plan.rows) == 2
    assert plan.class_counts == {"green": 1, "yellow": 1}
    assert plan.reason_counts == {"missed_detection": 1, "wrong_color": 1}
    assert plan.suggested_total_images == 140
    assert output_csv.exists()
    assert output_md.exists()
    with output_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["case_id"] for row in rows] == ["valid_0001", "test_0001"]
    assert rows[0]["recommended_focus"]


def test_create_addition_plan_cli(capsys, monkeypatch):
    script = _load_script()
    workspace = REPO_ROOT / "cache" / "pytest" / "test_create_traffic_light_data_addition_plan_cli" / uuid4().hex
    review_path = workspace / "review.csv"
    output_csv = workspace / "plan.csv"
    output_md = workspace / "plan.md"
    _write_review_csv(review_path, [_row("valid_0001", "green", "unknown", "too_small", "add_similar_data")])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_traffic_light_data_addition_plan.py",
            "--input",
            str(review_path),
            "--output-csv",
            str(output_csv),
            "--output-markdown",
            str(output_md),
        ],
    )

    code = script.main()
    output = capsys.readouterr().out

    assert code == 0
    assert "rows=1" in output
    assert "status=ok" in output
    assert output_csv.exists()


def _write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _row(case_id: str, gt: str, pred: str, reason: str, action: str) -> dict[str, str]:
    split = case_id.split("_", 1)[0]
    return {
        "case_id": case_id,
        "split": split,
        "gallery_image": f"cache/gallery/{case_id}.jpg",
        "source_image": f"data/source/{case_id}.jpg",
        "gt_states": gt,
        "primary_gt": gt,
        "predicted_state": pred,
        "confidence": "0.0",
        "status": "rendered",
        "reason": reason,
        "action": action,
        "priority": "",
        "reviewed_by": "",
        "notes": "The perspective of observing traffic lights may not necessarily be straight",
    }
