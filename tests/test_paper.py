import json
from pathlib import Path

from gnss_lidar_slam.paper import load_sequence_values, write_table3


def _write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_table3_is_derived_from_saved_results(tmp_path: Path):
    run = "RUN-TEST-0001.json"
    _write(tmp_path / "stage_6/runs/manifests" / run, {
        "dataset_id": "DATASET-001", "sequence_id": "seq_1", "variant": "baseline"
    })
    _write(tmp_path / "stage_6/runs/results" / run, {
        "status": "completed", "metrics": [{"name": "checkpoint_3d_rmse", "value": 1.234}]
    })
    candidate = "RUN-TEST-0002.json"
    _write(tmp_path / "stage_6/runs/manifests" / candidate, {
        "dataset_id": "DATASET-001", "sequence_id": "seq_1", "variant": "candidate"
    })
    _write(tmp_path / "stage_6/runs/results" / candidate, {
        "status": "completed", "metrics": [{"name": "checkpoint_3d_rmse", "value": 1.111}]
    })
    rows = load_sequence_values(tmp_path)
    assert rows[0]["KISS-SLAM"] == 1.234
    assert rows[0]["Proposed method"] == 1.111
    write_table3(rows, tmp_path / "out")
    csv_text = (tmp_path / "out/table3_sequence_accuracy.csv").read_text()
    tex_text = (tmp_path / "out/table3_sequence_accuracy.tex").read_text()
    assert "1.23" in csv_text and "1.11" in csv_text
    assert r"\textbf{1.11}" in tex_text

