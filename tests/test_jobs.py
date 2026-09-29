"""The logic behind the web app: checking inputs, and reading a run's state."""

import json
from datetime import datetime, timedelta

import pytest

from sscd_libs.jobs import check_input_dir, parse_angles, recent_runs, run_state
from sscd_libs.settings import DEFAULTS


def test_check_input_dir(example_scales, tmp_path):
    ok = check_input_dir(str(example_scales), DEFAULTS)
    assert ok["ok"] and ok["n_images"] == 3 and ok["by_ext"] == {".tif": 3}
    assert not check_input_dir("", DEFAULTS)["ok"]
    assert "not found" in check_input_dir(str(tmp_path / "missing"), DEFAULTS)["message"]
    assert "No images" in check_input_dir(str(tmp_path), DEFAULTS)["message"]


@pytest.mark.parametrize("text, expected", [("0, 45, 90", [0, 45, 90]), ("0 180", [0, 180]), ("359", [359])])
def test_parse_angles(text, expected):
    assert parse_angles(text) == expected


@pytest.mark.parametrize("text", ["", "0, 360", "a, b", "0, 0", "-5"])
def test_parse_angles_rejects(text):
    with pytest.raises(ValueError):
        parse_angles(text)


def _run(root, name, info=None, progress=None):
    d = root / "runs" / name
    d.mkdir(parents=True)
    if info is not None:
        (d / "run_info.json").write_text(json.dumps(info))
    if progress is not None:
        (d / "progress.json").write_text(json.dumps(progress))
    return d


def test_a_running_run_without_a_live_process_is_marked_failed(tmp_path):
    # status 'running', but no process holds a run slot for it (e.g. killed)
    d = _run(tmp_path, "2026-01-01_1000_x", {"status": "running", "created": "2026-01-01T10:00:00"},
             {"status": "running", "stage": "focus", "stages": {}, "counts": {}, "warnings": []})
    state = run_state(d, tmp_path)
    assert state["status"] == "failed"
    assert "ended unexpectedly" in state["error"]
    assert json.loads((d / "run_info.json").read_text())["status"] == "failed"   # recorded


def test_finished_status_in_run_info_wins_over_stale_progress(tmp_path):
    d = _run(tmp_path, "2026-01-01_1000_y", {"status": "completed", "created": "2026-01-01T10:00:00"},
             {"status": "running", "stage": "circuli", "stages": {}, "counts": {"scales": 3}, "warnings": []})
    assert run_state(d, tmp_path)["status"] == "completed"


def test_a_queued_run_without_heartbeat_is_marked_failed(tmp_path):
    old = (datetime.now() - timedelta(minutes=10)).isoformat(timespec="seconds")
    d = _run(tmp_path, "2026-01-01_1000_z", {"status": "queued", "created": old},
             {"status": "queued", "stages": {}, "counts": {}, "warnings": [], "updated": old})
    assert run_state(d, tmp_path)["status"] == "failed"


def test_recent_runs_newest_first(tmp_path):
    _run(tmp_path, "2026-01-01_1000_a", {"status": "completed", "created": "2026-01-01T10:00:00",
                                         "counts": {"scales": 3, "focus_found": 2, "circuli": 677}})
    _run(tmp_path, "2026-01-02_0900_b", {"status": "failed", "created": "2026-01-02T09:00:00"})
    rows = recent_runs(tmp_path)
    assert [r["run"] for r in rows] == ["2026-01-02_0900_b", "2026-01-01_1000_a"]
    assert rows[1]["circuli"] == 677
