"""Manual focus for scales where SSCD found no focus (manual_focus.py, sscd_libs/manual_focus.py)."""

import json
import subprocess
import sys

import pandas as pd
import pytest
from PIL import Image

from conftest import REPO_DIR, needs_weights
from sscd_libs import manual_focus as mf
from sscd_libs.data_processing import pascal_to_evaltxt
from sscd_libs.outputs import RunPaths

FOCUS_HEADER = ["scale_id", "class_name", "score", "focus_method", "n_focus_boxes", "xmin", "ymin", "xmax",
                "ymax", "x_px", "y_px"]


def _fake_run(tmp_path):
    """A finished run folder: scale A with a detected focus, B and C without."""
    paths = RunPaths(tmp_path / "run")
    paths.make_dirs()
    for s in ("A", "B", "C"):
        Image.new("RGB", (400, 300), (200, 200, 200)).save(paths.scales / f"{s}.jpg")
    pd.DataFrame([["A", "focus", 0.99, "standard", 1, 100, 100, 150, 160, 125.0, 130.0]],
                 columns=FOCUS_HEADER).to_csv(paths.focus_csv, index=False)
    pd.DataFrame({"scale_id": ["A", "B", "C"], "focus_found": [True, False, False],
                  "focus_method": ["standard", None, None], "review_note": [None, None, None]}
                 ).to_csv(paths.summary_csv, index=False)
    pd.DataFrame(columns=["scale_id"]).to_csv(paths.circuli_csv, index=False)
    paths.info.write_text(json.dumps({"status": "completed", "input_dir": str(tmp_path / "originals"),
                                      "parameters": {"transect_angles": [0], "transect_max_boxes": 500,
                                                     "plot_dets": False, "dets_separate_files": False,
                                                     "overlays": False}}))
    return paths


def test_decisions_and_review_queue(tmp_path):
    paths = _fake_run(tmp_path)
    assert [(q["scale_id"], q["state"]) for q in mf.review_queue(paths)] == [("B", "pending"), ("C", "pending")]

    mf.save_decision(paths, "B", "focus", 210.4, 140.6)
    mf.save_decision(paths, "C", "no_focus", note="regenerated")
    queue = {q["scale_id"]: q for q in mf.review_queue(paths)}
    assert queue["B"]["state"] == "focus" and (queue["B"]["x_px"], queue["B"]["y_px"]) == (210, 141)
    assert queue["C"]["state"] == "no_focus" and queue["C"]["note"] == "regenerated"

    mf.remove_decision(paths, "C")
    assert {q["scale_id"]: q["state"] for q in mf.review_queue(paths)}["C"] == "pending"

    mf.mark_applied(paths, ["B"])
    assert {q["scale_id"]: q["state"] for q in mf.review_queue(paths)}["B"] == "applied"
    with pytest.raises(ValueError, match="already been applied"):
        mf.save_decision(paths, "B", "no_focus")
    with pytest.raises(ValueError, match="needs its position"):
        mf.save_decision(paths, "C", "focus")


def test_focus_box_is_the_runs_median_and_centred_on_the_click(tmp_path):
    paths = _fake_run(tmp_path)
    assert mf.focus_box_size(paths) == (50, 60)
    decisions = pd.DataFrame([{"scale_id": "B", "decision": "focus", "x_px": 210, "y_px": 140}])
    row = mf.manual_focus_rows(decisions, (50, 60)).iloc[0]
    assert (row.xmin, row.ymin, row.xmax, row.ymax) == (185, 110, 235, 170)
    assert ((row.xmin + row.xmax) / 2, (row.ymin + row.ymax) / 2) == (210, 140)
    assert row.focus_method == "manual" and pd.isna(row.score)

    paths.focus_csv.unlink()   # no detected focus in the run: a fixed fraction of the image width
    assert mf.focus_box_size(paths, 3840) == (54, 54)


def test_export_writes_voc_annotations_sscd_can_read(tmp_path):
    paths = _fake_run(tmp_path)
    mf.save_decision(paths, "B", "focus", 210, 140)
    mf.save_decision(paths, "C", "no_focus", note="damaged")   # nothing to annotate
    out, n = mf.export_annotations(paths, input_dir=None)
    assert n == 1 and sorted(p.name for p in (out / "images").iterdir()) == ["B.jpg"]
    (tmp_path / "txt").mkdir()
    pascal_to_evaltxt(str(out / "xml"), "B", str(tmp_path / "txt"))
    assert (tmp_path / "txt" / "B.txt").read_text().split() == ["focus", "185", "110", "235", "170"]

    out, n = mf.export_annotations(paths, input_dir=None, include_detected=True)
    manifest = pd.read_csv(out / "manifest.csv")
    assert n == 2 and manifest.set_index("scale_id").loc["A", "source"] == "detected (unverified)"


def test_no_usable_focus_is_recorded_without_detection(tmp_path):
    import manual_focus
    from sscd_libs.runcontrol import RunControl

    paths = _fake_run(tmp_path)
    mf.save_decision(paths, "C", "no_focus", note="regenerated")
    info = json.loads(paths.info.read_text())
    record = manual_focus.apply_corrections(paths, info, RunControl(paths.root))
    summary = pd.read_csv(paths.summary_csv).set_index("scale_id")
    assert summary.loc["C", "review_note"] == "no usable focus: regenerated"
    assert not summary.loc["C", "focus_found"] and pd.isna(summary.loc["B", "review_note"])
    assert record["no_usable_focus"] == ["C"] and record["manual_focus"] == []
    assert len(pd.read_csv(paths.focus_csv)) == 1           # focus.csv unchanged
    assert mf.review_queue(paths)[1]["state"] == "applied"


@pytest.mark.slow
@needs_weights
def test_apply_a_manual_focus_to_the_example_run(tmp_path):
    run_dir = tmp_path / "run"
    subprocess.run([sys.executable, str(REPO_DIR / "sscd.py"), "--img_dir", str(REPO_DIR / "data" / "example_scales"),
                    "--run_dir", str(run_dir), "--output_root", str(tmp_path), "--plot_dets", "False",
                    "--overlays", "True"], cwd=REPO_DIR, check=True, capture_output=True)
    paths = RunPaths(run_dir)
    before = pd.read_csv(paths.summary_csv).set_index("scale_id")

    mf.save_decision(paths, "Tummel 60300", "focus", 2351, 1306)    # set by eye
    result = subprocess.run([sys.executable, str(REPO_DIR / "manual_focus.py"), "apply", "--run_dir", str(run_dir)],
                            cwd=REPO_DIR, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]

    after = pd.read_csv(paths.summary_csv).set_index("scale_id")
    tummel = after.loc["Tummel 60300"]
    assert tummel.focus_found and tummel.focus_method == "manual" and tummel.review_note == "focus set by hand"
    assert tummel.n_transects == 5 and tummel.total_n_circuli > 50
    # the scales found by the detector are unchanged
    found = ["N Esk NC_2018_186", "N Esk NC_2018_187"]
    pd.testing.assert_frame_equal(after.loc[found].drop(columns="review_note"),
                                  before.loc[found].drop(columns="review_note"))

    circuli = pd.read_csv(paths.circuli_csv)
    info = json.loads(paths.info.read_text())
    assert info["status"] == "completed" and info["counts"]["focus_found"] == 3
    assert info["counts"]["circuli"] == len(circuli) and (circuli["scale_id"] == "Tummel 60300").sum() > 50
    assert (paths.overlays / "Tummel 60300_overlay.jpg").exists()
    assert len(list(paths.transects.glob("Tummel 60300_*.jpg"))) == 5
    assert mf.review_queue(paths)[0]["state"] == "applied"
