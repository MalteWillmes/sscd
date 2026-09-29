"""
End-to-end regression tests: they run the detectors (a few minutes on a CPU) and need the
trained weights. Skipped when the weights are missing; run only these with
`uv run python -m pytest -m slow`.
"""

import json
import subprocess
import sys

import pandas as pd
import pytest

from conftest import REPO_DIR, needs_weights

pytestmark = [pytest.mark.slow, needs_weights]


def test_circuli_detector_reproduces_2021_detections(tmp_path):
    """The circuli detector on the bundled eval transects gives exactly the detections the
    original (2021, Keras 2) system produced: guards the Keras-2 weight loading and the
    model port."""
    import sscd
    from sscd_libs.detection import detect

    dets = detect(img_dir=str(REPO_DIR / "data" / "eval_example" / "imgs"), weights=sscd.WEIGHTS["circuli"],
                  classes_file=sscd.CLASS_FILES["circuli"], no_det_dir=str(tmp_path / "none"),
                  yolo_max_boxes=200, **sscd.CIRCULI_MODEL)
    ref = pd.read_csv(REPO_DIR / "data" / "eval_example" / "detections.csv")
    ref = ref[ref.img_id.isin(dets.img_id)]
    key = ["img_id", "detection_nr"]
    new, ref = dets.sort_values(key).reset_index(drop=True), ref.sort_values(key).reset_index(drop=True)
    assert len(new) == len(ref) == 493
    assert (new[["xmin", "ymin", "xmax", "ymax"]].to_numpy() == ref[["xmin", "ymin", "xmax", "ymax"]].to_numpy()).all()
    assert (new.score - ref.score).abs().max() < 1e-4


def test_full_run_of_the_example_scales(tmp_path):
    run_dir = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, str(REPO_DIR / "sscd.py"), "--img_dir", str(REPO_DIR / "data" / "example_scales"),
         "--run_dir", str(run_dir), "--output_root", str(tmp_path), "--plot_dets", "False", "--overlays", "True"],
        cwd=REPO_DIR, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["status"] == "completed"
    assert info["counts"] == {"scales": 3, "focus_found": 2, "transects": 10, "circuli": 677}

    summary = pd.read_csv(run_dir / "results" / "scales_summary.csv", dtype={"scale_id": str}).set_index("scale_id")
    assert summary.loc["N Esk NC_2018_186", "total_n_circuli"] == 363
    assert summary.loc["N Esk NC_2018_186", "mean_n_circuli"] == 72.6
    assert not summary.loc["Tummel 60300", "focus_found"]
    assert sorted(p.name for p in (run_dir / "overlays").iterdir()) == [
        "N Esk NC_2018_186_overlay.jpg", "N Esk NC_2018_187_overlay.jpg"]
    progress = json.loads((run_dir / "progress.json").read_text())
    assert progress["status"] == "completed" and progress["warnings"]   # the 'no focus' warning
