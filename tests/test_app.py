"""The Streamlit web app renders and validates its inputs (no run is started)."""

import pytest

from conftest import REPO_DIR

pytest.importorskip("streamlit", reason="GUI extra not installed (uv sync --extra gui)")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def app(output_root):
    return AppTest.from_file(str(REPO_DIR / "sscd_app.py"), default_timeout=60).run()


def _start_button(at):
    return next(b for b in at.button if b.label == "Start run")


def test_app_renders(app):
    assert not app.exception
    assert app.title[0].value == "Salmon Scale Circuli Detector"
    assert _start_button(app).disabled          # nothing chosen yet


def test_folder_and_angles_are_validated(app, example_scales, tmp_path):
    app.text_input(key="img_dir").set_value(str(tmp_path / "missing")).run()
    assert "not found" in app.error[0].value

    app.text_input(key="img_dir").set_value(str(example_scales)).run()
    assert app.success[0].value.startswith("3 images")
    assert not _start_button(app).disabled

    angles = next(t for t in app.text_input if t.label.startswith("Transect angles"))
    angles.set_value("0, 400").run()
    assert "between 0 and 359" in app.error[0].value
    assert _start_button(app).disabled


def test_recent_folder_fills_in_the_folder(output_root, example_scales):
    import json

    run = output_root / "runs" / "2026-09-01_1000"
    run.mkdir(parents=True)
    (run / "run_info.json").write_text(json.dumps({"input_dir": str(example_scales)}))
    at = AppTest.from_file(str(REPO_DIR / "sscd_app.py"), default_timeout=60).run()
    at.selectbox(key="recent_dir").set_value(str(example_scales)).run()
    assert not at.exception
    assert at.text_input(key="img_dir").value == str(example_scales)
    assert at.success[0].value.startswith("3 images")


def test_focus_review_shows_scales_without_focus(output_root):
    from test_manual_focus import _fake_run

    run = output_root / "runs" / "2026-10-07_1000_review"
    run.parent.mkdir(parents=True)
    paths = _fake_run(run.parent)            # creates <runs>/run
    paths.root.rename(run)
    at = AppTest.from_file(str(REPO_DIR / "sscd_app.py"), default_timeout=60)
    at.query_params["run"] = run.name
    at.run()
    assert not at.exception
    assert "Focus review" in [h.value for h in at.subheader]
    assert at.selectbox(key="mf_scale").options == ["B  (to review)", "C  (to review)"]
    assert next(b for b in at.button if b.label == "Set focus here").disabled   # no click yet


def test_run_that_is_still_starting_shows_without_errors(output_root):
    # just after "Start run" the run folder exists, but sscd.py has not written run_info.json yet
    run = output_root / "runs" / "2026-10-07_1402_starting"
    run.mkdir(parents=True)
    (run / "console.log").write_text("")
    at = AppTest.from_file(str(REPO_DIR / "sscd_app.py"), default_timeout=60)
    at.query_params["run"] = run.name
    at.run()
    assert not at.exception
    assert "Focus review" not in [h.value for h in at.subheader]
