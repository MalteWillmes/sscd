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
