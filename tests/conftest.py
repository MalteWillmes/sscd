"""
Shared test setup.

Run all tests:            uv run python -m pytest
Skip the slow ones:       uv run python -m pytest -m "not slow"

The slow tests run the detectors and need the trained weights
(uv run python -m sscd_libs.fetch weights); without them they are skipped.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))  # sscd.py, sscd_app.py etc. live at the repository root

# keep test runs quiet and independent of the user's settings / output root
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ["SSCD_SETTINGS"] = str(REPO_DIR / "tests" / "no_such_settings.toml")

WEIGHTS_PRESENT = (REPO_DIR / "data" / "yoloV3_checkpoints" / "circuli_detector" / "yolov3_train_22.tf.index").exists()
needs_weights = pytest.mark.skipif(not WEIGHTS_PRESENT, reason="trained weights not downloaded")


@pytest.fixture
def output_root(tmp_path, monkeypatch):
    """A fresh output root for the test, set as SSCD_OUTPUT_ROOT."""
    root = tmp_path / "sscd_outputs"
    monkeypatch.setenv("SSCD_OUTPUT_ROOT", str(root))
    return root


@pytest.fixture
def example_scales():
    return REPO_DIR / "data" / "example_scales"
