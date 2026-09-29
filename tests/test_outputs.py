"""Output folders: naming, never overwriting, keeping outputs out of the repository."""

import json
import os
from pathlib import Path

import pytest

from conftest import REPO_DIR
from sscd_libs import outputs


def test_run_folders_are_new_dated_and_named(output_root):
    a = outputs.new_run(name="N Esk/2018").root
    b = outputs.new_run(name="N Esk/2018").root
    assert a.parent == output_root / "runs"
    assert a.name.endswith("_N-Esk-2018")          # file-system friendly name
    assert b.name == a.name + "-2"                 # never reuses a folder
    for sub in ("results", "work/scales", "work/transects"):
        assert (a / sub).is_dir()


def test_outputs_inside_the_repository_are_refused(tmp_path):
    with pytest.raises(ValueError, match="inside the SSCD code folder"):
        outputs.new_run(root=REPO_DIR / "my_outputs")
    assert not (REPO_DIR / "my_outputs").exists()   # checked before anything is created


def test_explicit_run_dir_must_be_new_or_empty(tmp_path):
    used = tmp_path / "used"
    used.mkdir()
    (used / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        outputs.new_run(run_dir=used)
    assert (used / "keep.txt").exists()

    reserved = tmp_path / "reserved"   # a GUI-reserved folder holding only the launcher's log is fine
    reserved.mkdir()
    (reserved / "console.log").write_text("")
    (reserved / "STOP").write_text("")   # stop clicked while the run was still starting
    assert outputs.new_run(run_dir=reserved).root == reserved.resolve()


def test_output_root_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("SSCD_OUTPUT_ROOT", str(tmp_path / "env"))
    assert outputs.output_root() == (tmp_path / "env").resolve()
    assert outputs.output_root(tmp_path / "arg") == (tmp_path / "arg").resolve()


@pytest.mark.skipif(os.name != "nt", reason="Windows default")
def test_default_output_root_is_short_on_windows(tmp_path, monkeypatch):
    monkeypatch.setenv("SystemDrive", str(tmp_path / "drive"))
    (tmp_path / "drive").mkdir()
    assert outputs.default_output_root() == Path(str(tmp_path / "drive") + "\\") / "sscd_outputs"
    monkeypatch.setenv("SystemDrive", str(tmp_path / "missing"))        # cannot be created there
    assert outputs.default_output_root() == outputs.HOME_OUTPUT_ROOT


def test_latest_run_is_latest_started_completed(output_root):
    def make(name, started, status):
        d = output_root / "runs" / name
        d.mkdir(parents=True)
        (d / "run_info.json").write_text(json.dumps({"started": started, "status": status}))
        return d

    make("2026-01-01_1000_a", "2026-01-01T10:00:00", "completed")
    b = make("2026-01-01_1100_b", "2026-01-01T11:00:00", "completed")
    make("2026-01-01_1200_c", "2026-01-01T12:00:00", "failed")
    make("2026-01-01_1300_d", "2026-01-01T13:00:00", "running")
    (output_root / "runs" / "not-a-run").mkdir()
    assert outputs.latest_run().root == b
