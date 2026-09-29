"""Choosing the scale-image folder in the GUI (sscd_libs/folders.py)."""

import json
from pathlib import Path

from PIL import Image

from sscd_libs.folders import has_images, list_subfolders, path_parts, recent_input_dirs

ANY_FOLDER = {"allowed_input_roots": []}


def test_has_images(tmp_path):
    (tmp_path / "sub").mkdir()
    Image.new("RGB", (4, 4)).save(tmp_path / "sub" / "a.TIF")
    (tmp_path / "notes.txt").write_text("x")
    assert not has_images(tmp_path)              # only directly in the folder
    assert has_images(tmp_path / "sub")
    assert not has_images(tmp_path / "missing")


def test_subfolders_leave_out_hidden_and_system_folders(tmp_path):
    for name in ("b", "A", ".git", "$RECYCLE.BIN"):
        (tmp_path / name).mkdir()
    (tmp_path / "file.txt").write_text("x")
    assert [p.name for p in list_subfolders(tmp_path)] == ["A", "b"]


def test_path_parts_stop_at_the_allowed_root(tmp_path):
    folder = tmp_path / "archive" / "2024" / "scales"
    parts = path_parts(folder, ANY_FOLDER)
    assert parts[-1] == folder and parts[0] == Path(folder.anchor)
    allowed = {"allowed_input_roots": [str((tmp_path / "archive").resolve())]}
    assert [p.name for p in path_parts(folder.resolve(), allowed)] == ["archive", "2024", "scales"]


def test_recent_folders_newest_first_once_and_existing(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    runs = [("2026-09-01_1000", a), ("2026-09-02_1000", b), ("2026-09-03_1000", a),
            ("2026-09-04_1000", tmp_path / "deleted")]
    for name, folder in runs:
        run = tmp_path / "root" / "runs" / name
        run.mkdir(parents=True)
        (run / "run_info.json").write_text(json.dumps({"input_dir": str(folder)}))
    (tmp_path / "root" / "runs" / "2026-09-05_1000").mkdir()     # no run_info.json (yet)
    assert recent_input_dirs(tmp_path / "root", ANY_FOLDER) == [str(a), str(b)]
    assert recent_input_dirs(tmp_path / "no_root", ANY_FOLDER) == []
