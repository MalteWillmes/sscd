"""The SSCD icon replaces Streamlit's favicon.png (sscd_libs/app_icon.py)."""

import sys
import types

from sscd_libs.app_icon import ICON, install_favicon


def _fake_streamlit(tmp_path, monkeypatch, with_favicon=True):
    package = tmp_path / "streamlit"
    (package / "static").mkdir(parents=True)
    if with_favicon:
        (package / "static" / "favicon.png").write_bytes(b"streamlit logo")
    monkeypatch.setitem(sys.modules, "streamlit", types.SimpleNamespace(__file__=str(package / "__init__.py")))
    return package / "static" / "favicon.png"


def test_favicon_is_replaced(tmp_path, monkeypatch):
    favicon = _fake_streamlit(tmp_path, monkeypatch)
    assert install_favicon()
    assert favicon.read_bytes() == ICON.read_bytes()
    assert install_favicon()                     # already done: nothing to do


def test_other_streamlit_layout_is_left_alone(tmp_path, monkeypatch):
    favicon = _fake_streamlit(tmp_path, monkeypatch, with_favicon=False)
    assert not install_favicon()
    assert not favicon.exists()
