"""
Use the SSCD icon as the web app's browser-tab icon from the first moment.

Streamlit's page first loads the package's own favicon.png and only switches to
the app's icon (st.set_page_config) once the app has run, so the Streamlit logo
flashes up on every start. Streamlit has no setting for this, so the favicon.png
of the installed Streamlit package - in SSCD's own environment - is replaced by
the SSCD icon. Run by start_sscd.bat / .sh before the app starts:

    python -m sscd_libs.app_icon
"""

import shutil
from pathlib import Path

from sscd_libs.helpers import REPO_DIR

ICON = REPO_DIR / "assets" / "sscd_icon.png"


def install_favicon():
    """Copy the SSCD icon over Streamlit's favicon.png if it differs. Returns True if
    the favicon is (now) the SSCD icon; never fails the app over it."""
    try:
        import streamlit

        favicon = Path(streamlit.__file__).parent / "static" / "favicon.png"
        if not favicon.exists():  # a Streamlit version with another layout: leave it
            return False
        if favicon.read_bytes() != ICON.read_bytes():
            shutil.copyfile(ICON, favicon)
        return True
    except (ImportError, OSError):
        return False


if __name__ == "__main__":
    install_favicon()
