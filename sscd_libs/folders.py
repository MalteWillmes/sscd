"""
Choosing the scale-image folder in the GUI (sscd_app.py): the folder browser's
shortcuts and path, recently used folders, and the operating system's own
folder dialog (only when the app runs on the user's own computer).
"""

import os
import string
import subprocess
import sys
from pathlib import Path

from sscd_libs.data_processing import IMAGE_EXTENSIONS
from sscd_libs.runcontrol import read_json
from sscd_libs.settings import input_dir_allowed

# the folder dialog runs in its own process: Tk must own the main thread, which
# Streamlit's script thread is not
_DIALOG_SCRIPT = """
import sys, tkinter
from tkinter import filedialog
root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)   # in front of the browser
path = filedialog.askdirectory(initialdir=sys.argv[1] or None, mustexist=True,
                               title="Choose the folder with the scale images")
print(path or "")
"""


def dialog_available(settings):
    """The OS folder dialog opens on the computer running the app, so it is only
    offered when the app is used locally (no allowed_input_roots, i.e. not a server)."""
    return not settings["allowed_input_roots"]


def choose_folder_dialog(initial_dir=None):
    """Show the OS folder dialog; returns the chosen folder, or None if cancelled."""
    result = subprocess.run(  # noqa: S603 - our own script, run with this Python
        [sys.executable, "-c", _DIALOG_SCRIPT, str(initial_dir or "")],
        capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise OSError(result.stderr.strip().splitlines()[-1] if result.stderr.strip()
                      else "the folder dialog could not be opened")
    path = result.stdout.strip()
    return str(Path(path)) if path else None


def has_images(path):
    """True if the folder directly contains at least one scale image (stops at the first)."""
    try:
        with os.scandir(path) as entries:
            return any(e.is_file() and Path(e.name).suffix.lower() in IMAGE_EXTENSIONS for e in entries)
    except OSError:
        return False


def list_subfolders(path):
    """Sub-folders of `path`, by name; hidden and Windows system folders left out; empty if unreadable."""
    try:
        return sorted((p for p in Path(path).iterdir()
                       if p.is_dir() and not p.name.startswith((".", "$"))),
                      key=lambda p: p.name.lower())
    except OSError:
        return []


def browse_shortcuts(settings):
    """Starting points for the folder browser: the allowed roots on a server,
    otherwise the home folder and (on Windows) the drives."""
    if settings["allowed_input_roots"]:
        return [Path(r) for r in settings["allowed_input_roots"]]
    places = [Path.home()]
    if os.name == "nt":
        places += [Path(f"{d}:\\") for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
    else:
        places.append(Path("/"))
    return places


def path_parts(path, settings):
    """The folders from the top down to `path` (for a clickable path), leaving out
    those above the allowed roots."""
    path = Path(path)
    return [p for p in reversed([path, *path.parents]) if input_dir_allowed(p, settings)]


def recent_input_dirs(root, settings, n=8):
    """Image folders of the most recent runs (newest first, each once), if they still exist."""
    runs_dir = Path(root) / "runs"
    if not runs_dir.is_dir():
        return []
    folders = []
    for run in sorted((d for d in runs_dir.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True):
        folder = (read_json(run / "run_info.json") or {}).get("input_dir")
        if folder and folder not in folders and Path(folder).is_dir() and input_dir_allowed(folder, settings):
            folders.append(folder)
            if len(folders) == n:
                break
    return folders
