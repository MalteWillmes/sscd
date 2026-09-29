"""
Where SSCD writes its results, and how an output folder is organised.

All runs live under one output root (default: ~/sscd_outputs, change it with
the SSCD_OUTPUT_ROOT environment variable or --output_root):

    <output root>/
        runs/<YYYY-MM-DD_HHMM>[_<name>]/          one folder per sscd.py run
            run_info.json                          inputs, parameters, versions, counts
            sscd.log
            results/   circuli.csv, focus.csv, scales_summary.csv
            overlays/  <scale>_overlay.jpg         (overlay_detections.py)
            qc/        focus_plots/, circuli_plots/, no_detections/
            work/      scales/, transects/         intermediate images (deletable; overlays
                                                   then use the original input images)
        evaluations/<YYYY-MM-DD_HHMM>[_<name>]/   one folder per eval_detector.py run
            eval_info.json, eval.log, results/, plots/

Nothing is ever deleted: every run gets a new folder.
"""

import json
import logging
import os
import platform
import re
import subprocess  # noqa: S404 - only runs git to record the code version
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sscd_libs.helpers import REPO_DIR

DEFAULT_OUTPUT_ROOT = Path.home() / "sscd_outputs"

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def output_root(override=None):
    """The output root: `override` if given, else $SSCD_OUTPUT_ROOT, else output_root from
    the settings file (sscd_libs/settings.py), else ~/sscd_outputs."""
    from sscd_libs.settings import load_settings

    root = (override or os.environ.get("SSCD_OUTPUT_ROOT") or load_settings()["output_root"]
            or DEFAULT_OUTPUT_ROOT)
    return Path(root).expanduser().resolve()


def _check_outside_repo(path):
    path = Path(path).resolve()
    if path == REPO_DIR or REPO_DIR in path.parents:
        raise ValueError(
            f"'{path}' is inside the SSCD code folder ({REPO_DIR}). Keep outputs separate from the "
            f"code: use the default output root ({DEFAULT_OUTPUT_ROOT}), set SSCD_OUTPUT_ROOT, or "
            "pass another --output_root."
        )


def _new_dir(parent, name=None):
    """Create parent/<YYYY-MM-DD_HHMM>[_<name>], adding -2, -3 ... if it already exists."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    if name:
        # keep names file-system friendly and readable: letters, digits, - and _
        clean = re.sub(r"[^\w\-]+", "-", name.strip()).strip("-_")
        if clean:
            stamp = f"{stamp}_{clean}"
    parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, 1000):
        candidate = parent / (stamp if i == 1 else f"{stamp}-{i}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a new folder in {parent}")


# files a launcher (the GUI) may already have put in a reserved run folder: its log, and a
# stop request made while the run was still starting
LAUNCHER_FILES = {"console.log", "STOP"}


def _use_dir(path):
    """Use an explicitly given folder: it must be new or empty (nothing is ever overwritten)."""
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(p.name not in LAUNCHER_FILES for p in path.iterdir())):
        raise ValueError(f"'{path}' already exists and is not empty - SSCD never overwrites results. "
                         "Choose a new folder.")
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class RunPaths:
    """Paths inside one run folder."""

    root: Path

    @property
    def info(self):
        return self.root / "run_info.json"

    @property
    def log(self):
        return self.root / "sscd.log"

    @property
    def results(self):
        return self.root / "results"

    @property
    def circuli_csv(self):
        return self.results / "circuli.csv"

    @property
    def focus_csv(self):
        return self.results / "focus.csv"

    @property
    def summary_csv(self):
        return self.results / "scales_summary.csv"

    @property
    def per_image(self):
        return self.results / "per_image"

    @property
    def overlays(self):
        return self.root / "overlays"

    @property
    def qc(self):
        return self.root / "qc"

    @property
    def focus_plots(self):
        return self.qc / "focus_plots"

    @property
    def circuli_plots(self):
        return self.qc / "circuli_plots"

    @property
    def no_detections(self):
        return self.qc / "no_detections"

    @property
    def work(self):
        return self.root / "work"

    @property
    def scales(self):
        return self.work / "scales"

    @property
    def transects(self):
        return self.work / "transects"

    def make_dirs(self):
        for d in (self.results, self.qc, self.scales, self.transects):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EvalPaths:
    """Paths inside one evaluation folder."""

    root: Path

    @property
    def info(self):
        return self.root / "eval_info.json"

    @property
    def log(self):
        return self.root / "eval.log"

    @property
    def results(self):
        return self.root / "results"

    @property
    def plots(self):
        return self.root / "plots"

    @property
    def temp(self):
        return self.root / "temp"

    def make_dirs(self):
        for d in (self.results, self.temp):
            d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
def new_run(root=None, name=None, run_dir=None):
    """Create the folder for a new sscd.py run (or use `run_dir`, which must be new/empty)."""
    _check_outside_repo(run_dir or output_root(root))  # before creating anything
    path = _use_dir(run_dir) if run_dir else _new_dir(output_root(root) / "runs", name)
    paths = RunPaths(path)
    paths.make_dirs()
    return paths


def new_evaluation(root=None, name=None, eval_dir=None):
    """Create the folder for a new eval_detector.py run (or use `eval_dir`, new/empty)."""
    _check_outside_repo(eval_dir or output_root(root))  # before creating anything
    path = _use_dir(eval_dir) if eval_dir else _new_dir(output_root(root) / "evaluations", name)
    paths = EvalPaths(path)
    paths.make_dirs()
    return paths


def reserve_run_dir(root=None, name=None):
    """Create a new, empty dated run folder (for a caller that passes it to sscd.py --run_dir)."""
    _check_outside_repo(output_root(root))
    return _new_dir(output_root(root) / "runs", name)


def reserve_eval_dir(root=None, name=None):
    """Create a new, empty dated evaluation folder (for eval_detector.py --eval_dir)."""
    _check_outside_repo(output_root(root))
    return _new_dir(output_root(root) / "evaluations", name)


def open_run(run_dir):
    """An existing run folder (as written by sscd.py)."""
    paths = RunPaths(Path(run_dir).resolve())
    if not paths.info.exists():
        raise FileNotFoundError(f"'{paths.root}' is not an SSCD run folder (no run_info.json).")
    return paths


def latest_run(root=None):
    """
    The most recently started run under the output root that completed.

    Ordered by the start time recorded in run_info.json (file-system times change
    when files are added or folders copied). Newer runs that failed or are still
    running are skipped, with a warning.
    """
    runs_dir = output_root(root) / "runs"
    runs = []
    for p in runs_dir.glob("*"):
        try:
            info = json.loads((p / "run_info.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # not a run folder, or run_info.json unreadable
        runs.append((info.get("started", ""), p.name, info.get("status"), p))
    completed = [r for r in runs if r[2] == "completed"]
    if not completed:
        raise FileNotFoundError(f"No completed runs found in {runs_dir}")
    latest = max(completed)
    for started, name, status, _ in sorted(runs):
        if (started, name) > latest[:2]:
            logger.warning("Skipping newer run %s (status: %s)", name, status)
    return RunPaths(latest[3])


# ------------------------------------------------------------------------------
def code_version():
    """SSCD version and git commit of the code producing the results."""
    version = {"sscd": "unknown", "git_commit": None, "git_dirty": None}
    try:
        import tomllib

        with open(REPO_DIR / "pyproject.toml", "rb") as f:
            version["sscd"] = tomllib.load(f)["project"]["version"]
    except (ImportError, OSError, KeyError):
        pass
    try:
        git = ["git", "-C", str(REPO_DIR)]
        version["git_commit"] = subprocess.run(  # noqa: S603 - fixed git command
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        status = subprocess.run(  # noqa: S603 - fixed git command
            [*git, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
        version["git_dirty"] = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        pass  # not a git checkout, or git not installed
    return version


def environment():
    import tensorflow as tf

    return {"python": platform.python_version(), "tensorflow": tf.__version__,
            "platform": platform.platform(), "command": sys.argv}


def write_info(path, info):
    """Write (or rewrite) an info JSON file."""
    Path(path).write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
