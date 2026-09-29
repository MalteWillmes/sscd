"""
Starting, watching and stopping sscd.py runs - the logic behind the GUI
(sscd_app.py), kept free of Streamlit so it can be tested directly.

A run is a separate sscd.py process writing into its own run folder; its state
is read from the folder (run_info.json, progress.json, logs), so the GUI can be
closed, reloaded or used by several people without affecting the run.
"""

import collections
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from sscd_libs.data_processing import find_duplicate_images, list_input_images
from sscd_libs.helpers import REPO_DIR
from sscd_libs.outputs import output_root, reserve_run_dir
from sscd_libs.runcontrol import (
    PROGRESS_FILE,
    STAGES,
    read_json,
    request_stop,
    run_is_alive,
    slot_owner_pid,
)
from sscd_libs.settings import input_dir_allowed

# a queued run refreshes progress.json every few seconds; older means it died
QUEUED_HEARTBEAT_S = 60
# after a stop request, a run normally ends within seconds (between two images);
# if it is still alive after this long, it is terminated
STOP_GRACE_S = 60
FINISHED = ("completed", "failed", "stopped")


# ------------------------------------------------------------------------------
def check_input_dir(path, settings):
    """
    Validate a scale-image folder. Returns a dict with ok (bool), message and,
    if ok, n_images and a count per file extension.
    """
    path = (path or "").strip().strip('"')
    if not path:
        return {"ok": False, "message": "Enter the folder containing the scale images."}
    p = Path(path).expanduser()
    if not input_dir_allowed(p, settings):
        roots = ", ".join(settings["allowed_input_roots"])
        return {"ok": False, "message": f"Only folders inside {roots} can be used on this server."}
    if not p.is_dir():
        return {"ok": False, "message": f"Folder not found: {p}"}
    try:
        images = list_input_images(p)
    except (FileNotFoundError, ValueError) as err:
        return {"ok": False, "message": str(err)}
    except OSError as err:
        return {"ok": False, "message": f"Cannot read folder: {err}"}
    by_ext = collections.Counter(Path(f).suffix.lower() for f in images)
    try:
        duplicates = [[Path(f).name for f in group] for group in find_duplicate_images(images)]
    except OSError:
        duplicates = []
    return {"ok": True, "path": str(p.resolve()), "n_images": len(images), "by_ext": dict(by_ext),
            "duplicates": duplicates,
            "message": f"{len(images)} images (" + ", ".join(f"{n} {e}" for e, n in sorted(by_ext.items())) + ")"}


def parse_angles(text):
    """'0, 45 90' -> [0, 45, 90]; raises ValueError with a readable message."""
    parts = [p for p in text.replace(",", " ").split() if p]
    if not parts:
        raise ValueError("Enter at least one transect angle.")
    try:
        angles = [int(p) for p in parts]
    except ValueError:
        raise ValueError("Transect angles must be whole numbers of degrees, e.g. 0, 45, 90.") from None
    if any(not 0 <= a < 360 for a in angles):
        raise ValueError("Transect angles must be between 0 and 359 degrees.")
    if len(set(angles)) != len(angles):
        raise ValueError("Each transect angle can only be used once.")
    return angles


# ------------------------------------------------------------------------------
def start_run(input_dir, run_name, angles, settings, overlays=True, plot_dets=True,
              per_image_files=False, max_circuli=500, focus_retry=True, focus_low_threshold=0.1):
    """Start sscd.py in a new run folder as an independent process. Returns the run folder."""
    root = output_root(settings.get("output_root"))
    run_dir = reserve_run_dir(root, run_name or None)
    cmd = [
        sys.executable, str(REPO_DIR / "sscd.py"),
        "--img_dir", str(input_dir),
        "--run_dir", str(run_dir),
        "--output_root", str(root),
        "--transect_angles", *[str(a) for a in angles],
        "--overlays", str(bool(overlays)),
        "--plot_dets", str(bool(plot_dets)),
        "--dets_separate_files", str(bool(per_image_files)),
        "--transect_max_boxes", str(int(max_circuli)),
        "--focus_retry", str(bool(focus_retry)),
        "--focus_low_threshold", str(float(focus_low_threshold)),
    ]
    # detached from the GUI process: the run continues if the GUI is closed or restarted
    kwargs = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
    console = open(run_dir / "console.log", "w", encoding="utf-8")  # noqa: SIM115 - handed to the child
    try:
        subprocess.Popen(cmd, cwd=REPO_DIR, stdout=console, stderr=subprocess.STDOUT,  # noqa: S603
                         stdin=subprocess.DEVNULL, **kwargs)
    finally:
        console.close()
    return run_dir


def stop_run(run_dir):
    """Ask a run to stop (it finishes the current image, then stops and keeps its results)."""
    request_stop(run_dir)


# ------------------------------------------------------------------------------
def _tail(path, n=200):
    try:
        return "".join(collections.deque(open(path, encoding="utf-8", errors="replace"), maxlen=n))
    except OSError:
        return ""


def _mark_ended(run_dir, info, progress, status, error):
    """Record a run whose process ended without recording it itself (crash, kill)."""
    from sscd_libs.outputs import write_info
    from sscd_libs.runcontrol import _write_json_atomic

    now = datetime.now().isoformat(timespec="seconds")
    if info is not None:
        info.update(status=status, finished=now, error=error)
        write_info(Path(run_dir) / "run_info.json", info)
    if progress is not None:
        progress.update(status=status, updated=now)
        _write_json_atomic(Path(run_dir) / PROGRESS_FILE, progress)


def run_state(run_dir, root):
    """
    Everything the GUI shows about a run, read from its folder. Detects runs whose
    process died (status still 'running'/'queued') and records them as failed, and
    terminates runs that ignore a stop request for longer than STOP_GRACE_S.
    """
    run_dir = Path(run_dir)
    info = read_json(run_dir / "run_info.json")
    progress = read_json(run_dir / PROGRESS_FILE)
    # run_info.json is written first when a run ends, so a finished status there wins
    # over a progress.json that could not be updated in time
    status = (info or {}).get("status")
    if status not in FINISHED:
        status = (progress or {}).get("status") or status or "starting"
    stop_file = run_dir / "STOP"
    stop_requested = stop_file.exists()

    if status == "running" and not run_is_alive(root, run_dir):
        status = "stopped" if stop_requested else "failed"
        _mark_ended(run_dir, info, progress, status,
                    None if stop_requested else "The run's process ended unexpectedly (crashed or killed).")
    elif status == "starting" and "Traceback" in _tail(run_dir / "console.log", 60):
        status = "failed"  # sscd.py failed before it could record anything
        _mark_ended(run_dir, info, progress, status, "The run failed to start - see the console output.")
    elif status in ("queued", "starting"):
        updated = (progress or {}).get("updated") or (info or {}).get("created")
        age = (datetime.now() - datetime.fromisoformat(updated)).total_seconds() if updated else None
        started = datetime.fromtimestamp(run_dir.stat().st_ctime)
        if (age is not None and age > QUEUED_HEARTBEAT_S) or (
                age is None and (datetime.now() - started).total_seconds() > QUEUED_HEARTBEAT_S):
            status = "failed"
            _mark_ended(run_dir, info, progress, status,
                        "The run did not start or its process ended while waiting in the queue.")

    # stop fallback: still running long after the stop request -> terminate the process
    if stop_requested and status == "running":
        asked = datetime.fromtimestamp(stop_file.stat().st_mtime)
        pid = slot_owner_pid(root, run_dir)
        if (datetime.now() - asked).total_seconds() > STOP_GRACE_S and pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

    info = read_json(run_dir / "run_info.json") or info or {}
    progress = read_json(run_dir / PROGRESS_FILE) or progress or {}
    stages = [
        {"name": name, "label": label, **progress.get("stages", {}).get(name, {"done": 0, "total": 0})}
        for name, label in STAGES.items()
        if (name != "overlays" or info.get("parameters", {}).get("overlays"))
        and (name != "focus_retry" or name in progress.get("stages", {}))  # only when it ran
    ]
    return {
        "run_dir": str(run_dir),
        "name": run_dir.name,
        "status": status,
        "stop_requested": stop_requested and status not in FINISHED,
        "stage": progress.get("stage"),
        "stages": stages,
        "counts": progress.get("counts") or info.get("counts") or {},
        "warnings": progress.get("warnings") or info.get("warnings") or [],
        "error": info.get("error"),
        "input_dir": info.get("input_dir"),
        "created": info.get("created"),
        "runtime_min": info.get("runtime_min"),
        "log": _tail(run_dir / "sscd.log"),
        "console": _tail(run_dir / "console.log", 60),
    }


def recent_runs(root, n=10):
    """Most recent run folders under the output root, newest first (with status and counts)."""
    runs_dir = Path(root) / "runs"
    if not runs_dir.is_dir():
        return []
    rows = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        info = read_json(d / "run_info.json") or {}
        progress = read_json(d / PROGRESS_FILE) or {}
        status = info.get("status")
        if status not in FINISHED:
            status = progress.get("status") or status or "starting"
        counts = progress.get("counts") or info.get("counts") or {}
        rows.append({
            "run": d.name,
            "status": status,
            # runs made before the GUI existed only record 'started'
            "created": (info.get("created") or info.get("started") or "").replace("T", " "),
            "images": counts.get("scales"),
            "focus found": counts.get("focus_found"),
            "circuli": counts.get("circuli"),
            "run_dir": str(d),
        })
    rows.sort(key=lambda r: (r["created"], r["run"]), reverse=True)
    return rows[:n]


def wait_until_finished(run_dir, root, timeout_s=3600, poll_s=1.0):
    """Block until a run has finished (for scripts and tests). Returns its final state."""
    t0 = time.time()
    while True:
        state = run_state(run_dir, root)
        if state["status"] in FINISHED:
            return state
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Run {run_dir} did not finish within {timeout_s} s")
        time.sleep(poll_s)
