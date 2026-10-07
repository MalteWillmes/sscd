"""
Live control of an sscd.py run, shared by the pipeline and the GUI:

- progress.json in the run folder: current stage, images done / total per
  stage, and the warnings raised so far (written atomically, read by the GUI)
- stop requests: a STOP file in the run folder, checked between images
- run slots: at most N runs at a time per output root. A slot is an OS file
  lock held for the lifetime of the process, so a crashed or killed run can
  never keep a slot blocked.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

STAGES = {
    "convert": "Converting images",
    "focus": "Focus detection",
    "focus_retry": "Focus: second pass (scales without focus)",
    "transects": "Extracting transects",
    "circuli": "Circuli detection",
    "overlays": "Drawing overlays",
    # manual_focus.py apply: scales whose focus was set by hand after the run
    "manual_transects": "Manual focus: extracting transects",
    "manual_circuli": "Manual focus: circuli detection",
    "manual_overlays": "Manual focus: drawing overlays",
}
# stages shown only once a run has reached them
OPTIONAL_STAGES = ("focus_retry", "manual_transects", "manual_circuli", "manual_overlays")

PROGRESS_FILE = "progress.json"
# sscd.log lines carry the stage they were written in, e.g. "INFO (2026-10-07 14:02:11) [focus]: ...";
# "-" outside the stages. The GUI shows each stage's lines under its progress bar.
LOG_FORMAT = "%(levelname)s (%(asctime)s) [%(sscd_stage)s]: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def stage_log_formatter():
    """Formatter for sscd.log; use with RunControl.stage_filter on the same handler."""
    return logging.Formatter(LOG_FORMAT, LOG_DATEFMT, defaults={"sscd_stage": "-"})
STOP_FILE = "STOP"
SLOTS_DIR = ".run_slots"


class StopRequested(Exception):  # noqa: N818 - reads naturally as "raise StopRequested"
    """Raised inside the pipeline when a stop was requested for the run."""


# ------------------------------------------------------------------------------
def _write_json_atomic(path, data, attempts=10):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    # replace, so readers never see a half-written file. On Windows this fails while
    # another process (e.g. the GUI) has the file open for reading - which only
    # lasts milliseconds, so retry briefly.
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05)


def read_json(path):
    """Read a JSON file, or None if missing/unreadable (e.g. while being replaced)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def request_stop(run_dir):
    """Ask the run in `run_dir` to stop (checked between images)."""
    (Path(run_dir) / STOP_FILE).write_text(datetime.now().isoformat(timespec="seconds"))


# ------------------------------------------------------------------------------
class _WarningCollector(logging.Handler):
    """Collects WARNING+ log records (SSCD's QC warnings) for progress.json."""

    def __init__(self, control):
        super().__init__(level=logging.WARNING)
        self.control = control

    def emit(self, record):
        if record.name.startswith("tensorflow"):
            return
        self.control.warnings.append({"stage": self.control.stage, "message": record.getMessage().strip()})
        self.control.save()


class _StageTag(logging.Filter):
    """Adds the run's current stage to each log record (record.sscd_stage)."""

    def __init__(self, control):
        super().__init__()
        self.control = control

    def filter(self, record):
        record.sscd_stage = self.control.stage or "-"
        return True


class RunControl:
    """Progress reporting, warnings and stop checks for one run folder."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.stage = None
        self.stages = {}
        self.warnings = []
        self.counts = {}
        self.status = "queued"
        self._last_save = 0.0
        self.handler = _WarningCollector(self)
        self.stage_filter = _StageTag(self)

    @classmethod
    def resume(cls, run_dir):
        """Continue an existing run's progress record (stages, counts, warnings), e.g. for
        a step that adds to a finished run."""
        control = cls(run_dir)
        previous = read_json(control.run_dir / PROGRESS_FILE) or {}
        control.stages = previous.get("stages") or {}
        control.counts = previous.get("counts") or {}
        control.warnings = previous.get("warnings") or []
        return control

    # --- reporting ------------------------------------------------------------
    def save(self, force=True):
        now = time.monotonic()
        if not force and now - self._last_save < 0.5:
            return  # don't rewrite the file for every one of hundreds of transects
        self._last_save = now
        try:
            self._write()
        except OSError:
            # progress reporting is best-effort: never fail a run over it (the next
            # update writes the file again)
            pass

    def _write(self):
        _write_json_atomic(self.run_dir / PROGRESS_FILE, {
            "status": self.status,
            "stage": self.stage,
            "stages": self.stages,
            "counts": self.counts,
            "warnings": self.warnings,
            "updated": datetime.now().isoformat(timespec="seconds"),
        })

    def set_status(self, status):
        self.status = status
        self._last_save = time.monotonic()
        for _ in range(20):  # status changes matter: keep trying for a few seconds
            try:
                self._write()
                return
            except OSError:
                time.sleep(0.25)

    def end_stages(self):
        """Log messages and warnings after this belong to no stage (e.g. the summary)."""
        self.stage = None

    def start_stage(self, name, total):
        self.check_stop()
        self.stage = name
        self.stages[name] = {"label": STAGES.get(name, name), "done": 0, "total": total}
        self.save()

    def advance(self, done=None):
        """Mark one more item (or `done` items) of the current stage as done."""
        self.check_stop()
        stage = self.stages[self.stage]
        stage["done"] = stage["done"] + 1 if done is None else done
        self.save(force=stage["done"] >= stage["total"])

    def set_failed(self, name, n, what):
        """Record how many items of stage `name` failed, e.g. (12, 'no focus') - shown in the GUI."""
        if name in self.stages:
            self.stages[name].update(failed=int(n), failed_label=what)
            self.save(force=True)

    def progress_callback(self):
        """A callable(done, total) for functions that report their own progress."""
        def callback(done, total):
            self.stages[self.stage]["total"] = total
            self.advance(done)
        return callback

    # --- stopping -------------------------------------------------------------
    def stop_requested(self):
        return (self.run_dir / STOP_FILE).exists()

    def check_stop(self):
        if self.stop_requested():
            raise StopRequested("Run stopped on request")


# ------------------------------------------------------------------------------
def _try_lock(fh):
    """Non-blocking exclusive lock on an open file; True if acquired."""
    try:
        if sys.platform == "win32":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


class RunSlot:
    """
    One of `max_runs` slots under the output root, held while a run executes.
    The lock is released by the OS when the process ends, however it ends.
    """

    def __init__(self, output_root, max_runs):
        self.dir = Path(output_root) / SLOTS_DIR
        self.max_runs = max(1, int(max_runs))
        self.fh = None
        self.index = None

    def try_acquire(self, run_dir):
        self.dir.mkdir(parents=True, exist_ok=True)
        for i in range(self.max_runs):
            path = self.dir / f"slot-{i}.lock"
            fh = open(path, "a+", encoding="utf-8")  # noqa: SIM115 - kept open while holding the lock
            if _try_lock(fh):
                # record who holds it (informative only; the lock itself is the truth)
                with open(self.dir / f"slot-{i}.owner", "w", encoding="utf-8") as owner:
                    owner.write(f"{os.getpid()}\n{run_dir}\n")
                self.fh, self.index = fh, i
                return True
            fh.close()
        return False

    def wait(self, run_dir, control, poll_s=2.0, on_wait=None):
        """Wait for a free slot (checking for stop requests while queued)."""
        waited = False
        while not self.try_acquire(run_dir):
            if not waited and on_wait:
                on_wait()
            waited = True
            control.check_stop()
            control.save()  # heartbeat: shows the queued process is alive
            time.sleep(poll_s)


def _same_dir(a, b):
    """True if two paths are the same folder (also across 8.3 short names, case, symlinks)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def run_is_alive(output_root, run_dir, max_runs=16):
    """True if some process currently holds a slot for `run_dir`."""
    slots = Path(output_root) / SLOTS_DIR
    for i in range(max_runs):
        owner = slots / f"slot-{i}.owner"
        info = read_text_lines(owner)
        if len(info) >= 2 and _same_dir(info[1], run_dir):
            with open(slots / f"slot-{i}.lock", "a+", encoding="utf-8") as fh:
                if _try_lock(fh):
                    return False  # lock was free: the owner process is gone
                return True
    return False


def slot_owner_pid(output_root, run_dir, max_runs=16):
    """PID of the process holding a slot for `run_dir`, if any (see run_is_alive)."""
    slots = Path(output_root) / SLOTS_DIR
    for i in range(max_runs):
        info = read_text_lines(slots / f"slot-{i}.owner")
        if len(info) >= 2 and _same_dir(info[1], run_dir):
            return int(info[0])
    return None


def read_text_lines(path):
    try:
        return Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
