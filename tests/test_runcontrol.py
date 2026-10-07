"""Run control: progress file, stop requests, run slots (queue)."""

import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from conftest import REPO_DIR
from sscd_libs.runcontrol import (
    RunControl,
    RunSlot,
    StopRequested,
    read_json,
    request_stop,
    run_is_alive,
    slot_owner_pid,
)


def test_progress_file_tracks_stages_counts_and_warnings(tmp_path):
    rc = RunControl(tmp_path)
    rc.set_status("running")
    rc.start_stage("focus", 3)
    rc.advance()
    rc.progress_callback()(3, 3)
    rc.counts.update(scales=3)
    rc.warnings.append("No focus found")
    rc.save()
    p = read_json(tmp_path / "progress.json")
    assert p["status"] == "running"
    assert p["stage"] == "focus"
    assert p["stages"]["focus"] == {"label": "Focus detection", "done": 3, "total": 3}
    assert p["counts"] == {"scales": 3}
    assert p["warnings"] == ["No focus found"]

    rc.set_failed("focus", 1, "no focus")
    rc.set_failed("circuli", 5, "not started")          # unknown stage: ignored
    p = read_json(tmp_path / "progress.json")
    assert p["stages"]["focus"]["failed"] == 1 and p["stages"]["focus"]["failed_label"] == "no focus"
    assert "circuli" not in p["stages"]


def test_stop_request_is_raised_at_the_next_checkpoint(tmp_path):
    rc = RunControl(tmp_path)
    rc.start_stage("focus", 10)
    rc.advance()
    request_stop(tmp_path)
    with pytest.raises(StopRequested):
        rc.advance()


def test_slots_limit_concurrent_runs(tmp_path):
    first, second = RunSlot(tmp_path, 1), RunSlot(tmp_path, 1)
    assert first.try_acquire(tmp_path / "run-a")
    assert not second.try_acquire(tmp_path / "run-b")     # the single slot is taken
    assert RunSlot(tmp_path, 2).try_acquire(tmp_path / "run-b")  # a second slot is free


def test_a_killed_run_frees_its_slot(tmp_path):
    """The slot is an OS lock held by the run's process: killing the process frees it."""
    run_dir = tmp_path / "runs" / "crashing"
    run_dir.mkdir(parents=True)
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, r"{REPO_DIR}")
        from sscd_libs.runcontrol import RunSlot
        slot = RunSlot(r"{tmp_path}", 1)   # kept referenced, as sscd.py does: it holds the lock
        assert slot.try_acquire(r"{run_dir}")
        print("locked", flush=True)
        time.sleep(60)
    """)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        assert run_is_alive(tmp_path, run_dir)
        assert not RunSlot(tmp_path, 1).try_acquire(tmp_path / "other")
        # the recorded owner is the process holding the lock (on Windows, a venv's python.exe
        # is a launcher whose child runs the code) - kill it, as the GUI's stop fallback does
        os.kill(slot_owner_pid(tmp_path, run_dir), signal.SIGTERM)
    finally:
        holder.kill()
        holder.wait()
    time.sleep(0.5)
    assert not run_is_alive(tmp_path, run_dir)
    assert RunSlot(tmp_path, 1).try_acquire(tmp_path / "other")


def test_log_lines_and_warnings_carry_their_stage(tmp_path):
    import logging

    from sscd_libs.jobs import stage_logs
    from sscd_libs.runcontrol import stage_log_formatter

    rc = RunControl(tmp_path)
    log = logging.getLogger("test_stage_tags")
    log.propagate = False
    log.setLevel(logging.INFO)
    handler = logging.FileHandler(tmp_path / "sscd.log", encoding="utf-8")
    handler.setFormatter(stage_log_formatter())
    handler.addFilter(rc.stage_filter)
    log.addHandler(handler)
    log.addHandler(rc.handler)
    try:
        log.warning("2 duplicate images")              # before any stage
        rc.start_stage("focus", 3)
        log.info("Starting detection in 3 images")
        log.warning("No focus found in 1 scale image(s):\n\n\tTummel 60300")
        rc.end_stages()
        log.info("Summary")
    finally:
        handler.close()
        log.removeHandler(handler)
        log.removeHandler(rc.handler)

    assert [(w["stage"], w["message"].split(":")[0]) for w in rc.warnings] == [
        (None, "2 duplicate images"), ("focus", "No focus found in 1 scale image(s)")]
    logs = stage_logs(tmp_path / "sscd.log")
    assert "): Starting detection" in logs["focus"] and "[focus]" not in logs["focus"]
    assert "\tTummel 60300" in logs["focus"]             # continuation lines stay with their message
    assert "Summary" in logs["-"] and "duplicate" in logs["-"]
