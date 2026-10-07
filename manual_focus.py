"""
Manual focus for scales of a finished run where SSCD found no focus.

    # add the scales whose focus was set by hand (results/focus_corrections.csv, usually
    # written by the web app) to the run's results
    uv run python manual_focus.py apply --run_dir "<output root>/runs/2026-10-07_1000_VI"

    # export the manual foci as Pascal VOC annotations (training data / ground truth)
    uv run python manual_focus.py export --run_dir "<run folder>" [--include_detected]

`apply` cuts the transects and detects the circuli for each manually set focus, with
the run's own parameters, and adds them to results/focus.csv, circuli.csv and
scales_summary.csv (focus_method = manual) and to the overlays. Scales marked as
having no usable focus get a review_note in scales_summary.csv. Nothing already in
the results is changed: only scales without a focus can be corrected. See
sscd_libs/manual_focus.py for the corrections file.
"""

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import sscd
from sscd_libs import manual_focus as mf
from sscd_libs.detection import plot_detections
from sscd_libs.helpers import unpack_for_string
from sscd_libs.outputs import open_run, output_root, write_info
from sscd_libs.overlay import draw_run_overlays, read_results_csv
from sscd_libs.runcontrol import STOP_FILE, RunControl, RunSlot, StopRequested, read_json
from sscd_libs.settings import load_settings

logger = logging.getLogger(__name__)


def _output_root_of(paths):
    # a run made by the GUI or with --run_name lives in <output root>/runs/<run>
    return paths.root.parent.parent if paths.root.parent.name == "runs" else output_root()


def _setup_logging(paths):
    handlers = (logging.StreamHandler(), logging.FileHandler(paths.log, mode="a", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s (%(asctime)s): %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers, force=True)


def apply_corrections(paths, info, control):
    """Add the pending manual foci (and no-focus notes) to the run's results. Returns a
    short record of what was applied."""
    args = {**info["parameters"], "img_dir": info["input_dir"]}
    decisions = mf.pending(paths)
    focus_old = read_results_csv(paths.focus_csv)
    # only scales without a focus can be corrected: never change existing results
    already = decisions["scale_id"].isin(set(focus_old.dropna(subset=["xmin"])["scale_id"]))
    if already.any():
        logger.warning("Skipping corrections for scales that already have a focus: %s",
                       ", ".join(decisions.loc[already, "scale_id"]))
        decisions = decisions[~already]
    to_focus = decisions[decisions["decision"] == "focus"]
    excluded = decisions[decisions["decision"] == "no_focus"]

    box = mf.focus_box_size(paths)
    new_focus = mf.manual_focus_rows(to_focus, box).reindex(columns=sscd.FOCUS_COLUMNS)
    logger.info("Manual focus for %d scale(s) (focus box %d x %d px, the run's median); "
                "%d scale(s) marked as having no usable focus", len(new_focus), box[0], box[1], len(excluded))

    transect_ids, circuli = [], pd.DataFrame(columns=sscd.CIRCULI_COLUMNS)
    if len(new_focus):
        for scale_id in new_focus["scale_id"]:
            mf.scale_jpg(paths, args["img_dir"], scale_id)
        # cut into a staging folder, so only the new transects are detected
        staging = paths.focus_plots / "manual_transects"
        if staging.exists():
            shutil.rmtree(staging)
        transect_ids, circuli, _ = sscd.transects_and_circuli(
            args, paths, control, new_focus, staging, stages=("manual_transects", "manual_circuli"))
        paths.transects.mkdir(parents=True, exist_ok=True)
        for f in staging.glob("*.jpg"):
            f.replace(paths.transects / f.name)
        shutil.rmtree(staging, ignore_errors=True)

        for r in new_focus.to_dict("records"):
            (paths.no_focus / f"{r['scale_id']}.jpeg").unlink(missing_ok=True)  # it has a focus now
            if args["plot_dets"]:
                with Image.open(paths.scales / f"{r['scale_id']}.jpg") as im:
                    plot_detections(np.asarray(im.convert("RGB")), pd.DataFrame([r]), str(paths.focus_plots),
                                    plot_conf=False, fig_w=65, fig_h=60, img_id=r["scale_id"])

    # --- results tables: add rows, replace the summary rows of the corrected scales
    focus_all = pd.concat([focus_old, new_focus], ignore_index=True) if len(new_focus) else focus_old
    focus_all.reindex(columns=sscd.FOCUS_COLUMNS).to_csv(paths.focus_csv, index=False)
    if len(circuli):
        circuli_all = pd.concat([read_results_csv(paths.circuli_csv), circuli], ignore_index=True)
        circuli_all.reindex(columns=sscd.CIRCULI_COLUMNS).to_csv(paths.circuli_csv, index=False)

    summary = read_results_csv(paths.summary_csv)
    order = {scale_id: i for i, scale_id in enumerate(summary["scale_id"])}
    if "review_note" not in summary:
        summary["review_note"] = None
    summary["review_note"] = summary["review_note"].astype(object)
    if len(new_focus):
        new_rows = sscd.scales_summary(new_focus["scale_id"].tolist(), new_focus, transect_ids, circuli)
        new_rows["review_note"] = "focus set by hand"
        summary = pd.concat([summary[~summary["scale_id"].isin(new_rows["scale_id"])], new_rows],
                            ignore_index=True)
        summary = summary.sort_values("scale_id", key=lambda s: s.map(order)).reset_index(drop=True)
    for r in excluded.to_dict("records"):
        note = r["note"] if isinstance(r["note"], str) and r["note"] else "no reason given"
        summary.loc[summary["scale_id"] == r["scale_id"], "review_note"] = f"no usable focus: {note}"
    summary.to_csv(paths.summary_csv, index=False)

    # --- overlays of the new scales
    if args.get("overlays") and len(new_focus):
        control.start_stage("manual_overlays", len(new_focus))
        draw_run_overlays(paths, args["transect_angles"], args["img_dir"],
                          progress=control.progress_callback(), scale_ids=new_focus["scale_id"].tolist())

    mf.mark_applied(paths, decisions["scale_id"].tolist())
    control.counts["focus_found"] = control.counts.get("focus_found", 0) + len(new_focus)
    if len(new_focus):
        logger.warning("Focus set by hand for %d scale(s) (focus_method = manual):\n\n\t%s\n",
                       len(new_focus), unpack_for_string(new_focus["scale_id"]))
    return {"applied": datetime.now().isoformat(timespec="seconds"),
            "manual_focus": new_focus["scale_id"].tolist(), "no_usable_focus": excluded["scale_id"].tolist(),
            "focus_box_px": list(box), "circuli": len(circuli)}


def apply(run_dir):
    paths = open_run(run_dir)
    info = read_json(paths.info)
    if info.get("status") != "completed":
        raise RuntimeError(f"The run is {info.get('status')}; corrections can only be applied to a completed run.")
    if not len(mf.pending(paths)):
        print("No corrections to apply.")
        return

    _setup_logging(paths)
    (paths.root / STOP_FILE).unlink(missing_ok=True)  # an old stop request must not stop this
    control = RunControl.resume(paths.root)
    logging.getLogger().addHandler(control.handler)
    logger.info("Applying focus corrections (%s)", paths.focus_corrections)

    def finish(**fields):
        # the run itself stays completed; only what was added changes
        info.update(status="completed", counts=control.counts, warnings=control.warnings, **fields)
        write_info(paths.info, info)
        control.set_status("completed")
        (paths.root / STOP_FILE).unlink(missing_ok=True)

    try:
        slot = RunSlot(_output_root_of(paths), load_settings()["max_concurrent_runs"])
        info.update(status="queued")
        write_info(paths.info, info)
        control.set_status("queued")
        slot.wait(paths.root, control, on_wait=lambda: logger.info("Waiting for a free run slot..."))
        info.update(status="running")
        write_info(paths.info, info)
        control.set_status("running")
        record = apply_corrections(paths, info, control)
    except StopRequested:
        logger.warning("Applying the focus corrections was stopped - the results are unchanged")
        shutil.rmtree(paths.focus_plots / "manual_transects", ignore_errors=True)
        finish()
        return
    except BaseException as err:
        logger.exception("Applying the focus corrections failed - the results are unchanged")
        finish(manual_focus_error=f"{type(err).__name__}: {err}")
        raise
    info.setdefault("manual_focus", []).append(record)
    info.pop("manual_focus_error", None)
    finish()
    logger.info("Focus corrections applied: %d scale(s) added, %d marked as having no usable focus",
                len(record["manual_focus"]), len(record["no_usable_focus"]))
    logging.shutdown()


def export(run_dir, include_detected=False):
    paths = open_run(run_dir)
    info = read_json(paths.info)
    out, n = mf.export_annotations(paths, info["input_dir"], include_detected=include_detected)
    print(f"{n} focus annotation(s) written to {out}")


def main():
    parser = argparse.ArgumentParser(description="Manual focus for scales where SSCD found no focus.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_apply = sub.add_parser("apply", help="add the manually set foci to the run's results")
    p_apply.add_argument("--run_dir", required=True, help="the run folder")
    p_export = sub.add_parser("export", help="export the manual foci as Pascal VOC annotations")
    p_export.add_argument("--run_dir", required=True, help="the run folder")
    p_export.add_argument("--include_detected", action="store_true",
                          help="also export the foci found by the detector (unverified)")
    args = parser.parse_args()
    if args.command == "apply":
        apply(Path(args.run_dir))
    else:
        export(Path(args.run_dir), args.include_detected)


if __name__ == "__main__":
    main()
