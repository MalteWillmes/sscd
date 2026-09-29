"""
Overlay circuli detections back onto the original scale images.

Run after sscd.py, on one of its run folders (sscd.py --overlays True, the
GUI's default, already does this at the end of a run; use this script to add
overlays to a run made without them, or to redraw them e.g. with numbers).
For every scale with a detected focus it draws the focus box, the outline of
each radial transect and a tick across the transect at each detected circulus
(positions from the run's results/circuli.csv). Nothing is re-detected.

Usage:

uv run python overlay_detections.py --latest
uv run python overlay_detections.py --run_dir "<output root>/runs/2026-09-29_1412_N-Esk-2018"

Writes <run folder>/overlays/<scale>_overlay.jpg (re-running overwrites them).
"""

# import built-in modules
import argparse
import json
import logging

# import local modules
from sscd_libs.outputs import latest_run, open_run
from sscd_libs.overlay import draw_run_overlays, read_results_csv

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
def main():
    args_parser = argparse.ArgumentParser(
        description="Overlay circuli detections from an sscd.py run back onto the original scale images."
    )
    which = args_parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--run_dir", help="run folder of a finished sscd.py run")
    which.add_argument("--latest", action="store_true",
                       help="use the most recent completed run under the output root")
    args_parser.add_argument(
        "--output_root", default=None,
        help="output root to look for --latest in (default: $SSCD_OUTPUT_ROOT, else C:\\sscd_outputs on Windows, ~/sscd_outputs elsewhere)",
    )
    args_parser.add_argument(
        "--label_every", type=int, default=0,
        help="number every n-th circulus on the overlay (default 0: no numbers)",
    )
    args = args_parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    run = latest_run(args.output_root) if args.latest else open_run(args.run_dir)
    logger.info("Run folder: %s", run.root)
    info = json.loads(run.info.read_text(encoding="utf-8"))
    if info.get("status") != "completed":
        raise RuntimeError(f"Run {run.root.name} did not complete (status: {info.get('status')}).")

    draw_run_overlays(run, info["parameters"]["transect_angles"], info["input_dir"],
                      label_every=args.label_every)

    summary = read_results_csv(run.summary_csv)
    no_focus = sorted(summary.loc[~summary["focus_found"], "scale_id"])
    if no_focus:
        logger.warning("No focus detected (no overlay): %s", ", ".join(no_focus))
    logger.info("Overlays written to %s", run.overlays)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
