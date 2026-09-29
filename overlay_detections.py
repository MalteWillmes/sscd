"""
Overlay circuli detections back onto the original scale images.

Run after sscd.py, on one of its run folders. For every scale with a detected
focus it draws the focus box, the outline of each radial transect and a tick
across the transect at each detected circulus (positions from the run's
results/circuli.csv). Nothing is re-detected.

Usage:

uv run python overlay_detections.py --latest
uv run python overlay_detections.py --run_dir "<output root>/runs/2026-09-29_1412_N-Esk-2018"

Writes <run folder>/overlays/<scale>_overlay.jpg (re-running overwrites them).
"""

# import built-in modules
import argparse
import json
import logging
from pathlib import Path

# import installed packages
from PIL import Image

# import local modules
from sscd_libs.data_processing import list_input_images, to_8bit_rgb
from sscd_libs.outputs import latest_run, open_run
from sscd_libs.overlay import draw_overlay, focus_by_scale, read_results_csv

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
_originals = {}  # input dir -> {scale id: original image path}


def scale_image(run, input_dir, scale_id):
    """
    The scale image to draw on: the run's jpeg copy in work/scales, or - if work/
    was deleted - the original input image, converted the same way sscd.py does.
    """
    work_copy = run.scales / f"{scale_id}.jpg"
    if work_copy.exists():
        return work_copy
    if input_dir not in _originals:
        try:
            _originals[input_dir] = {Path(p).stem: p for p in list_input_images(input_dir)}
        except (FileNotFoundError, OSError):
            _originals[input_dir] = {}
    original = _originals[input_dir].get(scale_id)
    if original is None:
        raise FileNotFoundError(
            f"Scale image '{scale_id}' not found: neither {work_copy} nor an original in the run's "
            f"input folder {input_dir}."
        )
    with Image.open(original) as im:
        im.load()  # read the pixels before the file is closed
        return to_8bit_rgb(im)


# ------------------------------------------------------------------------------
def main():
    args_parser = argparse.ArgumentParser(
        description="Overlay circuli detections from an sscd.py run back onto the original scale images."
    )
    which = args_parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--run_dir", help="run folder of a finished sscd.py run")
    which.add_argument("--latest", action="store_true",
                       help="use the most recent run under the output root")
    args_parser.add_argument(
        "--output_root", default=None,
        help="output root to look for --latest in (default: $SSCD_OUTPUT_ROOT, else ~/sscd_outputs)",
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
    angles = info["parameters"]["transect_angles"]

    focus = focus_by_scale(read_results_csv(run.focus_csv))
    circuli = read_results_csv(run.circuli_csv)
    summary = read_results_csv(run.summary_csv)

    run.overlays.mkdir(exist_ok=True)
    for scale_id, focus_bbox in focus.items():
        scale_circuli = circuli[circuli["scale_id"] == scale_id]
        draw_overlay(scale_image(run, info["input_dir"], scale_id), focus_bbox, angles, scale_circuli,
                     run.overlays / f"{scale_id}_overlay.jpg", label_every=args.label_every)
        logger.info("%s: %d circuli", scale_id, len(scale_circuli))

    no_focus = sorted(summary.loc[~summary["focus_found"], "scale_id"])
    if no_focus:
        logger.warning("No focus detected (no overlay): %s", ", ".join(no_focus))
    logger.info("Overlays written to %s", run.overlays)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
