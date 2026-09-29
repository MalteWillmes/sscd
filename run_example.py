"""
Run SSCD end-to-end on the bundled example scales.

Downloads the trained weights on first use (~790 MB), then runs the full
pipeline (focus detection -> transects -> circuli detection -> spacings) on
data/example_scales, into a new run folder <output root>/runs/<date>_<time>_example.
Optionally also draws the detections onto the scales, and runs the evaluation
example against the annotated transects in data/eval_example.

Usage (from any directory):

    uv run python run_example.py             # detection example
    uv run python run_example.py --overlay   # ... plus detections drawn on the scales
    uv run python run_example.py --eval      # ... plus the evaluation example
    uv run python run_example.py --no_plots  # skip QC plot images (faster)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent


def run(script, *args, reserved_dir=None):
    # Each stage runs in its own process so it gets its own log file and
    # releases all TensorFlow memory when it finishes.
    cmd = [sys.executable, script, *args]
    print("\n>>> " + " ".join(cmd) + "\n", flush=True)
    result = subprocess.run(cmd, cwd=REPO_DIR)  # noqa: S603 - fixed scripts, no shell
    if result.returncode != 0:
        # don't leave behind the empty folder reserved for a run that never started
        if reserved_dir is not None and reserved_dir.is_dir() and not any(reserved_dir.iterdir()):
            reserved_dir.rmdir()
        # the actual error (traceback) was printed by the script itself, just above
        sys.exit(f"\n{script} failed (exit code {result.returncode}) - see its error message above.")


def main():
    parser = argparse.ArgumentParser(description="Run SSCD on the bundled example data.")
    parser.add_argument(
        "--output_root",
        default=None,
        help="root folder for SSCD outputs (default: $SSCD_OUTPUT_ROOT, else C:\\sscd_outputs on Windows, ~/sscd_outputs elsewhere)",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="also draw the circuli detections back onto the scale images (overlay_detections.py)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="also run the evaluation example (data/eval_example)",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="skip writing QC images with detections drawn on them",
    )
    args = parser.parse_args()

    # the weight fetcher and scripts resolve paths relative to the repo root
    os.chdir(REPO_DIR)
    from sscd_libs.fetch import fetch_weights
    from sscd_libs.outputs import reserve_eval_dir, reserve_run_dir

    fetch_weights()

    run_dir = reserve_run_dir(args.output_root, "example")
    run(
        "sscd.py",
        "--img_dir", "./data/example_scales",
        "--run_dir", str(run_dir),
        "--transect_angles", "0", "45", "90", "135", "180",
        "--plot_dets", str(not args.no_plots),
        reserved_dir=run_dir,
    )

    if args.overlay:
        run("overlay_detections.py", "--run_dir", str(run_dir))

    if args.eval:
        eval_dir = reserve_eval_dir(args.output_root, "example")
        run(
            "eval_detector.py",
            "--img_dir", "./data/eval_example/imgs/",
            "--anns_dir", "./data/eval_example/anns/",
            "--dets_csv", "./data/eval_example/detections.csv",
            "--iou_threshould", "0.5",
            "--eval_dir", str(eval_dir),
            "--plot_dets_vs_anns", str(not args.no_plots),
            "--sep_plots", "True",
            reserved_dir=eval_dir,
        )

    print("\nDone. Key outputs:")
    print(f"  run folder : {run_dir}")
    print(f"  results    : {run_dir / 'results'}  (circuli.csv, focus.csv, scales_summary.csv)")
    if args.overlay:
        print(f"  overlays   : {run_dir / 'overlays'}")
    if args.eval:
        print(f"  evaluation : {eval_dir / 'results' / 'evaluation_results.txt'}")


if __name__ == "__main__":
    main()
