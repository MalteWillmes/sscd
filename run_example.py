"""
Run SSCD end-to-end on the bundled example scales.

Downloads the trained weights on first use (~790 MB), then runs the full
pipeline (focus detection -> transects -> circuli detection -> spacings) on
data/example_scales. Optionally also runs the evaluation example against the
annotated transects in data/eval_example.

Usage (from any directory):

    uv run python run_example.py             # detection example
    uv run python run_example.py --eval      # ... plus the evaluation example
    uv run python run_example.py --overlay   # ... plus detections drawn on the scales
    uv run python run_example.py --no_plots  # skip detection plot images (faster)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent


def run(script, *args):
    # Each stage runs in its own process so it gets its own log file and
    # releases all TensorFlow memory when it finishes.
    cmd = [sys.executable, script, *args]
    print("\n>>> " + " ".join(cmd) + "\n", flush=True)
    result = subprocess.run(cmd, cwd=REPO_DIR)  # noqa: S603 - fixed scripts, no shell
    if result.returncode != 0:
        # the actual error (traceback) was printed by the script itself, just above
        sys.exit(f"\n{script} failed (exit code {result.returncode}) - see its error message above.")


def main():
    parser = argparse.ArgumentParser(description="Run SSCD on the bundled example data.")
    parser.add_argument(
        "--output_dir",
        default=str(REPO_DIR / "SSCD_temp_outputs"),
        help="where to write outputs (default: SSCD_temp_outputs/ in the repo)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="also run the evaluation example (data/eval_example)",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="also draw the circuli detections back onto the scale images (overlay_detections.py)",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="skip writing images with detections drawn on them",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    detection_dir = output_dir / "example"
    eval_dir = output_dir / "eval_example"

    # sscd.py and the weight fetcher resolve ./data relative to the repo root
    os.chdir(REPO_DIR)
    from sscd_libs.fetch import fetch_weights

    fetch_weights()

    run(
        "sscd.py",
        "--img_dir", "./data/example_scales",
        "--output_dir", str(detection_dir),
        "--transect_angles", "0", "45", "90", "135", "180",
        "--plot_dets", str(not args.no_plots),
    )

    if args.overlay:
        run("overlay_detections.py", "--output_dir", str(detection_dir))

    if args.eval:
        run(
            "eval_detector.py",
            "--img_dir", "./data/eval_example/imgs/",
            "--anns_dir", "./data/eval_example/anns/",
            "--dets_csv", "./data/eval_example/detections.csv",
            "--iou_threshould", "0.5",
            "--output_dir", str(eval_dir),
            "--plot_dets_vs_anns", str(not args.no_plots),
            "--sep_plots", "True",
        )

    print("\nDone. Key outputs:")
    print(f"  circuli spacings : {detection_dir / 'detections' / 'circuli' / 'circuli_spacings.csv'}")
    print(f"  focus detections : {detection_dir / 'detections' / 'focus' / 'detections.csv'}")
    print(f"  run log          : {detection_dir / 'log_sscd_detection.log'}")
    if args.overlay:
        print(f"  overlays         : {detection_dir / 'overlays'}")
    if args.eval:
        print(f"  evaluation       : {eval_dir / 'evaluation_results.txt'}")


if __name__ == "__main__":
    main()
