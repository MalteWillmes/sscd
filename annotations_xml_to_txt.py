"""
Convert Pascal VOC (LabelImg) XML annotation files into the whitespace-separated
txt format used by the evaluator: one line per box, "label xmin ymin xmax ymax".

Usage (from the repository root):

uv run python annotations_xml_to_txt.py \
    --anns_dir "<some_path>/anns_xml" \
    --output_dir "<some_path>/anns_txt"
"""

# import built-in modules
import argparse
import glob
import os
from pathlib import Path

# import local modules
from sscd_libs.data_processing import pascal_to_evaltxt


# ------------------------------------------------------------------------------
def main():
    args_parser = argparse.ArgumentParser(
        description="Convert Pascal VOC XML annotations to evaluator txt files"
    )
    args_parser.add_argument("--anns_dir", required=True, help="directory with .xml annotation files")
    args_parser.add_argument("--output_dir", required=True, help="directory to write .txt files to")
    args = args_parser.parse_args()

    ann_filepaths = glob.glob(os.path.join(glob.escape(args.anns_dir), "*.xml"))
    if not ann_filepaths:
        raise FileNotFoundError(f"No XML annotation files found in {args.anns_dir}")

    # --- Annotations (ground truth bounding boxes): convert from Pascal VOC xml to txt files
    os.makedirs(args.output_dir, exist_ok=True)
    for ann_filepath in ann_filepaths:
        pascal_to_evaltxt(args.anns_dir, Path(ann_filepath).stem, args.output_dir)

    print(f"Converted {len(ann_filepaths)} annotation files to {args.output_dir}")


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
