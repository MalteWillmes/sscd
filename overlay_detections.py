"""
Overlay circuli detections back onto the original scale images.

Run after sscd.py, on its output directory. For every scale with a detected
focus it draws the focus box, the outline of each radial transect and a tick
across the transect at each detected circulus, and it writes the circuli
positions in scale-image coordinates to a CSV file. Nothing is re-detected.

Usage (from the repository root):

uv run python overlay_detections.py --output_dir "./SSCD_temp_outputs/example"

Outputs, in <output_dir>/overlays/:
    <scale>_overlay.jpg     scale image with focus, transects and circuli ticks
    circuli_on_scale.csv    one row per circulus: position on the scale image
                            (x_px, y_px), distance from the focus along the
                            transect (dist_from_focus_px) and spacing_px
"""

# import built-in modules
import argparse
import logging
from pathlib import Path

# import installed packages
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# import local modules
from sscd_libs.data_processing import transect_geometry, transect_to_image
from sscd_libs.helpers import prepare_output_dir

logger = logging.getLogger(__name__)

# one colour per transect (colour-blind friendly, readable on the pale scale background)
TRANSECT_COLOURS = [
    (230, 25, 75), (0, 130, 200), (60, 180, 75), (245, 130, 48), (145, 30, 180),
    (240, 50, 230), (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128),
]
FOCUS_COLOUR = (255, 215, 0)


# ------------------------------------------------------------------------------
def overlay_scale(scale_img_path, focus_bbox, transects, out_path, label_every=0):
    """
    Draw focus, transect outlines and circuli ticks on one scale image.

    Args
    ----
        scale_img_path: path to the scale jpeg (sscd.py's jpegs/scales/)
        focus_bbox: dict-like with xmin, ymin, xmax, ymax of the focus
        transects: list of (angle_deg, circuli DataFrame) - the DataFrame holds
            this transect's detections (transect-image pixel coordinates)
        out_path: where to write the overlay jpeg
        label_every: number every n-th circulus (0 = no numbers)

    Returns
    -------
        DataFrame with the circuli positions in scale-image coordinates
    """
    with Image.open(scale_img_path) as im:
        im = im.convert("RGB")
    width, height = im.size
    draw = ImageDraw.Draw(im)

    # line widths and text size relative to the image size
    tick_width = max(2, round(max(width, height) / 1300))
    outline_width = max(1, tick_width // 2)
    font = ImageFont.load_default(size=max(12, round(max(width, height) / 110)))

    rows = []
    for i, (angle_deg, dets) in enumerate(transects):
        colour = TRANSECT_COLOURS[i % len(TRANSECT_COLOURS)]
        geom = transect_geometry(focus_bbox, angle_deg, width, height)

        # transect outline (corners in transect-image coordinates u along, v across)
        corners = [(0, 0), (geom.length, 0), (geom.length, geom.width), (0, geom.width)]
        outline = [transect_to_image(geom, u, v) for u, v in corners]
        draw.polygon(outline, outline=colour, width=outline_width)

        # angle label just beyond the far end of the transect, kept inside the image
        lx, ly = transect_to_image(geom, geom.length, geom.width / 2)
        label = f"{angle_deg}\N{DEGREE SIGN}"
        l, t, r, b = draw.textbbox((0, 0), label, font=font)
        lx = min(max(lx - (r - l) / 2, 0), width - (r - l))
        ly = min(max(ly - (b - t) / 2, 0), height - (b - t))
        draw.text((lx, ly), label, fill=colour, font=font, stroke_width=2, stroke_fill=(255, 255, 255))

        # a tick across the transect at each circulus, spanning its detection box
        for det in dets.itertuples():
            p0 = transect_to_image(geom, det.x_center, det.ymin)
            p1 = transect_to_image(geom, det.x_center, det.ymax)
            draw.line([p0, p1], fill=colour, width=tick_width)

            if label_every and det.circulus_nr % label_every == 0:
                tx, ty = transect_to_image(geom, det.x_center, -geom.width * 0.6)
                draw.text((tx, ty), str(det.circulus_nr), fill=colour, font=font, anchor="mm",
                          stroke_width=2, stroke_fill=(255, 255, 255))

            x, y = transect_to_image(geom, det.x_center, det.y_center)
            rows.append({
                "transect_id": det.img_id,
                "angle_deg": angle_deg,
                "circulus_nr": det.circulus_nr,
                "score": det.score,
                "x_px": round(x, 2),
                "y_px": round(y, 2),
                # the transect starts on the line through the focus centre, so
                # distance along the transect == distance from the focus
                "dist_from_focus_px": det.x_center,
                "spacing_px": det.spacing_px,
            })

    # focus box on top
    draw.rectangle(
        [focus_bbox["xmin"], focus_bbox["ymin"], focus_bbox["xmax"], focus_bbox["ymax"]],
        outline=FOCUS_COLOUR, width=tick_width,
    )

    im.save(out_path, "JPEG", quality=90)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
def main():
    args_parser = argparse.ArgumentParser(
        description="Overlay circuli detections from an sscd.py run back onto the original scale images."
    )
    args_parser.add_argument(
        "--output_dir", required=True,
        help="output directory of a finished sscd.py run (overlays are written to its overlays/ subfolder)",
    )
    args_parser.add_argument(
        "--label_every", type=int, default=0,
        help="number every n-th circulus on the overlay (default 0: no numbers)",
    )
    args = args_parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    run_dir = Path(args.output_dir)
    focus_csv = run_dir / "detections" / "focus" / "detections.csv"
    circuli_csv = run_dir / "detections" / "circuli" / "circuli_spacings.csv"
    scales_dir = run_dir / "jpegs" / "scales"
    for required in (focus_csv, scales_dir):
        if not required.exists():
            raise FileNotFoundError(f"{required} not found - is '{run_dir}' the output directory of an sscd.py run?")

    focus = pd.read_csv(focus_csv, dtype={"img_id": str})
    no_focus = sorted(focus.loc[focus.score.isna(), "img_id"])
    focus = focus.dropna(subset=["score"])

    # circuli per transect; transects without circuli appear as a single empty row
    if circuli_csv.exists():
        circuli = pd.read_csv(circuli_csv, dtype={"img_id": str})
    else:
        circuli = pd.DataFrame(columns=["img_id", "score"])
    # transect ids are '<scale id>_<angle>' (the scale id may itself contain '_')
    id_parts = circuli["img_id"].astype(str).str.rsplit("_", n=1)
    circuli["scale_id"] = id_parts.str[0]
    circuli["angle_deg"] = id_parts.str[1].astype(int)

    overlays_dir = run_dir / "overlays"
    prepare_output_dir(overlays_dir)

    tables = []
    for focus_bbox in focus.to_dict("records"):
        scale_id = focus_bbox["img_id"]
        scale_circuli = circuli[circuli.scale_id == scale_id]
        transects = [
            (int(angle), dets.dropna(subset=["score"]).astype({"circulus_nr": int}))
            for angle, dets in scale_circuli.groupby("angle_deg", sort=True)
        ]
        table = overlay_scale(
            scales_dir / f"{scale_id}.jpg", focus_bbox, transects,
            overlays_dir / f"{scale_id}_overlay.jpg", label_every=args.label_every,
        )
        table.insert(0, "scale_id", scale_id)
        tables.append(table)
        logger.info("%s: %d transects, %d circuli", scale_id, len(transects), len(table))

    if no_focus:
        logger.warning("No focus detected (no overlay): %s", ", ".join(no_focus))

    columns = ["scale_id", "transect_id", "angle_deg", "circulus_nr", "score",
               "x_px", "y_px", "dist_from_focus_px", "spacing_px"]
    tables = [t for t in tables if len(t)]
    on_scale = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=columns)
    on_scale[columns].to_csv(overlays_dir / "circuli_on_scale.csv", index=False)
    logger.info("Overlays and circuli_on_scale.csv written to %s", overlays_dir)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
