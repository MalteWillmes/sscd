"""
Map circuli detections from transect images back onto the scale image, and
draw them there.
"""

import logging
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from sscd_libs.data_processing import list_input_images, to_8bit_rgb, transect_geometry, transect_to_image

logger = logging.getLogger(__name__)

# one colour per transect (colour-blind friendly, readable on the pale scale background)
TRANSECT_COLOURS = [
    (230, 25, 75), (0, 130, 200), (60, 180, 75), (245, 130, 48), (145, 30, 180),
    (240, 50, 230), (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128),
]
FOCUS_COLOUR = (255, 215, 0)


# ------------------------------------------------------------------------------
def split_transect_ids(transect_ids):
    """Series of '<scale id>_<angle>' -> (scale ids, angles in degrees).
    The scale id may itself contain '_'."""
    parts = transect_ids.astype(str).str.rsplit("_", n=1)
    return parts.str[0], parts.str[1].astype(int)


def circuli_on_scale(circuli, focus_by_scale):
    """
    Add scale-image positions to circuli detections.

    Args
    ----
        circuli: DataFrame of circuli detections with columns transect_id,
            x_center, y_center (transect-image pixels); rows with detections only
        focus_by_scale: {scale_id: focus bbox dict with xmin, ymin, xmax, ymax}

    Returns
    -------
        copy of `circuli` with scale_id, angle_deg, x_px and y_px (centre of the
        detection box on the scale image) and dist_from_focus_px (distance from
        the focus centre along the transect) added
    """
    out = circuli.copy()
    out["scale_id"], out["angle_deg"] = split_transect_ids(out["transect_id"])
    out["x_px"] = float("nan")
    out["y_px"] = float("nan")
    for (scale_id, angle), rows in out.groupby(["scale_id", "angle_deg"]):
        geom = transect_geometry(focus_by_scale[scale_id], angle)
        x, y = transect_to_image(geom, rows["x_center"], rows["y_center"])
        out.loc[rows.index, "x_px"] = x.round(2)
        out.loc[rows.index, "y_px"] = y.round(2)
    # the transect starts on the line through the focus centre (perpendicular to
    # it), so distance along the transect == distance from the focus
    out["dist_from_focus_px"] = out["x_center"]
    return out


# ------------------------------------------------------------------------------
def draw_overlay(scale_img, focus_bbox, angles, circuli, out_path, label_every=0):
    """
    Draw focus, transect outlines and a tick across the transect at each circulus.

    Args
    ----
        scale_img: the scale image - a path (jpeg as processed by sscd.py) or a PIL image
        focus_bbox: dict with xmin, ymin, xmax, ymax of the focus
        angles: transect angles (degrees) that were extracted for this scale
        circuli: this scale's rows of results/circuli.csv
        out_path: where to write the overlay jpeg
        label_every: number every n-th circulus (0 = no numbers)
    """
    if isinstance(scale_img, Image.Image):
        im = scale_img.convert("RGB")
    else:
        with Image.open(scale_img) as im:
            im = im.convert("RGB")
    width, height = im.size
    draw = ImageDraw.Draw(im)

    # line widths and text size relative to the image size
    tick_width = max(2, round(max(width, height) / 1300))
    outline_width = max(1, tick_width // 2)
    font = ImageFont.load_default(size=max(12, round(max(width, height) / 110)))
    halo = {"stroke_width": 2, "stroke_fill": (255, 255, 255)}

    for i, angle_deg in enumerate(sorted(angles)):
        geom = transect_geometry(focus_bbox, angle_deg, width, height)
        if geom.length < 1:
            continue  # focus on the image border: this transect was not extracted
        colour = TRANSECT_COLOURS[i % len(TRANSECT_COLOURS)]

        # transect outline (corners in transect-image coordinates: u along, v across)
        corners = [(0, 0), (geom.length, 0), (geom.length, geom.width), (0, geom.width)]
        draw.polygon([transect_to_image(geom, u, v) for u, v in corners], outline=colour, width=outline_width)

        # angle label just beyond the far end of the transect, kept inside the image
        lx, ly = transect_to_image(geom, geom.length, geom.width / 2)
        label = f"{angle_deg}\N{DEGREE SIGN}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        lx = min(max(lx - (right - left) / 2, 0), width - (right - left))
        ly = min(max(ly - (bottom - top) / 2, 0), height - (bottom - top))
        draw.text((lx, ly), label, fill=colour, font=font, **halo)

        # a tick across the transect at each circulus, spanning its detection box
        for det in circuli[circuli["angle_deg"] == angle_deg].itertuples():
            p0 = transect_to_image(geom, det.dist_from_focus_px, det.ymin)
            p1 = transect_to_image(geom, det.dist_from_focus_px, det.ymax)
            draw.line([p0, p1], fill=colour, width=tick_width)
            if label_every and det.circulus_nr % label_every == 0:
                tx, ty = transect_to_image(geom, det.dist_from_focus_px, -geom.width * 0.6)
                draw.text((tx, ty), str(det.circulus_nr), fill=colour, font=font, anchor="mm", **halo)

    # focus box on top
    draw.rectangle(
        [focus_bbox["xmin"], focus_bbox["ymin"], focus_bbox["xmax"], focus_bbox["ymax"]],
        outline=FOCUS_COLOUR, width=tick_width,
    )
    im.save(out_path, "JPEG", quality=90)


def focus_by_scale(focus):
    """results/focus.csv (rows with a focus box) -> {scale_id: bbox dict}. Manually set
    foci (manual_focus.py) have a box but no detector score."""
    return {row["scale_id"]: row for row in focus.dropna(subset=["xmin"]).to_dict("records")}


def read_results_csv(path):
    """Read a results table keeping ids as text (numeric names keep leading zeros)."""
    return pd.read_csv(path, dtype={"scale_id": str, "transect_id": str, "img_id": str})


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


def draw_run_overlays(run, angles, input_dir, label_every=0, progress=None, scale_ids=None):
    """
    Draw <run>/overlays/<scale>_overlay.jpg for every scale with a focus, from the
    run's results. Returns the number of overlays drawn.

    Args
    ----
        run: RunPaths of the run
        angles: the transect angles of the run (run_info.json parameters)
        input_dir: the run's input folder (fallback if work/ was deleted)
        label_every: number every n-th circulus (0 = no numbers)
        progress: optional callable(done, total), called after each scale
        scale_ids: only draw these scales (default: all with a focus)
    """
    focus = focus_by_scale(read_results_csv(run.focus_csv))
    if scale_ids is not None:
        focus = {k: v for k, v in focus.items() if k in set(scale_ids)}
    circuli = read_results_csv(run.circuli_csv)
    run.overlays.mkdir(exist_ok=True)
    for i, (scale_id, focus_bbox) in enumerate(focus.items(), 1):
        scale_circuli = circuli[circuli["scale_id"] == scale_id]
        draw_overlay(scale_image(run, input_dir, scale_id), focus_bbox, angles, scale_circuli,
                     run.overlays / f"{scale_id}_overlay.jpg", label_every=label_every)
        logger.info("Overlay %s: %d circuli", scale_id, len(scale_circuli))
        if progress:
            progress(i, len(focus))
    return len(focus)
