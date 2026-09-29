"""
Second focus-detection pass, only for scales where the normal pass found no focus.

  A. padded:        the scale image padded (with its own background colour) to the
                    aspect ratio of the images the focus detector was trained on
                    (3840 x 2748), and detected with the normal threshold. The
                    detector squeezes every image into a square; wider images (e.g.
                    16:9) are squeezed more than anything it was trained on.
  B. low_threshold: the best box above a lower score threshold, for scales still
                    without a focus.

Each pass keeps only the single most confident box per scale. Scales rescued this way
are flagged (focus_method) so they can be checked or excluded.
"""

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

# height / width of the images the focus detector was trained on
TRAINING_ASPECT = 2748 / 3840


def pad_to_training_aspect(im):
    """
    Pad an image (PIL) to the training aspect ratio, centred, with its background
    colour (median of the outermost pixels). Returns (padded image, (dx, dy)) where
    (dx, dy) is the position of the original image inside the padded one.
    """
    im = im.convert("RGB")
    w, h = im.size
    if h / w < TRAINING_ASPECT:          # too wide: pad top and bottom
        new_w, new_h = w, round(w * TRAINING_ASPECT)
    else:                                # too tall: pad left and right
        new_w, new_h = round(h / TRAINING_ASPECT), h
    dx, dy = (new_w - w) // 2, (new_h - h) // 2
    a = np.asarray(im)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    fill = tuple(int(v) for v in np.median(border, axis=0))
    padded = Image.new("RGB", (new_w, new_h), fill)
    padded.paste(im, (dx, dy))
    return padded, (dx, dy)


def _best_box(dets):
    """The most confident detection per image."""
    dets = dets.dropna(subset=["score"])
    return dets.sort_values("score", ascending=False).groupby("img_id").head(1)


def retry_focus(scale_ids, scales_dir, work_dir, detect_focus, low_threshold, progress=None):
    """
    Look again for the focus in `scale_ids` (images <scales_dir>/<id>.jpg).

    Args
    ----
        detect_focus: callable(img_dir, no_det_dir, yolo_score_threshold, progress)
            running the focus detector (as sscd_libs.detection.detect) on a folder
        low_threshold: score threshold of pass B (None or 0: skip pass B)
        progress: optional callable(done, total), called per image in each pass

    Returns
    -------
        DataFrame of the rescued focus detections - one row per scale, in the
        original image's coordinates, as detect() returns them - with an extra
        column focus_method ('padded' or 'low_threshold').
    """
    work_dir = Path(work_dir)
    rescued = []
    remaining = list(scale_ids)

    # --- A. padded to the training aspect ratio, normal threshold
    padded_dir = work_dir / "padded"
    padded_dir.mkdir(parents=True, exist_ok=True)
    offsets, sizes = {}, {}
    for scale_id in remaining:
        with Image.open(Path(scales_dir) / f"{scale_id}.jpg") as im:
            sizes[scale_id] = im.size
            padded, offsets[scale_id] = pad_to_training_aspect(im)
        padded.save(padded_dir / f"{scale_id}.jpg", quality=95)
    found = _best_box(detect_focus(str(padded_dir), str(work_dir / "padded_none"), None, progress))
    if len(found):
        found = found.copy()
        dx = found["img_id"].map(lambda s: offsets[s][0])
        dy = found["img_id"].map(lambda s: offsets[s][1])
        w = found["img_id"].map(lambda s: sizes[s][0])
        h = found["img_id"].map(lambda s: sizes[s][1])
        found["xmin"] = (found["xmin"] - dx).clip(lower=0).combine(w, min)
        found["xmax"] = (found["xmax"] - dx).clip(lower=0).combine(w, min)
        found["ymin"] = (found["ymin"] - dy).clip(lower=0).combine(h, min)
        found["ymax"] = (found["ymax"] - dy).clip(lower=0).combine(h, min)
        found["img_prop"] = (found["xmax"] - found["xmin"]) * (found["ymax"] - found["ymin"]) / (w * h)
        found["focus_method"] = "padded"
        rescued.append(found)
        remaining = [s for s in remaining if s not in set(found["img_id"])]

    # --- B. lower score threshold, on the original images
    if remaining and low_threshold:
        low_dir = work_dir / "low_threshold"
        low_dir.mkdir(parents=True, exist_ok=True)
        for scale_id in remaining:
            shutil.copy(Path(scales_dir) / f"{scale_id}.jpg", low_dir / f"{scale_id}.jpg")
        found = _best_box(detect_focus(str(low_dir), str(work_dir / "low_threshold_none"), low_threshold, progress))
        if len(found):
            found = found.assign(focus_method="low_threshold")
            rescued.append(found)

    if not rescued:
        return pd.DataFrame()
    result = pd.concat(rescued, ignore_index=True)
    result["detection_nr"] = 1
    return result
