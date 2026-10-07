"""
Manual focus for the scales of a run where SSCD found no focus.

The user marks the focus centre of each such scale (in the web app, or by writing the
corrections file) or marks it as having no usable focus. manual_focus.py then cuts
the transects and detects the circuli for those scales and adds them to the run's
results, flagged as focus_method = manual.

    results/focus_corrections.csv   one row per reviewed scale:
        scale_id, decision (focus | no_focus), x_px, y_px (focus centre, pixels of the
        scale image), note, decided, applied (time, empty until applied)

The focus box of a manual focus - which sets the transect width (half its shorter
side) - has the median size of the boxes the detector found in the same run.

The corrections can also be exported as Pascal VOC annotations (training data for
the focus detector, and ground truth for eval_detector.py).
"""

import os
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sscd_libs.data_processing import list_input_images, tiff_to_jpg

COLUMNS = ["scale_id", "decision", "x_px", "y_px", "note", "decided", "applied"]
DECISIONS = ("focus", "no_focus")
MANUAL = "manual"
# focus box side as a fraction of the image width, if the run has no detected focus
# to take the size from (example scales: ~50 x 60 px on 3840 x 2748 px)
FALLBACK_BOX_FRACTION = 0.014


def _now():
    return datetime.now().isoformat(timespec="seconds")


# --- the corrections file -----------------------------------------------------
def read_corrections(paths):
    """The run's focus corrections (empty table if there are none yet)."""
    if not paths.focus_corrections.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(paths.focus_corrections, dtype={"scale_id": str, "decision": str, "note": str,
                                                      "decided": str, "applied": str})
    return df.reindex(columns=COLUMNS)


def _write_corrections(paths, df):
    paths.focus_corrections.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.focus_corrections.with_suffix(".tmp")
    df.reindex(columns=COLUMNS).to_csv(tmp, index=False)
    os.replace(tmp, paths.focus_corrections)


def _applied(row):
    return isinstance(row.get("applied"), str) and row["applied"] != ""


def save_decision(paths, scale_id, decision, x_px=None, y_px=None, note=""):
    """Record (or change) the decision for one scale. Applied decisions are final."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    if decision == "focus" and (x_px is None or y_px is None):
        raise ValueError("a focus needs its position (x_px, y_px)")
    df = read_corrections(paths)
    old = df[df["scale_id"] == scale_id]
    if len(old) and _applied(old.iloc[0].to_dict()):
        raise ValueError(f"The correction for '{scale_id}' has already been applied.")
    row = {"scale_id": scale_id, "decision": decision,
           "x_px": int(round(x_px)) if decision == "focus" else None,
           "y_px": int(round(y_px)) if decision == "focus" else None,
           "note": note or "", "decided": _now(), "applied": ""}
    df = pd.concat([df[df["scale_id"] != scale_id], pd.DataFrame([row])], ignore_index=True)
    _write_corrections(paths, df)


def remove_decision(paths, scale_id):
    """Undo a decision that has not been applied yet."""
    df = read_corrections(paths)
    old = df[df["scale_id"] == scale_id]
    if len(old) and _applied(old.iloc[0].to_dict()):
        raise ValueError(f"The correction for '{scale_id}' has already been applied.")
    _write_corrections(paths, df[df["scale_id"] != scale_id])


def mark_applied(paths, scale_ids):
    df = read_corrections(paths)
    df.loc[df["scale_id"].isin(scale_ids), "applied"] = _now()
    _write_corrections(paths, df)


def pending(paths):
    """Decisions not applied yet."""
    df = read_corrections(paths)
    return df[~df.apply(lambda r: _applied(r.to_dict()), axis=1)] if len(df) else df


# --- which scales to review ---------------------------------------------------
def review_queue(paths):
    """
    The scales of the run where SSCD found no focus, in the order of the run, each with
    its review state: pending (no decision yet), focus / no_focus (decided, not applied),
    or applied.
    """
    if not paths.summary_csv.exists():
        return []
    summary = pd.read_csv(paths.summary_csv, dtype={"scale_id": str, "focus_method": str})
    missed = summary[(~summary["focus_found"].astype(bool)) | (summary["focus_method"] == MANUAL)]
    decisions = {r["scale_id"]: r for r in read_corrections(paths).to_dict("records")}
    queue = []
    for scale_id in missed["scale_id"]:
        d = decisions.get(scale_id)
        state = "pending" if d is None else ("applied" if _applied(d) else d["decision"])
        queue.append({"scale_id": scale_id, "state": state, "decision": None if d is None else d["decision"],
                      "x_px": None if d is None else d["x_px"], "y_px": None if d is None else d["y_px"],
                      "note": "" if d is None or pd.isna(d["note"]) else d["note"]})
    return queue


# --- the focus box --------------------------------------------------------------
def focus_box_size(paths, image_width=None):
    """
    (width, height) of a manual focus box: the median size of the focus boxes the
    detector found in this run (same microscope and magnification), else a fixed
    fraction of the image width.
    """
    if paths.focus_csv.exists():
        focus = pd.read_csv(paths.focus_csv, dtype={"scale_id": str, "focus_method": str})
        detected = focus[(focus["focus_method"] != MANUAL) & focus["xmin"].notna()]
        if len(detected):
            return (int(round((detected["xmax"] - detected["xmin"]).median())),
                    int(round((detected["ymax"] - detected["ymin"]).median())))
    side = int(round(FALLBACK_BOX_FRACTION * (image_width or 3840)))
    return side, side


def manual_focus_rows(decisions, box_size):
    """Focus rows (as in results/focus.csv) for decisions with a focus position. The box
    is centred exactly on the clicked point (it may extend past the image border)."""
    w, h = box_size
    hw, hh = max(w // 2, 1), max(h // 2, 1)
    rows = []
    for d in decisions.to_dict("records"):
        if d["decision"] != "focus":
            continue
        x, y = int(d["x_px"]), int(d["y_px"])
        rows.append({"scale_id": d["scale_id"], "class_name": "focus", "score": np.nan,
                     "focus_method": MANUAL, "n_focus_boxes": 0,
                     "xmin": x - hw, "ymin": y - hh, "xmax": x + hw, "ymax": y + hh,
                     "x_px": float(x), "y_px": float(y)})
    return pd.DataFrame(rows)


# --- scale images -----------------------------------------------------------------
def scale_jpg(paths, input_dir, scale_id):
    """The run's 8-bit jpeg copy of a scale (work/scales), recreated from the original
    input image if work/ was deleted."""
    jpg = paths.scales / f"{scale_id}.jpg"
    if not jpg.exists():
        originals = {Path(p).stem: p for p in list_input_images(input_dir)}
        if scale_id not in originals:
            raise FileNotFoundError(f"Scale image '{scale_id}' not found in {jpg.parent} or {input_dir}")
        jpg.parent.mkdir(parents=True, exist_ok=True)
        tiff_to_jpg(originals[scale_id], str(jpg))
    return jpg


# --- training export ----------------------------------------------------------------
def _voc_xml(filename, size, box, label="focus"):
    w, h = size
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = "images"
    ET.SubElement(ann, "filename").text = filename
    ET.SubElement(ET.SubElement(ann, "source"), "database").text = "SSCD manual focus"
    s = ET.SubElement(ann, "size")
    for k, v in (("width", w), ("height", h), ("depth", 3)):
        ET.SubElement(s, k).text = str(v)
    ET.SubElement(ann, "segmented").text = "0"
    obj = ET.SubElement(ann, "object")
    for k, v in (("name", label), ("pose", "Unspecified"), ("truncated", "0"), ("difficult", "0")):
        ET.SubElement(obj, k).text = v
    bb = ET.SubElement(obj, "bndbox")
    # clipped to the image, as annotation tools do
    for k, v, hi in (("xmin", box[0], w), ("ymin", box[1], h), ("xmax", box[2], w), ("ymax", box[3], h)):
        ET.SubElement(bb, k).text = str(int(min(max(v, 0), hi)))
    ET.indent(ann)
    return ET.ElementTree(ann)


def export_annotations(paths, input_dir, include_detected=False):
    """
    Write the run's manual foci as Pascal VOC annotations, with the scale images, to
    <run>/annotations/focus/{images,xml}/ plus a manifest.csv. With include_detected,
    the foci found by the detector are added too, marked 'detected (unverified)' in the
    manifest - check them before training on them. An earlier export is replaced.

    Returns (export folder, number of annotations).
    """
    out = paths.root / "annotations" / "focus"
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "xml").mkdir()

    decisions = read_corrections(paths)
    focus = (pd.read_csv(paths.focus_csv, dtype={"scale_id": str, "focus_method": str})
             if paths.focus_csv.exists() else pd.DataFrame(columns=["scale_id", "focus_method"]))
    rows = manual_focus_rows(decisions, focus_box_size(paths))
    rows = rows.assign(source="manual")
    if include_detected:
        detected = focus[(focus["focus_method"] != MANUAL) & focus["xmin"].notna()]
        detected = detected[~detected["scale_id"].isin(rows["scale_id"] if len(rows) else [])]
        rows = pd.concat([rows, detected.assign(source="detected (unverified)")], ignore_index=True)

    manifest = []
    for r in rows.to_dict("records"):
        jpg = scale_jpg(paths, input_dir, r["scale_id"])
        with Image.open(jpg) as im:
            size = im.size
        name = f"{r['scale_id']}.jpg"
        shutil.copyfile(jpg, out / "images" / name)
        box = (r["xmin"], r["ymin"], r["xmax"], r["ymax"])
        _voc_xml(name, size, box).write(out / "xml" / f"{r['scale_id']}.xml", encoding="utf-8")
        manifest.append({"scale_id": r["scale_id"], "source": r["source"],
                         "focus_method": r.get("focus_method"),
                         "xmin": int(box[0]), "ymin": int(box[1]), "xmax": int(box[2]), "ymax": int(box[3])})
    pd.DataFrame(manifest, columns=["scale_id", "source", "focus_method", "xmin", "ymin", "xmax", "ymax"]
                 ).to_csv(out / "manifest.csv", index=False)
    return out, len(manifest)
