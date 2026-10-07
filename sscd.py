# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:54:54 2021

@author: Bruno Caneco

Purpose: Main script to run the Salmon Scale Circuli detector via a command-line interface. 

Brief pipeline description:
    (i) Convert scale image's format to jepgs
    (ii) Detect focus location on each scale image
    (iii) Extract scale transect images at different angles from detected focus
    (iv) Detect circuli bands locations on transect images
    (v) Calculate circuli spacings in each transect
    
    
Each run is written to a new folder <output root>/runs/<date>_<time>[_<run name>]
(output root: --output_root, else $SSCD_OUTPUT_ROOT, else C:\\sscd_outputs / ~/sscd_outputs); see
sscd_libs/outputs.py for the folder layout.

Usage:

uv run python sscd.py \
   --img_dir "./data/example_scales" \
   --run_name "example" \
   --transect_angles 0 45 90 135 180
"""

# import built-in modules
import argparse
import os
from datetime import datetime
from pathlib import Path
from time import time

# import installed/3rd-party modules
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image


# import local modules
from sscd_libs.helpers import (
    REPO_DIR,
    boolean_string,
    unpack_for_string,
    )
from sscd_libs.outputs import code_version, environment, new_run, output_root, write_info
from sscd_libs.overlay import circuli_on_scale, draw_run_overlays, focus_by_scale, split_transect_ids
from sscd_libs.runcontrol import RunControl, RunSlot, StopRequested, stage_log_formatter
from sscd_libs.settings import load_settings

from sscd_libs.detection import detect, plot_detections
from sscd_libs.focus_retry import best_focus_box, retry_focus

from sscd_libs.data_processing import (
    images_tiff_to_jpeg,
    find_duplicate_images,
    list_input_images,
    get_transects
    )


logger = logging.getLogger(__name__)

# model files, resolved relative to the repository so sscd.py runs from any directory
DATA_DIR = os.path.join(REPO_DIR, "data")
WEIGHTS = {
    "focus": os.path.join(DATA_DIR, "yoloV3_checkpoints", "focus_detector", "yolov3_train_190.tf"),
    "circuli": os.path.join(DATA_DIR, "yoloV3_checkpoints", "circuli_detector", "yolov3_train_22.tf"),
    }
CLASS_FILES = {
    "focus": os.path.join(DATA_DIR, "scales_label.names"),
    "circuli": os.path.join(DATA_DIR, "scale_transects_label.names"),
    }




# ------------------------------------------------------------------------------
def focus_checks(focus_dets_df):
        
    # QA for circuli spacings - flag up detection anomalies
    
    # issues counter
    issues = 0
    
    # Warning when the detector found more than one focus box in an image (only the most
    # confident box is used; the image may contain several scales or a false detection)
    multiple = focus_dets_df[focus_dets_df["n_focus_boxes"] > 1]
    if len(multiple) > 0:
        logger.warning("... More than one focus detected in %d image(s) - only the most confident box is "
                       "used. Check them (n_focus_boxes in focus.csv; all boxes are drawn in "
                       "work/focus). Do the images contain several scales?\n\n\t%s\n",
                       len(multiple), unpack_for_string(
                           f"{r.img_id} ({r.n_focus_boxes} boxes, kept score {r.score:.2f})"
                           for r in multiple.itertuples()))
        issues += 1

    # Warning when detection boxes cover more than a given proportion of the image
    det_prop_tolerance = 0.2  # 1/5 of the image (arbitrary at this stage. May need tunning with usage)
    large_dets = focus_dets_df[["img_id", "detection_nr", "score",
                                         "img_prop"]][focus_dets_df.img_prop > det_prop_tolerance]
    if len(large_dets) > 0:
                      
         logger.warning("... The size of some of the focus detections are unusually large "
                        f"(covering >{det_prop_tolerance*100}% of the image size):"                      
                       '\n\n\t'+ large_dets.to_string().replace('\n', '\n\t') +
                       "\n\n\tCheck focus detection images as something might have gone wrong "
                       "(e.g. unsuitable images; detection deterioration)\n\n")
         issues += 1
         
    
    # report if no issues found
    if issues == 0:
        logger.info("... no apparent issues")
         
    return 0





# ------------------------------------------------------------------------------
def circuli_checks(circuli_dets_df, circuli_max_boxes):
    
    # QA for circuli spacings - flag up detection anomalies
      
    # issues counter
    issues = 0
    
    # Cases with spacings greater than a given tolerance
    spacing_tolerance = 250   # 250 pixels
    large_spacings = circuli_dets_df[["img_id", "circulus_nr", "score",
                                         "spacing_px"]][circuli_dets_df.spacing_px > spacing_tolerance]
    
    if len(large_spacings) > 0:
        logger.warning(f"... Some of the extracted spacings are abnormally large (>{spacing_tolerance} pixels), "
                      "likely due to misdetections of debris on the scale's periphery:" 
                       '\n\n\t'+ large_spacings.to_string().replace('\n', '\n\t') +
                       "\n\n\tCheck circuli detection images to confirm debris misdetection."
                       "\n\tNOTE: Large spacings due to debris misdetections must be removed on post-processing\n\n")
        issues += 1
    
    
    # Warning when more than 20% of spacings are over the large spacing tolerance 
    # NOTE: 20% is arbitrary at this point. Should be tunned with more usage and better grasp of common problems
    prop_large_spacings = large_spacings.shape[0]/circuli_dets_df.shape[0]   
    if prop_large_spacings > 0.2:
        logger.warning("... Over 20% of extracted spacings are abnormally large. "
                       "Check circuli detection images as something might have gone wrong "
                       "(e.g. unsuitable images; detection deterioration)\n\n")
        issues += 1
    
    
    # Warning when more than 30% of spacings are <= 1 pixel
    # NOTE: 30% is arbitrary at this point. Should be tunned with more usage and better grasp of common problems
    prop_tiny_spacings = circuli_dets_df.query("spacing_px <= 1").shape[0]/circuli_dets_df.shape[0]   
    if prop_tiny_spacings > 0.3:
        logger.warning("...Over 30% of extracted spacings are abnormally small. "
                       "Check circuli detection images as something might have gone wrong "
                       "(e.g. unsuitable images; detection deterioration)\n\n")
        issues += 1

    
    # Warning when maximum number of detections in one image 
    hit_max_num_dets = circuli_dets_df[["img_id", "circulus_nr"]][circuli_dets_df.circulus_nr == circuli_max_boxes]    
    if len(hit_max_num_dets) > 0:
        logger.warning(f"... Current max number of detections permitted per transect ({circuli_max_boxes} boxes) "
                        "has been reached in the following images"
                        '\n\n\t'+ hit_max_num_dets.to_string().replace('\n', '\n\t') +
                        "\n\n\tCheck circuli detection images for visual inspection. "
                        "Cap on max number of circuli detections may need to be adjusted "
                        "to accomodate older individuals\n\n"
                        )   
        issues += 1
        
    
    # Warning when detection boxes cover more than a given proportion of the image
    det_prop_tolerance = 0.20  # 1/5 of the image (arbitrary at this stage. May need tunning with usage)
    large_dets = circuli_dets_df[["img_id", "circulus_nr", "score",
                                         "img_prop"]][circuli_dets_df.img_prop > det_prop_tolerance]
    
    if len(large_dets) > 0:
        
         logger.warning("... The size of some of the circuli detections are unusually large "
                        f"(covering >{det_prop_tolerance*100}% of the image size):"                      
                       '\n\n\t'+ large_dets.to_string().replace('\n', '\n\t') +
                       "\n\n\tCheck circuli detection images as something might have gone wrong "
                       "(e.g. unsuitable images; detection deterioration)\n\n")
         issues += 1
         
    
    # report if no issues found
    if issues == 0:
        logger.info("... no apparent issues")

    return 0
         

    
    
    





# ------------------------------------------------------------------------------
CIRCULI_COLUMNS = ["scale_id", "transect_id", "angle_deg", "circulus_nr", "class_name", "score",
                   "spacing_px", "dist_from_focus_px", "x_px", "y_px", "xmin", "ymin", "xmax", "ymax"]
FOCUS_COLUMNS = ["scale_id", "class_name", "score", "focus_method", "n_focus_boxes", "xmin", "ymin", "xmax", "ymax", "x_px", "y_px"]

FOCUS_MODEL = {"input_width": 1376, "input_height": 1376, "yolo_score_threshold": 0.5, "yolo_max_boxes": 100}
CIRCULI_MODEL = {"input_width": 3904, "input_height": 64, "yolo_score_threshold": 0.3}


def scales_summary(scale_ids, focus, transect_ids, circuli):
    """One row per input scale: focus found?, transects, circuli, median spacing."""
    t_scale, t_angle = (split_transect_ids(pd.Series(transect_ids, dtype=str))
                        if len(transect_ids) else (pd.Series(dtype=str), pd.Series(dtype=int)))
    focus_score = dict(zip(focus["scale_id"], focus["score"], strict=True))
    focus_method = dict(zip(focus["scale_id"], focus["focus_method"], strict=True))
    n_focus_boxes = dict(zip(focus["scale_id"], focus["n_focus_boxes"], strict=True))
    rows = []
    for scale_id in scale_ids:
        c = circuli[circuli["scale_id"] == scale_id]
        angles = sorted(t_angle[t_scale == scale_id])
        empty = [a for a in angles if not (c["angle_deg"] == a).any()]
        rows.append({
            "scale_id": scale_id,
            "focus_found": scale_id in focus_score,
            "focus_score": focus_score.get(scale_id),
            # standard, or found in the second pass: padded / low_threshold
            "focus_method": focus_method.get(scale_id),
            # boxes found by the detector; > 1: only the most confident one is used
            "n_focus_boxes": n_focus_boxes.get(scale_id),
            "n_transects": len(angles),
            "total_n_circuli": len(c),
            # per transect with at least one circulus (transects without circuli are not
            # counted); empty if there are none
            "mean_n_circuli": (round(len(c) / (len(angles) - len(empty)), 2)
                               if len(angles) > len(empty) else None),
            # empty if no spacing exists (e.g. only one circulus per transect)
            "median_spacing_px": (round(c["spacing_px"].dropna().median(), 2)
                                  if c["spacing_px"].notna().any() else None),
            "transects_without_circuli": ";".join(str(a) for a in empty),
            # manual focus review: why a scale has no focus (see manual_focus.py)
            "review_note": None,
        })
    return pd.DataFrame(rows)


def second_focus_pass(args, paths, control, focus_found, missed):
    """Look again for the focus in the `missed` scales; returns the updated
    (focus_found, missed). Rescued scales keep focus_method 'padded'/'low_threshold'."""
    control.start_stage("focus_retry", len(missed))
    logger.info("Second focus pass for %d scale(s) without a focus", len(missed))

    def detect_focus(img_dir, no_det_dir, threshold, progress):
        return detect(img_dir=img_dir, weights=WEIGHTS["focus"], classes_file=CLASS_FILES["focus"],
                      no_det_dir=no_det_dir, input_width=FOCUS_MODEL["input_width"],
                      input_height=FOCUS_MODEL["input_height"],
                      yolo_score_threshold=threshold or FOCUS_MODEL["yolo_score_threshold"],
                      yolo_max_boxes=5, report_no_detections=False, progress=progress)

    rescued = retry_focus(missed, paths.scales, paths.focus_retry, detect_focus,
                          args["focus_low_threshold"], progress=control.progress_callback())
    if len(rescued) == 0:
        return focus_found, missed

    for row in rescued.itertuples():
        # no longer a scale without detections
        (paths.no_focus / f"{row.img_id}.jpeg").unlink(missing_ok=True)
        if args["plot_dets"]:
            with Image.open(paths.scales / f"{row.img_id}.jpg") as im:
                plot_detections(np.asarray(im.convert("RGB")), rescued[rescued["img_id"] == row.img_id],
                                str(paths.focus_plots), fig_w=65, fig_h=60)
    logger.warning(
        "Focus found only in the second pass for %d scale(s) - check them (focus_method in focus.csv and "
        "scales_summary.csv):\n\n\t%s\n", len(rescued),
        unpack_for_string(f"{r.img_id} ({r.focus_method}, score {r.score:.2f})" for r in rescued.itertuples()))
    focus_found = pd.concat([focus_found, rescued], ignore_index=True)
    return focus_found, [s for s in missed if s not in set(rescued["img_id"])]


def transects_and_circuli(args, paths, control, focus, transects_dir, stages=("transects", "circuli")):
    """
    Steps 4-8 for the scales in `focus` (rows as in results/focus.csv): cut their radial
    transects from work/scales into `transects_dir`, detect the circuli on them, calculate
    the spacings and place each circulus on its scale image. The transect and circuli
    counts in `control` are added to, so a later step (manual_focus.py) adds to a run's totals.

    Returns (transect_ids, circuli, circuli_summary_stats).
    """
    angles = args["transect_angles"]
    transects_dir = Path(transects_dir)
    transects_dir.mkdir(parents=True, exist_ok=True)
    circuli = pd.DataFrame(columns=CIRCULI_COLUMNS)

    ## --- 4. Generate transect images off each focus
    logger.info("Extracting images of radial transects from focus in %d scales", len(focus))
    control.start_stage(stages[0], len(focus))
    transects_before = control.counts.get("transects", 0)
    n_transects = 0
    for focus_bbx in tqdm(focus.to_dict("records"), ascii=True, ncols=120):
        get_transects(focus_bbox=focus_bbx,
                      transect_degrees=angles,
                      img_filepath=str(paths.scales / (focus_bbx["scale_id"] + ".jpg")),
                      output_dir=str(transects_dir))
        # transects of this scale that could be cut (the others lie off the image edge)
        n_transects += sum((transects_dir / f"{focus_bbx['scale_id']}_{a}.jpg").exists() for a in angles)
        control.counts.update(transects=transects_before + n_transects)   # shown in the GUI as they are cut
        control.advance()
    logger.info("Finished extracting transect images")

    control.set_failed(stages[0], len(focus) * len(angles) - n_transects, "transects off the image edge")
    if not n_transects:
        logger.warning("No transect could be extracted (every focus lies on the image border "
                       "in the requested directions) - skipping circuli detection")
        return [], circuli, pd.DataFrame()

    ## --- 5. Circuli detections (model for non-padded images, for conf thresh of 0.3)
    logger.info("Gearing up circuli detector")
    control.start_stage(stages[1], n_transects)
    control.counts.setdefault("circuli", 0)

    def count_circuli(n):   # shown in the GUI as they are found
        control.counts["circuli"] += n

    circuli_dets = detect(
        img_dir=str(transects_dir),
        weights=WEIGHTS["circuli"],
        classes_file=CLASS_FILES["circuli"],
        no_det_dir=str(paths.no_circuli),
        yolo_max_boxes=args["transect_max_boxes"],
        plot_dir=str(paths.circuli_plots) if args["plot_dets"] else None,
        per_image_dir=str(paths.per_image / "circuli") if args["dets_separate_files"] else None,
        draw_det_num=True,
        fig_w=100,
        fig_h=5,
        progress=control.progress_callback(),
        on_detections=count_circuli,
        **CIRCULI_MODEL,
    )
    logger.info("Finished circuli detection")
    transect_ids = circuli_dets["img_id"].unique().tolist()
    control.set_failed(stages[1], circuli_dets.loc[circuli_dets["score"].isna(), "img_id"].nunique(),
                       "transects without circuli")

    ## --- 6. Calculate circuli spacings
    logger.info("Calculating intracirculus spacings (in pixels)")
    circuli_dets["x_center"] = (circuli_dets["xmin"] + circuli_dets["xmax"]) / 2
    circuli_dets["y_center"] = (circuli_dets["ymin"] + circuli_dets["ymax"]) / 2
    # detections are sorted by x within each transect (see detections_as_df)
    circuli_dets["spacing_px"] = circuli_dets.groupby("img_id")["x_center"].diff()
    circuli_dets = circuli_dets.rename(columns={"detection_nr": "circulus_nr"})

    ## --- 7. Sanity checks on circuli detections and spacings
    logger.info("Running sanity checks on circuli detections and spacings...")
    circuli_checks(circuli_dets, args["transect_max_boxes"])

    ## --- 8. Position of each circulus on the scale image
    found = circuli_dets.dropna(subset=["score"]).rename(columns={"img_id": "transect_id"})
    found = found.astype({c: int for c in ("circulus_nr", "xmin", "ymin", "xmax", "ymax")})
    circuli = circuli_on_scale(found, focus_by_scale(focus)).reindex(columns=CIRCULI_COLUMNS)

    circuli_summary_stats = circuli_dets[["score", "spacing_px"]].describe(percentiles=[0.05, .5, .95])
    circuli_summary_stats = circuli_summary_stats.rename(columns={"score": "det_conf_score"})
    circuli_summary_stats = circuli_summary_stats.round({"det_conf_score": 4, "spacing_px": 2})
    return transect_ids, circuli, circuli_summary_stats


def run_pipeline(args, paths, control):
    """Detection pipeline for one run; writes into the run folder `paths` and reports
    progress through `control` (RunControl). Returns counts."""

    per_image = args["dets_separate_files"]
    input_images = list_input_images(args["img_dir"])
    n_images = len(input_images)
    control.counts.update(scales=n_images)   # shown in the GUI from the start
    duplicates = find_duplicate_images(input_images)
    if duplicates:
        logger.warning("%d image(s) are identical copies of another image and are processed (and counted) "
                       "twice:\n\n\t%s\n", sum(len(g) - 1 for g in duplicates),
                       unpack_for_string(" = ".join(Path(p).name for p in g) for g in duplicates))

    ## --- 1. Convert image files to jpeg format (work/scales)
    control.start_stage("convert", n_images)
    images_tiff_to_jpeg(args["img_dir"], str(paths.scales), progress=control.progress_callback())

    ## --- 2. Focus detection
    logger.info("Gearing up focus detection")
    control.start_stage("focus", n_images)
    focus_dets = detect(
        img_dir=str(paths.scales),
        weights=WEIGHTS["focus"],
        classes_file=CLASS_FILES["focus"],
        no_det_dir=str(paths.no_focus),
        plot_dir=str(paths.focus_plots) if args["plot_dets"] else None,
        per_image_dir=str(paths.per_image / "focus") if per_image else None,
        draw_det_num=False,
        fig_w=65,
        fig_h=60,
        progress=control.progress_callback(),
        # with a second pass, scales without focus are reported after it
        report_no_detections=not args["focus_retry"],
        **FOCUS_MODEL,
    )
    logger.info("Finished focus detection")

    scale_ids = focus_dets["img_id"].tolist()   # every scale, with or without focus
    # the most confident box per scale (more than one is flagged by focus_checks)
    focus_found = best_focus_box(focus_dets).assign(focus_method="standard")
    missed = [s for s in scale_ids if s not in set(focus_found["img_id"])]
    control.set_failed("focus", len(missed), "no focus")

    ## --- 2b. Second pass, only for scales without a focus: padded to the training
    ##         aspect ratio, then a lower score threshold (see sscd_libs/focus_retry.py)
    if args["focus_retry"] and missed:
        focus_found, missed = second_focus_pass(args, paths, control, focus_found, missed)
        control.set_failed("focus_retry", len(missed), "still no focus")
    if missed:
        logger.warning("No focus found in %d scale image(s) - no circuli are detected on them:\n\n\t%s\n\n"
                       "\tImage(s) saved to %s\n", len(missed), unpack_for_string(missed),
                       paths.no_focus)
    control.counts.update(scales=len(scale_ids), focus_found=len(focus_found))

    focus = focus_found.rename(columns={"img_id": "scale_id"})
    focus["x_px"] = (focus["xmin"] + focus["xmax"]) / 2
    focus["y_px"] = (focus["ymin"] + focus["ymax"]) / 2
    focus = focus.astype({c: int for c in ("xmin", "ymin", "xmax", "ymax")})
    focus = focus.reindex(columns=FOCUS_COLUMNS)

    transect_ids = []
    circuli = pd.DataFrame(columns=CIRCULI_COLUMNS)
    circuli_summary_stats = pd.DataFrame()

    # Only proceed to circuli detection if there is at least one focus detection
    if len(focus_found) > 0:

        ## --- 3. Sanity checks on focus detections
        logger.info("Running sanity checks on focus detections...")
        focus_checks(focus_found)

        ## --- 4.-8. Transects, circuli, spacings and positions on the scale images
        transect_ids, circuli, circuli_summary_stats = transects_and_circuli(
            args, paths, control, focus, paths.transects)

    else:
        logger.warning("Focus detector failed to locate focus in any of the provided scale images - "
                       "system cannot proceed to the circuli detection stage")

    ## --- 9. Results tables
    control.end_stages()
    focus.to_csv(paths.focus_csv, index=False)
    circuli.to_csv(paths.circuli_csv, index=False)
    summary = scales_summary(scale_ids, focus, transect_ids, circuli)
    summary.to_csv(paths.summary_csv, index=False)

    counts = {"scales": len(scale_ids), "focus_found": len(focus), "transects": len(transect_ids),
              "circuli": len(circuli)}
    control.counts.update(counts)
    control.save()

    ## --- 10. Circuli drawn onto the scale images (overlays/)
    if args["overlays"] and len(focus):
        control.start_stage("overlays", len(focus))
        draw_run_overlays(paths, args["transect_angles"], str(Path(args["img_dir"]).resolve()),
                          progress=control.progress_callback())

    logger.info("Summary"
                "\n\n---------------------------------------------------------"
                f"\nScale images processed: {counts['scales']}"
                f"\nFocus detected: \t{counts['focus_found']}"
                f"\nTransects processed: \t{counts['transects']}"
                f"\nCirculi detected: \t{counts['circuli']}"
                "\nCirculi summary statistics:"
                "\n\t" + circuli_summary_stats.to_string().replace('\n', '\n\t') +
                "\n---------------------------------------------------------")
    return counts


# ------------------------------------------------------------------------------
def main():

    # parse the command line arguments
    args_parser = argparse.ArgumentParser(
        description='Salmon Scale Circuli Detector: locate the focus of each scale image, extract '
        'radial transects and detect circuli along them, reporting inter-circulus spacings. '
        'Each run is written to a new folder <output_root>/runs/<date>_<time>[_<run_name>].',
        formatter_class=argparse.MetavarTypeHelpFormatter
        )
    args_parser.add_argument(
        "--img_dir",
        required=True,
        type=str,
        help="directory containing the scale images (.tif/.tiff/.jpg/.jpeg)",
    )
    args_parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="root folder for all SSCD outputs (default: $SSCD_OUTPUT_ROOT, else C:\\sscd_outputs on Windows, ~/sscd_outputs elsewhere)",
    )
    args_parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="optional name appended to the run folder, e.g. 'N-Esk-2018'",
    )
    args_parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="write the run to exactly this (new or empty) folder instead of a new dated "
        "folder under <output_root>/runs",
    )
    args_parser.add_argument(
        "--transect_angles",
        required=False,
        type=int,
        nargs='+',
        default=[0, 45, 90, 135, 180],
        help="choice of angle(s) for radial transects relative to focus, in degrees",
    )
    args_parser.add_argument(
        "--dets_separate_files",
        required=False,
        type=boolean_string,
        default=False,
        help="also save the detections of each image in a separate txt file (results/per_image)",
    )
    args_parser.add_argument(
        "--plot_dets",
        required=False,
        type=boolean_string,
        default=True,
        help="save images with the detections drawn on them, for visual inspection (work/focus, work/circuli)",
    )
    args_parser.add_argument(
        "--transect_max_boxes",
        required=False,
        type=int,
        default=500,
        help="Maximum number of detections per transect image",
    )
    args_parser.add_argument(
        "--overlays",
        required=False,
        type=boolean_string,
        default=False,
        help="also draw the circuli onto the scale images at the end (overlays/); the same as "
        "running overlay_detections.py afterwards",
    )
    args_parser.add_argument(
        "--focus_retry",
        required=False,
        type=boolean_string,
        default=True,
        help="second focus pass for scales without a focus: padded to the training images' aspect "
        "ratio, then with --focus_low_threshold (rescued scales are flagged in focus_method)",
    )
    args_parser.add_argument(
        "--focus_low_threshold",
        required=False,
        type=float,
        default=0.1,
        help="score threshold of the second focus pass's last step (0: skip that step)",
    )
    args = vars(args_parser.parse_args())

    # --------------------------------------- #
    # --               Checks             --- #
    # --------------------------------------- #
    # (before creating the run folder)

    if not os.path.isdir(args["img_dir"]):
        raise FileNotFoundError(f"Image directory not found: {args['img_dir']}")
    list_input_images(args["img_dir"])  # fails early if there are no (or clashing) images

    for detector, weights_file in WEIGHTS.items():
        if not os.path.exists(weights_file + ".index"):
            raise FileNotFoundError(f"Checkpoint files for the {detector} detector not found "
                                    f"({weights_file}.*). Please refer to the README.md file and "
                                    "follow instructions on how to set up yolo weights")

    paths = new_run(args["output_root"], args["run_name"], args["run_dir"])

    # --------------------------------------- #
    # --       Logger Configuration       --- #
    # --------------------------------------- #

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(paths.log, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(stage_log_formatter())   # each line tagged with its stage
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s (%(asctime)s): %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=(console_handler, file_handler)
        )

    logger.info("Run folder: %s", paths.root)

    # --------------------------------------- #
    # --      Run record (run_info.json)  --- #
    # --------------------------------------- #

    info = {
        "status": "queued",
        "created": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(Path(args["img_dir"]).resolve()),
        "parameters": {k: args[k] for k in ("transect_angles", "transect_max_boxes", "plot_dets",
                                            "dets_separate_files", "overlays", "focus_retry",
                                            "focus_low_threshold")},
        "models": {
            "focus": {"weights": Path(WEIGHTS["focus"]).name, **FOCUS_MODEL},
            "circuli": {"weights": Path(WEIGHTS["circuli"]).name, **CIRCULI_MODEL,
                        "yolo_max_boxes": args["transect_max_boxes"]},
        },
        "code": code_version(),
        "environment": environment(),
        "pid": os.getpid(),
    }
    write_info(paths.info, info)

    # live progress, warnings and stop requests (progress.json / STOP in the run folder)
    control = RunControl(paths.root)
    logging.getLogger().addHandler(control.handler)
    file_handler.addFilter(control.stage_filter)
    control.save()

    def finish(status, **fields):
        info.update(status=status, finished=datetime.now().isoformat(timespec="seconds"),
                    warnings=control.warnings, **fields)
        write_info(paths.info, info)
        control.set_status(status)

    try:
        # at most max_concurrent_runs runs execute at a time; further runs wait here
        slot = RunSlot(output_root(args["output_root"]), load_settings()["max_concurrent_runs"])
        slot.wait(paths.root, control,
                  on_wait=lambda: logger.info("Waiting for a free run slot (other runs in progress)..."))
        run_start = time()
        info.update(status="running", started=datetime.now().isoformat(timespec="seconds"))
        write_info(paths.info, info)
        control.set_status("running")

        counts = run_pipeline(args, paths, control)

    except (StopRequested, KeyboardInterrupt) as err:
        logger.warning("Run stopped - results written so far are kept in %s", paths.root)
        finish("stopped", counts=control.counts)
        logging.shutdown()
        if isinstance(err, KeyboardInterrupt):
            raise
        return
    except BaseException as err:  # record failures before re-raising
        logger.exception("Run failed")
        finish("failed", counts=control.counts, error=f"{type(err).__name__}: {err}")
        logging.shutdown()
        raise

    run_duration = round((time() - run_start) / 60, 2)
    finish("completed", runtime_min=run_duration, counts=counts)

    logger.info("Run finished in %s mins. Results: %s", run_duration, paths.results)
    logging.shutdown()


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
