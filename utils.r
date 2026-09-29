#' Wrapper function to run python command-line function "eval_detector.py",
#' which is part of the SSCD toolkit. Results are written to a new folder
#' <output root>/evaluations/<date>_<time>[_<eval_name>] (output root: $SSCD_OUTPUT_ROOT,
#' else ~/sscd_outputs).
#'
#' dets_csv can be a run's results/circuli.csv (or results/focus.csv).
#' Requires the fs and glue packages, and uv on the PATH.
eval_detections <- function(sscd_path, img_dir, anns_dir, dets_csv, iou_threshould = 0.5,
                            eval_name = NULL, plot_dets_vs_anns = TRUE, sep_plots = TRUE) {

  # eval_name replaced the former output_dir argument (outputs now go to a new dated folder)
  if (!is.null(eval_name) && grepl("[/\\\\]", eval_name)) {
    stop("eval_name looks like a path: output_dir was replaced by eval_name, a short name ",
         "for the evaluation folder (outputs go to <output root>/evaluations/).")
  }

  quote_path <- function(x) glue::double_quote(normalizePath(x))

  # Build the command to run evaluation module (written in python), inside the
  # project's uv environment
  args <- c(
    "uv run --project", quote_path(sscd_path),
    "python", quote_path(file.path(sscd_path, "eval_detector.py")),
    "--img_dir", quote_path(fs::path_abs(img_dir)),
    "--anns_dir", quote_path(fs::path_abs(anns_dir)),
    "--dets_csv", quote_path(fs::path_abs(dets_csv)),
    "--iou_threshould", iou_threshould,
    "--plot_dets_vs_anns", ifelse(plot_dets_vs_anns, "True", "False"),
    "--sep_plots", ifelse(sep_plots, "True", "False")
  )
  if (!is.null(eval_name)) {
    args <- c(args, "--eval_name", glue::double_quote(eval_name))
  }

  # Invoke system command
  system(paste(args, collapse = " "))
}
