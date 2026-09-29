#' Wrapper function to run python command-line function "eval_detector.py", 
#' which is part of the SSCD toolkit. Results are written to a new folder
#' <output root>/evaluations/<date>_<time>_<eval_name> (output root: $SSCD_OUTPUT_ROOT,
#' else ~/sscd_outputs).
eval_detections <- function(sscd_path, img_dir, anns_dir, dets_csv, iou_threshould,
                            eval_name, plot_dets_vs_anns, sep_plots) {
  
  # list with absolute paths
  abs_paths <- list(
    eval_fun_path = file.path(sscd_path, "eval_detector.py"),
    img_dir = fs::path_abs(img_dir),
    anns_dir = fs::path_abs(anns_dir),
    dets_csv = fs::path_abs(dets_csv)
  )
  
  # Format paths for use in windows command prompt
  abs_paths_norm <- lapply(
    abs_paths, 
    function(x){
      double_quote(
        normalizePath(x)
      )
    }
  )
  
  # Build the command to run evaluation module (written in python), inside the
  # project's uv environment
  eval_py_call <- glue::glue(
    'uv run --project {double_quote(normalizePath(sscd_path))} python {abs_paths_norm$eval_fun_path}',
    '--img_dir {abs_paths_norm$img_dir}',
    '--anns_dir {abs_paths_norm$anns_dir}',
    '--dets_csv {abs_paths_norm$dets_csv}',
    '--iou_threshould {iou_threshould}',
    '--eval_name {double_quote(eval_name)}',
    '--plot_dets_vs_anns {ifelse(plot_dets_vs_anns, "True", "False")}',
    '--sep_plots {ifelse(sep_plots, "True", "False")}',
    .sep = " "
  )
  
  # Invoke system command
  system(eval_py_call)
  
}

