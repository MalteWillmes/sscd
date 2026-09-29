# SSCD - Salmon Scale Circuli Detector

Finds the focus (centre) of salmon scale images and detects the circuli (growth rings) along
radial transects, using two YOLOv3 detectors.

> **Please cite the original method and model:**
> Hanson NN, Ounsley JP, Henry J, Terzić K, Caneco B (2024). Automatic detection of fish scale
> circuli using deep learning. *Biology Methods and Protocols* 9(1): bpae056.
> https://doi.org/10.1093/biomethods/bpae056

This is a fork of [NINAnor/sscd](https://github.com/NINAnor/sscd) (original code by Bruno Caneco),
updated for TensorFlow 2.16+/Keras 3, with a web app, overlays and a new output layout.

- [Getting started](#getting-started)
- [Outputs](#outputs)
- [Web app](#web-app)
- [Command line](#command-line)
- [Evaluating the detectors](#evaluating-the-detectors)
- [Training](#training)
- [Development](#development)
- [Quick notes (Windows paths)](#quick-notes-windows-paths)


## Getting started

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (Windows PowerShell:
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`).
2. Get the code: `git clone https://github.com/MalteWillmes/sscd.git` (or download the ZIP).
3. Start the web app: double-click `start_sscd.bat` (Windows) or run `./start_sscd.sh` (Linux/macOS).

The first start installs everything (a few minutes) and offers to download the trained weights
(~790 MB). Windows: see [Quick notes](#quick-notes-windows-paths) for long paths and Dropbox.

**For the command line / development**, set up the environment in the SSCD folder:

```bash
uv sync --dev --extra gui
uv run python -m sscd_libs.fetch weights
```

Re-run `uv sync --dev --extra gui` after pulling changes (plain `uv sync` removes the GUI packages).


## Outputs

Every run gets a new folder; nothing is overwritten. The **output root** is `C:\sscd_outputs`
by default (`~/sscd_outputs` on Linux/macOS, or if `C:\` is not writable). Change it with the
`SSCD_OUTPUT_ROOT` environment variable, `output_root` in the [settings file](#settings-shared-server)
or `--output_root`.

```
<output root>
 ├── runs/<date>_<time>[_<run name>]/
 │    ├── run_info.json          input folder, parameters, code version, status, counts
 │    ├── sscd.log, progress.json (console.log, STOP: runs started from the GUI)
 │    ├── results/
 │    │    ├── circuli.csv        one row per circulus
 │    │    ├── focus.csv          focus per scale
 │    │    ├── scales_summary.csv one row per input scale
 │    │    └── per_image/         one txt per image (--dets_separate_files True)
 │    ├── overlays/              <scale>_overlay.jpg: circuli drawn on the scale
 │    └── work/                  images of each step - can be deleted once checked
 │         ├── scales/           jpeg copies of the input scales
 │         ├── focus/            focus drawn on each scale; no_focus/, second_pass/
 │         ├── transects/        the transect images
 │         └── circuli/          circuli drawn on each transect; no_circuli/
 └── evaluations/<date>_<time>[_<name>]/
```

**`circuli.csv`**: `scale_id`, `transect_id` (`<scale id>_<angle>`), `angle_deg`, `circulus_nr`
(from the focus outwards), `class_name`, `score`, `spacing_px` (to the previous circulus),
`dist_from_focus_px`, `x_px`/`y_px` (centre on the scale image), `xmin`...`ymax` (box in the
transect image). Distances are in pixels of the original image.

**`focus.csv`**: `scale_id`, `class_name`, `score`, `focus_method`, `n_focus_boxes`,
`xmin`...`ymax` (box), `x_px`/`y_px` (centre).

**`scales_summary.csv`**: `scale_id`, `focus_found`, `focus_score`, `focus_method`,
`n_focus_boxes`, `n_transects`, `total_n_circuli`, `mean_n_circuli` (per transect with at least
one circulus), `median_spacing_px`, `transects_without_circuli` (angles).

**Focus QC flags**
- `focus_method`: `standard`, or found in the **second pass**, which retries only scales without
  a focus: `padded` (image padded to the training aspect ratio, 3840 x 2748) and then
  `low_threshold` (best box above `--focus_low_threshold`, default 0.1). On 110 test scales it
  found 7 of 16 missed foci without adding false ones. Check these scales on the overlays.
- `n_focus_boxes` > 1: several focus boxes were found; the most confident one is used. All boxes
  are drawn in `work/focus` - check for several scales in one image or a false detection.
- Identical duplicate image files are reported (GUI and run warnings), as they would be counted twice.


## Web app

Start with `start_sscd.bat` / `start_sscd.sh`, the **SSCD: GUI (web app)** configuration in
VS Code, or `uv run --extra gui python -m streamlit run sscd_app.py`. It opens at
http://localhost:8501.

- Choose the image folder: type the path, *Choose folder...* (your computer's folder window),
  *Recent folders*, or *Browse folders*. The folder is checked immediately (images, clashing
  names, duplicates).
- Set a run name, transect angles and advanced options; start the run.
- Follow progress per stage, counts and warnings. *Stop run* ends after the current image and
  keeps the results.

Runs are separate processes: they keep going if the browser is closed. One run executes at a
time per output root (also counting command-line runs); others wait in a queue. Overlays are on
by default.

### Settings (shared server)

Optional `~/.sscd/settings.toml` (or the file in `SSCD_SETTINGS`):

```toml
output_root = "/data/sscd_outputs"
allowed_input_roots = ["/mnt/scale_archive"]   # restrict input folders; hides "Choose folder..."
max_concurrent_runs = 1
```

The app has no login and only listens on `localhost` (`.streamlit/config.toml`); expose it
through a reverse proxy with single sign-on.


## Command line

```bash
uv run python run_example.py                    # the 3 bundled example scales (--overlay, --eval, --no_plots)
uv run python sscd.py --img_dir "Z:\scales\2024" --run_name 2024-batch1
uv run python overlay_detections.py --latest    # draw detections of the latest run onto the scales
```

In VS Code: select the project's `.venv` as interpreter, then use the configurations under
*Run and Debug* (GUI, example, a folder of scales, overlays).

### `sscd.py` options

| Option | Description | Default |
|---|---|---|
| `--img_dir` | Folder of scale images (`.tif`/`.tiff`/`.jpg`/`.jpeg`, 8/16-bit, grey or colour); only read | |
| `--run_name` | Appended to the run folder name | |
| `--output_root` | Root folder for all outputs | `$SSCD_OUTPUT_ROOT`, else `C:\sscd_outputs` (Windows) / `~/sscd_outputs` |
| `--run_dir` | Write to exactly this new/empty folder instead | |
| `--transect_angles` | Transect directions in degrees (0 = right, 90 = up) | `0 45 90 135 180` |
| `--plot_dets` | Save the detections drawn on the images (`work/focus`, `work/circuli`) | `True` |
| `--dets_separate_files` | Also one txt per image (`results/per_image/`) | `False` |
| `--transect_max_boxes` | Maximum detections per transect | `500` |
| `--overlays` | Draw overlays at the end of the run | `False` (GUI: `True`) |
| `--focus_retry` | Second focus pass for scales without a focus | `True` |
| `--focus_low_threshold` | Score threshold of the second pass's last step (`0`: skip) | `0.1` |

### `overlay_detections.py` options

| Option | Description | Default |
|---|---|---|
| `--run_dir` / `--latest` | A run folder, or the latest completed run | |
| `--output_root` | Where to look for `--latest` | as `sscd.py` |
| `--label_every` | Number every n-th circulus (0: none) | `0` |

Overlays show the focus box (yellow), each transect's outline and a tick at each circulus, one
colour per transect. If `work/` was deleted, the original images are used.

### Speed and memory

About 1.2 GB RAM and well under a minute per scale on a laptop CPU. If it takes minutes per
scale (seen on Intel 12th-gen "U" laptop CPUs), disable oneDNN: `setx TF_ENABLE_ONEDNN_OPTS 0`
(Windows) or `export TF_ENABLE_ONEDNN_OPTS=0`. Results are unaffected; leave it on elsewhere.


## Evaluating the detectors

`eval_detector.py` compares detections with manual annotations (Pascal VOC XML) and reports
precision, recall, average precision, F1 and mean centre error. A clear drop compared to the
[metrics at the last training][5] suggests retraining. Step-by-step guide: [docs/sscd_evaluate.md][6].

```bash
uv run python eval_detector.py --img_dir "./data/eval_example/imgs/" --anns_dir "./data/eval_example/anns/" --dets_csv "./data/eval_example/detections.csv" --eval_name example --sep_plots True
```

| Option | Description | Default |
|---|---|---|
| `--img_dir` | Images to evaluate (`.jpg`, e.g. a run's `work/transects`) | |
| `--anns_dir` | Annotations: Pascal VOC XML (or evaluator `.txt`) | |
| `--dets_csv` | A run's `results/circuli.csv` or `results/focus.csv` (or a CSV with `img_id`, `class_name`, `score`, `xmin`, `ymin`, `xmax`, `ymax`) | |
| `--iou_threshould` | IoU needed for a true positive (spelling as in the code) | `0.5` |
| `--eval_name` / `--output_root` / `--eval_dir` | Where to write, as for `sscd.py` | |
| `--plot_dets_vs_anns` | Plot detections vs annotations | `True` |
| `--sep_plots` | Separate plots for detections and annotations (`False` recommended for focus) | `False` |
| `--get_details` | Per-detection and per-annotation details | `False` |

Outputs (`<output root>/evaluations/...`): `results/evaluation_results.txt`,
`results/results_by_image.csv`, `results/circulus_PRC.png`, `plots/`, `eval.log`, `eval_info.json`.

A detection is a true positive if its IoU (overlap / union with an annotation box) is at least
the threshold; extra detections on the same annotation are false positives. Annotations are
themselves imperfect - check the plots before concluding the detector got worse.
Metrics code adapted from [Object-Detection-Metrics][4].


## Training

The detectors are YOLOv3 models ([yolov3-tf2][7]); see [docs/sscd_setup_for_training.md][8]
and the paper above. The training code has not yet been updated for TensorFlow 2.16+/Keras 3.


## Development

```bash
uv run python -m pytest                  # all tests (~2 min; slow ones need the weights)
uv run python -m pytest -m "not slow"    # fast tests only
```

GitHub runs the tests (*Tests*) and the pre-commit checks (*CI*) on every push. To run the
checks locally: `uv tool install pre-commit`, `pre-commit install`, `pre-commit run --all-files`
(not possible where `.exe` launchers in user folders are blocked). Formatting on save is off in
VS Code so that files keep upstream's style.

Template updates: `uvx --with copier-template-extensions copier update --trust --defaults`
(`.copier-answers.yml` still has `notebook: true`; this fork has no notebooks).

## Quick notes (Windows paths)

Windows limits paths to 260 characters unless
[long paths are enabled](https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation)
(needs admin rights). Without that:

- **SSCD folder:** `start_sscd.bat` keeps its environment in `%LOCALAPPDATA%\sscd\envs\`, so the
  SSCD folder can be anywhere. With `uv sync` inside a long path, TensorFlow's deep file paths
  break the install (import errors); keep the environment elsewhere with
  `setx UV_PROJECT_ENVIRONMENT "%LOCALAPPDATA%\sscd\venv"`.
- **Output root:** output files are named after the image and angle, so long image file names
  plus a deep output root can fail. The default `C:\sscd_outputs` is short for this reason; keep
  any other root (`SSCD_OUTPUT_ROOT`) short too.
- **Dropbox / OneDrive:** uv may fail with "incompatible hardlinks": `setx UV_LINK_MODE copy`.

`setx` applies to new terminals; restart VS Code afterwards.

**References:** [yolov3-tf2][7] · [Diagonal crop][9] · [Object-Detection-Metrics][4] · [YOLOv3][10]

[4]: https://github.com/rafaelpadilla/Object-Detection-Metrics
[5]: /docs/sscd_evaluate.md#evaluation-metrics-on-test-set-on-latest-training
[6]: /docs/sscd_evaluate.md
[7]: https://github.com/zzh8829/yolov3-tf2
[8]: /docs/sscd_setup_for_training.md
[9]: https://github.com/jobevers/diagonal-crop
[10]: https://arxiv.org/pdf/1804.02767.pdf
