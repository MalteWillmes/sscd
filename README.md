# SSCD
 Salmon Scale Circuli Detector (SSCD)



## Getting started

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (Windows, in PowerShell:
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`).
2. Get SSCD: `git clone https://github.com/MalteWillmes/sscd.git`, or download the repository as
   a ZIP file from GitHub and unpack it - any folder is fine.
3. Start the web app:
   - **Windows:** double-click `start_sscd.bat` in the SSCD folder
   - **Linux / macOS:** run `./start_sscd.sh` in the SSCD folder

The first start installs everything it needs (a few minutes; on Windows the Python environment is
kept in `%LOCALAPPDATA%\sscd\envs\`, so the SSCD folder can be anywhere, e.g. in Dropbox or a
long path). The app then opens in your browser and offers to download the trained model weights
(~790 MB) if they're missing. Results are saved to `sscd_outputs` in your home folder (see
[Where results are saved](#where-results-are-saved)). The rest of this README covers the command
line, evaluation and development.


 __Table of Contents__

   - [Getting started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [How to run SSCD](#how-to-run-sscd)
   - [Evaluating SSCD's performance](#evaluating-sscds-performance)
   - [SSCD Training](#sscd-training)
   - [Development](#development)


## Prerequisites

In order to install and use SSCD the following programmes are required to be installed:
  - [uv](https://docs.astral.sh/uv/getting-started/installation/) (it also installs a suitable Python if needed)
  - [Git][2] (optional: you can also download the repository as a ZIP file from GitHub)


## Installation

#### 1. Clone the SSCD code repository

```bash
git clone https://github.com/MalteWillmes/sscd.git
cd sscd
```

#### 2. Set-up the project environment

This step creates a virtual environment for the SSCD tool, with all the required packages and Python dependencies being automatically installed.

```bash
uv sync --dev --extra gui
```

(`--extra gui` installs the web app's dependencies; leave it out if you only use the command
line. Use the same command whenever you update the environment, otherwise `uv sync` removes
them again.)

Windows notes (`start_sscd.bat` takes care of both automatically, by keeping its environment in
`%LOCALAPPDATA%\sscd\envs\` - separate from the `.venv` used by the commands below):

- Keep the repository in a folder with a reasonably short path. Some TensorFlow files have
  ~150-character paths inside the environment, and Windows' 260-character path limit then
  breaks the install (import errors from `tensorflow`) unless
  [long paths are enabled](https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation).
- In a Dropbox/OneDrive folder, `uv` may fail with "incompatible hardlinks": set
  `setx UV_LINK_MODE copy` once (then open a new terminal), or add `--link-mode=copy`.
- Alternatively keep the environment outside the repository, like the start script does: set
  `UV_PROJECT_ENVIRONMENT` to a short folder, e.g. `setx UV_PROJECT_ENVIRONMENT "%LOCALAPPDATA%\sscd\venv"`.

#### 3. Download YOLOv3 weights for focus and circuli detectors

  Run the following command to download and extract the trained weights (~790 MB) into `data/yoloV3_checkpoints/`:

  ```bash
  uv run python -m sscd_libs.fetch weights
  ```

  (`run_example.py` and the GUI also download the weights automatically if they are missing.)

  That's it: installation (hopefully) done!


#### 4. Updating the SSCD Environment

The SSCD's environment should be updated if project dependencies change (e.g. in `pyproject.toml`).
Once the most recent version has been pulled to the local repository, update the SSCD environment with:
```bash
uv sync --dev --extra gui
```
(`start_sscd.bat` / `start_sscd.sh` do this for their own environment on every start.)

#### 5. How to install a package
Run `uv add <package-name>` to install a package. For example:
```bash
uv add requests
```

## How to run SSCD

### Where results are saved

  Every run is saved to its own new folder; nothing is ever overwritten or deleted. All runs
  live under one **output root**, outside the code folder:

  - default: `sscd_outputs` in your home folder (e.g. `C:\Users\<you>\sscd_outputs`)
  - to change it permanently, set the `SSCD_OUTPUT_ROOT` environment variable
    to an absolute path (Windows: `setx SSCD_OUTPUT_ROOT "D:\SSCD results"`, then open a new
    terminal / restart VS Code), or set `output_root` in the optional settings file (see
    [Web app](#web-app-gui));
    for a single run, pass `--output_root`

  ```
  <output root>
     ├─── runs
     │     ├─── 2026-09-29_1412_N-Esk-2018         one folder per run: <date>_<time>[_<run name>]
     │     │     ├─── run_info.json                what was run: input folder, parameters, models,
     │     │     │                                 SSCD version (git commit), status, counts
     │     │     ├─── sscd.log
     │     │     ├─── results
     │     │     │     ├─── circuli.csv             one row per circulus (see below)
     │     │     │     ├─── focus.csv               focus position per scale
     │     │     │     ├─── scales_summary.csv      per scale: focus found?, transects, circuli,
     │     │     │     │                             median spacing, transects without circuli
     │     │     │     └─── per_image               focus/, circuli/: one txt per image
     │     │     │                                  (if --dets_separate_files True)
     │     │     ├─── progress.json                 live progress, counts and warnings (for the GUI)
     │     │     ├─── console.log, STOP             (runs started from the GUI: console output; a
     │     │     │                                  STOP file appears when a stop was requested)
     │     │     ├─── overlays                      <scale>_overlay.jpg (--overlays / overlay_detections.py)
     │     │     ├─── qc
     │     │     │     ├─── focus_plots             focus detections drawn on each scale
     │     │     │     ├─── circuli_plots           circuli detections drawn on each transect
     │     │     │     └─── no_detections           focus/: scales without focus,
     │     │     │                                  circuli/: transects without circuli
     │     │     └─── work
     │     │           ├─── scales                  8-bit RGB jpeg copies of the input scales
     │     │           └─── transects               the transect images (e.g. for annotating)
     │     └─── ...
     └─── evaluations
           └─── 2026-10-02_1030_circuli-vs-Bruno    one folder per eval_detector.py run
                 ├─── eval_info.json, eval.log
                 ├─── results                        evaluation_results.txt, results_by_image.csv,
                 │                                   circulus_PRC.png
                 └─── plots                          detections vs annotations per image
  ```

  If a folder of that name already exists (e.g. two runs with the same name in the same
  minute), `-2`, `-3`, ... is appended.

  `work/` holds most of a run's size. It can be deleted once you've picked any transect images
  you want to annotate for an evaluation; `overlay_detections.py` then draws on the original
  images from the run's input folder instead (if they're still there).

  `results/circuli.csv` columns: `scale_id`, `transect_id` (`<scale id>_<angle>`), `angle_deg`,
  `circulus_nr` (from the focus outwards), `class_name`, `score`, `spacing_px` (to the previous
  circulus on the transect), `dist_from_focus_px` (along the transect), `x_px`/`y_px` (centre of
  the detection on the scale image) and `xmin`/`ymin`/`xmax`/`ymax` (detection box in the
  transect image). All distances are in pixels of the original scale image.

  `results/focus.csv` columns: `scale_id`, `class_name`, `score`, `xmin`/`ymin`/`xmax`/`ymax`
  (focus box) and `x_px`/`y_px` (focus centre), for scales where a focus was found.
  `results/scales_summary.csv` lists every input scale: `focus_found`, `focus_score`,
  `n_transects`, `total_n_circuli` (all transects together), `mean_n_circuli` (per transect with
  at least one circulus; transects without circuli are not counted), `median_spacing_px` and
  `transects_without_circuli` (angles).

### Web app (GUI)

  Start it with `start_sscd.bat` / `start_sscd.sh` (see [Getting started](#getting-started)),
  **SSCD: GUI (web app)** in VS Code, or from the SSCD folder with

  ```bash
  uv run --extra gui python -m streamlit run sscd_app.py
  ```

  This opens SSCD in your browser, at
  http://localhost:8501 - it only runs on your own computer. In the app you:

  - choose the folder with the scale images (type or paste the path, or use *Browse folders*);
    it is checked straight away (number of images, file types, clashing names)
  - optionally give the run a name, and set the transect angles and advanced options
  - start the run, and follow it: a progress bar per stage, the numbers of images, scales with
    a focus, transects and circuli, and any warnings (e.g. scales where no focus was found) or
    errors
  - stop a run: it ends after the current image and keeps the results written so far
    (status *stopped*)

  Overlays are drawn at the end of each run by default (`overlay_detections.py` can still add
  or redraw them later). A run is an ordinary `sscd.py` run in its own process: it keeps going
  if you close the browser tab, and the app's *Recent runs* list shows it again. Only one run
  executes at a time per output root - including runs started from the command line; further
  runs wait in a queue (status *queued*).

  **Settings (e.g. for a shared server).** An optional settings file, `~/.sscd/settings.toml`
  (or the file named by the `SSCD_SETTINGS` environment variable), can fix the output root,
  restrict which folders the app accepts, and allow more simultaneous runs:

  ```toml
  output_root = "/data/sscd_outputs"
  allowed_input_roots = ["/mnt/scale_archive"]   # empty / missing: any folder
  max_concurrent_runs = 1
  ```

  On a server, scale images are read from folders on the server (e.g. a mounted network share
  holding the scale archive). The app has no login of its own and only listens on `localhost`
  (see `.streamlit/config.toml`): make it available to colleagues through the institute's
  reverse proxy / single sign-on rather than opening the port directly. On Linux, TensorFlow
  can also use a GPU.

### Quick start: run the bundled example

  ```bash
  uv run python run_example.py
  ```

  This downloads the weights if needed and runs the full pipeline on the three scales in
  `data/example_scales`, into a new run folder `<output root>/runs/<date>_<time>_example`.
  Add `--overlay` to also draw the detections onto the scales, `--eval` to also run the
  evaluation example (see below), or `--no_plots` to skip the QC plot images.

### From VS Code

  1. Open the repository folder in VS Code (with the Python extension installed).
  2. `Ctrl+Shift+P` → *Python: Select Interpreter* → choose the project's `.venv`.
  3. Open *Run and Debug* (`Ctrl+Shift+D`), pick a configuration and press `F5`:
     - **SSCD: GUI (web app)** (see above)
     - **SSCD: run example** / **SSCD: run example + evaluation** / **SSCD: run example + overlay**
     - **SSCD: run on a folder of scales**: asks for the image folder and an optional run name
     - **SSCD: overlay detections of the latest run**

  Formatting on save is deliberately disabled in `.vscode/settings.json`, so editing a file
  doesn't restyle it (this fork keeps upstream's formatting to stay easy to merge).

### On your own scale images

  The image folder can be anywhere (it is only read, never changed):

  ```bash
  uv run python sscd.py --img_dir "Z:\Adult_Salmon_Scales\2024\scales" --run_name "2024-batch1"
  ```

### Performance and memory

  A run needs roughly 1.2 GB of RAM and, on a typical laptop CPU, well under a minute
  per scale. If runs are dramatically slower than that (minutes per scale), try disabling
  TensorFlow's oneDNN optimizations, which can be pathologically slow on some Intel
  hybrid laptop CPUs (e.g. 12th-gen "U" series). Results are unaffected:

  ```bash
  # Windows (applies to new terminals; restart VS Code)
  setx TF_ENABLE_ONEDNN_OPTS 0
  # Linux/macOS
  export TF_ENABLE_ONEDNN_OPTS=0
  ```

  Leave oneDNN enabled on machines where it performs well (most servers and desktops).


### `sscd.py` inputs

| Argument               | Description                     | Type          | Default         |
|------------------------|---------------------------------|---------------|-----------------|
| `--img_dir`    | Directory containing the scale images: `.tif`/`.tiff`/`.jpg`/`.jpeg` (any case; 8/16-bit, greyscale or colour) | str  |      |
| `--run_name`   | Optional name appended to the run folder, e.g. `N-Esk-2018` | str  |      |
| `--output_root` | Root folder for all outputs | str  | `$SSCD_OUTPUT_ROOT`, else `~/sscd_outputs` |
| `--run_dir`    | Write the run to exactly this new (or empty) folder instead of a dated folder under the output root (`--run_name`/`--output_root` are then ignored) | str  |      |
| `--transect_angles` | Angle(s) of the radial transects from the focus, in degrees (0-359; 0 = right, 90 = up) | int (spaced) | `0 45 90 135 180` |
| `--plot_dets`    | Save images with the detections drawn on them (`qc/`), for visual inspection | bool (`True`/`False`)  | `True` |
| `--dets_separate_files` | Also write the detections of each image to a separate txt file (`results/per_image/`) | bool (`True`/`False`) | `False` |
| `--transect_max_boxes` | Maximum number of detections per transect image              | int    | `200`  |
| `--overlays` | Also draw the circuli onto the scale images at the end (`overlays/`), like `overlay_detections.py` | bool (`True`/`False`) | `False` (the GUI: on) |


### Circuli detections on the original scale image

After a run, `overlay_detections.py` draws the detections back onto the original scale images
(it re-uses the run's results; nothing is re-detected):

```bash
uv run python overlay_detections.py --latest
uv run python overlay_detections.py --run_dir "<output root>/runs/2026-09-29_1412_N-Esk-2018"
```

It writes `<run folder>/overlays/<scale>_overlay.jpg`: the scale image with the focus box
(yellow), the outline of each radial transect and a tick across the transect at each detected
circulus, one colour per transect (labelled with its angle).

| Argument        | Description                                                  | Type | Default |
|-----------------|--------------------------------------------------------------|------|---------|
| `--run_dir` / `--latest` | A run folder, or the most recently started completed run under the output root | str / flag |  |
| `--output_root` | Output root to look for `--latest` in                        | str  | as for `sscd.py` |
| `--label_every` | Number every n-th circulus on the overlay (0: no numbers)    | int  | `0`     |

Positions are mapped back with the same transect geometry used to cut the transects, accurate to
about 1 pixel.


## Evaluating SSCD's performance

Evaluating the performance of the SCCD is crucial to identify degradation in the system's capacity to produce reliable detections of circuli bands, and subsequently provide accurate intercirculi spacings. Consistent drops in evaluation metrics on new images, compared to [those][5] obtained when the system was last trained, indicates the system needs to be [retrained](#sscd-training) with fresh images.

The performance of each detector comprised in SSCD's pipeline can be evaluated via the `eval_detector.py` function. This tool combines outputs from the `sscd.py` script with annotation data (provided by the user) to produce standard object detection evaluation metrics.

Core computational tasks were adapted from [this project][4], where background information on evaluation methods for object detection algorithms and relevant performance metrics can also be found.

A more detailed guide for evaluating the performance of SSCD's detectors is available [here][6].

The following example evaluates the circulus detector on the bundled annotated transects (`uv run python run_example.py --eval` runs the same thing):

```bash
uv run python eval_detector.py --img_dir "./data/eval_example/imgs/" --anns_dir "./data/eval_example/anns/" --dets_csv "./data/eval_example/detections.csv" --eval_name "example" --sep_plots True
```

### `eval_detector.py` inputs


| Argument     | Description                                             | Type          | Default  |
|--------------|---------------------------------------------------------|---------------|----------|
| `--img_dir`  | Directory path to images for evaluation (`.jpg`, e.g. transect images from a run's `work/transects`) | str    |          |
| `--anns_dir` | Directory path to annotation files: Pascal VOC XML files (or evaluator `.txt` files, e.g. from `annotations_xml_to_txt.py`) | str  |       |
| `--dets_csv` | Detections to evaluate: a run's `results/circuli.csv` (or `results/focus.csv`), or a CSV with `img_id`, `class_name`, `score`, `xmin`, `ymin`, `xmax`, `ymax` columns | str | |
| `--iou_threshould` | IOU threshold (IOU<sub>thresh</sub>) determining if a detection is TP or FP (see "Metrics" section bellow) | float  | `0.5`  |
| `--eval_name` | Optional name appended to the evaluation folder | str |  |
| `--output_root` | Root folder for all outputs | str | `$SSCD_OUTPUT_ROOT`, else `~/sscd_outputs` |
| `--eval_dir` | Write the evaluation to exactly this new (or empty) folder (`--eval_name`/`--output_root` are then ignored) | str |  |
| `--plot_dets_vs_anns` | Option to generate image plots contrasting detections with annotations | bool   | `True` |
| `--sep_plots` | Option to produce separate plots for detections and annotations. If `False` draw both in the same plot (recommended for focus detections) | bool  | `False`  |
| `--get_details` | Also write per-detection and per-annotation evaluation details | bool  | `False`  |


### `eval_detector.py` outputs

Evaluation metrics are printed to the console and saved in a new folder
`<output root>/evaluations/<date>_<time>[_<eval name>]` (see [above](#where-results-are-saved)):

  - `results/evaluation_results.txt` - Main evaluation metrics
  - `results/results_by_image.csv` - Classification of detections by image
  - `results/circulus_PRC.png` - Precision-Recall curve for the object class under evaluation
  - `plots/` - detections vs. annotations image plots
  - `eval.log` / `eval_info.json` - log, and a record of the inputs, parameters and code version



#### Definitions and Metrics:
  - Intersection Over Union (IOU):  the overlapping area between the detection bounding box and the annotation bounding box divided by the area of union between them:

    ![](docs/images/iou.png)

  - IOU threshold (IOU<sub>thresh</sub>): determines if a detection is classified as True Positive or False Positive
  - True Positive (TP): a correct detection (i.e. a detection with IOU &ge; IOU<sub>thresh</sub>)
  - False Positive (FP): an incorrect detection (i.e. a detection with IOU &lt; IOU<sub>thresh</sub> **OR** an extra TP on the same annotation)
  - False Negative (FN): an undetected annotation
  - Precision: the proportion of correct positive detections = TP/(TP+FP)
  - Recall: the proportion of annotations correctly detected (*true positive rate*) = TP/(TP+FN)
  - Average precision (AP): a combination of precision and recall scores. Given by the area under the precision vs. recall curve (check [here][4] for more details).
  - F<sub>1</sub> score: the harmonic mean of precision and recall. Higher scores when both recall and precision are high.
  - Mean Centre Error (MCE): average of Euclidian distances (in pixels) between the centres of TP detection boxes and respective annotation boxes


> **Note of caution**
>
> Annotations are not ground truths in a strict sense. Target objects are marked manually and thence subject to human error and labelling ambiguity. Therefore, performance metrics are highly dependent not only on the accuracy of the detector, but also on the quality of annotations used on the evaluation. Image plots contrasting detections against annotations should help scrutinise if apparent drops in performance metrics are being driven by a deteriorating detector, by poor labelling, or both.

## SSCD Training

As mentioned above, retraining the SSCD's detectors might become necessary if/when performance levels on new set of scale images drop substantially from those observed after the latest training.

Each detector is a [YOLOv3][10] (*You Only Look Once*) model trained for its specific detection task. YOLOv3 models were implemented using [Tensorflow 2](https://www.tensorflow.org/) (an open-source deep learning library developed by Google), based on the excellent repository by [Zihao Zhang][7].

This [page][8] provides details on how to set up a workstation for (re)training the SSCD's detectors.

*The training protocol (previously a Jupyter notebook, `docs/SSCD Training Protocol.ipynb`) has been removed from this fork; it remains available in the git history. The training code path has not yet been updated for TensorFlow 2.16+/Keras 3.*


## Development

### Update from template
To update your project with the latest changes from the template, run the command below (note:
`.copier-answers.yml` still says `notebook: true`, so an update may try to add Jupyter files
again - this fork no longer uses notebooks):
```bash
uvx --with copier-template-extensions copier update --trust
```

You can keep your previous answers by using:
```bash
uvx --with copier-template-extensions copier update --trust --defaults
```

### (Optional) pre-commit
pre-commit is a set of tools that help you ensure code quality. It runs every time you make a commit.

First, install pre-commit:
```bash
uv tool install pre-commit
```

Then install pre-commit hooks:
```bash
pre-commit install
```

To run pre-commit on all files:
```bash
pre-commit run --all-files
```

On machines that block `.exe` launchers in user folders, pre-commit (and its hooks) cannot run
locally; the same checks run on GitHub (Actions, once enabled for the repository).


### References (supporting code)
- [YOLOv3 implementation in Tensorflow 2][7]
- [Diagonal crop][9]
- Object detection evaluation [tool](https://github.com/rafaelpadilla/Object-Detection-Metrics#how-to-use-this-project)

[2]: https://git-scm.com/downloads "Git Installers"
[4]: https://github.com/rafaelpadilla/Object-Detection-Metrics
[5]: /docs/sscd_evaluate.md#evaluation-metrics-on-test-set-on-latest-training
[6]: /docs/sscd_evaluate.md
[7]: https://github.com/zzh8829/yolov3-tf2
[8]: /docs/sscd_setup_for_training.md
[9]: https://github.com/jobevers/diagonal-crop
[10]: https://arxiv.org/pdf/1804.02767.pdf
