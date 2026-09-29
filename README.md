# SSCD
 Salmon Scale Circuli Detector (SSCD)



 __Table of Contents__

   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [How to run SSCD](#how-to-run-sscd)
   - [Evaluating SSCD's performance](#evaluating-sscds-performance)
   - [SSCD Training](#sscd-training)


## Prerequisites

In order to install and use SSCD the following programmes are required to be installed:
  - [uv](https://docs.astral.sh/uv/getting-started/installation/)
  - [Git][2]


## Installation

#### 1. Clone the SSCD code repository

```bash
git clone <repository-url>
cd sscd
```

#### 2. Set-up the project environment

This step creates a virtual environment for the SSCD tool, with all the required packages and Python dependencies being automatically installed.

```bash
uv sync --dev
```

#### 3. Download YOLOv3 weights for focus and circuli detectors

  Run the following command to download and extract the trained weights (~790 MB) into `data/yoloV3_checkpoints/`:

  ```bash
  uv run sscd-fetch weights
  ```

  (If your system blocks the `sscd-fetch` launcher, use `uv run python -m sscd_libs.fetch weights` instead.
  `run_example.py` below also downloads the weights automatically if they are missing.)

  That's it: installation (hopefully) done!


#### 4. Updating the SSCD Environment

The SSCD's environment should be updated if project dependencies change (e.g. in `pyproject.toml`).
Once the most recent version has been pulled to the local repository, update the SSCD environment with:
```bash
uv sync --dev
```

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
    (Windows: `setx SSCD_OUTPUT_ROOT "D:\SSCD results"`, then open a new terminal);
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
     │     │     │     └─── per_image               (if --dets_separate_files True)
     │     │     ├─── overlays                      <scale>_overlay.jpg (overlay_detections.py)
     │     │     ├─── qc
     │     │     │     ├─── focus_plots             focus detections drawn on each scale
     │     │     │     ├─── circuli_plots           circuli detections drawn on each transect
     │     │     │     └─── no_detections           scales without focus / transects without circuli
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

  `work/` holds most of a run's size. It can be deleted once you've picked any transect images
  you want to annotate for an evaluation; `overlay_detections.py` then draws on the original
  images from the run's input folder instead (if they're still there).

  `results/circuli.csv` columns: `scale_id`, `transect_id` (`<scale id>_<angle>`), `angle_deg`,
  `circulus_nr` (from the focus outwards), `class_name`, `score`, `spacing_px` (to the previous
  circulus on the transect), `dist_from_focus_px` (along the transect), `x_px`/`y_px` (centre of
  the detection on the scale image) and `xmin`/`ymin`/`xmax`/`ymax` (detection box in the
  transect image). All distances are in pixels of the original scale image.

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
  # Windows (new terminals only)
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
| `--run_dir`    | Write the run to exactly this new (or empty) folder instead of a dated folder under the output root | str  |      |
| `--transect_angles` | Choice of angle(s) for radial transects in degrees (0-360)  | int (spaced) | `0 45 90 135 180` |
| `--plot_dets`    | Save images with the detections drawn on them (`qc/`), for visual inspection | bool (`True`/`False`)  | `True` |
| `--dets_separate_files` | Also write the detections of each image to a separate txt file (`results/per_image/`) | bool (`True`/`False`) | `False` |
| `--transect_max_boxes` | Maximum number of detections per transect image              | int    | `200`  |


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
| `--run_dir` / `--latest` | A run folder, or the most recent run under the output root | str / flag |  |
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
uv run python eval_detector.py \
    --img_dir "./data/eval_example/imgs/" \
    --anns_dir "./data/eval_example/anns/" \
    --dets_csv "./data/eval_example/detections.csv" \
    --iou_threshould 0.5 \
    --eval_name "example" \
    --plot_dets_vs_anns True \
    --sep_plots True
```

### `eval_detector.py` inputs


| Argument     | Description                                             | Type          | Default  |
|--------------|---------------------------------------------------------|---------------|----------|
| `--img_dir`  | Directory path to images for evaluation. Expects JPEG images | str    |          |
| `--anns_dir` | Directory path to annotation files. Expects XML files with Pascal VOC format  | str  |       |
| `--dets_csv` | Detections to evaluate: a run's `results/circuli.csv` (or `results/focus.csv`), or a CSV with `img_id`, `class_name`, `score`, `xmin`, `ymin`, `xmax`, `ymax` columns | str | |
| `--iou_threshould` | IOU threshold (IOU<sub>thresh</sub>) determining if a detection is TP or FP (see "Metrics" section bellow) | float  | `0.5`  |
| `--eval_name` | Optional name appended to the evaluation folder | str |  |
| `--output_root` | Root folder for all outputs | str | `$SSCD_OUTPUT_ROOT`, else `~/sscd_outputs` |
| `--eval_dir` | Write the evaluation to exactly this new (or empty) folder | str |  |
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
To update your project with the latest changes from the template, run:
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
