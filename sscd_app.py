"""
SSCD web app: choose a folder of scale images and the parameters, start / stop a
run and follow its progress, summary and warnings.

Start it from the repository root (needs the optional GUI dependencies):

    uv sync --extra gui
    uv run --extra gui python -m streamlit run sscd_app.py

Runs are ordinary sscd.py runs in their own process and run folder: they keep
going if the browser tab is closed, and are listed again when the app reopens.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from sscd_libs.fetch import WEIGHTS_SENTINELS, fetch_weights
from sscd_libs.jobs import (
    FINISHED,
    check_input_dir,
    list_subfolders,
    parse_angles,
    recent_runs,
    run_state,
    start_run,
    stop_run,
)
from sscd_libs.outputs import output_root
from sscd_libs.settings import input_dir_allowed, load_settings, settings_path

st.set_page_config(
    page_title="SSCD - Salmon Scale Circuli Detector",
    page_icon=str(Path(__file__).parent / "assets" / "sscd_icon.png"),  # browser-tab icon
    layout="centered",
)

STATUS_LABELS = {
    "starting": "Starting",
    "queued": "Waiting for a free slot (another run is in progress)",
    "running": "Running",
    "completed": "Completed",
    "failed": "Failed",
    "stopped": "Stopped",
}

# --- settings -----------------------------------------------------------------
try:
    settings = load_settings()
except (OSError, ValueError) as err:
    st.error(f"Invalid settings file {settings_path()}: {err}")
    st.stop()
ROOT = output_root(settings["output_root"])
RUNS_DIR = ROOT / "runs"


def shown_run():
    """The run shown in this browser tab (?run=<run folder name>), if valid."""
    name = st.query_params.get("run")
    if not name or Path(name).name != name:  # a bare folder name only, never a path
        return None
    run_dir = RUNS_DIR / name
    return run_dir if run_dir.is_dir() else None


# --- header ---------------------------------------------------------------------
st.title("Salmon Scale Circuli Detector")
st.caption(f"Results are saved to `{ROOT}`")

weights_ok = all(p.exists() for p in WEIGHTS_SENTINELS)
if not weights_ok:
    st.warning("The trained detector weights (~790 MB) are not installed yet.")
    if st.button("Download weights"):
        with st.spinner("Downloading and extracting the weights - this takes a few minutes..."):
            fetch_weights()
        st.rerun()


# --- new run --------------------------------------------------------------------
st.header("New run")

if "img_dir" not in st.session_state:
    st.session_state.img_dir = ""


def _browse_start():
    current = Path(st.session_state.img_dir.strip().strip('"') or ".").expanduser()
    if current.is_dir() and input_dir_allowed(current, settings):
        return current.resolve()
    roots = settings["allowed_input_roots"]
    return Path(roots[0]) if roots else Path.home()


def _set_browse(path):
    st.session_state.browse_dir = str(path)


def _use_browse():
    st.session_state.img_dir = st.session_state.browse_dir


st.text_input("Scale images folder", key="img_dir",
              help="Folder containing the scale images (.tif/.tiff/.jpg/.jpeg). It is only read, never changed.")

with st.expander("Browse folders"):
    if "browse_dir" not in st.session_state:
        st.session_state.browse_dir = str(_browse_start())
    browse = Path(st.session_state.browse_dir)
    st.write(f"`{browse}`")
    col_up, col_use = st.columns(2)
    parent_ok = browse.parent != browse and input_dir_allowed(browse.parent, settings)
    col_up.button("Up one level", on_click=_set_browse, args=(browse.parent,), disabled=not parent_ok,
                  width="stretch")
    col_use.button("Use this folder", on_click=_use_browse, type="primary", width="stretch")
    subfolders = list_subfolders(browse)
    if subfolders:
        choice = st.selectbox("Sub-folders", subfolders, format_func=lambda p: p.name, index=None,
                              placeholder=f"{len(subfolders)} sub-folders - choose one to open")
        if choice is not None:
            _set_browse(choice)
            st.rerun()
    else:
        st.caption("No sub-folders.")

folder = check_input_dir(st.session_state.img_dir, settings)
if st.session_state.img_dir.strip():
    (st.success if folder["ok"] else st.error)(folder["message"])

run_name = st.text_input("Run name (optional)", placeholder="e.g. N-Esk-2018",
                         help="Appended to the run folder name: <date>_<time>_<run name>")
angles_text = st.text_input("Transect angles (degrees)", value="0, 45, 90, 135, 180",
                            help="Directions of the radial transects from the focus: 0 = right, 90 = up")
angles, angles_error = None, None
try:
    angles = parse_angles(angles_text)
except ValueError as err:
    angles_error = str(err)
    st.error(angles_error)

overlays = st.checkbox("Draw the detected circuli onto the scale images (overlays)", value=True)
with st.expander("Advanced options"):
    max_circuli = st.number_input("Maximum number of circuli per transect", min_value=1, max_value=2000,
                                  value=200, step=10)
    plot_dets = st.checkbox("Save QC images with the detections drawn on each scale and transect", value=True)
    per_image = st.checkbox("Also save the detections of each image as a separate text file", value=False)

can_start = weights_ok and folder["ok"] and angles is not None
if st.button("Start run", type="primary", disabled=not can_start):
    run_dir = start_run(folder["path"], run_name.strip(), angles, settings, overlays=overlays,
                        plot_dets=plot_dets, per_image_files=per_image, max_circuli=max_circuli)
    st.query_params["run"] = run_dir.name
    st.rerun()


# --- the run shown in this tab ----------------------------------------------------
@st.fragment(run_every="2s")
def run_panel(run_dir):
    s = run_state(run_dir, ROOT)
    status = s["status"]

    col_title, col_stop = st.columns([3, 1])
    col_title.subheader(s["name"])
    if status not in FINISHED:
        col_stop.button("Stop run", disabled=s["stop_requested"], width="stretch",
                        on_click=stop_run, args=(run_dir,))

    label = STATUS_LABELS.get(status, status)
    if s["stop_requested"]:
        label = "Stopping after the current image..."
    if status == "completed":
        st.success(f"{label}" + (f" in {s['runtime_min']} min" if s["runtime_min"] is not None else "")
                   + f". Results: `{s['run_dir']}`")
    elif status == "failed":
        st.error(f"{label}: {s['error'] or 'see the log below'}")
    elif status == "stopped":
        st.warning(f"{label}. The results written so far are kept in `{s['run_dir']}`")
    else:
        st.info(label)
    if s["input_dir"]:
        st.caption(f"Scale images: `{s['input_dir']}`")

    # progress per stage
    for stage in s["stages"]:
        done, total = stage.get("done", 0), stage.get("total", 0)
        if total:
            st.progress(min(done / total, 1.0), text=f"{stage['label']}: {done} / {total}")
        else:
            st.progress(0.0, text=f"{stage['label']}")

    # summary
    counts = s["counts"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images", counts.get("scales", "-"))
    c2.metric("Focus found", counts.get("focus_found", "-"))
    c3.metric("Transects", counts.get("transects", "-"))
    c4.metric("Circuli", counts.get("circuli", "-"))

    # QC warnings and errors
    for warning in s["warnings"]:
        first, _, rest = warning.partition("\n")
        st.warning(f"**{first.strip(' .')}**" + (f"\n\n```\n{rest.strip()}\n```" if rest.strip() else ""))

    with st.expander("Log", expanded=status == "failed"):
        st.code(s["log"] or s["console"] or "(no output yet)", language=None)
        if status == "failed" and s["console"] and s["console"] != s["log"]:
            st.caption("Console output")
            st.code(s["console"], language=None)


run_dir = shown_run()
if run_dir is not None:
    st.divider()
    run_panel(run_dir)


# --- recent runs ---------------------------------------------------------------
@st.fragment(run_every="5s")
def recent_runs_panel():
    runs = recent_runs(ROOT, n=10)
    if not runs:
        st.caption("No runs yet.")
        return
    table = pd.DataFrame(runs).drop(columns=["run_dir"])
    table["status"] = table["status"].map(lambda x: STATUS_LABELS.get(x, x).split(" (")[0])
    for col in ("images", "focus found", "circuli"):
        table[col] = table[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
    st.dataframe(table, hide_index=True, width="stretch")
    names = [r["run"] for r in runs]
    current = st.query_params.get("run")
    chosen = st.selectbox("Show run", names, index=names.index(current) if current in names else None,
                          placeholder="Choose a run to see its progress and summary")
    if chosen and chosen != current:
        st.query_params["run"] = chosen
        st.rerun()  # whole page, so the run panel above shows the chosen run


st.divider()
st.header("Recent runs")
recent_runs_panel()
