"""
SSCD web app: choose a folder of scale images and the parameters, start / stop a
run and follow its progress, summary and warnings.

Start it from the repository root (needs the optional GUI dependencies):

    uv sync --extra gui
    uv run --extra gui python -m streamlit run sscd_app.py

Runs are ordinary sscd.py runs in their own process and run folder: they keep
going if the browser tab is closed, and are listed again when the app reopens.
"""

import collections
from pathlib import Path

import pandas as pd
import streamlit as st

from sscd_libs.app_icon import install_favicon
from sscd_libs.fetch import WEIGHTS_SENTINELS, fetch_weights
from sscd_libs.folders import (
    browse_shortcuts,
    choose_folder_dialog,
    dialog_available,
    has_images,
    list_subfolders,
    path_parts,
    recent_input_dirs,
)
from sscd_libs.jobs import (
    FINISHED,
    check_input_dir,
    parse_angles,
    recent_runs,
    run_state,
    start_run,
    stop_run,
)
from sscd_libs.outputs import output_root
from sscd_libs.review_panel import focus_review_panel
from sscd_libs.settings import input_dir_allowed, load_settings, settings_path

st.set_page_config(
    page_title="SSCD - Salmon Scale Circuli Detector",
    page_icon=str(Path(__file__).parent / "assets" / "sscd_icon.png"),  # browser-tab icon
    layout="centered",
)

# for later starts (start_sscd.bat/.sh do it before the app starts); see sscd_libs/app_icon.py
install_favicon()

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
    if st.session_state.img_dir.strip() and current.is_dir() and input_dir_allowed(current, settings):
        return current.resolve()
    return browse_shortcuts(settings)[0]


def _set_browse(path):
    st.session_state.browse_dir = str(path)
    st.session_state.browse_filter = ""


def _use_folder(path):
    st.session_state.img_dir = str(path)
    _set_browse(path)


def _open_dialog():
    try:
        chosen = choose_folder_dialog(_browse_start())
    except OSError as err:
        st.session_state.dialog_error = str(err)
        return
    if chosen:
        _use_folder(chosen)


def _use_recent():
    if st.session_state.recent_dir:
        _use_folder(st.session_state.recent_dir)
    st.session_state.recent_dir = None


def _place_label(place):
    return "Home" if place == Path.home() else str(place)


if dialog_available(settings):
    col_path, col_dialog = st.columns([4, 1], vertical_alignment="bottom")
else:  # on a server the dialog would open on the server's screen
    col_path, col_dialog = st.container(), None
col_path.text_input("Scale images folder", key="img_dir",
                    help="Folder containing the scale images (.tif/.tiff/.jpg/.jpeg). It is only read, never changed.")
if col_dialog is not None:
    col_dialog.button("Choose folder...", on_click=_open_dialog, width="stretch",
                      help="Opens your computer's folder window (if you don't see it, it may be behind the browser)")
if dialog_error := st.session_state.pop("dialog_error", None):
    st.error(f"Could not open the folder window ({dialog_error}). Type the path or use *Browse folders* instead.")

recent = recent_input_dirs(ROOT, settings)
if recent:
    st.selectbox("Recent folders", recent, index=None, key="recent_dir", on_change=_use_recent,
                 placeholder=f"Folders used in earlier runs ({len(recent)})")

MAX_SUBFOLDER_BUTTONS = 100
with st.expander("Browse folders"):
    if "browse_dir" not in st.session_state:
        st.session_state.browse_dir = str(_browse_start())
    browse = Path(st.session_state.browse_dir)

    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        st.caption("Go to:", width="content")
        for place in browse_shortcuts(settings):
            st.button(_place_label(place), key=f"place:{place}", on_click=_set_browse, args=(place,),
                      icon=":material/home:" if place == Path.home() else ":material/hard_drive:")

    # the current path: each part opens that folder
    with st.container(horizontal=True, gap=None, vertical_alignment="center"):
        for i, part in enumerate(path_parts(browse, settings)):
            if i:
                st.markdown(":gray[/]", width="content")
            st.button(part.name or str(part).rstrip("\\/"), key=f"part:{part}", on_click=_set_browse,
                      args=(part,), type="tertiary")

    subfolders = list_subfolders(browse)
    if len(subfolders) > 30:
        wanted = st.text_input("Filter sub-folders", key="browse_filter", placeholder="Part of the folder name")
        subfolders = [p for p in subfolders if wanted.strip().lower() in p.name.lower()]
    if subfolders:
        with st.container(horizontal=True, gap="small"):
            for p in subfolders[:MAX_SUBFOLDER_BUTTONS]:
                st.button(p.name, key=f"sub:{p}", on_click=_set_browse, args=(p,), icon=":material/folder:")
        if len(subfolders) > MAX_SUBFOLDER_BUTTONS:
            st.caption(f"... and {len(subfolders) - MAX_SUBFOLDER_BUTTONS} more - use the filter above.")
    else:
        st.caption("No sub-folders.")

    images_here = has_images(browse)
    st.button("Use this folder", on_click=_use_folder, args=(browse,), type="primary", disabled=not images_here,
              help=None if images_here else "There are no scale images directly in this folder")

@st.cache_data(ttl=60, show_spinner="Checking the folder...")
def _check_folder(path, allowed_roots):
    # cached briefly: the check reads parts of the image files (duplicate detection),
    # which is slow on a network share if repeated on every click
    return check_input_dir(path, {**settings, "allowed_input_roots": list(allowed_roots)})


folder = _check_folder(st.session_state.img_dir, tuple(settings["allowed_input_roots"]))
if st.session_state.img_dir.strip():
    (st.success if folder["ok"] else st.error)(folder["message"])
if folder.get("duplicates"):
    n = sum(len(g) - 1 for g in folder["duplicates"])
    listed = "\n".join("- " + " = ".join(f"`{name}`" for name in group) for group in folder["duplicates"])
    st.warning(f"**{n} image(s) are identical copies of another image in this folder** and would be "
               f"processed - and counted in the results - twice. Consider removing the copies:\n\n{listed}")

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
                                  value=500, step=10)
    plot_dets = st.checkbox("Save QC images with the detections drawn on each scale and transect", value=True)
    per_image = st.checkbox("Also save the detections of each image as a separate text file", value=False)
    focus_retry = st.checkbox(
        "Second focus pass for scales where no focus was found", value=True,
        help="Retries those scales padded to the aspect ratio of the training images, then accepts the "
             "best focus above a lower score threshold. Scales found this way are flagged (focus_method).")
    focus_low_threshold = st.number_input(
        "Lower score threshold for the second pass", min_value=0.0, max_value=0.5, value=0.1, step=0.05,
        disabled=not focus_retry, help="0 skips this step (only the padded retry is done)")

can_start = weights_ok and folder["ok"] and angles is not None
if st.button("Start run", type="primary", disabled=not can_start):
    run_dir = start_run(folder["path"], run_name.strip(), angles, settings, overlays=overlays,
                        plot_dets=plot_dets, per_image_files=per_image, max_circuli=max_circuli,
                        focus_retry=focus_retry, focus_low_threshold=focus_low_threshold)
    st.query_params["run"] = run_dir.name
    st.rerun()


# --- the run shown in this tab ----------------------------------------------------
def _show_warning(message):
    first, _, rest = message.partition("\n")
    st.warning(f"**{first.strip(' .')}**" + (f"\n\n```\n{rest.strip()}\n```" if rest.strip() else ""))


@st.fragment(run_every="2s")
def run_panel(run_dir):
    s = run_state(run_dir, ROOT)
    status = s["status"]
    # when a run (or applying focus corrections) finishes, refresh the whole page, so the
    # focus review below shows the new state
    seen = st.session_state.setdefault("run_status_seen", {})
    previous, seen[run_dir.name] = seen.get(run_dir.name), status
    if previous is not None and previous != status and status in FINISHED:
        st.rerun()

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

    # warnings (QC) and log lines are shown under the stage they come from; the others first
    names = {stage["name"] for stage in s["stages"]}
    by_stage = collections.defaultdict(list)
    for w in s["warnings"]:
        by_stage[w["stage"] if w["stage"] in names else None].append(w["message"])
    for message in by_stage[None]:
        _show_warning(message)

    # progress per stage
    for stage in s["stages"]:
        done, total = stage.get("done", 0), stage.get("total", 0)
        text = f"{stage['label']}: {done} / {total}" if total else stage["label"]
        if stage.get("failed"):  # e.g. scales without a focus, transects without circuli
            text += f" &nbsp; :red[**{stage['failed']} {stage.get('failed_label', 'failed')}**]"
        st.progress(min(done / total, 1.0) if total else 0.0, text=text)
        for message in by_stage[stage["name"]]:
            _show_warning(message)
        if log := s["stage_logs"].get(stage["name"]):
            with st.expander(f"Log: {stage['label']}"):
                st.code(log, language=None)

    # summary
    counts = s["counts"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images", counts.get("scales", "-"))
    c2.metric("Focus found", counts.get("focus_found", "-"))
    c3.metric("Transects", counts.get("transects", "-"))
    c4.metric("Circuli", counts.get("circuli", "-"))

    with st.expander("Full log", expanded=status == "failed"):
        st.code(s["log"] or s["console"] or "(no output yet)", language=None)
        if status == "failed" and s["console"] and s["console"] != s["log"]:
            st.caption("Console output")
            st.code(s["console"], language=None)


run_dir = shown_run()
if run_dir is not None:
    st.divider()
    run_panel(run_dir)
    focus_review_panel(run_dir)


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
