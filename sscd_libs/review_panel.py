"""
The web app's focus review (sscd_app.py): set the focus by hand on the scales of a
completed run where SSCD found no focus, apply the corrections, and export them as
training annotations. The logic lives in sscd_libs/manual_focus.py and manual_focus.py.
"""

from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from sscd_libs import manual_focus as mf
from sscd_libs.jobs import start_manual_focus
from sscd_libs.outputs import open_run
from sscd_libs.runcontrol import read_json

OVERVIEW_WIDTH = 900   # px the whole scale is shown at
ZOOM_SIDE = 320        # px of the scale image shown in the enlarged view
ZOOM_WIDTH = 640       # px the enlarged view is shown at (2x)
MARK = (230, 30, 30)

STATE_LABELS = {
    "pending": "to review",
    "focus": "focus set",
    "no_focus": "no usable focus",
    "applied": "applied",
}


@st.cache_resource(max_entries=2, show_spinner=False)
def _load_scale(path, mtime):
    with Image.open(path) as im:
        full = im.convert("RGB")
    overview = full.copy()
    overview.thumbnail((OVERVIEW_WIDTH, OVERVIEW_WIDTH * 2))
    return full, overview


def _marked(im, point, box, scale, offset=(0, 0)):
    """A copy of `im` with the focus point and its box drawn (point/box in scale-image pixels)."""
    im = im.copy()
    if point is None:
        return im
    d = ImageDraw.Draw(im)
    x, y = (point[0] - offset[0]) * scale, (point[1] - offset[1]) * scale
    hw, hh = box[0] / 2 * scale, box[1] / 2 * scale
    d.rectangle([x - hw, y - hh, x + hw, y + hh], outline=MARK, width=2)
    r = max(6, hw)
    d.line([x - r * 1.6, y, x - r * 0.6, y], fill=MARK, width=2)
    d.line([x + r * 0.6, y, x + r * 1.6, y], fill=MARK, width=2)
    d.line([x, y - r * 1.6, x, y - r * 0.6], fill=MARK, width=2)
    d.line([x, y + r * 0.6, x, y + r * 1.6], fill=MARK, width=2)
    return im


def _new_click(key, click):
    """True once per click of an image (the component keeps returning its last click)."""
    if not click:
        return False
    stamp = click.get("unix_time")
    if st.session_state.get(f"{key}_seen") == stamp:
        return False
    st.session_state[f"{key}_seen"] = stamp
    return True


def _points():
    return st.session_state.setdefault("mf_points", {})


def _next_pending(queue, after):
    ids = [q["scale_id"] for q in queue if q["state"] == "pending"]
    later = [s for s in ids if s != after]
    return later[0] if later else None


def _save(paths, queue, scale_id, decision, point=None, note=""):
    try:
        if decision == "focus":
            mf.save_decision(paths, scale_id, "focus", *point)
        else:
            mf.save_decision(paths, scale_id, "no_focus", note=note)
    except ValueError as err:
        st.session_state.mf_error = str(err)
        return
    nxt = _next_pending(queue, scale_id)
    if nxt:
        st.session_state.mf_scale = nxt


def _undo(paths, scale_id):
    try:
        mf.remove_decision(paths, scale_id)
    except ValueError as err:
        st.session_state.mf_error = str(err)
    _points().pop(scale_id, None)


def _apply(run_dir):
    start_manual_focus(run_dir)
    st.session_state.mf_applying = True


@st.fragment
def focus_review_panel(run_dir):
    """Focus review for the run in `run_dir` (shown when it has scales without a focus)."""
    try:
        paths = open_run(run_dir)
    except FileNotFoundError:
        return  # a run that is still starting has no run_info.json yet
    info = read_json(paths.info) or {}
    queue = mf.review_queue(paths)
    if not queue:
        return

    st.divider()
    st.subheader("Focus review")
    open_items = [q for q in queue if q["state"] != "applied"]
    n_applied = len(queue) - len(open_items)
    st.caption(f"{len(queue)} scale(s) where SSCD found no focus: set the focus by hand, or mark the scale as "
               f"having no usable focus. Corrected scales are added to this run's results "
               f"(focus_method = manual). {n_applied} applied so far.")

    if info.get("status") != "completed":
        st.info("Focus corrections are being applied - see the progress above.")
        return
    if err := st.session_state.pop("mf_error", None):
        st.error(err)

    if open_items:
        _review_one(paths, run_dir, queue, open_items, info)
    else:
        st.success("All scales without a focus have been reviewed and applied.")
    _export(paths, info)


def _review_one(paths, run_dir, queue, open_items, info):
    ids = [q["scale_id"] for q in open_items]
    if st.session_state.get("mf_scale") not in ids:
        st.session_state.mf_scale = next((q["scale_id"] for q in open_items if q["state"] == "pending"), ids[0])
    states = {q["scale_id"]: q for q in open_items}
    scale_id = st.selectbox("Scale", ids, key="mf_scale",
                            format_func=lambda s: f"{s}  ({STATE_LABELS[states[s]['state']]})")
    item = states[scale_id]

    points = _points()
    if scale_id not in points and item["decision"] == "focus":
        points[scale_id] = (int(item["x_px"]), int(item["y_px"]))
    point = points.get(scale_id)

    jpg = mf.scale_jpg(paths, info["input_dir"], scale_id)
    full, overview = _load_scale(str(jpg), jpg.stat().st_mtime)
    box = mf.focus_box_size(paths, full.width)
    to_overview = overview.width / full.width

    st.caption("Click the centre of the focus. Then refine it on the enlarged view on the right, if needed.")
    col_all, col_zoom = st.columns([3, 2])
    with col_all:
        click = streamlit_image_coordinates(_marked(overview, point, box, to_overview),
                                            key=f"mf_over_{scale_id}", width="stretch", cursor="crosshair")
        if _new_click(f"mf_over_{scale_id}", click):
            k = full.width / click["width"]
            point = points[scale_id] = (int(round(click["x"] * k)), int(round(click["y"] * k)))
            st.rerun(scope="fragment")
    with col_zoom:
        if point is None:
            st.info("The enlarged view appears after the first click.")
        else:
            x0 = min(max(point[0] - ZOOM_SIDE // 2, 0), max(full.width - ZOOM_SIDE, 0))
            y0 = min(max(point[1] - ZOOM_SIDE // 2, 0), max(full.height - ZOOM_SIDE, 0))
            crop = full.crop((x0, y0, x0 + ZOOM_SIDE, y0 + ZOOM_SIDE)).resize((ZOOM_WIDTH, ZOOM_WIDTH))
            zoom_click = streamlit_image_coordinates(
                _marked(crop, point, box, ZOOM_WIDTH / ZOOM_SIDE, offset=(x0, y0)),
                key=f"mf_zoom_{scale_id}", width="stretch", cursor="crosshair")
            if _new_click(f"mf_zoom_{scale_id}", zoom_click):
                k = ZOOM_SIDE / zoom_click["width"]
                points[scale_id] = (int(round(x0 + zoom_click["x"] * k)), int(round(y0 + zoom_click["y"] * k)))
                st.rerun(scope="fragment")
            st.caption(f"Focus at x = {point[0]}, y = {point[1]} px. The box ({box[0]} x {box[1]} px, the "
                       "median of this run's detected foci) sets the transect width.")

    col_set, col_none = st.columns(2)
    with col_set:
        st.button("Set focus here", type="primary", disabled=point is None, width="stretch",
                  on_click=_save, args=(paths, queue, scale_id, "focus", point))
    with col_none:
        with st.popover("No usable focus", width="stretch"):
            reason = st.text_input("Reason (recorded in scales_summary.csv)", key=f"mf_reason_{scale_id}",
                                   placeholder="e.g. regenerated scale, focus damaged")
            st.button("Mark as no usable focus", key=f"mf_none_{scale_id}",
                      on_click=_save, args=(paths, queue, scale_id, "no_focus", None, reason))
    if item["state"] in ("focus", "no_focus"):
        st.button("Undo the decision for this scale", type="tertiary", on_click=_undo, args=(paths, scale_id))

    decided = [q for q in open_items if q["state"] in ("focus", "no_focus")]
    n_left = sum(q["state"] == "pending" for q in open_items)
    st.write(f"**{len(decided)}** decided, **{n_left}** still to review.")
    st.button(f"Apply {len(decided)} correction(s)", disabled=not decided, on_click=_apply, args=(run_dir,),
              help="Cuts the transects and detects the circuli for the corrected scales and adds them to this "
                   "run's results. Runs in the background, like a run.")


def _export(paths, info):
    with st.expander("Export as training data"):
        st.caption("Writes the manually set foci as Pascal VOC annotations, with the scale images, to "
                   "`annotations/focus/` in the run folder: training data for the focus detector, and "
                   "ground truth for eval_detector.py.")
        include = st.checkbox("Also export the foci SSCD detected itself (marked as unverified - check them "
                              "before training)", key="mf_export_detected")
        if st.button("Export focus annotations"):
            with st.spinner("Exporting..."):
                out, n = mf.export_annotations(paths, info["input_dir"], include_detected=include)
            st.success(f"{n} annotation(s) written to `{Path(out)}`")
