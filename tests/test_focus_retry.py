"""The second focus pass for scales without a focus."""

import pandas as pd
import pytest
from PIL import Image

from sscd_libs.focus_retry import TRAINING_ASPECT, best_focus_box, pad_to_training_aspect, retry_focus


@pytest.mark.parametrize("size", [(3840, 2160), (1000, 1500), (3840, 2748)])
def test_padding_reaches_training_aspect_and_keeps_the_image(size):
    im = Image.new("RGB", size, (200, 210, 220))
    im.putpixel((5, 7), (255, 0, 0))
    padded, (dx, dy) = pad_to_training_aspect(im)
    assert padded.height / padded.width == pytest.approx(TRAINING_ASPECT, abs=1e-3)
    assert padded.getpixel((5 + dx, 7 + dy)) == (255, 0, 0)   # original sits at the offset
    assert padded.getpixel((0, 0)) == (200, 210, 220)          # padding in the background colour


def test_boxes_map_back_and_passes_are_flagged(tmp_path):
    """Pass A (padded) boxes are shifted back to the original image; pass B only sees what
    pass A did not find; each rescued scale gets exactly one box and its focus_method."""
    for sid in ("a", "b", "c"):
        Image.new("RGB", (3840, 2160), (128, 128, 128)).save(tmp_path / f"{sid}.jpg")
    dy = (round(3840 * TRAINING_ASPECT) - 2160) // 2
    calls = []

    def fake_detect(img_dir, no_det_dir, threshold, progress):
        calls.append((img_dir, threshold))
        if threshold is None:   # pass A: 'a' found (two boxes; the best one wins), in padded coordinates
            return pd.DataFrame({"img_id": ["a", "a", "b", "c"], "score": [0.9, 0.6, None, None],
                                 "xmin": [100, 500, None, None], "ymin": [dy + 200, dy + 900, None, None],
                                 "xmax": [150, 550, None, None], "ymax": [dy + 260, dy + 960, None, None]})
        return pd.DataFrame({"img_id": ["b", "c"], "score": [0.2, None],   # pass B: 'b' found
                             "xmin": [10, None], "ymin": [20, None], "xmax": [60, None], "ymax": [80, None]})

    out = retry_focus(["a", "b", "c"], tmp_path, tmp_path / "work", fake_detect, 0.1).set_index("img_id")
    assert sorted(out.index) == ["a", "b"]
    assert out.loc["a", "focus_method"] == "padded"
    assert (out.loc["a", "xmin"], out.loc["a", "ymin"], out.loc["a", "ymax"]) == (100, 200, 260)
    assert out.loc["b", "focus_method"] == "low_threshold" and out.loc["b", "score"] == 0.2
    assert [t for _, t in calls] == [None, 0.1]
    assert sorted(p.stem for p in (tmp_path / "work" / "low_threshold").iterdir()) == ["b", "c"]


def test_best_focus_box_keeps_most_confident_and_counts_boxes():
    dets = pd.DataFrame({"img_id": ["b", "a", "a", "c"], "score": [0.9, 0.6, 0.8, None],
                         "xmin": [1, 2, 3, None]})
    best = best_focus_box(dets)
    assert best["img_id"].tolist() == ["b", "a"]          # input order kept, no-detection image dropped
    assert best["score"].tolist() == [0.9, 0.8]
    assert best["n_focus_boxes"].tolist() == [1, 2]
