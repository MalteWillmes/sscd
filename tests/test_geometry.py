"""Mapping transect-image coordinates back onto the scale image."""

import numpy as np
import pytest
from PIL import Image

from sscd_libs.data_processing import transect_geometry, transect_to_image
from tools.diagonal_crop import crop

W, H = 1200, 900
FOCUS = {"xmin": 560.0, "ymin": 410.0, "xmax": 610.0, "ymax": 470.0}


@pytest.fixture(scope="module")
def noise():
    """Random 24-bit colours: (nearly) every pixel value is unique, so it identifies its position."""
    return np.random.default_rng(1).integers(0, 256, (H, W, 3), dtype=np.uint8)


@pytest.mark.parametrize("angle", [0, 30, 45, 90, 135, 180, 200, 250, 270, 315, 359])
def test_transect_pixels_map_back_to_their_source(noise, angle):
    """Cut a transect with the real crop code, find where each transect pixel really came
    from, and compare with transect_to_image: within the crop's own pixel rounding."""
    key = noise[..., 0].astype(np.int64) << 16 | noise[..., 1].astype(np.int64) << 8 | noise[..., 2]
    geom = transect_geometry(FOCUS, angle, W, H)
    t = np.asarray(crop(Image.fromarray(noise), geom.base, geom.angle_rad, geom.width, geom.length)).astype(np.int64)
    tkey = t[..., 0] << 16 | t[..., 1] << 8 | t[..., 2]
    v, u = np.mgrid[0:t.shape[0], 0:t.shape[1]]
    x, y = transect_to_image(geom, u + 0.5, v + 0.5)

    errors = []
    for xi, yi, k in zip(x.ravel()[::11], y.ravel()[::11], tkey.ravel()[::11], strict=True):
        cx, cy = int(xi), int(yi)
        window = key[max(cy - 3, 0):cy + 4, max(cx - 3, 0):cx + 4]
        hits = np.argwhere(window == k)
        if len(hits):  # pixels from outside the image (black fill) have no source
            hy, hx = hits[0] + [max(cy - 3, 0), max(cx - 3, 0)]
            errors.append(np.hypot(hx + 0.5 - xi, hy + 0.5 - yi))
    errors = np.array(errors)
    assert len(errors) > 100
    assert np.median(errors) < 0.8
    assert errors.max() < 1.5


def test_transect_starts_at_focus_centre():
    """u (along the transect) is the distance from the focus centre, for any angle."""
    cx, cy = (FOCUS["xmin"] + FOCUS["xmax"]) / 2, (FOCUS["ymin"] + FOCUS["ymax"]) / 2
    for angle in range(0, 360, 15):
        geom = transect_geometry(FOCUS, angle)
        x, y = transect_to_image(geom, 100.0, geom.width / 2)  # 100 px out, on the centre line
        assert np.hypot(x - cx, y - cy) == pytest.approx(100.0)
