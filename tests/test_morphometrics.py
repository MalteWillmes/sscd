"""Scale outline and morphometrics (sscd_libs/morphometrics.py), on synthetic scales with known geometry."""

import math

import cv2
import numpy as np
import pytest

from sscd_libs.morphometrics import feret_diameters, find_outline, measure, radius

W, H = 3000, 2000


def _slide(seed=0):
    """A bright slide with vignetting and some fine noise."""
    yy, xx = np.mgrid[0:H, 0:W]
    vignette = 230 - 40 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)
    noise = np.random.default_rng(seed).normal(0, 2, (H, W))
    return np.clip(vignette + noise, 0, 255)


def _scale(img, centre, axes, ring_px=14):
    """A 'scale': an ellipse, slightly darker than the slide, with concentric rings (circuli)."""
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, centre, axes, 0, 0, 360, 255, cv2.FILLED)
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot((xx - centre[0]) / axes[0], (yy - centre[1]) / axes[1]) * min(axes)
    rings = np.where((r % ring_px) < 3, 90, 0)
    inside = mask > 0
    img[inside] = img[inside] - 25 - rings[inside]
    return img


def _rgb(gray):
    return np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


def test_ellipse_area_diameters_and_radii():
    a, b, centre = 600, 400, (1500, 1000)
    rgb = _rgb(_scale(_slide(), centre, (a, b)))
    points = find_outline(rgb, focus=centre)
    m = measure(points, (W, H), focus=centre, angles=(0, 90, 180))
    assert m["outline_flags"] == ""
    assert m["area_px2"] == pytest.approx(math.pi * a * b, rel=0.03)
    assert m["length_px"] == pytest.approx(2 * a, rel=0.02)          # maximum Feret diameter
    assert m["width_px"] == pytest.approx(2 * b, rel=0.02)           # minimum Feret diameter
    assert m["solidity"] > 0.97
    assert m["radius_0_px"] == pytest.approx(a, rel=0.02)
    assert m["radius_90_px"] == pytest.approx(b, rel=0.03)            # up
    assert m["radius_180_px"] == pytest.approx(a, rel=0.02)


def test_scale_cut_off_by_the_image_border_is_flagged():
    rgb = _rgb(_scale(_slide(), (1500, 1800), (600, 400)))
    m = measure(find_outline(rgb), (W, H))
    assert "touches the image border" in m["outline_flags"]


def test_focus_picks_the_scale_among_several_objects():
    gray = _scale(_slide(), (700, 1000), (500, 500))
    rgb = _rgb(_scale(gray, (2300, 1000), (300, 300)))
    points = find_outline(rgb, focus=(2300, 1000))
    assert measure(points, (W, H))["length_px"] == pytest.approx(600, rel=0.03)
    assert measure(find_outline(rgb), (W, H))["length_px"] == pytest.approx(1000, rel=0.03)  # else the largest


def test_feret_and_radius_on_simple_polygons():
    rectangle = np.array([[0, 0], [300, 0], [300, 100], [0, 100]], float)
    assert feret_diameters(rectangle) == pytest.approx((math.hypot(300, 100), 100))
    square = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], float)
    assert radius(square, (100, 100), 0) == pytest.approx(100)
    assert radius(square, (100, 100), 45) == pytest.approx(100 * math.sqrt(2))
    assert radius(square, (50, 150), 90) == pytest.approx(150)      # up = smaller y
    assert math.isnan(radius(square, (300, 100), 0))                # focus outside


def test_blank_image_has_no_outline():
    m = measure(find_outline(_rgb(np.full((H, W), 220.0))), (W, H), angles=(0,))
    assert not m["outline_found"] and m["outline_flags"] == "no outline found" and m["radius_0_px"] is None
