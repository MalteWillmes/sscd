"""
Scale outline and basic scale morphometrics.

The outline is found without a trained model, on a copy of the scale image reduced to
WORK_SIZE pixels (longest side):

  1. texture: local gradient energy - the circuli and the scale's surface make the
     scale far "busier" than the slide (Otsu threshold);
  2. darkness: the slide background, vignetting included, is fitted as a smooth
     surface on the untextured pixels; pixels clearly darker than it are part of the
     scale (this adds the smooth posterior field, which has few circuli);
  3. the union is closed and opened, the region containing the focus (else the
     largest) is kept with its holes filled, and thin attachments - debris and
     fibres touching the scale - are cut off by an opening of OPEN_ATTACH.

Measurements are in pixels of the scale image (not calibrated):

  area_px2, perimeter_px   of the outline
  length_px, width_px      maximum and minimum Feret diameter (caliper widths), so
                           independent of how the scale lies in the image
  solidity                 area / area of the convex hull (1 = convex)
  radius_<angle>_px        distance from the focus to the outline along each
                           transect angle (0 = right, 90 = up), to the first crossing

Flags (outline_flags) mark outlines to check: touching the image border (the scale
is cut off, so area and length are too small), irregular (solidity below
SOLIDITY_MIN), or the focus outside the outline.
"""

import math

import cv2
import numpy as np

WORK_SIZE = 1200          # px, longest side the outline is found at
TEXTURE_SIGMA = 0.012     # smoothing of the texture energy, fraction of WORK_SIZE
CLOSE = 0.02              # closing/opening of the scale region, fraction of WORK_SIZE
OPEN_ATTACH = 0.06        # opening that cuts off thin attachments, fraction of WORK_SIZE
SOLIDITY_MIN = 0.90       # below: "irregular outline" (VI scales: 0.90-0.99; notched Tummel 60300: 0.87)
EDGE_MARGIN = 2           # px; an outline this close to the image border touches it


def _kernel(fraction):
    size = int(WORK_SIZE * fraction) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _region(mask, seed):
    """The connected region containing `seed` (else nearest to it, else the largest),
    holes filled. Returns (region mask, contour) or (None, None)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    if seed is not None:
        contour = max(contours, key=lambda c: cv2.pointPolygonTest(c, seed, True))
    else:
        contour = max(contours, key=cv2.contourArea)
    region = np.zeros_like(mask)
    cv2.drawContours(region, [contour], -1, 255, cv2.FILLED)
    return region, contour


def find_outline(rgb, focus=None):
    """
    The outline of the scale in an RGB image (H x W x 3, uint8), as an (N, 2) float array
    of x, y in image pixels, or None. `focus` (x, y), if known, picks the scale when
    there is more than one object.
    """
    h, w = rgb.shape[:2]
    f = WORK_SIZE / max(h, w)
    small = cv2.resize(rgb, (max(round(w * f), 1), max(round(h * f), 1)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 1. texture
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    energy = cv2.GaussianBlur(gradient, (0, 0), WORK_SIZE * TEXTURE_SIGMA)
    energy8 = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, texture = cv2.threshold(energy8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. darkness relative to the fitted slide background
    mask = texture
    background = cv2.dilate(texture, np.ones((41, 41), np.uint8)) == 0
    residual = None
    if background.sum() > 1000:
        yy, xx = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
        xn, yn = xx / gray.shape[1] - 0.5, yy / gray.shape[0] - 0.5
        terms = [np.ones_like(xn), xn, yn, xn * xn, yn * yn, xn * yn, xn ** 3, yn ** 3, xn * xn * yn, xn * yn * yn]
        smooth = cv2.GaussianBlur(gray, (0, 0), 3)
        coef, *_ = np.linalg.lstsq(np.stack([t[background] for t in terms], 1), smooth[background], rcond=None)
        residual = sum(c * t for c, t in zip(coef, terms, strict=True)) - smooth
        noise = np.median(np.abs(residual[background])) * 1.4826
        dark = cv2.GaussianBlur(residual.astype(np.float32), (0, 0), 4) > max(4 * noise, 6)
        mask = cv2.bitwise_or(texture, dark.astype(np.uint8) * 255)

    # 3. the scale region, cleaned
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(CLOSE), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(CLOSE))
    seed = None if focus is None else (float(focus[0]) * f, float(focus[1]) * f)
    region, _ = _region(mask, seed)
    if region is None:
        return None
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, _kernel(OPEN_ATTACH))

    # 4. refine the edge: the smoothing above spreads strong features (e.g. the dark
    #    outermost circuli) past the scale's edge; place the edge with evidence that does not
    #    spread - the raw gradient (trimmed by the 1 px the Sobel filter reaches) and darkness
    #    after a 3x3 median - closed to bridge the gaps between circuli
    if residual is not None:
        def strong(values):
            ref = values[background]
            return values > np.median(ref) + 4 * np.median(np.abs(ref - np.median(ref))) * 1.4826

        edges = cv2.erode(strong(gradient).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        fit = sum(c * t for c, t in zip(coef, terms, strict=True))
        darker = fit - cv2.medianBlur(np.clip(gray, 0, 255).astype(np.uint8), 3).astype(np.float32)
        evidence = (edges | (strong(darker) & (darker > 6))).astype(np.uint8) * 255
        evidence = cv2.morphologyEx(evidence, cv2.MORPH_CLOSE, _kernel(CLOSE), iterations=2)
        refined, _ = _region(cv2.bitwise_and(region, evidence), seed)
        if refined is not None and refined.sum() > 0.5 * region.sum():   # never lose most of the scale
            region = cv2.morphologyEx(refined, cv2.MORPH_OPEN, _kernel(CLOSE))
    _, contour = _region(region, seed)
    if contour is None or cv2.contourArea(contour) < 100:
        return None
    contour = cv2.approxPolyDP(contour, 0.7, True)   # removes the pixel staircase
    # pixel centres: work pixel i covers image pixels [i/f, (i+1)/f)
    return (contour.reshape(-1, 2).astype(np.float64) + 0.5) / f


# ------------------------------------------------------------------------------
def feret_diameters(points):
    """(maximum, minimum) Feret diameter of a polygon: its largest and smallest caliper
    width, from the convex hull (rotating calipers for the minimum)."""
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    if len(hull) < 3:
        return 0.0, 0.0
    d = hull[:, None, :] - hull[None, :, :]
    maximum = float(np.sqrt((d ** 2).sum(-1)).max())
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    normals = np.stack([-edges[:, 1], edges[:, 0]], 1) / lengths[:, None]
    # for each hull edge: the largest distance of the hull from that edge's line
    widths = np.abs(((hull[None, :, :] - hull[:, None, :]) * normals[:, None, :]).sum(-1)).max(1)
    return maximum, float(widths[lengths > 0].min())


def radius(points, focus, angle_deg):
    """Distance from `focus` to the outline along `angle_deg` (0 = right, 90 = up, as the
    transects), to the first crossing; NaN if the focus is outside the outline."""
    if cv2.pointPolygonTest(points.astype(np.float32).reshape(-1, 1, 2), (float(focus[0]), float(focus[1])),
                            False) < 0:
        return math.nan
    a = math.radians(angle_deg)
    direction = np.array([math.cos(a), -math.sin(a)])   # image y points down
    p, q = points, np.roll(points, -1, axis=0)
    seg = q - p
    rel = p - np.asarray(focus, dtype=np.float64)
    denom = direction[0] * seg[:, 1] - direction[1] * seg[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (rel[:, 0] * seg[:, 1] - rel[:, 1] * seg[:, 0]) / denom     # along the ray
        u = (rel[:, 0] * direction[1] - rel[:, 1] * direction[0]) / denom  # along the edge
    hits = t[(np.abs(denom) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)]
    return float(hits.min()) if len(hits) else math.nan


def measure(points, image_size, focus=None, angles=()):
    """Morphometrics of an outline (see the module docstring); `points` None gives the
    "no outline" result."""
    result = {"outline_found": points is not None, "area_px2": None, "perimeter_px": None, "length_px": None,
              "width_px": None, "solidity": None}
    result.update({f"radius_{a}_px": None for a in angles})
    if points is None:
        result["outline_flags"] = "no outline found"
        return result
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    length, width = feret_diameters(points)
    result.update(area_px2=round(area), perimeter_px=round(cv2.arcLength(contour, True), 1),
                  length_px=round(length, 1), width_px=round(width, 1),
                  solidity=round(area / hull_area, 3) if hull_area else None)
    flags = []
    w, h = image_size
    if (points[:, 0].min() <= EDGE_MARGIN or points[:, 1].min() <= EDGE_MARGIN
            or points[:, 0].max() >= w - EDGE_MARGIN or points[:, 1].max() >= h - EDGE_MARGIN):
        flags.append("touches the image border")
    if result["solidity"] is not None and result["solidity"] < SOLIDITY_MIN:
        flags.append("irregular outline")
    if focus is not None:
        for a in angles:
            r = radius(points, focus, a)
            result[f"radius_{a}_px"] = None if math.isnan(r) else round(r, 1)
        if all(result[f"radius_{a}_px"] is None for a in angles) and angles:
            flags.append("focus outside the outline")
    result["outline_flags"] = "; ".join(flags)
    return result


def draw_outline(rgb, points, focus=None, angles=(), max_side=1600):
    """A reduced copy of the image with the outline, Feret length and focus radii drawn
    (QC image). Returns an RGB array."""
    h, w = rgb.shape[:2]
    f = min(1.0, max_side / max(h, w))
    vis = cv2.resize(rgb, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA).copy()
    if points is None:
        return vis
    pts = np.round(points * f).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], True, (230, 30, 30), 3, cv2.LINE_AA)
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    d = ((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(d.argmax(), d.shape)
    cv2.line(vis, tuple(np.round(hull[i] * f).astype(int)), tuple(np.round(hull[j] * f).astype(int)),
             (30, 120, 230), 2, cv2.LINE_AA)
    if focus is not None:
        fx, fy = focus
        for a in angles:
            r = radius(points, focus, a)
            if not math.isnan(r):
                ex, ey = fx + r * math.cos(math.radians(a)), fy - r * math.sin(math.radians(a))
                cv2.line(vis, (round(fx * f), round(fy * f)), (round(ex * f), round(ey * f)), (250, 200, 0), 2,
                         cv2.LINE_AA)
    return vis
