"""Qt-free core of the cube overlay viewer (hsapp.viewer.overlay_view): several cubes on one set of axes.

Conventions
    Everything the user points at (clicked pixel, ROI) lives in the REFERENCE frame, in um.  Every cube carries a shift
    (sx, sy) in um = where a feature of the reference appears in that cube minus where it is in the reference, so the
    reference point (x, y) is read in cube k at (x + sx_k, y + sy_k) with cube k's own pixel size.  Cubes with different
    pixel sizes (objectives) or image sizes can therefore share spectra; image overlays need the same grid.
    Spectra are always plotted against each cube's OWN wavelengths (raw 632-886 nm in 128 frames and registered 670-850 nm
    in 91 frames mix freely); a frame is chosen by wavelength, never by index.
    Metrics (peak / centroid / integral / max) are computed on the un-normalised spectrum inside a wavelength range.

Nothing here touches hardware, Qt, the hsapp settings file or the hsapp locks.
"""
import glob
import os
import re
from datetime import datetime

import numpy as np

from .cubefile import CubeFile, CubeFormatError, auto_levels

# Okabe-Ito (colour-blind safe), black replaced by a light grey for the dark canvas; assigned in this fixed order.
DISTINCT = ('#E69F00', '#56B4E9', '#009E73', '#F0E442', '#CC79A7', '#D55E00', '#0072B2', '#BBBBBB')
# viridis anchors (matplotlib), used from 0.12 on so the first cube of a series is not black on the dark canvas
_VIRIDIS = ((0.0, (68, 1, 84)), (0.125, (71, 44, 122)), (0.25, (59, 81, 139)), (0.375, (44, 113, 142)),
            (0.5, (33, 144, 141)), (0.625, (39, 173, 129)), (0.75, (92, 200, 99)), (0.875, (170, 220, 50)),
            (1.0, (253, 231, 37)))
METRICS = ('peak λ (nm)', 'centroid λ (nm)', 'integral in range', 'maximum')
METRIC_KEYS = ('peak_nm', 'centroid_nm', 'integral', 'max')
_TIME_FORMATS = ('%Y/%m/%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
                 '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S')


# ---------------------------------------------------------------------------------------------------------------- files
def natural_key(path):
    """Sort key: digits compare as numbers ('r2' < 'r10'), case ignored."""
    name = os.path.basename(str(path)).lower()
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]


def expand_paths(items):
    """File names and glob patterns -> existing .h5 paths, each pattern naturally sorted, duplicates dropped, order kept.

    Only '*' and '?' make a pattern (square brackets and parentheses in file names stay literal)."""
    out, seen = [], set()
    for item in items:
        item = str(item).strip().strip('"')
        if not item:
            continue
        if '*' in item or '?' in item:
            pat = item.replace('[', '[[]')                       # literal '[' for glob
            found = sorted(glob.glob(pat), key=natural_key)
        else:
            found = [item]
        for p in found:
            key = os.path.normcase(os.path.abspath(p))
            if key not in seen:
                seen.add(key)
                out.append(os.path.abspath(p))
    return out


def raw_sibling(path):
    """The raw cube that a processed file was made from ('x_dc(2).h5', 'x(1).h5', 'x_pc(1).h5' -> 'x.h5' in the same folder),
    or None when it does not exist or `path` is that raw file itself."""
    folder, name = os.path.split(os.path.abspath(path))
    stem, ext = os.path.splitext(name)
    base = re.sub(r'\(\d\)', '', stem)
    base = re.sub(r'_(dc|pc)(?=[(_.]|$)', '', base)
    base = re.sub(r'_corrected$', '', base)
    cand = os.path.join(folder, base + ext)
    if os.path.normcase(cand) == os.path.normcase(os.path.abspath(path)) or not os.path.isfile(cand):
        return None
    return cand


def _parse_time(text):
    text = str(text).strip()
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _time_from(cf):
    t = _parse_time(cf.info.get('Cube/CreationDate', '')) if cf.info.get('Cube/CreationDate') else None
    if t is not None:
        return t, 'Cube/CreationDate'
    if cf.timestamps:
        t = _parse_time(cf.timestamps[0])
        if t is not None:
            return t, 'first frame time stamp'
    for key in ('Acq/started_iso', 'Acq/finished_iso'):
        if cf.info.get(key):
            t = _parse_time(cf.info[key])
            if t is not None:
                return t, key
    return None, None


def cube_start_time(cf):
    """(datetime, source) of the start of the cube: /Cube/Info/Cube@CreationDate, else the first time stamp, else the raw
    sibling file, else the file modification time (source says which)."""
    t, src = _time_from(cf)
    if t is not None:
        return t, src
    sib = raw_sibling(cf.path)
    if sib is not None:
        try:
            t, src = _time_from(CubeFile(sib))
        except (CubeFormatError, OSError):
            t = None
        if t is not None:
            return t, 'raw file: ' + src
    return datetime.fromtimestamp(os.path.getmtime(cf.path)), 'file modification time'


def _gain_acc_from(cf):
    g, a = cf.info.get('Acq/emICCD/gain'), cf.info.get('Acq/emICCD/acc')
    try:
        if g is not None and a is not None:
            return float(g), float(a)
    except (TypeError, ValueError):
        pass
    return None


def cube_gain_acc(cf):
    """(gain, acc, source) of an emICCD cube from /Acq/emICCD (raw) or from its raw sibling; (None, None, None) otherwise."""
    ga = _gain_acc_from(cf)
    if ga is not None:
        return ga[0], ga[1], 'file'
    sib = raw_sibling(cf.path)
    if sib is not None:
        try:
            ga = _gain_acc_from(CubeFile(sib))
        except (CubeFormatError, OSError):
            ga = None
        if ga is not None:
            return ga[0], ga[1], 'raw file'
    return None, None, None


def comparability_warnings(entries):
    """Plain-text warnings for a list of dicts {name, units, dtype_kind, gain, acc, px, shape} of the SHOWN cubes:
    mixed units; integer (count) cubes with different gain x acc (their counts are not comparable); mixed pixel sizes /
    image sizes (spectra are mapped through um; colour overlay / difference images are off)."""
    out = []
    if len(entries) < 2:
        return out
    units = sorted({e['units'] for e in entries})
    if len(units) > 1:
        out.append('mixed units (%s): compare normalised spectra only' % ', '.join(units))
    counts = [e for e in entries if e.get('dtype_kind') in ('u', 'i')]
    prods = {round(e['gain'] * e['acc'], 6) for e in counts if e.get('gain') and e.get('acc')}
    if len(prods) > 1:
        out.append('integer cubes with different gain x acc (%s): their counts are not comparable; use the _dc files'
                   % ', '.join('%g' % p for p in sorted(prods)))
    pxs = {round(e['px'], 6) for e in entries}
    shapes = {tuple(e['shape']) for e in entries}
    if len(pxs) > 1 or len(shapes) > 1:
        out.append('different pixel sizes or image sizes: spectra are mapped through um; colour overlay and difference '
                   'images need the same grid')
    return out


# ---------------------------------------------------------------------------------------------------------------- colours
def _hex(rgb):
    return '#%02X%02X%02X' % tuple(int(round(v)) for v in rgb)


def sequence_color(t):
    """Viridis colour at t in [0, 1] (linear between the anchors)."""
    t = float(min(max(t, 0.0), 1.0))
    for (t0, c0), (t1, c1) in zip(_VIRIDIS[:-1], _VIRIDIS[1:]):
        if t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return _hex([a + f * (b - a) for a, b in zip(c0, c1)])
    return _hex(_VIRIDIS[-1][1])


def sequence_colors(n, start=0.12):
    """n colours along viridis from `start` to 1 (first cube dark blue-violet, last yellow)."""
    if n <= 0:
        return []
    if n == 1:
        return [sequence_color(0.6)]
    return [sequence_color(start + (1.0 - start) * i / (n - 1)) for i in range(n)]


def next_distinct(used):
    """The first DISTINCT colour not in `used`, None when all are taken."""
    used = {str(u).upper() for u in used}
    return next((c for c in DISTINCT if c.upper() not in used), None)


# ---------------------------------------------------------------------------------------------------------------- geometry
def to_pixel(x_um, y_um, shift_um, px_um, shape_hw):
    """(row, col) in a cube of `shape_hw` (h, w) and pixel size `px_um` for the reference point (x_um, y_um); None outside."""
    sx, sy = shift_um
    col = int(np.floor((x_um + sx) / px_um))
    row = int(np.floor((y_um + sy) / px_um))
    h, w = shape_hw
    if 0 <= row < h and 0 <= col < w:
        return row, col
    return None


def rect_window(x0_um, y0_um, x1_um, y1_um, shift_um, px_um, shape_hw):
    """(r0, r1, c0, c1) half-open pixel window of a reference rectangle in one cube: the pixels whose CENTRES lie inside
    [x0, x1) x [y0, y1) (clipped, at least 1 px); None when the rectangle lies completely outside that cube."""
    sx, sy = shift_um
    h, w = shape_hw
    xa, xb = sorted((x0_um + sx, x1_um + sx))
    ya, yb = sorted((y0_um + sy, y1_um + sy))
    c0, c1 = int(np.ceil(xa / px_um - 0.5)), int(np.ceil(xb / px_um - 0.5))
    r0, r1 = int(np.ceil(ya / px_um - 0.5)), int(np.ceil(yb / px_um - 0.5))
    if c1 <= 0 or r1 <= 0 or c0 >= w or r0 >= h:
        return None
    c0, r0 = max(0, c0), max(0, r0)
    c1, r1 = min(w, max(c1, c0 + 1)), min(h, max(r1, r0 + 1))
    return r0, r1, c0, c1


def probe_spectrum(cf, probe, shift_um, bin_px=1):
    """Spectrum (n,) of `cf` for one probe, None when the probe lies outside this cube.

    probe: ('pixel', x_um, y_um) -> bin x bin block around the mapped pixel | ('roi', x0, y0, x1, y1) in um |
    ('full',) -> mean over all valid pixels (the shift does not matter)."""
    kind = probe[0]
    h, w = cf.shape[1:]
    if kind == 'full':
        return cf.full_spectrum()
    if kind == 'pixel':
        rc = to_pixel(probe[1], probe[2], shift_um, cf.pixel_size_um, (h, w))
        return None if rc is None else cf.spectrum(rc[0], rc[1], bin=int(bin_px))
    if kind == 'roi':
        win = rect_window(probe[1], probe[2], probe[3], probe[4], shift_um, cf.pixel_size_um, (h, w))
        return None if win is None else cf.roi_mean(*win)
    raise ValueError('unknown probe %r' % (kind,))


# ---------------------------------------------------------------------------------------------------------------- spectra
def normalise(x, y, mode, band):
    """'none' | 'peak' (nanmax = 1) | 'band' (mean inside band (lo, hi) = 1); a zero / non-finite reference leaves y as is."""
    y = np.asarray(y, float)
    if mode == 'none':
        return y
    if mode == 'peak':
        ref = np.nanmax(y) if np.isfinite(y).any() else np.nan
    elif mode == 'band':
        lo, hi = min(band), max(band)
        sel = (x >= lo) & (x <= hi) & np.isfinite(y)
        ref = float(np.mean(y[sel])) if sel.any() else np.nan
    else:
        raise ValueError('unknown normalisation %r' % (mode,))
    return y / ref if (np.isfinite(ref) and ref != 0) else y


def _trapezoid(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))) if len(x) > 1 else 0.0


def spectral_metrics(x, y, lo, hi):
    """{'peak_nm', 'centroid_nm', 'integral', 'max'} of the spectrum inside lo..hi nm (finite points only, sorted by x;
    duplicate wavelengths of raw cubes are averaged).  peak_nm: 3-point parabola through the maximum (the maximum itself at
    the range edge); centroid: weights clipped at 0; integral: trapezoid in units x nm.  NaN when < 2 points."""
    nan = {k: float('nan') for k in METRIC_KEYS}
    x, y = np.asarray(x, float), np.asarray(y, float)
    lo, hi = min(lo, hi), max(lo, hi)
    sel = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    if sel.sum() < 2:
        return nan
    xs, ys = x[sel], y[sel]
    ux, inv = np.unique(xs, return_inverse=True)
    uy = np.bincount(inv, weights=ys) / np.bincount(inv)
    if len(ux) < 2:
        return nan
    i = int(np.argmax(uy))
    peak = float(ux[i])
    if 0 < i < len(ux) - 1:
        a, b, _c = np.polyfit(ux[i - 1:i + 2] - ux[i], uy[i - 1:i + 2], 2)
        if a < 0:
            v = -b / (2 * a)
            if abs(v) <= max(ux[i + 1] - ux[i], ux[i] - ux[i - 1]):
                peak = float(ux[i] + v)
    wgt = np.clip(uy, 0, None)
    cen = float(np.sum(wgt * ux) / np.sum(wgt)) if np.sum(wgt) > 0 else float('nan')
    return {'peak_nm': peak, 'centroid_nm': cen, 'integral': _trapezoid(uy, ux), 'max': float(uy[i])}


def plot_connect(x, y):
    """pyqtgraph 'connect' array: a segment only between finite points with increasing wavelength (raw cubes step back at
    every grating change)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return np.zeros(len(x), np.ubyte)
    ok = np.isfinite(y)
    return (np.append(x[1:] > x[:-1], False) & ok & np.append(ok[1:], False)).astype(np.ubyte)


# ---------------------------------------------------------------------------------------------------------------- images
def frame_for_wavelength(cf, wl):
    """Index of the frame nearest to `wl`, None when `wl` lies outside the cube's range by more than one frame step."""
    w = np.asarray(cf.wavelengths, float)
    lo, hi = float(np.nanmin(w)), float(np.nanmax(w))
    step = float(np.median(np.abs(np.diff(np.unique(w))))) if len(np.unique(w)) > 1 else 1.0
    if wl < lo - step or wl > hi + step:
        return None
    return cf.nearest_frame(wl)


def cube_image(cf, mode, wl=None, band=None):
    """(h, w) float32 image of one cube: mode 'single' (frame nearest to wl, None when wl is outside the cube) or 'band'
    (mean of the frames inside band; None when no frame of the cube lies inside it)."""
    if mode == 'single':
        i = frame_for_wavelength(cf, wl)
        return None if i is None else cf.frame(i)
    if mode == 'band':
        lo, hi = min(band), max(band)
        w = np.asarray(cf.wavelengths, float)
        if not ((w >= lo) & (w <= hi)).any():
            return None
        return cf.band(lo, hi)
    raise ValueError('unknown image mode %r' % (mode,))


def shift_into_reference(img, shift_um, px_um):
    """Cube image -> reference grid: out[r, c] = img[r + dy, c + dx] with the shift rounded to whole pixels; NaN where the
    cube has no data."""
    dy = int(round(shift_um[1] / px_um))
    dx = int(round(shift_um[0] / px_um))
    h, w = img.shape
    out = np.full((h, w), np.nan, np.float32)
    rs, re_ = max(0, -dy), min(h, h - dy)
    cs, ce = max(0, -dx), min(w, w - dx)
    if rs < re_ and cs < ce:
        out[rs:re_, cs:ce] = img[rs + dy:re_ + dy, cs + dx:ce + dx]
    return out


def phase_shift(ref, img, max_frac=0.25):
    """(dy, dx) in pixels (sub-pixel) such that a feature at (r, c) of `ref` appears at (r + dy, c + dx) in `img`
    (phase correlation with a Hann window; NaN filled with the median; shifts beyond max_frac of the side are not searched)."""
    a = np.array(ref, np.float64)
    b = np.array(img, np.float64)
    if a.shape != b.shape:
        raise ValueError('images differ in size: %s vs %s' % (a.shape, b.shape))
    for z in (a, b):
        med = np.nanmedian(z) if np.isfinite(z).any() else 0.0
        z[~np.isfinite(z)] = med
        z -= z.mean()
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    fa, fb = np.fft.fft2(a * win), np.fft.fft2(b * win)
    cross = fb * np.conj(fa)
    cross /= np.maximum(np.abs(cross), 1e-30)
    r = np.fft.ifft2(cross).real
    my, mx = max(1, int(max_frac * h)), max(1, int(max_frac * w))
    ys = np.r_[0:my + 1, h - my:h]
    xs = np.r_[0:mx + 1, w - mx:w]
    sub = r[np.ix_(ys, xs)]
    iy, ix = np.unravel_index(int(np.argmax(sub)), sub.shape)
    py, px = int(ys[iy]), int(xs[ix])

    def refine(c_m, c_0, c_p):
        den = c_m - 2 * c_0 + c_p
        return 0.0 if den >= 0 else 0.5 * (c_m - c_p) / den

    oy = refine(r[(py - 1) % h, px], r[py, px], r[(py + 1) % h, px])
    ox = refine(r[py, (px - 1) % w], r[py, px], r[py, (px + 1) % w])
    dy = py - h if py > h // 2 else py
    dx = px - w if px > w // 2 else px
    return float(dy + oy), float(dx + ox)


def color_overlay(images, colors, common=True, low=0.5, high=99.5):
    """(h, w, 3) uint8 additive blend: each image scaled to 0..1 (common percentiles over all images when `common`, else
    its own), multiplied by its colour and summed.  Where images overlap the sum exceeds 1; the whole blend is then divided
    by one factor (the `high` percentile of the brightest channel, at least 1), so a series of similar cubes does not clip
    to white while separate features keep their full colour.  NaN counts as 0."""
    if not images:
        raise ValueError('no image to overlay')
    if common:
        stacked = np.concatenate([np.asarray(im, np.float32).ravel()[::max(1, im.size // 200_000)] for im in images])
        lv = [auto_levels(stacked, low, high)] * len(images)
    else:
        lv = [auto_levels(im, low, high) for im in images]
    out = np.zeros(images[0].shape + (3,), np.float32)
    for im, col, (lo, hi) in zip(images, colors, lv):
        s = np.nan_to_num(np.clip((np.asarray(im, np.float32) - lo) / (hi - lo), 0, 1), nan=0.0)
        rgb = np.array([int(col[k:k + 2], 16) for k in (1, 3, 5)], np.float32) / 255.0
        out += s[..., None] * rgb
    top = out.max(axis=-1)
    scale = max(1.0, float(np.percentile(top, high))) if top.size else 1.0
    return (np.clip(out / scale, 0, 1) * 255).astype(np.uint8)


def change_image(sel, ref, kind, rel_floor=0.02):
    """'difference' -> sel - ref | 'relative' -> 100 (sel / ref - 1) % where ref > rel_floor x its 99th percentile (NaN
    elsewhere: dark pixels would give huge ratios)."""
    sel = np.asarray(sel, np.float32)
    ref = np.asarray(ref, np.float32)
    if kind == 'difference':
        return sel - ref
    if kind == 'relative':
        v = ref[np.isfinite(ref)]
        floor = rel_floor * float(np.percentile(v, 99)) if v.size else np.inf
        with np.errstate(all='ignore'):
            out = 100.0 * (sel / ref - 1.0)
        out[~(ref > floor)] = np.nan
        return out.astype(np.float32)
    raise ValueError('unknown change image %r' % (kind,))


def symmetric_levels(img, high=99.0):
    """(-m, m) with m = the `high` percentile of |img| (finite pixels), at least 1e-12."""
    v = np.abs(np.asarray(img, float))
    v = v[np.isfinite(v)]
    m = float(np.percentile(v, high)) if v.size else 1.0
    m = m if m > 0 else 1e-12
    return -m, m
