r"""trpl_explorer - interactive viewer for broadband TRPL data (PI-MAX4 emICCD), 2026-11-04.

    python trpl_explorer.py [file.h5]          (trpl_new or base env: numpy, h5py, matplotlib, tkinter)

One window: the PL map on the left, decays on the right.
  * Map: one frame at a chosen time (slider, or click on the decay plot), the time-integrated PL, or 1/e and 1/10
    lifetime maps (spatial binning 1-8 against noise; log colours).
  * Regions: draw rectangles, ellipses, freehand shapes or click points; each region's mean decay appears at once,
    in the region's colour, with t(1/e) and t(1/10) in the table.  Delete / Ctrl+Z / double-click to rename.
  * Hover probe: the decay of a small square under the mouse, live.
  * Compare: overlay the same regions from a second file (dashed), e.g. another run of the same spot.
  * tau_diff vs QFLS (2026-11-19): third panel, differential lifetime against the relative quasi-Fermi-level splitting of the
    full frame and of every region (hsapp.trpl_kinetics: a smooth positive-exponential fit over the real
    gate windows of a _corrected file).
  * Export: CSV of all decays, PNG of the figure, JSON of the regions (load them again on another run).

Files: <name>_corrected.h5 (new method), <name>_stitched.h5 (trpl_stitching) and plain notebook files.
Intensity = (Data - Metadata/dark_data) / (gain x accumulations x gate width), per pixel (the notebook normalisation).
In _corrected / _stitched files the gate-step (and drift) corrections are already in Data; plain files are not
corrected for gate-width steps (the window says so).  Time = gate centre - t0, with t0 = the scout's onset from
<name>_extras.h5 when it exists, else the 10 % point of the rise of the full-frame decay.
Nothing is ever written into the data files.
4D hyperspectral TRPL files (IMA filter x delay) open in their own window: trpl_explorer_4d.py (2026-11-07).
"""
import csv
import datetime
import json
import os
import sys
import time
from importlib import import_module, util

import numpy as np
import h5py
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.widgets import RectangleSelector, EllipseSelector, LassoSelector  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import ttk, filedialog, messagebox, simpledialog  # noqa: E402

from trpl_graph_tools import GraphTools  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_DEFAULTS = import_module('trpl_explorer_local') if util.find_spec('trpl_explorer_local') else None
try:                                                   # tau_diff vs QFLS panel (2026-11-19); the explorer works without it
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from hsapp import trpl_kinetics as TK
    TK_ERROR = ''
except Exception as _e:                                # noqa: BLE001
    TK, TK_ERROR = None, str(_e)
PIX_UM = {'2x': 6.5, '10x': 1.3, '20x': 0.645, '50x': 0.258, '63x': 0.165, '100x': 0.116}   # um per pixel
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f']
MAP_MODES = (('frame', 'Frame at time'), ('integrated', 'Time-integrated'), ('tau_e', 'Lifetime 1/e'), ('tau_10', 'Lifetime 1/10'))
TOOLS = (('rect', 'Rectangle'), ('ellipse', 'Ellipse'), ('lasso', 'Freehand'), ('point', 'Point'), ('none', 'None (zoom/pan)'))
HELP = """TRPL explorer

Draw regions on the map (left mouse):
  Rectangle / Ellipse: drag.   Freehand: draw the outline.   Point: click (square of 'size' pixels).
  Each region gets a colour; its mean decay appears on the right and in the table.
Hover probe: the grey curve is the decay under the mouse.
Click on the decay plot: the map shows that time.
Table: select a region to highlight it; Delete removes it, Ctrl+Z (or 'Remove last') removes the
last region drawn, double-click renames it.
Noise: the table shows the photon noise of one point at the peak. The default plan gives ~1 % for the
whole image, so small regions are noisy (7x7 px ~80 %, 100x100 px ~5 % on 4 Nov). The window opens on
the time-integrated map, binned to ~25 photons per map pixel.
Map: frame at a time (slider), time-integrated PL, or lifetime maps (time to 1/e or 1/10 of each
pixel's peak). Map titles give the detected photons per map pixel: raise the binning if they are few.
Graphs: scroll over a plot to zoom at the pointer. Hold Shift for time or x only, Ctrl for y only.
Use the hand or magnifier in the toolbar to pan or drag a zoom box. Reset zoom fits all graphs.
Choose Decay or Lifetime to see it on its own. Untick Graph only to restore the overview.
Hover near a sample to read its plotted values. For map navigation, select region tool 'None'.
Compare: overlays the same regions from a second file of the same image size (dashed lines).
Export CSV: all decays (time after onset, gate, full frame, regions), plus <name>_taudiff_qfls.csv with the
tau_diff vs QFLS curves of the right panel.  Save regions: JSON to reuse.
tau_diff vs QFLS (right panel): dQFLS = kT ln(PL / PL_peak), tau_diff = (-(1/2) d ln PL / dt)^-1, from a smooth
fit (positive sum of exponentials over the real gate windows), as in the hsapp TRPL viewer.  Dots mark 10 ns,
100 ns, 1 us ... after the onset (first solid curve).  Small regions are noisy: compare regions of similar size.

Intensity = (Data - dark) / (gain x accumulations x gate width) per pixel.  Open the _corrected.h5
file of a new-method run: plain files are not corrected for the gate-width steps."""


def default_data_dir():
    if LOCAL_DEFAULTS is None:
        return os.path.expanduser('~')
    path = LOCAL_DEFAULTS.DATA_DIR
    return path if os.path.isdir(path) else os.getcwd()


# ---------------------------------------------------------------------------------- data
def _decode(x):
    return x.decode('utf-8', 'replace') if isinstance(x, bytes) else str(x)


def run_base(path):
    for suf in ('_corrected.h5', '_stitched.h5', '.h5'):
        if path.endswith(suf):
            return path[:-len(suf)]
    return os.path.splitext(path)[0]


def file_kind(path):
    with h5py.File(path, 'r') as h:
        if 'Corrections' in h:
            return 'corrected'
        if 'Stitching' in h:
            return 'stitched'
        return 'raw'


# emICCD noise model (measured on this camera, trpl_schedule.noise_b): var(S) = b * S for a sum S of raw counts,
# b = 1.2e4 x (gain / 4600)^0.65 counts per detected photon.  S / b = number of detected photons.
def counts_per_photon(gain):
    return 1.2e4 * (np.asarray(gain, float) / 4600.0) ** 0.65


def photons(ds, value_per_pixel, npix, i):
    """Detected photons in npix pixels of frame i whose normalised value per pixel is value_per_pixel."""
    raw = np.asarray(value_per_pixel, float) * npix * ds.G[i] * ds.ACC[i] * ds.PW[i]
    return raw / counts_per_photon(ds.G[i])


def peak_value(y, ip, noisy=False):
    """Peak of a decay: max over frames ip-2..ip+2 for a clean curve; mean over ip-1..ip+1 for a noisy one (the
    maximum of noisy points is biased high, which makes every crossing time too early)."""
    seg = y[max(0, ip - 1):ip + 2] if noisy else y[max(0, ip - 2):ip + 3]
    seg = seg[np.isfinite(seg)]
    if not seg.size:
        return np.nan
    return float(seg.mean()) if noisy else float(seg.max())


def t_at(t, y, ip, frac, noisy=False):
    """Time after onset where the decay y first falls below frac x peak (peak_value) and the next point is also
    below (one noisy point does not count); log-linear interpolation."""
    p = peak_value(y, ip, noisy)
    if not (p > 0):
        return np.nan
    yn = y / p
    for i in range(ip + 1, len(yn)):
        nxt_ok = i + 1 >= len(yn) or yn[i + 1] < frac
        if yn[i] < frac and nxt_ok:
            ya, yb, ta, tb = yn[i - 1], yn[i], t[i - 1], t[i]
            if ya > 0 and yb > 0 and ya != yb:
                f = (np.log(ya) - np.log(frac)) / (np.log(ya) - np.log(yb))
            elif ya != yb:
                f = (ya - frac) / (ya - yb)
            else:
                f = 1.0
            return float(ta + np.clip(f, 0, 1) * (tb - ta))
    return np.nan


class Dataset:
    """One TRPL file, normalised to signal per pixel per (gain x accumulation x ns of gate)."""

    def __init__(self, path, progress=None):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(path)
        self.notes = []
        with h5py.File(self.path, 'r') as h:
            if 'Data' not in h or 'Metadata' not in h or 'pulse_delay' not in h['Metadata']:
                raise ValueError('not a broadband TRPL file: it needs /Data and /Metadata/pulse_delay '
                                 '(hyperspectral IMA cubes are not supported)')
            md = h['Metadata']
            D = h['Data']
            if D.ndim != 3:
                raise ValueError(f'/Data has shape {D.shape}; a (time, y, x) cube is needed')
            n, ny, nx = D.shape
            self.PD = np.asarray(md['pulse_delay'][()], float).reshape(-1)[:n]
            self.PW = np.asarray(md['pulse_width'][()], float).reshape(-1)[:n]
            self.G = np.asarray(md['emiccd_gain'][()], float).reshape(-1)[:n]
            self.ACC = np.asarray(md['accumulations'][()], float).reshape(-1)[:n]
            dark = np.asarray(md['dark_data'][()], np.float32) if 'dark_data' in md else np.zeros((ny, nx), np.float32)
            self.objective = _decode(md['objective'][()]).strip() if 'objective' in md else ''
            self.pix_um = PIX_UM.get(self.objective)
            norm = (self.G * self.ACC * self.PW).astype(np.float32)
            if np.any(~(norm > 0)):
                raise ValueError('gain x accumulations x gate width is not positive for some frames')
            cube = np.empty((n, ny, nx), np.float32)
            step = 8
            for i in range(0, n, step):
                blk = np.asarray(D[i:i + step], np.float32)
                cube[i:i + step] = (blk - dark) / norm[i:i + step, None, None]
                if progress:
                    progress(min(i + step, n), n)
            self.T = float(np.asarray(md['T'][()]).reshape(-1)[0]) if 'T' in md else 300.0
            self.t_eff_abs = self.ws_abs = self.we_abs = None      # real gate windows (absolute delay scale) of a _corrected file
            if 'Corrections' in h and all(k in h['Corrections'] for k in ('t_center_eff', 't_window_start_eff', 't_window_end_eff')):
                c_ = h['Corrections']
                te_, ws_, we_ = (np.asarray(c_[k][()], float).reshape(-1)[:n] for k in ('t_center_eff', 't_window_start_eff',
                                                                                        't_window_end_eff'))
                if len(te_) == n and np.all(np.isfinite(te_)) and np.all(we_ >= ws_):
                    self.t_eff_abs, self.ws_abs, self.we_abs = te_, ws_, we_
            if 'Corrections' in h:
                self.kind = 'corrected'
                if 'notes' in h['Corrections']:
                    self.notes += [_decode(x) for x in h['Corrections/notes'][()]]
            elif 'Stitching' in h:
                self.kind = 'stitched (' + _decode(h['Stitching'].attrs.get('method', '?')) + ')'
            else:
                self.kind = 'raw'
        order = np.argsort(self.PD, kind='stable')
        if np.any(order != np.arange(n)):
            cube = cube[order]
            self.PD, self.PW, self.G, self.ACC = self.PD[order], self.PW[order], self.G[order], self.ACC[order]
            if self.t_eff_abs is not None:
                self.t_eff_abs, self.ws_abs, self.we_abs = self.t_eff_abs[order], self.ws_abs[order], self.we_abs[order]
        self.cube = cube
        self.shape = (ny, nx)
        self.full = cube.reshape(n, -1).mean(axis=1)
        self.ip = int(np.nanargmax(self.full))
        self.t0, self.t0_source = self._find_t0()
        self.t = self.PD + self.PW / 2.0 - self.t0
        n_changes = int(np.sum(np.diff(self.PW) != 0))
        if self.kind == 'raw' and n_changes:
            self.notes.append(f'NOT corrected for gate-width steps ({n_changes} changes of gate width): open the '
                              f'_corrected.h5 (new method) or _stitched.h5 file instead')
        self._bins = {1: cube}
        self._maps = {}

    def _find_t0(self):
        ex = run_base(self.path) + '_extras.h5'
        if os.path.exists(ex):
            try:
                with h5py.File(ex, 'r') as h:
                    t0 = float(json.loads(h['plan_json'][()])['params']['t0'])
                return t0, 'scout onset (extras file)'
            except Exception:
                pass
        y = self.full / self.full[self.ip]
        for j in range(self.ip, 0, -1):
            if y[j - 1] < 0.1 <= y[j]:
                f = (0.1 - y[j - 1]) / (y[j] - y[j - 1])
                return float(self.PD[j - 1] + f * (self.PD[j] - self.PD[j - 1])), '10 % of the full-frame rise'
        return float(self.PD[self.ip]), 'delay of the brightest frame (no rise in the data)'

    def binned(self, b):
        if b not in self._bins:
            n, ny, nx = self.cube.shape
            y2, x2 = ny // b * b, nx // b * b
            self._bins[b] = self.cube[:, :y2, :x2].reshape(n, y2 // b, b, x2 // b, b).mean(axis=(2, 4))
        return self._bins[b]

    def map(self, mode, b, i=0):
        c = self.binned(b)
        if mode == 'frame':
            return c[int(i)]
        key = (mode, b)
        if key not in self._maps:
            if mode == 'integrated':
                o = np.argsort(self.t)
                trap = getattr(np, 'trapezoid', None) or np.trapz
                self._maps[key] = trap(c[o], self.t[o], axis=0).astype(np.float32)
            else:
                self._maps[key] = self._lifetime_map(c, np.exp(-1.0) if mode == 'tau_e' else 0.1)
        return self._maps[key]

    def _lifetime_map(self, c, frac):
        ip = self.ip
        b = int(round(np.sqrt(self.cube.shape[1] * self.cube.shape[2] / (c.shape[1] * c.shape[2]))))
        Pmax = c[max(0, ip - 2):ip + 3].max(axis=0)
        Pmean = c[max(0, ip - 1):ip + 2].mean(axis=0)
        n_ph = photons(self, Pmean, b * b, ip)                       # detected photons per map pixel, peak frame
        with np.errstate(divide='ignore', invalid='ignore'):
            noisy = ~(n_ph > 400)                                     # > 5 % noise: the maximum is biased
        P = np.where(noisy, Pmean, Pmax)
        self.last_map_photons = float(np.nanmedian(n_ph)) if np.isfinite(n_ph).any() else np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            y = c[ip:] / np.where(P > 0, P, np.nan)
        t = self.t[ip:]
        below = y < frac
        both = below.copy()
        both[:-1] &= below[1:]                                        # two consecutive points below frac
        has = both.any(axis=0)
        k = np.argmax(both, axis=0)
        km1 = np.clip(k - 1, 0, None)
        ya = np.take_along_axis(y, km1[None], 0)[0]
        yb = np.take_along_axis(y, k[None], 0)[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            lg = (ya > 0) & (yb > 0) & (ya != yb)
            f = np.where(lg, (np.log(ya) - np.log(frac)) / (np.log(ya) - np.log(yb)), (ya - frac) / (ya - yb))
        f = np.clip(np.nan_to_num(f, nan=1.0), 0, 1)
        out = (t[km1] + f * (t[k] - t[km1])).astype(np.float32)
        out[k == 0] = t[0]
        out[~has] = np.nan
        good = np.isfinite(P)
        if good.any():
            out[~(P > 0.05 * np.nanpercentile(P[good], 99))] = np.nan     # too dark for a lifetime
        return out

    def kinetics_axis(self):
        """(t, lo, hi) for hsapp.trpl_kinetics: time after the onset of every point and its gate window before / after it -- the
        real windows of a _corrected file (Corrections/t_window_*_eff), else the nominal gate (centre +- width / 2)."""
        if self.t_eff_abs is not None:
            return self.t_eff_abs - self.t0, self.t_eff_abs - self.ws_abs, self.we_abs - self.t_eff_abs
        return self.t, self.PW / 2.0, self.PW / 2.0

    def region_decay(self, mask):
        return self.cube[:, mask].mean(axis=1)

    def peak_noise(self, decay, npix):
        """Relative photon noise of one point at the peak frame for a region of npix pixels."""
        n = photons(self, decay[self.ip], npix, self.ip)
        return float(1.0 / np.sqrt(n)) if n > 0 else np.inf

    def map_photons(self, mode, b, i=0):
        """Detected photons per map pixel (bin b x b): in frame i, or summed over all frames (integrated map)."""
        if mode == 'frame':
            return float(photons(self, max(self.full[i], 0.0), b * b, i))
        return float(sum(photons(self, max(self.full[k], 0.0), b * b, k) for k in range(len(self.t))))

    def auto_bin(self, target=25.0):
        """Smallest binning with >= target photons per map pixel in the time-integrated map (~20 % noise)."""
        for b in (1, 2, 4, 8, 16, 32, 64):
            if self.map_photons('integrated', b) >= target:
                return b
        return 64

    def recommended_bin(self, target=400.0):
        """Smallest map binning with >= target detected photons per map pixel at the peak frame (5 % noise)."""
        per_px = photons(self, max(self.full[self.ip], 1e-30), 1, self.ip)
        for b in (1, 2, 4, 8, 16, 32, 64):
            if per_px * b * b >= target:
                return b
        return 64


def mask_from_verts(verts, ny, nx):
    v = np.asarray(verts, float)
    x0, x1 = int(max(0, np.floor(v[:, 0].min()))), int(min(nx - 1, np.ceil(v[:, 0].max())))
    y0, y1 = int(max(0, np.floor(v[:, 1].min()))), int(min(ny - 1, np.ceil(v[:, 1].max())))
    m = np.zeros((ny, nx), bool)
    if x1 < x0 or y1 < y0:
        return m
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    inside = Path(v).contains_points(np.column_stack([xx.ravel(), yy.ravel()]))
    m[y0:y1 + 1, x0:x1 + 1] = inside.reshape(yy.shape)
    return m


def rect_verts(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def ellipse_verts(x0, y0, x1, y1, n=72):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, abs(x1 - x0) / 2, abs(y1 - y0) / 2
    return [(cx + rx * np.cos(q), cy + ry * np.sin(q)) for q in a]


class Region:
    def __init__(self, name, kind, verts, color, mask):
        self.name, self.kind, self.verts, self.color, self.mask = name, kind, [tuple(map(float, p)) for p in verts], color, mask
        self.npix = int(mask.sum())
        self.decay = None
        self.decay_cmp = None
        self.patch = None
        self.label = None


# ---------------------------------------------------------------------------------- app
class Explorer(tk.Tk):
    def __init__(self, path=None, *, allow_correction=True):
        super().__init__()
        self.allow_correction = allow_correction
        self.title('TRPL explorer')
        self.geometry('1560x940')
        self.ds = None
        self.cmp = None
        self.regions = []
        self._n_made = 0
        self._last_hover = 0.0
        self._hover = None
        self._selected = None
        self.last_dir = default_data_dir()
        self.v_map = tk.StringVar(value='frame')
        self.v_bin = tk.StringVar(value='2')
        self.v_logc = tk.BooleanVar(value=False)
        self.v_tool = tk.StringVar(value='rect')
        self.v_psize = tk.IntVar(value=7)
        self.v_hover = tk.BooleanVar(value=True)
        self.v_norm = tk.BooleanVar(value=True)
        self.v_logx = tk.BooleanVar(value=False)
        self.v_logy = tk.BooleanVar(value=True)
        self.v_full = tk.BooleanVar(value=True)
        self.v_kin = tk.BooleanVar(value=True)
        self._kin_cache = {}
        self._build()
        self.protocol('WM_DELETE_WINDOW', self.destroy)
        if path:
            self.after(50, lambda: self.open_file(path))

    # ------------------------------------------------------------------ layout
    def _build(self):
        bar = ttk.Frame(self, padding=(6, 4))
        bar.pack(side=tk.TOP, fill=tk.X)
        for txt, cmd in (('Open...', self.open_dialog), ('Compare...', self.compare_dialog), ('Clear compare', self.clear_compare)):
            ttk.Button(bar, text=txt, command=cmd).pack(side=tk.LEFT, padx=2)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        for txt, cmd in (('Export CSV...', self.export_csv), ('Save figure...', self.save_figure),
                         ('Save regions...', self.save_regions), ('Load regions...', self.load_regions)):
            ttk.Button(bar, text=txt, command=cmd).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text='?', width=3, command=lambda: messagebox.showinfo('TRPL explorer', HELP)).pack(side=tk.LEFT, padx=6)
        self.l_file = ttk.Label(bar, text='no file', foreground='#555')
        self.l_file.pack(side=tk.LEFT, padx=10)

        opt = ttk.Frame(self, padding=(6, 0))
        opt.pack(side=tk.TOP, fill=tk.X)
        f = ttk.LabelFrame(opt, text='Map', padding=(4, 0))
        f.pack(side=tk.LEFT, padx=2)
        for key, lab in MAP_MODES:
            ttk.Radiobutton(f, text=lab, value=key, variable=self.v_map, command=self.update_map).pack(side=tk.LEFT)
        ttk.Label(f, text='  bin').pack(side=tk.LEFT)
        cb = ttk.Combobox(f, textvariable=self.v_bin, values=('1', '2', '4', '8', '16', '32', '64'), width=3, state='readonly')
        cb.pack(side=tk.LEFT)
        cb.bind('<<ComboboxSelected>>', lambda e: self.update_map())
        ttk.Checkbutton(f, text='log colours', variable=self.v_logc, command=self.update_map).pack(side=tk.LEFT, padx=4)
        f = ttk.LabelFrame(opt, text='Region tool', padding=(4, 0))
        f.pack(side=tk.LEFT, padx=6)
        for key, lab in TOOLS:
            ttk.Radiobutton(f, text=lab, value=key, variable=self.v_tool, command=self._set_tool).pack(side=tk.LEFT)
        ttk.Label(f, text='  point size').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=1, to=51, increment=2, width=4, textvariable=self.v_psize).pack(side=tk.LEFT)
        f = ttk.LabelFrame(opt, text='Decays', padding=(4, 0))
        f.pack(side=tk.LEFT, padx=6)
        for lab, var in (('normalise', self.v_norm), ('log time', self.v_logx), ('log PL', self.v_logy),
                         ('full frame', self.v_full), ('hover probe', self.v_hover), ('tau_diff vs QFLS', self.v_kin)):
            ttk.Checkbutton(f, text=lab, variable=var, command=self.redraw_decays).pack(side=tk.LEFT)

        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(15, 6.2), dpi=100)
        gs = self.fig.add_gridspec(1, 6, width_ratios=[1.0, 0.035, 0.26, 1.0, 0.20, 0.80], left=0.04, right=0.985, bottom=0.09,
                                   top=0.92, wspace=0.05)
        self.ax_map = self.fig.add_subplot(gs[0])
        self.ax_cb = self.fig.add_subplot(gs[1])
        self.ax_dec = self.fig.add_subplot(gs[3])
        self.ax_kin = self.fig.add_subplot(gs[5])      # tau_diff vs QFLS (2026-11-19)
        self.canvas = FigureCanvasTkAgg(self.fig, master=mid)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.graph_tools = GraphTools(mid, self.canvas, {'Decay': self.ax_dec, 'Lifetime': self.ax_kin})
        self.graph_tools.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = self.graph_tools.toolbar
        sl = ttk.Frame(mid)
        sl.pack(side=tk.TOP, fill=tk.X, padx=10)
        ttk.Label(sl, text='time').pack(side=tk.LEFT)
        self.s_time = tk.Scale(sl, from_=0, to=1, orient=tk.HORIZONTAL, showvalue=0, resolution=1, command=self._on_slider)
        self.s_time.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.l_time = ttk.Label(sl, text='', width=58)
        self.l_time.pack(side=tk.LEFT)

        bot = ttk.Frame(self, padding=(6, 2))
        bot.pack(side=tk.TOP, fill=tk.X)
        cols = ('name', 'tool', 'pixels', 'area', 'peak', 'noise', 't_e', 't_10', 't_e_cmp')
        heads = ('region', 'tool', 'pixels', 'area (um2)', 'peak signal', 'noise/point at peak', 't(1/e) ns', 't(1/10) ns',
                 't(1/e) compare')
        self.table = ttk.Treeview(bot, columns=cols, show='headings', height=5)
        for c, hdr in zip(cols, heads):
            self.table.heading(c, text=hdr)
            self.table.column(c, width=120 if c != 'name' else 140, anchor=tk.CENTER)
        self.table.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.table.bind('<<TreeviewSelect>>', self._on_table_select)
        self.table.bind('<Double-1>', self._rename)
        bb = ttk.Frame(bot)
        bb.pack(side=tk.LEFT, padx=6)
        ttk.Button(bb, text='Delete', command=self.delete_selected).pack(fill=tk.X, pady=1)
        ttk.Button(bb, text='Remove last', command=self.undo_last).pack(fill=tk.X, pady=1)
        ttk.Button(bb, text='Clear all', command=self.clear_regions).pack(fill=tk.X, pady=1)
        self.l_status = ttk.Label(self, text='Open a TRPL file (Open...).', anchor=tk.W, padding=(6, 2))
        self.l_status.pack(side=tk.BOTTOM, fill=tk.X)
        self.bind('<Delete>', lambda e: self.delete_selected())
        self.bind('<Control-z>', lambda e: self.undo_last())

        sel_props = dict(edgecolor='white', facecolor='none', linestyle='--', linewidth=1.2)
        self.sel_rect = RectangleSelector(self.ax_map, self._on_box_rect, useblit=True, button=[1], minspanx=2, minspany=2,
                                          spancoords='data', interactive=False, props=sel_props)
        self.sel_ell = EllipseSelector(self.ax_map, self._on_box_ell, useblit=True, button=[1], minspanx=2, minspany=2,
                                       spancoords='data', interactive=False, props=sel_props)
        self.sel_lasso = LassoSelector(self.ax_map, self._on_lasso, useblit=True, button=[1],
                                       props=dict(color='white', linestyle='--', linewidth=1.2))
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self._set_tool()
        self._empty_axes()

    def _empty_axes(self):
        for ax in (self.ax_map, self.ax_dec, self.ax_cb, self.ax_kin):
            ax.clear()
        self.ax_map.set_title('no file')
        self.ax_cb.set_axis_off()
        self.canvas.draw_idle()

    def status(self, msg):
        self.l_status.config(text=msg)
        self.update_idletasks()

    # ------------------------------------------------------------------ files
    def open_dialog(self):
        p = filedialog.askopenfilename(title='Open TRPL file', initialdir=self.last_dir,
                                       filetypes=[('TRPL HDF5', '*.h5'), ('all files', '*.*')])
        if p:
            self.open_file(p)

    def _resolve(self, path):
        """For a plain file of a new-method run, offer the corrected file (or to create it)."""
        try:
            kind = file_kind(path)
        except Exception as e:
            messagebox.showerror('TRPL explorer', f'cannot read {path}:\n{e}')
            return None
        if kind != 'raw':
            return path
        base = run_base(path)
        if os.path.exists(base + '_corrected.h5'):
            if messagebox.askyesno('TRPL explorer', 'This is the raw file (no gate-step correction).\n'
                                   'A corrected file exists next to it. Open the corrected file instead?'):
                return base + '_corrected.h5'
        elif (self.allow_correction and LOCAL_DEFAULTS is not None and LOCAL_DEFAULTS.CORRECTIONS_AVAILABLE
              and os.path.exists(base + '_extras.h5')):
            if messagebox.askyesno('TRPL explorer', 'This is the raw file (no gate-step correction).\n'
                                   'Create the corrected file from its extras file?'):
                try:
                    self.status('writing the corrected file ...')
                    return LOCAL_DEFAULTS.correct_file(path, base + '_extras.h5')
                except Exception as e:
                    messagebox.showerror('TRPL explorer', f'could not create the corrected file:\n{e}')
        return path

    def _load(self, path):
        t = time.time()
        ds = Dataset(path, progress=lambda i, n: self.status(f'loading {os.path.basename(path)}: frame {i}/{n} ...'))
        self.last_dir = os.path.dirname(os.path.abspath(path))
        self.status(f'loaded {ds.name} in {time.time() - t:.1f} s')
        return ds

    def _open_4d(self, path):
        """4D hyperspectral TRPL files (recognised by content) open in their own window (trpl_explorer_4d.py, a
        Toplevel of this app).  Returns False for anything else, which then takes the unchanged 3D path below."""
        try:
            import trpl_explorer_4d
        except Exception as e:                       # 4D module not next to this file: say so for 4D files only
            try:
                with h5py.File(path, 'r') as h:
                    is3d = (isinstance(h.get('Data'), h5py.Dataset) and h['Data'].ndim == 3 and 'Metadata' in h
                            and 'pulse_delay' in h['Metadata'])
                    is4d = not is3d and any(isinstance(v, h5py.Dataset) and v.ndim == 4 for v in h.values())
            except Exception:
                is4d = False
            if is4d:
                messagebox.showerror('TRPL explorer', f'{os.path.basename(path)} is a 4D hyperspectral file, but the 4D '
                                     f'window (trpl_explorer_4d.py, next to trpl_explorer.py) cannot be loaded:\n{e}')
                self.status('4D file not opened: trpl_explorer_4d.py missing')
            return is4d
        try:
            if not trpl_explorer_4d.is_4d_file(path):
                return False
        except Exception:
            return False
        try:
            win = trpl_explorer_4d.open_window(self, path)
        except Exception as e:
            messagebox.showerror('TRPL explorer', f'cannot open the 4D window for {path}:\n{e}')
            win = None
        self.status(f'{os.path.basename(path)} is a 4D hyperspectral file: '
                    + ('opened in the 4D window' if win is not None else 'not opened (see the message)'))
        return True

    def open_file(self, path):
        if self._open_4d(path):
            return
        path = self._resolve(path)
        if not path:
            return
        try:
            self.ds = None
            ds = self._load(path)
        except Exception as e:
            messagebox.showerror('TRPL explorer', f'cannot open {path}:\n{e}')
            self.status('open failed')
            return
        self.ds = ds
        self.graph_tools.clear()
        self.v_map.set('integrated')
        self.v_bin.set(str(ds.auto_bin()))
        if self.cmp is not None and self.cmp.shape != ds.shape:
            self.cmp = None
        n = len(ds.t)
        self.s_time.config(to=n - 1)
        self.s_time.set(ds.ip)
        self.ax_map.clear()
        self.ax_cb.clear()
        ny, nx = ds.shape
        self.im = self.ax_map.imshow(ds.map('frame', 1, ds.ip), origin='lower', cmap='inferno', interpolation='nearest',
                                     extent=(-0.5, nx - 0.5, -0.5, ny - 0.5))
        self.cbar = self.fig.colorbar(self.im, cax=self.ax_cb)
        self.ax_map.set_xlabel('x (pixel)')
        self.ax_map.set_ylabel('y (pixel)')
        self._scalebar()
        for r in self.regions:                      # keep the regions when they fit the new image
            r.patch = r.label = None
        kept = [r for r in self.regions if r.mask.shape == ds.shape]
        self.regions = []
        for r in kept:
            self._attach(r)
        self.l_file.config(text=f'{ds.name}   [{ds.kind}]   {n} frames, {nx}x{ny} px'
                                + (f', {ds.objective} ({ds.pix_um} um/px)' if ds.pix_um else ''))
        note = f't0 = {ds.t0:.2f} ns ({ds.t0_source}); peak at {ds.t[ds.ip]:.1f} ns.'
        if ds.notes:
            note += '  ' + ' | '.join(ds.notes)
        self.status(note)
        self.update_map()

    def compare_dialog(self):
        if self.ds is None:
            return
        p = filedialog.askopenfilename(title='Compare with (same image size)', initialdir=self.last_dir,
                                       filetypes=[('TRPL HDF5', '*.h5'), ('all files', '*.*')])
        if not p:
            return
        p = self._resolve(p)
        if not p:
            return
        try:
            c = self._load(p)
        except Exception as e:
            messagebox.showerror('TRPL explorer', f'cannot open {p}:\n{e}')
            return
        if c.shape != self.ds.shape:
            messagebox.showerror('TRPL explorer', f'image size {c.shape} differs from {self.ds.shape}')
            return
        self.cmp = c
        for r in self.regions:
            r.decay_cmp = c.region_decay(r.mask)
        self.status(f'compare: {c.name} [{c.kind}], t0 {c.t0:.2f} ns ({c.t0_source}) - dashed lines'
                    + ('  | ' + ' | '.join(c.notes) if c.notes else ''))
        self._refresh_table()
        self.redraw_decays()

    def clear_compare(self):
        self.cmp = None
        for r in self.regions:
            r.decay_cmp = None
        self._refresh_table()
        self.redraw_decays()

    # ------------------------------------------------------------------ map
    def _scalebar(self):
        ds = self.ds
        if not ds or not ds.pix_um:
            return
        try:
            from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        except Exception:
            return
        width_um = ds.shape[1] * ds.pix_um
        L = next((x for x in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if x >= 0.15 * width_um), 1000)
        self.ax_map.add_artist(AnchoredSizeBar(self.ax_map.transData, L / ds.pix_um, f'{L} um', 'lower right',
                                               color='white', frameon=False, size_vertical=max(1, ds.shape[0] // 150)))

    def update_map(self):
        ds = self.ds
        if ds is None:
            return
        mode, b = self.v_map.get(), int(self.v_bin.get())
        i = int(self.s_time.get())
        if mode in ('tau_e', 'tau_10'):
            self.status('computing the lifetime map ...')
        m = ds.map(mode, b, i)
        self.im.set_data(m)
        fin = m[np.isfinite(m)]
        lifetime = mode in ('tau_e', 'tau_10')
        self.im.set_cmap('viridis' if lifetime else 'inferno')
        if fin.size:
            lo, hi = np.percentile(fin, [1, 99])
            if hi <= lo:
                hi = lo + 1e-12
            if self.v_logc.get() and not lifetime:
                pos = fin[fin > 0]
                lo = max(np.percentile(pos, 1) if pos.size else hi * 1e-3, hi * 1e-4)
                self.im.set_norm(LogNorm(vmin=lo, vmax=hi))
            else:
                self.im.set_norm(Normalize(vmin=lo, vmax=hi))
        lab = {'frame': 'signal / (gain x acc x ns)', 'integrated': 'time-integrated signal (x ns)',
               'tau_e': 't(1/e) after onset (ns)', 'tau_10': 't(1/10) after onset (ns)'}[mode]
        self.cbar.update_normal(self.im)
        self.cbar.set_label(lab)
        if mode == 'frame':
            nph = ds.map_photons('frame', b, i)
            title = (f'frame {i + 1}/{len(ds.t)}: {ds.t[i]:.1f} ns after onset (gate {ds.PD[i]:g} + {ds.PW[i]:g} ns), bin {b}'
                     f'\n~{nph:.3g} photons per map pixel in this frame')
        elif mode == 'integrated':
            nph = ds.map_photons('integrated', b)
            title = f'time-integrated PL, bin {b}\n~{nph:.3g} photons per map pixel (noise ~{100 / np.sqrt(max(nph, 1e-9)):.0f} %)'
        else:
            nph = getattr(ds, 'last_map_photons', np.nan)
            rb = ds.recommended_bin()
            title = f'{lab} (bin {b}x{b}); grey = too dark'
            if np.isfinite(nph):
                title += f'\n~{nph:.0f} photons per map pixel at the peak (noise ~{100 / np.sqrt(max(nph, 1e-9)):.0f} %)'
                if b < rb:
                    title += f' - use bin >= {rb}'
        self.ax_map.set_title(title, fontsize=10)
        self.ax_map.set_facecolor('#808080')
        self._update_time_label()
        if mode in ('tau_e', 'tau_10'):
            fin_n = int(np.isfinite(m).sum())
            self.status(f'{lab}: {fin_n} of {m.size} map pixels have a value'
                        + (f'; median {np.nanmedian(m):.0f} ns' if fin_n else ''))
        self.redraw_decays()

    def _update_time_label(self):
        ds = self.ds
        if ds is None:
            return
        i = int(self.s_time.get())
        self.l_time.config(text=f'frame {i + 1}/{len(ds.t)}   t = {ds.t[i]:.1f} ns after onset   '
                                f'(gate {ds.PD[i]:g} + {ds.PW[i]:g} ns, {ds.ACC[i]:g} acc)')

    def _on_slider(self, _val):
        if self.ds is None:
            return
        if self.v_map.get() != 'frame':
            self._update_time_label()
            self.redraw_decays()
            return
        self.update_map()

    # ------------------------------------------------------------------ regions
    def _set_tool(self):
        t = self.v_tool.get()
        self.sel_rect.set_active(t == 'rect')
        self.sel_ell.set_active(t == 'ellipse')
        self.sel_lasso.set_active(t == 'lasso')

    def _on_box_rect(self, e0, e1):
        self._box(e0, e1, 'rect')

    def _on_box_ell(self, e0, e1):
        self._box(e0, e1, 'ellipse')

    def _box(self, e0, e1, kind):
        if None in (e0.xdata, e1.xdata, e0.ydata, e1.ydata):
            return
        x0, x1 = sorted((e0.xdata, e1.xdata))
        y0, y1 = sorted((e0.ydata, e1.ydata))
        v = rect_verts(x0, y0, x1, y1) if kind == 'rect' else ellipse_verts(x0, y0, x1, y1)
        self.add_region(kind, v)
        sel = self.sel_rect if kind == 'rect' else self.sel_ell
        self.after_idle(lambda: (sel.clear(), self.canvas.draw_idle()))

    def _on_lasso(self, verts):
        if len(verts) >= 3:
            self.add_region('lasso', verts)

    def _point(self, x, y):
        r = max(0, (int(self.v_psize.get()) - 1) // 2)
        cx, cy = int(round(x)), int(round(y))
        self.add_region('point', rect_verts(cx - r - 0.5, cy - r - 0.5, cx + r + 0.5, cy + r + 0.5))

    def add_region(self, kind, verts, name=None, color=None):
        if self.ds is None:
            return None
        m = mask_from_verts(verts, *self.ds.shape)
        if not m.any():
            self.status('the region contains no pixel centre - draw a larger one')
            return None
        self._n_made += 1
        r = Region(name or f'R{self._n_made}', kind, verts, color or COLORS[(self._n_made - 1) % len(COLORS)], m)
        self._attach(r)
        self._refresh_table()
        self.redraw_decays()
        ps = self.ds.pix_um
        self.status(f'{r.name}: {r.npix} pixels' + (f' ({r.npix * ps * ps:.0f} um2)' if ps else '')
                    + f'; noise per point at the peak ~{100 * self._stats(self.ds, r.decay, r.npix)[0]:.1f} %'
                    + f'; t(1/e) {self._stats(self.ds, r.decay, r.npix)[2]:.0f} ns')
        return r

    def _attach(self, r):
        r.decay = self.ds.region_decay(r.mask)
        r.decay_cmp = self.cmp.region_decay(r.mask) if self.cmp is not None else None
        r.patch = Polygon(r.verts, closed=True, fill=False, edgecolor=r.color, linewidth=1.6)
        self.ax_map.add_patch(r.patch)
        v = np.asarray(r.verts)
        r.label = self.ax_map.text(v[:, 0].min(), v[:, 1].max() + 3, r.name, color='white', fontsize=8,
                                   fontweight='bold', va='bottom',
                                   bbox=dict(boxstyle='round,pad=0.15', facecolor=r.color, edgecolor='none', alpha=0.85))
        self.regions.append(r)

    def _detach(self, r):
        for a in (r.patch, r.label):
            try:
                a.remove()
            except Exception:
                pass
        self.regions.remove(r)

    def delete_selected(self):
        sel = self.table.selection()
        if not sel:
            return
        for iid in sel:
            r = next((x for x in self.regions if x.name == iid), None)
            if r:
                self._detach(r)
        self._selected = None
        self._refresh_table()
        self.redraw_decays()

    def undo_last(self):
        if self.regions:
            self._detach(self.regions[-1])
            self._refresh_table()
            self.redraw_decays()

    def clear_regions(self):
        for r in list(self.regions):
            self._detach(r)
        self._refresh_table()
        self.redraw_decays()

    def _rename(self, _e):
        sel = self.table.selection()
        if not sel:
            return
        r = next((x for x in self.regions if x.name == sel[0]), None)
        if not r:
            return
        new = simpledialog.askstring('Rename region', 'New name:', initialvalue=r.name, parent=self)
        if new and new.strip() and all(x.name != new.strip() for x in self.regions):
            r.name = new.strip()
            r.label.set_text(r.name)
            self._refresh_table()
            self.redraw_decays()

    def _on_table_select(self, _e):
        sel = self.table.selection()
        self._selected = sel[0] if sel else None
        for r in self.regions:
            r.patch.set_linewidth(3.2 if r.name == self._selected else 1.6)
        self.redraw_decays()

    def _refresh_table(self):
        self.table.delete(*self.table.get_children())
        ds, ps = self.ds, (self.ds.pix_um if self.ds else None)
        for r in self.regions:
            nz, noisy, te, t10 = self._stats(ds, r.decay, r.npix)
            tec = self._stats(self.cmp, r.decay_cmp, r.npix)[2] if r.decay_cmp is not None else np.nan
            vals = (r.name, r.kind, r.npix, f'{r.npix * ps * ps:.0f}' if ps else '-', f'{peak_value(r.decay, ds.ip, noisy):.3g}',
                    f'{100 * nz:.1f} %' if np.isfinite(nz) else '-',
                    f'{te:.0f}' if np.isfinite(te) else '-', f'{t10:.0f}' if np.isfinite(t10) else '-',
                    f'{tec:.0f}' if np.isfinite(tec) else '-')
            self.table.insert('', tk.END, iid=r.name, values=vals, tags=(r.name,))
            self.table.tag_configure(r.name, foreground=r.color)

    # ------------------------------------------------------------------ decay plot
    @staticmethod
    def _stats(ds, decay, npix):
        """(noise per point at the peak, noisy flag, t(1/e), t(1/10)) of a decay averaged over npix pixels."""
        if decay is None or ds is None:
            return np.nan, False, np.nan, np.nan
        nz = ds.peak_noise(decay, npix)
        noisy = not (nz < 0.05)
        return nz, noisy, t_at(ds.t, decay, ds.ip, np.exp(-1), noisy), t_at(ds.t, decay, ds.ip, 0.1, noisy)

    def _curve(self, t, y, ip, noisy=False):
        if self.v_norm.get():
            p = peak_value(y, ip, noisy)
            y = y / p if p > 0 else y * np.nan
        if self.v_logx.get():
            k = t > 0
            return t[k], y[k]
        return t, y

    def redraw_decays(self):
        ax, ds = self.ax_dec, self.ds
        previous = self.graph_tools.capture_ranges()
        ax.clear()
        self._hover = None
        if ds is None:
            self.ax_kin.clear()
            self.canvas.draw_idle()
            return
        if self.v_full.get():
            ax.plot(*self._curve(ds.t, ds.full, ds.ip), '-', color='k', lw=1.0, alpha=0.6, label='full frame')
            if self.cmp is not None:
                ax.plot(*self._curve(self.cmp.t, self.cmp.full, self.cmp.ip), '--', color='k', lw=1.0, alpha=0.6,
                        label='full frame (compare)')
        for r in self.regions:
            lw = 2.6 if r.name == self._selected else 1.4
            nz, noisy, te, _ = self._stats(ds, r.decay, r.npix)
            ax.plot(*self._curve(ds.t, r.decay, ds.ip, noisy), 'o-', ms=2.5, lw=lw, color=r.color,
                    label=f'{r.name} ({r.npix} px)' + (f', t(1/e) {te:.0f} ns' if np.isfinite(te) else ''))
            if r.decay_cmp is not None:
                ax.plot(*self._curve(self.cmp.t, r.decay_cmp, self.cmp.ip, self._stats(self.cmp, r.decay_cmp, r.npix)[1]),
                        '--', lw=lw, color=r.color,
                        label=f'{r.name} compare')
        self._hover, = ax.plot([], [], '-', color='0.45', lw=1.0, label='hover' if self.v_hover.get() else None)
        if self.v_map.get() == 'frame':
            i = int(self.s_time.get())
            if not self.v_logx.get() or ds.t[i] > 0:
                ax.axvline(ds.t[i], color='0.6', lw=0.8, ls=':')
        if self.v_logy.get():
            ax.set_yscale('log', nonpositive='mask')
        if self.v_logx.get():
            ax.set_xscale('log')
        ax.set_xlabel('time after onset (ns)')
        ax.set_ylabel('PL (peak = 1)' if self.v_norm.get() else 'signal / (gain x acc x ns) per pixel')
        ax.grid(True, which='major', alpha=0.25)
        ax.legend(fontsize=7.5, loc='upper right', ncol=1 if len(self.regions) < 8 else 2)
        if self.v_norm.get() and self.v_logy.get():
            lo = 1e-5
            ys = [ln.get_ydata() for ln in ax.get_lines() if len(ln.get_ydata())]
            pos = np.concatenate([np.asarray(y, float)[np.asarray(y, float) > 0] for y in ys]) if ys else np.array([])
            if pos.size:
                lo = max(1e-6, np.percentile(pos, 0.5) * 0.5)
            ax.set_ylim(lo, 2.0)
        ax.set_title(f'{ds.name}  (click: map at that time)', fontsize=9)
        self.draw_kinetics()
        self.graph_tools.finish_redraw(previous)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ tau_diff vs QFLS (2026-11-19)
    def _kin(self, ds, y):
        """hsapp.trpl_kinetics.diff_kinetics of one decay of dataset ds (cached), or an error string."""
        y = np.asarray(y, float)
        key = (id(ds), y.tobytes())
        if key not in self._kin_cache:
            t, lo, hi = ds.kinetics_axis()
            try:
                self._kin_cache[key] = TK.diff_kinetics(t, y, lo_ns=lo, hi_ns=hi, temperature_k=ds.T)
            except Exception as e:                     # noqa: BLE001  (a failed fit never breaks the decays)
                self._kin_cache[key] = str(e)
            if len(self._kin_cache) > 200:
                self._kin_cache.pop(next(iter(self._kin_cache)))
        return self._kin_cache[key]

    def kinetics_curves(self):
        """[(label, color, dashed, noisy, result)] of the panel: full frame, regions, compare (same order as the decays)."""
        ds, out = self.ds, []
        if ds is None or TK is None:
            return out
        if self.v_full.get():
            out.append(('full frame', 'k', False, False, self._kin(ds, ds.full)))
            if self.cmp is not None:
                out.append(('full frame (compare)', 'k', True, False, self._kin(self.cmp, self.cmp.full)))
        for r in self.regions:
            if r.decay is not None:
                out.append((f'{r.name} ({r.npix} px)', r.color, False, self._stats(ds, r.decay, r.npix)[1], self._kin(ds, r.decay)))
            if r.decay_cmp is not None and self.cmp is not None:
                out.append((f'{r.name} compare', r.color, True, self._stats(self.cmp, r.decay_cmp, r.npix)[1],
                            self._kin(self.cmp, r.decay_cmp)))
        return out

    def draw_kinetics(self):
        ax = self.ax_kin
        ax.clear()
        if not self.v_kin.get():
            ax.set_axis_off()
            return
        ax.set_axis_on()
        if TK is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, 'tau_diff vs QFLS unavailable:\n' + TK_ERROR, ha='center', va='center', fontsize=8, transform=ax.transAxes)
            return
        failed = []
        marked = False
        for lab, col, dashed, noisy, res in self.kinetics_curves():
            if isinstance(res, str):
                failed.append(lab)
                continue
            lw = 2.6 if lab.split(' (')[0] == self._selected else (0.9 if noisy else 1.4)
            ax.plot(res['dqfls_meV'], res['tau_diff_ns'], '--' if dashed else '-', color=col, lw=lw, alpha=0.55 if noisy else 0.9,
                    label=lab + (' (noisy)' if noisy else ''))
            if not marked and not dashed and res['markers']:     # time marks on the first solid curve
                for tm, q, tau in res['markers']:
                    if tm >= 10:
                        ax.plot([q], [tau], 'o', ms=3.5, color=col, zorder=5)
                        ax.annotate(('%g ns' % tm) if tm < 1000 else ('%g us' % (tm / 1000)), (q, tau), textcoords='offset points',
                                    xytext=(3, -9), fontsize=7, color='0.35')
                marked = True
        ax.set_yscale('log', nonpositive='mask')
        ax.set_xlabel('dQFLS = kT ln(PL / PL_peak) (meV)')
        ax.set_ylabel('tau_diff (ns)')
        ax.grid(True, which='major', alpha=0.25)
        ttl = 'tau_diff vs QFLS'
        if failed:
            ttl += '  (no fit: ' + ', '.join(failed[:3]) + ('...' if len(failed) > 3 else '') + ')'
        ax.set_title(ttl, fontsize=9)
        if ax.get_lines():
            ax.legend(fontsize=7, loc='upper right')

    # ------------------------------------------------------------------ mouse
    def _box_at(self, x, y):
        r = max(0, (int(self.v_psize.get()) - 1) // 2)
        ny, nx = self.ds.shape
        cx, cy = int(round(x)), int(round(y))
        return slice(max(0, cy - r), min(ny, cy + r + 1)), slice(max(0, cx - r), min(nx, cx + r + 1))

    def _on_motion(self, ev):
        ds = self.ds
        if ds is None or ev.inaxes is not self.ax_map or ev.xdata is None:
            return
        x, y = ev.xdata, ev.ydata
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < ds.shape[1] and 0 <= iy < ds.shape[0]):
            return
        b = int(self.v_bin.get())
        m = self.im.get_array()
        try:
            val = float(m[min(iy // b, m.shape[0] - 1), min(ix // b, m.shape[1] - 1)])
        except Exception:
            val = np.nan
        um = f' = ({ix * ds.pix_um:.1f}, {iy * ds.pix_um:.1f}) um' if ds.pix_um else ''
        self.l_status.config(text=f'x {ix}, y {iy}{um}   map value {val:.4g}')
        if not self.v_hover.get() or self._hover is None or ev.button is not None:
            return
        now = time.time()
        if now - self._last_hover < 0.04:
            return
        self._last_hover = now
        sy, sx = self._box_at(x, y)
        d = ds.cube[:, sy, sx].mean(axis=(1, 2))
        npx = (sy.stop - sy.start) * (sx.stop - sx.start)
        self._hover.set_data(*self._curve(ds.t, d, ds.ip, self._stats(ds, d, npx)[1]))
        self.canvas.draw_idle()

    def _on_click(self, ev):
        if self.ds is None or self.toolbar.mode:
            return
        if ev.inaxes is self.ax_map and self.v_tool.get() == 'point' and ev.button == 1 and ev.xdata is not None:
            self._point(ev.xdata, ev.ydata)
        elif ev.inaxes is self.ax_dec and ev.button == 1 and ev.xdata is not None:
            i = int(np.argmin(np.abs(self.ds.t - ev.xdata)))
            if self.v_map.get() != 'frame':
                self.v_map.set('frame')
            self.s_time.set(i)
            self.update_map()

    # ------------------------------------------------------------------ export
    def export_csv(self):
        ds = self.ds
        if ds is None:
            return
        p = filedialog.asksaveasfilename(title='Export decays', initialdir=self.last_dir, defaultextension='.csv',
                                         initialfile=os.path.splitext(ds.name)[0] + '_regions.csv',
                                         filetypes=[('CSV', '*.csv')])
        if not p:
            return
        with open(p, 'w', newline='', encoding='utf-8') as fh:
            fh.write(f'# TRPL explorer export {datetime.datetime.now().isoformat(timespec="seconds")}\n')
            fh.write(f'# file: {ds.path} [{ds.kind}]\n')
            fh.write(f'# t0 = {ds.t0:.3f} ns ({ds.t0_source}); t_after_onset = gate start + width/2 - t0\n')
            fh.write('# values: mean over the region of (Data - dark) / (gain x accumulations x gate width), per pixel\n')
            for r in self.regions:
                v = np.asarray(r.verts)
                fh.write(f'# {r.name}: {r.kind}, {r.npix} px, x {v[:, 0].min():.0f}-{v[:, 0].max():.0f}, '
                         f'y {v[:, 1].min():.0f}-{v[:, 1].max():.0f}; noise per point at the peak '
                         f'{100 * self._stats(ds, r.decay, r.npix)[0]:.1f} %; t(1/e) {self._stats(ds, r.decay, r.npix)[2]:.1f} ns, '
                         f't(1/10) {self._stats(ds, r.decay, r.npix)[3]:.1f} ns\n')
            w = csv.writer(fh)
            w.writerow(['gate_start_ns', 'gate_width_ns', 't_after_onset_ns', 'full_frame'] + [r.name for r in self.regions])
            for i in range(len(ds.t)):
                w.writerow([f'{ds.PD[i]:g}', f'{ds.PW[i]:g}', f'{ds.t[i]:.3f}', f'{ds.full[i]:.6g}']
                           + [f'{r.decay[i]:.6g}' for r in self.regions])
        extra = ''
        if TK is not None:                             # tau_diff vs QFLS curves of the right panel (2026-11-19)
            pk = os.path.splitext(p)[0] + '_taudiff_qfls.csv'
            with open(pk, 'w', newline='', encoding='utf-8') as fh:
                fh.write(f'# TRPL explorer tau_diff vs QFLS {datetime.datetime.now().isoformat(timespec="seconds")}; file {ds.path}\n')
                fh.write('# hsapp.trpl_kinetics: dQFLS = kT ln(PL/PL_peak), tau_diff = (-(1/2) dlnPL/dt)^-1 of a smooth positive-exponential '
                         f'fit over the {"real" if ds.t_eff_abs is not None else "nominal"} gate windows; T = {ds.T:g} K\n')
                w = csv.writer(fh)
                w.writerow(['curve', 't_after_onset_ns', 'pl_rel', 'dqfls_meV', 'tau_diff_ns'])
                for lab, _c, _d, _n, res in self.kinetics_curves():
                    if isinstance(res, str):
                        w.writerow([lab, 'fit failed: ' + res, '', '', ''])
                        continue
                    for i in range(len(res['t'])):
                        w.writerow([lab, f"{res['t'][i]:.4g}", f"{res['pl'][i]:.6g}", f"{res['dqfls_meV'][i]:.4f}",
                                    f"{res['tau_diff_ns'][i]:.6g}"])
            extra = f' and tau_diff vs QFLS to {pk}'
        self.status(f'exported {len(self.regions)} regions to {p}{extra}')

    def save_figure(self):
        if self.ds is None:
            return
        p = filedialog.asksaveasfilename(title='Save figure', initialdir=self.last_dir, defaultextension='.png',
                                         initialfile=os.path.splitext(self.ds.name)[0] + '_explorer.png',
                                         filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')])
        if p:
            self.fig.savefig(p, dpi=150)
            self.status(f'saved {p}')

    def save_regions(self):
        if self.ds is None or not self.regions:
            return
        p = filedialog.asksaveasfilename(title='Save regions', initialdir=self.last_dir, defaultextension='.json',
                                         initialfile=os.path.splitext(self.ds.name)[0] + '_regions.json',
                                         filetypes=[('JSON', '*.json')])
        if p:
            with open(p, 'w', encoding='utf-8') as fh:
                json.dump({'image_shape': list(self.ds.shape), 'source': self.ds.path,
                           'regions': [dict(name=r.name, kind=r.kind, color=r.color, verts=r.verts) for r in self.regions]},
                          fh, indent=1)
            self.status(f'saved {len(self.regions)} regions to {p}')

    def load_regions(self):
        if self.ds is None:
            return
        p = filedialog.askopenfilename(title='Load regions', initialdir=self.last_dir, filetypes=[('JSON', '*.json')])
        if not p:
            return
        with open(p, encoding='utf-8') as fh:
            d = json.load(fh)
        if tuple(d.get('image_shape', self.ds.shape)) != tuple(self.ds.shape):
            messagebox.showerror('TRPL explorer', f'regions were drawn on a {d["image_shape"]} image, this one is {self.ds.shape}')
            return
        names = {r.name for r in self.regions}
        for x in d.get('regions', []):
            nm = x['name'] if x['name'] not in names else x['name'] + "'"
            self.add_region(x['kind'], x['verts'], name=nm, color=x.get('color'))
        self.status(f'loaded {len(d.get("regions", []))} regions from {p}')


def main(argv=None, *, allow_correction=True):
    argv = sys.argv[1:] if argv is None else argv
    app = Explorer(argv[0] if argv else None, allow_correction=allow_correction)
    if not argv:
        app.after(200, app.open_dialog)
    app.mainloop()


if __name__ == '__main__':
    main()
