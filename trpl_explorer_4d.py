r"""trpl_explorer_4d - viewer for 4D hyperspectral TRPL cubes (IMA filter x gate delay x y x x), 2026-11-07.

    python trpl_explorer_4d.py [file.h5]      (trpl_new env: numpy, h5py, matplotlib, tkinter)
    or open a 4D file in trpl_explorer.py / TRPL_explorer.bat: it opens this window instead of the 3D one.

Files (recognised by their content, not by their name):
  v2 raw cubes      hyperspectral_trpl_acquisition_v2 (root attr file_format 'hyperspectral_trpl_v2_4d'):
                    /data (filter, delay, y, x) = (raw - matched laser-off dark) / (gain x accumulations x gate ns);
                    filter_wavelength_nm = IMA filter settings, gate_delay_ns, gate_width_ns, grating in config_json.
  old raw cubes     notebook merge '4D_cube_raw_corrected' (filter, delay, y, x) + 'wavelength' (filter settings)
                    + 'time' (delays shifted to the peak, or an intensity sum Y_cor in older files).  The delays come
                    from a per-filter file next to it (e.g. 478.0nm.h5: Metadata/pulse_delay, pulse_width).
  rectified cubes   '4D_cube' (wavelength, delay, y, x) + 'wavelength' + 'time_delay' (old notebook output or
                    rectify_4d_destripe.py).  Old files can have the two labels swapped: detected by value and by the
                    axis lengths, fixed, and reported.
Raw cubes (filter settings) can be shown per pixel on the true wavelength: each pixel's wavelength is computed from the
IMA calibration XML with the equations of Rectify.py (copied below; tested equal to Rectify to < 1e-9 nm) and the data
are interpolated linearly per pixel onto one grid, with no extrapolation (NaN outside each pixel's range).
Nothing is ever written into the data files.
"""
import csv
import datetime
import hashlib
import json
import os
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# region helpers, colours and objective table of the main window (importing it also selects the TkAgg backend)
from trpl_explorer import (COLORS, LOCAL_DEFAULTS, PIX_UM, TOOLS, Region, default_data_dir,
                           ellipse_verts, mask_from_verts, rect_verts)  # noqa: E402

from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.widgets import RectangleSelector, EllipseSelector, LassoSelector  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
import tkinter as tk  # noqa: E402
from tkinter import ttk, filedialog, messagebox, simpledialog  # noqa: E402

from trpl_graph_tools import GraphTools  # noqa: E402

__version__ = '1.1.0 (2026-11-07)'
# 1.1.0 (review of 7 Nov): opens on the brightest point of the full-frame curve of the axis it shows (G1 opened with an
#   empty decay plot); fine grid + smaller bin no longer makes the file fail to open (coarser grid instead); closing
#   the window (or opening another file) stops a running read; hover curves redrawn by blitting (~150 ms -> few ms);
#   Delete / Ctrl+Z in number fields no longer remove regions; region names stay unique; swapped-label test is
#   order-free (descending wavelength axes) and warns when both axes look like wavelengths with the same length;
#   old raw cubes whose 'wavelength' lacks values take the filter settings from the per-filter files; 3D files are
#   recognised before the generic 4D test; converted v2 files ('source' attribute) are no longer called old protocol;
#   clearer XML messages; decay plot keeps the delay range and says why the full-frame decay can be empty.

CALIBRATION_XMLS = (os.path.join(HERE, '20205-IMA.xml'),)
if LOCAL_DEFAULTS is not None:
    CALIBRATION_XMLS += LOCAL_DEFAULTS.CURRENT_CALIBRATION_XMLS + LOCAL_DEFAULTS.OLDER_CALIBRATION_XMLS
GRATING_NAMES = {1: '[400 ; 600] nm', 2: '[600 ; 1000] nm'}           # Rectify grating_type -> XML GratingName
MAX_CUBE_BYTES = 2e9                                                   # default binning keeps the cube below this
PLAN_MAX_BYTES = 900e6                                                 # interpolation plan limit (grid x pixels)
RECT_CACHE_BYTES = 700e6                                               # cache of rectified frames (per delay)
CHEAP_FRAME_BYTES = 64e6                                               # full-res frame read: max bytes decompressed
MAP_MODES = (('frame', 'Frame'), ('wlint', 'Wavelength-integrated'), ('tint', 'Time-integrated (band)'),
             ('centroid', 'Spectral centroid'))
AXIS_MODES = (('filter', 'filter setting (raw frames)'), ('true', 'true wavelength (rectified per pixel)'))
V2_OK_STATUS = {'ok', 'low_signal', 'not_tested', 'no_reference'}
ENTRY_CLASSES = ('TEntry', 'Entry', 'TSpinbox', 'Spinbox', 'TCombobox', 'Text')   # keys typed here are not commands


class _Cancelled(Exception):
    """Raised by the progress callback when the read was stopped (window closed or another file opened)."""

HELP4D = """TRPL explorer 4D - hyperspectral TRPL (IMA filter x gate delay x image)

What the window shows
  Left: an image of the sample. Right, top: spectra at the chosen gate delay. Right, bottom: decays in
  the chosen wavelength band (shaded on the spectrum plot). The two sliders under the figure choose the
  wavelength and the gate delay. Click on the spectrum plot to choose a wavelength; click on the decay
  plot to choose a delay. The window opens at the brightest gate delay and at the brightest point of
  the full-frame spectrum. Large files are read in the background; choosing another file meanwhile
  stops that read, and closing the window stops it too.
  Scroll over a graph to zoom at the pointer (Shift for x only, Ctrl for y only). The hand pans
  and the magnifier zooms into a box. Reset zoom fits both graphs. Choose Decay or Spectrum
  to enlarge it, and untick Graph only to return. Hover near a sample to read its plotted values.

Wavelength axis
  'Filter setting (raw frames)': the wavelength set on the IMA filter for each raw frame. The light that
  reaches a pixel is NOT exactly at that wavelength: the IMA shifts it across the image (about -0.083 nm
  per column on grating index 0 and about +0.14 nm per column on grating index 1, so about +-21 nm or
  +-36 nm between the centre and the left/right edges). Spectra of large regions are smeared in this mode.
  'True wavelength (rectified per pixel)': each pixel is interpolated (linear, as Rectify.py) from its
  own wavelengths onto one common grid. Where a pixel has no measured wavelength the image is grey;
  nothing is extrapolated. The pixel wavelengths come from the IMA calibration XML shown in the status
  line (File > Choose IMA XML... to use another one).
  Rectified files are already on a wavelength grid: they are shown as they are.

Image (Map)
  Frame: one wavelength, one delay.
  Wavelength-integrated: the spectrum of each pixel integrated over wavelength, at the chosen delay.
  Time-integrated (band): each pixel integrated over all delays, in the chosen wavelength band.
  Spectral centroid: mean wavelength of each pixel's spectrum (negative values set to 0), at the delay.
  In true-wavelength mode the integral and the centroid use only the wavelengths that every pixel has
  measured, so that all pixels are compared over the same range.
  'bin' is the binning used when the file is read (changing it reads the file again; the window opens
  with the smallest binning that keeps the cube below 2 GB). 'full-res frame' shows single raw frames
  at full resolution when this is cheap to read.

Regions
  Draw rectangles, ellipses, freehand shapes or click points (left mouse) on the image. Each region
  gets a colour; its spectrum and its decay appear on the right and in the table. 'Delete' removes the
  selected region, Ctrl+Z (or 'Remove last') removes the last one, double-click renames it.
  Regions are evaluated on the binned data: their edges are exact to the binning only.
  In true-wavelength mode a spectrum point is shown only where every pixel of the region has data
  (for the full frame: 99.9 % of the pixels, so the full-frame decay is empty for a wavelength outside
  the full-frame spectrum); tick 'partial coverage' to accept points where at least half have data.
  The table gives each region's spectrum peak and spectral centroid at the current delay (the centroid
  uses the wavelengths every pixel has measured, so regions on both sides of the image compare) and
  the delay of its decay maximum. 'Measured range' is the wavelength range where the region has data:
  a peak at or near its end may lie outside it ('(edge)' marks a maximum at the very end). Small regions are noisy: with 1-ns gates a 9 x 9 pixel spectrum
  jumps by about 10 % from one filter setting to the next.
  The dashed curve follows the mouse (hover probe; 'point size' sets its square). Region files (JSON)
  are the same as in the main TRPL explorer: regions saved there load here and the other way round.
  Delete and Ctrl+Z act on regions except while you type in a number field.

Numbers and warnings
  Values are those stored in the file: for new (v2) files (raw - laser-off dark) / (gain x
  accumulations x gate width in ns), per pixel. Old notebook files are also divided by a gate-width
  factor. The status line says what has NOT been corrected (for example the readout offsets that make
  moving stripes after rectification, or the gate-width efficiency).
  Log axes cannot show zero or negative values: noisy points near zero (for example before the laser
  pulse) leave gaps in the log decay plot. Untick 'log decay' to see them.
  Export CSV writes two files: the spectra at the current delay and the decays in the current band,
  for the full frame and every region.
Nothing is ever written into the data files."""


# =============================================================================================== small helpers
def _dec(x):
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode('utf-8', 'replace')
    if isinstance(x, np.ndarray) and x.shape == ():
        return _dec(x[()])
    return str(x)


def _attr(h, key, default=None):
    if key not in h.attrs:
        return default
    v = h.attrs[key]
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode('utf-8', 'replace')
    if isinstance(v, np.generic):
        return v.item()
    return v


def md5_file(path, block=1 << 22):
    m = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(block), b''):
            m.update(chunk)
    return m.hexdigest()


_MD5_CACHE = {}


def md5_cached(path):
    key = (os.path.abspath(path), os.path.getmtime(path), os.path.getsize(path))
    if key not in _MD5_CACHE:
        _MD5_CACHE[key] = md5_file(path)
    return _MD5_CACHE[key]


def parse_datetime(s):
    """'20250115160953' (IMA XML) or ISO text ('2026-11-07T00:18:59.126', '2025-08-27T13:26:24.33+01:00') -> datetime."""
    s = str(s).strip()
    if re.fullmatch(r'\d{14}', s):
        return datetime.datetime.strptime(s, '%Y%m%d%H%M%S')
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})', s)
    if m:
        return datetime.datetime(*map(int, m.groups()))
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', s)
    if m:
        return datetime.datetime(*map(int, m.groups()))
    return None


def bin2d(a, b):
    """Mean over b x b pixel blocks of the last two axes (the image is cropped to a multiple of b)."""
    if b == 1:
        return a
    *lead, ny, nx = a.shape
    y2, x2 = ny // b * b, nx // b * b
    return a[..., :y2, :x2].reshape(*lead, y2 // b, b, x2 // b, b).mean(axis=(-3, -1))


def auto_bin(n_w, n_t, ny, nx, max_bytes=MAX_CUBE_BYTES):
    """Smallest binning of 1, 2, 4, 8 that keeps the float32 cube below max_bytes."""
    for b in (1, 2, 4, 8):
        if n_w * n_t * (ny // b) * (nx // b) * 4 < max_bytes:
            return b
    return 8


def _increasing(v):
    v = np.asarray(v, float).ravel()
    return v.size >= 1 and bool(np.all(np.isfinite(v))) and (v.size == 1 or bool(np.all(np.diff(v) > 0)))


def looks_like_wavelengths(v):
    """Strictly increasing, 250-2500 nm, steps 0.05-50 nm."""
    v = np.asarray(v, float).ravel()
    if not _increasing(v) or v.min() < 250 or v.max() > 2500:
        return False
    return v.size == 1 or (np.diff(v).min() >= 0.05 and np.diff(v).max() <= 50)


def looks_like_delays(v):
    """Strictly increasing, steps >= 0.01 ns, span >= 0.5 ns, |values| < 1e7 ns (an intensity trace fails this)."""
    v = np.asarray(v, float).ravel()
    if not _increasing(v) or np.abs(v).max() >= 1e7:
        return False
    return v.size == 1 or (np.diff(v).min() >= 0.01 and v.max() - v.min() >= 0.5)


def weights_from_mask(mask, b):
    """Fraction of each b x b map pixel that lies inside a full-resolution mask."""
    m = np.asarray(mask, np.float32)
    return bin2d(m, b) if b > 1 else m


# =============================================================================================== IMA wavelength map
class IMAWavelengthMap:
    """Per-pixel wavelength of a raw IMA frame.  The equations and the XML parsing are those of
    Rectify.RectificationSingleCore (copied, so that this module does not import Rectify.py; the test compares the two).
    grating_type 1 = '[400 ; 600] nm' (IMA grating index 0), 2 = '[600 ; 1000] nm' (index 1).
    cam_type 2 = emICCD 512 x 512, 13 um pixels (1 = 1024 x 1024, 6.5 um, as in Rectify)."""

    def __init__(self, xml_file, grating_type=1, cam_type=2):
        self.xml_file = os.path.abspath(xml_file)
        self.grating_type = int(grating_type)
        root = ET.parse(xml_file).getroot()
        if cam_type == 1:
            self.width, self.height, self.pixel_size = 1024, 1024, 6.5
        else:
            self.width, self.height, self.pixel_size = 512, 512, 13
        self.x, self.y = np.meshgrid(np.linspace(-(self.width - 1) / 2, (self.width - 1) / 2, self.width),
                                     np.linspace(-(self.height - 1) / 2, (self.height - 1) / 2, self.height))
        name = GRATING_NAMES[self.grating_type]
        for grating in root.findall('.//Grating'):
            if grating.find('GeneralInfo/GratingName').text.strip() == name:
                cal = grating.find("Calibration[@Type='Current']")
                self.factor = float(cal.find('Factor').text)
                self.period = int(grating.find('Period').text)
                self.stage_offset = float(grating.find('GeneralInfo/StageOffset').text)
                self.offset = float(cal.find('Offset').text)
                self.focal_length = float(cal.find('FocalLength').text)
                self.focal_length_coeff = float(cal.find('FocalLengthCoef').text)
                self.slope = float(cal.find('Slope').text)
                self.curve = float(cal.find('Curve').text)
                rng = grating.find('Range')
                self.reg_lower = int(rng.find('RegLower').text)
                self.reg_upper = int(rng.find('RegUpper').text)
                self.BeamSide = grating.find('GeneralInfo/BeamSide').text.strip()
                dt = cal.find('DateTime')
                self.cal_date = parse_datetime(dt.text) if dt is not None and dt.text else None
                break
        else:
            raise ValueError(f'grating {name} not found in {xml_file}')

    def _wl(self, wavelength, x, y):
        # = Rectify.RectificationSingleCore.calculate_2D_wavelength_breakdown, same operations in the same order
        if self.BeamSide == 'Right':
            A = 180 - self.stage_offset - self.offset + np.rad2deg(np.arcsin(wavelength / (2 * self.factor * self.period)))
        else:
            A = 180 - self.stage_offset - self.offset - np.rad2deg(np.arcsin(wavelength / (2 * self.factor * self.period)))
        theta = (180 - A - self.stage_offset - self.offset)
        f = self.focal_length * (1 + self.focal_length_coeff * (wavelength - (self.reg_lower + self.reg_upper) / 2))
        alpha = np.rad2deg(np.arctan((x - (self.slope * y) - (self.curve * y ** 2)) * self.pixel_size / f))
        return 2 * self.factor * self.period * abs(np.sin(np.deg2rad(theta + alpha)))

    def breakdown(self, wavelength):
        """(height, width) map of the true wavelength (nm) at filter setting `wavelength` (nm); [row, column]."""
        return self._wl(wavelength, self.x, self.y)

    def at_pixel(self, wavelength, col, row):
        """True wavelength at full-resolution pixel (col, row)."""
        x = np.asarray(col, float) - (self.width - 1) / 2
        y = np.asarray(row, float) - (self.height - 1) / 2
        return self._wl(wavelength, x, y)


def xml_calibration_date(xml_file, grating_type=None):
    """Latest 'Current' calibration DateTime in an IMA XML (of one grating when grating_type is given)."""
    root = ET.parse(xml_file).getroot()
    out = []
    for g in root.findall('.//Grating'):
        nm = g.find('GeneralInfo/GratingName')
        if grating_type is not None and (nm is None or nm.text.strip() != GRATING_NAMES[grating_type]):
            continue
        cal = g.find("Calibration[@Type='Current']")
        dt = cal.find('DateTime') if cal is not None else None
        if dt is not None and dt.text:
            d = parse_datetime(dt.text)
            if d:
                out.append(d)
    return max(out) if out else None


def choose_xml(date=None, md5=None, extra=()):
    """Pick the IMA XML for a raw cube.  Returns (path or None, reason, warning or None).
    md5 (v2 files): the XML whose md5 matches; else by date: the XML with the latest calibration not after the data."""
    known = []
    for p in CALIBRATION_XMLS + tuple(extra):
        if p and os.path.isfile(p) and os.path.abspath(p).lower() not in [os.path.abspath(q).lower() for q in known]:
            known.append(p)
    if not known:
        return None, 'no IMA XML found', 'no IMA calibration XML found: choose one (File > Choose IMA XML...)'
    if md5:
        for p in known:
            if md5_cached(p) == md5:
                return p, f"its md5 {md5[:8]}.. matches the file's ima_config_xml_md5", None
    by_date, why = None, ''
    if date is not None:
        cands = []
        for i, p in enumerate(known):
            try:
                d = xml_calibration_date(p)
            except Exception:
                d = None
            if d is not None and d <= date:
                cands.append((d, -i, p))                     # latest calibration; ties: the earlier entry
        if cands:
            d, _, by_date = max(cands)
            why = f'its calibration of {d:%d %b %Y} is the latest one before the data ({date:%d %b %Y})'
    if md5:
        p = by_date or known[0]
        return p, why or 'default', (f"the file's IMA XML md5 {md5[:8]}.. matches no known XML; using {p} "
                                     f"({why or 'default'}): check it (File > Choose IMA XML...)")
    if by_date:
        return by_date, why, None
    if date is not None:
        return known[0], f'default (no known XML has a calibration before the data, {date:%d %b %Y})', (
            f'no known IMA XML has a calibration before the data ({date:%d %b %Y}): {known[0]} is used; check it '
            '(File > Choose IMA XML...)')
    return known[0], 'default (acquisition date unknown)', ('acquisition date unknown: the default IMA XML is used; '
                                                           'check it (File > Choose IMA XML...)')


# =============================================================================================== file detection
def probe_4d(h):
    """(kind, text) for an open HDF5 file: kind 'v2', 'raw', 'rect', 'other4d' or None (not a 4D cube)."""
    fmt = _attr(h, 'file_format', '')
    if fmt == 'hyperspectral_trpl_v2_4d' and isinstance(h.get('data'), h5py.Dataset) and h['data'].ndim == 4:
        return 'v2', 'v2 4D raw cube'
    if isinstance(h.get('4D_cube_raw_corrected'), h5py.Dataset) and h['4D_cube_raw_corrected'].ndim == 4:
        return 'raw', 'old notebook 4D raw cube'
    if isinstance(h.get('4D_cube'), h5py.Dataset) and h['4D_cube'].ndim == 4:
        return 'rect', 'rectified 4D cube'
    # a 3D broadband file stays in the main window even if it also holds some other 4D dataset
    if isinstance(h.get('Data'), h5py.Dataset) and h['Data'].ndim == 3 and 'Metadata' in h and 'pulse_delay' in h['Metadata']:
        return None, '3D broadband TRPL file (time, y, x): open it in the main window (TRPL explorer).'
    for k, v in h.items():
        if isinstance(v, h5py.Dataset) and v.ndim == 4:
            return 'other4d', (f"4D dataset '{k}' {v.shape} of an unknown layout. Supported: v2 files "
                               "(file_format 'hyperspectral_trpl_v2_4d', /data), '4D_cube_raw_corrected' + 'wavelength', "
                               "and '4D_cube' + 'wavelength' + 'time_delay'. (The old save_data_4d 'Unrectified4D' "
                               "files keep raw counts without dark subtraction: open the per-filter files instead.)")
    if 'Cube' in h and 'Images' in h['Cube']:
        return None, 'steady-state hyperspectral cube (Cube/Images, no gate delays): not a time-resolved 4D file.'
    return None, ("not a 4D hyperspectral TRPL file: it needs /data (v2), '4D_cube_raw_corrected' (old raw) "
                  "or '4D_cube' (rectified)")


def detect_4d(path):
    """(kind, text) of a file; kind None for files that are not 4D (and for unreadable files)."""
    try:
        with h5py.File(path, 'r') as h:
            return probe_4d(h)
    except Exception as e:
        return None, f'cannot read {path}: {e}'


def is_4d_file(path):
    """True for a 4D hyperspectral TRPL file (supported or of an unknown 4D layout); never raises."""
    return detect_4d(path)[0] is not None


_PF_RE = re.compile(r'^(\d+(?:\.\d+)?)\s*nm\.h5$', re.IGNORECASE)


def per_filter_files(folder):
    """[(filter setting nm, path)] of per-filter files (e.g. '478.0nm.h5') in a folder, sorted."""
    out = []
    try:
        names = os.listdir(folder)
    except OSError:
        return out
    for f in names:
        m = _PF_RE.match(f)
        if m and os.path.isfile(os.path.join(folder, f)):
            out.append((float(m.group(1)), os.path.join(folder, f)))
    return sorted(out)


def read_per_filter_meta(path):
    """Acquisition metadata of one per-filter file (no image data)."""
    with h5py.File(path, 'r') as h:
        md = h['Metadata']
        meta = dict(path=path,
                    delay=np.asarray(md['pulse_delay'][()], float).ravel(),
                    width=np.asarray(md['pulse_width'][()], float).ravel() if 'pulse_width' in md else None,
                    gain=np.asarray(md['emiccd_gain'][()], float).ravel() if 'emiccd_gain' in md else None,
                    acc=np.asarray(md['accumulations'][()], float).ravel() if 'accumulations' in md else None,
                    frames=np.asarray(md['frames_combined'][()], float).ravel() if 'frames_combined' in md else None,
                    objective=_dec(md['objective'][()]).strip() if 'objective' in md else '',
                    n_frames=h['Data'].shape[0] if 'Data' in h else None)
        meta['date'] = None
        if 'xml_data' in h:
            mt = re.search(r'<Origin[^>]*created="([^"]+)"', _dec(h['xml_data'][()]))
            if mt:
                meta['date'] = parse_datetime(mt.group(1))
    m = _PF_RE.match(os.path.basename(path))
    meta['lc'] = float(m.group(1)) if m else None
    return meta


# =============================================================================================== dataset
class Dataset4D:
    """One 4D hyperspectral TRPL file, read binned into memory as float32 (n_w, n_t, ny/b, nx/b).

    Common interface for the three formats:
      lam (n_w,)      filter settings (raw cubes, is_raw True) or true wavelengths (rectified cubes), ascending, nm
      delay (n_t,)    gate delays (ns), ascending; width (n_t,) gate widths (ns) or None; delay_label says what it is
      cube            float32 (n_w, n_t, ny/b, nx/b), file units (see units / normalisation)
      kind            'v2' | 'raw' | 'rect';  kind_label, units, normalisation, warnings, notes
    True wavelength for raw cubes: grid (n_g,), rect_frame(it) (n_g, P) per-pixel linear interpolation, NaN outside.
    """

    def __init__(self, path, bin=None, xml=None, grating=None, grid_step=None, progress=None,
                 max_bytes=MAX_CUBE_BYTES):
        t_start = time.time()
        self.path = os.path.abspath(path)
        self.name = os.path.basename(path)
        self.folder = os.path.dirname(self.path)
        self.warnings, self.notes, self.info = [], [], OrderedDict()
        self.width = None
        self.objective, self.pix_um = '', None
        self.date, self.date_source = None, ''
        self.grating_index, self.grating_source = None, ''
        self.xml_path, self.xml_reason, self.xml_warning, self.xml_md5_file = None, '', None, None
        self.xml_hint = None
        self.frame_info = {}
        self.per_filter = None
        self.transposed = False
        self.grid_note = ''
        self.xml_cal_date = None
        self.counts_factor = None            # v2: (n_w, n_t) gain x acc x width (file units -> camera counts)
        with h5py.File(self.path, 'r') as h:
            kind, why = probe_4d(h)
            if kind is None or kind == 'other4d':
                raise ValueError(why)
            self.kind = kind
            getattr(self, '_axes_' + kind)(h)
            self.n_w, self.n_t = len(self.lam), len(self.delay)
            ny, nx = self.full_shape
            self.b = int(bin) if bin else auto_bin(self.n_w, self.n_t, ny, nx, max_bytes)
            if ny % self.b or nx % self.b:
                self.notes.append(f'image {nx}x{ny} is cropped to a multiple of the binning {self.b}')
            self._read(h, progress)
        self.read_s = time.time() - t_start
        self._finish(xml, grating, grid_step)
        self.load_s = time.time() - t_start

    # ------------------------------------------------------------------------------------------- axes per format
    def _axes_v2(self, h):
        d = h['data']
        self.dset = 'data'
        n_w, n_t, ny, nx = d.shape
        self.full_shape = (ny, nx)
        lam = np.asarray(h['filter_wavelength_nm'][()], float).ravel()
        delay = np.asarray(h['gate_delay_ns'][()], float).ravel()
        width = np.asarray(h['gate_width_ns'][()], float).ravel() if 'gate_width_ns' in h else None
        if lam.size != n_w or delay.size != n_t:
            raise ValueError(f'axis lengths {lam.size}, {delay.size} do not match /data {d.shape}')
        self.kind_label = f"v2 4D raw cube (code {_attr(h, 'code_version', '?')}): IMA filter settings x gate delays"
        self.is_raw = True
        self.normalisation = _attr(h, 'normalisation', '(raw - matched laser-off dark) / (gain * accumulations * gate_width_ns)')
        self.units = _attr(h, 'units', 'counts / (gain x accumulations x gate width ns)')
        self.unit_short = 'signal / (gain x acc x gate ns)'
        self.delay_label = 'gate delay (ns)'
        self.objective = str(_attr(h, 'objective', '') or '').strip()
        created = _attr(h, 'created', None)
        if created:
            self.date, self.date_source = parse_datetime(created), "file attribute 'created'"
        cfg = {}
        if 'config_json' in h:
            try:
                cfg = json.loads(_dec(h['config_json'][()]))
            except Exception:
                cfg = {}
        if 'grating' in cfg and cfg['grating'] is not None:
            self.grating_index, self.grating_source = int(cfg['grating']), 'config_json'
        self.xml_md5_file = _attr(h, 'ima_config_xml_md5', None)
        if 'sessions_json' in h:
            try:
                ses = json.loads(_dec(h['sessions_json'][()]))
                ima = (ses[-1] if isinstance(ses, list) and ses else {}).get('ima', {}) if ses else {}
                self.xml_hint = ima.get('config_xml')
                self.info['IMA gratings (session)'] = json.dumps(ima.get('gratings'))
            except Exception:
                pass
        self.info['notes (config)'] = str(cfg.get('notes', ''))
        for k in ('code_version', 'created', 'complete', 'partial', 'gw_efficiency_applied', 'strip_offset_subtracted',
                  'schedule_hash', 'ima_config_xml_md5', 'temperature_K', 'run_dir'):
            if k in h.attrs:
                self.info[k] = str(_attr(h, k))
        self.order_w = np.argsort(lam, kind='stable')
        self.order_t = np.argsort(delay, kind='stable')
        self.lam, self.delay = lam[self.order_w], delay[self.order_t]
        self.width = width[self.order_t] if width is not None else None
        if 'filter_wavelength_readback_nm' in h:
            rb = np.asarray(h['filter_wavelength_readback_nm'][()], float).ravel()[self.order_w]
            dev = np.abs(rb - self.lam)
            if np.isfinite(dev).any():
                self.info['filter read-back - set (max)'] = f'{np.nanmax(dev):.3f} nm'
                if np.nanmax(dev) > 0.1:
                    self.warnings.append(f'filter read-back differs from the setting by up to {np.nanmax(dev):.2f} nm')
        fi = h.get('frame_info')
        if isinstance(fi, h5py.Group):
            for k in ('gain', 'accumulations', 'frames_combined', 'sync_master_xml', 'readback_delay_ns', 'readback_width_ns'):
                if k in fi:
                    self.frame_info[k] = np.asarray(fi[k][()], float)[self.order_w][:, self.order_t]
            if 'check_status' in fi:
                st = np.array([_dec(x) for x in fi['check_status'][()].ravel()]).reshape(fi['check_status'].shape)
                vals, cnt = np.unique(st, return_counts=True)
                self.info['frame checks'] = ', '.join(f'{v} {c}' for v, c in zip(vals, cnt))
                bad = int(sum(c for v, c in zip(vals, cnt) if v not in V2_OK_STATUS))
                if bad:
                    self.warnings.append(f'{bad} frames have a check status other than ok / low_signal / not_tested')
        if all(k in self.frame_info for k in ('gain', 'accumulations')) and self.width is not None:
            self.counts_factor = self.frame_info['gain'] * self.frame_info['accumulations'] * self.width[None, :]
        sm = self.frame_info.get('sync_master_xml')
        if sm is not None and np.isfinite(sm).any() and np.nanmin(sm) < 1:
            self.warnings.append(f'{int(np.sum(sm < 1))} data frames were read back with the laser trigger OFF')
        if 'refs' in h and 'ratio_repeat_over_first' in h['refs']:
            r = np.asarray(h['refs/ratio_repeat_over_first'][()], float)
            if np.isfinite(r).any():
                self.info['reference repeats / first (min-max)'] = f'{np.nanmin(r):.3f} - {np.nanmax(r):.3f}'
        # warnings about what the file does not correct
        self.warnings.append('raw frames: the readout offsets (they make moving stripes after rectification) are NOT corrected')
        if not bool(_attr(h, 'gw_efficiency_applied', False)):
            if self.width is not None and np.unique(self.width).size == 1:
                self.notes.append(f'gate-width efficiency not applied (all gates {self.width[0]:g} ns: one global scale)')
            else:
                self.warnings.append('gate-width efficiency NOT applied: frames with different gate widths are not on one scale')
        if _attr(h, 'partial', False) or not _attr(h, 'complete', True):
            self.warnings.append('PARTIAL file (the run did not finish)')

    def _find_per_filter(self, lam_settings, n_t):
        """Metadata of a per-filter file next to the cube whose filter setting is one of the cube's and whose number of
        delays matches; None when there is none."""
        cands = per_filter_files(self.folder)
        if not cands:
            return None
        ref = lam_settings[0] if lam_settings is not None and len(lam_settings) else None
        if ref is not None:
            cands.sort(key=lambda c: abs(c[0] - ref))
        for lc, p in cands[:8]:
            if lam_settings is not None and not np.any(np.isclose(lam_settings, lc, atol=0.05)):
                continue
            try:
                meta = read_per_filter_meta(p)
            except Exception:
                continue
            if meta['delay'].size == n_t:
                return meta
        return None

    def _check_per_filter(self, h, lam_file, pf=None):
        """Does the per-filter file belong to this cube?  At two delays, (Data - dark) / (gain x acc x width) of the
        per-filter file divided by the cube slice of the same filter setting must be one constant (the gate-width
        factor of the old notebook).  A file of another run gives a random ratio."""
        pf = pf if pf is not None else self.per_filter
        name = os.path.basename(pf['path'])
        try:
            k = int(np.argmin(np.abs(lam_file - pf['lc'])))
            if abs(lam_file[k] - pf['lc']) > 0.05:
                return
            ds = h[self.dset]
            res = []
            with h5py.File(pf['path'], 'r') as g:
                D, md = g['Data'], g['Metadata']
                dark = np.asarray(md['dark_data'][()], np.float64) if 'dark_data' in md else 0.0
                n = D.shape[0]
                for t in sorted({n // 2, n - 1}):
                    norm = (np.asarray(D[t], np.float64) - dark) / (pf['gain'][t] * pf['acc'][t] * pf['width'][t])
                    c = np.asarray(ds[k, t], np.float64)
                    ok = np.isfinite(c) & (np.abs(c) > 1e-3 * np.nanmax(np.abs(c)))
                    r = norm[ok] / c[ok]
                    med = float(np.median(r))
                    res.append((t, med, float(np.median(np.abs(r / med - 1)))))
            worst = max(x[2] for x in res)
            if not worst < 0.02:
                self.warnings.append(f'the per-filter file {name} does NOT match the cube (relative spread {worst:.2g}): '
                                     'its delays may belong to another run')
            else:
                key = 'per-filter file check' if 'per-filter file check' not in self.info else f'per-filter file check ({name})'
                self.info[key] = (f'{name} matches the cube slice {lam_file[k]:g} nm (spread '
                                                      f'{worst:.1e}); file value = (raw - dark)/(gain x acc x width) / f, '
                                                      + ', '.join(f'f = {m:.4g} at frame {t}' for t, m, _ in res))
        except Exception as e:
            self.notes.append(f'per-filter file check skipped ({e})')

    def _delays_from(self, t_file, t_name, n_t):
        """Delay axis for old files: per-filter file > the file's own axis (if it looks like delays) > frame index."""
        pf = self.per_filter
        t_ok = t_file is not None and t_file.size == n_t and looks_like_delays(t_file)
        if pf is not None:
            delay, width = pf['delay'], pf['width']
            src = os.path.basename(pf['path'])
            if t_ok:
                off = delay - t_file
                if np.allclose(off, off[0], atol=1e-6):
                    if abs(off[0]) > 1e-9:
                        self.notes.append(f"'{t_name}' in the file = gate delay - {off[0]:g} ns; gate delays from {src}")
                else:
                    self.warnings.append(f"'{t_name}' in the file is not the gate delays of {src} (not a constant shift); "
                                         f"the delays of {src} are used")
            elif t_file is not None and t_file.size == n_t:
                self.warnings.append(f"'{t_name}' in the file holds an intensity trace (Y_cor), not delays; "
                                     f"gate delays from {src}")
            else:
                self.notes.append(f'gate delays from {src}')
            self.delay_label = 'gate delay (ns)'
            return np.asarray(delay, float), (np.asarray(width, float) if width is not None and width.size == n_t else None)
        if t_ok:
            self.warnings.append(f"no per-filter file next to the cube: the time axis is '{t_name}' from the file "
                                 "(it may be shifted to the PL peak); gate widths unknown")
            self.delay_label = f"'{t_name}' from the file (ns)"
            return np.asarray(t_file, float), None
        self.warnings.append(f"no delay axis: '{t_name}' is not a delay axis and no per-filter file (e.g. 478.0nm.h5) "
                             "is next to the cube; the delay axis is the frame index")
        self.delay_label = 'frame index'
        return np.arange(n_t, dtype=float), None

    def _old_meta_common(self):
        pf = self.per_filter
        if pf is None:
            return
        self.objective = pf.get('objective') or ''
        if pf.get('date'):
            self.date, self.date_source = pf['date'], f"LightField XML of {os.path.basename(pf['path'])}"
        if pf.get('gain') is not None:
            g, a = pf['gain'], pf['acc']
            self.info['gain (per-filter file)'] = f'{np.min(g):g}-{np.max(g):g}'
            if a is not None:
                self.info['accumulations (per-filter file)'] = f'{np.min(a):g}-{np.max(a):g}'

    def _date_fallback(self):
        if self.date is not None:
            return
        m = re.search(r'(20\d\d)[_-](\d\d)[_-](\d\d)', self.path)
        if m:
            try:
                self.date = datetime.datetime(*map(int, m.groups()))
                self.date_source = 'folder name'
                return
            except ValueError:
                pass
        self.date = datetime.datetime.fromtimestamp(os.path.getmtime(self.path))
        self.date_source = 'file modification time'

    def _axes_raw(self, h):
        d = h['4D_cube_raw_corrected']
        self.dset = '4D_cube_raw_corrected'
        n_w, n_t, ny, nx = d.shape
        self.full_shape = (ny, nx)
        if 'wavelength' not in h:
            raise ValueError("'4D_cube_raw_corrected' without 'wavelength' (filter settings)")
        lam = np.asarray(h['wavelength'][()], float).ravel()
        extra_pf = []
        if lam.size != n_w:
            # old notebooks sometimes lost values of 'wavelength' (e.g. 26 values for 27 filters, 2024_12_11): the
            # per-filter files next to the cube give the settings when there is one per filter and they contain 'wavelength'
            pf_all = per_filter_files(self.folder)
            lcs = np.array([lc for lc, _ in pf_all])
            if (lcs.size == n_w and lam.size and looks_like_wavelengths(lcs)
                    and all(np.any(np.isclose(lcs, v, atol=0.05)) for v in lam)):
                self.warnings.append(f"'wavelength' in the file has {lam.size} values for {n_w} filter settings: the "
                                     f"filter settings are taken from the {n_w} per-filter files next to the cube "
                                     f"({lcs.min():g}-{lcs.max():g} nm)")
                extra_pf = [p for lc, p in pf_all if not np.any(np.isclose(lam, lc, atol=0.05))]
                lam = lcs
            else:
                raise ValueError(f"'wavelength' has {lam.size} values, the cube has {n_w} filter settings")
        if not looks_like_wavelengths(np.sort(lam)):
            raise ValueError(f"'wavelength' does not hold filter settings in nm ({lam[:3]} ...)")
        t_file = np.asarray(h['time'][()], float).ravel() if 'time' in h else None
        self.kind_label = "old notebook 4D raw cube ('4D_cube_raw_corrected'): IMA filter settings x gate delays"
        self.is_raw = True
        src = _attr(h, 'source', None)
        if src:                                        # written by a converter (e.g. v2_to_destripe.py), not the notebook
            self.kind_label = f"4D raw cube ('4D_cube_raw_corrected') converted from {os.path.basename(str(src))}"
            self.normalisation = f'as in the source file {src}'
            self.units = 'as in the source file'
            self.unit_short = 'signal (source units)'
        else:
            self.normalisation = ('(raw - dark of each filter setting) / (gain x accumulations x gate width x gate-width '
                                  'factor) (old notebook merge)')
            self.units = 'old notebook units: counts / (gain x acc x gate ns x GW factor)'
            self.unit_short = 'signal / (gain x acc x gate ns x GW factor)'
        self.order_w = np.argsort(lam, kind='stable')
        self.per_filter = self._find_per_filter(lam, n_t)
        if self.per_filter is not None:
            self._check_per_filter(h, lam)
        for p in extra_pf[:2]:                         # filter settings taken from the per-filter files: check them too
            try:
                m = read_per_filter_meta(p)
                if m['delay'].size == n_t and m.get('gain') is not None and m.get('acc') is not None:
                    self._check_per_filter(h, lam, m)
            except Exception as e:
                self.notes.append(f'check of {os.path.basename(p)} skipped ({e})')
        delay, width = self._delays_from(t_file, 'time', n_t)
        self.order_t = np.argsort(delay, kind='stable')
        self.lam, self.delay = lam[self.order_w], delay[self.order_t]
        self.width = width[self.order_t] if width is not None else None
        self._old_meta_common()
        self._date_fallback()
        if 'wavelength_after_rectificiation' in h:
            war = np.asarray(h['wavelength_after_rectificiation'][()], float).ravel()
            self.info['notebook rectification grid'] = f'{war.min():g}-{war.max():g} nm ({war.size} points)'
        self.warnings.append('raw frames: the readout offsets (moving stripes after rectification) are NOT corrected')
        if src:
            self.notes.append(f'converted from {src} (attribute source); its own warnings apply')
        else:
            self.notes.append('old protocol: one dark per filter setting; after the first filter the darks were taken '
                              'with the laser trigger ON (collect_data_spectral_var)')

    def _axes_rect(self, h):
        d = h['4D_cube']
        self.dset = '4D_cube'
        n0, n1, ny, nx = d.shape
        self.full_shape = (ny, nx)
        a = np.asarray(h['wavelength'][()], float).ravel() if 'wavelength' in h else None
        bt = np.asarray(h['time_delay'][()], float).ravel() if 'time_delay' in h else None
        def wl_like(v):                                # order-free: a descending wavelength axis is still one
            return v is not None and v.size >= 1 and looks_like_wavelengths(np.sort(v))
        lab_ok = a is not None and a.size == n0 and wl_like(a)
        swap_ok = bt is not None and bt.size == n0 and wl_like(bt)
        tr_ok = a is not None and a.size == n1 and n1 != n0 and wl_like(a)
        if lab_ok:
            lam, t_file, t_name = a, bt, 'time_delay'
            if swap_ok and n0 == n1 and not _attr(h, 'labels', ''):
                self.warnings.append("'wavelength' and 'time_delay' have the same length and both look like wavelengths "
                                     "(nm): the labels are taken as they are - check the two axes (File info)")
        elif swap_ok:
            lam, t_file, t_name = bt, a, 'wavelength'
            self.warnings.append("labels SWAPPED in the file: 'time_delay' holds the wavelengths and 'wavelength' the "
                                 "delay axis - fixed here")
        elif tr_ok:
            self.transposed = True
            lam, t_file, t_name = a, bt, 'time_delay'
            n0, n1 = n1, n0
            self.notes.append("cube stored as (delay, wavelength, y, x): read transposed")
        else:
            raise ValueError("cannot identify the wavelength axis of '4D_cube' (neither 'wavelength' nor 'time_delay' "
                             f"holds {n0} increasing wavelengths in nm)")
        self.is_raw = False
        self.order_w = np.argsort(lam, kind='stable')
        src_meta = _attr(h, 'source_meta', None)
        lab_attr = _attr(h, 'labels', '')
        script = _attr(h, 'script', '')
        if script:
            ver = _attr(h, 'version', '?')
            self.kind_label = f"rectified 4D cube ({script} {ver}): true wavelength x gate delays"
            done = [k for k in ('offsets_removed', 'baseline_removed', 'throughput_corrected') if _attr(h, k, False)]
            self.notes.append('corrections in the file: ' + (', '.join(done) if done else 'none listed'))
            if not _attr(h, 'offsets_removed', False):
                self.warnings.append('readout offsets (stripes) NOT removed')
            self.xml_hint = _attr(h, 'device_xml', None)
            for k in ('device_xml', 'device_xml_md5', 'interpolation', 'throughput_status', 'created', 'source_raw4d'):
                if k in h.attrs:
                    self.info[k] = str(_attr(h, k))
        else:
            self.kind_label = 'rectified 4D cube (old notebook): true wavelength x gate delays'
            self.warnings.append('old notebook rectification: the readout offsets (moving vertical stripes) are NOT corrected')
        self.units = _attr(h, 'units', 'old notebook units: counts / (gain x acc x gate ns x GW factor)')
        self.normalisation = ('as the 4D raw cube it was made from: (raw - dark) / (gain x accumulations x gate width '
                              'x gate-width factor)')
        self.unit_short = 'signal / (gain x acc x gate ns x GW factor)'
        # delays: correct-label files with a real delay axis need no per-filter file, but use it when it is there
        n_t = n1
        meta_path = None
        if src_meta and os.path.isfile(src_meta):
            meta_path = src_meta
        if meta_path:
            try:
                m = read_per_filter_meta(meta_path)
                self.per_filter = m if m['delay'].size == n_t else None
            except Exception:
                self.per_filter = None
        if self.per_filter is None:
            self.per_filter = self._find_per_filter(None, n_t)
        if lab_attr and t_file is not None and t_file.size == n_t and looks_like_delays(t_file):
            delay = t_file
            width = np.asarray(h['pulse_width'][()], float).ravel() if 'pulse_width' in h else None
            if width is not None and width.size != n_t:
                width = None
            self.delay_label = 'gate delay (ns)'
            if self.per_filter is not None:
                self._old_meta_common()
        else:
            delay, width = self._delays_from(t_file, t_name, n_t)
            self._old_meta_common()
        self.order_t = np.argsort(delay, kind='stable')
        self.lam, self.delay = lam[self.order_w], np.asarray(delay, float)[self.order_t]
        self.width = width[self.order_t] if width is not None else None

    # ------------------------------------------------------------------------------------------- reading
    def _read(self, h, progress):
        ds = h[self.dset]
        b = self.b
        ny, nx = self.full_shape
        nyb, nxb = ny // b, nx // b
        n_w, n_t = self.n_w, self.n_t
        self.cube = np.empty((n_w, n_t, nyb, nxb), np.float32)
        inv_w = np.empty(n_w, np.intp)
        inv_w[self.order_w] = np.arange(n_w)
        inv_t = np.empty(n_t, np.intp)
        inv_t[self.order_t] = np.arange(n_t)
        ch = ds.chunks or ((1, 1) + ds.shape[2:])
        cw, ct = (ch[1], ch[0]) if self.transposed else (ch[0], ch[1])
        frame_bytes = ny * nx * ds.dtype.itemsize
        step_t = n_t
        if cw * n_t * frame_bytes > 256e6:
            step_t = max(ct, int(256e6 // (cw * frame_bytes)) // ct * ct)
        total, done = n_w * n_t, 0
        for k0 in range(0, n_w, cw):
            k1 = min(n_w, k0 + cw)
            for t0 in range(0, n_t, step_t):
                t1 = min(n_t, t0 + step_t)
                if self.transposed:
                    blk = np.asarray(ds[t0:t1, k0:k1], np.float32).swapaxes(0, 1)
                else:
                    blk = np.asarray(ds[k0:k1, t0:t1], np.float32)
                self.cube[inv_w[k0:k1][:, None], inv_t[t0:t1][None, :]] = bin2d(blk, b)
                del blk
                done += (k1 - k0) * (t1 - t0)
                if progress:
                    progress(done, total)
        # cost of reading one full-resolution frame (chunks touched)
        cy, cx = ch[2], ch[3]
        n_ch = -(-ny // cy) * -(-nx // cx)
        self.frame_read_bytes = n_ch * int(np.prod(ch)) * ds.dtype.itemsize
        self.full_frame_cheap = self.b > 1 and self.frame_read_bytes <= CHEAP_FRAME_BYTES

    def _finish(self, xml, grating, grid_step):
        self.shape = self.cube.shape[2:]
        self.centre = self.delay + self.width / 2.0 if self.width is not None else self.delay.copy()
        self.pix_um = PIX_UM.get(self.objective)
        self.full_spec = np.empty((self.n_w, self.n_t))                   # full-frame mean, per filter (low memory)
        nan_frames = 0
        for k in range(self.n_w):
            blk = self.cube[k].reshape(self.n_t, -1)
            fin = np.isfinite(blk)
            nan_frames += int(np.sum(~fin.all(axis=1)))
            n = fin.sum(axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                self.full_spec[k] = np.where(fin, blk, 0).sum(axis=1, dtype=np.float64) / n
        if nan_frames and self.kind != 'rect':
            self.warnings.append(f'{nan_frames} of {self.n_w * self.n_t} frames contain missing values (NaN)')
        tot = np.nansum(self.full_spec, axis=0)
        self.it_peak = int(np.nanargmax(tot)) if np.isfinite(tot).any() else 0
        col = self.full_spec[:, self.it_peak]
        self.iw_peak = int(np.nanargmax(col)) if np.isfinite(col).any() else 0
        steps = np.diff(self.lam)
        self.lam_step = float(np.median(steps)) if steps.size else 2.0
        self._full_cache = OrderedDict()
        self._rect_cache = OrderedDict()
        self._ff_cache = OrderedDict()
        self._plan = None
        self.imap = None
        self.W = None
        self.grid = None
        self.true_ok, self.true_reason = False, ''
        if self.kind == 'rect':
            fin = np.isfinite(self.cube[:, self.it_peak]).all(axis=(1, 2))
            self.rect_common = np.flatnonzero(fin) if fin.any() else np.arange(self.n_w)
            if not fin.all():
                self.notes.append(f'{int((~fin).sum())} of {self.n_w} wavelengths have empty pixels (outside the measured '
                                  'range): integrals and centroids use the other wavelengths')
            self.true_reason = 'the file is already rectified (true wavelength axis)'
        if self.is_raw:
            self._setup_map(xml, grating)
            if self.true_ok:
                self._set_grid_safe(grid_step)

    # ------------------------------------------------------------------------------------------- wavelength map
    def _guess_grating(self):
        lo, hi = float(self.lam.min()), float(self.lam.max())
        if hi <= 600:
            return 0, f'filter settings {lo:g}-{hi:g} nm <= 600 nm'
        if lo >= 600:
            return 1, f'filter settings {lo:g}-{hi:g} nm >= 600 nm'
        if hi > 710:
            return 1, f'filter settings up to {hi:g} nm (beyond grating index 0, 370-710 nm)'
        if lo < 450:
            return 0, f'filter settings from {lo:g} nm (below grating index 1, 450-1025 nm)'
        return (0 if (600 - lo) >= (hi - 600) else 1), f'filter settings {lo:g}-{hi:g} nm cross 600 nm: GUESSED'

    def _setup_map(self, xml=None, grating=None):
        self.imap, self.W, self.grid, self._plan = None, None, None, None
        self._rect_cache.clear()
        self._ff_cache.clear()
        self.true_ok = False
        self.xml_cal_date = None                       # never show the calibration date of a previous XML
        if grating is not None:
            self.grating_index, self.grating_source = int(grating), 'chosen by the user'
        elif self.grating_index is None:
            gi, why = self._guess_grating()
            self.grating_index, self.grating_source = gi, 'from the filter range (' + why + ')'
            if 'GUESSED' in why:
                self.warnings.append('IMA grating guessed from the filter range: check it (File > IMA grating)')
        self.rect_grating = 1 if self.grating_index == 0 else 2
        if xml:
            self.xml_path, self.xml_reason, self.xml_warning = os.path.abspath(xml), 'chosen by the user', None
            try:
                m5 = md5_cached(xml)
                if self.xml_md5_file and m5 != self.xml_md5_file:
                    self.xml_warning = (f'the chosen IMA XML (md5 {m5[:8]}..) is not the one recorded in the file '
                                        f'(md5 {self.xml_md5_file[:8]}..)')
            except OSError as e:
                self.xml_warning = f'cannot read the chosen IMA XML: {e}'
        else:
            extra = (self.xml_hint,) if self.xml_hint else ()
            self.xml_path, self.xml_reason, self.xml_warning = choose_xml(self.date, self.xml_md5_file, extra)
        if self.n_w < 2:
            self.true_reason = 'one filter setting only: no spectrum to interpolate'
        elif self.full_shape != (512, 512):
            self.true_reason = f'image {self.full_shape} is not the full 512 x 512 emICCD frame of the IMA map'
        elif not self.xml_path:
            self.true_reason = 'no IMA XML'
        try:
            if self.xml_path:
                self.imap = IMAWavelengthMap(self.xml_path, self.rect_grating, cam_type=2)
                self.xml_cal_date = self.imap.cal_date
        except Exception as e:
            self.imap = None
            self.true_reason = f'cannot read the IMA XML: {e}'
        if self.imap is None or self.n_w < 2 or self.full_shape != (512, 512):
            return
        b = self.b
        W = np.empty((self.n_w,) + self.shape, np.float64)
        for k, lc in enumerate(self.lam):
            W[k] = bin2d(self.imap.breakdown(float(lc)), b)
        if not np.all(np.diff(W, axis=0) > 0):
            self.true_reason = 'the per-pixel wavelengths do not increase with the filter setting'
            return
        self.W = W
        self.common = (float(W[0].max()), float(W[-1].min()))
        self.extended = (float(W[0].min()), float(W[-1].max()))
        dmap = W[self.n_w // 2] - self.lam[self.n_w // 2]
        self.offset_range = (float(dmap.min()), float(dmap.max()))
        self.true_ok = True
        self.true_reason = ''

    def set_xml(self, xml):
        self._setup_map(xml=xml, grating=self.grating_index if self.grating_source == 'chosen by the user' else None)
        if self.true_ok:
            self._set_grid_safe(getattr(self, 'grid_step', None))

    def set_grating(self, index):
        self._setup_map(xml=self.xml_path if self.xml_reason == 'chosen by the user' else None, grating=index)
        if self.true_ok:
            self._set_grid_safe(getattr(self, 'grid_step', None))

    def _set_grid_safe(self, step=None):
        """set_grid(step); when that grid needs more than PLAN_MAX_BYTES at this binning (e.g. a 0.5-nm grid kept
        through a change to bin 1), use the filter step or a coarser one instead of failing, and say so."""
        self.grid_note = ''
        s0 = float(step) if step else self.lam_step
        cands = [s0]                                   # the requested step first, then coarser ones only
        for s in [s0 * 2 ** k for k in range(1, 6)] + [self.lam_step * m for m in (1, 2, 4, 8)]:
            if s > s0 + 1e-9 and all(abs(s - c) > 1e-9 for c in cands):
                cands.append(s)
        cands = [cands[0]] + sorted(cands[1:])
        for s in cands:
            try:
                self.set_grid(s)
            except MemoryError:
                continue
            if step and abs(s - float(step)) > 1e-9:
                self.grid_note = (f'a {float(step):g}-nm true-wavelength grid needs too much memory at bin {self.b}: '
                                  f'the grid is {s:g} nm')
            return
        self.true_ok = False
        self.true_reason = f'the true-wavelength grid needs too much memory at bin {self.b}: choose a larger bin'

    def set_grid(self, step=None):
        """Common true-wavelength grid (default step = the filter step) over the range seen by any pixel."""
        if not self.true_ok:
            return
        step = float(step) if step else self.lam_step
        lo, hi = self.extended
        start = np.ceil(lo / step - 1e-9) * step
        stop = np.floor(hi / step + 1e-9) * step
        grid = np.round(np.arange(start, stop + step / 2, step), 9)
        if grid.size * int(np.prod(self.shape)) * 9 > PLAN_MAX_BYTES:
            raise MemoryError(f'a {step:g}-nm grid at bin {self.b} needs more than {PLAN_MAX_BYTES / 1e6:.0f} MB: '
                              'choose a coarser grid or a larger bin')
        self.grid_step, self.grid = step, grid
        self._plan = None
        self._rect_cache.clear()
        self._ff_cache.clear()
        c0, c1 = self.common
        self.common_idx_true = np.flatnonzero((grid >= c0 - 1e-6) & (grid <= c1 + 1e-6))

    def _get_plan(self):
        """For every grid wavelength and map pixel: index k of the lower node and weight of node k+1 (linear, the
        same as Rectify.Linear_Interpolation inside the data range); k = -1 outside the pixel's range (NaN)."""
        if self._plan is not None:
            return self._plan
        W = self.W.reshape(self.n_w, -1)
        g = self.grid
        n_g, P = g.size, W.shape[1]
        g0, dg = float(g[0]), float(self.grid_step)
        kk = np.full((n_g, P), -1, np.int32)
        ww = np.zeros((n_g, P), np.float32)
        u = (W - g0) / dg                                    # fractional grid index of every node
        first = np.ceil(u - 1e-9).astype(np.int64)           # first grid index at or above each node
        pidx = np.arange(P)
        for k in range(self.n_w - 1):
            j0 = first[k]
            j1 = first[k + 1] if k < self.n_w - 2 else np.floor(u[k + 1] + 1e-9).astype(np.int64) + 1
            dW = W[k + 1] - W[k]
            n_m = int((j1 - j0).max()) if P else 0
            for m in range(max(n_m, 0)):
                j = j0 + m
                sel = (j < j1) & (j >= 0) & (j < n_g)
                if not sel.any():
                    continue
                p, jj = pidx[sel], j[sel]
                kk[jj, p] = k
                ww[jj, p] = ((g[jj] - W[k, p]) / dW[p]).astype(np.float32)
        self._plan = (kk, ww)
        return self._plan

    def rect_frame(self, it):
        """All grid wavelengths at delay it: (n_g, P) float32, NaN outside each pixel's measured range (cached)."""
        it = int(it)
        if it in self._rect_cache:
            self._rect_cache.move_to_end(it)
            return self._rect_cache[it]
        kk, ww = self._get_plan()
        F = self.cube[:, it].reshape(self.n_w, -1)
        valid = kk >= 0
        k0 = np.where(valid, kk, 0)
        lo = np.take_along_axis(F, k0, 0)
        hi = np.take_along_axis(F, np.minimum(k0 + 1, self.n_w - 1), 0)
        out = lo + (hi - lo) * ww
        out[~valid] = np.nan
        self._rect_cache[it] = out
        while len(self._rect_cache) > 1 and sum(a.nbytes for a in self._rect_cache.values()) > RECT_CACHE_BYTES:
            self._rect_cache.popitem(last=False)
        return out

    def _interp_subset(self, J, P, T):
        """True-wavelength values at grid indices J, flat map pixels P, delay indices T: (nJ, nP, nT), NaN outside."""
        kk, ww = self._get_plan()
        J, P, T = np.asarray(J, np.intp), np.asarray(P, np.intp), np.asarray(T, np.intp)
        k = kk[np.ix_(J, P)]
        w = ww[np.ix_(J, P)]
        valid = k >= 0
        k0 = np.where(valid, k, 0)
        F = self.cube.reshape(self.n_w, self.n_t, -1)
        pp = np.broadcast_to(P[None, :], k0.shape)[..., None]
        tt = T[None, None, :]
        lo = F[k0[..., None], tt, pp]
        hi = F[np.minimum(k0 + 1, self.n_w - 1)[..., None], tt, pp]
        out = lo + (hi - lo) * w[..., None]
        out[np.broadcast_to(~valid[..., None], out.shape)] = np.nan
        return out

    # ------------------------------------------------------------------------------------------- axes and access
    def use_interp(self, mode):
        return self.is_raw and mode == 'true' and self.true_ok

    def axis(self, mode):
        """(values, label) of the wavelength axis in this mode."""
        if self.use_interp(mode):
            return self.grid, 'true wavelength (nm)'
        if self.is_raw:
            return self.lam, 'filter setting (nm)'
        return self.lam, 'wavelength (nm, rectified file)'

    def common_idx(self, mode):
        """Axis indices that every pixel has measured (true mode), all filter settings (filter mode) or the
        wavelengths without empty pixels (rectified files)."""
        if self.use_interp(mode):
            return self.common_idx_true
        if self.kind == 'rect':
            return self.rect_common
        return np.arange(self.n_w)

    def band_idx(self, mode, lam0, width):
        x = self.axis(mode)[0]
        J = np.flatnonzero(np.abs(x - lam0) <= width / 2.0 + 1e-6)
        if J.size == 0:
            J = np.array([int(np.argmin(np.abs(x - lam0)))])
        return J

    def frame_values(self, mode, it):
        """(axis, V) with V (n_axis, P) the whole image at delay it."""
        if self.use_interp(mode):
            return self.grid, self.rect_frame(it)
        return self.lam, self.cube[:, int(it)].reshape(self.n_w, -1)

    def band_values(self, mode, J, P=None):
        """(nJ, nP, n_t) values at axis indices J for map pixels P (all when None)."""
        T = np.arange(self.n_t)
        if P is None:
            P = np.arange(int(np.prod(self.shape)))
        if self.use_interp(mode):
            return self._interp_subset(J, P, T)
        F = self.cube.reshape(self.n_w, self.n_t, -1)
        return F[np.ix_(np.asarray(J, np.intp), T, np.asarray(P, np.intp))].transpose(0, 2, 1)

    # ------------------------------------------------------------------------------------------- curves
    def _ff_put(self, key, val):
        self._ff_cache[key] = val
        while len(self._ff_cache) > 96:
            self._ff_cache.popitem(last=False)

    @staticmethod
    def _wmean(V, w, thr):
        """Weighted mean over axis 1 of V (..., nP, ...) with weights w (nP); NaN where the valid weight < thr."""
        ok = np.isfinite(V)
        wb = w.reshape((1, -1) + (1,) * (V.ndim - 2))
        wsum = (ok * wb).sum(axis=1)
        num = (np.where(ok, V, 0.0) * wb).sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            y = num / wsum
        y[~(wsum >= thr * w.sum() - 1e-9)] = np.nan
        return y

    def spectrum(self, mode, it, wb=None, partial=False):
        """Mean spectrum of a region (map-pixel weights wb, None = full frame) at delay it: (axis, y)."""
        x = self.axis(mode)[0]
        thr = 0.5 if partial else 0.999
        if wb is None:
            key = ('s', self.use_interp(mode), int(it), bool(partial))
            if key not in self._ff_cache:
                V = self.frame_values(mode, it)[1]
                self._ff_put(key, self._wmean(V, np.ones(V.shape[1], np.float32), thr))
            return x, self._ff_cache[key]
        wr = wb.ravel()
        P = np.flatnonzero(wr > 0)
        if P.size == 0:
            return x, np.full(x.size, np.nan)
        if self.use_interp(mode):
            V = self._interp_subset(np.arange(x.size), P, [int(it)])[:, :, 0]
        else:
            V = self.cube[:, int(it)].reshape(self.n_w, -1)[:, P]
        return x, self._wmean(V, wr[P].astype(np.float64), thr)

    def decay(self, mode, lam0, width, wb=None, partial=False):
        """Mean decay of a region in the band lam0 +- width/2: mean over the band's axis points of the region mean."""
        J = self.band_idx(mode, lam0, width)
        thr = 0.5 if partial else 0.999
        if wb is None:
            key = ('d', self.use_interp(mode), tuple(int(j) for j in J), bool(partial))
            if key not in self._ff_cache:
                V = self.band_values(mode, J, None)
                with np.errstate(invalid='ignore'), _quiet():
                    self._ff_put(key, np.nanmean(self._wmean(V, np.ones(V.shape[1]), thr), axis=0))
            return self._ff_cache[key]
        else:
            wr = wb.ravel()
            P = np.flatnonzero(wr > 0)
            if P.size == 0:
                return np.full(self.n_t, np.nan)
            w = wr[P].astype(np.float64)
        V = self.band_values(mode, J, P)                     # (nJ, nP, n_t)
        if w is None:
            w = np.ones(V.shape[1])
        y = self._wmean(V, w, thr)                           # (nJ, n_t)
        with np.errstate(invalid='ignore'), _quiet():
            return np.nanmean(y, axis=0)

    # ------------------------------------------------------------------------------------------- maps
    def map_frame(self, mode, ia, it):
        if self.use_interp(mode):
            return self.rect_frame(it)[int(ia)].reshape(self.shape)
        return self.cube[int(ia), int(it)]

    def map_wl_integrated(self, mode, it):
        x, V = self.frame_values(mode, it)
        J = self.common_idx(mode)
        if J.size < 2:
            return np.full(self.shape, np.nan, np.float32)
        out = _trapz(V[J], x[J], axis=0)
        return out.reshape(self.shape).astype(np.float32)

    def map_centroid(self, mode, it):
        x, V = self.frame_values(mode, it)
        J = self.common_idx(mode)
        if J.size < 2:
            return np.full(self.shape, np.nan, np.float32)
        Vc = np.clip(V[J], 0, None)
        with np.errstate(invalid='ignore', divide='ignore'):
            out = (Vc * x[J][:, None]).sum(0) / Vc.sum(0)
        return out.reshape(self.shape).astype(np.float32)

    def map_time_integrated(self, mode, lam0, width):
        J = self.band_idx(mode, lam0, width)
        V = self.band_values(mode, J)                        # (nJ, P, n_t)
        if self.n_t < 2:
            out = V[..., 0]
        else:
            out = _trapz(V, self.centre, axis=2)             # (nJ, P)
        with np.errstate(invalid='ignore'), _quiet():
            out = np.nanmean(out, axis=0)
        return out.reshape(self.shape).astype(np.float32)

    def full_frame(self, ia, it):
        """Full-resolution frame (ny, nx) float32 read from the file (raw filter axis or rectified file), cached."""
        key = (int(ia), int(it))
        if key in self._full_cache:
            self._full_cache.move_to_end(key)
            return self._full_cache[key]
        kf, tf = int(self.order_w[int(ia)]), int(self.order_t[int(it)])
        with h5py.File(self.path, 'r') as h:
            ds = h[self.dset]
            a = ds[tf, kf] if self.transposed else ds[kf, tf]
        a = np.asarray(a, np.float32)
        self._full_cache[key] = a
        while len(self._full_cache) > 6:
            self._full_cache.popitem(last=False)
        return a

    def pixel_wavelength(self, ia, col, row):
        """True wavelength (nm) of full-resolution pixel (col, row) for filter setting index ia (raw cubes)."""
        if self.imap is None or not self.is_raw:
            return np.nan
        return float(self.imap.at_pixel(float(self.lam[int(ia)]), col, row))

    # ------------------------------------------------------------------------------------------- text
    def xml_text(self):
        if not self.is_raw:
            return 'IMA XML: not needed (rectified file)' + (f'; made with {self.xml_hint}' if self.xml_hint else '')
        if not self.xml_path:
            return 'IMA XML: none'
        cal = getattr(self, 'xml_cal_date', None)
        why = ('chosen because ' + self.xml_reason) if self.xml_reason.startswith('its ') else self.xml_reason
        return (f'IMA XML: {self.xml_path}' + (f' (calibration {cal:%d %b %Y})' if cal else '') +
                f', grating index {self.grating_index} = {GRATING_NAMES[self.rect_grating]} ({self.grating_source}); '
                + why)

    def summary_lines(self):
        """Lines for the status area: kind, numbers, XML, warnings."""
        ny, nx = self.full_shape
        L = [f'{self.kind_label}.  {self.n_w} x {self.n_t} frames, {nx}x{ny} px read at bin {self.b} '
             f'({self.cube.nbytes / 1e9:.2f} GB, {self.load_s:.1f} s).',
             f'Values: {self.normalisation}; units {self.units}.',
             self.xml_text() + '.']
        if self.is_raw and not self.true_ok and self.true_reason:
            L.append('True-wavelength mode not available: ' + self.true_reason + '.')
        warn = list(self.warnings) + ([self.xml_warning] if self.xml_warning else [])
        if warn:
            L.append('WARNING: ' + ' | '.join(warn))
        notes = list(self.notes) + ([self.grid_note] if self.grid_note else [])
        if notes:
            L.append('Notes: ' + ' | '.join(notes))
        return L

    def info_text(self):
        L = [f'file: {self.path}', f'kind: {self.kind_label}', f'cube in memory: {self.cube.shape} float32, bin {self.b}, '
             f'{self.cube.nbytes / 1e9:.2f} GB, read in {self.load_s:.1f} s',
             f'wavelength axis: {self.lam.min():g}-{self.lam.max():g} nm, {self.n_w} values, step {self.lam_step:g} nm '
             f"({'filter settings' if self.is_raw else 'true wavelength'})",
             f'delay axis: {self.delay_label}: {self.delay.min():g}-{self.delay.max():g}, {self.n_t} values'
             + (f'; gate widths {np.unique(self.width).tolist()} ns' if self.width is not None else '; gate widths unknown'),
             f'normalisation: {self.normalisation}', f'units: {self.units}',
             f"date: {self.date:%Y-%m-%d %H:%M} ({self.date_source})" if self.date else 'date: unknown',
             self.xml_text()]
        if self.is_raw and self.true_ok:
            L.append(f'per-pixel true wavelength - filter setting (middle filter): {self.offset_range[0]:+.1f} to '
                     f'{self.offset_range[1]:+.1f} nm; range seen by every pixel {self.common[0]:.1f}-{self.common[1]:.1f} nm, '
                     f'by any pixel {self.extended[0]:.1f}-{self.extended[1]:.1f} nm')
        if self.per_filter is not None:
            L.append(f"per-filter file used: {self.per_filter['path']}")
        if self.objective:
            L.append(f'objective: {self.objective}' + (f' ({self.pix_um} um/px)' if self.pix_um else ''))
        for k, v in self.info.items():
            L.append(f'{k}: {v}')
        if self.warnings or self.xml_warning:
            L.append('')
            L += ['WARNING: ' + w for w in self.warnings + ([self.xml_warning] if self.xml_warning else [])]
        notes = list(self.notes) + ([self.grid_note] if self.grid_note else [])
        if notes:
            L.append('')
            L += ['note: ' + n for n in notes]
        return '\n'.join(L)


class _quiet:
    """Silence 'Mean of empty slice' warnings inside a with-block."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter('ignore', RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def _trapz(y, x, axis=0):
    f = getattr(np, 'trapezoid', None) or np.trapz
    return f(y, x, axis=axis)


# =============================================================================================== window
class Explorer4D(tk.Toplevel):
    """4D window: image at (wavelength, delay), spectra at the delay, decays in a wavelength band, regions."""

    def __init__(self, master=None, path=None, own_root=False, background=True):
        if master is None:
            master = tk.Tk()
            master.withdraw()
            own_root = True
        super().__init__(master)
        self._own_root = own_root
        self.title('TRPL explorer 4D')
        self.geometry('1600x990')
        self.ds = None
        self.regions = []
        self._n_made = 0
        self._last_hover = 0.0
        self._hover_s = None
        self._hover_d = None
        self._selected = None
        self._loading = False
        self._cancel = False                 # set to stop a running read (window closed, another file opened)
        self._pending_open = None            # file to open once a stopped read has ended
        self._drawn_key = None               # view state of the last map/curve redraw (skips repeated slider callbacks)
        self._bg = {}                        # plot backgrounds for redrawing the hover curves by blitting
        self._xml_override = None
        self._xml_next = None
        self._grating_override = None
        self.im = None
        self.cbar = None
        self.last_dir = default_data_dir()
        self.v_axis = tk.StringVar(value='filter')
        self.v_grid = tk.StringVar(value='')
        self.v_map = tk.StringVar(value='frame')
        self.v_bin = tk.StringVar(value='')
        self.v_logc = tk.BooleanVar(value=False)
        self.v_band = tk.StringVar(value='2')
        self.v_fullres = tk.BooleanVar(value=True)
        self.v_tool = tk.StringVar(value='rect')
        self.v_psize = tk.IntVar(value=7)
        self.v_hover = tk.BooleanVar(value=True)
        self.v_norm = tk.BooleanVar(value=True)
        self.v_logy = tk.BooleanVar(value=True)
        self.v_full = tk.BooleanVar(value=True)
        self.v_partial = tk.BooleanVar(value=False)
        self.v_grating = tk.IntVar(value=0)
        self._build()
        self.protocol('WM_DELETE_WINDOW', self.close)
        if path:
            self.after(50, lambda: self.open_file(path, background=background))

    # ------------------------------------------------------------------ layout
    def _build(self):
        menubar = tk.Menu(self)
        fm = tk.Menu(menubar, tearoff=0)
        fm.add_command(label='Open 4D file...', command=self.open_dialog)
        fm.add_command(label='Choose IMA XML...', command=self.choose_xml_dialog)
        gm = tk.Menu(fm, tearoff=0)
        gm.add_radiobutton(label='grating index 0: [400 ; 600] nm', value=0, variable=self.v_grating, command=self._grating_changed)
        gm.add_radiobutton(label='grating index 1: [600 ; 1000] nm', value=1, variable=self.v_grating, command=self._grating_changed)
        fm.add_cascade(label='IMA grating', menu=gm)
        fm.add_command(label='File info...', command=self.show_info)
        fm.add_separator()
        fm.add_command(label='Export CSV...', command=self.export_csv)
        fm.add_command(label='Save figure...', command=self.save_figure)
        fm.add_command(label='Save regions...', command=self.save_regions)
        fm.add_command(label='Load regions...', command=self.load_regions)
        fm.add_separator()
        fm.add_command(label='Close', command=self.close)
        menubar.add_cascade(label='File', menu=fm)
        hm = tk.Menu(menubar, tearoff=0)
        hm.add_command(label='How to use', command=self.show_help)
        menubar.add_cascade(label='Help', menu=hm)
        self.config(menu=menubar)

        bar = ttk.Frame(self, padding=(6, 4))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text='Open 4D...', command=self.open_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text='File info...', command=self.show_info).pack(side=tk.LEFT, padx=2)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        for txt, cmd in (('Export CSV...', self.export_csv), ('Save figure...', self.save_figure),
                         ('Save regions...', self.save_regions), ('Load regions...', self.load_regions)):
            ttk.Button(bar, text=txt, command=cmd).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text='?', width=3, command=self.show_help).pack(side=tk.LEFT, padx=6)
        self.l_file = ttk.Label(bar, text='no file', foreground='#555')
        self.l_file.pack(side=tk.LEFT, padx=10)

        opt = ttk.Frame(self, padding=(6, 0))
        opt.pack(side=tk.TOP, fill=tk.X)
        f = ttk.LabelFrame(opt, text='Wavelength axis', padding=(4, 0))
        f.pack(side=tk.LEFT, padx=2)
        self.rb_axis = {}
        for key, lab in AXIS_MODES:
            self.rb_axis[key] = ttk.Radiobutton(f, text=lab, value=key, variable=self.v_axis, command=self._axis_changed)
            self.rb_axis[key].pack(side=tk.LEFT)
        ttk.Label(f, text='  grid (nm)').pack(side=tk.LEFT)
        self.cb_grid = ttk.Combobox(f, textvariable=self.v_grid, values=('0.5', '1', '2'), width=4, state='readonly')
        self.cb_grid.pack(side=tk.LEFT)
        self.cb_grid.bind('<<ComboboxSelected>>', lambda e: self._grid_changed())
        f = ttk.LabelFrame(opt, text='Map', padding=(4, 0))
        f.pack(side=tk.LEFT, padx=6)
        self.rb_map = {}
        for key, lab in MAP_MODES:
            self.rb_map[key] = ttk.Radiobutton(f, text=lab, value=key, variable=self.v_map, command=self.update_map)
            self.rb_map[key].pack(side=tk.LEFT)
        ttk.Label(f, text='  bin').pack(side=tk.LEFT)
        cb = ttk.Combobox(f, textvariable=self.v_bin, values=('1', '2', '4', '8'), width=3, state='readonly')
        cb.pack(side=tk.LEFT)
        cb.bind('<<ComboboxSelected>>', lambda e: self._bin_changed())
        ttk.Checkbutton(f, text='log colours', variable=self.v_logc, command=self.update_map).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(f, text='full-res frame', variable=self.v_fullres, command=self.update_map).pack(side=tk.LEFT)
        f = ttk.LabelFrame(opt, text='Band', padding=(4, 0))
        f.pack(side=tk.LEFT, padx=6)
        ttk.Label(f, text='width (nm)').pack(side=tk.LEFT)
        sp = ttk.Spinbox(f, from_=0.5, to=200, increment=1, width=5, textvariable=self.v_band, command=self._band_changed)
        sp.pack(side=tk.LEFT)
        sp.bind('<Return>', lambda e: self._band_changed())
        sp.bind('<FocusOut>', lambda e: self._band_changed())

        opt2 = ttk.Frame(self, padding=(6, 2))
        opt2.pack(side=tk.TOP, fill=tk.X)
        f = ttk.LabelFrame(opt2, text='Region tool', padding=(4, 0))
        f.pack(side=tk.LEFT, padx=2)
        for key, lab in TOOLS:
            ttk.Radiobutton(f, text=lab, value=key, variable=self.v_tool, command=self._set_tool).pack(side=tk.LEFT)
        ttk.Label(f, text='  point size').pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=1, to=51, increment=2, width=4, textvariable=self.v_psize).pack(side=tk.LEFT)
        f = ttk.LabelFrame(opt2, text='Curves', padding=(4, 0))
        f.pack(side=tk.LEFT, padx=6)
        for lab, var in (('normalise', self.v_norm), ('log decay', self.v_logy), ('full frame', self.v_full),
                         ('hover probe', self.v_hover), ('partial coverage', self.v_partial)):
            ttk.Checkbutton(f, text=lab, variable=var, command=self.redraw_curves).pack(side=tk.LEFT)

        mid = ttk.Frame(self)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.fig = Figure(figsize=(15.5, 6.4), dpi=100)
        gs = self.fig.add_gridspec(2, 4, width_ratios=[1.0, 0.035, 0.27, 1.12], height_ratios=[1, 1], left=0.04,
                                   right=0.985, bottom=0.08, top=0.92, wspace=0.05, hspace=0.42)
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_cb = self.fig.add_subplot(gs[:, 1])
        self.ax_spec = self.fig.add_subplot(gs[0, 3])
        self.ax_dec = self.fig.add_subplot(gs[1, 3])
        self.canvas = FigureCanvasTkAgg(self.fig, master=mid)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.graph_tools = GraphTools(mid, self.canvas, {'Decay': self.ax_dec, 'Spectrum': self.ax_spec})
        self.graph_tools.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = self.graph_tools.toolbar
        sl = ttk.Frame(mid)
        sl.pack(side=tk.TOP, fill=tk.X, padx=10)
        r1 = ttk.Frame(sl)
        r1.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(r1, text='wavelength', width=11).pack(side=tk.LEFT)
        self.s_wl = tk.Scale(r1, from_=0, to=1, orient=tk.HORIZONTAL, showvalue=0, resolution=1, command=self._on_wl)
        self.s_wl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.l_wl = ttk.Label(r1, text='', width=70)
        self.l_wl.pack(side=tk.LEFT)
        r2 = ttk.Frame(sl)
        r2.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(r2, text='gate delay', width=11).pack(side=tk.LEFT)
        self.s_dl = tk.Scale(r2, from_=0, to=1, orient=tk.HORIZONTAL, showvalue=0, resolution=1, command=self._on_dl)
        self.s_dl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.l_dl = ttk.Label(r2, text='', width=70)
        self.l_dl.pack(side=tk.LEFT)

        bot = ttk.Frame(self, padding=(6, 2))
        bot.pack(side=tk.TOP, fill=tk.X)
        cols = ('name', 'tool', 'pixels', 'area', 'spec_peak', 'range', 'centroid', 'dec_peak')
        heads = ('region', 'tool', 'pixels', 'area (um2)', 'spectrum peak (nm)', 'measured range (nm)',
                 'spectral centroid (nm)', 'decay maximum at (ns)')
        self.table = ttk.Treeview(bot, columns=cols, show='headings', height=4)
        for c, hdr in zip(cols, heads):
            self.table.heading(c, text=hdr)
            self.table.column(c, width=150 if c in ('spec_peak', 'range', 'centroid', 'dec_peak') else 110, anchor=tk.CENTER)
        self.table.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.table.bind('<<TreeviewSelect>>', self._on_table_select)
        self.table.bind('<Double-1>', self._rename)
        bb = ttk.Frame(bot)
        bb.pack(side=tk.LEFT, padx=6)
        ttk.Button(bb, text='Delete', command=self.delete_selected).pack(fill=tk.X, pady=1)
        ttk.Button(bb, text='Remove last', command=self.undo_last).pack(fill=tk.X, pady=1)
        ttk.Button(bb, text='Clear all', command=self.clear_regions).pack(fill=tk.X, pady=1)
        self.l_status = ttk.Label(self, text='Open a 4D hyperspectral TRPL file (Open 4D...).', anchor=tk.W, padding=(6, 1))
        self.l_status.pack(side=tk.BOTTOM, fill=tk.X)
        self.l_info = tk.Label(self, text='', anchor=tk.W, justify=tk.LEFT, padx=6, pady=1, fg='#333', wraplength=1560)
        self.l_info.pack(side=tk.BOTTOM, fill=tk.X)
        # Delete / Ctrl+Z act on regions, but not while typing in a number field (band width, point size, grid): the
        # fields' key events also reach this window's bindings (their bindtags include the toplevel)
        self.bind('<Delete>', self._key_delete)
        self.bind('<Control-z>', self._key_undo)

        sel_props = dict(edgecolor='white', facecolor='none', linestyle='--', linewidth=1.2)
        self.sel_rect = RectangleSelector(self.ax_map, self._on_box_rect, useblit=True, button=[1], minspanx=2, minspany=2,
                                          spancoords='data', interactive=False, props=sel_props)
        self.sel_ell = EllipseSelector(self.ax_map, self._on_box_ell, useblit=True, button=[1], minspanx=2, minspany=2,
                                       spancoords='data', interactive=False, props=sel_props)
        self.sel_lasso = LassoSelector(self.ax_map, self._on_lasso, useblit=True, button=[1],
                                       props=dict(color='white', linestyle='--', linewidth=1.2))
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('draw_event', self._on_draw)
        self._set_tool()
        self._empty_axes()

    @staticmethod
    def _in_entry(e):
        try:
            return e.widget.winfo_class() in ENTRY_CLASSES
        except Exception:
            return False

    def _key_delete(self, e):
        if not self._in_entry(e):
            self.delete_selected()

    def _key_undo(self, e):
        if not self._in_entry(e):
            self.undo_last()

    def _on_draw(self, _ev):
        """After every full redraw: keep the backgrounds of the two plots (full redraws skip the hover curves, which
        are kept invisible) and draw the hover curves on top; the hover probe then only blits them."""
        self._bg = {}
        try:
            for ax, ln in ((self.ax_spec, self._hover_s), (self.ax_dec, self._hover_d)):
                if ax.get_visible() and ln is not None and ln.axes is ax:
                    self._bg[ax] = self.canvas.copy_from_bbox(ax.bbox)
                    self._draw_hidden(ax, ln)
        except Exception:
            self._bg = {}

    @staticmethod
    def _draw_hidden(ax, ln):
        """Draw a hover curve.  It stays invisible for full redraws: not 'animated', because matplotlib's region
        selectors (useblit) redraw the whole figure again for every visible animated artist (~4x slower redraws)."""
        if not len(ln.get_xdata()):
            return
        ln.set_visible(True)
        try:
            ax.draw_artist(ln)
        finally:
            ln.set_visible(False)

    def _bg_valid(self):
        """The kept backgrounds still fit the plots (not after a resize, or a figure saved at another dpi)."""
        if set(self._bg) != {ax for ax in (self.ax_spec, self.ax_dec) if ax.get_visible()}:
            return False
        try:
            for ax, bg in self._bg.items():
                x0, y0, x1, y1 = bg.get_extents()
                if abs((x1 - x0) - ax.bbox.width) > 2 or abs((y1 - y0) - ax.bbox.height) > 2:
                    return False
            return True
        except Exception:
            return False

    def _blit_hover(self):
        """Redraw only the hover curves (a full redraw of the figure takes ~150 ms)."""
        if not self._bg_valid():
            self._bg = {}
            self.canvas.draw_idle()
            return
        try:
            for ax, ln in ((self.ax_spec, self._hover_s), (self.ax_dec, self._hover_d)):
                if ax not in self._bg:
                    continue
                self.canvas.restore_region(self._bg[ax])
                self._draw_hidden(ax, ln)
                self.canvas.blit(ax.bbox)
        except Exception:
            self._bg = {}
            self.canvas.draw_idle()

    def _empty_axes(self):
        self.graph_tools.clear()
        for ax in (self.ax_map, self.ax_spec, self.ax_dec, self.ax_cb):
            ax.clear()
        self.ax_map.set_title('no file')
        self.ax_cb.set_axis_off()
        self.im = None
        self.canvas.draw_idle()

    def status(self, msg):
        try:
            self.l_status.config(text=msg)
            self.update_idletasks()
        except tk.TclError:
            pass

    def close(self):
        self._cancel = True                  # a running read stops at its next block (and its memory is released)
        self._pending_open = None
        self.ds = None
        self._result = None
        try:
            self.update_idletasks()          # run pending redraws now, not after the window is gone
        except tk.TclError:
            pass
        try:
            self.destroy()
        finally:
            if self._own_root:
                try:
                    self.master.destroy()
                except tk.TclError:
                    pass

    def show_help(self):
        messagebox.showinfo('TRPL explorer 4D', HELP4D, parent=self)

    def show_info(self):
        if self.ds is None:
            return
        w = tk.Toplevel(self)
        w.title('File info - ' + self.ds.name)
        t = tk.Text(w, width=130, height=34, wrap='word')
        t.insert('1.0', self.ds.info_text())
        t.config(state='disabled')
        t.pack(fill=tk.BOTH, expand=True)
        ttk.Button(w, text='Close', command=w.destroy).pack(pady=3)

    # ------------------------------------------------------------------ files
    def open_dialog(self):
        p = filedialog.askopenfilename(parent=self, title='Open 4D hyperspectral TRPL file', initialdir=self.last_dir,
                                       filetypes=[('HDF5', '*.h5'), ('all files', '*.*')])
        if p:
            self.open_file(p)

    def open_file(self, path, background=True, bin=None, keep_view=False):
        """Read a 4D file (in a background thread unless background=False); the window stays usable meanwhile.
        A file chosen while another one is still being read stops that read and is opened when it has ended."""
        if self._loading:
            self._cancel = True
            self._pending_open = (path, background, bin, keep_view)
            self.status(f'stopping the read of the previous file; {os.path.basename(path)} opens next ...')
            return
        kind, why = detect_4d(path)
        if kind is None or kind == 'other4d':
            messagebox.showerror('TRPL explorer 4D', f'{os.path.basename(path)}:\n{why}', parent=self)
            self.status('not opened: ' + why)
            return
        if not keep_view:                                   # a new file: XML chosen before it (if any), file's grating
            self._xml_override, self._xml_next = self._xml_next, None
            self._grating_override = None
        prev = self._view_state() if keep_view else None
        self.ds = None
        self._empty_axes()
        self._loading = True
        self._cancel = False
        params = dict(bin=bin, xml=self._xml_override, grating=self._grating_override,
                      grid_step=(float(self.v_grid.get()) if keep_view and self.v_grid.get() else None))
        name = os.path.basename(path)
        self.status(f'reading {name} ...')
        if not background:
            try:
                ds = Dataset4D(path, progress=lambda i, n: self.status(f'reading {name}: frame {i}/{n} ...'), **params)
            except Exception as e:
                self._loading = False
                messagebox.showerror('TRPL explorer 4D', f'cannot open {path}:\n{e}', parent=self)
                self.status('open failed')
                return
            self._loading = False
            self._loaded(ds, prev)
            return
        self._prog = (0, 1)
        self._result = None

        def prog(i, n):
            if self._cancel:
                raise _Cancelled()
            self._prog = (i, n)

        def work():
            try:
                self._result = Dataset4D(path, progress=prog, **params)
            except _Cancelled:
                self._result = None
            except Exception as e:                       # reported in the Tk thread
                self._result = e

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        self._t_load = time.time()
        self.after(150, lambda: self._poll(path, prev))

    def _poll(self, path, prev):
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        if self._thread.is_alive():
            i, n = self._prog
            self.status(f'reading {os.path.basename(path)}: frame {i}/{n} ({time.time() - self._t_load:.0f} s) ...')
            self.after(200, lambda: self._poll(path, prev))
            return
        self._loading = False
        res = self._result
        self._result = None
        if self._cancel:                                 # stopped (or finished just after the stop): drop it
            self._cancel = False
            res = None
            nxt, self._pending_open = self._pending_open, None
            self.status(f'stopped reading {os.path.basename(path)}')
            if nxt:
                self.open_file(*nxt)
            return
        if isinstance(res, Exception) or res is None:
            messagebox.showerror('TRPL explorer 4D', f'cannot open {path}:\n{res}', parent=self)
            self.status('open failed')
            return
        self._loaded(res, prev)

    def _view_state(self):
        if self.ds is None:
            return None
        return dict(axis=self.v_axis.get(), map=self.v_map.get(), lam=self._cur_lam(), it=int(self.s_dl.get()),
                    band=self.v_band.get())

    def _loaded(self, ds, prev=None):
        self.ds = ds
        self.last_dir = ds.folder
        self.v_bin.set(str(ds.b))
        self.v_grating.set(ds.grating_index if ds.grating_index is not None else 0)
        if ds.is_raw and ds.true_ok:
            self.v_grid.set(f'{ds.grid_step:g}')
            vals = sorted({0.5, 1.0, 2.0, ds.lam_step})
            self.cb_grid.config(values=[f'{v:g}' for v in vals], state='readonly')
        else:
            self.v_grid.set('')
            self.cb_grid.config(state='disabled')
        self.v_band.set(prev['band'] if prev and prev.get('band') else f'{ds.lam_step:g}')
        # axis modes
        if not ds.is_raw:
            self.v_axis.set('true')
            self.rb_axis['filter'].config(state='disabled')
            self.rb_axis['true'].config(state='disabled', text='true wavelength (rectified file)')
        else:
            self.rb_axis['filter'].config(state='normal')
            self.rb_axis['true'].config(state='normal' if ds.true_ok else 'disabled',
                                        text='true wavelength (rectified per pixel)')
            self.v_axis.set('true' if ds.true_ok else 'filter')
        for key in ('wlint', 'centroid'):
            self.rb_map[key].config(state='normal' if ds.n_w >= 2 else 'disabled')
        self.v_map.set('frame')
        if prev:
            if prev['axis'] in ('filter', 'true') and ds.is_raw and (prev['axis'] == 'filter' or ds.true_ok):
                self.v_axis.set(prev['axis'])
            self.v_map.set(prev['map'])
        # sliders
        self.s_dl.config(to=max(ds.n_t - 1, 0))
        self.s_dl.set(prev['it'] if prev and prev['it'] < ds.n_t else ds.it_peak)
        self._config_wl_slider(prev['lam'] if prev else self._default_lam(ds))
        # image
        self.ax_map.clear()
        self.ax_cb.clear()
        ny, nx = ds.full_shape
        self.im = self.ax_map.imshow(np.zeros(ds.shape, np.float32), origin='lower', cmap='inferno',
                                     interpolation='nearest', extent=(-0.5, nx - 0.5, -0.5, ny - 0.5))
        self.cbar = self.fig.colorbar(self.im, cax=self.ax_cb)
        self.ax_map.set_xlabel('x (pixel)')
        self.ax_map.set_ylabel('y (pixel)')
        self._scalebar()
        for r in self.regions:
            r.patch = r.label = None
        kept = [r for r in self.regions if r.mask.shape == ds.full_shape]
        self.regions = []
        for r in kept:
            self._attach(r)
        self.l_file.config(text=f'{ds.name}   [{ds.kind}]   {ds.n_w} x {ds.n_t} frames, bin {ds.b}'
                                + (f', {ds.objective} ({ds.pix_um} um/px)' if ds.pix_um else ''))
        self.l_info.config(text='\n'.join(ds.summary_lines()))
        self.update_map()
        self.status(f'read {ds.name} in {ds.load_s:.1f} s ({ds.cube.nbytes / 1e9:.2f} GB at bin {ds.b})')

    def _default_lam(self, ds):
        """Wavelength the window opens on: the brightest point of the full-frame spectrum at the brightest delay, on
        the axis it opens with.  (In true-wavelength mode the full-frame curves exist only where nearly every pixel
        has data; the brightest filter setting can lie outside that range, e.g. 594 nm on G1, which left the decay
        plot empty.)"""
        try:
            x, s = ds.spectrum(self._mode(), ds.it_peak, None, False)
            if np.isfinite(s).any():
                return float(x[int(np.nanargmax(s))])
        except Exception:
            pass
        return float(ds.lam[ds.iw_peak])

    def _bin_changed(self):
        if self.ds is None or not self.v_bin.get():
            return
        b = int(self.v_bin.get())
        if b == self.ds.b:
            return
        ny, nx = self.ds.full_shape
        need = self.ds.n_w * self.ds.n_t * (ny // b) * (nx // b) * 4
        if need > 3.5e9 and not messagebox.askyesno('TRPL explorer 4D', f'bin {b} needs {need / 1e9:.1f} GB of memory '
                                                    'for the cube. Read the file at this binning?', parent=self):
            self.v_bin.set(str(self.ds.b))
            return
        self.open_file(self.ds.path, bin=b, keep_view=True)

    def choose_xml_dialog(self):
        p = filedialog.askopenfilename(parent=self, title='Choose the IMA calibration XML', initialdir=HERE,
                                       filetypes=[('IMA XML', '*.xml'), ('all files', '*.*')])
        if not p:
            return
        if self.ds is None or not self.ds.is_raw:
            self._xml_next = p
            self.status(f'IMA XML for the next raw file: {p}')
            return
        self._xml_override = p
        try:
            self.ds.set_xml(p)
        except Exception as e:
            messagebox.showerror('TRPL explorer 4D', f'cannot use {p}:\n{e}', parent=self)
            return
        self._after_map_change()

    def _grating_changed(self):
        if self.ds is None or not self.ds.is_raw:
            return
        g = int(self.v_grating.get())
        if g == self.ds.grating_index:
            return
        if self.ds.grating_source == 'config_json' and not messagebox.askyesno(
                'TRPL explorer 4D', f'The file says grating index {self.ds.grating_index}. Use index {g} anyway?', parent=self):
            self.v_grating.set(self.ds.grating_index)
            return
        self._grating_override = g
        self.ds.set_grating(g)
        self._after_map_change()

    def _after_map_change(self):
        ds = self.ds
        lam = self._cur_lam()
        self.rb_axis['true'].config(state='normal' if ds.true_ok else 'disabled')
        if not ds.true_ok:
            self.v_axis.set('filter')
        else:
            self.v_grid.set(f'{ds.grid_step:g}')
        self._config_wl_slider(lam)
        self.l_info.config(text='\n'.join(ds.summary_lines()))
        self.update_map()

    def _grid_changed(self):
        ds = self.ds
        if ds is None or not ds.true_ok or not self.v_grid.get():
            return
        lam = self._cur_lam()
        try:
            ds.set_grid(float(self.v_grid.get()))
        except MemoryError as e:
            messagebox.showerror('TRPL explorer 4D', str(e), parent=self)
            self.v_grid.set(f'{ds.grid_step:g}')
            return
        self._config_wl_slider(lam)
        self.update_map()

    # ------------------------------------------------------------------ wavelength / delay
    def _mode(self):
        return self.v_axis.get() if self.ds is not None and self.ds.is_raw else 'true'

    def _axis(self):
        return self.ds.axis(self._mode())

    def _cur_lam(self):
        if self.ds is None:
            return np.nan
        x = self._axis()[0]
        return float(x[min(int(self.s_wl.get()), x.size - 1)])

    def _band(self):
        try:
            w = float(self.v_band.get())
        except ValueError:
            w = self.ds.lam_step if self.ds else 2.0
        return max(w, 1e-3)

    def _config_wl_slider(self, lam):
        x = self._axis()[0]
        self.s_wl.config(to=max(x.size - 1, 0), state='normal' if x.size > 1 else 'disabled')
        self.s_wl.set(int(np.argmin(np.abs(x - lam))) if np.isfinite(lam) else 0)
        self._update_labels()

    def _axis_changed(self):
        if self.ds is None:
            return
        lam = getattr(self, '_last_lam', None)
        lam = lam if lam is not None else float(self.ds.lam[self.ds.iw_peak])
        self._config_wl_slider(lam)
        self.update_map()

    def _band_changed(self):
        if self.ds is None:
            return
        if self.v_map.get() == 'tint':
            self.update_map()
        else:
            self.redraw_curves()

    def _update_labels(self):
        ds = self.ds
        if ds is None:
            return
        x, lab = self._axis()
        i = min(int(self.s_wl.get()), x.size - 1)
        self._last_lam = float(x[i])
        txt = f'{lab.split(" (")[0]} {x[i]:.1f} nm   ({i + 1}/{x.size})'
        if self._mode() == 'filter' and ds.is_raw and ds.imap is not None:
            lo = ds.pixel_wavelength(i, 0, ds.full_shape[0] // 2)
            hi = ds.pixel_wavelength(i, ds.full_shape[1] - 1, ds.full_shape[0] // 2)
            txt += f'   true wavelength across the image {min(lo, hi):.1f}-{max(lo, hi):.1f} nm'
        txt += f'   band {x[i] - self._band() / 2:.1f}-{x[i] + self._band() / 2:.1f} nm'
        self.l_wl.config(text=txt)
        it = int(self.s_dl.get())
        w = f', width {ds.width[it]:g} ns' if ds.width is not None else ''
        self.l_dl.config(text=f'{ds.delay_label.split(" (")[0]} {ds.delay[it]:g}{" ns" if "ns" in ds.delay_label else ""}'
                              f'{w}   ({it + 1}/{ds.n_t})')

    def _view_key(self):
        """What the image and the curves depend on (a slider callback with an unchanged key does nothing: Tk calls a
        scale's command also after scale.set(), once the window is redrawn, i.e. after the explicit update)."""
        return (id(self.ds), self._mode(), self.v_map.get(), int(self.s_wl.get()), int(self.s_dl.get()), self._band(),
                getattr(self.ds, 'grid_step', None), bool(self.v_partial.get()))

    def _on_wl(self, _v):
        if self.ds is None:
            return
        self._update_labels()
        key = self._view_key()
        if key == self._drawn_key:
            return
        if self.v_map.get() in ('frame', 'tint'):
            self.update_map()
        else:
            self.redraw_curves()
            self._drawn_key = key

    def _on_dl(self, _v):
        if self.ds is None:
            return
        self._update_labels()
        key = self._view_key()
        if key == self._drawn_key:
            return
        if self.v_map.get() != 'tint':
            self.update_map()
        else:
            self.redraw_curves()
            self._drawn_key = key

    # ------------------------------------------------------------------ map
    def _scalebar(self):
        ds = self.ds
        if not ds or not ds.pix_um:
            return
        try:
            from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        except Exception:
            return
        width_um = ds.full_shape[1] * ds.pix_um
        L = next((x for x in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if x >= 0.15 * width_um), 1000)
        self.ax_map.add_artist(AnchoredSizeBar(self.ax_map.transData, L / ds.pix_um, f'{L} um', 'lower right',
                                               color='white', frameon=False, size_vertical=max(1, ds.full_shape[0] // 150)))

    def current_map(self):
        """(image, colour label, title, is_centroid) for the current settings."""
        ds, mode, mm = self.ds, self._mode(), self.v_map.get()
        x, lab = self._axis()
        ia = min(int(self.s_wl.get()), x.size - 1)
        it = int(self.s_dl.get())
        lam = float(x[ia])
        band = self._band()
        J = ds.common_idx(mode)
        rng = f'{x[J[0]]:.0f}-{x[J[-1]]:.0f} nm' if J.size else '-'
        dtxt = f'{ds.delay_label.split(" (")[0]} {ds.delay[it]:g}' + (' ns' if 'ns' in ds.delay_label else '')
        axname = 'filter setting' if mode == 'filter' and ds.is_raw else 'wavelength'
        if mm == 'frame':
            if (self.v_fullres.get() and ds.full_frame_cheap and not ds.use_interp(mode)):
                img = ds.full_frame(ia, it)
                res = 'full resolution'
            else:
                img = ds.map_frame(mode, ia, it)
                res = f'bin {ds.b}'
            title = f'{axname} {lam:.1f} nm, {dtxt} ({res})'
            if ds.use_interp(mode):
                title += '\nrectified per pixel (grey = not measured at this wavelength)'
            elif ds.is_raw:
                title += '\nraw frame: the true wavelength changes across the image'
            return img, ds.unit_short, title, False
        if mm == 'wlint':
            img = ds.map_wl_integrated(mode, it)
            title = f'wavelength-integrated at {dtxt}, over {rng}'
            title += ('\n(range measured by every pixel)' if ds.use_interp(mode) else
                      '\n(filter axis: each pixel integrates its own shifted range)' if ds.is_raw else '')
            return img, ds.unit_short + ' x nm', title, False
        if mm == 'centroid':
            img = ds.map_centroid(mode, it)
            title = f'spectral centroid at {dtxt}, over {rng}'
            title += ('\n(range measured by every pixel)' if ds.use_interp(mode) else
                      '\n(FILTER axis: includes the IMA wavelength shift across the image)' if ds.is_raw else '')
            return img, 'centroid (nm)', title, True
        img = ds.map_time_integrated(mode, lam, band)
        tl = ds.delay_label.split(' (')[0]
        title = (f'time-integrated over {tl} {ds.delay.min():g}-{ds.delay.max():g}'
                 + (' ns' if 'ns' in ds.delay_label else '') + f', {axname} band '
                 f'{lam - band / 2:.1f}-{lam + band / 2:.1f} nm')
        return img, ds.unit_short + ' x ns', title, False

    def update_map(self):
        ds = self.ds
        if ds is None or self.im is None:
            return
        self._update_labels()
        try:
            m, clab, title, centroid = self.current_map()
        except Exception as e:
            self.status(f'cannot compute this map: {e}')
            return
        self.im.set_data(m)
        fin = m[np.isfinite(m)]
        self.im.set_cmap('RdYlBu_r' if centroid else 'inferno')
        if fin.size:
            lo, hi = np.percentile(fin, [2, 98] if centroid else [1, 99])
            if hi <= lo:
                hi = lo + 1e-12
            if self.v_logc.get() and not centroid:
                pos = fin[fin > 0]
                lo = max(np.percentile(pos, 1) if pos.size else hi * 1e-3, hi * 1e-4)
                if hi > lo > 0:
                    self.im.set_norm(LogNorm(vmin=lo, vmax=hi))
                else:
                    self.im.set_norm(Normalize(vmin=lo, vmax=hi))
            else:
                self.im.set_norm(Normalize(vmin=lo, vmax=hi))
        self.cbar.update_normal(self.im)
        self.cbar.set_label(clab)
        fmt = self.cbar.formatter
        if not centroid and hasattr(fmt, 'set_powerlimits'):            # linear colours only (log has its own)
            fmt.set_powerlimits((-2, 3))
            self.cbar.update_ticks()
        self.ax_map.set_title(title, fontsize=9.5)
        self.ax_map.set_facecolor('#808080')
        self.redraw_curves()
        self._drawn_key = self._view_key()

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
        m = mask_from_verts(verts, *self.ds.full_shape)
        if not m.any():
            self.status('the region contains no pixel centre - draw a larger one')
            return None
        self._n_made += 1
        r = Region(self._unique_name(name or f'R{self._n_made}'), kind, verts,
                   color or COLORS[(self._n_made - 1) % len(COLORS)], m)
        self._attach(r)
        self.redraw_curves()
        ps = self.ds.pix_um
        self.status(f'{r.name}: {r.npix} pixels' + (f' ({r.npix * ps * ps:.0f} um2)' if ps else '')
                    + (f'; evaluated on the bin-{self.ds.b} data' if self.ds.b > 1 else ''))
        return r

    def _unique_name(self, name):
        """Region names are the keys of the table rows: a name already in use gets ' appended (R1, R1', R1'', ...)."""
        names = {r.name for r in self.regions}
        name = str(name)
        while name in names:
            name += "'"
        return name

    def _attach(self, r):
        r.wb = weights_from_mask(r.mask, self.ds.b)
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
        self.redraw_curves(recompute=False)

    def undo_last(self):
        if self.regions:
            self._detach(self.regions[-1])
            self.redraw_curves(recompute=False)

    def clear_regions(self):
        for r in list(self.regions):
            self._detach(r)
        self.redraw_curves(recompute=False)

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
            self.redraw_curves(recompute=False)

    def _on_table_select(self, _e):
        sel = self.table.selection()
        self._selected = sel[0] if sel else None
        for r in self.regions:
            if r.patch is not None:
                r.patch.set_linewidth(3.2 if r.name == self._selected else 1.6)
        self.redraw_curves(recompute=False, table=False)

    def _region_curves(self, r):
        ds, mode = self.ds, self._mode()
        it = int(self.s_dl.get())
        part = self.v_partial.get()
        x, s = ds.spectrum(mode, it, r.wb, part)
        d = ds.decay(mode, self._cur_lam(), self._band(), r.wb, part)
        return x, s, d

    def _compute_regions(self):
        """Spectrum (at the delay) and decay (in the band) of every region for the current settings."""
        if self.ds is None:
            return
        for r in self.regions:
            try:
                _, r.spec, r.dec = self._region_curves(r)
            except Exception as e:
                r.spec, r.dec = np.full(self._axis()[0].size, np.nan), np.full(self.ds.n_t, np.nan)
                self.status(f'{r.name}: {e}')

    def _refresh_table(self):
        self.table.delete(*self.table.get_children())
        ds = self.ds
        if ds is None:
            return
        ps = ds.pix_um
        x = self._axis()[0]
        for r in self.regions:
            s, d = getattr(r, 'spec', None), getattr(r, 'dec', None)
            if s is None or d is None or len(s) != x.size or len(d) != ds.n_t:
                _, s, d = self._region_curves(r)
                r.spec, r.dec = s, d
            fin = np.isfinite(s)
            pk = '-'
            if fin.any():
                ip, fi = int(np.nanargmax(s)), np.flatnonzero(fin)
                pk = f'{x[ip]:.1f}' + (' (edge)' if x.size > 1 and ip in (fi[0], fi[-1]) else '')
            common = np.zeros(x.size, bool)                 # centroid over the range every pixel has measured,
            common[ds.common_idx(self._mode())] = True      # so that regions on both sides of the image compare
            sc = np.clip(np.where(fin & common, s, 0), 0, None)
            cen = f'{(sc * x).sum() / sc.sum():.1f}' if sc.sum() > 0 and x.size > 1 else '-'
            if self._mode() == 'filter' and ds.is_raw and cen != '-':
                cen += ' (filter axis)'
            dp = f'{ds.delay[np.nanargmax(d)]:g}' if np.isfinite(d).any() else '-'
            rg = f'{x[fin][0]:.0f}-{x[fin][-1]:.0f}' if fin.sum() > 1 else '-'
            vals = (r.name, r.kind, r.npix, f'{r.npix * ps * ps:.0f}' if ps else '-', pk, rg, cen, dp)
            self.table.insert('', tk.END, iid=r.name, values=vals, tags=(r.name,))
            self.table.tag_configure(r.name, foreground=r.color)

    # ------------------------------------------------------------------ curves
    def _norm(self, y):
        if not self.v_norm.get():
            return y
        p = np.nanmax(y) if np.isfinite(y).any() else np.nan
        return y / p if p > 0 else y * np.nan

    def redraw_curves(self, recompute=True, table=True):
        """Recompute the region curves (unless recompute=False), redraw both plots and refill the table."""
        ds = self.ds
        previous = self.graph_tools.capture_ranges()
        for ax in (self.ax_spec, self.ax_dec):
            ax.clear()
        self._hover_s = self._hover_d = None
        self._bg = {}                                    # backgrounds are taken again at the next full redraw
        if ds is None:
            self.canvas.draw_idle()
            return
        if recompute:
            self._compute_regions()
        mode = self._mode()
        it = int(self.s_dl.get())
        lam, band = self._cur_lam(), self._band()
        part = self.v_partial.get()
        x, xlab = self._axis()
        a, b = self.ax_spec, self.ax_dec
        full_dec = None
        if self.v_full.get():
            try:
                a.plot(x, self._norm(ds.spectrum(mode, it, None, part)[1]), '-', color='k', lw=1.0, alpha=0.6,
                       label='full frame')
                full_dec = ds.decay(mode, lam, band, None, part)
                b.plot(ds.delay, self._norm(full_dec), '-', color='k', lw=1.0, alpha=0.6, label='full frame')
            except Exception as e:
                self.status(f'full-frame curves: {e}')
        for r in self.regions:
            lw = 2.6 if r.name == self._selected else 1.4
            s, d = getattr(r, 'spec', None), getattr(r, 'dec', None)
            if s is None or d is None or len(s) != x.size or len(d) != ds.n_t:
                _, s, d = self._region_curves(r)
                r.spec, r.dec = s, d
            a.plot(x, self._norm(s), 'o-', ms=2.5, lw=lw, color=r.color, label=f'{r.name} ({r.npix} px)')
            b.plot(ds.delay, self._norm(d), 'o-', ms=2.5, lw=lw, color=r.color, label=f'{r.name}')
        # hover curves: dashed (the full-frame curve is solid grey), invisible to full redraws, drawn by blitting
        self._hover_s, = a.plot([], [], '--', color='0.15', lw=1.0, visible=False)
        self._hover_d, = b.plot([], [], '--', color='0.15', lw=1.0, visible=False)
        if x.size > 1:
            a.axvspan(lam - band / 2, lam + band / 2, color='0.8', alpha=0.45, lw=0)
            pad = 0.01 * float(x.max() - x.min())
            a.set_xlim(float(x.min()) - pad, float(x.max()) + pad)
        if ds.n_t > 1:                                   # keep the whole delay range, also when no curve has data
            pad = 0.02 * float(ds.delay.max() - ds.delay.min())
            b.set_xlim(float(ds.delay.min()) - pad, float(ds.delay.max()) + pad)
        if full_dec is not None and not np.isfinite(full_dec).any():
            why = ('not (nearly) every pixel has data in this band in true-wavelength mode:\n'
                   "tick 'partial coverage' or choose a wavelength inside the full-frame spectrum"
                   if ds.use_interp(mode) else 'no data in this band')
            b.text(0.02, 0.03, 'full frame: ' + why, transform=b.transAxes, ha='left', va='bottom', fontsize=7.5,
                   color='0.25', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='0.8', alpha=0.85))
        a.axvline(lam, color='0.4', lw=0.8, ls=':')
        a.set_xlabel(xlab)
        ylab = 'PL (peak = 1)' if self.v_norm.get() else ds.unit_short + ' per pixel'
        a.set_ylabel(ylab)
        dtxt = f'{ds.delay[it]:g}' + (' ns' if 'ns' in ds.delay_label else '')
        a.set_title(f'spectra at {ds.delay_label.split(" (")[0]} {dtxt}  (click: image at that wavelength)', fontsize=9)
        a.grid(True, alpha=0.25)
        if x.size == 1:
            a.text(0.5, 0.5, f'one filter setting ({x[0]:g} nm): no spectrum', transform=a.transAxes, ha='center',
                   va='center', color='0.3')
        b.axvline(ds.delay[it], color='0.4', lw=0.8, ls=':')
        if self.v_logy.get():
            b.set_yscale('log', nonpositive='mask')
        b.set_xlabel(ds.delay_label)
        b.set_ylabel(ylab)
        bt = 'filter-setting' if mode == 'filter' and ds.is_raw else 'wavelength'
        b.set_title(f'decays, {bt} band {lam - band / 2:.1f}-{lam + band / 2:.1f} nm  (click: image at that delay)',
                    fontsize=9)
        b.grid(True, which='major', alpha=0.25)
        if self.v_norm.get() and self.v_logy.get():
            ys = [np.asarray(ln.get_ydata(), float) for ln in b.get_lines() if len(ln.get_ydata())]
            pos = np.concatenate([y[np.isfinite(y) & (y > 0)] for y in ys]) if ys else np.array([])
            lo = max(1e-4, np.percentile(pos, 1) * 0.5) if pos.size else 1e-3
            b.set_ylim(lo, 2.0)
        for ax in (a, b):
            hs, ls = ax.get_legend_handles_labels()
            if self.v_hover.get():                       # the hover curve itself is invisible to full redraws
                hs.append(Line2D([], [], ls='--', color='0.15', lw=1.0))
                ls.append('hover')
            if hs:
                ax.legend(hs, ls, fontsize=7.5, loc='upper right', ncol=1 if len(self.regions) < 7 else 2)
        self.graph_tools.finish_redraw(previous)
        self.canvas.draw_idle()
        if table:
            self._refresh_table()

    # ------------------------------------------------------------------ mouse
    def _on_motion(self, ev):
        ds = self.ds
        if ds is None or self.im is None or ev.inaxes is not self.ax_map or ev.xdata is None:
            return
        ny, nx = ds.full_shape
        ix, iy = int(round(ev.xdata)), int(round(ev.ydata))
        if not (0 <= ix < nx and 0 <= iy < ny):
            return
        m = self.im.get_array()
        bd = max(1, ny // m.shape[0])
        try:
            val = float(m[min(iy // bd, m.shape[0] - 1), min(ix // bd, m.shape[1] - 1)])
        except Exception:
            val = np.nan
        um = f' = ({ix * ds.pix_um:.1f}, {iy * ds.pix_um:.1f}) um' if ds.pix_um else ''
        txt = f'x {ix}, y {iy}{um}   value {val:.4g}'
        if ds.is_raw and ds.imap is not None and self._mode() == 'filter':
            ia = min(int(self.s_wl.get()), ds.n_w - 1)
            txt += f'   true wavelength here {ds.pixel_wavelength(ia, ix, iy):.2f} nm (filter {ds.lam[ia]:g} nm)'
        self.l_status.config(text=txt)
        if not self.v_hover.get() or self._hover_s is None or ev.button is not None:
            return
        now = time.time()
        if now - self._last_hover < 0.05:
            return
        self._last_hover = now
        r = max(0, (int(self.v_psize.get()) - 1) // 2)
        mk = np.zeros(ds.full_shape, bool)
        mk[max(0, iy - r):min(ny, iy + r + 1), max(0, ix - r):min(nx, ix + r + 1)] = True
        wb = weights_from_mask(mk, ds.b)
        mode, part = self._mode(), self.v_partial.get()
        try:
            x, s = ds.spectrum(mode, int(self.s_dl.get()), wb, part)
            d = ds.decay(mode, self._cur_lam(), self._band(), wb, part)
        except Exception:
            return
        self._hover_s.set_data(x, self._norm(s))
        self._hover_d.set_data(ds.delay, self._norm(d))
        self._blit_hover()

    def _on_click(self, ev):
        if self.ds is None or self.toolbar.mode:
            return
        if ev.inaxes is self.ax_map and self.v_tool.get() == 'point' and ev.button == 1 and ev.xdata is not None:
            self._point(ev.xdata, ev.ydata)
        elif ev.inaxes is self.ax_spec and ev.button == 1 and ev.xdata is not None:
            x = self._axis()[0]
            self.s_wl.set(int(np.argmin(np.abs(x - ev.xdata))))
            if self.v_map.get() not in ('frame', 'tint'):
                self.v_map.set('frame')
            self.update_map()
        elif ev.inaxes is self.ax_dec and ev.button == 1 and ev.xdata is not None:
            self.s_dl.set(int(np.argmin(np.abs(self.ds.delay - ev.xdata))))
            if self.v_map.get() == 'tint':
                self.v_map.set('frame')
            self.update_map()

    # ------------------------------------------------------------------ export
    def _header(self, fh, what):
        ds = self.ds
        fh.write(f'# TRPL explorer 4D {__version__} export {datetime.datetime.now().isoformat(timespec="seconds")}: {what}\n')
        fh.write(f'# file: {ds.path}\n# kind: {ds.kind_label}\n# values: {ds.normalisation}; units {ds.units}\n')
        fh.write(f'# {ds.xml_text()}\n')
        mode = self._mode()
        fh.write(f'# wavelength axis: {ds.axis(mode)[1]}'
                 + (f' (per-pixel linear interpolation, grid {ds.grid_step:g} nm, no extrapolation)' if ds.use_interp(mode) else '')
                 + f'; binning {ds.b}; region values = mean over the region (weighted by pixel coverage of the bins)'
                 + ('; points where < 50 % of the region has data are empty' if self.v_partial.get() else
                    '; points where not every pixel of the region has data are empty') + '\n')
        for w in ds.warnings + ([ds.xml_warning] if ds.xml_warning else []):
            fh.write(f'# WARNING: {w}\n')
        for r in self.regions:
            v = np.asarray(r.verts)
            fh.write(f'# {r.name}: {r.kind}, {r.npix} px, x {v[:, 0].min():.0f}-{v[:, 0].max():.0f}, '
                     f'y {v[:, 1].min():.0f}-{v[:, 1].max():.0f}\n')

    def export_csv(self):
        ds = self.ds
        if ds is None:
            return
        p = filedialog.asksaveasfilename(parent=self, title='Export spectra and decays (two CSV files)',
                                         initialdir=self.last_dir, defaultextension='.csv',
                                         initialfile=os.path.splitext(ds.name)[0] + '_4D.csv', filetypes=[('CSV', '*.csv')])
        if not p:
            return
        out = self.write_csv(p)
        self.status('exported ' + ' and '.join(out))

    def write_csv(self, p):
        """Write <p>_spectra.csv and <p>_decays.csv; returns the two paths."""
        ds = self.ds
        mode = self._mode()
        it = int(self.s_dl.get())
        lam, band = self._cur_lam(), self._band()
        part = self.v_partial.get()
        base = os.path.splitext(p)[0]
        p_s, p_d = base + '_spectra.csv', base + '_decays.csv'
        x, xlab = ds.axis(mode)
        full_s = ds.spectrum(mode, it, None, part)[1]
        curves = [self._region_curves(r) for r in self.regions]
        with open(p_s, 'w', newline='', encoding='utf-8') as fh:
            self._header(fh, f'spectra at {ds.delay_label} {ds.delay[it]:g}'
                         + (f' (gate width {ds.width[it]:g} ns)' if ds.width is not None else ''))
            w = csv.writer(fh)
            w.writerow([xlab.replace(' ', '_'), 'full_frame'] + [r.name for r in self.regions])
            for i in range(x.size):
                w.writerow([f'{x[i]:.4f}', f'{full_s[i]:.6g}'] + [f'{c[1][i]:.6g}' for c in curves])
        full_d = ds.decay(mode, lam, band, None, part)
        with open(p_d, 'w', newline='', encoding='utf-8') as fh:
            self._header(fh, f'decays in the band {lam - band / 2:.2f}-{lam + band / 2:.2f} nm ({ds.axis(mode)[1]})')
            w = csv.writer(fh)
            w.writerow(['gate_delay_ns' if 'gate delay' in ds.delay_label else 'delay_axis', 'gate_width_ns',
                        'gate_centre_ns', 'full_frame'] + [r.name for r in self.regions])
            for t in range(ds.n_t):
                w.writerow([f'{ds.delay[t]:g}', f'{ds.width[t]:g}' if ds.width is not None else '',
                            f'{ds.centre[t]:g}' if ds.width is not None else '', f'{full_d[t]:.6g}']
                           + [f'{c[2][t]:.6g}' for c in curves])
        return p_s, p_d

    def save_figure(self):
        if self.ds is None:
            return
        p = filedialog.asksaveasfilename(parent=self, title='Save figure', initialdir=self.last_dir, defaultextension='.png',
                                         initialfile=os.path.splitext(self.ds.name)[0] + '_explorer4d.png',
                                         filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')])
        if p:
            self.fig.savefig(p, dpi=150)
            self._bg = {}                                # saving drew at 150 dpi: redraw the screen, new backgrounds
            self.canvas.draw_idle()
            self.status(f'saved {p}')

    def regions_dict(self):
        return {'image_shape': list(self.ds.full_shape), 'source': self.ds.path,
                'regions': [dict(name=r.name, kind=r.kind, color=r.color, verts=r.verts) for r in self.regions]}

    def save_regions(self):
        if self.ds is None or not self.regions:
            return
        p = filedialog.asksaveasfilename(parent=self, title='Save regions', initialdir=self.last_dir,
                                         defaultextension='.json',
                                         initialfile=os.path.splitext(self.ds.name)[0] + '_regions.json',
                                         filetypes=[('JSON', '*.json')])
        if p:
            with open(p, 'w', encoding='utf-8') as fh:
                json.dump(self.regions_dict(), fh, indent=1)
            self.status(f'saved {len(self.regions)} regions to {p}')

    def load_regions(self, p=None):
        if self.ds is None:
            return
        if p is None:
            p = filedialog.askopenfilename(parent=self, title='Load regions', initialdir=self.last_dir,
                                           filetypes=[('JSON', '*.json')])
        if not p:
            return
        with open(p, encoding='utf-8') as fh:
            d = json.load(fh)
        if tuple(d.get('image_shape', self.ds.full_shape)) != tuple(self.ds.full_shape):
            messagebox.showerror('TRPL explorer 4D', f'regions were drawn on a {d["image_shape"]} image, '
                                 f'this one is {list(self.ds.full_shape)}', parent=self)
            return
        for x in d.get('regions', []):                   # add_region makes names unique (R1 -> R1')
            self.add_region(x['kind'], x['verts'], name=x['name'], color=x.get('color'))
        self.status(f'loaded {len(d.get("regions", []))} regions from {p}')


def open_window(master, path=None):
    """Open a 4D window as a Toplevel of a running Tk application (used by trpl_explorer.py)."""
    if path:
        kind, why = detect_4d(path)
        if kind is None or kind == 'other4d':
            messagebox.showerror('TRPL explorer 4D', f'{os.path.basename(path)}:\n{why}', parent=master)
            return None
    return Explorer4D(master, path)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = tk.Tk()
    root.withdraw()
    win = Explorer4D(root, argv[0] if argv else None, own_root=True)
    if not argv:
        win.after(200, win.open_dialog)
    root.mainloop()


if __name__ == '__main__':
    main()
