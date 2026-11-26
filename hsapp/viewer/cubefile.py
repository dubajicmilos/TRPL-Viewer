"""CubeFile: read-only access to PHySpec-layout hyperspectral HDF5 cubes (no Qt).

Understands PHySpec's own files, the files written by `hyperspec_finalize` / `hyperspec_emiccd_finalize` and the
hsapp processing worker: raw, rectified `(1)`, registered `(2)`, dark-subtracted `_dc(`, photometric `_pc(`, `_dark`.

Dataset layout:
    /Cube/Images (n, h, w) uint16 | float32     /Cube/Wavelength (n,)   /Cube/GratingID (n,)   /Cube/TimeExposure (n,)
    /Cube/Timestamp (n,) S23 (raw)              /Cube/Angle (n,) (raw cubes only)
    /Cube/Info/Camera/XAxis@Coefs = [0, um per pixel]   /Cube/Info/Optics@Objective   /Cube/Info/Camera@Model, PixelSizeNm
    /Cube/Info/Cube@PixelUnit (photometric cubes)       /Cube/Info/Processings/NNN (Name, Unit, ...)
    time-series files (`hyperspec_acq.py timeseries`) have NO /Cube group: the frames are in /Acq/TimeSeries/Images
    (same companion datasets); CubeFile then shows those (`source_group == 'Acq/TimeSeries'`).

Invalid pixels: exact zeros (rectification / registration fill) and NaN are NaN in everything this class returns.
The camera offset (98.5 counts CMOS, 575 counts emICCD for integer cubes; INFERRED from hyperspec_viewer.py, the
values it applies for the same files) is NOT subtracted unless `offset` is set; `offset_guess()` returns the suggestion.
"""
import os
import re
from datetime import datetime

import h5py
import numpy as np

# Objective magnifications as used by Hyperspectral_analysis.hf (CLAUDE.md table); only a fallback for files
# without /Cube/Info/Camera/XAxis@Coefs.  Leading number of the objective name selects the entry ('63x-0.7mm' -> 63).
MAGNIFICATION = {2: 2.0, 10: 10.0, 20: 20.155, 50: 50.3875, 63: 78.7496, 100: 111.972}
CMOS_OFFSET_COUNTS = 98.5
EMICCD_OFFSET_COUNTS = 575.0
DEFAULT_BUDGET_BYTES = 16 * 2 ** 30
CHUNK_BYTES = 64 * 2 ** 20                       # frames are read / converted in blocks of about this size

_TOKEN = r'_%s(?=[(_.]|$)'                      # '_dc' / '_pc' / '_dark' as a name token: sweep1_dc(1), sweep1(1)_pc_corrected
_STAGE = re.compile(r'\((\d)\)')


class CubeFormatError(ValueError):
    """The file is not a PHySpec-layout cube (no readable /Cube/Images or /Acq/TimeSeries/Images)."""


class CubeTooLarge(MemoryError):
    """load_all() refused: the float32 cube would exceed the RAM budget (use lazy access)."""


def _decode(value):
    """h5py attribute -> plain Python (bytes -> str, 1-element arrays -> scalars, numpy scalars -> int/float)."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in 'SO':
            items = [v.decode('utf-8', 'replace') if isinstance(v, bytes) else str(v) for v in value.ravel()]
            return items[0] if len(items) == 1 else items
        if value.size == 1:
            return value.ravel()[0].item()
        return value.tolist() if value.size <= 16 else value
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    if isinstance(value, np.generic):
        return value.item()
    return value


def nest_flat(flat, sep='/'):
    """{'Camera/Model': 'x', 'Camera/Bits': 16} -> {'Camera': {'Model': 'x', 'Bits': 16}} (insertion order kept)."""
    out = {}
    for key, value in flat.items():
        node = out
        parts = str(key).split(sep)
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = node[part] = {}
            node = nxt
        node[parts[-1]] = value
    return out


def classify_name(path):
    """(kind, stage, corrections) from the file name.

    kind: 'raw' | 'rectified' | 'registered' | 'dark-subtracted' | 'photometric' | 'dark', first match of the name
    tokens `_dark`, `_pc`, `_dc`, then `(1)`, `(2)`, else raw (e.g. `x_dark.h5`, `x_pc(1).h5`, `x(1)_pc_corrected.h5`,
    `x_dc(2).h5`).  stage: 'raw' | 'rectified' | 'registered' from the `(n)` token.
    corrections: subset of ('dc', 'pc') in processing order."""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    has = {t: re.search(_TOKEN % t, stem) is not None for t in ('dark', 'pc', 'dc')}
    m = _STAGE.search(stem)
    stage = {'1': 'rectified', '2': 'registered'}.get(m.group(1), 'raw') if m else 'raw'
    if has['dark']:
        kind = 'dark'
    elif has['pc']:
        kind = 'photometric'
    elif has['dc']:
        kind = 'dark-subtracted'
    else:
        kind = stage
    return kind, stage, tuple(t for t in ('dc', 'pc') if has[t])


def auto_levels(img, low=0.5, high=99.5, max_samples=250_000):
    """(lo, hi) display levels from percentiles of the finite pixels (hot pixels ignored).

    The image is strided to about `max_samples` pixels first, so this is fast on 1 Mpx frames.  Always hi > lo."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = a.reshape(a.shape[0], -1)
    step = max(1, int(np.ceil(np.sqrt(a.size / float(max_samples)))))
    sub = a[::step, ::step] if a.ndim == 2 and step > 1 else a
    v = sub[np.isfinite(sub)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [low, high])
    lo, hi = float(lo), float(hi)
    if not hi > lo:
        span = max(abs(lo) * 1e-6, 1e-12)
        lo, hi = lo - span, hi + span
    return lo, hi


def to_rgb_uint8(rgb, low=0.5, high=99.5):
    """(h, w, 3) float -> uint8 with each channel scaled between its own percentiles; NaN becomes 0."""
    out = np.zeros(rgb.shape, np.uint8)
    for c in range(3):
        ch = rgb[..., c]
        lo, hi = auto_levels(ch, low, high)
        scaled = (ch - lo) / (hi - lo)
        out[..., c] = np.nan_to_num(np.clip(scaled, 0.0, 1.0) * 255.0, nan=0.0).astype(np.uint8)
    return out


class CubeFile:
    """One cube file.  Metadata is read in the constructor; pixel data stays on disk until `load_all()`.

    Public attributes: path, name, kind, stage, corrections, wavelengths (n,), grating_ids (n,), exposure (n,),
    objective, pixel_size_um (+ pixel_size_calibrated), units, info (flat 'Group/Sub/key' -> value, decoded),
    shape (n, h, w), dtype, is_emiccd, timeseries (/Acq/TimeSeries present), source_group, timestamps (list of str or
    None), offset (subtracted from every returned value, default 0.0).

    Thread note: a CubeFile holds no open h5py handle (every disk read opens and closes the file), so it can be
    created in a worker thread and used in the GUI thread."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.name = os.path.splitext(os.path.basename(self.path))[0]
        self.kind, self.stage, self.corrections = classify_name(self.path)
        self.offset = 0.0
        self._data = None
        self._integrated = None
        self._full_spectrum = None
        try:
            with h5py.File(self.path, 'r') as f:
                self._read_meta(f)
        except OSError as exc:
            raise CubeFormatError('cannot open %s as HDF5: %s' % (self.path, exc)) from exc

    # ------------------------------------------------------------------ metadata
    def _read_meta(self, f):
        group = None
        for cand in ('Cube', 'Acq/TimeSeries'):
            if cand + '/Images' in f and f[cand + '/Images'].ndim == 3 and f[cand + '/Images'].shape[0] > 0:
                group = cand
                break
        if group is None:
            raise CubeFormatError('%s: no non-empty 3-D /Cube/Images or /Acq/TimeSeries/Images' % self.path)
        self.source_group = group
        self.timeseries = 'Acq/TimeSeries/Images' in f
        ds = f[group + '/Images']
        self.shape = tuple(int(s) for s in ds.shape)
        self.dtype = ds.dtype
        self.compression = ds.compression
        n = self.shape[0]
        g = f[group]
        self.wavelengths = g['Wavelength'][()].astype(float) if 'Wavelength' in g else np.arange(n, dtype=float)
        self.grating_ids = g['GratingID'][()].astype(int) if 'GratingID' in g else np.zeros(n, int)
        self.exposure = g['TimeExposure'][()].astype(float) if 'TimeExposure' in g else np.full(n, np.nan)
        for name in ('Wavelength', 'GratingID', 'TimeExposure'):
            if name in g and g[name].shape[0] != n:
                raise CubeFormatError('%s: /%s/%s has %d entries but there are %d frames'
                                      % (self.path, group, name, g[name].shape[0], n))
        self.timestamps = ([t.decode('utf-8', 'replace') for t in g['Timestamp'][()]]
                           if 'Timestamp' in g and g['Timestamp'].shape[0] == n else None)
        self.has_angle = 'Angle' in g
        self.info = self._flatten_info(f, group)
        self.camera_model = str(self.info.get('Camera/Model', ''))
        self.is_emiccd = ('PI-MAX' in self.camera_model.upper()
                          or float(self.info.get('Camera/PixelSizeNm', 0) or 0) == 13000.0)
        self.objective = str(self.info.get('Optics/Objective', '')) or None
        self.pixel_size_um, self.pixel_size_calibrated = self._pixel_size(f, group)
        self.units = self._units(f, group)

    def _flatten_info(self, f, group):
        """/<group>/Info attrs as {'Camera/Model': 'C13440-20CU', ...}.  Registration blocks of other objectives and
        empty groups are skipped; /Acq scalar attributes (and /Acq/emICCD) are added under 'Acq/'."""
        flat = {}
        objective = None
        if group + '/Info/Optics' in f:
            raw = f[group + '/Info/Optics'].attrs.get('Objective')
            objective = _decode(raw) if raw is not None else None

        def walk(node, prefix):
            for key, val in node.attrs.items():
                flat[prefix + key] = _decode(val)
            for name, child in node.items():
                if not isinstance(child, h5py.Group):
                    continue
                if node.name.endswith('/Registration') and objective is not None and name != objective:
                    continue
                walk(child, prefix + name + '/')

        if group + '/Info' in f:
            walk(f[group + '/Info'], '')
        for sub in ('Acq', 'Acq/emICCD'):
            if sub in f:
                for key, val in f[sub].attrs.items():
                    flat['Acq/' + ('emICCD/' if sub.endswith('emICCD') else '') + key] = _decode(val)
        return {k: v for k, v in flat.items() if not (isinstance(v, str) and v == '')}

    def _pixel_size(self, f, group):
        """(um per pixel, calibrated?) from XAxis@Coefs[1]; else PixelSizeNm / magnification of the objective;
        else (1.0, False) and the viewer labels its axes in pixels."""
        coefs = self.info.get('Camera/XAxis/Coefs')
        if coefs is not None:
            coefs = np.atleast_1d(coefs)
            if coefs.size >= 2 and np.isfinite(coefs[1]) and float(coefs[1]) > 0:
                return float(coefs[1]), True
        cam_nm = self.info.get('Camera/PixelSizeNm')
        m = re.match(r'\s*(\d+)', self.objective or '')
        mag = MAGNIFICATION.get(int(m.group(1))) if m else None
        if cam_nm and mag:
            return float(cam_nm) / 1000.0 / mag, True
        return 1.0, False

    def _units(self, f, group):
        unit = self.info.get('Cube/PixelUnit')
        if unit:
            return str(unit)
        keys = sorted((k for k in self.info if k.startswith('Processings/') and k.endswith('/Unit')), reverse=True)
        return str(self.info[keys[0]]) if keys else 'counts'      # the last processing step that names a unit

    def offset_guess(self):
        """Camera offset to subtract from integer cubes: 98.5 (CMOS) / 575 (emICCD); 0.0 for float cubes."""
        if self.dtype.kind not in 'ui':
            return 0.0
        return EMICCD_OFFSET_COUNTS if self.is_emiccd else CMOS_OFFSET_COUNTS

    # ------------------------------------------------------------------ geometry / labels
    @property
    def n_frames(self):
        return self.shape[0]

    @property
    def size_um(self):
        """(height, width) of the field of view in um."""
        return self.shape[1] * self.pixel_size_um, self.shape[2] * self.pixel_size_um

    def nearest_frame(self, wavelength):
        """Index of the frame whose wavelength is closest to `wavelength` (first one on ties)."""
        return int(np.argmin(np.abs(self.wavelengths - wavelength)))

    @property
    def elapsed_s(self):
        """(n,) seconds since the first frame from /Timestamp ('YYYY/MM/DD HH:MM:SS.mmm' local time), or None when
        the file has no (parsable) time stamps."""
        if self.timestamps is None:
            return None
        try:
            t = [datetime.strptime(s.strip(), '%Y/%m/%d %H:%M:%S.%f') for s in self.timestamps]
        except ValueError:
            return None
        return np.array([(x - t[0]).total_seconds() for x in t])

    def frame_label(self, i):
        """'λ = 520.0 nm (G0)' (raw cubes: filter wavelength); time series add the time stamp."""
        text = '\u03bb = %.1f nm (G%d)' % (self.wavelengths[i], self.grating_ids[i])
        if self.timestamps is not None and self.source_group != 'Cube':
            text += '  ' + self.timestamps[i]
        return text

    def default_rgb_bands(self):
        """[(lo, hi)] for R, G, B: the cube's wavelength range in thirds (longest third = red)."""
        lo, hi = float(np.nanmin(self.wavelengths)), float(np.nanmax(self.wavelengths))
        t = (hi - lo) / 3.0
        return [(lo + 2 * t, hi), (lo + t, lo + 2 * t), (lo, lo + t)]

    # ------------------------------------------------------------------ data access
    @property
    def in_ram(self):
        return self._data is not None

    def ram_bytes(self):
        """Size of the float32 cube in RAM."""
        return int(np.prod(self.shape)) * 4

    def fits_in_ram(self, budget_bytes=DEFAULT_BUDGET_BYTES):
        return self.ram_bytes() <= budget_bytes

    def release(self):
        """Drop the RAM copy and the cached integrated image / full-field spectrum."""
        self._data = None
        self._integrated = None
        self._full_spectrum = None

    def _prep(self, block):
        """Raw block -> float32 with zeros as NaN (no offset)."""
        out = np.array(block, dtype=np.float32)
        out[out == 0] = np.nan
        return out

    def _read_frames(self, index):
        """float32 (k, h, w) of the given frame indices (no offset), from RAM or disk."""
        index = np.asarray(index, dtype=int)
        if self._data is not None:
            return self._data[index]
        order = np.argsort(index, kind='stable')
        out = np.empty((len(index),) + self.shape[1:], np.float32)
        with h5py.File(self.path, 'r') as f:
            ds = f[self.source_group + '/Images']
            for pos in order:
                out[pos] = self._prep(ds[int(index[pos])])
        return out

    def _read_window(self, y0, y1, x0, x1):
        """float32 (n, y1-y0, x1-x0) of all frames inside the window (no offset)."""
        if self._data is not None:
            return self._data[:, y0:y1, x0:x1]
        with h5py.File(self.path, 'r') as f:
            block = f[self.source_group + '/Images'][:, y0:y1, x0:x1]
        return self._prep(block)

    def load_all(self, progress=None, budget_bytes=DEFAULT_BUDGET_BYTES):
        """Read the whole cube into RAM as float32 (read-only) and return it.

        Raises CubeTooLarge when the float32 cube exceeds `budget_bytes` (nothing is read; use `scan()` and lazy access).
        `progress(done_frames, n_frames)` is called after every block; an exception raised by it aborts the load and
        leaves the object unloaded.  Also fills the cached `integrated()` image and `full_spectrum()`."""
        if self._data is not None:
            return self._data
        if not self.fits_in_ram(budget_bytes):
            raise CubeTooLarge('%s needs %.1f GB as float32, budget %.1f GB'
                               % (self.name, self.ram_bytes() / 2 ** 30, budget_bytes / 2 ** 30))
        data = np.empty(self.shape, np.float32)
        try:
            self._stream(progress, store=data)
        except BaseException:
            self.release()
            raise
        data.flags.writeable = False
        self._data = data
        return data

    def scan(self, progress=None):
        """One streaming pass over the file computing `integrated()` and `full_spectrum()` without keeping the
        cube (for cubes above the RAM budget)."""
        self._stream(progress, store=None)

    def _stream(self, progress, store):
        n, h, w = self.shape
        per = max(1, CHUNK_BYTES // (h * w * 4))
        acc = np.zeros((h, w), np.float64)
        cnt = np.zeros((h, w), np.int32)
        spec = np.full(n, np.nan)
        with h5py.File(self.path, 'r') as f:
            ds = f[self.source_group + '/Images']
            for i0 in range(0, n, per):
                i1 = min(n, i0 + per)
                blk = self._prep(ds[i0:i1])
                if store is not None:
                    store[i0:i1] = blk
                ok = np.isfinite(blk)
                z = np.where(ok, blk, np.float32(0))
                acc += z.sum(axis=0, dtype=np.float64)
                cnt += ok.sum(axis=0, dtype=np.int32)
                cnts = ok.reshape(i1 - i0, -1).sum(axis=1)
                sums = z.reshape(i1 - i0, -1).sum(axis=1, dtype=np.float64)
                with np.errstate(all='ignore'):
                    spec[i0:i1] = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
                if progress is not None:
                    progress(i1, n)
        with np.errstate(all='ignore'):
            self._integrated = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
        self._full_spectrum = spec

    def frame(self, i):
        """float32 (h, w) of frame i (offset subtracted, invalid = NaN).  With offset 0 and the cube in RAM this is a
        read-only view, otherwise a new array."""
        if not 0 <= i < self.n_frames:
            raise IndexError('frame %d out of range 0..%d' % (i, self.n_frames - 1))
        if self._data is not None:
            fr = self._data[i]
        else:
            with h5py.File(self.path, 'r') as f:
                fr = self._prep(f[self.source_group + '/Images'][int(i)])
        return fr - np.float32(self.offset) if self.offset else fr

    def roi_mean(self, y0, y1, x0, x1):
        """Mean spectrum (n,) float64 over the pixel window rows y0:y1, cols x0:x1 (NaN pixels ignored; offset
        subtracted).  The window is clipped to the image; an empty window or all-NaN frames give NaN."""
        n, h, w = self.shape
        y0, y1, x0, x1 = max(0, int(y0)), min(h, int(y1)), max(0, int(x0)), min(w, int(x1))
        if y1 <= y0 or x1 <= x0:
            return np.full(n, np.nan)
        win = self._read_window(y0, y1, x0, x1).reshape(n, -1)
        ok = np.isfinite(win)
        cnt = ok.sum(axis=1)
        sums = np.where(ok, win, np.float32(0)).sum(axis=1, dtype=np.float64)
        with np.errstate(all='ignore'):
            spec = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
        return spec - self.offset

    def spectrum(self, y, x, bin=1):
        """Mean spectrum of the bin x bin pixel block centred on pixel (row y, col x) (block clipped at the edges)."""
        if bin < 1:
            raise ValueError('bin must be >= 1')
        lo, hi = (bin - 1) // 2, bin // 2 + 1
        return self.roi_mean(y - lo, y + hi, x - lo, x + hi)

    def full_spectrum(self):
        """Mean spectrum over all valid pixels (cached after load_all / scan)."""
        if self._full_spectrum is None:
            self.scan()
        return self._full_spectrum - self.offset

    def integrated(self):
        """(h, w) float32 mean over all frames (NaN pixels ignored; cached after load_all / scan)."""
        if self._integrated is None:
            self.scan()
        return self._integrated - np.float32(self.offset) if self.offset else self._integrated

    def band_indices(self, lo, hi):
        """Frame indices with lo <= wavelength <= hi; if none, the single frame nearest to the band centre."""
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        idx = np.nonzero((self.wavelengths >= lo) & (self.wavelengths <= hi))[0]
        return idx if idx.size else np.array([self.nearest_frame(0.5 * (lo + hi))])

    def band(self, lo, hi):
        """(h, w) float32 mean of the frames inside the wavelength band [lo, hi] nm (see band_indices)."""
        idx = self.band_indices(lo, hi)
        n, h, w = self.shape
        per = max(1, CHUNK_BYTES // (h * w * 4))
        acc = np.zeros((h, w), np.float64)
        cnt = np.zeros((h, w), np.int32)
        for k in range(0, len(idx), per):
            blk = self._read_frames(idx[k:k + per])
            ok = np.isfinite(blk)
            acc += np.where(ok, blk, np.float32(0)).sum(axis=0, dtype=np.float64)
            cnt += ok.sum(axis=0, dtype=np.int32)
        with np.errstate(all='ignore'):
            out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
        return out - np.float32(self.offset) if self.offset else out

    def rgb(self, bands):
        """(h, w, 3) float32: channels R, G, B = band means of three (lo, hi) tuples (see to_rgb_uint8 for display)."""
        if len(bands) != 3:
            raise ValueError('rgb() needs exactly three (lo, hi) bands, got %d' % len(bands))
        return np.stack([self.band(lo, hi) for lo, hi in bands], axis=-1)

    # ------------------------------------------------------------------ summary
    def describe(self):
        """Nested dict for the metadata panel: file summary + the decoded /Cube/Info attributes."""
        size = os.path.getsize(self.path)
        file_info = {
            'path': self.path, 'kind': self.kind, 'stage': self.stage,
            'corrections': ', '.join(self.corrections) or 'none',
            'frames': self.n_frames, 'image (h x w)': '%d x %d' % self.shape[1:], 'dtype': str(self.dtype),
            'compression': self.compression or 'none', 'size on disk (MB)': round(size / 2 ** 20, 1),
            'source group': self.source_group, 'units': self.units,
            'wavelength range (nm)': '%.1f - %.1f' % (np.nanmin(self.wavelengths), np.nanmax(self.wavelengths)),
            'pixel size (um)': ('%.4g' % self.pixel_size_um) + ('' if self.pixel_size_calibrated else ' (UNCALIBRATED)'),
            'detector': 'emICCD' if self.is_emiccd else 'sCMOS',
            'time series present': self.timeseries,
        }
        return {'File': file_info, 'Info': nest_flat(self.info)}
