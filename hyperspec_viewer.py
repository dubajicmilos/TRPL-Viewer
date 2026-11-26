r"""hyperspec_viewer.py -- side-by-side viewer for PHySpec-layout hyperspectral cubes (CMOS, emICCD, PHySpec).

Opens any number of rectified (or registered) cubes in the PHySpec HDF5 layout (/Cube/Images (lambda, y, x),
/Cube/Wavelength, /Cube/GratingID, /Cube/Info/Camera/XAxis@Coefs = um per pixel) and shows

  * top row   : one image per cube at the current wavelength (slider / left-right arrow keys), on a common
                physical grid in um (each cube is aligned to the FIRST cube by phase correlation of the
                wavelength-integrated images, after binning to a common pixel size),
  * bottom    : the spectrum of the SAME physical region in every cube (click an image to choose the region,
                right-click = whole common field), normalised (peak / 480-500 nm band / none), and the ratio
                of each cube to the first one.

Use rectified, dark-subtracted float32 cubes where available. Integer cubes use a default camera offset
of 98.5 counts for CMOS and 575 counts for emICCD. Zero-filled pixels are masked. Compare normalised
spectra and images across detectors because their absolute intensity units can differ.

    python hyperspec_viewer.py A.h5 B.h5
    python hyperspec_viewer.py A.h5 B.h5 --save compare.png --wl 500 --roi 0 0 20     (no window; ROI centre x y and size, um)
Options: --roi-um 20 (click-ROI size), --no-align, --offset FILE_INDEX=VALUE (override the integer-cube offset).
Notes: images are shown in camera row order (origin upper, PHySpec orientation); the CMOS<->emICCD alignment is a
translation only (orientation 'none' and scale 1.00 verified 2026-11-06).
"""
import argparse
import os
import sys

import h5py
import numpy as np

BAND = (480.0, 500.0)          # normalisation band (nm)
GRATING_SWITCH_NM = 599.0      # G0 -> G1 switch (PHySpec: G0 < 600 nm <= G1)


class Cube:
    """One PHySpec-layout cube held in memory as float32 (lambda, y, x) with NaN for invalid pixels."""

    def __init__(self, path, offset=None):
        self.path = path
        self.name = os.path.basename(path)
        with h5py.File(path, 'r') as f:
            c = f['Cube']
            raw = c['Images']
            # Photon etc. single images (e.g. *_Broadband.h5) hold one frame and no /Cube/Wavelength
            self.is_image = raw.shape[0] == 1 and 'Wavelength' not in c
            self.wl = c['Wavelength'][()].astype(float) if 'Wavelength' in c else np.full(raw.shape[0], np.nan)
            self.gid = c['GratingID'][()] if 'GratingID' in c else np.zeros(len(self.wl), int)
            self.integer = raw.dtype.kind in 'ui'
            cam = c['Info/Camera'].attrs
            model = cam.get('Model', [b'?'])[0]
            self.model = model.decode() if isinstance(model, bytes) else str(model)
            try:
                self.px_um = float(c['Info/Camera/XAxis'].attrs['Coefs'][1])
            except Exception:                                   # fall back: pixel pitch / objective (20x)
                self.px_um = float(cam.get('PixelSizeNm', [6500.0])[0]) / 1000.0 / 20.0
            data = raw[()].astype(np.float32)
        if self.integer:
            if offset is None:
                offset = 575.0 if 'PI-MAX' in self.model else 98.5
            data[data == 0] = np.nan                            # rectification zero-fill
            data -= np.float32(offset)
            b = 8                                               # registration fill sits in the outermost rows/cols
            data[:, :b, :] = np.nan; data[:, -b:, :] = np.nan; data[:, :, :b] = np.nan; data[:, :, -b:] = np.nan
        self.offset = offset
        self.data = data
        self.ny, self.nx = data.shape[1:]
        self.shift_um = np.zeros(2)                             # (dy, dx) added to this cube's physical coordinates

    def coords(self):
        """Physical pixel-centre coordinates (um) incl. the alignment shift; (0, 0) = field centre of cube 0."""
        y = (np.arange(self.ny) - (self.ny - 1) / 2.0) * self.px_um + self.shift_um[0]
        x = (np.arange(self.nx) - (self.nx - 1) / 2.0) * self.px_um + self.shift_um[1]
        return y, x

    def extent(self):
        y, x = self.coords()
        h = self.px_um / 2.0
        return [x[0] - h, x[-1] + h, y[-1] + h, y[0] - h]      # origin='upper' (camera row order)

    def integrated(self):
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                return np.nan_to_num(np.nanmean(self.data, axis=0))

    def roi_spectrum(self, roi):
        """Mean spectrum inside roi = (yc, xc, size_um) in physical um, or the whole valid field if roi is None."""
        if roi is None:
            sub = self.data[:, 16:-16, 16:-16]
        else:
            yc, xc, s = roi
            y, x = self.coords()
            iy = np.where(np.abs(y - yc) <= s / 2.0)[0]
            ix = np.where(np.abs(x - xc) <= s / 2.0)[0]
            if len(iy) == 0 or len(ix) == 0:
                return np.full(len(self.wl), np.nan)
            sub = self.data[:, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
        with np.errstate(all='ignore'):
            return np.nanmean(sub.reshape(len(self.wl), -1), axis=1)


def _bin(img, k):
    """k x k block mean (crop to a multiple of k)."""
    if k <= 1:
        return img
    h, w = (img.shape[0] // k) * k, (img.shape[1] // k) * k
    return img[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def _highpass(a):
    from scipy.ndimage import gaussian_filter
    a = a.astype(np.float64)
    return gaussian_filter(a, 1.0) - gaussian_filter(a, 12.0)


def align_to_first(cubes, log=print):
    """Translation of every cube relative to cube 0 (phase correlation of integrated, high-passed images on the
    coarser pixel grid).  Sets cube.shift_um so that equal physical coordinates show the same sample point."""
    from skimage.registration import phase_cross_correlation
    ref = cubes[0]
    for c in cubes[1:]:
        coarse = max(ref.px_um, c.px_um)
        kr, kc = int(round(coarse / ref.px_um)), int(round(coarse / c.px_um))
        a, b = _bin(ref.integrated(), kr), _bin(c.integrated(), kc)
        n = min(a.shape[0], b.shape[0], a.shape[1], b.shape[1])
        def crop(m):
            y0, x0 = (m.shape[0] - n) // 2, (m.shape[1] - n) // 2
            return m[y0:y0 + n, x0:x0 + n]
        a, b = _highpass(crop(a)), _highpass(crop(b))
        shift, err, _ = phase_cross_correlation(a, b, upsample_factor=20, normalization=None)
        # shift = (dy, dx) to apply to b so that it matches a -> b's content sits at +shift in a's frame
        c.shift_um = np.array(shift) * coarse
        bb = np.roll(np.roll(b, int(round(shift[0])), 0), int(round(shift[1])), 1)
        m = 16
        aa, bc = a[m:-m, m:-m], bb[m:-m, m:-m]
        ncc = float(((aa - aa.mean()) * (bc - bc.mean())).sum() / np.sqrt(((aa - aa.mean()) ** 2).sum() * ((bc - bc.mean()) ** 2).sum()))
        log(f'  align {c.name} -> {ref.name}: shift (dy, dx) = ({c.shift_um[0]:+.2f}, {c.shift_um[1]:+.2f}) um '
            f'[{shift[0]:+.2f}, {shift[1]:+.2f} px at {coarse:.3f} um/px], NCC after shift {ncc:.3f}')


def normalise(wl, s, mode):
    if mode == 'none':
        return s
    if mode == 'peak':
        d = np.nanmax(s)
    else:
        m = (wl >= BAND[0]) & (wl <= BAND[1])
        d = np.nanmean(s[m]) if m.any() else np.nanmax(s)
    return s / d if d and np.isfinite(d) else s


def frame_index(c, wl):
    """Frame of cube c nearest to wavelength wl; a single image is its own frame at every wavelength."""
    return 0 if c.is_image else int(np.argmin(np.abs(c.wl - wl)))


def pct_limits(img):
    v = img[np.isfinite(img)]
    v = v[v > 0] if (v > 0).sum() > 100 else v
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    if hi / max(lo, 1e-30) > 50:
        lo, hi = np.percentile(v, 5), np.percentile(v, 95)
    return lo, hi


class Viewer:
    def __init__(self, cubes, roi_um=20.0, norm='band', wl0=500.0, roi=None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, RadioButtons
        self.plt = plt
        self.cubes = cubes
        self.roi_um = roi_um
        self.norm = norm
        self.roi = roi
        n = len(cubes)
        self.colors = ['tab:blue', 'tab:purple', 'k', 'tab:orange', 'tab:green', 'tab:red'][:max(n, 1)]
        self.fig = plt.figure(figsize=(4.6 * max(n, 2) + 1.5, 9.0))
        gs = self.fig.add_gridspec(2, max(n, 2), height_ratios=[1.0, 0.85], hspace=0.32, wspace=0.55,
                                   left=0.06, right=0.86, top=0.94, bottom=0.12)
        self.ax_img = [self.fig.add_subplot(gs[0, i]) for i in range(n)]
        half = max(n, 2) // 2 if max(n, 2) > 2 else 1
        self.ax_spec = self.fig.add_subplot(gs[1, :half])
        self.ax_ratio = self.fig.add_subplot(gs[1, half:])
        self.spec_cubes = [c for c in cubes if not c.is_image]      # single images have no spectrum
        self.wl = self.spec_cubes[0].wl if self.spec_cubes else np.zeros(1)
        self.k = int(np.argmin(np.abs(self.wl - wl0)))
        self.ims, self.rects = [], []
        for ax, c in zip(self.ax_img, cubes):
            j = frame_index(c, self.wl[self.k])
            img = c.data[j]
            lo, hi = pct_limits(img)
            im = ax.imshow(img, cmap='inferno', origin='upper', extent=c.extent(), vmin=lo, vmax=hi, interpolation='nearest')
            cb = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, format='%.2g')
            cb.ax.tick_params(labelsize=7)
            ax.set_xlabel('x (um)'); ax.set_ylabel('y (um)')
            self.ims.append(im)
        self.slider = None
        if self.spec_cubes and len(self.wl) > 1:
            ax_sl = self.fig.add_axes([0.10, 0.035, 0.62, 0.025])
            self.slider = Slider(ax_sl, 'wavelength', float(self.wl[0]), float(self.wl[-1]), valinit=float(self.wl[self.k]),
                                 valstep=np.asarray(self.wl, float))
            self.slider.on_changed(self._on_slider)
        ax_rb = self.fig.add_axes([0.88, 0.14, 0.11, 0.14])
        ax_rb.set_title('normalise', fontsize=9)
        self.radio = RadioButtons(ax_rb, ('band', 'peak', 'none'), active=('band', 'peak', 'none').index(norm))
        self.radio.on_clicked(self._on_norm)
        self.fig.text(0.88, 0.44, 'click image: ROI\n(%.0f um square)\nright-click: whole field\n<- / -> : wavelength'
                      % roi_um, fontsize=8, va='top')
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._update_images()
        self._update_spectra()

    # ---------------------------------------------------------------- callbacks
    def _on_slider(self, val):
        self.k = int(np.argmin(np.abs(self.wl - val)))
        self._update_images(); self._update_markers(); self.fig.canvas.draw_idle()

    def _on_norm(self, label):
        self.norm = label
        self._update_spectra(); self.fig.canvas.draw_idle()

    def _on_key(self, ev):
        if ev.key in ('left', 'right') and self.slider is not None:
            self.k = int(np.clip(self.k + (1 if ev.key == 'right' else -1), 0, len(self.wl) - 1))
            self.slider.set_val(float(self.wl[self.k]))

    def _on_click(self, ev):
        if ev.inaxes not in self.ax_img or ev.xdata is None:
            return
        self.roi = None if ev.button == 3 else (float(ev.ydata), float(ev.xdata), self.roi_um)
        self._update_spectra(); self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------- drawing
    def _update_images(self):
        for ax, im, c in zip(self.ax_img, self.ims, self.cubes):
            j = frame_index(c, self.wl[self.k])
            img = c.data[j]
            im.set_data(img)
            lo, hi = pct_limits(img)
            im.set_clim(lo, hi)
            what = 'single image' if c.is_image else f'{c.wl[j]:.0f} nm, G{int(c.gid[j])}'
            ax.set_title(f'{c.name}\n{c.model} | {what}', fontsize=9)

    def _update_markers(self):
        for ln in getattr(self, '_wl_lines', []):
            ln.set_xdata([self.wl[self.k], self.wl[self.k]])

    def _update_spectra(self):
        import matplotlib.patches as mp
        for r in self.rects:
            r.remove()
        self.rects = []
        if self.roi is not None:
            yc, xc, s = self.roi
            for ax in self.ax_img:
                r = mp.Rectangle((xc - s / 2, yc - s / 2), s, s, fill=False, ec='cyan', lw=1.5)
                ax.add_patch(r); self.rects.append(r)
        a, b = self.ax_spec, self.ax_ratio
        a.clear(); b.clear()
        spectra = []
        spec = [(c, col) for c, col in zip(self.cubes, self.colors) if not c.is_image]
        for c, col in spec:
            s = normalise(c.wl, c.roi_spectrum(self.roi), self.norm)
            spectra.append(s)
            for g in np.unique(c.gid):                         # separate lines per grating (no line across 600 nm)
                m = c.gid == g
                a.plot(c.wl[m], s[m], '-', color=col, lw=1.6, label=f'{c.name}' if g == np.unique(c.gid)[0] else None)
        ref = spec[0][0] if spec else None
        for (c, col), s in zip(spec[1:], spectra[1:]):
            si = np.interp(ref.wl, c.wl, s) if not np.array_equal(c.wl, ref.wl) else s
            with np.errstate(all='ignore'):
                r = si / spectra[0]
            for g in np.unique(ref.gid):
                m = ref.gid == g
                b.plot(ref.wl[m], r[m], '-', color=col, lw=1.6, label=f'{c.name} / {ref.name}' if g == np.unique(ref.gid)[0] else None)
        for ax in (a, b):
            ax.axvline(GRATING_SWITCH_NM, color='grey', ls=':', lw=1)
            ax.grid(alpha=.3); ax.set_xlabel('wavelength (nm)')
        self._wl_lines = [a.axvline(self.wl[self.k], color='red', lw=0.8, alpha=0.6),
                          b.axvline(self.wl[self.k], color='red', lw=0.8, alpha=0.6)]
        where = 'whole common field' if self.roi is None else f'ROI {self.roi[2]:.0f} um at (x={self.roi[1]:.1f}, y={self.roi[0]:.1f}) um'
        a.set_title(f'spectra, {where}; normalise: {self.norm}', fontsize=9)
        a.set_ylabel('signal (normalised)' if self.norm != 'none' else 'signal (each file in its own units)')
        if spec:
            a.legend(fontsize=8)
        else:
            a.text(0.5, 0.5, 'single images only: no spectra', transform=a.transAxes, ha='center', va='center')
        b.set_title('ratio to the first cube with a spectrum (shape comparison)', fontsize=9)
        b.set_ylabel('ratio')
        if len(spec) > 1:
            b.legend(fontsize=8)

    def show(self):
        plt = self.plt
        try:                                                    # lab convention: bring the Tk window to the front
            win = self.fig.canvas.manager.window
            win.lift(); win.attributes('-topmost', True); win.after(500, lambda: win.attributes('-topmost', False))
        except Exception:
            pass
        plt.show()


def main(argv=None):
    ap = argparse.ArgumentParser(description='Side-by-side viewer for PHySpec-layout hyperspectral cubes (CMOS / emICCD / PHySpec).',
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('cubes', nargs='+', help='PHySpec-layout .h5 cubes (rectified recommended; the first one is the reference)')
    ap.add_argument('--roi-um', type=float, default=20.0, help='side of the click ROI square (um), default 20')
    ap.add_argument('--norm', choices=('band', 'peak', 'none'), default='band', help='spectrum normalisation (band = %g-%g nm)' % BAND)
    ap.add_argument('--wl', type=float, default=500.0, help='initial wavelength (nm)')
    ap.add_argument('--roi', type=float, nargs=3, metavar=('X_UM', 'Y_UM', 'SIZE_UM'), help='initial ROI (default whole field)')
    ap.add_argument('--no-align', action='store_true', help='do not align the cubes (shift 0)')
    ap.add_argument('--offset', action='append', default=[], metavar='INDEX=VALUE',
                    help='camera offset to subtract from integer cube INDEX (0-based), e.g. --offset 2=98.5')
    ap.add_argument('--save', metavar='PNG', help='write the figure and exit (no window)')
    a = ap.parse_args(argv)

    offsets = {}
    for o in a.offset:
        i, v = o.split('='); offsets[int(i)] = float(v)
    import matplotlib
    matplotlib.use('Agg' if a.save else 'TkAgg')
    cubes = []
    for i, p in enumerate(a.cubes):
        print(f'loading {p} ...', flush=True)
        c = Cube(p, offsets.get(i))
        spectral = 'single image' if c.is_image else f'{c.data.shape[0]} wavelengths {c.wl[0]:.0f}-{c.wl[-1]:.0f} nm'
        print(f'  {c.model}: {spectral}, {c.nx}x{c.ny} px at {c.px_um:.4f} um/px'
              + (f', integer cube: offset {c.offset} subtracted' if c.integer else ''))
        cubes.append(c)
    if len(cubes) > 1 and not a.no_align:
        align_to_first(cubes)
    roi = None if a.roi is None else (a.roi[1], a.roi[0], a.roi[2])
    v = Viewer(cubes, roi_um=a.roi_um, norm=a.norm, wl0=a.wl, roi=roi)
    if a.save:
        v.fig.savefig(a.save, dpi=120)
        print('saved', a.save)
    else:
        v.show()


if __name__ == '__main__':
    main()
