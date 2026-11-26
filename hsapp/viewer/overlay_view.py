"""Cube overlay viewer: several hyperspectral cubes on one set of axes (PySide6 through pyqtgraph.Qt).

    python -m hsapp.viewer.overlay_view cube-a.h5 cube-b.h5

What it shows
    left    the cube list: show / hide, colour (double-click the swatch), start time, wavelength range, units, gain x acc,
            shift x / y in um (editable; "Align to reference" fills it by phase correlation of the integrated images).
            One cube is the REFERENCE (bold): clicked pixels and ROIs are in its frame, in um.
    middle  the image: the selected cube | colour overlay of all shown cubes | difference / relative change selected vs
            reference, at one wavelength or as the mean over a band.  "Same levels for all cubes" keeps the levels fixed
            while you step through the list (PageUp / PageDown, or Cycle).
    right   the spectra of every shown cube at the clicked pixel, up to three ROIs and the full field (colour = cube, line
            style = probe), and below one metric (peak / centroid wavelength, integral, maximum inside the band) against
            the cube order or the start time.

Read-only: no hardware, no hsapp lock, no settings file is written.  The core maths is in overlay_core (Qt-free, tested).
"""
import argparse
import csv
import os
import sys
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from hsapp.gui import theme

from . import overlay_core as OC
from .cubefile import CubeFile, auto_levels
from .widgets import ColormapCombo, KeyValueTree, make_spinbox

IMAGE_MODES = ('selected cube', 'colour overlay of the shown cubes', 'difference: selected \u2212 reference',
               'change: selected / reference \u2212 1 (%)')
IMG_SELECTED, IMG_OVERLAY, IMG_DIFF, IMG_REL = range(4)
LAMBDA_MODES = ('single \u03bb', 'band \u03bb1\u2013\u03bb2')
NORMS = ('none', 'peak', 'band')
MAX_ROIS = 3
PROBE_STYLES = OrderedDict([('pixel', QtCore.Qt.PenStyle.SolidLine), ('ROI 1', QtCore.Qt.PenStyle.DashLine),
                            ('ROI 2', QtCore.Qt.PenStyle.DotLine), ('ROI 3', QtCore.Qt.PenStyle.DashDotLine),
                            ('full field', QtCore.Qt.PenStyle.DashDotDotLine)])
STYLE_NOTE = ('line style = probe:  \u2500\u2500 pixel   - - ROI 1   \u00b7\u00b7\u00b7 ROI 2   -\u00b7- ROI 3   '
              '-\u00b7\u00b7- full field;  colour = cube')
NEUTRAL = '#E0E0E0'
DIVERGING = pg.ColorMap([0.0, 0.5, 1.0], [(33, 102, 172), (235, 235, 235), (178, 24, 43)])
CYCLE_MS = 700
IMG_CACHE = 48
COL_SHOW, COL_NAME, COL_TIME, COL_WL, COL_UNITS, COL_GA, COL_DX, COL_DY = range(8)
HEADERS = ('show', 'cube', 'start', '\u03bb (nm)', 'units', 'gain\u00d7acc', 'dx (\u00b5m)', 'dy (\u00b5m)')


class CubeEntry:
    """One loaded cube in the list."""

    def __init__(self, uid, cf, storage, t0, t_src, gain, acc, ga_src):
        self.uid, self.cf, self.storage = uid, cf, storage
        self.t0, self.t_src = t0, t_src
        self.gain, self.acc, self.ga_src = gain, acc, ga_src
        self.color = None
        self.custom_color = False                            # picked by hand: kept until the colour scheme is changed
        self.shown = True
        self.shift = [0.0, 0.0]                              # um, see overlay_core

    @property
    def name(self):
        return self.cf.name

    def meta(self):
        cf = self.cf
        return {'name': cf.name, 'units': cf.units, 'dtype_kind': cf.dtype.kind, 'gain': self.gain, 'acc': self.acc,
                'px': cf.pixel_size_um, 'shape': cf.shape[1:]}


class _LoadThread(QtCore.QThread):
    """Opens and loads cubes one after the other off the GUI thread (RAM while the shared budget lasts, lazy after that)."""

    loaded = QtCore.Signal(str, object)
    failed = QtCore.Signal(str, str)
    progress = QtCore.Signal(str, int, int)

    def __init__(self, paths, budget_bytes, parent=None):
        super().__init__(parent)
        self._paths, self.budget = list(paths), int(budget_bytes)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        for k, path in enumerate(self._paths):
            if self._cancel:
                return
            name = os.path.basename(path)
            try:
                cf = CubeFile(path)

                def tick(i, n, nm=name, k=k):
                    if self._cancel:
                        raise InterruptedError()
                    self.progress.emit('%s (%d/%d): frame %d/%d' % (nm, k + 1, len(self._paths), i, n), i, n)
                if cf.ram_bytes() <= self.budget:
                    cf.load_all(tick, self.budget)
                    self.budget -= cf.ram_bytes()
                    storage = 'RAM'
                else:
                    cf.scan(tick)
                    storage = 'lazy (from disk)'
                t0, t_src = OC.cube_start_time(cf)
                gain, acc, ga_src = OC.cube_gain_acc(cf)
            except InterruptedError:
                return
            except Exception as exc:                          # reported in the window, the other files still load
                self.failed.emit(path, '%s: %s' % (type(exc).__name__, exc))
                continue
            self.loaded.emit(path, (cf, storage, t0, t_src, gain, acc, ga_src))


@contextmanager
def busy():
    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()


def _swatch(color, shown=True, size=14):
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtGui.QColor(0, 0, 0, 0))
    p = QtGui.QPainter(pm)
    c = QtGui.QColor(color)
    if not shown:
        c.setAlpha(70)
    p.fillRect(1, 1, size - 2, size - 2, c)
    p.end()
    return QtGui.QIcon(pm)


class OverlayViewer(QtWidgets.QMainWindow):
    """Main window.  Scriptable: add_paths(), wait_loaded(), select_point(x_um, y_um), add_roi(...), set_band(lo, hi),
    set_wavelength(wl), align_to_reference(), current_spectra(), current_metrics(), export_*()."""

    def __init__(self, ram_budget_gb=16.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Cube overlay viewer')
        self.setAcceptDrops(True)
        self._budget = int(ram_budget_gb * 2 ** 30)
        self.entries = []
        self._uid = 0
        self.ref_uid = None
        self._pix = None                                     # (x_um, y_um) in the reference frame
        self._rois = []                                      # dict(label, roi)
        self._img_cache = OrderedDict()
        self._shown_img = None
        self._loader = None
        self._queue = []
        self._failed = []
        self._curves = []
        self._metric_items = []
        self._metrics = {}
        self._guard = False
        self._setting_levels = False
        self._build_ui()
        self._set_enabled(False)

    # ================================================================================================== UI
    def _build_ui(self):
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.setCentralWidget(split)

        # ---------------------------------------------------------------- left: cube list
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton('Add cubes\u2026')
        self.add_btn.setToolTip('Add .h5 cubes (several at once; they are sorted naturally: r2 before r10).  You can also drop '
                                'files onto the window.')
        self.remove_btn = QtWidgets.QPushButton('Remove')
        self.ref_btn = QtWidgets.QPushButton('Set reference')
        self.ref_btn.setToolTip('The selected cube becomes the reference: pixels / ROIs / shifts are in its frame')
        for w in (self.add_btn, self.remove_btn, self.ref_btn):
            row.addWidget(w)
        lv.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.align_btn = QtWidgets.QPushButton('Align to reference')
        self.align_btn.setToolTip('Shift of every shown cube against the reference by phase correlation of the integrated images '
                                  '(same pixel size and image size only).  The shifts go into dx / dy and can be edited.')
        self.zero_btn = QtWidgets.QPushButton('Zero shifts')
        row.addWidget(self.align_btn)
        row.addWidget(self.zero_btn)
        row.addStretch(1)
        lv.addLayout(row)
        self.table = QtWidgets.QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.setToolTip('Tick = shown in the spectra / overlay.  Double-click the swatch to change the colour.  dx / dy: '
                              'where the reference\'s features appear in this cube (um), editable.')
        lv.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel('Colours'))
        self.scheme_combo = QtWidgets.QComboBox()
        self.scheme_combo.addItems(['distinct (up to 8)', 'sequence (viridis, list order)'])
        self.scheme_combo.setToolTip('distinct: a fixed colour per cube (colour-blind safe); sequence: dark -> yellow along the list, '
                                     'for time series.  More than 8 cubes always use the sequence.')
        row.addWidget(self.scheme_combo)
        row.addStretch(1)
        lv.addLayout(row)
        self.offset_check = QtWidgets.QCheckBox('Subtract camera offset (integer cubes)')
        self.offset_check.setToolTip('98.5 counts (sCMOS) / 575 counts (emICCD) from raw, rectified and registered integer cubes; '
                                     '_dc / _pc cubes are not touched')
        lv.addWidget(self.offset_check)
        self.warn_label = QtWidgets.QLabel('')
        self.warn_label.setWordWrap(True)
        theme.set_tone(self.warn_label, 'warn')
        lv.addWidget(self.warn_label)
        self.info_tree = KeyValueTree()
        self.info_tree.setMaximumHeight(220)
        lv.addWidget(QtWidgets.QLabel('Metadata of the selected cube'))
        lv.addWidget(self.info_tree)
        split.addWidget(left)

        # ---------------------------------------------------------------- middle: image
        mid = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(mid)
        mv.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        self.img_combo = QtWidgets.QComboBox()
        self.img_combo.addItems(IMAGE_MODES)
        self.cmap_combo = ColormapCombo(default='inferno')
        self.auto_check = QtWidgets.QCheckBox('Auto levels')
        self.auto_check.setChecked(True)
        self.same_check = QtWidgets.QCheckBox('Same levels for all cubes')
        self.same_check.setChecked(True)
        self.same_check.setToolTip('Levels from all shown cubes together, so stepping through the list shows real brightness '
                                   'changes (colour overlay: one common scale)')
        self.fit_btn = QtWidgets.QPushButton('Fit')
        for w in (self.img_combo, self.cmap_combo, self.auto_check, self.same_check, self.fit_btn):
            row.addWidget(w)
        row.addStretch(1)
        mv.addLayout(row)
        imgrow = QtWidgets.QHBoxLayout()
        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.setMenuEnabled(False)
        self.plot.setLabel('bottom', 'x (\u00b5m, reference frame)')
        self.plot.setLabel('left', 'y (\u00b5m)')
        self.img = pg.ImageItem()
        self.img.setOpts(axisOrder='row-major')
        self.img_rgb = pg.ImageItem()
        self.img_rgb.setOpts(axisOrder='row-major')
        self.img_rgb.setVisible(False)
        self.plot.addItem(self.img)
        self.plot.addItem(self.img_rgb)
        self.hist = pg.HistogramLUTWidget()
        self.hist.setMinimumWidth(110)
        self.hist.setMaximumWidth(140)
        self.hist.setImageItem(self.img)
        self.hist.item.gradient.setColorMap(self.cmap_combo.colormap())
        pen = pg.mkPen('#ffffff', width=1, style=QtCore.Qt.PenStyle.DashLine)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        for ln in (self.vline, self.hline):
            ln.setZValue(50)
            ln.setVisible(False)
            self.plot.addItem(ln, ignoreBounds=True)
        imgrow.addWidget(self.glw, 1)
        imgrow.addWidget(self.hist)
        mv.addLayout(imgrow, 1)
        self.img_label = QtWidgets.QLabel(' ')
        mv.addWidget(self.img_label)
        self.readout = QtWidgets.QLabel(' ')
        theme.set_mono(self.readout)
        mv.addWidget(self.readout)
        row = QtWidgets.QHBoxLayout()
        self.lam_combo = QtWidgets.QComboBox()
        self.lam_combo.addItems(LAMBDA_MODES)
        self.wl_spin = make_spinbox(750.0, 0.0, 3000.0, 1.0, 1, 'nm', 'Wavelength of the image (each cube: its nearest frame)')
        self.lo_spin = make_spinbox(670.0, 0.0, 3000.0, 1.0, 1, 'nm', 'Band / metric range: lower end')
        self.hi_spin = make_spinbox(850.0, 0.0, 3000.0, 1.0, 1, 'nm', 'Band / metric range: upper end')
        self.prev_btn = QtWidgets.QPushButton('\u25c0')
        self.next_btn = QtWidgets.QPushButton('\u25b6')
        for b in (self.prev_btn, self.next_btn):
            b.setFixedWidth(34)
            theme.set_compact(b)
        self.prev_btn.setToolTip('Previous cube (PageUp)')
        self.next_btn.setToolTip('Next cube (PageDown)')
        self.cycle_check = QtWidgets.QCheckBox('Cycle')
        self.cycle_check.setToolTip('Step through the shown cubes every %.1f s' % (CYCLE_MS / 1000))
        for w in (self.lam_combo, self.wl_spin, QtWidgets.QLabel('  range'), self.lo_spin, QtWidgets.QLabel('\u2013'),
                  self.hi_spin, self.prev_btn, self.next_btn, self.cycle_check):
            row.addWidget(w)
        row.addStretch(1)
        mv.addLayout(row)
        split.addWidget(mid)

        # ---------------------------------------------------------------- right: spectra + metric
        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        top = QtWidgets.QWidget()
        tv = QtWidgets.QVBoxLayout(top)
        tv.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        self.bin_combo = QtWidgets.QComboBox()
        for b in (1, 3, 5, 9, 15):
            self.bin_combo.addItem('%d\u00d7%d px' % (b, b), b)
        self.bin_combo.setCurrentIndex(1)
        self.bin_combo.setToolTip('Binning of the clicked-pixel spectrum (pixels of each cube)')
        self.roi_btn = QtWidgets.QPushButton('Add ROI')
        self.roi_btn.setToolTip('Add a rectangle (at most %d); drag / resize it on the image' % MAX_ROIS)
        self.clear_roi_btn = QtWidgets.QPushButton('Clear ROIs')
        self.full_check = QtWidgets.QCheckBox('full field')
        self.full_check.setChecked(True)
        self.norm_combo = QtWidgets.QComboBox()
        self.norm_combo.addItems(['no normalisation', 'peak = 1', 'range mean = 1'])
        self.log_check = QtWidgets.QCheckBox('log y')
        for w in (self.bin_combo, self.roi_btn, self.clear_roi_btn, self.full_check, self.norm_combo, self.log_check):
            row.addWidget(w)
        row.addStretch(1)
        tv.addLayout(row)
        self.spec_plot = pg.PlotWidget()
        self.spec_plot.showGrid(x=True, y=True, alpha=0.25)
        self.legend = self.spec_plot.addLegend(offset=(-10, 10))
        self.spec_plot.setLabel('bottom', 'wavelength (nm)')
        for ax in ('bottom', 'left'):
            self.spec_plot.getAxis(ax).enableAutoSIPrefix(False)
        self.band_region = pg.LinearRegionItem([670, 850], brush=pg.mkBrush(255, 215, 64, 30), movable=True)
        self.band_region.setZValue(-5)
        self.band_region.setToolTip('band of the image (band mode) and range of the metrics / range normalisation')
        self.wl_line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen('#ffd740', width=1.5))
        self.wl_line.setZValue(20)
        self.spec_plot.addItem(self.band_region, ignoreBounds=True)
        self.spec_plot.addItem(self.wl_line, ignoreBounds=True)
        tv.addWidget(self.spec_plot, 1)
        note = QtWidgets.QLabel(STYLE_NOTE)
        theme.set_caption(note)
        tv.addWidget(note)
        right.addWidget(top)
        bot = QtWidgets.QWidget()
        bv = QtWidgets.QVBoxLayout(bot)
        bv.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        self.metric_combo = QtWidgets.QComboBox()
        self.metric_combo.addItems(OC.METRICS)
        self.metric_combo.setToolTip('computed on the un-normalised spectra inside the range (yellow band)')
        self.xaxis_combo = QtWidgets.QComboBox()
        self.xaxis_combo.addItems(['vs cube order', 'vs start time (min)'])
        for w in (QtWidgets.QLabel('Metric'), self.metric_combo, self.xaxis_combo):
            row.addWidget(w)
        row.addStretch(1)
        bv.addLayout(row)
        self.metric_plot = pg.PlotWidget()
        self.metric_plot.showGrid(x=True, y=True, alpha=0.25)
        for ax in ('bottom', 'left'):
            self.metric_plot.getAxis(ax).enableAutoSIPrefix(False)
        bv.addWidget(self.metric_plot, 1)
        right.addWidget(bot)
        right.setSizes([560, 300])
        split.addWidget(right)
        split.setSizes([430, 640, 620])

        theme.style_plot(self.glw, 'image')
        theme.style_plot(self.hist, 'image')
        theme.style_plot(self.spec_plot, 'plot')
        theme.style_plot(self.metric_plot, 'plot')

        # ---------------------------------------------------------------- menu / status
        menu = self.menuBar().addMenu('&File')
        act = menu.addAction('Add cubes\u2026', self.add_dialog)
        act.setShortcut(QtGui.QKeySequence('Ctrl+O'))
        menu.addSeparator()
        menu.addAction('Export spectra as CSV\u2026', lambda: self._save_dialog('spectra', 'CSV (*.csv)', self.export_spectra_csv))
        menu.addAction('Export metrics as CSV\u2026', lambda: self._save_dialog('metrics', 'CSV (*.csv)', self.export_metrics_csv))
        menu.addAction('Save window as PNG\u2026', lambda: self._save_dialog('window', 'PNG (*.png)', self.export_window_png))
        menu.addSeparator()
        menu.addAction('Close', self.close)
        self.status = self.statusBar()

        self.cycle_timer = QtCore.QTimer(self)
        self.cycle_timer.setInterval(CYCLE_MS)
        self.spec_timer = QtCore.QTimer(self)
        self.spec_timer.setSingleShot(True)
        self.spec_timer.setInterval(60)
        for seq, fn in (('PgUp', lambda: self.step_cube(-1)), ('PgDown', lambda: self.step_cube(1))):
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.activated.connect(fn)
        sc = QtGui.QShortcut(QtGui.QKeySequence('Delete'), self.table)       # only in the cube list (removes from the list,
        sc.setContext(QtCore.Qt.ShortcutContext.WidgetShortcut)              # never from the disk)
        sc.activated.connect(self.remove_selected)

        # ---------------------------------------------------------------- wiring
        self.add_btn.clicked.connect(self.add_dialog)
        self.remove_btn.clicked.connect(self.remove_selected)
        self.ref_btn.clicked.connect(self.set_reference_selected)
        self.align_btn.clicked.connect(self.align_to_reference)
        self.zero_btn.clicked.connect(self.zero_shifts)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        self.scheme_combo.currentIndexChanged.connect(lambda _i: self._recolor(force=True))
        self.offset_check.toggled.connect(self._on_offset)
        self.img_combo.currentIndexChanged.connect(lambda _i: self.refresh_image())
        self.cmap_combo.colormapChanged.connect(lambda _n: self.refresh_image())
        self.auto_check.toggled.connect(lambda on: on and self.refresh_image())
        self.hist.item.sigLevelChangeFinished.connect(self._on_user_levels)
        self.same_check.toggled.connect(lambda _on: self.refresh_image())
        self.fit_btn.clicked.connect(self.fit_view)
        self.lam_combo.currentIndexChanged.connect(self._on_lam_mode)
        self.wl_spin.valueChanged.connect(self._on_wl_spin)
        self.lo_spin.valueChanged.connect(self._on_band_spin)
        self.hi_spin.valueChanged.connect(self._on_band_spin)
        self.band_region.sigRegionChangeFinished.connect(self._on_band_region)
        self.wl_line.sigPositionChangeFinished.connect(self._on_wl_line)
        self.prev_btn.clicked.connect(lambda: self.step_cube(-1))
        self.next_btn.clicked.connect(lambda: self.step_cube(1))
        self.cycle_check.toggled.connect(lambda on: self.cycle_timer.start() if on else self.cycle_timer.stop())
        self.cycle_timer.timeout.connect(lambda: self.step_cube(1))
        self.bin_combo.currentIndexChanged.connect(lambda _i: self.schedule_spectra())
        self.roi_btn.clicked.connect(lambda: self.add_roi())
        self.clear_roi_btn.clicked.connect(self.clear_rois)
        self.full_check.toggled.connect(lambda _on: self.schedule_spectra())
        self.norm_combo.currentIndexChanged.connect(lambda _i: self.schedule_spectra())
        self.log_check.toggled.connect(lambda _on: self.schedule_spectra())
        self.metric_combo.currentIndexChanged.connect(lambda _i: self.refresh_metric_plot())
        self.xaxis_combo.currentIndexChanged.connect(lambda _i: self.refresh_metric_plot())
        self.spec_timer.timeout.connect(self.refresh_spectra)
        self.plot.scene().sigMouseClicked.connect(self._on_scene_click)
        self.plot.scene().sigMouseMoved.connect(self._on_scene_move)

    def _set_enabled(self, on):
        for w in (self.remove_btn, self.ref_btn, self.align_btn, self.zero_btn, self.img_combo, self.cmap_combo,
                  self.auto_check, self.same_check, self.fit_btn, self.lam_combo, self.wl_spin, self.lo_spin, self.hi_spin,
                  self.prev_btn, self.next_btn, self.cycle_check, self.bin_combo, self.roi_btn, self.clear_roi_btn,
                  self.full_check, self.norm_combo, self.log_check, self.metric_combo, self.xaxis_combo, self.offset_check):
            w.setEnabled(on)
        if not on:
            self.img_label.setText('Add cubes with Ctrl+O or drop .h5 files onto the window')

    # ================================================================================================== loading
    def add_dialog(self):
        start = os.path.dirname(self.entries[-1].cf.path) if self.entries else ''
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Add cubes', start, 'HDF5 cubes (*.h5 *.hdf5);;All files (*)')
        if paths:
            self.add_paths(sorted(paths, key=OC.natural_key))

    def add_paths(self, items):
        """Queue files / glob patterns for loading (already listed or queued files are skipped)."""
        have = {os.path.normcase(e.cf.path) for e in self.entries} | {os.path.normcase(p) for p in self._queue}
        new = [p for p in OC.expand_paths(items) if os.path.normcase(p) not in have]
        if not new:
            self.status.showMessage('nothing new to add', 5000)
            return 0
        self._queue.extend(new)
        self._start_loader()
        return len(new)

    def _used_bytes(self):
        return sum(e.cf.ram_bytes() for e in self.entries if e.cf.in_ram)

    def _start_loader(self):
        if self._loader is not None or not self._queue:
            return
        paths, self._queue = self._queue, []
        th = _LoadThread(paths, max(0, self._budget - self._used_bytes()), self)
        th.loaded.connect(self._on_loaded)
        th.failed.connect(self._on_failed)
        th.progress.connect(lambda text, i, n: self.status.showMessage('loading ' + text))
        th.finished.connect(self._on_loader_finished)
        self._loader = th
        th.start()

    @property
    def is_loading(self):
        return self._loader is not None or bool(self._queue)

    def wait_loaded(self, timeout_s=300.0):
        """Run the event loop until every queued cube is loaded (scripts / tests).  True when nothing is pending."""
        import time
        t_end = time.monotonic() + timeout_s
        while self.is_loading and time.monotonic() < t_end:
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
        QtWidgets.QApplication.processEvents()
        return not self.is_loading

    def _on_loader_finished(self):
        th, self._loader = self._loader, None
        if th is not None:
            th.deleteLater()
        if self._queue:
            self._start_loader()
        else:
            n = len(self.entries)
            msg = '%d cube(s) loaded' % n
            if self._failed:
                msg += ';  NOT loaded: ' + ';  '.join('%s (%s)' % (os.path.basename(p), m) for p, m in self._failed[-3:])
            self.status.showMessage(msg, 0 if self._failed else 8000)

    def _on_failed(self, path, msg):
        self._failed.append((path, msg))
        self.status.showMessage('NOT loaded: %s (%s)' % (os.path.basename(path), msg))

    def _on_loaded(self, path, payload):
        cf, storage, t0, t_src, gain, acc, ga_src = payload
        self._uid += 1
        e = CubeEntry(self._uid, cf, storage, t0, t_src, gain, acc, ga_src)
        if self.offset_check.isChecked():
            cf.offset = cf.offset_guess()
        first = not self.entries
        self.entries.append(e)
        if first:
            self.ref_uid = e.uid
        self._recolor()
        self._rebuild_table(select_uid=e.uid if first else None)
        self._set_enabled(True)
        if first:
            self._init_ranges()
            self.fit_view()
        else:
            self._extend_ranges()
        self._update_warnings()
        self.refresh_image()
        self.schedule_spectra()

    # ================================================================================================== list
    def entry(self, uid):
        return next((e for e in self.entries if e.uid == uid), None)

    @property
    def reference(self):
        return self.entry(self.ref_uid)

    @property
    def selected(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if rows and 0 <= rows[0].row() < len(self.entries):
            return self.entries[rows[0].row()]
        return self.reference

    def shown(self):
        return [e for e in self.entries if e.shown]

    def _recolor(self, force=False):
        """Colours by the scheme.  Hand-picked colours stay unless `force` (the scheme was changed)."""
        if force:
            for e in self.entries:
                e.custom_color = False
                e.color = None
        seq = self.scheme_combo.currentIndex() == 1 or len(self.entries) > len(OC.DISTINCT)
        if seq:
            for e, c in zip(self.entries, OC.sequence_colors(len(self.entries))):
                if not e.custom_color:
                    e.color = c
        else:
            for e in self.entries:
                if not e.custom_color and (e.color is None or e.color not in OC.DISTINCT):
                    e.color = None
            for e in self.entries:
                if e.color is None:
                    e.color = OC.next_distinct([x.color for x in self.entries if x is not e and x.color]) or '#BBBBBB'
        if force:
            self._rebuild_table()
            self.refresh_image()
            self.schedule_spectra()

    def _rebuild_table(self, select_uid=None):
        cur = select_uid if select_uid is not None else (self.selected.uid if self.entries and self.selected else None)
        self._guard = True
        try:
            self.table.setRowCount(len(self.entries))
            t_first = min((e.t0 for e in self.entries), default=None)
            for r, e in enumerate(self.entries):
                cf = e.cf
                show = QtWidgets.QTableWidgetItem(_swatch(e.color, e.shown), '')
                show.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                              | QtCore.Qt.ItemFlag.ItemIsSelectable)
                show.setCheckState(QtCore.Qt.CheckState.Checked if e.shown else QtCore.Qt.CheckState.Unchecked)
                name = QtWidgets.QTableWidgetItem(cf.name + ('  (ref)' if e.uid == self.ref_uid else ''))
                f = name.font()
                f.setBold(e.uid == self.ref_uid)
                name.setFont(f)
                name.setToolTip('%s\n%s, %d frames, %d x %d px of %.4g um, %s' % (cf.path, cf.kind, cf.shape[0], cf.shape[2],
                                                                                  cf.shape[1], cf.pixel_size_um, e.storage))
                dt = (e.t0 - t_first).total_seconds() / 60.0 if t_first is not None else 0.0
                tm = QtWidgets.QTableWidgetItem('%s  (+%.1f min)' % (e.t0.strftime('%H:%M:%S'), dt))
                tm.setToolTip('from ' + e.t_src)
                wl = QtWidgets.QTableWidgetItem('%.0f\u2013%.0f /%d' % (np.nanmin(cf.wavelengths), np.nanmax(cf.wavelengths),
                                                                        cf.shape[0]))
                un = QtWidgets.QTableWidgetItem(cf.units)
                ga = QtWidgets.QTableWidgetItem('%g\u00d7%g' % (e.gain, e.acc) if e.gain else '\u2013')
                if e.ga_src:
                    ga.setToolTip('from the ' + e.ga_src)
                items = [show, name, tm, wl, un, ga]
                for k, it in enumerate(items):
                    if k:
                        it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
                    self.table.setItem(r, k, it)
                for col, v in ((COL_DX, e.shift[0]), (COL_DY, e.shift[1])):
                    it = QtWidgets.QTableWidgetItem('%.3f' % v)
                    it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
                                | QtCore.Qt.ItemFlag.ItemIsEditable)
                    if e.uid == self.ref_uid:
                        it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
                    self.table.setItem(r, col, it)
            if cur is not None:
                row = next((i for i, e in enumerate(self.entries) if e.uid == cur), None)
                if row is not None:
                    self.table.selectRow(row)
        finally:
            self._guard = False
        sel = self.selected
        if sel is not None:
            self.info_tree.set_data(sel.cf.describe())

    def _on_item_changed(self, item):
        if self._guard:
            return
        r, c = item.row(), item.column()
        if not 0 <= r < len(self.entries):
            return
        e = self.entries[r]
        if c == COL_SHOW:
            e.shown = item.checkState() == QtCore.Qt.CheckState.Checked
            self._guard = True
            item.setIcon(_swatch(e.color, e.shown))
            self._guard = False
            self._update_warnings()
            self.refresh_image()
            self.schedule_spectra()
        elif c in (COL_DX, COL_DY):
            try:
                v = float(item.text().replace(',', '.'))
            except ValueError:
                v = e.shift[c - COL_DX]
            e.shift[c - COL_DX] = v
            self._guard = True
            item.setText('%.3f' % v)
            self._guard = False
            self.refresh_image()
            self.schedule_spectra()

    def _on_double_click(self, r, c):
        if c != COL_SHOW or not 0 <= r < len(self.entries):
            return
        e = self.entries[r]
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor(e.color), self, 'Colour of %s' % e.name)
        if col.isValid():
            e.color = col.name().upper()
            e.custom_color = True
            self._rebuild_table()
            self.refresh_image()
            self.schedule_spectra()

    def _on_selection(self):
        if self._guard:
            return
        sel = self.selected
        if sel is not None:
            self.info_tree.set_data(sel.cf.describe())
        self.refresh_image()
        self.schedule_spectra()

    def select_uid(self, uid):
        row = next((i for i, e in enumerate(self.entries) if e.uid == uid), None)
        if row is not None:
            self.table.selectRow(row)

    def step_cube(self, d):
        """Select the next / previous SHOWN cube (wraps around)."""
        shown = self.shown()
        if not shown:
            return
        cur = self.selected
        idx = next((i for i, e in enumerate(shown) if cur is not None and e.uid == cur.uid), -1)
        self.select_uid(shown[(idx + d) % len(shown)].uid)

    def set_reference_selected(self):
        sel = self.selected
        if sel is None or sel.uid == self.ref_uid:
            return
        # keep every cube where it is in space: a point at x (old frame) is at x + s_new in the new reference, and cube k,
        # which read it at x + s_k, now reads the new-frame point x' at x' + (s_k - s_new)
        sx, sy = sel.shift
        for e in self.entries:
            e.shift = [e.shift[0] - sx, e.shift[1] - sy]
        if self._pix is not None:
            self._pix = (self._pix[0] + sx, self._pix[1] + sy)
        for r in self._rois:
            p = r['roi'].pos()
            r['roi'].setPos((p.x() + sx, p.y() + sy), finish=False)
        self.ref_uid = sel.uid
        self._rebuild_table()
        self._update_crosshair()
        self.refresh_image()
        self.schedule_spectra()

    def remove_selected(self):
        sel = self.selected
        if sel is None:
            return
        self.entries = [e for e in self.entries if e.uid != sel.uid]
        sel.cf.release()
        self._img_cache = OrderedDict((k, v) for k, v in self._img_cache.items() if k[0] != sel.uid)
        if sel.uid == self.ref_uid:
            self.ref_uid = self.entries[0].uid if self.entries else None
        self._recolor()
        self._rebuild_table(select_uid=self.ref_uid)
        if not self.entries:
            self.clear_rois()
            self._pix = None
            self._update_crosshair()
            self.img.clear()
            self.img_rgb.clear()
            self._clear_curves()
            self._set_enabled(False)
            return
        self._update_warnings()
        self.refresh_image()
        self.schedule_spectra()

    def _update_warnings(self):
        warn = OC.comparability_warnings([e.meta() for e in self.shown()])
        lazy = [e.name for e in self.entries if not e.cf.in_ram]
        if lazy:
            warn.append('read from disk (RAM budget used up): %s' % ', '.join(lazy))
        self.warn_label.setText('\n'.join('\u26a0 ' + w for w in warn))

    def _on_offset(self, on):
        for e in self.entries:
            e.cf.offset = e.cf.offset_guess() if on else 0.0
        self._img_cache.clear()
        self.refresh_image()
        self.schedule_spectra()

    def align_to_reference(self):
        """Phase-correlation shifts of every shown cube against the reference (integrated images, same grid only)."""
        ref = self.reference
        if ref is None:
            return []
        out = []
        with busy():
            a = ref.cf.integrated()
            for e in self.shown():
                if e.uid == ref.uid:
                    continue
                if e.cf.shape[1:] != ref.cf.shape[1:] or abs(e.cf.pixel_size_um - ref.cf.pixel_size_um) > 1e-9:
                    out.append('%s: other grid, not aligned' % e.name)
                    continue
                dy, dx = OC.phase_shift(a, e.cf.integrated())
                px = e.cf.pixel_size_um
                e.shift = [dx * px, dy * px]
                out.append('%s: dx %+.2f px, dy %+.2f px' % (e.name, dx, dy))
        self._rebuild_table()
        self.refresh_image()
        self.schedule_spectra()
        self.status.showMessage('aligned: ' + ';  '.join(out), 0)
        return out

    def zero_shifts(self):
        for e in self.entries:
            e.shift = [0.0, 0.0]
        self._rebuild_table()
        self.refresh_image()
        self.schedule_spectra()

    # ================================================================================================== wavelength controls
    def _union_range(self):
        lo = min(float(np.nanmin(e.cf.wavelengths)) for e in self.entries)
        hi = max(float(np.nanmax(e.cf.wavelengths)) for e in self.entries)
        return lo, hi

    def _init_ranges(self):
        lo, hi = self._union_range()
        ref = self.reference
        fs = ref.cf.full_spectrum()
        wl = float(ref.cf.wavelengths[int(np.nanargmax(fs))]) if np.isfinite(fs).any() else 0.5 * (lo + hi)
        step = float(np.median(np.abs(np.diff(np.unique(ref.cf.wavelengths))))) if ref.cf.shape[0] > 1 else 1.0
        self._guard = True
        try:
            for sp in (self.wl_spin, self.lo_spin, self.hi_spin):
                sp.setRange(lo, hi)
                sp.setSingleStep(step if step > 0 else 1.0)
            self.wl_spin.setValue(wl)
            self.lo_spin.setValue(lo)
            self.hi_spin.setValue(hi)
        finally:
            self._guard = False
        self._sync_decorations()

    def _extend_ranges(self):
        lo, hi = self._union_range()
        self._guard = True
        try:
            for sp in (self.wl_spin, self.lo_spin, self.hi_spin):
                sp.setRange(min(lo, sp.minimum()), max(hi, sp.maximum()))
        finally:
            self._guard = False

    @property
    def band(self):
        a, b = self.lo_spin.value(), self.hi_spin.value()
        return (a, b) if a <= b else (b, a)

    def set_band(self, lo, hi):
        self._guard = True
        try:
            self.lo_spin.setValue(float(lo))
            self.hi_spin.setValue(float(hi))
        finally:
            self._guard = False
        self._on_band_spin()

    def set_wavelength(self, wl):
        self.wl_spin.setValue(float(wl))

    def set_lambda_mode(self, mode):
        self.lam_combo.setCurrentIndex(LAMBDA_MODES.index(mode) if isinstance(mode, str) else int(mode))

    def set_image_mode(self, mode):
        self.img_combo.setCurrentIndex(IMAGE_MODES.index(mode) if isinstance(mode, str) else int(mode))

    def _sync_decorations(self):
        self._guard = True
        try:
            self.band_region.setRegion(self.band)
            self.wl_line.setValue(self.wl_spin.value())
            single = self.lam_combo.currentIndex() == 0
            self.wl_line.setVisible(single)
            self.wl_spin.setEnabled(single and bool(self.entries))
        finally:
            self._guard = False

    def _on_lam_mode(self, _i):
        self._sync_decorations()
        self.refresh_image()

    def _on_wl_spin(self, _v):
        if self._guard:
            return
        self._sync_decorations()
        if self.lam_combo.currentIndex() == 0:
            self.refresh_image()

    def _on_band_spin(self, *_):
        if self._guard:
            return
        self._sync_decorations()
        if self.lam_combo.currentIndex() == 1:
            self.refresh_image()
        self.schedule_spectra()

    def _on_band_region(self):
        if self._guard:
            return
        lo, hi = self.band_region.getRegion()
        self.set_band(lo, hi)

    def _on_wl_line(self):
        if self._guard:
            return
        self.wl_spin.setValue(float(self.wl_line.value()))

    # ================================================================================================== image
    def _image(self, e):
        """Single-λ or band image of one cube (cached), None when the cube has no data at that wavelength / band."""
        single = self.lam_combo.currentIndex() == 0
        sel = ('single', round(self.wl_spin.value(), 3)) if single else ('band', tuple(round(v, 3) for v in self.band))
        key = (e.uid, sel, float(e.cf.offset))
        if key in self._img_cache:
            self._img_cache.move_to_end(key)
            return self._img_cache[key]
        img = OC.cube_image(e.cf, sel[0], wl=self.wl_spin.value(), band=self.band)
        self._img_cache[key] = img
        while len(self._img_cache) > IMG_CACHE:
            self._img_cache.popitem(last=False)
        return img

    def _same_grid(self, cubes):
        ref = self.reference
        return all(c.cf.shape[1:] == ref.cf.shape[1:] and abs(c.cf.pixel_size_um - ref.cf.pixel_size_um) < 1e-9 for c in cubes)

    def _what(self):
        if self.lam_combo.currentIndex() == 0:
            return '\u03bb %.1f nm' % self.wl_spin.value()
        return 'band %.1f\u2013%.1f nm' % self.band

    def refresh_image(self):
        if not self.entries or self.reference is None:
            return
        mode = self.img_combo.currentIndex()
        ref, sel = self.reference, self.selected
        with busy():
            if mode == IMG_OVERLAY:
                self._show_overlay(ref)
            elif mode in (IMG_DIFF, IMG_REL):
                self._show_change(ref, sel, 'difference' if mode == IMG_DIFF else 'relative')
            else:
                self._show_cube(sel)

    def _place(self, item, e, shifted=True):
        cf = e.cf
        h, w = cf.shape[1:]
        px = cf.pixel_size_um
        sx, sy = e.shift if shifted else (0.0, 0.0)
        item.setRect(QtCore.QRectF(-sx, -sy, w * px, h * px))

    def _show_scalar(self, img, cmap, levels):
        self.img_rgb.setVisible(False)
        self.img.setVisible(True)
        self.hist.setEnabled(True)
        self.hist.item.gradient.setColorMap(cmap)
        self.img.setImage(img, autoLevels=False)
        if self.auto_check.isChecked():
            lo, hi = levels if levels is not None else auto_levels(img)
            span = hi - lo
            self._setting_levels = True
            try:
                self.hist.item.setHistogramRange(lo - 0.5 * span, hi + 0.5 * span, padding=0)
                self.hist.item.setLevels(lo, hi)
            finally:
                self._setting_levels = False
        self.img.setLevels(tuple(self.hist.item.getLevels()))
        self._shown_img = img

    def _on_user_levels(self):
        """Dragging the histogram handles switches 'Auto levels' off (the levels then stay while you step through cubes)."""
        if not self._setting_levels and self.auto_check.isChecked():
            self.auto_check.blockSignals(True)
            self.auto_check.setChecked(False)
            self.auto_check.blockSignals(False)

    def _show_cube(self, e, note=None):
        """The selected-cube image; `note` (why another mode fell back to this one) goes in front of the label."""
        pre = (note + '  —  showing ') if note else ''
        img = self._image(e)
        if img is None:
            self.img.clear()
            self._shown_img = None
            self.img_label.setText(pre + '%s has no frame at %s' % (e.name, self._what()))
            return
        levels = None
        if self.same_check.isChecked():
            imgs = [im for im in (self._image(x) for x in self.shown() + [e]) if im is not None]
            stacked = np.concatenate([im.ravel()[::max(1, im.size // 200_000)] for im in imgs])
            levels = auto_levels(stacked)
        self._show_scalar(img, self.cmap_combo.colormap(), levels)
        self._place(self.img, e)
        self.img_label.setText(pre + '%s  \u00b7  %s  \u00b7  %s  \u00b7  shift %+.2f, %+.2f \u00b5m'
                               % (e.name, self._what(), e.cf.units, e.shift[0], e.shift[1]))

    def _show_overlay(self, ref):
        cubes = self.shown()
        if not cubes:
            self.img_label.setText('no cube is shown (tick at least one)')
            return
        if not self._same_grid(cubes):
            self._show_cube(self.selected, note='colour overlay needs the same image size and pixel size for all shown cubes '
                                                '(untick the others)')
            return
        imgs, cols, names = [], [], []
        px = ref.cf.pixel_size_um
        for e in cubes:
            im = self._image(e)
            if im is None:
                continue
            imgs.append(OC.shift_into_reference(im, e.shift, px))
            cols.append(e.color)
            names.append(e.name)
        if not imgs:
            self.img_label.setText('no shown cube has data at ' + self._what())
            return
        rgb = OC.color_overlay(imgs, cols, common=self.same_check.isChecked())
        self.img.setVisible(False)
        self.img_rgb.setVisible(True)
        self.hist.setEnabled(False)
        self.img_rgb.setImage(rgb, autoLevels=False, levels=(0, 255))
        self._place(self.img_rgb, ref, shifted=False)
        self._shown_img = rgb
        self.img_label.setText('colour overlay of %d cube(s) at %s (%s scale)'
                               % (len(imgs), self._what(), 'common' if self.same_check.isChecked() else 'own'))

    def _show_change(self, ref, sel, kind):
        if sel is None or sel.uid == ref.uid:
            self._show_cube(ref, note='select a cube other than the reference (PageDown) for the %s image' % kind)
            return
        if not self._same_grid([sel]):
            self._show_cube(sel, note='%s image: %s needs the same image size and pixel size as the reference'
                                      % (kind, sel.name))
            return
        a, b = self._image(ref), self._image(sel)
        if a is None or b is None:
            self.img_label.setText('no data at %s in %s' % (self._what(), ref.name if a is None else sel.name))
            return
        px = ref.cf.pixel_size_um
        img = OC.change_image(OC.shift_into_reference(b, sel.shift, px), OC.shift_into_reference(a, ref.shift, px), kind)
        self._show_scalar(img, DIVERGING, OC.symmetric_levels(img))
        self._place(self.img, ref, shifted=False)
        unit = ref.cf.units if kind == 'difference' else '%'
        self.img_label.setText('%s %s %s at %s (%s)' % (sel.name, '\u2212' if kind == 'difference' else '/', ref.name,
                                                       self._what(), unit))

    def fit_view(self):
        ref = self.reference
        if ref is None:
            return
        h, w = ref.cf.shape[1:]
        px = ref.cf.pixel_size_um
        self.plot.setRange(xRange=(0, w * px), yRange=(0, h * px), padding=0.02)

    # ================================================================================================== pointer / ROIs
    def _on_scene_click(self, ev):
        if not self.entries or ev.button() != QtCore.Qt.MouseButton.LeftButton or ev.double():
            return
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = vb.mapSceneToView(ev.scenePos())
        self.select_point(p.x(), p.y())

    def select_point(self, x_um, y_um):
        """Clicked point in the reference frame (um): crosshair + pixel spectra of every shown cube."""
        self._pix = (float(x_um), float(y_um))
        self._update_crosshair()
        self.schedule_spectra()

    def _update_crosshair(self):
        on = self._pix is not None
        for ln in (self.vline, self.hline):
            ln.setVisible(on)
        if on:
            self.vline.setValue(self._pix[0])
            self.hline.setValue(self._pix[1])

    def _on_scene_move(self, pos):
        if not self.entries or self._shown_img is None:
            return
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            return
        p = vb.mapSceneToView(pos)
        parts = ['x %.2f \u00b5m  y %.2f \u00b5m' % (p.x(), p.y())]
        mode = self.img_combo.currentIndex()
        e = self.selected if mode == IMG_SELECTED else self.reference
        if e is not None:
            shift = e.shift if mode == IMG_SELECTED else (0.0, 0.0)
            rc = OC.to_pixel(p.x(), p.y(), shift, e.cf.pixel_size_um, e.cf.shape[1:])
            if rc is not None and self._shown_img.shape[:2] == e.cf.shape[1:]:
                v = self._shown_img[rc]
                parts.append('[%s row %d col %d]  %s' % (e.name, rc[0], rc[1],
                                                         ', '.join('%d' % x for x in v) if np.ndim(v) else '%.5g' % v))
        self.readout.setText('   '.join(parts))

    def add_roi(self, x0_um=None, y0_um=None, x1_um=None, y1_um=None):
        """Add a rectangle in the reference frame (um); default: 10 % of the reference side, centred in the view."""
        if not self.entries:
            raise RuntimeError('no cube loaded')
        if len(self._rois) >= MAX_ROIS:
            self.status.showMessage('at most %d ROIs' % MAX_ROIS, 5000)
            return None
        ref = self.reference
        h, w = ref.cf.shape[1:]
        px = ref.cf.pixel_size_um
        if x0_um is None:
            (vx0, vx1), (vy0, vy1) = self.plot.viewRange()
            side = max(2, int(0.1 * min(h, w))) * px
            cx, cy = 0.5 * (vx0 + vx1), 0.5 * (vy0 + vy1)
            x0_um, y0_um, x1_um, y1_um = cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2
        used = {r['label'] for r in self._rois}
        label = next(lb for lb in ('ROI 1', 'ROI 2', 'ROI 3') if lb not in used)
        pen = pg.mkPen(NEUTRAL, width=2, style=PROBE_STYLES[label])
        roi = pg.RectROI((min(x0_um, x1_um), min(y0_um, y1_um)), (abs(x1_um - x0_um), abs(y1_um - y0_um)), pen=pen)
        roi.setZValue(60)
        self.plot.addItem(roi)
        text = pg.TextItem(label, color=NEUTRAL, anchor=(0, 1))
        text.setParentItem(roi)
        self._rois.append({'label': label, 'roi': roi, 'text': text})
        self._rois.sort(key=lambda r: r['label'])
        roi.sigRegionChanged.connect(self.schedule_spectra)
        self.schedule_spectra()
        return label

    def clear_rois(self):
        for r in self._rois:
            self.plot.removeItem(r['roi'])
        self._rois = []
        self.schedule_spectra()

    def _probes(self):
        out = []
        if self._pix is not None:
            out.append(('pixel', ('pixel',) + self._pix))
        for r in self._rois:
            pos, size = r['roi'].pos(), r['roi'].size()
            out.append((r['label'], ('roi', pos.x(), pos.y(), pos.x() + size.x(), pos.y() + size.y())))
        if self.full_check.isChecked():
            out.append(('full field', ('full',)))
        return out

    # ================================================================================================== spectra / metrics
    def schedule_spectra(self, *_):
        if self.entries:
            self.spec_timer.start()

    def _clear_curves(self):
        for c in self._curves:
            self.spec_plot.removeItem(c)
        self._curves = []
        self.legend.clear()

    def current_spectra(self):
        """[(entry, probe label, x, y raw)] of the shown cubes x probes (y None when the probe lies outside that cube)."""
        out = []
        b = int(self.bin_combo.currentData() or 1)
        probes = self._probes()
        for e in self.shown():
            for label, probe in probes:
                y = OC.probe_spectrum(e.cf, probe, e.shift, b)
                out.append((e, label, np.asarray(e.cf.wavelengths, float), y))
        return out

    def refresh_spectra(self):
        self._clear_curves()
        self._metrics = {}
        if not self.entries:
            self.refresh_metric_plot()
            return
        norm = NORMS[self.norm_combo.currentIndex()]
        logy = self.log_check.isChecked()
        lo, hi = self.band
        sel = self.selected
        named = set()
        outside = []
        for e, label, x, y in self.current_spectra():
            if y is None:
                outside.append('%s (%s)' % (e.name, label))
                continue
            self._metrics[(e.uid, label)] = OC.spectral_metrics(x, y, lo, hi)
            yn = OC.normalise(x, y, norm, (lo, hi))
            if logy:
                yn = np.where(yn > 0, yn, np.nan)
            width = 3.0 if (sel is not None and e.uid == sel.uid) else 1.6
            pen = pg.mkPen(theme.plot_color(e.color), width=width, style=PROBE_STYLES[label])
            name = None
            if e.uid not in named:
                name = e.name
                named.add(e.uid)
            item = self.spec_plot.plot(x, yn, pen=pen, name=name, connect=OC.plot_connect(x, yn))
            self._curves.append(item)
        self.spec_plot.setLogMode(x=False, y=logy)
        units = {e.cf.units for e in self.shown()}
        if norm != 'none':
            self.spec_plot.setLabel('left', 'normalised (%s)' % ('peak' if norm == 'peak' else 'range mean'))
        else:
            self.spec_plot.setLabel('left', units.pop() if len(units) == 1 else 'mixed units (see list)')
        self.spec_plot.enableAutoRange()
        if outside:
            self.status.showMessage('outside the cube: ' + ', '.join(outside[:6]), 6000)
        if not self._probes():
            self.status.showMessage('click the image, add a ROI or tick "full field" to see spectra', 6000)
        self.refresh_metric_plot()

    def current_metrics(self):
        """[(entry, probe label, metrics dict)] in list order (shown cubes only)."""
        order = {e.uid: i for i, e in enumerate(self.entries)}
        out = [(self.entry(uid), label, m) for (uid, label), m in self._metrics.items() if self.entry(uid) is not None]
        return sorted(out, key=lambda t: (list(PROBE_STYLES).index(t[1]), order[t[0].uid]))

    def _metric_x(self, e):
        if self.xaxis_combo.currentIndex() == 1:
            t_first = min(x.t0 for x in self.entries)
            return (e.t0 - t_first).total_seconds() / 60.0
        return float(self.entries.index(e) + 1)

    def refresh_metric_plot(self):
        for it in self._metric_items:
            self.metric_plot.removeItem(it)
        self._metric_items = []
        key = OC.METRIC_KEYS[self.metric_combo.currentIndex()]
        rows = self.current_metrics()
        for label in PROBE_STYLES:
            pts = [(self._metric_x(e), m[key], e.color) for e, lb, m in rows if lb == label and np.isfinite(m[key])]
            if not pts:
                continue
            pts.sort(key=lambda p: p[0])
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            line = pg.PlotDataItem(xs, ys, pen=pg.mkPen(theme.plot_color(NEUTRAL), width=1.2, style=PROBE_STYLES[label]))
            sc = pg.ScatterPlotItem(xs, ys, size=9, pen=pg.mkPen(theme.plot_color(NEUTRAL), width=0.8),
                                    brush=[pg.mkBrush(theme.plot_color(p[2])) for p in pts])
            for it in (line, sc):
                self.metric_plot.addItem(it)
                self._metric_items.append(it)
        unit_y = OC.METRICS[self.metric_combo.currentIndex()]
        units = {e.cf.units for e in self.shown()}
        if key in ('integral', 'max'):
            unit_y += ' (%s%s)' % (units.pop() if len(units) == 1 else 'mixed units', ' \u00d7 nm' if key == 'integral' else '')
        self.metric_plot.setLabel('left', unit_y)
        self.metric_plot.setLabel('bottom', 'cube (list order)' if self.xaxis_combo.currentIndex() == 0
                                  else 'start time (min after the first cube)')
        self.metric_plot.enableAutoRange()

    # ================================================================================================== export
    def _save_dialog(self, what, filt, fn):
        if not self.entries:
            return
        start = os.path.join(os.path.dirname(self.reference.cf.path), 'overlay_%s%s' % (what, filt[-5:-1]))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save %s' % what, start, filt)
        if not path:
            return
        try:
            fn(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, 'Export failed', '%s: %s' % (type(exc).__name__, exc))
        else:
            self.status.showMessage('wrote ' + path, 8000)

    def _probe_comments(self, f):
        if self._pix is not None:
            f.write('# pixel: x %.3f um, y %.3f um (reference frame), %d x %d px binning\n'
                    % (self._pix + (self.bin_combo.currentData(),) * 2))
        for label, probe in self._probes():
            if probe[0] == 'roi':
                f.write('# %s: x %.3f-%.3f um, y %.3f-%.3f um (reference frame)\n' % (label, probe[1], probe[3], probe[2],
                                                                                       probe[4]))
        f.write('# reference: %s\n# range: %.2f-%.2f nm\n' % (self.reference.cf.path, self.band[0], self.band[1]))
        for e in self.shown():
            f.write('# cube %s: %s  shift %.4f, %.4f um  offset %g  start %s (%s)\n'
                    % (e.name, e.cf.path, e.shift[0], e.shift[1], e.cf.offset, e.t0.isoformat(sep=' '), e.t_src))

    def export_spectra_csv(self, path):
        """Long format: cube, probe, frame, wavelength_nm, value (raw, offset as set), value_normalised."""
        spectra = self.current_spectra()
        if not spectra:
            raise ValueError('nothing to export: click a pixel, add a ROI or tick "full field"')
        norm = NORMS[self.norm_combo.currentIndex()]
        with open(path, 'w', newline='', encoding='utf-8') as f:
            self._probe_comments(f)
            f.write('# normalisation of value_normalised: %s\n' % norm)
            wr = csv.writer(f)
            wr.writerow(['cube', 'probe', 'frame', 'wavelength_nm', 'value', 'value_normalised', 'units'])
            for e, label, x, y in spectra:
                if y is None:
                    continue
                yn = OC.normalise(x, y, norm, self.band)
                for i in range(len(x)):
                    wr.writerow([e.name, label, i, '%.4f' % x[i], '%.9g' % y[i], '%.9g' % yn[i], e.cf.units])

    def export_metrics_csv(self, path):
        rows = self.current_metrics()
        if not rows:
            raise ValueError('no metrics yet: click a pixel, add a ROI or tick "full field"')
        t_first = min(e.t0 for e in self.entries)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            self._probe_comments(f)
            wr = csv.writer(f)
            wr.writerow(['cube', 'order', 'start', 'minutes', 'probe'] + list(OC.METRIC_KEYS) + ['units'])
            for e, label, m in rows:
                wr.writerow([e.name, self.entries.index(e) + 1, e.t0.isoformat(sep=' '),
                             '%.3f' % ((e.t0 - t_first).total_seconds() / 60.0), label]
                            + ['%.9g' % m[k] for k in OC.METRIC_KEYS] + [e.cf.units])

    def export_window_png(self, path):
        if not self.grab().save(path):
            raise OSError('could not write %s' % path)

    # ================================================================================================== window
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        paths = [u.toLocalFile() for u in ev.mimeData().urls() if u.toLocalFile().lower().endswith(('.h5', '.hdf5'))]
        if paths:
            self.add_paths(sorted(paths, key=OC.natural_key))
            ev.acceptProposedAction()

    def closeEvent(self, ev):
        self.cycle_timer.stop()
        self._queue = []
        if self._loader is not None:
            self._loader.cancel()
            self._loader.wait(30000)
        for e in self.entries:
            e.cf.release()
        super().closeEvent(ev)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m hsapp.viewer.overlay_view',
                                 description='Overlay several hyperspectral cubes (read-only viewer).')
    ap.add_argument('files', nargs='*', help='cube files or glob patterns, e.g. "cube-*.h5"')
    ap.add_argument('--ram-gb', type=float, default=16.0, help='RAM for all cubes together (beyond: read from disk), default 16')
    ap.add_argument('--theme', choices=('dark', 'light', 'auto'), default='dark')
    opts = ap.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    theme.apply(app, opts.theme)
    win = OverlayViewer(ram_budget_gb=opts.ram_gb)
    win.resize(1750, 980)
    win.show()
    if opts.files:
        n = win.add_paths(opts.files)
        if not n:
            win.status.showMessage('no file matched: ' + ' '.join(opts.files), 0)
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
