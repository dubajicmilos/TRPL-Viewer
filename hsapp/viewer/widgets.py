"""Small shared widgets of the hsapp viewer (Qt only through pyqtgraph.Qt, never touches hardware).

    ColormapCombo     combo box of the viewer colormaps with gradient icons
    ScaleBar          scale bar for an image PlotItem, length in um (picks a 1-2-5 length itself)
    ProgressOverlay   translucent "loading ..." overlay with a progress bar covering a parent widget
    KeyValueTree      read-only tree for (nested) metadata dictionaries
    make_spinbox      QSpinBox / QDoubleSpinBox with unit suffix, range, step, tooltip
"""
import math

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from .cubefile import nest_flat

COLORMAP_NAMES = ('gray', 'inferno', 'viridis', 'magma', 'turbo')


def get_colormap(name):
    """pg.ColorMap for one of COLORMAP_NAMES.  'gray' is built here; inferno/viridis/magma ship with pyqtgraph;
    turbo needs matplotlib (ValueError if it is not importable)."""
    if name == 'gray':
        return pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])
    if name not in COLORMAP_NAMES:
        raise ValueError('unknown colormap %r (known: %s)' % (name, ', '.join(COLORMAP_NAMES)))
    try:
        if name == 'turbo':
            return pg.colormap.get('turbo', source='matplotlib')
        return pg.colormap.get(name)
    except Exception as exc:                                    # missing matplotlib / colormap file
        raise ValueError('colormap %r is not available: %s' % (name, exc)) from exc


def _available_colormaps():
    out = []
    for name in COLORMAP_NAMES:
        try:
            out.append((name, get_colormap(name)))
        except ValueError:
            continue                                            # turbo without matplotlib is simply not offered
    return out


class ColormapCombo(QtWidgets.QComboBox):
    """Combo box of colormap names with a gradient icon each.  `colormapChanged(str)` is emitted on change."""

    colormapChanged = QtCore.Signal(str)

    def __init__(self, parent=None, default='inferno'):
        super().__init__(parent)
        self._maps = dict(_available_colormaps())
        for name, cmap in self._maps.items():
            lut = cmap.getLookupTable(0.0, 1.0, 64, alpha=False)
            img = QtGui.QImage(64, 1, QtGui.QImage.Format.Format_RGB888)
            for i in range(64):
                img.setPixelColor(i, 0, QtGui.QColor(int(lut[i, 0]), int(lut[i, 1]), int(lut[i, 2])))
            self.addItem(QtGui.QIcon(QtGui.QPixmap.fromImage(img.scaled(48, 12))), name)
        self.setIconSize(QtCore.QSize(48, 12))
        if default in self._maps:
            self.setCurrentText(default)
        self.currentTextChanged.connect(self.colormapChanged)
        self.setToolTip('Colormap of the image')

    def colormap(self):
        """The currently selected pg.ColorMap."""
        return self._maps[self.currentText()]

    def set_colormap(self, name):
        """Select a colormap by name (ValueError if it is not offered)."""
        if name not in self._maps:
            raise ValueError('colormap %r not offered (offered: %s)' % (name, ', '.join(self._maps)))
        self.setCurrentText(name)


def nice_length(target):
    """Largest 1-2-5 x 10^k value <= target (target > 0)."""
    if not (target > 0 and math.isfinite(target)):
        raise ValueError('target length must be positive and finite, got %r' % (target,))
    exp = math.floor(math.log10(target))
    for m in (5.0, 2.0, 1.0):
        v = m * 10.0 ** exp
        if v <= target * (1 + 1e-9):
            return v
    return 10.0 ** exp


class ScaleBar(pg.ScaleBar):
    """Scale bar anchored at the bottom right of an image ViewBox/PlotItem, labelled in um.

    Call `set_fraction_of(width_um)` once the image width is known: the bar length becomes the largest 1-2-5 value
    not above 20 % of that width.  The bar scales with the view (zoom) like pg.ScaleBar."""

    def __init__(self, length_um=10.0, parent_item=None):
        super().__init__(size=length_um, width=4, brush=pg.mkBrush('w'), pen=pg.mkPen(None), suffix='um',
                         offset=(-14, -14))
        self.setZValue(100)
        self.set_length(length_um)
        if parent_item is not None:
            self.setParentItem(parent_item)

    def set_length(self, length_um):
        """Bar length in um (> 0); the label reads e.g. '20 µm' or '0.5 µm'."""
        if not (length_um > 0 and math.isfinite(length_um)):
            raise ValueError('scale bar length must be positive and finite, got %r' % (length_um,))
        self.size = float(length_um)
        self.text.setText('%g µm' % length_um)
        self.updateBar()

    def set_fraction_of(self, width_um, fraction=0.2):
        """Choose a nice length of at most `fraction` of the image width (um)."""
        self.set_length(nice_length(width_um * fraction))


class ProgressOverlay(QtWidgets.QWidget):
    """Translucent overlay with a message and a progress bar that covers its parent widget.

    `show_progress(text, done, total)` shows/updates it (total <= 0 gives a busy bar), `finish()` hides it."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet('ProgressOverlay { background-color: rgba(20, 20, 20, 170); }'
                           'QLabel { color: white; font-size: 12pt; }')
        lay = QtWidgets.QVBoxLayout(self)
        lay.addStretch(1)
        self._label = QtWidgets.QLabel('')
        self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._bar = QtWidgets.QProgressBar()
        self._bar.setMaximumWidth(360)
        self._bar.setTextVisible(True)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._bar)
        row.addStretch(1)
        lay.addWidget(self._label)
        lay.addLayout(row)
        lay.addStretch(1)
        parent.installEventFilter(self)
        self.hide()

    def eventFilter(self, obj, event):
        if obj is self.parent() and event.type() == QtCore.QEvent.Type.Resize:
            self.setGeometry(self.parent().rect())
        return False

    def show_progress(self, text, done=0, total=0):
        self._label.setText(text)
        if total > 0:
            self._bar.setRange(0, int(total))
            self._bar.setValue(int(min(done, total)))
        else:
            self._bar.setRange(0, 0)
        self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()

    def finish(self):
        self.hide()


def format_value(value):
    """Compact text of a metadata value (arrays are summarised, long text is cut at 160 characters)."""
    if isinstance(value, (np.ndarray, list, tuple)):
        arr = np.asarray(value)
        if arr.size > 8:
            text = '[%s ... %s] (%d values)' % (arr.flat[0], arr.flat[-1], arr.size)
        else:
            text = '[' + ', '.join(str(v) for v in arr.tolist()) + ']'
    elif isinstance(value, float):
        text = '%.6g' % value
    else:
        text = str(value)
    return text if len(text) <= 160 else text[:157] + '...'


class KeyValueTree(QtWidgets.QTreeWidget):
    """Read-only two-column tree (key | value) of a nested dict; nested dicts become expandable branches."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderLabels(['Key', 'Value'])
        self.setAlternatingRowColors(True)
        self.setUniformRowHeights(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.header().setStretchLastSection(True)

    def set_data(self, data, expand_level=1):
        """Replace the contents with `data` (nested dict); branches up to `expand_level` deep are expanded."""
        self.clear()
        self._fill(self.invisibleRootItem(), data)
        self.expandToDepth(expand_level - 1)

    def set_flat(self, flat, sep='/', expand_level=1):
        """Like set_data for a flat dict whose keys are 'a/b/c' paths."""
        self.set_data(nest_flat(flat, sep), expand_level)

    def _fill(self, parent, data):
        for key, value in data.items():
            if isinstance(value, dict):
                item = QtWidgets.QTreeWidgetItem(parent, [str(key), ''])
                self._fill(item, value)
            else:
                text = format_value(value)
                item = QtWidgets.QTreeWidgetItem(parent, [str(key), text])
                item.setToolTip(1, str(value) if len(str(value)) < 2000 else text)

    def as_dict(self):
        """The displayed tree as nested dict of strings (used by tests)."""
        def rec(item):
            if item.childCount():
                return {item.child(i).text(0): rec(item.child(i)) for i in range(item.childCount())}
            return item.text(1)
        return rec(self.invisibleRootItem())


def make_spinbox(value, minimum, maximum, step=1.0, decimals=0, suffix='', tooltip='', parent=None):
    """Spin box with a unit suffix.  decimals == 0 and integral limits give a QSpinBox, otherwise a QDoubleSpinBox.

    The suffix is shown with a leading space (' nm'); keyboard tracking is off so signals fire on commit only."""
    if decimals == 0 and float(minimum).is_integer() and float(maximum).is_integer() and float(step).is_integer():
        box = QtWidgets.QSpinBox(parent)
        box.setRange(int(minimum), int(maximum))
        box.setSingleStep(int(step))
        box.setValue(int(value))
    else:
        box = QtWidgets.QDoubleSpinBox(parent)
        box.setDecimals(int(decimals))
        box.setRange(float(minimum), float(maximum))
        box.setSingleStep(float(step))
        box.setValue(float(value))
    if suffix:
        box.setSuffix(suffix if suffix.startswith(' ') else ' ' + suffix)
    if tooltip:
        box.setToolTip(tooltip)
    box.setKeyboardTracking(False)
    return box
