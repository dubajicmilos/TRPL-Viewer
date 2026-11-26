"""hsapp.gui.theme - the ONE place that decides how hsapp looks (2026-11-11).

Public API (other modules rely on exactly these names)::

    apply(app, mode)             mode 'dark' | 'light' | 'auto': Fusion style, QPalette, ONE application style sheet built from the token
                                 dict, the application font and the pyqtgraph defaults.  Safe to call again to switch at run time (existing
                                 widgets are re-polished by Qt; widgets with an ``apply_theme()`` method, e.g. the viewers, restyle their
                                 pyqtgraph canvases; icons set with ``set_icon`` are re-tinted).
    current() -> 'dark' | 'light'   the variant that is active ('light' = the plain Fusion look while ``apply`` was never called)
    tokens(variant=None) -> dict    colour / size tokens (a copy) of the active (or the named) variant
    color(name, variant=None)       one token as '#rrggbb'
    set_role(widget, role)       role 'primary' | 'danger' | 'ghost' | None: dynamic property "role", re-polished
    set_tone(widget, tone)       tone 'muted' | 'safe' | 'warn' | 'danger' | 'sim' | 'accent' | None: dynamic property "tone", re-polished
    make_chip(text, tone=None) -> QLabel (property chip=true: a rounded status pill);  set_chip(label, text, tone)
    icon(name) -> QIcon          Qt standard pixmaps for 'warning' / 'info' / 'error', otherwise a small inline SVG rendered with QtSvg and
                                 tinted with the text colour; an unknown name returns an empty QIcon, never raises
    style_plot(widget, kind)     kind 'plot' | 'image': background / axis colours of a pyqtgraph widget; image canvases stay dark in BOTH
                                 variants, line plots follow the variant.  A no-op until ``apply`` was called (the tests keep pyqtgraph's look)
    TONES                        the one semantic colour map {tone: {'dark': hex, 'light': hex}}; STEP_COLORS, STATE_COLORS, FILTER_STYLES ...
                                 of the old modules are ``ToneMap`` aliases that read it

Additions (not required by the design, used by the migrated modules): ``set_solid`` (filled tone background), ``set_prop`` / ``set_mono`` /
``set_caption`` / ``set_heading`` (dynamic properties for the type scale), ``set_icon`` (icon that is re-tinted on a theme switch and takes the
text colour of a primary / danger button), ``rich(text, tone)`` and ``plot_color(hex)``, ``contrast_ratio``, ``reset(app)``.

The style sheet targets widget types and dynamic properties only (``QPushButton[role="primary"]``, ``QLabel[tone="warn"]``,
``QLabel[chip="true"]`` ...), never object names.  Nothing here touches hardware or the settings file.
"""
from __future__ import annotations

import os
import re
import tempfile
import weakref
from collections.abc import Mapping
from string import Template
from typing import Dict, Optional

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtSvg, QtWidgets

MODES = ('dark', 'light', 'auto')
ROLES = ('primary', 'danger', 'ghost')
TONE_NAMES = ('muted', 'safe', 'warn', 'danger', 'sim', 'accent')
BODY_FONT = 'Segoe UI'
MONO_FONT = 'Consolas'
BODY_PT = 9
CAPTION_PT = 8
TITLE_PT = 10
IMAGE_BG = '#0f1216'                   # image canvases (camera frames, maps) stay dark in both variants
IMAGE_AXES = '#9aa4b2'


# ---------------------------------------------------------------------------------------------------------------------
# colour helpers
# ---------------------------------------------------------------------------------------------------------------------
def _rgb(hexcolor: str):
    h = hexcolor.lstrip('#')
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(rgb) -> str:
    return '#%02x%02x%02x' % tuple(max(0, min(255, int(round(v)))) for v in rgb)


def blend(fg: str, bg: str, alpha: float) -> str:
    """``fg`` drawn over ``bg`` with opacity ``alpha`` -> '#rrggbb'."""
    f, b = _rgb(fg), _rgb(bg)
    return _hex([alpha * f[i] + (1.0 - alpha) * b[i] for i in range(3)])


def relative_luminance(hexcolor: str) -> float:
    def lin(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(v) for v in _rgb(hexcolor))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG contrast ratio of two '#rrggbb' colours (1 .. 21)."""
    la, lb = relative_luminance(fg), relative_luminance(bg)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


# ---------------------------------------------------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------------------------------------------------
_DARK = dict(window='#14171c', surface='#1b1f26', input='#232831', hover='#2b313c', border='#323a46', border_strong='#465062',
             text='#e6e9ef', muted='#9aa4b2', disabled='#5d6673',
             accent='#4f9cff', accent_hover='#6fb0ff', accent_pressed='#3b82e0', on_accent='#0b1220',
             safe='#3ecf8e', warn='#f5b942', danger='#ef5350', sim='#b388ff',
             on_safe='#0b1220', on_warn='#0b1220', on_danger='#0b1220', on_sim='#0b1220',
             plot_bg='#0f1216', plot_axes='#9aa4b2', tooltip_bg='#0b0e12', tooltip_text='#e6e9ef')
_LIGHT = dict(window='#f4f6f9', surface='#ffffff', input='#ffffff', hover='#eef2f7', border='#d7dde6', border_strong='#b8c2d0',
              text='#1a2230', muted='#5b6675', disabled='#a0a9b6',
              accent='#2563eb', accent_hover='#3b76f0', accent_pressed='#1d4fd0', on_accent='#ffffff',
              safe='#15803d', warn='#b45309', danger='#c62828', sim='#6d28d9',
              on_safe='#ffffff', on_warn='#ffffff', on_danger='#ffffff', on_sim='#ffffff',
              plot_bg='#ffffff', plot_axes='#334155', tooltip_bg='#1a2230', tooltip_text='#f4f6f9')


def _derive(t: dict, variant: str) -> dict:
    """Tokens computed from the base ones (blends: selection, tinted chip backgrounds, hover / pressed fills)."""
    t = dict(t)
    dark = variant == 'dark'
    t['selection'] = blend(t['accent'], t['input'], 0.35)                   # = the accent at about 35 % alpha over an input field
    t['progress_chunk'] = blend(t['accent'], t['input'], 0.55 if dark else 0.45)
    for tone in ('safe', 'warn', 'danger', 'sim', 'accent'):
        t[tone + '_bg'] = blend(t[tone], t['surface'], 0.20 if dark else 0.14)      # tinted pill background, the text stays t['text']
    # hover / pressed fills of a danger button: lighter in the dark variant (dark text on it), a little lighter / darker in the light one (white text)
    t['danger_hover'] = blend('#ffffff', t['danger'], 0.14 if dark else 0.08)
    t['danger_pressed'] = blend('#ffffff', t['danger'], 0.26) if dark else blend('#000000', t['danger'], 0.12)
    t['placeholder'] = blend(t['muted'], t['input'], 0.75)
    t['white'] = '#ffffff'
    t['variant'] = variant
    t['font_body'] = BODY_FONT
    t['font_mono'] = MONO_FONT
    t['pt_body'] = BODY_PT
    t['pt_caption'] = CAPTION_PT
    t['pt_title'] = TITLE_PT
    t['radius_input'] = 4
    t['radius_button'] = 5
    t['radius_card'] = 6
    t['button_min_height'] = 24
    t['grid'] = 4
    return t


TOKENS: Dict[str, dict] = {'dark': _derive(_DARK, 'dark'), 'light': _derive(_LIGHT, 'light')}

# the one semantic colour map: tone -> {variant: '#rrggbb'}
TONES: Dict[str, Dict[str, str]] = {
    tone: {v: TOKENS[v][tone] for v in ('dark', 'light')} for tone in ('muted', 'safe', 'warn', 'danger', 'sim', 'accent', 'disabled')}
TONES['neutral'] = {v: TOKENS[v]['text'] for v in ('dark', 'light')}

_variant = 'light'          # active variant; 'light' = the plain Fusion look until apply() is called
_mode = None                # the mode passed to the last apply()
_applied = False
_scheme_hooked = False


def current() -> str:
    return _variant


def is_applied() -> bool:
    return _applied


def tokens(variant: Optional[str] = None) -> dict:
    return dict(TOKENS[variant or _variant])


def color(name: str, variant: Optional[str] = None) -> str:
    """One token ('text', 'danger', 'plot_bg' ...) or tone as '#rrggbb' for the active (or the named) variant."""
    t = TOKENS[variant or _variant]
    if name in t:
        return t[name]
    if name in TONES:
        return TONES[name][variant or _variant]
    raise KeyError('unknown theme colour %r' % (name,))


class ToneMap(Mapping):
    """Read-only mapping key -> '#rrggbb' of the ACTIVE variant; the values come from the token table (the old per-module colour tables
    STEP_COLORS / STATE_COLORS are aliases of it).  ``mapping`` maps each key to a token or tone name."""

    def __init__(self, mapping: Dict[str, str]):
        self._m = dict(mapping)

    def __getitem__(self, key):
        return color(self._m[key])

    def __iter__(self):
        return iter(self._m)

    def __len__(self):
        return len(self._m)

    def tone_of(self, key) -> str:
        return self._m[key]


class QssMap(ToneMap):
    """Like ToneMap but the values are QSS strings 'color: #rrggbb; font-weight: bold;' (alias of the old FILTER_STYLES)."""

    def __init__(self, mapping: Dict[str, str], bold=()):
        super().__init__(mapping)
        self._bold = set(bold)

    def __getitem__(self, key):
        return 'color: %s;%s' % (color(self._m[key]), ' font-weight: bold;' if key in self._bold else '')


def rich(text: str, tone: str, bold: bool = False) -> str:
    """``text`` (already HTML-escaped) in the colour of ``tone`` for rich-text labels / the log; the colour is the one of the variant that is
    active when the text is built."""
    inner = '<b>%s</b>' % text if bold else text
    return '<span style="color:%s">%s</span>' % (color(tone), inner)


def plot_color(hexcolor: str) -> str:
    """A colour designed for a dark canvas, made legible on the active variant's line-plot background: unchanged in the dark variant; in the
    light variant colours that are too bright for white are darkened until they reach a contrast ratio of 3:1."""
    if _variant == 'dark' or not _applied:
        return hexcolor
    bg = TOKENS['light']['plot_bg']
    c = _rgb(hexcolor)
    k = 1.0
    while k > 0.15 and contrast_ratio(_hex([v * k for v in c]), bg) < 3.0:
        k -= 0.05
    return _hex([v * k for v in c])


# ---------------------------------------------------------------------------------------------------------------------
# dynamic properties
# ---------------------------------------------------------------------------------------------------------------------
def _repolish(widget):
    try:
        st = widget.style()
        st.unpolish(widget)
        st.polish(widget)
        widget.update()
    except RuntimeError:                      # the C++ object is gone
        pass


def set_prop(widget, name: str, value) -> None:
    """Set the dynamic property ``name`` (None / False -> removed) and re-polish when it changed (cheap when it did not)."""
    if widget is None:
        return
    if isinstance(value, bool):
        value = 'true' if value else None
    elif value == '':
        value = None
    old = widget.property(name)
    if old == '':
        old = None
    if old == value:
        return
    widget.setProperty(name, value)
    _repolish(widget)


def set_role(widget, role: Optional[str]) -> None:
    if role not in ROLES and role is not None:
        raise ValueError('role must be one of %s or None, got %r' % (ROLES, role))
    set_prop(widget, 'role', role)
    _retint(widget)


def set_tone(widget, tone: Optional[str]) -> None:
    if tone not in TONE_NAMES and tone is not None:
        raise ValueError('tone must be one of %s or None, got %r' % (TONE_NAMES, tone))
    set_prop(widget, 'tone', tone)


def set_solid(widget, on: bool = True) -> None:
    """Filled background in the colour of the tone (with the 'on' text colour); needs a tone."""
    set_prop(widget, 'solid', True if on else None)


def set_mono(widget, on: bool = True) -> None:
    set_prop(widget, 'mono', True if on else None)


def set_caption(widget, on: bool = True) -> None:
    set_prop(widget, 'caption', True if on else None)


def set_heading(widget, on: bool = True) -> None:
    set_prop(widget, 'heading', True if on else None)


def set_card(widget, on: bool = True) -> None:
    """A QFrame drawn as a flat card (surface colour, 1-px border, rounded): the TRPL step boxes and the emICCD live strip."""
    set_prop(widget, 'card', True if on else None)


def set_compact(widget, on: bool = True) -> None:
    """Narrow button (the '...' browse buttons, play, quick times): small horizontal padding so the text is not clipped."""
    set_prop(widget, 'compact', True if on else None)


def make_chip(text: str, tone: Optional[str] = None) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setProperty('chip', 'true')
    lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    if tone:
        set_tone(lab, tone)
    return lab


def set_chip(label: QtWidgets.QLabel, text: str, tone: Optional[str] = None) -> None:
    label.setText(text)
    if label.property('chip') != 'true':
        label.setProperty('chip', 'true')
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    set_tone(label, tone)


# ---------------------------------------------------------------------------------------------------------------------
# icons
# ---------------------------------------------------------------------------------------------------------------------
_STD = {'warning': QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning, 'info': QtWidgets.QStyle.StandardPixmap.SP_MessageBoxInformation,
        'error': QtWidgets.QStyle.StandardPixmap.SP_MessageBoxCritical}

_SVG = {
    'play': '<polygon points="7 4 20 12 7 20 7 4" fill="C"/>',
    'stop': '<rect x="6" y="6" width="12" height="12" rx="1.5" fill="C"/>',
    'hold': '<rect x="6" y="4" width="4" height="16" fill="C"/><rect x="14" y="4" width="4" height="16" fill="C"/>',
    'refresh': '<polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>',
    'open': '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>',
    'save': '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/>'
            '<polyline points="7 3 7 8 15 8"/>',
    'settings': '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/>'
                '<line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/>'
                '<line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    'snap': '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
    'live': '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    'probe': '<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="1.5" fill="C"/><line x1="12" y1="1" x2="12" y2="5"/>'
             '<line x1="12" y1="19" x2="12" y2="23"/><line x1="1" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="23" y2="12"/>',
    'crosshair': '<circle cx="12" cy="12" r="7"/><line x1="12" y1="1" x2="12" y2="7"/><line x1="12" y1="17" x2="12" y2="23"/>'
                 '<line x1="1" y1="12" x2="7" y2="12"/><line x1="17" y1="12" x2="23" y2="12"/>',
    'light': '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>'
             '<line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>'
             '<line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>'
             '<line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>',
    'shutter': '<circle cx="12" cy="12" r="10"/><line x1="14.31" y1="8" x2="20.05" y2="17.94"/><line x1="9.69" y1="8" x2="21.17" y2="8"/>'
               '<line x1="7.38" y1="12" x2="13.12" y2="2.06"/><line x1="9.69" y1="16" x2="3.95" y2="6.06"/>'
               '<line x1="14.31" y1="16" x2="2.83" y2="16"/><line x1="16.62" y1="12" x2="10.88" y2="21.94"/>',
    'laser': '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    'filter': '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    'zoom': '<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/>'
            '<line x1="3" y1="21" x2="10" y2="14"/>',
    'browse': '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    'trpl': '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
    'compress': '<polyline points="21 8 21 21 3 21 3 8"/><rect x="1" y="3" width="22" height="5"/><line x1="10" y1="12" x2="14" y2="12"/>',
    'data': '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>'
            '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    'processing': '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    'cube': '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>'
            '<polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>',
    'quit': '<path d="M18.36 6.64a9 9 0 1 1-12.73 0"/><line x1="12" y1="2" x2="12" y2="12"/>',
    'close': '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    'help': '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    'guide': '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    'read': '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>',
    'go': '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
    'abort': '<polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/>'
             '<line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
    'info': '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    'up': '<polyline points="18 15 12 9 6 15"/>',
    'down': '<polyline points="6 9 12 15 18 9"/>',
    'left': '<polyline points="15 18 9 12 15 6"/>',
    'right': '<polyline points="9 18 15 12 9 6"/>',
    'home': '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    'tools': '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    'sim': '<rect x="3" y="4" width="18" height="12" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="16" x2="12" y2="21"/>',
    'trpl4d': '<polyline points="2 4 7 12 13 16 22 18"/><polyline points="2 9 7 17 13 21 22 22"/><line x1="2" y1="2" x2="2" y2="22"/>',
}
_SVG_FRAME = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="C" stroke-width="2" stroke-linecap="round" '
              'stroke-linejoin="round">%s</svg>')
_ICON_SIZES = (16, 20, 24, 32, 48)
_icon_cache: Dict[tuple, QtGui.QIcon] = {}
_icon_users: list = []                       # [(weakref to widget / action, name, explicit colour)]


def icon_names():
    return sorted(set(_SVG) | set(_STD))


def _svg_pixmap(name: str, hexcolor: str, size: int) -> QtGui.QPixmap:
    svg = (_SVG_FRAME % _SVG[name]).replace('"C"', '"%s"' % hexcolor).encode('utf-8')
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg))
    pm = QtGui.QPixmap(size, size)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pm)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    return pm


def icon(name: str, color_name: Optional[str] = None) -> QtGui.QIcon:
    """Icon ``name`` (see ``icon_names()``) tinted with the text colour (disabled state: the disabled colour); ``color_name`` picks another
    token (for example 'on_accent' for a filled primary button).  Unknown name -> empty QIcon (never raises)."""
    try:
        if QtGui.QGuiApplication.instance() is None or not isinstance(name, str):
            return QtGui.QIcon()
        if name in _STD:
            app = QtWidgets.QApplication.instance()
            return app.style().standardIcon(_STD[name]) if app is not None else QtGui.QIcon()
        if name not in _SVG:
            return QtGui.QIcon()
        key = (name, _variant, color_name or 'text')
        ic = _icon_cache.get(key)
        if ic is None:
            ic = QtGui.QIcon()
            normal = color(color_name or 'text')
            dis = color('disabled')
            for s in _ICON_SIZES:
                ic.addPixmap(_svg_pixmap(name, normal, s), QtGui.QIcon.Mode.Normal)
                ic.addPixmap(_svg_pixmap(name, normal, s), QtGui.QIcon.Mode.Active)
                ic.addPixmap(_svg_pixmap(name, dis, s), QtGui.QIcon.Mode.Disabled)
            _icon_cache[key] = ic
        return ic
    except Exception:                                     # noqa: BLE001 - an icon must never break a window
        return QtGui.QIcon()


def _icon_colour_for(widget, explicit):
    if explicit:
        return explicit
    role = widget.property('role') if hasattr(widget, 'property') else None
    return {'primary': 'on_accent', 'danger': 'on_danger'}.get(role)


def set_icon(target, name: str, color_name: Optional[str] = None) -> None:
    """Set the icon ``name`` on a button / action / tool button; it is re-tinted when the theme or the role of the button changes.
    The texts are never touched."""
    try:
        target.setIcon(icon(name, _icon_colour_for(target, color_name)))
    except (RuntimeError, AttributeError):
        return
    for ref, nm, _c in _icon_users:
        if ref() is target:
            _icon_users.remove((ref, nm, _c))
            break
    try:
        _icon_users.append((weakref.ref(target), name, color_name))
    except TypeError:
        pass


def _retint(widget=None):
    """Re-set the icons of the registered widgets (all of them, or only ``widget``)."""
    alive = []
    for ref, nm, col in _icon_users:
        w = ref()
        if w is None:
            continue
        alive.append((ref, nm, col))
        if widget is not None and w is not widget:
            continue
        try:
            w.setIcon(icon(nm, _icon_colour_for(w, col)))
        except RuntimeError:
            pass
    if widget is None:
        _icon_users[:] = alive


# ---------------------------------------------------------------------------------------------------------------------
# pyqtgraph
# ---------------------------------------------------------------------------------------------------------------------
def _plot_colours(kind: str):
    if kind == 'image':
        return IMAGE_BG, IMAGE_AXES
    t = TOKENS[_variant]
    return t['plot_bg'], t['plot_axes']


def style_plot(widget, kind: str = 'plot') -> None:
    """Background and axis colours of a pyqtgraph widget (PlotWidget, GraphicsLayoutWidget, HistogramLUTWidget ...).  ``kind`` 'image' = a
    camera frame / map canvas (dark in both variants), 'plot' = line plots (follow the variant).  No-op until ``apply`` was called."""
    if not _applied or widget is None:
        return
    if kind not in ('plot', 'image'):
        raise ValueError("kind must be 'plot' or 'image'")
    bg, axes = _plot_colours(kind)
    try:
        widget.setBackground(bg)
        scene = widget.scene() if hasattr(widget, 'scene') else None
        items = scene.items() if scene is not None else []
        for it in items:
            if isinstance(it, pg.AxisItem):
                it.setPen(pg.mkPen(axes))
                it.setTextPen(pg.mkPen(axes))
                try:
                    it.label.setDefaultTextColor(QtGui.QColor(axes))
                except Exception:                         # noqa: BLE001
                    pass
            elif isinstance(it, pg.LabelItem):
                it.setText(it.text, color=axes) if getattr(it, 'text', None) else None
        widget.setProperty('plot_kind', kind)
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------------------------------------------------
# style sheet
# ---------------------------------------------------------------------------------------------------------------------
_QSS = Template(r'''
QToolTip { background-color: $tooltip_bg; color: $tooltip_text; border: 1px solid $border_strong; padding: 4px 6px; }

QMainWindow::separator, QDialog::separator { background: $window; width: 5px; height: 5px; }
QMainWindow::separator:hover { background: $accent; }
QSplitter::handle { background: $window; }
QSplitter::handle:hover { background: $border_strong; }

QMenuBar { background: $window; border-bottom: 1px solid $border; padding: 1px 4px; }
QMenuBar::item { background: transparent; padding: 4px 10px; border-radius: 4px; }
QMenuBar::item:selected { background: $hover; }
QMenuBar::item:pressed { background: $selection; }
QMenu { background: $surface; border: 1px solid $border_strong; padding: 4px; }
QMenu::item { padding: 5px 28px 5px 26px; margin: 1px 2px; border-radius: 4px; background: transparent; }
QMenu::item:selected { background: $selection; color: $text; }
QMenu::item:disabled { color: $disabled; background: transparent; }
QMenu::icon { padding-left: 6px; }
QMenu::separator { height: 1px; background: $border; margin: 4px 8px; }
QMenu::indicator { width: 14px; height: 14px; left: 6px; }
$img_menucheck

QToolBar { background: $window; border: none; border-bottom: 1px solid $border; spacing: 4px; padding: 3px 6px; }
QToolBar::separator { background: $border; width: 1px; margin: 4px 6px; }
QToolButton { background: transparent; border: 1px solid transparent; border-radius: $radius_button px; padding: 4px 8px; color: $text; }
QToolButton:hover { background: $hover; border-color: $border; }
QToolButton:pressed { background: $selection; border-color: $accent; }
QToolButton:checked { background: $selection; border-color: $accent; }
QToolButton:disabled { color: $disabled; background: transparent; border-color: transparent; }
QToolButton[role="danger"] { background: $danger; color: $on_danger; border: 1px solid $danger; font-weight: 600; }
QToolButton[role="danger"]:hover { background: $danger_hover; border-color: $danger_hover; }
QToolButton[role="danger"]:pressed { background: $danger_pressed; border-color: $danger_pressed; }
QToolButton[role="danger"]:disabled { background: $border; color: $disabled; border-color: $border; }

QPushButton { background: $input; color: $text; border: 1px solid $border_strong; border-radius: $radius_button px; padding: 3px 12px;
              min-height: 16px; }
QPushButton:hover { background: $hover; border-color: $muted; }
QPushButton:pressed { background: $border; }
QPushButton:checked { background: $selection; border-color: $accent; }
QPushButton:focus { border: 1px solid $accent; }
QPushButton:default { border: 1px solid $accent; }
QPushButton:disabled { background: $window; color: $disabled; border-color: $border; }
QPushButton[role="primary"] { background: $accent; color: $on_accent; border: 1px solid $accent; font-weight: 600; }
QPushButton[role="primary"]:hover { background: $accent_hover; border-color: $accent_hover; }
QPushButton[role="primary"]:pressed { background: $accent_pressed; border-color: $accent_pressed; }
QPushButton[role="primary"]:disabled { background: $border; color: $disabled; border-color: $border; }
QPushButton[role="danger"] { background: $danger; color: $on_danger; border: 1px solid $danger; font-weight: 600; }
QPushButton[role="danger"]:hover { background: $danger_hover; border-color: $danger_hover; }
QPushButton[role="danger"]:pressed { background: $danger_pressed; border-color: $danger_pressed; }
QPushButton[role="danger"]:disabled { background: $border; color: $disabled; border-color: $border; }
QPushButton[compact="true"] { padding: 3px 4px; min-width: 0px; }
QPushButton[tight="true"] { padding: 3px 7px; }
QPushButton[role="ghost"] { background: transparent; border: 1px solid transparent; color: $text; }
QPushButton[role="ghost"]:hover { background: $hover; border-color: $border; }
QPushButton[role="ghost"]:pressed { background: $border; }
QPushButton[role="ghost"]:disabled { background: transparent; color: $disabled; border-color: transparent; }

QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QComboBox {
    background: $input; color: $text; border: 1px solid $border_strong; border-radius: $radius_input px; padding: 3px 6px;
    selection-background-color: $selection; selection-color: $text; }
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QAbstractSpinBox:focus, QComboBox:focus { border: 2px solid $accent; padding: 2px 5px; }
QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {
    background: $window; color: $disabled; border-color: $border; }
QLineEdit:read-only { background: $surface; }
QAbstractSpinBox { padding-right: 18px; }
QAbstractSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 16px; border: none; background: transparent;
                              border-top-right-radius: $radius_input px; }
QAbstractSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 16px; border: none; background: transparent;
                                border-bottom-right-radius: $radius_input px; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: $hover; }
QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed { background: $border; }
$img_spin
QComboBox { padding-right: 22px; }
QComboBox:hover, QAbstractSpinBox:hover { border-color: $muted; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 22px; border: none; background: transparent; }
$img_combo
QComboBox QAbstractItemView { background: $surface; border: 1px solid $border_strong; outline: 0; padding: 2px;
                              selection-background-color: $selection; selection-color: $text; }
QComboBox QAbstractItemView::item { min-height: 22px; padding: 2px 6px; }

QCheckBox, QRadioButton { spacing: 6px; background: transparent; }
QCheckBox:disabled, QRadioButton:disabled { color: $disabled; }
QCheckBox::indicator, QGroupBox::indicator { width: 14px; height: 14px; border: 1px solid $border_strong; border-radius: 3px; background: $input; }
QRadioButton::indicator { width: 14px; height: 14px; border: 1px solid $border_strong; border-radius: 8px; background: $input; }
QCheckBox::indicator:hover, QGroupBox::indicator:hover, QRadioButton::indicator:hover { border-color: $accent; }
QCheckBox::indicator:checked, QGroupBox::indicator:checked { background: $accent; border-color: $accent; $img_check }
QRadioButton::indicator:checked { background: $accent; border-color: $accent; $img_dot }
QCheckBox::indicator:indeterminate { background: $selection; border-color: $accent; }
QCheckBox::indicator:disabled, QGroupBox::indicator:disabled, QRadioButton::indicator:disabled { background: $window; border-color: $border; }
QCheckBox::indicator:checked:disabled, QGroupBox::indicator:checked:disabled, QRadioButton::indicator:checked:disabled { background: $border_strong;
                                                                                                                          border-color: $border_strong; }

QDockWidget { color: $text; font-size: ${pt_title}pt; font-weight: 600; }
QDockWidget * { font-size: ${pt_body}pt; font-weight: 400; }
QGroupBox { background: $surface; border: 1px solid $border; border-radius: $radius_card px; margin-top: 16px; padding: 0px 6px 4px 6px;
            font-size: ${pt_title}pt; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 2px; padding: 0 4px; color: $muted; }
QGroupBox * { font-size: ${pt_body}pt; font-weight: 400; }

QTabWidget::pane { border: none; border-top: 1px solid $border; top: -1px; }
QTabBar { background: transparent; }
QTabBar::tab { background: transparent; color: $muted; padding: 6px 14px; margin-right: 2px; border: none; border-bottom: 2px solid transparent; }
QTabBar::tab:hover { color: $text; background: $hover; }
QTabBar::tab:selected { color: $text; border-bottom: 2px solid $accent; background: transparent; }
QTabBar::tab:disabled { color: $disabled; }
$img_tabclose

QDockWidget::title { background: $surface; border-bottom: 1px solid $border; padding: 6px 8px; text-align: left; }
$img_dock

QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 11px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 11px; margin: 0; }
QScrollBar::handle:vertical { background: $border_strong; border-radius: 4px; min-height: 28px; margin: 1px 2px; }
QScrollBar::handle:horizontal { background: $border_strong; border-radius: 4px; min-width: 28px; margin: 2px 1px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: $muted; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; background: none; border: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

QAbstractItemView { background: $input; alternate-background-color: $surface; color: $text; border: 1px solid $border; border-radius: $radius_input px;
                    gridline-color: $border; outline: 0; selection-background-color: $selection; selection-color: $text; }
QAbstractItemView:disabled { color: $disabled; }
QAbstractItemView::item { padding: 2px 4px; }
QAbstractItemView::item:hover { background: $hover; }
QAbstractItemView::item:selected { background: $selection; color: $text; }
QTableCornerButton::section { background: $surface; border: none; }
QHeaderView { background: $surface; }
QHeaderView::section { background: $surface; color: $muted; border: none; border-bottom: 1px solid $border; border-right: 1px solid $border;
                       padding: 4px 8px; font-weight: 600; }
QHeaderView::section:last { border-right: none; }
$img_tree

QProgressBar { background: $input; border: 1px solid $border; border-radius: 4px; min-height: 14px; max-height: 16px; text-align: center; color: $text;
               font-size: ${pt_caption}pt; }
QProgressBar::chunk { background: $progress_chunk; border-radius: 3px; margin: 0; }

QSlider::groove:horizontal { height: 4px; background: $border_strong; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $accent; border-radius: 2px; }
QSlider::handle:horizontal { background: $accent; border: 2px solid $surface; width: 10px; margin: -6px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: $accent_hover; }
QSlider::groove:vertical { width: 4px; background: $border_strong; border-radius: 2px; }
QSlider::sub-page:vertical { background: $border_strong; }
QSlider::add-page:vertical { background: $accent; border-radius: 2px; }
QSlider::handle:vertical { background: $accent; border: 2px solid $surface; height: 10px; margin: 0 -6px; border-radius: 7px; }

QStatusBar { background: $surface; border-top: 1px solid $border; color: $muted; }
QStatusBar::item { border: none; }
QMessageBox QPushButton, QDialogButtonBox QPushButton { min-width: 72px; }

/* type scale and semantic labels (dynamic properties, never object names) */
QFrame[card="true"] { background: $surface; border: 1px solid $border; border-radius: $radius_card px; }
QFrame[header="true"] { background: $surface; border: none; border-bottom: 1px solid $border; }
QFrame[notice="true"] { background: $accent_bg; border: none; border-bottom: 1px solid $accent; }
QFrame[notice="true"][tone="warn"] { background: $warn_bg; border-bottom: 1px solid $warn; }
QFrame[notice="true"] QLabel { background: transparent; border: none; }
QFrame[stripe="true"] { background: transparent; border: none; }
QFrame[stripe="true"][tone="warn"] { background: $warn; }
QFrame[stripe="true"][tone="danger"] { background: $danger; }
QLabel[caption="true"] { font-size: ${pt_caption}pt; color: $muted; }
QLabel[heading="true"] { font-weight: 600; padding: 2px 4px; }
*[mono="true"] { font-family: "$font_mono", "Cascadia Mono", monospace; }
QLabel[tone="muted"] { color: $muted; }
QLabel[tone="safe"] { color: $safe; font-weight: 600; }
QLabel[tone="warn"] { color: $warn; font-weight: 600; }
QLabel[tone="danger"] { color: $danger; font-weight: 600; }
QLabel[tone="sim"] { color: $sim; font-weight: 600; }
QLabel[tone="accent"] { color: $accent; font-weight: 600; }
QLabel[solid="true"][tone="safe"] { background: $safe; color: $on_safe; padding: 2px 8px; }
QLabel[solid="true"][tone="warn"] { background: $warn; color: $on_warn; padding: 2px 8px; }
QLabel[solid="true"][tone="danger"] { background: $danger; color: $on_danger; padding: 2px 8px; }
QLabel[solid="true"][tone="sim"] { background: $sim; color: $on_sim; padding: 2px 8px; }
QLabel[solid="true"][tone="accent"] { background: $accent; color: $on_accent; padding: 2px 8px; }
QLabel[chip="true"] { background: $hover; color: $text; border: 1px solid $border_strong; border-radius: 10px; padding: 2px 10px; font-weight: 600; }
QLabel[chip="true"][tone="muted"] { background: $hover; color: $muted; border-color: $border_strong; }
QLabel[chip="true"][tone="safe"] { background: $safe_bg; color: $text; border-color: $safe; }
QLabel[chip="true"][tone="warn"] { background: $warn_bg; color: $text; border-color: $warn; }
QLabel[chip="true"][tone="danger"] { background: $danger_bg; color: $text; border-color: $danger; }
QLabel[chip="true"][tone="sim"] { background: $sim_bg; color: $text; border-color: $sim; }
QLabel[chip="true"][tone="accent"] { background: $accent_bg; color: $text; border-color: $accent; }
QLabel[chip="true"][solid="true"][tone="safe"] { background: $safe; color: $on_safe; }
QLabel[chip="true"][solid="true"][tone="warn"] { background: $warn; color: $on_warn; }
QLabel[chip="true"][solid="true"][tone="danger"] { background: $danger; color: $on_danger; }
QLabel[chip="true"][solid="true"][tone="sim"] { background: $sim; color: $on_sim; }
QLabel[chip="true"][solid="true"][tone="accent"] { background: $accent; color: $on_accent; }
QLabel:disabled { color: $disabled; }
QLabel[dot="true"] { font-size: 14pt; padding: 0; background: transparent; }
''')


_ASSET_DIR = os.path.join(tempfile.gettempdir(), 'hsapp_theme_assets_v1')


def _svg_file(name: str, body: str, hexcolor: str, variant: str) -> Optional[str]:
    """Write (when missing / changed) a small SVG used by the style sheet; -> forward-slash path or None when it cannot be written."""
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%s" stroke-width="3" stroke-linecap="round" '
           'stroke-linejoin="round">%s</svg>' % (hexcolor, body))
    path = os.path.join(_ASSET_DIR, '%s_%s_%s.svg' % (variant, name, hexcolor.lstrip('#')))
    try:
        os.makedirs(_ASSET_DIR, exist_ok=True)
        if not os.path.isfile(path):
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(svg)
        return path.replace('\\', '/')
    except OSError:
        return None


def _url(path: Optional[str]) -> Optional[str]:
    return 'url("%s")' % path if path else None


def build_qss(variant: str) -> str:
    """The application style sheet of ``variant`` (pure text; the image rules use SVG files written to the temp folder)."""
    t = dict(TOKENS[variant])
    muted, on_acc, text = t['muted'], t['on_accent'], t['text']
    up = _url(_svg_file('up', '<polyline points="6 15 12 9 18 15"/>', muted, variant))
    down = _url(_svg_file('down', '<polyline points="6 9 12 15 18 9"/>', muted, variant))
    check = _url(_svg_file('check', '<polyline points="5 12.5 10 17.5 19 7.5"/>', on_acc, variant))
    dot = _url(_svg_file('dot', '<circle cx="12" cy="12" r="3.5" fill="%s" stroke="none"/>' % on_acc, on_acc, variant))
    closed = _url(_svg_file('branch_closed', '<polyline points="9 6 15 12 9 18"/>', muted, variant))
    opened = _url(_svg_file('branch_open', '<polyline points="6 9 12 15 18 9"/>', muted, variant))
    close = _url(_svg_file('close', '<line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/>', muted, variant))
    close_hot = _url(_svg_file('close_hot', '<line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/>', text, variant))
    floating = _url(_svg_file('float', '<rect x="5" y="9" width="10" height="10" rx="1"/><polyline points="9 9 9 5 19 5 19 15 15 15"/>', muted, variant))
    t['img_spin'] = ('QAbstractSpinBox::up-arrow { image: %s; width: 9px; height: 9px; }\n'
                     'QAbstractSpinBox::down-arrow { image: %s; width: 9px; height: 9px; }' % (up, down)) if up and down else ''
    t['img_combo'] = 'QComboBox::down-arrow { image: %s; width: 10px; height: 10px; }' % down if down else ''
    t['img_check'] = 'image: %s;' % check if check else ''
    menucheck = _url(_svg_file('menucheck', '<polyline points="5 12.5 10 17.5 19 7.5"/>', text, variant))
    t['img_menucheck'] = 'QMenu::indicator:checked { image: %s; }' % menucheck if menucheck else ''
    t['img_dot'] = 'image: %s;' % dot if dot else ''
    t['img_tabclose'] = ('QTabBar::close-button { image: %s; subcontrol-position: right; padding: 2px; border-radius: 3px; }\n'
                         'QTabBar::close-button:hover { image: %s; background: %s; }' % (close, close_hot, t['hover'])) if close and close_hot else ''
    t['img_dock'] = ('QDockWidget { titlebar-close-icon: %s; titlebar-normal-icon: %s; }\n'
                     'QDockWidget::close-button, QDockWidget::float-button { background: transparent; border: none; padding: 2px; border-radius: 3px; }\n'
                     'QDockWidget::close-button:hover, QDockWidget::float-button:hover { background: %s; }' % (close, floating, t['hover'])
                     ) if close and floating else ''
    t['img_tree'] = ('QTreeView::branch:has-children:!has-siblings:closed, QTreeView::branch:closed:has-children:has-siblings { image: %s; }\n'
                     'QTreeView::branch:open:has-children:!has-siblings, QTreeView::branch:open:has-children:has-siblings { image: %s; }'
                     % (closed, opened)) if closed and opened else ''
    return _QSS.substitute(t)


def _palette(variant: str) -> QtGui.QPalette:
    t = TOKENS[variant]
    P, R, G = QtGui.QPalette, QtGui.QPalette.ColorRole, QtGui.QPalette.ColorGroup
    pal = P()
    c = QtGui.QColor
    roles = {R.Window: t['window'], R.WindowText: t['text'], R.Base: t['input'], R.AlternateBase: t['surface'], R.ToolTipBase: t['tooltip_bg'],
             R.ToolTipText: t['tooltip_text'], R.Text: t['text'], R.Button: t['input'], R.ButtonText: t['text'], R.BrightText: '#ffffff',
             R.Link: t['accent'], R.LinkVisited: t['sim'], R.Highlight: t['selection'], R.HighlightedText: t['text'],
             R.PlaceholderText: t['placeholder'], R.Light: t['hover'], R.Midlight: t['surface'], R.Mid: t['border'], R.Dark: t['border_strong'],
             R.Shadow: '#000000'}
    for role, hx in roles.items():
        pal.setColor(role, c(hx))
    for role in (R.WindowText, R.Text, R.ButtonText):
        pal.setColor(G.Disabled, role, c(t['disabled']))
    pal.setColor(G.Disabled, R.Base, c(t['window']))
    pal.setColor(G.Disabled, R.Button, c(t['window']))
    pal.setColor(G.Disabled, R.Highlight, c(t['border']))
    pal.setColor(G.Disabled, R.HighlightedText, c(t['disabled']))
    return pal


def _system_variant(app) -> str:
    """'dark' when Windows is in dark mode (QStyleHints.colorScheme, PySide6 >= 6.5), else 'light' (also when the platform cannot tell)."""
    try:
        hints = QtGui.QGuiApplication.styleHints()
        scheme = hints.colorScheme()
        if scheme == QtCore.Qt.ColorScheme.Dark:
            return 'dark'
    except (AttributeError, RuntimeError):
        pass
    return 'light'


def _hook_scheme(app) -> None:
    global _scheme_hooked
    if _scheme_hooked:
        return
    try:
        QtGui.QGuiApplication.styleHints().colorSchemeChanged.connect(lambda *_a: _on_scheme_changed())
        _scheme_hooked = True
    except (AttributeError, RuntimeError):
        pass


def _on_scheme_changed():
    if _mode == 'auto':
        app = QtWidgets.QApplication.instance()
        if app is not None:
            apply(app, 'auto')


def apply(app, mode: str = 'dark') -> str:
    """Apply the theme to the QApplication ``app`` (Fusion style, palette, style sheet, font, pyqtgraph defaults); -> the variant used.
    ``mode`` 'dark' | 'light' | 'auto' ('auto' follows Windows through QStyleHints.colorScheme(), otherwise light).  May be called again to
    switch at run time."""
    global _variant, _mode, _applied
    if mode not in MODES:
        mode = 'dark'
    variant = _system_variant(app) if mode == 'auto' else mode
    _mode = mode
    _variant = variant
    _applied = True
    app.setStyle('Fusion')
    app.setPalette(_palette(variant))
    font = QtGui.QFont(BODY_FONT, BODY_PT)
    font.setStyleHint(QtGui.QFont.StyleHint.SansSerif)
    app.setFont(font)
    app.setStyleSheet(build_qss(variant))
    t = TOKENS[variant]
    try:
        pg.setConfigOptions(background=t['plot_bg'], foreground=t['plot_axes'])
    except Exception:                                              # noqa: BLE001
        pass
    _icon_cache.clear()
    _retint()
    if mode == 'auto':
        _hook_scheme(app)
    for w in app.allWidgets():                                    # viewers restyle their pyqtgraph canvases
        fn = getattr(w, 'apply_theme', None)
        if callable(fn):
            try:
                fn()
            except RuntimeError:
                pass
    return variant


def reset(app) -> None:
    """Back to the plain Fusion look (what the tests use): empty application style sheet, the style's own palette, pyqtgraph defaults."""
    global _variant, _mode, _applied
    app.setStyleSheet('')
    app.setStyle('Fusion')
    app.setPalette(app.style().standardPalette())
    _variant, _mode, _applied = 'light', None, False
    _icon_cache.clear()
    try:
        pg.setConfigOptions(background='k', foreground='d')
    except Exception:                                              # noqa: BLE001
        pass
    _retint()


# old per-module colour tables, now aliases of the token table ---------------------------------------------------------------
STEP_COLORS = ToneMap({'todo': 'border_strong', 'running': 'accent', 'done': 'safe', 'stale': 'warn', 'failed': 'danger', 'optional': 'muted'})
STATE_COLORS = ToneMap({'pending': 'muted', 'running': 'accent', 'done': 'safe', 'failed': 'danger', 'stopped': 'warn', 'skipped': 'muted'})
FILTER_TONES = {'ok': 'safe', 'warn': 'warn', 'idle': 'muted', 'bad': 'danger'}
FILTER_STYLES = QssMap({'ok': 'safe', 'warn': 'warn', 'idle': 'muted', 'bad': 'danger'}, bold=('ok', 'warn', 'bad'))
