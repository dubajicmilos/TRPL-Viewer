"""Navigation and sample readout for the standalone TRPL explorer graphs."""
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk


class GraphTools(ttk.Frame):
    def __init__(self, parent, canvas, graphs):
        super().__init__(parent)
        self.canvas = canvas
        self.graphs = graphs
        self._home = {}
        self._positions = None

        self.toolbar = NavigationToolbar2Tk(canvas, self, pack_toolbar=False)
        self.toolbar.pack(side=tk.LEFT)
        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.graph = tk.StringVar(self, value=next(iter(graphs)))
        self.graph_choice = ttk.Combobox(self, width=max(map(len, graphs)), textvariable=self.graph,
                                         state='disabled', values=tuple(graphs))
        self.graph_choice.pack(side=tk.LEFT)
        self.graph_choice.bind('<<ComboboxSelected>>', self._change_graph)
        self.focused = tk.BooleanVar(self, value=False)
        self.focus_button = ttk.Checkbutton(self, text='Graph only', variable=self.focused, command=self._layout)
        self.focus_button.pack(side=tk.LEFT, padx=5)
        self.reset_button = ttk.Button(self, text='Reset zoom', command=self.reset_zoom)
        self.reset_button.pack(side=tk.LEFT, padx=2)
        self.readout = ttk.Label(self)
        self.readout.pack(side=tk.LEFT, padx=8)
        self.clear()

        canvas.mpl_connect('scroll_event', self._wheel)
        canvas.mpl_connect('motion_notify_event', self._inspect)
        canvas.mpl_connect('figure_leave_event', lambda _event: self.readout.config(text=''))

    @staticmethod
    def _ranges(ax):
        return ax.get_xlim(), ax.get_ylim()

    def capture_ranges(self):
        """Save the manual limits before the explorer rebuilds the curves."""
        return {ax: self._ranges(ax) for ax in self._home}

    def finish_redraw(self, previous):
        """Record full extents and retain zoom on axes with the same scale and units."""
        available = []
        for name, ax in self.graphs.items():
            if not ax.axison:
                self._home.pop(ax, None)
                continue
            available.append(name)
            full = self._ranges(ax)
            axes = ((ax.get_xlabel(), ax.get_xscale()), (ax.get_ylabel(), ax.get_yscale()))
            if ax in previous and ax in self._home:
                old_full, old_axes = self._home[ax]
                for current, home, scale, old_scale, setter in zip(
                    previous[ax], old_full, axes, old_axes, (ax.set_xlim, ax.set_ylim),
                ):
                    if scale == old_scale and current != home:
                        setter(current)
                if old_axes != axes:
                    self.toolbar.update()
            self._home[ax] = full, axes
        self.graph_choice.config(values=available)
        if available and self.graph.get() not in available:
            self.graph.set(available[0])
            if self.focused.get():
                self._layout()
        self.graph_choice.config(state='readonly' if available else 'disabled')
        self._update_focus_button()
        self.reset_button.state(('!disabled',) if available else ('disabled',))

    def clear(self):
        self._home.clear()
        self.toolbar.update()
        self.graph_choice.config(state='disabled')
        self._update_focus_button()
        self.reset_button.state(('disabled',))
        self.readout.config(text='')

    def _update_focus_button(self):
        # Keep the way back to the overview available while a file loads or after a load error.
        self.focus_button.state(('!disabled',) if self.focused.get() or self._home else ('disabled',))

    def reset_zoom(self):
        for ax, (limits, _) in self._home.items():
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
        self.toolbar.update()
        self.canvas.draw_idle()

    def _change_graph(self, _event):
        if not self.focused.get():
            self.focused.set(True)
        self._layout()

    def _layout(self):
        if self.focused.get():
            if self._positions is None:
                self._positions = {ax: ax.get_position(original=True).frozen() for ax in self.canvas.figure.axes}
            graph = self.graphs[self.graph.get()]
            for ax, position in self._positions.items():
                ax.set_visible(ax is graph)
                ax.set_position((0.09, 0.13, 0.87, 0.78) if ax is graph else position)
        elif self._positions is not None:
            for ax, position in self._positions.items():
                ax.set_visible(True)
                ax.set_position(position)
            self._positions = None
        self._update_focus_button()
        self.toolbar.update()  # Navigation history also stores axes positions.
        self.readout.config(text='')
        self.canvas.draw_idle()

    def _wheel(self, event):
        ax = event.inaxes
        if ax not in self._home or self.toolbar.mode or not event.step:
            return
        if self.toolbar._nav_stack() is None:
            self.toolbar.push_current()
        factor = 1.25 ** -np.clip(event.step, -8, 8)
        modifiers = getattr(event, 'modifiers', ())
        for coord, axis, limits, setter, skip in (
            (event.xdata, ax.xaxis, ax.get_xlim(), ax.set_xlim, 'ctrl'),
            (event.ydata, ax.yaxis, ax.get_ylim(), ax.set_ylim, 'shift'),
        ):
            if skip in modifiers or coord is None:
                continue
            transform = axis.get_transform()
            start, end, anchor = transform.transform([*limits, coord])
            next_limits = transform.inverted().transform(anchor + factor * (np.array([start, end]) - anchor))
            if np.isfinite(next_limits).all() and next_limits[0] != next_limits[1]:
                if axis.get_scale() != 'log' or np.all(next_limits > 0):
                    setter(next_limits)
        self.toolbar.push_current()
        self.canvas.draw_idle()

    def _inspect(self, event):
        ax = event.inaxes
        sample = ''
        distance = 12 ** 2
        if ax in self._home and not self.toolbar.mode and event.button is None:
            for line in ax.get_lines():
                label = line.get_label()
                if not line.get_visible() or label.startswith('_') or label == 'hover':
                    continue
                points = line.get_xydata()
                if not len(points):
                    continue
                with np.errstate(invalid='ignore', over='ignore'):
                    screen = ax.transData.transform(points)
                    d2 = np.sum((screen - (event.x, event.y)) ** 2, axis=1)
                d2[~np.isfinite(d2)] = np.inf
                index = np.argmin(d2)
                if d2[index] < distance and ax.bbox.contains(*screen[index]):
                    distance = d2[index]
                    x, y = points[index]
                    sample = f'{label}   x {x:.6g}   y {y:.6g}'
        self.readout.config(text=sample)
