"""Windows launcher for the standalone TRPL and hyperspectral viewers."""

import argparse
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import subprocess
import sys
from threading import Thread
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser

from viewer_release.update_check import UpdateCheckError, __version__, available_update


APP_NAME = 'TRPL Viewer'
TOOLS = {
    'trpl': ('Broadband TRPL', 'Explore decay curves, maps and regions.'),
    'trpl-4d': ('4D TRPL', 'Explore wavelength and time-resolved image stacks.'),
    'cube-overlay': ('Cube overlay', 'Overlay spectra, maps and ratios from several cubes.'),
    'hyperspectral': ('Hyperspectral comparison', 'Compare cubes side by side.'),
}


def _run_tool(tool, args):
    if tool == 'trpl':
        from trpl_explorer import main

        return main(args, allow_correction=False)
    if tool == 'trpl-4d':
        from trpl_explorer_4d import main
    elif tool == 'cube-overlay':
        from hsapp.viewer.overlay_view import main
    else:
        from hyperspec_viewer import main
    return main(args)


def _start_tool(tool, paths):
    args = ['--tool', tool, *paths]
    if getattr(sys, 'frozen', False):
        # Viewers may keep running independently of the launcher window.
        subprocess.Popen([sys.executable, *args], env={**os.environ, 'PYINSTALLER_RESET_ENVIRONMENT': '1'})
    else:
        subprocess.Popen([sys.executable, '-m', 'viewer_release.launcher', *args],
                         cwd=Path(__file__).resolve().parents[1])


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.minsize(575, 270)

        menu = tk.Menu(self)
        self.help_menu = tk.Menu(menu, tearoff=False)
        self.help_menu.add_command(label='Check for updates…', command=self.check_for_updates)
        menu.add_cascade(label='Help', menu=self.help_menu, underline=0)
        self.configure(menu=menu)

        content = ttk.Frame(self, padding=20)
        content.pack(fill='both', expand=True)
        ttk.Label(content, text=APP_NAME, font=('Segoe UI', 19, 'bold')).pack(anchor='w', pady=(0, 16))
        tools = ttk.Frame(content)
        tools.pack(fill='x')
        tools.columnconfigure(1, weight=1)
        for row, (tool, (label, description)) in enumerate(TOOLS.items()):
            ttk.Button(tools, text=label, width=27, command=lambda key=tool: self.open_tool(key)).grid(
                row=row, column=0, sticky='ew', pady=3)
            ttk.Label(tools, text=description, wraplength=300).grid(
                row=row, column=1, sticky='w', padx=(12, 0), pady=3)

        footer = ttk.Frame(content)
        footer.pack(side='bottom', fill='x', pady=(16, 0))
        ttk.Label(footer, text=f'Version {__version__}').pack(side='right')
        self.status = ttk.Label(footer)
        self.status.pack(side='left')

    def open_tool(self, tool):
        paths = ()
        if tool == 'hyperspectral':
            paths = filedialog.askopenfilenames(
                parent=self, title='Select hyperspectral cubes to compare',
                filetypes=[('HDF5 cubes', '*.h5 *.hdf5'), ('All files', '*')],
            )
            if not paths:
                return
        try:
            _start_tool(tool, paths)
        except OSError as exc:
            messagebox.showerror(APP_NAME, f'Could not open {TOOLS[tool][0]}.\n\n{exc}', parent=self)

    def check_for_updates(self):
        self.help_menu.entryconfigure(0, state='disabled')
        self.status.configure(text='Checking GitHub for updates…')
        results = SimpleQueue()

        def check():
            try:
                results.put(available_update())
            except UpdateCheckError as exc:
                results.put(exc)

        Thread(target=check, daemon=True).start()
        self.after(100, self.show_update_result, results)

    def show_update_result(self, results):
        try:
            result = results.get_nowait()
        except Empty:
            self.after(100, self.show_update_result, results)
            return

        self.help_menu.entryconfigure(0, state='normal')
        self.status.configure(text='')
        if isinstance(result, UpdateCheckError):
            messagebox.showinfo('Check for updates', str(result), parent=self)
        elif result is None:
            messagebox.showinfo('Check for updates', f'Version {__version__} is up to date.', parent=self)
        elif messagebox.askyesno(
            'Update available', f'Version {result.version} is available.\n\nOpen the GitHub release to download the installer?',
            parent=self,
        ):
            try:
                if webbrowser.open(result.release_page):
                    return
            except (OSError, webbrowser.Error):
                pass
            messagebox.showinfo('Update available', f'Open this page in your browser:\n\n{result.release_page}', parent=self)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='TRPL-Viewer', description=__doc__)
    parser.add_argument('--tool', choices=TOOLS, help='open a viewer directly, followed by its usual arguments')
    parser.add_argument('--smoke-test', metavar='RESULT', type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args(args[:2] if args and args[0] == '--tool' else args)
    if options.smoke_test is not None:
        from viewer_release.packaging.smoke import run_smoke_test

        return run_smoke_test(options.smoke_test)
    if options.tool is not None:
        try:
            return _run_tool(options.tool, args[2:])
        except Exception as exc:
            if not getattr(sys, 'frozen', False):
                raise
            messagebox.showerror(APP_NAME, f'Could not open {TOOLS[options.tool][0]}.\n\n{exc}')
            return 1
    Launcher().mainloop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
