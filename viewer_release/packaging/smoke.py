"""Exercise the frozen viewer using small synthetic files."""

import json
import sys
import tempfile
import traceback
from pathlib import Path

import h5py
import numpy as np


def _write_test_files(folder):
    time = np.linspace(0, 220, 24)
    intensity = np.exp(-time / 55)
    y, x = np.mgrid[-1:1:40j, -1:1:48j]
    image = np.exp(-5 * (x * x + (y + 0.15) ** 2))
    wavelengths = np.linspace(480, 530, 8)
    spectral = np.exp(-((wavelengths - 510) / 20) ** 2)
    decay = folder / "decay.h5"
    cube4d = folder / "cube4d.h5"
    cubes = [folder / "reference.h5", folder / "shifted.h5"]
    with h5py.File(decay, "w") as h5:
        h5["Data"] = 400 * intensity[:, None, None] * image
        meta = h5.create_group("Metadata")
        meta["pulse_delay"] = time
        meta["pulse_width"] = np.full_like(time, 3)
        meta["emiccd_gain"] = np.full_like(time, 5)
        meta["accumulations"] = np.ones_like(time)
    with h5py.File(cube4d, "w") as h5:
        h5.attrs["labels"] = "wavelength, time_delay, y, x"
        h5["wavelength"] = wavelengths
        h5["time_delay"] = time
        h5["4D_cube"] = spectral[:, None, None, None] * intensity[None, :, None, None] * image
    for shift, path in enumerate(cubes):
        with h5py.File(path, "w") as h5:
            cube = h5.create_group("Cube")
            cube["Images"] = spectral[:, None, None] * np.roll(image, shift, axis=1)
            cube["Wavelength"] = wavelengths
            cube["TimeExposure"] = np.ones_like(wavelengths)
            camera = cube.create_group("Info/Camera")
            camera.attrs["Model"] = np.array([b"Synthetic"])
            camera.create_group("XAxis").attrs["Coefs"] = np.array([0, 0.5])
    snapshot = folder / "snapshot.h5"                 # Photon etc. broadband image: one frame, no /Cube/Wavelength
    with h5py.File(snapshot, "w") as h5:
        cube = h5.create_group("Cube")
        cube["Images"] = image[None]
        camera = cube.create_group("Info/Camera")
        camera.attrs["Model"] = np.array([b"Synthetic"])
        camera.create_group("XAxis").attrs["Coefs"] = np.array([0, 0.5])
        cube.create_group("Info/Cube").attrs["BroadBand"] = np.array([1], dtype=np.int32)
    cubes.append(snapshot)
    return decay, cube4d, cubes


def _smoke_trpl(decay, cube4d):
    from trpl_explorer import Dataset, Explorer
    from trpl_explorer_4d import Dataset4D, Explorer4D

    # Preload the fixtures so a malformed one raises instead of opening a modal dialog.
    if Dataset(decay).shape != (40, 48) or Dataset4D(cube4d).cube.shape != (8, 24, 40, 48):
        raise RuntimeError("Unexpected shape from the TRPL data loaders")
    window = Explorer(allow_correction=False)
    window.withdraw()
    try:
        window.open_file(str(decay))
        window.update_idletasks()
        if not window.ds or len(window.ds.t) != 24:
            raise RuntimeError("Broadband TRPL explorer did not open the fixture")
        view4d = Explorer4D(master=window, background=False)
        view4d.withdraw()
        try:
            view4d.open_file(str(cube4d), background=False)
            view4d.update_idletasks()
            if not view4d.ds or len(view4d.ds.lam) != 8:
                raise RuntimeError("4D TRPL explorer did not open the fixture")
        finally:
            view4d.destroy()
    finally:
        window.destroy()


def _smoke_cubes(cubes, folder):
    from hyperspec_viewer import main

    output = folder / "comparison.png"
    main([*(str(path) for path in cubes), "--save", str(output)])
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Hyperspectral comparison did not render its figure")

    from hsapp.viewer.overlay_view import OverlayViewer
    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication([])
    widget = OverlayViewer()
    try:
        widget.add_paths([str(path) for path in cubes])
        widget.wait_loaded(timeout_s=15)
        app.processEvents()
        if len(widget.entries) != len(cubes):
            raise RuntimeError("Cube overlay did not load both fixtures")
    finally:
        widget.close()


def run_smoke_test(result_file):
    result = {"status": "failed"}
    try:
        from viewer_release.update_check import __version__

        result.update(version=__version__, frozen=bool(getattr(sys, "frozen", False)))
        with tempfile.TemporaryDirectory(prefix="trpl-viewer-smoke-") as temp:
            folder = Path(temp)
            decay, cube4d, cubes = _write_test_files(folder)
            _smoke_trpl(decay, cube4d)
            _smoke_cubes(cubes, folder)
        # Exercise the native BLAS/LAPACK library used by TRPL lifetime fitting.
        coefficients, *_ = np.linalg.lstsq([[1, 0], [1, 1], [1, 2]], [5, 7, 9], rcond=None)
        np.testing.assert_allclose(coefficients, [5, 2])
        result["status"] = "passed"
    except Exception:
        result["error"] = traceback.format_exc()
    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["status"] == "passed" else 1
