# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from PyInstaller.utils.hooks import collect_data_files

root = Path(SPECPATH).resolve().parents[1]
version = os.environ["TRPL_VIEWER_BUILD_VERSION"]
generated = Path(workpath) / "viewer_build"
generated.mkdir(parents=True, exist_ok=True)
version_module = generated / "_version.py"
version_module.write_text(f"__version__ = {version!r}\n", encoding="utf-8")

data = []
license_dir = "THIRD-PARTY-LICENSES"
notices = [
    "Licenses and notices for the third-party software distributed with TRPL Viewer.",
    "Upstream licence files supplied by the Python packages are in the adjoining directories.",
    "",
    "This application uses PySide6, Shiboken and Qt. Qt is supplied as replaceable DLLs.",
    "The Qt/GPL and Qt/LGPL license texts accompany these notices.",
    "Qt licensing: https://doc.qt.io/qt-6/licensing.html",
    "Qt for Python source: https://code.qt.io/cgit/pyside/pyside-setup.git/",
    "Qt 6.7.2 source: https://download.qt.io/official_releases/qt/6.7/6.7.2/submodules/",
    "CPython licence: https://docs.python.org/3.11/license.html",
    "Tcl/Tk licence: https://www.tcl-lang.org/software/tcltk/license.html",
    "",
]
pending = [
    "numpy", "scipy", "h5py", "matplotlib", "scikit-image", "PySide6-Essentials", "shiboken6",
    "pyqtgraph", "PyInstaller",
]
distributions = {}
while pending:
    name = canonicalize_name(pending.pop())
    if name in distributions:
        continue
    dist = metadata.distribution(name)
    distributions[name] = dist
    for requirement in (Requirement(value) for value in dist.requires or ()):
        if requirement.marker is None or requirement.marker.evaluate():
            pending.append(requirement.name)
for name, dist in sorted(distributions.items()):
    notices.append(f"{name} {dist.version}: https://pypi.org/project/{name}/{dist.version}/")
    for path in dist.files or ():
        source = Path(dist.locate_file(path))
        filename = path.name.lower()
        if source.is_file() and (
            "licenses" in path.parts or filename.startswith(("license", "copying", "notice"))
        ):
            data.append((str(source), f"{license_dir}/{name}/{path.parent.as_posix()}"))
for name, path in (
    ("CPython", Path(sys.base_prefix) / "LICENSE.txt"),
    ("CPython", Path(sys.base_prefix) / "LICENSE_PYTHON.txt"),
    ("Tcl", Path(sys.base_prefix) / "tcl/tcl8.6/license.terms"),
    ("Tk", Path(sys.base_prefix) / "tcl/tk8.6/license.terms"),
):
    if path.is_file():
        data.append((str(path), f"{license_dir}/{name}"))
notice_file = generated / "NOTICE.txt"
notice_file.write_text("\n".join(notices) + "\n", encoding="utf-8")
data.append((str(notice_file), license_dir))
for license_name in ("GPL-3.0.txt", "LGPL-3.0.txt"):
    data.append((str(Path(SPECPATH) / "licenses" / license_name), f"{license_dir}/Qt"))
data += collect_data_files("pyqtgraph", includes=["colors/maps/*.csv", "colors/maps/*.txt"])

a = Analysis(
    [str(root / "viewer_release/launcher.py")],
    pathex=[str(root)],
    datas=data,
    hiddenimports=["matplotlib.backends.backend_agg", "PySide6.QtSvg"],
    excludes=["hsapp.__main__", "trpl_corrections", "trpl_explorer_local", "PyQt5", "PyQt6", "PySide2"],
    hooksconfig={"matplotlib": {"backends": ["TkAgg"]}},
)

# Compile the generated version into the bundle without editing the checkout.
a.pure.append(("viewer_release._version", str(version_module), "PYMODULE"))

# Reject accidental additions to the viewer's first-party module set.
expected = {
    "hsapp", "hsapp.gui", "hsapp.gui.theme", "hsapp.trpl_kinetics", "hsapp.viewer",
    "hsapp.viewer.cubefile", "hsapp.viewer.overlay_core", "hsapp.viewer.overlay_view",
    "hsapp.viewer.widgets", "viewer_release", "viewer_release.launcher", "viewer_release._version",
    "viewer_release.update_check", "viewer_release.packaging", "viewer_release.packaging.smoke",
    "trpl_explorer", "trpl_explorer_4d", "trpl_graph_tools", "hyperspec_viewer",
}
private = []
for name, path, _ in a.pure:
    source = Path(path).resolve()
    if source.is_relative_to(root):
        top = source.relative_to(root).parts[0]
        if (top.endswith(".py") or top in {"hsapp", "viewer_release"}) and name not in expected:
            private.append(name)
if private:
    raise RuntimeError(f"Unexpected project modules in viewer bundle: {sorted(private)}")

pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="TRPL-Viewer", console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="TRPL-Viewer", upx=False)
