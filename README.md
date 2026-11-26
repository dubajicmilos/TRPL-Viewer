# TRPL Viewer

Open and inspect existing time-resolved photoluminescence and hyperspectral HDF5 files on Windows.

## Install and update

Download `TRPL-Viewer-Setup-<version>-win-x64.exe` from [Releases](https://github.com/dubajicmilos/TRPL-Viewer/releases/latest), run it, and open **TRPL Viewer** from the Start menu. The installer includes Python and the viewing libraries.

Use **Help > Check for updates** to check for the latest stable release. Open the release page from the prompt and run the new installer to upgrade. Viewing local files works offline.

## Open your data

- **Broadband TRPL** shows decay curves, lifetime plots, images and regions. It accepts raw, corrected and stitched files.
- **4D TRPL** shows spectra and decays across time and wavelength. For older raw files, use **File > Choose IMA XML** to supply wavelength calibration when needed.
- **Cube overlay** aligns and compares spectra and images across hyperspectral cubes.
- **Hyperspectral comparison** opens at least two cubes side by side and plots their spectra and ratios.

In the TRPL explorers, scroll to zoom, use the toolbar to pan or draw a zoom box, and choose **Graph only** to enlarge the selected graph. **Reset zoom** restores the full range.
