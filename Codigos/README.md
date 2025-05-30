# *Codigos* folder

The codes here are meant to compute MHW metrics from temperature data and produce figures displaying these metrics. It includes two Python notebooks showing one way to use the Python scripts inside the `pyscripts` folder.

## What is in here?

 - `pyscripts/` :
      Python scripts containing all the functions used by the Python Notebooks.

 - `compute_mhws.ipynb` :
      Python notebook computing MHW metrics and saving them to netCDF files.

 - `figure_plotter.ipynb` :
      Python notebook loading netCDF files produced by `compute_mhws.ipynb` in order to produce figures displaying MHW metrics.

 - `check_packages.ipynb` :
      Python notebook meant to help the Python environment setup process.

## How should this code be runned?

...

Make sure that the *Datos* folder is at the right place (see External files section) for the codes to run smoothly.

Normally, you only need to run any script (except the utils) using python3 and the required python packages installed (see *requirements.txt*).
These scripts haven't been tested under Windows. Even if some precautions were taken, some errors may occur regarding file paths.

If issues occur with Python interpreter or packages, please run *utils/check_packages_version.py* to check versions.

Note: Visual Studio Code was used during development, the workflow being thought to integrate easily with it.

## External files

To run these codes smoothly, external files are expected in the Datos folder. See `README.md` of the root folder.

