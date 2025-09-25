# Codigos folder

The code here is meant to compute MHW metrics from temperature data and produce figures displaying these metrics. It includes Python notebooks showing one way to use the Python scripts inside the `pyscripts/` folder.

## What is in here?

 - `pyscripts/` :
     Python scripts containing all the functions used by the Python notebooks.

 - `01_check_packages.ipynb` :
     Python notebook meant to help the Python environment setup process.

 - `02_download_data.ipynb` :
     Python notebook that automates data downloading of the REP and MEDREA datasets.

 - `03_mhws_computing.ipynb` :
     Python notebook computing MHW metrics and saving them to netCDF files.

 - `04_report_plotting.ipynb` :
     Python notebook loading pre-computed MHW netCDF files in order to produce figures displaying MHW metrics (for the report).

 - `05_presentation_plotting.ipynb` :
     Python notebook loading pre-computed MHW netCDF files in order to produce figures displaying MHW metrics (for the presentation).

## How should this code be runned?

 1. Make sure to have correctly setup the Python environment by running `01_check_packages.ipynb`.
 1. Make sure to have correctly downloaded the required data by running `02_download_data.ipynb`.
 3. Run `03_mhws_computing.ipynb` to compute MHW datasets.
 4. Finally, run `04_report_plotting.ipynb` or `05_presentation_plotting.ipynb` to generate the desired figures.

*Note: Visual Studio Code was used during development, the workflow being thought to integrate easily with it.*
