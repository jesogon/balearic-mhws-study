# Codigos folder

The code here is meant to compute MHW metrics from temperature data and produce figures displaying these metrics. It includes three Python notebooks showing one way to use the Python scripts inside the `pyscripts/` folder.

## What is in here?

 - `pyscripts/` :
     Python scripts containing all the functions used by the Python notebooks.

 - `mhws_computing.ipynb` :
     Python notebook computing MHW metrics and saving them to netCDF files.

 - `report_plotting.ipynb` :
     Python notebook loading pre-computed MHW netCDF files in order to produce figures displaying MHW metrics (for the report).

 - `presentation_plotting.ipynb` :
     Python notebook loading pre-computed MHW netCDF files in order to produce figures displaying MHW metrics (for the presentation).

 - `check_packages.ipynb` :
     Python notebook meant to help the Python environment setup process.

## How should this code be runned?

 1. Make sure to have correctly downloaded the required data, as described in the `README.md` of the Datos folder.
 2. Make sure to have correctly setup the Python environment to be able to run Python notebooks.
 3. Make sure to have downloaded all the required packages.

 > Run `check_packages.ipynb` to check your Python environment.

 4. Run `mhws_computing.ipynb` to compute MHW datasets (see next sections for more details).
 5. Run `report_plotting.ipynb` or `presentation_plotting.ipynb` to get the desired figures (see next sections for more details).

*Note: Visual Studio Code was used during development, the workflow being thought to integrate easily with it.*
