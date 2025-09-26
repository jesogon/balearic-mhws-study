# Code/pyscripts folder

In this folder, Python scripts define functions that are used in the Python notebooks. The code here has been intented to be reusable.

## What is in here?

 - `load_save_dataset.py` :
     Code used to load or save data from the Data folder.

 - `mhws_computer.py` :
     Code used to compute MHW metrics from temperature datasets.

 - `basic_plotter.py` :
     Code used to generate figures from datasets.

 - `options.py` :
     Code containing options used in the other scripts.

 - `utils.py` :
     Useful code used in the other scripts.

 - `marineHeatWaves.py` :
     Modified version of *marineHeatWaves* module for Python developed by Eric C. J. Oliver. (see License section)

## License

The *marineHeatWaves* module for python developped by Eric C. J. Oliver has been modified for the purpose of the thesis. The modifications are the following :

 1. **Add severity metrics**: The severity metric has been added as described in the report.
 2. **Calculate means by days and not by event**: This modification makes longer event have more impact on the annual mean (of intensity or severity).
 3. **Option to cut events between 31st December and 1st January**: This modification makes that for a given year, the annual metrics are only based on what happened this given year. This introduces a bias in the mean duration metric, as some event would be split and thus show a lower duration.
