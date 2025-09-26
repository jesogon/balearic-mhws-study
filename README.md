# <h1 style="text-align: center;"> Balearic MHWs study </h1>

This repository contains all the code used to produce figures for the master thesis entitled “Marine heatwaves in the Balearic Islands region”.

Please note that the repository is not in its final state. Modifications for enhancing the code readability are planned during the whole month of September 2025.

## Overview

This code is meant to compute and visualise marine heatwaves (MHWs) in the Balearic Islands region. It is intented to provide a reproducible, modular and extensible workflow for:

- Downloading and handling large oceanographic datasets
- Computing MHW metrics using a l
- Generating figures for reports and presentations

## What is in here?

 - `Code/` :
    Code used to compute MHW metrics and produce figures from them.

 - `Data/` :
    Data used by the code (namely, temperature and bathymetry data). Those files are not included to the GitHub repository.

 - `Documentos/` :
    Folder containing the original master thesis report and presentation using the code herein.

## How should this code be runned?

See the `README.md` inside the Code folder for more details. 

## External data

For most of the code here, external data are expected. Due to the large size of those files, they are not included in the GitHub repository.

See the `README.md` inside the Data folder for more details.

## License

This code have been developped by Arthur Gonnet, and are licensed under the GNU General Public License v3.0 (GPLv3).

This code include a modified version of the *marineHeatWaves* module for python developped by Eric C. J. Oliver (see https://github.com/ecjoliver/marineHeatWaves), under GPLv3 license.

This code include the *pyMannKendall* module for python developped by Md. Manjurul Hussain and Ishtiak Mahmud (see https://github.com/mmhs013/pyMannKendall), under MIT license.

This work make use of E.U. Copernicus Marine Service Information; https://doi.org/10.48670/moi-00173; https://doi.org/10.25423/CMCC/MEDSEA_MULTIYEAR_PHY_006_004_E3R1, under a permissive license.

## Contact

> Arthur Gonnet <br>
> br.arthur.gonnet@gmail.com <br>
> https://github.com/arthur-gonnet/
