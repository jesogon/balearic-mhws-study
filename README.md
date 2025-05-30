# <h1 style="text-align: center;"> Balearic MHWs study </h1>

This repository contains all the codes used to produce figures for the "Coastal marine heatwaves in the Balearic Sea" thesis report.

Please note that the repository is not in its final state. Modifications for enhancing the code readability are planned during the whole month of September 2025.

## What is in here?

 - `Codigos/` :
      Codes used to compute MHW metrics and produce figures from them.

 - `Datos/` :
      Data used by the codes (namely, temperature and bathymetry data).

## How should this code be runned?

See the `README.md` inside the Codigos folder for more details. 

## External files

For most of the code here, external files are expected, namely data files. Inside the Datos folder are expected: data of **bathymetry**, of **REP** (satellite-derived SST), of **MEDREA** (physical reanalysis of temperature) and of **mhws** (pre-computed dataset of MHW). Due to the large size of those files, they are not included in the GitHub repository. See the `README.md` inside the Datos folder for more details.

This code can also be adapted to work on other data.

## License

These codes have been developped by Arthur Gonnet, and are licensed under GPLv3 license.

These codes use a modified version of the *marineHeatWaves* module for python developped by Eric C. J. Oliver (see https://github.com/ecjoliver/marineHeatWaves), under GPLv3 license.

These codes use the *pyMannKendall* module for python developped by Md. Manjurul Hussain and Ishtiak Mahmud (see https://github.com/mmhs013/pyMannKendall), under MIT license.

These codes use E.U. Copernicus Marine Service Information; https://doi.org/10.48670/moi-00173; https://doi.org/10.25423/CMCC/MEDSEA_MULTIYEAR_PHY_006_004_E3R1, under a permissive license.
