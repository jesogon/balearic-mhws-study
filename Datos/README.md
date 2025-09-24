# Datos folder

The data here is meant to be used to compute MHW metrics in the Mediterranean Sea. Most of the data used has been provided by the Copernicus Marine Service.

## What is in here?

 - `bathymetry/` (Input) :
     Bathymetry files, from MEDREA or GEBCO.

 - `REP/` (Input) :
     Reprocessed satellite-derived SST over the Mediterranean Sea from 1982 to 2023 provided by Copernicus Marine Service.

 - `MEDREA/` (Input) :
     Physical reanalysis over the Mediterranean Sea from 1987 to 2022 provided by Copernicus Marine Service. For the purpose of the study, only a subset of the data in the Balearic region was used.

 - `mhws/` (Output) :
     NetCDF files containing pre-computed MHW metrics, to be used to produce figures.

*Please note that this code can also be adapted to work with other data.*

## How to download external data?

In the GitHub repository, these folders are empty. To get the data used in the study, please download them online, on the Copernicus Marine Data Store (https://data.marine.copernicus.eu/products), or contact the author.

Detailed information for downloading each dataset can be found in the `README.md` within its respective folder.

## Datos folders structure

In order to use the code without any modifications, the Datos folder must be provided using the original folder structure. If the data structure is modified, please modify the `Codigos/pyscripts/load_save_dataset.py` folder paths.

The expected file tree inside the Datos folder should be as follow:

> `Datos/`
> > `bathymetry/` <br>
> >  ├─ Bathymetry_GEBCO_2023_IBERIAN.nc <br>
> >  └─ Bathymetry_MEDREA_006_004_mask_bathy_BalearicIslands.nc <br>
>
> > `REP/` <br>
> >  └─ MEDITERRANEAN/SST-L4-REP-HR/DATA-NEW/DAILY/ <br>
> >    ├─ 1982/ <br>
> >    │  ├─ SST_MED_SST_L4_REP_OBSERVATIONS_010_021_y1982m01.nc <br>
> >    │  └─ ... <br>
> >    ├─ 1983/ <br>
> >    └─ ... <br>
>
> > `MEDREA/` <br>
> >  └─ MEDITERRANEAN/REANALYSIS/DATA/DAILY/BalearicIslands/ <br>
> >    ├─ 1987/ <br>
> >    │  ├─ TEMP_MEDSEA_MULTIYEAR_PHY_006_004_y1987m01_BalearicIslands.nc <br>
> >    │  └─ ... <br>
> >    ├─ 1988/ <br>
> >    └─ ... <br>
>
> > `mhws/` <br>
> >  ├─ yearly/ <br>
> >  │  ├─ rep_mhws_balears_1987_2021.nc <br>
> >  │  └─ medrea_mhws_balears_1987_2021.nc <br>
> >  └─ all_events/ <br>
> >     ├─ rep_mean_mhws_balears_1987_2021.nc <br>
> >     └─ medrea_mean_mhws_balears_1987_2021.nc <br>

## Licenses

This work make use of E.U. Copernicus Marine Service Information; https://doi.org/10.48670/moi-00173; https://doi.org/10.25423/CMCC/MEDSEA_MULTIYEAR_PHY_006_004_E3R1, under a permissive license.
