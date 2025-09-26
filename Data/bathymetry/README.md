# MEDREA dataset 

## Where does the data come from?

The MEDREA dataset has been provided by E.U. Copernicus Marine Service Information; https://doi.org/10.25423/CMCC/MEDSEA_MULTIYEAR_PHY_006_004_E3R1.

## How to download?

### Automated download

Inside the `Code/` folder, the Python notebook `02_download_data.ipynb` automates the downloading of this dataset.

### Manual download

 1. Go on https://doi.org/10.25423/CMCC/MEDSEA_MULTIYEAR_PHY_006_004_E3R1
 2. Click on `Data access`
 3. Download "Statics (bathymetry)"
 
 > When downloading, either download the whole files using `Browse` link, or download only subsets using `Form` link. For the second option, you will require a Copernicus Marine Data Store account. Note that in this study, the subset used had the coordinates: Lat 37.69° to 40.98°, Lon -1.042° to 5.458°.

 4. Structure the data under the following data structure to be used directly with the code provided here. If the data structure is modified, please modify the `Code/pyscripts/load_save_dataset.py` folder paths.

```
bathymetry/
└── Bathymetry_MEDREA_006_004_mask_bathy_BalearicIslands.nc
```

## Overview

> "The topography is created starting from the GEBCO 30arc-second grid (Weatherall et al., 2015), filtered (using a Shapiro filter) and manually modified in critical areas such as: islands along the Eastern Adriatic coasts, Gibraltar and Messina straits, Atlantic box edge."
>
> -- *Escudier et al., (2021)*

## Specifications (on 22 Sep. 2025)

|   |   |
| - | - |
| Full name             | Mediterranean Sea Physics Reanalysis    |
| Product ID            | MEDSEA_MULTIYEAR_PHY_006_004    |
| Source                | Numerical models    |
| Spatial extent        | Mediterranean Sea · Lat 30.19° to 45.98° · Lon -6° to 36.29° |
| Spatial resolution    | 0.042° × 0.042°    |
| Temporal extent       | 1 Jan 1987 to 1 Aug 2025 (here downloaded until 2022)     |
| Temporal resolution   | Daily    |
| Elevation (depth) levels | 141    |
| Processing level      | Level 4    |
| Variables             | Sea floor depth below geoid · ...   |
| Feature type          | Grid    |
| Blue markets          | Policy & Governance · Science & Innovation · Extremes & Hazards & Safety · Coastal Services · Natural Resources & Energy · Trade & Marine Navigation    |
| Projection            | WGS 84 (EPSG:4326)    |
| Data assimilation     | In-Situ TS Profiles · SST · Sea Level    |
| Update frequency      | Daily    |
| Format                | NetCDF-4    |
| Originating centre    | CMCC (Italy)    |
| Last metadata update  | 24 June 2025    |
|   |   |

## References

DOI (Product): https://doi.org/10.25423/CMCC/MEDSEA_MULTIYEAR_PHY_006_004_E3R1

Escudier, R., Clementi, E., Omar, M., Cipollone, A., Pistoia, J., Aydogdu, A., Drudi, M., Grandi, A., Lyubartsev, V., Lecci, R., Cretí, S., Masina, S., Coppini, G., & Pinardi, N. (2020). Mediterranean Sea Physical Reanalysis (CMEMS MED-Currents) (Version 1) [Data set]. Copernicus Monitoring Environment Marine Service (CMEMS). https://doi.org/10.25423/CMCC/MEDSEA_MULTIYEAR_PHY_006_004_E3R1

Escudier, R., Clementi, E., Cipollone, A., Pistoia, J., Drudi, M., Grandi, A., Lyubartsev, V., Lecci, R., Aydogdu, A., Delrosso, D., Omar, M., Masina, S., Coppini G., Pinardi, N. (2021). A High Resolution Reanalysis for the Mediterranean Sea. Frontiers in Earth Science, 9, 1060, https://www.frontiersin.org/article/10.3389/feart.2021.702285, DOI=10.3389/feart.2021.702285

Nigam, T., Escudier, R., Pistoia, J., Aydogdu, A., Omar, M., Clementi, E., Cipollone, A., Drudi, M., Grandi, A., Mariani, A., Lyubartsev, V., Lecci, R., Cretí, S., Masina, S., Coppini, G., & Pinardi, N. (2021). Mediterranean Sea Physical Reanalysis INTERIM (CMEMS MED-Currents, E3R1i system) (Version 1) [Data set]. Copernicus Monitoring Environment Marine Service (CMEMS). https://doi.org/10.25423/CMCC/MEDSEA_MULTIYEAR_PHY_006_004_E3R1I

