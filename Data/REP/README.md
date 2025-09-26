# REP dataset

## Where does the data come from?

The REP dataset has been provided by E.U. Copernicus Marine Service Information; https://doi.org/10.48670/moi-00173.

## How to download?

### Automated download

Inside the `Code/` folder, the Python notebook `02_download_data.ipynb` automates the downloading of this dataset.

### Manual download

However, to download it manually, here is how:

 1. Go on https://doi.org/10.48670/moi-00173
 2. Click on `Data access`
 3. Download "All data"
 
 > When downloading, either download the whole files using `Browse` link, or download only subsets using `Form` link. For the second option, you will require a Copernicus Marine Data Store account.

 4. (Optional) Structure the data under the following data structure to be used directly with the code provided here. If the data structure is modified, please modify the `Code/pyscripts/load_save_dataset.py` folder paths.

```
REP/
└── MEDITERRANEAN/SST-L4-REP-HR/DATA-NEW/DAILY/
    ├── 1982/
    │   ├── SST_MED_SST_L4_REP_OBSERVATIONS_010_021_y1982m01.nc
    │   └── ...
    ├── 1983/
    └── ...
```

## Overview

The Reprocessed (REP) Mediterranean (MED) dataset provides a stable and consistent long-term Sea Surface Temperature (SST) time series over the Mediterranean Sea (and the adjacent North Atlantic box) developed for climate applications. This product consists of daily (nighttime), optimally interpolated (L4), satellite-based estimates of the foundation SST (namely, the temperature free, or nearly-free, of any diurnal cycle) at 0.05° resolution grid covering the period from 1st January 1981 to present (approximately one month before real time). The MED-REP-L4 product is built from a consistent reprocessing of the collated level-3 (merged single-sensor, L3C) climate data record (CDR) v.3.0, provided by the ESA Climate Change Initiative (CCI) and covering the period up to 2021, and its interim extension (ICDR) that allows the regular temporal extension for 2022 onwards.

## Specifications (on 22 Sep. 2025)

|   |   |
| - | - |
| Full name             | Mediterranean Sea - High Resolution L4 Sea Surface Temperature Reprocessed    |
| Product ID            | SST_MED_SST_L4_REP_OBSERVATIONS_010_021    |
| Source                | Satellite observations    |
| Spatial extent        | Mediterranean Sea · Lat 30.13° to 46.03° · Lon -18.12° to 36.33° |
| Spatial resolution    | 0.05° × 0.05°    |
| Temporal extent       | 1 Jan 1982 to 23 Aug 2025 (here downloaded until 2023)     |
| Temporal resolution   | Daily    |
| Processing level      | Level 4    |
| Variables             | Sea surface temperature (SST)    |
| Feature type          | Grid    |
| Blue markets          | Climate & Adaptation · Policy & Governance · Science & Innovation · Extremes & Hazards & Safety · Coastal Services    |
| Projection            | WGS84 / Simple Mercator (EPSG:41001)    |
| Update frequency      | Daily    |
| Format                | NetCDF-4    |
| Originating centre    | MET Norway    |
| Last metadata update  | 26 November 2024    |
|   |   |

## References

DOI (product): https://doi.org/10.48670/moi-00173

Pisano, A., Nardelli, B. B., Tronconi, C., & Santoleri, R. (2016). The new Mediterranean optimally interpolated pathfinder AVHRR SST Dataset (1982–2012). Remote Sensing of Environment, 176, 107-116. doi: https://doi.org/10.1016/j.rse.2016.01.019

Embury, O., Merchant, C.J., Good, S.A., Rayner, N.A., Høyer, J.L., Atkinson, C., Block, T., Alerskans, E., Pearson, K.J., Worsfold, M., McCarroll, N., Donlon, C., (2024). Satellite-based time-series of sea-surface temperature since 1980 for climate applications. Sci Data 11, 326. doi: https://doi.org/10.1038/s41597-024-03147-w
