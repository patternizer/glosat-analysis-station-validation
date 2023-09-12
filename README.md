![image](https://github.com/patternizer/glosat-analysis-station-validation/blob/main/tas_median_2018-03.png)

# glosat-analysis-station-validation

Comparison of monthly GloSAT in-filled 5x5 gridded HadCRUT-style analysis with station anomalies and the climate indicies NAO and SOI. Part of land surface air temperature station data record validation efforts and ongoing work for the [GloSAT](https://www.glosat.org) project: www.glosat.org. 

## Contents

* `merge_netcdfs.py` - Python helper to concatenate yearly HadCRUT5-style netcdf files.
* `stations_to_gridded_netcdf.py` - Python reader to load in CRUTEM5 station archive and output raw gridded .nc.
* `analysis_2_pkl.py` - Python reader to load gridded HadCRUT5-style analysis and output to .pkl. Used to provide gridded background for MAT and LAT overlays.
* `mat_2_pkl.py` - Python reader to load gridded MAT and output to .pkl. Used to provide gridded MAT overlay.
* `lat_2_pkl.py` - Python reader to load gridded LAT, area-average and output to .pkl. Used to provide gridded LAT overlay.
* `nao_2_pkl.py` - Python reader to load Luterbacher and Jones monthly NAO indices and output to .pkl a merged timeseries from 1658-2022. Used to display monthly NAO value in each monthly comparison map.
* `soi_2_pkl.py` - Python reader to load monthly SOI and output to .pkl. Used to display monthly SOI value in each monthly comparison map.
* `aod_2_pkl.py` - Python reader to load volcanic AOD(550nm) and output to .pkl. Used to provide a perform large volcano detection (independent analysis for Emily Wallis).
* `plot_glosat_analysis_vs_stations_vs_mat.py` - python script to read in the GloSAT.analysis.alpha.4 in-filled 5x5 gridded median netCDFs, the GloSAT.p04c.EBC.LEKnormals station anomalies, the gridded LAT, the gridded MAT and the merged NAO and SOI series, and plot a map for each month. 
* `plot_glosat_stations_vs_gridded.py` - python script to read in the GloSAT stations and overlay on gridded LAT, and plot a map for each month. 
* `crutem_2_pkl.py` - Python reader to load gridded CRUTEM5 and output area-weighted LAT to .pkl.
* `crutem_gridded_2_pkl.py` - Python reader to load gridded CRUTEM5, mask to land and compute GMST (LAT) with various area-weighting schema, and output to .pkl. Used in area-weighted GMST comparisons.
* `wgs84_area_weighting_vs_cosine_error.py` - Python research code to compare various area-averaging schema.
* `make_contactsheet.py` - Python plotting function to generate a per calendar month contactsheet for a year of data.
* `make_facetgrid.py` - Python plotting function to generate a small multiples plot of all momthly gridded CRUTEM5 maps.
* `make_gif.py` - Python function to generate an animated .gif of monthly quadruple overlay maps.

## Instructions for use

The first step is to clone the latest glosat-analysis-station-validation code and step into the installed Github directory: 

    $ git clone https://github.com/patternizer/glosat-analysis-station-validation.git
    $ cd glosat-analysis-station-validation

Then create a DATA/ directory and copy to it the required datasets listed in plot_glosat_analysis_vs_stations.py (available on request).

### Using Standard Python

The code is designed to run in an environment using Miniconda3-latest-Linux-x86_64.

    $ python merge_netcdfs.py
    $ python stations_to_gridded_netcdf.py
    $ python crutem_2_pkl.py
    $ python lat_2_pkl.py
    $ python mat_2_pkl.py
    $ python analysis_2_pkl.py
    $ python nao_2_pkl.py
    $ python soi_2_pkl.py
    $ python plot_glosat_analysis_vs_stations_vs_mat.py
    $ python plot_glosat_stations_vs_gridded.py (optional)    
    $ python crutem_gridded_2_pkl.py (optional)
    $ python aod_2_pkl.py (optional)
    $ python wgs84_area_weighting_vs_cosine_error.py (optional)
    $ python plot_stats.py (optional)
    $ python plot_contactsheet.py (optional)
    $ python plot_facetgrid.py (optional)
    $ python make_gif.py (optional)
    
### A note on Cartopy and Conda environments

Due to a Cartopy conflict with recent package versions of Matplotlib and GDAL, it is recommended that you work-around this by creating a new Conda environment and not install GDAL for this code. Dependencies: pandas, matplotlib, cartopy, scipy.

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)


