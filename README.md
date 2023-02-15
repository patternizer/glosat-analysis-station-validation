![image](https://github.com/patternizer/glosat-analysis-station-validation/blob/main/tas_median_1848-07.png)
![image](https://github.com/patternizer/glosat-analysis-station-validation/blob/main/nao_yearly_mean.png)
![image](https://github.com/patternizer/glosat-analysis-station-validation/blob/main/ssim.png)

# glosat-analysis-station-validation

Comparison of monthly GloSAT in-filled 5x5 gridded analysis with station anomalies and NAO. Part of land surface air temperature data record validation efforts and ongoing work for the [GloSAT](https://www.glosat.org) project: www.glosat.org. 

## Contents

* `plot_glosat_analysis_vs_stations.py` - python script to read in the GloSAT.analysis.alpha.4 in-filled 5x5 gridded median netCDFs, the GloSAT.p04c.EBC.LEKnormals station anomalies, the Luterbacher et al reconstructed NAO (1658-1900), the Phil Jones NAO series (1821-2022), and plot the station monthly anomalies on the analysis background map. For each map, the structural similarity image (SSIM) index from Scipy is calculated as an indication of the monthly correlation between the analysis and its previous month.
* `plot_glosat_analysis_vs_stations_yearly.py` - Yearly-averaged version of `plot_glosat_analysis_vs_stations.py`. NB: this is prone to error due to temporal coverage biasing the yearly mean. 
* `nao-2-pkl.py` - Python reader to load Luterbacher and Jones monthly NAO indices and output to .pkl a merged timeseries from 1658-2022.
* `make_gif.py` - Python function to generate an animated .gif of monthly map .png output files.

## Instructions for use

The first step is to clone the latest glosat-analysis-station-validation code and step into the installed Github directory: 

    $ git clone https://github.com/patternizer/glosat-analysis-station-validation.git
    $ cd glosat-analysis-station-validation

Then create a DATA/ directory and copy to it the required datasets listed in plot_glosat_analysis_vs_stations.py (available on request).

### Using Standard Python

The code is designed to run in an environment using Miniconda3-latest-Linux-x86_64.

    $ python nao-2-pkl.py
    $ python plot_glosat_analysis_vs_stations.py
    $ python make_gif.py
    
### A note on Cartopy and Conda environments

Due to a Cartopy conflict with Matplotlib with GDAL installed, it is recommended that you work-around this by creating a new Conda environment and not installing GDAL for this code. Dependencies include: pandas, matplotlib, cartopy, scipy.

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)


