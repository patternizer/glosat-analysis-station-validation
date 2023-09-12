#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: merge_netcdfs.py
#------------------------------------------------------------------------------
# Version 0.1
# 5 March, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

import numpy
import xarray
import netCDF4

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

input_nc_files = 'DATA/glosat-analysis-alpha-4-infilled/median_fields/GloSATref.1.0.0.0-alpha.4.analysis.analysis.anomalies.ensemble_median*.nc'
output_nc = 'GloSATref.1.0.0.0-alpha.4.analysis.analysis.anomalies.ensemble_median.nc'

#----------------------------------------------------------------------------
# CONCATENATE: analysis into single netCDF
#----------------------------------------------------------------------------

ds = xarray.open_mfdataset( input_nc_files, combine = 'nested', concat_dim="time" )
ds.to_netcdf( output_nc )

# -----------------------------------------------------------------------------
print('** END')

