#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: stations_to_gridded_netcdf.py
#------------------------------------------------------------------------------
# Version 0.1
# 5 March, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import pickle
from datetime import datetime
import netCDF4

# Statisticslibraries:
from skimage.metrics import structural_similarity

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

year_start = 1781
year_end = 2022

latstep = 5
lonstep = 5
n_lat = int(180/latstep)
n_lon = int(360/lonstep)        
lats = np.arange( -90 + (latstep/2), 90 + (latstep/2), latstep )
lons = np.arange( -180 + (lonstep/2), 180 + (lonstep/2), lonstep )
lat_edges = np.arange( -90, 90 + latstep, latstep )        
lon_edges = np.arange( -180, 180 + lonstep, lonstep )     
 
filename_anom = 'DATA/df_anom_qc.pkl'     	# INPUT
filename_nc = 'GloSATLAT_stations_gridded.nc' # OUTPUT
        
#----------------------------------------------------------------------------
# LOAD: GloSAT anomalies station dataframe
#----------------------------------------------------------------------------

print('loading station anomalies ...')

df_anom = pd.read_pickle( filename_anom, compression='bz2' ) 

#----------------------------------------------------------------------------
# SET: time vector
#----------------------------------------------------------------------------

t_vec = pd.date_range(start=str(year_start), end=str(year_end), freq='MS' )[0:-1]
#t_vec = pd.date_range(start=str(1781), end=str(2022), freq='MS' )[0:-1]

#------------------------------------------------------------------------------
# INITIALISE: NaN array to hold gridded data and Int vec to hold counts
#------------------------------------------------------------------------------

tas_median = np.zeros([ len(t_vec), n_lat, n_lon]) * np.nan
n_stations_vec = np.zeros( len(t_vec) ).astype(int)

#----------------------------------------------------------------------------
# AREA-AVERAGE: stations and store in array
#----------------------------------------------------------------------------

for year in np.arange(year_start,year_end):
 
    w = df_anom[ df_anom['year'] == year ]
    X = w.stationlon.values
    Y = w.stationlat.values

    for j in range(12):

        k = ( (year-1781) ) * 12 + j
        
        # EXTRACT: station anomalies
 
        Z = w[str(j+1)].values   
        
        # FILTER: stations with no data
    
        da = pd.DataFrame({'X':X, 'Y':Y, 'Z':Z}).dropna()        
        count_stations = len(da)
        
        # REGRID & AVERAGE: station data        
            
#       db = da.groupby( [ pd.cut(da.X, lon_edges, include_lowest=True, right=True), pd.cut(da.Y, lat_edges, include_lowest=True, right=True) ] )['Z'].mean()
        db = da.groupby( [ pd.cut(da.X, lon_edges, include_lowest=True), pd.cut(da.Y, lat_edges, include_lowest=True) ] )['Z'].mean()
        v_stations = np.reshape(db.values, [n_lon, n_lat]).T            
    
        '''    
    
        v_stations = np.zeros( [ n_lat, n_lon ] ) * np.nan        
        for ii in range(len(lon_edges)-1):
    
            for jj in range(len(lat_edges)-1):

                # INEQUALITY: conditions to avoid double counting on grid cell boundary                

                if ( ii == range(len(lon_edges)-1)[-1] ) & ( jj < range(len(lat_edges)-1)[-1] ):  
                    db = da[ ( ( da.X >= lon_edges[ii] ) & ( da.X <= lon_edges[ii+1] ) ) & ( ( da.Y >= lat_edges[jj] ) & ( da.Y < lat_edges[jj+1] ) ) ]
                elif ( ii == range(len(lon_edges)-1)[-1] ) & ( jj == range(len(lat_edges)-1)[-1] ):  
                    db = da[ ( ( da.X >= lon_edges[ii] ) & ( da.X <= lon_edges[ii+1] ) ) & ( ( da.Y >= lat_edges[jj] ) & ( da.Y <= lat_edges[jj+1] ) ) ]
                elif ( ii < range(len(lon_edges)-1)[-1] ) & ( jj == range(len(lat_edges)-1)[-1] ):  
                    db = da[ ( ( da.X >= lon_edges[ii] ) & ( da.X < lon_edges[ii+1] ) ) & ( ( da.Y >= lat_edges[jj] ) & ( da.Y <= lat_edges[jj+1] ) ) ]
                else:                    
                    db = da[ ( ( da.X >= lon_edges[ii] ) & ( da.X < lon_edges[ii+1] ) ) & ( ( da.Y >= lat_edges[jj] ) & ( da.Y < lat_edges[jj+1] ) ) ]

                if len(db) == 0:                       
                    grid_cell_mean = np.nan
                elif len(db) == 1:
                    grid_cell_mean = db['Z']
                else:                    
                    grid_cell_mean = np.nanmean(db['Z'])
    
                v_stations[jj,ii] = grid_cell_mean

        '''
    
        tas_median[k,:,:] = v_stations
        n_stations_vec[k] = count_stations

#------------------------------------------------------------------------------
# SAVE: station count timeseries
#------------------------------------------------------------------------------
    
df_count_stations = pd.DataFrame({'count':n_stations_vec}, index=t_vec)
df_count_stations.to_pickle( 'df_count_stations.pkl', compression='bz2' )

#------------------------------------------------------------------------------
# EXPORT: Xarray dataset to netCDF-4
#------------------------------------------------------------------------------

df_stations_gridded = xr.Dataset(
    { "tas_median": ( ("time", "latitude", "longitude"), tas_median, ), },
    coords={"latitude":lats, "longitude":lons, 'time':t_vec }
)
df_stations_gridded.to_netcdf( filename_nc, mode='w', format='NETCDF4')

# -----------------------------------------------------------------------------
# Print library verions
# -----------------------------------------------------------------------------

print("numpy      : ", np.__version__) 
print("pandas     : ", pd.__version__) 
print("xarray     : ", xr.__version__)

# -----------------------------------------------------------------------------
print('** END')



