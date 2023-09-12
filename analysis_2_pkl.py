#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: analysis_gmst_counts_ssim_2_pkl.py
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
 
nc_file = 'DATA/glosat-analysis/GloSATref.1.0.0.0-alpha.4.analysis.analysis.anomalies.ensemble_median.nc'
nc_var = 'tas_median'

#----------------------------------------------------------------------------
# METHODS
#----------------------------------------------------------------------------

def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid defined by WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    
    Input
    -----
    lat: vector or latitudes in degrees      
    
    Output
    ------
    r: vector of radius in meters
    
    '''

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric (see equation 3-110 in WGS84)
    lat = np.deg2rad( lat )
    lat_gc = np.arctan( (1-e2) * np.tan(lat) )

    # radius equation (see equation 3-107 in WGS84)
    r = ( a * (1 - e2)**0.5 ) / ( 1 - (e2 * np.cos(lat_gc)**2) )**0.5 
        
    return r

def area_grid(lat, lon):
    """
    Calculate the area of each grid cell (in meters)
    Based on the function in https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    
    Input 
    -----
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output (Xarray)
    ------
    area: grid-cell area in square-meters with dimensions [lat,lon]    
    """

    xlon, ylat = np.meshgrid( lon, lat )
    R = earth_radius( ylat )
    dlat = np.deg2rad( np.gradient( ylat, axis=0) ) 
    dlon = np.deg2rad( np.gradient( xlon, axis=1) )
    dy = dlat * R
    dx = dlon * R * np.cos( np.deg2rad( ylat ) )
    area = dy * dx

    xda = xr.DataArray(
        area,
        dims=[ "latitude", "longitude" ],
        coords={ "latitude": lat, "longitude": lon },
        attrs={ "long_name": "area_per_pixel", "description": "area per pixel", "units": "m^2",},
    )
    return xda
        
#--------------------------------------------------------------------------
# COMPUTE: area weights ( using WGS84 oblate sphere )
#--------------------------------------------------------------------------

lats = np.arange( -90 + (latstep/2), 90 + (latstep/2), latstep )
lons = np.arange( -180 + (lonstep/2), 180 + (lonstep/2), lonstep )
grid_cell_area = area_grid( lats, lons ) 

#----------------------------------------------------------------------------
# LOAD: GloSAT analysis netCDF
#----------------------------------------------------------------------------

ds = xr.open_dataset( nc_file, decode_cf=True)
par = ds[nc_var]

#----------------------------------------------------------------------------
# SET: time vector
#----------------------------------------------------------------------------

t_vec = pd.date_range(start=str(year_start), end=str(year_end), freq='MS' )[0:-1]

#----------------------------------------------------------------------------
# INITIALISE: empty dec map for SSIM calculation
#----------------------------------------------------------------------------

dec_map = np.ones([ n_lat, n_lon ]) * np.nan # initialise zero map for SSIM crossing year boundary

gmst_vec = []
count_vec = []
ssim_vec = []

#----------------------------------------------------------------------------
# COMPUTE: area-weighted GMST and grid cell count
#----------------------------------------------------------------------------

for k in range(par.shape[0]):

    # COMPUTE: SSIM    

    if k == 0:

        map1 = dec_map
        map2 = np.array( par[k,:,:] )
            
    else:

        map1 = np.array( par[k-1,:,:] )
        map2 = np.array( par[k,:,:] )

    # CATER: for first map having no prior map (December previous year) to correlate against

    if np.nansum(map1) == 0.0:    

        image1 = map1
        image2 = np.nan_to_num(map2, copy=True, nan=0.0, posinf=None, neginf=None)
        
    else:

        image1 = np.nan_to_num(map1, copy=True, nan=0.0, posinf=None, neginf=None)
        image2 = np.nan_to_num(map2, copy=True, nan=0.0, posinf=None, neginf=None)
            
    ssim = structural_similarity(image1, image2)

    # COMPUTE: GMST and extract count    

    v = par[k,:,:]     
    mask = np.isfinite( v )
    masked_area = grid_cell_area.where(mask).sum(['latitude','longitude'])
    gmst = ( ( v.where(mask) * grid_cell_area.where(mask) ).sum(['latitude','longitude']) / masked_area ).values + 0
    count = mask.sum().values + 0

    gmst_vec.append( gmst )
    count_vec.append( count )
    ssim_vec.append( ssim )

df_gmst_analysis = pd.DataFrame({ 'gmst':gmst_vec }, index=t_vec)
df_count_analysis = pd.DataFrame({ 'count':count_vec }, index=t_vec)
df_ssim_analysis = pd.DataFrame({ 'ssim':ssim_vec }, index=t_vec)

#----------------------------------------------------------------------------
# SAVE: GMST, count and SSIM timeseries to .pkl
#----------------------------------------------------------------------------

df_gmst_analysis.to_pickle( 'df_gmst_analysis.pkl', compression='bz2' )
df_count_analysis.to_pickle( 'df_count_analysis.pkl', compression='bz2' )
df_ssim_analysis.to_pickle( 'df_ssim_analysis.pkl', compression='bz2' )

# -----------------------------------------------------------------------------
# Print library verions
# -----------------------------------------------------------------------------

print("numpy      : ", np.__version__) 
print("pandas     : ", pd.__version__) 
print("xarray     : ", xr.__version__)

# -----------------------------------------------------------------------------
print('** END')



