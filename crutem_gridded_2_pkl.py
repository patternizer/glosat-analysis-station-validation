#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: crutem_gridded_gmst_counts_ssim_2_pkl.py
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

fontsize = 16
cmap = 'bwr'
#cmap = 'RdBu_r'
dpi = 300 # 144,300,600

nsmooth = 10 # years for yearly means

year_start = 1781
year_end = 2022

latstep = 5
lonstep = 5
n_lat = int(180/latstep)
n_lon = int(360/lonstep)  

sftof_nc = 'DATA/sftof.nc'
crutem_nc = 'DATA/GloSAT.p04c.EBCv0.6.LEKnorms21Nov22_alternativegrid-178101-202112.nc'

#----------------------------------------------------------------------------
# METHODS
#----------------------------------------------------------------------------

def weighted_temporal_mean(ds, var):
    """
    weight by days in each month
    """
    month_length = ds.time.dt.days_in_month
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    obs = ds[var]
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)
    obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")

    return obs_sum / ones_out

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

def array_to_xarray(lat, lon, da):

    xda = xr.DataArray(
        da,
        dims=[ "latitude", "longitude" ],
        coords={ "latitude": lat, "longitude": lon },
        attrs={ "long_name": "data_per_pixel", "description": "data per pixel", "units": "",},
    )
    return xda
        
#--------------------------------------------------------------------------
# COMPUTE: area weights ( using WGS84 oblate sphere )
#--------------------------------------------------------------------------

lats = np.arange( -90 + (latstep/2), 90 + (latstep/2), latstep )
lons = np.arange( -180 + (lonstep/2), 180 + (lonstep/2), lonstep )
grid_cell_area = area_grid( lats, lons ) 

# CHECK: compare standard and weighted means for e.g. masked NH

total_area = grid_cell_area.sum(['latitude','longitude'])
grid_cell_weights_total = grid_cell_area / total_area

ones = grid_cell_area * 0.0 + 1.0
mask = ones.latitude > 0
masked_area = grid_cell_area[mask].sum(['latitude','longitude'])
grid_cell_weights_masked = grid_cell_area[mask] / masked_area

masked_fraction = masked_area / total_area # = 0.5

# CHECK: weight closure condition (sum to 1)

print( 'Sum of weights (lat) =', grid_cell_weights_masked.values.sum(axis=0).sum() ) # = 1
print( 'Sum of weights (lon) =', grid_cell_weights_masked.values.sum(axis=1).sum() ) # = 1

# CHECK: standard and weighted mean equivalence for uniform field

ones_standard_mean = ones[mask].mean(['latitude','longitude']) # = 1
ones_weighted_mean_1 = ( ones[mask] * grid_cell_area[mask] ).sum(['latitude','longitude']) / masked_area # = 1
ones_weighted_mean_2 = ( ones[mask] * grid_cell_weights_masked ).sum(['latitude','longitude']) # = 1

# COMPUTE: equatorial weight

equatorial_weight = grid_cell_weights_total.values.max() 

#--------------------------------------------------------------------------
# COMPUTE: zonal latitude weights from Cos( latitude ) * land/sea mask
#--------------------------------------------------------------------------

lat_edges = np.arange( -90, 90 + latstep, latstep )        
lon_edges = np.arange( -180, 180 + lonstep, lonstep )        

zones = np.arange( -90, 90 + latstep, latstep) # zone boundaries
zones_per_degree = np.arange( -90, 90 + 1, 1) # zone boundaries oer degree
zone_bins = np.arange( -90 + (latstep/2), 90 + (latstep/2), latstep) # zone midpoints
zone_bins_per_degree = np.arange( -90 + (1/2), 90 + (1/2), 1) # zone midpoints per degree
        
zonal_lat_weight_per_degree = []
for idx in zone_bins_per_degree:
    zonal_lat_weight_per_degree.append( np.abs( np.cos( (idx/180) * np.pi ) ) )        
zonal_lat_weight_per_degree = np.array( zonal_lat_weight_per_degree )

zonal_lat_weight = []
for idx in zone_bins:
    zonal_lat_weight.append( np.abs( np.cos( (idx/180) * np.pi ) ) ) 
zonal_lat_weight = np.array( zonal_lat_weight )
        
# COMPUTE: Zonal land fraction weights
        
nc = netCDF4.Dataset( sftof_nc, "r") # 1x1 degree
lats = nc.variables["lat"][:]
lons = nc.variables["lon"][:]
sftof = np.ma.filled(nc.variables["sftof"][:,:],0.0)
nc.close()
zonal_land_weight_per_degree = np.nanmean( sftof, axis=1 ) # per degree
zonal_land_weight_per_degree = zonal_land_weight_per_degree[::-1]/100

zonal_land_weight = []
zones_dict = dict( zip( zones_per_degree, np.arange( len( zonal_land_weight_per_degree ) + 1 ) ) )
for i in range(len(zones)-1):  
    weight_land = np.nanmean( zonal_land_weight_per_degree[ zones_dict[zones[i]] : zones_dict[zones[i]+latstep] ] )
    zonal_land_weight.append( weight_land ) 
zonal_land_weight = np.array( zonal_land_weight )

# COMPUTE: Zonal ocean fraction weights

zonal_ocean_weight_per_degree = 1.0 - np.array( zonal_land_weight_per_degree )
zonal_ocean_weight = 1.0 - np.array( zonal_land_weight )
        
# COMBINE: weights
        
zonal_lat_land_weight = zonal_lat_weight * zonal_land_weight
zonal_lat_ocean_weight = zonal_lat_weight * zonal_ocean_weight

zonal_lat_land_weight_per_degree = zonal_lat_weight_per_degree * zonal_land_weight_per_degree
zonal_lat_ocean_weight_per_degree = zonal_lat_weight_per_degree * zonal_ocean_weight_per_degree

# HEMISPHERICAL: weights

zonal_weight_hemisphere = np.array([ 0.49577966, 0.43426979]) # NB: NH,SH weights computed from prior latstep = 90

# COMPUTE: weight array (lat,lon)

zonal_lat_weight_array = np.tile( zonal_lat_weight.reshape(len(zonal_lat_weight),1), n_lon )
zonal_lat_land_weight_array = np.tile( zonal_lat_land_weight.reshape(len(zonal_lat_land_weight),1), n_lon )
zonal_lat_ocean_weight_array = np.tile( zonal_lat_ocean_weight.reshape(len(zonal_lat_ocean_weight),1), n_lon )

lat = np.arange( -90 + (latstep/2), 90 + (latstep/2), latstep )
lon = np.arange( -180 + (lonstep/2), 180 + (lonstep/2), lonstep )
zonal_lat_weight_array = array_to_xarray( lat, lon, zonal_lat_weight_array)

#----------------------------------------------------------------------------
# LOAD: CRUTEM alternative grid netCDF
#----------------------------------------------------------------------------

ds = xr.open_dataset( crutem_nc, decode_cf=True)
par = ds.temperature_anomaly
 
#----------------------------------------------------------------------------
# SET: time vector
#----------------------------------------------------------------------------

t_vec = pd.date_range(start=str(year_start), end=str(year_end), freq='MS' )[0:-1]

#----------------------------------------------------------------------------
# INITIALISE: empty dec map for SSIM calculation
#----------------------------------------------------------------------------

dec_map = np.ones([ n_lat, n_lon ]) * np.nan # initialise zero map for SSIM crossing year boundary

gmst_crutem_gridded_vec = []
gmst_crutem_gridded_cosine_vec = []
counts_crutem_gridded_vec = []
ssim_crutem_gridded_vec = []

#----------------------------------------------------------------------------
# COMPUTE: area-weighted GMST (CRUTEM gridded and cosine weighted)
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
    masked_area_cosine = zonal_lat_weight_array.where(mask).sum(['latitude','longitude'])
    gmst = ( ( v.where(mask) * grid_cell_area.where(mask) ).sum(['latitude','longitude']) / masked_area ).values + 0
    gmst_cosine = ( ( v.where(mask) * zonal_lat_weight_array.where(mask) ).sum(['latitude','longitude']) / masked_area_cosine ).values + 0
    count = mask.sum().values + 0

    gmst_crutem_gridded_vec.append( gmst )
    gmst_crutem_gridded_cosine_vec.append( gmst_cosine )
    counts_crutem_gridded_vec.append( count )
    ssim_crutem_gridded_vec.append( ssim )

df_gmst_crutem_gridded = pd.DataFrame({ 'gmst':gmst_crutem_gridded_vec }, index=t_vec)
df_gmst_crutem_gridded_cosine = pd.DataFrame({ 'gmst':gmst_crutem_gridded_cosine_vec }, index=t_vec)
df_count_crutem_gridded = pd.DataFrame({ 'count':counts_crutem_gridded_vec }, index=t_vec)
df_ssim_crutem_gridded = pd.DataFrame({ 'ssim':ssim_crutem_gridded_vec }, index=t_vec)

#----------------------------------------------------------------------------
# SAVE: CRUTEM gridded and cosine GMST timeseries to .pkl
#----------------------------------------------------------------------------

df_gmst_crutem_gridded.to_pickle( 'df_gmst_crutem_gridded.pkl', compression='bz2' )
df_gmst_crutem_gridded_cosine.to_pickle( 'df_gmst_crutem_gridded_cosine.pkl', compression='bz2' )
df_count_crutem_gridded.to_pickle( 'df_count_crutem_gridded.pkl', compression='bz2' )
df_ssim_crutem_gridded.to_pickle( 'df_ssim_crutem_gridded.pkl', compression='bz2' )

# -----------------------------------------------------------------------------
# Print library verions
# -----------------------------------------------------------------------------

print("numpy      : ", np.__version__) 
print("pandas     : ", pd.__version__) 
print("xarray     : ", xr.__version__)

# -----------------------------------------------------------------------------
print('** END')



