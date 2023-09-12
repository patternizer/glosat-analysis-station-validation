#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: wgs84_area_weighting_vs_cosine_error.py
#------------------------------------------------------------------------------
# Version 0.1
# 21 February, 2023
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

# Plotting libraries:
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib import colors as mcolors
# %matplotlib inline # for Jupyter Notebooks

import seaborn as sns; sns.set()

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

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

'''
#----------------------------------------------------------------------------
# LOAD: CRUTEM alternative grid netCDF
#----------------------------------------------------------------------------

ds_crutem = xr.open_dataset( crutem_nc, decode_cf=True)
par_crutem = ds_crutem.temperature_anomaly
lon_crutem = ds_crutem.longitude
lat_crutem = ds_crutem.latitude
time_crutem = ds_crutem.time    

gmst_crutem_gridded = []
gmst_crutem_gridded_cosine = []
for k in range(par_crutem.shape[0]):
    v_crutem = par_crutem[k,:,:]     
    mask_crutem = np.isfinite( v_crutem )
    masked_area = grid_cell_area.where(mask_crutem).sum(['latitude','longitude'])
    masked_area_cosine = zonal_lat_weight_array.where(mask_crutem).sum(['latitude','longitude'])
    gmst = ( ( v_crutem.where(mask_crutem) * grid_cell_area.where(mask_crutem) ).sum(['latitude','longitude']) / masked_area ).values + 0
    gmst_cosine = ( ( v_crutem.where(mask_crutem) * zonal_lat_weight_array.where(mask_crutem) ).sum(['latitude','longitude']) / masked_area_cosine ).values + 0
    gmst_crutem_gridded.append(gmst)
    gmst_crutem_gridded_cosine.append(gmst_cosine)

df_gmst_crutem_gridded = pd.DataFrame({'gmst_global':gmst_crutem_gridded}, index=t_vec)
df_gmst_crutem_gridded_cosine = pd.DataFrame({'gmst_global':gmst_crutem_gridded_cosine}, index=t_vec)
'''

#==============================================================================
# PLOTS
#==============================================================================

#------------------------------------------------------------------------------
# PLOT: zonal area-weights
#------------------------------------------------------------------------------

figstr = 'zonal-weighting-scheme.png'
titlestr = 'Zonal weighting: for ' + str(1) + r'$^{\circ}$ and ' + str(latstep) + r'$^{\circ}$ bins with and without land and ocean fraction weighting'
xstr = 'Weight'
ystr = 'Latitude, °N'

fig, ax = plt.subplots(figsize=(15,10))          
plt.fill_betweenx( x1=0, x2=zonal_lat_weight_per_degree * equatorial_weight, y=zone_bins_per_degree, color='white')
plt.fill_betweenx( x1=0, x2=zonal_lat_ocean_weight_per_degree * equatorial_weight, y=zone_bins_per_degree, color='blue', alpha=0.2)
plt.fill_betweenx( x1=0, x2=zonal_lat_land_weight_per_degree * equatorial_weight, y=zone_bins_per_degree, color='green', alpha=0.2)
plt.plot( zonal_lat_weight_per_degree * equatorial_weight, zone_bins_per_degree, marker='o', markerfacecolor='white', ls='-', lw=0.5, color='black', label='Cos(latitude): ' + str(1) + r'$^{\circ}$' )
plt.plot( zonal_lat_land_weight_per_degree * equatorial_weight, zone_bins_per_degree, marker='o', markerfacecolor='white', ls='-', lw=0.5, color='green', label='Cos(latitude) x land fraction: ' + str(1) + r'$^{\circ}$' )
plt.plot( zonal_lat_land_weight * equatorial_weight, zone_bins, marker='.', ls='none', lw=0.5, color='green', label='Cos(latitude) x land fraction: ' + str(latstep) + r'$^{\circ}$' )
plt.plot( zonal_lat_ocean_weight_per_degree * equatorial_weight, zone_bins_per_degree, marker='o', markerfacecolor='white', ls='-', lw=0.5, color='blue', label='Cos(latitude) x ocean fraction: ' + str(1) + r'$^{\circ}$' )
plt.plot( zonal_lat_ocean_weight * equatorial_weight, zone_bins, marker='.', ls='none', lw=0.5, color='blue', label='Cos(latitude) x ocean fraction: ' + str(latstep) + r'$^{\circ}$' )
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='lower right', ncol=1, fontsize=12)
plt.xlabel(xstr, fontsize=fontsize)
plt.ylabel(ystr, fontsize=fontsize)
plt.title( titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=300, bbox_inches='tight')
plt.close(fig)
        
#------------------------------------------------------------------------------
# PLOT: cosine error
#------------------------------------------------------------------------------

figstr = 'zonal-weighting-scheme-cosine_error.png'
titlestr = 'Zonal weighting: cosine error'
xstr = 'Weight'
ystr = 'Latitude, °N'

fig, ax = plt.subplots(figsize=(15,10))          
plt.fill_betweenx( x1=0, x2=zonal_lat_weight * equatorial_weight, y=zone_bins, color='white')
plt.plot( zonal_lat_weight * equatorial_weight, zone_bins, marker='o', markerfacecolor='white', ls='-', lw=0.5, color='black', label='Cos(latitude): ' + str(latstep) + r'$^{\circ}$' )
plt.plot(grid_cell_weights_total.values[:,0], zone_bins, marker='.', color='red', ls='none', label='Oblate sphere: ' + str(latstep) + r'$^{\circ}$' )
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='lower right', ncol=1, fontsize=12)
plt.xlabel(xstr, fontsize=fontsize)
plt.ylabel(ystr, fontsize=fontsize)
plt.title( titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=300, bbox_inches='tight')
plt.close(fig)

# -----------------------------------------------------------------------------
# Print library verions
# -----------------------------------------------------------------------------

print("numpy      : ", np.__version__) 
print("pandas     : ", pd.__version__) 
print("xarray     : ", xr.__version__)
print("matplotlib : ", matplotlib.__version__)
print("seaborn    : ", sns.__version__)

# -----------------------------------------------------------------------------
print('** END')



