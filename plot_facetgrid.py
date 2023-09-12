#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_facetgrid.py
#------------------------------------------------------------------------------
# Version 0.3
# 6 March, 2023
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

# Color libraries:
import cmocean
    
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

# plt.style.use('fivethirtyeight')

import seaborn as sns; sns.set()

# Mapping libraries:
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cf
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

path_analysis = 'DATA/glosat-analysis/'
path_lat = 'DATA/glosat-lat/'
path_mat = 'DATA/glosat-mat/'

fontsize = 12
vmin = -3.0        
vmax = 3.0    

#year_start = 1870 # WMO 150th anniversary
year_start = 1781
year_end = 2022

dpi = 600                   # [144,300,600]
resolution = '110m'         # ['110m','50m','10m']
use_gridlines = True        # [True,False]
use_cmocean = True          # [True,False] False --> cmap = 'bwr' 
use_dataset = 'lat'    # ['analysis','lat','mat']
use_lat = True              # [True,False] --> prefer data-driven LAT to alternative grid LAT
use_darktheme = True        # [True,False]
use_projection = 'robinson' # see projection list below

# SET: projection
    
if use_projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0)
if use_projection == 'europp': p = ccrs.EuroPP()
if use_projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0)
if use_projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0)
if use_projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0)
if use_projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0)
if use_projection == 'northpolarstereo': p = ccrs.NorthPolarStereo()
if use_projection == 'orthographic': p = ccrs.Orthographic(0,0)
if use_projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0)
if use_projection == 'robinson': p = ccrs.Robinson(central_longitude=0)
if use_projection == 'southpolarstereo': p = ccrs.SouthPolarStereo()    

# LOAD: Natural Earth features

borders = cf.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resolution, facecolor='none', alpha=1)
land = cf.NaturalEarthFeature('physical', 'land', scale=resolution, edgecolor='k', facecolor=cf.COLORS['land'])
ocean = cf.NaturalEarthFeature('physical', 'ocean', scale=resolution, edgecolor='none', facecolor=cf.COLORS['water'])
lakes = cf.NaturalEarthFeature('physical', 'lakes', scale=resolution, edgecolor='b', facecolor=cf.COLORS['water'])
rivers = cf.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=resolution, edgecolor='b', facecolor='none')

# SET: cmap

if use_cmocean == True:
    #cmap_full = cmocean.cm.curl
    cmap_full = cmocean.cm.balance
    cmap = cmocean.tools.crop_by_percent(cmap_full, 20, which='both') # clip 20% from ends
else:
    #cmap = 'RdBu_r'
    cmap = 'bwr'

# SET: theme

if use_darktheme == True:
	default_color = 'white'
else:
	default_color = 'black'     
    
# Calculate current time

now = datetime.now()
currentdy = str(now.day).zfill(2)
currentmn = str(now.month).zfill(2)
currentyr = str(now.year)
titletime = str(currentdy) + '/' + currentmn + '/' + currentyr    
    
#------------------------------------------------------------------------------
# THEME
#------------------------------------------------------------------------------

if use_darktheme == True:
	
    matplotlib.rcParams['text.usetex'] = False
    rcParams['font.family'] = ['Lato']
#    rcParams['font.family'] = ['sans-serif']
#    rcParams['font.sans-serif'] = ['Avant Garde', 'Lucida Grande', 'Verdana', 'DejaVu Sans' ]    
    plt.rc('text',color='white')
    plt.rc('lines',color='white')
    plt.rc('patch',edgecolor='white')
    plt.rc('grid',color='lightgray')
    plt.rc('xtick',color='white')
    plt.rc('ytick',color='white')
    plt.rc('axes',edgecolor='lightgray')
    plt.rc('axes',facecolor='black')
    plt.rc('axes',labelcolor='white')
    plt.rc('figure',facecolor='black')
    plt.rc('figure',edgecolor='black')
    plt.rc('savefig',edgecolor='black')
    plt.rc('savefig',facecolor='black')

else:

    matplotlib.rcParams['text.usetex'] = False
    rcParams['font.family'] = ['Lato']
#    rcParams['font.family'] = ['sans-serif']
#    rcParams['font.sans-serif'] = ['Avant Garde', 'Lucida Grande', 'Verdana', 'DejaVu Sans' ]    
    plt.rc('savefig',facecolor='white')
    plt.rc('axes',edgecolor='black')
    plt.rc('xtick',color='black')
    plt.rc('ytick',color='black')
    plt.rc('axes',labelcolor='black')
    plt.rc('axes',facecolor='white')

#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------

def weighted_temporal_mean(ds, var):

    """
    weight by days in each month:
    https://ncar.github.io/esds/posts/2021/yearly-averages-xarray/
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)

    # Subset our dataset for our variable
    obs = ds[var]

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")

    # Return the weighted average
    return obs_sum / ones_out
    
#------------------------------------------------------------------------------
# LOAD: GloSAT.analysis.alpha.4 (infilled)
#------------------------------------------------------------------------------

print('loading Analysis ...')
    
ncfile_analysis = path_analysis + 'GloSATref.1.0.0.0-alpha.4.analysis.analysis.anomalies.ensemble_median.nc'    
ds_analysis = xr.open_dataset(ncfile_analysis, decode_cf=True)

# COMPUTE: annual mean (weighted)

par_analysis = weighted_temporal_mean( ds_analysis, 'tas_median')

#------------------------------------------------------------------------------
# LOAD: GloSAT.p04c.EBC.LEKnormals (alternative grid)
#------------------------------------------------------------------------------

print('loading LAT ...')
    
if use_lat == True:
        
    ncfile_lat = path_lat + 'GloSATLAT_stations_gridded.nc'    
    var = 'tas_median'
    
else:
    
    ncfile_lat = path_lat + 'GloSAT.p04c.EBCv0.6.LEKnorms21Nov22_alternativegrid-178101-202112.nc'    
    var = 'temperature_anomaly'
    
ds_lat = xr.open_dataset(ncfile_lat, decode_cf=True)

# COMPUTE: annual mean (weighted)

par_lat = weighted_temporal_mean( ds_lat, var )

#------------------------------------------------------------------------------
# LOAD: GloSATMAT.2.4.0.0
#------------------------------------------------------------------------------

print('loading MAT ...')
    
#ncfile_mat = path_mat + 'GloSATMAT_2.4.0.0_anomaly_ensmean_b1961_1990.nc'
ncfile_mat = path_mat + 'GloSATMAT_2.4.0.0_anomaly_b1961_1990.nc'    
ds_mat = xr.open_dataset(ncfile_mat, decode_cf=True)

# COMPUTE: annual mean (weighted)

par_mat = weighted_temporal_mean( ds_mat, 't2m_anomaly')

#------------------------------------------------------------------------------
# SELECT: dataset
#------------------------------------------------------------------------------

if use_dataset == 'analysis':
    par = par_analysis
elif use_dataset == 'lat':
    par = par_lat
elif use_dataset == 'mat':
    par = par_mat

#------------------------------------------------------------------------------
# EXTRACT: time slice
#------------------------------------------------------------------------------

par = par.isel( time=par.time.dt.year.isin( np.arange(year_start,year_end+1) ) )
#par = par.isel( time=par.time.dt.month.isin( [1] ) )

N = par.shape[0]

#------------------------------------------------------------------------------
# CREDITS
#------------------------------------------------------------------------------

if use_dataset == 'analysis':
    datastr = r'$\bf{Data}$' + ' (analysis □): GloSAT.analysis.alpha.4 (UKMO-HC)'  
elif use_dataset == 'lat':
    datastr = r'$\bf{Data}$' + ' (lat □): GloSAT.p04c.EBC.LEKnormals (CRU/UEA, UYork)'  
elif use_dataset == 'mat':
    datastr = r'$\bf{Data}$' + ' (mat □): GloSATMAT.2.4.0.0 ensemble mean (NOC)'  

baselinestr = r'$\bf{Baseline}$' + ': 1961-1990'        
authorstr = r'$\bf{Graphic}$' + ': Michael Taylor, CRU/UEA' + ' -- ' + titletime

#------------------------------------------------------------------------------
# PLOT: facetgrid
#------------------------------------------------------------------------------

i=0
j=0
#col_wrap = 12 # months
col_wrap = 10 # years 

if use_dataset == 'analysis':
    figstr = 'glosat_analysis_facetgrid_yearly_1781_2021.png'        
elif use_dataset == 'lat':
    figstr = 'glosat_lat_facetgrid_yearly_1781_2021.png'        
elif use_dataset == 'mat':
    figstr = 'glosat_mat_facetgrid_yearly_1781_2021.png'        

fig, axs = plt.subplots( nrows=int(N/col_wrap) + 1, ncols=col_wrap + 1, figsize=(8,10), subplot_kw=dict(projection=p), constrained_layout=False )    
#fig, axs = plt.subplots( nrows=int(N/col_wrap) + 1, ncols=col_wrap + 1, figsize=(15,10), subplot_kw=dict(projection=p), constrained_layout=False )    

for ax in axs.flat:

    if i % (col_wrap + 1) == 0:

        # REMOVE: Cartopy projection axis and re-add with no projection (for text annotation in figure space)

        rows, cols, start, stop = ax.get_subplotspec().get_geometry()
        ax.remove()
        ax = fig.add_subplot(rows, cols, start+1)
        ax.xaxis.set_ticks([]) 
        ax.yaxis.set_ticks([]) 
        ax.spines[['left','right','top','bottom']].set_color('none')
        tx = ax.text(0.1, 0.3, str( par.time.dt.year[i-j].values+0 )[0:3] + '0s', fontsize=fontsize, fontweight='bold' )

        j += 1            

    else:
        
        if ( (i-j) == 0 ) & (year_start==1781):
            
            print('adding blank 1780')
            ax.set_title(None)           
            
#        elif (i-j-1) < len(par):
        elif (i-j) < len(par):

            #ax.set_global()  
            ax.add_feature(land, facecolor='grey', linestyle='-', linewidth=0.1, edgecolor='k', alpha=1, zorder=1)
            ax.add_feature(ocean, facecolor='grey', alpha=0.7, zorder=1)            
#            im = par[i-1-j,:,:].plot(ax=ax, alpha=1, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree(), add_colorbar=False, zorder=10 ) 
            im = par[i-j,:,:].plot(ax=ax, alpha=1, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree(), add_colorbar=False, zorder=10 ) 
            ax.coastlines(resolution=resolution, color='k', linestyle='-', linewidth=0.1, edgecolor='k', alpha=1, zorder=100)                                                                                  
            ax.add_feature(borders, linestyle='-', linewidth=0.05, edgecolor='k', alpha=1, zorder=100)         
            #ax.set_title( str(par.time[i-1-j].values)[0:4], fontsize=8 )
            ax.set_title(None)           

            if use_gridlines:
                
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.01, color='purple', alpha=1, linestyle='-', zorder=1000)
                gl.top_labels = False; gl.bottom_labels = False; gl.left_ylabels = False; gl.right_ylabels = False
                gl.xlines = True; gl.ylines = True
                gl.xlocator = mticker.FixedLocator(np.linspace(-180,180,73)) # every 5 degrees
                gl.ylocator = mticker.FixedLocator(np.linspace(-90,90,37))   # every 5 degrees
                gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER

        else:

            print('reached end of dataset')
            ax.set_title(None)           

        # REMOVE: Cartopy boundary
        
        ax.spines['geo'].set_edgecolor('pink')
        ax.spines['geo'].set_linewidth(0.1)

    i += 1    
        
fig.text(0.7, 0.08, 'Michael Taylor, CRU/UEA -- ' + titletime, fontsize=8, fontweight='bold' )
    
fig.subplots_adjust( left=None, right=None, top=None, bottom=None, wspace=None, hspace=None )

cbar_ax = fig.add_axes([0.3, 0.02, 0.5, 0.01]) # left, bottom, width, height
cb = fig.colorbar( im, cax=cbar_ax, orientation="horizontal", extend='both', shrink=0.5, pad=0, drawedges=False )
cb.set_label(label=r'Anomaly (from 1961-1990) [$^{\circ}$C]', fontsize=fontsize)
cb.ax.tick_params(labelsize=fontsize, width=1)

'''
if use_dataset == 'analysis':
    plt.suptitle( 'GloSAT( Analysis) ' + str(year_start) + '-' + str(year_end) + ': yearly weighted mean', x=0.24, y=0.92, horizontalalignment='left', verticalalignment='top', fontsize=16, fontweight='bold' )
elif use_dataset == 'lat':
    plt.suptitle( 'GloSAT (LAT) ' + str(year_start) + '-' + str(year_end) + ': yearly weighted mean', x=0.26, y=0.92, horizontalalignment='left', verticalalignment='top', fontsize=16, fontweight='bold' )
elif use_dataset == 'mat':
    plt.suptitle( 'GloSAT (MAT) ' + str(year_start) + '-' + str(year_end) + ': yearly weighted mean', x=0.26, y=0.92, horizontalalignment='left', verticalalignment='top', fontsize=16 , fontweight='bold' )
'''
    
#plt.tight_layout()
plt.savefig(figstr, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
plt.close()

# -----------------------------------------------------------------------------
# Print library verions
# -----------------------------------------------------------------------------

print("numpy      : ", np.__version__) 
print("pandas     : ", pd.__version__) 
print("xarray     : ", xr.__version__)
print("matplotlib : ", matplotlib.__version__)
print("cartopy    : ", cartopy.__version__)
                 
#------------------------------------------------------------------------------
print('** END')
