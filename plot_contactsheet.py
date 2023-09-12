#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_contactsheet.py
#------------------------------------------------------------------------------
# Version 0.1
# 24 February, 2023
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

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

fontsize = 12
cmap = 'bwr'

vmin = -6.0        
vmax = 6.0    
resolution = '110m' # 110, 50 or 10km
dpi = 600 # 144,300,600

#year_start = 1781
year_start = 2021
year_end = 2022

filename_anom = 'DATA/df_anom_qc.pkl'     
filename_nao = 'DATA/df_nao_1781_2022.pkl' 
filename_soi = 'DATA/df_soi_1866_2022.pkl' 
sftof_file = 'DATA/sftof.nc'
path_analysis = 'DATA/glosat-analysis-alpha-4-infilled/median_fields/'
path_mat = 'DATA/glosat-mat/'
glosat_version = 'GloSAT.p04c.EBC.LEKnormals'

use_darktheme = False
use_credits = False
    
projection = 'robinson'

if projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0)
if projection == 'europp': p = ccrs.EuroPP()
if projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0)
if projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0)
if projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0)
if projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0)
if projection == 'northpolarstereo': p = ccrs.NorthPolarStereo()
if projection == 'orthographic': p = ccrs.Orthographic(0,0)
if projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0)
if projection == 'robinson': p = ccrs.Robinson(central_longitude=0)
if projection == 'southpolarstereo': p = ccrs.SouthPolarStereo()    

use_darktheme = False
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

#----------------------------------------------------------------------------
# CREDITS
#----------------------------------------------------------------------------

datastr = r'$\bf{Data}$' + ' (mat ▫): GloSATMAT.2.4.0.0 ensemble mean (NOC)'  
baselinestr = r'$\bf{Baseline}$' + ': 1961-1990'        
authorstr = r'$\bf{Graphic}$' + ': Michael Taylor, CRU/UEA' + ' -- ' + titletime

#--------------------------------------------------------------------------
# LOAD: GloSATMAT.2.4.0.0
#--------------------------------------------------------------------------

print('loading MAT ...')
    
#ncfile_mat = path_mat + 'GloSATMAT_2.4.0.0_anomaly_ensmean_b1961_1990.nc'
ncfile_mat = path_mat + 'GloSATMAT_2.4.0.0_anomaly_b1961_1990.nc'
    
ds_mat = xr.open_dataset(ncfile_mat, decode_cf=True)
par_mat = ds_mat.t2m_anomaly
lon_mat = ds_mat.longitude
lat_mat = ds_mat.latitude
time_mat = ds_mat.time    


#----------------------------------------------------------------------------
# EXTRACT: time slice
#----------------------------------------------------------------------------

par = par_mat.isel( time=par_mat.time.dt.year.isin( np.arange(year_start,year_end+1) ) )

#----------------------------------------------------------------------------
# PLOT: contactsheet
#----------------------------------------------------------------------------

'''
borders = cf.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resolution, facecolor='none', alpha=1)
land = cf.NaturalEarthFeature('physical', 'land', scale=resolution, edgecolor='k', facecolor=cf.COLORS['land'])
ocean = cf.NaturalEarthFeature('physical', 'ocean', scale=resolution, edgecolor='none', facecolor=cf.COLORS['water'])
lakes = cf.NaturalEarthFeature('physical', 'lakes', scale=resolution, edgecolor='b', facecolor=cf.COLORS['water'])
rivers = cf.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=resolution, edgecolor='b', facecolor='none')
'''

n_years = np.floor(par.shape[0]/12).astype(int)
#col_wrap = 4
#n_cols = col_wrap
#n_rows = int( np.ceil( n_years / col_wrap ) )

n_cols = 3
n_rows = 4

for i in range(n_years):
        
    year = par.time[i*12].dt.year.values + 0

    titlestr = str( year )
    figstr = 'mat_contactsheet' + '_' + str(year) +'.png'        

    fig, axs = plt.subplots( n_rows, n_cols, figsize=(13.33,7.5), subplot_kw=dict(projection=p))    
    # PowerPoint:            fontsize = 18; fig = plt.figure(figsize=(13.33,7.5), dpi=144); plt.savefig('figure.png', bbox_inches='tight')
    # Posters  (vectorized): fontsize = 18; fig = plt.figure(figsize=(13.33,7.5), dpi=600); plt.savefig('my_figure.svg', bbox_inches='tight')                          
    # Journals (vectorized): fontsize = 18; fig = plt.figure(figsize=(3.54,3.54), dpi=300); plt.savefig('my_figure.svg', bbox_inches='tight')     
        
    for j in range(12):

        r = j // n_cols # row index
        c = j % n_cols  # col index

        v = par[i*12+j,:,:]
        vmin = -6.0
        vmax = 6.0
        #x, y = np.meshgrid(lon_mat,lat_mat)        

        '''             
        axs[r,c].set_global()  
        axs[r,c].add_feature(land, facecolor='grey', linestyle='-', linewidth=0.1, edgecolor='k', alpha=1, zorder=1)
        axs[r,c].add_feature(ocean, facecolor='cyan', alpha=1, zorder=1)
        #axs[r,c].add_feature(lakes)
        #axs[r,c].add_feature(rivers, linewidth=0.5)
        #axs[r,c].add_feature(borders, linestyle='-', linewidth=0.1, edgecolor='k', alpha=1, zorder=2)         
        '''
        
        g = v.plot( ax = axs[r,c], transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap=cmap, add_colorbar=True, cbar_kwargs={'orientation':'vertical','extend':'both','shrink':1, 'pad':0.05}) 
        cb = g.colorbar
        #if ( j % ncols + 1 ) != n_cols: cb.remove()   
        cb.set_label(label=r'Anomaly [$^{\circ}$C]', fontsize=fontsize)
        cb.ax.tick_params(labelsize=fontsize)

        '''
        axs[r,c].coastlines(resolution=resolution, color='k', linestyle='-', linewidth=0.5, edgecolor='k', alpha=1, zorder=100)                                                                                  
        axs[r,c].add_feature(borders, linestyle='-', linewidth=0.1, edgecolor='k', alpha=1, zorder=100)                           
        gl = axs[r,c].gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.1, color='purple', alpha=1, linestyle='-', zorder=1000)
        gl.top_labels = False; gl.bottom_labels = False; gl.left_ylabels = False; gl.right_ylabels = False
        gl.xlines = True; gl.ylines = True
        gl.xlocator = mticker.FixedLocator(np.linspace(-180,180,73)) # every 5 degrees
        gl.ylocator = mticker.FixedLocator(np.linspace(-90,90,37))   # every 5 degrees
        gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
        '''
        
        parallels = np.linspace(-90,90,37)
        meridians = np.linspace(-180,180,73)
        gl = axs[r,c].gridlines(crs=ccrs.PlateCarree(), xlocs=meridians, ylocs=parallels, linestyle="-", linewidth=0.1, color='purple', alpha=1)        
        axs[r,c].add_feature(cf.LAND, facecolor='grey')
        axs[r,c].add_feature(cf.OCEAN, facecolor='cyan')
        axs[r,c].add_feature(cf.COASTLINE, edgecolor="k", linewidth=0.5)
        axs[r,c].add_feature(cf.BORDERS, edgecolor="k", linewidth=0.1)        
        
        axs[r,c].set_title( str(year) + '-' + str(j+1).zfill(2), fontsize=fontsize, color=default_color, y=1.0, fontweight='bold') 
    
    if use_credits == True:
        
        if dpi == 144: xstart = dpi; ystart=10; ystep = 20
        elif dpi == 300: xstart = dpi; ystart=10; ystep = 40
        elif dpi == 600: xstart = dpi; ystart=10; ystep = 80
                
        plt.annotate(datastr, xy=(xstart,ystart+ystep*3), xycoords='figure pixels', color=default_color, fontsize=8)  
        plt.annotate(baselinestr, xy=(xstart,ystart+ystep*2), xycoords='figure pixels', color=default_color, fontsize=8)  
        plt.annotate(authorstr, xy=(xstart,ystart+ystep*1), xycoords='figure pixels', color=default_color, fontsize=8)           
           
    fig.subplots_adjust(left=None, bottom=0.1, right=None, top=None, wspace=None, hspace=None)
    plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
print('** END')
