#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_glosat_stations_vs_gridded.py
#------------------------------------------------------------------------------
# Version 0.5
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

# Colour libraries:
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

filename_anom = 'DATA/df_anom_qc.pkl'     
filename_nao = 'OUT/df_nao.pkl' 
filename_soi = 'OUT/df_soi.pkl' 
filename_count_stations = 'OUT/df_count_stations.pkl'
filename_count_lat = 'OUT/df_count_lat.pkl'
filename_count_crutem_gridded = 'OUT/df_count_crutem_gridded.pkl'
filename_gmst_lat = 'OUT/df_gmst_lat.pkl' 
filename_gmst_crutem_gridded = 'OUT/df_gmst_crutem_gridded.pkl' 
lat_stations_nc = 'DATA/glosat-lat/GloSATLAT_stations_gridded.nc'
lat_crutem_gridded_nc = 'DATA/glosat-lat/GloSAT.p04c.EBCv0.6.LEKnorms21Nov22_alternativegrid-178101-202112.nc'

fontsize = 10
vmin = -6.0        
vmax = 6.0    

#year_start = 1781
#year_end = 2022
year_start = 2018
year_end = 2019

latstep = 5
lonstep = 5
n_lat = int(180/latstep)
n_lon = int(360/lonstep)        

dpi = 300                   # [144,300,600]
resolution = '10m'          # ['110m','50m','10m']
use_gridlines = True        # [True,False]
use_cmocean = True          # [True,False] False --> 'bwr' 
use_cyan = False             # [True,False] False --> 'grey'
use_dataset = 'lat'         # ['analysis','lat','mat']
use_lat = True              # [True,False] --> prefer data-driven LAT to alternative grid LAT
use_darktheme = False       # [True,False]
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
    
# CALCULATE: current time

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
# LOAD: GloSAT anomalies station dataframe
#----------------------------------------------------------------------------

print('loading station anomalies ...')

df_anom = pd.read_pickle( filename_anom, compression='bz2' ) 

#------------------------------------------------------------------------------    
# LOAD: NAO indices monthly timeseries
#------------------------------------------------------------------------------
# https://crudata.uea.ac.uk/cru/data/nao/viz.htm

print('loading NAO indices ...')

# LOAD: merged pkl dataframe of monthly NAO: Luterbacher reconstructions 1658-1821 and Phil Jones 1821-2022
     
df_nao = pd.read_pickle( filename_nao, compression='bz2' )      

#------------------------------------------------------------------------------    
# LOAD: SOI indices monthly timeseries
#------------------------------------------------------------------------------
# https://crudata.uea.ac.uk/cru/data/soi/

print('loading SOI indices ...')

# LOAD: pkl dataframe of monthly SOI: Phil Jones 1866-2022
     
df_soi = pd.read_pickle( filename_soi, compression='bz2' )      

#------------------------------------------------------------------------------    
# LOAD: GMST(LAT) monthly timeseries calculated from area-averaged stations + WGS84 area-weights
#------------------------------------------------------------------------------

print('loading GMST(LAT) timeseries ...')

# LOAD: pkl dataframe of monthly GMST(LAT) calculated from stations
     
df_gmst_lat = pd.read_pickle( filename_gmst_lat, compression='bz2' )      

#------------------------------------------------------------------------------    
# LOAD: GMST(LAT) monthly timeseries calculated from GloSAT.p04c.EBC.LEKnormals (alternative grid)
#------------------------------------------------------------------------------

print('loading GMST(LAT) CRUTEM gridded timeseries ...')

# LOAD: pkl dataframe of monthly GMST(LAT) calculated GloSAT.p04c.EBC.LEKnormals (alternative grid)
     
df_gmst_crutem_gridded = pd.read_pickle( filename_gmst_crutem_gridded, compression='bz2' )      

#------------------------------------------------------------------------------    
# LOAD: counts
#------------------------------------------------------------------------------

print('loading station and grid cell counts ...')
     
df_count_lat = pd.read_pickle( filename_count_lat, compression='bz2' )      
df_count_crutem_gridded = pd.read_pickle( filename_count_crutem_gridded, compression='bz2' )      
df_count_stations = pd.read_pickle( filename_count_stations, compression='bz2' )      

#----------------------------------------------------------------------------
# LOAD: GloSAT.p04c.EBC.LEKnormals (alternative grid) Xarray --> for station vs gridded comparison
#----------------------------------------------------------------------------

if use_lat == True:

    ds = xr.open_dataset( lat_stations_nc, decode_cf=True)
    par = ds.tas_median
    
else:

    ds = xr.open_dataset( lat_crutem_gridded_nc, decode_cf=True)
    par = ds.temperature_anomaly
      
#==============================================================================
# LOOP: over all years
#==============================================================================

for year in np.arange(year_start,year_end):
    
    #----------------------------------------------------------------------------
    # LOOP: over months in year
    #----------------------------------------------------------------------------
        
#    for j in range(12):
    for j in range(2,3):
                                
        # EXTRACT: GloSATLAT from GloSAT.p04c.EBC.LEKnormals (alternative grid) for given year and month

        k = ( (year-1781) ) * 12 + j
        v = par[k,:,:]     
            		    		            
        # EXTRACT: station anomalies
        
        w = df_anom[ df_anom['year'] == year ]
        X = w.stationlon.values
        Y = w.stationlat.values
        Z = w[str(j+1)].values    
        
        # FILTER: stations with no data

        da = pd.DataFrame({'X':X, 'Y':Y, 'Z':Z}).dropna()

        X = da.X.values
        Y = da.Y.values
        Z = da.Z.values

        # EXTRACT: counts (stations and gridded)

        count_lat = df_count_lat[df_count_lat.index.year==year].values.ravel()[j]
        count_crutem_gridded = df_count_crutem_gridded[df_count_crutem_gridded.index.year==year].values.ravel()[j]
        count_stations = df_count_stations[df_count_stations.index.year==year].values.ravel()[j]
        
        # EXTRACT: NAO index

        nao = df_nao[df_nao.index.year==year].values.ravel()[j]

        # EXTRACT: SOI index

        soi = df_soi[df_soi.index.year==year].values.ravel()[j]

        # EXTRACT: GMST index from GloSAT.p04c.EBC.LEKnormals (alternative grid)

        gmst_lat = df_gmst_lat[df_gmst_lat.index.year==year]['gmst'].values.ravel()[j]
        gmst_crutem_gridded = df_gmst_crutem_gridded[df_gmst_crutem_gridded.index.year==year]['gmst'].values.ravel()[j]

        # APPLY: run type
        
        if use_lat == True:
            
            count = count_lat
            gmst = gmst_lat
            datastr1 = r'$\bf{Data}$' + r' (area-weighted mean □): GloSAT.p04c.EBC.LEKnormals (CRU/UEA, UYork)' + ': N=' + str( count ) + '/' + str(n_lat*n_lon)

        else:
            
            count = count_crutem_gridded
            gmst = gmst_crutem_gridded
            datastr1 = r'$\bf{Data}$' + r' (area-weighted mean □): GloSAT.p04c.EBC.LEKnormals (alternative grid) (CRU/UEA, UYork)' + ': N=' + str( count ) + '/' + str(n_lat*n_lon)
                                    
        #----------------------------------------------------------------------------
        # MAP: station points and gridded station area-weighted means
        #----------------------------------------------------------------------------
                        
        titlestr = str(year) + '-' + str(j+1).zfill(2) + ' NAO=' + str( np.round( nao, 3 ) ) + ' SOI=' + str( np.round( soi, 3 ) ) + ' GMST=' + str( np.round( gmst, 3 ) )            
        figstr = 'crutem_' + str(year) + '-' + str(j+1).zfill(2) +'.png'        

        # CREDITS
        
        datastr2 = r'$\bf{Data}$' + r' (stations ◦): GloSAT.p04c.EBC.LEKnormals (CRU/UEA, UYork)' + ': N(stations)=' + str( count_stations ) + '/' + str(11865)        
        datastr3 = r'$\bf{Data}$' + ' (NAO): 1781-1820 (Juerg Luterbacher), 1821-2022 (Phil Jones)'
        datastr4 = r'$\bf{Data}$' + ' (SOI): 1866-2022 (Phil Jones)'
        baselinestr = r'$\bf{Baseline}$' + ': 1961-1990'        
        authorstr = r'$\bf{Graphic}$' + ': Michael Taylor, CRU/UEA' + ' -- ' + titletime

        
        fig, ax = plt.subplots(figsize=(13.33,7.5), subplot_kw=dict(projection=p))    
        # PowerPoint:            fontsize = 18; fig = plt.figure(figsize=(13.33,7.5), dpi=144); plt.savefig('figure.png', bbox_inches='tight')
        # Posters  (vectorized): fontsize = 18; fig = plt.figure(figsize=(13.33,7.5), dpi=600); plt.savefig('my_figure.svg', bbox_inches='tight')                          
        # Journals (vectorized): fontsize = 18; fig = plt.figure(figsize=(3.54,3.54), dpi=300); plt.savefig('my_figure.svg', bbox_inches='tight')     
                                                             
        ax.set_global()  
        ax.add_feature(land, facecolor='grey', linestyle='-', linewidth=0.1, edgecolor='k', alpha=1, zorder=1)
        if use_cyan == True:
            ax.add_feature(ocean, facecolor='cyan', alpha=1, zorder=1)
        else:
            ax.add_feature(ocean, facecolor='grey', alpha=0.7, zorder=1)
        # ax.add_feature(lakes)
        # ax.add_feature(rivers, linewidth=0.5)
        # ax.add_feature(borders, linestyle='-', linewidth=0.1, edgecolor='k', alpha=1, zorder=2)         
    
        # PLOT: 5x5 gridded station data area-averaged mean
    
        g_stations = v.plot(ax=ax, alpha=1, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree(), cbar_kwargs={'orientation':'horizontal','extend':'both','shrink':0.5, 'pad':0.05}, zorder=10)                 
        cb = g_stations.colorbar; cb.ax.tick_params(labelsize=fontsize); cb.set_label(label=r'Anomaly (from 1961-1990), $^{\circ}$C', size=fontsize); cb.ax.set_title(None, fontsize=fontsize)
            
        ax.coastlines(resolution=resolution, color='k', linestyle='-', linewidth=0.2, edgecolor='k', alpha=1, zorder=100)                                                                                  
        ax.add_feature(borders, linestyle='-', linewidth=0.1, edgecolor='k', alpha=1, zorder=100)         
             
        if use_gridlines == True:
            
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.1, color='purple', alpha=1, linestyle='-', zorder=1000)
            gl.top_labels = False; gl.bottom_labels = False; gl.left_ylabels = False; gl.right_ylabels = False
            gl.xlines = True; gl.ylines = True
            gl.xlocator = mticker.FixedLocator(np.linspace(-180,180,73)) # every 5 degrees
            gl.ylocator = mticker.FixedLocator(np.linspace(-90,90,37))   # every 5 degrees
            gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    
        # PLOT: station data
       
        h1 = plt.scatter(x=X, y=Y, c=Z, s=2, marker='o', edgecolor='k', lw=0.2, alpha=1, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree(), zorder=10000)         
    
        ax.set_title(titlestr, color=default_color, fontsize=fontsize)                                                
        
        if dpi == 144: xstart = 325; ystart=10; ystep = 20
        elif dpi == 300: xstart = 675; ystart=10; ystep = 40
        elif dpi == 600: xstart = 1350; ystart=10; ystep = 80
        
        plt.annotate(datastr1, xy=(xstart,ystart+ystep*5), xycoords='figure pixels', color=default_color, fontsize=8)  
        plt.annotate(datastr2, xy=(xstart,ystart+ystep*4), xycoords='figure pixels', color=default_color, fontsize=8) 
        plt.annotate(datastr3, xy=(xstart,ystart+ystep*3), xycoords='figure pixels', color=default_color, fontsize=8) 
        plt.annotate(datastr4, xy=(xstart,ystart+ystep*2), xycoords='figure pixels', color=default_color, fontsize=8) 
        plt.annotate(authorstr, xy=(xstart,ystart+ystep*1), xycoords='figure pixels', color=default_color, fontsize=8)   
        
        fig.subplots_adjust(left=None, bottom=0.1, right=None, top=None, wspace=None, hspace=None)
        plt.savefig(figstr, dpi=dpi, bbox_inches='tight', pad_inches=0.2)
        plt.close()

# -----------------------------------------------------------------------------
# Print library verions
# -----------------------------------------------------------------------------

print("numpy      : ", np.__version__) 
print("pandas     : ", pd.__version__) 
print("xarray     : ", xr.__version__)
print("matplotlib : ", matplotlib.__version__)
print("cartopy    : ", cartopy.__version__)

# -----------------------------------------------------------------------------
print('** END')
