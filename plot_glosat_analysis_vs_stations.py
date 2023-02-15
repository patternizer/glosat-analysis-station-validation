#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_glosat_analysis_vs_stations.py
#------------------------------------------------------------------------------
# Version 0.5
# 1 February, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import pickle
from datetime import datetime

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

fontsize = 10
cmap = 'bwr'
#cmap = 'RdBu_r'
vmin = -6.0        
vmax = 6.0    
resolution = '10m' # 110, 50 or 10km
dpi = 300 # 144,300,600

year_start = 1781
year_end = 2022

filename_anom = 'DATA/df_anom_qc.pkl'     
filename_nao = 'DATA/df_nao_1781_2022.pkl' 
path = 'DATA/glosat-analysis-alpha-4-infilled/median_fields/'
glosat_version = 'GloSAT.p04c.EBC.LEKnormals'

use_darktheme = False
make_gif = False
    
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

    matplotlib.rcParams['text.usetex'] = True
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

datastr1 = r'$\bf{Data}$' + ' (5x5 Gridded): GloSAT.analysis.alpha.4 (UKMO-HC / Colin Morice)'        
datastr2 = r'$\bf{Data}$' + ' (stations): GloSAT.p04c.EBC.LEKnormals (CRU/UEA)'  
datastr3 = r'$\bf{Data}$' + ' (NAO): 1781-1820 (Juerg Luterbacher), 1821-2022 (Phil Jones)'
baselinestr = r'$\bf{Baseline}$' + ': 1961-1990'        
authorstr = r'$\bf{Graphic}$' + ': Michael Taylor, CRU/UEA' + ' -- ' + titletime

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

#----------------------------------------------------------------------------
# LOAD: GloSAT anomalies
#----------------------------------------------------------------------------

print('loading station anomalies ...')

df_anom = pd.read_pickle( filename_anom, compression='bz2' ) 

#------------------------------------------------------------------------------    
# LOAD: NAO indices
#------------------------------------------------------------------------------
# https://crudata.uea.ac.uk/cru/data/nao/viz.htm

print('loading NAO indices ...')

# LOAD: merged pkl dataframe of monthly NAO: Luterbacher reconstructions 1658-1821 and Phil Jones 1821-2022
     
df_nao = pd.read_pickle( filename_nao, compression='bz2' )      

#----------------------------------------------------------------------------
# INITIALISE: arrays to store timeseries of NAO, SSIM and analysis grid counts
#----------------------------------------------------------------------------

dec_map = np.ones([36,72]) * np.nan # initialise zero map for SSIM crossing year boundary
        
ssim_vec = []
n_analysis_vec = []
n_stations_vec = []

#----------------------------------------------------------------------------
# LOOP: over all years
#----------------------------------------------------------------------------

for year in np.arange(year_start,year_end):

    # LOAD: GloSAT.analysis.alpha.4 for a given year
    
    print('loading analysis for year=' + str(year) + ' ...')
    
    ncfile = path + 'GloSATref.1.0.0.0-alpha.4.analysis.analysis.anomalies.ensemble_median.' + str(year) + '.nc'
    
    ds = xr.open_dataset(ncfile, decode_cf=True)
    par = ds.tas_median
    lon = ds.longitude
    lat = ds.latitude
    time = ds.time
    
    N = par.shape[0]

    #----------------------------------------------------------------------------
    # LOOP: over months in year
    #----------------------------------------------------------------------------
        
    for j in range(N):
            
        # EXTRACT: analysis 5x5 gridded anomaly background
 
        v = par[j,:,:]
        x, y = np.meshgrid(lon,lat)        
    		    		    
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

        # COUNT: contributing analysis gridcells and stations
        
        count_analysis = int(np.isfinite(v).sum())
        count_stations = len(Z)

        # EXTRACT: NAO index

        nao = df_nao[df_nao.index.year==year].values.ravel()[j]

        # CORRELATE: with previous month
                
        if j == 0:

            map1 = dec_map
            map2 = np.array( par[j,:,:] )
            
        else:

            map1 = np.array( par[j-1,:,:] )
            map2 = np.array( par[j,:,:] )

        # CATER: for first map having no prior map (December previous year) to correlate against

        if np.nansum(map1) == 0.0:    

            image1 = map1
            image2 = np.nan_to_num(map2, copy=True, nan=0.0, posinf=None, neginf=None)
        
        else:

            image1 = np.nan_to_num(map1, copy=True, nan=0.0, posinf=None, neginf=None)
            image2 = np.nan_to_num(map2, copy=True, nan=0.0, posinf=None, neginf=None)
            
        ssim = structural_similarity(image1, image2)

        # APPEND: SSIM index and analysis counts
        
        ssim_vec.append(ssim)
        n_analysis_vec.append(count_analysis)
        n_stations_vec.append(count_stations)

        if j == (N-1): dec_map = np.array( par[j,:,:] ) # temporarily store December map (for next year's January SSIM calculation)

        #----------------------------------------------------------------------------
        # PLOT: map of analysis overlaid with station points
        #----------------------------------------------------------------------------
                    
        titlestr = str(year) + '-' + str(j+1).zfill(2) + ' NAO=' + str( np.round( nao, 3 ) ) + ' SSIM=' + str( np.round( ssim, 3 ) ) + ' N(analysis)=' + str(count_analysis) + '/' + str(36*72) + ' N(stations)=' + str( count_stations ) + '/' + str(11865) 
        figstr = 'tas_median_' + str(year) + '-' + str(j+1).zfill(2) +'.png'        
    
        fig, ax = plt.subplots(figsize=(13.33,7.5), subplot_kw=dict(projection=p))    
        # PowerPoint:            fontsize = 18; fig = plt.figure(figsize=(13.33,7.5), dpi=144); plt.savefig('figure.png', bbox_inches='tight')
        # Posters  (vectorized): fontsize = 18; fig = plt.figure(figsize=(13.33,7.5), dpi=600); plt.savefig('my_figure.svg', bbox_inches='tight')                          
        # Journals (vectorized): fontsize = 18; fig = plt.figure(figsize=(3.54,3.54), dpi=300); plt.savefig('my_figure.svg', bbox_inches='tight')     
                                            
        borders = cf.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resolution, facecolor='none', alpha=1)
        land = cf.NaturalEarthFeature('physical', 'land', scale=resolution, edgecolor='k', facecolor=cf.COLORS['land'])
        ocean = cf.NaturalEarthFeature('physical', 'ocean', scale=resolution, edgecolor='none', facecolor=cf.COLORS['water'])
        lakes = cf.NaturalEarthFeature('physical', 'lakes', scale=resolution, edgecolor='b', facecolor=cf.COLORS['water'])
        rivers = cf.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=resolution, edgecolor='b', facecolor='none')
         
        ax.set_global()  
        ax.add_feature(land, facecolor='grey', linestyle='-', linewidth=0.1, edgecolor='k', alpha=1, zorder=1)
        ax.add_feature(ocean, facecolor='cyan', alpha=1, zorder=1)
        # ax.add_feature(lakes)
        # ax.add_feature(rivers, linewidth=0.5)
        # ax.add_feature(borders, linestyle='-', linewidth=0.1, edgecolor='k', alpha=1, zorder=2)         

        g = v.plot(ax=ax, alpha=1, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree(), cbar_kwargs={'orientation':'horizontal','extend':'both','shrink':0.5, 'pad':0.05}, zorder=10)                 
        cb = g.colorbar; cb.ax.tick_params(labelsize=fontsize); cb.set_label(label=r'Anomaly,$^{\circ}$C', size=fontsize); cb.ax.set_title(None, fontsize=fontsize)

        ax.coastlines(resolution=resolution, color='k', linestyle='-', linewidth=0.2, edgecolor='k', alpha=1, zorder=100)                                                                                  
        ax.add_feature(borders, linestyle='-', linewidth=0.1, edgecolor='k', alpha=1, zorder=100)         
              
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.1, color='purple', alpha=1, linestyle='-', zorder=1000)
        gl.top_labels = False; gl.bottom_labels = False; gl.left_ylabels = False; gl.right_ylabels = False
        gl.xlines = True; gl.ylines = True
        gl.xlocator = mticker.FixedLocator(np.linspace(-180,180,73)) # every 5 degrees
        gl.ylocator = mticker.FixedLocator(np.linspace(-90,90,37))   # every 5 degrees
        gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER

        h = plt.scatter(x=X, y=Y, c=Z, s=2, marker='o', edgecolor='k', lw=0.1, alpha=1, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree(), zorder=10000)         

        ax.set_title(titlestr, fontsize=fontsize)                                                
    #   fig.suptitle(titlestr, fontsize=36, color=default_color, fontweight='bold')        
    
        if dpi == 144: xstart = 333; ystart=10; ystep = 20
        elif dpi == 300: xstart = 700; ystart=20; ystep = 40
        elif dpi == 600: xstart = 1400; ystart=40; ystep = 80
    
        plt.annotate(datastr1, xy=(xstart,ystart+ystep*5), xycoords='figure pixels', color=default_color, fontsize=10)  
        plt.annotate(datastr2, xy=(xstart,ystart+ystep*4), xycoords='figure pixels', color=default_color, fontsize=10) 
        plt.annotate(datastr3, xy=(xstart,ystart+ystep*3), xycoords='figure pixels', color=default_color, fontsize=10) 
        plt.annotate(baselinestr, xy=(xstart,ystart+ystep*2), xycoords='figure pixels', color=default_color, fontsize=10)   
        plt.annotate(authorstr, xy=(xstart,ystart+ystep*1), xycoords='figure pixels', color=default_color, fontsize=10)   
    
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()

# SAVE: NAO timeseries

df_nao.to_pickle( 'df_nao.pkl', compression='bz2' )

# SAVE: SSIM timeseries
    
t_vec = pd.date_range(start=str(year_start), end=str(year_end), freq='MS' )[0:-1]
df_ssim = pd.DataFrame({'ssim':ssim_vec}, index=t_vec)
df_ssim.to_pickle( 'df_ssim.pkl', compression='bz2' )

# SAVE: analysis count timeseries
    
t_vec = pd.date_range(start=str(year_start), end=str(year_end), freq='MS' )[0:-1]
df_count_analysis = pd.DataFrame({'count':n_analysis_vec}, index=t_vec)
df_count_analysis.to_pickle( 'df_count_analysis.pkl', compression='bz2' )

# SAVE: station count timeseries
    
t_vec = pd.date_range(start=str(year_start), end=str(year_end), freq='MS' )[0:-1]
df_count_stations = pd.DataFrame({'count':n_stations_vec}, index=t_vec)
df_count_stations.to_pickle( 'df_count_stations.pkl', compression='bz2' )

#----------------------------------------------------------------------------
# PLOT: SSIM & count timerseries
#----------------------------------------------------------------------------

# RESET: plot vars

fontsize = 16
import seaborn as sns; sns.set()

figstr = 'ssim_count.png'
titlestr = 'GloSAT.analysis.alpha.4: SSIM(t-1,t) and Coverage(t)'
                        
fig, ax = plt.subplots(figsize=(15,10))     
plt.plot( df_ssim.index, df_ssim.ssim, marker='o', ls='-', lw=0.5, color='blue')
ax1 = plt.gca()
ax2 = ax.twinx()
ax2.plot( df_count_analysis.index, (1.0 - df_count_analysis['count']/2592), marker='o', ls='-', lw=0.5, color='red', label='GloSAT analysis')
ax2.plot( df_count_stations.index, (1.0 - df_count_stations['count']/11865), marker='s', markerfacecolor='none', ls='--', lw=0.5, color='red', label='GloSAT stations')
ax1.set_ylabel('SSIM(t-1,t)', fontsize=fontsize, color='blue')
ax1.tick_params(labelsize=fontsize, color='blue')    
ax1.set_ylim(0,1)
ax2.spines['left'].set_color('blue')
ax2.set_ylabel('Coverage(t)', fontsize=fontsize, color='red')
ax2.tick_params(labelsize=fontsize, colors='red')    
ax2.spines['right'].set_color('red')
ax2.set_ylim(0,1)
ax2.legend(loc='lower right', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
plt.close('all')

#----------------------------------------------------------------------------
# PLOT: NAO timerseries
#----------------------------------------------------------------------------

figstr = 'nao.png'
titlestr = 'Merged NAO: 1781-1820 (Juerg Luterbacher), 1821-2022 (Phil Jones)'
                        
fig, ax = plt.subplots(figsize=(15,10))    
plt.fill_between( df_nao[df_nao.index.year<1821].index, df_nao[df_nao.index.year<1821].nao, ls='-', lw=0.5, color='red', label='NAO: 1781-1820 (Juerg Luterbacher)')
plt.fill_between( df_nao[df_nao.index.year>=1821].index, df_nao[df_nao.index.year>=1821].nao, ls='-', lw=0.5, color='green', label='NAO: 1821-2022 (Phil Jones)')
plt.ylim(-7,7)
plt.legend(loc='upper left', fontsize=fontsize)    
plt.tick_params(labelsize=fontsize)    
plt.ylabel('NAO(t)', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
plt.close('all')

# -----------------------------------------------------------------------------
# Print library verions
# -----------------------------------------------------------------------------
print("numpy      : ", np.__version__) 
print("pandas     : ", pd.__version__) 
print("xarray     : ", xr.__version__)
print("matplotlib : ", matplotlib.__version__)
print("cartopy    : ", cartopy.__version__)
print("seaborn    : ", sns.__version__)

# -----------------------------------------------------------------------------
print('** END')
