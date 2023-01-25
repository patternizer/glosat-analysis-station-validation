#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_glosat_analysis_vs_stations.py
#------------------------------------------------------------------------------
# Version 0.3
# 25 January, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

import os, glob
import imageio
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import pickle
from datetime import datetime
import calendar as cal

# Statisticslibraries:
from scipy import signal
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

print("numpy   : ", np.__version__) 
print("xarray   : ", xr.__version__)
print("matplotlib   : ", matplotlib.__version__)
print("cartopy   : ", cartopy.__version__)

# %matplotlib inline # for Jupyter Notebooks

'''
# Calculate current time 

now = datetime.now()
currentmn = str(now.month)
if now.day == 1:
    currentdy = str(cal.monthrange(now.year,now.month-1)[1])
    currentmn = str(now.month-1)
else:
    currentdy = str(now.day-1)
if int(currentdy) < 10:
    currentdy = '0' + currentdy    
currentyr = str(now.year)
if int(currentmn) < 10:
    currentmn = '0' + currentmn
titletime = str(currentdy) + '/' + currentmn + '/' + currentyr
'''

# Calculate current time

now = datetime.now()
currentdy = str(now.day).zfill(2)
currentmn = str(now.month).zfill(2)
currentyr = str(now.year)
titletime = str(currentdy) + '/' + currentmn + '/' + currentyr    

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

fontsize = 10
cmap = 'bwr'
vmin = -6.0        
vmax = 6.0    

year_start = 1781
year_end = 2022

filename_temp = 'DATA/df_temp_qc.pkl'        
filename_anom = 'DATA/df_anom_qc.pkl'     
filename_nao = 'DATA/naomonjurg.dat' # Juerg Luterbacher reconstructed NAO 1658-2021
path = 'DATA/glosat-analysis-alpha-4-infilled/median_fields/'
glosat_version = 'GloSAT.p04c.EBC.LEKnormals'

use_darktheme = False
make_gif = False
    
projection = 'platecarree'

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
    
#------------------------------------------------------------------------------
# THEME
#------------------------------------------------------------------------------

if use_darktheme == True:
	
    matplotlib.rcParams['text.usetex'] = False
#    rcParams['font.family'] = ['DejaVu Sans']
#    rcParams['font.sans-serif'] = ['Avant Garde']
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
#    rcParams['font.family'] = ['DejaVu Sans']
#    rcParams['font.sans-serif'] = ['Avant Garde']
    plt.rc('savefig',facecolor='white')
    plt.rc('axes',edgecolor='black')
    plt.rc('xtick',color='black')
    plt.rc('ytick',color='black')
    plt.rc('axes',labelcolor='black')
    plt.rc('axes',facecolor='white')

#----------------------------------------------------------------------------
# CREDITS
#----------------------------------------------------------------------------

datastr1 = r'$\bf{Data}$' + ' (5x5 Gridded): GloSAT.analysis.alpha.4 (UKMO-HC/Colin Morice)'        
datastr2 = r'$\bf{Data}$' + ' (stations): GloSAT.p04c.EBC.LEKnormals (CRU/UEA)'  
datastr3 = r'$\bf{Data}$' + ' (NAO): 1658-2001 (Juerg Luterbacher, Bern)' 
baselinestr = r'$\bf{Baseline}$' + ': 1961-1990'        
authorstr = r'$\bf{Graphic}$' + ': Michael Taylor, CRU/UEA' + ' -- ' + titletime

#----------------------------------------------------------------------------
# LOAD: GloSAT anomalies
#----------------------------------------------------------------------------

df_temp = pd.read_pickle( filename_temp, compression='bz2' ) 
df_anom = pd.read_pickle( filename_anom, compression='bz2' ) 

#------------------------------------------------------------------------------    
# LOAD: NAO indices
#------------------------------------------------------------------------------
# https://crudata.uea.ac.uk/cru/data/nao/viz.htm

print('loading NAO indices ...')

#Year	Mon	NAO
#1658	Dec	-0.30

nheader = 26
f = open(filename_nao)
lines = f.readlines()
years = []
months = [] 
obs = []

month_dict = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

for i in range(nheader,len(lines)):
    words = lines[i].split()   
    year = int(words[0])
    month = month_dict[ words[1] ]
    val = float(words[2])
    years.append(year)                                     
    months.append(month)            
    obs.append(val)            
f.close()    
obs = np.array(obs)

da = pd.DataFrame()             
da['year'] = years
da['month'] = months
da['nao'] = obs

# TRIM: to start of Pandas datetime range ( if needed )

da = da[da.year >= 1678].reset_index(drop=True)
dates = [ pd.to_datetime( str(da.year[i]) + '-' + str(da.month[i]).zfill(2) + '-01', format='%Y-%m-%d' ) for i in range(len(da)) ]
df_nao = pd.DataFrame({'nao':da.nao.values}, index=dates )

#----------------------------------------------------------------------------
# PLOT: Ensemble mean temperature anomalies
#----------------------------------------------------------------------------

if use_darktheme == True:
	default_color = 'white'
else:
	default_color = 'black'               

dec_map = np.ones([36,72]) * np.nan # initialise zero map for SSIM crossing year boundary
        
for year in np.arange(year_start,year_end):

    print(year)    

    #----------------------------------------------------------------------------
    # LOAD: GloSAT.analysis.alpha.4
    #----------------------------------------------------------------------------
    
    ncfile = path + 'GloSATref.1.0.0.0-alpha.4.analysis.analysis.anomalies.ensemble_median.' + str(year) + '.nc'
    
    ds = xr.open_dataset(ncfile, decode_cf=True)
    par = ds.tas_median
    lon = ds.longitude
    lat = ds.latitude
    time = ds.time
    
    N = par.shape[0]
        
    for j in range(N):
            
        # EXTRACT: analysis 5x5 gridded anomaly background
 
        v = par[j,:,:]
        x, y = np.meshgrid(lon,lat)        
    
        # EXTRACT: station anomalies
        
        w = df_anom[ df_anom['year'] == year ]
        X = w.stationlon.values
        Y = w.stationlat.values
        Z = w[str(j+1)].values    

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

        if j == (N-1): dec_map = np.array( par[j,:,:] )
                
        titlestr = str(year) + '-' + str(j+1).zfill(2) + ' NAO=' + str(nao) + ' SSIM=' + str( np.round( ssim, 3 ) )
        figstr = 'tas_median_' + str(year) + '-' + str(j+1).zfill(2) +'.png'        
    
        fig, ax = plt.subplots(figsize=(13.33,7.5), subplot_kw=dict(projection=p))    
        # PowerPoint:            fontsize = 18; fig = plt.figure(figsize=(13.33,7.5), dpi=144); plt.savefig('figure.png', bbox_inches='tight')
        # Posters  (vectorized): fontsize = 18; fig = plt.figure(figsize=(13.33,7.5), dpi=600); plt.savefig('my_figure.svg', bbox_inches='tight')                          
        # Journals (vectorized): fontsize = 18; fig = plt.figure(figsize=(3.54,3.54), dpi=300); plt.savefig('my_figure.svg', bbox_inches='tight')     
    
        # NB: for other projections we need to transform with ccrs.Geodetic()
            
        # g = plt.scatter(x=x, y=y, c=v, s=1, alpha=1.0, transform=p)         
        # g = plt.contourf(x, y, v, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, extend='both')   
        g = v.plot(ax=ax, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap=cmap, cbar_kwargs={'orientation':'horizontal','extend':'both','shrink':0.5, 'pad':0.05})             
        cb = g.colorbar; cb.ax.tick_params(labelsize=fontsize); cb.set_label(label=r'Anomaly,$^{\circ}$C', size=fontsize); cb.ax.set_title(None, fontsize=fontsize)
    
        h = plt.scatter(x=X, y=Y, c=Z, s=10, marker='o', edgecolor='k', lw=0.5, vmin=vmin, vmax=vmax, cmap=cmap, transform=p)         
                                    
        resolution = '10m' # 110, 50 or 10km
        ax.set_global()  
    
        borders = cf.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resolution, facecolor='none', alpha=0.7)
    #    land = cf.NaturalEarthFeature('physical', 'land', scale=resolution, edgecolor='k', facecolor=cf.COLORS['land'])
    #    ocean = cf.NaturalEarthFeature('physical', 'ocean', scale=resolution, edgecolor='none', facecolor=cf.COLORS['water'])
    #    lakes = cf.NaturalEarthFeature('physical', 'lakes', scale=resolution, edgecolor='b', facecolor=cf.COLORS['water'])
    #    rivers = cf.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=resolution, edgecolor='b', facecolor='none')
         
    #    ax.add_feature(land, facecolor='lightgrey', linestyle='-', linewidth=0.1, edgecolor='k', alpha=0.5)
    #    ax.add_feature(ocean, linestyle='-', linewidth=0.1, edgecolor='b', alpha=1 )
    #    ax.add_feature(lakes)
    #    ax.add_feature(rivers, linewidth=0.5)
        ax.add_feature(borders, linestyle='-', linewidth=0.1, edgecolor='k', alpha=1)         
        ax.coastlines(resolution=resolution, color='k', linestyle='-', linewidth=0.2, edgecolor='k', alpha=1)                                                                              
    
        ax.set_title(titlestr, fontsize=fontsize)    
              
    #    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='lightgrey', alpha=1, linestyle='-')
    #    gl.top_labels = False; gl.bottom_labels = False; gl.left_ylabels = False; gl.right_ylabels = False
    #    gl.xlines = True; gl.ylines = True
    #    gl.xlocator = mticker.FixedLocator(np.linspace(-180,180,73)) # every 5 degrees
    #    gl.ylocator = mticker.FixedLocator(np.linspace(-90,90,37))   # every 5 degrees
    #    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
                                              
    #   fig.suptitle(titlestr, fontsize=36, color=default_color, fontweight='bold')        
        plt.annotate(datastr1, xy=(300,120), xycoords='figure pixels', color=default_color, fontsize=10) 
        plt.annotate(datastr2, xy=(300,100), xycoords='figure pixels', color=default_color, fontsize=10) 
        plt.annotate(datastr3, xy=(300,80), xycoords='figure pixels', color=default_color, fontsize=10) 
        plt.annotate(baselinestr, xy=(300,60), xycoords='figure pixels', color=default_color, fontsize=10)   
        plt.annotate(authorstr, xy=(300,40), xycoords='figure pixels', color=default_color, fontsize=10)     
    
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    #   plt.savefig(figstr, bbox_inches='tight')
        plt.savefig(figstr, dpi=144)
        plt.close()


#if make_gif == True:

#    images = sorted(glob.glob('*.png'))
#    var = [imageio.imread(file) for file in images]
#    imageio.mimsave('plot.gif', var, fps = 10)

# COVERT GIF to MP4
# ffmpeg -i plot.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" plot.mp4

# -----------------------------------------------------------------------------
print('** END')
