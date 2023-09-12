#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: plot_stats.py
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
dpi = 300 # 144,300,600

nsmooth = 10 # years for yearly means
use_weighted_yearly_mean = False # [True,False]

year_start = 1781
year_end = 2022

latstep = 5
lonstep = 5
n_lat = int(180/latstep)
n_lon = int(360/lonstep)        

nao_file = 'OUT/df_nao.pkl'
soi_file = 'OUT/df_soi.pkl'
aod_file = 'OUT/df_aod.pkl'

ssim_analysis_file = 'OUT/df_ssim_analysis.pkl'
ssim_lat_file = 'OUT/df_ssim_lat.pkl'
ssim_mat_file = 'OUT/df_ssim_mat.pkl'
ssim_crutem_gridded_file = 'OUT/df_ssim_crutem_gridded.pkl'

count_analysis_file = 'OUT/df_count_analysis.pkl'
count_lat_file = 'OUT/df_count_lat.pkl'
count_mat_file = 'OUT/df_count_mat.pkl'
count_crutem_gridded_file = 'OUT/df_count_crutem_gridded.pkl'
count_stations_file = 'OUT/df_count_stations.pkl'

gmst_analysis_file = 'OUT/df_gmst_analysis.pkl'
gmst_lat_file = 'OUT/df_gmst_lat.pkl'
gmst_mat_file = 'OUT/df_gmst_mat.pkl'
gmst_crutem_gridded_file = 'OUT/df_gmst_crutem_gridded.pkl'
gmst_crutem_gridded_cosine_file = 'OUT/df_gmst_crutem_gridded_cosine.pkl'
gmst_crutem_file = 'OUT/df_gmst_crutem.pkl' # ['gmst_global','gmst_nh','gmst_sh']

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
 
def update_yearly_mean( ds, var):
    """
    update Pandas dataframe yearly mean with weighted yearly mean
    """

    dt = ds.to_xarray()
    dt = dt.rename( {'index':'time'} )    
    du = weighted_temporal_mean( dt, var ).values
    
    return du
       
#----------------------------------------------------------------------------
# LOAD: stats timerseries
#----------------------------------------------------------------------------

# LOAD: NAO timeseries
df_nao = pd.read_pickle( nao_file, compression='bz2' )

# LOAD: SOI timeseries
df_soi = pd.read_pickle( soi_file, compression='bz2' )

# LOAD: stratospheric AOD timeseries
df_aod = pd.read_pickle( aod_file, compression='bz2' )

# LOAD: SSIM (analysis) timeseries    
df_ssim_analysis = pd.read_pickle( ssim_analysis_file, compression='bz2' )

# LOAD: SSIM (LAT) timeseries    
df_ssim_lat = pd.read_pickle( ssim_lat_file, compression='bz2' )

# LOAD: SSIM (MAT) timeseries    
df_ssim_mat = pd.read_pickle( ssim_mat_file, compression='bz2' )

# LOAD: SSIM (MAT) timeseries    
df_ssim_crutem_gridded = pd.read_pickle( ssim_crutem_gridded_file, compression='bz2' )

# LOAD: analysis count timeseries    
df_count_analysis = pd.read_pickle( count_analysis_file, compression='bz2' ) / (n_lat * n_lon)

# LOAD: lat count timeseries    
df_count_lat = pd.read_pickle( count_lat_file, compression='bz2' ) / (n_lat * n_lon)

# LOAD: mat count timeseries    
df_count_mat = pd.read_pickle( count_mat_file, compression='bz2' ) / (n_lat * n_lon)

# LOAD: crutem count timeseries    
df_count_crutem_gridded = pd.read_pickle( count_crutem_gridded_file, compression='bz2' ) / (n_lat * n_lon)

# LOAD: station count timeseries    
df_count_stations = pd.read_pickle( count_stations_file, compression='bz2' )

# LOAD: GMST (analysis) timeseries    
df_gmst_analysis = pd.read_pickle( gmst_analysis_file, compression='bz2' )

# LOAD: GMST (lat) timeseries    
df_gmst_lat = pd.read_pickle( gmst_lat_file, compression='bz2' )

# LOAD: GMST (mat) timeseries    
df_gmst_mat = pd.read_pickle( gmst_mat_file, compression='bz2' )

# LOAD: GMST (CRUTEM) timeseries    
df_gmst_crutem = pd.read_pickle( gmst_crutem_file, compression='bz2' )

# LOAD: GMST (CRUTEM gridded) timeseries    
df_gmst_crutem_gridded = pd.read_pickle( gmst_crutem_gridded_file, compression='bz2' )

# LOAD: GMST (CRUTEM gridded cosine) timeseries    
df_gmst_crutem_gridded_cosine = pd.read_pickle( gmst_crutem_gridded_cosine_file, compression='bz2' )
    
#----------------------------------------------------------------------------
# SET: time vectors
#----------------------------------------------------------------------------

t_vec = pd.date_range(start=str(year_start), end=str(year_end), freq='MS' )[0:-1]
t_vec_yearly = pd.date_range(start=str(year_start), end=str(year_end), freq='AS' )[0:-1]
        
#----------------------------------------------------------------------------
# COMPUTE: yearly means
#----------------------------------------------------------------------------

df_nao_yearly = df_nao.groupby(df_nao.index.year).mean()
df_soi_yearly = df_soi.groupby(df_soi.index.year).mean()
df_aod_yearly = df_aod.groupby(df_aod.index.year).mean()

df_ssim_analysis_yearly = df_ssim_analysis.groupby(df_ssim_analysis.index.year).mean()
df_ssim_lat_yearly = df_ssim_lat.groupby(df_ssim_lat.index.year).mean()
df_ssim_mat_yearly = df_ssim_mat.groupby(df_ssim_mat.index.year).mean()
df_ssim_crutem_gridded_yearly = df_ssim_crutem_gridded.groupby(df_ssim_crutem_gridded.index.year).mean()

df_count_analysis_yearly = df_count_analysis.groupby(df_count_analysis.index.year).mean()
df_count_lat_yearly = df_count_lat.groupby(df_count_lat.index.year).mean()
df_count_mat_yearly = df_count_mat.groupby(df_count_mat.index.year).mean()
df_count_crutem_gridded_yearly = df_count_crutem_gridded.groupby(df_count_crutem_gridded.index.year).mean()
df_count_stations_yearly = df_count_stations.groupby(df_count_stations.index.year).mean()

df_gmst_analysis_yearly = df_gmst_analysis.groupby(df_gmst_analysis.index.year).mean()
df_gmst_lat_yearly = df_gmst_lat.groupby(df_gmst_lat.index.year).mean()
df_gmst_mat_yearly = df_gmst_mat.groupby(df_gmst_mat.index.year).mean()
df_gmst_crutem_yearly = df_gmst_crutem.groupby(df_gmst_crutem.index.year).mean()
df_gmst_crutem_gridded_yearly = df_gmst_crutem_gridded.groupby(df_gmst_crutem_gridded.index.year).mean()
df_gmst_crutem_gridded_cosine_yearly = df_gmst_crutem_gridded_cosine.groupby(df_gmst_crutem_gridded_cosine.index.year).mean()

#----------------------------------------------------------------------------
# COMPUTE: yearly SD
#----------------------------------------------------------------------------

df_nao_yearly_sd = df_nao.groupby(df_nao.index.year).std()
df_soi_yearly_sd = df_soi.groupby(df_soi.index.year).std()
df_aod_yearly_sd = df_aod.groupby(df_aod.index.year).std()

df_ssim_analysis_yearly_sd = df_ssim_analysis.groupby(df_ssim_analysis.index.year).std()
df_ssim_lat_yearly_sd = df_ssim_lat.groupby(df_ssim_lat.index.year).std()
df_ssim_mat_yearly_sd = df_ssim_mat.groupby(df_ssim_mat.index.year).std()
df_ssim_crutem_gridded_yearly_sd = df_ssim_crutem_gridded.groupby(df_ssim_crutem_gridded.index.year).std()

df_count_analysis_yearly_sd = df_count_analysis.groupby(df_count_analysis.index.year).std()
df_count_lat_yearly_sd = df_count_lat.groupby(df_count_lat.index.year).std()
df_count_mat_yearly_sd = df_count_mat.groupby(df_count_mat.index.year).std()
df_count_crutem_gridded_yearly_sd = df_count_crutem_gridded.groupby(df_count_crutem_gridded.index.year).std()
df_count_stations_yearly_sd = df_count_stations.groupby(df_count_stations.index.year).std()

df_gmst_analysis_yearly_sd = df_gmst_analysis.groupby(df_gmst_analysis.index.year).std()
df_gmst_lat_yearly_sd = df_gmst_lat.groupby(df_gmst_lat.index.year).std()
df_gmst_mat_yearly_sd = df_gmst_mat.groupby(df_gmst_mat.index.year).std()
df_gmst_crutem_yearly_sd = df_gmst_crutem.groupby(df_gmst_crutem.index.year).std()
df_gmst_crutem_gridded_yearly_sd = df_gmst_crutem_gridded.groupby(df_gmst_crutem_gridded.index.year).std()
df_gmst_crutem_gridded_cosine_yearly_sd = df_gmst_crutem_gridded_cosine.groupby(df_gmst_crutem_gridded_cosine.index.year).std()

#----------------------------------------------------------------------------
# UPDATE: yearly means (weighted by number of days in month)
#----------------------------------------------------------------------------

if use_weighted_yearly_mean == True:
    
    df_nao_yearly['nao'] = update_yearly_mean( df_nao, 'nao')
    df_soi_yearly['soi'] = update_yearly_mean( df_soi, 'soi')
    df_aod_yearly['aod'] = update_yearly_mean( df_aod, 'aod')
    
    df_ssim_analysis_yearly['ssim'] = update_yearly_mean( df_ssim_analysis, 'ssim' )
    df_ssim_lat_yearly['ssim'] = update_yearly_mean( df_ssim_lat, 'ssim' )
    df_ssim_mat_yearly['ssim'] = update_yearly_mean( df_ssim_mat, 'ssim' )
    df_ssim_crutem_gridded_yearly['ssim'] = update_yearly_mean( df_ssim_crutem_gridded, 'ssim' )
    
    df_count_analysis_yearly['count'] = update_yearly_mean( df_count_analysis, 'count' )
    df_count_lat_yearly['count'] = update_yearly_mean( df_count_lat, 'count' )
    df_count_mat_yearly['count'] = update_yearly_mean( df_count_mat, 'count' )
    df_count_crutem_gridded_yearly['count'] = update_yearly_mean( df_count_crutem_gridded, 'count' )
    df_count_stations_yearly['count'] = update_yearly_mean( df_count_stations, 'count' )
    
    df_gmst_analysis_yearly['gmst'] = update_yearly_mean( df_gmst_analysis, 'gmst' )
    df_gmst_lat_yearly['gmst'] = update_yearly_mean( df_gmst_lat, 'gmst' )
    df_gmst_mat_yearly['gmst'] = update_yearly_mean( df_gmst_mat, 'gmst' )
    df_gmst_crutem_yearly['gmst_global'] = update_yearly_mean( df_gmst_crutem, 'gmst_global' )
    df_gmst_crutem_gridded_yearly['gmst'] = update_yearly_mean( df_gmst_crutem_gridded, 'gmst' )
    df_gmst_crutem_gridded_cosine_yearly['gmst'] = update_yearly_mean( df_gmst_crutem_gridded_cosine, 'gmst' )

#==============================================================================
# PLOTS
#==============================================================================

#----------------------------------------------------------------------------
# PLOT: NAO timerseries (yearly mean)
#----------------------------------------------------------------------------

figstr = 'nao_yearly.png'
titlestr = 'Merged NAO: 1781-2022 (yearly mean)'
                        
fig, ax = plt.subplots(figsize=(15,10))     
plt.fill_between( df_nao_yearly[df_nao_yearly.index<1821].index, df_nao_yearly[df_nao_yearly.index<1821].nao, ls='-', lw=0.5, color='red', label='NAO: 1781-1820 (Juerg Luterbacher)')
plt.fill_between( df_nao_yearly[df_nao_yearly.index>=1821].index, df_nao_yearly[df_nao_yearly.index>=1821].nao, ls='-', lw=0.5, color='green', label='NAO: 1821-2022 (Phil Jones)')
plt.ylim(-3,3)
plt.legend(loc='upper left', fontsize=12)    
plt.tick_params(labelsize=fontsize)    
plt.ylabel('NAO', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
plt.close('all')

#----------------------------------------------------------------------------
# PLOT: SOI timerseries (yearly mean)
#----------------------------------------------------------------------------

figstr = 'soi_yearly.png'
titlestr = 'SOI: 1866-2022 (yearly mean)'
                        
fig, ax = plt.subplots(figsize=(15,10))     
plt.fill_between( df_soi_yearly.index, df_soi_yearly.soi, ls='-', lw=0.5, color='k', label='SOI: 1866-2022 (Phil Jones)')
plt.ylim(-3,3)
plt.legend(loc='upper left', fontsize=12)    
plt.tick_params(labelsize=fontsize)    
plt.ylabel('SOI', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
plt.close('all')

#----------------------------------------------------------------------------
# PLOT: stratospheric AOD timerseries (yearly mean)
#----------------------------------------------------------------------------

figstr = 'aod_yearly.png'
titlestr = 'AOD: 1781-2022 (yearly mean)'
                        
fig, ax = plt.subplots(figsize=(15,10))     
plt.fill_between( df_aod_yearly.index, df_aod_yearly.aod, ls='-', lw=0.5, color='k', label='AOD: 1781-2022 (eVolv2k)')
#plt.ylim(-3,3)
plt.legend(loc='upper right', fontsize=12)    
plt.tick_params(labelsize=fontsize)    
plt.ylabel('AOD(550)', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
plt.close('all')

#----------------------------------------------------------------------------
# PLOT: SSIM timerseries (yearly mean and SD)
#----------------------------------------------------------------------------

figstr = 'ssim_yearly.png'
titlestr = 'SSIM(t-1,t): yearly mean and SD' + ' (' + str(nsmooth) + 'yr MA)'
                        
fig, ax = plt.subplots(figsize=(15,10))     

plt.fill_between( df_ssim_analysis_yearly.index, df_ssim_analysis_yearly.ssim.rolling(nsmooth, center=True).mean() - df_ssim_analysis_yearly_sd.ssim.rolling(nsmooth, center=True).mean(), df_ssim_analysis_yearly.ssim.rolling(nsmooth, center=True).mean() + df_ssim_analysis_yearly_sd.ssim.rolling(nsmooth, center=True).mean(), color='k', alpha=0.1)
plt.fill_between( df_ssim_lat_yearly.index, df_ssim_lat_yearly.ssim.rolling(nsmooth, center=True).mean() - df_ssim_lat_yearly_sd.ssim.rolling(nsmooth, center=True).mean(), df_ssim_lat_yearly.ssim.rolling(nsmooth, center=True).mean() + df_ssim_lat_yearly_sd.ssim.rolling(nsmooth, center=True).mean(), color='r', alpha=0.1)
plt.fill_between( df_ssim_mat_yearly.index, df_ssim_mat_yearly.ssim.rolling(nsmooth, center=True).mean() - df_ssim_mat_yearly_sd.ssim.rolling(nsmooth, center=True).mean(), df_ssim_mat_yearly.ssim.rolling(nsmooth, center=True).mean() + df_ssim_mat_yearly_sd.ssim.rolling(nsmooth, center=True).mean(), color='b', alpha=0.1)
plt.fill_between( df_ssim_crutem_gridded_yearly.index, df_ssim_crutem_gridded_yearly.ssim.rolling(nsmooth, center=True).mean() - df_ssim_crutem_gridded_yearly_sd.ssim.rolling(nsmooth, center=True).mean(), df_ssim_crutem_gridded_yearly.ssim.rolling(nsmooth, center=True).mean() + df_ssim_crutem_gridded_yearly_sd.ssim.rolling(nsmooth, center=True).mean(), color='purple', alpha=0.1)
plt.plot( df_ssim_analysis_yearly.index, df_ssim_analysis_yearly.ssim.rolling(nsmooth, center=True).mean(), ls='-', lw=1, color='k', label='GloSAT (analysis)')
plt.plot( df_ssim_lat_yearly.index, df_ssim_lat_yearly.ssim.rolling(nsmooth, center=True).mean(), ls='-', lw=1, color='r', label='GloSAT (LAT)')
plt.plot( df_ssim_mat_yearly.index, df_ssim_mat_yearly.ssim.rolling(nsmooth, center=True).mean(), ls='-', lw=1, color='b', label='GloSAT (MAT)')
plt.plot( df_ssim_crutem_gridded_yearly.index, df_ssim_crutem_gridded_yearly.ssim.rolling(nsmooth, center=True).mean(), ls='-', lw=1, color='purple', label='GloSAT.p04c.EBCv0.6.LEKnorms21Nov22_alternativegrid')
plt.ylabel('SSIM(t-1,t)', fontsize=fontsize, color='k')
plt.tick_params(labelsize=fontsize, color='k')    
plt.ylim(0,1)
plt.legend(loc='lower left', fontsize=12)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
plt.close('all')

#----------------------------------------------------------------------------
# PLOT: coverage timerseries (yearly mean and SD)
#----------------------------------------------------------------------------

figstr = 'coverage_yearly.png'
titlestr = 'Global coverage fraction (' + str(latstep) + r'$^{\circ}$' + 'x ' + str(latstep) + r'$^{\circ}$ gridcells): yearly mean and SD'  + ' (' + str(nsmooth) + 'yr MA)'
                        
fig, ax = plt.subplots(figsize=(15,10))     

plt.fill_between( df_count_analysis_yearly.index, df_count_analysis_yearly['count'].rolling(nsmooth, center=True).mean() - df_count_analysis_yearly_sd['count'].rolling(nsmooth, center=True).mean(), df_count_analysis_yearly['count'].rolling(nsmooth, center=True).mean() + df_count_analysis_yearly_sd['count'].rolling(nsmooth, center=True).mean(), color='k', alpha=0.1)
plt.fill_between( df_count_lat_yearly.index, df_count_lat_yearly['count'].rolling(nsmooth, center=True).mean() - df_count_lat_yearly_sd['count'].rolling(nsmooth, center=True).mean(), df_count_lat_yearly['count'].rolling(nsmooth, center=True).mean() + df_count_lat_yearly_sd['count'].rolling(nsmooth, center=True).mean(), color='r', alpha=0.1)
plt.fill_between( df_count_mat_yearly.index, df_count_mat_yearly['count'].rolling(nsmooth, center=True).mean() - df_count_mat_yearly_sd['count'].rolling(nsmooth, center=True).mean(), df_count_mat_yearly['count'].rolling(nsmooth, center=True).mean() + df_count_mat_yearly_sd['count'].rolling(nsmooth, center=True).mean(), color='b', alpha=0.1)
plt.plot( df_count_analysis_yearly.index, (df_count_analysis_yearly['count']).rolling(nsmooth, center=True).mean(), ls='-', lw=1, color='k', label='GloSAT (analysis)')
plt.plot( df_count_lat_yearly.index, (df_count_lat_yearly['count']).rolling(nsmooth, center=True).mean(), ls='-', lw=1, color='r', label='GloSAT (LAT)')
plt.plot( df_count_mat_yearly.index, (df_count_mat_yearly['count']).rolling(nsmooth, center=True).mean(), ls='-', lw=1, color='b', label='GloSAT (MAT)')

plt.fill_between( df_count_crutem_gridded_yearly.index, df_count_crutem_gridded_yearly['count'].rolling(nsmooth, center=True).mean() - df_count_crutem_gridded_yearly_sd['count'].rolling(nsmooth, center=True).mean(), df_count_crutem_gridded_yearly['count'].rolling(nsmooth, center=True).mean() + df_count_crutem_gridded_yearly_sd['count'].rolling(nsmooth, center=True).mean(), color='purple', alpha=0.1)
plt.plot( df_count_crutem_gridded_yearly.index, (df_count_crutem_gridded_yearly['count']).rolling(nsmooth, center=True).mean(), ls='-', lw=1, color='purple', label='GloSAT.p04c.EBCv0.6.LEKnorms21Nov22_alternativegrid')

plt.ylabel('Coverage fraction', fontsize=fontsize, color='k')
plt.tick_params(labelsize=fontsize, colors='k')    
plt.ylim(0,1)
plt.legend(loc='upper left', fontsize=12)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=dpi, bbox_inches='tight')
plt.close('all')

#------------------------------------------------------------------------------
# PLOT: Area-averaged GMST (yearly mean and SD)
#------------------------------------------------------------------------------

figstr = 'gmst_yearly.png'
titlestr = 'Area-averaged GMST (' + str(latstep) + r'$^{\circ}$' + 'x ' + str(latstep) + r'$^{\circ}$ gridcells): yearly mean and SD' + ' (' + str(nsmooth) + 'yr MA)'
ystr = 'GMST, °C'

fig, ax = plt.subplots(figsize=(15,10))          

plt.fill_between( df_gmst_analysis_yearly.index, df_gmst_analysis_yearly.gmst.rolling(nsmooth, center=True).mean() - df_gmst_analysis_yearly_sd.gmst.rolling(nsmooth, center=True).mean(), df_gmst_analysis_yearly.gmst.rolling(nsmooth, center=True).mean() + df_gmst_analysis_yearly_sd.gmst.rolling(nsmooth, center=True).mean(), color='k', alpha=0.1)
plt.fill_between( df_gmst_lat_yearly.index, df_gmst_lat_yearly.gmst.rolling(nsmooth, center=True).mean() - df_gmst_lat_yearly_sd.gmst.rolling(nsmooth, center=True).mean(), df_gmst_lat_yearly.gmst.rolling(nsmooth, center=True).mean() + df_gmst_lat_yearly_sd.gmst.rolling(nsmooth, center=True).mean(), color='r', alpha=0.1)
plt.fill_between( df_gmst_mat_yearly.index, df_gmst_mat_yearly.gmst.rolling(nsmooth, center=True).mean() - df_gmst_mat_yearly_sd.gmst.rolling(nsmooth, center=True).mean(), df_gmst_mat_yearly.gmst.rolling(nsmooth, center=True).mean() + df_gmst_mat_yearly_sd.gmst.rolling(nsmooth, center=True).mean(), color='b', alpha=0.1)
plt.plot( df_gmst_analysis_yearly.index, df_gmst_analysis_yearly.gmst.rolling(nsmooth, center=True).mean(), ls='-', lw=1, color='k', label='GloSAT (analysis)')
plt.plot( df_gmst_lat_yearly.index, df_gmst_lat_yearly.gmst.rolling(nsmooth, center=True).mean(), ls='-', lw=1, color='r', label='GloSAT (LAT)')
plt.plot( df_gmst_mat_yearly.index, df_gmst_mat_yearly.gmst.rolling(nsmooth, center=True).mean(), ls='-', lw=1, color='b', label='GloSAT (MAT)')

plt.plot( df_gmst_crutem_yearly.index, df_gmst_crutem_yearly.gmst_global.rolling(nsmooth, center=True).mean(), ls='-', lw=3, color='orange', label='GloSAT.p04c.EBCv0.6.LEKnorms21Nov22_alternativegrid (global timeseries)')
plt.plot( df_gmst_crutem_gridded_yearly.index, df_gmst_crutem_gridded_yearly.gmst.rolling(nsmooth, center=True).mean(), ls='-', lw=2, color='purple', label='GloSAT.p04c.EBCv0.6.LEKnorms21Nov22_alternativegrid (WGS84 weighting)')
plt.plot( df_gmst_crutem_gridded_cosine_yearly.index, df_gmst_crutem_gridded_cosine_yearly.gmst.rolling(nsmooth, center=True).mean(), ls='-', lw=1, color='pink', label='GloSAT.p04c.EBCv0.6.LEKnorms21Nov22_alternativegrid (Cosine weighting)')

plt.ylim(-2,2)
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='upper left', ncol=1, fontsize=12)
plt.ylabel(ystr, fontsize=fontsize)
plt.title( titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=300, bbox_inches='tight')
plt.close(fig)

#----------------------------------------------------------------------------
# PLOT: GMST (LAT) and [NH,SH] timerseries (yearly mean)
#----------------------------------------------------------------------------

figstr = 'crutem_yearly.png'
titlestr = 'GloSAT.p04c.EBCv0.6.LEKnorms21Nov22_alternativegrid: 1781-2022 (yearly mean)'
                        
fig, ax = plt.subplots(figsize=(15,10))     
plt.fill_between( df_gmst_crutem_yearly.index, df_gmst_crutem_yearly.gmst_nh, ls='-', lw=0.2, color='r', alpha=1, label='NH')
plt.fill_between( df_gmst_crutem_yearly.index, df_gmst_crutem_yearly.gmst_sh, ls='-', lw=0.2, color='b', alpha=1, label='SH')
plt.plot( df_gmst_crutem_yearly.index, df_gmst_crutem_yearly.gmst_global, ls='-', lw=3, color='k', label='Global')
plt.ylim(-3,3)
plt.legend(loc='upper left', fontsize=12)    
plt.tick_params(labelsize=fontsize)    
plt.ylabel(r'GMST, $^{\circ}$C', fontsize=fontsize)
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
print("seaborn    : ", sns.__version__)

# -----------------------------------------------------------------------------
print('** END')



