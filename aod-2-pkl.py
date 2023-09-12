#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: aod-2-pkl.py
#------------------------------------------------------------------------------
# Version 0.1
# 7 March, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import rcParams
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns; sns.set()

from scipy import stats
from scipy.signal import find_peaks

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

filename_aod1 = 'DATA/aod/volcanic_sAOD_monthly_-50001-201912_new.csv' # 
filename_aod2 = 'DATA/aod/tau.line_2012.12.txt' # 

fontsize = 16
year_start = 1781
year_end = 2022

# -----------------------------------------------------------------------------
# METHODS
# -----------------------------------------------------------------------------

def convert_cftime_year_decimal ( t_monthly ):
    
    year = [t_monthly[i].year for i in range(len(t_monthly))]
    year_frac = []
    for i in range(len(t_monthly)):
        if i%12 == 0:
            istart = i
            iend = istart+11   
            frac = np.cumsum([t_monthly[istart+j].day for j in range(12)])
            year_frac += list(frac/frac[-1])
        else:
            i += 1
    year_decimal = [float(year[i])+year_frac[i] for i in range(len(year))]    
        
    return year_decimal

#------------------------------------------------------------------------------    
# LOAD: AOD550 indices from dataset 1
#------------------------------------------------------------------------------

print('loading AOD indices #1...')

nheader = 1
f = open(filename_aod1)
lines = f.readlines()
year_dec = []
obs = [] 

for i in range(nheader,len(lines)):

    words = lines[i].strip().split(',')
    date = float(words[0]) 
    val = float(words[1])

    year_dec.append(date)                                     
    obs.append(val)            

f.close()    
obs = np.array(obs)

df1 = pd.DataFrame()             
df1['year_dec'] = year_dec
df1['aod'] = obs

# ROUND: year decimal to 3dp

df1['year_dec'] = df1['year_dec'].round(3)

#------------------------------------------------------------------------------    
# LOAD: AOD550 indices from dataset 2
#------------------------------------------------------------------------------

print('loading AOD indices #2...')

nheader = 4
f = open(filename_aod2)
lines = f.readlines()
year_dec = []
obs_global = [] 
obs_nh = []
obs_sh = []

for i in range(nheader,len(lines)):

    words = lines[i].strip().split()
    date = float(words[0]) 
    val_global = float(words[1])
    val_nh = float(words[2])
    val_sh = float(words[3])

    year_dec.append( date )                                     
    obs_global.append( val_global )            
    obs_nh.append( val_nh )            
    obs_sh.append( val_sh )            

f.close()    
obs_global = np.array(obs_global)
obs_nh = np.array(obs_nh)
obs_sh = np.array(obs_sh)

df2 = pd.DataFrame()             
df2['year_dec'] = year_dec
df2['aod_global'] = obs_global
df2['aod_nh'] = obs_nh
df2['aod_sh'] = obs_sh
             
# MERGE: dataframes

df = df1.copy().merge(df2, how='left', on='year_dec')

# TRIM: to GloSAT 1781-2022 date range

df = df[df['year_dec'] >= year_start ].reset_index(drop=True)

# Solve Y1677-Y2262 Pandas bug with xarray:        
# t_monthly_hadcet = pd.date_range(start=str(db['year'].iloc[0]), periods=len(ts_monthly_hadcet), freq='MS')                  
#t_monthly_hadcet = xr.cftime_range(start=str(db['year'].iloc[0]), periods=len(ts_monthly_hadcet), freq='MS', calendar='noleap')     
#t_monthly_hadcet = convert_cftime_year_decimal( t_monthly_hadcet )

# CONVERT: 30/360  day count convention time fractions back to datetimes!

years  = np.floor( df['year_dec'] ).astype(int)
fractions = df['year_dec'] - years

midmonths = np.round( [ 30/360/2 + i*(30/360) for i in range(12) ], 3 )
midmonths_dict = dict( zip( midmonths, range(1 ,13) ) )
months = [ midmonths_dict[i] for i in np.round(fractions.values, 3) ]

dates = [ pd.to_datetime( str(years[i]) + '-' + str(months[i]).zfill(2) + '-15', format='%Y-%m-%d' ) for i in range(len(df)) ]

# CONSTRUCT: dataframe and set datetime index

df_aod = pd.DataFrame({'aod':df.aod.values, 'aod_global':df.aod_global.values, 'aod_nh':df.aod_nh.values, 'aod_sh':df.aod_sh.values }, index=dates )

# SAVE: AOD merged timeseries to pkl

df_aod.to_pickle( 'df_aod.pkl', compression='bz2' )

#------------------------------------------------------------------------------
# DETECT: peaks
#------------------------------------------------------------------------------

aod_threshold = 0.05
#peaks = df['aod'][ (np.abs( stats.zscore(df['aod']) ) >= 2 ) ].index # 2 sd
peaks, properties = find_peaks(df_aod['aod'], prominence=aod_threshold, width=1)
#peaks_times = df_aod['year_dec'][peaks]
peaks_times = df_aod.index[peaks]

#DatetimeIndex(['1784-01-15', '1809-08-15', '1815-11-15', '1831-07-15',
#               '1835-07-15', '1862-06-15', '1884-01-15', '1902-12-15',
#               '1912-11-15', '1963-08-15', '1983-01-15', '1992-06-15'],
#              dtype='datetime64[ns]', freq=None)

#------------------------------------------------------------------------------
# PLOT
#------------------------------------------------------------------------------

titlestr = 'Stratospheric aerosol: ' + str(year_start) + '-' + str(year_end)
figstr = 'aod_peaks.png'

fig, axs = plt.subplots(4,1, figsize=(15,10), sharex=True, sharey=True)          
axs[0].fill_between( df['year_dec'], df['aod'], ls='-', lw=0.5, color='k', label='eVolv2k')
axs[1].fill_between( df['year_dec'], df['aod_global'], ls='-', lw=0.5, color='k', label='NASA GISS: global')
axs[2].fill_between( df['year_dec'], df['aod_nh'], ls='-', lw=0.5, color='k', label='NASA GISS: NH')
axs[3].fill_between( df['year_dec'], df['aod_sh'], ls='-', lw=0.5, color='k', label='NASA GISS: SH')
#axs[0].axvline( x=1781, color='r', label='start of GloSAT')    

mask_giss = df['year_dec'] >= 1850
mask_giss_peaks = peaks >= df['year_dec'].index[mask_giss][0]
peaks_giss = peaks[mask_giss_peaks]

axs[0].plot( df['year_dec'].loc[peaks], df['aod'].loc[peaks], 'rx', label=r'peak(AOD$\geq$' + str(aod_threshold) + ')')
axs[1].plot( df['year_dec'].loc[peaks_giss], df['aod'].loc[peaks_giss], 'rx', label=r'eVolv2k peak')
axs[2].plot( df['year_dec'].loc[peaks_giss], df['aod'].loc[peaks_giss], 'rx', label=r'eVolv2k peak')
axs[3].plot( df['year_dec'].loc[peaks_giss], df['aod'].loc[peaks_giss], 'rx', label=r'eVolv2k peak')

for ax in axs.flat:

    ax.tick_params(labelsize=fontsize)    
    ax.legend(loc='upper right', ncol=1, fontsize=12)
    ax.set_ylabel( 'AOD(550)', fontsize=fontsize)

plt.xlim( year_start, year_end )
plt.ylim(0,0.4)
axs[0].set_title( titlestr, fontsize=fontsize)
plt.savefig( figstr, dpi=300, bbox_inches='tight')
plt.close()

#------------------------------------------------------------------------------
# VERSIONING:
#------------------------------------------------------------------------------

print("numpy      : ", np.__version__) 
print("pandas     : ", pd.__version__) 

#------------------------------------------------------------------------------
print('** END')


