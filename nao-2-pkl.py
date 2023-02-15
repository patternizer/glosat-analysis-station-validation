#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: nao-2-pkl.py
#------------------------------------------------------------------------------
# Version 0.1
# 31 January, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pickle

# Plotting libraries:
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt; plt.close('all')
import seaborn as sns; sns.set()

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

print("numpy      : ", np.__version__) 
print("pandas     : ", pd.__version__) 
print("matplotlib : ", matplotlib.__version__)
print("seaborn    : ", sns.__version__) 

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

filename_nao1 = 'DATA/naomonjurg.dat' # Juerg Luterbacher reconstructed NAO 1658-1900
filename_nao2 = 'DATA/nao.dat' # Phil Jones 1821-2022

fontsize = 16
dpi = 300

#------------------------------------------------------------------------------    
# LOAD: NAO indices
#------------------------------------------------------------------------------
# https://crudata.uea.ac.uk/cru/data/nao/viz.htm

print('loading NAO indices ...')

# LOAD: Luterbacher reconstructions 1658-1900
     
nheader = 26
f = open(filename_nao1)
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

# TRIM: to GloSAT range up to 1821 when Phil Jones' NAO dataset begins

da = da[ (da.year >= 1781) & (da.year < 1821) ].reset_index(drop=True)
dates = [ pd.to_datetime( str(da.year[i]) + '-' + str(da.month[i]).zfill(2) + '-01', format='%Y-%m-%d' ) for i in range(len(da)) ]
df_nao1 = pd.DataFrame({'nao':da.nao.values}, index=dates )

# LOAD: Phil Jones CRU/UEA NAO

nheader = 0
f = open(filename_nao2)
lines = f.readlines()
years = []
months = []
obs = [] 

for i in range(nheader,len(lines)):

    words = lines[i].split()   
    year = int(words[0])

    for j in range(1,13):

        month = j
        val = float(words[j])

        years.append(year)                                     
        months.append(month)            
        obs.append(val)            

f.close()    
obs = np.array(obs)

da = pd.DataFrame()             
da['year'] = years
da['month'] = months
da['nao'] = obs

dates = [ pd.to_datetime( str(da.year[i]) + '-' + str(da.month[i]).zfill(2) + '-01', format='%Y-%m-%d' ) for i in range(len(da)) ]
df_nao2 = pd.DataFrame({'nao':da.nao.values}, index=dates )

# REPLACE: fill value -99.99 with np.nan

df_nao2 = df_nao2.replace(-99.99,np.nan)

# MERGE: dataframes

df_nao = pd.concat([df_nao1, df_nao2], ignore_index=False)

# SAVE: NAO merged timeseries to pkl

df_nao.to_pickle( 'df_nao_1781_2022.pkl', compression='bz2' )

#----------------------------------------------------------------------------
# PLOT: NAO timerseries
#----------------------------------------------------------------------------

figstr = 'nao_yearly_mean.png'
titlestr = 'Merged NAO: 1781-1820 (Juerg Luterbacher), 1821-2022 (Phil Jones): yearly mean'
                        
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
print('** END')
