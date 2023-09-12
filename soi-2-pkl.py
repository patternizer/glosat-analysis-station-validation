#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: soi-2-pkl.py
#------------------------------------------------------------------------------
# Version 0.1
# 24 February, 2023
# Michael Taylor
# michael DOT a DOT taylor AT uea DOT ac DOT uk 
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pickle

#----------------------------------------------------------------------------
# SETTINGS
#----------------------------------------------------------------------------

filename_soi = 'DATA/soi.dat' # Phil Jones 1866-2022

#------------------------------------------------------------------------------    
# LOAD: SOI indices
#------------------------------------------------------------------------------
# https://crudata.uea.ac.uk/cru/data/soi/

print('loading SOI indices ...')

# LOAD: Phil Jones CRU/UEA SOI

nheader = 0
f = open(filename_soi)
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
da['soi'] = obs

dates = [ pd.to_datetime( str(da.year[i]) + '-' + str(da.month[i]).zfill(2) + '-01', format='%Y-%m-%d' ) for i in range(len(da)) ]
df_soi = pd.DataFrame({'soi':da.soi.values}, index=dates )

# REPLACE: fill value -99.99 with np.nan

df_soi = df_soi.replace(-99.99,np.nan)

# SAVE: NAO merged timeseries to pkl

df_soi.to_pickle( 'df_soi.pkl', compression='bz2' )

#------------------------------------------------------------------------------
# VERSIONING:
#------------------------------------------------------------------------------

print("numpy      : ", np.__version__) 
print("pandas     : ", pd.__version__) 

#------------------------------------------------------------------------------
print('** END')


