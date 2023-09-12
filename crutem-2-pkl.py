#! /usr/bin python

#------------------------------------------------------------------------------
# PROGRAM: crutem-2-pkl.py
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

filename_crutem = 'DATA/GloSAT.p04c.EBCv0.6.LEKnorms21Nov22_alternativegrid-178101-202112.timeseries.txt'

#------------------------------------------------------------------------------    
# LOAD: CRUTEM5 global and hemispherocal GMST
#------------------------------------------------------------------------------

print('loading CRUTEM5 GMST ...')

nheader = 5
f = open(filename_crutem)
lines = f.readlines()
years = []
months = []
months = []
gmst_global = [] 
gmst_nh = [] 
gmst_sh = [] 

for i in range(nheader,len(lines)):

    words = lines[i].split()   
    year = int(words[0])
    month = int(words[1])
    earth = float(words[2])
    nh = float(words[3])
    sh = float(words[4])

    years.append(year)                                     
    months.append(month)            
    gmst_global.append(earth)            
    gmst_nh.append(nh)            
    gmst_sh.append(sh)            

f.close()    
gmst_global = np.array(gmst_global)
gmst_nh = np.array(gmst_nh)
gmst_sh = np.array(gmst_sh)

da = pd.DataFrame()             
da['year'] = years
da['month'] = months
da['gmst_global'] = gmst_global
da['gmst_nh'] = gmst_nh
da['gmst_sh'] = gmst_sh

dates = [ pd.to_datetime( str(da.year[i]) + '-' + str(da.month[i]).zfill(2) + '-01', format='%Y-%m-%d' ) for i in range(len(da)) ]
df_crutem = pd.DataFrame({'gmst_global':da.gmst_global.values, 'gmst_nh':da.gmst_nh.values, 'gmst_sh':da.gmst_sh.values }, index=dates )

# REPLACE: fill value -99.99 with np.nan

df_crutem = df_crutem.replace(-9.999,np.nan)

'''
# REPLACE: global with NH value pre-1857

mask = df_crutem['gmst_nh'].index.year < 1857
df_crutem['gmst_global'][mask] = df_crutem['gmst_nh'][mask]
'''

# SAVE: GMST (CRUTEM) merged timeseries to pkl

df_crutem.to_pickle( 'df_gmst_crutem.pkl', compression='bz2' )

# -----------------------------------------------------------------------------
# VERSIONING:
# -----------------------------------------------------------------------------

print("numpy      : ", np.__version__) 
print("pandas     : ", pd.__version__) 

# -----------------------------------------------------------------------------
print('** END')
