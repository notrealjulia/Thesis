# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:45:30 2020

@author: JULSP

Columns into a list
"""

import pandas as pd
from os.path import dirname
import numpy as np


#%%
"""
Loading data
Define path to dynamic data and load the static data file
"""
path = dirname(__file__)
dynamic_data = pd.read_csv(path + "/311045200.csv",  sep='	', index_col=None, error_bad_lines=False) 


#%%

dynamic = dynamic_data.set_index('timestamp')
#transforming index into resampable form 
dynamic.index = pd.to_datetime(dynamic.index)

# Get one day (first day)            
date = sorted(dynamic.index)[0].isoformat()[:10] 

first_day = dynamic.loc[date]

# resample data takes nearest available value
data_resample = first_day.resample('T').bfill()
data_resample = data_resample.dropna()

#only keeping speed
data_resample = data_resample[['sog [kn]']]


#%%
sequence=data_resample.aggregate(lambda x: [x.tolist()], axis=0).map(lambda x:x[0]).reset_index(drop = True)
# print('ser_aggCol (collapse each column to a list)',sequence, sep='\n', end='\n\n\n')

print(sequence[0])
#%%


df = pd.DataFrame( columns = ['date', 'mmsi', 'speed']) 

s = list(sequence[0])

df['speed'] = s

