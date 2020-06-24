# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:44:42 2020

@author: JULSP

getting a list of static data for time series for sequence classification
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
data = pd.read_csv(path + "/Trips_and_VesselStatus_Static.csv",  sep='	', index_col=None, error_bad_lines=False) 

#%%
"""
Selecting mmsi subset
active ships with over a 5000 signals
of a certain size 
"""
active_subset = data[(data['status']== 'active') & (data['signals']>= 5000)]
data_static = active_subset[['mmsi', 'iwrap_type_from_dataset','org_type_info_from_data', 'length_from_data_set', 'width', 'trips', 'signals']]
data_static_cleaned = data_static[(data_static['length_from_data_set'] > 2) & (data_static['width'] > 2)]
data_static_cleaned = data_static_cleaned[data_static_cleaned['length_from_data_set'] <= 400]
data_static_cleaned = data_static_cleaned.reset_index(drop =True) #reset index

#%%
"""
0.
Limiting each amount of vessels 
for testing the neural network
"""

amount_limit = 10

df1 = data_static_cleaned[data_static_cleaned['iwrap_type_from_dataset'] == 'General cargo ship'][:amount_limit]
df2 = data_static_cleaned[data_static_cleaned['iwrap_type_from_dataset'] == 'Oil products tanker'][:amount_limit]
df3 = data_static_cleaned[data_static_cleaned['iwrap_type_from_dataset'] == 'Passenger ship'][:amount_limit]
df4 = data_static_cleaned[data_static_cleaned['iwrap_type_from_dataset'] == 'Pleasure boat'][:amount_limit]
df5 = data_static_cleaned[data_static_cleaned['iwrap_type_from_dataset'] == 'Other ship'][:amount_limit]
df6 = data_static_cleaned[data_static_cleaned['iwrap_type_from_dataset'] == 'Fishing ship'][:amount_limit]
df7 = data_static_cleaned[data_static_cleaned['iwrap_type_from_dataset'] == 'Support ship'][:amount_limit]
df8 = data_static_cleaned[data_static_cleaned['iwrap_type_from_dataset'] == 'Fast ferry'][:amount_limit]

data_static_cleaned = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8]) #put it all together
data_static_cleaned = data_static_cleaned.sample(frac=1).reset_index(drop=True) #shuffle
static_subset = data_static_cleaned.reset_index(drop =True) #reset index

#%%

print(static_subset.columns)

static_subset = static_subset[['mmsi','iwrap_type_from_dataset','length_from_data_set','width' ]]

static_subset.to_csv(path + '/static_subset.csv', index = False)