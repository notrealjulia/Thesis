# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:14:51 2020

@author: JULSP
"""

import pandas as pd
from os.path import dirname
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import mglearn
from sklearn.metrics import mean_absolute_error
#%%
"""
Loading Data
& Pre-Processing
"""

path = dirname(__file__)
data = pd.read_csv(path + "/DYNAMIC_DAYS_test.csv") #column 3 to 12 is dynamic data

"""
dropping nan
"""

data_clean = data.drop(['trips'], axis=1)
#data_clean = data_clean[data_clean.iwrap_type_from_dataset != 'Other ship']
data_clean = data_clean.dropna()
data_processed = data_clean[data_clean.length_from_data_set < 400] #vessels that are over 400m do not exist
data_processed = data_processed[data_processed.length_from_data_set > 3] #do not want to look at tiny boats
data_processed = data_processed[data_processed.Speed_mean != 0] #boats that stad still
data_processed = data_processed[data_processed.Speed_max <65] #boats that go faster than that are helicopters
data_processed = data_processed[data_processed.Speed_max >2] #boats that stand still

data_processed = data_processed.reset_index(drop =True) #reset index