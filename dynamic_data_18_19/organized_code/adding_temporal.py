# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:13:03 2020

@author: JULSP
"""

import pandas as pd
from os.path import dirname
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow import keras

#%%

"""
Loading and pre-processing
"""
path = dirname(__file__)
# data without "Unidetified" vessels
data_all = pd.read_csv(path + "/data_jun_jul_aug_sep_oct.csv")
data_clean = data_all.drop(['trips'], axis=1)
data_clean = data_clean.dropna()
# data_clean = data_clean[data_clean.iwrap_type_from_dataset != 'Other ship'] 
data_processed = data_clean[data_clean.length_from_data_set < 400] #vessels that are over 400m do not exist
data_processed = data_processed[data_processed.Speed_max <65] #boats that go faster than that are helicopters
data_processed = data_processed[data_processed.Speed_max >0] #boats that stand still

data_processed = data_processed.reset_index(drop =True) #reset index

 #%%

data_processed['Date'] = pd.to_datetime(data_processed['date'], errors='coerce')

data_processed['weekday'] = data_processed['Date'].dt.dayofweek


data_processed.to_csv('data_jun_jul_aug_sep_oct_withdays.csv', index=False)