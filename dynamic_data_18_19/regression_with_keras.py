# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:27:04 2020

@author: JULSP
"""

import pandas as pd
from os.path import dirname
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import RobustScaler

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
data_clean = data_clean[data_clean.iwrap_type_from_dataset != 'Other ship'] 
data_processed = data_clean[data_clean.length_from_data_set < 400] #vessels that are over 400m do not exist
data_processed = data_processed[data_processed.Speed_max <65] #boats that go faster than that are helicopters
data_processed = data_processed[data_processed.Speed_max >0] #boats that stand still

data_processed = data_processed.reset_index(drop =True) #reset index
#%%
"""
0.
Picking the x and y variables
"""
X = data_processed[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
y = data_processed[['length_from_data_set']]
y = y.values.ravel() #somethe model wants this to be an array

#need this later for visualisation
feature_names = ['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']

#%%
"""
0.
Splitting the data
"""
#split into test and train+val 
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2,  random_state=0)
# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

print("\nSize of training set: {}   size of validation set: {}   size of test set:" " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))