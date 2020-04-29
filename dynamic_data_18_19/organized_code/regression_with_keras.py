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
from sklearn.preprocessing import StandardScaler

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
Basing on the results we got from Gradient Booster feature prioritization
"""

X = data_processed[['Speed_median', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_std']]
y = data_processed[['length_from_data_set']]
y = y.values.ravel() #somethe model wants this to be an array

#need this later for visualisation
feature_names = ['Speed_median', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_std']
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

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


#%%
"""
1.
Simple Model architecture 
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#create model
model = Sequential()

#get number of columns in training data
n_cols = X_train.shape[1]

#add model layers
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#compile model using mse as a measure of model performance
model.compile(optimizer='adam', loss='mean_squared_error')

from tensorflow.keras.callbacks import EarlyStopping
#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)
#train model
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50, callbacks=[early_stopping_monitor])

#%%
"""
 Learning curves visualisation
"""
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
#plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

#%%

"""
1.
More powerful model
"""

#training a new model on the same data to show the effect of increasing model capacity

#create model
model_mc = Sequential()

#add model layers
model_mc.add(Dense(200, activation='relu', input_shape=(n_cols,)))
model_mc.add(Dense(200, activation='relu'))
model_mc.add(Dense(200, activation='relu'))
model_mc.add(Dense(1))

#compile model using mse as a measure of model performance
model_mc.compile(optimizer='adam', loss='mean_squared_error')
#train model
history = model_mc.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50, callbacks=[early_stopping_monitor])

#%%

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
#plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()


