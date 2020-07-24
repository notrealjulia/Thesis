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
from tensorflow.keras.utils import normalize
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from tensorflow_docs.modeling import EpochDots

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

X = data_processed[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
y = data_processed[['length_from_data_set']]
y = y.values.ravel() #somethe model wants this to be an array

#need this later for visualisation
feature_names = ['Speed_median', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_std']

X = normalize(X, axis = 1)
#%%
"""
0.
Splitting the data
"""
#split into test and train+val 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=0)

print("\nSize of training set: {}    size of test set:" " {}\n".format(X_train.shape[0], X_test.shape[0]))

#%%


print(X_train.shape[1])

#%%
"""
1.
Simple Model architecture 
"""


# #%%

# def build_model():
#   model = keras.Sequential([
#     keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(1)
#   ])

#   optimizer = tf.keras.optimizers.RMSprop(0.001)

#   model.compile(loss='mse',
#                 optimizer=optimizer,
#                 metrics=['mae', 'mse'])
#   return model

# #%%
# model = build_model()
# model.summary()


# #%%
# EPOCHS = 100

# history = model.fit(
#   X_train, y_train,
#   epochs=EPOCHS, validation_split = 0.2, verbose=0,
#    callbacks=[tfdocs.modeling.EpochDots()])

# #%%

# scores = model.evaluate(X_test, y_test, verbose=0)
# print(scores)

#%%
model = Sequential()
#get number of columns in training data
n_cols = X_train.shape[1]
#add model layers
model.add(keras.layers.Flatten(input_shape = [10]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(32, activation="relu"))
model.add(keras.layers.Dense(24, activation="relu"))
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1))

#compile model using mse as a measure of model performance
model.compile(loss=Huber(delta=1.0, reduction="auto", name="huber_loss"), #don't change the loss function
              optimizer=RMSprop(0.001),
              metrics=["mse", "mae"])

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=10)


#train model
history = model.fit(X_train, y_train, validation_split = 0.2, epochs=100, callbacks=[early_stopping_monitor])

#%%
y_pred = model.predict(X_test)
# scores = model.evaluate(X_test, y_test, verbose=0)
# print(scores)
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error

MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)

print("testing score: {:.3f}\n".format(model.score(X_test, y_test)))

#%%
"""
 Learning curves visualisation
"""
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

#%%

"""
1.
More powerful model
the model stops epochs automatically when it stops improving the results
currelt stops at epoch 18
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
plt.show()

"""
 Learning curves visualisation
"""

