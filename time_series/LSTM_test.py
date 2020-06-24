# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:31:48 2020

@author: JULSP

First test of the small dataset training
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing import sequence

# fix random seed for reproducibility
np.random.seed(0)

import pandas as pd
from os.path import dirname
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow import keras

#%%
"""
Loading 

and pre-processing
"""
path = dirname(__file__)
data = pd.read_csv(path + "/test_speed_sequence_mul100.csv")

#Ecoding the labels
lb = LabelEncoder()
data['iwrap_cat'] = lb.fit_transform(data['iwrap_type_from_dataset'])

#picking X and y and splitting
X = data[['padded_sequences']]
y = data[['iwrap_cat']]
y = y.values.ravel() #somethe model wants this to be an array
unique_class = np.unique(y)
#finishing encoding
y = keras.utils.to_categorical(y)

#%%
X_reshaped = X.values.ravel()

X_reshaped = X_reshaped.reshape(80, 1140, 1)

#%%

#split into test and train+val 
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2,  random_state=0, stratify = y)
# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1, stratify = y_trainval)

print("\nSize of training set: {}   size of validation set: {}   size of test set:" " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

#scale iof more dimensions
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_valid = scaler.transform(X_valid)
# X_test = scaler.transform(X_test)

#%%

#create model

model = Sequential()
#get number of columns in training data
n_cols = X_train.shape[1]
#add model layers
model.add(keras.layers.Flatten(input_shape = [1]))
model.add(LSTM(100))
model.add(keras.layers.Dense(8, activation="softmax"))

#compile model using mse as a measure of model performance
model.compile(loss="categorical_crossentropy", #don't change the loss function
              optimizer="adam",
              metrics=["accuracy"])

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=10)


#%%

print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
