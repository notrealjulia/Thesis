# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:25:30 2020

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
"""
0.
Picking the x and y variables
Basing on the results we got from Gradient Booster feature prioritization
"""
from sklearn.preprocessing import LabelEncoder
#Ecoding the labels
lb = LabelEncoder()
data_processed['iwrap_cat'] = lb.fit_transform(data_processed['iwrap_type_from_dataset'])
#picking X and y and splitting
X = data_processed[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
y = data_processed[['iwrap_cat']]
y = y.values.ravel() #somethe model wants this to be an array
unique_class = np.unique(y)
#%%

#calculating class weights

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = unique_class, y=y)

#%%

#need to use categorical_crossentropy as the loss function with categorical classes

y = keras.utils.to_categorical(y)

#%%

#split into test and train+val 
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2,  random_state=0, stratify = y)
# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1, stratify = y_trainval)

print("\nSize of training set: {}   size of validation set: {}   size of test set:" " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)



#%%

#create model

model = Sequential()
#get number of columns in training data
n_cols = X_train.shape[1]
#add model layers
model.add(keras.layers.Flatten(input_shape = [10]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(8, activation="softmax"))

#compile model using mse as a measure of model performance
model.compile(loss="categorical_crossentropy", #don't change the loss function
              optimizer="adam",
              metrics=["accuracy"])

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=10)


#%%

#train model

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=30, class_weight=class_weights, callbacks=[early_stopping_monitor])

#Ends on
#Epoch 21/30
#28797/28797 [==============================] - 5s 165us/sample - loss: 0.7868 - accuracy: 0.7209 - val_loss: 0.8270 - val_accuracy: 0.7114

#%%

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.xlabel("Epochs")
plt.title("Learning rate of a Neural Network built on TensorFlow")
plt.show()

#%%
#TODO add >>> model.evaluate(X_test, y_test)
