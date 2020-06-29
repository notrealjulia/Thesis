# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:58:20 2020

@author: JULSP
"""

import pandas as pd
from os.path import dirname
import numpy as np


seq_path = "D:/Thesis_data/data"        

# df_test_y = pd.read_csv(seq_path + "/df_seq_test_full.csv") 
df_train_y =  pd.read_csv(seq_path + "/df_train_y.csv") 

# df_test_speed_min = pd.read_csv(seq_path + "/df_test_speed_min.csv") 
# df_test_cog_min = pd.read_csv(seq_path + "/df_test_cog_min.csv") 
# df_test_lat_min = pd.read_csv(seq_path + "/df_test_lat_min.csv") 
# df_test_lon_min = pd.read_csv(seq_path + "/df_test_lon_min.csv")


df_train_speed_min = pd.read_csv(seq_path + "/df_train_speed_min.csv") 
df_train_cog_min = pd.read_csv(seq_path + "/df_train_cog_min.csv") 
df_train_lat_min = pd.read_csv(seq_path + "/df_train_lat_min.csv") 
df_train_lon_min = pd.read_csv(seq_path + "/df_train_lon_min.csv")


#%%


X_train_array_speed = df_train_speed_min.to_numpy() #turning it into an array 
X_train_array_cog = df_train_cog_min.to_numpy() #turning it into an array 
X_train_array_lat = df_train_lat_min.to_numpy() #turning it into an array 
X_train_array_lon = df_train_lon_min.to_numpy() #turning it into an array 

X_train_array = np.concatenate((X_train_array_speed,
                                X_train_array_cog,
                                X_train_array_lat,
                                X_train_array_lon), axis=1)


length_train = len(X_train_array_cog)

X_train = X_train_array.reshape(length_train, 500, 4, order = 'F')



#%%


from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, RNN
from tensorflow.keras.callbacks import EarlyStopping

lb = LabelEncoder()
#Ecoding the labels for training

df_train_y['iwrap_cat'] = lb.fit_transform(df_train_y['iwrap_type_from_dataset'])
y_train = df_train_y[['iwrap_cat']]
y_train = y_train.values.ravel() #somethe model wants this to be an array
unique_class = np.unique(y_train)
y_train = keras.utils.to_categorical(y_train)

#Encoding labels for testing

# df_seq_1['iwrap_cat'] = lb.fit_transform(df_seq_1['iwrap_type_from_dataset'])
# y_test = df_seq_1[['iwrap_cat']]
# y_test = y_test.values.ravel() #somethe model wants this to be an array
# unique_class = np.unique(y_test)
# y_test = keras.utils.to_categorical(y_test)

#%%

model = Sequential()

model.add(LSTM(200))
model.add(Dense(8, activation="softmax"))

#compile model using mse as a measure of model performance
model.compile(loss="categorical_crossentropy", #don't change the loss function
              optimizer="adam",
              metrics=["accuracy"])

#set early stopping monitor so the model stops training when it won't improve anymore
# early_stopping_monitor = EarlyStopping(patience=10)


#%%

model.fit(X_train, y_train, epochs=3)
# Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

