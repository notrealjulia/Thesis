# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:18:10 2020

@author: JULSP
"""
import pandas as pd
from os.path import dirname
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, RNN
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import normalize
from sklearn.model_selection import train_test_split

 
seq_path = "D:/Thesis_data/data/all"

df_y_1 =  pd.read_csv(seq_path + "/df_Y_june.csv") 
df_sog_1 =  pd.read_csv(seq_path + "/df_speed_june.csv") 
df_cog_1 =  pd.read_csv(seq_path + "/df_cog_june.csv") 
df_lat_1 =  pd.read_csv(seq_path + "/df_lat_june.csv") 
df_lon_1 =  pd.read_csv(seq_path + "/df_lon_june.csv") 


df_y_2 =  pd.read_csv(seq_path + "/df_Y_july.csv") 
df_sog_2 =  pd.read_csv(seq_path + "/df_speed_july.csv") 
df_cog_2 =  pd.read_csv(seq_path + "/df_cog_july.csv") 
df_lat_2 =  pd.read_csv(seq_path + "/df_lat_july.csv") 
df_lon_2 =  pd.read_csv(seq_path + "/df_lon_july.csv") 


limit = 500

#%%
#adding all the data together

df_y = pd.concat([df_y_1, df_y_2], ignore_index=True)
df_sog = pd.concat([df_sog_1, df_sog_2], ignore_index=True)
df_cog = pd.concat([df_cog_1, df_cog_2], ignore_index=True)
df_lat = pd.concat([df_lat_1, df_lat_2], ignore_index=True)
df_lon = pd.concat([df_lon_1, df_lon_2], ignore_index=True)


#%%
#getting the X values
#limiting the amont of data for LSTM, filling NaN and limiting to 300

def reshaping_seq(df, limit):
    
    df = df.fillna(0)
    df = df.drop(df.iloc[:, limit:], axis = 1) #cut off after 500 values, LSTM can't handle 1440
    array_from_df = df.to_numpy()
    
    return array_from_df

sog_array = reshaping_seq(df_sog, limit)
cog_array = reshaping_seq(df_cog, limit)
lat_array = reshaping_seq(df_lat, limit)
lon_array = reshaping_seq(df_lon, limit)


#%%
# concatinating all data together and reshaping for the LSTM unit 

X_array = np.concatenate((sog_array,
                          cog_array,
                          lat_array,
                          lon_array), axis=1)


length_for_reshape = len(sog_array)

X = X_array.reshape(length_for_reshape, limit, 4, order = 'F')

X_speed = sog_array.reshape(length_for_reshape, limit, 1)

#%%
#normalizing 

X_norm = normalize(X, axis = 1)

#%%
#getting the Y values

lb = LabelEncoder()
#Ecoding the labels for training

df_y['iwrap_cat'] = lb.fit_transform(df_y['iwrap_type_from_dataset'])
y = df_y[['iwrap_cat']]
y = y.values.ravel() #somethe model wants this to be an array
# unique_class = np.unique(y) #for report
y = keras.utils.to_categorical(y)


#%%
#Train - test split

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

#%%


model = Sequential()

model.add(LSTM(units = 100))
model.add(Dense(8, activation="softmax"))

#compile model using mse as a measure of model performance
model.compile(loss="categorical_crossentropy", #don't change the loss function
              optimizer="adam",
              metrics=["accuracy"])

#set early stopping monitor so the model stops training when it won't improve anymore
# early_stopping_monitor = EarlyStopping(patience=10)


#%%

model.fit(X_train, y_train, epochs=3, batch_size=4, validation_split=0.2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))