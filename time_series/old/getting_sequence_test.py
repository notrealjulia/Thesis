# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:45:30 2020

@author: JULSP

Filling out a df to look like it should for LSTM
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
#dynamic_data_path = "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"
dynamic_data_path = "D:/Thesis_data/export"
data = pd.read_csv(path + "/static_subset.csv") 
data['speed_sequence'] = np.nan
data['date'] = np.nan

data['speed_sequence'] = data['speed_sequence'].astype('object')

# data = data[:2]

#%%

#go through the static data fill out speed and dates

def get_dynamic_first_day(static_data):
    
    for i in range(len(static_data)):
        print(i)
        first_day = pd.DataFrame()
            #getting mmsi name of the ship from the static data 
        mmsi = static_data.mmsi.iloc[i]
        print("getting dynamic data for ship ID", mmsi)
            #loading the dynamic file for that specific ship
        dynamic = pd.read_csv(dynamic_data_path +"/{}.csv".format(mmsi), sep="\t", index_col=None, error_bad_lines=True)
            #using the date time stamp as the index to sort by days
        dynamic = dynamic.set_index('timestamp')
            #transforming index into resampable form 
        dynamic.index = pd.to_datetime(dynamic.index)
            #get the first day (can change to specific dates later)
        date = sorted(dynamic.index)[0].isoformat()[:10] 
        print('date is', date)

        try:    
            first_day = dynamic.loc[date] #only look at the data recorded in specified date
            # print(first_day.head())
    
        except Exception as inst:
            #if the ship doesn't have any data for this date, just set all the values to 0
            print("couldn't get data for ship", mmsi)
            print (inst)

        #resampling data
        try:
            # resample data takes nearest available value
            data_resample = first_day.resample('T').mean().round(2) #not mean
            data_resample = data_resample.dropna()
        
        except Exception as inst:
            #if the ship doesn't have any data for this date, just set all the values to 0
            print("couldn't resample data for ship", mmsi)
            print (inst)
            
        #getting speed sequence
        try: 
            

            #only keeping speed and transforming the column into a list
            data_resample_speed = data_resample[['sog [kn]']] * 100 #can still use data_resample to get other features
            print(data_resample_speed.head())
            # data_resample_speed['sog [kn]'] = data_resample_speed['sog [kn]'] * 100
            # print('after multiplication',data_resample_speed.head())
            sequence=data_resample_speed.aggregate(lambda x: [x.tolist()], axis=0).map(lambda x:x[0])
            
            #adding the sequence and the date to the static dataset
        
            static_data.at[i,"speed_sequence"] =  sequence[0]
            static_data.loc[i,"date"] =  date

        except Exception as inst:
            print("couldn't get speed data on ship", mmsi)
            # print (inst.args)
            print (inst)
            
    return static_data

#%%

df_sequence = get_dynamic_first_day(data)

#%%
#Reshaping

df_ = pd.DataFrame(df_sequence['speed_sequence'].values.tolist()) #splitting the list of speed values into individual columns

df_ = df_.fillna(0) #padding with 0

X_array = df_.to_numpy() #turning it into an array 

#%%

X_reshaped = X_array.reshape(80, 1440, 1)


#%%

from os.path import dirname
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow import keras


#Ecoding the labels
lb = LabelEncoder()
df_sequence['iwrap_cat'] = lb.fit_transform(df_sequence['iwrap_type_from_dataset'])

#picking X and y and splitting
X = X_reshaped


y = data[['iwrap_cat']]
y = y.values.ravel() #somethe model wants this to be an array
unique_class = np.unique(y)
#finishing encoding
y = keras.utils.to_categorical(y)

#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping

#create model

model = Sequential()
#get number of columns in training data
# n_cols = X_train.shape[1]
#add model layers
# model.add(keras.layers.Flatten(input_shape = [1]))
model.add(LSTM(1000))
model.add(Dense(8, activation="softmax"))

#compile model using mse as a measure of model performance
model.compile(loss="categorical_crossentropy", #don't change the loss function
              optimizer="adam",
              metrics=["accuracy"])

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=10)


#%%

history = model.fit(X, y, epochs=3, callbacks=[early_stopping_monitor])



#%%
# #getting the length of the max pad

# s_pad_with_nan = pd.DataFrame(df_sequence['speed_sequence'].values.tolist()).agg(list, 1)

# length_for_padding = len(s_pad_with_nan[0])


# #1440

# #%%
# #padding the sequence

# from tensorflow.keras.preprocessing import sequence

# df_sequence['padded_sequences']=np.nan

# # df_sequence['speed_sequence'] = df_sequence['speed_sequence'] * 100

# df_sequence['padded_sequences'] = sequence.pad_sequences(df_sequence.speed_sequence, maxlen=length_for_padding, padding='post', value=0.0).tolist()

# #%%

# #prepare X, because pandas store lists as strings 

# X = df_sequence[['padded_sequences']]

# my_array = X.to_numpy()

# my_array_ = sequence.pad_sequences(my_array, maxlen=length_for_padding)

# # my_array_ = np.concatenate(X).astype(None)


# #%%




# # saving
# df_sequence.to_csv(path + '/test_speed_sequence_mul100.csv', index = False)


#%%

# for i, row in df_sequence.iterrows():
#     df_sequence['speed_sequence'][i] = df_sequence['speed_sequence'][i] + [0.0]*(length_for_padding - len(df_sequence['speed_sequence'][i])

# # speed_1_longer = speed_1 + [0.0]*(length_for_padding - len(speed_1))

# #%%

# for index, value in speed.items():
#     value = value + [0.0]*(length_for_padding - len(value))

# #%%

# from itertools import chain, repeat, islice

# def pad_infinite(iterable, padding=None):
#    return chain(iterable, repeat(padding))

# def pad(iterable, size, padding=None):
#    return islice(pad_infinite(iterable, padding), size)

# #%%
# padding_test =  df_sequence['speed_sequence'][0]

# padding_result = list(pad(padding_test, length_for_padding, padding = 0))

# #%%

# df_sequence['padded_speed_sequence'] = list(pad(df_sequence['speed_sequence'], length_for_padding, padding = 0))



#    #* (length_for_padding - len(df_sequence['speed_sequence']))

# #%%

# where_are_NaNs = np.isnan(x)
# x[where_are_NaNs] = 0

# #%%

# x2 =x[:5]

# for index, value in x2.items():
#     for number in value :
#         # if number == np.nan:
#         #     number = 0
#         print(number)


#%%

# for i in range(len(data)):
#     print(i)
#     first_day = pd.DataFrame()
#         #getting mmsi name of the ship from the static data 
#     mmsi = data.mmsi.iloc[i]
#     print("getting dynamic data for ship ID", mmsi)
#         #loading the dynamic file for that specific ship
#     dynamic = pd.read_csv(dynamic_data_path +"/{}.csv".format(mmsi), sep="\t", index_col=None, error_bad_lines=True)
#         #using the date time stamp as the index to sort by days
#     dynamic = dynamic.set_index('timestamp')
#         #transforming index into resampable form 
#     dynamic.index = pd.to_datetime(dynamic.index)
#         #get the first day (can change to specific dates later)
#     date = sorted(dynamic.index)[0].isoformat()[:10] 
#     print('date is', date)
    
#     first_day = dynamic.loc[date]
#     print(first_day.head())

#     data_resample = first_day.resample('T').bfill() #not mean
#     data_resample = data_resample.dropna()
    
#     print(data_resample.shape)

#     #only keeping speed and transforming the column into a list
#     data_resample_speed = data_resample[['sog [kn]']]  #can still use data_resample to get other features
#     sequence=data_resample_speed.aggregate(lambda x: [x.tolist()], axis=0).map(lambda x:x[0]).reset_index(drop = True)


#%%
# """
# Unstructred test code - delete after completing the above 
# """
# dynamic = dynamic_data.set_index('timestamp')
# #transforming index into resampable form 
# dynamic.index = pd.to_datetime(dynamic.index)

# # Get one day (first day)            
# date = sorted(dynamic.index)[0].isoformat()[:10] 

# first_day = dynamic.loc[date]

# # resample data takes nearest available value
# data_resample = first_day.resample('T').bfill()
# data_resample = data_resample.dropna()

# #only keeping speed
# data_resample = data_resample[['sog [kn]']]


# #%%
# sequence=data_resample.aggregate(lambda x: [x.tolist()], axis=0).map(lambda x:x[0])
# print('ser_aggCol (collapse each column to a list)',sequence, sep='\n', end='\n\n\n')

# #%%


# df = pd.DataFrame( columns = ['date', 'mmsi', 'speed']) 

# df['speed'] = sequence

