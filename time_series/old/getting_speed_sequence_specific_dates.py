# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:57:59 2020

@author: JULSP

GETTING SPEED SEQUENCES FROM 1 MONTH OF DYNAMIC DATA FILES
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
static_data = pd.read_csv(path + "/Trips_and_VesselStatus_Static.csv",  sep='	', index_col=None, error_bad_lines=False) 

#%%
"""
Selecting mmsi subset
active ships with over a 5000 signals
of a certain size 
"""
active_subset = static_data[(static_data['status']== 'active') & (static_data['signals']>= 5000)]
active_subset = active_subset[['mmsi', 'iwrap_type_from_dataset','org_type_info_from_data', 'length_from_data_set', 'width', 'trips', 'signals']]
active_subset_cleaned = active_subset[(active_subset['length_from_data_set'] > 2) & (active_subset['width'] > 2)]
active_subset_cleaned = active_subset_cleaned[active_subset_cleaned['length_from_data_set'] <= 400]
data = active_subset_cleaned.reset_index(drop =True) #reset index

#%%
data['speed_sequence'] = np.nan
data['date'] = np.nan

data['speed_sequence'] = data['speed_sequence'].astype('object')


# data = data[:4] #to test the function

#%%

#go through the static data fill out speed and dates

def get_dynamic_first_day(static_data, date):
    
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
        # date = sorted(dynamic.index)[0].isoformat()[:10] 
        # print('\nDate is ', date)

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
            data_resample_speed = data_resample[['sog [kn]']]  #can still use data_resample to get other features
            # print(data_resample_speed.head())
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

"""
Getting data for specific dates
Going by monthly basis
"""    

first_day = '2019-06-11'
last_day = '2019-06-21'

daterange = pd.date_range(first_day, last_day)
df_sequences_2 = pd.DataFrame()

for single_date in daterange:
    day_format = single_date.strftime("%Y-%m-%d")
    print (day_format)
    try:
        day_dynamic_data = get_dynamic_first_day(data, day_format)
        df_sequences_2 = df_sequences_2.append(day_dynamic_data, sort = False)
    except:
        print('no data for this date', day_format)
        
        #%%
#this is going to be X_test
df_seq_1 = df_sequences.dropna()
df_seq_1 = df_seq_1.reset_index(drop =True)

#%%
#this is going to be the X_train
df_seq_2 = df_sequences_2.dropna()
df_seq_2 = df_seq_2.reset_index(drop = True)

#%%

df_train = pd.DataFrame(df_seq_2['speed_sequence'].values.tolist()) #splitting the list of speed values into individual columns
df_test = pd.DataFrame(df_seq_1['speed_sequence'].values.tolist())

df_train = df_train.fillna(0) #padding with 0
df_test = df_test.fillna(0)

#%%
# Remove all columns after 500 
df_train_min = df_train.drop(df_train.iloc[:, 500:], axis = 1) 
df_test_min = df_test.drop(df_test.iloc[:, 500:], axis = 1) 

#%%
X_train_array = df_train_min.to_numpy() #turning it into an array 
X_test_array = df_test_min.to_numpy() #turning it into an array 

print(len(X_train_array))

X_train = X_train_array.reshape(len(X_train_array), 500, 1)
X_test = X_test_array.reshape(len(X_test_array), 500, 1)

#%%
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping

lb = LabelEncoder()
#Ecoding the labels for training

df_seq_2['iwrap_cat'] = lb.fit_transform(df_seq_2['iwrap_type_from_dataset'])
y_train = df_seq_2[['iwrap_cat']]
y_train = y_train.values.ravel() #somethe model wants this to be an array
unique_class = np.unique(y_train)
y_train = keras.utils.to_categorical(y_train)

#Encoding labels for testing

df_seq_1['iwrap_cat'] = lb.fit_transform(df_seq_1['iwrap_type_from_dataset'])
y_test = df_seq_1[['iwrap_cat']]
y_test = y_test.values.ravel() #somethe model wants this to be an array
unique_class = np.unique(y_test)
y_test = keras.utils.to_categorical(y_test)

#%%

model = Sequential()

model.add(LSTM(100))
model.add(Dense(8, activation="softmax"))

#compile model using mse as a measure of model performance
model.compile(loss="categorical_crossentropy", #don't change the loss function
              optimizer="adam",
              metrics=["accuracy"])

#set early stopping monitor so the model stops training when it won't improve anymore
# early_stopping_monitor = EarlyStopping(patience=10)


#%%

model.fit(X_train, y_train, epochs=10)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#%%
#TODO save the data df_train, df_test, and df_seq_1 and df_seq_2

df_train.to_csv(path +'/df_train.csv', index = False)
df_test.to_csv(path +'/df_test.csv', index = False)
df_seq_1.to_csv(path +'/df_seq_test.csv', index = False)
df_seq_2.to_csv(path +'/df_seq_train.csv', index = False)




