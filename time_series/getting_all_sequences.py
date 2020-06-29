# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:07:35 2020

@author: JULSP

getting sequential data from dynamic files
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
dynamic_data_path = "D:/Thesis_data/export"

#DF of active vessels, this is the DF that we are going to expand and enrich with dynamic data 
static_data = pd.read_csv(path + "/Trips_and_VesselStatus_Static.csv",  sep='	', index_col=None, error_bad_lines=False) 

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
data['cog_sequence'] = np.nan
data['lat_sequence'] = np.nan
data['lon_sequence'] = np.nan
data['date'] = np.nan

data['speed_sequence'] = data['speed_sequence'].astype('object')
data['cog_sequence'] = data['cog_sequence'].astype('object')
data['lat_sequence'] = data['lat_sequence'].astype('object')
data['lon_sequence'] = data['lon_sequence'].astype('object')

# data = data[:4] #to test the function

#%%

#go through the static data fill out speed and dates

def get_dynamic_first_day(static_data, date):
    
    for i in range(len(static_data)):
        print(i)
        first_day = pd.DataFrame()
            #getting mmsi name of the ship from the static data 
        mmsi = static_data.mmsi.iloc[i]
        # print("getting dynamic data for ship ID", mmsi)
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
            pass
            # if the ship doesn't have any data for this date, just set all the values to 0
            # print("couldn't get data for ship", mmsi)
            # print (inst)

        #resampling data
        try:
            # resample data takes nearest available value
            data_resample = first_day.resample('T').mean().round(2) #not mean
            data_resample = data_resample.dropna()
        
        except Exception as inst:
            pass
            # if the ship doesn't have any data for this date, just set all the values to 0
            # print("couldn't resample data for ship", mmsi)
            # print (inst)
            
            
        #GETTING SPEED SEQUENCES
        try: 
            #only keeping speed and transforming the column into a list
            data_resample_speed = data_resample[['sog [kn]']]  #can still use data_resample to get other features
            sequence_speed=data_resample_speed.aggregate(lambda x: [x.tolist()], axis=0).map(lambda x:x[0])
            
            #adding the sequence and the date to the static dataset
            static_data.at[i,"speed_sequence"] =  sequence_speed[0]
            static_data.loc[i,"date"] =  date

        except Exception as inst:
            pass
            # print("couldn't get speed data on ship", mmsi)
            # print (inst.args)
            # print (inst)
            
            
        #GETTING COG SEQUENCES
        try: 
            #only keeping speed and transforming the column into a list
            data_resample_cog = data_resample[['cog [deg]']]  #can still use data_resample to get other features
            sequence_cog = data_resample_cog.aggregate(lambda x: [x.tolist()], axis=0).map(lambda x:x[0])
            
            #adding the sequence and the date to the static dataset
            static_data.at[i,"cog_sequence"] =  sequence_cog[0]

        except Exception as inst:
            pass
            # print("couldn't get cog data on ship", mmsi)
            # print (inst.args)
            # print (inst)
            
        #GETTING COG SEQUENCES
        try: 
            #only keeping speed and transforming the column into a list
            data_resample_lat = data_resample[['lat [deg]']]  #can still use data_resample to get other features
            sequence_lat = data_resample_lat.aggregate(lambda x: [x.tolist()], axis=0).map(lambda x:x[0])
            
            #adding the sequence and the date to the static dataset
            static_data.at[i,"lat_sequence"] =  sequence_lat[0]
            
            #only keeping speed and transforming the column into a list
            data_resample_lon = data_resample[['lon [deg]']]  #can still use data_resample to get other features
            sequence_lon = data_resample_lon.aggregate(lambda x: [x.tolist()], axis=0).map(lambda x:x[0])
            
            #adding the sequence and the date to the static dataset
            static_data.at[i,"lon_sequence"] =  sequence_lon[0]
            print(i, ' yes')

        except Exception as inst:
            pass
            # print("couldn't get lat/lon data on ship", mmsi)
            # print (inst.args)
            # print (inst)
            
    return static_data

#%%

"""
Getting data for specific dates
Going by monthly basis
"""    

def get_dynamic_data_for_dates(first_day, last_day): #'2019-06-11'

    # first_day = '2019-07-01'
    # last_day = '2019-07-20'

    daterange = pd.date_range(first_day, last_day)
    df_sequences = pd.DataFrame()

    for single_date in daterange:
        day_format = single_date.strftime("%Y-%m-%d")
        print (day_format)
        try:
            day_dynamic_data = get_dynamic_first_day(data, day_format)
            df_sequences = df_sequences.append(day_dynamic_data, sort = False)
        except:
            print('no data for this date', day_format)
            
    return df_sequences
        
#%%
        
#get the sequence
 
big_data  = get_dynamic_data_for_dates(first_day = '2019-07-01', last_day = '2019-07-30')
           
            
#%%

#save the sequence            

#this is going to be X_test
df_seq = big_data.dropna() #get rid of empty rows
df_seq = df_seq.reset_index(drop =True) #reset the index - we will need this later to get the y values            
            
#%%
        
#this is going to be X_test
# df_seq_1 = df_sequences.dropna() #get rid of empty rows
# df_seq_1 = df_seq_1.reset_index(drop =True) #reset the index - we will need this later to get the y values

df_speed = pd.DataFrame(df_seq['speed_sequence'].values.tolist()) #spread the sequences into a DF
# df_test_speed = df_test_speed.fillna(0) #padd with 0

df_cog = pd.DataFrame(df_seq['cog_sequence'].values.tolist())
# df_test_cog = df_test_cog.fillna(0)

df_lat = pd.DataFrame(df_seq['lat_sequence'].values.tolist())
# df_test_lat = df_test_lat.fillna(0)

df_lon = pd.DataFrame(df_seq['lon_sequence'].values.tolist())
# df_test_lon = df_test_lon.fillna(0)

# df_test_speed_min = df_test_speed.drop(df_test_speed.iloc[:, 500:], axis = 1) #cut off after 500 values, LSTM can't handle 1440
# df_test_cog_min = df_test_cog.drop(df_test_cog.iloc[:, 500:], axis = 1)
# df_test_lat_min = df_test_lat.drop(df_test_lat.iloc[:, 500:], axis = 1)
# df_test_lon_min = df_test_lon.drop(df_test_lon.iloc[:, 500:], axis = 1)
 
seq_path = "D:/Thesis_data/data"

df_speed.to_csv(seq_path +'/all/df_speed_july.csv', index = False)
df_cog.to_csv(seq_path +'/all/df_cog_july.csv', index = False)
df_lat.to_csv(seq_path +'/all/df_lat_july.csv', index = False)
df_lon.to_csv(seq_path +'/all/df_lon_july.csv', index = False)


df_seq.to_csv(seq_path +'/all/df_Y_july.csv', index = False)



#%%
#this is going to be X_test
X_test_array_speed = df_test_speed_min.to_numpy() #turning it into an array 
X_test_array_cog = df_test_cog_min.to_numpy() #turning it into an array 
X_test_array_lat = df_test_lat_min.to_numpy() #turning it into an array 
X_test_array_lon = df_test_lon_min.to_numpy() #turning it into an array 


X_test_array = np.concatenate((X_test_array_speed, X_test_array_cog, X_test_array_lat, X_test_array_lon), axis=1)

#%%
# length_test  = len(X_test_array_cog)

# X_test = X_test_array.reshape(length_test, 500, 4, order = 'F')

# df_test.to_csv(path +'/df_test_full.csv', index = False)
# df_seq_1.to_csv(path +'/df_seq_test_full.csv', index = False)

df_test_speed_min.to_csv(seq_path +'/df_test_speed_min.csv', index = False)
df_test_cog_min.to_csv(seq_path +'/df_test_cog_min.csv', index = False)
df_test_lat_min.to_csv(seq_path +'/df_test_lat_min.csv', index = False)
df_test_lon_min.to_csv(seq_path +'/df_test_lon_min.csv', index = False)


#%%
#this is going to be X_train
df_seq_2 = df_sequences_2.dropna()
df_seq_2 = df_seq_2.reset_index(drop =True)

df_seq_2.to_csv(seq_path +'/df_train_y.csv', index = False)

df_train_speed = pd.DataFrame(df_seq_2['speed_sequence'].values.tolist())
df_train_speed = df_train_speed.fillna(0)

df_train_cog = pd.DataFrame(df_seq_2['cog_sequence'].values.tolist())
df_train_cog = df_train_cog.fillna(0)

df_train_lat = pd.DataFrame(df_seq_2['lat_sequence'].values.tolist())
df_train_lat = df_train_lat.fillna(0)

df_train_lon = pd.DataFrame(df_seq_2['lon_sequence'].values.tolist())
df_train_lon = df_train_lon.fillna(0)


df_train_speed_min = df_train_speed.drop(df_train_speed.iloc[:, 500:], axis = 1) #cut off after 500 values, LSTM can't handle 1440
df_train_cog_min = df_train_cog.drop(df_train_cog.iloc[:, 500:], axis = 1)
df_train_lat_min = df_train_lat.drop(df_train_lat.iloc[:, 500:], axis = 1)
df_train_lon_min = df_train_lon.drop(df_train_lon.iloc[:, 500:], axis = 1)


df_train_speed_min.to_csv(seq_path +'/df_train_speed_min.csv', index = False)
df_train_cog_min.to_csv(seq_path +'/df_train_cog_min.csv', index = False)
df_train_lat_min.to_csv(seq_path +'/df_train_lat_min.csv', index = False)
df_train_lon_min.to_csv(seq_path +'/df_train_lon_min.csv', index = False)
#%%
X_train_array_speed = df_train_speed_min.to_numpy() #turning it into an array 
X_train_array_cog = df_train_cog_min.to_numpy() #turning it into an array 
X_train_array_lat = df_train_lat_min.to_numpy() #turning it into an array 
X_train_array_lon = df_train_lon_min.to_numpy() #turning it into an array 


X_train_array = np.concatenate((X_train_array_speed, X_train_array_cog, X_train_array_lat, X_train_array_lon), axis=1)

length_train = len(X_train_array_cog)

X_train = X_train_array.reshape(length_train, 500, 4, order = 'F')