# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:10:55 2020

@author: JULSP

Script for getting dynamic data for active vessels 

"""
import pandas as pd
from os.path import dirname
import numpy as np


#%%
"""
Loading data
"""
path = dirname(__file__)
#dynamic_data_path = "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"
dynamic_data_path = "D:/Thesis_data/export"
data = pd.read_csv(path + "/Trips_and_VesselStatus_Static.csv",  sep='	', index_col=None, error_bad_lines=False) 

#%%
"""
Selecting mmsi subset
active ships with over a 5000 signals
of a certain size 
"""
active_subset = data[(data['status']== 'active') & (data['signals']>= 5000)]
data_static = active_subset[['mmsi', 'iwrap_type_from_dataset', 'length_from_data_set', 'width', 'trips', 'signals']]
data_static_cleaned = data_static[(data_static['length_from_data_set'] > 2) & (data_static['width'] > 2)]
data_static_cleaned = data_static_cleaned[data_static_cleaned['length_from_data_set'] <= 400]
data_static_cleaned = data_static_cleaned.reset_index(drop =True) #reset index

#%%
"""
Adding columns to store dynamic variables in the static dataframe
"""
test_data = data_static_cleaned

test_data['Speed_mean'] = np.nan
test_data['Speed_median'] = np.nan
test_data['Speed_min'] = np.nan
test_data['Speed_max'] = np.nan
test_data['Speed_std'] = np.nan
test_data['ROT_mean'] = np.nan
test_data['ROT_median'] = np.nan
test_data['ROT_min'] = np.nan
test_data['ROT_max'] = np.nan
test_data['ROT_std'] = np.nan
test_data['number_of_signals'] = np.nan
test_data['date'] = np.nan

#%%
print(len(test_data))

#%%

def get_dynamic_first_day(static_data_ships_local, date):
    
    for i in range(len(static_data_ships_local)):
        print(i)
        first_day = pd.DataFrame()
            #getting mmsi name of the ship from the static data 
        mmsi = static_data_ships_local.mmsi.iloc[i]
        print("getting dynamic data for ship ID", mmsi)
            #loading the dynamic file for that specific ship
        dynamic = pd.read_csv(dynamic_data_path +"/{}.csv".format(mmsi), sep="\t", index_col=None, error_bad_lines=True)
            #using the date time stamp as the index to sort by days
        dynamic = dynamic.set_index('timestamp')
            #transforming index into resampable form 
        dynamic.index = pd.to_datetime(dynamic.index)
            
            ### comment the line below and uncomment the following line to get data from first day
            #first_date = date
            #first_date = sorted(dynamic.index)[0].isoformat()[:10] #since it is a string pick first XXXX-XX-XX ten digits, to not look at time
            # print(first_date)
        try:    
            first_day = dynamic.loc[date] #only look at the data recorded in specified date
    
        except:
            #if the ship doesn't have any data for this date, just set all the values to 0
            print("couldn't get data for ship", mmsi)
            static_data_ships_local.loc[i,"Speed_mean"] = 0
            static_data_ships_local.loc[i,"Speed_median"] = 0
            static_data_ships_local.loc[i,"Speed_min"] = 0
            static_data_ships_local.loc[i,"Speed_max"] = 0
            static_data_ships_local.loc[i,"Speed_std"] = 0
            static_data_ships_local.loc[i,"number_of_signals"] = 0
            static_data_ships_local.loc[i,"date"] = date
            static_data_ships_local.loc[i,"ROT_mean"] = 0
            static_data_ships_local.loc[i,"ROT_median"] = 0
            static_data_ships_local.loc[i,"ROT_min"] = 0
            static_data_ships_local.loc[i,"ROT_max"] = 0
            static_data_ships_local.loc[i,"ROT_std"] = 0
            
        #getting speed parameters
        try: 
            first_day = first_day[first_day['sog [kn]'] != 0] #dropping zero, this fucks up our values
            static_data_ships_local.loc[i,"Speed_mean"] = first_day.iloc[:,2].mean()
            static_data_ships_local.loc[i,"Speed_median"] = first_day.iloc[:,2].median()
            static_data_ships_local.loc[i,"Speed_min"] = first_day.iloc[:,2].min()
            static_data_ships_local.loc[i,"Speed_max"] = first_day.iloc[:,2].max()
            static_data_ships_local.loc[i,"Speed_std"] = first_day.iloc[:,2].std()
            static_data_ships_local.loc[i,"number_of_signals"] = len(first_day)
            static_data_ships_local.loc[i,"date"] = date

        except Exception as inst:
            print("couldn't get speed data on ship", mmsi)
            print (inst.args)
            print (inst)

        
        #calculating Rate of Turn from COG
        try:
            data_resample = first_day.resample('T').mean()
            data_resample = data_resample.dropna()
            data_resample['Cog_diff'] = data_resample['cog [deg]'].diff()
            data_resample = data_resample[data_resample['Cog_diff'] != 0] #dropping zero, this fucks up our values                     
            data_resample['Cog_diff_abs'] = data_resample['Cog_diff'].abs() #it can either go to negative number so for some calculations we want absolute values

            #series to locate time holes
            time_difference = data_resample.index.to_series().diff()
            #add differences in timestamps to the original df
            data_resample_with_time_difference = pd.concat([data_resample, time_difference], axis=1, sort=False)
            #drop all lines that do not deal with 1 minute intervals
            data_resample_with_time_difference_1min = data_resample_with_time_difference[data_resample_with_time_difference.timestamp == '00:01:00']
            # print("it's a miracle this resampling works")
            
            static_data_ships_local.loc[i,"ROT_mean"] = data_resample_with_time_difference_1min.Cog_diff_abs.mean()
            static_data_ships_local.loc[i,"ROT_median"] = data_resample_with_time_difference_1min.Cog_diff_abs.median()
            static_data_ships_local.loc[i,"ROT_min"] = data_resample_with_time_difference_1min.Cog_diff_abs.min()
            static_data_ships_local.loc[i,"ROT_max"] = data_resample_with_time_difference_1min.Cog_diff_abs.max()
            static_data_ships_local.loc[i,"ROT_std"] = data_resample_with_time_difference_1min.Cog_diff.std() #the only one where we do not use absolute values because we want to capture the range

        except:
            print("couldn't get ROT values for ship", mmsi)
            
    return static_data_ships_local


#%%
# test_data = test_data[:30]
# september_01_manual = get_dynamic_first_day(test_data, '2019-09-01')

#%%
    
"""
Getting data
"""    

first_day = '2019-08-01'
last_day = '2019-08-31'

daterange = pd.date_range(first_day, last_day)


# test_data = test_data[:30]


for single_date in daterange:
    day_format = single_date.strftime("%Y-%m-%d")
    print (day_format)
    try:
        day_dynamic_data = get_dynamic_first_day(test_data, day_format)
        day_dynamic_data.to_csv('ships_AUG_{}.csv'.format(day_format), index=False)
    except:
        print('no data for this date', day_format)
        

#%%
"""
Putting all the dates together
"""
# dynamic_data_01 = pd.read_csv("C:/Users/julsp/Documents/GitHub/Thesis/dynamic_data_18_19/organized_code/Dynamic_data/ships_SEPT_v22019-09-01.csv", error_bad_lines=False) 
# dynamic_data_02 = pd.read_csv("C:/Users/julsp/Documents/GitHub/Thesis/dynamic_data_18_19/organized_code/Dynamic_data/ships_SEPT_v22019-09-02.csv", error_bad_lines=False) 

# #dynamic_data_01 = dynamic_data_01.dropna()
# #dynamic_data_02 = dynamic_data_02.dropna()

# frames = [dynamic_data_01, dynamic_data_02]
# dynamic_data_all = pd.concat(frames)
# #dynamic_data_all.trips.fillna(0, inplace = True)
# dynamic_data_all = dynamic_data_all.dropna()

# dynamic_data_all = dynamic_data_all.reset_index(drop =True) #reset index


#%%

first_day = '2019-10-01'
last_day = '2019-10-31'

# TODO get rid of rows where all dynamic values are 0

daterange = pd.date_range(first_day, last_day)

df_for_appeding = pd.DataFrame()

for single_date in daterange:
    open_df = pd.read_csv("C:/Users/julsp/Documents/GitHub/Thesis/dynamic_data_18_19/organized_code/Dynamic_data/ships_OCT_{}.csv".format(single_date.strftime("%Y-%m-%d")), error_bad_lines=False) 
    df_for_appeding = df_for_appeding.append(open_df, sort = False)

df_for_appeding.trips.fillna(0, inplace = True)
result_df = df_for_appeding.dropna()
result_df = result_df[(result_df['Speed_mean'] != 0) & (result_df['ROT_mean'] != 0)]
#result_df = df_for_appeding.drop_duplicates(subset=['Speed_mean', 'number_of_signals'], keep='first')
# result_df.trips.fillna(0, inplace = True)
# result_df = result_df.dropna()
# result_df = result_df.reset_index(drop =True) #reset index

result_df.to_csv('dynamic_data_oct.csv', index=False)

# df_all.to_csv('Dynamic_data_Sept.csv', index=False)
