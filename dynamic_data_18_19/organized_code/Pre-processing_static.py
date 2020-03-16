# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:10:55 2020

@author: JULSP
"""
import pandas as pd
from os.path import dirname
import numpy as np


#%%
"""
Loading data
"""
path = dirname(__file__)
data = pd.read_csv(path + "/Trips_and_VesselStatus_Static.csv",  sep='	', index_col=None, error_bad_lines=False) 

#%%
"""
Selecting mmsi subset
active ships with over a 1000 signals
"""
active_subset = data[(data['status']== 'active') & (data['signals']>= 1000)]
#getting rid of unecessary rows
data_static = active_subset[['mmsi', 'iwrap_type_from_dataset', 'length_from_data_set', 'width', 'trips', 'signals']]

data_static_cleaned = data_static[(data_static['length_from_data_set'] > 2) & (data_static['width'] > 2)]
data_static_cleaned = data_static_cleaned[data_static_cleaned['length_from_data_set'] <= 400]
#data_static_cleaned = data_static_cleaned[data_static_cleaned['iwrap_type_from_dataset'] != "Other ship"]

data_static_cleaned = data_static_cleaned.reset_index(drop =True) #reset index


#%%

# """
# Visualising ship type distribution 
# """

# trips_for_types = data_static_cleaned.groupby('iwrap_type_from_dataset')['trips'].sum()
# ax1 = trips_for_types.plot.pie(y = 'trips', label = 'iwrap_type_from_dataset', autopct='%1.0f%%', pctdistance=0.8, labeldistance=1.2)
# #%%

# trips_for_types = data_static_cleaned.groupby('iwrap_type_from_dataset')['trips'].sum().reset_index() 
# sns.barplot(x = 'iwrap_type_from_dataset', y ='trips', data = trips_for_types)

#%%
"""Getting dynamic data for 1 day"""
dynamic_data_path = "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"


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

list_of_first_dates = []


#%%

def get_dynamic_first_day(static_data_ships_local, day_number):
    
    for i in range(len(static_data_ships_local)):
        print(i)
        try:
            #getting mmsi name of the ship from the static data 
            mmsi = static_data_ships_local.mmsi.iloc[i]
            print("getting dynamic data for ship ID", mmsi)
            #print('mmsi ',mmsi)
            #loading the dynamic file for that specific ship
            dynamic = pd.read_csv(dynamic_data_path +"/{}.csv".format(mmsi), sep="\t", index_col=None, error_bad_lines=True)
        except:
            print("couldn't open file for ship", mmsi)
            pass
        try:
            #using the date time stamp as the index to sort by days
            dynamic = dynamic.set_index('timestamp')
            #transforming index into resampable form 
            dynamic.index = pd.to_datetime(dynamic.index)
            
            first_date = sorted(dynamic.index)[day_number].isoformat()[:10] #since it is a string pick first XXXX-XX-XX ten digits, to not look at time
            # print(first_date)
            list_of_first_dates.append(first_date)
            
            first_day = dynamic.loc[first_date] #only look at the data recorded in the first day of sailing
            first_day = first_day[first_day['sog [kn]'] != 0] #dropping zero, this fucks up our values
        except:
            print("couldn't get data from the first day for ship", mmsi)
            pass
        try:
            
            static_data_ships_local.loc[i,"Speed_mean"] = first_day.iloc[:,2].mean()
            static_data_ships_local.loc[i,"Speed_median"] = first_day.iloc[:,2].median()
            static_data_ships_local.loc[i,"Speed_min"] = first_day.iloc[:,2].min()
            static_data_ships_local.loc[i,"Speed_max"] = first_day.iloc[:,2].max()
            static_data_ships_local.loc[i,"Speed_std"] = first_day.iloc[:,2].std()

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
"""
Getting more data for the rarer boats
"""    
rare_boats = [ 'Passenger ship','Fishing ship', 'Support ship' ,'Fast ferry']
data_set_few =test_data[test_data['iwrap_type_from_dataset'].isin(rare_boats)]  
data_set_few = data_set_few.reset_index(drop =True) #reset index
#%%    
day_1_few_vessels = get_dynamic_first_day(data_set_few, day_number = 1)  
day_1_few_vessels.to_csv("rare_ships_day_1.csv", index = False)

#%%

day_2_few_vessels = get_dynamic_first_day(data_set_few, day_number = 2)  
day_2_few_vessels.to_csv("rare_ships_day_2.csv", index = False)

day_3_few_vessels = get_dynamic_first_day(data_set_few, day_number = 3)  
day_3_few_vessels.to_csv("rare_ships_day_3.csv", index = False)

day_4_few_vessels = get_dynamic_first_day(data_set_few, day_number = 4)  
day_4_few_vessels.to_csv("rare_ships_day_4.csv", index = False)



#%%


test_speed_only = get_dynamic_first_day(test_data)
test_speed_only.to_csv("DYNAMIC_DAYS_test.csv", index = False) #saves to C:\Users\julsp - probably something with spyder

#%%
test_speed_only.to_csv("DYNAMIC_DAYS_1.csv", index = False) #saves to C:\Users\julsp - probably something with spyder