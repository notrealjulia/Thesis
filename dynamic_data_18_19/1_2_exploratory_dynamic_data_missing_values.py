# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:26:11 2020

@author: JULSP

getting infromation for exploratory analysis
"""

import pandas as pd
from os.path import dirname
import numpy as np
from scipy.stats.mstats import mode
import winsound
frequency = 300  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second

path = dirname(__file__)
dynamic_data_path = "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"

raw_static_data = pd.read_csv(path + "/static_ship_data.csv", sep='	', index_col=None, error_bad_lines=False)

#print(raw_static_data['iwrap_type_from_dataset'].unique())

#%%

"""
Getting basic info out our data, how many ships are there of each kind
"""

#splitting dataframe into sub sections based on vessel types
#TODO later split it into trips
cargo = raw_static_data[raw_static_data['iwrap_type_from_dataset'] == 'General cargo ship']
tanker = raw_static_data[raw_static_data['iwrap_type_from_dataset'] == 'Oil products tanker']
fishing = raw_static_data[raw_static_data['iwrap_type_from_dataset'] == 'Fishing ship']
passenger = raw_static_data[raw_static_data['iwrap_type_from_dataset'] == 'Passenger ship']
boat = raw_static_data[raw_static_data['iwrap_type_from_dataset'] == 'Pleasure boat']
support = raw_static_data[raw_static_data['iwrap_type_from_dataset'] == 'Support ship']
ferry = raw_static_data[raw_static_data['iwrap_type_from_dataset'] == 'Fast ferry']
other = raw_static_data[raw_static_data['iwrap_type_from_dataset'] == 'Other ship']

#put all different vessel types into a dictionary
ship_type_dict = {'cargo' : cargo, 'tanker': tanker, 'fishing':fishing, 'passenger':passenger, 'boat':boat, 'support':support, 'ferry':ferry, 'other':other}

#print the amount of vessels for each vessel type
for key, value in ship_type_dict.items():
    print(key, len(value))
    #print(value.mmsi)


#%%

"""
Adding extra columns for collecting dynamic data
"""

def add_columns(ship_data_frame):
    ship_data_frame['mean_speed'] = np.nan
    ship_data_frame['mean_draught'] = np.nan
    ship_data_frame['mean_heading'] = np.nan
    ship_data_frame['mean_course'] = np.nan
    ship_data_frame['mean_ROT'] = np.nan
    # ship_data_frame['mean_easting'] = np.nan
    # ship_data_frame['mean_northing'] = np.nan
    ship_data_frame = ship_data_frame.reset_index(drop =True) #trust me on this one
    return ship_data_frame



#%%

"""
Fail proof function to get dynamic data
"""

def get_dynamic(static_data_ships_local):
    
    for i in range(len(static_data_ships_local)):
        print(i+1)
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
        
        #getting speed
        try:
            static_data_ships_local.loc[i,"mean_speed"] = dynamic.iloc[:,3].mean()
        except Exception as inst:
            print("couldn't get speed data on ship", mmsi)
            print (inst.args)
            print (inst)
            
        #getting draught
        try:
            static_data_ships_local.loc[i,"mean_draught"] = dynamic.iloc[:,4].mean() 
        except Exception as inst:
            print("couldn't get draught data on ship", mmsi)
            print (inst.args)
            print (inst)
            
        #getting heading
        try:
            static_data_ships_local.loc[i,"mean_heading"] = dynamic.iloc[:,5].mean() 
        except Exception as inst:
            print("couldn't get heading data on ship", mmsi)
            print (inst.args)
            print (inst)

        #getting course
        try:
            static_data_ships_local.loc[i,"mean_course"] = dynamic.iloc[:,6].mean() 
        except Exception as inst:
            print("couldn't get course data on ship", mmsi)
            print (inst.args)
            print (inst)
            
        #getting ROT
        try:
            static_data_ships_local.loc[i,"mean_ROT"] = dynamic.iloc[:,7].mean() 
        except Exception as inst:
            print("couldn't get ROT data on ship", mmsi)
            print (inst.args)
            print (inst)
            
        # #getting ROT
        # try:
        #     static_data_ships_local.loc[i,"mean_easting"] = dynamic.iloc[:,8].mean() 
        # except Exception as inst:
        #     print("couldn't get Easting data on ship", mmsi)
        #     print (inst.args)
        #     print (inst)
            
        # #getting ROT
        # try:
        #     static_data_ships_local.loc[i,"mean_northing"] = dynamic.iloc[:,9].mean() 
        # except Exception as inst:
        #     print("couldn't get Northing data on ship", mmsi)
        #     print (inst.args)
        #     print (inst)
            
        
    return static_data_ships_local

    
#%%

"""
calculates % of missing data for each vessel
"""

def detect_missing(df):
    df.replace(0, np.nan, inplace=True)
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})
    return missing_value_df

#%%
    
cargo = add_columns(cargo)
tanker = add_columns(tanker)
fishing = add_columns(fishing)
passenger = add_columns(passenger)
boat = add_columns(boat)
support = add_columns(support)
ferry = add_columns(ferry)
other = add_columns(other)

   
#%% 

#TODO combine with the rest of the get_dynamic vessel types when done
ferry_dynamic = get_dynamic(ferry)
winsound.Beep(frequency, duration) #make some noise when done

#%%
cargo_dynamic = get_dynamic(cargo)
winsound.Beep(frequency, duration)
tanker_dynamic = get_dynamic(tanker)
winsound.Beep(frequency, duration)
fishing_dynamic = get_dynamic(fishing)
winsound.Beep(frequency, duration)
passenger_dynamic = get_dynamic(passenger)
winsound.Beep(frequency, duration)
boat_dynamic = get_dynamic(boat)
winsound.Beep(frequency, duration)
support_dynamic = get_dynamic(support)
winsound.Beep(frequency, duration)
other_dynamic = get_dynamic(other)
winsound.Beep(frequency, duration)
#%%

cargo_missing = detect_missing(cargo_dynamic)

cargo_dynamic.to_csv(path + 'cargo_dynamic.csv', encoding ='utf-8', index=False)
tanker_dynamic.to_csv(path + 'tanker_dynamic.csv', encoding ='utf-8', index=False)

ferry_dynamic.to_csv(path + 'ferry_dynamic.csv', encoding ='utf-8', index=False)

fishing_dynamic.to_csv(path + 'fishing_dynamic.csv', encoding ='utf-8', index=False)

passenger_dynamic.to_csv(path + 'passenger_dynamic.csv', encoding ='utf-8', index=False)

boat_dynamic.to_csv(path + 'boat_dynamic.csv', encoding ='utf-8', index=False)
support_dynamic.to_csv(path + 'support_dynamic.csv', encoding ='utf-8', index=False)

other_dynamic.to_csv(path + 'other_dynamic.csv', encoding ='utf-8', index=False)

#%%
dynamic_test_file = pd.read_csv(dynamic_data_path + "/211727510.csv", sep='	', index_col=None, error_bad_lines=True)
#%%

tanker_missing = detect_missing(tanker_dynamic)

ferry_missing = detect_missing(ferry_dynamic)

fishing_missing = detect_missing(fishing_dynamic)

passenger_missing = detect_missing(passenger_dynamic)
boat_missing = detect_missing(boat_dynamic)
support_missing = detect_missing(support_dynamic)
other_missing = detect_missing(other_dynamic)


#%%
"""
attempt to put all of the above into a loop
"""   
for key, value in ship_type_dict.items():
    
    value = add_columns(value)
    #print(value.mmsi)
    
#%%
""" no!"""    
ship_type_dict['ferry'] = get_dynamic(ship_type_dict['ferry'])
    
#%%
import sys
print(sys.path) 

