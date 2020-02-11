# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:09:14 2020

@author: JULSP
"""

import pandas as pd
from os.path import dirname
import numpy as np
from scipy.stats.mstats import mode

path = dirname(__file__)
dynamic_data_path = "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"

static_data_full = pd.read_csv(path + "/static_ship_data.csv", sep='	', index_col=None, error_bad_lines=False)
static_data = static_data_full[static_data_full.length_from_data_set != -1] #drop -1 in width

print(round((len(static_data)/len(static_data_full)*100), 1), 'percent of ships have information about vessel length') #85.4

#%%

#print(static_data['iwrap_type_from_dataset'].unique())

cargo = static_data[static_data['iwrap_type_from_dataset'] == 'General cargo ship']
tanker = static_data[static_data['iwrap_type_from_dataset'] == 'Oil products tanker']
fishing = static_data[static_data['iwrap_type_from_dataset'] == 'Fishing ship']
passenger = static_data[static_data['iwrap_type_from_dataset'] == 'Passenger ship']
boat = static_data[static_data['iwrap_type_from_dataset'] == 'Pleasure boat']
support = static_data[static_data['iwrap_type_from_dataset'] == 'Support ship']
ferry = static_data[static_data['iwrap_type_from_dataset'] == 'Fast ferry']

"""
print('tanker is', round(tanker['length_from_data_set'].mean(), 1)) # 178.7m
print('cargo is', round(cargo['length_from_data_set'].mean(), 1)) # 157.1m
print('passenger is', round(passenger['length_from_data_set'].mean(), 1)) # 139.1m
print('support is', round(support['length_from_data_set'].mean(), 1)) #39.3m
print('ferry is', round(ferry['length_from_data_set'].mean(), 1)) #22.6m
print('fishing is', round(fishing['length_from_data_set'].mean(), 1)) #17.9m
print('boat is', round(boat['length_from_data_set'].mean(), 1)) #12.8m
"""
static_data_ships = pd.concat([tanker, cargo, passenger, support, ferry, fishing, boat])
print(round((len(static_data_ships)/len(static_data_full)*100), 1), 'percent of ships have information about vessel type') #80.4

static_data_ships = static_data_ships.reset_index(drop =True) #reset index
static_data_ships = static_data_ships[static_data_ships.width != -1] #drop -1 in width

static_data_ships['Speed_mean'] = np.nan
static_data_ships['Speed_std'] = np.nan
static_data_ships['Speed_median'] = np.nan
static_data_ships['Speed_mode'] = np.nan
static_data_ships['ROT_max'] = np.nan
static_data_ships['ROT_min'] = np.nan
 #%%

"""
getting dynamic data for all known ship types
"""

#static_data_ships = static_data_ships[:15] #test
def get_dynamic(static_data_ships):
    
    for i in range(len(static_data_ships)):
        try:
            print("getting dynamic data for ship number", i+1)
            #getting mmsi name of the ship from the static data 
            mmsi = static_data.mmsi[i]
            #loading the dynamic file for that specific ship
            dynamic = pd.read_csv(dynamic_data_path +"/{}.csv".format(mmsi), sep="\t", index_col=None, error_bad_lines=True)
        except:
            print("couldn't open file for ship", i+1)
            pass
        try:
            #calculating mean, max, std speed for that specific ship and saving it in the static DF
            dynamic = dynamic[(dynamic[['sog [kn]']] != 0).all(axis=1)] #dropping all 0 values
            static_data_ships["Speed_mean"].loc[i] = dynamic.iloc[:,3].mean()
            static_data_ships["Speed_std"].loc[i] = dynamic.iloc[:,3].std()
            static_data_ships["Speed_median"].loc[i] = dynamic.iloc[:,3].median()
            dynamic['rounded_speed'] = np.nan
            dynamic['rounded_speed'] = dynamic.iloc[:,3].round(0) #rounding up to integers
        #print(dynamic['rounded_speed'][0])
            static_data_ships['Speed_mode'].loc[i] = dynamic.mode()['rounded_speed'][0]
        except Exception as inst:
            print("couldn't get speed data on ship", mmsi)
            print (inst.args)
            print (inst)
        
        #calculating max and min ROT values
        try:
            dynamic = dynamic[(dynamic[['rot [deg/min]']] != 0).all(axis=1)] #dropping all 0 values
            static_data_ships["ROT_max"].loc[i] = dynamic.iloc[:,7].max()
            static_data_ships["ROT_min"].loc[i] = dynamic.iloc[:,7].min()
        except:
            print("couldn't get ROT values for ship", mmsi)
        
            
    return static_data_ships
        
#%%       
import winsound
frequency = 1000  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
   
static_data_ships_1 = static_data_ships[9:11]

dynamic_1 = get_dynamic(static_data_ships_1)
winsound.Beep(frequency, duration)

#%%

#data = static_data_ships.dropna(subset = ['Speed_mean'])       
dynamic_1_vessel =pd.read_csv(dynamic_data_path + '/245405000.csv', sep="\t", index_col=None, error_bad_lines=False)

atd_1 = dynamic_1_vessel.iloc[:,3].std()