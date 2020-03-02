# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:49:43 2020

@author: KORAL
"""


"""

Goes trough MMSI numbers (all or separated by type) from static file and opens corresponding dynamic file.
Then goes trough all the speeds in the dynamic file, when speed is smaller than 0.1, timestamps are added to the list.
When vessel starts moving (speed > 0.1), previous idle time is recorded and added to the list of stops.
Stops that pass the threshold (e.g. 15 min or an hour) are counted as trips and this number is added to the static file under 'trips' column

"""

#%%
# ALL vessels 

from datetime import datetime
import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt
import numpy as np
import sys
import progressbar

print(sys.path)
path = dirname(__file__)
dynamic_path= "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"
static = pd.read_csv(r"C:\Users\KORAL\OneDrive - Ramboll\Documents\GitHub\Thesis\Thesis\Static Data 2018-2019/static_ship_data.csv", sep='	', index_col=None, error_bad_lines=False)
dynamic = pd.read_csv(dynamic_path + "/211727510.csv", sep='	', index_col=None, error_bad_lines=True)
dictionary = static.groupby('iwrap_type_from_dataset')['mmsi'].apply(lambda g: g.values.tolist()).to_dict()

#%%
static['trips'] = np.nan
memoryErrors = []
parserErrors = []
for mmsi in progressbar.progressbar(dictionary.get('Passenger ship')): 
    
    try:
        dynamic = pd.read_csv(dynamic_path +"/{}.csv".format(str(mmsi)), sep='	', index_col=None, error_bad_lines=False)    
        idleTimes = []
        stops = []
        trips = 0
        
                   
        for i, speed in enumerate(dynamic['sog [kn]']):
            try:
                if speed < 0.1:
                    stops.append(dynamic['timestamp'].iloc[i])
                        
                elif speed > 0.1:
                    firstStop = datetime.strptime(stops[0], "%Y-%m-%dT%H:%M:%S")
                    lastStop = datetime.strptime(stops[-1], "%Y-%m-%dT%H:%M:%S")
                    idleTime = (lastStop - firstStop)
                    idleTimes.append(idleTime)
                    stops = []
            except:
                pass
        
        for idle in idleTimes:
            if idle.total_seconds() > 900:  # stop time defined in seconds, currently = 15 minutes 
                
                trips += 1

        static['trips'][(static['mmsi'] == mmsi)] = trips  
    except MemoryError as error:
        memoryErrors.append((mmsi,error))
    except pd.errors.ParserError as er:
        parserErrors.append((mmsi,er))
#%%

### TESTING 
#print(static.loc[static['mmsi'] == 265874000])
#%%
#dynamic = pd.read_csv(dynamic_path + "/265874000.csv", sep='	', index_col=None, error_bad_lines=True)
#plt.plot(dynamic['lat [deg]'], dynamic['lon [deg]'])

#%%
"""
# 1 VESSEL 
idleTimes = []
stops = []
trips = 0
static['trips'] = np.nan
          
for i, speed in enumerate(dynamic['sog [kn]']):
    try:
        if speed < 0.1:
            stops.append(dynamic['timestamp'].iloc[i])
                
        elif speed > 0.1:
            firstStop = datetime.strptime(stops[0], "%Y-%m-%dT%H:%M:%S")
            lastStop = datetime.strptime(stops[-1], "%Y-%m-%dT%H:%M:%S")
            idleTime = (lastStop - firstStop)
            idleTimes.append(idleTime)
            
            stops = []
    except:
        pass

for time in idleTimes:
    if time.total_seconds() > 1800:
        print(time)
        
        trips += 1

static['trips'][(static['mmsi'] == mmsi)] = trips      
"""