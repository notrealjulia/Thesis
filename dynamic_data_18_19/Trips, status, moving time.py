# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:49:15 2020

@author: KORAL
"""






"""

Goes trough MMSI numbers from static file:
    - Opens corresponding dynamic file:
        - Goes trough all the speeds in the dynamic file
            - When speed is smaller than 0.1, counter of timestamps is initiated and timestamps are added to the 'stops' list. 
            - When ship starts to move again, the time difference between first and last time are found 
            (this corresponds to duration of continuous idleTime and is added to idleTimes list). Stops list is reset.  
            
            
            - When speed is equal or bigger than 0.1, counter of timestamps is initiated and timestamps are added to the 'moves' list. 
            - When ship stops again, the time difference between first and last time are found 
           (this corresponds to duration of continuous moveTime and is added to moveTimes list). Moves list is reset. 
    
    
    - total moving/idle time found by adding timedelta objects from movingTimes and idleTimes lists.
            
    - if proportion of idle time is equal or more than 80%, the ship status is changed to 'harboured' in the static file
    - if proportion of idle time is equal or more than 80%, the ship status is changed to 'active' in the static file
    
    - total moving and idle times are added to the static file.
    
    - the number of stops from idleTimes list that exceed certain limit (.e.g. 15 minutes´) and added as 'trips' in the static file. 
    

            

"""




from datetime import datetime
import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt
import numpy as np
import sys
import progressbar
import operator
import functools 


print(sys.path)
path = dirname(__file__)
dynamic_path= "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"
static = pd.read_csv(r"C:\Users\KORAL\OneDrive - Ramboll\Documents\GitHub\Thesis\Thesis\Static Data 2018-2019/static_ship_data.csv", sep='	', index_col=None, error_bad_lines=False)
dynamic = pd.read_csv(dynamic_path + "/211727510.csv", sep='	', index_col=None, error_bad_lines=True)
dictionary = static.groupby('iwrap_type_from_dataset')['mmsi'].apply(lambda g: g.values.tolist()).to_dict()


#%%
static['trips'] = np.nan
static['status'] = np.nan
static['idleTime']= np.nan
static['movingTime']= np.nan

memoryErrors = []
parserErrors = []
otherErrors = []
for mmsi in progressbar.progressbar(enumerate(static['mmsi'])): 
    
    try:
        dynamic = pd.read_csv(dynamic_path +"/{}.csv".format(str(mmsi)), sep='	', index_col=None, error_bad_lines=False)    
        idleTimes = []
        stops = []
        moves = []
        movingTimes = []
        trips = 0
        
                   
        for i, speed in enumerate(dynamic['sog [kn]']):
            try:
                if speed < 0.1:
                    stops.append(dynamic['timestamp'].iloc[i])
                    firstMove = datetime.strptime(moves[0], "%Y-%m-%dT%H:%M:%S")
                    lastMove = datetime.strptime(moves[-1], "%Y-%m-%dT%H:%M:%S")
                    movingTime = (lastMove - firstMove)
                    movingTimes.append(movingTime)
                    moves = []
                        
                elif speed >= 0.1:
                    moves.append(dynamic['timestamp'].iloc[i])
                    firstStop = datetime.strptime(stops[0], "%Y-%m-%dT%H:%M:%S")
                    lastStop = datetime.strptime(stops[-1], "%Y-%m-%dT%H:%M:%S")
                    idleTime = (lastStop - firstStop)
                    idleTimes.append(idleTime)
                    stops = []
            except:
                pass
            # Finding harboured vessels 
        idle = functools.reduce(operator.add, idleTimes)
        moving = functools.reduce(operator.add, movingTimes)
        status = np.round(idle * 100 / (idle + moving))
        print(status)
        static['idleTime'][(static['mmsi'] == mmsi)] = idle
        static['movingTime'][(static['mmsi'] == mmsi)] = moving
        if status >= 80:
            static['status'][(static['mmsi'] == mmsi)] = 'harboured'
        else:
            static['status'][(static['mmsi'] == mmsi)] = 'active'
            
            # finding number of trips:
        for idleTime in idleTimes:
            if idleTime.total_seconds() > 900:  # stop time defined in seconds, currently = 15 minutes    
                trips += 1
        static['trips'][(static['mmsi'] == mmsi)] = trips  
        
        
    except MemoryError as error:
        memoryErrors.append((mmsi,error))
    except pd.errors.ParserError as er:
        parserErrors.append((mmsi,er))
    except:
        otherErrors.append(mmsi)
        print('Other error at', mmsi)
        
static.to_csv('static, trips and status', sep='\t')

#%%

### TESTING 
print(static.loc[static['mmsi'] == mmsi][['trips', 'status', 'idleTime', 'movingTime']])
dynamic = pd.read_csv(dynamic_path + "/{}.csv".format(mmsi), sep='	', index_col=None, error_bad_lines=True)
plt.plot(dynamic['lat [deg]'], dynamic['lon [deg]'])



#%%
static.to_csv('static with status and trips', sep='\t')

