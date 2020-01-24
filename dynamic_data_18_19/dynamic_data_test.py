# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:55:07 2020

@author: JULSP
"""

import pandas as pd
from os.path import dirname

path = dirname(__file__)
data = pd.read_csv(path + "/246669000.csv", sep='	', index_col=None, error_bad_lines=False)

#transforming knots into m/s
data['speed'] = data['sog [kn]'].apply(lambda x: x*0.51444)

#plotting all data points
ax1 = data.plot.scatter(x = 'lat [deg]', y = 'lon [deg]', c = 'speed', colormap ='ocean') # colormap ='ocean'

#making hte timestamp into the index
data = data.set_index('timestamp')
#%%

#transforming index into resampable form 
data.index = pd.to_datetime(data.index)

#reaampling for every 10minutes and dropping empty values, taking mean value of the new sample bin
data_lon = data['lon [deg]'].resample('10T', how='mean')
data_lon = data_lon.dropna()

data_lat = data['lat [deg]'].resample('10T', how='mean')
data_lat = data_lat.dropna()

data_speed = data['speed'].resample('10T', how='mean')
data_speed = data_speed.dropna()

#concat and plot downsampled data
data_minute = pd.concat([data_lat, data_lon, data_speed], axis=1)
ax2= data_minute.plot.scatter(x = 'lat [deg]', y = 'lon [deg]', c = 'speed', colormap ='ocean') # colormap ='ocean'

#%%

#TODO: make the above into a function to run all data ??

sample_rate = '10T'

def dynamic_data_processing(dataset, sample_rate, ):
    
    for data in dataset:
        data['speed'] = data['sog [kn]'].apply(lambda x: x*0.51444)#transforming knots into m/s
        data = data.set_index('timestamp')
        data.index = pd.to_datetime(data.index)
        
        data_lon = data['lon [deg]'].resample(sample_rate, how='mean')
        data_lon = data_lon.dropna()
        data_lat = data['lat [deg]'].resample(sample_rate, how='mean')
        data_lat = data_lat.dropna()
        data_speed = data['speed'].resample(sample_rate, how='mean')
        data_speed = data_speed.dropna()
        
        return dataset