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

ax1 = data.plot.scatter(x = 'lat [deg]', y = 'lon [deg]', c = 'speed', colormap ='ocean')

#%%
data_time = data
data_time['timestamp'] = data_time['timestamp'].str.replace(r'\D', '')
data_time = data_time.set_index(['timestamp'])
data_time.index = pd.to_datetime(data_time.index, format='%Y%m%d%H%M%S')
#format='%Y%m%d%H%M%S'
data_small = data['timestamp'].resample('3T')