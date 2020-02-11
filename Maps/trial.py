# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:16:24 2020

@author: Kornelija
"""
from os.path import dirname
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt  


path = dirname(__file__)
frame = pd.read_csv(path + "/245176000.csv", sep='	', index_col=None, error_bad_lines=False)


minlon = max(-180, min(frame['lon [deg]'])-1)
minlat = max(-90, min(frame['lat [deg]'])-1)
maxlon = min(180, max(frame['lon [deg]'])+1)
maxlat = min(90, max(frame['lat [deg]'])+1)

lat0 = (maxlat + minlat)/2
lon0 = (maxlon+minlon)/2
lat1 = (maxlat+minlat)/2-20

fig,ax=plt.subplots(figsize=(15,15))
m = Basemap(llcrnrlon=minlon,llcrnrlat=minlat,urcrnrlon=maxlon,
            urcrnrlat=maxlat, rsphere=(6378137.00,6356752.3142),
            resolution='i',projection='cyl',lat_0=lat0,lon_0=lon0,
            lat_ts = lat1)
m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary(fill_color='white')
m.fillcontinents(color='lightgrey',lake_color='white')
x, y = m(frame['lon [deg]'],frame['lat [deg]'])
m.scatter(x,y,s=0.1, marker='.',c='red')
ax.set_title('245176000 route')
plt.xlabel('lat', fontsize = 14)
plt.ylabel('lon', fontsize = 14)

plt.show()

