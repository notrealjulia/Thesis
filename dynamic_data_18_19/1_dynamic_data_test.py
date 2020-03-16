# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:55:07 2020

@author: JULSP


exploring the data
"""

import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt
import seaborn as sn

path = dirname(__file__)
dynamic_data_path = "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"

data = pd.read_csv(path + "/248595000.csv", sep='	', index_col=None, error_bad_lines=True)

#%%

#pd.read_csv(dynamic_data_path +"/{}.csv".format(mmsi), sep="\t", index_col=None, error_bad_lines=True)
#transforming knots into m/s
data['speed'] = data['sog [kn]'].apply(lambda x: x*0.51444)

#plotting all data points
ax1 = data.plot.scatter(y = 'lat [deg]', x = 'lon [deg]', c = 'speed', alpha=0.1, colormap ='rainbow') # colormap ='ocean'
ax1.set_ylabel("Latitude in Degrees")
ax1.set_xlabel("Longitude in Degrees")
ax1.set_title("All AIS lon/lat data points for one vessel")
ax1.set_clabel("Speed in knots")
#making the timestamp into the index
print(data.head())
#%%

data = data.set_index('timestamp')
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
data_minute = data_minute.rename(columns={'lat [deg]': 'lat', 'lon [deg]': 'lon'})

#BBox = (data_minute.lon.min(), data_minute.lon.max(), data_minute.lat.min(), data_minute.lat.max())

        
ax2 = data_minute.plot.scatter(x = 'lon', y = 'lat', c = 'speed', alpha=0.3, colormap ='rainbow')        
ax2.set_ylabel("Latitude in Degrees")
ax2.set_xlabel("Longitude in Degrees")
ax2.set_title("Resampled AIS lon/lat data points for one vessel for every 10 minutes")

#%%

print(data.head())


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
    
#%%
import seaborn as sn  

#corr_test_df = data_dynamic_test.drop(columns = 'rot [deg/min]')
ax = plt.axes()
data_for_corr = data[['sog [kn]', 'rot [deg/min]', 'lat [deg]', 'lon [deg]', 'cog [deg]']]
corrMatrix = data_for_corr.corr()
sn.heatmap(corrMatrix, ax = ax, center = 0, annot=True)
ax.set_title('Correlation Matrix of Dynamic Variables from one vessel')
plt.show()
      
    
    
#%%
        
        
"""
Testing Linear Regression Assumptions 
"""

data_2 = pd.read_csv(path + "/static_all_with_speed_rot.csv")

data_2 = data_2[data_2['length_from_data_set'] < 401]
data_2 = data_2[data_2['Speed_mean'] < 50]

#data_2["iwrap_type_from_dataset"] = data_2["iwrap_type_from_dataset"].astype('category')
#data_2["iwrap_n"] = data_2["iwrap_type_from_dataset"].cat.codes

""" Linear Relationship"""
# ax1 = data_2.plot.scatter( x = 'length_from_data_set', y = 'Speed_mean', c='iwrap_n', colormap='viridis',  alpha = 0.3)
# ax1.set_title('Relationship between speed and size of vessels in m')
# ax1.set_ylabel('Mean speed of vessel in knots')
# ax1.set_xlabel('Length of vessel')

data_2 = data_2.rename({'iwrap_type_from_dataset': 'Vessel Type'}, axis=1)

ax = sn.scatterplot(x="length_from_data_set", y="Speed_mean", hue="Vessel Type", style = "Vessel Type",  data=data_2)
ax.set_xlabel('Length of vessel in m')
ax.set_ylabel('Mean speed of vessel in kn')
ax.set_title('Relationship between speed and size of vessels')
ax.show()
#%%
""" Normal distribution"""
#normalized data plos exactly the same
data_2_subset = data_2[['length_from_data_set', 'Speed_mean']]
data_norm=((data_2_subset-data_2_subset.min())/(data_2_subset.max()-data_2_subset.min()))*100

# ax2 = data_norm.plot.scatter( x = 'length_from_data_set', y = 'Speed_mean', alpha = 0.3)
# ax2.set_title('Relationship between speed and size of vessels in m')
# ax2.set_ylabel('Mean speed of vessel in knots')
# ax2.set_xlabel('Length of vessel')

#ax3 = data_2_subset.length_from_data_set.plot.hist(alpha = 0.3, bins =20)
ax3 = data_2_subset.length_from_data_set.plot.kde()
ax3.set_title('KDE of vessel lengths')

# ax4 = data_2_subset.Speed_mean.plot.hist(alpha = 0.3, bins =50)
# ax4 = data_2_subset.Speed_mean.plot.kde()
# ax4.set_title('KDE of vessel mean speed')

#data_subset_length = data_2_subset[(data_2_subset['length_from_data_set'] < 200) & (data_2_subset['length_from_data_set'] > 10)]
#ax4 = data_subset_length.length_from_data_set.plot.hist(alpha = 0.3, bins =20)
#ax4.set_title('Distribution of vessel lengths limited')



#%%
"""ROT linear regression"""
data_3 = pd.read_csv(path + "/dynamic_for_linear_models.csv")

#ax5 = plt.axes()
ROT_norm = data_3[[ 'ROT_median', 'ROT_min', 'ROT_max']]
#ROT_normal=((ROT_norm-ROT_norm.min())/(ROT_norm.max()-ROT_norm.min()))*20


ax5 = ROT_norm.plot.hist(alpha = 0.3, bins =50)


#corrMatrix = data_for_corr.corr()
#sn.heatmap(corrMatrix, ax = ax, center = 0, annot=True)
#ax.set_title('Correlation Matrix of Dynamic Variables from one vessel')
#plt.show()

