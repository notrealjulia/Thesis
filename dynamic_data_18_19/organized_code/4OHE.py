# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 10:07:23 2020

@author: KORAL
"""

import numpy as np
import pandas as pd 

"""
We set here the resolution of the latitude and longitude coordinate at about 1km, 
the resolution of SOG at 1knot and the resolution of COG at 5


SOG:  Our range is from 0 to 102.2, with 99.98% of the data falling into 0 to 30 kn range, so the remaining is removed.
Also, speed usually does not exceed 30kn per hour (same approach used by reference authors)
This leaves us to 30 bins, 1 kn each. 

COG: Our range is 0 to 360. Recommended precision is 5 degrees (360/5 = 72 bins)



The valid range of latitude in degrees is -90 and +90 for the southern and northern hemisphere respectively. 
Longitude is in the range -180 and +180 specifying coordinates west and east of the Prime Meridian, respectively.
As per reference article, we're using 1 km precision.

LAT: Each degree of latitude is approximately 69 miles (111 kilometers) apart.
Our range is approximately 1 deg (from 54.0927 to 54.9735), so we can use 110 bins 

LON: Distance depends on LAT degrees. Find distance using haversine's formula. 
Our range is about 4 deg (from 9.8153 to 13.4101), which combined with 
LAT translates to 229 - 251 miles depending on LAT positions. We can use 250 bins.  


    ### e.g. for calclating LON distance 

import mpu

# Point one
lat1 = 54.0927
lon1 = 9.8153

# Point two
lat2 = 54.0927
lon2 = 13.4101

# What you were looking for
dist = mpu.haversine_distance((lat1, lon1), (lat2, lon2))
print(dist)



"""



active = pd.read_parquet(r'C:\Users\KORAL\OneDrive - Ramboll\Documents\GitHub\Thesis\Thesis\dynamic_data_18_19\organized_code\active_dynamic_September.parquet')
activeSpeed = active.loc[active['sog [kn]'] <= 30]  ## only around 1500 signals were more than 35 kn, average speed usualy does not exceed 30 
subset = active[:1000000]


#%%
    ## BINS used by reference authors (our LON and LAT bins are different because of different ROI size):
# LAT_BINS = 300; LON_BINS = 300; SOG_BINS = 30; COG_BINS = 72 

# Binning 
COG_bins = np.linspace(0, 360, 72)

SOG_bins = np.linspace(0, 30, 30)

LAT_bins = np.linspace(54.0927, 54.9735, 110)

LON_bins = np.linspace(9.8153, 13.4101, 250)

#%%

    # EXAMPLE 
sog = subset['sog [kn]'].values.reshape(-1, 1) 
which_bin = np.digitize(sog, bins=SOG_bins)
print("\nData points:\n", sog[:5])
print("\nBin membership for data points:\n", which_bin[:5])


# Encoding (OHE)
from sklearn.preprocessing import OneHotEncoder
# transform using the OneHotEncoder
encoder = OneHotEncoder(sparse=False)
# encoder.fit finds the unique values that appear in which_bin
encoder.fit(which_bin)
# transform creates the one-hot encoding
X_binned = encoder.transform(which_bin)
print(X_binned[:5])

#%%
# TO DO (K): 

# Resample data every 10 minutes, bin and  encode the remaining features (4), should result in 462 length vector. 
# Take a sample of clean data (Category and size) for ML. 
# Remember to separate Other from Undefined using original static data!!!



