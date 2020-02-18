# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:17:28 2020

@author: KORAL
"""


"""
    Four-hot-encoding
Concatenating one-hot vectors of [lat, lon, SOG, COG]
To create the one-hot vector, we divide the entire value range into N_attribute_i equal-width bins


    Pre-processing:
Remove vessels whose speed is smaller than 0.1 knots for more than 80% of the time
Downsample to 10 minutes scale
For training only choose tracks that don't have more than 1 hour time interval between two succesful AIS messages.

The hyper-parameters are the resolution of each bin in
the one-hot vectors

    Encoding resolutions: 
LAT and LON: 0.01◦
SOG: 1 knot 
COG: 1°

"""
from os.path import dirname
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = dirname(__file__)
frame = pd.read_csv(path + "/dynamic_sample/245176000.csv", sep='	', index_col=None, error_bad_lines=False)

360/5
#LAT_BINS = 300; LON_BINS = 300; SOG_BINS = 30; COG_BINS = 72 
plt.hist(x=frame['cog [deg]'], bins=72)

