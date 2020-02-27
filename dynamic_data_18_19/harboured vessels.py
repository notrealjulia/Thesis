# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:02:59 2020

@author: KORAL
"""

"""
Remove vessels whose speed is smaller than 0.1 knots for more than 80% of the time and find helicopters 

"""

import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt
import numpy as np

path = dirname(__file__)
static = pd.read_csv(r"C:\Users\KORAL\OneDrive - Ramboll\Documents\GitHub\Thesis\Thesis\Static Data 2018-2019/static_ship_data.csv", sep='	', index_col=None, error_bad_lines=False)
dynamic_path= "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"

dictionary = static.groupby('iwrap_type_from_dataset')['mmsi'].apply(lambda g: g.values.tolist()).to_dict()

harboured_vessels = []
errors = []

cut_labels = ['still', 'moving', 'helicopter']
cut_bins = [-0.1, 0.1, 60, 120]
for mmsi in dictionary.get('Other ship'):
    try:
        dynamic = pd.read_csv(dynamic_path +"/{}.csv".format(str(mmsi)), sep='	', index_col=None, error_bad_lines=False)
        dynamic['bins'] = pd.cut(dynamic['sog [kn]'], bins=cut_bins, labels=cut_labels)
        bins_count = dynamic.bins.value_counts(normalize=True)
        if bins_count.still > .8:
            harboured_vessels.append(mmsi)
        elif bins_count.helicopter > 0.1:
            print(mmsi, 'Helicopter!')
    except:
        print ('memory error at', mmsi) 
        errors.append(mmsi)     # find how to deal with those 
