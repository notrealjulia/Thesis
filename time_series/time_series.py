# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:16:09 2020

@author: JULSP
"""

from numpy import array

data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

	
data = data.reshape((1, len(data), 1))

print(data.shape)

#%%

data_2d = array([
	[0.1, 1.0],
	[0.2, 0.9],
	[0.3, 0.8],
	[0.4, 0.7],
	[0.5, 0.6],
	[0.6, 0.5],
	[0.7, 0.4],
	[0.8, 0.3],
	[0.9, 0.2],
	[1.0, 0.1]])


print(data_2d.shape)

data_2d = data_2d.reshape(1, 10, 2)