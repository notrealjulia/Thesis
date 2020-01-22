# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 18:20:43 2020

@author: Kornelija
"""

"""
missingMMSI = frame.loc[(frame['mmsi'] == 0.0) | (frame['mmsi'].isna())]
missingLenght = frame.loc[(frame['length'] == 0.0) | (frame['length'].isna())]
missingWidth = frame.loc[(frame['width'] == 0.0) | (frame['width'].isna())]
missingMinDraught = frame.loc[(frame['minDraught'] == 0.0) | (frame['minDraught'].isna())]
missingMaxDraught = frame.loc[(frame['maxDraught'] == 0.0) | (frame['maxDraught'].isna())]
missingTypeMin = frame.loc[(frame['typeMin'] == 0.0) | (frame['typeMin'].isna())]
missingTypeMax = frame.loc[(frame['typeMax'] == 0.0) | (frame['typeMax'].isna())]
missingIMO = frame.loc[(frame['imo'] == 0.0) | (frame['imo'].isna())]
missingShipName = frame.loc[(frame['shipName'] == 0.0) | (frame['shipName'].isna())]
missingAISType = frame.loc[(frame['aisType'] == 0.0) | (frame['aisType'].isna())]
missingCallSign = frame.loc[(frame['callSign'] == 0.0) | (frame['callSign'].isna())]
missingA = frame.loc[(frame['a'] == 0.0) | (frame['a'].isna())]
missingB = frame.loc[(frame['b'] == 0.0) | (frame['b'].isna())]
missingC = frame.loc[(frame['c'] == 0.0) | (frame['c'].isna())]
missingD = frame.loc[(frame['d'] == 0.0) | (frame['d'].isna())]

names = ['mmsi', 'length', 'width', 'minDraught', 'maxDraught', 'typeMin', 'typeMax', 'imo', 'shipName', 'aisType', 'callSign', 'a', 'b', 'c', 'd']
missingSize = frame.loc[(frame['length'] == 0) & (frame['width'] == 0)]
missingSizeAndType = missingSize.loc[missingSize['typeMax'] == 0.0]

variables = [missingMMSI, missingLenght, missingWidth, missingMinDraught, missingMaxDraught, missingTypeMin, missingTypeMax, missingIMO, missingShipName, missingAISType, missingCallSign, missingA, missingB, missingC, missingD]
"""