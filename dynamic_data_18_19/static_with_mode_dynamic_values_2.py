# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 17:18:59 2020

@author: JULSP
"""

import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import mode


path = dirname(__file__)
dynamic_data_path = "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"

static_data = pd.read_csv(path + "/static_with_ROT_Speed_1464.csv")

example_df =pd.read_csv(path + '/245176000.csv', sep='	', index_col=None, error_bad_lines=False)

#static_data = static_data[:5]
#example_df['speed_rounded'] = np.nan
#example_df['speed_rounded'] = example_df.iloc[:,3].round(0)
#mode_of_mean_speed = example_df.mode()['speed_rounded'][0]

#%%

"""
Getting mode speed and ROT
"""
#For ROT drop 0 values
static_data['Speed_mode'] = np.nan
static_data['ROT_mode'] = np.nan

static_data['Speed_mode'] = static_data['Speed_mode'].astype(object)
static_data['ROT_mode'] = static_data['ROT_mode'].astype(object)

f = lambda x: mode(x, axis=None)[0]

for i in range(len(static_data)):
    try:
        print("getting dynamic data for ship number", i+1)
    #getting mmsi name of the ship from the static data 
        mmsi = static_data.mmsi[i]
    #loading the dynamic file for that specific ship
        dynamic = pd.read_csv(dynamic_data_path +"/{}.csv".format(mmsi), sep='	', index_col=None, error_bad_lines=False)
    #calculating mean, max, std speed for that specific ship and saving it in the static DF
        dynamic['rounded_speed'] = np.nan
        dynamic['rounded_speed'] = dynamic.iloc[:,3].round(0) #rounding up to integers
        #print(dynamic['rounded_speed'][0])
        static_data['Speed_mode'].loc[i] = dynamic.mode()['rounded_speed'][0]
        
        dynamic['rounded_ROT'] = np.nan
        dynamic['rounded_ROT'] = dynamic.iloc[:,7].round(0)
        dynamic = dynamic[(dynamic[['rounded_ROT']] != 0).all(axis=1)] #dropping all 0 values
        static_data['ROT_mode'].loc[i] = dynamic.mode()['rounded_ROT'][0]
        static_data['ROT_median'].loc[i] = dynamic.iloc[:,13].median()
#        static_data['ROT_median'].loc[i] = dynamic.iloc[:,7].median()
#        static_data["ROT_min"].loc[i] = dynamic.iloc[:,7].min()
#        static_data["Speed_median"].loc[i] = dynamic.iloc[:,3].median()
    except:
        print("couldn't get data on", i+1)
        
#%%
"""
Pre-processing
"""   
simple_dynamic = static_data[static_data.length_from_data_set != -1]

simple_dynamic = simple_dynamic[['length_from_data_set', 'width', 'iwrap_type_from_dataset', 'mean_speed', 'top_speed', 'std_speed', 'Speed_mode', 'Speed_median', 'ROT_mean', 'ROT_max', 'ROT_median', 'ROT_min', 'ROT_mode']]
    
#%%    
"""
Plotting the new stuff
"""

import seaborn as sn

#static_data['length/width'] = static_data['length_from_data_set'] / static_data['width']
#list_of_columns = [9,15,16,17,18,19,21,28,30,22,23,24,26,25,27,31]
list_of_columns = [9,15,21,28,30,22,23,24,26,25,27,31]

correlation_df = static_data.iloc[:,list_of_columns]
correlation_df.rename({'length_from_data_set':'vessel_length', 'width':'vessel_width', 'mean_speed':'Speed_mean', 'top_speed':'Speed_max','std_speed':'Speed_std'}, axis = 1, inplace = True)

corrMatrix = correlation_df.corr()
sn.heatmap(corrMatrix,center = 0, annot=True)

#%%
import seaborn as sn
#list_of_CH = [5,6]
#COG_heading = example_df.iloc[:,list_of_CH]

simple_dynamic = simple_dynamic.fillna(0)
simple_dynamic_corr = simple_dynamic.drop('iwrap_type_from_dataset', axis=1)

corrMatrix = simple_dynamic_corr.corr()
sn.heatmap(corrMatrix,center = 0, annot=True)

#%%

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn import metrics

def evaluation(model, predicted, actual):
    errors = abs(predicted-actual)
    averageError = np.mean(abs(predicted-actual))
    mape = 100 * (errors / actual)
    accuracy = 100-np.mean(mape)
    mse = metrics.mean_squared_error(predicted, actual)
    print(np.mean(mape))
    print(model, '\nAverage Absolute Error = ', round(averageError),'m', '\nMSE = ', round(mse, 2), '\nAccuracy = ', round(accuracy, 2), '%')
    return({'averageAbsoluteError': averageError, 'mse':mse, 'accuracy':accuracy})

def LR(output,features, labels):
    X = features.values#.reshape(-1,1) # features
    y = labels.values#.reshape(-1,1) # labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train) #training the algorithm
    y_pred = regressor.predict(X_test) #predicting

    # visualiation of 10 vessels (predicted vs actual value)
    lr = np.round(pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()}))
    lr = lr.head(10)
    lr.plot(kind='bar',figsize=(10,6))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title('Actual vs Predicted {0} (10 vessels), \n Linear Regression'.format(output), fontsize=12)
    plt.show()
    #evaluation metrics
    lrEvaluation = evaluation('Linear Regression {0}'.format(output), y_pred, y_test)
    return(lrEvaluation, lr)

#%%
    
#X = pd.DataFrame(np.c_[static_data['ROT_min'], static_data['mean_speed']], columns=['ROT_min','mean_speed']
speed_and_rot = simple_dynamic[['mean_speed', 'ROT_min']]
#speed_and_rot2 = np.c_[simple_dynamic['mean_speed'], simple_dynamic['ROT_min']], columns=['mean_speed','ROT_min']
lrLength = LR('Length',speed_and_rot, simple_dynamic['length_from_data_set'])

#%%

#simple_dynamic = simple_dynamic['ROT_mode'].dropna(how='all')    
#X = pd.DataFrame(np.c_[static_data['ROT_min'], static_data['mean_speed']], columns=['ROT_min','mean_speed']
speed_and_rot = simple_dynamic[['mean_speed', 'top_speed', 'std_speed', 'Speed_mode', 'Speed_median', 'ROT_mean', 'ROT_max', 'ROT_median', 'ROT_min']]
#speed_and_rot2 = np.c_[simple_dynamic['mean_speed'], simple_dynamic['ROT_min']], columns=['mean_speed','ROT_min']
lrLength = LR('Length',speed_and_rot, simple_dynamic['length_from_data_set'])

#%%
simple_dynamic.to_csv(path+'dynamic_for_linear_models.csv', encoding ='utf-8', index=False)