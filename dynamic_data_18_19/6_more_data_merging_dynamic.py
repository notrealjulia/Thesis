# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:09:14 2020

@author: JULSP
"""

import pandas as pd
from os.path import dirname
import numpy as np
from scipy.stats.mstats import mode

path = dirname(__file__)
dynamic_data_path = "//garbo/Afd-681/05-Technical Knowledge/05-03-Navigational Safety/00 Udviklingsprojekter/01 ML - Missing properties/02 Work/01 IWRAP/ML South Baltic Sea vA/export"

static_data_full = pd.read_csv(path + "/static_ship_data.csv", sep='	', index_col=None, error_bad_lines=False)
static_data = static_data_full[static_data_full.length_from_data_set != -1] #drop -1 in width

data_raw =  pd.read_csv(path + "/static_all_with_speed_rot.csv")

print(round((len(static_data)/len(static_data_full)*100), 1), 'percent of ships have information about vessel length') #85.4
#%%
"""
Pre-processing dynamic data
"""

data_processed = data_raw[data_raw.length_from_data_set < 600] #vessels that are over 400m are usually helicopters
data_processed = data_processed[data_processed.length_from_data_set > 10]
data_processed = data_processed[data_processed.Speed_mean != 0]
data_processed = data_processed[data_processed.iwrap_type_from_dataset != 'Pleasure boat']

data_with_ROT = data_processed[data_processed.ROT_max != 0]

print(len(data_with_ROT)/len(data_processed)) #16 %, 26% without pleasure boats
#%%

#print(static_data['iwrap_type_from_dataset'].unique())

cargo = static_data[static_data['iwrap_type_from_dataset'] == 'General cargo ship']
tanker = static_data[static_data['iwrap_type_from_dataset'] == 'Oil products tanker']
fishing = static_data[static_data['iwrap_type_from_datadata_processed = data[data.length_from_data_set > 10]set'] == 'Fishing ship']
passenger = static_data[static_data['iwrap_type_from_dataset'] == 'Passenger ship']
boat = static_data[static_data['iwrap_type_from_dataset'] == 'Pleasure boat']
support = static_data[static_data['iwrap_type_from_dataset'] == 'Support ship']
ferry = static_data[static_data['iwrap_type_from_dataset'] == 'Fast ferry']

"""
print('tanker is', round(tanker['length_from_data_set'].mean(), 1)) # 178.7m
print('cargo is', round(cargo['length_from_data_set'].mean(), 1)) # 157.1m
print('passenger is', round(passenger['length_from_data_set'].mean(), 1)) # 139.1m
print('support is', round(support['length_from_data_set'].mean(), 1)) #39.3m
print('ferry is', round(ferry['length_from_data_set'].mean(), 1)) #22.6m
print('fishing is', round(fishing['length_from_data_set'].mean(), 1)) #17.9m
print('boat is', round(boat['length_from_data_set'].mean(), 1)) #12.8m
"""
static_data_ships = pd.concat([tanker, cargo, passenger, support, ferry, fishing, boat])
print(round((len(static_data_ships)/len(static_data_full)*100), 1), 'percent of ships have information about vessel type') #80.4

static_data_ships = static_data_ships[static_data_ships.width != -1] #drop -1 in width
static_data_ships = static_data_ships.reset_index(drop =True) #reset index

static_data_ships['Speed_mean'] = np.nan
static_data_ships['Speed_std'] = np.nan
static_data_ships['Speed_median'] = np.nan
static_data_ships['Speed_max'] = np.nan
static_data_ships['Speed_min'] = np.nan
static_data_ships['Speed_mode'] = np.nan

static_data_ships['ROT_mean'] = np.nan
static_data_ships['ROT_std'] = np.nan
static_data_ships['ROT_median'] = np.nan
static_data_ships['ROT_max'] = np.nan
static_data_ships['ROT_min'] = np.nan
 #%%

"""
getting dynamic data for all known ship types
"""

#static_data_ships = static_data_ships[:15] #test
def get_dynamic(static_data_ships_local):
    
    for i in range(len(static_data_ships_local)):
        print(i)
        try:
            #getting mmsi name of the ship from the static data 
            mmsi = static_data_ships_local.mmsi.iloc[i]
            print("getting dynamic data for ship ID", mmsi)
            #print('mmsi ',mmsi)
            #loading the dynamic file for that specific ship
            dynamic = pd.read_csv(dynamic_data_path +"/{}.csv".format(mmsi), sep="\t", index_col=None, error_bad_lines=True)
        except:
            print("couldn't open file for ship", mmsi)
            pass
        try:
            #calculating mean, max, std speed for that specific ship and saving it in the static DF
            #dynamic = dynamic[(dynamic[['sog [kn]']] != 0).all(axis=1)] #dropping all 0 values
            #TODO NOTE this way of working with pandas, the commented out version is the bad way, the following line is the correct way, 10x faster
            #static_data_ships["Speed_mean"].loc[i] = dynamic.iloc[:,3].mean()
            static_data_ships_local.loc[i,"Speed_mean"] = dynamic.iloc[:,3].mean()
            static_data_ships_local.loc[i,"Speed_std"] = dynamic.iloc[:,3].std()
            static_data_ships_local.loc[i,"Speed_median"] = dynamic.iloc[:,3].median()
            static_data_ships_local.loc[i,"Speed_max"] = dynamic.iloc[:,3].max()
            static_data_ships_local.loc[i,"Speed_min"] = dynamic.iloc[:,3].min()
            dynamic['rounded_speed'] = np.nan
            dynamic['rounded_speed'] = dynamic.iloc[:,3].round(0) #rounding up to integers
        #print(dynamic['rounded_speed'][0])
            static_data_ships_local.loc[i,'Speed_mode'] = dynamic.mode()['rounded_speed'][0]
        except Exception as inst:
            print("couldn't get speed data on ship", mmsi)
            print (inst.args)
            print (inst)
        
        #calculating max and min ROT values
        try:
            #dynamic = dynamic[(dynamic[['rot [deg/min]']] != 0).all(axis=1)] #dropping all 0 values
            static_data_ships_local.loc[i,"ROT_mean"] = dynamic.iloc[:,7].mean()
            static_data_ships_local.loc[i,"ROT_std"] = dynamic.iloc[:,7].std()
            static_data_ships_local.loc[i,"ROT_median"] = dynamic.iloc[:,7].median()
            static_data_ships_local.loc[i,"ROT_max"] = dynamic.iloc[:,7].max()
            static_data_ships_local.loc[i,"ROT_min"] = dynamic.iloc[:,7].min()
        except:
            print("couldn't get ROT values for ship", mmsi)
            
    return static_data_ships_local
        
#%%       
import winsound
frequency = 300  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
   
#static_data_ships_1 = static_data_ships[:1000]
#static_data_ships_1 = static_data_ships_1.reset_index(drop =True) #reset index

#dynamic_1 = get_dynamic(static_data_ships_1)
dynamic_all = get_dynamic(static_data_ships)
#winsound.Beep(frequency, duration)

#%%
#data = static_data_ships.dropna(subset = ['Speed_mean'])       
#dynamic_1_vessel =pd.read_csv(dynamic_data_path + '/245405000.csv', sep="\t", index_col=None, error_bad_lines=False)
#atd_1 = dynamic_1_vessel.iloc[:,3].std()

dynamic_all.to_csv(path + 'static_all_with_speed_rot.csv', encoding ='utf-8', index=False)

#%%

import seaborn as sn

df = dynamic_all[(dynamic_all[['Speed_mean']] != 0).all(axis=1)]

list_of_columns = [9,15,21,22,23,24,25,26]

correlation_df = dynamic_all.iloc[:,list_of_columns]
correlation_df.rename({'length_from_data_set':'vessel_length', 'width':'vessel_width'}, axis = 1, inplace = True)

corrMatrix = correlation_df.corr()

#%%
sn.heatmap(corrMatrix,center = 0, annot=True)

#%%

import pandas as pd
from os.path import dirname
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.mstats import mode
import seaborn as sn
from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge, HuberRegressor
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn import metrics

path = dirname(__file__)
data = data_processed

def evaluation(model, predicted, actual):
    errors = abs(predicted-actual)
    averageError = np.mean(abs(predicted-actual))
    mape = 100 * (errors / actual)
    accuracy = 100-np.mean(mape)
    mse = metrics.mean_squared_error(predicted, actual)
    print(np.mean(mape))
    print(model, '\nAverage Absolute Error = ', round(averageError),'m', '\nMSE = ', round(mse, 2), '\nAccuracy = ', round(accuracy, 2), '%')
    return({'averageAbsoluteError': averageError, 'mse':mse, 'accuracy':accuracy})

#%%
    
"""
Linear Regression
"""

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


data = data[(data[['Speed_mean']] != 0).all(axis=1)]


#Two most correlated features
speed_and_rot = data[['Speed_mean']] #accuracy = 71.4 %
lrLength = LR('Length',speed_and_rot, data['length_from_data_set'])

speed_and_rot_all = data[['Speed_mean', 'Speed_median', 'Speed_mode','Speed_min','Speed_max']]
lrLengthall = LR('Length',speed_and_rot_all, data['length_from_data_set']) 