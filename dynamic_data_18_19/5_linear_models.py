# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:46:31 2020

@author: JULSP
"""


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
data = pd.read_csv(path + "/dynamic_for_linear_models.csv")

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

#Two most correlated features
    
speed_and_rot = data[['mean_speed', 'ROT_min']] #accuracy = 71.4 %
speed_and_rot=((speed_and_rot - speed_and_rot.min()) / (speed_and_rot.max() - speed_and_rot.min())) * 20
speed_and_rot.plot.hist()

lrLength = LR('Length',speed_and_rot, data['length_from_data_set'])

speed_and_rot_all = data[['mean_speed', 'top_speed', 'std_speed', 'Speed_mode', 'Speed_median', 'ROT_mean', 'ROT_max', 'ROT_median', 'ROT_min']]
speed_and_rot_all=((speed_and_rot_all - speed_and_rot_all.min()) / (speed_and_rot_all.max() - speed_and_rot_all.min())) * 20
lrLengthall = LR('Length',speed_and_rot_all, data['length_from_data_set']) #accuracy = 72.75 %

#%%
"""
Lasso Regression
"""
def LassoRegression(output,features, labels):
    X = features.values#.reshape(-1,1) # features
    y = labels.values#.reshape(-1,1) # labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = Lasso()  
    regressor.fit(X_train, y_train) #training the algorithm
    y_pred = regressor.predict(X_test) #predicting

    # visualiation of 10 vessels (predicted vs actual value)
    lr = np.round(pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()}))
    lr = lr.head(10)
    lr.plot(kind='bar',figsize=(10,6))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title('Actual vs Predicted {0} (10 vessels), \n Lasso Regression'.format(output), fontsize=12)
    plt.show()
    #evaluation metrics
    lrEvaluation = evaluation('Lasso Regression {0}'.format(output), y_pred, y_test)
    return(lrEvaluation, lr)


#Two most correlated features
#speed_and_rot = data[['mean_speed', 'ROT_min']] #accuracy = 71.38 %
Lasso_Length = LassoRegression('Length',speed_and_rot, data['length_from_data_set'])

#speed_and_rot_all = data[['mean_speed', 'top_speed', 'std_speed', 'Speed_mode', 'Speed_median', 'ROT_mean', 'ROT_max', 'ROT_median', 'ROT_min']]
Lasso_Length_all = LassoRegression('Length',speed_and_rot_all, data['length_from_data_set']) #accuracy = 72.77 %

#%%
"""
Bayesian Ridge
"""
def BayesianRidgeRegression(output,features, labels):
    X = features.values#.reshape(-1,1) # features
    y = labels.values#.reshape(-1,1) # labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = BayesianRidge()  
    regressor.fit(X_train, y_train) #training the algorithm
    y_pred = regressor.predict(X_test) #predicting

    # visualiation of 10 vessels (predicted vs actual value)
    lr = np.round(pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()}))
    lr = lr.head(10)
    lr.plot(kind='bar',figsize=(10,6))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title('Actual vs Predicted {0} (10 vessels), \n BayesianRidge Regression'.format(output), fontsize=12)
    plt.show()
    #evaluation metrics
    lrEvaluation = evaluation('BayesianRidge Regression {0}'.format(output), y_pred, y_test)
    return(lrEvaluation, lr)

#Two most correlated features
#speed_and_rot = data[['mean_speed', 'ROT_min']] #accuracy = 71.39 %
Lasso_Length = BayesianRidgeRegression('Length',speed_and_rot, data['length_from_data_set'])

#speed_and_rot_all = data[['mean_speed', 'top_speed', 'std_speed', 'Speed_mode', 'Speed_median', 'ROT_mean', 'ROT_max', 'ROT_median', 'ROT_min']]
Lasso_Length_all = BayesianRidgeRegression('Length',speed_and_rot_all, data['length_from_data_set']) #accuracy = 72.77 %

#%%

"""
HuberRegressor - should be better for datasets with strong outliers
"""
def HuberReg(output,features, labels):
    X = features.values#.reshape(-1,1) # features
    y = labels.values#.reshape(-1,1) # labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = HuberRegressor()  
    regressor.fit(X_train, y_train) #training the algorithm
    y_pred = regressor.predict(X_test) #predicting

    # visualiation of 10 vessels (predicted vs actual value)
    lr = np.round(pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()}))
    lr = lr.head(10)
    lr.plot(kind='bar',figsize=(10,6))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title('Actual vs Predicted {0} (10 vessels), \n Huber Regressor'.format(output), fontsize=12)
    plt.show()
    #evaluation metrics
    lrEvaluation = evaluation('Huber Regressor {0}'.format(output), y_pred, y_test)
    return(lrEvaluation, lr)

#Two most correlated features
#speed_and_rot = data[['mean_speed', 'ROT_min']] #accuracy = 72.79 %
Lasso_Length = HuberReg('Length',speed_and_rot, data['length_from_data_set'])

#speed_and_rot_all = data[['mean_speed', 'top_speed', 'std_speed', 'Speed_mode', 'Speed_median', 'ROT_mean', 'ROT_max', 'ROT_median', 'ROT_min']]
Lasso_Length_all = HuberReg('Length',speed_and_rot_all, data['length_from_data_set']) #accuracy = 73.83 %

#%%

"""
DecisionTreeRegressor - THE BEST
"""

from sklearn.tree import DecisionTreeRegressor

def TreesReg(output,features, labels):
    X = features.values#.reshape(-1,1) # features
    y = labels.values#.reshape(-1,1) # labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = DecisionTreeRegressor()  
    regressor.fit(X_train, y_train) #training the algorithm
    y_pred = regressor.predict(X_test) #predicting

    # visualiation of 10 vessels (predicted vs actual value)
    lr = np.round(pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()}))
    lr = lr.head(10)
    lr.plot(kind='bar',figsize=(10,6))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title('Actual vs Predicted {0} (10 vessels), \n Decision Tree Regressor'.format(output), fontsize=12)
    plt.show()
    #evaluation metrics
    lrEvaluation = evaluation('Decision Tree Regressor {0}'.format(output), y_pred, y_test)
    return(lrEvaluation, lr)

#Two most correlated features
#speed_and_rot = data[['mean_speed', 'ROT_min']] #accuracy = 75.1 %
Lasso_Length = TreesReg('Length',speed_and_rot, data['length_from_data_set'])

#speed_and_rot_all = data[['mean_speed', 'top_speed', 'std_speed', 'Speed_mode', 'Speed_median', 'ROT_mean', 'ROT_max', 'ROT_median', 'ROT_min']]
Lasso_Length_all = TreesReg('Length',speed_and_rot_all, data['length_from_data_set']) #accuracy = 79.97 %