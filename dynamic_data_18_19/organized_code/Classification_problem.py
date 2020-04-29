# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:47:47 2020

@author: JULSP
"""
import pandas as pd
import sklearn 
import numpy as np
from os.path import dirname

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

#%%
"""
0.
Loading Data
& Pre-Processing
"""

path = dirname(__file__)

data_all = pd.read_csv(path + "/data_jun_jul_aug_sep_oct.csv")

data_clean = data_all.drop(['trips'], axis=1)
#data_clean = data_clean[data_clean.iwrap_type_from_dataset != 'Other ship']
data_clean = data_clean.dropna() #drop nan values, our models can't handle this
data_processed = data_clean[data_clean.length_from_data_set < 400] #vessels that are over 400m do not exist
data_processed = data_processed[data_processed.Speed_max <65] #boats that go faster than that are helicopters
data_processed = data_processed[data_processed.Speed_max >0] #boats that stand still

data_processed = shuffle(data_processed) #shuffle the data to get different boats for limited versions
data_processed = data_processed.reset_index(drop =True) #reset index

types_of_vessels = data_processed.iwrap_type_from_dataset.value_counts()
print(types_of_vessels)

#%%
# """
# 0.
# Ecoding the labels
# Picking X and y variables
# Splitting train/test/validation
# Scaling
# """

# #Ecoding the labels
# lb = LabelEncoder()
# data_processed['iwrap_cat'] = lb.fit_transform(data_processed['iwrap_type_from_dataset'])
# # picking X and y and splitting
# X = data_processed[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
# y = data_processed[['iwrap_cat']]
# print(y[:10])
# y = y.values.ravel() #some of the model wants this to be an array

# print(y[:10])
# #split into test and train+val 
# X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2,  random_state=0, stratify = y)
# # split train+validation set into training and validation sets
# X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1, stratify = y_trainval)

# print("\nSize of training set: {}   size of validation set: {}   size of test set:" " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

# #Scaling the data
# scaler = RobustScaler() #accounts for outliers
# scaler.fit(X_trainval)
# X_trainval_scaled = scaler.transform(X_trainval)
# X_valid_scaled = scaler.transform(X_valid)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# #make sure that we have all classes represented
# print(np.unique(y_trainval))
# print(np.unique(y_test))
# print(np.unique(y_train))
# print(np.unique(y_valid))

# print(lb.classes_)

#%%
# """
# 1.
# Logistic Regression 
# """
# logreg = LogisticRegression(max_iter=10000,C=0.5)
# # Validation
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# corss_val_log = cross_val_score(logreg, X_valid_scaled, y_valid, cv=kfold)
# print("Cross-validation scores for Logistic reg:\n{}".format(corss_val_log))
# print("Mean cross validation for Logistic reg {:.3f}".format(np.mean(corss_val_log)))

# #Training and test
# logreg.fit(X_train_scaled, y_train)
# print("Training set score: {:.3f}".format(logreg.score(X_train_scaled, y_train)))
# print("Test set score: {:.3f}\n".format(logreg.score(X_test_scaled, y_test)))
# y_pred = logreg.predict(X_test_scaled)

# #Classification report
# print(classification_report(y_test, y_pred, target_names=lb.classes_))
# print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

#%%

"""
VERSION 2
Ecoding the labels
Picking X and y variables
Splitting train/test/validation
Scaling
"""


#Ecoding the labels with one-hot-encoder
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()

lb.fit(data_processed['iwrap_type_from_dataset'])
labels = lb.transform(data_processed['iwrap_type_from_dataset'])
X = data_processed[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
y = labels

print(y[:10])

#split into test and train+val 
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2,  random_state=0, stratify = y)
# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1, stratify = y_trainval)

print("\nSize of training set: {}   size of validation set: {}   size of test set:" " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

#Scaling the data
scaler = RobustScaler() #accounts for outliers
scaler.fit(X_trainval)
X_trainval_scaled = scaler.transform(X_trainval)
X_valid_scaled = scaler.transform(X_valid)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#make sure that we have all classes represented
# print(np.unique(y_trainval))
# print(np.unique(y_test))
# print(np.unique(y_train))
# print(np.unique(y_valid))

print(lb.classes_)
print(lb.y_type_)
#%%

# # X_valid_scaled = np.array(X_valid_scaled)
# y_valid = y_valid.values.ravel()
# # X_train_scaled = np.array(X_train_scaled)
# y_train = y_train.flatten('F')
# # X_test_scaled = np.array(X_test_scaled)
# # X_test_scaled = np.array(X_test_scaled)
# y_test = y_test.flatten('F')

#%%
"""
1.
Logistic Regression 
"""
from sklearn.linear_model import LogisticRegressionCV
logreg = LogisticRegressionCV(multi_class= 'multinomial')
# Validation
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# corss_val_log = cross_val_score(logreg, X_valid_scaled, y_valid, cv=kfold)
# print("Cross-validation scores for Logistic reg:\n{}".format(corss_val_log))
# print("Mean cross validation for Logistic reg {:.3f}".format(np.mean(corss_val_log)))

#Training and test
logreg.fit(X_train_scaled, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}\n".format(logreg.score(X_test_scaled, y_test)))
y_pred = logreg.predict(X_test_scaled)

# #Classification report
print(classification_report(y_test, y_pred, target_names=lb.classes_))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')


#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

knn = KNeighborsClassifier()

#Validation
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


#validation and grid search
        
param_grid = {'n_neighbors': [4, 5, 6, 7],
              'leaf_size': [30, 35, 10]}

grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs =-1) #incorporates Cross validation, , n_jobs =-1 uses all PC cores
grid_search.fit(X_valid_scaled, y_valid)
print("Best parameters: {}".format(grid_search.best_params_))
print("Mean cross-validated score of the best_estimator: {:.2f}".format(grid_search.best_score_))
best_parameters_knn = grid_search.best_params_
            

knn_model = KNeighborsClassifier(**best_parameters_knn, weights = 'distance')

# corss_val_knn = cross_val_score(knn, X_valid_scaled, y_valid, cv=kfold)
# print("Cross-validation scores for Logistic reg:\n{}".format(corss_val_knn))
# print("Mean cross validation for Logistic reg {:.3f}".format(np.mean(corss_val_knn)))

#Training and test
knn_model.fit(X_train_scaled, y_train)
print("Training set score: {:.3f}".format(knn_model.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}\n".format(knn_model.score(X_test_scaled, y_test)))
y_pred = knn_model.predict(X_test_scaled)

#Classification report
print(classification_report(y_test, y_pred, target_names=lb.classes_))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

# Best parameters: {'leaf_size': 30, 'n_neighbors': 6}
# Mean cross-validated score of the best_estimator: 0.66
# Training set score: 1.000
# Test set score: 0.693

#                      precision    recall  f1-score   support

#          Fast ferry       0.86      0.46      0.60        39
#        Fishing ship       0.42      0.13      0.20       147
#  General cargo ship       0.69      0.80      0.74      2959
# Oil products tanker       0.47      0.34      0.40       874
#          Other ship       0.59      0.38      0.46      1002
#      Passenger ship       0.72      0.69      0.71      1409
#       Pleasure boat       0.76      0.91      0.83      2659
#        Support ship       0.58      0.36      0.45       510

#            accuracy                           0.69      9599
#           macro avg       0.64      0.51      0.55      9599
#        weighted avg       0.68      0.69      0.68      9599

# Precision is positive predictive value 
# Recall is true positive rate or sensitivity 
