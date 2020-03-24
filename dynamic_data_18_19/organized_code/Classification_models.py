# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:14:51 2020

@author: JULSP
"""

import pandas as pd
import sklearn 
import numpy as np

from os.path import dirname

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import mglearn

#%%
"""
0.
Loading Data
& Pre-Processing
"""

path = dirname(__file__)
data_sept = pd.read_csv(path + "/dynamic_data_sept.csv") 
data_oct = pd.read_csv(path + "/dynamic_data_oct.csv")



data_all = data_sept.append(data_oct, sort = False)

data_clean = data_all.drop(['trips'], axis=1)
#data_clean = data_clean[data_clean.iwrap_type_from_dataset != 'Other ship']
data_clean = data_clean.dropna() #drop nan values, our models can't handle this
data_processed = data_clean[data_clean.length_from_data_set < 400] #vessels that are over 400m do not exist
data_processed = data_processed[data_processed.Speed_max <65] #boats that go faster than that are helicopters
data_processed = data_processed[data_processed.Speed_max >0] #boats that stand still

data_processed = data_processed.reset_index(drop =True) #reset index

types_of_vessels = data_processed.iwrap_type_from_dataset.value_counts()
print(types_of_vessels)

#%%
"""
0.
Limiting each amount of vessels 
"""

amount_limit = 1000

df1 = data_processed[data_processed['iwrap_type_from_dataset'] == 'General cargo ship'][:amount_limit]
df2 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Oil products tanker'][:amount_limit]
df3 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Passenger ship'][:amount_limit]
df4 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Pleasure boat'][:amount_limit]
df5 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Other ship'][:amount_limit]
df6 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Fishing ship'][:amount_limit]
df7 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Support ship'][:amount_limit]
df8 = data_processed[data_processed['iwrap_type_from_dataset'] == 'Fast ferry'][:amount_limit]

data_processed = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8]) #put it all together
data_processed = data_processed.sample(frac=1).reset_index(drop=True) #shuffle
data_processed = data_processed.reset_index(drop =True) #reset index

#%%
"""
0.
Ecoding the labels
Picking X and y variables
Splitting train/test/validation
Scaling
"""

#Ecoding the labels
lb = LabelEncoder()
data_processed['iwrap_cat'] = lb.fit_transform(data_processed['iwrap_type_from_dataset'])
#picking X and y and splitting
X = data_processed[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
y = data_processed[['iwrap_cat']]
y = y.values.ravel() #somethe model wants this to be an array


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
print(np.unique(y_trainval))
print(np.unique(y_test))
print(np.unique(y_train))
print(np.unique(y_valid))
#%%
"""
0.
Retrieving vessel names for Classification report
LabelEncoder splits them up differently each time 
There is probably an automatic way of doing this
"""
class_0 = data_processed.loc[data_processed['iwrap_cat'] == 0, 'iwrap_type_from_dataset'].iloc[0]
class_1 = data_processed.loc[data_processed['iwrap_cat'] == 1, 'iwrap_type_from_dataset'].iloc[0]
class_2 = data_processed.loc[data_processed['iwrap_cat'] == 2, 'iwrap_type_from_dataset'].iloc[0]
class_3 = data_processed.loc[data_processed['iwrap_cat'] == 3, 'iwrap_type_from_dataset'].iloc[0]
class_4 = data_processed.loc[data_processed['iwrap_cat'] == 4, 'iwrap_type_from_dataset'].iloc[0]
class_5 = data_processed.loc[data_processed['iwrap_cat'] == 5, 'iwrap_type_from_dataset'].iloc[0]
class_6 = data_processed.loc[data_processed['iwrap_cat'] == 6, 'iwrap_type_from_dataset'].iloc[0]
class_7 = data_processed.loc[data_processed['iwrap_cat'] == 7, 'iwrap_type_from_dataset'].iloc[0]

target_names = [class_0, class_1, class_2, class_3, class_4, class_5, class_6, class_7]

#%%
"""
1.
Logistic Regression 
"""
logreg = LogisticRegression(max_iter=10000)
#Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
corss_val_log = cross_val_score(logreg, X_valid_scaled, y_valid, cv=kfold)
print("Cross-validation scores for Logistic reg:\n{}".format(corss_val_log))
print("Mean cross validation for Logistic reg {:.3f}".format(np.mean(corss_val_log)))

#Training and test
logreg.fit(X_train_scaled, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}\n".format(logreg.score(X_test_scaled, y_test)))
y_pred = logreg.predict(X_test_scaled)

#Classification report
print(classification_report(y_test, y_pred, target_names=target_names))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

# RESULTS:
## sept + oct, with limit of 1000 vessels per category 
# Cross-validation scores for Logistic reg:
# [0.36538462 0.41312741 0.3976834  0.32432432 0.36293436]
# Mean cross validation for Logistic reg 0.373
# Training set score: 0.393
# Test set score: 0.406

#                      precision    recall  f1-score   support

#          Fast ferry       0.67      0.10      0.17        21
#        Fishing ship       0.69      0.15      0.24        75
#  General cargo ship       0.37      0.56      0.44       200
# Oil products tanker       0.32      0.27      0.29       200
#          Other ship       0.39      0.29      0.34       200
#      Passenger ship       0.49      0.41      0.45       200
#       Pleasure boat       0.45      0.73      0.56       200
#        Support ship       0.37      0.30      0.33       200

#            accuracy                           0.41      1296
#           macro avg       0.47      0.35      0.35      1296
#        weighted avg       0.42      0.41      0.39      1296


#WITHOUT THE 1000 vessel limit
# Cross-validation scores for Logistic reg:
# [0.54289373 0.5262484  0.51856594 0.52752881 0.52240717]
# Mean cross validation for Logistic reg 0.528
# Training set score: 0.524
# Test set score: 0.536

#                      precision    recall  f1-score   support

#          Fast ferry       0.50      0.05      0.09        21
#        Fishing ship       0.00      0.00      0.00        75
#  General cargo ship       0.55      0.92      0.69      1356
# Oil products tanker       0.00      0.00      0.00       420
#          Other ship       0.46      0.29      0.36       494
#      Passenger ship       0.42      0.26      0.32       610
#       Pleasure boat       0.58      0.79      0.67       695
#        Support ship       0.14      0.00      0.01       234

#            accuracy                           0.54      3905
#           macro avg       0.33      0.29      0.27      3905
#        weighted avg       0.43      0.54      0.45      3905

#%%
"""
1.
Probabilistic results for test dataset
"""
prob_test_scores_logreg = logreg.predict_proba(X_test)

print("Predicted probabilities:\n{:.2f}%".format, prob_test_scores_logreg[:6])
print("Sums: {}".format(logreg.predict_proba(X_test)[:6].sum(axis=1)))
#%%
"""
2.
SVM 
TODO look up kernel trick
"""

best_score = 0
#validation
for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # for each combination of parameters, train an SVC
        svm = SVC(gamma=gamma, C=C, random_state = 0)
        svm.fit(X_valid_scaled, y_valid)
        # evaluate the SVC on the test set
        score = svm.score(X_valid, y_valid)
        # if we got a better score, store the score and parameters
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
            
print("Mean cross-validated score of the best_estimator: {:.2f}".format(best_score))
print("Best parameters: ", best_parameters)

#%%
"""
2.
SVM
"""
# rebuild a model on best parameters
svm_best = SVC(**best_parameters, random_state = 0)

#Training and test - is it useless?
svm_best.fit(X_train_scaled, y_train)
print("Training set score: {:.3f}".format(svm_best.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}\n".format(svm_best.score(X_test_scaled, y_test)))

#Classification report
print(classification_report(y_test, svm_best.predict(X_test_scaled), target_names=target_names))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

#RESULTS WITH 1000 limit
# Training set score: 0.419
# Test set score: 0.413

#                      precision    recall  f1-score   support

#          Fast ferry       0.00      0.00      0.00        21
#        Fishing ship       0.00      0.00      0.00        75
#  General cargo ship       0.37      0.65      0.47       200
# Oil products tanker       0.58      0.21      0.31       200
#          Other ship       0.44      0.26      0.33       200
#      Passenger ship       0.45      0.49      0.47       200
#       Pleasure boat       0.39      0.79      0.52       200
#        Support ship       0.41      0.28      0.33       200

#            accuracy                           0.41      1296
#           macro avg       0.33      0.33      0.30      1296
#        weighted avg       0.41      0.41      0.38      1296
#%%
"""
3.
Random Forest Classifier
Validation
"""
from sklearn.ensemble import RandomForestClassifier

forest_class = RandomForestClassifier(random_state = 42)

#parameteres for RF
param_grid = {'n_estimators': [ 70, 80, 90],
              'max_depth': [15, 20, 35],
              'max_features': [2, 3, 5],
              'ccp_alpha': [2, 3, 5]}

grid_search = GridSearchCV(forest_class, param_grid, cv=5, n_jobs =-1)
grid_search.fit(X_valid_scaled, y_valid)
print("Best parameters: {}".format(grid_search.best_params_))
best_parameters = grid_search.best_params_
print("Mean cross-validated score of the best_estimator: {:.2f}".format(grid_search.best_score_))

#%%
"""
3. 
Random Forest
Classification report
"""

forest_class_best = RandomForestClassifier(**best_parameters, random_state = 42)

#Train/test
forest_class_best.fit(X_train_scaled, y_train)
print("Random Forest Training set score: {:.2f}".format(forest_class_best.score(X_train_scaled, y_train)))
print("Random Forest Test set score: {:.2f}\n".format(forest_class_best.score(X_test_scaled, y_test)))

#Classification report
print(classification_report(y_test, forest_class_best.predict(X_test_scaled), target_names=target_names))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

#RESULTS with 1000 limit
# Random Forest Training set score: 0.15
# Random Forest Test set score: 0.15

#                      precision    recall  f1-score   support

#          Fast ferry       0.00      0.00      0.00        21
#        Fishing ship       0.00      0.00      0.00        75
#  General cargo ship       0.00      0.00      0.00       200
# Oil products tanker       0.15      1.00      0.27       200
#          Other ship       0.00      0.00      0.00       200
#      Passenger ship       0.00      0.00      0.00       200
#       Pleasure boat       0.00      0.00      0.00       200
#        Support ship       0.00      0.00      0.00       200

#            accuracy                           0.15      1296
#           macro avg       0.02      0.12      0.03      1296
#        weighted avg       0.02      0.15      0.04      1296
#%%
"""
4.
NN
"""

from sklearn.neural_network import MLPClassifier
nn_class = MLPClassifier(random_state =0, max_iter = 20000)

"""
Neural Network
Grid Search - for Neural Network
!Takes a long time
"""


param_grid = {'hidden_layer_sizes': [(100,50,10), (100,100), (100,50)],
              'alpha': [0.1, 0.05, 0.01],}

grid_search = GridSearchCV(nn_class, param_grid, cv=5, n_jobs =-1) #incorporates Cross validation, , n_jobs =-1 uses all PC cores
grid_search.fit(X_valid_scaled, y_valid)
print("Best parameters: {}".format(grid_search.best_params_))
print("Mean cross-validated score of the best_estimator: {:.2f}".format(grid_search.best_score_))
best_parameters3 = grid_search.best_params_

# Best parameters: {'alpha': 0.1, 'hidden_layer_sizes': (100, 50, 10)}
# Mean cross-validated score of the best_estimator: 0.64
#%%

nn_class = MLPClassifier(random_state =0, max_iter = 20000, hidden_layer_sizes = (100,100), alpha = 0.1)


#Training and test 
#NEEDS more data overfits
nn_class.fit(X_train_scaled, y_train)
print("Training set score: {:.3f}".format(nn_class.score(X_train_scaled, y_train)))
print("Test set score: {:.3f}\n".format(nn_class.score(X_test_scaled, y_test)))

#Classification report
print(classification_report(y_test, nn_class.predict(X_test_scaled), target_names=target_names))
print('Precision is positive predictive value \nRecall is true positive rate or sensitivity \n')

# RESULTS: with 1000 limiting each class
# Training set score: 0.563
# Test set score: 0.532

#                      precision    recall  f1-score   support

#          Fast ferry       0.67      0.19      0.30        21
#        Fishing ship       0.51      0.31      0.38        75
#  General cargo ship       0.55      0.47      0.51       200
# Oil products tanker       0.55      0.69      0.61       200
#          Other ship       0.49      0.30      0.37       200
#      Passenger ship       0.58      0.53      0.55       200
#       Pleasure boat       0.54      0.77      0.64       200
#        Support ship       0.47      0.56      0.51       200

#            accuracy                           0.53      1296
#           macro avg       0.55      0.48      0.48      1296
#        weighted avg       0.53      0.53      0.52      1296

# Precision is positive predictive value 
# Recall is true positive rate or sensitivity 


# RESULTS without the limit wit architecture 100,50
# Training set score: 0.708
# Test set score: 0.670

#                      precision    recall  f1-score   support

#          Fast ferry       0.83      0.48      0.61        21
#        Fishing ship       0.46      0.23      0.30        75
#  General cargo ship       0.65      0.88      0.75      1356
# Oil products tanker       0.60      0.21      0.31       420
#          Other ship       0.61      0.48      0.53       494
#      Passenger ship       0.79      0.60      0.68       610
#       Pleasure boat       0.70      0.90      0.79       695
#        Support ship       0.62      0.35      0.45       234

#            accuracy                           0.67      3905
#           macro avg       0.66      0.52      0.55      3905
#        weighted avg       0.67      0.67      0.64      3905

# Precision is positive predictive value 
# Recall is true positive rate or sensitivity 
