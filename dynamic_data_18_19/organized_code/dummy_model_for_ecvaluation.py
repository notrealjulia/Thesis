# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:33:38 2020

@author: JULSP
"""

import pandas as pd
import numpy as np
from os.path import dirname

from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from sklearn.dummy import DummyClassifier

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

data_processed = shuffle(data_processed)

data_processed = data_processed.reset_index(drop =True) #reset index

types_of_vessels = data_processed.iwrap_type_from_dataset.value_counts()
print(types_of_vessels)

#%%
"""
0.
Ecoding the labels
Picking X and y variables
Splitting train/test/validation
Scaling
"""

#Ecoding the labels with one-hot-encoder
# from sklearn import preprocessing
# lb = preprocessing.LabelBinarizer()
# labels = lb.fit_transform(data_processed['iwrap_type_from_dataset'])

#Ecoding the labels
lb = LabelEncoder()
data_processed['iwrap_cat'] = lb.fit_transform(data_processed['iwrap_type_from_dataset'])
#picking X and y and splitting
X = data_processed[['Speed_mean', 'Speed_median', 'Speed_min', 'Speed_max', 'Speed_std', 'ROT_mean', 'ROT_median', 'ROT_min', 'ROT_max', 'ROT_std']]
y = data_processed[['iwrap_cat']]
y = y.values.ravel() #somethe model wants this to be an array

# y = lb.inverse_transform(labels)

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

print(lb.classes_)


#%%


dummy_clf = DummyClassifier(strategy="stratified")
dummy_clf.fit(X_train_scaled, y_train)

y_pred = dummy_clf.predict(X_test_scaled)

# print(dummy_clf.score(X,y))

print(classification_report(y_test, y_pred, target_names=lb.classes_))


#%%

# strategies = ['most_frequent', 'stratified', 'uniform', 'constant'] 
  
# test_scores = [] 
# for s in strategies: 
#     if s =='constant': 
#         dclf = DummyClassifier(strategy = s, random_state = 0, constant ='M') 
#     else: 
#         dclf = DummyClassifier(strategy = s, random_state = 0) 
#     dclf.fit(X_train_scaled, y_train) 
#     score = dclf.score(X_test_scaled, y_test) 
#     test_scores.append(score)
    
# #%%
    
# import matplotlib.pyplot as plt  
# import seaborn as sns    
    
# ax = sns.stripplot(strategies, test_scores); 
# ax.set(xlabel ='Strategy', ylabel ='Test Score') 
# plt.show() 