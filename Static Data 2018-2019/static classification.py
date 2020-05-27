# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:00:31 2020

@author: KORAL
"""

from os.path import dirname
import pandas as pd
import numpy as np
import glob
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import plot_confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

#%%
path = dirname(__file__)
frame = pd.read_csv(path + "\static_ship_data.csv", sep='	', index_col=None, error_bad_lines=False)

#%%
"""

PRE-PROCESSING


•	Removing irrelevant columns (where 100% of values are missing)
•	Renaming columns for convenience
•	Separating 'Undefined' category from 'Other' category in iwrap column using mask and org_type column

"""

mask = (frame['org_type_info_from_data'] == 'Undefined') & (frame['iwrap_type_from_dataset'] == 'Other ship')
frame['iwrap_type_from_dataset'][mask] = 'Undefined'

frame = frame[['mmsi', 'imo_from_data_set', 'name_from_data_set', 'iwrap_type_from_dataset', 'org_type_info_from_data', 'length_from_data_set', 'width', 'size_a','size_b','size_c','size_d']]
frame = frame.rename(columns={"imo_from_data_set":"imo", "name_from_data_set": "name", "iwrap_type_from_dataset": "iwrapType", "org_type_info_from_data": "orgType", "length_from_data_set": "length", "iwrap_type_from_dataset": "iwrapType"})


"""

fixed outliers:
    257073000 LIVITA Fishipng ship - 'General Cargo Ship'
    636015743 ARCHIMIDIS Other ship - 'General Cargo Ship'     
    241289000 MARAN GAS SPARTA Other ship - 'Oil products tanker'
    241354000 NISSOS THERASSIA Other ship - 'Oil products tanker'
    477962200 GOLDEN ENDEAVOUR Other ship - 'General cargo ship'
    253403000 LEIV EIRIKSSON Other ship - 'Support ship' 
    1193046 - remove
"""    


frame.loc[(frame['mmsi'] == 257073000)] = frame.loc[(frame['mmsi'] == 257073000)].replace(to_replace ="Fishing ship",  
                            value ="General cargo ship") 

frame.loc[(frame['mmsi'] == 636015743)] = frame.loc[(frame['mmsi'] == 636015743)].replace(to_replace ="Other ship",  
                            value ="General cargo ship") 

frame.loc[(frame['mmsi'] == 241289000)] = frame.loc[(frame['mmsi'] == 241289000)].replace(to_replace ="Other ship",  
                            value ="Oil products tanker") 

frame.loc[(frame['mmsi'] == 241354000)] = frame.loc[(frame['mmsi'] == 241354000)].replace(to_replace ="Other ship",  
                            value ="Oil products tanker") 

frame.loc[(frame['mmsi'] == 477962200)] = frame.loc[(frame['mmsi'] == 477962200)].replace(to_replace ="Other ship",  
                            value ="General cargo ship") 

frame.loc[(frame['mmsi'] == 253403000)] = frame.loc[(frame['mmsi'] == 253403000)].replace(to_replace ="Other ship",  
                            value ="Support ship") 


frame = frame[frame.mmsi != 1193046]

"""
Preparing Dataframe for training ML models:
    Removing entries with missing size
    Removing entries with missing type
    Sanity check: removing entries over 400m
"""

frameCopy = frame.loc[(frame['length']  != -1) | (frame['width'] != -1)]
frameCopy = frameCopy.loc[(frameCopy['iwrapType'] != 'Undefined')]
frameCopy = frameCopy.loc[(frameCopy['length'] <= 400)]

#%%
"""
 ---CLASSIFICATION TRAINING---

    PREDICTION:
        Predicting IWRAP type using Length x Width as variables
        
        1) Defining X and Y    
        2) Splitting data into train/test
        3) SMOTE (Synthetic Minority Over-sampling Technique)
        4) Fitting the model
        5) Results 


"""


"""


    PREPARATION, SPLITTING 
    
"""


y = frameCopy[['iwrapType']] # Labels
X = frameCopy[['length' , 'width']] #Features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0) #Splitting


    # Uncomment for balanced classification:
    
#from imblearn.over_sampling import SMOTE
#smote = SMOTE('all')
#X_train, y_train = smote.fit_sample(np.array(X_train), np.array(y_train))



"""

    CLASSIFICATION EVALUATION FUNCTION (input - fitted model)
    
    
"""


def classifier_evaluation(model):
    class_names = ['Cargo',
                   'Fast ferry',
                   'Fishing',
                   'Other',
                   'Passenger',
                   'Pleasure',
                   'Support',
                   'Tanker']
    # making predictions:
    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    cross_val_predictions = cross_val_predict(model, X, y, cv=5)
    
    # plot confusion matrix:
    confusion = metrics.confusion_matrix(y_test, test_pred, labels = class_names)

    disp = plot_confusion_matrix(model, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues, xticks_rotation = 'vertical', normalize = 'true', values_format='.2f')
    disp.ax_.set_title('Imbalanced Classification')
    
    # print scores:
    print("Test Accuracy:",metrics.accuracy_score(y_test, test_pred))
    print("Train Accuracy:",metrics.accuracy_score(y_train, train_pred))
    print("Average Cross Validation Score:", np.mean(cross_val_score(model, X, y, cv=5)))
    print('CLASSIFICATION REPORT: \n', classification_report(y_test, test_pred))  #, target_names=lb.classes_))
    plt.savefig((str(model)[:10]+'_imbal.png'), dpi=1000, format='png', bbox_inches='tight')
    plt.show()
    return (confusion)



#%%
"""

    DECISION TREE CLASSIFIER
    
"""

DTC = DecisionTreeClassifier()
#DTC.fit(X_train, y_train) # without oversampling 
DTC.fit(X_train, y_train)
print("DECISION TREE")
DTC_evaluation = classifier_evaluation(DTC)

#%%

"""

    RANDOM FOREST CLASSIFIER
    
"""

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=100)
RFC.fit(X_train,y_train)
print("RANDOM FOREST CLASSIFIER")
RFC_evaluation = classifier_evaluation(RFC)


#%%

"""
    K-Nearest Neighbors (KNN)
    
"""

from sklearn.neighbors import KNeighborsClassifier 

#   THIS FUNCTION FINDS THE BEST K (NUMBER OF NEIGBORS) FOR KNN ALGORITHM

from operator import itemgetter
def best_k():
    accuracies =  []
    neighbors = list(range(1,10))
    for i in neighbors:
        knn = KNeighborsClassifier(n_neighbors = i).fit(X_train, y_train) 
        accuracies.append((i, knn.score(X_test, y_test)))
    return max(accuracies,key=itemgetter(1))[0]

KNC = KNeighborsClassifier(n_neighbors = best_k())
KNC.fit(X_train, y_train)
print("K-N CLASSIFIER")
KNC_evaluation = classifier_evaluation(KNC)

