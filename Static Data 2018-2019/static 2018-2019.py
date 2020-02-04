# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from os.path import dirname
import pandas as pd
import numpy as np
import glob
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

path = dirname(__file__)
frame = pd.read_csv(path + "/static_ship_data.csv", sep='	', index_col=None, error_bad_lines=False)
#frame.drop_duplicates(subset=['imo'], keep='last', inplace=True) # TO DO: Find on which identifier to drop the duplicates



#%%
    
"""

PRE-PREOCESSING

•	Removing irrelevant columns (100% missing values)
•	Renaming columns
•	Separating 'Undefined' category from 'Other' category in iwrap column using mask and org_type column
•	Finding type values

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
    iwrap and org types:
    
"""

orgTypes = frame['orgType'].value_counts(dropna=False) # number of AIS (org) types in the dataset
orgTypes_normalized = frame['orgType'].value_counts(normalize=True, dropna=False) # % of AIS (org) types in the dataset

iwrapTypes = frame['iwrapType'].value_counts(dropna=False) # number of AIS (iwrap) types in the dataset
iwrapTypes_normalized = frame['iwrapType'].value_counts(normalize=True, dropna=False) # % of AIS (iwrap) types in the dataset
#%%


"""

MISSING VALUES

Missing values are those, that are equal to  0, -1 or NaN.

•	First loop goes trough variable names and appends missing values to a list
•	Second loop goes trough missing values and finds counts/percentages relative to all data

"""


names = list(frame)

missingValues = []
missingCounts = []
missingPercentages = []

for el in names:
    missingVal=frame.loc[(frame[el]==0)|(frame[el]==-1)|(frame[el].isna())]
    missingValues.append(missingVal)

for el in missingValues:
    missingCounts.append(len(el))
    missingPercentages.append(len(el)*100/len(frame))

resultsMissingValues = pd.DataFrame({'Variable': names,
                       'Number of missing':missingCounts,
                       '% of missing': np.round_(missingPercentages, decimals = 2)})

    
iwrapUndefined  = frame.loc[(frame['iwrapType']=='Undefined')]
#%%

"""

ANALYSIS:


•	Finding/Removing outliers
•	Finding Means, STD, Variance
•	Kernel Density Estimation


"""

types = list(iwrapTypes.index.values)


frameCopy = frame.loc[(frame['length']  != -1) & (frame['width'] != -1)] ##  a copy without missing length and width properties and lengths > 400 
frameCopy = frameCopy.loc[(frameCopy['length'] <= 400)]

#%%
"""

OUTLIERS:
        
•   Visualise box plots with outliers      
•	Find z-scores, threshold = 3.
•	Outliers added to separate DF for checking
•	Scatter plots


""" 
plt.figure(figsize=(10,5))    
L = sns.boxplot(x="iwrapType", y="length",
                # hue="smoker", col="time",
                 data=frameCopy)
L.set_xticklabels(L.get_xticklabels(), rotation=45)


plt.figure(figsize=(10,5))    
W = sns.boxplot(x="iwrapType", y="width",
                # hue="smoker", col="time",
                 data=frameCopy)
W.set_xticklabels(W.get_xticklabels(), rotation=45)


outliersDF = []
for el in types: 
    selected = frameCopy.loc[(frameCopy['iwrapType'] == el)]
    selected['z'] = np.abs(stats.zscore(selected['length']))
    outliersDF.append(selected.loc[(selected['z'] > 3)])

    ax = sns.scatterplot(x="length", y="width", data=selected).set_title(el)
    plt.show()



"""

MEAN, STD 

""" 

meansAIS = frameCopy.groupby(['iwrapType'])['length', 'width'].mean()
stdAIS = frameCopy.groupby(['iwrapType'])['length', 'width'].std()





"""

KDE (Kernel Density Estimation of size by vessel type)

""" 

types = list(iwrapTypes.index.values)
cols =  []

for el in types:
    cols.append(frameCopy[frameCopy.iwrapType == el])

for col in cols:
    plt.figure(figsize=(5,5))
    ax = sns.kdeplot(col.length, col.width,
                      cmap="Blues", shade=True, shade_lowest=False)
    sns.despine()
    ax.set(xlabel='Length', ylabel='Width')
    ax.set_title(col.iwrapType.iloc[0])
    #plt.savefig('kde_{}.svg'.format(col.iwrap_type_from_dataset.iloc[0]), format='svg', dpi=1000)
    


#%%
"""

VISUALISATION

"""

## VESSEl TYPES (IWRAP)

types=frame.iwrapType.value_counts()
fig, ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = np.arange(len(types))  # the x locations for the groups
ax.barh(ind, types, width)
ax.set_yticks(ind+width/2)
ax.set_yticklabels(types.index, minor=False)
plt.title('title')
for i, v in enumerate(types):
    plt.text(v, i, " "+str(v), va='center', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.tick_params(axis='both', which='both', bottom='off',labelbottom='off')
plt.title('iWrap vessel types, \n South Baltic Sea 2018-2019', fontsize=12)

#%%

""" 

    PREDICTION (SIZE):
        Predicting Length based on width and vice versa
        
        
missingLengthAndWidth = frame.loc[(frame['length']  == -1) & (frame['width']  == -1)]
missingLength = frame.loc[(frame['length']  == -1) & (frame['width']  != -1)]
missingWidth = frame.loc[(frame['width']  == -1) & (frame['length']  != -1)]
"""
    # Evaluation function 
def evaluation(model, predicted, actual):
    errors = abs(predicted-actual)
    averageError = np.mean(abs(predicted-actual))
    mape = 100 * (errors / actual)
    accuracy = 100-np.mean(mape)
    mse = metrics.mean_squared_error(predicted, actual)
    print(model, '\nAverage Absolute Error = ', round(averageError),'m', '\nMSE = ', round(mse, 2), '\nAccuracy = ', round(accuracy, 2), '%')
    return({'averageAbsoluteError': averageError, 'mse':mse, 'accuracy':accuracy})
    

#%%

""" 

    MODEL 1. Baseline Model. Using mean size of each vessel type as predictedion.
    
"""

def baseline(df,output): # output - what we want to predict (length or width)
    df['predicted'] = np.round_(df[output].groupby(df.iwrapType).transform('mean')) 
    
    # visualiation of 10 vessels (predicted vs actual value)
    average = np.round(pd.DataFrame({'Actual': df[output], 'Predicted': df['predicted']}))
    average = average.head(10)
    average.plot(kind='bar',figsize=(10,6))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title('Actual vs Predicted {0} (10 vessels), \n Baseline Model'.format(output), fontsize=12)
    plt.show()
    
    # evalution metrics
    baseEvaluation = evaluation('Baseline ({0})'.format(output), df['predicted'], df[output])
    return (df, baseEvaluation, average)

"""
    for el in types: 
        vessels = df.loc[(df['iwrapType']  == el)]
        vesselEvaluation = evaluation(el, vessels['predicted'], vessels[output])
        vesselEvaluation = pd.DataFrame(dict)
"""        

      
baseWidth = baseline(frameCopy, 'width')
baseLength = baseline(frameCopy, 'length')


#%%

""" 

    MODEL 2. Using Linear regression for size prediction
    
"""

def LR(output,features, labels):
    X = features.values.reshape(-1,1) # features
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

lrLength = LR('Length',frameCopy.width, frameCopy.length)
lrWidth = LR('Width',frameCopy.length, frameCopy.width)


#%%

""" 
    PREDICTION:
        Predicting iWrap type using Length x Width as variables
        
        1) Preparation (binary labels, SMOTE (Synthetic Minority Over-sampling Technique))
        2) Defining X and Y    
        3) Splitting data into train/test
        4) Fitting the model
        5) Results 
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
#%%
"""
    PREPARATION, SPLITTING 
    
"""

frameCopy = frameCopy.loc[(frameCopy['iwrapType']  != 'Undefined')] 
y = frameCopy[['iwrapType']] # lABELS
X = frameCopy[['length' , 'width']] #features
X_ratio = frameCopy['length']/frameCopy['width']   #length/width as ratio
X_ratio = np.array(X_ratio).reshape(-1, 1)



lb = preprocessing.LabelBinarizer()
y =pd.DataFrame(lb.fit_transform(y['iwrapType']),
                        columns=lb.classes_)
smote = SMOTE('minority')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_sm, y_sm = smote.fit_sample(np.array(X_train), np.array(y_train))

#%%
"""
    
    DECISION TREE CLASSIFIER
    

"""

classifier = DecisionTreeClassifier()
#classifier.fit(X_train, y_train) # without oversampling 82% 
classifier.fit(X_sm, y_sm) # with oversampling 80%

y_pred_DT= classifier.predict(X_test)
#y_pred_DT = pd.DataFrame(y_pred_DT, columns=lb.classes_)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_DT))
print('DECISION TREE CLASSIFICATION REPORT: \n', classification_report(y_test, y_pred_DT,  target_names=lb.classes_))

#%%

"""

    RANDOM FOREST CLASSIFIER
    
"""

from sklearn.ensemble import RandomForestClassifier


classifier=RandomForestClassifier(n_estimators=100)
classifier.fit(X_train,y_train)

y_pred_RFC=classifier.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_RFC))
print('Random forest CLASSIFICATION REPORT: \n', classification_report(y_test, y_pred_RFC, target_names=lb.classes_))



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

knn = KNeighborsClassifier(n_neighbors = best_k()).fit(X_train, y_train)
y_pred_KNN= knn.predict(X_test)  

print(metrics.accuracy_score(y_test, y_pred_KNN))
print('knn CLASSIFICATION REPORT: \n', classification_report(y_test, y_pred_KNN,  target_names=lb.classes_))


#  L/W RATIO as feature PERFORMS WORSE THAN L and W as features
