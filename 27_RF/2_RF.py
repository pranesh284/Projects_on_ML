# -*- coding: utf-8 -*-
"""
Wed May  9 18:43:30 2018: Dhiraj
"""
#https://github.com/alexhwoods/Machine-Learning/tree/master/Random%20Forest
# RandomForests
# First let's import the dataset, using Pandas.
import pandas as pd
import numpy as np

train = pd.read_csv("train-iris.csv")    # make sure you're in the right directory if using iPython!
test = pd.read_csv("test-iris.csv") 

train.head()             # ignore the first column, it's how I split the data.
train.describe(include='all')
train.columns
train.petal_length.describe()   #class not working because of keyword
train.describe(include=[np.object])
#train.describe(include=['category'])  #nothing so far
train.describe(exclude=[np.number])
train['class'].value_counts()
test['class'].value_counts()

from sklearn.ensemble import RandomForestClassifier


#%%
# however, are data has to be in a numpy array in order for the random forest algorithm to except it!
cols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
colsRes = ['class']

trainArr = train[['petal_length', 'petal_width', 'sepal_length', 'sepal_width']].values
#trainArr = train.as_matrix(cols)    # training array
trainArr

trainRes=train[['class']].values
#trainRes = train.as_matrix(colsRes) # training results
trainRes
## Training!

#Estimators -- How many different trees for finding optimized one

rf = RandomForestClassifier(n_estimators=100)    # 100 decision trees is a good enough number
rf.fit(trainArr, trainRes)          # finally, we fit the data to the algorithm!!! :)

# note - you might get an warning saying you entered a 2 column vector..ignore it.

#%%
## Testing!

# put the test results in the same format!
#testArr = test.as_matrix(cols)
testArr=trainArr

y_pred = rf.predict(testArr)


from sklearn.metrics import classification_report
print(classification_report(trainRes, y_pred))
