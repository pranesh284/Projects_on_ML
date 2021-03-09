# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:01:44 2020

@author: vikas
"""

import numpy as np
import pandas as pd


train = pd.read_csv("train-iris.csv")    # make sure you're in the right directory if using iPython!
test = pd.read_csv("test-iris.csv") 

train.head()             # ignore the first column, it's how I split the data.
train
#train.describe(include='all')
train.columns
train.describe()   #class not working because of keyword
#train.describe(include=[np.object])
#train.describe(include=['category'])  #nothing so far
#train.describe(exclude=[np.number])
train['class'].value_counts()
test['class'].value_counts()

from sklearn.ensemble import RandomForestClassifier

cols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
colsRes = ['class']

x_train = train[cols].values
y_train = train[colsRes].values
y_train


rf = RandomForestClassifier(n_estimators = 100)

rf.fit(x_train, y_train)


x_test = test[cols].values
y_test = test[colsRes].values

y_pred = rf.predict(x_test)

y_pred

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test, y_pred)
acc

pre = precision_score(y_test, y_pred, average='micro')
pre

rec = recall_score(y_test, y_pred, average='micro')
rec

f1 = f1_score(y_test, y_pred, average='micro')

f1











#Logistic Regression Testing


import numpy as np
import pandas as pd


train = pd.read_csv("train-iris.csv")    # make sure you're in the right directory if using iPython!
test = pd.read_csv("test-iris.csv") 

train.head()             # ignore the first column, it's how I split the data.
train
#train.describe(include='all')
train.columns
train.describe()   #class not working because of keyword
#train.describe(include=[np.object])
#train.describe(include=['category'])  #nothing so far
#train.describe(exclude=[np.number])
train['class'].value_counts()
test['class'].value_counts()

from sklearn.linear_model import LogisticRegression

cols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
colsRes = ['class']

x_train = train[cols].values
y_train = train[colsRes].values
y_train


rf = LogisticRegression()

rf.fit(x_train, y_train)


x_test = test[cols].values
y_test = test[colsRes].values

y_pred = rf.predict(x_test)

y_pred

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test, y_pred)
acc

pre = precision_score(y_test, y_pred, average='micro')
pre

rec = recall_score(y_test, y_pred, average='micro')
rec

f1 = f1_score(y_test, y_pred, average='micro')

f1



