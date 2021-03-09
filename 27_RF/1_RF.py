


import numpy as np
import pandas as pd

train = pd.read_csv("train-iris.csv")    # make sure you're in the right directory if using iPython!
test = pd.read_csv("test-iris.csv")

train.head()             # ignore the first column, it's how I split the data.
train

train.columns
train.describe()   #class not working because of keyword

train['class'].value_counts()
test['class'].value_counts()

from sklearn.ensemble import RandomForestClassifier


cols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
colsRes = ['class']


trainArr = train[cols].values    # training array
#trainArr = train.as_matrix(cols)    # training array
trainArr

trainRes = train[colsRes].values
(colsRes) # training results
trainRes
## Training!

rf = RandomForestClassifier(n_estimators=100)    # 100 decision trees is a good enough number

rf.fit(trainArr, trainRes)          # finally, we fit the data to the algorithm!!! :)


y_pred = rf.predict(trainArr)

y_pred

from sklearn.metrics import classification_report

cr=classification_report(y_pred, trainRes)

cr

from sklearn.tree import DecisionTreeClassifier

clsModel = DecisionTreeClassifier()  #model with parameter

clsModel.fit(trainArr, trainRes)

#predict
y_pred = clsModel.predict(trainArr)
y_pred

from sklearn.metrics import classification_report

cr=classification_report(ypred1, trainRes)

cr




testArr = test[cols].values
y_pred = rf.predict(testArr)

y_test = test['class'].values

# add it back to the dataframe, so I can compare side-by-side
test['predictions'] = results
test




print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average = 'weighted'))
print("Recall:",metrics.recall_score(y_test, y_pred, average = 'weighted'))

