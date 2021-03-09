#Topic: DT - Diabetis Data Set
#-----------------------------
#https://www.datacamp.com/community/tutorials/decision-tree-classification-python
#As a marketing manager, you want a set of customers who are most likely to purchase your product. This is how you can save your marketing budget by finding your audience. As a loan manager, you need to identify risky loan applications to achieve a lower loan default rate. This process of classifying customers into a group of potential and non-potential customers or safe or risky loan applications is known as a classification problem. Classification is a two-step process, learning step and prediction step. In the learning step, the model is developed based on given training data. In the prediction step, the model is used to predict the response for given data. Decision Tree is one of the easiest and popular classification algorithms to understand and interpret. It can be utilized for both classification and regression kind of problem.
# Load libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
from graphviz import Source
from IPython.display import Image  
import pydotplus
from IPython.display import SVG
from sklearn.ensemble import RandomForestClassifier


#%%%% : Load Data
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
url='https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/pima-indians-diabetes.csv'
pima = pd.read_csv(url, header=None, names=col_names)

pima.label.value_counts()
pima.shape


feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols] # Features - bmi, age etc
y = pima.label # Target variable : has diabetes =1
X.head(5)
y.head(5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1) 
X_test.shape

clf=RandomForestClassifier(n_estimators=100)

#clf=DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

y_train

y_pred = clf.predict(X_test)

y_pred

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)
y_pred

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))





from sklearn.tree import DecisionTreeClassifier
# instantiate the model (using the default parameters)
logreg = DecisionTreeClassifier()
# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)
y_pred

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))






X_val=[[5,112,26.2,30,121,72,0.245]]
X_val
y_val = clf.predict(X_val)
y_val