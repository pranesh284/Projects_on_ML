

# Load libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
from graphviz import Source
from IPython.display import Image, SVG
import pydotplus
import matplotlib.pyplot as plt

#%%%%
url='https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/pima-indians-diabetes.csv'

#https://www.kaggle.com/uciml/pima-indians-diabetes-database

# load dataset
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header=None, names=col_names)
pima.head()
pima.label.value_counts() #how many are diabetic - 268
pima.shape

#%%% : Feature Selection

feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols] # Features - bmi, age etc
y = pima.label # Target variable : has diabetes =1

#%%% Splitting Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

X_train.head()

#%%%
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
y_train
#Predict the response for test dataset
y_pred = clf.predict(X_test)
y_pred
#%%% :
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))





y_pred_proba = clf.predict_proba(X_test)[::,1]
y_pred_proba
y_pred_proba.shape


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

fpr
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

#together

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show();







from sklearn.linear_model import LogisticRegression

LR =LogisticRegression()

LR.fit(X_train, y_train)

#predict
y_pred = LR.predict(X_test)
y_pred







print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))



y_pred_proba = LR.predict_proba(X_test)[::,1]
y_pred_proba
y_pred_proba.shape


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

fpr
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

#together

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show();





from sklearn.ensemble import RandomForestClassifier

RF=RandomForestClassifier()

RF.fit(X_train, y_train)

#predict
ypred3 = RF.predict(X_test)
ypred3

print("Accuracy:",metrics.accuracy_score(y_test, ypred3))







#%%%
y_test.shape, y_pred.shape
y_test.head()
y_pred[0:6]




clf3 = DecisionTreeClassifier(max_depth=3)
# Train Decision Tree Classifer
clf3 = clf3.fit(X_train,y_train)
#Visualise

from graphviz import Source
from sklearn import tree
tree.plot_tree(decision_tree=clf3, fontsize=8)


#display(SVG(graph3b.pipe(format='svg')))
X_train[0:1]
#Class:1 : glucose > 127, glucose < 158, bmi, age,
#Predict the response for test dataset
y_pred3 = clf3.predict(X_test)
len(X_test)
y_pred3
len(y_pred3)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred3))
#classification rate increased to 77.05%, which is better accuracy than the previous model.

clf4 = DecisionTreeClassifier(criterion="gini", max_depth=3)
# Train Decision Tree Classifer
clf4 = clf4.fit(X_train,y_train)
y_pred4 = clf4.predict(X_test)
tree.plot_tree(decision_tree= clf4, fontsize=8)

fig = plt.figure(figsize=(10,8))
_ = tree.plot_tree(clf4, filled=True, fontsize=7)  #see plot

print("Accuracy:",metrics.accuracy_score(y_test, y_pred4))



#%%%%
clf = DecisionTreeClassifier(max_depth=3)


clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))