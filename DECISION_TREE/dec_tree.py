
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/DUanalytics/pyAnalytics/master/data/bill_authentication.csv')
data.head()
data.shape
data.columns
data
#data preparation : X & Y
X= data.drop('Class', axis=1) #axis=1 -> column
y= data['Class']
X
y
y.value_counts()



#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.10)
X_train.shape
X_test.shape


#model
from sklearn.tree import DecisionTreeClassifier

clsModel = DecisionTreeClassifier()  #model with parameter

clsModel.fit(X_train, y_train)

#predict
ypred1 = clsModel.predict(X_test)
ypred1



#metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics

cr=classification_report(y_test, ypred1)

cr




confusion_matrix(y_true=y_test, y_pred=ypred1)
accuracy_score(y_true=y_test, y_pred=ypred1)




print("Accuracy:",metrics.accuracy_score(y_test, ypred1))
print("Precision:",metrics.precision_score(y_test, ypred1))
print("Recall:",metrics.recall_score(y_test, ypred1))




y_pred_proba = clsModel.predict_proba(X_test)[::,1]
y_pred_proba
y_pred_proba.shape


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

fpr
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc



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
ypred2 = LR.predict(X_test)
ypred2



#metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

cr=classification_report(y_test, ypred2)

cr

confusion_matrix(y_true=y_test, y_pred=ypred2)
accuracy_score(y_true=y_test, y_pred=ypred2)




print("Accuracy:",metrics.accuracy_score(y_test, ypred2))
print("Precision:",metrics.precision_score(y_test, ypred2))
print("Recall:",metrics.recall_score(y_test, ypred2))




y_pred_proba = clsModel.predict_proba(X_test)[::,1]
y_pred_proba
y_pred_proba.shape


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

fpr
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc



plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show();






from sklearn import tree
tree.plot_tree(decision_tree=clsModel, fontsize=6)

tree.plot_tree(decision_tree=clsModel, max_depth=2, feature_names=['Var', 'Skew', ' Kur',  'Ent'], class_names=['Orgiginal','Fake'], fontsize=8)
tree.plot_tree(decision_tree=clsModel, max_depth=3, feature_names=['Var', 'Skew', ' Kur',  'Ent'], class_names=['Orgiginal','Fake'], fontsize=8)
