
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python
#data
url='https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/pima-indians-diabetes.csv'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv(url, header=None, names=col_names)

pima.head()

#%%%Selecting Feature


feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']

pima.columns

X = pima[feature_cols] # Features
X
y = pima.label # Target variable
y

#%%%Splitting Data
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

#%%%

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)
y_pred
y_test

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

metrics.accuracy_score(y_test, y_pred)


y_test = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 1])

y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

print(classification_report(y_test, y_pred))


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



chk= [[1,68,63.1,33,137,40,2.288], [0,18,23.1,13,95,10,1.288]]
y_val= logreg.predict(chk)
y_val


#%%%Model Evaluation using Confusion Matrix

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
(119 + 36)/(119 + 36 + 26 + 11)
#Here, you can see the confusion matrix in the form of the array object. The dimension of this matrix is 2*2 because this model is binary classification. You have two classes 0 and 1. Diagonal values represent accurate predictions, while non-diagonal elements are inaccurate predictions. In the output, 119 and 36 are actual predictions, and 26 and 11 are incorrect predictions.



#%%Visualizing Confusion Matrix using Heatmap

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline # for Jupiter

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show();


#%%%Confusion Matrix Evaluation Metrics


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))



#%%ROC Curve


y_pred_proba = logreg.predict_proba(X_test)[::,1]
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
auc
#AUC score for the case is 0.86. AUC score 1 represents perfect classifier, and 0.5 represents a worthless classifier.




