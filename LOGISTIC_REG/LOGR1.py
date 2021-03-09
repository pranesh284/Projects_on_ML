
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
import seaborn as sns # used to create the Confusion Matrix
import matplotlib.pyplot as plt

#data from csv
url = "https://stats.idre.ucla.edu/stat/data/binary.csv"
df = pd.read_csv(url)

df
df.head()
df.dtypes

#set the independent variables (represented as X) and the dependent variable (represented as y):

X = df[['gre', 'gpa']] #array
y = df['admit']
X
y

#split data : Then, apply train_test_split.

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
X_train.shape, X_test.shape
y_train.shape, y_test.shape

#apply logistic regression
logistic_regression= LogisticRegression()

logistic_regression.fit(X_train,y_train)

y_pred=logistic_regression.predict(X_test)

y_test

y_pred

#print the Accuracy and plot the Confusion Matrix:
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred)*100)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
confusion_matrix

sns.heatmap(confusion_matrix, annot=True, cmap='coolwarm', annot_kws={'size':20}, cbar=True)
plt.show();


# TP = True Positives
# TN = True Negatives
# FP = False Positives
# FN = False Negatives




print (X_test) #test dataset

print (y_pred) #predicted values

y_test, y_pred
type(y_test), type(y_pred)


#predict on new data set
#use the existing logistic regression model to predict whether the new candidates will get admitted. The new set of data can then be captured in a second DataFrame called df2:

new_candidates = {'gmat': [590,740,680,610,710], 'gpa': [2,3.7,3.3,2.3,3] }
new_candidates
df2 = pd.DataFrame(new_candidates,columns= ['gmat', 'gpa'])
df2

y_pred2=logistic_regression.predict(df2)

print (df2)
print (y_pred2)
#The first and fourth candidates are not expected to be admitted, while the other candidates are expected to be admitted.
#df.concat(y_pred2)
pd.concat([df2, pd.Series(y_pred2)], axis=1, sort=False)
df2
