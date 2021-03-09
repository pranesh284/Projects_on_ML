

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
df.drop_duplicates()

X = df[['gpa', 'gre']].values

y = df['admit'].values


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
X_train.shape, X_test.shape
y_train.shape, y_test.shape





model= LogisticRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)




from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(y_test, y_pred)

cf =confusion_matrix(y_test, y_pred)

sns.heatmap(cf, annot=True, cmap='coolwarm', annot_kws={'size':20}, cbar=True)
plt.show();


y_pred=logistic_regression.predict(X_test)

y_pred

















