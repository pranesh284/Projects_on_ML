

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = "https://stats.idre.ucla.edu/stat/data/binary.csv"
df = pd.read_csv(url)

df

x= df[['gre', 'gpa']]

y = df['admit']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
X_train.shape, X_test.shape
y_train.shape, y_test.shape

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
acc

confusion_matrix = pd.crosstab(y_test,y_pred)

confusion_matrix


import seaborn as sns # used to create the Confusion Matrix

sns.heatmap(confusion_matrix, annot=True, cmap='coolwarm', annot_kws={'size':20}, cbar=True)
plt.show();





