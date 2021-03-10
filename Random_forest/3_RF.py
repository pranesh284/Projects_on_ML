
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data
from pydataset import data
mtcars = data('mtcars')
mtcars.head()
df = mtcars.copy()
df
df[['wt','hp']]


X = df[['wt','hp']].values
X
y = df['mpg'].values  #s
y

from sklearn.metrics import r2_score


## Fitting Random Forest Regression to the dataset
# import the regressor

from sklearn.ensemble import RandomForestRegressor

# create regressor object
regressor = RandomForestRegressor(n_estimators = 100)

# fit the regressor with x and y data
regressor.fit(X, y)
ypred= regressor.predict(X)

r2_score(y, ypred)


from sklearn.tree import DecisionTreeRegressor

 # create regressor object
DTregressor = DecisionTreeRegressor()

# fit the regressor with x and y data
DTregressor.fit(X, y)
ypred2= DTregressor.predict(X)

print(r2_score(y, ypred2))


from sklearn.linear_model import LinearRegression

LR= LinearRegression()

LR.fit(X,y)
ypred1= LR.predict(X)

print(r2_score(y,ypred1))




pd.DataFrame({'actual':df['mpg'], 'RF_predict': regressor.predict(X),'RF_diff':df['mpg'] - regressor.predict(X), 'LR_predict': LR.predict(X),'LR_diff':df['mpg'] - LR.predict(X) })
df[['wt','hp']].head()
newData = np.array([2.7, 120]).reshape(1, 2)
newData
ypred1 = regressor.predict(newData)  # test the output by changing values
ypred1


#classification


df.columns
X2 = df[['wt','hp']].values
X2
y2 = df['am'].values  #s
y2

## Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestClassifier

 # create classifier object
classifier = RandomForestClassifier(n_estimators = 100, random_state = 0)

# fit the regressor with x and y data
classifier.fit(X2, y2)
classifier.predict(X2)
classifier.predict_proba(X2)
classifier.score(X2,y2)
pd.DataFrame({'actual':df['am'], 'predict': classifier.predict(X2),'diff':df['am'] - classifier.predict(X2) })

from sklearn.metrics import confusion_matrix
confusion_matrix(y2, classifier.predict(X2))


df[['wt','hp']].head()
newData = np.array([2.7, 120]).reshape(1, 2)
newData
ypred2 = classifier.predict(newData)  # test the output by changing values
ypred2




l1 = ['india', 'nepal', 'china']

cnt = 0

s1 =set({})



for i in l1:
    s1.add(i.upper())

s1



e3 = enumerate (list(range(10,20)),start=10)

c=0

for i in e3:
    print(i)
    if(c>5):
        break

    c=c+1
print (c)

for i in e3:
    print(i)

city = np.random.choice(a=['Delhi', 'Gurugram','Noida','Faridabad'], size=10, p=[.4,.2,.2,.2])
city






while (cnt<len(l1)):
    l1[cnt] = l1[cnt].upper()
    cnt= cnt+1

l1


friends = ["Rolf", "Bob", "Jen", "Anne"]
time_since_seen = [3, 7, 15, 11]


long_timers ={
    friends[i]: time_since_seen[i]
    for i in range(len(friends))
        if time_since_seen[i] > 5
}














