

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

data = {'x': [25,34,22,27,33,33,31, 22,35,34,67,54,57,43,50,57,59,52,65, 47,49,48,35,33,44,45,38,43,51,46],'y': [79,51,53,78,59,74,73,57,69,75,51,32, 40,47,53,36,35,58, 59,50,25,20,14,12,20,5,29,27,8,7]       }

df = pd.DataFrame(data,columns=['x','y'])
print (df)

plt.scatter(df['x'], df['y'])



kmeans = KMeans(n_clusters=10)
kmeans.fit(df)

centroids = kmeans.cluster_centers_

print(centroids)

kmeans.labels_.astype(float)


plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50)

plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=80, marker="*")
plt.show()



sse=[]
kmeans_kwargs = {'init':'random', 'n_init':10, 'max_iter': 300}

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)
sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();



kmeans = KMeans(n_clusters=3)
kmeans.fit(df)

centroids = kmeans.cluster_centers_

print(centroids)

kmeans.labels_.astype(float)


plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50)

cl = ['r','g','b']
plt.scatter(centroids[:, 0], centroids[:, 1], c=cl, s=80, marker="*")
plt.show()




!pip install kneed
from kneed import KneeLocator


kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow




kmeans.predict([[22,66]])

kmeans.predict([[66,99]])

kmeans.predict([[40,20]])



kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()




from pydataset import data
mtcars = data('mtcars')
data = mtcars.copy()

data.head(2)


sse=[]
kmeans_kwargs = {'init':'random', 'n_init':10, 'max_iter': 300}

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(data)
    sse.append(kmeans.inertia_)
sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();



from kneed import KneeLocator
kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow





kmeans = KMeans( init = 'random', n_clusters=2,  max_iter=300)
kmeans
kmeans.fit(data)

kmeans.cluster_centers_  #average or rep values
kmeans.labels_

data['labels'] = kmeans.labels_

data


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_features = scaler.fit_transform(data)

scaled_features[:5]  #values between -3 to +3

kmeans = KMeans( init = 'random', n_clusters=3,  max_iter=300)
kmeans
kmeans.fit(scaled_features)

kmeans.inertia_

kmeans.cluster_centers_  #average or rep values
kmeans.n_iter_  #in 6 times, clusters stabilised

kmeans.labels_
kmeans.cluster_centers_.shape
kmeans.cluster_centers_[0:1]


data1=data

data["labels1"] =kmeans.labels_
data
