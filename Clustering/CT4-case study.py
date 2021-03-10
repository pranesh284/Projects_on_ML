

import numpy as np
X = np.array([[5,3],  [10,15], [15,12],  [24,10],  [30,30],  [85,70],   [71,80],  [60,78],    [70,55],    [80,91],])
X

import matplotlib.pyplot as plt

labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')
for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(  label,   xy=(x, y), xytext=(-3, 3), textcoords='offset points', ha='right', va='bottom')
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(X, 'single')
linked
labelList = range(1, 11)

plt.figure(figsize=(10, 7))
dendrogram(linked,  orientation='top',  labels=labelList,  distance_sort='descending', show_leaf_counts=True)
plt.show();




#%%%
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')

cluster.fit_predict(X)

print(cluster.labels_)

plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')


#  segment customers into different groups based on their shopping trends.

url='https://stackabuse.s3.amazonaws.com/files/hierarchical-clustering-with-python-and-scikit-learn-shopping-data.csv'
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib inline
import numpy as np
df = pd.read_csv(url)
#customer_data = pd.read_csv('data/shopping-data.csv')
customer_data = df.copy()
customer_data.head()
#explore data
customer_data
customer_data.head()
customer_data.shape
customer_data.describe()
customer_data.columns

#rename columns
customer_data.columns = ['customerID', 'genre', 'age', 'income', 'spendscore']
customer_data.head()

# two-dimensional feature space, we will retain only two of these five columns. We can remove CustomerID column, Genre, and Age column.
# will retain the Annual Income
 # (in thousands of dollars) and Spending Score (1-100) columns. The Spending Score column signifies
 # how often a person spends money in a mall on a scale of 1 to 100 with 100 being the highest spender.

data = customer_data.iloc[:, 2:5]
data

import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))
plt.show();

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')

pred = cluster.fit_predict(data)


data['labels'] = pred
data

data.to_csv("income.csv")


#%%%kmeans

import matplotlib.pyplot as plt
import numpy as np
X2 = np.array([[1,2], [1.5, 1.8], [5,8], [8,8], [1, 0.6], [9,11]])
X2
from matplotlib import style

plt.scatter(X2[:,0], X2[:,1], s=150)
plt.show();

from sklearn.cluster import KMeans
Kmean2 = KMeans(n_clusters=2)
Kmean2.fit(X2)
centers2 = Kmean2.cluster_centers_
X2
centers2
Kmean2.labels_

plt.scatter(X2[:,0], X2[:,1], s=50, c = Kmean2.labels_)
plt.scatter(centers2[:,0], centers2[:,1], s=100, marker='*', color =['red'])
plt.show();

#%%%iris
import matplotlib.pyplot as plt
#pip install kneed
from kneed import KneeLocator
from sklearn.cluster import KMeans

from pydataset import data
iris = data('iris')
data = iris.copy()
data.head()
#how many groups
data.Species.value_counts()
data.columns
data_train= data.iloc [ :,:-1]



X3 = data[['Sepal.Length','Sepal.Width']]
X3
y3 = data.Species.values
y3
X3.shape

plt.scatter(X3['Sepal.Length'], X3['Sepal.Width'], s=50)



#%%choosing no of clusters
kmeans_kwargs = {'init':'random', 'n_init':10, 'max_iter': 300}
sse=[]

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(data_train)
    sse.append(kmeans.inertia_)
sse

plt.style.use('fivethirtyeight')
plt.plot(range(1,11), sse)
plt.xticks(range(1,11))
plt.xlabel('No of clusters')
plt.ylabel('SSE')
plt.show();

kl = KneeLocator(x=range(1,11), y=sse, curve='convex', direction='decreasing')
kl.elbow



#group them into 3 categories
irisCluster = KMeans(n_clusters=3)
irisCluster.fit(data_train)
irisCenters= irisCluster.cluster_centers_
irisCenters
irisCluster.labels_

data[ 'pred'] = irisCluster.labels_
data


data.pred.value_counts()


from sklearn.cluster import AgglomerativeClustering

aggCluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

aggCluster

data[ 'predHerar']  = aggCluster.fit_predict(data_train)

data

data.predHerar.value_counts()




irisCluster.labels_
import collections
collections.Counter(irisCluster.labels_)

#optimal no of clusters
from sklearn.cluster import KMeans
X3
SSdistance = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X3)
    SSdistance.append(km.inertia_)

SSdistance  #inertia for 10 differents sets of clusters

plt.plot(K, SSdistance, 'bx-')
plt.xlabel('k -no of clusters')
plt.ylabel('Sum of Sq Distance')
plt.title('Elbow method to find optimal no of clusters')
plt.show();



