#---------------------------------------------------------------------------
# Clustring 
#---------------------------------------------------------------------------
import pandas as pd

dataset = pd.read_csv('E:/Data_Science/Machine_Learning/15) Clustering/studentclusters.csv')
x = dataset.copy()

# Visualization the using Pandas
x.plot.scatter(x='marks',y='shours')

# Normalize the data standard or min-max
from sklearn.preprocessing import minmax_scale
x_scale = minmax_scale(x)

# import K_means clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5,random_state=1234)
# fit the data
kmeans.fit(x_scale)
labels = kmeans.labels_
# visual the cluster
labels = pd.DataFrame(labels)
df = pd.concat([x,labels],axis=1)
df = df.rename(columns={0:"labels"})

# Visualization the using Pandas
df.plot.scatter(x='marks',y='shours',c='labels',colormap='Set1')

# Elbow Method for optimize No_of_Clusters
inertia = []
for i in range(2,15+1):
    kmeans =KMeans(n_clusters=i)
    kmeans.fit(x_scale)
    inertia.append(kmeans.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(2,16),inertia, marker='o')
plt.title('Elbow_curve')
plt.xlabel('NO_OF_Clusters')
plt.ylabel('Inertia')