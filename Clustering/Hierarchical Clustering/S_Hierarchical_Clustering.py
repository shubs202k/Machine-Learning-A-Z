# Hierarchical Clustering

#%reset -f

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3 , 4]].values

# Using the Dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward')) 

# Method = ward ===== A method that tries to minimize the variance
# within each cluster
plt.figure(1)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eulidean Distance')
plt.show()

# Applying HC to our Dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5,affinity = 'euclidean',
                             linkage = 'ward')
y_hc = hc.fit_predict(X) 

# Visualizing the Clustering Results 
plt.figure(2)
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1], s= 100,c='red',label='Careful')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1], s= 100,c='blue',label='Standard')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1], s= 100,c='cyan',label='Target')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1], s= 100,c='green',label='Careless')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1], s= 100,c='magenta',label='Sensible')
plt.title('Clusters_of_Clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()    












  








