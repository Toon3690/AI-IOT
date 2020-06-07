import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Setup for visualisation with matplotlib
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

#Load the dataset with pd from "data/stats_14-17.csv"
#Save it to a variable called data
file_loc = "stats.csv"
data = pd.read_csv(file_loc)

#Now you can use these functions to print parts of your table
#print(data.describe(include="all")
#print(data.columns)

#Clean the data
data = data.drop(['Unnamed: 0', 'MP', '3PAr'], axis=1)
#print('dit dus', data)

X = data.drop(['Player', 'Pos', 'G', 'Player_ID','url', 'Status'], axis=1)
y = data['Pos']

# Scale the data
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Reduce dimensions with PCA
pca = PCA(n_components=3)
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
#print("Cumulative Explained Variance:", pca.explained_variance_ratio_.sum())

#Cluster using K-means
model = KMeans(n_clusters=4)
model.fit_transform(X_pca)

#print(X_pca)
#Draw visual
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])
#plt.show()
#print(X_pca[:,0])

for name, label in [('TRB_100', 0), ('ORB_100', 1), ('TOV_100', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))


# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show() 