# -*- coding: utf-8 -*-
"""
Elbow method and silhouette coefficient

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples,silhouette_score
# Create dataset
x1 = np.array([3,1,1,2,1,6,6,6,5,6,7,8,9,7,9,9,8])
x2 = np.array([5,4,5,6,5,8,4,7,4,7,1,2,1,2,3,2,3])
X = np.array(list(zip(x1,x2))).reshape(len(x1),2)
plt.plot()
plt.xlim([0,10])
plt.ylim([0,10])
plt.title("Dataset a clasificar")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x1,x2)
plt.show()
# Prepare clustering
max_k = 10
K = range(1,max_k)
ssw = []
color_palette = [plt.cm.Spectral(float(i)/max_k) for i in K]
centroid = [sum(X)/len(X) for i in K]
sst = sum(np.min(cdist(X,centroid,"euclidean"),axis = 1))
# Clutering
for k in K:
    kmeanmodel = KMeans(n_clusters=k).fit(X)
    centers = pd.DataFrame(kmeanmodel.cluster_centers_)
    labels = kmeanmodel.labels_
    ssw_k = sum(np.min(cdist(X,kmeanmodel.cluster_centers_,"euclidean"),axis = 1))
    ssw.append(ssw_k)
    label_color = [color_palette[i] for i in labels]
    # Get silhouette
    if 1 < k < len(X):
        fig, (axis1,axis2) = plt.subplots(1,2)
        fig.set_size_inches(20,8)
        axis1.set_xlim([-0.1,1.0])
        axis1.set_ylim([0,len(X) + (k+1)*10])
        silhouette_avg = silhouette_score(X,labels)
        print("* para k = ",k, "el promedio de la silueta es de :",silhouette_avg)
        sample_silhouette_values = silhouette_samples(X,labels)
        y_lower = 10
        for i in range(k):
            # Add silhouette
            ith_cluster_silval = sample_silhouette_values[labels == i]
            print("   - para i = ",i+1, " la silueta del cluster vale: ",np.mean(ith_cluster_silval))
            ith_cluster_silval.sort()
            # Get position
            ith_cluster_size = ith_cluster_silval.shape[0]
            y_upper = y_lower + ith_cluster_size
            # Get color
            color = color_palette[i]
            # Paint silhouette
            axis1.fill_betweenx(np.arange(y_lower,y_upper),0,ith_cluster_silval,facecolor=color,alpha=0.7)
            # Label cluster
            axis1.text(-0.05,y_lower+.5*ith_cluster_size,str(i+1))
            # Get new y_lower
            y_lower = y_upper+10
        axis1.set_title("Representación de la silueta para k = %s"%str(k))
        axis1.set_xlabel("S(i)")
        axis1.set_ylabel("ID del cluster")
    plt.plot()
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.title("Clustering para k = %s"%str(k))
    plt.scatter(x1,x2,c=label_color)
    plt.scatter(centers[0],centers[1],c=color_palette,marker="x")
    plt.show()
# Elbow method
plt.plot(K,ssw,"bx-")
plt.xlabel("k")
plt.ylabel("SSw(k)")
plt.title("La tecnica del codo para encontrar el k ópimo")
# Elbow method (normalized)
plt.plot(K,1-ssw/sst,"bx-")
plt.xlabel("k")
plt.ylabel("i-norm(SSw(k))")
plt.title("La tecnica del codo  normalizado para encontrar el k ópimo")
plt.show()