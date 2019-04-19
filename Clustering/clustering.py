# -*- coding: utf-8 -*-
"""
Hierarchical clustering, full implementation
with random generated values

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, inconsistent,fcluster
from scipy.spatial.distance import pdist
import numpy as np
# Create datasets
np.random.seed(4711)
a = np.random.multivariate_normal([10,0],[[3,1],[1,4]], size = [100,])
b = np.random.multivariate_normal([0,20],[[3,1],[1,4]], size = [50,])
X = np.concatenate((a,b))
print(X.shape)
# Visualiza
plt.scatter(X[:,0],X[:,1])
# Linking
Z = linkage(X,"ward")
print(Z)
c, coph_dist = cophenet(Z, pdist(X))
# Dendrogram
plt.figure(figsize = (25,10))
plt.title("Dendrograma del clustering jerárquico")
plt.xlabel("Indices de muestra")
plt.ylabel("Distancias")
dendrogram(Z,leaf_rotation=90.,leaf_font_size = 8.0,color_threshold = 0.7*180) #180 is the global distance
# Last clusters
print(Z[-4:,])
# Truncate dendrogram
plt.figure(figsize = (25,10))
plt.title("Dendrograma del clustering jerárquico")
plt.xlabel("Indices de muestra")
plt.ylabel("Distancias")
dendrogram(Z,leaf_rotation=90.,leaf_font_size = 8.0,color_threshold = 0.7*180, truncate_mode="lastp", p = 12, show_leaf_counts = False, show_contracted = True) 
# Better dendrogram
def dendrogram_tune(*args,**kwargs):
    max_d = kwargs.pop("max_d",None)
    if max_d and "color_threshold" not in kwargs:
        kwargs["color_threshold"] = max_d
    annotate_above = kwargs.pop("annotate_above",0)
    ddata = dendrogram(*args,**kwargs)
    if not kwargs.get("no_plot",False):
        plt.title("Clustering jerárquico con dendrograma truncado")
        plt.xlabel("Indice del dataset (tamaño del cluster")
        plt.ylabel("Distancia")
        for i,d,c in zip(ddata["icoord"],ddata["dcoord"],ddata["color_list"]):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x,y,'o',c=c)
                plt.annotate("%.3g"%y,(x,y),xytext=(0,-5),textcoords = "offset points",va="top",ha = "center")
    if max_d:
        plt.axhline(y=max_d,c ="k")
    return ddata
dendrogram_tune(Z,truncate_mode="lastp",p=12,leaf_rotation=90.,leaf_font_size=12.,show_contracted=True,annotate_above=10,max_d=50)
# Cutting the dendrogram (selecting clusters)
depth = 5
incons = inconsistent(Z,depth)
print(incons)
# Elbow method
last = Z[-10:,2]
last_rev = last[::-1]
idx = np.arange(1,len(last)+1)
plt.figure(figsize = (25,10))
plt.title("Elbow method")
plt.xlabel("Clusters")
plt.ylabel("Distancias")
plt.plot(idx,last_rev)
acc = np.diff(last,2)
acc_rev =acc[::-1]
plt.plot(idx[:-2]+1,acc_rev)
k = acc_rev.argmax() + 2
print("El numero optimo de clusters es %s"%str(k))
# try with disperse data
c = np.random.multivariate_normal([40,40],[[20,1],[1,30]],size=[200,])
d = np.random.multivariate_normal([80,80],[[30,1],[1,30]],size=[200,])
e = np.random.multivariate_normal([0,100],[[100,1],[1,100]],size=[200,])
X2 = np.concatenate((X,c,d,e),)
Z2 = linkage(X2,"ward")
plt.figure(figsize=(25,10))
dendrogram_tune(Z2,truncate_mode="lastp",p=30,leaf_rotation=90.,leaf_font_size=10.,show_contracted=True,annotate_above=40,max_d=170)
last = Z2[-10:,2]
last_rev = last[::-1]
idx = np.arange(1,len(last)+1)
plt.figure(figsize = (25,10))
plt.title("Elbow method")
plt.xlabel("Clusters")
plt.ylabel("Distancias")
plt.plot(idx,last_rev)
acc = np.diff(last,2)
acc_rev =acc[::-1]
plt.plot(idx[:-2]+1,acc_rev)
k = acc_rev.argmax() + 2
print("El numero optimo de clusters es %s"%str(k))
# Final visualization
# By distance
max_d = 20
clusters = fcluster(Z,max_d,criterion="distance")
# By clusters
k = 3
clusters = fcluster(Z,k,criterion="maxclust")
# Plotting
plt.figure(figsize=(25,10))
plt.scatter(X[:,0],X[:,1],c = clusters,cmap="prism")
max_d = 170
clusters = fcluster(Z2,max_d,criterion="distance")
plt.figure(figsize=(25,10))
plt.scatter(X2[:,0],X2[:,1],c = clusters,cmap="prism")
plt.show()