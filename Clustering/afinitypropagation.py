# -*- coding: utf-8 -*-
"""
Affinity propagation

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle
# Create dataset
centers = [[1,1],[-1,-1],[1,-1]]
X, labels = make_blobs(n_samples=300,centers=centers,cluster_std=0.5, random_state=0)
plt.scatter(X[:,0],X[:,1],c=labels,s=5,cmap="autumn")
plt.show()
af = AffinityPropagation(preference=-50).fit(X)
cluster_center_ids = af.cluster_centers_indices_
labels = af.labels_
n_clust = len(cluster_center_ids)
# Make it a function to report
def report_affinity_propagation(X):
    af = AffinityPropagation(preference=-50).fit(X)
    cluster_center_ids = af.cluster_centers_indices_
    n_clust = len(cluster_center_ids)
    clust_labels = af.labels_
    print("Numero estimado de clusters: %d"%n_clust)
    print("Homogeneidad: %0.3f"%metrics.homogeneity_score(labels,clust_labels))
    print("Completitud: %0.3f"%metrics.completeness_score(labels,clust_labels))
    print("V-measure: %0.3f"%metrics.v_measure_score(labels,clust_labels))
    print("R2 ajustado: %0.3f"%metrics.adjusted_rand_score(labels,clust_labels))
    print("Informacion mutua ajustada: %0.3f"%metrics.adjusted_mutual_info_score(labels,clust_labels))
    print("Coeficiente de la silueta: %0.3f"%metrics.silhouette_score(X,labels,metric="sqeuclidean"))
    plt.figure(figsize=(16,9))
    plt.clf()
    colors=cycle("bgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk")
    for k, col in zip(range(n_clust),colors):
        class_members = (clust_labels==k)
        clust_center = X[cluster_center_ids[k]]
        plt.plot(X[class_members,0], X[class_members,1],col + ".")
        plt.plot(clust_center[0],clust_center[1], 'o', markerfacecolor="k",markersize=14)
        for x in X[class_members]:
            plt.plot([clust_center[0],x[0]],[clust_center[1],x[1]],col)
    plt.title("numero estimado de clusters %d"%n_clust)
    plt.show()
# Call function
report_affinity_propagation(X)