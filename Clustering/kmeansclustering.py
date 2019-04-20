# -*- coding: utf-8 -*-
"""
K-means clustering

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import numpy as numpy
from scipy.cluster.vq import vq,kmeans
# Create datasets
data = np.random.random(90).reshape(30,3)
c1 = np.random.choice(range(len(data)))
c2 = np.random.choice(range(len(data)))
# Getting k
clust_centers = np.vstack([data[c1],data[c2]])
print(clust_centers)
print(vq(data,clust_centers))
# K-means
kmeans(data,clust_centers)
