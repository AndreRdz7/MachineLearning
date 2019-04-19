# -*- coding: utf-8 -*-
"""
Hierarchical clustering

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
# Get dataset
data = pd.read_csv("./movies.csv", sep=";")
print(data.head())
movies = data.columns.values.tolist()[1:]
print(movies)
# Ward link
Z = linkage(data[movies], "ward")
print(Z)
plt.figure(figsize=(25,10))
plt.title("Dendograma jerárquico para el clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90, leaf_font_size = 10)
# Average link
Z = linkage(data[movies], "average")
print(Z)
plt.figure(figsize=(25,10))
plt.title("Dendograma jerárquico para el clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90, leaf_font_size = 10)
# Complete link
Z = linkage(data[movies], "complete")
print(Z)
plt.figure(figsize=(25,10))
plt.title("Dendograma jerárquico para el clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90, leaf_font_size = 10)
# Single link
Z = linkage(data[movies], "simple")
print(Z)
plt.figure(figsize=(25,10))
plt.title("Dendograma jerárquico para el clustering")
plt.xlabel("ID de los usuarios de Netflix")
plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90, leaf_font_size = 10)
plt.show()