# -*- coding: utf-8 -*-
"""
Principal componen analysis

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sk_pca
# Get dataset
df = pd.read_csv("./iris.csv")
X = df.iloc[:,0:4].values
Y = df.iloc[:,4].values
# Make it standard
X_std=StandardScaler().fit_transform(X)
# PCA
acp = sk_pca(n_components=2)
Y = acp.fit_transform(X_std)