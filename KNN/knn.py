# -*- coding: utf-8 -*-
"""
K Nearest Neighbours

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
# Get dataset
df = pd.read_csv("./breast-cancer-wisconsin.data.txt",header=None)
print(df.head())
print(df.describe())
print(df.head())
df = df.drop([0],1)
df.replace("?",-99999,inplace=True)
Y = df[10]
X = df[[1,2,3,4,5,6,7,8,9]]
# Classifier
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,Y_train)
# Test
accuracy = clf.score(X_test,Y_test)
print("Dataset cleaned: accuracy = ",accuracy*100,"%")
## Cleanless classifier
df = pd.read_csv("./breast-cancer-wisconsin.data.txt",header=None)
df.replace("?",-99999,inplace=True)
Y = df[10]
X = df[[0,1,2,3,4,5,6,7,8,9]]
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,Y_train)
accuracy = clf.score(X_test,Y_test)
print("Dataset not cleaned: accuracy = ",accuracy*100,"%")