# -*- coding: utf-8 -*-
"""
K Nearest Neighbours (no dataset)

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from math import sqrt
from collections import Counter
import random
from sklearn import neighbors

# Create dataset
dataset = {
    'k':[[1,2],[2,3],[3,1]],
    'r':[[6,5],[7,7],[8,6]]
}
new_point = [5,7]
[[plt.scatter(ii[0],ii[1],s=50,color = i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_point[0],new_point[1],s=100)
plt.show()
# KNN
def k_nearest_neighbors(data,predict,k=3):
    if len(data) >= k:
        warnings.warn("K es un valor menor que el numero total de elementos a votar")
    distances = []
    for group in data:
        for feature in data[group]:
            # d = sqrt((feature[0]-predict[0])**2 + (feature[1]-predict[1])**2)
            d = np.sqrt(np.sum((np.array(feature) - np.array(predict))**2)) 
            """
            d = np.linalg.norm(np.array(feature)-np.array(predict))
            """
            distances.append([d,group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

result = k_nearest_neighbors(dataset,new_point)
print(result)
# Algorithm testing on dataset
df = pd.read_csv("./breast-cancer-wisconsin.data.txt",header=None)
df.replace("?",-99999,inplace=True)
df = df.drop([0],1,inplace=True)
full_data = df.astype(float).values.tolist()
test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data))]
for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])
correct = 0
total = 0
for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set,data,k = 5)
        if group == vote:
            correct += 1
        total += 1
print("Efficiency = ",correct/total*100)