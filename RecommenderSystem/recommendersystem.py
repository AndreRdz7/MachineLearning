# -*- coding: utf-8 -*-
"""
Movielens Recommender system
(using knn-ish method)

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import NearestNeighbors
# Get dataset
df = pd.read_csv("./u.data", sep='\t', header=None)
df.columns = ["UserID","ItemID","Rating","TimeStamp"]
print(df.head())
# Data visualization
plt.hist(df.Rating)
plt.show()
plt.hist(df.TimeStamp)
plt.show()
print(df.groupby(["Rating"])["UserID"].count())
plt.hist(df.groupby(["ItemID"])["ItemID"].count())
plt.show()
# Matrix representation
n_users = df.UserID.unique().shape[0]
n_items = df.ItemID.unique().shape[0]
ratings = np.zeros((n_users,n_items))
# Sparse matrix
for row in df.itertuples():
    ratings[row[1]-1,row[2]-1] = row[3]
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0]*ratings.shape[1])
sparsity *= 100
print("Sparsity: {:4.2f}%".format(sparsity))
# Splitting data
ratings_train, ratings_test = train_test_split(ratings,test_size=0.3,random_state=42)
print(ratings_train.shape)
# User filter
sim_matrix = 1 - sklearn.metrics.pairwise.cosine_distances(ratings_train)
print(sim_matrix.shape)
users_predictions = sim_matrix.dot(ratings_train) / np.array([np.abs(sim_matrix).sum(axis=1)]).T
def get_mse(preds, actuals):
    if preds.shape[0] != actuals.shape[1]:
        actuals = actuals.T
    preds = preds[actuals.nonzero()].flatten()
    actuals = actuals[actuals.nonzero()].flatten()
    return mean_squared_error(preds,actuals)

get_mse(users_predictions,ratings_train)
get_mse(users_predictions,ratings_test)
k = 10
neighbors = NearestNeighbors(k,'cosine')
neighbors.fit(ratings_train)
top_k_distances, top_k_users = neighbors.kneighbors(ratings_train,return_distance=True)
users_predicts_k = np.zeros(ratings_train.shape)
for i in range(ratings_train.shape[0]):
    users_predicts_k[i,:] = top_k_distances[i].T.dot(ratings_train[top_k_users][i]) / np.array([np.abs(top_k_distances[i].T).sum(axis=0)]).T
get_mse(users_predicts_k, ratings_train)
get_mse(users_predicts_k,ratings_test)
# Item filter
n_movies = ratings_train.shape[1]
neighbors = NearestNeighbors(n_movies,'cosine')
neighbors.fit(ratings_train.T)
top_k_distances,top_k_items = neighbors.kneighbors(ratings_train.T,return_distance=True)
item_prediction = ratings_train.dot(top_k_distances) / np.array([np.abs(top_k_distances).sum(axis=1)])
get_mse(item_prediction,ratings_train)
get_mse(item_prediction,ratings_test)
# KNN filter
k = 30
neighbors = NearestNeighbors(k,'cosine')
neighbors.fit(ratings_train.T)
top_k_distances,top_k_items = neighbors.kneighbors(ratings_train.T,return_distance=True)
preds = np.zeros(ratings_train.T.shape)
for i in range(ratings_train.T.shape[0]):
    den = 1
    if(np.abs(top_k_distances[i]).sum(axis=0)>0):
        den = np.abs(top_k_distances[i]).sum(axis=0)
    preds[i,:] = top_k_distances[i].dot(ratings_train.T[top_k_items][i]) / np.array([den]).T
get_mse(preds,ratings_train)
get_mse(preds,ratings_test)