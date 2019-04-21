# -*- coding: utf-8 -*-
"""
Boston's house prices regression tree

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor,export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold,cross_val_score
from graphviz import Source
import os
# Get dataset
data = pd.read_csv("./Boston.csv")
colnames = data.columns.values.tolist()
predictors = colnames[:13]
target = colnames[13]
X = data[predictors]
Y = data[target]
# Build tree
regtree = DecisionTreeRegressor(min_samples_split=30,min_samples_leaf=10,random_state=0)
regtree.fit(X,Y)
preds = regtree.predict(data[predictors])
data["preds"] = preds
print(data[["preds","medv"]])
# Visulize tree
with open("./boston.dot","w") as dotfile:
    export_graphviz(regtree,out_file=dotfile,feature_names=predictors)
    dotfile.close()
file = open("./boston.dot","r")
text = file.read()
Source(text)
# Cross validation
cv = KFold(n=X.shape[0],n_folds=10,shuffle=True,random_state=1)
scores = cross_val_score(regtree,X,Y,scoring="neg_mean_squared_error",cv = cv, n_jobs=1)
print(scores)
score = np.mean(scores)
print(score)
print(list(zip(predictors,regtree.feature_importances_)))
# Random forest
forest = RandomForestRegressor(n_jobs=20,oob_score=True,n_estimators=50000)
forest.fit(X,Y)
data["rforest_pred"] = forest.oob_prediction_
data[["rforest_pred","medv"]]
data["rforest_error"] = (data["rforest_pred"]-data["medv"])**2
res = sum(data["rforest_error"])/len(data)
print(res)
print(forest.oob_score_)