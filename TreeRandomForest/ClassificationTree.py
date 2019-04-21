# -*- coding: utf-8 -*-
"""
Flower classification tree

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cross_validation import KFold,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from graphviz import Source
# Get dataset
data = pd.read_csv("./iris.csv")
print(data.head())
# Visualize data
print(data.Species.unique())
plt.hist(data.Species)
# Prepare data
colnames = data.columns.values.tolist()
predictors = colnames[:4]
target = colnames[4]
# Split dataset
data["is_train"] = np.random.uniform(0,1,len(data)) <= 0.75
train,test = data[data["is_train"]==True], data[data["is_train"]==False]
# Create model
tree = DecisionTreeClassifier(criterion="entropy", min_samples_split=20,random_state=99)
tree.fit(train[predictors],train[target])
preds = tree.predict(test[predictors])
# Verify results
pd.crosstab(test[target],preds,rownames=["Actual"], colnames=["Predictions"])
# Visualization
with open("./iris_dtree.dot","w") as dotfile:
    export_graphviz(tree,out_file=dotfile, feature_names=predictors)
    dotfile.close()
file = open("./iris_dtree.dot","r")
text = file.read()
Source(text)
# Cross validation
X = data[predictors]
Y = data[target]
tree = DecisionTreeClassifier(criterion="entropy",max_depth=3,min_samples_split=20,random_state=99)
tree.fit(X,Y)
cv = KFold(n= X.shape[0], n_folds=10,shuffle=True,random_state=1)
score = np.mean(cross_val_score(tree,X,Y,scoring="accuracy",cv = cv, n_jobs=1))
print(score)
# Random Forest
forest = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=10)
forest.fit(X,Y)
print(forest.oob_decision_function_)
print(forest.oob_score_)