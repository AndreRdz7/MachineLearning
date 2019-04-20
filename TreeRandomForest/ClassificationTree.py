# -*- coding: utf-8 -*-
"""
Flower classification tree

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
# Get dataset
data = pd.read_csv("./iris.csv")
print(data.head())
# Visualize data
print(data.Species.unique())
plt.hist(data.Species)
plt.show()
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
