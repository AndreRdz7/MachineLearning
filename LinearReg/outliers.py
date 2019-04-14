# -*- coding: utf-8 -*-
"""
Outliers

Author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
# Import dataset
data = pd.read_csv("./auto-mpg.csv")
X = data["displacement"].fillna(data["displacement"].mean())
X = X[:,np.newaxis]
Y = data["mpg"].fillna(data["mpg"].mean())
linearmodel = LinearRegression()
linearmodel.fit(X,Y)
linearmodel.score(X,Y)
# Visualize linear regression
plt.plot(X,Y,"ro")
plt.plot(X,linearmodel.predict(X),color="blue")
data[(data["displacement"]>250) & (data["mpg"]>35)]
data[(data["displacement"]>300) & (data["mpg"]>20)]
data_clean = data.drop([395,258,305,372])
# Try new data
X = data_clean["displacement"].fillna(data_clean["displacement"].mean())
X = X[:,np.newaxis]
Y = data_clean["mpg"].fillna(data_clean["mpg"].mean())
linearmodel = LinearRegression()
linearmodel.fit(X,Y)
linearmodel.score(X,Y)
# Further cleaning... by using boxplotting and deleting those values
