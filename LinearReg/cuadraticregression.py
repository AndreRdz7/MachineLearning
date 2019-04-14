# -*- coding: utf-8 -*-
"""
Variable transformation

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
# Check head
data.head()
# Clean dataframe
data["mpg"] = data["mpg"].dropna()
data["horsepower"] = data["horsepower"].dropna()
plt.plot(data["horsepower"],data["mpg"],"ro")
plt.xlabel("Caballos de potencia")
plt.ylabel("Consumo (millas por galeón)")
plt.title("CV vs MPG")
# Linear model
X = data["horsepower"].fillna(data["horsepower"].mean())
Y = data["mpg"].fillna(data["mpg"].mean())
# Adjust X
X_adj = X[:,np.newaxis]
linearmodel = LinearRegression()
# Adjust x to ndarray and train
linearmodel.fit(X_adj,Y)
plt.plot(X,Y,"ro")
plt.plot(X,linearmodel.predict(X_adj,color="blue")
linearmodel.score(X_adj,Y)
# Error
SSD = np.sum((Y-linearmodel.predict(X_adj))**2)
RSE = np.sqrt(SSD/(len(X_adj)-1))
y_mean = np.mean(Y)
error = RSE/y_mean
error
# Cuadratic model
# mpg = a * b * hp**2
X_data = X**2
X_data = X_data[:,np.newaxis]
lm = LinearRegression()
lm.fit(X_data,Y)
lm.score(X_data,Y)
# Error
SSD = np.sum((Y-lm.predict(X_data))**2)
RSE = np.sqrt(SSD/(len(X_data)-1))
y_mean = np.mean(Y)
error = RSE/y_mean
error
# Combine Linear and Cuadratic
# mpg = a + b*hp + c*hp**2
poly = PolynomialFeatures(degree=2)
X_data = poly.fit_transform(X[:,np.newaxis])
linearcombined = LinearRegression()
linearcombined.fit(X_data,Y)
linearcombined.score(X_data, Y)
linearcombined.intercept_
linearcombined.coef_
# Try many degrees
for degree in range(2,10):
	poly = PolynomialFeatures(degree=2)
	X_data = poly.fit_transform(X[:,np.newaxis])
	linearcombined = LinearRegression()
	linearcombined.fit(X_data,Y)
	print("on degree ",degree)
	print(linearcombined.score(X_data, Y))