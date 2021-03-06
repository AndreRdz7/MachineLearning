# -*- coding: utf-8 -*-
"""
SVM for regression

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
# set dataset
X = np.sort(5*np.random.rand(200,1),axis=0)
Y = np.sin(X).ravel()
Y[::5] += 3*(0.5-np.random.rand(40))
# Plot
plt.scatter(X,Y,color="darkorange",label="data")
# Make regression
C = 1e3
svr_lin = SVR(kernel="linear",C=C)
svr_rbf = SVR(kernel="rbf",C=C,gamma=0.1)
svr_pol = SVR(kernel="poly",C=C,degree=3)
# Make predictions
y_lin = svr_lin.fit(X,Y).predict(X)
y_rbf = svr_rbf.fit(X,Y).predict(X)
y_pol = svr_pol.fit(X,Y).predict(X)
# Plot
lw = 2
plt.figure(figsize=(16,9))
plt.scatter(X,Y,color="darkorange",label="data")
plt.plot(X,y_lin, color="navy",lw=lw,label="SVM Linear")
plt.plot(X,y_rbf, color="c",lw=lw,label="SVM Radial")
plt.plot(X,y_pol, color="cornflowerblue",lw=lw,label="SVM Polynomial")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Support Vector Regression")
plt.legend()
plt.show()