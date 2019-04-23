# -*- coding: utf-8 -*-
"""
Simple linear support vector classifier

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
# Make dataset
X = [1,5,1.5,8,1,9]
Y = [2,8,1.8,8,0.6,11]
plt.scatter(X,Y)
plt.show()
data = np.array(list(zip(X,Y)))
# Lower is 0, upper is 1
target = [0,1,0,1,0,1]
# Make model
classifier = svm.SVC(kernel="linear",C = 1.0)
classifier.fit(data,target)
# Try model
point = np.array([0.57,0.67]).reshape(1,2)
classifier.predict(point)
point2 = np.array([10.57,10.67]).reshape(1,2)
classifier.predict(point2)
# SVM Graphical representation
# model = w0x + w1y + e = 0
w = classifier.coef_[0]
print("weights",w)
a = -w[0]/w[1]
b = -classifier.intercept_[0]/w[1]
# y = ax + b
xx = np.linspace(0,10)
yy = a * xx + b
plt.plot(xx,yy,"k-",label="Hiperplano de separación")
plt.scatter(X,Y,c = target)
plt.legend()
plt.show()