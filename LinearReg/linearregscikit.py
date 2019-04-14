# -*- coding: utf-8 -*-
"""
Linear regression
with Scikit-learn

Author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
# Import dataset
data = pd.read_csv("./Advertising.csv")
# Prediction cols
feature_cols = ["TV","Radio","Newspaper"]
# Predictor and predictions
X = data[feature_cols]
Y = data["Sales"]
# Create model to get variables to train
estimator = SVR(kernel="linear")
selector = RFE(estimator, 2, step = 1)
selector = selector.fit(X,Y)
# check cols
selector.support_
# col ranking
selector.ranking_
# ML Model
X_pred = X[["TV", "Radio"]]
linearmodel = LinearRegression()
linearmodel.fit(X_pred,Y)
# alpha
linearmodel.intercept_
# betas
linearmodel.coef_
# r squared
linearmodel.score(X_pred,Y)