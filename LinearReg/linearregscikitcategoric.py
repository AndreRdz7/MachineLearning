# -*- coding: utf-8 -*-
"""
Linear regression
with Scikit-learn and categoric variables

Author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# Import dataset
data = pd.read_csv("./Ecom Expense.csv")
# Check head
data.head()
# Make dummies
dummy_gender = pd.get_dummies(data["Gender"],prefix = "Gender")
dummy_city_tier = pd.get_dummies(data["City Tier"], prefix = "City")
# Join dataframes
column_names = data.columns.values.tolist()
data_new = data[column_names].join(dummy_gender)
column_names = data_new.columns.values.tolist()
data_new = data_new[column_names].join(dummy_city_tier)
# Get useful cols
feature_cols = ["Monthly Income", "Transaction Time", "Record","Gender_Male", "City_Tier 2","City_Tier 3"]
X = data_new[feature_cols]
Y = data_new["Total Spend"]
# Create model
linearmodel = LinearRegression()
linearmodel.fit(X,Y)
linearmodel.intercept_
linearmodel.coef_
list(zip(feature_cols,linearmodel.coef_))
linearmodel.score(X,Y)
data_new["prediction"] = -79.41713030137271 + data_new["Monthly Income"] * 0.14753898049205744 + data_new["Transaction Time"] * 0.15494612549589615 + data_new["Gender_Female"] * -94.15779883032012 + data_new["Gender_Male"] * 94.15779883032015 + data_new["Record"] * 772.2334457445644 + data_new["City_Tier 1"] * 76.76432601049521 + data_new["City_Tier 2"] * 55.13897430923251 + data_new["City_Tier 3"] * -131.90330031972783
# Error
SSD = np.sum((data_new["prediction"]-data_new["Total Spend"])**2)
RSE = np.sqrt(SSD/(len(data_new)-len(feature_cols)-1))
RSE
sales_mean = np.mean(data_new["Total Spend"])
sales_mean
error = RSE/sales_mean
error
# Delete dummy variables (use IPython console to execute previous code with this dummies, instead of regulars)
dummy_gender = pd.get_dummies(data["Gender"],prefix = "Gender").iloc[:,1:]
dummy_city_tier = pd.get_dummies(data["City Tier"], prefix = "City").iloc[:,1:]
