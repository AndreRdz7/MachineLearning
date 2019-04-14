# -*- coding: utf-8 -*-
"""
Model validation

Author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np
# Import dataset
data = pd.read_csv("./Advertising.csv")
# Split on train and test data
a = np.random.randn(len(data))
check = (a<0.8)
training = data[check]
testing = data[~check]
# Create model on best variables
linearmodel = smf.ols(formula="Sales~TV+Radio",data = training).fit()
linearmodel.summary()
# Sales =  2.9205 +  0.0463 * TV +  0.1861 * Radio
# Model validation with testing data
sales_prediction = linearmodel.predict(testing)
sales_prediction
# Errors
SSD = sum((testing["Sales"]-sales_prediction)**2)
SSD
RSE = np.sqrt(SSD/(len(testing)-2-1))
RSE
sales_mean = np.mean(testing["Sales"])
error = RSE/sales_mean
error
