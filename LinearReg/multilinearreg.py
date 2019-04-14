# -*- coding: utf-8 -*-
"""
Multiple linear regression with Python,
get relation between ads on tv, radio, newspaper and sales

Author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
# Import dataset
data = pd.read_csv("./Advertising.csv")
# Linear model of sales in function of TV ads
linearmodel = smf.ols(formula="Sales~TV", data=data).fit()
# See parameters
# Where intercept means the alpha
# Where TV means means the beta
# Such as sale_predicted = intercept + TV * tv(x)
# We get the linear model
linearmodel.params
# p-values shoud be near 0
linearmodel.pvalues
# r squared should be near 1
linearmodel.rsquared
# Regression results
linearmodel.summary()
# Predict
sales_prediction = linearmodel.predict(pd.DataFrame(data["TV"]))
# Graph it
data.plot(kind="scatter", x="TV", y="Sales")
plt.plot(pd.DataFrame(data["TV"]), sales_prediction, c="red", linewidth=2)
# SRE
data["sales_prediction"] = 7.032594 + 0.047537 * data["TV"]
data["RSE"] = (data["Sales"]-data["sales_prediction"])**2
SSD = sum(data["RSE"])
RSE = np.sqrt(SSD/(len(data)-2))
# Sales average
sales_mean = np.mean(data["Sales"])
# Error percentage
error = RSE/sales_mean
# Error histogram (should be normal)
plt.hist((data["Sales"]-data["sales_prediction"]))

# Add newspaper to the model
linearmodel2 = smf.ols(formula="Sales~TV+Newspaper", data=data).fit()
# Show params
linearmodel2.params
# Shoe p-values
linearmodel2.pvalues
# New model
# Sales = 5.774948 + 0.046901 * TV + 0.044219 * Newspaper
# Verify r squared
linearmodel2.rsquared
linearmodel2.rsquared_adj
# Make predictions
sales_prediction2 = linearmodel2.predict(data[["TV", "Newspaper"]])
# Show predictions
sales_prediction2
# Get SSD
SSD = sum((data["Sales"]-sales_prediction2)**2)
# Show SSD
RSE = np.sqrt(SSD/(len(data)-2-1))
# Show RSE
RSE
# Show linearmodel2 summary
linearmodel2.summary()

# Add newspaper to the model
linearmodel3 = smf.ols(formula="Sales~TV+Radio", data=data).fit()
# Get summary
linearmodel3.summary()
# Make predictions
sales_prediction3 = linearmodel3.predict(data[["TV", "Radio"]])
# Show predictions
sales_prediction3
# Get SSD
SSD = sum((data["Sales"]-sales_prediction3)**2)
# Show SSD
RSE = np.sqrt(SSD/(len(data)-2-1))
# Show RSE
RSE

# Try every variable
linearmodel4 = smf.ols(formula="Sales~TV+Radio+Newspaper", data=data).fit()
# Get summary
linearmodel4.summary()
# Make predictions
sales_prediction4 = linearmodel4.predict(data[["TV", "Radio", "Newspaper"]])
# Show predictions
sales_prediction4
# Get SSD
SSD = sum((data["Sales"]-sales_prediction4)**2)
# Show SSD
RSE = np.sqrt(SSD/(len(data)-3-1))
# Show RSE
RSE
