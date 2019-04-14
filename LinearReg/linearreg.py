# -*- coding: utf-8 -*-
"""
Simple linear regression with Python,
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

"""
Since our R squared value is low and our error is high, we need to
implement another solution, by considerating the other two variables
respect to sales, with multiple linear regression.
"""
