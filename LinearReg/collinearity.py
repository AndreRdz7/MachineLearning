# -*- coding: utf-8 -*-
"""
collinearity

@author: David André Rodríguez Méndez (AndreRdz7)
"""

# Import libraries
import pandas as pd
import statsmodels.formula.api as smf
# Import dataset
data = pd.read_csv("./Advertising.csv")

# Check collinearity on Newpaper vs TV and Radio
linearmodel_n = smf.ols(formula="Newspaper~TV+Radio",data = data).fit()
rsquared_n = linearmodel_n.rsquared
VIF = 1/(1-rsquared_n)
VIF

# Check collinearity on TV vs Newspaper and Radio
linearmodel_n = smf.ols(formula="TV~Newspaper+Radio",data = data).fit()
rsquared_n = linearmodel_n.rsquared
VIF = 1/(1-rsquared_n)
VIF

# Check collinearity on Radio vs TV and Newspaper
linearmodel_n = smf.ols(formula="Radio~TV+Newspaper",data = data).fit()
rsquared_n = linearmodel_n.rsquared
VIF = 1/(1-rsquared_n)
VIF