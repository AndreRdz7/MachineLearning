# -*- coding: utf-8 -*-
"""
Logistic regression

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import pandas as pd
# Get dataset
df = pd.read_csv("./Gender Purchase.csv")
print(df.head())
# print(df.shape)
# Crate crosstable
contingency_table = pd.crosstab(df["Gender"],df["Purchase"])
# Totals
contingency_table.sum(axis = 1)
# Proportion
contingency_table.astype("float").div(contingency_table.sum(axis = 1),axis = 0)

pm = 121/246
pf = 159/256
odds_m = pm/(1-pm)
odds_f = pf(1-pf)
odds_r = odds_m/odds_f

