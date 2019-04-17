# -*- coding: utf-8 -*-
"""
Data exploration analysis and
logistic regression for bank predictions

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
import statsmodel.api as sm
from ggplot import *
# Get dataset
data = pd.read_csv("./bank.csv", sep=";")
# Data conversion
data["y"] = (data["y"]=="yes").astype(int)
data["education"] = np.where(data["education"]=="basic.4y","Basic",data["education"])
data["education"] = np.where(data["education"]=="basic.6y","Basic",data["education"])
data["education"] = np.where(data["education"]=="basic.9y","Basic",data["education"])
data["education"] = np.where(data["education"]=="high.school","High School",data["education"])
data["education"] = np.where(data["education"]=="professional","Professional Course",data["education"])
data["education"] = np.where(data["education"]=="university.degree","University Degree",data["education"])
data["education"] = np.where(data["education"]=="illiterate","Illiterate",data["education"])
data["education"] = np.where(data["education"]=="unknown","Unknown",data["education"])
# Data analysis
pd.crosstab(data.education, data.y).plot(kind="bar")
plt.title("Frecuencia de compra en funcion del nivel de educación")
plt.xlabel("Nivel de educación")
plt.ylabel("Frecuencia de compra del producto")
table = pd.crosstab(data.marital, data.y)
table.div(table.sum(1).astype(float),axis = 0).plot(kind="bar",stacked=True)
plt.title("Diagrama apilado de estado civil contra el nivel de compras")
plt.xlabel("Estado civil")
plt.ylabel("Proporción de clientes")
table2 = pd.crosstab(data.month, data.y).plot(kind="bar")
plt.title("Diagrama apilado de estado civil contra el mes")
plt.xlabel("Mes del año")
plt.ylabel("Proporción de clientes")
table3 = pd.crosstab(data.poutcome,data.y).plot(kind="bar")
# Category conversion
categories = {"job", "marital", "education", "housing", "loan", "contact", "month", "day_of_week", "poutcome"}
for category in categories:
    cat_list = "cat"+"_"+category
    cat_dummies = pd.get_dummies(data[category],prefix=category)
    data_new = data.join(cat_dummies)
    data = data_new
data_vars = data.columns.values.tolist()
to_keep = [v for v in data_vars if v not in categories]
to_keep = [v for v in data_vars if v not in ["default"]]
bank_data = data[to_keep]
# Getting final data
bank_data_vars = bank_data.columns.values.tolist()
Y = ['y']
X = [v for v in bank_data_vars if v not in Y]
# Variable selection
n = 12
lr = LogisticRegression()
rfe = RFE(lr,n)
rfe = rfe.fit(bank_data[X],bank_data[Y].values.ravel())
z = zip(bank_data_vars,rfe.support_,rfe.ranking_)
print(list(z))
# Variable selection
cols = ["previous","euribor3m","job_blue-collar","job_retired","month_aug","month_dec","month_jul","month_jun","month_mar","month_nov","day_of_week_wed", "poutcome_nonexistent"]
X = bank_data[cols]
Y = bank_data["y"]
# Implementation of the model
logit_model = sm.Logit(Y,X)
result = logit_model.fit()
print(result.summary2())
# Implementation of the model with scikit
lm = linear_model.LogisticRegression()
lm.fit(X,Y)
print(lm.score())
print(pd.DataFrame(list(zip(X.columns, np.transpose(lm.coef_)))))
# Model validation
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 0)
lm = linear_model.LogisticRegression()
lm.fit(X_train,Y_train)
probabilities = lm.predict_proba(X_test)
print(probabilities)
prediction = lm.predict(X_test)
print(prediction)
prob = probabilities[:,1]
prob_df = pd.DataFrame(prob)
threshold = 0.1
prob_df["prediction"] = np.where(prob_df[0]>threshold,1,0)
prob_df["actual"] = list(Y_test)
print(prob_df.head())
counter = pd.crosstab(prob_df.prediction,columns = "count")
# Metrics
print(metrics.accuracy_score(Y_test,prediction))
# Cross validation
scores = cross_val_score(linear_model.LogisticRegression(),X,Y,scoring="accuracy",cv=10)
print(scores)
print(scores.mean())
# ROC Curve Creation
confusion_matrix = pd.crosstab(prob_df.prediction,prob_df.actual)
print(confusion_matrix)
TN = confusion_matrix[0][0]
TP = confusion_matrix[1][1]
FP = confusion_matrix[0][1]
FN = confusion_matrix[1][0]
sens = TP/(TP+FN)
print(sens)
espec_1 = 1-TN/(TN+FP)
print(espec_1)
# Try different thresholds
thresholds = [0.04,0.05,0.07,0.1,0.12,0.18,0.2,0.25]
sensitivities = [1]
especifities_1 = [1]
for t in thresholds:
    prob_df["prediction"] = np.where(prob_df[0]>=t,1,0)
    prob_df["actual"] = list(Y_test)
    confusion_matrix = pd.crosstab(prob_df.prediction,prob_df.actual)
    print(confusion_matrix)
    TN = confusion_matrix[0][0]
    TP = confusion_matrix[1][1]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    sens = TP/(TP+FN)
    sensitivities.append(sens)
    espec_1 = 1-TN/(TN+FP)
    especifities_1.append(espec_1)
sensitivities.append(0)
especifities_1.append(0)
print(sensitivities)
print(especifities_1)
# Plot ROC
plt.plot(especifities_1,sensitivities,marker="o", linestyle="--",color="r")
x = [i*0.01 for i in range(100)]
y = [i*0.01 for i in range(100)]
plt.plot(x,y)
plt.title("Curva ROC")
plt.xlabel("1-Especificidad")
plt.ylabel("Sensibilidad")
plt.show()
# New ggplot
espec_1,sensit,_ = metrics.roc_curve(Y_test,prob)
df = pd.DataFrame({
    "x":espec_1,
    "y":sensit
})
ggplot(df,aes(x ="x",y = "y")) + geom_line() + geom_abline(linetype = "dashed")
auc = metrics.auc(espec_1,sensit)
print(auc)
# Area under the curve
ggplot(df, aes(x="x",y="y")) + geom_area(alpha = 0.25) + geom_line(aes(y="y")) + ggtitle("Curva ROC y ABC=%s" % str(auc))