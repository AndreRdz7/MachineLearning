# -*- coding: utf-8 -*-
"""
Logistic regression with max likelyhood method

Author: David André Rodríguez Méndez (AndreRdz7)
"""
""" 

From scratch

# Import libraries
import numpy as np
from numpy import linalg
# Likelihood
def likelihood(y, pi):
    total_sum = 1
    sum_in = list(range(1,len(y)+1))
    for i in range(len(y)):
        sum_in[i] = np.where(y[i] == 1, pi[i], 1-pi[i])
        total_sum = total_sum * sum_in[i]
    return total_sum
# Probabilities
def logitprobs (X, beta):
    n_rows = np.shape(X)[0]
    n_cols = np.shape(X)[1]
    pi = list(range(1,n_rows+1))
    exponent = list(range(1,n_rows+1))
    for i in range(n_rows):
        exponent[i] = 0
        for j in range(n_cols):
            ex = X[i][j] * beta[j]
            exponent[i] = ex + exponent[i]
        with np.errstate(divide="ignore",invalid="ignore"):
            pi[i]= 1 / (np.exp(-exponent[i]))
    return pi
# Diagonal matriz W
def findW(pi):
    n = len(pi)
    W = np.zeros(n*n).reshape(n,n)
    for i in range(n):
        print(i)
        W[i,i] = pi[i]*(1-pi[i])
        W[i,i].astype(float)
    return W
# Obtain solution of logistic function
def logistics(X,Y,limit):
    nrow = np.shape(X)[0]
    bias = np.ones(nrow).reshape(nrow,1)
    X_new = np.append(X, bias, axis = 1)
    ncol = np.shape(X_new)[1]
    beta = np.zeros(ncol).reshape(ncol,1)
    root_dif = np.array(range(1,ncol+1)).reshape(ncol,1)
    iter_i = 10000
    while(iter_i > limit):
        print("Iteration: i " + str(iter_i) + "," + str(limit))
        pi = logitprobs(X_new,beta)
        print("pi: " + str(pi))
        W = findW(pi)
        print("W: " + str(W))
        num = (np.transpose(np.matrix(X_new))*np.matrix(Y-np.transpose(pi)).transpose())
        den = (np.matrix(np.transpose(X_new))*np.matrix(W)*np.matrix(X_new))
        root_dif = np.array(linalg.inv(den)*num)
        beta = beta + root_dif
        print("beta: " + str(beta))
        iter_i = np.sum(root_dif*root_dif)
        li = likelihood(Y,pi)
    return beta

# Handmade testing
X = np.array(range(10)).reshape(10,1)
Y = [0,0,0,0,1,0,1,0,1,1]
bias = np.ones(10).reshape(10,1)
X_new = np.append(X, bias, axis = 1)

a = logistics(X,Y,0.00001)

"""

# Using libraries
# Import library
import statsmodels.api as sm
import numpy as np
# Define arrays
X = np.array(range(10)).reshape(10,1)
Y = [0,0,0,0,1,0,1,0,1,1]
bias = np.ones(10).reshape(10,1)
X_new = np.append(X, bias, axis = 1)
# Create model
logit_model = sm.Logit(Y,X_new)
# train
result = logit_model.fit()
# Show results
print(result.summary2())