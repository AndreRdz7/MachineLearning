# -*- coding: utf-8 -*-
"""
Principal componen analysis

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import pandas as pd
import numpy as np
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
from sklearn.preprocessing import StandardScaler
#  Plotly login
tls.set_credentials_file(username='AndreRdz7',api_key='vhsv2rzIYldslgzAAeMO')
# Get dataset
df = pd.read_csv("./iris.csv")
X = df.iloc[:,0:4].values
Y = df.iloc[:,4].values
"""
traces = []
legend = {0: False, 1: False, 2:False, 3:False}
colors = {
    'setosa': 'rgb(255,217,20)',
    'versicolor': 'rgb(31,220,120)',
    'virginica': 'rgb(44,50,180)'
}
for col in range(4):
    for key in colors:
       traces.append(Histogram(x=X[y==key,col],opacity=0.7,xaxis="x%s"%(col+1),marker=Marker(color=colors[key]),name=key,showlegend=legend[col]))
data = Data(traces)
layout = Layout(barmode="overlay",xaxis=XAxis(domain=[0,0.25],title="Long. Sépalos (cm)"),xaxis2=XAxis(domain=[0.3,0.5],title="Anch. Sépalos (cm)"),xaxis3=XAxis(domain=[0.55,0.75],title="Long. Pétalos (cm)"),xaxis4=XAxis(domain=[0.8,1.0],title="Anch. Pétalos (cm)"),yaxis=YAxis(title="Número de ejemplares"),title="Distribución de los rasgos de las flores iris")

fig = Figure(data=data,layout=layout)
py.iplot(fig)
"""
# Make it standard
X_std=StandardScaler().fit_transform(X)
# Value and vector decomposition
# a
mean_vect = np.mean(X_std,axis=0)
print(mean_vect)
cov_matrix = (X_std - mean_vect).T.dot((X_std - mean_vect))/(X_std.shape[0]-1)
print(cov_matrix)
np.cov(X_std.T)
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
# b
corr_matrix = np.corrcoef(X_std.T)
print(corr_matrix)
eigen_values, eigen_vectors = np.linalg.eig(corr_matrix)
# c
u,s,v = np.linalg.svd(X_std.T)
# PCA
for ev in eigen_vectors:
    print("La longitud del VP es : %s"%np.linalg.norm(ev))
eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
print(eigen_pairs)
eigen_pairs.sort()
eigen_pairs.reverse()
print("Valores propios en orden ascendente")
for ep in eigen_pairs:
    print(ep[0])
total_sum = sum(eigen_values)
var_exp = [(i/total_sum)*100 for i in sorted(eigen_values,reverse=True)]
cum_var_exp = np.cumsum(var_exp)
W = np.hstack((eigen_pairs[0][1].reshape(4,1),eigen_pairs[1][1].reshape(4,1)))
print(W)
# Project variables
Y = X_std.dot(W)
results = []
for name in ('setosa','versicolor','virginica'):
    result = Scatter(x=Y[y==name,0],y=[y==name,1],mode="markers",name=name,marker=Marker(size=12,line=Line(color="rgba(220,220,220,0.15",width=0.5),opacity=0.8))
    results.append(result)
data = Data(results)
layout = Layout(showlegend=True,scene=Scene(xaxis=XAxis(title="Componente principal 1"),yaxis=YAxis(title="Componente principal 2")))
fig = Figure(data=data,layout=layout)
py.iplot(fig)