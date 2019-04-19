# -*- coding: utf-8 -*-
"""
Distance matrix for clustering

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
# Get the dataset
data = pd.read_csv("./movies.csv", sep=";")
print(data.head())
movies = data.columns.values.tolist()[1:]
print(movies)
# Distnaces
dd1 = distance_matrix(data[movies], data[movies],p=1)
dd2 = distance_matrix(data[movies], data[movies],p=2)
dd10 = distance_matrix(data[movies], data[movies],p=10)
def dm_to_df(dd,col_name):
    return pd.DataFrame(dd,index = col_name, columns = col_name)
dm_to_df(dd1,data["user_id"])
dm_to_df(dd2,data["user_id"])
dm_to_df(dd10,data["user_id"])
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
ax.scatter(xs = data["star_wars"], ys = data["lord_of_the_rings"],zs =["harry_potter"])
# Data binding
df1 = dm_to_df(dd1,data["user_id"])
Z = []
df[11] = df[1] + df[10]
df.loc[11] = df.loc[1] + df.loc[10]
Z.append([1,10,0.7,2]) # id1, id2, d , cluster elements
# Cluster
for i in df.columns,values,tolist():
    df.loc[11][i] = min(df.loc[1][i], df.loc[10][i])
    df.loc[i][11] = min(df.loc[i][1], df.loc[i][10])
df = drop([1,10])
df = drop([1,10], axis = 1)
print(df)
# Repeat the clustering
x = 2
y = 7
n = 12
df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
Z.append([x,y,df.loc[x][y],2]) # id1, id2, d , cluster elements
# Cluster
for i in df.columns,values,tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])
print(df)
df = drop([x,y])
df = drop([x,y], axis = 1)
print(df)
# And so on and so forth (the best is to automate via function)
x = 5
y = 6
n = 13
df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
Z.append([x,y,df.loc[x][y],2]) # id1, id2, d , cluster elements
for i in df.columns,values,tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])
print(df)
df = drop([x,y])
df = drop([x,y], axis = 1)
print(df)
x = 11
y = 13
n = 14
df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
Z.append([x,y,df.loc[x][y],2]) # id1, id2, d , cluster elements
for i in df.columns,values,tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])
print(df)
df = drop([x,y])
df = drop([x,y], axis = 1)
print(df)
x = 9
y = 12
z = 14
n = 15
df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
Z.append([x,y,df.loc[x][y],3]) # id1, id2, d , cluster elements
for i in df.columns,values,tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i],df.loc[z][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y],df.loc[i][z])
print(df)
df = drop([x,y,z])
df = drop([x,y,z], axis = 1)
print(df)
x = 4
y = 6
z = 15
n = 16
df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
Z.append([x,y,df.loc[x][y],3]) # id1, id2, d , cluster elements
for i in df.columns,values,tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i],df.loc[z][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y],df.loc[i][z])
print(df)
df = drop([x,y,z])
df = drop([x,y,z], axis = 1)
print(df)
x = 3
y = 16
n = 17
df[n] = df[x] + df[y]
df.loc[n] = df.loc[x] + df.loc[y]
Z.append([x,y,df.loc[x][y],2]) # id1, id2, d , cluster elements
for i in df.columns,values,tolist():
    df.loc[n][i] = min(df.loc[x][i], df.loc[y][i])
    df.loc[i][n] = min(df.loc[i][x], df.loc[i][y])
print(df)
df = drop([x,y])
df = drop([x,y], axis = 1)
print(df)