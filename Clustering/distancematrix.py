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
