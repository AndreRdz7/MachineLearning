# -*- coding: utf-8 -*-
"""
SVM non linear kernel

@author: David André Rodríguez Méndez (AndreRdz7)
"""
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_circles,make_blobs
from mpl_toolkits import mplot3d
from ipywidgets import interact,fixed
# Make model
X, Y = make_circles(100,factor=.1,noise=.1)
def plt_svc(model, ax=None,plot_support=True):
    # Plotting classification in 2D with SVC
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Evaluate model
    xx = np.linspace(xlim[0],xlim[1],30)
    yy = np.linspace(ylim[0],ylim[1],30)
    Y,X = np.meshgrid(yy,xx)
    xy = np.vstack([X.ravel(),Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    # Frontier representation
    ax.contour(X,Y,P,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=["--","-","--"])
    # Get support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=300, linewidth=1,facecolors="none")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.scatter(X[:,0],X[:,1],c=Y,s=50,cmap="autumn")
plt_svc(SVC(kernel="linear").fit(X,Y),plot_support=False)
plt.show()

r = np.exp(-(X**2).sum(1))
def plot_3D(elev=30,azim=30,X=X,Y=Y,r=r):
    ax = plt.subplot(projection="3d")
    ax.scatter3D(X[:,0],X[:,1],r,c=Y,s=50,cmap="autum")
    ax.view_init(elev=elev,azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("r")

interact(plot_3D,elev=[-90,-60,-30,0,30,60,90],azim=[-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180], X=fixed(X), Y=fixed(Y),r=fixed(r))
plt.show()

rbf = SVC(kernel="rbf",C=1E6)
rbf.fit(X,Y)
plt.scatter(X[:,0],X[:,1],c=Y,s=50,cmap="autumn")
plt_svc(rbf)
plt.scatter(rbf.support_vectors_[:,0],rbf.support_vectors_[:,1],s=300,lw=1,facecolors="none")
plt.show()

# Adjust parameters
X,Y = make_blobs(n_samples=100,centers=2,random_state=0,cluster_std=0.8)
plt.scatter(X[:,0],X[:,1],c=Y,s=50,cmap="autumn")
fig, ax = plt.subplots(1,2,figsize=(16,6))
fig.subplots_adjust(left=0.06,right=0.95,wspace=0.1)
for ax_i,C in zip(ax,[10.0,0.1]):
    model = SVC(kernel="linear",C=C)
    model.fit(X,Y)
    ax_i.scatter(X[:,0],X[:,1],c=Y,s=50,cmap="autumn")
    plt_svc(model,ax_i)
    ax_i.set_title("C={0:.1f}".format(C),size=15)