#%%
import sys
import matplotlib
import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.neighbors import KernelDensity as KD
from matplotlib.colors import ListedColormap
# Fix random state for reproducibility
np.random.seed(1978081)
# Matplotlib setting
plt.rcParams['text.usetex'] = True
matplotlib.rcParams['figure.dpi']= 300

mm0 = np.array([2,2])
mm1= np.array([4,4])
Sig0 = 4*np.identity(2)
Sig1 = 4*np.identity(2)
for N in [50, 100, 250, 500]:
    X0 = mvn.rvs(mm0,Sig0,N)
    x0,y0 = np.split(X0,2,1)
    X1 = mvn.rvs(mm1,Sig1,N)
    x1,y1 = np.split(X1,2,1)
    X = np.concatenate((X0,X1),axis=0)
    y = np.concatenate((np.zeros(N),np.ones(N)))
    cmap_light = ListedColormap(["#FFE0C0","#B7FAFF"])
    s = .01  # mesh step size
    x_min,x_max = (-3,9)
    y_min,y_max = (-3,9)
    for h in [0.1,0.3,0.5,1, 2, 5]:
        clf0 = KD(bandwidth=h)
        clf0.fit(X0)
        clf1 = KD(bandwidth=h)
        clf1.fit(X1)
        xx,yy = np.meshgrid(np.arange(x_min,x_max,s),np.arange(y_min,y_max,s))
        Z0 = clf0.score_samples(np.c_[xx.ravel(), yy.ravel()])
        Z1 = clf1.score_samples(np.c_[xx.ravel(), yy.ravel()])
        Z = Z0<=Z1
        Z = Z.reshape(xx.shape)
        fig,ax=plt.subplots(figsize=(8,8),dpi=150)
        plt.rc("xtick",labelsize=16)
        plt.rc("ytick",labelsize=16)
        plt.plot(x0,y0,".r",markersize=8) # class 0
        plt.plot(x1,y1,".b",markersize=8) # class 1
        plt.title("N={},h={}".format(N,h))
        plt.xlim([-3,9])
        plt.ylim([-3,9])
        plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
        ax.contour(xx,yy,Z,colors="black",linewidths=0.5)
        plt.show()
        fig.savefig("img/c05_kernel"+"_h_"+str(int(10*h))+"_N_"+str(int(N))+".png",bbox_inches="tight",facecolor="white")