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

def plot_kd(ax, x0, y0, x1, y1, Z):
    cmap_light = ListedColormap(["#FFE0C0","#B7FAFF"])
    plt.rc("xtick",labelsize=16)
    plt.rc("ytick",labelsize=16)
    ax.plot(x0,y0,".r",markersize=8) # class 0
    ax.plot(x1,y1,".b",markersize=8) # class 1
    ax.set_title("N={},h={}".format(Ns[i],h))
    ax.set_xlim([-3,9])
    ax.set_ylim([-3,9])
    ax.pcolormesh(xx,yy,Z,cmap=cmap_light)
    ax.contour(xx,yy,Z,colors="black",linewidths=0.5)
    return ax


mm0 = np.array([2,2])
mm1= np.array([4,4])
Sig0 = 4*np.identity(2)
Sig1 = 4*np.identity(2)
Ns = [50, 100, 250, 500]
#Ns = [50]
hs = [0.1,0.3,0.5,1, 2, 5]
#hs = [0.1]
Xs = [[mvn.rvs(mm0, Sig0, n), mvn.rvs(mm1,Sig1,n)] for n in Ns]

clf0s = [[KD() for i in range(0, len(hs))] for j in range(0, len(Ns))]
clf1s = [[KD() for i in range(0, len(hs))] for j in range(0, len(Ns))]

s = .1 #0.01 # mesh step size
plts = [plt.subplots(figsize=(8,8), dpi=150) for i in range(0, len(Ns)*len(hs))]
figs, axs = list(zip(*plts))

for (i, X) in enumerate(Xs):
    x0,y0 = np.split(X[0],2,1)
    x1,y1 = np.split(X[1],2,1)
    y = np.concatenate((np.zeros(Ns[i]),np.ones(Ns[i])))
    x_min,x_max = (-3,9)
    y_min,y_max = (-3,9)
    for (j, h) in enumerate(hs):
        clf0s[i][j] = KD(bandwidth=h)
        clf0s[i][j].fit(X[0])
        clf1s[i][j] = KD(bandwidth=h)
        clf1s[i][j].fit(X[1])
        xx,yy = np.meshgrid(np.arange(x_min,x_max,s),np.arange(y_min,y_max,s))
        Z0 = clf0s[i][j].score_samples(np.c_[xx.ravel(), yy.ravel()])
        Z1 = clf1s[i][j].score_samples(np.c_[xx.ravel(), yy.ravel()])
        Z = Z0<=Z1
        Z = Z.reshape(xx.shape)
        plot_kd(axs[i+j], x0, y0, x1, y1, Z)
        figs[i+j].savefig("img/c05_kernel"+"_h_"+str(int(10*h))+"_N_"+str(int(Ns[i]))+".png",bbox_inches="tight",facecolor="white")
# %% Test error

def measure_test_error(clf0, clf1, xxs, yys, ys):
    Z0 = clf0.score_samples(np.c_[xxs, yys])
    Z1 = clf1.score_samples(np.c_[xxs, yys])
    Z = Z0<=Z1
    return np.count_nonzero(Z != ys) / len(ys)

nt = 500
X_test = [mvn.rvs(mm0, Sig0, nt), mvn.rvs(mm1,Sig1,nt)] 
x0,y0 = np.split(X_test[0],2,1)
x1,y1 = np.split(X_test[1],2,1)
ys = np.concatenate((np.zeros(nt),np.ones(nt)))

pltts = [plt.subplots(figsize=(8,8), dpi=150) for i in range(0, len(Ns)*len(hs))]

for i in range(0, len(Ns)):
    xxs = np.concatenate((x0, x1))
    yys = np.concatenate((y0, y1))
    ys = np.concatenate((np.zeros(Ns[i]),np.ones(Ns[i])))
    for (j, h) in enumerate(hs):
        err = measure_test_error(clf0s[i][j], clf1s[i][j], xxs, yys, ys)

        axs[i+j].annotate("Test Error: {}\n Bayes Error: {}".format(err, 2), (6,8))
        figs[i+j].savefig("img/c05_kernel_test"+"_h_"+str(int(10*h))+"_N_"+str(int(Ns[i]))+".png",bbox_inches="tight",facecolor="white")


# %%
