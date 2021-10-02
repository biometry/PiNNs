# !/usr/bin/env python
# coding: utf-8
import utils
import HP
import utils
import trainloaded
import torch
import pandas as pd
import numpy as np

x, y, xt = utils.loaddata('NAS', 0, dir="./data/", raw=True)
yp = xt.GPPp
swmn = np.mean(xt.SWp)
swstd = np.std(xt.SWp)
print(yp, swmn, swstd)
xt = xt.drop(['date', 'ET', 'Unnamed: 0', 'GPP', 'SWp', 'GPPp', 'ETp'], axis=1)
print(xt)
ypp = yp
splits = len(x.index.year.unique())
x.index, y.index, ypp.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(ypp))

y = y.to_frame()
print(x,y,xt)
#arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 140, 4, emb=True)

# architecture search
#layersizes, ag = HP.ArchitectureSearch(arch_grid, {'epochs': 100, 'batchsize': 16, 'lr':0.001, 'eta': 0.2}, x, y, splits, "arSemb2", reg=yp, emb=True, raw = xt, embtp=2, hp=True, sw=(swmn, swstd))
#ag.to_csv("./Nemb2AS.csv")

layersizes = [[64, 4], [2]]
# Hyperparameter Search Space
hpar_grid = HP.HParSearchSpace(140, reg=True, emb=True)

# Hyperparameter search
hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpemb2", reg=ypp, emb=True, raw=xt, embtp=2, hp=True, sw=(swmn, swstd))

print( 'hyperparameters: ', hpars)
grid.to_csv("./Nemb2HP.csv")

