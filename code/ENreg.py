# !/usr/bin/env python
# coding: utf-8
import utils
import HP
import utils
import trainloaded
import embtraining
import torch
import pandas as pd
import numpy as np

x, y, mn, std, xt = utils.loaddata('NAS', 1, dir="./data/", raw=True)

yp_tr = pd.read_csv("./data/train_soro.csv")
yp_te = pd.read_csv("./data/test_soro.csv")
yp_tr.index = pd.DatetimeIndex(yp_tr['date'])
yp_te.index = pd.DatetimeIndex(yp_te['date'])
yptr = yp_tr.drop(yp_tr.columns.difference(['GPPp']), axis=1)
ypte = yp_te.drop(yp_te.columns.difference(['GPPp']), axis=1)
#yp = yptr.merge(ypte, how="outer")

#print(len(yptr), len(ypte))
#print(yptr, ypte)
yp = pd.concat([yptr, ypte])
#print(yp)

reg = (yp[1:]-mn['GPP'])/std['GPP']

#print(len(x), len(y))
splits = len(x.index.year.unique())
#print(splits)
x.index, y.index, reg.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(reg))

arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 150, 4)

# architecture search
layersizes, ag = HP.ArchitectureSearch(arch_grid, {'epochs': 300, 'batchsize': 8, 'lr':0.01, "eta": 0.9}, x, y, splits, "arSreg", reg)

# Hyperparameter Search Space
hpar_grid = HP.HParSearchSpace(150, True)

# Hyperparameter search
hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpreg", reg)

print( 'hyperparameters: ', hpars)


grid.to_csv("./NregHP.csv")
ag.to_csv("./NregAS.csv")
