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

x, y, xt = utils.loaddata('NAS', 1, dir="./data/", raw=True)

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

reg = yp[1:]
y = y.to_frame()
#print(len(x), len(y))
splits = len(x.index.year.unique())
#print(splits)
print(x,y, reg)

x.index, y.index, reg.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(reg))

arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 800, 4)

# architecture search
layersizes, ag = HP.ArchitectureSearch(arch_grid, {'epochs': 100, 'batchsize': 8, 'lr':0.01, "eta": 0.5}, x, y, splits, "arSreg", reg)
ag.to_csv("./NregAS.csv")

#layersizes = [4, 32, 2, 16]
# Hyperparameter Search Space
hpar_grid = HP.HParSearchSpace(800, True)

# Hyperparameter search
hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpreg", reg)

print( 'hyperparameters: ', hpars)


grid.to_csv("./NregHP.csv")

