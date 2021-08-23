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

x, y, mn, std, xt = utils.loaddata('NAS', 0, dir="./data/", raw=True)

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

x = (yp[1:]-mn['GPP'])/std['GPP']


#print(len(x), len(y))
splits = len(x.index.year.unique())
#print(splits)
x.index, y.index = np.arange(0, len(x)), np.arange(0, len(y))

arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 150, 4)

# architecture search
layersizes, agrid = HP.ArchitectureSearch(arch_grid, {'epochs': 300, 'batchsize': 8, 'lr':0.01}, x, y, splits, "arSres")
agrid.to_csv("./NresAS.csv")

# Hyperparameter Search Space
hpar_grid = HP.HParSearchSpace(150)

# Hyperparameter search
hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpres")

print( 'hyperparameters: ', hpars)


grid.to_csv("./NresHP.csv")


