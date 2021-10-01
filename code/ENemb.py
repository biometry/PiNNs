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
<<<<<<< HEAD
xt = xt.drop(['date', 'GPP', 'ET', 'GPPp', 'ETp', 'SWp', 'Unnamed: 0'], axis=1)
print(xt)
l, m, yp = utils.loaddata('NAS', 0, dir="./data/", raw=True)

yp = yp.drop(yp.columns.difference(['GPPp']), axis=1)
print(yp)
splits = len(x.index.year.unique())
x.index, y.index, yp.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(yp))

y = y.to_frame()

arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 140, 4, emb=True)

# architecture search
layersizes, ag = HP.ArchitectureSearch(arch_grid, {'epochs': 100, 'batchsize': 16, 'lr':0.001, 'eta': 0.2}, x, y, splits, "arSemb", reg=yp, emb=True, raw = xt, hp=True)
ag.to_csv("./NembAS.csv")

# Hyperparameter Search Space
hpar_grid = HP.HParSearchSpace(140, reg=True, emb=True)

# Hyperparameter search
hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpemb", reg=yp, emb=True, raw=xt, hp=True)

print( 'hyperparameters: ', hpars)
grid.to_csv("./NembHP.csv")
=======
xt = xt.drop(['date', 'GPP', 'ET'], axis=1)
yp_tr = pd.read_csv("./data/train_soro.csv")
yp_te = pd.read_csv("./data/test_soro.csv")
yp_tr.index = pd.DatetimeIndex(yp_tr['date'])
yp_te.index = pd.DatetimeIndex(yp_te['date'])
yptr = yp_tr.drop(yp_tr.columns.difference(['GPPp']), axis=1)
ypte = yp_te.drop(yp_te.columns.difference(['GPPp']), axis=1)
yp = pd.concat([yptr, ypte])
ypp = yp
splits = len(x.index.year.unique())
x.index, y.index, ypp.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(ypp))

y = y.to_frame()
print(x,y,xt)
arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 400, 4, emb=True)

# architecture search
layersizes, ag = HP.ArchitectureSearch(arch_grid, {'epochs': 100, 'batchsize': 32, 'lr':0.01, 'eta': 0.5}, x, y, splits, "arSemb", reg=ypp, emb=True, raw = xt)
ag.to_csv("./NembAS.csv")

# Hyperparameter Search Space
#hpar_grid = HP.HParSearchSpace(400, reg=True, emb=True)

# Hyperparameter search
#hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpemb", reg=ypp, emb=True, raw=xt)

#print( 'hyperparameters: ', hpars)
#grid.to_csv("./NembHP.csv")
>>>>>>> origin/main

