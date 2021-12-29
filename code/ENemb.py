# !/usr/bin/env python
# coding: utf-8
import utils
import HP
import utils
import training
import torch
import pandas as pd
import numpy as np

x, y, xt = utils.loaddata('NAS', 0, dir="./data/", raw=True)

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
